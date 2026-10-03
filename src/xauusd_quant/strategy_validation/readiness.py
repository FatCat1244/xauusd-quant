"""Historical criteria and prospective evidence are separate; missing null/spec proof blocks.

Stage 12 promotion remains unchanged. This stage can assess declared historical
reconstruction, but never turn inspected development outcomes into prospective ones.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..execution.config import ExecutionConfig, content_hash
from ..execution.io import parse_utc
from ..execution.readiness import ADAPTIVE_CHOICES, code_identity, inventory, sha256
from ..utils.config import Config
from .plan import ExperimentPlan

REQUIRED_GATES = (
    "data_provenance",
    "fold_local_pipeline",
    "null_evidence",
    "execution_specification",
    "evidence_identity",
    "evaluation_history_declared",
)


def validate_adaptive_evidence(records: dict[str, Any], cutoff: datetime) -> str:
    """Used for any externally asserted selection/ensemble creation path, never trust its label."""
    status = "passed"
    for choice in ADAPTIVE_CHOICES:
        value = records.get(choice)
        if value is None:
            if status != "failed":
                status = "unknown"
        elif parse_utc(value) >= cutoff:
            status = "failed"
    return status


def assess(
    config: Config,
    execution: ExecutionConfig,
    plan: ExperimentPlan,
    plan_path: Path,
    evidence_path: Path | None = None,
) -> dict[str, Any]:
    facts = inventory(config)
    gates = {
        "data_provenance": facts["data_metadata_status"],
        "fold_local_pipeline": "passed",  # restricted new path, enforced in fit_fold
        "null_evidence": "unknown",
        "execution_specification": "unknown",
        "evidence_identity": "unknown",
        "evaluation_history_declared": "passed",
    }
    evidence: dict[str, Any] = {}
    errors: list[str] = []
    if evidence_path is not None:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        plan_hash = json.loads(plan_path.read_text())["content_sha256"]
        if (
            evidence.get("plan_sha256") != plan_hash
            or evidence.get("dataset_version") != facts["dataset_version"]
            or evidence.get("pipeline_source_identity")
            != code_identity(config.project_root)["source_identity"]
        ):
            errors.append("evidence plan/data/current source identity differs")
        files = evidence.get("files") or {}
        verified = set()
        for name, expected in files.items():
            path = config.project_root / name
            if (
                not path.resolve().is_relative_to(config.project_root.resolve())
                or not path.is_file()
                or sha256(path) != expected
            ):
                errors.append(f"evidence hash/path invalid: {name}")
            else:
                verified.add(name)
        gates["evidence_identity"] = "passed" if verified and not errors else "unknown"
        statuses = []
        for fold in plan.folds:
            row = (evidence.get("folds") or {}).get(fold.fold_id) or {}
            # Old global ensemble/feature evidence needs every adaptive choice proven.
            if "external_adaptive_choices" in row:
                current = validate_adaptive_evidence(row["external_adaptive_choices"], fold.start)
                previous = gates["fold_local_pipeline"]
                gates["fold_local_pipeline"] = (
                    "failed"
                    if "failed" in (current, previous)
                    else "unknown"
                    if "unknown" in (current, previous)
                    else "passed"
                )
            for null in ("random_walk", "sign_flip"):
                record = (row.get("nulls") or {}).get(null) or {}
                status = "unknown"
                if record.get("status") == "failed":
                    status = "failed"
                elif (
                    record.get("status") == "passed"
                    and record.get("file") in verified
                    and record.get("as_of_utc")
                    and parse_utc(record["as_of_utc"]) < fold.start
                    and record.get("plan_sha256") == plan_hash
                ):
                    status = "passed"
                statuses.append(status)
        gates["null_evidence"] = (
            "failed"
            if "failed" in statuses
            else "passed"
            if statuses and all(s == "passed" for s in statuses)
            else "unknown"
        )
        spec = evidence.get("execution_specification") or {}
        if (
            execution.specification_status == "verified_supplied"
            and spec.get("file") in verified
            and spec.get("execution_sha256") == content_hash(execution.resolved())
        ):
            gates["execution_specification"] = "passed"
        if errors:
            gates["null_evidence"] = gates["execution_specification"] = "unknown"
    return {
        "version": "STRATEGY_READINESS_V001",
        "gates": gates,
        "inventory": facts,
        "evidence_file_sha256": sha256(evidence_path) if evidence_path else None,
        "evidence_errors": errors,
        "classification": plan.evaluation_history_classification,
        "prospective_status": "not_established; inspected history and development-only reader",
        "legacy_stage12_status": "historical_diagnostic_only; global selection path not reused",
        "legacy_creation_path_limitations": facts["scientific_limitations"],
        "minimum_repair": "new pipeline-matched prior-only null evidence and verified execution specifications; legacy model reuse additionally needs fold-local feature counts/identities and ensemble universe/screening",
    }
