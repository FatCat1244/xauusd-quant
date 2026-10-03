"""Versioned evidence gates; historical fit chronology alone never establishes OOS selection.

No outcome values or reserved feature rows are read. Footer checks verify identity
and coverage, not the previously recorded full-content digests. Missing is unknown.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ..data.converter import dataset_version
from ..utils.config import Config
from .config import ExecutionConfig, content_hash
from .policy import utc_time

READINESS_VERSION = "EXECUTION_READINESS_V001"
ADAPTIVE_CHOICES = (
    "feature_identities",
    "feature_count",
    "model_fit",
    "preprocessing",
    "tuning",
    "calibration",
    "policy",
    "eligibility",
    "universe",
    "correlation",
    "weights",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_identity(root: Path) -> dict[str, Any]:
    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL)

    files = sorted([*root.glob("src/**/*.py"), *root.glob("config/*.yaml")])
    source = {str(p.relative_to(root)): sha256(p) for p in files}
    try:
        commit = git("rev-parse", "HEAD").decode().strip()
        status = git("status", "--porcelain").decode().splitlines()
        diff = hashlib.sha256(git("diff", "HEAD", "--binary")).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        commit, status, diff = None, ["git identity unavailable"], None
    return {
        "commit": commit,
        "working_tree_status": status,
        "diff_sha256": diff,
        "source_sha256": source,
        "source_identity": content_hash(source),
    }


def _verify_spec(path: Path, root: Path) -> dict[str, Any]:
    """Use the repository's actual hash-key declaration without importing fitted models."""
    body = json.loads(path.read_text(encoding="utf-8"))
    namespace = "ensemble" if path.name.startswith("ENSEMBLE_SPEC") else "ml"
    tree = ast.parse(
        (root / "src/xauusd_quant" / namespace / "registry.py").read_text(encoding="utf-8")
    )
    fields = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_HASHED" for t in node.targets)
    )
    blob = json.dumps(
        {k: body.get(k) for k in fields}, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    expected = hashlib.blake2b(blob, digest_size=16).hexdigest()
    if body.get("frozen") is not True or body.get("content_hash") != expected:
        raise ValueError(f"invalid frozen spec: {path}")
    return body


def inventory(config: Config) -> dict[str, Any]:
    """Bounded metadata audit of actual files, without another research or final-test run."""
    root, directory = config.project_root, config.processed_data_path
    path = directory / "_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    failures, rows, boundaries = [], 0, []
    if manifest.get("tick_fingerprint") != config.tick_fingerprint():
        failures.append("current tick/timezone settings differ from manifest")
    declared_paths = set()
    for entry in manifest["partitions"]:
        file = directory / entry["path"]
        if not file.resolve().is_relative_to(directory.resolve()):
            raise ValueError("partition path escapes tick dataset")
        declared_paths.add(file.resolve())
        if not file.exists():
            failures.append(f"missing {entry['path']}")
            continue
        metadata = pq.read_metadata(file)
        rows += metadata.num_rows
        if metadata.num_rows != entry["rows"] or file.stat().st_size != entry["bytes"]:
            failures.append(f"row count / file size mismatch {entry['path']}")
        column = pq.read_schema(file).names.index("timestamp")
        stats = [
            metadata.row_group(i).column(column).statistics for i in range(metadata.num_row_groups)
        ]
        if stats and all(s is not None and s.has_min_max for s in stats):
            first = min(s.min for s in stats)
            last = max(s.max for s in stats)
            boundaries.append((first, last))
            if first != datetime.fromisoformat(
                entry["first_timestamp"]
            ) or last != datetime.fromisoformat(entry["last_timestamp"]):
                failures.append(f"timestamp footer mismatch {entry['path']}")
        else:
            failures.append(f"timestamp statistics unavailable {entry['path']}")
        if entry.get("status") != "complete" or entry.get("validated") is not True:
            failures.append(f"partition validation unknown {entry['path']}")
    actual_paths = {p.resolve() for p in directory.glob("year=*/month=*/*.parquet")}
    if actual_paths != declared_paths:
        failures.append("partition files differ from manifest")
    if dataset_version(manifest) != manifest.get("dataset_version"):
        failures.append("manifest content identity differs from dataset version")
    if (
        rows != manifest.get("total_rows")
        or manifest.get("status") != "complete"
        or manifest.get("truncated_by_limit")
        or manifest.get("missing_partitions")
    ):
        failures.append("partial or incomplete tick dataset")
    if boundaries:
        first, last = min(a for a, _ in boundaries), max(b for _, b in boundaries)
        expected_months = {
            (i // 12, i % 12 + 1)
            for i in range(first.year * 12 + first.month - 1, last.year * 12 + last.month)
        }
        if {(int(e["year"]), int(e["month"])) for e in manifest["partitions"]} != expected_months:
            failures.append("missing calendar month")
    raw = config.raw_data_path
    raw_identity = {
        "path": str(raw),
        "exists": raw.exists(),
        "size": raw.stat().st_size if raw.is_file() else None,
        "fresh_raw_hash_computed": False,
    }
    source_size = sum(s["size"] for s in manifest.get("source_signature", []))
    if not raw.is_file() or raw.stat().st_size != source_size:
        failures.append("raw file size differs or raw input absent")
    specs, spec_errors = [], []
    for kind, prefix in (("ml", "MODEL_SPEC"), ("ensemble", "ENSEMBLE_SPEC")):
        for file in sorted(
            (root / "results" / f"{kind}_research").glob(f"*/frozen/{prefix}*.json")
        ):
            try:
                body = _verify_spec(file, root)
                specs.append(
                    {
                        "path": str(file.relative_to(root)),
                        "id": body["spec_id"],
                        "sha256": sha256(file),
                        "hash_verified": True,
                        "target": body["target"],
                        "horizon": body["horizon"],
                    }
                )
            except ValueError as exc:
                spec_errors.append(str(exc))
    fitted, artifact_errors = [], []
    for file in sorted((root / "data/models").glob("*/manifest.json")):
        body = json.loads(file.read_text(encoding="utf-8"))
        bad = []
        for name, expected in body.get("files", {}).items():
            payload = file.parent / name
            if (
                not payload.resolve().is_relative_to(file.parent.resolve())
                or not payload.is_file()
                or sha256(payload) != expected
            ):
                bad.append(name)
        if bad or not body.get("files"):
            artifact_errors.append({"path": str(file), "bad_files": bad})
        fitted.append(
            {
                "path": str(file.relative_to(root)),
                "sha256": sha256(file),
                "model_id": body.get("model_id"),
                "files_verified": not bad,
            }
        )
    selections = []
    for file in sorted((root / "results/feature_selection").glob("*/manifests/*.json")):
        body = json.loads(file.read_text(encoding="utf-8"))
        content = [
            {k: row[k] for k in ("name", "feature_id", "feature_version")}
            for row in body["features"]
        ]
        digest = hashlib.blake2b(
            json.dumps(content, sort_keys=True, separators=(",", ":"), default=str).encode(),
            digest_size=16,
        ).hexdigest()
        selections.append(
            {
                "path": str(file.relative_to(root)),
                "sha256": sha256(file),
                "id": body["feature_set_id"],
                "hash_verified": digest == body["content_hash"],
                "selection_period": body.get("created_from_period"),
                "size_validation_period": body.get("selection_validation_period"),
                "chronology_status": "historical_selection_not_fold_local",
            }
        )
    history = []
    for file in sorted((root / "results").glob("*/*/final_test/access_log.jsonl")):
        entries = [
            json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line
        ]
        history.append(
            {
                "path": str(file.relative_to(root)),
                "sha256": sha256(file),
                "events": dict(Counter(e.get("event") for e in entries)),
            }
        )
    forecasts = [
        {
            "path": str(f.relative_to(root)),
            "rows": pq.read_metadata(f).num_rows,
            "columns": pq.read_schema(f).names,
        }
        for f in sorted(
            (root / "results/ensemble_research").glob("*/joint_predictive_state.parquet")
        )
    ]
    return {
        "readiness_version": READINESS_VERSION,
        "data_metadata_status": "passed" if not failures else "failed",
        "data_failures": failures,
        "manifest_sha256": sha256(path),
        "dataset_version": manifest.get("dataset_version"),
        "actual_footer_rows": rows,
        "partitions": len(declared_paths),
        "first_timestamp": str(min(a for a, _ in boundaries)) if boundaries else None,
        "last_timestamp": str(max(b for _, b in boundaries)) if boundaries else None,
        "full_content_reverified": False,
        "raw": raw_identity,
        "timezone": manifest.get("timezone"),
        "frozen_specs": specs,
        "spec_errors": spec_errors,
        "fitted_artifacts": fitted,
        "artifact_errors": artifact_errors,
        "feature_manifests": selections,
        "forecast_artifacts": forecasts,
        "evaluation_history": history,
        "evaluation_history_classification": "historical_inspected_2022_plus"
        if history
        else "unknown",
        "scientific_limitations": [
            "V001 feature identities use 2003-2017 outcomes; sizes use 2018-2021 validation",
            "ML 2011-2021 scoring precedes some of those adaptive choices",
            "ensemble global eligibility and correlation use all five blocks",
            "wf_universe retains globally outcome-dependent null/structural statuses",
            "earlier 2022+ inspection includes feature research, ML and ensemble tests",
            "no independently established untouched evaluation period",
        ],
    }


def scientific_gates(
    metadata: dict[str, Any],
    evaluation_start: datetime,
    config: ExecutionConfig,
    *,
    evaluation_end: datetime | None = None,
) -> dict[str, str]:
    """All adaptive choices and null evidence must precede every scored fold.

    This validates supplied audit records, not the truth of unverifiable assertions.
    Real promotion additionally requires verified evidence-file hashes and history.
    """
    start = utc_time(evaluation_start)
    gates: dict[str, str] = {}
    for key in ("forecast_chronology", "null_evidence", "untouched_evaluation"):
        gates[key] = "unknown"
    folds = metadata.get("folds") or []
    if folds:
        chronology = "passed"
        null_status = "passed"
        for fold in folds:
            boundary = utc_time(datetime.fromisoformat(fold["evaluation_start_utc"]))
            if "evaluation_end_utc" not in fold:
                chronology = "unknown"
            for choice in ADAPTIVE_CHOICES:
                value = (fold.get("adaptive_as_of_utc") or {}).get(choice)
                if value is None:
                    if chronology != "failed":
                        chronology = "unknown"
                elif utc_time(datetime.fromisoformat(value)) >= boundary:
                    chronology = "failed"
            nulls = fold.get("nulls") or {}
            for name in ("random_walk", "sign_flip"):
                if nulls.get(name) == "failed":
                    null_status = "failed"
                elif nulls.get(name) != "passed" and null_status != "failed":
                    null_status = "unknown"
        gates["forecast_chronology"] = chronology
        gates["null_evidence"] = null_status
    if folds and not any(
        utc_time(datetime.fromisoformat(fold["evaluation_start_utc"]))
        <= start
        < utc_time(datetime.fromisoformat(fold["evaluation_end_utc"]))
        for fold in folds
        if fold.get("evaluation_end_utc")
    ):
        gates["forecast_chronology"] = "failed"
    if folds and evaluation_end is not None:
        # Cover the entire declared interval, including gaps between audited folds.
        cursor = start
        for lower, upper in sorted(
            (
                utc_time(datetime.fromisoformat(fold["evaluation_start_utc"])),
                utc_time(datetime.fromisoformat(fold["evaluation_end_utc"])),
            )
            for fold in folds
            if fold.get("evaluation_end_utc")
        ):
            if upper <= lower:
                gates["forecast_chronology"] = "failed"
            if lower <= cursor < upper:
                cursor = upper
        if cursor < utc_time(evaluation_end):
            gates["forecast_chronology"] = "failed"
    # The known historical manifests can never pass through a new assertion.
    if any(str(x).endswith("_V001") for x in metadata.get("feature_set_ids", [])):
        gates["forecast_chronology"] = "failed"
    if metadata.get("evaluation_history_classification") == "historical_inspected":
        gates["untouched_evaluation"] = "failed"
    elif metadata.get("evaluation_history_classification") == "untouched_verified":
        gates["untouched_evaluation"] = "passed"
    gates["broker_specification"] = (
        "passed" if config.specification_status == "verified_supplied" else "unknown"
    )
    gates["forecast_units_and_horizon"] = (
        "passed"
        if (
            metadata.get("timeframe"),
            metadata.get("target"),
            metadata.get("units"),
            metadata.get("horizon_bars"),
        )
        == (config.timeframe, "future_return", config.policy_units, config.horizon_bars)
        else "failed"
    )
    return gates


def promotion_status(gates: dict[str, str]) -> str:
    # No empty set or missing mandatory gate can pass vacuously.
    required = {
        "forecast_chronology",
        "null_evidence",
        "untouched_evaluation",
        "broker_specification",
        "forecast_units_and_horizon",
        "data_provenance",
        "evidence_identity",
    }
    return (
        "eligible_for_economic_review"
        if required <= gates.keys() and all(gates[k] == "passed" for k in required)
        else "historical_diagnostic_only"
    )
