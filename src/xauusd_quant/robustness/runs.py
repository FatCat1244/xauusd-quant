"""Immutable Stage14 entry points; metadata audit precedes bounded saved-outcome analysis."""

from __future__ import annotations

import json
import re
import time
from datetime import UTC, datetime
from importlib.metadata import version
from typing import Any

from ..execution.io import DiskRecorder, write_new_json
from ..execution.readiness import code_identity, sha256
from ..execution.runs import process_memory
from ..strategy_validation.runs import TrialLedger
from ..utils.config import Config
from .inventory import build_inventory
from .plan import RobustnessPlan, freeze_plan, load_plan


def run(config: Config, plan: RobustnessPlan, run_id: str, mode: str) -> dict[str, Any]:
    started = time.perf_counter()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*_V\d{3,}", run_id):
        raise ValueError("versioned run id required")
    frozen = freeze_plan(config.project_root, plan)
    directory = config.project_root / "results/robustness/runs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    write_new_json(
        directory / "manifest.json",
        {
            "stage": 14,
            "mode": mode,
            "registered_utc": datetime.now(UTC),
            "plan": str(frozen),
            "plan_sha256": sha256(frozen),
            "source_index_sha256": sha256(frozen.with_name(f"{plan.plan_id}_SOURCES.json")),
            "resolved_plan": plan.resolved(),
            "code": code_identity(config.project_root),
            "dependencies": {
                name: version(name) for name in ("numpy", "polars", "pyarrow", "scipy", "PyYAML")
            },
            "data_configuration": config.to_dict(),
            "classification": "software_correctness" if mode == "smoke" else plan.classification,
        },
    )

    def check_budget() -> None:
        memory = process_memory()
        if (
            time.perf_counter() - started > plan.wall_budget_seconds
            or int(memory.get("private_bytes") or 0) > plan.private_memory_budget_bytes
        ):
            raise RuntimeError("registered resource budget exhausted; no expansion")

    result: dict[str, Any] = {"output": str(directory)}
    with DiskRecorder(directory) as sink:
        ledger = TrialLedger(sink, plan.trial_budget)
        if mode in ("inventory", "evaluate"):
            evidence = build_inventory(config)
            write_new_json(directory / "inventory.json", evidence)
            check_budget()
            result["selection_exposure"] = evidence["selection_exposure_quantification"]
        if mode == "evaluate":
            from .reports import historical_reports

            result.update(historical_reports(config, plan, directory, ledger))
            check_budget()
        if mode == "smoke":
            from .studies import synthetic_studies

            result.update(synthetic_studies(plan, directory, ledger, check_budget))
        result["attempted_trials"] = ledger.attempts
    result["resources"] = {"wall_seconds": time.perf_counter() - started, **process_memory()}
    write_new_json(directory / "verdict.json", result)
    return result


def cli_command(config: Config, args: Any) -> int:
    plan = load_plan(args.plan or config.project_root / "config/robustness.yaml")
    if args.command == "robustness-plan":
        print(freeze_plan(config.project_root, plan))
        return 0
    mode = args.command.removeprefix("robustness-")
    result = run(config, plan, args.run_id, mode)
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in ("output", "candidate_verdicts", "attempted_trials", "resources")
            },
            indent=2,
        )
    )
    return 0
