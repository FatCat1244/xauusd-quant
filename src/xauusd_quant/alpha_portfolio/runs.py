"""Versioned offline entry points; empty evidence yields an inactive shared account."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from ..execution.config import load_execution_config
from ..execution.io import DiskRecorder, write_new_json
from ..execution.readiness import code_identity, sha256
from ..execution.runs import process_memory
from ..strategy_validation.runs import TrialLedger
from ..utils.config import Config
from .allocation import fit
from .plan import PortfolioPlan, freeze, freeze_plan, load_plan, versioned
from .portfolio import Portfolio
from .registry import Alpha, freeze_registry


def universe_at(candidates: list[Alpha], cutoff: datetime) -> tuple[Alpha, ...]:
    return tuple(c for c in candidates if c.eligible_at(cutoff))


def access_history(root: Path, registry: dict[str, Any]) -> dict[str, str]:
    return {
        entry["path"]: sha256(root / entry["path"])
        for entry in registry["dataset"]["evaluation_history"]
    }


def run(config: Config, plan: PortfolioPlan, run_id: str, mode: str) -> dict[str, Any]:
    versioned(run_id)
    if mode not in ("smoke", "evaluate"):
        raise ValueError("declared smoke/evaluate modes only")
    started = time.perf_counter()
    frozen = freeze_plan(config.project_root, plan)
    registry_path = freeze_registry(config.project_root, plan)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))["body"]
    logs = access_history(config.project_root, registry)
    path = config.project_root / "results/alpha_portfolio/runs" / run_id
    path.mkdir(parents=True, exist_ok=False)
    execution = load_execution_config(config.project_root / "config/execution.yaml")
    write_new_json(
        path / "manifest.json",
        {
            "stage": 15,
            "mode": mode,
            "registered_utc": datetime.now(UTC),
            "schema_version": "PORTFOLIO_RUN_V001",
            "plan_sha256": sha256(frozen),
            "registry_sha256": sha256(registry_path),
            "resolved_plan": plan.resolved(),
            "code": code_identity(config.project_root),
            "dependencies": {
                name: version(name) for name in ("numpy", "polars", "pyarrow", "PyYAML")
            },
            "execution": execution.resolved(),
            "data_configuration": config.to_dict(),
            "reserved_access_logs": logs,
            "classification": "software_correctness"
            if mode == "smoke"
            else "retrospective_previously_inspected",
        },
    )
    write_new_json(
        path / "exposure_contract_V001.json",
        {
            "units": "signed_budget_fraction",
            "instrument": "XAUUSD",
            "fields": [
                "alpha_id",
                "specification_id",
                "decision_utc",
                "available_utc",
                "valid_until_utc",
                "target",
                "state",
                "rationale",
                "timeframe",
                "horizon_bars",
            ],
            "invalid_or_expired": "zero contribution; committed actual exposure still awaits Stage12 exit",
            "budget_lots": plan.budget_lots,
            "normalization": "fixed sign; no full-history scaling",
        },
    )

    def check_budget() -> None:
        memory = process_memory()
        if (
            time.perf_counter() - started > plan.wall_budget_seconds
            or int(memory.get("private_bytes") or 0) > plan.private_memory_budget_bytes
        ):
            raise RuntimeError("frozen resource budget exhausted; do not expand")

    with DiskRecorder(path) as sink:
        ledger = TrialLedger(sink, plan.trial_budget)
        if mode == "smoke":
            from .studies import synthetic_studies

            result = synthetic_studies(plan, path, ledger, check_budget)
        else:
            # No data loader is invoked: Stage14 gave no scientific eligibility.
            if registry["eligible_alphas"]:
                raise ValueError(
                    "real fold-local economic eligibility/input audit required before market evaluation"
                )

            def inactive() -> dict[str, Any]:
                portfolio = Portfolio(execution, {}, plan.budget_lots, sink)
                allocation = fit(
                    (), "equal", datetime(2021, 6, 1, tzinfo=UTC), [], [], np.empty((0, 0))
                )
                freeze(path / "allocation_V001.json", allocation.resolved())
                freeze(path / "normalization_V001.json", {"method": "fixed sign", "active": False})
                report = portfolio.finish(datetime(2021, 6, 1, tzinfo=UTC))
                write_new_json(path / "shared_account_V001.json", report)
                return {
                    "result": "NO_ELIGIBLE_ALPHAS",
                    "active": False,
                    "market_candidates_eligible": [],
                    "market_study_executed": False,
                    "economic_evidence": "unavailable, not a zero-return trading study",
                    "missing_prerequisites": registry["missing_prerequisites"],
                }

            result = ledger.call(
                "inactive_empty_universe", {"registry": plan.registry_id}, inactive
            )
            if result is None:
                raise RuntimeError("inactive-account verification failed")
        check_budget()
        result["attempted_trials"] = ledger.attempts
    if access_history(config.project_root, registry) != logs:
        raise RuntimeError("reserved-period access history changed")
    result.update(
        {
            "resources": {"wall_seconds": time.perf_counter() - started, **process_memory()},
            "software_status": "completed",
            "reserved_outcomes_read": False,
            "output": str(path),
            "selection_exposure": "partially recorded; historical discovery uncertainty unknown",
        }
    )
    write_new_json(path / "verdict.json", result)
    specification = config.project_root / "docs/stage15_strategy_specification.md"
    if specification.is_file():
        with (path / "strategy_specification_V001.md").open("x", encoding="utf-8") as stream:
            stream.write(specification.read_text(encoding="utf-8"))
    return result


def cli_command(config: Config, args: Any) -> int:
    plan = load_plan(args.plan or config.project_root / "config/alpha_portfolio.yaml")
    if args.command == "portfolio-plan":
        print(freeze_plan(config.project_root, plan))
        return 0
    if args.command == "alpha-registry":
        print(freeze_registry(config.project_root, plan))
        return 0
    result = run(config, plan, args.run_id, args.command.removeprefix("portfolio-"))
    print(json.dumps(result, indent=2))
    return 0
