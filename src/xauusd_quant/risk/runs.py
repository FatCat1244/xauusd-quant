"""Immutable risk plans/readiness and sequential synthetic replay; no market loader."""

from __future__ import annotations

import json
import time
from dataclasses import fields
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ..alpha_portfolio.allocation import Allocation
from ..alpha_portfolio.intents import Intent
from ..alpha_portfolio.plan import freeze, versioned
from ..alpha_portfolio.portfolio import Portfolio, WeightUpdate
from ..execution.config import ExecutionConfig, content_hash
from ..execution.engine import Quote
from ..execution.io import DiskRecorder, write_new_json
from ..execution.readiness import code_identity, sha256
from ..execution.runs import process_memory
from ..strategy_validation.runs import TrialLedger
from ..utils.config import Config
from .contracts import AccountSpec, HealthSnapshot, InstrumentSpec, RiskRequest, resolved
from .engine import TRANSITIONS
from .policy import RiskConfiguration, RiskPolicy, load_configuration
from .portfolio import RiskPortfolio

PLAN: dict[str, Any] = {
    "plan_id": "RISK_REPLAY_PLAN_V001", "stage": 16,
    "universe": "two synthetic sleeves A/B; real Stage15 eligibility empty, no promotion",
    "risk_configuration": "RISK_SYNTHETIC_V001; fixed values, no PnL tuning",
    "scenarios": ["agreement", "opposition", "pending_change", "adverse_jump", "loss_halt", "invalid_feed", "invalid_health"],
    "period": "synthetic 2021-01-04 12:00 UTC, four seconds per fixture; no reserved rows",
    "primary": "causal risk boundary, reconciliation, reservations, retained halts, unresolved exits",
    "comparisons": "same inputs Stage15 baseline versus risk-governed Stage12 execution; no winner selection",
    "sizing": "declared horizon stress, no invented stop, toward-zero rounding; not a loss guarantee",
    "acceptance": "software invariants only; real replay BLOCKED without eligible artifacts and supplied settings",
    "budget": {"trials": 14, "events_per_trial": 16, "wall_seconds": 120, "private_bytes": 1073741824},
    "history": "synthetic software evidence; earlier market history inspected and partially recorded",
    "stop": "Stage16 only; no broker, deployment, commits, pushes or Stage17",
    "source": "deterministic hand-constructed quote paths, not measured execution distributions",
}


def evidence(root: Path) -> dict[str, Any]:
    """Metadata/records only. No forecast, features, tick values or account credentials."""
    relative = [
        "results/alpha_portfolio/registries/ALPHA_REGISTRY_V003.json",
        "results/alpha_portfolio/runs/PORTFOLIO_INACTIVE_V003/verdict.json",
        "results/alpha_portfolio/runs/PORTFOLIO_SMOKE_V004/verdict.json",
        "results/alpha_portfolio/FINAL_VERIFICATION_V003.json",
        "results/robustness/runs/ROBUST_HISTORICAL_V002/verdict.json",
    ]
    sources = {name: sha256(root / name) if (root / name).is_file() else None for name in relative}
    registry_path = root / relative[0]
    registry: dict[str, Any] = {}
    if registry_path.is_file():
        envelope = json.loads(registry_path.read_text(encoding="utf-8"))
        if envelope.get("content_sha256") != content_hash(envelope["body"]):
            raise ValueError("Stage15 registry identity invalid")
        registry = envelope["body"]
    access = {entry["path"]: sha256(root / entry["path"])
              for entry in registry.get("dataset", {}).get("evaluation_history", [])}
    return {"sources": sources, "registry_id": registry.get("registry_id"),
            "eligible_alphas": registry.get("eligible_alphas", []),
            "registry_available": bool(registry), "reserved_access_logs": access,
            "missing_prerequisites": registry.get("missing_prerequisites", ["Stage15 registry unavailable"]),
            "previous_results_reproduced": False,
            "source_scope": "saved metadata and result records only; pre-risk source identities retained"}


def freeze_plan(root: Path) -> Path:
    return freeze(root / "results/risk/plans/RISK_REPLAY_PLAN_V001.json", PLAN)


def readiness(root: Path, configuration: RiskConfiguration) -> dict[str, Any]:
    audit = evidence(root)
    required = {
        "policy": [f.name for f in fields(RiskPolicy)] if configuration.policy is None else [],
        "account": [f.name for f in fields(AccountSpec)] if configuration.account is None else [],
        "instrument": [f.name for f in fields(InstrumentSpec)] if configuration.instrument is None else [],
    }
    return {"status": "BLOCKED", "configuration_state": "WARMING_UP_RECONCILIATION_REQUIRED" if configuration.configured else "UNCONFIGURED",
            "configuration": resolved(configuration), "missing_settings": required,
            "alpha_status": "NO_ELIGIBLE_ALPHAS" if not audit["eligible_alphas"] else "ELIGIBILITY_REPLAY_AUDIT_REQUIRED",
            "real_data_replay_completed": False, "strategy_active": False, "evidence": audit,
            "remaining": ["supplied instrument/account/cost/margin terms", "eligible Stage15 strategy and prior-only health provenance",
                          "Stage17 feed/shadow and asynchronous reconciliation validation", "Stage18 separately authorized demo validation"],
            "offline_software_is_live_validation": False}


def synthetic_inputs(scenario: str) -> list[Quote | Intent | WeightUpdate]:
    t = datetime(2021, 1, 4, 12, tzinfo=UTC)
    def quote(seconds: int, mid: float = 1800, valid: bool = True) -> Quote:
        at = t + timedelta(seconds=seconds)
        return Quote(at, at.replace(tzinfo=None), mid - .1, mid + .1 if valid else mid - .2, 0)
    def intent(alpha: str, seconds: float, target: float) -> Intent:
        at = t + timedelta(seconds=seconds)
        return Intent(alpha, f"{alpha}_V001", at, at, t + timedelta(seconds=30), target,
                      "active" if target else "flat", "synthetic fixture", "5m", 1)
    allocation = WeightUpdate(Allocation("EQUAL_V001", t, t - timedelta(seconds=1),
                                         {"A": .5, "B": .5}, "equal", {}))
    b = -1 if scenario == "opposition" else 1
    inputs: list[Quote | Intent | WeightUpdate] = [allocation, quote(0), intent("A", 1, 1), intent("B", 1, b)]
    if scenario == "pending_change":
        inputs += [intent("A", 1.5, -1)]
    inputs += [quote(2, 1810 if scenario == "adverse_jump" else 1800)]
    if scenario in ("loss_halt", "invalid_feed"):
        inputs += [quote(3, 1700), quote(4, 1700, scenario != "invalid_feed")]
    else:
        inputs += [intent("A", 3, 0), intent("B", 3, 0), quote(4, 1802)]
    return inputs


def fixture_health(alpha: str, now: datetime, *, valid: bool = True) -> HealthSnapshot:
    return HealthSnapshot(alpha, f"{alpha}_V001", f"MODEL_{alpha}_V001", f"FEATURES_{alpha}_V001",
        now, now, now, now + timedelta(seconds=300), .001 if valid else float("nan"),
        "log_mid_return", 300, True, True, True, True, True, True, .01, .01)


def replay(scenario: str, governed: bool, configuration: RiskConfiguration, directory: Path) -> dict[str, Any]:
    execution = ExecutionConfig(quantity_lots=.02)
    directory.mkdir(parents=True, exist_ok=False)
    with DiskRecorder(directory) as sink:
        portfolio: Portfolio = RiskPortfolio(execution, {"A": "A_V001", "B": "B_V001"}, .02, sink, configuration) if governed else Portfolio(execution, {"A": "A_V001", "B": "B_V001"}, .02, sink)
        events = synthetic_inputs(scenario)
        for event in events:
            now = event.available_utc if isinstance(event, Intent) else event.timestamp_utc
            if isinstance(portfolio, RiskPortfolio):
                for alpha in ("A", "B"):
                    portfolio.health_update(fixture_health(alpha, now, valid=scenario != "invalid_health"))
            portfolio.consume([event])
        end = datetime(2021, 1, 4, 12, tzinfo=UTC) + timedelta(seconds=4)
        summary = portfolio.finish(end)
        freeze(directory / "account_state_V001.json", portfolio.state())
        if isinstance(portfolio, RiskPortfolio):
            freeze(directory / "risk_checkpoint_V001.json", portfolio.risk_state())
            summary["risk"] = portfolio.risk.summary()
        write_new_json(directory / "summary.json", summary)
    return {"scenario": scenario, "governed": governed,
            "net_closed_pnl": summary["execution"]["closed_net_pnl_account"],
            "marked_equity": summary["execution"]["final_mark"]["equity_account"],
            "commission": summary["execution"]["total_commission_account"],
            "risk": summary.get("risk"), "result": summary["result"]}


def run(config: Config, run_id: str, mode: str, configuration_path: Path) -> dict[str, Any]:
    versioned(run_id)
    if mode not in ("readiness", "replay"):
        raise ValueError("offline readiness/replay only")
    started = time.perf_counter()
    frozen = freeze_plan(config.project_root)
    configuration = load_configuration(configuration_path)
    directory = config.project_root / "results/risk/runs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    audit = readiness(config.project_root, configuration)
    freeze(directory / "risk_configuration_V001.json", resolved(configuration))
    freeze(config.project_root / "results/risk/configurations" / f"{configuration.configuration_id}.json",
           resolved(configuration))
    freeze(directory / "risk_schema_V001.json", {name: [f.name for f in fields(cls)] for name, cls in (
        ("policy", RiskPolicy), ("account", AccountSpec), ("instrument", InstrumentSpec), ("decision_request", RiskRequest))})
    freeze(directory / "risk_state_machine_V001.json", {key: sorted(values) for key, values in TRANSITIONS.items()})
    specification = config.project_root / "docs/stage16_risk_engine.md"
    if specification.is_file():
        with (directory / "risk_specification_V001.md").open("x", encoding="utf-8") as stream:
            stream.write(specification.read_text(encoding="utf-8"))
    write_new_json(directory / "manifest.json", {"stage": 16, "mode": mode, "registered_utc": datetime.now(UTC),
        "plan_sha256": sha256(frozen), "configuration_sha256": configuration.identity,
        "code": code_identity(config.project_root), "evidence": audit["evidence"],
        "classification": "synthetic_software_validation" if mode == "replay" else "metadata_only_readiness"})
    result: dict[str, Any] = {"readiness": audit}
    if mode == "replay":
        if not configuration.configured or configuration.policy is None or configuration.policy.status != "synthetic":
            raise ValueError("replay requires explicitly synthetic fully specified configuration")
        comparisons = []
        with DiskRecorder(directory) as ledger_sink:
            ledger = TrialLedger(ledger_sink, 14)
            for scenario in PLAN["scenarios"]:
                for governed in (False, True):
                    name = f"{scenario}_{'risk' if governed else 'baseline'}"
                    def execute(scenario_name: str = scenario, use_risk: bool = governed,
                                trial_name: str = name) -> dict[str, Any]:
                        return replay(scenario_name, use_risk, configuration, directory / trial_name)
                    trial = ledger.call(name, {"scenario": scenario, "governed": governed},
                                        execute)
                    if trial is None:
                        raise RuntimeError("risk replay failed; retain ledger, do not expand")
                    comparisons.append(trial)
                    if time.perf_counter() - started > 120 or int(process_memory().get("private_bytes") or 0) > 1073741824:
                        raise RuntimeError("frozen risk replay resource budget exhausted")
            result["attempted_trials"] = ledger.attempts
        result["synthetic_comparisons"] = comparisons
    if evidence(config.project_root)["reserved_access_logs"] != audit["evidence"]["reserved_access_logs"]:
        raise RuntimeError("reserved access logs changed")
    result.update({"real_data_replay_completed": False, "strategy_active": False, "reserved_outcomes_read": False,
                   "resources": {"wall_seconds": time.perf_counter() - started, **process_memory()}, "output": str(directory)})
    write_new_json(directory / "verdict.json", result)
    return result


def cli_command(config: Config, args: Any) -> int:
    if args.command == "risk-plan":
        print(freeze_plan(config.project_root))
        return 0
    configuration = args.risk_config or config.project_root / "config/risk.yaml"
    result = run(config, args.run_id, args.command.removeprefix("risk-"), configuration)
    print(json.dumps(result, indent=2))
    return 1 if args.command == "risk-readiness" else 0
