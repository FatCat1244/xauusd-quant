"""Verify Stage18 continuation records offline, preserving blocked conclusions.

Reads immutable run metadata, the existing hash-chained journal and the bounded
Exness capture. Never opens MT5, a model binary or historical market partitions.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from xauusd_quant.alpha_portfolio.plan import freeze
from xauusd_quant.demo.config import load_demo_config
from xauusd_quant.demo.journal import Journal
from xauusd_quant.demo.runs import account_directory
from xauusd_quant.execution.engine import MemoryRecorder
from xauusd_quant.execution.readiness import sha256
from xauusd_quant.risk.engine import RiskEngine
from xauusd_quant.risk.policy import load_configuration
from xauusd_quant.risk.runs import evidence
from xauusd_quant.shadow.config import load_shadow_config
from xauusd_quant.shadow.runs import code_identity, compare, frozen_inventory, read_frozen


def verify(root: Path, config_path: Path, output: Path) -> dict[str, Any]:
    before = evidence(root)
    config = load_demo_config(config_path)
    terminal = load_shadow_config(root / str(config.terminal_config_path))
    risk = load_configuration(root / str(config.risk_config_path))
    source = code_identity(root)
    run_id = "DEMO_NATIVE_DIAGNOSTIC_V001"
    directory = root / "results/demo/runs" / run_id
    verdict = read_frozen(directory / "verdict.json")
    provenance = read_frozen(directory / "provenance.json")
    plan = read_frozen(directory / "plan.json")
    preflight = read_frozen(root / "results/demo/runs/DEMO_CONTINUE_PREFLIGHT_V001/preflight.json")
    if (plan["plan_id"] != "DEMO_NATIVE_DIAGNOSTIC_PLAN_V001"
        or provenance["execution_identity"] != config.identity
        or provenance["risk_identity"] != risk.identity
        or provenance["code_identity"] != verdict["code_identity"]
        or provenance["code_identity"] != source
        or provenance["script_sha256"] != verdict["script_sha256"]):
        raise ValueError("continuation source, configuration or diagnostic plan mismatch")
    if (not preflight["demo_mode_verified"] or not preflight["execution_permissions_verified"]
        or not preflight["fresh_quote"] or preflight["account_has_positions"]
        or preflight["account_has_orders"]):
        raise ValueError("continuation preflight evidence missing")
    journal = Journal(account_directory(root, terminal))
    try:
        if not journal.rows or journal.rows[-1]["event"] != "checkpoint":
            raise ValueError("intact final checkpoint required")
        saved = journal.rows[-1]["data"]
        RiskEngine.restore(risk, MemoryRecorder(), saved["risk"])
        if (saved["config_sha256"] != config.identity or saved["terminal_sha256"] != terminal.identity
            or saved["code_identity"] != source or saved["run_id"] != run_id):
            raise ValueError("final checkpoint incompatible with this observation")
        selected = [r for r in journal.rows if r["data"].get("run_id") == run_id]
        entry = saved["intents"][f"{run_id}:entry"]
        decision = entry["decision"]
        if (entry["state"] != "REJECTED" or entry.get("check") or entry.get("submission_utc")
            or decision["decision"] != "REJECT" or decision["approved_change_lots"] != 0
            or set(decision["rules"]) != {"SESSION_CLOSED", "OVERNIGHT_RESTRICTION"}
            or any(r["event"] in {"broker_precheck", "broker_response"} for r in selected)
            or any(v.get("submission_utc") for r in journal.rows if r["event"] == "checkpoint"
                   for v in r["data"].get("intents", {}).values())
            or verdict["failure"] is not None or verdict["submissions"] != 0
            or verdict["prechecks"] != 0 or verdict["verified_deals"] != 0
            or not verdict["verified_flat"] or verdict["reservations"] != 0
            or verdict["unresolved_intents"] or saved["halts"]):
            raise ValueError("session rejection must remain a zero-order blocked outcome")
        rearms = [r for r in selected if r["event"] == "execution_rearm"]
        reconciliations = [r for r in selected if r["event"] == "reconciliation"]
        if (len(rearms) != 1 or not rearms[0]["data"].get("recovery_evidence")
            or not reconciliations or not reconciliations[-1]["data"]["verified_flat"]
            or reconciliations[-1]["data"]["balance_discrepancy"] != 0):
            raise ValueError("explicit recovery and flat reconciliation evidence incomplete")
        rearm_index = journal.rows.index(rearms[0])
        prior_risk = [r["data"]["risk"]["body"] for r in journal.rows[:rearm_index]
                      if r["event"] == "checkpoint"][-1]
        rearmed_risk = next(r["data"]["risk"]["body"] for r in journal.rows[rearm_index + 1:]
                           if r["event"] == "checkpoint")
        retained_fields = ("high_water", "daily_start", "daily_key", "daily_loss", "drawdown",
                           "order_times", "turnover", "decisions", "terminal", "acknowledgements")
        if any(prior_risk[k] != rearmed_risk[k] for k in retained_fields):
            raise ValueError("explicit rearm changed retained financial or execution history")
        prior_failure = read_frozen(root / "results/demo/runs/DEMO_FRESH_QUOTE_SMOKE_V001/verdict.json")
        if prior_failure["failure"] != "ValueError" or prior_failure["submissions"] != 0:
            raise ValueError("original failed native attempt must be preserved")
        journal_identity = sha256(journal.path)
    finally:
        journal.close()
    capture = root / "results/shadow/runs/EXNESS_CONTINUE_CAPTURE_V001"
    replay = root / "results/shadow/runs/EXNESS_CONTINUE_REPLAY_V001"
    equality = root / "results/shadow/runs/EXNESS_CONTINUE_EQUALITY_V001/comparison.json"
    summary = read_frozen(capture / "summary.json")
    manifest = read_frozen(capture / "manifest.json")
    for name, digest in manifest["files"].items():
        if Path(name).name != name or sha256(capture / name) != digest:
            raise ValueError("capture manifest changed")
    comparison = compare(capture, replay)
    if comparison != read_frozen(equality) or summary["code_identity"] != source:
        raise ValueError("recomputed observed-table equality or source identity differs")
    bars: Counter[str] = Counter()
    all_backfill = True
    partial = 0
    ticks = 0
    with (capture / "records.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record["table"] == "ticks":
                ticks += 1
            elif record["table"] == "bars":
                row = record["row"]
                bars[row["timeframe"]] += 1
                all_backfill &= row["backfill"]
                partial += int("PARTIAL_STARTUP_BAR" in row["reasons"])
    if ticks != summary["accepted_committed_ticks"] or sum(bars.values()) != summary["bars"]:
        raise ValueError("capture counts do not reconcile")
    registry = read_frozen(root / "results/alpha_portfolio/registries/ALPHA_REGISTRY_V003.json")
    statuses = dict(Counter(c["eligibility"] for c in registry["candidates"]))
    models = frozen_inventory(root)
    after = evidence(root)
    if before != after:
        raise ValueError("scientific metadata or reserved access logs changed during verification")
    report = {"verification_id": output.stem,
        "status": "RISK_BLOCKED_NO_ORDERS", "original_verdict_preserved": verdict["status"],
        "risk_rules": decision["rules"], "decision_utc": decision["decision_utc"],
        "native_source_identity": source, "native_script_identity": provenance["script_sha256"],
        "current_operator_script_identity": sha256(root / "scripts/run_unsubmitted_demo_smoke.py"),
        "completed_prechecks": 0, "submissions": 0, "entry_deals": 0, "close_deals": 0,
        "verified_flat_at_run_end": True, "cash_discrepancy": 0, "reservations": 0,
        "unresolved_intents": [], "current_flatness": "historical snapshot only; not a new broker read",
        "journal_identity": journal_identity, "risk_history_preserved": True,
        "original_native_failure": "UNCLASSIFIED_VALUE_ERROR_NOT_REACHED_BY_THIS_DIAGNOSTIC",
        "capture": {"fresh_ticks": summary["fresh_live_ticks"], "backfill_ticks": summary["backfill_ticks"],
            "total_ticks": ticks, "bars_by_timeframe": dict(bars), "all_bars_backfill": all_backfill,
            "partial_startup_bars": partial, "wall_seconds": summary["wall_seconds"],
            "memory": summary["memory"], "halts": summary["halts"],
            "records_sha256": sha256(capture / "records.jsonl")},
        "observed_equality": comparison, "features": "monitoring ret_1 only; no fitted model bindings",
        "alpha_statuses": statuses, "eligible_alphas": before["eligible_alphas"],
        "frozen_model_specs": models["models"], "frozen_ensemble_specs": models["ensembles"],
        "scientific_evidence": before, "reserved_log_check_scope": "before/after this metadata verification only",
        "missing_prerequisites": before["missing_prerequisites"],
        "strategy_active": False, "strategy_risk_configuration": "unconfigured; supplied risk is SMOKE only",
        "stage19_started": False, "verifier_sha256": sha256(Path(__file__))}
    freeze(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(Path(__file__).resolve().parents[1], args.demo_config, args.output)
    print(report["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
