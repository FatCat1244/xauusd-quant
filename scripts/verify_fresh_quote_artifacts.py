"""Verify the preserved failed native smoke evidence without connecting to MT5."""
from __future__ import annotations

import argparse
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
from xauusd_quant.shadow.config import load_shadow_config
from xauusd_quant.shadow.runs import code_identity, read_frozen


def verify(root: Path, config_path: Path, output: Path) -> dict[str, Any]:
    config = load_demo_config(config_path)
    terminal = load_shadow_config(root / str(config.terminal_config_path))
    risk = load_configuration(root / str(config.risk_config_path))
    run_id = "DEMO_FRESH_QUOTE_SMOKE_V001"
    directory = root / "results/demo/runs" / run_id
    verdict = read_frozen(directory / "verdict.json")
    provenance = read_frozen(directory / "provenance.json")
    preflight = read_frozen(root / "results/demo/runs/DEMO_FRESH_QUOTE_PREFLIGHT_V001/preflight.json")
    if not (preflight["demo_mode_verified"] and preflight["execution_permissions_verified"]
            and preflight["fresh_quote"] and not preflight["account_has_positions"]
            and not preflight["account_has_orders"]):
        raise ValueError("preserved preflight must match the documented account gates")
    journal = Journal(account_directory(root, terminal))
    try:
        if not journal.rows or journal.rows[-1]["event"] != "checkpoint":
            raise ValueError("intact final checkpoint required")
        saved = journal.rows[-1]["data"]
        RiskEngine.restore(risk, MemoryRecorder(), saved["risk"])
        if saved["config_sha256"] != config.identity or saved["terminal_sha256"] != terminal.identity:
            raise ValueError("unchanged configurations required for evidence verification")
        selected = [r for r in journal.rows if r["data"].get("run_id") == run_id]
        intents = [v for v in saved["intents"].values() if v.get("run_id") == run_id]
        # The verdict's zero submissions must agree with the full durable history,
        # not merely the last broker snapshot or a terminal rejected status.
        if (any(v.get("submission_utc") for r in selected if r["event"] == "checkpoint"
                for v in r["data"]["intents"].values())
            or any(r["event"] in {"broker_response", "broker_precheck"} for r in selected)):
            raise ValueError("native evidence no longer matches the failed precheck-only attempt")
        if (verdict["submissions"] != 0 or verdict["verified_deals"] != 0
            or not verdict["verified_flat"] or verdict["reservations"] != 0
            or verdict["status"] != "NO_VERIFIED_LIFECYCLE"
            or saved["halts"] != ["EXPLICIT_REARM_SMOKE_ABORTED"]):
            raise ValueError("failed lifecycle/retained halt must not be promoted or erased")
        migrations = [r for r in selected if r["event"] == "execution_code_migration"]
        quote_checks = [r for r in selected if r["event"] == "quote_revalidation"]
        reconciliations = [r for r in selected if r["event"] == "reconciliation"]
        if (len(migrations) != 1 or len(quote_checks) != 1 or len(intents) != 1
            or saved["code_identity"] != provenance["code_identity"]
            or verdict["code_identity"] != provenance["code_identity"]
            or verdict["script_sha256"] != provenance["script_sha256"]
            or migrations[0]["data"]["new_code_identity"] != provenance["code_identity"]
            or quote_checks[0]["data"]["decision"]["decision"] != "APPROVE"
            or not reconciliations or not reconciliations[-1]["data"]["verified_flat"]
            or reconciliations[-1]["data"]["balance_discrepancy"] != 0
            or any(v["state"] != "REJECTED" for v in intents)):
            raise ValueError("migration, fresh risk and flat accounting evidence incomplete")
        report = {"verification_id": "DEMO_FRESH_QUOTE_VERIFICATION_V001",
            "run_id": run_id, "native_run_code_identity": provenance["code_identity"],
            "native_run_script_sha256": provenance["script_sha256"],
            "current_code_identity": code_identity(root),
            "current_script_sha256": sha256(root / "scripts/run_unsubmitted_demo_smoke.py"),
            "journal_sha256": sha256(journal.path), "intact_journal": True,
            "preflight_gates_verified": True, "explicit_migration_audited": True,
            "parent_fresh_risk_decisions": len(quote_checks),
            "completed_broker_prechecks": 0, "native_order_check_invocations": None,
            "broker_changing_submissions": 0, "entry_deals": 0, "close_deals": 0,
            "verified_flat": True, "cash_discrepancy": 0, "halts_preserved": True,
            "native_failure_reason": "UNCLASSIFIED_NATIVE_WORKER_VALUE_ERROR",
            "post_run_clock_fix_native_validated": False, "strategy_active": False,
            "status": "FAILED_NATIVE_LIFECYCLE_PRESERVED", "stage19_started": False}
        freeze(output, report)
        return report
    finally:
        journal.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = verify(root, args.demo_config, args.output)
    print(report["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
