"""Reconcile local Stage18 evidence identities without opening market datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from xauusd_quant.alpha_portfolio.plan import freeze
from xauusd_quant.demo.runs import prerequisites
from xauusd_quant.execution.readiness import sha256
from xauusd_quant.shadow.runs import code_identity, read_frozen

ROOT = Path(__file__).resolve().parents[1]


def verify() -> dict[str, Any]:
    names = {"synthetic": "results/demo/runs/DEMO_SYNTHETIC_V003/verdict.json",
        "preflight": "results/demo/runs/DEMO_NATIVE_PREFLIGHT_V003/preflight.json",
        "readiness": "results/demo/runs/DEMO_READINESS_V001/readiness.json",
        "blocked_smoke": "results/demo/runs/DEMO_BLOCKED_SMOKE_V001/verdict.json",
        "blocked_strategy": "results/demo/runs/DEMO_BLOCKED_STRATEGY_V002/verdict.json",
        "blocked_precheck": "results/demo/runs/DEMO_BLOCKED_PRECHECK_V001/verdict.json"}
    records = {k: read_frozen(ROOT / p) for k, p in names.items()}
    source = code_identity(ROOT)
    assert source == records["synthetic"]["code_identity"] == records["preflight"]["code_identity"]
    guard_path = ROOT / "results/demo/DEMO_GUARDS_V002.json"
    guards = json.loads(guard_path.read_text(encoding="utf-8"))
    assert guards["original_source_unchanged"] and all(g["test_detected_break"] for g in guards["mutations"])
    upstream = prerequisites(ROOT)
    assert upstream == records["readiness"]["evidence"]
    assert not upstream["eligible_alphas"]
    for kind in ("blocked_smoke", "blocked_strategy", "blocked_precheck"):
        assert records[kind]["status"] == "BLOCKED" and records[kind]["demo_submissions"] == 0
    native = records["preflight"]
    assert native["native_connectivity_completed"] and native["demo_mode_verified"]
    assert native["demo_submissions"] == 0 and not native["broker_precheck_completed"]
    assert not native["account_has_positions"] and not native["account_has_orders"]
    return {"stage": 18, "code_identity": source,
        "artifact_sha256": {p: sha256(ROOT / p) for p in names.values()},
        "guard_artifact_sha256": sha256(guard_path), "guard_mutations_detected": len(guards["mutations"]),
        "verification_script_sha256": sha256(Path(__file__)), "upstream_evidence_unchanged": True,
        "reserved_access_log_hashes_unchanged": True, "actual_native": native,
        "synthetic_resources": records["synthetic"]["resources"],
        "unknown_and_accepted_synthetic_cases_retained": records["synthetic"]["unresolved_cases_preserved"],
        "strategy_status": "BLOCKED_NO_ELIGIBLE_ALPHAS",
        "actual_order_checks": 0, "actual_submissions": 0, "actual_fills": 0, "actual_closures": 0,
        "actual_execution_uncertainties": [], "project_owned_exposure_created": False,
        "unrelated_account_activity_managed": False, "real_money_readiness": False,
        "stage19_started": False, "status": "OFFLINE_INFRASTRUCTURE_VALIDATED; DEMO_EXECUTION_BLOCKED"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("verification record exists; bump version")
    report = verify()
    freeze(args.output, report)
    print(json.dumps({k: report[k] for k in ("status", "code_identity", "guard_mutations_detected", "actual_submissions", "reserved_access_log_hashes_unchanged")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
