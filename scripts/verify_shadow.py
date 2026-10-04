"""Bounded synthetic Stage17 study; never imports or connects to native MT5."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from xauusd_quant.alpha_portfolio.plan import freeze
from xauusd_quant.execution.readiness import sha256
from xauusd_quant.execution.runs import process_memory
from xauusd_quant.shadow.config import ShadowConfig
from xauusd_quant.shadow.runs import PLAN, code_identity, compare, readiness, replay

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from shadow_synth import feed, governed, tick, trend  # noqa: E402
from test_shadow_runs import run  # noqa: E402


def study(directory: Path) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=False)
    freeze(directory / "plan.json", PLAN)
    source_code = code_identity(ROOT)
    before = time.perf_counter()
    config, adapter, capture_summary = run(directory)
    replay_summary = replay(ROOT, config, directory / "CAPTURE_V001", directory / "REPLAY_V001")
    comparison = compare(directory / "CAPTURE_V001", directory / "REPLAY_V001")
    assert comparison["equal"] and not comparison["full_model_equality"]
    assert adapter.shutdown_count == 1
    pipeline, recorder = governed()
    feed(pipeline, trend())
    assert pipeline.portfolio is not None
    freeze(directory / "synthetic_pipeline_records.json", recorder.tables)
    state = pipeline.portfolio.risk_state()
    freeze(directory / "synthetic_governed_checkpoint.json", state)
    # Independent cash/entry/exit cost arithmetic from the authoritative cash flows.
    cash_change = sum(r["amount_account"] for r in recorder.tables.get("cash_flows", []))
    assert math.isclose(pipeline.portfolio.engine.cash, 10000 + cash_change, abs_tol=1e-8)
    for metric, total in pipeline.portfolio.account_totals.items():
        assert math.isclose(total, sum(a.get(metric, 0) for a in pipeline.portfolio.attribution.values()), abs_tol=1e-8)
    fills = recorder.tables.get("fills", [])
    orders = {r["order_id"]: r for r in recorder.tables.get("orders", []) if r["status"] == "submitted"}
    for fill in fills:
        order = orders[fill["order_id"]]
        assert order["arrival_utc"] < fill["timestamp_utc"] < order["expires_utc"]
        assert fill["commission_account"] == fill["quantity_lots"] * 3
    missing, missing_records = governed(diagnostics=False)
    feed(missing, trend())
    assert not missing_records.tables.get("fills")
    flat, flat_records = governed()
    feed(flat, [tick(i * 10, i + 1) for i in range(26)])
    assert not flat_records.tables.get("fills")
    report = {"stage": 17, "classification": "synthetic software validation only",
        "fixture_source_sha256": {relative: sha256(ROOT / relative) for relative in (
            "scripts/verify_shadow.py", "tests/shadow_synth.py", "tests/test_shadow_runs.py")},
        "code_identity": source_code, "capture": capture_summary, "replay": replay_summary,
        "comparison": comparison, "readiness": readiness(ROOT, ShadowConfig()),
        "synthetic_governed": {"ticks": len(trend()), "bars": pipeline.counts["bars"],
            "predictions": len(recorder.tables["predictions"]), "risk_decisions": len(recorder.tables["risk_decisions"]),
            "local_fills": len(fills), "cash_reconciled": True, "attribution_reconciled": True,
            "missing_uncertainty_blocks": True, "flat_control_no_trade": True, "broker_orders": 0},
        "resources": {"wall_seconds": time.perf_counter() - before, **process_memory()},
        "native_connectivity_completed": False, "live_observation_completed": False,
        "actual_frozen_model_equality_completed": False, "strategy_active": False,
        "stage18_ready": False, "selection_or_profitability_claim": None}
    assert code_identity(ROOT) == source_code
    assert report["resources"]["wall_seconds"] < 120
    assert int(report["resources"].get("private_bytes") or 0) < 1073741824
    freeze(directory / "verdict.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    from xauusd_quant.alpha_portfolio.plan import versioned
    versioned(args.run_id)
    if "/" in args.run_id or "\\" in args.run_id:
        raise ValueError("path-safe identity required")
    result = study(ROOT / "results/shadow/runs" / args.run_id)
    print(json.dumps({k: result[k] for k in ("synthetic_governed", "resources", "native_connectivity_completed", "stage18_ready")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
