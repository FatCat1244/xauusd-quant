"""Sequential bounded fake-broker lifecycle studies, never native MT5."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

from xauusd_quant.alpha_portfolio.plan import freeze, versioned
from xauusd_quant.demo.runs import PLAN
from xauusd_quant.execution.readiness import sha256
from xauusd_quant.execution.runs import process_memory
from xauusd_quant.shadow.runs import code_identity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from demo_synth import create, propose  # noqa: E402

CASES = ("full", "partial", "timeout_after_fill", "accepted", "none_without_fill", "reject")


def study(directory: Path) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=False)
    freeze(directory / "plan.json", {**PLAN, "synthetic_cases": CASES,
        "budget": "one representative case first; then five fixed cases sequentially; <=120s and <=1GiB private memory",
        "selection": "no search, seeds or outcome-based change; deterministic known quotes/deals/cash",
        "synthetic_exit": "one close after verified fill; held open/uncertain cases retained"})
    source = code_identity(ROOT)
    started = time.perf_counter()
    results = []
    representative = None
    for mode in CASES:
        case_start = time.perf_counter()
        (directory / mode).mkdir()
        c, b, journal = create(directory / mode, mode=mode, code=source)
        try:
            entry = propose(c, b)
            c.submit(entry["intent_id"], b.snapshot(), b.now)
            c.reconcile(b.snapshot(), b.now)
            before_close = c.summary()
            # Only verified fills permit a closing proposal. Unknowns remain unknown.
            if before_close["verified_entry_deals"]:
                b.mode = "full"
                b.now += timedelta(seconds=1)
                b.bid += 2
                b.ask += 2
                close = propose(c, b, close=True)
                c.submit(close["intent_id"], b.snapshot(), b.now)
                c.reconcile(b.snapshot(), b.now)
                volume = .01 if mode == "partial" else .02
                expected = (1801.9 - 1800.1) * volume * 100 - 2 * volume * 3
                assert math.isclose(b.balance - 10000, expected, abs_tol=1e-8)
                assert math.isclose(c.summary()["broker_reported_cash"], expected, abs_tol=1e-8)
                assert c.summary()["verified_flat"]
            else:
                assert not b.deals
                assert len(b.sent) == 1
            if mode in ("accepted", "none_without_fill"):
                assert c.risk.state.reservations and not c.summary()["verified_flat"]
            result = {"case": mode, "before_close": before_close, "final": c.summary(),
                "actual_native_submissions": 0, "fake_requests": b.sent, "fake_deals": b.deals,
                "configuration_sha256": c.config.identity, "risk_configuration_sha256": c.risk.configuration.identity,
                "resources": {"wall_seconds": time.perf_counter() - case_start, **process_memory()}}
            freeze(directory / mode / "summary.json", result)
            results.append(result)
            if representative is None:
                representative = result["resources"]
                freeze(directory / "representative_resources.json", representative)
                assert representative["wall_seconds"] < 30
                assert int(representative.get("private_bytes") or 0) < 1073741824
        except Exception:
            freeze(directory / mode / "failure.json", {"status": "FAILED_SYNTHETIC_CHECK", "code_identity": source})
            raise
        finally:
            journal.close()
    assert code_identity(ROOT) == source
    report = {"stage": 18, "status": "OFFLINE_INFRASTRUCTURE_VALIDATED",
        "classification": "synthetic lifecycle/accounting only; no profitability or strategy claim",
        "code_identity": source,
        "fixture_source_sha256": {p: sha256(ROOT / p) for p in ("scripts/verify_demo.py", "tests/demo_synth.py")},
        "cases": [{"case": r["case"], **r["final"]} for r in results],
        "representative_resources": representative,
        "resources": {"wall_seconds": time.perf_counter() - started, **process_memory()},
        "unresolved_cases_preserved": ["accepted", "none_without_fill"],
        "actual_broker_submissions": 0, "strategy_active": False, "stage19_started": False}
    assert report["resources"]["wall_seconds"] < 120
    assert int(report["resources"].get("private_bytes") or 0) < 1073741824
    freeze(directory / "verdict.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    versioned(args.run_id)
    if "/" in args.run_id or "\\" in args.run_id:
        raise ValueError("path-safe identity required")
    report = study(ROOT / "results/demo/runs" / args.run_id)
    print(json.dumps({k: report[k] for k in ("status", "resources", "unresolved_cases_preserved", "actual_broker_submissions")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
