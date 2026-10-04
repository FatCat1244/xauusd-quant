"""Verify Stage14 saved identities, ledgers, timing/cost records and unchanged access history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from xauusd_quant.execution.config import content_hash
from xauusd_quant.execution.io import write_new_json
from xauusd_quant.execution.readiness import sha256
from xauusd_quant.robustness.inventory import ledger_view
from xauusd_quant.robustness.reports import read_records


def verify(root: Path, run_ids: list[str]) -> dict[str, Any]:
    results = []
    for run_id in run_ids:
        directory = root / "results/robustness/runs" / run_id
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        verdict = json.loads((directory / "verdict.json").read_text(encoding="utf-8"))
        plan_file = Path(manifest["plan"])
        if sha256(plan_file) != manifest["plan_sha256"]:
            raise AssertionError("saved plan changed")
        if (
            manifest.get("source_index_sha256") is not None
            and sha256(plan_file.with_name(f"{plan_file.stem}_SOURCES.json"))
            != manifest["source_index_sha256"]
        ):
            raise AssertionError("saved source identity index changed")
        frozen = json.loads(plan_file.read_text(encoding="utf-8"))
        if content_hash(frozen["plan"]) != frozen["content_sha256"]:
            raise AssertionError("saved plan content hash invalid")
        ledger_file = directory / "trial_ledger.jsonl"
        ledger = (
            ledger_view(ledger_file) if ledger_file.exists() else {"attempts": 0, "statuses": {}}
        )
        if (
            ledger["attempts"] != verdict["attempted_trials"]
            or ledger["attempts"] > manifest["resolved_plan"]["trial_budget"]
        ):
            raise AssertionError("saved trial budget/attempt count mismatch")
        if ledger["statuses"].get("interrupted_or_abandoned", 0):
            raise AssertionError("unfinished attempt cannot pass saved verification")
        counts = {"orders": 0, "fills": 0, "trades": 0}
        for file in directory.glob("*/summary.json"):
            summary = json.loads(file.read_text(encoding="utf-8"))
            if abs(summary["execution"]["final_mark"]["reconciliation_error_account"]) > 1e-7:
                raise AssertionError("saved cash/position accounting mismatch")
            orders = (
                read_records(file.parent / "orders.jsonl", 25000)
                if (file.parent / "orders.jsonl").exists()
                else []
            )
            submitted = {r["order_id"]: r for r in orders if r["status"] == "submitted"}
            fills = (
                read_records(file.parent / "fills.jsonl", 25000)
                if (file.parent / "fills.jsonl").exists()
                else []
            )
            trades = (
                read_records(file.parent / "trades.jsonl", 25000)
                if (file.parent / "trades.jsonl").exists()
                else []
            )
            for fill in fills:
                order = submitted[fill["order_id"]]
                if not order["arrival_utc"] < fill["timestamp_utc"] < order["expires_utc"]:
                    raise AssertionError("saved execution violates causal arrival/expiry")
            for trade in trades:
                net = (
                    trade["gross_price_pnl_account"]
                    - trade["commission_account"]
                    - trade["financing_account"]
                )
                gross = (
                    trade["matched_midpoint_pnl_account"]
                    - trade["spread_cost_account"]
                    - trade["slippage_account"]
                )
                if (
                    abs(net - trade["net_pnl_account"]) > 1e-7
                    or abs(gross - trade["gross_price_pnl_account"]) > 1e-7
                ):
                    raise AssertionError("saved cost decomposition double-counted or inconsistent")
            for key, rows in (("orders", orders), ("fills", fills), ("trades", trades)):
                counts[key] += len(rows)
        inventory = directory / "inventory.json"
        access_unchanged = None
        if inventory.exists():
            metadata = json.loads(inventory.read_text(encoding="utf-8"))["metadata"]
            access_unchanged = all(
                sha256(root / r["path"]) == r["sha256"] for r in metadata["evaluation_history"]
            )
            if not access_unchanged:
                raise AssertionError("reserved-access history changed during Stage14")
        results.append(
            {
                "run": run_id,
                "verified": True,
                "attempts": ledger["attempts"],
                "statuses": ledger["statuses"],
                "engine_records": counts,
                "reserved_access_logs_unchanged": access_unchanged,
                "source_matches_current": all(
                    sha256(root / name) == digest
                    for name, digest in manifest["code"]["source_sha256"].items()
                ),
                "artifact_hashes": {
                    str(file.relative_to(directory)): sha256(file)
                    for file in directory.glob("*.json")
                },
            }
        )
    return {"runs": results, "scope": "saved audit streams only; no new market/reserved outcomes"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_new_json(args.output, verify(Path(__file__).resolve().parents[1], args.run_id))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
