"""Independently reconcile saved synthetic risk/account records without market access."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from xauusd_quant.execution.config import content_hash
from xauusd_quant.execution.io import write_new_json
from xauusd_quant.execution.readiness import sha256
from xauusd_quant.risk.runs import evidence

ROOT = Path(__file__).resolve().parents[1]


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.is_file() else []


def check_run(run_id: str) -> dict[str, Any]:
    path = ROOT / "results/risk/runs" / run_id
    manifest = json.loads((path / "manifest.json").read_text())
    verdict = json.loads((path / "verdict.json").read_text())
    for relative, expected in manifest["code"]["source_sha256"].items():
        assert sha256(ROOT / relative) == expected, f"source changed: {relative}"
    assert evidence(ROOT)["reserved_access_logs"] == manifest["evidence"]["reserved_access_logs"]
    assert not verdict["real_data_replay_completed"] and not verdict["strategy_active"]
    frozen = json.loads((path / "risk_configuration_V001.json").read_text())
    assert frozen["content_sha256"] == content_hash(frozen["body"])
    summaries = []
    for directory in sorted(p for p in path.iterdir() if p.is_dir()):
        summary = json.loads((directory / "summary.json").read_text())
        mark = summary["execution"]["final_mark"]
        cash = sum(r["amount_account"] for r in rows(directory / "cash_flows.jsonl"))
        assert math.isclose(mark["cash_account"], 10000 + cash, abs_tol=1e-7)
        for metric, total in summary["attribution_totals"].items():
            allocated = sum(v.get(metric, 0) for v in summary["attribution"].values())
            assert math.isclose(allocated, total, abs_tol=1e-7)
        fills = rows(directory / "fills.jsonl")
        orders = rows(directory / "orders.jsonl")
        submitted = {r["order_id"]: r for r in orders if r["status"] == "submitted"}
        risk_decisions = {r["intent_id"]: r for r in rows(directory / "risk_decisions.jsonl")}
        acks = rows(directory / "risk_execution_updates.jsonl")
        for fill in fills:
            order = submitted[fill["order_id"]]
            assert datetime.fromisoformat(order["arrival_utc"]) < datetime.fromisoformat(fill["timestamp_utc"]) < datetime.fromisoformat(order["expires_utc"])
            side = fill["ask"] if fill["direction"] > 0 else fill["bid"]
            assert math.isclose(fill["price_usd_per_ounce"], side + fill["direction"] * .02, abs_tol=1e-8)
            assert math.isclose(fill["commission_account"], fill["quantity_lots"] * 3, abs_tol=1e-8)
            if risk_decisions:
                acknowledgement = next(r for r in acks if r["order_id"] == str(fill["order_id"]) and r["status"] == "filled")
                decision = risk_decisions[acknowledgement["intent_id"]]
                assert math.isclose(abs(decision["approved_change_lots"]), fill["quantity_lots"], abs_tol=1e-9)
                assert datetime.fromisoformat(decision["received_utc"]) < datetime.fromisoformat(fill["timestamp_utc"])
                if fill["purpose"] == "entry":
                    assert decision["decision"] in ("APPROVE", "RESIZE")
                    assert decision["measurements"]["post_round_limits_valid"]
        for trade in rows(directory / "trades.jsonl"):
            gross = trade["direction"] * (trade["exit_price"] - trade["entry_price"]) * trade["quantity_lots"] * 100
            assert math.isclose(gross, trade["gross_price_pnl_account"], abs_tol=1e-7)
            assert math.isclose(trade["net_pnl_account"], gross - trade["commission_account"] - trade["financing_account"], abs_tol=1e-7)
        for frozen_path in directory.glob("*_V001.json"):
            envelope = json.loads(frozen_path.read_text())
            assert envelope["content_sha256"] == content_hash(envelope["body"]), frozen_path
        if "risk" in summary:
            checkpoint = json.loads((directory / "risk_checkpoint_V001.json").read_text())["body"]
            assert checkpoint["content_sha256"] == content_hash(checkpoint["body"])
            state = checkpoint["body"]["risk"]
            assert state["content_sha256"] == content_hash(state["body"])
            assert state["configuration_sha256"] == manifest["configuration_sha256"]
            if summary["execution"]["open_position"] is not None:
                assert summary["risk"]["actual_account"]["signed_lots"] != 0
        summaries.append({"name": directory.name, "fills": len(fills), "cash_flows": len(rows(directory / "cash_flows.jsonl")),
                          "risk_decisions": len(risk_decisions)})
    if manifest["mode"] == "replay":
        ledger = rows(path / "trial_ledger.jsonl")
        assert len([r for r in ledger if r["status"] == "attempted"]) == 14
        assert len([r for r in ledger if r["status"] == "completed"]) == 14
        assert len(summaries) == 14
    return {"run_id": run_id, "passed": True, "records": summaries,
            "reserved_access_logs_unchanged": True, "market_values_read": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--readiness", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"runs": [check_run(args.replay), check_run(args.readiness)], "scope": "saved synthetic records, current source and evidence identities"}
    write_new_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
