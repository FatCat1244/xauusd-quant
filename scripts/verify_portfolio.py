"""Independently reconcile saved Stage15 records, identities and matched cost stress."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from xauusd_quant.execution.config import content_hash
from xauusd_quant.execution.io import parse_utc, write_new_json
from xauusd_quant.execution.readiness import code_identity, sha256

ROOT = Path(__file__).resolve().parents[1]


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(directory: Path, table: str) -> list[dict[str, Any]]:
    file = directory / f"{table}.jsonl"
    return (
        [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines()]
        if file.is_file()
        else []
    )


def close(a: float, b: float) -> None:
    if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-7):
        raise AssertionError(f"reconciliation failed: {a} != {b}")


def verify_account(directory: Path, report: dict[str, Any]) -> dict[str, int]:
    execution = report["execution"]
    submitted = {
        row["order_id"]: row for row in rows(directory, "orders") if row["status"] == "submitted"
    }
    fills = rows(directory, "fills")
    for fill in fills:
        order = submitted[fill["order_id"]]
        assert (
            parse_utc(order["created_utc"])
            <= parse_utc(order["arrival_utc"])
            < parse_utc(fill["timestamp_utc"])
            < parse_utc(order["expires_utc"])
        )
        assert 0 < fill["bid"] <= fill["ask"]
        side = fill["ask"] if fill["direction"] == 1 else fill["bid"]
        assert fill["direction"] * (fill["price_usd_per_ounce"] - side) >= -1e-10
    flows = rows(directory, "cash_flows")
    close(
        10000 + sum(row["amount_account"] for row in flows), execution["final_mark"]["cash_account"]
    )
    close(sum(fill["commission_account"] for fill in fills), execution["total_commission_account"])
    for trade in rows(directory, "trades"):
        close(
            trade["matched_midpoint_pnl_account"]
            - trade["spread_cost_account"]
            - trade["slippage_account"],
            trade["gross_price_pnl_account"],
        )
        close(
            trade["gross_price_pnl_account"]
            - trade["commission_account"]
            - trade["financing_account"],
            trade["net_pnl_account"],
        )
    for metric, total in report["attribution_totals"].items():
        close(sum(v.get(metric, 0) for v in report["attribution"].values()), total)
        close(
            sum(
                row["amount_account"]
                for row in rows(directory, "sleeve_cash_attribution")
                if row["metric"] == metric
            ),
            total,
        )
    for row in rows(directory, "sleeve_equity_attribution"):
        if row["portfolio_equity_account"] is not None:
            close(sum(row["pnl"].values()), row["portfolio_equity_account"] - 10000)
    for row in rows(directory, "allocation_updates"):
        assert parse_utc(row["information_as_of_utc"]) < parse_utc(row["effective_utc"])
    return {
        "fills": len(fills),
        "cash_flows": len(flows),
        "closed_trades": len(rows(directory, "trades")),
    }


def verify_run(run_id: str) -> dict[str, Any]:
    path = ROOT / "results/alpha_portfolio/runs" / run_id
    manifest, verdict = read(path / "manifest.json"), read(path / "verdict.json")
    assert manifest["code"]["source_identity"] == code_identity(ROOT)["source_identity"]
    plan = manifest["resolved_plan"]
    for kind, key, folder in (
        ("plan_sha256", "plan_id", "plans"),
        ("registry_sha256", "registry_id", "registries"),
    ):
        file = ROOT / "results/alpha_portfolio" / folder / f"{plan[key]}.json"
        assert sha256(file) == manifest[kind]
        artifact = read(file)
        assert content_hash(artifact["body"]) == artifact["content_sha256"]
    registry = read(ROOT / "results/alpha_portfolio/registries" / f"{plan['registry_id']}.json")[
        "body"
    ]
    for relative, digest in registry["evidence_files"].items():
        assert sha256(ROOT / relative) == digest
    processed = Path(manifest["data_configuration"]["processed_data_path"])
    assert sha256(processed / "_manifest.json") == registry["dataset"]["manifest_sha256"]
    for relative, digest in manifest["reserved_access_logs"].items():
        assert sha256(ROOT / relative) == digest
    ledger = rows(path, "trial_ledger")
    attempts = {r["trial"] for r in ledger if r["status"] == "attempted"}
    completed = {r["trial"] for r in ledger if r["status"] == "completed"}
    assert attempts == completed and len(attempts) == verdict["attempted_trials"]
    assert verdict["result"] == "NO_ELIGIBLE_ALPHAS" and not verdict["reserved_outcomes_read"]
    totals = {"fills": 0, "cash_flows": 0, "closed_trades": 0}
    if manifest["mode"] == "smoke":
        for file in sorted(path.glob("*/summary.json")):
            counts = verify_account(file.parent, read(file))
            totals = {key: totals[key] + counts[key] for key in totals}
        cost_checks = 0
        for base in path.glob("*_BASE"):
            adverse = path / base.name.replace("_BASE", "_ADVERSE_COST")
            a, b = rows(base, "fills"), rows(adverse, "fills")
            assert [(r["timestamp_utc"], r["direction"], r["quantity_lots"]) for r in a] == [
                (r["timestamp_utc"], r["direction"], r["quantity_lots"]) for r in b
            ]
            left, right = rows(base, "trades"), rows(adverse, "trades")
            assert len(left) == len(right)
            for first, second in zip(left, right, strict=True):
                extra = (
                    second["commission_account"]
                    - first["commission_account"]
                    + second["slippage_account"]
                    - first["slippage_account"]
                )
                close(first["net_pnl_account"] - second["net_pnl_account"], extra)
                cost_checks += 1
        totals["matched_trade_cost_checks"] = cost_checks
    else:
        assert verdict["active"] is False and verdict["market_study_executed"] is False
        verify_account(path, read(path / "shared_account_V001.json"))
    for file in path.glob("**/*_V001.json"):
        artifact = read(file)
        if "content_sha256" in artifact:
            assert content_hash(artifact["body"]) == artifact["content_sha256"]
    return {"run_id": run_id, "passed": True, "attempts": len(attempts), **totals}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        results = [verify_run(run_id) for run_id in args.run_id]
    except Exception as error:
        write_new_json(
            args.output,
            {
                "status": "failed",
                "run_ids": args.run_id,
                "error_type": type(error).__name__,
                "reason": str(error),
            },
        )
        raise
    write_new_json(
        args.output,
        {
            "runs": results,
            "reserved_access_logs_unchanged": True,
            "scope": "saved records; current source/immutable identities; no new outcomes",
        },
    )
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
