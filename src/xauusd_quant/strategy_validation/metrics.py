"""Continuous fold accounting and conditional dependence-aware uncertainty, without annualization."""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from ..execution.engine import ExecutionEngine, Recorder
from .plan import ExperimentPlan


class AuditRecorder:
    """Write every event, retain trades/period summaries only; never retain the tick stream."""

    def __init__(self, sink: Recorder) -> None:
        self.sink = sink
        self.trades: list[dict[str, Any]] = []
        self.seen_positions: set[int] = set()
        self.maximum_holding_seconds = 0.0

    def __call__(self, table: str, row: dict[str, Any]) -> None:
        if table == "trades":
            position = row["position_id"]
            if position in self.seen_positions:
                raise ValueError("duplicate closed position cannot be aggregated")
            self.seen_positions.add(position)
            self.trades.append(row.copy())
            self.maximum_holding_seconds = max(
                self.maximum_holding_seconds, (row["exit_utc"] - row["entry_utc"]).total_seconds()
            )
        self.sink(table, row)


def snapshot(engine: ExecutionEngine, at: datetime) -> dict[str, Any]:
    mark = engine.checkpoint(at)
    return {
        "mark": mark,
        "counts": dict(engine.counts),
        "gross": engine.total_gross,
        "closed_net": engine.realized,
        "commission": engine.total_commission,
        "financing": engine.total_financing,
        "spread": engine.total_spread,
        "slippage": engine.total_slippage,
        "midpoint": engine.midpoint_gross,
        "turnover": engine.turnover,
        "exposure": engine.lot_seconds,
    }


def fold_metrics(fold_id: str, before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    a, b = before["mark"], after["mark"]
    equity_delta = (
        b["equity_account"] - a["equity_account"]
        if b["equity_account"] is not None and a["equity_account"] is not None
        else None
    )
    counts = Counter(after["counts"])
    counts.subtract(before["counts"])
    return {
        "fold_id": fold_id,
        "start_utc": a["timestamp_utc"],
        "end_utc": b["timestamp_utc"],
        "marked_equity_change_account": equity_delta,
        "cash_change_account": b["cash_account"] - a["cash_account"],
        "counts": dict(counts),
        "start_mark": a,
        "end_mark": b,
        **{
            f"period_{key}": after[key] - before[key]
            for key in (
                "gross",
                "closed_net",
                "commission",
                "financing",
                "spread",
                "slippage",
                "midpoint",
                "turnover",
                "exposure",
            )
        },
        "attribution": "equity/cash increments carry exposure; closed trades attributed by exit; not reset experiments",
    }


def aggregate_folds(folds: list[dict[str, Any]], initial_cash: float) -> dict[str, Any]:
    if not folds:
        raise ValueError("no fold metrics")
    for previous, current in zip(folds, folds[1:], strict=False):
        if (
            previous["end_utc"] != current["start_utc"]
            or previous["end_mark"] != current["start_mark"]
        ):
            raise ValueError("fold aggregation would conceal a reset or duplicated interval")
    first, last = folds[0]["start_mark"], folds[-1]["end_mark"]
    net = (
        last["equity_account"] - first["equity_account"]
        if first["equity_account"] is not None and last["equity_account"] is not None
        else None
    )
    values = [row["marked_equity_change_account"] for row in folds]
    if (
        all(v is not None for v in values)
        and net is not None
        and not math.isclose(sum(values), net, abs_tol=max(1e-7, 8 * math.ulp(initial_cash)))
    ):
        raise ValueError("fold equity changes do not reconcile")
    return {
        "marked_equity_change_account": net,
        "return_on_initial_cash": net / initial_cash if net is not None else None,
        "return_definition": "UTC-boundary marked-equity increments / fixed initial cash; no compounding or annualization",
        "folds": len(folds),
        "known_fold_marks": sum(v is not None for v in values),
        "positive_fold_fraction": sum(v > 0 for v in values if v is not None) / len(values)
        if all(v is not None for v in values)
        else None,
        "fold_pnl_std_account": float(np.std(values, ddof=1))
        if len(values) > 1 and all(v is not None for v in values)
        else None,
        "counts": dict(sum((Counter(f["counts"]) for f in folds), Counter())),
        **{
            f"period_{key}": sum(row[f"period_{key}"] for row in folds)
            for key in (
                "gross",
                "closed_net",
                "commission",
                "financing",
                "spread",
                "slippage",
                "midpoint",
                "turnover",
                "exposure",
            )
        },
    }


def daily_uncertainty(
    marks: list[dict[str, Any]], plan: ExperimentPlan, max_holding_seconds: float
) -> dict[str, Any]:
    """Noncircular moving blocks of complete UTC days, retaining dependence within each block.

    Five days is a declared sensitivity convention, not an estimated optimal block.
    Missing calendar days or longer holdings prevent a numeric interval. Bootstrap
    conditions on this policy and cannot correct all previous model/policy searches.
    """
    values = []
    for before, after in zip(marks, marks[1:], strict=False):
        if after["timestamp_utc"] - before["timestamp_utc"] != timedelta(days=1):
            return {"status": "unknown", "reason": "incomplete UTC-day spacing"}
        if (
            before["equity_account"] is None
            or after["equity_account"] is None
            or not math.isfinite(before["equity_account"])
            or not math.isfinite(after["equity_account"])
        ):
            return {"status": "unknown", "reason": "unknown liquidation marks; no imputation"}
        values.append(after["equity_account"] - before["equity_account"])
    block = plan.bootstrap_days
    if len(values) < max(plan.minimum_daily_observations, 4 * block):
        return {
            "status": "insufficient",
            "daily_observations": len(values),
            "required": max(plan.minimum_daily_observations, 4 * block),
        }
    if max_holding_seconds >= block * 86400:
        return {"status": "unknown", "reason": "holding dependence spans frozen block length"}
    x = np.asarray(values)
    rng = np.random.default_rng(plan.seed)
    means = []
    for _ in range(plan.bootstrap_replicates):
        pieces = [
            x[i : i + block]
            for i in rng.integers(0, len(x) - block + 1, size=math.ceil(len(x) / block))
        ]
        means.append(float(np.concatenate(pieces)[: len(x)].mean()))
    return {
        "status": "estimated",
        "daily_observations": len(values),
        "block_days": block,
        "replicates": plan.bootstrap_replicates,
        "seed": plan.seed,
        "mean_daily_pnl_account": float(x.mean()),
        "mean_daily_pnl_95_interval": [float(v) for v in np.quantile(means, [0.025, 0.975])],
        "scope": plan.uncertainty_scope,
        "annualization": "none",
        "limitation": "prespecified heuristic block; residual long dependence and discovery search remain",
    }


def concentration(
    trades: list[dict[str, Any]], folds: list[dict[str, Any]], plan: ExperimentPlan
) -> dict[str, Any]:
    positive = sorted(
        (float(t["net_pnl_account"]) for t in trades if t["net_pnl_account"] > 0), reverse=True
    )
    fold_positive = [
        f["marked_equity_change_account"]
        for f in folds
        if f["marked_equity_change_account"] is not None and f["marked_equity_change_account"] > 0
    ]
    largest = sum(positive[: plan.remove_largest_trades])
    return {
        "largest_positive_trades_account": largest,
        "largest_trade_positive_share": largest / sum(positive) if positive else None,
        "largest_fold_positive_share": max(fold_positive) / sum(fold_positive)
        if fold_positive
        else None,
        "closed_net_without_largest_positive_trades_account": sum(
            float(t["net_pnl_account"]) for t in trades
        )
        - largest,
        "definition": "remove frozen number of largest positive closed trades; descriptive, not independent samples",
    }


def evidence_verdict(
    gates: dict[str, str],
    result: dict[str, Any],
    plan: ExperimentPlan,
    neighbors: list[dict[str, Any]],
) -> dict[str, Any]:
    required = {
        "data_provenance",
        "fold_local_pipeline",
        "null_evidence",
        "execution_specification",
        "evidence_identity",
        "evaluation_history_declared",
    }
    failures = sorted(k for k in required if gates.get(k) != "passed")
    if failures:
        return {"status": "BLOCKED", "reasons": failures}
    if plan.evaluation_history_classification == "software_correctness":
        return {
            "status": "INCONCLUSIVE",
            "reasons": ["synthetic correctness is not market evidence"],
        }
    aggregate, engine = result["aggregate"], result["execution"]
    pnl = aggregate["marked_equity_change_account"]
    uncertainty = result["uncertainty"]
    if (
        pnl is None
        or not math.isfinite(pnl)
        or aggregate["folds"] < plan.minimum_folds
        or engine["counts"].get("closed_positions", 0) < plan.minimum_closed_trades
        or uncertainty["status"] != "estimated"
        or engine["open_position"] is not None
        or aggregate["known_fold_marks"] != aggregate["folds"]
        or any(
            n["aggregate"]["marked_equity_change_account"] is None
            or not math.isfinite(n["aggregate"]["marked_equity_change_account"])
            for n in neighbors
        )
    ):
        return {
            "status": "INCONCLUSIVE",
            "reasons": ["insufficient folds/trades/daily marks or unresolved open exposure"],
        }
    checks = {
        "positive_after_cost": pnl > 0,
        "positive_fold_fraction": aggregate["positive_fold_fraction"]
        >= plan.minimum_positive_fold_fraction,
        "conditional_uncertainty": uncertainty["mean_daily_pnl_95_interval"][0] > 0,
        "trade_concentration": (
            result["concentration"]["largest_trade_positive_share"] is not None
            and result["concentration"]["largest_trade_positive_share"]
            <= plan.maximum_positive_trade_concentration
        ),
        "fold_concentration": (
            result["concentration"]["largest_fold_positive_share"] is not None
            and result["concentration"]["largest_fold_positive_share"]
            <= plan.maximum_positive_fold_concentration
        ),
        "without_largest_trades": result["concentration"][
            "closed_net_without_largest_positive_trades_account"
        ]
        > 0,
        "all_registered_robustness": all(
            n["aggregate"]["marked_equity_change_account"] > 0 for n in neighbors
        ),
    }
    return {
        "status": "PASSES DECLARED HISTORICAL CRITERIA" if all(checks.values()) else "REJECTED",
        "checks": checks,
        "scope": "retrospective frozen-family criteria; not prospective profitability or trading permission",
    }
