"""Matched target losses and conditional contiguous-block effects, without decorative p-values."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from .plan import EconometricPlan


def loss(value: float, actual: float, target: str) -> float:
    if not math.isfinite(value) or not math.isfinite(actual):
        raise ValueError("nonfinite comparison value")
    if target == "future_return":
        return (actual - value) ** 2
    if target != "realized_variance" or value <= 0 or actual < 0:
        raise ValueError("invalid target/variance forecast")
    return math.log(value) + actual / value


def compare(
    records: list[dict[str, Any]], candidate: str, baseline: str, plan: EconometricPlan
) -> dict[str, Any]:
    c = {
        (r["fold_id"], r["row_id"]): r
        for r in records
        if r["model"] == candidate and r["status"] == "scored"
    }
    b = {
        (r["fold_id"], r["row_id"]): r
        for r in records
        if r["model"] == baseline and r["status"] == "scored"
    }
    if len(c) != sum(r["model"] == candidate and r["status"] == "scored" for r in records) or len(
        b
    ) != sum(r["model"] == baseline and r["status"] == "scored" for r in records):
        raise ValueError("duplicate forecast comparison identity")
    pairs = [(c[k], b[k]) for k in sorted(c.keys() & b.keys())]
    if not pairs:
        return {
            "candidate": candidate,
            "baseline": baseline,
            "status": "untested",
            "matched_rows": 0,
        }
    for a, z in pairs:
        if any(
            a[k] != z[k]
            for k in (
                "target",
                "units",
                "horizon_bars",
                "grid_seconds",
                "available_utc",
                "actual",
                "target_end_utc",
            )
        ):
            raise ValueError("comparison target/horizon/observations/availability mismatch")
    cl = np.asarray([loss(a["value"], a["actual"], a["target"]) for a, _ in pairs])
    bl = np.asarray([loss(z["value"], z["actual"], z["target"]) for _, z in pairs])
    improvement = bl - cl
    by_fold: dict[str, list[float]] = defaultdict(list)
    for (a, _), value in zip(pairs, improvement, strict=True):
        by_fold[a["fold_id"]].append(float(value))
    # QLIKE can be negative: report its matched difference, never divide by negative QLIKE.
    target = pairs[0][0]["target"]
    baseline_mse = float(bl.mean())
    relative = (
        float(improvement.mean() / baseline_mse)
        if target == "future_return" and baseline_mse > 0
        else None
    )
    report: dict[str, Any] = {
        "candidate": candidate,
        "baseline": baseline,
        "status": "measured",
        "target": target,
        "loss": "MSE"
        if target == "future_return"
        else "log(predicted variance)+realized variance/predicted variance (QLIKE)",
        "matched_rows": len(pairs),
        "candidate_scored_rows": len(c),
        "baseline_scored_rows": len(b),
        "candidate_mean_loss": float(cl.mean()),
        "baseline_mean_loss": baseline_mse,
        "mean_loss_improvement": float(improvement.mean()),
        "return_relative_mse_improvement": relative,
        "fold_improvements": {k: float(np.mean(v)) for k, v in by_fold.items()},
        "positive_fold_fraction": sum(np.mean(v) > 0 for v in by_fold.values()) / len(plan.folds),
        "scope": "same matched target/availability rows; conditional frozen model sequence; not whole discovery uncertainty",
        "named_test": "no Diebold-Mariano: adaptive/nested reference, short sample, dependence/heavy tails and prior search",
    }
    times = [datetime.fromisoformat(a["available_utc"]) for a, _ in pairs]
    days = len({t.date() for t in times})
    block = max(plan.horizon_bars, math.ceil(plan.comparison_block_minutes * 60 / plan.seconds))
    blocks = []
    for i in range(len(times) - block + 1):
        if all(
            times[j + 1] - times[j] == timedelta(seconds=plan.seconds)
            and pairs[j][0]["fold_id"] == pairs[j + 1][0]["fold_id"]
            for j in range(i, i + block - 1)
        ):
            blocks.append(improvement[i : i + block])
    if (
        days < plan.minimum_evaluation_days
        or len(pairs) < plan.minimum_comparison_rows
        or len(blocks) < 4
    ):
        report["uncertainty"] = {
            "status": "insufficient",
            "calendar_days": days,
            "minimum_days": plan.minimum_evaluation_days,
            "block_observations": block,
            "available_contiguous_blocks": len(blocks),
            "reason": "calendar/row/block coverage below frozen minima; no interval or p-value",
        }
    else:
        rng = np.random.default_rng(plan.seed)
        means = [
            float(
                np.concatenate(
                    [blocks[k] for k in rng.integers(0, len(blocks), math.ceil(len(pairs) / block))]
                )[: len(pairs)].mean()
            )
            for _ in range(plan.bootstrap_replicates)
        ]
        report["uncertainty"] = {
            "status": "estimated",
            "mean_loss_improvement_95_interval": list(np.quantile(means, [0.025, 0.975])),
            "block_observations": block,
            "seed": plan.seed,
            "replicates": plan.bootstrap_replicates,
            "limitation": "blocks stay inside observed contiguous folds; heuristic dependence length; not full selection bootstrap",
        }
    return report


def addition_verdict(
    comparisons: list[dict[str, Any]],
    gates: dict[str, str],
    plan: EconometricPlan,
    *,
    failed: bool = False,
) -> dict[str, Any]:
    if failed or not comparisons or any(c.get("matched_rows", 0) == 0 for c in comparisons):
        return {
            "status": "BLOCKED BY MISSING INPUTS OR EVIDENCE",
            "reason": "missing/failed model or comparable observations",
        }
    ready = all(
        gates.get(k) == "passed"
        for k in ("data_provenance", "fold_local_pipeline", "evidence_identity", "null_evidence")
    )
    adequate = all(c["uncertainty"]["status"] == "estimated" for c in comparisons)
    if (
        not ready
        or not adequate
        or plan.evaluation_history_classification == "software_correctness"
    ):
        return {
            "status": "INCONCLUSIVE",
            "promotion": "blocked",
            "reason": "insufficient calendar evidence or unverified matching provenance/null evidence; fixture correctness is not market evidence",
        }
    success = all(
        c["mean_loss_improvement"] > 0
        and c["positive_fold_fraction"] >= 0.6
        and c["uncertainty"]["mean_loss_improvement_95_interval"][0] > 0
        and (
            c["return_relative_mse_improvement"] >= plan.minimum_loss_improvement
            if c["return_relative_mse_improvement"] is not None
            else c["mean_loss_improvement"] >= plan.minimum_loss_improvement
        )
        for c in comparisons
    )
    return {
        "status": "PROMISING WITHIN DECLARED HISTORICAL SCOPE"
        if success
        else "NO DEMONSTRATED INCREMENTAL VALUE",
        "scope": "fixed retrospective forecasts; economic value remains separately gated",
    }
