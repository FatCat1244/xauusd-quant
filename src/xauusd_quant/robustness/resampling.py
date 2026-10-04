"""Segment-stratified moving blocks preserve grid dependence and never bridge gaps."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

import numpy as np


def segments(times: list[datetime], groups: list[str], seconds: int) -> list[slice]:
    if not times or len(times) != len(groups) or seconds <= 0:
        raise ValueError("aligned nonempty clocks/groups required")
    if any(t.tzinfo is None or t.utcoffset() is None for t in times):
        raise ValueError("aware clocks required")
    if any(b <= a for a, b in zip(times, times[1:], strict=False)):
        raise ValueError("strict chronological sampling required")
    starts = [0]
    for i in range(1, len(times)):
        if groups[i] != groups[i - 1] or times[i] - times[i - 1] != timedelta(seconds=seconds):
            starts.append(i)
    return [slice(a, b) for a, b in zip(starts, [*starts[1:], len(times)], strict=True)]


def block_indices(length: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Overlapping blocks drawn uniformly; truncate final block to original length."""
    if type(block) is not int or not 1 <= block <= length:
        raise ValueError("block exceeds segment; no excluded short segments or IID fallback")
    starts = rng.integers(0, length - block + 1, math.ceil(length / block))
    return (starts[:, None] + np.arange(block)).ravel()[:length]


def bootstrap(
    values: np.ndarray,
    times: list[datetime],
    groups: list[str],
    *,
    seconds: int,
    horizon: int,
    block: int,
    replicates: int,
    seed: int,
    minimum_days: int,
    minimum_rows: int,
    minimum_blocks: int,
) -> dict[str, Any]:
    """Conditional mean intervals; locally stationary weak dependence assumed.

    Boundary strata retain their observation weights. A segment shorter than the
    block makes this choice unavailable: never discard it and claim full coverage.
    Horizons overlap, so block >= horizon. This is not a significance p-value.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) != len(times) or not np.isfinite(x).all():
        raise ValueError("finite aligned values required, no silent drop")
    if type(horizon) is not int or horizon < 1 or type(block) is not int or block < horizon:
        raise ValueError("block must cover positive overlapping target horizon")
    if not 2 <= replicates <= 999 or min(minimum_days, minimum_rows, minimum_blocks) < 1:
        raise ValueError("bounded replicates and declared evidence minima required")
    strata = segments(times, groups, seconds)
    lengths = [s.stop - s.start for s in strata]
    independent_blocks = sum(n // block for n in lengths)
    days = len({t.date() for t in times})
    report: dict[str, Any] = {
        "status": "insufficient",
        "block": block,
        "seed": seed,
        "replicates": replicates,
        "rows": len(x),
        "days": days,
        "segment_lengths": lengths,
        "nonoverlapping_block_capacity": independent_blocks,
        "mean": float(x.mean()),
        "interval": None,
        "scope": "conditional frozen series; no strategy-discovery correction",
    }
    if (
        days < minimum_days
        or len(x) < minimum_rows
        or independent_blocks < minimum_blocks
        or min(lengths) < block
    ):
        report["reason"] = "coverage/block minima or short segment; all observations retained"
        return report
    if float(np.var(x)) == 0:
        report.update(status="unavailable", reason="zero variance; no degenerate significance")
        return report
    rng = np.random.default_rng(seed)
    means = np.empty(replicates)
    for r in range(replicates):
        total = 0.0
        for s in strata:
            piece = x[s]
            total += float(piece[block_indices(len(piece), block, rng)].sum())
        means[r] = total / len(x)
    report.update(
        status="estimated",
        interval=list(np.quantile(means, [0.025, 0.975])),
        bootstrap_mean_sd=float(np.std(means, ddof=1)),
        assumptions="within-segment weak dependence/local stationarity; blocks resample within folds and sessions; joins artificial; percentile interval approximate",
    )
    return report
