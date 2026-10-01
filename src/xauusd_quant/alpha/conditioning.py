r"""IC within conditions: volatility, regime probability, time of day, a second feature
(Prompt #8, Steps 28-31, 45).

Conditions are never chosen by an outcome. Volatility quartiles and the
quartiles of a conditioning feature use **causal** edges: the quantiles of all
bars *before* the calendar quarter, refreshed each quarter - a 2003-2026 rank
would place early bars by the distribution of later years. Regime conditions
use the Prompt #7 filtered probabilities at thresholds fixed in the config
(descriptive, never optimised). Intraday groups are the hour, the configured
session and the weekday.

:func:`condition_moments` keeps the month axis, so the batch-means standard
errors of :mod:`.information_coefficient` apply within each condition.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from .information_coefficient import SX, SXX, SXY, SY, SYY, N, ic_with_errors, target_columns

__all__ = [
    "causal_quantile_buckets",
    "condition_ics",
    "condition_moments",
    "month_codes",
]


def month_codes(starts: np.ndarray, n_rows: int) -> np.ndarray:
    """Month index (0..M-1) of every row from the month start rows."""
    codes = np.zeros(n_rows, dtype=np.int64)
    codes[starts[1:]] = 1
    return np.cumsum(codes)


def causal_quantile_buckets(values: np.ndarray, timestamps: pl.Series, q: int, *,
                            min_history: int) -> np.ndarray:
    """Bucket 0..q-1 by quantile edges of all earlier bars, refreshed each calendar quarter.

    Rows before *min_history* finite earlier values, and missing values, get -1.
    """
    x = np.asarray(values, dtype=np.float64)
    quarter = (timestamps.dt.year().cast(pl.Int64) * 4
               + (timestamps.dt.month().cast(pl.Int64) - 1) // 3).to_numpy()
    starts = np.concatenate(([0], np.flatnonzero(np.diff(quarter) != 0) + 1))
    ends = np.concatenate((starts[1:], [x.size]))
    out = np.full(x.size, -1, dtype=np.int64)
    finite = np.isfinite(x)
    sorted_hist = np.empty(0)
    last = 0
    probs = np.linspace(0, 1, q + 1)[1:-1]
    for lo, hi in zip(starts, ends, strict=True):
        new = x[last:lo][finite[last:lo]]
        if new.size:
            sorted_hist = np.sort(np.concatenate((sorted_hist, new)))
        last = lo
        if sorted_hist.size < min_history:
            continue
        edges = np.quantile(sorted_hist, probs)
        seg = x[lo:hi]
        ok = np.isfinite(seg)
        out[lo:hi][ok] = np.searchsorted(edges, seg[ok], side="right")
    return out


def condition_moments(x: np.ndarray, ys: np.ndarray | Sequence[np.ndarray], months: np.ndarray,
                      cond: np.ndarray, n_months: int, n_cond: int) -> np.ndarray:
    """(n_months, n_cond, targets, 6) pairwise-complete sums per month and condition.

    Rows with ``cond < 0`` are left out. *ys* may be a sequence of column views.
    """
    x = np.asarray(x, dtype=np.float64)
    columns = target_columns(ys)
    fx = np.isfinite(x) & (cond >= 0)
    out = np.zeros((n_months, n_cond, len(columns), 6), dtype=np.float64)
    size = n_months * n_cond
    for j, column in enumerate(columns):
        y = np.asarray(column, dtype=np.float64)
        mask = fx & np.isfinite(y)
        code = months[mask] * n_cond + cond[mask]
        xv, yv = x[mask], y[mask]
        for k, w in ((N, None), (SX, xv), (SY, yv), (SXX, xv * xv), (SYY, yv * yv),
                     (SXY, xv * yv)):
            out[:, :, j, k] = np.bincount(code, weights=w, minlength=size).reshape(
                n_months, n_cond)
    return out


def condition_ics(moments: np.ndarray, *, min_obs: int, hac_lags: int = 2
                  ) -> dict[str, np.ndarray]:
    """IC, n, batch-means SE and z per condition (axis 1) from :func:`condition_moments`."""
    n_cond = moments.shape[1]
    keys = ("ic", "n", "se", "z", "p")
    out = {k: np.full((n_cond, *moments.shape[2:-1]), np.nan) for k in keys}
    for c in range(n_cond):
        stats = ic_with_errors(moments[:, c], hac_lags=hac_lags)
        enough = stats["n"] >= min_obs
        for k in keys:
            out[k][c] = np.where(enough, stats[k], np.nan) if k != "n" else stats[k]
    return out
