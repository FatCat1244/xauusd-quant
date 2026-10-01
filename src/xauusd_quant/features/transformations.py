r"""Causal scaling and winsorisation of feature values (Prompt #8, Steps 12-13).

Research only: the factory stores raw feature values, and these variants show
whether a scaling or a clip changes what a feature says about an outcome. Each
is causal - the value at ``t`` uses bars ``<= t`` only - so none can leak the
scale of later years into earlier bars (the leakage tests append wild future
values and require every earlier value unchanged):

``raw``                the stored value
``signed_log``         :math:`\mathrm{sign}(x) \ln(1 + |x|)` (pointwise)
``trailing_zscore``    :math:`(x_t - \bar x_{t,W}) / s_{t,W}` over the trailing ``W`` bars
``trailing_percentile`` rank of :math:`x_t` among the trailing ``W`` bars
``expanding_robust``   :math:`(x_t - \mathrm{median}) / \mathrm{IQR}` of *all earlier*
                       bars, the edges refreshed each calendar quarter

:func:`winsorize_causal` clips at the lower / upper quantiles of all earlier
bars, refreshed each calendar quarter. A whole-sample quantile or z-score -
the usual shortcut - would place 2005 by the distribution of 2025 and is never
used on features.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from ..regimes.dataset import _rolling_percentile
from .interactions import causal_zscore

__all__ = [
    "expanding_quantiles",
    "quarter_blocks",
    "scaling_variants",
    "signed_log",
    "winsorize_causal",
]


def signed_log(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    return np.sign(x) * np.log1p(np.abs(x))


def quarter_blocks(timestamps: pl.Series) -> tuple[np.ndarray, np.ndarray]:
    """(starts, ends) row ranges of the calendar quarters of a time-sorted series."""
    quarter = (timestamps.dt.year().cast(pl.Int64) * 4
               + (timestamps.dt.month().cast(pl.Int64) - 1) // 3).to_numpy()
    starts = np.concatenate(([0], np.flatnonzero(np.diff(quarter) != 0) + 1)).astype(np.int64)
    ends = np.concatenate((starts[1:], [quarter.size])).astype(np.int64)
    return starts, ends


def expanding_quantiles(values: np.ndarray, timestamps: pl.Series, probs: tuple[float, ...], *,
                        min_history: int) -> np.ndarray:
    """(n, len(probs)) quantiles of every finite value *before* each bar's quarter.

    NaN until *min_history* earlier finite values exist. One partition per
    quarter over the growing history (linear time each).
    """
    x = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(x)
    history = x[finite]
    count_before = np.concatenate(([0], np.cumsum(finite)))      # finite values before row i
    out = np.full((x.size, len(probs)), np.nan)
    starts, ends = quarter_blocks(timestamps)
    for lo, hi in zip(starts, ends, strict=True):
        k = int(count_before[lo])
        if k < min_history:
            continue
        out[lo:hi] = np.quantile(history[:k], probs)
    return out


def winsorize_causal(values: np.ndarray, timestamps: pl.Series, lower: float, upper: float, *,
                     min_history: int = 1000) -> np.ndarray:
    """Clip at the (lower, upper) quantiles of all earlier bars; NaN before *min_history*."""
    x = np.asarray(values, dtype=np.float64)
    q = expanding_quantiles(x, timestamps, (lower, upper), min_history=min_history)
    return np.clip(x, q[:, 0], q[:, 1])


def scaling_variants(values: np.ndarray, timestamps: pl.Series, window: int, *,
                     min_history: int = 1000) -> dict[str, np.ndarray]:
    """Every causal scaling of one feature (see the module docstring)."""
    x = np.asarray(values, dtype=np.float64)
    q = expanding_quantiles(x, timestamps, (0.25, 0.5, 0.75), min_history=min_history)
    iqr = q[:, 2] - q[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        robust = np.where(iqr > 0, (x - q[:, 1]) / np.where(iqr > 0, iqr, 1.0), np.nan)
    return {
        "raw": x,
        "signed_log": signed_log(x),
        "trailing_zscore": causal_zscore(x, window, clip=math.inf),
        "trailing_percentile": _rolling_percentile(x, window),
        "expanding_robust": robust,
    }
