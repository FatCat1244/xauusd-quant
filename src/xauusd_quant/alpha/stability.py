r"""Stability of an IC through time (Prompt #8, Steps 25-27, 32-34, 62).

Everything here is a sum of the monthly moments of
:mod:`.information_coefficient`, so it is exact (no resampling) and costs no
second pass over the bars:

* yearly and quarterly IC (Steps 25-26; a year / quarter with too few pairs is
  not reported, and tiny quarters are not over-read);
* consistency (Step 27): mean / median / sd of the yearly ICs, the share of
  years with each sign and with the pooled sign, worst and best year, the
  recent-period IC;
* eras (Step 32): the pair's active months cut into equal chronological
  thirds - defined by the calendar, never by where the IC looks good;
* subsamples (Step 62): odd / even years, first / second half, alternating
  quarters - fragility checks, not a substitute for out-of-sample tests;
* a causal rolling IC (Steps 33-34): the IC over the trailing ``W`` months,
  dated at the month *after* the window (its last pairs' outcomes end within
  ``h`` bars of that month's start), and alpha-health descriptors from it.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from .information_coefficient import MonthIndex, N, correlation_from_sums, ic_with_errors

__all__ = [
    "consistency",
    "era_ics",
    "grouped_ics",
    "rolling_ics",
    "subsample_ics",
]


def grouped_ics(months: np.ndarray, groups: np.ndarray, *, min_obs: int
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(labels, ic (G, ...), n (G, ...)) for contiguous month groups (years, quarters)."""
    labels, starts = np.unique(groups, return_index=True)
    order = np.argsort(starts)
    labels, starts = labels[order], starts[order]
    sums = np.add.reduceat(months, starts, axis=0)
    ic = correlation_from_sums(sums)
    n = sums[..., N]
    return labels, np.where(n >= min_obs, ic, np.nan), n


def consistency(yearly_ic: np.ndarray, pooled_ic: np.ndarray, recent_ic: np.ndarray
                ) -> dict[str, np.ndarray]:
    """Step 27 descriptors over the years (axis 0) for every pair (NaN where no year)."""
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN pairs -> NaN, masked below
        valid = np.isfinite(yearly_ic)
        count = valid.sum(axis=0)
        sign = np.sign(pooled_ic)
        pos = np.where(valid, yearly_ic > 0, False).sum(axis=0)
        neg = np.where(valid, yearly_ic < 0, False).sum(axis=0)
        same = np.where(valid, np.sign(yearly_ic) == sign, False).sum(axis=0)
        aligned = np.where(valid, yearly_ic * sign, np.nan)
        with np.errstate(all="ignore"):
            out = {
                "years": count,
                "mean_ic": np.nanmean(yearly_ic, axis=0),
                "median_ic": np.nanmedian(yearly_ic, axis=0),
                "std_ic": np.nanstd(yearly_ic, axis=0, ddof=1),
                "positive_year_fraction": pos / np.maximum(count, 1),
                "negative_year_fraction": neg / np.maximum(count, 1),
                "sign_consistency": same / np.maximum(count, 1),
                "worst_year_ic": np.nanmin(aligned, axis=0) * sign,
                "best_year_ic": np.nanmax(aligned, axis=0) * sign,
                "recent_ic": recent_ic,
            }
    for key in ("mean_ic", "median_ic", "std_ic", "worst_year_ic", "best_year_ic"):
        out[key] = np.where(count > 0, out[key], np.nan)
    return out


def _active_months(months: np.ndarray) -> np.ndarray:
    """(M, pairs) bool: months in which a pair has observations."""
    return months[..., N] > 0


def era_ics(months: np.ndarray, eras: int, *, min_obs: int) -> np.ndarray:
    """(eras, ...) IC over equal chronological thirds of each pair's active months."""
    flat = months.reshape(months.shape[0], -1, 6)
    active = _active_months(flat)
    out = np.full((eras, flat.shape[1]), np.nan)
    for j in range(flat.shape[1]):
        idx = np.flatnonzero(active[:, j])
        if idx.size < eras:
            continue
        bounds = np.linspace(0, idx.size, eras + 1).astype(int)
        for e in range(eras):
            chunk = idx[bounds[e]:bounds[e + 1]]
            total = flat[chunk, j].sum(axis=0)
            if total[N] >= min_obs:
                out[e, j] = correlation_from_sums(total)
    return out.reshape((eras, *months.shape[1:-1]))


def subsample_ics(months: np.ndarray, index: MonthIndex, *, min_obs: int
                  ) -> dict[str, np.ndarray]:
    """IC in odd / even years, first / second half and alternating quarters."""
    flat = months.reshape(months.shape[0], -1, 6)
    active = _active_months(flat)
    q_seq = (index.years - index.years[0]) * 4 + (index.quarters - 1)
    masks = {
        "odd_years": index.years % 2 == 1,
        "even_years": index.years % 2 == 0,
        "odd_quarters": q_seq % 2 == 1,
        "even_quarters": q_seq % 2 == 0,
    }
    out: dict[str, np.ndarray] = {}
    for name, mask in masks.items():
        total = flat[mask].sum(axis=0)
        ic = correlation_from_sums(total)
        out[name] = np.where(total[..., N] >= min_obs, ic, np.nan)
    first = np.full(flat.shape[1], np.nan)
    second = np.full(flat.shape[1], np.nan)
    for j in range(flat.shape[1]):
        idx = np.flatnonzero(active[:, j])
        if idx.size < 2:
            continue
        half = idx.size // 2
        a = flat[idx[:half], j].sum(axis=0)
        b = flat[idx[half:], j].sum(axis=0)
        if a[N] >= min_obs:
            first[j] = correlation_from_sums(a)
        if b[N] >= min_obs:
            second[j] = correlation_from_sums(b)
    out["first_half"] = first
    out["second_half"] = second
    shape = months.shape[1:-1]
    return {k: v.reshape(shape) for k, v in out.items()}


def rolling_ics(months: np.ndarray, *, window: int, min_months: int, min_obs: int,
                hac_lags: int = 2) -> dict[str, np.ndarray]:
    """Causal rolling IC: value at month m uses months [m - window, m - 1] only.

    Returns ``ic`` and ``z`` (batch-means, Newey-West over *hac_lags* months) of
    shape (M, ...), NaN where fewer than *min_months* months or *min_obs* pairs
    are in the window.
    """
    total_m = months.shape[0]
    ic = np.full((total_m, *months.shape[1:-1]), np.nan)
    z = np.full_like(ic, np.nan)
    for m in range(min_months, total_m):
        lo = max(0, m - window)
        sel = months[lo:m]
        stats = ic_with_errors(sel, hac_lags=hac_lags)
        enough = ((sel[..., N] > 0).sum(axis=0) >= min_months) & (stats["n"] >= min_obs)
        ic[m] = np.where(enough, stats["ic"], np.nan)
        z[m] = np.where(enough, stats["z"], np.nan)
    return {"ic": ic, "z": z}


def alpha_health(rolling_ic: np.ndarray, rolling_z: np.ndarray, pooled_ic: np.ndarray
                 ) -> dict[str, Any]:
    """Step 34 descriptors: last rolling IC and z, its gap to the pooled IC, sign stability."""
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(rolling_ic)
        count = valid.sum(axis=0)
        same = np.where(valid, np.sign(rolling_ic) == np.sign(pooled_ic), False).sum(axis=0)
        last_ic = np.full(pooled_ic.shape, np.nan)
        last_z = np.full(pooled_ic.shape, np.nan)
        flat_ic = rolling_ic.reshape(rolling_ic.shape[0], -1)
        flat_z = rolling_z.reshape(rolling_z.shape[0], -1)
        li, lz = last_ic.reshape(-1), last_z.reshape(-1)
        for j in range(flat_ic.shape[1]):
            idx = np.flatnonzero(np.isfinite(flat_ic[:, j]))
            if idx.size:
                li[j] = flat_ic[idx[-1], j]
                lz[j] = flat_z[idx[-1], j]
    return {"rolling_windows": count,
            "rolling_sign_stability": same / np.maximum(count, 1),
            "last_rolling_ic": last_ic, "last_rolling_z": last_z,
            "recent_minus_pooled": last_ic - pooled_ic}
