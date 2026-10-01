r"""Information coefficients from monthly sufficient statistics (Prompt #8, Steps 21-27).

For one feature ``x`` and a block of targets ``Y`` (one column per horizon),
:func:`block_moments` returns, for every calendar month and every target, the
pair count and the sums

.. math:: n,\ \sum x,\ \sum y,\ \sum x^2,\ \sum y^2,\ \sum xy

over the bars where *both* are finite (pairwise-complete). Every temporal
slice - the pooled sample, a year, a quarter, an era, a trailing window of
months, odd / even years - is a sum of months, so one pass over the data
serves them all, in float64.

**Pearson IC** is the correlation of the raw values; **rank IC** is the same
correlation of rank scores :math:`u = (\mathrm{rank} - \tfrac12)/n`, each
column ranked once over its own finite values (a close, documented
approximation of Spearman's correlation on each pairwise-complete subsample).

**Standard errors** are batch means over months. With pooled means and
standard deviations, the correlation's influence function is
:math:`u_t = z_x z_y - \tfrac r2 (z_x^2 + z_y^2)`; its month sums
:math:`U_m` would be independent if the serial dependence (of overlapping
targets and persistent features) were shorter than a month. It is not always,
so neighbouring months enter with Bartlett weights (Newey-West over the month
sums, ``hac_lags = 2``):

.. math:: \widehat{\mathrm{Var}}(r) = \frac{\sum_m U_m^2
          + 2 \sum_{l=1}^{L} (1 - \tfrac{l}{L+1}) \sum_m U_m U_{m+l}}{n^2}.

The naive :math:`(1-r^2)/\sqrt{n-3}` is kept beside it, labelled i.i.d.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import norm

__all__ = [
    "MonthIndex",
    "block_moments",
    "correlation_from_sums",
    "ic_with_errors",
    "month_index",
    "month_moments",
    "rank_scores",
    "target_columns",
]

#: Order of the moment axis in every stats array.
N, SX, SY, SXX, SYY, SXY = range(6)


@dataclass(frozen=True)
class MonthIndex:
    """Contiguous month blocks of a time-sorted table."""

    starts: np.ndarray            # first row of each month
    labels: list[str]             # "YYYY-MM"
    years: np.ndarray             # year of each month
    quarters: np.ndarray          # 1..4
    months: np.ndarray            # 1..12

    @property
    def size(self) -> int:
        return int(self.starts.size)


def month_index(timestamps: pl.Series) -> MonthIndex:
    """Month blocks of a time-sorted timestamp series (refuses unsorted input)."""
    frame = pl.DataFrame({"ts": timestamps}).with_columns(
        pl.col("ts").dt.year().cast(pl.Int32).alias("y"),
        pl.col("ts").dt.month().cast(pl.Int32).alias("m"))
    if not timestamps.is_sorted():
        raise ValueError("timestamps must be sorted")
    key = (frame["y"] * 100 + frame["m"]).to_numpy()
    change = np.flatnonzero(np.diff(key) != 0) + 1
    starts = np.concatenate(([0], change)).astype(np.int64)
    ym = key[starts]
    years = (ym // 100).astype(np.int64)
    months = (ym % 100).astype(np.int64)
    return MonthIndex(starts=starts, labels=[f"{y:04d}-{m:02d}" for y, m in zip(years, months,
                                                                                strict=True)],
                      years=years, quarters=((months - 1) // 3 + 1).astype(np.int64),
                      months=months)


def rank_scores(values: np.ndarray) -> np.ndarray:
    """(rank - 0.5) / n over the finite values (average ties), NaN elsewhere; float64."""
    x = np.asarray(values, dtype=np.float64)
    out = np.full(x.size, np.nan)
    finite = np.isfinite(x)
    n = int(finite.sum())
    if n == 0:
        return out
    s = pl.Series("x", x[finite]).rank(method="average").to_numpy()
    out[finite] = (s - 0.5) / n
    return out


def _first_finite(values: np.ndarray) -> float:
    v = np.asarray(values)
    idx = np.flatnonzero(np.isfinite(v))
    return float(v[idx[0]]) if idx.size else 0.0


def _center(values: np.ndarray) -> np.ndarray:
    """Values minus their first finite value (float64).

    The correlation of every slice is unchanged and the raw sums lose no digits
    to a large common level (ln RV sits around -6). The constant is the
    *earliest* value, never a full-sample mean: a causal slice (a trailing
    window of months) then depends on earlier data only, bit for bit.
    """
    v = np.asarray(values, dtype=np.float64)
    return v - _first_finite(v)


def target_columns(ys: np.ndarray | Sequence[np.ndarray]) -> list[np.ndarray]:
    """A 1-D array, a 2-D array (one target per column) or a sequence of columns -> columns."""
    if isinstance(ys, np.ndarray):
        return [ys] if ys.ndim == 1 else [ys[:, j] for j in range(ys.shape[1])]
    return list(ys)


def block_moments(x: np.ndarray, ys: np.ndarray | Sequence[np.ndarray],
                  starts: np.ndarray) -> np.ndarray:
    """(months, targets, 6) pairwise-complete sums n, Sx, Sy, Sxx, Syy, Sxy per month.

    Each series is centred by its own first finite value first; every
    correlation built from these sums is shift-invariant, and no slice (a
    year, a trailing window) reads a later value through that constant.
    Targets are processed one column at
    a time to keep the working memory at a few arrays of one column. *ys* may
    be a sequence of column views (no copy of a column-major table).
    """
    x = _center(x)
    columns = target_columns(ys)
    fx = np.isfinite(x)
    x0 = np.where(fx, x, 0.0)
    out = np.empty((starts.size, len(columns), 6), dtype=np.float64)
    for j, column in enumerate(columns):
        y = _center(column)
        mask = fx & np.isfinite(y)
        y0 = np.where(mask, y, 0.0)
        xm = np.where(mask, x0, 0.0)
        out[:, j, N] = np.add.reduceat(mask.astype(np.float64), starts)
        out[:, j, SX] = np.add.reduceat(xm, starts)
        out[:, j, SY] = np.add.reduceat(y0, starts)
        out[:, j, SXX] = np.add.reduceat(xm * xm, starts)
        out[:, j, SYY] = np.add.reduceat(y0 * y0, starts)
        out[:, j, SXY] = np.add.reduceat(xm * y0, starts)
    # reduceat of an empty slice returns the element at the index: guard empty months
    empty = np.diff(np.concatenate((starts, [x.size]))) == 0
    if empty.any():
        out[empty] = 0.0
    return out


def _side(block: np.ndarray, means: np.ndarray) -> np.ndarray:
    """``[finite, v0, v0^2]`` of a (rows, k) block centred by *means*, v0 = 0 where missing."""
    v = block.astype(np.float64) - means
    finite = np.isfinite(v)
    v0 = np.where(finite, v, 0.0)
    return np.hstack((finite.astype(np.float64), v0, v0 * v0))


def month_moments(x_columns: Sequence[np.ndarray], y_columns: Sequence[np.ndarray],
                  starts: np.ndarray) -> np.ndarray:
    """(features, targets, months, 6) pairwise-complete sums, one matrix product per month.

    With ``A = [f_x, x_0, x_0^2]`` and ``B = [f_y, y_0, y_0^2]`` for the rows of
    one month (``f`` the finite masks, ``*_0`` the centred values with missing
    set to 0), ``A^T B`` holds every sum at once: ``n = f_x.f_y``,
    ``Sx = x_0.f_y``, ``Sxx = x_0^2.f_y``, ``Sy = f_x.y_0``, ``Syy = f_x.y_0^2``,
    ``Sxy = x_0.y_0`` - exactly the pairwise-complete sums of :func:`block_moments`
    (the same centring by each column's first finite value), for a batch of
    features against every target column.
    """
    x_cols = list(x_columns)
    y_cols = list(y_columns)
    nf, nt = len(x_cols), len(y_cols)
    n = x_cols[0].size
    xm = np.array([_first_finite(c) for c in x_cols])
    ym = np.array([_first_finite(c) for c in y_cols])
    bounds = np.concatenate((starts, [n])).astype(np.int64)
    out = np.zeros((nf, nt, starts.size, 6))
    for m in range(starts.size):
        s, e = int(bounds[m]), int(bounds[m + 1])
        if e <= s:
            continue
        a = _side(np.column_stack([c[s:e] for c in x_cols]), xm)
        b = _side(np.column_stack([c[s:e] for c in y_cols]), ym)
        g = a.T @ b
        out[:, :, m, N] = g[:nf, :nt]
        out[:, :, m, SX] = g[nf:2 * nf, :nt]
        out[:, :, m, SXX] = g[2 * nf:, :nt]
        out[:, :, m, SY] = g[:nf, nt:2 * nt]
        out[:, :, m, SYY] = g[:nf, 2 * nt:]
        out[:, :, m, SXY] = g[nf:2 * nf, nt:2 * nt]
    return out


def correlation_from_sums(s: np.ndarray) -> np.ndarray:
    """Correlation from summed moments (..., 6); NaN where undefined."""
    n = s[..., N]
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = s[..., SXY] - s[..., SX] * s[..., SY] / n
        vx = s[..., SXX] - s[..., SX] ** 2 / n
        vy = s[..., SYY] - s[..., SY] ** 2 / n
        r = cov / np.sqrt(vx * vy)
    r = np.where((n > 2) & (vx > 0) & (vy > 0), r, np.nan)
    return np.clip(r, -1.0, 1.0)


def ic_with_errors(months: np.ndarray, *, select: np.ndarray | None = None,
                   hac_lags: int = 2) -> dict[str, np.ndarray]:
    """Pooled IC over the chosen months with batch-means and i.i.d. standard errors.

    *months* is (M, ..., 6); *select* a boolean mask over M (default all). Returns
    arrays over the trailing axes: ``ic``, ``n``, ``se`` (batch means with a
    Newey-West correction over *hac_lags* neighbouring months, Bartlett
    weights), ``se_iid``, ``z``, ``p`` (two-sided, normal), ``ci_low``,
    ``ci_high`` and ``months`` (months with pairs).

    Plain batch means assume independent months; a persistent feature against
    an overlapping target carries dependence across month boundaries (the
    noise features of the research found ~1.5x too many p < 0.05 without the
    correction at 1h).
    """
    sel = months if select is None else months[select]
    total = sel.sum(axis=0)
    r = correlation_from_sums(total)
    n = total[..., N]
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = total[..., SX] / n
        my = total[..., SY] / n
        sx = np.sqrt(np.maximum(total[..., SXX] / n - mx ** 2, 0.0))
        sy = np.sqrt(np.maximum(total[..., SYY] / n - my ** 2, 0.0))
        nm = sel[..., N]
        # month sums of z_x z_y, z_x^2 and z_y^2 under the pooled moments
        zxy = (sel[..., SXY] - mx * sel[..., SY] - my * sel[..., SX] + nm * mx * my) / (sx * sy)
        zxx = (sel[..., SXX] - 2 * mx * sel[..., SX] + nm * mx ** 2) / sx ** 2
        zyy = (sel[..., SYY] - 2 * my * sel[..., SY] + nm * my ** 2) / sy ** 2
        u = np.nan_to_num(zxy - 0.5 * r * (zxx + zyy), nan=0.0, posinf=0.0, neginf=0.0)
        acc = (u ** 2).sum(axis=0)
        for lag in range(1, min(hac_lags, u.shape[0] - 1) + 1):
            weight = 1.0 - lag / (hac_lags + 1.0)
            acc = acc + 2.0 * weight * (u[lag:] * u[:-lag]).sum(axis=0)
        var = np.maximum(acc, 0.0) / n ** 2
        se = np.where(np.isfinite(r), np.sqrt(var), np.nan)
        se_iid = (1.0 - r ** 2) / np.sqrt(np.maximum(n - 3.0, 1.0))
        z = r / se
    p = 2.0 * norm.sf(np.abs(z))
    return {"ic": r, "n": n, "se": se, "se_iid": se_iid, "z": z,
            "p": np.where(np.isfinite(z), p, np.nan),
            "ci_low": r - 1.959964 * se, "ci_high": r + 1.959964 * se,
            "months": (nm > 0).sum(axis=0)}
