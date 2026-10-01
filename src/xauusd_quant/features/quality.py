r"""Missingness, distributions and drift of every feature (Prompt #8, Steps 8-11).

Nothing is imputed here - missing values are characterised: overall, by year,
by month (through time) and by regime (the Prompt #7 filtered state), because
a feature missing mostly in one era or one state is not missing at random.

Distributions: count, missing, mean, median, sd, skewness, excess kurtosis,
quantiles, min / max, and flags for near-constant features (IQR zero or one
value > 95 % of the bars), extreme outliers (a robust Z = (x - median) / (1.4826
MAD) beyond 50), unstable scale (yearly sd ratio max / min > 10) and numerical
explosions (non-finite or |x| > 1e6).

Drift (Step 11) is measured three ways, because one metric can mislead: the
yearly / quarterly / era medians in pooled-IQR units, the population
stability index (PSI) of each year against the pooled deciles, the
Kolmogorov-Smirnov statistic and the Wasserstein distance (in pooled-sd
units) of each year against the pooled sample. Samples are evenly spaced
within each year and the KS / Wasserstein values are read on a 2000-point
quantile grid (estimates, labelled; scipy's exact versions re-sort the
reference for every year and cost 30x more for no difference that matters here).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

__all__ = ["distribution_row", "drift_rows", "ks_and_wasserstein", "missingness_tables"]


def missingness_tables(frame: pl.DataFrame, names: list[str], regime_state: np.ndarray | None
                       ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(overall + by regime, by year, by month) missing shares per feature."""
    years = frame["timestamp"].dt.year().to_numpy()
    months = frame["timestamp"].dt.strftime("%Y-%m").to_numpy()
    overall, by_year, by_month = [], [], []
    uniq_years = np.unique(years)
    month_labels, month_inv = np.unique(months, return_inverse=True)
    states = None if regime_state is None else np.asarray(regime_state)
    for name in names:
        miss = ~np.isfinite(frame[name].cast(pl.Float64).fill_null(np.nan).to_numpy())
        row: dict[str, Any] = {"feature": name, "missing_share": float(miss.mean()),
                               "missing_bars": int(miss.sum())}
        if states is not None:
            for s in np.unique(states[states >= 0]):
                sel = states == s
                row[f"missing_share_state_{int(s)}"] = float(miss[sel].mean())
            row["missing_share_no_state"] = float(miss[states < 0].mean()) if (states < 0).any() \
                else 0.0
        overall.append(row)
        by_year.append({"feature": name, **{str(int(y)): float(miss[years == y].mean())
                                           for y in uniq_years}})
        counts = np.bincount(month_inv, minlength=month_labels.size)
        missing = np.bincount(month_inv, weights=miss.astype(np.float64),
                              minlength=month_labels.size)
        by_month.append({"feature": name, **{m: float(missing[i] / counts[i])
                                            for i, m in enumerate(month_labels)}})
    return (pl.DataFrame(overall, infer_schema_length=None),
            pl.DataFrame(by_year, infer_schema_length=None),
            pl.DataFrame(by_month, infer_schema_length=None))


def distribution_row(name: str, values: np.ndarray, years: np.ndarray) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(x)
    f = x[finite]
    row: dict[str, Any] = {"feature": name, "count": int(f.size),
                           "missing": int((~finite).sum()),
                           "non_finite_values": int((~np.isfinite(x) & ~np.isnan(x)).sum())}
    if f.size == 0:
        return row
    q = np.quantile(f, [0.0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0])
    mean, sd = float(f.mean()), float(f.std(ddof=1)) if f.size > 1 else 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        c = f - mean
        m2 = float((c ** 2).mean())
        skew = float((c ** 3).mean() / m2 ** 1.5) if m2 > 0 else float("nan")
        kurt = float((c ** 4).mean() / m2 ** 2 - 3.0) if m2 > 0 else float("nan")
    med = float(q[5])
    mad = float(np.median(np.abs(f - med)))
    robust_z = (np.abs(f - med) / (1.4826 * mad)) if mad > 0 else np.zeros(f.size)
    values_top = pl.Series(f).value_counts(sort=True)
    top_share = float(values_top["count"][0] / f.size) if values_top.height else 0.0
    yearly_sd = []
    for y in np.unique(years[finite]):
        s = f[years[finite] == y]
        if s.size > 30:
            yearly_sd.append(float(s.std(ddof=1)))
    ys = np.array([v for v in yearly_sd if v > 0])
    row.update({
        "mean": mean, "median": med, "std": sd, "skewness": skew, "excess_kurtosis": kurt,
        "min": float(q[0]), "p001": float(q[1]), "p01": float(q[2]), "p05": float(q[3]),
        "p25": float(q[4]), "p75": float(q[6]), "p95": float(q[7]), "p99": float(q[8]),
        "p999": float(q[9]), "max": float(q[10]), "iqr": float(q[6] - q[4]),
        "top_value_share": top_share,
        "max_robust_z": float(robust_z.max()) if robust_z.size else float("nan"),
        "yearly_sd_ratio": float(ys.max() / ys.min()) if ys.size > 1 else float("nan"),
    })
    flags = []
    if row["iqr"] == 0 or top_share > 0.95:
        flags.append("near_constant")
    if row["max_robust_z"] > 50:
        flags.append("extreme_outliers")
    if np.isfinite(row["yearly_sd_ratio"]) and row["yearly_sd_ratio"] > 10:
        flags.append("unstable_scale")
    if row["non_finite_values"] > 0 or float(np.abs(f).max()) > 1e6:
        flags.append("numerical_explosion")
    row["flags"] = ",".join(flags)
    return row


def _proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts = np.histogram(values, bins=edges)[0].astype(np.float64)
    return np.maximum(counts / max(counts.sum(), 1.0), 1e-4)


def _quantile_grid(sorted_values: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Empirical quantile function Q(u) = x_(floor(u n)) of sorted values."""
    idx = np.minimum((u * sorted_values.size).astype(np.int64), sorted_values.size - 1)
    return sorted_values[idx]


def ks_and_wasserstein(sample_sorted: np.ndarray, ref_sorted: np.ndarray, *,
                       grid: int = 2000) -> tuple[float, float]:
    """(KS statistic, Wasserstein-1) of two sorted samples, evaluated on a quantile grid.

    KS is the largest CDF gap over the grid quantiles of both samples; W1 is
    the mean absolute gap between the quantile functions on ``grid`` equal
    steps of probability. Both are estimates within ~1/grid of the exact values.
    """
    u = (np.arange(grid) + 0.5) / grid
    qs, qr = _quantile_grid(sample_sorted, u), _quantile_grid(ref_sorted, u)
    points = np.concatenate((qs, qr))
    fs = np.searchsorted(sample_sorted, points, side="right") / sample_sorted.size
    fr = np.searchsorted(ref_sorted, points, side="right") / ref_sorted.size
    return float(np.abs(fs - fr).max()), float(np.abs(qs - qr).mean())


def drift_rows(name: str, values: np.ndarray, years: np.ndarray, *, sample_per_year: int = 10_000,
               reference_size: int = 50_000, seed: int = 0) -> list[dict[str, Any]]:
    """Per-year drift of one feature against its pooled distribution (estimates).

    The pooled reference and each year are evenly spaced samples (at most
    *reference_size* and *sample_per_year* values); the reference is sorted and
    binned once.
    """
    del seed                                   # evenly spaced samples: nothing is random
    x = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(x)
    pooled = x[finite]
    if pooled.size < 100:
        return []
    ref = pooled if pooled.size <= reference_size else pooled[
        np.linspace(0, pooled.size - 1, reference_size).astype(np.int64)]
    ref = np.sort(ref)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, 11)))
    if edges.size < 3:
        edges = np.array([ref[0] - 1e-12, float(np.median(ref)), ref[-1] + 1e-12])
    edges[0], edges[-1] = -np.inf, np.inf
    expected = _proportions(ref, edges)
    med = float(np.median(ref))
    iqr = float(np.quantile(ref, 0.75) - np.quantile(ref, 0.25))
    sd = float(ref.std(ddof=1)) or 1.0
    year_f = years[finite]
    rows = []
    for y in np.unique(year_f):
        s = pooled[year_f == y]
        if s.size < 100:
            continue
        if s.size > sample_per_year:
            s = s[np.linspace(0, s.size - 1, sample_per_year).astype(np.int64)]
        s = np.sort(s)
        actual = _proportions(s, edges)
        ks, w1 = ks_and_wasserstein(s, ref)
        rows.append({"feature": name, "year": int(y), "n": int(s.size),
                     "median_shift_iqr": (float(np.median(s)) - med) / iqr if iqr > 0 else
                     float("nan"),
                     "psi": float(((actual - expected) * np.log(actual / expected)).sum()),
                     "ks": ks, "wasserstein_sd": w1 / sd})
    return rows
