"""Return distribution analysis.

Descriptive moments, quantiles, tail behaviour against a Gaussian reference,
and optional normality tests.

A standing caution runs through this module: with millions of observations,
Jarque-Bera and D'Agostino return p-values of exactly zero for departures from
normality far too small to matter for anything. Every report therefore carries
both the test result *and* an effect size, and
:func:`normality_tests` attaches an explicit warning above the configured
sample-size threshold. A p-value here answers "is it exactly Gaussian?" - to
which the answer for financial returns is always no - not "does the difference
matter?".
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

__all__ = [
    "DistributionSummary",
    "NormalityResult",
    "describe_distribution",
    "gaussian_tail_comparison",
    "normality_tests",
    "summary_to_frame",
]


@dataclass
class DistributionSummary:
    """Moments, quantiles and tail behaviour for one return series."""

    name: str
    count: int
    mean: float
    median: float
    std: float
    variance: float
    minimum: float
    maximum: float
    skewness: float
    excess_kurtosis: float
    quantiles: dict[str, float] = field(default_factory=dict)
    tails: list[dict[str, Any]] = field(default_factory=list)
    normality: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def describe_distribution(
    values: pl.Series | np.ndarray,
    *,
    name: str = "returns",
    quantiles: tuple[float, ...] = (
        0.001, 0.005, 0.01, 0.025, 0.05, 0.25, 0.50, 0.75, 0.95, 0.975, 0.99, 0.995, 0.999
    ),
) -> DistributionSummary:
    """Descriptive statistics for a return series.

    Skewness and excess kurtosis are the bias-corrected sample estimators
    (``scipy`` with ``bias=False``); excess kurtosis is relative to the
    Gaussian, so 0 means Gaussian-tailed and positive means fatter.
    """
    array = _as_array(values)
    if array.size == 0:
        raise ValueError(f"{name}: no finite observations to describe")

    summary = DistributionSummary(
        name=name,
        count=int(array.size),
        mean=float(np.mean(array)),
        median=float(np.median(array)),
        std=float(np.std(array, ddof=1)) if array.size > 1 else float("nan"),
        variance=float(np.var(array, ddof=1)) if array.size > 1 else float("nan"),
        minimum=float(np.min(array)),
        maximum=float(np.max(array)),
        skewness=float(stats.skew(array, bias=False)) if array.size > 2 else float("nan"),
        excess_kurtosis=(
            float(stats.kurtosis(array, fisher=True, bias=False))
            if array.size > 3
            else float("nan")
        ),
    )
    levels = np.quantile(array, list(quantiles))
    summary.quantiles = {
        _quantile_key(q): float(v) for q, v in zip(quantiles, levels, strict=True)
    }
    summary.notes.append(
        "excess_kurtosis is relative to the Gaussian: 0 = Gaussian tails, "
        "positive = heavier tails."
    )
    return summary


def gaussian_tail_comparison(
    values: pl.Series | np.ndarray,
    *,
    sigma_levels: tuple[float, ...] = (2, 3, 4, 5, 6),
) -> list[dict[str, Any]]:
    """Compare observed tail frequencies with a Gaussian of the same mean/sd.

    For each level ``k`` this reports how often ``|x - mean| > k * sd`` actually
    occurs against how often a Gaussian would produce it. The ratio is the
    honest way to express "heavy tails": a ratio of 50 at 5 sigma says such
    moves happen fifty times more often than a Gaussian would allow, which is a
    statement about magnitude rather than a p-value.
    """
    array = _as_array(values)
    if array.size < 2:
        return []
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1))
    if not math.isfinite(sd) or sd == 0:
        return []

    out: list[dict[str, Any]] = []
    for k in sigma_levels:
        observed = int(np.count_nonzero(np.abs(array - mean) > k * sd))
        observed_rate = observed / array.size
        gaussian_rate = float(2.0 * stats.norm.sf(k))
        out.append({
            "sigma": float(k),
            "observed_count": observed,
            "observed_rate": observed_rate,
            "gaussian_rate": gaussian_rate,
            "gaussian_expected_count": gaussian_rate * array.size,
            "ratio_observed_to_gaussian": (
                observed_rate / gaussian_rate if gaussian_rate > 0 else float("inf")
            ),
        })
    return out


@dataclass
class NormalityResult:
    """One normality test, reported with the context needed to read it."""

    test: str
    statistic: float
    p_value: float
    observations: int
    reject_at_5pct: bool
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normality_tests(
    values: pl.Series | np.ndarray,
    *,
    jarque_bera: bool = True,
    dagostino_k2: bool = True,
    large_sample_threshold: int = 100_000,
) -> dict[str, Any]:
    """Run the configured normality tests and label their limitations.

    The returned ``interpretation`` block is not decoration. With n in the
    millions these tests reject normality essentially always, so the effect
    sizes (skewness, excess kurtosis, tail ratios) are what carry information
    about whether the departure matters.
    """
    array = _as_array(values)
    results: list[NormalityResult] = []
    warning = None
    if array.size >= large_sample_threshold:
        warning = (
            f"n = {array.size:,} exceeds {large_sample_threshold:,}: normality tests "
            "have enormous power at this size and will reject for departures far too "
            "small to matter. Read the effect sizes, not the p-value."
        )

    if jarque_bera and array.size >= 2:
        stat, p = stats.jarque_bera(array)
        results.append(NormalityResult(
            test="jarque_bera", statistic=float(stat), p_value=float(p),
            observations=int(array.size), reject_at_5pct=bool(p < 0.05), warning=warning,
        ))
    if dagostino_k2 and array.size >= 20:
        stat, p = stats.normaltest(array)
        results.append(NormalityResult(
            test="dagostino_k2", statistic=float(stat), p_value=float(p),
            observations=int(array.size), reject_at_5pct=bool(p < 0.05), warning=warning,
        ))

    return {
        "tests": [r.to_dict() for r in results],
        "observations": int(array.size),
        "large_sample_warning": warning,
        "interpretation": (
            "A rejection means the series is not exactly Gaussian. It says nothing "
            "about how far from Gaussian it is, and nothing about tradability. Use "
            "excess kurtosis and the Gaussian tail ratios for magnitude."
        ),
    }


def summary_to_frame(summary: DistributionSummary) -> pl.DataFrame:
    """Flatten a summary into a one-row frame for Parquet/CSV output."""
    row: dict[str, Any] = {
        "name": summary.name,
        "count": summary.count,
        "mean": summary.mean,
        "median": summary.median,
        "std": summary.std,
        "variance": summary.variance,
        "min": summary.minimum,
        "max": summary.maximum,
        "skewness": summary.skewness,
        "excess_kurtosis": summary.excess_kurtosis,
    }
    row.update(summary.quantiles)
    for tail in summary.tails:
        k = f"{tail['sigma']:g}".replace(".", "_")
        row[f"tail_{k}sigma_observed_rate"] = tail["observed_rate"]
        row[f"tail_{k}sigma_ratio_vs_gaussian"] = tail["ratio_observed_to_gaussian"]
    return pl.DataFrame([row])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _as_array(values: pl.Series | np.ndarray) -> np.ndarray:
    """Dense float64 array with nulls and non-finite values removed."""
    array = (
        values.drop_nulls().to_numpy()
        if isinstance(values, pl.Series)
        else np.asarray(values)
    )
    array = array.astype(np.float64, copy=False)
    return array[np.isfinite(array)]


def _quantile_key(q: float) -> str:
    """Column name for a quantile, as a percentile: 0.001 -> ``p0_1``, 0.5 -> ``p50``.

    Percentiles read unambiguously; a bare ``q5`` could be the 5th percentile
    or the median.
    """
    percent = f"{q * 100:.4f}".rstrip("0").rstrip(".")
    return "p" + percent.replace(".", "_")
