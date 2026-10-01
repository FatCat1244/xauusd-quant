r"""Descriptive analysis of rolling-regression outputs.

Slope, :math:`R^2`, the residual distribution and residual autocorrelation.
Everything here describes what the detrending produced; whether the residual
*reverts* is :mod:`xauusd_quant.research.residual_decay`.

A caution that applies to every autocorrelation in this module: a
rolling-regression residual is **autocorrelated by construction**. Consecutive
windows share ``N-1`` of their ``N`` bars, so consecutive residuals are fitted
against almost the same line. A large positive lag-1 residual ACF is therefore
expected and carries little information on its own. The differenced series
:math:`\Delta\epsilon_t` is the more informative object, and
:func:`residual_autocorrelation` computes all three side by side for exactly
that reason.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import polars as pl

from ..features.config import RegressionConfig
from .autocorrelation import acf_frame, autocorrelation, ljung_box_frame, ljung_box_test
from .distributions import describe_distribution, gaussian_tail_comparison

__all__ = [
    "ResidualDistribution",
    "SlopeStatistics",
    "residual_autocorrelation",
    "residual_distribution",
    "r_squared_statistics",
    "slope_statistics",
]


# ---------------------------------------------------------------------------
# Slope
# ---------------------------------------------------------------------------
@dataclass
class SlopeStatistics:
    """Distribution of the rolling regression slope.

    ``slope`` is the per-bar change in log price implied by the local trend.
    ``pct_per_bar`` is :math:`e^\\beta - 1`, the same thing as a percentage.
    Neither is annualised: that needs an explicit bars-per-year assumption,
    which this project does not make silently.
    """

    timeframe: str
    window: int
    observations: int
    mean: float
    median: float
    std: float
    quantiles: dict[str, float] = field(default_factory=dict)
    positive_frequency: float = float("nan")
    negative_frequency: float = float("nan")
    mean_pct_per_bar: float = float("nan")
    notes: list[str] = field(default_factory=list)

    def to_frame(self) -> pl.DataFrame:
        row: dict[str, Any] = {
            "timeframe": self.timeframe,
            "window": self.window,
            "observations": self.observations,
            "slope_mean": self.mean,
            "slope_median": self.median,
            "slope_std": self.std,
            "positive_frequency": self.positive_frequency,
            "negative_frequency": self.negative_frequency,
            "mean_pct_per_bar": self.mean_pct_per_bar,
        }
        row.update({f"slope_{k}": v for k, v in self.quantiles.items()})
        return pl.DataFrame([row])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def slope_statistics(
    frame: pl.DataFrame, *, timeframe: str, window: int, config: RegressionConfig
) -> SlopeStatistics:
    """Describe the rolling slope over the whole sample."""
    values = _finite(frame["regression_slope"])
    if values.size == 0:
        return SlopeStatistics(
            timeframe=timeframe, window=window, observations=0,
            mean=float("nan"), median=float("nan"), std=float("nan"),
            notes=["No valid regression slopes."],
        )

    summary = describe_distribution(
        values, name="slope", quantiles=config.distribution.quantiles
    )
    return SlopeStatistics(
        timeframe=timeframe,
        window=window,
        observations=int(values.size),
        mean=summary.mean,
        median=summary.median,
        std=summary.std,
        quantiles=summary.quantiles,
        positive_frequency=float(np.mean(values > 0)),
        negative_frequency=float(np.mean(values < 0)),
        mean_pct_per_bar=float(np.mean(np.expm1(values))),
        notes=[
            "slope is the per-bar change in log price of the local trend.",
            "mean_pct_per_bar = mean(exp(beta) - 1). Not annualised: that needs an "
            "explicit bars-per-year assumption.",
        ],
    )


def _grouped_slope(
    frame: pl.DataFrame, *, group_column: str, label: str
) -> pl.DataFrame:
    """Slope statistics within each level of a grouping column."""
    usable = frame.filter(
        pl.col("regression_slope").is_not_null() & pl.col(group_column).is_not_null()
    )
    if usable.is_empty():
        return pl.DataFrame()
    return (
        usable.group_by(group_column)
        .agg(
            pl.len().alias("observations"),
            pl.col("regression_slope").mean().alias("slope_mean"),
            pl.col("regression_slope").median().alias("slope_median"),
            pl.col("regression_slope").std().alias("slope_std"),
            (pl.col("regression_slope") > 0).mean().alias("positive_frequency"),
            pl.col("regression_slope").quantile(0.05).alias("slope_p5"),
            pl.col("regression_slope").quantile(0.95).alias("slope_p95"),
        )
        .sort(group_column)
        .rename({group_column: label})
    )


def slope_by_group(frame: pl.DataFrame, *, group_column: str) -> pl.DataFrame:
    """Public wrapper: slope statistics grouped by an existing column."""
    return _grouped_slope(frame, group_column=group_column, label=group_column)


# ---------------------------------------------------------------------------
# R-squared
# ---------------------------------------------------------------------------
def r_squared_statistics(
    frame: pl.DataFrame, *, timeframe: str, window: int, config: RegressionConfig
) -> pl.DataFrame:
    r"""Distribution of the rolling :math:`R^2`, plus residual behaviour per quartile.

    A high :math:`R^2` means the window was well described by a straight line;
    a low one means the local path was not trend-like. Whether the residual
    behaves differently in the two cases is a genuine question, so the residual
    standard deviation and extreme frequency are reported per quartile.
    """
    usable = frame.filter(pl.col("r_squared").is_not_null())
    if usable.is_empty():
        return pl.DataFrame()

    values = _finite(usable["r_squared"])
    summary = describe_distribution(
        values, name="r_squared", quantiles=config.distribution.quantiles
    )
    overall = {
        "timeframe": timeframe,
        "window": window,
        "bucket": "all",
        "observations": int(values.size),
        "r2_mean": summary.mean,
        "r2_median": summary.median,
        "r2_std": summary.std,
        "r2_p5": summary.quantiles.get("p5"),
        "r2_p95": summary.quantiles.get("p95"),
        "residual_std": float(np.nanstd(_finite(usable["residual"]), ddof=1)),
        "abs_z_gt_2_frequency": _extreme_frequency(usable, config.conditioning.extreme_abs_z),
    }

    rows = [overall]
    bucketed = add_quantile_buckets(
        usable, column="r_squared", label="r2_bucket",
        buckets=config.conditioning.quantile_buckets,
    )
    for bucket in sorted(bucketed["r2_bucket"].unique().drop_nulls().to_list()):
        chunk = bucketed.filter(pl.col("r2_bucket") == bucket)
        r2_values = _finite(chunk["r_squared"])
        residuals = _finite(chunk["residual"])
        rows.append({
            "timeframe": timeframe,
            "window": window,
            "bucket": bucket,
            "observations": int(r2_values.size),
            "r2_mean": float(np.mean(r2_values)) if r2_values.size else float("nan"),
            "r2_median": float(np.median(r2_values)) if r2_values.size else float("nan"),
            "r2_std": float(np.std(r2_values, ddof=1)) if r2_values.size > 1 else float("nan"),
            "r2_p5": float(np.quantile(r2_values, 0.05)) if r2_values.size else float("nan"),
            "r2_p95": float(np.quantile(r2_values, 0.95)) if r2_values.size else float("nan"),
            "residual_std": (
                float(np.std(residuals, ddof=1)) if residuals.size > 1 else float("nan")
            ),
            "abs_z_gt_2_frequency": _extreme_frequency(
                chunk, config.conditioning.extreme_abs_z
            ),
        })
    return pl.DataFrame(rows, infer_schema_length=None)


# ---------------------------------------------------------------------------
# Residual distribution
# ---------------------------------------------------------------------------
@dataclass
class ResidualDistribution:
    """Distribution of the residual and of both Z-score variants."""

    timeframe: str
    window: int
    series: dict[str, Any] = field(default_factory=dict)
    tails: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_frame(self) -> pl.DataFrame:
        rows = []
        for name, summary in self.series.items():
            row = {
                "timeframe": self.timeframe, "window": self.window, "series": name,
                "count": summary["count"], "mean": summary["mean"],
                "median": summary["median"], "std": summary["std"],
                "min": summary["minimum"], "max": summary["maximum"],
                "skewness": summary["skewness"],
                "excess_kurtosis": summary["excess_kurtosis"],
            }
            row.update(summary["quantiles"])
            rows.append(row)
        return pl.DataFrame(rows, infer_schema_length=None)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def residual_distribution(
    frame: pl.DataFrame, *, timeframe: str, window: int, config: RegressionConfig
) -> ResidualDistribution:
    """Moments, quantiles and tail behaviour of the residual and Z-scores."""
    result = ResidualDistribution(timeframe=timeframe, window=window)
    targets = ("residual", "residual_pct", "residual_zscore_fit", "residual_zscore_rolling")

    for name in targets:
        if name not in frame.columns:
            continue
        values = _finite(frame[name])
        if values.size < 10:
            result.warnings.append(f"{name}: only {values.size} finite values; skipped.")
            continue
        summary = describe_distribution(
            values, name=name, quantiles=config.distribution.quantiles
        )
        result.series[name] = summary.to_dict()
        if config.distribution.gaussian_reference:
            result.tails[name] = gaussian_tail_comparison(
                values, sigma_levels=config.distribution.tail_sigma_levels
            )

    fit = result.series.get("residual_zscore_fit")
    if fit:
        skew = fit["skewness"]
        result.notes.append(
            f"residual_zscore_fit skewness = {skew:+.4f}: "
            + ("close to symmetric." if abs(skew) < 0.1
               else f"{'right' if skew > 0 else 'left'}-skewed, so positive and negative "
                    "extremes are not mirror images.")
        )
        result.notes.append(
            "The in-window residual mean is exactly zero by OLS construction, so "
            "residual_zscore_fit has no mean subtracted; its own mean over the sample "
            "need not be zero."
        )
    result.notes.append(
        "Gaussianity is not assumed anywhere. Excess kurtosis is relative to the "
        "Gaussian and the tail table gives observed-versus-Gaussian frequency ratios."
    )
    return result


# ---------------------------------------------------------------------------
# Residual autocorrelation
# ---------------------------------------------------------------------------
def residual_autocorrelation(
    frame: pl.DataFrame, *, timeframe: str, window: int, config: RegressionConfig
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    r"""ACF and Ljung-Box for the residual, its difference and :math:`|\epsilon|`.

    The three are computed together because the raw residual ACF is dominated by
    window overlap: successive windows share ``N-1`` bars, so a high lag-1
    value says almost nothing. :math:`\Delta\epsilon` removes that mechanical
    component.
    """
    residual = _finite(frame["residual"])
    if residual.size < 100:
        return pl.DataFrame(), pl.DataFrame(), [
            f"Only {residual.size} residuals; autocorrelation skipped."
        ]

    available = {
        "residual": residual,
        "residual_diff": np.diff(residual),
        "abs_residual": np.abs(residual),
    }
    cfg = config.autocorrelation
    results, lb_rows = [], []
    for name in cfg.series:
        values = available.get(name)
        if values is None or values.size < 100:
            continue
        results.append(
            autocorrelation(
                values, max_lag=cfg.max_lag, series_name=name,
                confidence_level=cfg.confidence_level,
            )
        )
        lb_rows += ljung_box_test(values, lags=cfg.ljung_box_lags, series_name=name)

    notes = [
        f"Overlapping windows: consecutive residuals share {window - 1} of {window} "
        "bars, so a high lag-1 residual ACF is mechanical and expected.",
        "residual_diff is the informative series for reversion; compare it against "
        "the raw residual ACF rather than reading the latter alone.",
        "Ljung-Box p-values are reported next to mean |rho| because at these sample "
        "sizes the test rejects for negligible correlation.",
    ]
    # Every other table in this module carries its model; these two did not,
    # which made them the only outputs that could not be read on their own.
    tag = [
        pl.lit(timeframe).alias("timeframe"),
        pl.lit(window, dtype=pl.Int64).alias("window"),
    ]
    acf = acf_frame(results)
    ljung = ljung_box_frame(lb_rows)
    if not acf.is_empty():
        acf = acf.with_columns(tag).select("timeframe", "window", pl.exclude("timeframe", "window"))
    if not ljung.is_empty():
        ljung = ljung.with_columns(tag).select("timeframe", "window", pl.exclude("timeframe", "window"))
    return acf, ljung, notes


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def add_quantile_buckets(
    frame: pl.DataFrame, *, column: str, label: str, buckets: int = 4
) -> pl.DataFrame:
    """Label each row by which quantile bucket of *column* it falls in.

    Quantile edges come from the whole sample, which is fine for describing
    history but would be look-ahead in a live rule. The reports say so.
    """
    usable = frame.filter(pl.col(column).is_not_null())
    if usable.is_empty():
        return frame.with_columns(pl.lit(None, dtype=pl.Utf8).alias(label))

    edges = [
        float(usable.select(pl.col(column).quantile(i / buckets)).item())
        for i in range(1, buckets)
    ]
    # Built outward: the LAST `when` applied is evaluated first, so iterating the
    # edges in reverse means the highest edge gets the second-highest label.
    # Q1 is the lowest bucket and Q{buckets} the highest.
    expr = pl.lit(f"Q{buckets}")
    for i, edge in enumerate(reversed(edges)):
        bucket = f"Q{buckets - 1 - i}"
        expr = pl.when(pl.col(column) <= edge).then(pl.lit(bucket)).otherwise(expr)
    return frame.with_columns(
        pl.when(pl.col(column).is_null()).then(None).otherwise(expr).alias(label)
    )


def _extreme_frequency(frame: pl.DataFrame, threshold: float) -> float:
    """Share of rows whose fit Z-score exceeds *threshold* in magnitude."""
    if "residual_zscore_fit" not in frame.columns:
        return float("nan")
    values = _finite(frame["residual_zscore_fit"])
    return float(np.mean(np.abs(values) > threshold)) if values.size else float("nan")


def _finite(series: pl.Series) -> np.ndarray:
    array = series.drop_nulls().to_numpy().astype(np.float64, copy=False)
    return array[np.isfinite(array)]
