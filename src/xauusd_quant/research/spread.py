"""Spread and tick-activity analysis.

The spread matters more here than in most research projects, because the
eventual backtester fills longs at the ask and shorts at the bid. An effect
smaller than the typical spread is not an effect anyone can capture, so the
spread's level, its distribution and *when* it widens are all first-class
results rather than footnotes.

Two cautions carried in the output:

* ``tick_count`` is the number of quote updates in a bar. It is **not** traded
  volume. The ``volume`` column in this dataset is broker-reported liquidity
  units, not contracts, so neither should be read as turnover.
* Correlations here are contemporaneous. Spread and volatility rising together
  within the same bar says nothing about which leads the other.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import polars as pl

from .config import ResearchConfig

__all__ = [
    "ActivityAnalysis",
    "SpreadAnalysis",
    "activity_analysis",
    "spread_analysis",
    "spread_correlations",
]


@dataclass
class SpreadAnalysis:
    """Distribution of the spread, and how it moves with other quantities."""

    timeframe: str
    observations: int
    mean: float
    median: float
    std: float
    minimum: float
    maximum: float
    quantiles: dict[str, float] = field(default_factory=dict)
    wide_threshold: float | None = None
    wide_frequency: float | None = None
    correlations: dict[str, float] = field(default_factory=dict)
    by_hour: pl.DataFrame = field(default_factory=pl.DataFrame)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("by_hour", None)
        return payload

    def to_frame(self) -> pl.DataFrame:
        row: dict[str, Any] = {
            "timeframe": self.timeframe,
            "observations": self.observations,
            "mean_spread": self.mean,
            "median_spread": self.median,
            "std_spread": self.std,
            "min_spread": self.minimum,
            "max_spread": self.maximum,
            "wide_threshold": self.wide_threshold,
            "wide_frequency": self.wide_frequency,
        }
        row.update({f"spread_{k}": v for k, v in self.quantiles.items()})
        row.update({f"corr_{k}": v for k, v in self.correlations.items()})
        return pl.DataFrame([row])


def spread_analysis(
    frame: pl.DataFrame,
    *,
    timeframe: str,
    config: ResearchConfig,
    spread_column: str = "mean_spread",
    return_column: str = "ret_log",
    volatility_column: str | None = None,
    time_basis: str = "timestamp",
) -> SpreadAnalysis | None:
    """Describe the spread distribution and its co-movement with activity."""
    if spread_column not in frame.columns:
        return None
    usable = frame.filter(pl.col(spread_column).is_not_null())
    if usable.is_empty():
        return None

    series = usable[spread_column]
    cfg = config.spread
    quantiles = {
        f"p{q * 100:g}".replace(".", "_"): _f(series.quantile(q)) for q in cfg.quantiles
    }
    wide_threshold = _f(series.quantile(cfg.wide_quantile))
    analysis = SpreadAnalysis(
        timeframe=timeframe,
        observations=int(series.len()),
        mean=_f(series.mean()),
        median=_f(series.median()),
        std=_f(series.std()) if series.len() > 1 else float("nan"),
        minimum=_f(series.min()),
        maximum=_f(series.max()),
        quantiles=quantiles,
        wide_threshold=wide_threshold,
        wide_frequency=_f((series > wide_threshold).mean()),
    )
    analysis.correlations = spread_correlations(
        usable, spread_column=spread_column, return_column=return_column,
        volatility_column=volatility_column, config=config,
    )

    if time_basis in usable.columns:
        analysis.by_hour = (
            usable.with_columns(pl.col(time_basis).dt.hour().alias("hour"))
            .group_by("hour")
            .agg(
                pl.len().alias("bars"),
                pl.col(spread_column).mean().alias("mean_spread"),
                pl.col(spread_column).median().alias("median_spread"),
                pl.col(spread_column).quantile(0.95).alias("p95_spread"),
                pl.col(spread_column).max().alias("max_spread"),
            )
            .sort("hour")
        )

    analysis.notes += [
        f"Spread is the per-bar mean of (ask - bid), in price units, from "
        f"`{spread_column}`.",
        f"'Unusually wide' means above the {cfg.wide_quantile:.1%} quantile "
        f"({wide_threshold:.6g}).",
        "Correlations are contemporaneous within a bar and imply no lead/lag.",
        "Any effect smaller than the typical spread is not capturable; compare "
        "conditional-return magnitudes against median_spread before reading "
        "anything into them.",
    ]
    return analysis


def spread_correlations(
    frame: pl.DataFrame,
    *,
    spread_column: str,
    return_column: str,
    volatility_column: str | None,
    config: ResearchConfig,
) -> dict[str, float]:
    """Pearson correlations of the spread with the configured quantities."""
    wanted = set(config.spread.correlate_with)
    out: dict[str, float] = {}
    candidates: dict[str, pl.Expr] = {}

    if "abs_return" in wanted and return_column in frame.columns:
        candidates["abs_return"] = pl.col(return_column).abs()
    if "volatility" in wanted and volatility_column and volatility_column in frame.columns:
        candidates["volatility"] = pl.col(volatility_column)
    if "tick_count" in wanted and "tick_count" in frame.columns:
        candidates["tick_count"] = pl.col("tick_count").cast(pl.Float64)

    for name, expr in candidates.items():
        paired = frame.select(
            pl.col(spread_column).alias("spread"), expr.alias("other")
        ).drop_nulls()
        if paired.height < 3:
            continue
        a = paired["spread"].to_numpy().astype(np.float64)
        b = paired["other"].to_numpy().astype(np.float64)
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        out[name] = float(np.corrcoef(a, b)[0, 1])
    return out


@dataclass
class ActivityAnalysis:
    """Quote-update activity per bar - explicitly not traded volume."""

    timeframe: str
    observations: int
    mean_ticks_per_bar: float
    median_ticks_per_bar: float
    min_ticks_per_bar: int
    max_ticks_per_bar: int
    quantiles: dict[str, float] = field(default_factory=dict)
    low_activity_threshold: float | None = None
    low_activity_bars: int = 0
    by_hour: pl.DataFrame = field(default_factory=pl.DataFrame)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("by_hour", None)
        return payload

    def to_frame(self) -> pl.DataFrame:
        row: dict[str, Any] = {
            "timeframe": self.timeframe,
            "observations": self.observations,
            "mean_ticks_per_bar": self.mean_ticks_per_bar,
            "median_ticks_per_bar": self.median_ticks_per_bar,
            "min_ticks_per_bar": self.min_ticks_per_bar,
            "max_ticks_per_bar": self.max_ticks_per_bar,
            "low_activity_threshold": self.low_activity_threshold,
            "low_activity_bars": self.low_activity_bars,
        }
        row.update({f"ticks_{k}": v for k, v in self.quantiles.items()})
        return pl.DataFrame([row])


def activity_analysis(
    frame: pl.DataFrame, *, timeframe: str, config: ResearchConfig,
    time_basis: str = "timestamp",
) -> ActivityAnalysis | None:
    """Describe quote activity per bar and flag unusually quiet bars."""
    if "tick_count" not in frame.columns:
        return None
    usable = frame.filter(pl.col("tick_count").is_not_null())
    if usable.is_empty():
        return None

    series = usable["tick_count"]
    cfg = config.activity
    threshold = _f(series.quantile(cfg.low_activity_quantile))
    analysis = ActivityAnalysis(
        timeframe=timeframe,
        observations=int(series.len()),
        mean_ticks_per_bar=_f(series.mean()),
        median_ticks_per_bar=_f(series.median()),
        min_ticks_per_bar=_i(series.min()),
        max_ticks_per_bar=_i(series.max()),
        quantiles={
            f"p{q * 100:g}".replace(".", "_"): _f(series.quantile(q))
            for q in cfg.quantiles
        },
        low_activity_threshold=threshold,
        low_activity_bars=int((series <= threshold).sum()),
    )

    if time_basis in usable.columns:
        analysis.by_hour = (
            usable.with_columns(pl.col(time_basis).dt.hour().alias("hour"))
            .group_by("hour")
            .agg(
                pl.len().alias("bars"),
                pl.col("tick_count").mean().alias("mean_ticks"),
                pl.col("tick_count").median().alias("median_ticks"),
                pl.col("tick_count").sum().alias("total_ticks"),
            )
            .sort("hour")
        )

    single_tick = int((series <= 1).sum())
    if single_tick:
        analysis.warnings.append(
            f"{single_tick:,} bars contain a single quote update. Their OHLC is a "
            "single price and their return is mechanical rather than informative."
        )
    analysis.notes += [
        "tick_count is the number of QUOTE UPDATES in the bar, not traded volume.",
        "The dataset's `volume` column is broker liquidity units, not contracts, "
        "so neither column measures turnover.",
        f"Bars at or below the {cfg.low_activity_quantile:.1%} activity quantile "
        f"({threshold:.6g} updates) are counted as low-activity.",
    ]
    return analysis


def _f(value: Any) -> float:
    """Coerce a Polars aggregate to float.

    Polars aggregates are Optional in the type stubs because an empty frame
    yields null. Callers here guard against that, but an empty group would give
    NaN rather than a crash, which is the right failure mode for a statistic.
    """
    return float("nan") if value is None else float(value)


def _i(value: Any) -> int:
    """Integer counterpart of :func:`_f`; zero when the aggregate is null."""
    return 0 if value is None else int(value)
