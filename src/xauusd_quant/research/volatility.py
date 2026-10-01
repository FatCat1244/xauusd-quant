"""Volatility estimation and clustering diagnostics.

Rolling standard deviation, realized volatility and EWMA, plus descriptive
volatility-regime buckets.

Annualisation
-------------
Deliberately opt-in. XAUUSD trades roughly 23 hours a day, five days a week,
so the equity convention of 252 days x 6.5 hours is simply the wrong scaling
factor and would overstate annualised volatility by a large margin. Nothing is
annualised unless ``volatility.annualization.enabled`` is set, and when it is,
:func:`annualization_factor` records every assumption in the report.

Everything here is backward-looking: the value at ``t`` uses bars up to and
including ``t`` and never beyond.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import polars as pl

from ..data.resampler import parse_timeframe
from .config import ResearchConfig

__all__ = [
    "AnnualizationInfo",
    "VolatilityRegimes",
    "annualization_factor",
    "ewma_volatility",
    "realized_volatility",
    "rolling_volatility",
    "volatility_clustering",
    "volatility_regimes",
]


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------
def rolling_volatility(
    returns: pl.Expr, window: int, *, min_periods: int | None = None, ddof: int = 1
) -> pl.Expr:
    r"""Rolling standard deviation :math:`\sigma_t = \mathrm{Std}(r_{t-N+1:t})`.

    Backward-looking and inclusive of the current bar, so the value at ``t`` is
    knowable at ``t``.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    return returns.rolling_std(
        window_size=window, min_samples=min_periods or window, ddof=ddof
    )


def realized_volatility(
    returns: pl.Expr, window: int, *, min_periods: int | None = None
) -> pl.Expr:
    r"""Realized volatility :math:`RV_t = \sqrt{\sum_{i=t-N+1}^{t} r_i^2}`.

    Unlike :func:`rolling_volatility` this does not subtract a mean; it is the
    square root of the summed squared returns over the window, which is the
    usual realized-variance construction.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    return (
        (returns**2)
        .rolling_sum(window_size=window, min_samples=min_periods or window)
        .sqrt()
    )


def ewma_volatility(
    returns: pl.Series | np.ndarray, lambda_: float = 0.94, *, warmup: int = 20
) -> np.ndarray:
    r"""EWMA volatility.

    :math:`\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1 - \lambda) r_{t-1}^2`

    Note the ``t-1`` on the right-hand side: :math:`\sigma_t` is built from
    returns strictly *before* ``t``, so it is a genuine one-step-ahead forecast
    and can be used as a feature at ``t`` without look-ahead. The seed variance
    is the sample variance of the first *warmup* returns.

    Returns a float array aligned with the input, with the warm-up prefix as
    NaN.
    """
    if not 0 < lambda_ < 1:
        raise ValueError(f"lambda must be in (0, 1), got {lambda_}")
    array = _as_array(returns, keep_shape=True)
    n = array.size
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out

    warmup = max(2, min(int(warmup), n))
    seed = array[:warmup]
    seed = seed[np.isfinite(seed)]
    if seed.size < 2:
        return out
    variance = float(np.var(seed, ddof=1))

    for i in range(warmup, n):
        previous = array[i - 1]
        if np.isfinite(previous):
            variance = lambda_ * variance + (1.0 - lambda_) * previous**2
        out[i] = math.sqrt(variance) if variance >= 0 else np.nan
    return out


# ---------------------------------------------------------------------------
# Annualisation
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class AnnualizationInfo:
    """The factor used to annualise, and every assumption behind it."""

    enabled: bool
    factor: float | None
    bars_per_year: float | None
    timeframe: str
    trading_days_per_year: float | None
    hours_per_trading_day: float | None
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def annualization_factor(timeframe: str, config: ResearchConfig) -> AnnualizationInfo:
    """Compute :math:`\\sqrt{\\text{bars per year}}`, or decline to.

    Returns a disabled result rather than guessing when annualisation is off,
    so a report can state plainly that its volatility figures are per-bar.
    """
    annual = config.volatility.annualization
    if not annual.enabled:
        return AnnualizationInfo(
            enabled=False, factor=None, bars_per_year=None, timeframe=timeframe,
            trading_days_per_year=None, hours_per_trading_day=None,
            note=(
                "Annualisation disabled. Volatility figures are per-bar at the stated "
                "timeframe. XAUUSD trades ~23h/day, 5 days/week, so equity conventions "
                "(252 x 6.5h) do not apply; enable volatility.annualization only with "
                "an explicit, stated assumption."
            ),
        )

    if annual.bars_per_year is not None:
        bars_per_year = float(annual.bars_per_year)
        basis = "bars_per_year taken directly from configuration"
    else:
        seconds = parse_timeframe(timeframe).total_seconds()
        bars_per_day = (annual.hours_per_trading_day * 3600.0) / seconds
        bars_per_year = bars_per_day * annual.trading_days_per_year
        basis = (
            f"{annual.trading_days_per_year:g} trading days x "
            f"{annual.hours_per_trading_day:g} h/day at {timeframe} bars"
        )

    return AnnualizationInfo(
        enabled=True,
        factor=float(math.sqrt(bars_per_year)),
        bars_per_year=bars_per_year,
        timeframe=timeframe,
        trading_days_per_year=annual.trading_days_per_year,
        hours_per_trading_day=annual.hours_per_trading_day,
        note=f"Annualised by sqrt({bars_per_year:,.0f}); assumption: {basis}.",
    )


# ---------------------------------------------------------------------------
# Clustering and regimes
# ---------------------------------------------------------------------------
def volatility_clustering(
    returns: pl.Series | np.ndarray, *, max_lag: int = 100, confidence_level: float = 0.95
) -> pl.DataFrame:
    r"""Compare :math:`\mathrm{Corr}(|r_t|, |r_{t-k}|)` with the raw-return ACF.

    Putting all three series in one table is the point: volatility clustering
    shows as ``|r|`` and ``r^2`` autocorrelations that stay clearly positive
    over many lags while the raw-return ACF sits near zero.
    """
    from .autocorrelation import acf_absolute_returns, acf_frame, acf_returns, acf_squared_returns

    array = _as_array(returns)
    results = [
        acf_returns(array, max_lag=max_lag, confidence_level=confidence_level),
        acf_absolute_returns(array, max_lag=max_lag, confidence_level=confidence_level),
        acf_squared_returns(array, max_lag=max_lag, confidence_level=confidence_level),
    ]
    return acf_frame(results)


@dataclass
class VolatilityRegimes:
    """Descriptive buckets of a rolling-volatility series.

    Descriptive only. This is not a regime *model* and carries no notion of
    switching, persistence or prediction - it answers "how did returns behave
    when trailing volatility was in its lowest quartile?".
    """

    window: int
    quantiles: list[float]
    thresholds: dict[str, float]
    table: pl.DataFrame = field(default_factory=pl.DataFrame)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "quantiles": self.quantiles,
            "thresholds": self.thresholds,
            "notes": self.notes,
        }


def volatility_regimes(
    frame: pl.DataFrame,
    *,
    return_column: str = "ret_log",
    volatility_column: str,
    quantiles: tuple[float, ...] = (0.25, 0.50, 0.75),
    window: int = 50,
    spread_column: str | None = "mean_spread",
) -> VolatilityRegimes:
    """Bucket bars by trailing volatility and summarise each bucket.

    The volatility used for bucketing is *trailing* - it is known at the bar
    being classified - so the grouping itself contains no look-ahead. The
    summary statistics are contemporaneous with the bucket.
    """
    usable = frame.filter(
        pl.col(volatility_column).is_not_null() & pl.col(return_column).is_not_null()
    )
    if usable.is_empty():
        return VolatilityRegimes(
            window=window, quantiles=list(quantiles), thresholds={},
            notes=["No bars had both a return and a trailing volatility value."],
        )

    edges = [
        float(usable.select(pl.col(volatility_column).quantile(q)).item()) for q in quantiles
    ]
    thresholds = {f"q{q:g}": e for q, e in zip(quantiles, edges, strict=True)}

    bucket = pl.lit(f"Q{len(edges) + 1}_highest")
    for i, edge in enumerate(reversed(edges)):
        index = len(edges) - i
        label = "Q1_lowest" if index == 1 else f"Q{index}"
        bucket = pl.when(pl.col(volatility_column) <= edge).then(pl.lit(label)).otherwise(bucket)

    aggs = [
        pl.len().alias("observations"),
        pl.col(return_column).mean().alias("mean_return"),
        pl.col(return_column).median().alias("median_return"),
        pl.col(return_column).std().alias("return_std"),
        pl.col(return_column).abs().mean().alias("mean_abs_return"),
        pl.col(volatility_column).mean().alias("mean_trailing_volatility"),
        pl.col(volatility_column).min().alias("min_trailing_volatility"),
        pl.col(volatility_column).max().alias("max_trailing_volatility"),
    ]
    if spread_column and spread_column in usable.columns:
        aggs.append(pl.col(spread_column).mean().alias("mean_spread"))
    if "tick_count" in usable.columns:
        aggs.append(pl.col("tick_count").mean().alias("mean_tick_count"))

    table = (
        usable.with_columns(bucket.alias("volatility_regime"))
        .group_by("volatility_regime")
        .agg(aggs)
        .sort("volatility_regime")
    )

    return VolatilityRegimes(
        window=window,
        quantiles=list(quantiles),
        thresholds=thresholds,
        table=table,
        notes=[
            f"Buckets are quantiles of the {window}-bar trailing volatility "
            f"({volatility_column}), which is known at the bar it classifies.",
            "Descriptive only: this is not a regime model and implies nothing about "
            "persistence or predictability.",
        ],
    )


def _as_array(values: pl.Series | np.ndarray, *, keep_shape: bool = False) -> np.ndarray:
    if isinstance(values, pl.Series):
        array = values.to_numpy() if keep_shape else values.drop_nulls().to_numpy()
    else:
        array = np.asarray(values)
    array = array.astype(np.float64, copy=False)
    return array if keep_shape else array[np.isfinite(array)]
