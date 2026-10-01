r"""Rolling OLS detrending of a single price series.

Inside each window of ``N`` bars ending at ``t``:

.. math::

    y_i = \alpha_t + \beta_t x_i + \epsilon_i,
    \qquad x_i = 0, 1, \ldots, N-1,
    \qquad y_i = \ln P_i

The fitted value at the window's last point and the residual there are:

.. math::

    \hat y_t = \alpha_t + \beta_t (N-1), \qquad \epsilon_t = y_t - \hat y_t

Causality
---------
The window is ``[t-N+1, t]``. The value at ``t`` is a function of those ``N``
bars and nothing else, so appending later data cannot change it. ``x`` is a
local coordinate restarting at 0 in every window, so a window's absolute
position in history never enters the fit either. There is no centred window
anywhere in this module.

What this is not
----------------
This is local detrending of one price series. It is **not** cointegration and
**not** statistical arbitrage: there is no second asset and no equilibrium
relationship being estimated. A residual is simply the distance from a locally
fitted line, and calling it a "spread" would be misleading.

Numerical approach
------------------
The closed-form OLS solution is used, but computed from *centred* window
values rather than from raw power sums. With :math:`y = \ln P \approx 6` and
:math:`N = 512`, the textbook form :math:`\sum y^2 - N\bar y^2` subtracts two
numbers near 18,432 to obtain a variance that can be ~1e-6, discarding ten
significant digits. Centring first avoids that cancellation entirely, at the
cost of materialising the window matrix - which is chunked, so peak memory is
bounded and independent of series length.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import polars as pl

from ..utils.logging import get_logger
from .config import RegressionConfig, RegressionModelConfig

__all__ = [
    "FEATURE_COLUMNS",
    "RollingRegressionResult",
    "fit_window",
    "rolling_ols",
    "rolling_regression_features",
    "theil_sen_slope",
    "window_design",
]

LOGGER = get_logger("features.rolling_regression")

#: Columns produced by :func:`rolling_regression_features`, in order.
FEATURE_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "close",
    "log_price",
    "regression_intercept",
    "regression_slope",
    "fitted_log_price",
    "fitted_price",
    "residual",
    "residual_pct",
    "residual_std_fit",
    "r_squared",
    "n_observations",
    "residual_zscore_fit",
    "residual_zscore_rolling",
)


# ---------------------------------------------------------------------------
# Window geometry - constants that depend only on N
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class WindowDesign:
    r"""The fixed design matrix quantities for a window of ``N`` bars.

    Because ``x`` is always ``0..N-1``, every ``x``-side quantity is a constant
    and is computed once per window size rather than once per bar.
    """

    n: int
    x: np.ndarray
    x_mean: float
    x_centred: np.ndarray
    sxx_centred: float

    @property
    def last_x_centred(self) -> float:
        """``x_{N-1} - xbar``, i.e. the leverage of the newest point."""
        return float(self.n - 1) - self.x_mean


def window_design(n: int) -> WindowDesign:
    r"""Fixed ``x`` quantities for a window of ``n`` bars.

    :math:`\bar x = (N-1)/2` and :math:`S_{xx} = N(N^2-1)/12`, both exact.
    """
    if n < 3:
        raise ValueError(f"A regression window needs at least 3 bars, got {n}")
    x = np.arange(n, dtype=np.float64)
    x_mean = (n - 1) / 2.0
    x_centred = x - x_mean
    # Closed form, kept beside the computed value as a cross-check.
    sxx = n * (n * n - 1) / 12.0
    return WindowDesign(
        n=n, x=x, x_mean=x_mean, x_centred=x_centred, sxx_centred=float(sxx)
    )


# ---------------------------------------------------------------------------
# Single-window fit - the reference implementation
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class WindowFit:
    """OLS fit of one window, used as the reference in tests."""

    intercept: float
    slope: float
    fitted_last: float
    residual_last: float
    residual_std: float
    r_squared: float
    n: int


def fit_window(y: np.ndarray, *, epsilon: float = 1e-12) -> WindowFit:
    r"""Fit :math:`y_i = \alpha + \beta x_i` on one window, ``x = 0..N-1``.

    The straightforward implementation, kept deliberately simple so the
    vectorised path in :func:`rolling_ols` can be checked against it.
    """
    values = np.asarray(y, dtype=np.float64)
    n = values.size
    design = window_design(n)

    y_mean = float(values.mean())
    y_centred = values - y_mean
    sxy = float(y_centred @ design.x_centred)
    slope = sxy / design.sxx_centred if design.sxx_centred > epsilon else np.nan
    intercept = y_mean - slope * design.x_mean

    fitted = intercept + slope * design.x
    residuals = values - fitted
    sse = float(residuals @ residuals)
    sst = float(y_centred @ y_centred)

    return WindowFit(
        intercept=float(intercept),
        slope=float(slope),
        fitted_last=float(fitted[-1]),
        residual_last=float(residuals[-1]),
        residual_std=float(np.sqrt(sse / (n - 2))) if n > 2 else np.nan,
        r_squared=float(1.0 - sse / sst) if sst > epsilon else np.nan,
        n=n,
    )


# ---------------------------------------------------------------------------
# Vectorised rolling fit
# ---------------------------------------------------------------------------
@dataclass
class RollingRegressionResult:
    """Per-bar regression outputs, aligned with the input series."""

    window: int
    intercept: np.ndarray
    slope: np.ndarray
    fitted: np.ndarray
    residual: np.ndarray
    residual_std: np.ndarray
    r_squared: np.ndarray
    n_observations: np.ndarray
    degenerate_windows: int = 0
    warmup_bars: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            k: v for k, v in asdict(self).items() if not isinstance(v, np.ndarray)
        }
        payload["valid_observations"] = int(np.isfinite(self.residual).sum())
        return payload


def rolling_ols(
    y: np.ndarray,
    window: int,
    *,
    min_residual_std: float = 1e-10,
    min_total_variance: float = 1e-16,
    chunk_elements: int = 4_000_000,
) -> RollingRegressionResult:
    r"""Rolling OLS of ``y`` on ``0..N-1``, one fit per bar.

    Every output at index ``t`` uses only ``y[t-N+1 : t+1]``. The first
    ``N-1`` entries are NaN: a partial window is a different model and is not
    silently fitted, nor backfilled from later data.

    Degenerate windows - flat prices, or a total variance below
    ``min_total_variance`` - yield NaN for :math:`R^2` and the Z-score rather
    than an infinity, and are counted in ``degenerate_windows``.
    """
    values = np.asarray(y, dtype=np.float64)
    n_total = values.size
    design = window_design(window)

    out = {
        name: np.full(n_total, np.nan, dtype=np.float64)
        for name in ("intercept", "slope", "fitted", "residual", "residual_std", "r_squared")
    }
    counts = np.zeros(n_total, dtype=np.int64)
    result = RollingRegressionResult(
        window=window,
        intercept=out["intercept"], slope=out["slope"], fitted=out["fitted"],
        residual=out["residual"], residual_std=out["residual_std"],
        r_squared=out["r_squared"], n_observations=counts,
        warmup_bars=min(window - 1, n_total),
    )
    if n_total < window:
        result.notes.append(
            f"Series has {n_total} bars, fewer than the {window}-bar window: no fits."
        )
        return result

    from numpy.lib.stride_tricks import sliding_window_view

    windows = sliding_window_view(values, window)          # (n_total-window+1, window)
    n_windows = windows.shape[0]
    rows_per_chunk = max(1, chunk_elements // window)
    degenerate = 0

    for start in range(0, n_windows, rows_per_chunk):
        stop = min(start + rows_per_chunk, n_windows)
        block = np.ascontiguousarray(windows[start:stop])   # (R, N)

        # Centring before forming any sum of squares is what keeps this stable.
        y_mean = block.mean(axis=1)
        y_centred = block - y_mean[:, None]
        sxy = y_centred @ design.x_centred                  # (R,)
        sst = np.einsum("ij,ij->i", y_centred, y_centred)   # (R,)

        with np.errstate(invalid="ignore", divide="ignore"):
            slope = sxy / design.sxx_centred
            intercept = y_mean - slope * design.x_mean
            # Residuals across the whole window, computed directly rather than
            # via a difference of large sums.
            residuals = y_centred - slope[:, None] * design.x_centred[None, :]
            sse = np.einsum("ij,ij->i", residuals, residuals)
            r_squared = 1.0 - sse / sst
            residual_std = np.sqrt(sse / (window - 2))

        flat = sst <= min_total_variance
        if flat.any():
            degenerate += int(flat.sum())
            r_squared[flat] = np.nan
            residual_std[flat] = np.nan
        residual_std[residual_std < min_residual_std] = np.nan

        # The newest bar of each window is the one the features describe.
        lo, hi = window - 1 + start, window - 1 + stop
        out["slope"][lo:hi] = slope
        out["intercept"][lo:hi] = intercept
        out["fitted"][lo:hi] = intercept + slope * (window - 1)
        out["residual"][lo:hi] = residuals[:, -1]
        out["residual_std"][lo:hi] = residual_std
        out["r_squared"][lo:hi] = r_squared
        counts[lo:hi] = window

    result.degenerate_windows = degenerate
    result.notes.append(
        f"Window [t-{window - 1}, t]; the first {result.warmup_bars} bars have no fit "
        "and are left NaN rather than backfilled."
    )
    if degenerate:
        result.notes.append(
            f"{degenerate:,} window(s) had total variance <= {min_total_variance:g} "
            "(flat prices); their R^2 and residual std are NaN, not infinite."
        )
    return result


# ---------------------------------------------------------------------------
# Optional robust comparison
# ---------------------------------------------------------------------------
def theil_sen_slope(y: np.ndarray) -> float:
    """Theil-Sen slope for one window: the median of all pairwise slopes.

    O(N^2), so only ever used on a small sample of windows as a check that the
    OLS slope is not being driven by a handful of outliers. OLS remains the
    model.
    """
    values = np.asarray(y, dtype=np.float64)
    n = values.size
    if n < 2:
        return float("nan")
    i, j = np.triu_indices(n, k=1)
    return float(np.median((values[j] - values[i]) / (j - i)))


# ---------------------------------------------------------------------------
# Feature table
# ---------------------------------------------------------------------------
def rolling_regression_features(
    bars: pl.DataFrame,
    *,
    window: int,
    config: RegressionConfig,
    timeframe: str,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    r"""Build the full regression feature table for one timeframe and window.

    Returns the frame and a diagnostics dictionary. Both Z-score methods are
    emitted, in separate columns, and both are causal:

    ``residual_zscore_fit``
        :math:`\epsilon_t / s_{\text{fit}}` where
        :math:`s_{\text{fit}} = \sqrt{SSE/(N-2)}` is the residual standard
        error of the *current* fit. The in-window residual mean is exactly zero
        by construction, so no mean is subtracted.

    ``residual_zscore_rolling``
        :math:`(\epsilon_t - \mu_t)/\sigma_t` where the moments are rolling
        statistics of the residual *series*. A different question: how unusual
        is this residual against recent residuals?
    """
    model = config.regression
    numerical = config.numerical

    if bars.is_empty():
        raise ValueError(f"No bars supplied for {timeframe}")
    price_column = _resolve_price_column(model, bars.columns)

    frame = bars.sort("timestamp")
    price = frame[price_column].to_numpy().astype(np.float64)

    if model.price_transform == "log":
        if np.any(price <= 0):
            raise ValueError(
                f"{timeframe}: {int((price <= 0).sum())} non-positive prices cannot be "
                "log-transformed. Re-check the bar data."
            )
        y = np.log(price)
    else:
        y = price.copy()

    # `numerical.epsilon` guards the scalar reference path in `fit_window`. The
    # vectorised path has no use for it: Sxx is the constant N(N^2-1)/12, and
    # the two variance floors below cover the divisions that can degenerate.
    fit = rolling_ols(
        y, window,
        min_residual_std=numerical.min_residual_std,
        min_total_variance=numerical.min_total_variance,
        chunk_elements=numerical.chunk_elements,
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        zscore_fit = fit.residual / fit.residual_std
    zscore_fit[~np.isfinite(zscore_fit)] = np.nan

    columns: dict[str, Any] = {
        "timestamp": frame["timestamp"],
        "close": pl.Series("close", price),
        "log_price": pl.Series("log_price", y),
        "regression_intercept": pl.Series("regression_intercept", fit.intercept),
        "regression_slope": pl.Series("regression_slope", fit.slope),
        "fitted_log_price": pl.Series("fitted_log_price", fit.fitted),
        "fitted_price": pl.Series(
            "fitted_price",
            np.exp(fit.fitted) if model.price_transform == "log" else fit.fitted,
        ),
        "residual": pl.Series("residual", fit.residual),
        # For a log-price model the residual is already a relative deviation;
        # exp(eps)-1 expresses it as a straightforward percentage.
        "residual_pct": pl.Series(
            "residual_pct",
            np.expm1(fit.residual) if model.price_transform == "log"
            else fit.residual / np.where(price != 0, price, np.nan),
        ),
        "residual_std_fit": pl.Series("residual_std_fit", fit.residual_std),
        "r_squared": pl.Series("r_squared", fit.r_squared),
        "n_observations": pl.Series("n_observations", fit.n_observations),
        "residual_zscore_fit": pl.Series("residual_zscore_fit", zscore_fit),
    }
    out = pl.DataFrame(columns)

    # Method B, for every configured rolling window.
    primary = config.zscore.resolve_primary(window)
    for rolling_window in sorted(set(config.zscore.resolve(window))):
        out = out.with_columns(
            _rolling_zscore(
                "residual", rolling_window,
                min_periods=max(3, int(round(rolling_window * config.zscore.min_periods_ratio))),
                min_std=numerical.min_residual_std,
            ).alias(f"residual_zscore_rolling_{rolling_window}")
        )
    out = out.with_columns(
        pl.col(f"residual_zscore_rolling_{primary}").alias("residual_zscore_rolling")
    )

    # Trailing volatility of log returns, for the conditioning studies. Causal.
    vol_window = config.conditioning.volatility_window
    out = out.with_columns(
        pl.col("log_price").diff().alias("log_return")
    ).with_columns(
        pl.col("log_return")
        .rolling_std(window_size=vol_window, min_samples=vol_window, ddof=1)
        .alias("trailing_volatility")
    )

    # Warm-up and degenerate windows come out of numpy as NaN. Polars treats
    # NaN and null as different things, and `is_not_null()` keeps NaN, so every
    # downstream `filter(is_not_null())` would silently retain the first N-1
    # rows -- inflating the reported fit count and, because NaN sorts high,
    # dragging every quantile edge upward. Missing is represented as null
    # throughout the feature table, matching `trailing_volatility`.
    out = out.with_columns(
        pl.col(name).fill_nan(None)
        for name, dtype in out.schema.items()
        if dtype in (pl.Float64, pl.Float32)
    )

    diagnostics = {
        "timeframe": timeframe,
        "window": window,
        "bars_in": frame.height,
        "warmup_bars": fit.warmup_bars,
        "valid_fits": int(np.isfinite(fit.residual).sum()),
        "degenerate_windows": fit.degenerate_windows,
        "nan_zscore_fit": int(np.isnan(zscore_fit).sum()),
        "price_transform": model.price_transform,
        "price_column": price_column,
        "zscore_rolling_primary_window": primary,
        "notes": fit.notes,
    }
    if fit.degenerate_windows:
        LOGGER.warning(
            "[%s w=%d] %d degenerate window(s) produced NaN rather than an infinite "
            "Z-score", timeframe, window, fit.degenerate_windows,
        )
    return out, diagnostics


def _resolve_price_column(
    model: RegressionModelConfig, available: list[str]
) -> str:
    """Bar column supplying the configured research price."""
    mapping = {
        "mid": {"close": "close", "open": "open"},
        "bid": {"close": "last_bid", "open": "first_bid"},
        "ask": {"close": "last_ask", "open": "first_ask"},
    }
    column = mapping[model.price_source].get(model.price_column)
    if column is None:
        raise KeyError(
            f"price_source={model.price_source!r} has no {model.price_column!r} column"
        )
    if column not in available:
        raise KeyError(
            f"Bars have no {column!r} column (price_source={model.price_source}); "
            f"available: {available}"
        )
    return column


def _rolling_zscore(
    column: str, window: int, *, min_periods: int, min_std: float
) -> pl.Expr:
    """Causal rolling Z-score of *column* over its own past values.

    The window ends at ``t`` inclusive, which is causal because the residual at
    ``t`` is itself known at ``t``. A near-zero rolling standard deviation
    yields null, never an enormous Z.
    """
    mean = pl.col(column).rolling_mean(window_size=window, min_samples=min_periods)
    std = pl.col(column).rolling_std(window_size=window, min_samples=min_periods, ddof=1)
    return (
        pl.when(std > min_std)
        .then((pl.col(column) - mean) / std)
        .otherwise(None)
    )
