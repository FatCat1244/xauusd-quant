"""Backward-looking rolling statistics.

Every statistic here obeys one rule: the value at ``t`` may use only
:math:`X_{t-N+1}, \\ldots, X_t`. Nothing from after ``t`` ever enters.

``rolling.include_current_bar`` controls whether ``X_t`` itself is included:

* ``True`` (default) — window is ``[t-N+1, t]``. Causal, and correct when the
  feature describes the state *as of* the close of bar ``t``.
* ``False`` — window is ``[t-N, t-1]``, i.e. the whole thing shifted one bar.
  Use this when the feature must be knowable *before* bar ``t`` forms, which is
  what a decision taken at the open of ``t`` would require.

Both are causal; they differ in what "now" means. Which one was used is
recorded in :class:`RollingSpec` and written into every report, because the
distinction quietly changes results by one bar and is a classic source of
inflated backtests later on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import polars as pl

from .config import ResearchConfig

__all__ = [
    "RollingSpec",
    "rolling_autocorrelation",
    "rolling_frame",
    "rolling_kurtosis",
    "rolling_mean",
    "rolling_skewness",
    "rolling_statistics",
    "rolling_variance",
]


@dataclass(frozen=True, slots=True)
class RollingSpec:
    """How a set of rolling features was computed."""

    window: int
    min_periods: int
    include_current_bar: bool
    lags: tuple[int, ...]

    @property
    def window_description(self) -> str:
        return "[t-N+1, t]" if self.include_current_bar else "[t-N, t-1]"

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "window_description": self.window_description,
            "causality": (
                "Backward-looking only; no value at or after t+1 is used."
                if self.include_current_bar
                else "Backward-looking and lagged one bar; knowable before bar t forms."
            ),
        }


def _causal(expr: pl.Expr, *, include_current_bar: bool) -> pl.Expr:
    """Shift a rolling expression by one bar when the current bar is excluded."""
    return expr if include_current_bar else expr.shift(1)


def rolling_mean(
    values: pl.Expr, window: int, *, min_periods: int | None = None,
    include_current_bar: bool = True,
) -> pl.Expr:
    """Rolling mean over the trailing window."""
    return _causal(
        values.rolling_mean(window_size=window, min_samples=min_periods or window),
        include_current_bar=include_current_bar,
    )


def rolling_variance(
    values: pl.Expr, window: int, *, min_periods: int | None = None,
    include_current_bar: bool = True, ddof: int = 1,
) -> pl.Expr:
    """Rolling sample variance over the trailing window."""
    return _causal(
        values.rolling_var(window_size=window, min_samples=min_periods or window, ddof=ddof),
        include_current_bar=include_current_bar,
    )


def rolling_skewness(
    values: pl.Series, window: int, *, min_periods: int | None = None,
    include_current_bar: bool = True,
) -> np.ndarray:
    """Rolling sample skewness.

    Computed with NumPy strides rather than a Polars expression because Polars
    has no rolling third moment; the windowing is identical.
    """
    return _rolling_moment(values, window, min_periods, include_current_bar, order=3)


def rolling_kurtosis(
    values: pl.Series, window: int, *, min_periods: int | None = None,
    include_current_bar: bool = True,
) -> np.ndarray:
    """Rolling excess kurtosis (Gaussian = 0)."""
    return _rolling_moment(values, window, min_periods, include_current_bar, order=4)


def rolling_autocorrelation(
    values: pl.Series, window: int, lag: int = 1, *, min_periods: int | None = None,
    include_current_bar: bool = True,
) -> np.ndarray:
    r"""Rolling lag-``k`` autocorrelation within each trailing window.

    The correlation at ``t`` uses only the window ending at ``t`` (or ``t-1``
    when the current bar is excluded), so it stays causal.
    """
    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}")
    array = values.to_numpy().astype(np.float64, copy=False)
    n = array.size
    need = min_periods or window
    out = np.full(n, np.nan, dtype=np.float64)
    if window <= lag + 1 or n < window:
        return _shift(out, include_current_bar)

    windows = _sliding(array, window)
    finite = _sliding(np.isfinite(array).astype(np.int32), window).sum(axis=1)
    complete = finite == window

    rows_per_chunk = max(1, _WINDOW_CHUNK_ELEMENTS // window)
    for start in range(0, windows.shape[0], rows_per_chunk):
        stop = min(start + rows_per_chunk, windows.shape[0])
        mask = complete[start:stop]
        if not mask.any():
            continue
        block = np.ascontiguousarray(windows[start:stop][mask])
        a, b = block[:, lag:], block[:, :-lag]
        a_centred = a - a.mean(axis=1, keepdims=True)
        b_centred = b - b.mean(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            numerator = np.sum(a_centred * b_centred, axis=1)
            denominator = np.sqrt(
                np.sum(a_centred**2, axis=1) * np.sum(b_centred**2, axis=1)
            )
            rho = numerator / denominator
        rho[denominator <= 0] = np.nan
        out[window - 1 + start : window - 1 + stop][mask] = rho

    partial = np.flatnonzero((~complete) & (finite >= max(need, lag + 2)))
    for index in partial:
        chunk = windows[index]
        values_only = chunk[np.isfinite(chunk)]
        if values_only.size < lag + 2:
            continue
        a, b = values_only[lag:], values_only[:-lag]
        if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
            continue
        out[window - 1 + index] = float(np.corrcoef(a, b)[0, 1])
    return _shift(out, include_current_bar)


def rolling_statistics(
    frame: pl.DataFrame,
    *,
    column: str,
    window: int,
    config: ResearchConfig,
    prefix: str | None = None,
) -> tuple[pl.DataFrame, RollingSpec]:
    """Attach the full configured set of rolling features for one window.

    Returns the frame plus the :class:`RollingSpec` describing exactly how the
    features were windowed, so the report can state it rather than imply it.
    """
    cfg = config.rolling
    include_current = cfg.include_current_bar
    min_periods = max(2, int(round(window * cfg.min_periods_fraction)))
    tag = prefix or f"roll{window}"
    spec = RollingSpec(
        window=window,
        min_periods=min_periods,
        include_current_bar=include_current,
        lags=tuple(cfg.autocorr_lags),
    )

    series = frame[column]
    out = frame.with_columns(
        rolling_mean(pl.col(column), window, min_periods=min_periods,
                     include_current_bar=include_current).alias(f"{tag}_mean"),
        rolling_variance(pl.col(column), window, min_periods=min_periods,
                         include_current_bar=include_current).alias(f"{tag}_var"),
        _causal(
            pl.col(column).rolling_std(window_size=window, min_samples=min_periods, ddof=1),
            include_current_bar=include_current,
        ).alias(f"{tag}_volatility"),
    )
    out = out.with_columns(
        pl.Series(f"{tag}_skew",
                  rolling_skewness(series, window, min_periods=min_periods,
                                   include_current_bar=include_current)),
        pl.Series(f"{tag}_excess_kurtosis",
                  rolling_kurtosis(series, window, min_periods=min_periods,
                                   include_current_bar=include_current)),
    )
    for lag in cfg.autocorr_lags:
        out = out.with_columns(
            pl.Series(
                f"{tag}_acf{lag}",
                rolling_autocorrelation(series, window, lag=int(lag),
                                        min_periods=min_periods,
                                        include_current_bar=include_current),
            )
        )
    return out, spec


def rolling_frame(
    frame: pl.DataFrame, *, column: str, config: ResearchConfig,
    time_column: str = "timestamp",
) -> tuple[pl.DataFrame, list[RollingSpec]]:
    """Rolling features for every configured window, in one frame."""
    out = frame
    specs: list[RollingSpec] = []
    for window in config.rolling.windows:
        out, spec = rolling_statistics(out, column=column, window=int(window), config=config)
        specs.append(spec)
    keep = [time_column, column] + [
        c for c in out.columns if c.startswith("roll")
    ]
    return out.select([c for c in keep if c in out.columns]), specs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
#: Elements per chunk of the sliding-window matrix. Caps the temporary at
#: roughly 64 MB per array so memory does not scale with series length.
_WINDOW_CHUNK_ELEMENTS = 8_000_000


def _sliding(array: np.ndarray, window: int) -> np.ndarray:
    """Read-only (n - window + 1, window) view of the trailing windows."""
    from numpy.lib.stride_tricks import sliding_window_view

    return sliding_window_view(array, window)


def _standardised_moments(block: np.ndarray, order: int) -> np.ndarray:
    """Bias-corrected skewness or excess kurtosis for each row of *block*.

    Matches ``scipy.stats.skew(..., bias=False)`` and
    ``scipy.stats.kurtosis(..., fisher=True, bias=False)``.
    """
    n = block.shape[1]
    centred = block - block.mean(axis=1, keepdims=True)
    m2 = np.mean(centred**2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        if order == 3:
            m3 = np.mean(centred**3, axis=1)
            g1 = m3 / m2**1.5
            out = g1 * np.sqrt(n * (n - 1.0)) / (n - 2.0)
        else:
            m4 = np.mean(centred**4, axis=1)
            g2 = m4 / m2**2 - 3.0
            out = ((n + 1.0) * g2 + 6.0) * (n - 1.0) / ((n - 2.0) * (n - 3.0))
    out[m2 <= 0] = np.nan
    return out


def _rolling_moment(
    values: pl.Series, window: int, min_periods: int | None,
    include_current_bar: bool, *, order: int,
) -> np.ndarray:
    """Rolling third/fourth standardised moment, vectorised."""
    array = values.to_numpy().astype(np.float64, copy=False)
    n = array.size
    need = min_periods or window
    out = np.full(n, np.nan, dtype=np.float64)
    floor = 3 if order == 3 else 4
    if n < window or window < floor:
        return _shift(out, include_current_bar)

    windows = _sliding(array, window)
    finite = _sliding(np.isfinite(array).astype(np.int32), window).sum(axis=1)
    complete = finite == window

    rows_per_chunk = max(1, _WINDOW_CHUNK_ELEMENTS // window)
    for start in range(0, windows.shape[0], rows_per_chunk):
        stop = min(start + rows_per_chunk, windows.shape[0])
        mask = complete[start:stop]
        if not mask.any():
            continue
        block = np.ascontiguousarray(windows[start:stop][mask])
        out[window - 1 + start : window - 1 + stop][mask] = _standardised_moments(
            block, order
        )

    # Windows holding a null keep the original semantics: drop the non-finite
    # values and compute on what is left, provided enough of them remain.
    partial = np.flatnonzero((~complete) & (finite >= max(need, floor)))
    if partial.size:
        from scipy import stats

        fn = (
            (lambda x: stats.skew(x, bias=False))
            if order == 3
            else (lambda x: stats.kurtosis(x, fisher=True, bias=False))
        )
        for index in partial:
            chunk = windows[index]
            values_only = chunk[np.isfinite(chunk)]
            if values_only.size < floor or np.std(values_only) == 0:
                continue
            out[window - 1 + index] = float(fn(values_only))
    return _shift(out, include_current_bar)


def _shift(array: np.ndarray, include_current_bar: bool) -> np.ndarray:
    """Lag the whole series one bar when the current bar must be excluded."""
    if include_current_bar:
        return array
    shifted = np.full_like(array, np.nan)
    shifted[1:] = array[:-1]
    return shifted
