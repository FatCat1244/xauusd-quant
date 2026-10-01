r"""Compute every registered feature family from one bar series (Prompt #8, Step 4).

One code path for the real series and for the pipeline null controls: a
:class:`BarSeries` carries the prices (a null has a synthetic close path and
no intrabar highs / lows), the exogenous spread and activity, and the
timestamps. A :class:`SourceProvider` supplies what earlier layers compute:
the real series reads the versioned Prompt #3 regression store and the stored
Prompt #5 / #6 / #7 feature sets; a null runs the same engines on its own path.

Every output at bar ``t`` is a function of bars ``<= t`` only. Rolling windows
are trailing and complete (a partial window is NaN, never a shorter window),
EWMA recursions only look back, percentiles rank against the trailing window,
and nothing is normalised by a statistic of the whole sample. The leakage tests
append wild future bars to a prefix and require every earlier value unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import polars as pl
from scipy.signal import lfilter

from ..models.ornstein_uhlenbeck import rolling_ou
from ..regimes.dataset import _rolling_percentile
from .factory_config import FeatureFactoryConfig

__all__ = [
    "BarSeries",
    "SourceProvider",
    "autocorrelation_features",
    "microstructure_features",
    "ou_features",
    "prefix_sum",
    "returns_features",
    "rolling_mean",
    "time_features",
    "trailing_std",
    "volatility_features",
]

_LN2 = math.log(2.0)


@dataclass
class BarSeries:
    """One source's bars, time-ordered, one row per bar (real or a null path)."""

    name: str
    timeframe: str
    timestamps: pl.Series
    close: np.ndarray
    bar_seconds: float
    open: np.ndarray | None = None
    high: np.ndarray | None = None
    low: np.ndarray | None = None
    median_spread: np.ndarray | None = None
    tick_count: np.ndarray | None = None
    missing_slots: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return int(self.close.size)

    def log_returns(self) -> np.ndarray:
        lp = np.log(self.close)
        r = np.full(lp.size, np.nan)
        r[1:] = np.diff(lp)
        return r


class SourceProvider(Protocol):
    """Earlier-layer outputs for one source, aligned to its bars."""

    def regression(self, window: int) -> dict[str, np.ndarray]:
        """slope, r_squared, residual, residual_std_fit, residual_zscore_rolling, sigma."""
        ...

    def fft(self, series: str, window: int) -> dict[str, np.ndarray] | None:
        """spectral_entropy, flatness, top3 share, low/high share, period_1, centroid_norm."""
        ...

    def wavelet(self, series: str, window: int) -> dict[str, np.ndarray] | None:
        """wavelet entropy, fast/slow log ratio, dominant period, run length, top3, drift."""
        ...

    def regime(self) -> dict[str, np.ndarray] | None:
        """Filtered state probabilities, entropy, confidence, age, next-state, leave."""
        ...


# ---------------------------------------------------------------------------
# Causal rolling primitives (NaN-aware: a window with any NaN is NaN)
# ---------------------------------------------------------------------------
def prefix_sum(values: np.ndarray, width: int) -> np.ndarray:
    """Sum of the last *width* values; NaN if any is NaN or fewer than *width* exist."""
    x = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(x)
    total = np.concatenate(([0.0], np.cumsum(np.where(finite, x, 0.0))))
    bad = np.concatenate(([0], np.cumsum(~finite, dtype=np.int64)))
    out = np.full(x.size, np.nan)
    if 0 < width <= x.size:
        span = total[width:] - total[:-width]
        clean = (bad[width:] - bad[:-width]) == 0
        out[width - 1:] = np.where(clean, span, np.nan)
    return out


def rolling_mean(values: np.ndarray, width: int) -> np.ndarray:
    return prefix_sum(values, width) / float(width)


def _centered(values: np.ndarray) -> tuple[np.ndarray, float]:
    """Values minus a constant to keep the moment sums well conditioned.

    Subtracting one constant changes no moment about the window mean. The
    constant is the *first* finite value, so it never depends on later bars
    (a sample median would, and would break bit-for-bit prefix invariance).
    """
    x = np.asarray(values, dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(x))
    shift = float(x[finite[0]]) if finite.size else 0.0
    return x - shift, shift


def trailing_std(values: np.ndarray, width: int, ddof: int = 1) -> np.ndarray:
    """Rolling standard deviation over the last *width* values (float64 moment sums)."""
    x, _ = _centered(values)
    s1 = prefix_sum(x, width)
    s2 = prefix_sum(x * x, width)
    with np.errstate(invalid="ignore"):
        var = (s2 - s1 * s1 / width) / (width - ddof)
    var = np.where(var < 0, 0.0, var)
    return np.sqrt(var)


def _rolling_moments(values: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling population skewness and excess kurtosis."""
    x, _ = _centered(values)
    s = [prefix_sum(x ** p, width) / width for p in (1, 2, 3, 4)]
    m1, e2, e3, e4 = s
    with np.errstate(invalid="ignore", divide="ignore"):
        m2 = e2 - m1 ** 2
        m3 = e3 - 3 * m1 * e2 + 2 * m1 ** 3
        m4 = e4 - 4 * m1 * e3 + 6 * m1 ** 2 * e2 - 3 * m1 ** 4
        ok = m2 > 1e-30
        skew = np.where(ok, m3 / np.where(ok, m2, 1.0) ** 1.5, np.nan)
        kurt = np.where(ok, m4 / np.where(ok, m2, 1.0) ** 2 - 3.0, np.nan)
    return skew, kurt


def _rolling_extreme(values: np.ndarray, width: int, kind: str) -> np.ndarray:
    s = pl.Series("x", np.asarray(values, dtype=np.float64)).fill_nan(None)
    rolled = (s.rolling_max(window_size=width, min_samples=width) if kind == "max"
              else s.rolling_min(window_size=width, min_samples=width))
    return rolled.cast(pl.Float64).fill_null(np.nan).to_numpy()


def _rolling_corr(a: np.ndarray, b: np.ndarray, width: int) -> np.ndarray:
    """corr over the last *width* complete pairs (both finite), else NaN."""
    frame = pl.DataFrame({"a": pl.Series(np.asarray(a, dtype=np.float64)).fill_nan(None),
                          "b": pl.Series(np.asarray(b, dtype=np.float64)).fill_nan(None)})
    out = frame.select(pl.rolling_corr(pl.col("a"), pl.col("b"), window_size=width,
                                       min_samples=width)).to_series()
    arr = np.array(out.cast(pl.Float64).fill_null(np.nan).to_numpy(), dtype=np.float64)
    arr[~np.isfinite(arr)] = np.nan
    return arr


def _lagged(values: np.ndarray, k: int) -> np.ndarray:
    out = np.full(values.size, np.nan)
    if k < values.size:
        out[k:] = values[:-k] if k > 0 else values
    return out


def _half_log(v: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v > 0, 0.5 * np.log(v), np.nan)


def _ewma(values: np.ndarray, lam: float) -> np.ndarray:
    """Causal EWMA v_t = lam v_(t-1) + (1 - lam) x_t, seeded by the first span's mean."""
    x = np.asarray(values, dtype=np.float64)
    out = np.full(x.size, np.nan)
    finite = np.flatnonzero(np.isfinite(x))
    if finite.size == 0:
        return out
    start = int(finite[0])
    span = int(math.ceil(3.0 / (1.0 - lam)))
    seg = x[start:]
    if np.isnan(seg).any():              # bars have no gaps in r after the first; be safe
        seg = np.where(np.isfinite(seg), seg, 0.0)
    if seg.size <= span:
        return out
    seed = float(np.mean(seg[:span]))
    y, _ = lfilter([1.0 - lam], [1.0, -lam], seg[span:], zi=[lam * seed])
    out[start + span:] = y
    return out


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------
def returns_features(bs: BarSeries, cfg: FeatureFactoryConfig) -> dict[str, np.ndarray]:
    f = cfg.families.returns
    lp = np.log(bs.close)
    r = bs.log_returns()
    sigma = trailing_std(r, f.volatility_window)
    out: dict[str, np.ndarray] = {}
    for h in f.horizons:
        out[f"ret_{h}"] = lp - _lagged(lp, h)
    for h in f.z_horizons:
        move = lp - _lagged(lp, h)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"ret_z_{h}"] = np.where(sigma > 0, move / (sigma * math.sqrt(h)), np.nan)
    for w in f.moment_windows:
        skew, kurt = _rolling_moments(r, w)
        out[f"ret_skew_{w}"] = skew
        out[f"ret_kurt_{w}"] = kurt
    for w in f.range_windows:
        if bs.high is None or bs.low is None:
            out[f"range_pos_{w}"] = np.full(bs.size, np.nan)
            continue
        hi = _rolling_extreme(np.log(bs.high), w, "max")
        lo = _rolling_extreme(np.log(bs.low), w, "min")
        with np.errstate(invalid="ignore", divide="ignore"):
            span = hi - lo
            out[f"range_pos_{w}"] = np.where(span > 0, (lp - lo) / span, np.nan)
    return out


def volatility_features(bs: BarSeries, cfg: FeatureFactoryConfig) -> dict[str, np.ndarray]:
    f = cfg.families.volatility
    r = bs.log_returns()
    r2 = r * r
    out: dict[str, np.ndarray] = {}
    for w in f.rv_windows:
        out[f"log_rv_{w}"] = _half_log(rolling_mean(r2, w))
    for lam in f.ewma_lambdas:
        out[f"log_ewma_vol_{int(round(lam * 100))}"] = _half_log(_ewma(r2, lam))
    rv20 = out.get("log_rv_20")
    if rv20 is None:
        rv20 = _half_log(rolling_mean(r2, 20))
    out[f"vol_change_{f.change_lag}"] = rv20 - _lagged(rv20, f.change_lag)
    a, b = f.ratio_pair
    out[f"vol_ratio_{a}_{b}"] = out[f"log_rv_{a}"] - out[f"log_rv_{b}"]
    out["rv_percentile"] = _rolling_percentile(rv20, cfg.percentile_window(bs.bar_seconds))
    w = f.range_window
    if bs.high is not None and bs.low is not None and bs.open is not None:
        hl = np.log(bs.high) - np.log(bs.low)
        co = np.log(bs.close) - np.log(bs.open)
        out[f"log_parkinson_{w}"] = _half_log(rolling_mean(hl * hl / (4.0 * _LN2), w))
        gk = 0.5 * hl * hl - (2.0 * _LN2 - 1.0) * co * co
        out[f"log_garman_klass_{w}"] = _half_log(rolling_mean(gk, w))
    else:
        out[f"log_parkinson_{w}"] = np.full(bs.size, np.nan)
        out[f"log_garman_klass_{w}"] = np.full(bs.size, np.nan)
    return out


def autocorrelation_features(bs: BarSeries, cfg: FeatureFactoryConfig) -> dict[str, np.ndarray]:
    f = cfg.families.autocorrelation
    r = bs.log_returns()
    out: dict[str, np.ndarray] = {}
    for k in f.return_lags:
        out[f"acf_lag{k}_{f.window}"] = _rolling_corr(r, _lagged(r, k), f.window)
    for w in f.lag1_windows:
        out[f"acf_lag1_{w}"] = _rolling_corr(r, _lagged(r, 1), w)
    a = np.abs(r)
    out[f"abs_acf_lag1_{f.window}"] = _rolling_corr(a, _lagged(a, 1), f.window)
    s = r * r
    out[f"sq_acf_lag1_{f.window}"] = _rolling_corr(s, _lagged(s, 1), f.window)
    return out


def regression_features(provider: SourceProvider, cfg: FeatureFactoryConfig
                        ) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for n in cfg.families.regression.windows:
        cols = provider.regression(n)
        sigma = cols["sigma"]
        with np.errstate(invalid="ignore", divide="ignore"):
            ok = sigma > 0
            out[f"reg_slope_vol_{n}"] = np.where(ok, cols["slope"] / np.where(ok, sigma, 1), np.nan)
            out[f"reg_r2_{n}"] = cols["r_squared"]
            out[f"reg_resid_z_{n}"] = cols["residual_zscore_rolling"]
            out[f"reg_resid_vol_{n}"] = np.where(ok, cols["residual"] / np.where(ok, sigma, 1),
                                                 np.nan)
            ratio = np.where(ok, cols["residual_std_fit"] / np.where(ok, sigma, 1), np.nan)
            out[f"reg_resid_std_ratio_{n}"] = np.where(ratio > 0, np.log(ratio), np.nan)
    return out


def ou_features(residual: np.ndarray, cfg: FeatureFactoryConfig, *, rules: Any,
                chunk_rows: int = 16_384) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """OU features of the residual for every window, and the equilibrium series per window.

    Returns (features, extras); extras holds ``ou_mu_<M>`` for the OU-relative
    target (the equilibrium known at ``t``), never a feature column.
    """
    f = cfg.families.ou
    out: dict[str, np.ndarray] = {}
    extras: dict[str, np.ndarray] = {}
    for m in f.windows:
        fit = rolling_ou(residual, m, rules=rules, chunk_rows=chunk_rows)
        mp = fit.mapping
        valid = mp.valid & np.isfinite(mp.half_life_bars) & (mp.half_life_bars > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"ou_log_half_life_{m}"] = np.where(valid, np.log(np.where(valid, mp.half_life_bars,
                                                                           1.0)), np.nan)
            out[f"ou_theta_{m}"] = np.where(valid, mp.theta, np.nan)
            out[f"ou_zscore_{m}"] = fit.zscore
            if m == f.full_window:
                out[f"ou_b_{m}"] = fit.fits.b
                sd = fit.fits.window_std
                out[f"ou_mu_norm_{m}"] = np.where(valid & (sd > 0), mp.mu / np.where(sd > 0, sd,
                                                                                    1.0), np.nan)
                out[f"ou_fit_r2_{m}"] = fit.fits.r_squared
                defined = np.isfinite(fit.fits.b)
                out[f"ou_valid_{m}"] = np.where(defined, valid.astype(np.float64), np.nan)
                out[f"ou_std_ratio_{m}"] = fit.std_ratio
                out[f"ou_innov_z_{m}"] = fit.innovation_standardized
                h = f.decay_horizon
                out[f"ou_decay_ratio{h}_{m}"] = np.where(valid, fit.fits.b ** h, np.nan)
        extras[f"ou_mu_{m}"] = np.where(valid, mp.mu, np.nan)
        extras[f"ou_innovation_{m}"] = fit.innovation
        del fit
    for arr in out.values():
        arr[~np.isfinite(arr)] = np.nan
    return out, extras


def fft_features(provider: SourceProvider, cfg: FeatureFactoryConfig,
                 bar_seconds: float) -> dict[str, np.ndarray]:
    f = cfg.families.fft
    out: dict[str, np.ndarray] = {}

    def high_low(cols: dict[str, np.ndarray]) -> np.ndarray:
        hi, lo = cols["high_power_share"], cols["low_power_share"]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where((hi > 0) & (lo > 0), np.log(hi) - np.log(lo), np.nan)

    for n in f.windows:
        cols = provider.fft(f.series, n)
        if cols is None:
            continue
        out[f"fft_entropy_{n}"] = cols["spectral_entropy"]
        out[f"fft_high_low_{n}"] = high_low(cols)
        if n == f.full_window:
            out[f"fft_flatness_{n}"] = cols["spectral_flatness"]
            out[f"fft_top3_share_{n}"] = cols["top3_power_share"]
            out[f"fft_dominant_period_{n}"] = cols["fft_period_bars_1"]
            out[f"fft_centroid_{n}"] = cols["spectral_centroid_norm"]
    cols = provider.fft("regression_residual", f.residual_window)
    if cols is not None:
        out[f"fft_resid_entropy_{f.residual_window}"] = cols["spectral_entropy"]
        out[f"fft_resid_high_low_{f.residual_window}"] = high_low(cols)
    if bar_seconds >= 900:
        cols = provider.fft("abs_ou_innovation", f.abs_innovation_window)
        if cols is not None:
            out[f"fft_abs_eta_entropy_{f.abs_innovation_window}"] = cols["spectral_entropy"]
            out[f"fft_abs_eta_dominant_period_{f.abs_innovation_window}"] = \
                cols["fft_period_bars_1"]
    return out


def wavelet_features(provider: SourceProvider, cfg: FeatureFactoryConfig
                     ) -> dict[str, np.ndarray]:
    f = cfg.families.wavelet
    out: dict[str, np.ndarray] = {}
    for n in f.windows:
        cols = provider.wavelet(f.series, n)
        if cols is None:
            continue
        out[f"wav_entropy_{n}"] = cols["wavelet_entropy"]
        out[f"wav_fast_slow_{n}"] = cols["wavelet_fast_slow_log_ratio"]
        if n == f.full_window:
            out[f"wav_dominant_period_{n}"] = cols["wavelet_dominant_period_bars"]
            out[f"wav_run_length_{n}"] = cols["wavelet_dominant_run_length"]
            out[f"wav_top3_share_{n}"] = cols["wavelet_top3_scale_share"]
            out[f"wav_scale_drift_{n}"] = cols["wavelet_scale_drift"]
    return out


def regime_features(provider: SourceProvider, cfg: FeatureFactoryConfig,
                    size: int) -> dict[str, np.ndarray]:
    k = cfg.families.regime.states
    cols = provider.regime()
    names = ([f"regime_p{j}" for j in range(k)]
             + ["regime_entropy", "regime_confidence", "regime_age"]
             + [f"regime_next_p{j}" for j in range(k)] + ["regime_leave_prob"])
    if cols is None:
        return {name: np.full(size, np.nan) for name in names}
    return {name: cols[name] for name in names}


def microstructure_features(bs: BarSeries, cfg: FeatureFactoryConfig) -> dict[str, np.ndarray]:
    w = cfg.families.microstructure.change_window
    pct = cfg.percentile_window(bs.bar_seconds)
    n = bs.size
    if bs.median_spread is None or bs.tick_count is None:
        return {name: np.full(n, np.nan) for name in
                ("spread_rel", "spread_percentile", f"spread_change_{w}", "log_tick_count",
                 "activity_percentile", f"activity_change_{w}")}
    spread = np.asarray(bs.median_spread, dtype=np.float64)
    ticks = np.asarray(bs.tick_count, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where(bs.close > 0, 1e4 * spread / bs.close, np.nan)
        mean_spread = rolling_mean(spread, w)
        change = np.where((spread > 0) & (mean_spread > 0), np.log(spread / mean_spread), np.nan)
        act = 1.0 + ticks
        act_change = np.log(act / rolling_mean(act, w))
    return {
        "spread_rel": rel,
        "spread_percentile": _rolling_percentile(spread, pct),
        f"spread_change_{w}": change,
        "log_tick_count": np.log1p(ticks),
        "activity_percentile": _rolling_percentile(ticks, pct),
        f"activity_change_{w}": act_change,
    }


def time_features(bs: BarSeries, cfg: FeatureFactoryConfig,
                  research: Any | None) -> dict[str, np.ndarray]:
    ts = pl.DataFrame({"timestamp": bs.timestamps})
    parts = ts.select(
        (pl.col("timestamp").dt.hour().cast(pl.Float64)
         + pl.col("timestamp").dt.minute().cast(pl.Float64) / 60.0).alias("hour"),
        (pl.col("timestamp").dt.weekday().cast(pl.Float64) - 1.0).alias("weekday"))
    hour = parts["hour"].to_numpy()
    dow = parts["weekday"].to_numpy()
    out = {
        "tod_sin": np.sin(2 * np.pi * hour / 24.0),
        "tod_cos": np.cos(2 * np.pi * hour / 24.0),
        "dow_sin": np.sin(2 * np.pi * dow / 5.0),
        "dow_cos": np.cos(2 * np.pi * dow / 5.0),
    }
    if cfg.families.time.session_indicators and research is not None:
        from ..research.intraday import assign_sessions

        labels = ts.select(assign_sessions(pl.col("timestamp"), research).alias("s"))["s"]
        for session in ("off_hours", "asia", "london", "london_ny_overlap", "new_york"):
            out[f"session_{session}"] = (labels == session).cast(pl.Float64).to_numpy()
    return out
