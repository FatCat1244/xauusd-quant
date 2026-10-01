r"""Wavelet research that is not per-bar features: controls, boundaries, slices, FFT.

* **Positive controls** (Steps 37-39) - known signals through the engines:
  a sinusoid active only part of the time (the wavelet must localise it,
  the rolling FFT smears it over its window), a chirp (the dominant scale
  must follow the changing period) and a slow cycle with a fast burst (the
  two must land in different bands).
* **Boundary study** (Step 60) - how the newest coefficient of a *windowed*
  transform depends on its padding mode, against the one-sided MODWT, which
  needs none. The default is chosen on synthetic accuracy and stability.
* **Offline slices** (Steps 7, 8, 15) - scalograms and ridges of whole
  slices, centred and therefore NON-CAUSAL, for visual understanding only.
  Slices are chosen by volatility and era, never by outcome.
* **Wavelet vs FFT** (Step 23) - are the wavelet features transformations of
  the Fourier features of the same window?
* **Cost** (Step 49) - throughput and per-update latency.
"""

from __future__ import annotations

import time
import tracemalloc
from typing import Any

import numpy as np
import polars as pl

from ..features.spectral import RollingSpectrum, rolling_spectrum
from ..features.spectral_config import SpectralConfig
from ..features.wavelet import (
    NON_CAUSAL_LABEL,
    OfflineScalogram,
    causal_modwt,
    cwt_scales,
    extract_ridges,
    max_level,
    modwt_filters,
    offline_cwt,
    windowed_dwt_edge,
)
from ..features.wavelet_causal import RollingWavelet, ic_feature_columns, rolling_wavelet
from ..features.wavelet_config import WaveletConfig
from ..features.wavelet_energy import band_layout
from .fft_analysis import window_spectra
from .ou_estimation import num
from .spectral_ic import FEATURES as FFT_FEATURES
from .spectral_ic import feature_columns as fft_feature_columns
from .spectral_nulls import SourceData
from .wavelet_predictiveness import _dominant_run

__all__ = [
    "benchmark_wavelet",
    "boundary_study",
    "chirp_control",
    "dyadic_comparison",
    "fft_comparison",
    "localized_oscillation_control",
    "multiscale_control",
    "offline_ridges",
    "offline_scalogram",
    "select_slices",
]

_DB4_PERIOD = 1.4        # db4 nominal period of level j is ~1.4 * 2^j bars


def _half_max_interval(values: np.ndarray, baseline_end: int) -> tuple[int | None, int | None]:
    """First and last bar where *values* exceed half-way from the baseline to the peak."""
    v = np.asarray(values, dtype=np.float64)
    base = float(np.nanmedian(v[:baseline_end]))
    peak = float(np.nanmax(v))
    above = np.flatnonzero(v >= base + 0.5 * (peak - base))
    if above.size == 0:
        return None, None
    return int(above[0]), int(above[-1])


def _local_energy(values: np.ndarray, wavelet: str, levels: int) -> np.ndarray:
    """Causal local energy per detail level: mean of W_j^2 over the last 2^j bars."""
    coeffs = causal_modwt(values, modwt_filters(wavelet, levels))
    n = len(values)
    out = np.full((levels, n), np.nan)
    for j in range(levels):
        width = 2 ** (j + 1)
        csum = np.concatenate(([0.0], np.cumsum(coeffs.detail[j] ** 2)))
        out[j, width - 1:] = (csum[width:] - csum[:n - width + 1]) / width
        out[j, :coeffs.filters.lengths[j] - 1] = np.nan
    return out


# ---------------------------------------------------------------------------
# Positive controls
# ---------------------------------------------------------------------------
def localized_oscillation_control(config: WaveletConfig, spectral: SpectralConfig, *,
                                  n: int = 8192, period: float = 22.4, active: tuple[int, int] =
                                  (3000, 3600), noise_std: float = 0.3, seed: int = 11
                                  ) -> pl.DataFrame:
    """A cycle present only in *active*: where does each method say it is?

    Half-maximum interval of: the causal local energy of the band holding
    *period*; the trailing rolling-FFT power share of the matching bin; the
    offline CWT power at the matching period.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    amp = ((t >= active[0]) & (t < active[1])).astype(np.float64)
    x = amp * np.sin(2 * np.pi * t / period) + noise_std * rng.normal(size=n)
    window = config.representative_window
    wavelet = config.dwt.wavelet
    levels = max_level(window, wavelet, config.causal_features.min_level_coefficients)
    level = int(np.argmin(np.abs(_DB4_PERIOD * 2.0 ** np.arange(1, levels + 1) - period)))
    local = _local_energy(x, wavelet, levels)[level]
    ends = np.arange(window - 1, n)
    shares = window_spectra(x, ends, window, spectral)
    fft_bin = int(round(window / period)) - 1          # usable bins start at k = 1
    fft_power = np.full(n, np.nan)
    fft_power[ends] = shares[:, fft_bin]
    scales = cwt_scales(config.cwt.scales.min, config.cwt.scales.max, config.cwt.scales.count,
                        config.cwt.scales.mode)
    scal = offline_cwt(x, scales, wavelet=config.cwt.wavelet, bar_seconds=1.0)
    row = int(np.argmin(np.abs(scal.periods_bars - period)))
    rows = []
    for method, series, causal in (("causal_modwt_local_energy", local, True),
                                   (f"rolling_fft_share_N{window}", fft_power, True),
                                   ("offline_cwt_power", scal.power[row].astype(np.float64),
                                    False)):
        on, off = _half_max_interval(series, active[0] - 200)
        rows.append({"method": method, "causal": causal, "true_start": active[0],
                     "true_end": active[1] - 1, "detected_start": on, "detected_end": off,
                     "start_error": None if on is None else on - active[0],
                     "end_error": None if off is None else off - (active[1] - 1),
                     "width_error": None if on is None or off is None
                     else (off - on) - (active[1] - 1 - active[0]),
                     "label": None if causal else NON_CAUSAL_LABEL})
    return pl.DataFrame(rows, infer_schema_length=None)


def chirp_control(config: WaveletConfig, spectral: SpectralConfig, *, n: int = 8192,
                  periods: tuple[float, float] = (8.0, 128.0), noise_std: float = 0.1,
                  seed: int = 12) -> pl.DataFrame:
    """A cycle whose period grows exponentially: does each method follow it?

    Reported: correlation between the estimated and the true log period (and
    the RMSE of the log period) over the reliable part, for the offline CWT
    ridge, the causal local-energy centroid and the rolling-FFT dominant period.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    true_period = periods[0] * (periods[1] / periods[0]) ** (t / (n - 1))
    phase = np.cumsum(2 * np.pi / true_period)
    x = np.sin(phase) + noise_std * rng.normal(size=n)
    wavelet = config.dwt.wavelet
    window = config.representative_window
    levels = max_level(window, wavelet, config.causal_features.min_level_coefficients)
    local = _local_energy(x, wavelet, levels)
    level_periods = _DB4_PERIOD * 2.0 ** np.arange(1, levels + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        weights = local / np.nansum(local, axis=0)
        centroid = np.exp(np.nansum(weights * np.log(level_periods)[:, None], axis=0))
    centroid[np.isnan(local).any(axis=0)] = np.nan
    spec = rolling_spectrum(x, fft_window=window, config=spectral, bar_seconds=1.0)
    fft_period = np.where(spec.valid, spec.dominant_period_bars, np.nan)
    scales = cwt_scales(config.cwt.scales.min, config.cwt.scales.max, config.cwt.scales.count,
                        config.cwt.scales.mode)
    scal = offline_cwt(x, scales, wavelet=config.cwt.wavelet, bar_seconds=1.0)
    ridges = extract_ridges(scal, power_ratio=config.cwt.ridge_power_ratio)
    ridge_period = np.full(n, np.nan)
    if ridges:
        longest = max(ridges, key=lambda r: r.length)
        ridge_period[longest.start:longest.end + 1] = longest.periods_bars
    rows = []
    for method, estimate, causal in (("offline_cwt_ridge", ridge_period, False),
                                     ("causal_modwt_local_centroid", centroid, True),
                                     (f"rolling_fft_dominant_N{window}", fft_period, True)):
        ok = np.isfinite(estimate) & (true_period <= periods[1])
        if ok.sum() < 10:
            rows.append({"method": method, "causal": causal, "points": int(ok.sum())})
            continue
        a, b = np.log(estimate[ok]), np.log(true_period[ok])
        lag_fit = np.polyfit(b, a, 1)
        rows.append({"method": method, "causal": causal, "points": int(ok.sum()),
                     "log_period_correlation": float(np.corrcoef(a, b)[0, 1]),
                     "log_period_rmse": float(np.sqrt(np.mean((a - b) ** 2))),
                     "slope_vs_true": float(lag_fit[0]),
                     "label": None if causal else NON_CAUSAL_LABEL})
    return pl.DataFrame(rows, infer_schema_length=None)


def multiscale_control(config: WaveletConfig, *, n: int = 8192, slow_period: float = 128.0,
                       fast_period: float = 4.0, burst: tuple[int, int] = (4000, 4400),
                       noise_std: float = 0.2, seed: int = 13) -> pl.DataFrame:
    """A slow cycle throughout and a fast burst: are they told apart?"""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    inside = (t >= burst[0]) & (t < burst[1])
    x = (np.sin(2 * np.pi * t / slow_period) + inside * np.sin(2 * np.pi * t / fast_period)
         + noise_std * rng.normal(size=n))
    result = rolling_wavelet(x, window=config.representative_window, config=config,
                             bar_seconds=1.0)
    layout = result.layout
    slow_mask = layout.period_bars >= slow_period / 2
    fast_mask = layout.period_bars <= fast_period * 2
    local = _local_energy(x, config.dwt.wavelet, layout.levels)
    fast_local = np.nansum(local[fast_mask[:layout.levels]], axis=0)
    delay = layout.filter_lengths[0]
    during = inside & result.valid
    after = (t >= burst[1] + 4 * delay) & result.valid
    before = (t < burst[0]) & result.valid
    slow_share = np.nansum(result.shares[slow_mask], axis=0)
    scales = cwt_scales(config.cwt.scales.min, config.cwt.scales.max, config.cwt.scales.count,
                        config.cwt.scales.mode)
    scal = offline_cwt(x, scales, wavelet=config.cwt.wavelet, bar_seconds=1.0)
    fast_row = int(np.argmin(np.abs(scal.periods_bars - fast_period)))
    slow_row = int(np.argmin(np.abs(scal.periods_bars - slow_period)))
    cwt_fast = scal.power[fast_row].astype(np.float64)
    cwt_slow = scal.power[slow_row].astype(np.float64)
    mid = (t > 600) & (t < n - 600)
    return pl.DataFrame([
        {"check": "slow bands' median window share", "value": float(np.nanmedian(
            slow_share[before | after])), "causal": True},
        {"check": "fast local energy, burst / outside", "value": float(
            np.nanmedian(fast_local[during]) / np.nanmedian(fast_local[before | after])),
         "causal": True},
        {"check": "offline CWT fast power, burst / outside", "value": float(
            np.median(cwt_fast[inside & mid]) / np.median(cwt_fast[~inside & mid])),
         "causal": False},
        {"check": "offline CWT slow power, burst / outside", "value": float(
            np.median(cwt_slow[inside & mid]) / np.median(cwt_slow[~inside & mid])),
         "causal": False},
    ])


# ---------------------------------------------------------------------------
# Boundary study (Step 60)
# ---------------------------------------------------------------------------
def boundary_study(config: WaveletConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Accuracy and padding-sensitivity of latest-state energy estimates.

    Each trial: a sinusoid in one band (random level, period within 15 % of
    the band centre, random phase), amplitude constant or stepped (x4 or
    x1/4) 16-128 bars before the evaluation bar, plus noise. The target is
    the band energy of that signal at its *new* amplitude in steady state
    (a long stationary run through the same filters). Estimators:

    * ``causal_modwt`` - mean W_j^2 over the last 2^j one-sided coefficients;
    * ``windowed_dwt_<mode>`` - the last level-j coefficient of the window's
      decimated DWT with that padding mode (energy per bar).

    Returns per-trial rows and a summary: median |ln(estimate/target)|,
    median signed error (bias), and, for windowed estimates, the spread of
    one window's estimate across padding modes.
    """
    cfg = config.boundary
    rng = np.random.default_rng(cfg.seed)
    window = config.representative_window
    wavelet = config.dwt.wavelet
    levels = max_level(window, wavelet, config.causal_features.min_level_coefficients)
    filters = modwt_filters(wavelet, levels)
    rows: list[dict[str, Any]] = []
    for trial in range(cfg.trials):
        level = int(rng.integers(1, levels + 1))
        period = _DB4_PERIOD * 2 ** level * float(rng.uniform(0.85, 1.15))
        phase = float(rng.uniform(0, 2 * np.pi))
        old = float(rng.uniform(0.5, 2.0))
        step = trial % 2 == 1
        new = old * float(rng.choice([0.25, 4.0])) if step else old
        total = 2 * window
        t = np.arange(total)
        change = total - int(rng.integers(16, 129)) if step else total
        amp = np.where(t >= change, new, old)
        noise = 0.3 * rng.normal(size=total)
        x = amp * np.sin(2 * np.pi * t / period + phase) + noise
        steady = new * np.sin(2 * np.pi * np.arange(8 * window) / period + phase) \
            + 0.3 * rng.normal(size=8 * window)
        w_steady = causal_modwt(steady, filters).detail[level - 1][filters.lengths[level - 1]:]
        target = float(np.mean(w_steady ** 2))
        coeffs = causal_modwt(x, filters).detail[level - 1]
        estimates = {"causal_modwt": float(np.mean(coeffs[-2 ** level:] ** 2))}
        edge = x[-window:]
        for mode in cfg.padding_modes:
            estimates[f"windowed_dwt_{mode}"] = float(
                windowed_dwt_edge(edge, wavelet, levels, mode)[level - 1])
        for method, value in estimates.items():
            rows.append({"trial": trial, "level": level, "step": step, "method": method,
                         "target": target, "estimate": value,
                         "log_error": float(np.log(max(value, 1e-300) / target))})
    trials = pl.DataFrame(rows)
    windowed = trials.filter(pl.col("method").str.starts_with("windowed_dwt_"))
    spread = (windowed.group_by("trial").agg(
        (pl.col("log_error").max() - pl.col("log_error").min()).alias("padding_spread")))
    summary = (trials.group_by("method", "step")
               .agg(pl.len().alias("trials"),
                    pl.col("log_error").abs().median().alias("median_abs_log_error"),
                    pl.col("log_error").median().alias("median_log_error"),
                    pl.col("log_error").abs().quantile(0.9).alias("p90_abs_log_error"))
               .sort("step", "median_abs_log_error"))
    summary = summary.with_columns(
        pl.when(pl.col("method").str.starts_with("windowed_dwt_"))
        .then(pl.lit(num(spread["padding_spread"].median()))).otherwise(pl.lit(0.0))
        .alias("median_padding_spread"))
    return trials, summary


# ---------------------------------------------------------------------------
# Offline slices: scalograms and ridges (NON-CAUSAL)
# ---------------------------------------------------------------------------
def select_slices(source: SourceData, series: str, *, slice_bars: int, eras: int = 3
                  ) -> list[dict[str, Any]]:
    """Deterministic, outcome-free slices: per-era median volatility, and the extremes.

    Candidates are consecutive non-overlapping blocks of *slice_bars* bars in
    which the series is complete; each block's volatility is the standard
    deviation of its log returns.
    """
    values = source.columns[series]
    returns = source.columns["log_return"]
    n = values.size
    blocks = []
    for start in range(0, n - slice_bars + 1, slice_bars):
        seg = values[start:start + slice_bars]
        ret = returns[start:start + slice_bars]
        if np.isfinite(seg).all() and np.isfinite(ret).sum() > slice_bars // 2:
            blocks.append((start, float(np.nanstd(ret))))
    if not blocks:
        return []
    starts = np.array([b[0] for b in blocks])
    vols = np.array([b[1] for b in blocks])
    chosen: dict[str, int] = {}
    for e in range(eras):
        lo, hi = n * e // eras, n * (e + 1) // eras
        inside = np.flatnonzero((starts >= lo) & (starts < hi))
        if inside.size:
            med = np.median(vols[inside])
            chosen[f"era{e + 1}_median_volatility"] = int(
                starts[inside[np.argmin(np.abs(vols[inside] - med))]])
    chosen["highest_volatility"] = int(starts[np.argmax(vols)])
    chosen["lowest_volatility"] = int(starts[np.argmin(vols)])
    out = []
    for label, start in chosen.items():
        out.append({"slice": label, "start": start, "end": start + slice_bars - 1,
                    "first_timestamp": source.timestamps[start],
                    "last_timestamp": source.timestamps[start + slice_bars - 1],
                    "volatility": float(vols[np.flatnonzero(starts == start)[0]])})
    return out


def offline_scalogram(values: np.ndarray, config: WaveletConfig, *, bar_seconds: float,
                      wavelet: str | None = None) -> OfflineScalogram:
    """NON-CAUSAL scalogram of one slice (visualisation and ridge research only)."""
    scales = cwt_scales(config.cwt.scales.min, config.cwt.scales.max, config.cwt.scales.count,
                        config.cwt.scales.mode)
    return offline_cwt(values, scales, wavelet=wavelet or config.cwt.wavelet,
                       bar_seconds=bar_seconds)


def offline_ridges(sources: dict[str, SourceData], series: str, slices: list[dict[str, Any]],
                   config: WaveletConfig) -> pl.DataFrame:
    """Ridge statistics of every slice, for the real series and each null, per CWT wavelet."""
    wavelets = [config.cwt.wavelet] + ([config.cwt.compare_wavelet]
                                       if config.cwt.compare_wavelet else [])
    rows = []
    for spec in slices:
        start, end = spec["start"], spec["end"] + 1
        for name, source in sources.items():
            if series not in source.columns:
                continue
            values = source.columns[series][start:end]
            if not np.isfinite(values).all():
                continue
            for wavelet in wavelets:
                scal = offline_scalogram(values, config, bar_seconds=source.bar_seconds,
                                         wavelet=wavelet)
                ridges = extract_ridges(scal, power_ratio=config.cwt.ridge_power_ratio)
                cycles = np.array([r.cycles for r in ridges]) if ridges else np.zeros(0)
                drift = (np.array([abs(r.log_period_drift) * r.mean_period for r in ridges])
                         if ridges else np.zeros(0))
                rows.append({
                    "slice": spec["slice"], "source": name, "wavelet": wavelet,
                    "first_timestamp": spec["first_timestamp"], "bars": end - start,
                    "ridges": len(ridges),
                    "median_cycles": float(np.median(cycles)) if cycles.size else None,
                    "p90_cycles": float(np.quantile(cycles, 0.9)) if cycles.size else None,
                    "share_at_least_3_cycles": (float(np.mean(cycles >= 3))
                                                if cycles.size else None),
                    "median_abs_log_drift_per_cycle": (float(np.median(drift))
                                                       if drift.size else None),
                    "label": NON_CAUSAL_LABEL})
    return pl.DataFrame(rows, infer_schema_length=None)


# ---------------------------------------------------------------------------
# Wavelet vs FFT (Step 23)
# ---------------------------------------------------------------------------
_PAIRS: tuple[tuple[str, str], ...] = (
    ("wavelet_entropy", "fft_spectral_entropy"),
    ("wavelet_log_dominant_period", "fft_log_dominant_period"),
    ("wavelet_top3_scale_share", "fft_top3_power_share"),
    ("wavelet_dominant_energy_share", "fft_top1_power_share"),
    ("wavelet_log_run_length", "fft_log_run_length"),
    ("wavelet_log_centroid_period", "fft_spectral_centroid_norm"),
    ("wavelet_fast_slow_log_ratio", "fft_high_power_share"),
)


def _fft_columns(spectrum: RollingSpectrum, rows: np.ndarray) -> dict[str, np.ndarray]:
    out = {f"fft_{k}": v for k, v in fft_feature_columns(spectrum, FFT_FEATURES, rows).items()}
    run = _dominant_run(spectrum.dominant_bin.astype(np.int64), spectrum.valid)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["fft_log_run_length"] = np.log(np.minimum(run, spectrum.fft_window))[rows]
    return out


def fft_comparison(result: RollingWavelet, spectrum: RollingSpectrum, config: WaveletConfig, *,
                   source: str, max_rows: int = 200_000) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Paired correlations, and how well FFT features predict each wavelet feature.

    Determinism: each wavelet feature is regressed (OLS, standardised) on all
    Fourier features of the same window over the first half of the sample
    and scored on the second half; an out-of-sample R^2 near 1 means the
    wavelet feature is close to a function of the FFT features.
    """
    both = result.valid & spectrum.valid
    rows = np.flatnonzero(both)
    if rows.size > max_rows:
        rows = rows[np.linspace(0, rows.size - 1, max_rows).round().astype(np.int64)]
    if rows.size < 200:
        return pl.DataFrame(), pl.DataFrame()
    wav = ic_feature_columns(result, config.ic.features, rows)
    fft = _fft_columns(spectrum, rows)
    frame = pl.DataFrame({**wav, **fft}).with_columns(pl.all().fill_nan(None))
    pairs = []
    for a, b in _PAIRS:
        if a in frame.columns and b in frame.columns:
            sub = frame.select(a, b).drop_nulls()
            pairs.append({"source": source, "wavelet_feature": a, "fft_feature": b,
                          "observations": sub.height,
                          "pearson": sub.select(pl.corr(a, b)).item() if sub.height > 10 else None,
                          "spearman": sub.select(pl.corr(a, b, method="spearman")).item()
                          if sub.height > 10 else None})
    x_cols = [np.nan_to_num(np.asarray(v, dtype=np.float64)) for v in fft.values()]
    x = np.column_stack([np.ones(rows.size), *x_cols])
    half = rows.size // 2
    mu, sd = x[:half].mean(axis=0), x[:half].std(axis=0)
    sd[0], mu[0] = 1.0, 0.0
    keep = sd > 0
    xs = (x[:, keep] - mu[keep]) / sd[keep]
    det = []
    for name, values in wav.items():
        y = np.asarray(values, dtype=np.float64)
        ok = np.isfinite(y)
        tr, te = ok & (np.arange(rows.size) < half), ok & (np.arange(rows.size) >= half)
        if tr.sum() < 100 or te.sum() < 100:
            continue
        beta = np.linalg.lstsq(xs[tr], y[tr], rcond=None)[0]
        resid = y[te] - xs[te] @ beta
        sst = ((y[te] - y[tr].mean()) ** 2).sum()
        det.append({"source": source, "wavelet_feature": name,
                    "oos_r2_from_fft": float(1 - (resid ** 2).sum() / sst) if sst > 0 else None,
                    "train_rows": int(tr.sum()), "test_rows": int(te.sum())})
    return (pl.DataFrame(pairs, infer_schema_length=None),
            pl.DataFrame(det, infer_schema_length=None))


def dyadic_comparison(values: np.ndarray, result: RollingWavelet, spectral: SpectralConfig, *,
                      source: str, windows: int = 4000) -> pl.DataFrame:
    """Wavelet window shares against FFT power pooled into the same octave bands.

    Detail band ``j`` <-> frequencies ``[2^-(j+1), 2^-j)`` cycles per bar; the
    approximation <-> everything slower (DC excluded).
    """
    layout = result.layout
    n = result.window
    ends = np.flatnonzero(result.valid)
    if ends.size == 0:
        return pl.DataFrame()
    ends = ends[np.linspace(0, ends.size - 1, min(windows, ends.size)).round().astype(np.int64)]
    power = window_spectra(values, ends, n, spectral)
    freq = np.arange(1, power.shape[1] + 1) / n
    pooled = np.zeros((layout.bands, ends.size))
    for j in range(layout.levels):
        band = (freq >= 2.0 ** -(j + 2)) & (freq < 2.0 ** -(j + 1))
        pooled[j] = np.nansum(power[:, band], axis=1)
    pooled[layout.levels] = np.nansum(power[:, freq < 2.0 ** -(layout.levels + 1)], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled /= pooled.sum(axis=0, keepdims=True)
    rows = []
    for i, name in enumerate(layout.band_names()):
        w = result.shares[i, ends]
        f = pooled[i]
        ok = np.isfinite(w) & np.isfinite(f)
        if ok.sum() < 20:
            continue
        rows.append({"source": source, "band": name,
                     "period_bars": float(layout.period_bars[i]),
                     "mean_wavelet_share": float(w[ok].mean()),
                     "mean_fft_octave_share": float(f[ok].mean()),
                     "spearman": float(pl.DataFrame({"a": w[ok], "b": f[ok]}).select(
                         pl.corr("a", "b", method="spearman")).item()),
                     "mean_abs_difference": float(np.mean(np.abs(w[ok] - f[ok])))})
    return pl.DataFrame(rows, infer_schema_length=None)


# ---------------------------------------------------------------------------
# Cost (Step 49)
# ---------------------------------------------------------------------------
def benchmark_wavelet(config: WaveletConfig, *, bars: int = 1_000_000,
                      seed: int = 5) -> pl.DataFrame:
    """Throughput of the causal features and of offline scalograms; live update latency.

    ``live_update_ms`` recomputes every feature for one new bar from a
    buffer of ``2N`` bars (the longest lookback) - the naive live cost, an
    upper bound for an incremental implementation.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=bars)
    rows = []
    for window in config.windows:
        tracemalloc.start()
        start = time.perf_counter()
        rolling_wavelet(x, window=window, config=config, bar_seconds=300.0)
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        buffer = x[-2 * window - 1:]
        start = time.perf_counter()
        repeats = 20
        for _ in range(repeats):
            rolling_wavelet(buffer, window=window, config=config, bar_seconds=300.0)
        live = (time.perf_counter() - start) / repeats
        layout = band_layout(window, wavelet=config.dwt.wavelet, bar_seconds=300.0,
                             min_coefficients=config.causal_features.min_level_coefficients,
                             fast_slow_boundary_seconds=3600.0,
                             band_edges_seconds=(3600.0, 28800.0))
        rows.append({"task": "causal_features", "window": window, "levels": layout.levels,
                     "bars": bars, "seconds": elapsed, "bars_per_second": bars / elapsed,
                     "peak_memory_mb": peak / 2 ** 20, "live_update_ms": live * 1000})
    for length in (2048, 8192):
        segment = x[:length]
        start = time.perf_counter()
        offline_scalogram(segment, config, bar_seconds=300.0)
        elapsed = time.perf_counter() - start
        rows.append({"task": "offline_cwt_scalogram", "window": length, "levels":
                     config.cwt.scales.count, "bars": length, "seconds": elapsed,
                     "bars_per_second": length / elapsed, "peak_memory_mb": None,
                     "live_update_ms": None})
    return pl.DataFrame(rows, infer_schema_length=None)
