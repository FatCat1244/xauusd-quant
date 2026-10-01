r"""Causal rolling FFT features.

For each bar ``t`` the window :math:`x_{t-N+1}, \dots, x_t` - and nothing
after ``t`` - is preprocessed and transformed:

1. **validity** - every value finite, and the window's bars span unscheduled
   gaps (anything but weekends and the daily break) of at most
   ``missing_bar_tolerance`` of its slots. The FFT assumes regular sampling;
   nothing is filled in. The trading-time axis is the bar sequence, exactly as
   in the regression and OU layers;
2. **mean removal** (and optionally a linear detrend) inside the window;
3. **taper**: Hann by default,
   :math:`w_n = \tfrac12\big(1-\cos\tfrac{2\pi n}{N-1}\big)`, so a window that
   does not hold a whole number of cycles leaks less power into neighbouring
   bins than it would with the implicit rectangular window;
4. **real FFT** :math:`X_k = \sum_n x'_n e^{-i 2\pi k n / N}`, ``k = 0..N/2``.

Units and conventions, fixed here once:

``power``
    raw FFT power :math:`P_k = |X_k|^2` of the tapered, mean-removed window,
    over the usable bins (DC excluded; Nyquist excluded when configured).
``power share``
    :math:`p_k = P_k / \sum_j P_j` over the same bins. Not a PSD: a density
    in physical units is computed on demand by
    :func:`xauusd_quant.research.fft_analysis.power_spectral_density`.
``amplitude``
    :math:`2|X_k| / \sum_n w_n` - the amplitude of a cosine in input units,
    corrected for the taper's coherent gain (``|X_k| / \sum w`` at Nyquist).
``phase``
    the component's phase at the window's **last** bar,
    :math:`\arg X_k + 2\pi k (N-1)/N`, wrapped to :math:`[-\pi, \pi)`. Left
    null when the component holds less than ``phase_min_power_share`` of the
    power: the phase of a negligible component is noise.
``frequency``
    cycles per bar, :math:`k/N`; period ``N/k`` bars; the longest period a
    window can show is ``N`` bars and the shortest is 2 bars.
``component``
    a local maximum of the power spectrum. Components are ranked by power;
    a bin beside a stronger one is Hann leakage of it, not a second cycle.
    Concentration (top-1/3/5 share) is instead over the raw sorted bins, as
    its definition says.

Estimates sit on the bin grid. A cycle ``delta`` bins off the grid is
reported at the nearest bin, with up to ~15 % less amplitude (Hann
scalloping) and a phase biased by about :math:`\pi\,\delta` - the positive
controls report such an off-bin case explicitly.

The first ``N-1`` bars have no complete window and stay null. Appending bars
cannot change any earlier value: each window's features depend on that
window's values alone, so chunking and history length never matter.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

from .spectral_bands import NYQUIST, band_matrix, band_shares, bin_frequencies, usable_bins
from .spectral_bands import spectral_centroid as _centroid
from .spectral_config import SpectralConfig
from .spectral_entropy import _TINY

try:  # scipy's pocketfft runs batches on several threads
    import scipy.fft as _fft

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - scipy is part of the research extra
    import numpy.fft as _fft

    _HAS_SCIPY = False

__all__ = [
    "RollingSpectrum",
    "SpectralLayout",
    "batch_spectrum",
    "rolling_spectrum",
    "spectral_feature_frame",
    "spectral_layout",
    "window_function",
    "window_validity",
]

TWO_PI = 2.0 * np.pi


def window_function(name: str, n: int) -> np.ndarray:
    """Symmetric taper of length *n* (``rectangular`` is all ones)."""
    if n < 2:
        return np.ones(n)
    k = np.arange(n, dtype=np.float64)
    phase = TWO_PI * k / (n - 1)
    if name == "hann":
        return 0.5 * (1.0 - np.cos(phase))
    if name == "hamming":
        return 0.54 - 0.46 * np.cos(phase)
    if name == "blackman":
        return 0.42 - 0.5 * np.cos(phase) + 0.08 * np.cos(2.0 * phase)
    if name == "rectangular":
        return np.ones(n)
    raise ValueError(f"unknown window function {name!r}")


@dataclass(frozen=True)
class SpectralLayout:
    """Everything about a window length that does not depend on the data."""

    fft_window: int
    bins: np.ndarray             # usable bin indices k
    freqs: np.ndarray            # cycles per bar
    taper: np.ndarray
    taper_sum: float
    band_names: tuple[str, ...]
    bands: np.ndarray            # (bins, bands) indicator
    nyquist_in_bins: bool
    top_k: int                   # components kept per window
    window_function: str

    @property
    def resolution(self) -> float:
        """Bin spacing in cycles per bar, 1/N."""
        return 1.0 / self.fft_window


def spectral_layout(fft_window: int, config: SpectralConfig) -> SpectralLayout:
    bins = usable_bins(fft_window, config.spectrum)
    taper = window_function(config.preprocessing.window_function, fft_window)
    return SpectralLayout(
        fft_window=fft_window,
        bins=bins,
        freqs=bin_frequencies(bins, fft_window),
        taper=taper,
        taper_sum=float(taper.sum()),
        band_names=config.power_bands.names if config.power_bands.enabled else (),
        bands=(band_matrix(bins, fft_window, config.power_bands) if config.power_bands.enabled
               else np.zeros((bins.size, 0))),
        nyquist_in_bins=bool(fft_window % 2 == 0 and bins[-1] == fft_window // 2),
        top_k=min(config.spectrum.top_frequencies, bins.size),
        window_function=config.preprocessing.window_function,
    )


def window_validity(
    values: np.ndarray, fft_window: int, *, missing_slots: np.ndarray | None,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per window (indexed by its first bar): finite, gap share, and valid.

    ``missing_slots[i]`` is the number of expected-but-absent bar slots
    between bar ``i-1`` and bar ``i`` inside *unscheduled* gaps. Only gaps
    between the window's own bars count - the gap before its first bar lies
    outside it. The window's missing fraction is ``S / (N + S)``.
    """
    n = values.size
    windows = max(n - fft_window + 1, 0)
    finite = np.isfinite(values)
    csum = np.concatenate(([0], np.cumsum(finite, dtype=np.int64)))
    complete = (csum[fft_window:] - csum[:-fft_window]) == fft_window if windows else np.zeros(0, bool)
    if missing_slots is None:
        fraction = np.zeros(windows, dtype=np.float64)
    else:
        slots = np.nan_to_num(np.asarray(missing_slots, dtype=np.float64), nan=0.0)
        cs = np.concatenate(([0.0], np.cumsum(slots)))
        # internal gaps of the window starting at s: before bars s+1 .. s+N-1
        inside = cs[fft_window:] - cs[1:windows + 1] if windows else np.zeros(0)
        fraction = inside / (fft_window + inside)
    return complete, fraction, complete & (fraction <= tolerance)


def batch_spectrum(
    block: np.ndarray, layout: SpectralLayout, config: SpectralConfig, *,
    workers: int | None = None, keep_power: bool = False,
) -> dict[str, np.ndarray]:
    """Spectral features of each row of *block* (rows are windows, oldest first).

    *block* is modified in place (it is the caller's private copy). Rows whose
    power is effectively zero - a constant window - come back with
    ``ok = False`` and meaningless values that the caller must discard.
    """
    pre = config.preprocessing
    n = layout.fft_window
    if pre.remove_mean or pre.detrend_input:
        block -= block.mean(axis=1, keepdims=True)
    if pre.detrend_input:
        ramp = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
        slope = (block @ ramp) / float(ramp @ ramp)
        block -= slope[:, None] * ramp[None, :]
    block *= layout.taper
    spectrum = (_fft.rfft(block, axis=1, workers=workers) if _HAS_SCIPY
                else _fft.rfft(block, axis=1))
    usable = spectrum[:, layout.bins[0]:layout.bins[-1] + 1]
    power = np.square(usable.real)
    power += np.square(usable.imag)
    total = power.sum(axis=1)
    ok = total > config.spectrum.zero_power_tolerance
    width = power.shape[1]

    # Concentration: the largest raw bins, whatever their neighbours.
    kk = min(5, width)
    raw = -np.sort(-np.partition(power, width - kk, axis=1)[:, -kk:], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cumulative = np.cumsum(raw, axis=1) / total[:, None]

    # Components: local maxima of the spectrum, strongest first.
    is_peak = np.ones(power.shape, dtype=bool)
    is_peak[:, 1:] &= power[:, 1:] > power[:, :-1]
    is_peak[:, :-1] &= power[:, :-1] >= power[:, 1:]
    peak_power = np.where(is_peak, power, -1.0)
    k = layout.top_k
    if k < width:
        top = np.argpartition(peak_power, width - k, axis=1)[:, -k:]
    else:
        top = np.broadcast_to(np.arange(width), power.shape).copy()
    top_peak = np.take_along_axis(peak_power, top, axis=1)
    order = np.argsort(-top_peak, axis=1, kind="stable")
    top = np.take_along_axis(top, order, axis=1)
    is_component = np.take_along_axis(top_peak, order, axis=1) >= 0.0
    top_power = np.take_along_axis(power, top, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        top_shares = np.where(is_component, top_power / total[:, None], np.nan)
    top_bins = np.where(is_component, layout.bins[top], -1)
    scale = np.where(top_bins * 2 == n, 1.0, 2.0) / layout.taper_sum
    amplitude = np.where(is_component, np.sqrt(top_power) * scale, np.nan)

    dominant = np.take_along_axis(usable, top[:, :1], axis=1)[:, 0]
    phase = np.angle(dominant) + TWO_PI * top_bins[:, 0] * (n - 1) / n
    phase = (phase + np.pi) % TWO_PI - np.pi

    out: dict[str, np.ndarray] = {
        "ok": ok,
        "total_power": total,
        "concentration": cumulative,
        "top_bins": top_bins,
        "top_shares": top_shares,
        "top_amplitude": amplitude,
        "dominant_phase": phase,
        "centroid": _centroid(power, layout.freqs, total),
    }
    # One log serves both shape measures (see spectral_entropy for the maths).
    with np.errstate(invalid="ignore", divide="ignore"):
        log_power = np.log(np.maximum(power, _TINY))
        log_total = np.log(total)
        if config.spectral_entropy.enabled:
            entropy = log_total - (power * log_power).sum(axis=1) / total
            if config.spectral_entropy.normalise:
                entropy = entropy / np.log(width) if width > 1 else np.full_like(total, np.nan)
            out["entropy"] = entropy
        out["flatness"] = np.exp(log_power.mean(axis=1) - (log_total - np.log(width)))
    if layout.band_names:
        out["bands"] = band_shares(power, layout.bands, total)
    if keep_power:
        out["power"] = power
    return out


@dataclass
class RollingSpectrum:
    """Per-bar spectral features of one series for one window length.

    Arrays are indexed by bar; a value at bar ``t`` describes the window that
    ends at ``t``. Invalid and warm-up bars hold NaN (bins: -1).
    """

    layout: SpectralLayout
    bar_seconds: float
    valid: np.ndarray
    missing_fraction: np.ndarray
    total_power: np.ndarray
    entropy: np.ndarray
    flatness: np.ndarray
    centroid: np.ndarray
    band_shares: dict[str, np.ndarray]
    concentration: dict[int, np.ndarray]
    top_bins: np.ndarray
    top_amplitude: np.ndarray
    top_shares: np.ndarray
    dominant_phase: np.ndarray
    dominant_phase_valid: np.ndarray
    mean_share: np.ndarray
    dominant_histogram: np.ndarray
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def fft_window(self) -> int:
        return self.layout.fft_window

    @property
    def dominant_bin(self) -> np.ndarray:
        return self.top_bins[:, 0]

    @property
    def dominant_period_bars(self) -> np.ndarray:
        bins = self.dominant_bin.astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(bins > 0, self.fft_window / bins, np.nan)

    @property
    def dominant_period_seconds(self) -> np.ndarray:
        return self.dominant_period_bars * self.bar_seconds

    @property
    def centroid_norm(self) -> np.ndarray:
        return self.centroid / np.float32(NYQUIST)


def rolling_spectrum(
    values: np.ndarray,
    *,
    fft_window: int,
    config: SpectralConfig,
    bar_seconds: float,
    missing_slots: np.ndarray | None = None,
) -> RollingSpectrum:
    """Rolling FFT features of *values* (float array, NaN where missing)."""
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    layout = spectral_layout(fft_window, config)
    k, c = layout.top_k, min(config.features_output.top_components, layout.top_k)
    dtype = np.float32 if config.features_output.float32 else np.float64

    def empty() -> np.ndarray:
        return np.full(n, np.nan, dtype=dtype)

    result = RollingSpectrum(
        layout=layout, bar_seconds=float(bar_seconds),
        valid=np.zeros(n, dtype=bool), missing_fraction=empty(), total_power=empty(),
        entropy=empty(), flatness=empty(), centroid=empty(),
        band_shares={name: empty() for name in layout.band_names},
        concentration={m: empty() for m in (1, 3, 5) if m <= layout.bins.size},
        top_bins=np.full((n, k), -1, dtype=np.int16),
        top_amplitude=np.full((n, c), np.nan, dtype=dtype),
        top_shares=np.full((n, c), np.nan, dtype=dtype),
        dominant_phase=empty(), dominant_phase_valid=np.zeros(n, dtype=bool),
        mean_share=np.zeros(layout.bins.size), dominant_histogram=np.zeros(layout.bins.size,
                                                                         dtype=np.int64),
    )
    if n < fft_window:
        result.counts = {"windows": 0, "valid": 0}
        return result

    complete, fraction, candidates = window_validity(
        x, fft_window, missing_slots=missing_slots,
        tolerance=config.preprocessing.missing_bar_tolerance)
    result.missing_fraction[fft_window - 1:] = fraction
    windows = sliding_window_view(x, fft_window)
    threads = _threads(config.rolling_fft.workers)
    # Each thread takes whole chunks and writes only its own bars, so the
    # outputs never depend on how the work was split.
    rows = max(1, config.rolling_fft.chunk_elements // (fft_window * threads))
    min_share = config.spectrum.phase_min_power_share

    def work(start: int) -> tuple[np.ndarray, np.ndarray, int]:
        share_sum = np.zeros(layout.bins.size)
        histogram = np.zeros(layout.bins.size, dtype=np.int64)
        stop = min(start + rows, windows.shape[0])
        chosen = np.flatnonzero(candidates[start:stop]) + start
        if chosen.size == 0:
            return share_sum, histogram, 0
        spec = batch_spectrum(windows[chosen], layout, config, workers=1, keep_power=True)
        ok = spec["ok"]
        zero = int((~ok).sum())
        chosen = chosen[ok]
        if chosen.size == 0:
            return share_sum, histogram, zero
        ends = chosen + fft_window - 1
        total = spec["total_power"][ok]
        share_sum = spec["power"][ok].T @ (1.0 / total)
        top_bins = spec["top_bins"][ok]
        histogram = np.bincount(top_bins[:, 0] - layout.bins[0], minlength=layout.bins.size)
        result.valid[ends] = True
        result.total_power[ends] = total
        result.centroid[ends] = spec["centroid"][ok]
        result.flatness[ends] = spec["flatness"][ok]
        if "entropy" in spec:
            result.entropy[ends] = spec["entropy"][ok]
        for j, name in enumerate(layout.band_names):
            result.band_shares[name][ends] = spec["bands"][ok, j]
        shares = spec["top_shares"][ok]
        cumulative = spec["concentration"][ok]
        for m, arr in result.concentration.items():
            arr[ends] = cumulative[:, min(m, cumulative.shape[1]) - 1]
        result.top_bins[ends] = top_bins.astype(np.int16)
        result.top_amplitude[ends] = spec["top_amplitude"][ok, :c]
        result.top_shares[ends] = shares[:, :c]
        phase_ok = shares[:, 0] >= min_share
        result.dominant_phase[ends[phase_ok]] = spec["dominant_phase"][ok][phase_ok]
        result.dominant_phase_valid[ends[phase_ok]] = True
        return share_sum, histogram, zero

    starts = range(0, windows.shape[0], rows)
    if threads > 1:
        with ThreadPoolExecutor(max_workers=threads) as pool:
            parts = list(pool.map(work, starts))
    else:
        parts = [work(start) for start in starts]
    share_total = np.sum([p[0] for p in parts], axis=0) if parts else np.zeros(layout.bins.size)
    result.dominant_histogram = (np.sum([p[1] for p in parts], axis=0) if parts
                                 else result.dominant_histogram)
    zero_power = int(sum(p[2] for p in parts))
    valid_count = int(result.valid.sum())
    if valid_count:
        result.mean_share = share_total / valid_count
    result.counts = {
        "windows": int(windows.shape[0]),
        "valid": valid_count,
        "incomplete": int((~complete).sum()),
        "too_many_missing_bars": int((complete & ~candidates).sum()),
        "zero_power": zero_power,
    }
    return result


def _threads(workers: int) -> int:
    """Threads for the rolling pass: *workers*, or every core when it is -1."""
    if workers and workers > 0:
        return workers
    return max(1, os.cpu_count() or 1)


def spectral_feature_frame(
    result: RollingSpectrum, timestamps: pl.Series, config: SpectralConfig,
) -> pl.DataFrame:
    """The compact per-bar feature table written for later research and ML.

    One row per bar, null wherever the window is invalid; only ``timestamp``
    as the join key - prices, residuals and OU columns live in their own
    datasets and are joined by timestamp and dataset version.
    """
    n_top = result.top_amplitude.shape[1]
    dtype = pl.Float32 if config.features_output.float32 else pl.Float64
    bins = result.top_bins.astype(np.float64)
    bins[bins < 0] = np.nan
    columns: dict[str, pl.Series] = {
        "timestamp": timestamps.alias("timestamp"),
        "spectral_window_valid": pl.Series(result.valid),
        "missing_bar_fraction": pl.Series(result.missing_fraction, nan_to_null=True),
    }
    n = result.fft_window
    for j in range(n_top):
        freq = bins[:, j] / n
        with np.errstate(divide="ignore", invalid="ignore"):
            period = n / bins[:, j]
        power = result.top_shares[:, j].astype(np.float64) * result.total_power.astype(np.float64)
        for name, values in (
            (f"fft_freq_{j + 1}", freq), (f"fft_period_bars_{j + 1}", period),
            (f"fft_period_seconds_{j + 1}", period * result.bar_seconds),
            (f"fft_amp_{j + 1}", result.top_amplitude[:, j]),
            (f"fft_power_{j + 1}", power),
            (f"fft_power_share_{j + 1}", result.top_shares[:, j]),
        ):
            columns[name] = pl.Series(name, values, nan_to_null=True).cast(dtype)
    phase = result.dominant_phase.astype(np.float64)
    for name, values in (("fft_phase_1", phase), ("fft_phase1_sin", np.sin(phase)),
                         ("fft_phase1_cos", np.cos(phase))):
        columns[name] = pl.Series(name, values, nan_to_null=True).cast(dtype)
    columns["fft_phase1_valid"] = pl.Series(result.dominant_phase_valid)
    for name, values in (
        ("spectral_entropy", result.entropy), ("spectral_flatness", result.flatness),
        ("spectral_centroid", result.centroid), ("spectral_centroid_norm", result.centroid_norm),
        *((f"{band}_power_share", arr) for band, arr in result.band_shares.items()),
        *((f"top{m}_power_share", arr) for m, arr in result.concentration.items()),
        ("total_power", result.total_power),
    ):
        columns[name] = pl.Series(name, values, nan_to_null=True).cast(dtype)
    return pl.DataFrame(columns)
