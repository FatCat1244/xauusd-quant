r"""Single-window Fourier analysis, positive controls and throughput.

The rolling engine (:mod:`xauusd_quant.features.spectral`) keeps only a few
components per bar. This module looks at whole spectra - of one window, of a
chosen set of windows, or averaged - and holds the checks that the machinery
recovers what it should:

* **positive controls**: a sinusoid of known frequency, amplitude and phase,
  alone or in pairs, with noise, must come back where it was put;
* **resolution**: two frequencies closer than about ``2/N`` cycles per bar
  (the Hann main-lobe half-width) cannot be told apart by an ``N``-bar window
  however strong they are - conclusions must respect that;
* **aliasing**: bars are samples. Nothing faster than the Nyquist frequency of
  1/2 cycle per bar can be represented, and a faster oscillation that bar
  aggregation did not average away shows up at a *wrong*, lower frequency
  (:func:`aliased_frequency`). No sub-bar cycle is ever reported.

Power conventions are those of the engine: raw FFT power ``|X_k|^2``, power
share, and - only here, and only under that name - a one-sided periodogram
PSD in (input units)^2 per (cycle per second) or per (cycle per bar).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import polars as pl

from ..features.spectral import (
    TWO_PI,
    batch_spectrum,
    spectral_layout,
    window_function,
)
from ..features.spectral_config import SpectralConfig

try:
    import scipy.fft as _sfft
except ImportError:  # pragma: no cover
    _sfft = None

__all__ = [
    "aliased_frequency",
    "benchmark_rolling_fft",
    "mean_spectrum",
    "positive_control_table",
    "power_spectral_density",
    "recover_components",
    "resolution_table",
    "sinusoid",
    "window_spectra",
    "window_spectrum",
]


def sinusoid(n: int, components: list[tuple[float, float, float]], *, noise_std: float = 0.0,
             seed: int = 0) -> np.ndarray:
    r""":math:`\sum_j A_j \cos(2\pi f_j t + \phi_j) + \epsilon_t`, ``t = 0..n-1``.

    *components* holds ``(frequency in cycles per bar, amplitude, phase)``.
    """
    t = np.arange(n, dtype=np.float64)
    x = np.zeros(n)
    for freq, amp, phase in components:
        x += amp * np.cos(TWO_PI * freq * t + phase)
    if noise_std > 0:
        x += np.random.default_rng(seed).normal(0.0, noise_std, n)
    return x


def aliased_frequency(frequency: float) -> float:
    """Where a frequency (cycles per bar) appears once sampled once per bar."""
    folded = frequency % 1.0
    return min(folded, 1.0 - folded)


def window_spectrum(window: np.ndarray, config: SpectralConfig, *,
                    bar_seconds: float | None = None) -> pl.DataFrame:
    """Every usable bin of one window: frequency, period, power, share, amplitude, phase.

    Preprocessing is the engine's (mean removal, taper). Phase is at the
    window's last bar and is null for bins below ``phase_min_power_share``.
    """
    n = window.size
    layout = spectral_layout(n, config)
    block = np.asarray(window, dtype=np.float64)[None, :].copy()
    pre = config.preprocessing
    if pre.remove_mean or pre.detrend_input:
        block -= block.mean(axis=1, keepdims=True)
    if pre.detrend_input:
        ramp = np.arange(n) - (n - 1) / 2.0
        block -= ((block @ ramp) / float(ramp @ ramp))[:, None] * ramp[None, :]
    block *= layout.taper
    spectrum = np.fft.rfft(block[0])[layout.bins]
    power = spectrum.real ** 2 + spectrum.imag ** 2
    total = power.sum()
    share = power / total if total > 0 else np.full_like(power, np.nan)
    scale = np.where(layout.bins * 2 == n, 1.0, 2.0) / layout.taper_sum
    phase = (np.angle(spectrum) + TWO_PI * layout.bins * (n - 1) / n + np.pi) % TWO_PI - np.pi
    phase = np.where(share >= config.spectrum.phase_min_power_share, phase, np.nan)
    left = np.concatenate(([-np.inf], power[:-1]))
    right = np.concatenate((power[1:], [-np.inf]))
    frame = pl.DataFrame({
        "bin": layout.bins,
        "is_peak": (power > left) & (power >= right),
        "frequency_cycles_per_bar": layout.freqs,
        "period_bars": n / layout.bins.astype(np.float64),
        "power": power,
        "power_share": share,
        "amplitude": np.sqrt(power) * scale,
        "phase_at_last_bar": phase,
    })
    if bar_seconds is not None:
        frame = frame.with_columns(
            (pl.col("period_bars") * bar_seconds).alias("period_seconds"))
    return frame.with_columns(pl.col("phase_at_last_bar").fill_nan(None))


def recover_components(window: np.ndarray, config: SpectralConfig, k: int) -> pl.DataFrame:
    """The *k* strongest components (spectral peaks) of one window, strongest first."""
    return (window_spectrum(window, config).filter(pl.col("is_peak"))
            .sort("power", descending=True).head(k))


def power_spectral_density(window: np.ndarray, *, bar_seconds: float | None = None,
                           taper: str = "hann") -> pl.DataFrame:
    r"""One-sided periodogram of one window, in density units.

    :math:`\mathrm{PSD}_k = c_k |X_k|^2 / (f_s \sum_n w_n^2)` with ``c_k = 2``
    except at DC and Nyquist. ``f_s`` is 1 per bar, or ``1/bar_seconds`` per
    second when *bar_seconds* is given, so the density's area is the
    (tapered) variance. Kept apart from raw power and power share on purpose.
    """
    x = np.asarray(window, dtype=np.float64)
    n = x.size
    w = window_function(taper, n)
    spectrum = np.fft.rfft((x - x.mean()) * w)
    fs = 1.0 if bar_seconds is None else 1.0 / bar_seconds
    c = np.full(spectrum.size, 2.0)
    c[0] = 1.0
    if n % 2 == 0:
        c[-1] = 1.0
    psd = c * np.abs(spectrum) ** 2 / (fs * float((w ** 2).sum()))
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    unit = "per_second" if bar_seconds is not None else "per_bar"
    return pl.DataFrame({f"frequency_cycles_{unit}": freq, f"psd_{unit}": psd})


def window_spectra(values: np.ndarray, ends: np.ndarray, fft_window: int,
                   config: SpectralConfig) -> np.ndarray:
    """Power share of every usable bin for the windows ending at *ends* (rows)."""
    x = np.asarray(values, dtype=np.float64)
    layout = spectral_layout(fft_window, config)
    ends = np.asarray(ends, dtype=np.int64)
    ends = ends[(ends >= fft_window - 1) & (ends < x.size)]
    idx = ends[:, None] - (fft_window - 1) + np.arange(fft_window)[None, :]
    block = x[idx]
    keep = np.isfinite(block).all(axis=1)
    out = np.full((ends.size, layout.bins.size), np.nan)
    if keep.any():
        spec = batch_spectrum(block[keep].copy(), layout, config, keep_power=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[keep] = spec["power"] / spec["total_power"][:, None]
    return out


def mean_spectrum(shares: np.ndarray, layout_bins: np.ndarray, fft_window: int) -> pl.DataFrame:
    """Average power share per bin over windows (rows of *shares*)."""
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(shares, axis=0)
    freq = layout_bins / fft_window
    return pl.DataFrame({"bin": layout_bins, "frequency_cycles_per_bar": freq,
                         "normalised_frequency": freq / 0.5,
                         "period_bars": fft_window / layout_bins.astype(np.float64),
                         "mean_power_share": mean})


def positive_control_table(config: SpectralConfig, *, noise_std: float = 0.5,
                           seed: int = 7) -> pl.DataFrame:
    """Known sinusoids through the engine: is what went in what comes out?

    For every configured window length: one on-bin and one off-bin cosine of
    amplitude 1 with noise, and a two-component signal. Reports the injected
    and recovered frequency, amplitude and phase (at the last bar), the phase
    error, and whether the strongest bins are the injected ones.
    """
    rows: list[dict[str, Any]] = []
    for n in config.fft_windows:
        on_bin = 8.0 / n
        cases = {
            "single on-bin": [(on_bin, 1.0, 0.7)],
            "single off-bin": [(8.4 / n, 1.0, -1.1)],
            "two components": [(on_bin, 1.0, 0.3), (24.0 / n, 0.6, 2.0)],
        }
        for label, components in cases.items():
            x = sinusoid(4 * n, components, noise_std=noise_std, seed=seed)
            window = x[-n:]
            got = recover_components(window, config, len(components))
            t_last = 4 * n - 1
            for j, (freq, amp, phase) in enumerate(sorted(components, key=lambda c: -c[1])):
                rec = got.row(j, named=True) if j < got.height else {}
                true_phase = (TWO_PI * freq * t_last + phase + np.pi) % TWO_PI - np.pi
                rec_phase = rec.get("phase_at_last_bar")
                phase_error = (None if rec_phase is None else
                               float(abs((rec_phase - true_phase + np.pi) % TWO_PI - np.pi)))
                rows.append({
                    "fft_window": n, "case": label, "component": j + 1,
                    "injected_frequency": freq, "recovered_frequency": rec.get(
                        "frequency_cycles_per_bar"),
                    "frequency_error_bins": (None if not rec else
                                             abs(rec["frequency_cycles_per_bar"] - freq) * n),
                    "injected_amplitude": amp, "recovered_amplitude": rec.get("amplitude"),
                    "injected_phase_at_last_bar": float(true_phase),
                    "recovered_phase_at_last_bar": rec_phase,
                    "phase_error_radians": phase_error,
                    "noise_std": noise_std,
                })
    return pl.DataFrame(rows, infer_schema_length=None)


def resolution_table(config: SpectralConfig, *, base_bin_of_1024: float = 64.0) -> pl.DataFrame:
    """Can each window length separate two equal cosines ``delta`` apart?

    Two noise-free unit cosines at ``f`` and ``f + delta`` (cycles per bar).
    They count as resolved when the Hann spectrum has two distinct local
    maxima within one bin of each. The frequency resolution is ``1/N``; the
    Hann main lobe makes about ``2/N`` the practical limit.
    """
    rows = []
    f = base_bin_of_1024 / 1024.0
    for delta_bins_1024 in (2, 4, 8, 16, 32):
        delta = delta_bins_1024 / 1024.0
        for n in config.fft_windows:
            x = sinusoid(n, [(f, 1.0, 0.0), (f + delta, 1.0, 1.0)])
            spec = window_spectrum(x, config)
            power = spec["power"].to_numpy()
            freq = spec["frequency_cycles_per_bar"].to_numpy()
            peaks = np.flatnonzero((power[1:-1] > power[:-2]) & (power[1:-1] > power[2:])) + 1
            near = [bool(np.any(np.abs(freq[peaks] - g) <= 1.0 / n)) for g in (f, f + delta)]
            distinct = bool(all(near)) and np.unique(
                [int(np.argmin(np.abs(freq[peaks] - g))) for g in (f, f + delta)]).size == 2
            rows.append({"fft_window": n, "separation_cycles_per_bar": delta,
                         "separation_in_bins": delta * n, "resolved": distinct})
    return pl.DataFrame(rows)


def benchmark_rolling_fft(config: SpectralConfig, *, windows: int = 20_000,
                          seed: int = 3) -> pl.DataFrame:
    """Throughput of the pieces of the rolling engine, per window length.

    ``numpy_rfft`` and ``scipy_rfft_1_thread`` time the bare transform of a
    batch; ``scipy_rfft_all_threads`` uses every core; ``engine_features``
    times :func:`~xauusd_quant.features.spectral.batch_spectrum` end to end
    (preprocessing, transform, top-K, entropy, flatness, bands, phase).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for n in config.fft_windows:
        batch = rng.normal(size=(windows, n))
        layout = spectral_layout(n, config)
        timings: dict[str, float] = {}
        start = time.perf_counter()
        np.fft.rfft(batch, axis=1)
        timings["numpy_rfft"] = time.perf_counter() - start
        if _sfft is not None:
            start = time.perf_counter()
            _sfft.rfft(batch, axis=1, workers=1)
            timings["scipy_rfft_1_thread"] = time.perf_counter() - start
            start = time.perf_counter()
            _sfft.rfft(batch, axis=1, workers=-1)
            timings["scipy_rfft_all_threads"] = time.perf_counter() - start
        start = time.perf_counter()
        batch_spectrum(batch.copy(), layout, config, workers=config.rolling_fft.workers)
        timings["engine_features"] = time.perf_counter() - start
        for stage, seconds in timings.items():
            rows.append({"fft_window": n, "stage": stage, "windows": windows,
                         "seconds": seconds, "windows_per_second": windows / seconds,
                         "bars_per_second_equivalent": windows / seconds})
    return pl.DataFrame(rows)
