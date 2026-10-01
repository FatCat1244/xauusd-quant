r"""Frequency bins, power bands and the spectral centroid.

Frequencies are in **cycles per bar**. For a window of ``N`` bars the real FFT
has bins :math:`k = 0 \dots \lfloor N/2 \rfloor` at :math:`f_k = k/N`; the
Nyquist frequency is :math:`f_N = 1/2` cycle per bar (a period of 2 bars), so
no cycle shorter than two bars can be represented - and bar aggregation has
already removed or aliased anything faster than that.

Bands are defined on the *normalised* frequency :math:`f/f_N \in (0, 1]`, so
one set of edges means the same thing for every window length. A bin belongs
to the band whose half-open interval ``(lo, hi]`` contains it.
"""

from __future__ import annotations

import numpy as np

from .spectral_config import PowerBandsConfig, SpectrumConfig

__all__ = [
    "NYQUIST",
    "band_matrix",
    "band_shares",
    "bin_frequencies",
    "spectral_centroid",
    "usable_bins",
]

#: Nyquist frequency in cycles per bar.
NYQUIST = 0.5


def usable_bins(fft_window: int, spectrum: SpectrumConfig) -> np.ndarray:
    """Bin indices a component may come from: DC excluded, Nyquist optionally.

    For an even window the Nyquist bin holds only a cosine (its phase is 0 or
    pi by construction), so ``exclude_nyquist_if_needed`` drops it. An odd
    window has no Nyquist bin and nothing is dropped.
    """
    last = fft_window // 2
    if spectrum.exclude_nyquist_if_needed and fft_window % 2 == 0:
        last -= 1
    first = spectrum.minimum_frequency_index
    if last < first:
        raise ValueError(f"FFT window {fft_window} leaves no usable frequency bins")
    return np.arange(first, last + 1, dtype=np.int64)


def bin_frequencies(bins: np.ndarray, fft_window: int) -> np.ndarray:
    """Frequency of each bin in cycles per bar."""
    return bins.astype(np.float64) / float(fft_window)


def band_matrix(bins: np.ndarray, fft_window: int, bands: PowerBandsConfig) -> np.ndarray:
    """Indicator matrix (bins x bands): which band each usable bin falls in."""
    normalised = bin_frequencies(bins, fft_window) / NYQUIST
    edges = bands.edges
    matrix = np.zeros((bins.size, len(edges) - 1), dtype=np.float64)
    for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        matrix[:, j] = (normalised > lo) & (normalised <= hi)
    return matrix


def band_shares(power: np.ndarray, matrix: np.ndarray, total: np.ndarray) -> np.ndarray:
    """Share of each window's power in each band (rows sum to 1 over bands)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return (power @ matrix) / total[:, None]


def spectral_centroid(power: np.ndarray, freqs: np.ndarray, total: np.ndarray) -> np.ndarray:
    r"""Power-weighted mean frequency :math:`\sum f_k P_k / \sum P_k` (cycles per bar)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return (power @ freqs) / total
