r"""Wavelet bands: their physical periods, energy shares, groups and concentration.

A rolling window of ``N`` bars is decomposed into ``J`` detail bands and one
approximation band (:func:`band_layout`):

* detail band ``j`` (``j = 1..J``) covers periods of roughly
  :math:`[2^j, 2^{j+1}]` bars; its nominal period is
  :math:`2^j / f_c` bars with :math:`f_c` the wavelet's centre frequency
  (``pywt.central_frequency``; 0.714 for ``db4``, so about
  :math:`1.4 \cdot 2^j`);
* the approximation band covers :math:`[2^{J+1}, N]` bars (the window bounds
  the slowest variation it can see); its nominal period is the geometric
  mean of that range.

Every band also has a period in seconds (``bars x bar length``), and bands are
grouped by that physical period, never by level number, so a 1-hour scale is
compared with a 1-hour scale across timeframes:

* ``fast`` / ``slow``: nominal period below / at-or-above a boundary;
* ``high`` / ``mid`` / ``low``: two boundaries. A group with no band at a
  given timeframe (no band under 1 h on hourly bars) is undefined (NaN),
  not zero.

Energy shares :math:`p_j = E_j / \sum_k E_k` sum to one over all ``J+1``
bands.

**White-noise baseline.** Band energies are not flat for white noise: an
orthogonal MODWT gives detail band ``j`` exactly :math:`2^{-j}` of a white
series' variance, so "the band with the most energy" is always the finest
one and says nothing. :func:`white_noise_shares` computes the exact share
each band's estimator expects under white noise (for the approximation band,
the within-window variance of the smooth, from the scaling filter's
autocorrelation). The *dominant* band is the one whose share most exceeds
that baseline - white noise has no preferred band, as in the FFT.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pywt

from .wavelet import equivalent_filter_length, max_level, modwt_filters

__all__ = [
    "WaveletLayout",
    "active_scales",
    "band_layout",
    "concentration",
    "energy_shares",
    "group_share",
    "white_noise_shares",
]


@dataclass(frozen=True)
class WaveletLayout:
    """Bands of one (wavelet, window, bar size): periods, filter lengths, groups."""

    wavelet: str
    window: int
    bar_seconds: float
    levels: int                       # J detail bands; band J+1 is the approximation
    filter_lengths: tuple[int, ...]   # L_1 .. L_J, then the scaling filter's
    period_bars: np.ndarray           # nominal period of each band (J+1,)
    band_low_bars: np.ndarray         # shortest period of each band
    band_high_bars: np.ndarray        # longest period of each band
    fast: np.ndarray                  # bool (J+1,)
    groups: dict[str, np.ndarray]     # "high" / "mid" / "low" -> bool (J+1,)
    fast_slow_boundary_seconds: float
    band_edges_seconds: tuple[float, float]

    @property
    def bands(self) -> int:
        return self.levels + 1

    @property
    def period_seconds(self) -> np.ndarray:
        return self.period_bars * self.bar_seconds

    @property
    def coefficients_in_window(self) -> np.ndarray:
        """Complete (boundary-free) coefficients each band has inside the window."""
        return np.array([self.window - length + 1 for length in self.filter_lengths])

    def band_names(self) -> list[str]:
        return [f"d{j}" for j in range(1, self.levels + 1)] + ["a"]

    def describe(self) -> list[dict[str, float | int | str | bool]]:
        """One row per band, for metadata and reports."""
        rows: list[dict[str, float | int | str | bool]] = []
        for i, name in enumerate(self.band_names()):
            group = next((g for g, mask in self.groups.items() if mask[i]), "")
            rows.append({
                "band": name, "index": i + 1, "is_approximation": name == "a",
                "period_bars": float(self.period_bars[i]),
                "period_seconds": float(self.period_seconds[i]),
                "band_low_bars": float(self.band_low_bars[i]),
                "band_high_bars": float(self.band_high_bars[i]),
                "filter_length": int(self.filter_lengths[i]),
                "coefficients_in_window": int(self.coefficients_in_window[i]),
                "speed": "fast" if self.fast[i] else "slow", "group": group,
            })
        return rows


def band_layout(window: int, *, wavelet: str, bar_seconds: float, min_coefficients: int,
                fast_slow_boundary_seconds: float,
                band_edges_seconds: tuple[float, float]) -> WaveletLayout:
    """The bands a causal MODWT of *window* bars resolves, with physical periods."""
    levels = max_level(window, wavelet, min_coefficients)
    if levels < 1:
        raise ValueError(f"window {window} is too short for {wavelet!r}")
    length = pywt.Wavelet(wavelet).dec_len
    centre = float(pywt.central_frequency(wavelet))
    lengths = tuple(equivalent_filter_length(length, j) for j in range(1, levels + 1))
    lengths = (*lengths, lengths[-1])            # the level-J scaling filter has L_J taps
    j = np.arange(1, levels + 1, dtype=np.float64)
    low = np.append(2.0 ** j, 2.0 ** (levels + 1))
    high = np.append(2.0 ** (j + 1), float(max(window, 2 ** (levels + 1))))
    period = np.append(2.0 ** j / centre, np.sqrt(low[-1] * high[-1]))
    seconds = period * bar_seconds
    lo_edge, hi_edge = band_edges_seconds
    groups = {
        "high": seconds < lo_edge,
        "mid": (seconds >= lo_edge) & (seconds < hi_edge),
        "low": seconds >= hi_edge,
    }
    return WaveletLayout(wavelet=wavelet, window=window, bar_seconds=bar_seconds, levels=levels,
                         filter_lengths=lengths, period_bars=period, band_low_bars=low,
                         band_high_bars=high, fast=seconds < fast_slow_boundary_seconds,
                         groups=groups, fast_slow_boundary_seconds=fast_slow_boundary_seconds,
                         band_edges_seconds=(float(lo_edge), float(hi_edge)))


def white_noise_shares(layout: WaveletLayout) -> np.ndarray:
    """Expected share of each band's energy estimator for unit white noise.

    Detail band ``j``: the mean square of MODWT coefficients has expectation
    :math:`\\sum_l \\tilde h_{j,l}^2 = 2^{-j}`. Approximation band: the
    within-window variance of ``M`` consecutive smooth values,
    :math:`R(0) - M^{-2}\\sum_{s,s'} R(s-s')` with :math:`R` the scaling
    filter's autocorrelation. Normalised to sum to one.
    """
    filters = modwt_filters(layout.wavelet, layout.levels)
    expected = [float(np.sum(f ** 2)) for f in filters.detail]
    g = filters.scaling
    span = layout.window - g.size + 1
    acf = np.correlate(g, g, mode="full")[g.size - 1:]          # R(0), R(1), ...
    lags = np.arange(min(span, acf.size))
    weights = (span - lags) * np.where(lags == 0, 1.0, 2.0)     # pairs at each |lag|
    mean_square = float((weights * acf[:lags.size]).sum()) / span ** 2
    expected.append(max(float(acf[0]) - mean_square, 1e-12))
    shares = np.asarray(expected)
    return shares / shares.sum()


def energy_shares(energies: np.ndarray, *, tolerance: float = 1e-30) -> np.ndarray:
    """Row-normalised energies :math:`p_j = E_j / \\sum_k E_k`; NaN where the total is ~0.

    *energies* is ``(bands, bars)``.
    """
    e = np.asarray(energies, dtype=np.float64)
    total = e.sum(axis=0)
    ok = total > tolerance
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ok[None, :], e / np.where(ok, total, 1.0)[None, :], np.nan)


def group_share(shares: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Sum of the shares of the bands in *mask*; NaN if the group has no band."""
    if not np.asarray(mask).any():
        return np.full(shares.shape[1], np.nan)
    return shares[np.asarray(mask)].sum(axis=0)


def concentration(shares: np.ndarray, k: int) -> np.ndarray:
    """Share held by the *k* largest bands (NaN where shares are undefined)."""
    k = min(k, shares.shape[0])
    ordered = -np.sort(-np.nan_to_num(shares, nan=-1.0), axis=0)
    out = ordered[:k].sum(axis=0)
    return np.where(np.isnan(shares).any(axis=0), np.nan, out)


def active_scales(shares: np.ndarray) -> np.ndarray:
    """Bands holding more than the uniform share ``1/(J+1)``: how many scales carry energy."""
    bands = shares.shape[0]
    count = (shares > 1.0 / bands).sum(axis=0).astype(np.float64)
    return np.where(np.isnan(shares).any(axis=0), np.nan, count)
