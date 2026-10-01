r"""Wavelet transforms for the time-frequency layer, causal and offline kept apart.

**Causal** - :func:`causal_modwt`. The maximal-overlap DWT (MODWT, the
undecimated "stationary" wavelet transform) computed with one-sided filters:

.. math:: W_{j,t} = \sum_{l=0}^{L_j-1} \tilde h_{j,l}\, x_{t-l}, \qquad
          V_{J,t} = \sum_{l=0}^{L_J-1} \tilde g_{J,l}\, x_{t-l}

with the level-``j`` equivalent filters built from PyWavelets' orthogonal
filter pair (:func:`modwt_filters`), :math:`\tilde h = h/\sqrt2`,
:math:`\tilde g = g/\sqrt2`, and :math:`L_j = (2^j-1)(L-1)+1`. The coefficient
at bar ``t`` is a fixed linear combination of ``x[t-L_j+1 .. t]`` - nothing
later, and no padding at the right edge, because the filter never reaches
past ``t``. The price of causality is a delay: a level-``j`` coefficient
describes the signal roughly ``(L_j-1)/2`` bars back. The first ``L_j-1``
coefficients have an incomplete history and are never used.

PyWavelets has no one-sided MODWT (``pywt.swt`` is circular, so its last
coefficients wrap around to the start of the series); this module builds the
filters from ``pywt.Wavelet`` and applies them with direct convolution
(``numpy.convolve``), whose output at ``t`` depends on the same inputs in the
same order however long the series is - appending bars changes no earlier
coefficient. The tests check the energy decomposition against ``pywt.swt``.

**Offline** - :func:`offline_cwt`. The continuous transform of a whole slice,
centred at every point, so each coefficient uses bars on *both* sides of it.
It is for scalograms, ridge research and positive controls, is labelled
:data:`NON_CAUSAL_LABEL`, and can never enter a feature table (see
:mod:`xauusd_quant.features.wavelet_causal`).

:func:`windowed_dwt_edge` is the decimated DWT of one trailing window with a
chosen padding mode - used only by the boundary study, which measures how
much the newest coefficients of a windowed transform depend on padding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pywt

__all__ = [
    "NON_CAUSAL_LABEL",
    "ModwtCoefficients",
    "OfflineScalogram",
    "Ridge",
    "WaveletFilters",
    "causal_modwt",
    "circular_modwt",
    "cone_of_influence",
    "cwt_scales",
    "dwt_components",
    "equivalent_filter_length",
    "extract_ridges",
    "max_level",
    "modwt_filters",
    "offline_cwt",
    "scale_to_period",
    "windowed_dwt_edge",
]

#: Label carried by everything computed from both sides of a point in time.
NON_CAUSAL_LABEL = "NON-CAUSAL — VISUALIZATION ONLY"


# ---------------------------------------------------------------------------
# Causal MODWT
# ---------------------------------------------------------------------------
def equivalent_filter_length(filter_length: int, level: int) -> int:
    """Length :math:`L_j = (2^j - 1)(L - 1) + 1` of the level-``j`` MODWT filter."""
    return (2 ** level - 1) * (filter_length - 1) + 1


def max_level(window: int, wavelet: str, min_coefficients: int = 16) -> int:
    """Deepest level whose filter leaves *min_coefficients* complete coefficients in *window*."""
    length = pywt.Wavelet(wavelet).dec_len
    level = 0
    while window - equivalent_filter_length(length, level + 1) + 1 >= min_coefficients:
        level += 1
    return level


def _upsample(values: np.ndarray, factor: int) -> np.ndarray:
    out = np.zeros((values.size - 1) * factor + 1)
    out[::factor] = values
    return out


@dataclass(frozen=True)
class WaveletFilters:
    """Level-1..J equivalent MODWT wavelet filters and the level-J scaling filter."""

    wavelet: str
    filter_length: int
    detail: tuple[np.ndarray, ...]
    scaling: np.ndarray

    @property
    def levels(self) -> int:
        return len(self.detail)

    @property
    def lengths(self) -> tuple[int, ...]:
        """Support of each detail filter (the bars a coefficient reads)."""
        return tuple(int(f.size) for f in self.detail)


def modwt_filters(wavelet: str, levels: int) -> WaveletFilters:
    """Equivalent MODWT filters of an orthogonal PyWavelets wavelet."""
    if levels < 1:
        raise ValueError("levels must be >= 1")
    base = pywt.Wavelet(wavelet)
    if not base.orthogonal:
        raise ValueError(f"{wavelet!r} is not orthogonal")
    h = np.asarray(base.dec_hi, dtype=np.float64) / np.sqrt(2.0)
    g = np.asarray(base.dec_lo, dtype=np.float64) / np.sqrt(2.0)
    detail = []
    smooth = np.array([1.0])
    for j in range(1, levels + 1):
        step = 2 ** (j - 1)
        detail.append(np.convolve(smooth, _upsample(h, step)))
        smooth = np.convolve(smooth, _upsample(g, step))
    return WaveletFilters(wavelet=wavelet, filter_length=int(h.size), detail=tuple(detail),
                          scaling=smooth)


@dataclass
class ModwtCoefficients:
    """Causal MODWT output: ``detail[j-1, t]`` is :math:`W_{j,t}`, ``scaling[t]`` :math:`V_{J,t}`.

    Missing inputs are read as zero by the filters; ``finite_history[j, t]``
    says whether the ``L_j`` inputs behind a coefficient were all finite and
    present (row ``J`` is the scaling filter). A coefficient without a
    complete, finite history must not be used.
    """

    filters: WaveletFilters
    detail: np.ndarray
    scaling: np.ndarray
    finite_history: np.ndarray


def causal_modwt(values: np.ndarray, filters: WaveletFilters) -> ModwtCoefficients:
    """One-sided MODWT of *values*: every coefficient uses bars at or before its own."""
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    finite = np.isfinite(x)
    x0 = np.where(finite, x, 0.0)
    csum = np.concatenate(([0], np.cumsum(finite, dtype=np.int64)))
    lengths = [*filters.lengths, int(filters.scaling.size)]
    history = np.zeros((len(lengths), n), dtype=bool)
    for row, length in enumerate(lengths):
        if n >= length:
            full = csum[length:] - csum[:-length] == length
            history[row, length - 1:] = full
    detail = np.empty((filters.levels, n), dtype=np.float64)
    for j, f in enumerate(filters.detail):
        detail[j] = np.convolve(x0, f)[:n]
    scaling = np.convolve(x0, filters.scaling)[:n]
    return ModwtCoefficients(filters=filters, detail=detail, scaling=scaling,
                             finite_history=history)


def circular_modwt(values: np.ndarray, filters: WaveletFilters) -> tuple[np.ndarray, np.ndarray]:
    """The circular (periodic) MODWT - for checking the energy decomposition only.

    For an orthogonal wavelet :math:`\\sum_j \\|W_j\\|^2 + \\|V_J\\|^2 = \\|x\\|^2`.
    """
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    spectrum = np.fft.fft(x)

    def filt(f: np.ndarray) -> np.ndarray:
        kernel = np.zeros(n)
        for i, c in enumerate(f):
            kernel[i % n] += c
        return np.real(np.fft.ifft(spectrum * np.fft.fft(kernel)))

    return np.stack([filt(f) for f in filters.detail]), filt(filters.scaling)


# ---------------------------------------------------------------------------
# Decimated DWT: multiresolution decomposition (offline) and the right edge
# ---------------------------------------------------------------------------
def dwt_components(values: np.ndarray, wavelet: str, level: int,
                   mode: str = "periodization") -> dict[str, np.ndarray]:
    """Multiresolution decomposition :math:`x = A_J + \\sum_j D_j` of a whole series.

    Each component is reconstructed from one level's coefficients alone (the
    others zeroed), so components can be inspected or summed selectively; with
    ``periodization`` and a length divisible by ``2**level`` they add up to
    the series exactly. The decimated DWT of a whole series uses bars on both
    sides of every point: offline analysis only (:data:`NON_CAUSAL_LABEL`).
    """
    x = np.asarray(values, dtype=np.float64)
    coeffs = pywt.wavedec(x, wavelet, mode=mode, level=level)
    out = {}
    for i in range(len(coeffs)):
        parts = [np.zeros_like(c) for c in coeffs]
        parts[i] = coeffs[i]
        name = f"a{level}" if i == 0 else f"d{level - i + 1}"
        out[name] = pywt.waverec(parts, wavelet, mode=mode)[:x.size]
    return out



def windowed_dwt_edge(window: np.ndarray, wavelet: str, level: int, mode: str) -> np.ndarray:
    """Per-bar energy of the newest coefficient at each level of one window's DWT.

    ``pywt.wavedec`` pads the window according to *mode*; the last detail
    coefficient at level ``j`` straddles the right edge, so its value
    depends on the padding. Returned: :math:`c_{j,last}^2 / 2^j` for
    ``j = 1..level`` (the decimated coefficient spans ``2^j`` bars).
    """
    pad = "zero" if mode == "zero" else mode
    coeffs = pywt.wavedec(np.asarray(window, dtype=np.float64), wavelet, mode=pad, level=level)
    details = coeffs[1:][::-1]          # cD_1 .. cD_level
    return np.array([float(d[-1]) ** 2 / 2 ** j for j, d in enumerate(details, start=1)])


# ---------------------------------------------------------------------------
# Offline continuous transform (NON-CAUSAL)
# ---------------------------------------------------------------------------
def cwt_scales(minimum: float, maximum: float, count: int, mode: str = "logarithmic"
               ) -> np.ndarray:
    """The scale grid: logarithmic (constant ratio) or linear."""
    if mode == "logarithmic":
        return np.geomspace(minimum, maximum, count)
    return np.linspace(minimum, maximum, count)


def scale_to_period(wavelet: str, scales: np.ndarray) -> np.ndarray:
    """Approximate period in bars of each scale: ``1 / pywt.scale2frequency``.

    The mapping depends on the wavelet's centre frequency - a scale number
    is not a period by itself.
    """
    return 1.0 / np.asarray(pywt.scale2frequency(wavelet, np.asarray(scales, dtype=np.float64)))


def _efold_factor(wavelet: str) -> float:
    """e-folding time of the wavelet's power per unit scale (Torrence & Compo)."""
    if wavelet.startswith("cmor"):
        bandwidth = float(wavelet[4:].split("-")[0])
        return float(np.sqrt(bandwidth / 2.0))      # |psi|^2 ~ exp(-2 t^2 / B)
    if wavelet == "morl":
        return 1.0                                   # |psi|^2 envelope ~ exp(-t^2)
    if wavelet == "mexh":
        return float(np.sqrt(2.0))                   # DOG m=2
    raise ValueError(f"no cone of influence defined for {wavelet!r}")


def cone_of_influence(n: int, wavelet: str) -> np.ndarray:
    """Largest period (bars) at each point that is not reached by the slice edges.

    A coefficient at scale ``s`` feels the edge within ``f_e s`` bars of it
    (``f_e`` the e-folding factor); inside the cone the scalogram is
    unreliable and is shaded, and ridges are only traced outside it.
    """
    distance = np.minimum(np.arange(n), np.arange(n)[::-1]).astype(np.float64)
    max_scale = distance / _efold_factor(wavelet)
    with np.errstate(divide="ignore"):
        return np.where(max_scale > 0, scale_to_period(wavelet, np.maximum(max_scale, 1e-9)),
                        0.0)


@dataclass
class OfflineScalogram:
    """|W|^2 of a whole slice. ``label`` is always :data:`NON_CAUSAL_LABEL`."""

    wavelet: str
    scales: np.ndarray
    periods_bars: np.ndarray
    power: np.ndarray               # (scales, time), float32
    coi_period_bars: np.ndarray     # (time,)
    bar_seconds: float
    label: str = NON_CAUSAL_LABEL

    @property
    def periods_seconds(self) -> np.ndarray:
        return self.periods_bars * self.bar_seconds

    @property
    def reliable(self) -> np.ndarray:
        """(scales, time) mask: outside the cone of influence."""
        return self.periods_bars[:, None] <= self.coi_period_bars[None, :]


def offline_cwt(values: np.ndarray, scales: np.ndarray, *, wavelet: str,
                bar_seconds: float) -> OfflineScalogram:
    """Continuous transform of a whole slice - centred, NON-CAUSAL, visualisation only.

    Missing values are filled with the slice mean for the transform (the
    slice is chosen complete in practice); the mean is removed first.
    """
    x = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(x)
    x = np.where(finite, x, np.nanmean(x) if finite.any() else 0.0)
    x = x - x.mean()
    coef, _ = pywt.cwt(x, np.asarray(scales, dtype=np.float64), wavelet, method="fft")
    power = (np.abs(coef) ** 2).astype(np.float32)
    return OfflineScalogram(wavelet=wavelet, scales=np.asarray(scales, dtype=np.float64),
                            periods_bars=scale_to_period(wavelet, scales), power=power,
                            coi_period_bars=cone_of_influence(x.size, wavelet),
                            bar_seconds=bar_seconds)


@dataclass(frozen=True)
class Ridge:
    """A path of locally maximal power through (time, scale)."""

    start: int
    end: int
    scale_index: np.ndarray
    periods_bars: np.ndarray
    power_ratio: np.ndarray

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    @property
    def mean_period(self) -> float:
        return float(np.exp(np.mean(np.log(self.periods_bars))))

    @property
    def cycles(self) -> float:
        """Duration in cycles of its own period - comparable across scales."""
        return self.length / self.mean_period

    @property
    def log_period_drift(self) -> float:
        """Least-squares slope of ln(period) per bar along the ridge."""
        if self.length < 3:
            return 0.0
        t = np.arange(self.length, dtype=np.float64)
        return float(np.polyfit(t, np.log(self.periods_bars), 1)[0])


def extract_ridges(scalogram: OfflineScalogram, *, power_ratio: float,
                   min_length: int = 2) -> list[Ridge]:
    """Ridges of an offline scalogram.

    1. At each scale, a point qualifies if its power is at least
       *power_ratio* times that scale's median power over the reliable part
       of the slice (a relative threshold, so the same rule applies to data
       and nulls of any scale), and it lies outside the cone of influence.
    2. Among qualifying points, keep local maxima across scale at each time.
    3. Link maxima left to right: a maximum at time ``t`` continues the
       ridge whose last point is at ``t-1`` within one scale step (the
       nearest; ties to the stronger), otherwise it starts a new ridge.

    Ridges shorter than *min_length* bars are dropped.
    """
    power = scalogram.power.astype(np.float64)
    reliable = scalogram.reliable
    n_scales, n_times = power.shape
    medians = np.array([np.median(row[mask]) if mask.any() else np.inf
                        for row, mask in zip(power, reliable, strict=True)])
    ratio = power / np.where(medians > 0, medians, np.inf)[:, None]
    candidate = (ratio >= power_ratio) & reliable
    peak = candidate.copy()
    peak[1:] &= power[1:] >= power[:-1]
    peak[:-1] &= power[:-1] >= power[1:]
    active: dict[int, list[tuple[int, int]]] = {}      # last scale -> points
    finished: list[list[tuple[int, int]]] = []
    for t in range(n_times):
        scales_t = np.flatnonzero(peak[:, t])
        order = scales_t[np.argsort(-power[scales_t, t])]
        next_active: dict[int, list[tuple[int, int]]] = {}
        used: set[int] = set()
        for s in order:
            options = sorted((abs(k - s), -power[k, t - 1], k) for k in active
                             if abs(k - s) <= 1 and k not in used)
            if options:
                k = options[0][2]
                used.add(k)
                path = active[k]
                path.append((t, int(s)))
            else:
                path = [(t, int(s))]
            next_active[int(s)] = path
        finished += [p for k, p in active.items() if k not in used]
        active = next_active
    finished += list(active.values())
    ridges = []
    for path in finished:
        if len(path) < min_length:
            continue
        times = np.array([p[0] for p in path])
        idx = np.array([p[1] for p in path])
        ridges.append(Ridge(start=int(times[0]), end=int(times[-1]), scale_index=idx,
                            periods_bars=scalogram.periods_bars[idx],
                            power_ratio=ratio[idx, times]))
    return sorted(ridges, key=lambda r: r.start)
