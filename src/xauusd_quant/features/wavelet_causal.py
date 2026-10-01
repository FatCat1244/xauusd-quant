r"""Causal rolling wavelet features: every value at bar ``t`` from bars ``<= t`` only.

For a rolling window of ``N`` bars and the one-sided MODWT
(:func:`~xauusd_quant.features.wavelet.causal_modwt`):

**Window state** (the window ``t-N+1 .. t``). The level-``j`` coefficients
whose whole filter lies inside the window are ``W_{j,s}`` for
``s = t-M_j+1 .. t`` with ``M_j = N - L_j + 1``. Their mean square is the
standard boundary-free (unbiased MODWT) estimate of the wavelet variance at
that scale - no padding anywhere, because no coefficient reaches outside the
window at either end. The approximation band contributes the variance of
``V_{J,s}`` over its complete coefficients. From these ``J+1`` band energies:

``wavelet_energy_share_<band>``   :math:`p_j`
``wavelet_total_energy``          :math:`\sum_j E_j`
``wavelet_entropy``               :math:`-\sum p_j \ln p_j / \ln(J+1)`
``wavelet_dominant_band``         :math:`\arg\max_j p_j / b_j` - the band whose share most
                                  exceeds its white-noise share :math:`b_j`
                                  (1..J details, J+1 approximation)
``wavelet_dominant_period_*``     the dominant band's nominal period, bars and seconds
``wavelet_dominant_energy_share`` :math:`p_{j^*}`, and ``wavelet_dominant_excess`` :math:`p_{j^*}/b_{j^*}`
``wavelet_top1/top3_scale_share`` share of the largest / three largest bands (raw)
``wavelet_active_scales``         bands above the uniform share
``wavelet_fast_* / slow_*``       shares of bands by physical period (``fast_slow_boundary``),
                                  and :math:`\ln(E_{fast}/E_{slow})`
``wavelet_high/mid/low_share``    shares by the two physical band edges (NaN if empty)
``wavelet_centroid_period_bars``  :math:`\exp\sum_j p_j \ln P_j`

**Change and persistence** (lookback beyond one window, still causal):

``wavelet_scale_drift``           :math:`\ln C_t - \ln C_{t-k}`, centroid ``C``, ``k = N/4``
``wavelet_fast_slow_change``      the fast/slow log ratio's change over the same ``k``
``wavelet_energy_log_change``     :math:`\ln E_t - \ln E_{t-N}` (the previous window)
``wavelet_dominant_run_length``   consecutive valid bars with this dominant band, capped at ``N``

**Latest state** (the newest coefficients - what the FFT cannot localise):

``wavelet_local_*``               shares of the local energies
                                  :math:`e_{j,t} = \operatorname{mean}(W_{j,t-2^j+1..t}^2)`
``wavelet_burst_z_max`` / ``_band`` the largest
                                  :math:`Z_{j,t} = (\ln e_{j,t} - \mu)/\sigma`, with
                                  :math:`\mu,\sigma` from that band's earlier local energies
                                  inside the same window (strictly before ``t``)

A local coefficient is delayed by about half its filter length: level ``j``
of ``db4`` describes the signal roughly ``3.5 (2^j - 1)`` bars back. That is
the price of using no future data, and it is recorded, not hidden.

**Validity.** A window is valid when all its values are finite, its
unscheduled gaps are at most ``missing_bar_tolerance`` of its slots (the
trading-time axis of Prompt #5), and its energy is non-zero; everything else
is NaN, never filled. Constant input has zero energy, so it yields no
entropy and no dominant scale rather than a misleading one.

Everything is built from direct convolution and prefix sums, so appending
bars changes no earlier value; the tests verify it bit for bit and at random
cut-offs. :data:`CAUSAL_FEATURES` is the schema a feature table may contain -
:func:`wavelet_feature_frame` refuses anything else, so offline
(``offline_*``) or non-causal values cannot be stored as features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from .spectral import window_validity
from .wavelet import causal_modwt, modwt_filters
from .wavelet_config import WaveletConfig
from .wavelet_energy import (
    WaveletLayout,
    active_scales,
    band_layout,
    concentration,
    energy_shares,
    group_share,
    white_noise_shares,
)
from .wavelet_entropy import wavelet_entropy

__all__ = [
    "CAUSAL_FEATURES",
    "RollingWavelet",
    "feature_lookback",
    "ic_feature_columns",
    "rolling_wavelet",
    "wavelet_feature_frame",
]

_TINY = 1e-300


@dataclass(frozen=True)
class FeatureSpec:
    """What a stored causal feature is, how far back it reads, whether it is live-safe."""

    description: str
    lookback: str          # "N", "N+k", "2N"
    live_safe: bool = True
    causal: bool = True


#: Every column a causal wavelet feature table may hold (per-band shares are
#: ``wavelet_energy_share_d<j>`` / ``_a`` and are validated by pattern).
CAUSAL_FEATURES: dict[str, FeatureSpec] = {
    "wavelet_window_valid": FeatureSpec("window complete, gaps within tolerance, energy > 0",
                                        "N"),
    "missing_bar_fraction": FeatureSpec("unscheduled missing slots / (N + missing)", "N"),
    "wavelet_total_energy": FeatureSpec("sum of band energies (wavelet variance)", "N"),
    "wavelet_entropy": FeatureSpec("normalised entropy of the band energy shares", "N"),
    "wavelet_dominant_band": FeatureSpec("band whose share most exceeds its white-noise "
                                         "share (J+1 = approx)", "N"),
    "wavelet_dominant_period_bars": FeatureSpec("dominant band's nominal period, bars", "N"),
    "wavelet_dominant_period_seconds": FeatureSpec("dominant band's nominal period, s", "N"),
    "wavelet_dominant_energy_share": FeatureSpec("energy share of the dominant band", "N"),
    "wavelet_dominant_excess": FeatureSpec("dominant band's share / its white-noise share",
                                           "N"),
    "wavelet_top1_scale_share": FeatureSpec("share of the largest band", "N"),
    "wavelet_top3_scale_share": FeatureSpec("share of the three largest bands", "N"),
    "wavelet_active_scales": FeatureSpec("bands above the uniform share", "N"),
    "wavelet_fast_energy_share": FeatureSpec("share of bands with period < boundary", "N"),
    "wavelet_slow_energy_share": FeatureSpec("share of bands with period >= boundary", "N"),
    "wavelet_fast_slow_log_ratio": FeatureSpec("ln(fast energy / slow energy)", "N"),
    "wavelet_fast_slow_change": FeatureSpec("fast/slow log ratio minus its value k bars ago",
                                            "N+k"),
    "wavelet_high_share": FeatureSpec("share of bands with period < first edge", "N"),
    "wavelet_mid_share": FeatureSpec("share of bands between the edges", "N"),
    "wavelet_low_share": FeatureSpec("share of bands with period >= second edge", "N"),
    "wavelet_centroid_period_bars": FeatureSpec("energy-weighted geometric mean period", "N"),
    "wavelet_scale_drift": FeatureSpec("ln centroid minus its value k bars ago", "N+k"),
    "wavelet_dominant_run_length": FeatureSpec("bars the dominant band has held (cap N)",
                                               "2N"),
    "wavelet_energy_log_change": FeatureSpec("ln total energy minus the previous window's",
                                             "2N"),
    "wavelet_burst_z_max": FeatureSpec("largest local-energy Z across bands (trailing)", "N"),
    "wavelet_burst_band": FeatureSpec("band of the largest local-energy Z", "N"),
    "wavelet_local_entropy": FeatureSpec("entropy of the latest local band energies", "N"),
    "wavelet_local_fast_slow_log_ratio": FeatureSpec("latest local ln(fast / slow), details",
                                                     "N"),
}
_SHARE_PREFIX = "wavelet_energy_share_"


def feature_lookback(name: str, window: int, drift_lag: int) -> int:
    """Bars of history a feature reads at ``t`` (the live buffer it needs)."""
    spec = CAUSAL_FEATURES.get(name)
    if spec is None and not name.startswith(_SHARE_PREFIX):
        raise KeyError(f"{name!r} is not a registered causal wavelet feature")
    kind = spec.lookback if spec is not None else "N"
    return {"N": window, "N+k": window + drift_lag, "2N": 2 * window}[kind]


@dataclass
class RollingWavelet:
    """Per-bar causal wavelet state of one series for one rolling window."""

    layout: WaveletLayout
    drift_lag: int
    valid: np.ndarray
    missing_fraction: np.ndarray
    energy: np.ndarray                 # (bands, n) float64
    shares: np.ndarray                 # (bands, n)
    total_energy: np.ndarray
    entropy: np.ndarray
    dominant: np.ndarray               # int16 band index 0..J, -1 invalid
    dominant_share: np.ndarray
    dominant_excess: np.ndarray
    top1_share: np.ndarray
    top3_share: np.ndarray
    active: np.ndarray
    fast_share: np.ndarray
    slow_share: np.ndarray
    fast_slow_log_ratio: np.ndarray
    fast_slow_change: np.ndarray
    group_shares: dict[str, np.ndarray]
    log_centroid_period: np.ndarray
    scale_drift: np.ndarray
    run_length: np.ndarray
    energy_log_change: np.ndarray
    burst_z_max: np.ndarray
    burst_band: np.ndarray             # int16 detail level 1..J, -1 none
    burst_eligible: np.ndarray         # bool (J,) bands with enough history for a Z
    local_entropy: np.ndarray
    local_fast_slow_log_ratio: np.ndarray
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def window(self) -> int:
        return self.layout.window

    @property
    def dominant_period_bars(self) -> np.ndarray:
        return np.where(self.dominant >= 0,
                        self.layout.period_bars[np.maximum(self.dominant, 0)], np.nan)

    @property
    def dominant_period_seconds(self) -> np.ndarray:
        return self.dominant_period_bars * self.layout.bar_seconds

    @property
    def centroid_period_bars(self) -> np.ndarray:
        return np.exp(self.log_centroid_period)


def _prefix(values: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))


def _trailing_mean(prefix: np.ndarray, width: int, n: int) -> np.ndarray:
    """Mean of the *width* values ending at each t (NaN before a full span)."""
    out = np.full(n, np.nan)
    if width <= n:
        out[width - 1:] = (prefix[width:] - prefix[:n - width + 1]) / width
    return out


def _lagged_difference(values: np.ndarray, lag: int) -> np.ndarray:
    out = np.full(values.size, np.nan)
    if 0 < lag < values.size:
        out[lag:] = values[lag:] - values[:-lag]
    return out


def _run_length(dominant: np.ndarray, valid: np.ndarray, cap: int) -> np.ndarray:
    """Consecutive valid bars (ending at t) with the same dominant band, capped."""
    n = dominant.size
    idx = np.arange(n)
    change = np.ones(n, dtype=bool)
    change[1:] = (dominant[1:] != dominant[:-1]) | ~valid[:-1]
    start = np.maximum.accumulate(np.where(change, idx, 0))
    run = (idx - start + 1).astype(np.float64)
    return np.where(valid, np.minimum(run, cap), np.nan)


def rolling_wavelet(values: np.ndarray, *, window: int, config: WaveletConfig,
                    bar_seconds: float, missing_slots: np.ndarray | None = None,
                    wavelet: str | None = None) -> RollingWavelet:
    """Causal rolling wavelet features of *values* for a window of *window* bars."""
    cf = config.causal_features
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    layout = band_layout(window, wavelet=wavelet or config.dwt.wavelet, bar_seconds=bar_seconds,
                         min_coefficients=cf.min_level_coefficients,
                         fast_slow_boundary_seconds=cf.fast_slow_boundary_seconds,
                         band_edges_seconds=(cf.band_edges_seconds[0],
                                             cf.band_edges_seconds[1]))
    levels = layout.levels
    filters = modwt_filters(layout.wavelet, levels)
    coeffs = causal_modwt(x, filters)

    valid = np.zeros(n, dtype=bool)
    missing = np.full(n, np.nan)
    complete = np.zeros(0, dtype=bool)
    if n >= window:
        complete, fraction, ok = window_validity(x, window, missing_slots=missing_slots,
                                                 tolerance=cf.missing_bar_tolerance)
        valid[window - 1:] = ok
        missing[window - 1:] = fraction

    # --- band energies: mean square of the complete coefficients in the window
    energy = np.full((levels + 1, n), np.nan)
    for j in range(levels):
        span = window - filters.lengths[j] + 1
        energy[j] = _trailing_mean(_prefix(coeffs.detail[j] ** 2), span, n)
    span = window - filters.scaling.size + 1
    v = coeffs.scaling
    shift = v[filters.scaling.size - 1] if n >= filters.scaling.size else 0.0
    mean = _trailing_mean(_prefix(v - shift), span, n)
    square = _trailing_mean(_prefix((v - shift) ** 2), span, n)
    energy[levels] = np.maximum(square - mean ** 2, 0.0)
    energy[:, ~valid] = np.nan
    total = energy.sum(axis=0)
    zero = valid & ~(total > cf.zero_energy_tolerance)
    valid &= ~zero
    energy[:, ~valid] = np.nan
    total = np.where(valid, total, np.nan)

    shares = energy_shares(energy, tolerance=cf.zero_energy_tolerance)
    entropy = wavelet_entropy(shares, normalise=config.entropy.normalise)
    excess = shares / white_noise_shares(layout)[:, None]
    dominant = np.where(valid, np.argmax(np.nan_to_num(excess, nan=-1.0), axis=0), -1
                        ).astype(np.int16)
    pick = np.maximum(dominant, 0)[None, :]
    dominant_share = np.where(valid, np.take_along_axis(shares, pick, axis=0)[0], np.nan)
    dominant_excess = np.where(valid, np.take_along_axis(excess, pick, axis=0)[0], np.nan)
    top1 = concentration(shares, 1)
    top3 = concentration(shares, 3)
    active = active_scales(shares)
    fast_share = group_share(shares, layout.fast)
    slow_share = group_share(shares, ~layout.fast)
    with np.errstate(divide="ignore", invalid="ignore"):
        fs_log = np.where((fast_share > 0) & (slow_share > 0),
                          np.log(fast_share) - np.log(slow_share), np.nan)
        log_centroid = np.where(valid, np.nansum(shares * np.log(layout.period_bars)[:, None],
                                                 axis=0), np.nan)
        log_total = np.log(total)
    groups = {name: group_share(shares, mask) for name, mask in layout.groups.items()}
    drift_lag = max(1, int(round(window * cf.drift_lag_fraction)))
    scale_drift = _lagged_difference(log_centroid, drift_lag)
    fs_change = _lagged_difference(fs_log, drift_lag)
    energy_change = _lagged_difference(log_total, window)
    run = _run_length(dominant, valid, window)

    # --- latest state: local energies and trailing burst Z-scores
    local = np.full((levels, n), np.nan)
    z = np.full((levels, n), np.nan)
    eligible = np.zeros(levels, dtype=bool)
    for j in range(levels):
        period = 2 ** (j + 1)                       # level j+1 spans ~2^(j+1) bars
        span = window - filters.lengths[j] + 1
        if period > span:
            continue
        e = _trailing_mean(_prefix(coeffs.detail[j] ** 2), period, n)
        local[j] = e
        history = span - period
        if history < cf.burst_min_history:
            continue
        eligible[j] = True
        y = np.log(np.maximum(np.nan_to_num(e, nan=0.0), cf.zero_energy_tolerance))
        base = y[period - 1] if n >= period else 0.0
        p1, p2 = _prefix(y - base), _prefix((y - base) ** 2)
        m = np.full(n, np.nan)
        s2 = np.full(n, np.nan)
        if history < n:
            # strictly earlier local energies: t-history .. t-1
            m[history:] = (p1[history:n] - p1[:n - history]) / history
            s2[history:] = (p2[history:n] - p2[:n - history]) / history - m[history:] ** 2
        sd = np.sqrt(np.maximum(s2, 0.0) * history / (history - 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            z[j] = np.where(sd > 0, (y - base - m) / sd, np.nan)
    local[:, ~valid] = np.nan
    z[:, ~valid] = np.nan
    usable = z[eligible]
    if usable.size:
        any_z = ~np.isnan(usable).all(axis=0)
        burst_max = np.where(any_z, np.nanmax(np.where(np.isnan(usable), -np.inf, usable),
                                              axis=0), np.nan)
        burst_idx = np.flatnonzero(eligible)[np.argmax(np.where(np.isnan(usable), -np.inf,
                                                                usable), axis=0)]
        burst_band = np.where(any_z, burst_idx + 1, -1).astype(np.int16)
    else:
        burst_max = np.full(n, np.nan)
        burst_band = np.full(n, -1, dtype=np.int16)
    present = ~np.isnan(local).all(axis=1)
    local_shares = energy_shares(local[present], tolerance=cf.zero_energy_tolerance)
    local_entropy = wavelet_entropy(local_shares, normalise=config.entropy.normalise)
    detail_fast = layout.fast[:levels][present]
    lf = group_share(local_shares, detail_fast)
    ls = group_share(local_shares, ~detail_fast)
    with np.errstate(divide="ignore", invalid="ignore"):
        local_fs = np.where((lf > 0) & (ls > 0), np.log(lf) - np.log(ls), np.nan)

    windows = max(n - window + 1, 0)
    counts = {
        "windows": windows,
        "valid": int(valid.sum()),
        "incomplete": int((~complete).sum()) if windows else 0,
        "too_many_missing_bars": int(windows - int(valid.sum()) - int(zero.sum())
                                     - (int((~complete).sum()) if windows else 0)),
        "zero_energy": int(zero.sum()),
    }
    return RollingWavelet(
        layout=layout, drift_lag=drift_lag, valid=valid, missing_fraction=missing,
        energy=energy, shares=shares, total_energy=total, entropy=entropy, dominant=dominant,
        dominant_share=dominant_share, dominant_excess=dominant_excess, top1_share=top1,
        top3_share=top3, active=active, fast_share=fast_share,
        slow_share=slow_share, fast_slow_log_ratio=fs_log, fast_slow_change=fs_change,
        group_shares=groups, log_centroid_period=log_centroid, scale_drift=scale_drift,
        run_length=run, energy_log_change=energy_change, burst_z_max=burst_max,
        burst_band=burst_band, burst_eligible=eligible, local_entropy=local_entropy,
        local_fast_slow_log_ratio=local_fs, counts=counts)


def ic_feature_columns(result: RollingWavelet, names: tuple[str, ...],
                       rows: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """The causal features an IC / model study reads (logs where scale-like), NaN if undefined."""
    idx: slice | np.ndarray = slice(None) if rows is None else rows

    def at(values: np.ndarray) -> np.ndarray:
        return np.asarray(values[idx], dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        table = {
            "wavelet_entropy": lambda: at(result.entropy),
            "wavelet_dominant_energy_share": lambda: at(result.dominant_share),
            "wavelet_dominant_excess": lambda: np.log(at(result.dominant_excess)),
            "wavelet_top1_scale_share": lambda: at(result.top1_share),
            "wavelet_top3_scale_share": lambda: at(result.top3_share),
            "wavelet_log_dominant_period": lambda: np.log(at(result.dominant_period_bars)),
            "wavelet_active_scales": lambda: at(result.active),
            "wavelet_fast_slow_log_ratio": lambda: at(result.fast_slow_log_ratio),
            "wavelet_fast_slow_change": lambda: at(result.fast_slow_change),
            "wavelet_log_centroid_period": lambda: at(result.log_centroid_period),
            "wavelet_scale_drift": lambda: at(result.scale_drift),
            "wavelet_log_run_length": lambda: np.log(at(result.run_length)),
            "wavelet_energy_log_change": lambda: at(result.energy_log_change),
            "wavelet_burst_z_max": lambda: at(result.burst_z_max),
            "wavelet_local_entropy": lambda: at(result.local_entropy),
            "wavelet_local_fast_slow_log_ratio": lambda: at(result.local_fast_slow_log_ratio),
        }
        unknown = [n for n in names if n not in table]
        if unknown:
            raise KeyError(f"unknown wavelet IC feature(s) {unknown}; known: {sorted(table)}")
        return {name: table[name]() for name in names}


def wavelet_feature_frame(result: RollingWavelet, timestamps: pl.Series, *,
                          float32: bool = True) -> pl.DataFrame:
    """Per-bar causal features, joinable on ``timestamp``; only registered columns."""
    ftype = pl.Float32 if float32 else pl.Float64
    layout = result.layout
    columns: dict[str, Any] = {
        "timestamp": timestamps,
        "wavelet_window_valid": pl.Series(result.valid),
        "missing_bar_fraction": result.missing_fraction,
    }
    for i, name in enumerate(layout.band_names()):
        columns[f"{_SHARE_PREFIX}{name}"] = result.shares[i]
    columns.update({
        "wavelet_total_energy": result.total_energy,
        "wavelet_entropy": result.entropy,
        "wavelet_dominant_band": np.where(result.dominant >= 0, result.dominant + 1, -1),
        "wavelet_dominant_period_bars": result.dominant_period_bars,
        "wavelet_dominant_period_seconds": result.dominant_period_seconds,
        "wavelet_dominant_energy_share": result.dominant_share,
        "wavelet_dominant_excess": result.dominant_excess,
        "wavelet_top1_scale_share": result.top1_share,
        "wavelet_top3_scale_share": result.top3_share,
        "wavelet_active_scales": result.active,
        "wavelet_fast_energy_share": result.fast_share,
        "wavelet_slow_energy_share": result.slow_share,
        "wavelet_fast_slow_log_ratio": result.fast_slow_log_ratio,
        "wavelet_fast_slow_change": result.fast_slow_change,
        "wavelet_high_share": result.group_shares["high"],
        "wavelet_mid_share": result.group_shares["mid"],
        "wavelet_low_share": result.group_shares["low"],
        "wavelet_centroid_period_bars": result.centroid_period_bars,
        "wavelet_scale_drift": result.scale_drift,
        "wavelet_dominant_run_length": result.run_length,
        "wavelet_energy_log_change": result.energy_log_change,
        "wavelet_burst_z_max": result.burst_z_max,
        "wavelet_burst_band": result.burst_band,
        "wavelet_local_entropy": result.local_entropy,
        "wavelet_local_fast_slow_log_ratio": result.local_fast_slow_log_ratio,
    })
    _enforce_schema(columns)
    frame = pl.DataFrame(columns)
    ints = {"wavelet_dominant_band", "wavelet_burst_band"}
    exprs = []
    for name in frame.columns:
        if name in ("timestamp", "wavelet_window_valid"):
            continue
        if name in ints:
            exprs.append(pl.when(pl.col(name) > 0).then(pl.col(name)).otherwise(None)
                         .cast(pl.Int8).alias(name))
        elif name in ("wavelet_active_scales", "wavelet_dominant_run_length"):
            exprs.append(pl.col(name).fill_nan(None).cast(pl.Int32).alias(name))
        else:
            exprs.append(pl.col(name).cast(pl.Float64).fill_nan(None).cast(ftype).alias(name))
    return frame.with_columns(exprs)


def _enforce_schema(columns: dict[str, Any]) -> None:
    """A feature table holds causal, registered columns only - never offline values."""
    for name in columns:
        if name == "timestamp":
            continue
        if name.startswith("offline_"):
            raise ValueError(f"{name!r}: offline (non-causal) values can never be stored "
                             "as features")
        if name not in CAUSAL_FEATURES and not name.startswith(_SHARE_PREFIX):
            raise ValueError(f"{name!r} is not a registered causal wavelet feature")
