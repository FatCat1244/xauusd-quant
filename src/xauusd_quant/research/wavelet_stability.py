r"""Is a wavelet state a property of the market, or of one window?

Everything here is computed identically on the real series and on each null
control, from the causal features of :mod:`~xauusd_quant.features.wavelet_causal`:

* **scale persistence** - :math:`P(\text{band}_{t+h} = \text{band}_t)`, exactly and
  within ``band_tolerance`` bands. Windows fewer than ``N`` bars apart share
  data, so only lags of at least ``N`` measure persistence beyond overlap;
* **runs** - how long the dominant band holds, uncapped, and how often it
  switches from one bar to the next;
* **frequency drift** - the distribution of the change in the energy-weighted
  (log) centroid period at several lags, and whether a drift continues: the
  correlation of consecutive disjoint drifts;
* **bursts** - how often, in which band and for how long the local energy of
  a band stands more than ``burst_z_threshold`` trailing standard deviations
  above its own recent level;
* **stability through time and conditioning** - the Prompt #5 tables
  (:mod:`spectral_stability`) with wavelet metrics: year, quarter,
  equal-length era, volatility / trend / OU state, hour and session.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from ..features.wavelet_causal import RollingWavelet
from .spectral_stability import summary_exprs

__all__ = [
    "WAVELET_METRICS",
    "band_persistence",
    "burst_summary",
    "drift_summary",
    "run_summary",
    "wavelet_metrics_frame",
    "wavelet_summaries",
]

#: Per-bar wavelet metrics summarised throughout the study.
WAVELET_METRICS: tuple[str, ...] = (
    "wavelet_entropy", "wavelet_dominant_energy_share", "wavelet_dominant_excess",
    "wavelet_top1_scale_share", "wavelet_top3_scale_share", "wavelet_active_scales", "wavelet_fast_energy_share", "wavelet_fast_slow_log_ratio",
    "wavelet_high_share", "wavelet_mid_share", "wavelet_low_share",
    "wavelet_centroid_period_bars", "wavelet_local_entropy", "wavelet_burst_z_max",
    "wavelet_dominant_period_bars", "wavelet_dominant_period_seconds",
)
_MAX_PAIRS = 500_000


def wavelet_summaries() -> list[pl.Expr]:
    """Group summaries of :data:`WAVELET_METRICS` (medians, period IQR, modal band)."""
    return summary_exprs(WAVELET_METRICS, period="wavelet_dominant_period_bars",
                         seconds="wavelet_dominant_period_seconds", mode="wavelet_dominant_band")


def wavelet_metrics_frame(result: RollingWavelet, timestamps: pl.Series,
                          extra: dict[str, np.ndarray] | None = None) -> pl.DataFrame:
    """Valid windows only: timestamp, the wavelet metrics, any *extra* columns."""
    rows = np.flatnonzero(result.valid)
    idx = pl.Series(rows)

    def f32(values: np.ndarray) -> np.ndarray:
        return np.asarray(values[rows], dtype=np.float32)

    columns: dict[str, Any] = {
        "timestamp": timestamps.gather(idx),
        "wavelet_entropy": f32(result.entropy),
        "wavelet_dominant_energy_share": f32(result.dominant_share),
        "wavelet_dominant_excess": f32(result.dominant_excess),
        "wavelet_top1_scale_share": f32(result.top1_share),
        "wavelet_top3_scale_share": f32(result.top3_share),
        "wavelet_active_scales": f32(result.active),
        "wavelet_fast_energy_share": f32(result.fast_share),
        "wavelet_fast_slow_log_ratio": f32(result.fast_slow_log_ratio),
        "wavelet_high_share": f32(result.group_shares["high"]),
        "wavelet_mid_share": f32(result.group_shares["mid"]),
        "wavelet_low_share": f32(result.group_shares["low"]),
        "wavelet_centroid_period_bars": f32(result.centroid_period_bars),
        "wavelet_local_entropy": f32(result.local_entropy),
        "wavelet_burst_z_max": f32(result.burst_z_max),
        "wavelet_dominant_period_bars": f32(result.dominant_period_bars),
        "wavelet_dominant_period_seconds": f32(result.dominant_period_seconds),
        "wavelet_dominant_band": (result.dominant[rows] + 1).astype(np.int16),
    }
    for i, name in enumerate(result.layout.band_names()):
        columns[f"share_{name}"] = f32(result.shares[i])
    for name, values in (extra or {}).items():
        columns[name] = values[rows]
    return pl.DataFrame(columns).with_columns(pl.col(pl.Float32).fill_nan(None))


def _pairs(valid: np.ndarray, lag: int) -> np.ndarray:
    start = np.flatnonzero(valid[:-lag] & valid[lag:]) if lag < valid.size else np.zeros(0, int)
    if start.size > _MAX_PAIRS:
        start = start[np.linspace(0, start.size - 1, _MAX_PAIRS).round().astype(np.int64)]
    return start


def band_persistence(result: RollingWavelet, lags: list[int], *, tolerance: int,
                     source: str) -> pl.DataFrame:
    """P(same dominant band h bars later), exactly and within *tolerance* bands."""
    rows = []
    for lag in lags:
        start = _pairs(result.valid, lag)
        if start.size == 0:
            continue
        a, b = result.dominant[start], result.dominant[start + lag]
        rows.append({"source": source, "lag": lag, "lag_in_windows": lag / result.window,
                     "pairs": int(start.size), "same_band": float(np.mean(a == b)),
                     "similar_band": float(np.mean(np.abs(a - b) <= tolerance)),
                     "beyond_overlap": lag >= result.window})
    return pl.DataFrame(rows, infer_schema_length=None)


def run_summary(result: RollingWavelet, *, source: str) -> dict[str, Any]:
    """Uncapped runs of the dominant band over consecutive valid bars."""
    valid, dom = result.valid, result.dominant
    if valid.sum() < 2:
        return {"source": source, "runs": 0}
    change = np.ones(dom.size, dtype=bool)
    change[1:] = (dom[1:] != dom[:-1]) | ~valid[:-1] | ~valid[1:]
    starts = np.flatnonzero(change & valid)
    ends = np.append(starts[1:], dom.size)
    lengths = np.array([int(valid[s:e].sum()) if not valid[s:e].all() else e - s
                        for s, e in zip(starts, ends, strict=True)], dtype=np.float64)
    lengths = lengths[lengths > 0]
    both = valid[1:] & valid[:-1]
    switching = float(np.mean(dom[1:][both] != dom[:-1][both])) if both.any() else None
    return {
        "source": source, "runs": int(lengths.size),
        "run_length_median": float(np.median(lengths)),
        "run_length_mean": float(lengths.mean()),
        "run_length_p90": float(np.quantile(lengths, 0.9)),
        "run_length_max": float(lengths.max()),
        "runs_at_least_one_window": float(np.mean(lengths >= result.window)),
        "switching_fraction_lag1": switching,
    }


def drift_summary(result: RollingWavelet, lags: list[int], quantiles: tuple[float, ...], *,
                  source: str) -> pl.DataFrame:
    """Distribution of centroid drift at each lag, and whether a drift continues.

    ``continuation`` is the correlation of the drift over ``[t-h, t]`` with the
    drift over ``[t-2h, t-h]`` - at ``h >= N`` the two use disjoint windows.
    """
    c = np.where(result.valid, result.log_centroid_period, np.nan)
    rows = []
    for lag in lags:
        if lag <= 0 or 2 * lag >= c.size:
            continue
        drift = np.full(c.size, np.nan)
        drift[lag:] = c[lag:] - c[:-lag]
        prev = np.full(c.size, np.nan)
        prev[lag:] = drift[:-lag]
        finite = np.isfinite(drift)
        values = drift[finite]
        if values.size < 10:
            continue
        both = finite & np.isfinite(prev)
        cont = (float(np.corrcoef(drift[both], prev[both])[0, 1])
                if both.sum() > 10 and np.std(drift[both]) > 0 else None)
        q = np.quantile(values, quantiles)
        rows.append({"source": source, "lag": lag, "lag_in_windows": lag / result.window,
                     "observations": int(values.size), "std": float(values.std()),
                     **{f"q{int(round(p * 100)):02d}": float(v)
                        for p, v in zip(quantiles, q, strict=True)},
                     "share_positive": float(np.mean(values > 0)),
                     "continuation": cont, "beyond_overlap": lag >= result.window})
    return pl.DataFrame(rows, infer_schema_length=None)


def burst_summary(result: RollingWavelet, threshold: float, *, source: str) -> pl.DataFrame:
    """Burst frequency overall and by band, and the length of burst episodes."""
    valid = result.valid & np.isfinite(result.burst_z_max)
    if not valid.any():
        return pl.DataFrame()
    burst = valid & (result.burst_z_max > threshold)
    edges = np.diff(np.concatenate(([0], burst.astype(np.int8), [0])))
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    rows = [{"source": source, "band": "any", "valid_bars": int(valid.sum()),
             "burst_fraction": float(burst.sum() / valid.sum()),
             "episodes": int(lengths.size),
             "episode_length_median": float(np.median(lengths)) if lengths.size else None,
             "episode_length_p90": float(np.quantile(lengths, 0.9)) if lengths.size else None}]
    for level in np.flatnonzero(result.burst_eligible) + 1:
        mask = burst & (result.burst_band == level)
        rows.append({"source": source, "band": f"d{level}", "valid_bars": int(valid.sum()),
                     "burst_fraction": float(mask.sum() / valid.sum()),
                     "episodes": None, "episode_length_median": None,
                     "episode_length_p90": None})
    return pl.DataFrame(rows, infer_schema_length=None)
