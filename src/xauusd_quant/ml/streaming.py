r"""Historical streaming simulation of a frozen model (Prompt #10, Steps 92-94).

At each replayed bar ``t`` the bars up to ``t`` are appended to a rolling
buffer (never a later bar), every feature family the model needs is recomputed
from the buffer alone, the newest row goes through the model's *frozen*
preprocessing, the model and its calibrator, and the result is compared with
the batch prediction computed from the stored feature matrix at ``t``.

Agreement is required within float32 rounding of the features (the buffer
starts at a different bar than the full history, see :mod:`..selection.live`).
Regime features are the Prompt #7 walk-forward filter's output; they are read
from the stored matrix, as the live pipeline would read them from the regime
service - recorded, not recomputed.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import numpy as np

from ..features.factory import compute_families
from ..features.families import BarSeries
from ..features.interactions import interaction_columns
from ..features.joins import EngineProvider
from .calibration import Calibrator
from .models import Model
from .preprocessing import Preprocessor

__all__ = ["stream_predictions"]

_ORDER = ("returns", "volatility", "autocorrelation", "regression", "ou", "fft", "wavelet",
          "microstructure", "time")


def _slice(bs: BarSeries, lo: int, hi: int) -> BarSeries:
    def cut(v: np.ndarray | None) -> np.ndarray | None:
        return None if v is None else v[lo:hi]

    return replace(bs, name=f"{bs.name}_stream", timestamps=bs.timestamps.slice(lo, hi - lo),
                   close=bs.close[lo:hi], open=cut(bs.open), high=cut(bs.high), low=cut(bs.low),
                   median_spread=cut(bs.median_spread), tick_count=cut(bs.tick_count),
                   missing_slots=cut(bs.missing_slots), notes=[])


def _families(features: list[str], registry: dict[str, dict[str, Any]]) -> tuple[str, ...]:
    fams = {str(registry[f].get("family")) for f in features} - {"regime", "interaction"}
    if any(registry[f].get("family") == "interaction" for f in features):
        fams |= {"returns", "volatility", "regression", "ou", "fft", "wavelet",
                 "microstructure", "time"}
    if "ou" in fams or "fft" in fams:
        fams.add("regression")
    if any(f.startswith("fft_abs_eta") for f in features):
        fams.add("ou")
    return tuple(f for f in _ORDER if f in fams)


def stream_predictions(model: Model, pre: Preprocessor, cal: Calibrator, features: list[str],
                       registry: dict[str, dict[str, Any]], bars: BarSeries,
                       stored: dict[str, np.ndarray], rows: list[int], *, buffer: int,
                       fcfg: Any, ou: Any, research: Any, regression: Any, spectral: Any,
                       wavelet: Any, rtol: float = 1e-4, atol: float = 1e-6) -> dict[str, Any]:
    """Replay *rows*; compare streaming and batch predictions of one frozen model."""
    fams = _families(features, registry)
    regime_like = [f for f in features if registry[f].get("family") == "regime"
                   or (registry[f].get("family") == "interaction" and "regime" in f)]
    batch_x = np.column_stack([stored[f][rows] for f in features]).astype(np.float32)
    batch = cal.apply(model.predict(pre.transform(batch_x, features)))
    live = np.empty(len(rows))
    started = time.perf_counter()
    worst_feature = 0.0
    for k, t in enumerate(rows):
        lo = max(0, t + 1 - buffer)
        window = _slice(bars, lo, t + 1)
        provider = EngineProvider(bars=window, regression_config=regression,
                                  spectral_config=spectral, wavelet_config=wavelet)
        values, _ = compute_families(window, provider, fcfg, ou, research, families=fams,
                                     log=False, collect=False)
        if any(registry[f].get("family") == "interaction" for f in features):
            values.update(interaction_columns(values, fcfg, window.bar_seconds))
        row = np.empty((1, len(features)), dtype=np.float32)
        for j, f in enumerate(features):
            row[0, j] = stored[f][t] if f in regime_like else np.float32(values[f][-1])
            b = float(batch_x[k, j])
            if np.isfinite(b) and np.isfinite(row[0, j]):
                worst_feature = max(worst_feature, abs(float(row[0, j]) - b))
        live[k] = float(cal.apply(model.predict(pre.transform(row, features)))[0])
    seconds = time.perf_counter() - started
    diff = np.abs(live - batch)
    tol = atol + rtol * np.abs(batch)
    failed = int((diff > tol).sum())
    return {"steps": len(rows), "families_recomputed": list(fams),
            "read_from_regime_service": regime_like, "buffer_bars": buffer,
            "max_abs_prediction_difference": float(diff.max()) if diff.size else 0.0,
            "max_abs_feature_difference": worst_feature, "mismatches": failed,
            "passed": failed == 0, "seconds_per_bar": seconds / max(1, len(rows)),
            "tolerance": {"rtol": rtol, "atol": atol},
            "live": [float(v) for v in live], "batch": [float(v) for v in batch]}
