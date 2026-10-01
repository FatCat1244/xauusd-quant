r"""Historical streaming simulation of a frozen ensemble (Prompt #11, Step 67).

At each replayed bar ``t``:

1. the bars up to ``t`` (never a later one) fill a rolling buffer and every
   feature family the constituents need is recomputed from the buffer alone,
   once for all constituents;
2. each constituent runs its frozen preprocessing, model and calibrator;
3. the frozen combination merges them (weights, meta-model, state weights - the
   context at ``t`` is passed exactly as the batch path receives it);
4. the frozen final calibrator is applied;
5. the result is stored and compared with the batch prediction from the stored
   feature matrix at ``t``.

Agreement within float32 rounding of the features is required (the buffer
starts at a different bar than the full history). Regime features are the
Prompt #7 walk-forward filter's output and are read from the stored matrix, as
the live pipeline would read them from the regime service.
"""

from __future__ import annotations

import time
from datetime import date, datetime
from typing import Any

import numpy as np
import polars as pl

from ..data.resampler import parse_timeframe
from ..features.factory import compute_families
from ..features.families import BarSeries
from ..features.interactions import interaction_columns
from ..features.joins import EngineProvider, SourceMismatchError
from ..ml.streaming import _families, _slice
from ..research.spectral_nulls import missing_bar_slots
from ..utils.config import Config
from .inference import EnsembleModel

__all__ = ["load_development_bars", "stream_ensemble"]


def load_development_bars(config: Config, timeframe: str, before: date) -> BarSeries:
    """The bars strictly before *before*; later bars are filtered in the scan and never
    collected (the reserved period's prices stay out of memory).

    A copy of :func:`~..features.factory.load_bar_series` with the cut-off, on purpose:
    that module's source is part of the feature-matrix build key, so a new parameter there
    would mark the stored matrix stale.
    """
    files = sorted(config.bars_dir(timeframe).rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {timeframe} bars under {config.bars_dir(timeframe)}")
    cut = datetime(before.year, before.month, before.day)
    bars = (pl.scan_parquet(files)
            .select("timestamp", "open", "high", "low", "close", "tick_count", "median_spread")
            .filter(pl.col("timestamp") < pl.lit(cut))
            .sort("timestamp").collect())
    if bars["timestamp"].is_duplicated().any():
        raise SourceMismatchError(f"{timeframe} bars: duplicate timestamps")
    bar_seconds = parse_timeframe(timeframe).total_seconds()

    def f64(name: str) -> np.ndarray:
        return bars[name].cast(pl.Float64).fill_null(np.nan).to_numpy()

    return BarSeries(
        name="real", timeframe=timeframe, timestamps=bars["timestamp"], close=f64("close"),
        bar_seconds=bar_seconds, open=f64("open"), high=f64("high"), low=f64("low"),
        median_spread=f64("median_spread"), tick_count=f64("tick_count"),
        missing_slots=missing_bar_slots(bars["timestamp"], bar_seconds, config))


def stream_ensemble(model: EnsembleModel, registry: dict[str, dict[str, Any]], bars: BarSeries,
                    stored: dict[str, np.ndarray], rows: list[int], *, buffer: int, fcfg: Any,
                    ou: Any, research: Any, regression: Any, spectral: Any, wavelet: Any,
                    context: dict[str, np.ndarray] | None = None, rtol: float = 1e-4,
                    atol: float = 1e-6) -> dict[str, Any]:
    """Replay *rows*; compare the streaming and batch outputs of one frozen ensemble.

    *context* holds one value per replayed row (in *rows* order) for the inputs the
    combination needs (regime probabilities, volatility bucket, trailing weights).
    """
    features = model.required_features()
    fams = _families(features, registry)
    regime_like = [f for f in features if registry[f].get("family") == "regime"
                   or (registry[f].get("family") == "interaction" and "regime" in f)]
    batch_x = {f: np.asarray(stored[f][rows], dtype=np.float32) for f in features}
    batch = model.predict_batch(batch_x, context)
    live = np.empty(len(rows))
    live_members = np.empty((len(rows), len(model.constituents)))
    worst_feature = 0.0
    started = time.perf_counter()
    for k, t in enumerate(rows):
        lo = max(0, t + 1 - buffer)
        window = _slice(bars, lo, t + 1)
        provider = EngineProvider(bars=window, regression_config=regression,
                                  spectral_config=spectral, wavelet_config=wavelet)
        values, _ = compute_families(window, provider, fcfg, ou, research, families=fams,
                                     log=False, collect=False)
        if any(registry[f].get("family") == "interaction" for f in features):
            values.update(interaction_columns(values, fcfg, window.bar_seconds))
        x = {}
        for f in features:
            v = np.float32(stored[f][t]) if f in regime_like else np.float32(values[f][-1])
            x[f] = np.array([v], dtype=np.float32)
            b = float(batch_x[f][k])
            if np.isfinite(b) and np.isfinite(v):
                worst_feature = max(worst_feature, abs(float(v) - b))
        ctx = None if context is None else {n: np.asarray(a)[k:k + 1] for n, a in
                                            context.items()}
        res = model.predict_batch(x, ctx)
        live[k] = float(res["prediction"][0])
        live_members[k] = res["constituents"][0]
    seconds = time.perf_counter() - started
    batch_pred = np.asarray(batch["prediction"], dtype=np.float64)
    diff = np.abs(live - batch_pred)
    tol = atol + rtol * np.abs(batch_pred)
    # a value on one path and none on the other is a mismatch; NaN on both (an invalid
    # row on both paths) is agreement
    failed = int(((diff > tol) | (np.isfinite(live) != np.isfinite(batch_pred))).sum())
    member_diff = np.abs(live_members - batch["constituents"])
    return {"steps": len(rows), "families_recomputed": list(fams),
            "read_from_regime_service": regime_like, "buffer_bars": buffer,
            "max_abs_prediction_difference": float(np.nanmax(diff)) if diff.size else 0.0,
            "max_abs_constituent_difference": float(np.nanmax(member_diff))
            if member_diff.size else 0.0,
            "max_abs_feature_difference": worst_feature, "mismatches": failed,
            "passed": failed == 0, "seconds_per_bar": seconds / max(1, len(rows)),
            "tolerance": {"rtol": rtol, "atol": atol}}
