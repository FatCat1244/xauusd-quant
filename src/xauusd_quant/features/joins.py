r"""Joining feature sources on the canonical timestamp, with version checks (Steps 2, 7).

Every source is aligned to one bar sequence - the bars of the current bar
dataset version - by exact timestamp: duplicates are refused, a source row
without a bar is refused, and a bar without a source row becomes missing
(never filled). A stored feature set is read only if its manifest carries the
current tick and bar versions and engine fingerprint; otherwise it is
recomputed with the same engine, or - for the regime store, which cannot be
recomputed cheaply - the gate stops the factory and says what to rebuild.

:class:`RealProvider` serves the real series (versioned Prompt #3 store,
stored Prompt #5 / #6 / #7 sets); :class:`EngineProvider` runs the same
engines on a pipeline null's own price path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..features.config import RegressionConfig
from ..features.rolling_regression import rolling_regression_features
from ..features.spectral import rolling_spectrum, spectral_feature_frame
from ..features.spectral_config import SpectralConfig
from ..features.store import RegressionFeatureStore, StaleFeaturesError
from ..features.wavelet_causal import rolling_wavelet, wavelet_feature_frame
from ..features.wavelet_config import WaveletConfig
from ..utils.logging import get_logger
from .families import BarSeries

__all__ = [
    "EngineProvider",
    "RealProvider",
    "SourceMismatchError",
    "align_to_bars",
    "stored_set_problems",
]

LOGGER = get_logger("features.joins")

_FFT_COLUMNS = ("spectral_entropy", "spectral_flatness", "top3_power_share", "low_power_share",
                "high_power_share", "fft_period_bars_1", "spectral_centroid_norm")
_WAVELET_COLUMNS = ("wavelet_entropy", "wavelet_fast_slow_log_ratio",
                    "wavelet_dominant_period_bars", "wavelet_dominant_run_length",
                    "wavelet_top3_scale_share", "wavelet_scale_drift")
_REGIME_BASE = ("regime_age_bars", "leave_probability")


class SourceMismatchError(RuntimeError):
    """A source cannot be joined to the current bars (version, rows or timestamps)."""


def align_to_bars(frame: pl.DataFrame, timestamps: pl.Series, what: str) -> pl.DataFrame:
    """*frame* reindexed to *timestamps* exactly; missing bars become null.

    Refuses duplicate timestamps in the source and source rows that match no
    bar (a different bar version would show up exactly this way).
    """
    if frame.height == timestamps.len() and frame["timestamp"].equals(timestamps):
        return frame
    if frame["timestamp"].is_duplicated().any():
        raise SourceMismatchError(f"{what}: duplicate timestamps in the source")
    extra = frame.join(pl.DataFrame({"timestamp": timestamps}), on="timestamp", how="anti")
    if extra.height:
        raise SourceMismatchError(f"{what}: {extra.height} source rows match no bar of the current "
                             f"bar dataset (first {extra['timestamp'][0]})")
    joined = pl.DataFrame({"timestamp": timestamps}).join(frame, on="timestamp", how="left")
    missing = joined.height - frame.height
    if missing:
        LOGGER.info("%s: %d bars have no source row (kept as missing)", what, missing)
    return joined


def stored_set_problems(manifest: dict[str, Any] | None, lineage: dict[str, Any], *,
                        engine: str | None = None, rows: bool = True) -> list[str]:
    """What makes a stored feature set unusable for the current dataset (empty = usable)."""
    if manifest is None:
        return ["no manifest"]
    problems = []
    if manifest.get("tick_dataset_version") != lineage.get("tick_dataset_version"):
        problems.append("tick version")
    if manifest.get("bar_dataset_version") != lineage.get("bar_dataset_version"):
        problems.append("bar version")
    if engine is not None and manifest.get("engine_fingerprint") != engine:
        problems.append("engine")
    if rows and manifest.get("rows") != lineage.get("bar_rows"):
        problems.append("rows")
    return problems


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _arrays(frame: pl.DataFrame, columns: tuple[str, ...]) -> dict[str, np.ndarray]:
    return {c: frame[c].cast(pl.Float64).fill_null(np.nan).to_numpy() for c in columns}


def _fft_arrays(values: np.ndarray, bs: BarSeries, window: int,
                spectral: SpectralConfig) -> dict[str, np.ndarray]:
    result = rolling_spectrum(values, fft_window=window, config=spectral,
                              bar_seconds=bs.bar_seconds, missing_slots=bs.missing_slots)
    frame = spectral_feature_frame(result, bs.timestamps, spectral)
    return _arrays(frame, _FFT_COLUMNS)


def _wavelet_arrays(values: np.ndarray, bs: BarSeries, window: int,
                    wavelet: WaveletConfig) -> dict[str, np.ndarray]:
    result = rolling_wavelet(values, window=window, config=wavelet, bar_seconds=bs.bar_seconds,
                             missing_slots=bs.missing_slots)
    frame = wavelet_feature_frame(result, bs.timestamps, float32=wavelet.features_output.float32)
    return _arrays(frame, _WAVELET_COLUMNS)


@dataclass
class EngineProvider:
    """Everything computed from the source's own path (a pipeline null, or a fallback)."""

    bars: BarSeries
    regression_config: RegressionConfig
    spectral_config: SpectralConfig
    wavelet_config: WaveletConfig
    series: dict[str, np.ndarray] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    _regression_cache: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)

    def regression(self, window: int) -> dict[str, np.ndarray]:
        if window not in self._regression_cache:
            frame = pl.DataFrame({"timestamp": self.bars.timestamps, "close": self.bars.close})
            out, _ = rolling_regression_features(frame, window=window,
                                                 config=self.regression_config,
                                                 timeframe=self.bars.timeframe)
            self._regression_cache[window] = _regression_columns(out)
        return self._regression_cache[window]

    def drop_regression(self, window: int) -> None:
        self._regression_cache.pop(window, None)

    def fft(self, series: str, window: int) -> dict[str, np.ndarray] | None:
        values = self.series.get(series)
        if values is None:
            return None
        return _fft_arrays(values, self.bars, window, self.spectral_config)

    def wavelet(self, series: str, window: int) -> dict[str, np.ndarray] | None:
        values = self.series.get(series)
        if values is None:
            return None
        return _wavelet_arrays(values, self.bars, window, self.wavelet_config)

    def regime(self) -> dict[str, np.ndarray] | None:
        return None                   # regime features are not computed on the nulls


def _regression_columns(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    names = {"slope": "regression_slope", "r_squared": "r_squared", "residual": "residual",
             "residual_std_fit": "residual_std_fit",
             "residual_zscore_rolling": "residual_zscore_rolling",
             "sigma": "trailing_volatility"}
    return {k: frame[v].cast(pl.Float64).fill_null(np.nan).to_numpy() for k, v in names.items()}


@dataclass
class RealProvider(EngineProvider):
    """The real series: stored, versioned sets where current, the same engines otherwise."""

    config: Any = None
    lineage: dict[str, Any] = field(default_factory=dict)
    regime_dir: Path | None = None
    regime_states: int = 3
    sources: dict[str, Any] = field(default_factory=dict)

    def regression(self, window: int) -> dict[str, np.ndarray]:
        if window in self._regression_cache:
            return self._regression_cache[window]
        store = RegressionFeatureStore(self.config, self.regression_config)
        try:
            frame = store.load(self.bars.timeframe, window,
                               ["timestamp", "regression_slope", "r_squared", "residual",
                                "residual_std_fit", "residual_zscore_rolling",
                                "trailing_volatility"])
        except (FileNotFoundError, StaleFeaturesError) as exc:
            raise SourceMismatchError(f"regression store N={window}: {exc}") from exc
        if not frame["timestamp"].equals(self.bars.timestamps):
            raise SourceMismatchError(f"regression store N={window}: timestamps differ from the bars")
        manifest = store.manifest(self.bars.timeframe) or {}
        entry = (manifest.get("windows") or {}).get(str(window)) or {}
        self.sources[f"regression_{window}"] = {"kind": "stored",
                                                "feature_version": entry.get("feature_version")}
        self._regression_cache[window] = _regression_columns(frame)
        return self._regression_cache[window]

    def _stored(self, root: Path, pattern: str, columns: tuple[str, ...], engine: str,
                what: str) -> dict[str, np.ndarray] | None:
        manifest = _read_manifest(root / "_manifest.json")
        problems = stored_set_problems(manifest, self.lineage, engine=engine)
        if problems:
            return None
        files = sorted(root.glob(pattern))
        if not files:
            return None
        frame = pl.read_parquet(files, columns=["timestamp", *columns]).sort("timestamp")
        frame = align_to_bars(frame, self.bars.timestamps, what)
        self.sources[what] = {"kind": "stored", "path": root.as_posix(),
                              "generated_utc": (manifest or {}).get("generated_utc"),
                              "engine_fingerprint": engine}
        return _arrays(frame, columns)

    def fft(self, series: str, window: int) -> dict[str, np.ndarray] | None:
        root = (self.spectral_config.features_path / f"timeframe={self.bars.timeframe}"
                / f"source={series}" / f"fft_window={window}")
        what = f"fft_{series}_{window}"
        engine = self.spectral_config.engine_fingerprint()
        stored = self._stored(root, "year=*/features.parquet", _FFT_COLUMNS, engine, what)
        if stored is not None:
            return stored
        computed = super().fft(series, window)
        if computed is not None:
            self.sources[what] = {"kind": "computed", "engine_fingerprint": engine}
        return computed

    def wavelet(self, series: str, window: int) -> dict[str, np.ndarray] | None:
        root = (self.wavelet_config.features_path / f"timeframe={self.bars.timeframe}"
                / f"source={series}" / f"window={window}")
        what = f"wavelet_{series}_{window}"
        engine = self.wavelet_config.engine_fingerprint()
        stored = self._stored(root, "year=*/part-0.parquet", _WAVELET_COLUMNS, engine, what)
        if stored is not None:
            return stored
        computed = super().wavelet(series, window)
        if computed is not None:
            self.sources[what] = {"kind": "computed", "engine_fingerprint": engine}
        return computed

    def regime(self) -> dict[str, np.ndarray] | None:
        if self.regime_dir is None:
            return None
        manifest = _read_manifest(self.regime_dir / "_manifest.json")
        problems = stored_set_problems(manifest, self.lineage, rows=False)
        if problems or not (manifest or {}).get("live_safe"):
            raise SourceMismatchError(f"regime features at {self.regime_dir}: "
                                 f"{', '.join(problems) or 'not live-safe'} - rebuild them with "
                                 "`xq regime-walk-forward`")
        k = self.regime_states
        files = sorted(self.regime_dir.glob("year=*/part-0.parquet"))
        model = str(manifest.get("model") if manifest else "hmm")
        probs = [f"{model}_p{j}" for j in range(k)]
        nxt = [f"next_state_p{j}" for j in range(k)]
        cols = ["timestamp", *probs, f"{model}_entropy", f"{model}_state_confidence",
                *_REGIME_BASE, *nxt]
        frame = pl.read_parquet(files, columns=cols).sort("timestamp")
        frame = align_to_bars(frame, self.bars.timestamps, "regime features")
        arr = _arrays(frame, tuple(cols[1:]))
        age = arr["regime_age_bars"]
        out = {f"regime_p{j}": arr[probs[j]] for j in range(k)}
        out["regime_entropy"] = arr[f"{model}_entropy"]
        out["regime_confidence"] = arr[f"{model}_state_confidence"]
        out["regime_age"] = np.where(np.isfinite(age) & (age >= 0), np.log1p(age), np.nan)
        for j in range(k):
            out[f"regime_next_p{j}"] = arr[nxt[j]]
        out["regime_leave_prob"] = arr["leave_probability"]
        self.sources["regime"] = {"kind": "stored", "path": self.regime_dir.as_posix(),
                                  "regime_model_version": (manifest or {}).get(
                                      "regime_model_version"),
                                  "first_timestamp": (manifest or {}).get("first_timestamp")}
        return out
