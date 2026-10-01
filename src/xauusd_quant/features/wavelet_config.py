"""Typed configuration for the wavelet / time-frequency research layer.

Same contract as :mod:`xauusd_quant.features.spectral_config`: frozen
dataclasses, unknown keys are an error, and the settings hash to a
fingerprint recorded in every report and in the version of every cached
feature set.

``config/wavelet.yaml`` governs only the wavelet study. The inputs keep their
own definitions (the residual: ``config/regression.yaml``; the OU
innovation: ``config/ou.yaml``; sessions: ``config/research.yaml``), and the
Fourier features it is compared with are computed by the Prompt #5 engine
under ``config/spectral.yaml``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pywt
import yaml

from ..utils.config import ConfigError, _as_dict, _build, _expand, _one_of
from ..utils.paths import find_project_root, resolve_path
from .spectral_config import INPUT_SERIES, ConditioningConfig, StabilityConfig

__all__ = [
    "IC_FEATURES",
    "WaveletConfig",
    "default_wavelet_config_path",
    "load_wavelet_config",
]

DEFAULT_WAVELET_RELPATH = "config/wavelet.yaml"
WAVELET_CONFIG_ENV_VAR = "XAUUSD_WAVELET_CONFIG"

#: Causal features an IC / incremental-information study may use (the default set).
IC_FEATURES: tuple[str, ...] = (
    "wavelet_entropy", "wavelet_dominant_excess", "wavelet_top3_scale_share",
    "wavelet_log_dominant_period", "wavelet_active_scales", "wavelet_fast_slow_log_ratio",
    "wavelet_fast_slow_change", "wavelet_log_centroid_period", "wavelet_scale_drift",
    "wavelet_log_run_length", "wavelet_energy_log_change", "wavelet_burst_z_max",
    "wavelet_local_entropy", "wavelet_local_fast_slow_log_ratio",
)
#: Every feature name the IC and model studies accept.
IC_FEATURE_NAMES: tuple[str, ...] = (*IC_FEATURES, "wavelet_dominant_energy_share",
                                     "wavelet_top1_scale_share")
IC_TARGETS: tuple[str, ...] = (
    "future_return", "future_residual_change", "future_abs_residual_reduction",
    "future_volatility", "future_innovation_magnitude",
)
BINARY_TARGETS: tuple[str, ...] = ("residual_shrinks", "ou_zscore_shrinks")
PADDING_MODES: tuple[str, ...] = ("zero", "symmetric", "reflect", "constant", "periodization",
                                  "smooth")


def _ints(values: Any, where: str, *, minimum: int = 1) -> tuple[int, ...]:
    out = tuple(int(v) for v in values)
    if any(v < minimum for v in out):
        raise ConfigError(f"{where}: every value must be >= {minimum}")
    return out


def _check_discrete(name: str, where: str) -> None:
    if name not in pywt.wavelist(kind="discrete"):
        raise ConfigError(f"{where}: {name!r} is not a PyWavelets discrete wavelet")
    if not pywt.Wavelet(name).orthogonal:
        raise ConfigError(f"{where}: {name!r} is not orthogonal; the MODWT energy "
                          "decomposition needs an orthogonal wavelet")


def _check_continuous(name: str, where: str) -> None:
    family = name.split("-")[0].rstrip("0123456789.")
    if family not in ("cmor", "morl", "mexh"):
        raise ConfigError(f"{where}: supported CWT wavelets are cmorB-C, morl and mexh; "
                          f"got {name!r}")
    try:
        pywt.ContinuousWavelet(name)
    except ValueError as exc:
        raise ConfigError(f"{where}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    cwt_enabled: bool = True
    dwt_enabled: bool = True


@dataclass(frozen=True, slots=True)
class CWTScalesConfig:
    mode: str = "logarithmic"
    min: float = 2.0
    max: float = 256.0
    count: int = 48

    def __post_init__(self) -> None:
        _one_of(self.mode, ("logarithmic", "linear"), "cwt.scales.mode")
        if not 0 < float(self.min) < float(self.max):
            raise ConfigError("cwt.scales: need 0 < min < max")
        if self.count < 4:
            raise ConfigError("cwt.scales.count: must be >= 4")


@dataclass(frozen=True, slots=True)
class CWTConfig:
    """Offline continuous transform: scalograms and ridges, never features."""

    wavelet: str = "cmor1.5-1.0"
    compare_wavelet: str | None = "mexh"
    scales: CWTScalesConfig = field(default_factory=CWTScalesConfig)
    ridge_power_ratio: float = 3.0
    slice_bars: int = 2048

    def __post_init__(self) -> None:
        _check_continuous(self.wavelet, "cwt.wavelet")
        if self.compare_wavelet is not None:
            _check_continuous(self.compare_wavelet, "cwt.compare_wavelet")
        if self.ridge_power_ratio <= 1.0:
            raise ConfigError("cwt.ridge_power_ratio: must be > 1")
        if self.slice_bars < 256:
            raise ConfigError("cwt.slice_bars: must be >= 256")


@dataclass(frozen=True, slots=True)
class DWTConfig:
    wavelet: str = "db4"
    compare_wavelets: tuple[str, ...] = ("haar", "sym4")
    levels: str = "auto"

    def __post_init__(self) -> None:
        object.__setattr__(self, "compare_wavelets", tuple(self.compare_wavelets))
        _check_discrete(self.wavelet, "dwt.wavelet")
        for name in self.compare_wavelets:
            _check_discrete(name, "dwt.compare_wavelets")
        if str(self.levels) != "auto":
            raise ConfigError("dwt.levels: only 'auto' (the deepest level whose causal filter "
                              "fits the rolling window) is supported")


@dataclass(frozen=True, slots=True)
class CausalFeaturesConfig:
    enabled: bool = True
    method: str = "causal_modwt"
    rolling_windows: tuple[int, ...] = (128, 256, 512, 1024)
    representative_window: int = 512
    min_level_coefficients: int = 16
    missing_bar_tolerance: float = 0.05
    fast_slow_boundary_seconds: float = 14_400.0
    band_edges_seconds: tuple[float, ...] = (3_600.0, 28_800.0)
    drift_lag_fraction: float = 0.25
    burst_z_threshold: float = 3.0
    burst_min_history: int = 32
    zero_energy_tolerance: float = 1e-30

    def __post_init__(self) -> None:
        object.__setattr__(self, "rolling_windows", _ints(self.rolling_windows,
                                                          "causal_features.rolling_windows",
                                                          minimum=32))
        object.__setattr__(self, "band_edges_seconds",
                           tuple(float(v) for v in self.band_edges_seconds))
        _one_of(self.method, ("causal_modwt",), "causal_features.method")
        if self.representative_window not in self.rolling_windows:
            raise ConfigError("causal_features.representative_window: must be one of "
                              "rolling_windows")
        if len(self.band_edges_seconds) != 2 or not (
                0 < self.band_edges_seconds[0] < self.band_edges_seconds[1]):
            raise ConfigError("causal_features.band_edges_seconds: two increasing edges "
                              "(high|mid, mid|low)")
        if not 0 <= self.missing_bar_tolerance < 1:
            raise ConfigError("causal_features.missing_bar_tolerance: must be in [0, 1)")
        if not 0 < self.drift_lag_fraction <= 1:
            raise ConfigError("causal_features.drift_lag_fraction: must be in (0, 1]")
        if self.min_level_coefficients < 4 or self.burst_min_history < 8:
            raise ConfigError("causal_features: min_level_coefficients >= 4, "
                              "burst_min_history >= 8")


@dataclass(frozen=True, slots=True)
class EnergyConfig:
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class EntropyConfig:
    enabled: bool = True
    normalise: bool = True


@dataclass(frozen=True, slots=True)
class BoundaryConfig:
    padding_modes: tuple[str, ...] = ("zero", "symmetric", "reflect", "constant",
                                      "periodization")
    trials: int = 200
    seed: int = 20260927

    def __post_init__(self) -> None:
        object.__setattr__(self, "padding_modes", tuple(self.padding_modes))
        for mode in self.padding_modes:
            _one_of(mode, PADDING_MODES, "boundary.padding_modes")
        if self.trials < 10:
            raise ConfigError("boundary.trials: must be >= 10")


@dataclass(frozen=True, slots=True)
class PersistenceConfig:
    lags_in_windows: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    lags_in_bars: tuple[int, ...] = (1,)
    band_tolerance: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "lags_in_windows", tuple(float(v) for v in self.lags_in_windows))
        object.__setattr__(self, "lags_in_bars", _ints(self.lags_in_bars,
                                                       "persistence.lags_in_bars"))
        if self.band_tolerance < 0:
            raise ConfigError("persistence.band_tolerance: must be >= 0")


@dataclass(frozen=True, slots=True)
class OutcomesConfig:
    horizons: tuple[int, ...] = (1, 5, 10, 20)
    residual_extremes: tuple[float, ...] = (1.0, 2.0, 3.0)
    ou_extreme: float = 2.0
    buckets: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(self, "horizons", _ints(self.horizons, "outcomes.horizons"))
        object.__setattr__(self, "residual_extremes",
                           tuple(float(v) for v in self.residual_extremes))
        if self.buckets < 2:
            raise ConfigError("outcomes.buckets: must be >= 2")


@dataclass(frozen=True, slots=True)
class ICConfig:
    features: tuple[str, ...] = IC_FEATURES
    targets: tuple[str, ...] = IC_TARGETS
    horizons: tuple[int, ...] = (1, 2, 3, 5, 10, 20)
    max_rows: int = 1_000_000
    max_rows_control: int = 300_000
    controls: tuple[str, ...] = ("random_walk", "shuffled_returns", "block_bootstrap")

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", tuple(self.features))
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "controls", tuple(self.controls))
        object.__setattr__(self, "horizons", _ints(self.horizons, "ic.horizons"))
        for name in self.features:
            _one_of(name, IC_FEATURE_NAMES, "ic.features")
        for name in self.targets:
            _one_of(name, IC_TARGETS, "ic.targets")
        if self.max_rows < 1000 or self.max_rows_control < 1000:
            raise ConfigError("ic: max_rows and max_rows_control must be >= 1000")


@dataclass(frozen=True, slots=True)
class IncrementalConfig:
    """Nested explanatory models A-D, fitted only to measure incremental information."""

    enabled: bool = True
    targets: tuple[str, ...] = IC_TARGETS
    binary_targets: tuple[str, ...] = BINARY_TARGETS
    horizons: tuple[int, ...] = (1, 5, 20)
    test_blocks: tuple[tuple[int, int], ...] = ((2011, 2014), (2015, 2018), (2019, 2022),
                                                (2023, 2026))
    max_rows: int = 400_000
    ridge: float = 1.0
    binary_extreme: float = 2.0
    min_train_rows: int = 2_000
    controls: tuple[str, ...] = ("random_walk", "block_bootstrap")

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "binary_targets", tuple(self.binary_targets))
        object.__setattr__(self, "controls", tuple(self.controls))
        object.__setattr__(self, "horizons", _ints(self.horizons, "incremental.horizons"))
        blocks = tuple((int(a), int(b)) for a, b in self.test_blocks)
        object.__setattr__(self, "test_blocks", blocks)
        for name in self.targets:
            _one_of(name, IC_TARGETS, "incremental.targets")
        for name in self.binary_targets:
            _one_of(name, BINARY_TARGETS, "incremental.binary_targets")
        if any(a > b for a, b in blocks) or any(
                blocks[i][1] >= blocks[i + 1][0] for i in range(len(blocks) - 1)):
            raise ConfigError("incremental.test_blocks: increasing, non-overlapping "
                              "[first_year, last_year] pairs")
        if self.ridge < 0:
            raise ConfigError("incremental.ridge: must be >= 0")


@dataclass(frozen=True, slots=True)
class ControlsConfig:
    white_noise: bool = True
    random_walk: bool = True
    shuffled_returns: bool = True
    block_bootstrap: bool = True
    block_size: int = 256
    block_sizes: tuple[int, ...] = (64, 256, 1024)
    seed: int = 20260925

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_sizes", _ints(self.block_sizes,
                                                      "null_controls.block_sizes", minimum=2))
        if self.block_size < 2:
            raise ConfigError("null_controls.block_size: must be >= 2")

    def enabled(self) -> list[str]:
        """Names of every null series to build, in report order.

        ``block_bootstrap`` uses ``block_size`` (the Prompt #5 draw); every
        other size in ``block_sizes`` is ``block_bootstrap_<k>``.
        """
        out = [name for name, on in (("white_noise", self.white_noise),
                                     ("random_walk", self.random_walk),
                                     ("shuffled_returns", self.shuffled_returns)) if on]
        if self.block_bootstrap:
            out.append("block_bootstrap")
            out += [f"block_bootstrap_{k}" for k in self.block_sizes if k != self.block_size]
        return out


@dataclass(frozen=True, slots=True)
class FeaturesOutputConfig:
    write: str = "representative"
    path: str = "data/features/wavelet"
    float32: bool = True

    def __post_init__(self) -> None:
        _one_of(self.write, ("representative", "all", "none"), "features_output.write")


@dataclass(frozen=True, slots=True)
class PlotsConfig:
    enabled: bool = True
    format: str = "png"
    dpi: int = 130
    figsize: tuple[float, float] = (11.0, 6.0)
    sample_bars: int = 3000
    max_series_points: int = 20_000

    def __post_init__(self) -> None:
        object.__setattr__(self, "figsize", tuple(self.figsize))


@dataclass(frozen=True, slots=True)
class OfflineConfig:
    """Offline (non-causal) scalogram slices: deterministic, never by outcome."""

    slices_per_timeframe: int = 5
    null_slices: int = 3

    def __post_init__(self) -> None:
        if self.slices_per_timeframe < 1 or self.null_slices < 1:
            raise ConfigError("offline: slice counts must be >= 1")


@dataclass(frozen=True, slots=True)
class OutputConfig:
    write_parquet: bool = True
    write_csv: bool = True


@dataclass(frozen=True, slots=True)
class GridConfig:
    max_bars_without_override: int = 3_000_000


@dataclass(frozen=True, slots=True)
class LedgerConfig:
    enabled: bool = True
    path: str = "results/research_ledger.parquet"


@dataclass(frozen=True, slots=True)
class WaveletConfig:
    """Fully-resolved wavelet research configuration."""

    project_root: Path
    config_path: Path | None
    schema_version: str
    results_path: Path
    features_path: Path
    ledger_path: Path
    timeframes: tuple[str, ...]
    regression_window: int
    ou_window: int
    input_series: tuple[str, ...]
    analysis: AnalysisConfig
    cwt: CWTConfig
    dwt: DWTConfig
    causal_features: CausalFeaturesConfig
    energy: EnergyConfig
    entropy: EntropyConfig
    boundary: BoundaryConfig
    persistence: PersistenceConfig
    outcomes: OutcomesConfig
    ic: ICConfig
    incremental: IncrementalConfig
    null_controls: ControlsConfig
    conditioning: ConditioningConfig
    stability: StabilityConfig
    features_output: FeaturesOutputConfig
    plots: PlotsConfig
    offline: OfflineConfig
    output: OutputConfig
    grid: GridConfig
    ledger: LedgerConfig

    def __post_init__(self) -> None:
        for name in self.input_series:
            _one_of(name, INPUT_SERIES, "input_series")
        unknown = [c for c in (*self.ic.controls, *self.incremental.controls)
                   if c not in self.null_controls.enabled() or c == "white_noise"]
        if unknown:
            raise ConfigError(f"ic/incremental controls {unknown} must be enabled pipeline "
                              "null controls (white noise has no pipeline)")

    @property
    def controls(self) -> ControlsConfig:
        """The null-control settings (the name the shared source builders read)."""
        return self.null_controls

    @property
    def windows(self) -> tuple[int, ...]:
        return self.causal_features.rolling_windows

    @property
    def representative_window(self) -> int:
        return self.causal_features.representative_window

    def results_dir(self, timeframe: str, series: str, window: int) -> Path:
        return self.results_path / timeframe / series / f"window_{window}"

    def plots_dir(self, timeframe: str, series: str, window: int) -> Path:
        return self.results_dir(timeframe, series, window) / "plots"

    def to_dict(self) -> dict[str, Any]:
        return _as_dict(self)

    def fingerprint(self) -> str:
        """Stable digest of the settings that affect output."""
        payload = self.to_dict()
        for key in ("project_root", "config_path", "results_path", "features_path",
                    "ledger_path", "plots", "output", "grid", "ledger"):
            payload.pop(key, None)
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.blake2b(blob, digest_size=8).hexdigest()

    def engine_fingerprint(self) -> str:
        """Digest of only what changes per-bar causal feature values."""
        payload = {
            "schema_version": self.schema_version,
            "regression_window": self.regression_window,
            "ou_window": self.ou_window,
            "dwt": _as_dict(self.dwt),
            "causal_features": _as_dict(self.causal_features),
            "entropy": _as_dict(self.entropy),
            "float32": self.features_output.float32,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.blake2b(blob, digest_size=8).hexdigest()


_SECTIONS: dict[str, type] = {
    "analysis": AnalysisConfig,
    "cwt": CWTConfig,
    "dwt": DWTConfig,
    "causal_features": CausalFeaturesConfig,
    "energy": EnergyConfig,
    "entropy": EntropyConfig,
    "boundary": BoundaryConfig,
    "persistence": PersistenceConfig,
    "outcomes": OutcomesConfig,
    "ic": ICConfig,
    "incremental": IncrementalConfig,
    "null_controls": ControlsConfig,
    "conditioning": ConditioningConfig,
    "stability": StabilityConfig,
    "features_output": FeaturesOutputConfig,
    "plots": PlotsConfig,
    "offline": OfflineConfig,
    "output": OutputConfig,
    "grid": GridConfig,
    "ledger": LedgerConfig,
}
_TOP_LEVEL = ("schema_version", "timeframes", "regression_window", "ou_window", "input_series",
              "results_path")


def default_wavelet_config_path(root: Path | None = None) -> Path:
    root = root or find_project_root()
    return resolve_path(os.environ.get(WAVELET_CONFIG_ENV_VAR) or DEFAULT_WAVELET_RELPATH, root)


def load_wavelet_config(
    path: str | os.PathLike[str] | None = None,
    *,
    root: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> WaveletConfig:
    """Load, expand, validate and resolve ``config/wavelet.yaml``.

    *overrides* replace whole top-level keys or sections (a section given
    here is built from exactly what is passed, over the dataclass defaults).
    """
    root = root or find_project_root()
    cfg_path = (resolve_path(path, root) if path is not None
                else default_wavelet_config_path(root))
    if not cfg_path.exists():
        raise ConfigError(f"Wavelet configuration file not found: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{cfg_path}: top level must be a mapping")
    data: dict[str, Any] = _expand(raw)
    for key, value in (overrides or {}).items():
        if value is not None:
            data[key] = value

    cwt_raw = dict(data.pop("cwt", {}) or {})
    cwt_raw["scales"] = _build(CWTScalesConfig, cwt_raw.get("scales", {}), "cwt.scales")
    sections: dict[str, Any] = {"cwt": _build(CWTConfig, cwt_raw, "cwt")}
    sections.update({name: _build(cls, data.pop(name, {}), name)
                     for name, cls in _SECTIONS.items() if name != "cwt"})
    top = {key: data.pop(key) for key in list(data) if key in _TOP_LEVEL}
    if data:
        known = sorted({*_SECTIONS, *_TOP_LEVEL})
        raise ConfigError(f"{cfg_path}: unknown top-level option(s) {sorted(data)}. Known: {known}")
    features_output: FeaturesOutputConfig = sections["features_output"]
    ledger: LedgerConfig = sections["ledger"]
    return WaveletConfig(
        project_root=root,
        config_path=cfg_path,
        schema_version=str(top.get("schema_version", "1.0.0")),
        results_path=resolve_path(top.get("results_path", "results/wavelet_research"), root),
        features_path=resolve_path(features_output.path, root),
        ledger_path=resolve_path(ledger.path, root),
        timeframes=tuple(top.get("timeframes", ("1m", "5m", "15m", "30m", "1h"))),
        regression_window=int(top.get("regression_window", 128)),
        ou_window=int(top.get("ou_window", 256)),
        input_series=tuple(top.get("input_series",
                                   ("regression_residual", "ou_innovation", "log_return"))),
        **sections,
    )
