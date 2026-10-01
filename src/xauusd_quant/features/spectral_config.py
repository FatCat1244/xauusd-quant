"""Typed configuration for the spectral (Fourier) research layer.

Same contract as :mod:`xauusd_quant.models.config`: frozen dataclasses,
unknown keys are an error, and the whole thing hashes to a fingerprint that is
recorded in every report and in the version of every cached feature set.

``config/spectral.yaml`` governs only the spectral study. The inputs keep
their own definitions: the residual comes from ``config/regression.yaml``, the
OU innovation from ``config/ou.yaml``, sessions from ``config/research.yaml``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..utils.config import ConfigError, _as_dict, _build, _expand, _one_of
from ..utils.paths import find_project_root, resolve_path

__all__ = [
    "INPUT_SERIES",
    "WINDOW_FUNCTIONS",
    "SpectralConfig",
    "default_spectral_config_path",
    "load_spectral_config",
]

DEFAULT_SPECTRAL_RELPATH = "config/spectral.yaml"
SPECTRAL_CONFIG_ENV_VAR = "XAUUSD_SPECTRAL_CONFIG"

#: Series the study can analyse. Raw price is deliberately absent: its FFT is
#: dominated by non-stationarity and says nothing about cycles.
INPUT_SERIES: tuple[str, ...] = (
    "regression_residual", "ou_innovation", "log_return", "abs_ou_innovation",
)
WINDOW_FUNCTIONS: tuple[str, ...] = ("hann", "rectangular", "hamming", "blackman")
IC_TARGETS: tuple[str, ...] = (
    "future_return", "future_residual_change", "future_abs_residual_reduction",
)


def _ints(values: Any, where: str, *, minimum: int = 1) -> tuple[int, ...]:
    out = tuple(int(v) for v in values)
    if any(v < minimum for v in out):
        raise ConfigError(f"{where}: every value must be >= {minimum}")
    return out


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    window_function: str = "hann"
    remove_mean: bool = True
    detrend_input: bool = False
    missing_bar_tolerance: float = 0.05

    def __post_init__(self) -> None:
        _one_of(self.window_function, WINDOW_FUNCTIONS, "preprocessing.window_function")
        if not 0 <= self.missing_bar_tolerance < 1:
            raise ConfigError("preprocessing.missing_bar_tolerance: must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class SpectrumConfig:
    minimum_frequency_index: int = 1
    exclude_nyquist_if_needed: bool = True
    top_frequencies: int = 5
    phase_min_power_share: float = 0.02
    zero_power_tolerance: float = 1.0e-30

    def __post_init__(self) -> None:
        if self.minimum_frequency_index < 1:
            raise ConfigError("spectrum.minimum_frequency_index: must be >= 1 (DC is excluded)")
        if self.top_frequencies < 1:
            raise ConfigError("spectrum.top_frequencies: must be >= 1")
        if not 0 <= self.phase_min_power_share < 1:
            raise ConfigError("spectrum.phase_min_power_share: must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class PowerBandsConfig:
    enabled: bool = True
    edges: tuple[float, ...] = (0.0, 0.1, 0.4, 1.0)
    names: tuple[str, ...] = ("low", "mid", "high")

    def __post_init__(self) -> None:
        object.__setattr__(self, "edges", tuple(float(e) for e in self.edges))
        object.__setattr__(self, "names", tuple(str(n) for n in self.names))
        if len(self.edges) < 2 or self.edges[0] != 0.0 or self.edges[-1] != 1.0:
            raise ConfigError("power_bands.edges: must start at 0.0 and end at 1.0")
        if any(b <= a for a, b in zip(self.edges, self.edges[1:], strict=False)):
            raise ConfigError("power_bands.edges: must be strictly increasing")
        if len(self.names) != len(self.edges) - 1:
            raise ConfigError("power_bands.names: need one name per band (len(edges) - 1)")


@dataclass(frozen=True, slots=True)
class EntropyConfig:
    enabled: bool = True
    normalise: bool = True


@dataclass(frozen=True, slots=True)
class RollingFFTConfig:
    enabled: bool = True
    chunk_elements: int = 4_000_000
    workers: int = -1

    def __post_init__(self) -> None:
        if self.chunk_elements < 1024:
            raise ConfigError("rolling_fft.chunk_elements: must be >= 1024")


@dataclass(frozen=True, slots=True)
class PersistenceConfig:
    top_k: int = 5
    relative_tolerance: float = 0.10
    lags_in_windows: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    lags_in_bars: tuple[int, ...] = (1,)

    def __post_init__(self) -> None:
        object.__setattr__(self, "lags_in_windows", tuple(float(v) for v in self.lags_in_windows))
        object.__setattr__(self, "lags_in_bars", _ints(self.lags_in_bars, "persistence.lags_in_bars"))
        if self.top_k < 1:
            raise ConfigError("persistence.top_k: must be >= 1")
        if not 0 <= self.relative_tolerance < 1:
            raise ConfigError("persistence.relative_tolerance: must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class ReconstructionConfig:
    components: tuple[int, ...] = (1, 3, 5, 10)
    horizons: tuple[int, ...] = (1, 2, 3, 5, 10)
    max_windows: int = 100_000

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", _ints(self.components, "reconstruction.components"))
        object.__setattr__(self, "horizons", _ints(self.horizons, "reconstruction.horizons"))


@dataclass(frozen=True, slots=True)
class PhaseConfig:
    buckets: int = 8
    horizons: tuple[int, ...] = (1, 2, 5, 10)

    def __post_init__(self) -> None:
        object.__setattr__(self, "horizons", _ints(self.horizons, "phase.horizons"))
        if self.buckets < 2:
            raise ConfigError("phase.buckets: must be >= 2")


@dataclass(frozen=True, slots=True)
class OutcomesConfig:
    horizons: tuple[int, ...] = (1, 5, 10, 20)
    extreme_abs_z: float = 2.0
    buckets: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(self, "horizons", _ints(self.horizons, "outcomes.horizons"))
        if self.extreme_abs_z <= 0 or self.buckets < 2:
            raise ConfigError("outcomes: extreme_abs_z must be > 0 and buckets >= 2")


@dataclass(frozen=True, slots=True)
class ICConfig:
    features: tuple[str, ...] = ()
    targets: tuple[str, ...] = IC_TARGETS
    horizons: tuple[int, ...] = (1, 2, 3, 5, 10, 20)
    max_rows: int = 1_000_000
    max_rows_control: int = 300_000

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", tuple(str(f) for f in self.features))
        object.__setattr__(self, "targets", tuple(str(t) for t in self.targets))
        object.__setattr__(self, "horizons", _ints(self.horizons, "ic.horizons"))
        for target in self.targets:
            _one_of(target, IC_TARGETS, "ic.targets")


@dataclass(frozen=True, slots=True)
class ConditioningConfig:
    volatility_buckets: int = 4
    trend_buckets: int = 5
    ou_speed_buckets: int = 3
    time_basis: str = "timestamp"
    by_hour: bool = True
    by_session: bool = True
    min_samples_warning: int = 200

    def __post_init__(self) -> None:
        _one_of(self.time_basis, ("timestamp", "timestamp_utc"), "conditioning.time_basis")
        if self.trend_buckets != 5:
            raise ConfigError("conditioning.trend_buckets: the trend labels need exactly 5")
        if self.volatility_buckets < 2 or self.ou_speed_buckets < 2:
            raise ConfigError("conditioning: bucket counts must be >= 2")


@dataclass(frozen=True, slots=True)
class StabilityConfig:
    eras: int = 3
    change_quantiles: tuple[float, ...] = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)

    def __post_init__(self) -> None:
        object.__setattr__(self, "change_quantiles", tuple(float(q) for q in self.change_quantiles))
        if self.eras < 2:
            raise ConfigError("stability.eras: must be >= 2")


@dataclass(frozen=True, slots=True)
class ControlsConfig:
    white_noise: bool = True
    random_walk: bool = True
    shuffled_returns: bool = True
    block_bootstrap: bool = True
    block_size: int = 256
    seed: int = 20260925

    def __post_init__(self) -> None:
        if self.block_size < 2:
            raise ConfigError("controls.block_size: must be >= 2")

    def enabled(self) -> list[str]:
        """Names of the null series to build, in report order."""
        out = []
        if self.white_noise:
            out.append("white_noise")
        if self.random_walk:
            out.append("random_walk")
        if self.shuffled_returns:
            out.append("shuffled_returns")
        if self.block_bootstrap:
            out.append("block_bootstrap")
        return out


@dataclass(frozen=True, slots=True)
class FeaturesOutputConfig:
    write: str = "representative"
    path: str = "data/features/spectral"
    top_components: int = 3
    float32: bool = True

    def __post_init__(self) -> None:
        _one_of(self.write, ("representative", "all", "none"), "features_output.write")
        if self.top_components < 1:
            raise ConfigError("features_output.top_components: must be >= 1")


@dataclass(frozen=True, slots=True)
class PlotsConfig:
    enabled: bool = True
    format: str = "png"
    dpi: int = 130
    figsize: tuple[float, float] = (11.0, 6.0)
    sample_bars: int = 3000
    spectrogram_bars: int = 1500
    max_series_points: int = 20_000

    def __post_init__(self) -> None:
        object.__setattr__(self, "figsize", tuple(self.figsize))


@dataclass(frozen=True, slots=True)
class OutputConfig:
    write_parquet: bool = True
    write_csv: bool = True
    write_json: bool = True


@dataclass(frozen=True, slots=True)
class GridConfig:
    max_bars_without_override: int = 3_000_000


@dataclass(frozen=True, slots=True)
class LedgerConfig:
    enabled: bool = True
    path: str = "results/research_ledger.parquet"


@dataclass(frozen=True, slots=True)
class SpectralConfig:
    """Fully-resolved spectral research configuration."""

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
    fft_windows: tuple[int, ...]
    representative_fft_window: int
    preprocessing: PreprocessingConfig
    spectrum: SpectrumConfig
    power_bands: PowerBandsConfig
    spectral_entropy: EntropyConfig
    rolling_fft: RollingFFTConfig
    persistence: PersistenceConfig
    reconstruction: ReconstructionConfig
    phase: PhaseConfig
    outcomes: OutcomesConfig
    ic: ICConfig
    conditioning: ConditioningConfig
    stability: StabilityConfig
    controls: ControlsConfig
    features_output: FeaturesOutputConfig
    plots: PlotsConfig
    output: OutputConfig
    grid: GridConfig
    ledger: LedgerConfig

    def __post_init__(self) -> None:
        for name in self.input_series:
            _one_of(name, INPUT_SERIES, "input_series")
        if any(n < 8 for n in self.fft_windows):
            raise ConfigError("fft_windows: every window must be >= 8 bars")
        if self.representative_fft_window not in self.fft_windows:
            raise ConfigError("representative_fft_window: must be one of fft_windows")
        if self.persistence.top_k > self.spectrum.top_frequencies:
            raise ConfigError("persistence.top_k: cannot exceed spectrum.top_frequencies")
        if self.features_output.top_components > self.spectrum.top_frequencies:
            raise ConfigError("features_output.top_components: cannot exceed top_frequencies")

    def results_dir(self, timeframe: str, series: str, fft_window: int) -> Path:
        return self.results_path / timeframe / series / f"fft_{fft_window}"

    def plots_dir(self, timeframe: str, series: str, fft_window: int) -> Path:
        return self.results_dir(timeframe, series, fft_window) / "plots"

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
        """Digest of only what changes per-bar feature values (for the feature cache)."""
        payload = {
            "schema_version": self.schema_version,
            "regression_window": self.regression_window,
            "ou_window": self.ou_window,
            "preprocessing": _as_dict(self.preprocessing),
            "spectrum": _as_dict(self.spectrum),
            "power_bands": _as_dict(self.power_bands),
            "spectral_entropy": _as_dict(self.spectral_entropy),
            "top_components": self.features_output.top_components,
            "float32": self.features_output.float32,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.blake2b(blob, digest_size=8).hexdigest()


_SECTIONS: dict[str, type] = {
    "preprocessing": PreprocessingConfig,
    "spectrum": SpectrumConfig,
    "power_bands": PowerBandsConfig,
    "spectral_entropy": EntropyConfig,
    "rolling_fft": RollingFFTConfig,
    "persistence": PersistenceConfig,
    "reconstruction": ReconstructionConfig,
    "phase": PhaseConfig,
    "outcomes": OutcomesConfig,
    "ic": ICConfig,
    "conditioning": ConditioningConfig,
    "stability": StabilityConfig,
    "controls": ControlsConfig,
    "features_output": FeaturesOutputConfig,
    "plots": PlotsConfig,
    "output": OutputConfig,
    "grid": GridConfig,
    "ledger": LedgerConfig,
}
_TOP_LEVEL = ("schema_version", "timeframes", "regression_window", "ou_window", "input_series",
              "fft_windows", "representative_fft_window", "results_path")


def default_spectral_config_path(root: Path | None = None) -> Path:
    root = root or find_project_root()
    return resolve_path(os.environ.get(SPECTRAL_CONFIG_ENV_VAR) or DEFAULT_SPECTRAL_RELPATH, root)


def load_spectral_config(
    path: str | os.PathLike[str] | None = None,
    *,
    root: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> SpectralConfig:
    """Load, expand, validate and resolve ``config/spectral.yaml``."""
    root = root or find_project_root()
    cfg_path = (resolve_path(path, root) if path is not None
                else default_spectral_config_path(root))
    if not cfg_path.exists():
        raise ConfigError(f"Spectral configuration file not found: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{cfg_path}: top level must be a mapping")
    data: dict[str, Any] = _expand(raw)
    for key, value in (overrides or {}).items():
        if value is not None:
            data[key] = value

    sections: dict[str, Any] = {
        name: _build(cls, data.pop(name, {}), name) for name, cls in _SECTIONS.items()
    }
    top = {key: data.pop(key) for key in list(data) if key in _TOP_LEVEL}
    if data:
        known = sorted({*_SECTIONS, *_TOP_LEVEL})
        raise ConfigError(f"{cfg_path}: unknown top-level option(s) {sorted(data)}. Known: {known}")
    features_output: FeaturesOutputConfig = sections["features_output"]
    ledger: LedgerConfig = sections["ledger"]
    return SpectralConfig(
        project_root=root,
        config_path=cfg_path,
        schema_version=str(top.get("schema_version", "1.0.0")),
        results_path=resolve_path(top.get("results_path", "results/spectral_research"), root),
        features_path=resolve_path(features_output.path, root),
        ledger_path=resolve_path(ledger.path, root),
        timeframes=tuple(top.get("timeframes", ("1m", "5m", "15m", "30m", "1h"))),
        regression_window=int(top.get("regression_window", 128)),
        ou_window=int(top.get("ou_window", 256)),
        input_series=tuple(top.get("input_series",
                                   ("regression_residual", "ou_innovation", "log_return"))),
        fft_windows=_ints(top.get("fft_windows", (64, 128, 256, 512, 1024)), "fft_windows",
                          minimum=8),
        representative_fft_window=int(top.get("representative_fft_window", 256)),
        **sections,
    )
