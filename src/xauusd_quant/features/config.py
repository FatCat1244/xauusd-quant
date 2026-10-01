"""Typed configuration for the rolling-regression layer.

Same contract as :mod:`xauusd_quant.utils.config` and
:mod:`xauusd_quant.research.config`: frozen dataclasses, unknown keys are an
error, and the whole thing hashes to a fingerprint recorded in every report.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..utils.config import ConfigError, _as_dict, _build, _expand, _one_of
from ..utils.paths import find_project_root, resolve_path

__all__ = [
    "AutocorrelationConfig",
    "BootstrapConfig",
    "ConditioningConfig",
    "DecayConfig",
    "DistributionConfig",
    "ExtremesConfig",
    "NumericalConfig",
    "OutputConfig",
    "PlotsConfig",
    "RegressionConfig",
    "RegressionModelConfig",
    "RobustConfig",
    "StationarityConfig",
    "ZScoreConfig",
    "default_regression_config_path",
    "load_regression_config",
]

DEFAULT_REGRESSION_RELPATH = "config/regression.yaml"
REGRESSION_CONFIG_ENV_VAR = "XAUUSD_REGRESSION_CONFIG"

#: Sentinel meaning "reuse the regression window for this purpose".
SAME_AS_REGRESSION = "same_as_regression"


@dataclass(frozen=True, slots=True)
class RegressionModelConfig:
    """The rolling OLS model itself."""

    windows: tuple[int, ...] = (32, 64, 128, 256, 512)
    price_transform: str = "log"
    price_source: str = "mid"
    price_column: str = "close"
    include_current_bar: bool = True
    min_periods_ratio: float = 1.0
    flag_session_spanning_windows: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "windows", tuple(int(w) for w in self.windows))
        _one_of(self.price_transform, ("log", "price"), "regression.price_transform")
        _one_of(self.price_source, ("mid", "bid", "ask"), "regression.price_source")
        if any(w < 3 for w in self.windows):
            raise ConfigError(
                "regression.windows: a window must hold at least 3 bars; OLS with "
                "two points is exact and leaves no residual."
            )
        if not 0 < self.min_periods_ratio <= 1:
            raise ConfigError("regression.min_periods_ratio: must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class ZScoreConfig:
    """How residuals are normalised - see the YAML for the two methods."""

    rolling_windows: tuple[str | int, ...] = (SAME_AS_REGRESSION, 64, 128, 256)
    primary_rolling_window: str | int = SAME_AS_REGRESSION
    min_periods_ratio: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rolling_windows",
            tuple(w if w == SAME_AS_REGRESSION else int(w) for w in self.rolling_windows),
        )
        if self.primary_rolling_window not in self.rolling_windows:
            raise ConfigError(
                f"zscore.primary_rolling_window={self.primary_rolling_window!r} is not "
                f"in zscore.rolling_windows={list(self.rolling_windows)}"
            )

    def resolve(self, regression_window: int) -> list[int]:
        """Concrete rolling windows for a given regression window."""
        return [
            regression_window if w == SAME_AS_REGRESSION else int(w)
            for w in self.rolling_windows
        ]

    def resolve_primary(self, regression_window: int) -> int:
        if self.primary_rolling_window == SAME_AS_REGRESSION:
            return regression_window
        return int(self.primary_rolling_window)


@dataclass(frozen=True, slots=True)
class NumericalConfig:
    epsilon: float = 1e-12
    min_residual_std: float = 1e-10
    min_total_variance: float = 1e-16
    chunk_elements: int = 4_000_000

    def __post_init__(self) -> None:
        if self.chunk_elements < 10_000:
            raise ConfigError("numerical.chunk_elements: must be >= 10000")


@dataclass(frozen=True, slots=True)
class DistributionConfig:
    quantiles: tuple[float, ...] = (
        0.001, 0.005, 0.01, 0.025, 0.05, 0.25, 0.50, 0.75, 0.95, 0.975, 0.99, 0.995, 0.999
    )
    gaussian_reference: bool = True
    tail_sigma_levels: tuple[float, ...] = (2, 3, 4, 5)

    def __post_init__(self) -> None:
        object.__setattr__(self, "quantiles", tuple(self.quantiles))
        object.__setattr__(self, "tail_sigma_levels", tuple(self.tail_sigma_levels))


@dataclass(frozen=True, slots=True)
class AutocorrelationConfig:
    max_lag: int = 100
    series: tuple[str, ...] = ("residual", "residual_diff", "abs_residual")
    confidence_level: float = 0.95
    ljung_box_lags: tuple[int, ...] = (5, 10, 20, 50)

    def __post_init__(self) -> None:
        object.__setattr__(self, "series", tuple(self.series))
        object.__setattr__(self, "ljung_box_lags", tuple(self.ljung_box_lags))
        for name in self.series:
            _one_of(
                name, ("residual", "residual_diff", "abs_residual"),
                "autocorrelation.series",
            )


@dataclass(frozen=True, slots=True)
class StationarityConfig:
    adf_enabled: bool = True
    kpss_enabled: bool = True
    adf_regression: str = "c"
    adf_autolag: str | None = "AIC"
    kpss_regression: str = "c"
    kpss_nlags: str | int = "auto"
    max_observations: int = 120_000
    segmented_by: str = "year"
    segment_min_observations: int = 2000

    def __post_init__(self) -> None:
        _one_of(self.adf_regression, ("c", "ct", "ctt", "n"), "stationarity.adf_regression")
        _one_of(self.kpss_regression, ("c", "ct"), "stationarity.kpss_regression")
        _one_of(self.segmented_by, ("year", "quarter"), "stationarity.segmented_by")


@dataclass(frozen=True, slots=True)
class DecayConfig:
    newey_west_lags: str | int = "auto"
    min_observations: int = 500


@dataclass(frozen=True, slots=True)
class BootstrapConfig:
    enabled: bool = True
    iterations: int = 1000
    confidence_level: float = 0.95
    seed: int = 20260924
    max_samples: int = 200_000

    def __post_init__(self) -> None:
        if not 0 < self.confidence_level < 1:
            raise ConfigError("bootstrap.confidence_level: must be in (0, 1)")


@dataclass(frozen=True, slots=True)
class ExtremesConfig:
    zscore_bins: tuple[float, ...] = (-1000, -3, -2, -1, 0, 1, 2, 3, 1000)
    forward_horizons: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 50)
    max_crossing_horizon: int = 200
    crossing_checkpoints: tuple[int, ...] = (5, 10, 20, 50, 100, 200)
    mare_horizon: int = 50
    persistence_levels: tuple[float, ...] = (2.5, 3.0, 3.5)
    persistence_revert_to: float = 1.0
    min_samples_warning: int = 200
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "zscore_bins", tuple(float(b) for b in self.zscore_bins))
        object.__setattr__(
            self, "forward_horizons", tuple(int(h) for h in self.forward_horizons)
        )
        object.__setattr__(
            self, "crossing_checkpoints", tuple(int(c) for c in self.crossing_checkpoints)
        )
        object.__setattr__(
            self, "persistence_levels", tuple(float(p) for p in self.persistence_levels)
        )
        if list(self.zscore_bins) != sorted(self.zscore_bins):
            raise ConfigError("extremes.zscore_bins: must be in ascending order")
        if any(h < 1 for h in self.forward_horizons):
            raise ConfigError("extremes.forward_horizons: every horizon must be >= 1")
        if max(self.crossing_checkpoints) > self.max_crossing_horizon:
            raise ConfigError(
                "extremes.crossing_checkpoints: a checkpoint exceeds "
                f"max_crossing_horizon ({self.max_crossing_horizon})"
            )

    def bin_labels(self) -> list[str]:
        """Readable labels for the Z bins, e.g. ``-2<=Z<-1``."""
        out: list[str] = []
        edges = list(self.zscore_bins)
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            if lo <= -999:
                out.append(f"Z<{hi:g}")
            elif hi >= 999:
                out.append(f"Z>={lo:g}")
            else:
                out.append(f"{lo:g}<=Z<{hi:g}")
        return out


@dataclass(frozen=True, slots=True)
class ConditioningConfig:
    extreme_abs_z: float = 2.0
    quantile_buckets: int = 4
    volatility_window: int = 50
    by_year: bool = True
    by_hour: bool = True
    time_basis: str = "timestamp"

    def __post_init__(self) -> None:
        _one_of(self.time_basis, ("timestamp", "timestamp_utc"), "conditioning.time_basis")
        if self.quantile_buckets < 2:
            raise ConfigError("conditioning.quantile_buckets: must be >= 2")


@dataclass(frozen=True, slots=True)
class RobustConfig:
    enabled: bool = False
    method: str = "theil_sen"
    max_windows: int = 2000
    seed: int = 20260924

    def __post_init__(self) -> None:
        _one_of(self.method, ("theil_sen", "huber"), "robust.method")


@dataclass(frozen=True, slots=True)
class PlotsConfig:
    enabled: bool = True
    format: str = "png"
    dpi: int = 130
    figsize: tuple[float, float] = (11.0, 6.0)
    sample_bars: int = 1500
    max_scatter_points: int = 40_000

    def __post_init__(self) -> None:
        object.__setattr__(self, "figsize", tuple(self.figsize))


@dataclass(frozen=True, slots=True)
class OutputConfig:
    write_parquet: bool = True
    write_csv: bool = True
    write_json: bool = True
    cache_features: bool = True


@dataclass(frozen=True, slots=True)
class RegressionConfig:
    """Fully-resolved rolling-regression configuration."""

    project_root: Path
    config_path: Path | None
    schema_version: str
    features_path: Path
    results_path: Path
    timeframes: tuple[str, ...]

    regression: RegressionModelConfig
    zscore: ZScoreConfig
    numerical: NumericalConfig
    distribution: DistributionConfig
    autocorrelation: AutocorrelationConfig
    stationarity: StationarityConfig
    decay: DecayConfig
    extremes: ExtremesConfig
    conditioning: ConditioningConfig
    robust: RobustConfig
    plots: PlotsConfig
    output: OutputConfig

    def feature_dir(self, timeframe: str, window: int) -> Path:
        return self.features_path / f"timeframe={timeframe}" / f"window={window}"

    def results_dir(self, timeframe: str, window: int) -> Path:
        return self.results_path / timeframe / f"window_{window}"

    def plots_dir(self, timeframe: str, window: int) -> Path:
        return self.results_dir(timeframe, window) / "plots"

    def to_dict(self) -> dict[str, Any]:
        return _as_dict(self)

    def fingerprint(self) -> str:
        """Stable digest of the settings that affect output."""
        payload = self.to_dict()
        for key in ("project_root", "config_path", "features_path", "results_path"):
            payload.pop(key, None)
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.blake2b(blob, digest_size=8).hexdigest()


_SECTIONS: dict[str, type] = {
    "zscore": ZScoreConfig,
    "numerical": NumericalConfig,
    "distribution": DistributionConfig,
    "autocorrelation": AutocorrelationConfig,
    "stationarity": StationarityConfig,
    "decay": DecayConfig,
    "conditioning": ConditioningConfig,
    "robust": RobustConfig,
    "plots": PlotsConfig,
    "output": OutputConfig,
}


def default_regression_config_path(root: Path | None = None) -> Path:
    root = root or find_project_root()
    return resolve_path(
        os.environ.get(REGRESSION_CONFIG_ENV_VAR) or DEFAULT_REGRESSION_RELPATH, root
    )


def load_regression_config(
    path: str | os.PathLike[str] | None = None,
    *,
    root: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> RegressionConfig:
    """Load, expand, validate and resolve ``config/regression.yaml``."""
    root = root or find_project_root()
    cfg_path = (
        resolve_path(path, root) if path is not None else default_regression_config_path(root)
    )
    if not cfg_path.exists():
        raise ConfigError(f"Regression configuration file not found: {cfg_path}")

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

    regression = _build(
        RegressionModelConfig, data.pop("regression", {}), "regression"
    )
    ext_raw = dict(data.pop("extremes", {}) or {})
    bootstrap = _build(
        BootstrapConfig, ext_raw.pop("bootstrap", {}), "extremes.bootstrap"
    )
    extremes = _build(ExtremesConfig, {**ext_raw, "bootstrap": bootstrap}, "extremes")

    features_path = resolve_path(data.pop("features_path", "data/features/regression"), root)
    results_path = resolve_path(data.pop("results_path", "results/regression_research"), root)
    schema_version = str(data.pop("schema_version", "1.0.0"))
    timeframes = tuple(data.pop("timeframes", ("1m", "5m", "15m", "30m", "1h")))

    if data:
        known = sorted({
            *_SECTIONS, "regression", "extremes", "features_path", "results_path",
            "schema_version", "timeframes",
        })
        raise ConfigError(
            f"{cfg_path}: unknown top-level option(s) {sorted(data)}. Known: {known}"
        )

    return RegressionConfig(
        project_root=root,
        config_path=cfg_path,
        schema_version=schema_version,
        features_path=features_path,
        results_path=results_path,
        timeframes=timeframes,
        regression=regression,
        extremes=extremes,
        **sections,
    )
