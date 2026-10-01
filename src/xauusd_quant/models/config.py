"""Typed configuration for the Ornstein-Uhlenbeck layer.

Same contract as :mod:`xauusd_quant.features.config`: frozen dataclasses,
unknown keys are an error, and the whole thing hashes to a fingerprint that is
recorded in every report.

``config/ou.yaml`` only governs the OU modelling of the residual. The residual
itself is still defined by ``config/regression.yaml``, and the session windows
used by the intraday study still come from ``config/research.yaml``.
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
from .ornstein_uhlenbeck import OUValidityRules

__all__ = [
    "ConditioningConfig",
    "ControlsConfig",
    "DiagnosticsConfig",
    "EquilibriumConfig",
    "ExpectedPathConfig",
    "ExtremesConfig",
    "HalfLifeConfig",
    "NearUnitRootConfig",
    "NumericalConfig",
    "OUConfig",
    "OutputConfig",
    "PlotsConfig",
    "RepresentativeConfig",
    "RollingEstimationConfig",
    "SegmentedEstimationConfig",
    "StabilityConfig",
    "ValidBRange",
    "default_ou_config_path",
    "load_ou_config",
]

DEFAULT_OU_RELPATH = "config/ou.yaml"
OU_CONFIG_ENV_VAR = "XAUUSD_OU_CONFIG"

#: Estimators that exist. Anything else in `estimation_methods` is an error.
KNOWN_METHODS: tuple[str, ...] = ("ar1_ols",)


@dataclass(frozen=True, slots=True)
class ValidBRange:
    """Where an AR(1) coefficient is given an OU reading. May only narrow (0, 1)."""

    min: float = 0.0
    max: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.min < self.max <= 1.0:
            raise ConfigError(
                "valid_b_range: need 0 <= min < max <= 1. The OU mapping "
                "theta = -ln(b) only exists on (0, 1); this range may narrow that "
                f"interval but never widen it (got min={self.min}, max={self.max})."
            )


@dataclass(frozen=True, slots=True)
class RepresentativeConfig:
    """The (N, M) pair shown in single-row summaries. A default, not a choice."""

    regression_window: int = 128
    ou_window: int = 256


@dataclass(frozen=True, slots=True)
class HalfLifeConfig:
    max_reportable_bars: float = 10_000.0
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95)

    def __post_init__(self) -> None:
        object.__setattr__(self, "quantiles", tuple(float(q) for q in self.quantiles))
        if self.max_reportable_bars <= 1:
            raise ConfigError("half_life.max_reportable_bars: must be > 1")
        if any(not 0 < q < 1 for q in self.quantiles):
            raise ConfigError("half_life.quantiles: every quantile must be in (0, 1)")


@dataclass(frozen=True, slots=True)
class NearUnitRootConfig:
    thresholds: tuple[float, ...] = (0.95, 0.99)
    flag_threshold: float = 0.99

    def __post_init__(self) -> None:
        object.__setattr__(self, "thresholds", tuple(float(t) for t in self.thresholds))
        if any(not 0 < t < 1 for t in (*self.thresholds, self.flag_threshold)):
            raise ConfigError("near_unit_root: thresholds must be in (0, 1)")


@dataclass(frozen=True, slots=True)
class NumericalConfig:
    unit_root_tolerance: float = 1e-9
    min_regressor_variance: float = 1e-20
    zero_innovation_tolerance: float = 1e-12
    chunk_rows: int = 16_384

    def __post_init__(self) -> None:
        if not 0 <= self.unit_root_tolerance < 1e-3:
            raise ConfigError("numerical.unit_root_tolerance: must be in [0, 1e-3)")
        if self.min_regressor_variance < 0 or self.zero_innovation_tolerance < 0:
            raise ConfigError("numerical: variance floors cannot be negative")
        if self.chunk_rows < 1024:
            raise ConfigError("numerical.chunk_rows: must be >= 1024")


@dataclass(frozen=True, slots=True)
class RollingEstimationConfig:
    enabled: bool = True
    # Which (N, M) pairs get a per-bar parameters.parquet: "representative"
    # (one per timeframe), "all", or "none". On 23 years of 1-minute bars a
    # table is ~0.5 GB, so writing all 100 pairs would cost ~15 GB for data
    # that can be regenerated in seconds.
    write_parameters: str = "representative"

    def __post_init__(self) -> None:
        _one_of(self.write_parameters, ("representative", "all", "none"),
                "rolling_estimation.write_parameters")


@dataclass(frozen=True, slots=True)
class SegmentedEstimationConfig:
    yearly: bool = True
    quarterly: bool = True
    min_observations: int = 500


@dataclass(frozen=True, slots=True)
class EquilibriumConfig:
    confidence_level: float = 0.95
    newey_west_lags: str | int = "auto"
    material_fraction_of_std: float = 0.10

    def __post_init__(self) -> None:
        if not 0 < self.confidence_level < 1:
            raise ConfigError("equilibrium.confidence_level: must be in (0, 1)")
        if self.newey_west_lags != "auto" and (
            not isinstance(self.newey_west_lags, int) or self.newey_west_lags < 0
        ):
            raise ConfigError("equilibrium.newey_west_lags: 'auto' or an integer >= 0")

    def resolve_lags(self, n: int) -> int:
        """Newey-West lag count for a sample of ``n`` pairs."""
        if self.newey_west_lags == "auto":
            return max(1, int(4.0 * (max(n, 1) / 100.0) ** (2.0 / 9.0)))
        return int(self.newey_west_lags)


@dataclass(frozen=True, slots=True)
class ExpectedPathConfig:
    horizons: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 50)
    decay_bars: tuple[int, ...] = (1, 5, 10)
    decay_half_lives: tuple[float, ...] = (1, 2, 3)

    def __post_init__(self) -> None:
        object.__setattr__(self, "horizons", tuple(int(h) for h in self.horizons))
        object.__setattr__(self, "decay_bars", tuple(int(h) for h in self.decay_bars))
        object.__setattr__(
            self, "decay_half_lives", tuple(float(h) for h in self.decay_half_lives)
        )
        if any(h < 1 for h in (*self.horizons, *self.decay_bars)):
            raise ConfigError("expected_path: every horizon must be >= 1 bar")


@dataclass(frozen=True, slots=True)
class ExtremesConfig:
    abs_z_levels: tuple[float, ...] = (1.0, 2.0, 3.0)
    primary_abs_z: float = 2.0
    forward_horizons: tuple[int, ...] = (1, 5, 10, 20)
    max_horizon: int = 500
    adverse_horizon: int = 50
    first_passage_fractions: tuple[float, ...] = (0.75, 0.50, 0.25)
    min_samples_warning: int = 200
    write_events: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "abs_z_levels", tuple(sorted(float(z) for z in self.abs_z_levels))
        )
        object.__setattr__(
            self, "forward_horizons", tuple(int(h) for h in self.forward_horizons)
        )
        object.__setattr__(
            self, "first_passage_fractions",
            tuple(float(f) for f in self.first_passage_fractions),
        )
        if not self.abs_z_levels or any(z <= 0 for z in self.abs_z_levels):
            raise ConfigError("extremes.abs_z_levels: need at least one level > 0")
        if self.primary_abs_z not in self.abs_z_levels:
            raise ConfigError(
                f"extremes.primary_abs_z={self.primary_abs_z} is not one of "
                f"extremes.abs_z_levels={list(self.abs_z_levels)}"
            )
        if any(h < 1 for h in self.forward_horizons):
            raise ConfigError("extremes.forward_horizons: every horizon must be >= 1")
        if max(self.forward_horizons) > self.max_horizon:
            raise ConfigError("extremes.forward_horizons: a horizon exceeds max_horizon")
        if not 1 <= self.adverse_horizon <= self.max_horizon:
            raise ConfigError("extremes.adverse_horizon: must be in [1, max_horizon]")
        if any(not 0 < f < 1 for f in self.first_passage_fractions):
            raise ConfigError("extremes.first_passage_fractions: each must be in (0, 1)")
        if 0.5 not in self.first_passage_fractions:
            raise ConfigError(
                "extremes.first_passage_fractions: must include 0.5, which defines "
                "the realized half-decay time compared against the estimated half-life"
            )


@dataclass(frozen=True, slots=True)
class StabilityConfig:
    rolling_window: int = 1000
    segment_shift_factor: float = 2.0
    jump_factor: float = 1.5

    def __post_init__(self) -> None:
        if self.rolling_window < 10:
            raise ConfigError("stability.rolling_window: must be >= 10")
        if self.segment_shift_factor <= 1 or self.jump_factor <= 1:
            raise ConfigError("stability: shift and jump factors must be > 1")


@dataclass(frozen=True, slots=True)
class ConditioningConfig:
    volatility_buckets: int = 4
    trend_buckets: int = 5
    r_squared_buckets: int = 4
    time_basis: str = "timestamp"
    by_hour: bool = True
    by_session: bool = True
    min_samples_warning: int = 200

    def __post_init__(self) -> None:
        _one_of(self.time_basis, ("timestamp", "timestamp_utc"), "conditioning.time_basis")
        for name in ("volatility_buckets", "trend_buckets", "r_squared_buckets"):
            if getattr(self, name) < 2:
                raise ConfigError(f"conditioning.{name}: must be >= 2")


@dataclass(frozen=True, slots=True)
class DiagnosticsConfig:
    max_lag: int = 50
    ljung_box_lags: tuple[int, ...] = (1, 5, 10, 20, 50)
    structural_lags: bool = True
    arch_lags: int = 10
    confidence_level: float = 0.95
    max_observations: int = 500_000

    def __post_init__(self) -> None:
        object.__setattr__(self, "ljung_box_lags", tuple(int(k) for k in self.ljung_box_lags))
        if self.max_lag < 1 or self.arch_lags < 1:
            raise ConfigError("diagnostics: max_lag and arch_lags must be >= 1")
        if not 0 < self.confidence_level < 1:
            raise ConfigError("diagnostics.confidence_level: must be in (0, 1)")


@dataclass(frozen=True, slots=True)
class ControlsConfig:
    random_walk: bool = True
    ou_reference: bool = True
    shuffled_returns: bool = False
    seed: int = 20260924

    def enabled(self) -> list[str]:
        """Names of the comparison series to build, in report order."""
        out = []
        if self.random_walk:
            out.append("control_random_walk")
        if self.ou_reference:
            out.append("reference_ou")
        if self.shuffled_returns:
            out.append("control_shuffled_returns")
        return out


@dataclass(frozen=True, slots=True)
class PlotsConfig:
    enabled: bool = True
    format: str = "png"
    dpi: int = 130
    figsize: tuple[float, float] = (11.0, 6.0)
    sample_bars: int = 2000
    max_series_points: int = 20_000
    max_scatter_points: int = 40_000

    def __post_init__(self) -> None:
        object.__setattr__(self, "figsize", tuple(self.figsize))


@dataclass(frozen=True, slots=True)
class OutputConfig:
    write_parquet: bool = True
    write_csv: bool = True
    write_json: bool = True


@dataclass(frozen=True, slots=True)
class OUConfig:
    """Fully-resolved Ornstein-Uhlenbeck research configuration."""

    project_root: Path
    config_path: Path | None
    schema_version: str
    results_path: Path
    timeframes: tuple[str, ...]
    regression_windows: tuple[int, ...]
    ou_estimation_windows: tuple[int, ...]
    estimation_methods: tuple[str, ...]
    min_observations: int
    dt: float

    valid_b_range: ValidBRange
    representative: RepresentativeConfig
    half_life: HalfLifeConfig
    near_unit_root: NearUnitRootConfig
    numerical: NumericalConfig
    rolling_estimation: RollingEstimationConfig
    segmented_estimation: SegmentedEstimationConfig
    equilibrium: EquilibriumConfig
    expected_path: ExpectedPathConfig
    extremes: ExtremesConfig
    stability: StabilityConfig
    conditioning: ConditioningConfig
    diagnostics: DiagnosticsConfig
    controls: ControlsConfig
    plots: PlotsConfig
    output: OutputConfig

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "regression_windows", tuple(int(w) for w in self.regression_windows)
        )
        object.__setattr__(
            self, "ou_estimation_windows", tuple(int(w) for w in self.ou_estimation_windows)
        )
        object.__setattr__(self, "estimation_methods", tuple(self.estimation_methods))
        if any(w < 3 for w in self.regression_windows):
            raise ConfigError("regression_windows: a regression window needs >= 3 bars")
        if any(w < 4 for w in self.ou_estimation_windows):
            raise ConfigError(
                "ou_estimation_windows: an OU window needs >= 4 observations "
                "(3 pairs, so the innovation variance has a degree of freedom)"
            )
        if not self.estimation_methods:
            raise ConfigError("estimation_methods: at least one method is required")
        for method in self.estimation_methods:
            _one_of(method, KNOWN_METHODS, "estimation_methods")
        if self.min_observations < 10:
            raise ConfigError("min_observations: must be >= 10")
        if self.dt <= 0:
            raise ConfigError("dt: must be > 0")

    # -- validity rules shared by every estimator ---------------------------
    def validity_rules(self) -> OUValidityRules:
        """The thresholds that decide which AR(1) fits receive an OU reading."""
        return OUValidityRules(
            dt=float(self.dt),
            valid_b_min=float(self.valid_b_range.min),
            valid_b_max=float(self.valid_b_range.max),
            unit_root_tolerance=float(self.numerical.unit_root_tolerance),
            max_half_life_bars=float(self.half_life.max_reportable_bars),
            near_unit_root_threshold=float(self.near_unit_root.flag_threshold),
            min_regressor_variance=float(self.numerical.min_regressor_variance),
            zero_innovation_tolerance=float(self.numerical.zero_innovation_tolerance),
        )

    # -- paths ----------------------------------------------------------------
    def results_dir(self, timeframe: str, regression_window: int, ou_window: int) -> Path:
        return (
            self.results_path / timeframe / f"regression_{regression_window}"
            / f"ou_{ou_window}"
        )

    def plots_dir(self, timeframe: str, regression_window: int, ou_window: int) -> Path:
        return self.results_dir(timeframe, regression_window, ou_window) / "plots"

    def to_dict(self) -> dict[str, Any]:
        return _as_dict(self)

    def fingerprint(self) -> str:
        """Stable digest of the settings that affect output."""
        payload = self.to_dict()
        for key in ("project_root", "config_path", "results_path"):
            payload.pop(key, None)
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.blake2b(blob, digest_size=8).hexdigest()


_SECTIONS: dict[str, type] = {
    "valid_b_range": ValidBRange,
    "representative": RepresentativeConfig,
    "half_life": HalfLifeConfig,
    "near_unit_root": NearUnitRootConfig,
    "numerical": NumericalConfig,
    "rolling_estimation": RollingEstimationConfig,
    "segmented_estimation": SegmentedEstimationConfig,
    "equilibrium": EquilibriumConfig,
    "expected_path": ExpectedPathConfig,
    "extremes": ExtremesConfig,
    "stability": StabilityConfig,
    "conditioning": ConditioningConfig,
    "diagnostics": DiagnosticsConfig,
    "controls": ControlsConfig,
    "plots": PlotsConfig,
    "output": OutputConfig,
}


def default_ou_config_path(root: Path | None = None) -> Path:
    root = root or find_project_root()
    return resolve_path(os.environ.get(OU_CONFIG_ENV_VAR) or DEFAULT_OU_RELPATH, root)


def load_ou_config(
    path: str | os.PathLike[str] | None = None,
    *,
    root: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> OUConfig:
    """Load, expand, validate and resolve ``config/ou.yaml``."""
    root = root or find_project_root()
    cfg_path = resolve_path(path, root) if path is not None else default_ou_config_path(root)
    if not cfg_path.exists():
        raise ConfigError(f"OU configuration file not found: {cfg_path}")

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
    results_path = resolve_path(data.pop("results_path", "results/ou_research"), root)
    schema_version = str(data.pop("schema_version", "1.0.0"))
    timeframes = tuple(data.pop("timeframes", ("1m", "5m", "15m", "30m", "1h")))
    regression_windows = tuple(data.pop("regression_windows", (32, 64, 128, 256, 512)))
    ou_windows = tuple(data.pop("ou_estimation_windows", (64, 128, 256, 512)))
    methods = tuple(data.pop("estimation_methods", ("ar1_ols",)))
    min_observations = int(data.pop("min_observations", 50))
    dt = float(data.pop("dt", 1.0))

    if data:
        known = sorted({
            *_SECTIONS, "results_path", "schema_version", "timeframes", "regression_windows",
            "ou_estimation_windows", "estimation_methods", "min_observations", "dt",
        })
        raise ConfigError(
            f"{cfg_path}: unknown top-level option(s) {sorted(data)}. Known: {known}"
        )

    return OUConfig(
        project_root=root,
        config_path=cfg_path,
        schema_version=schema_version,
        results_path=results_path,
        timeframes=timeframes,
        regression_windows=regression_windows,
        ou_estimation_windows=ou_windows,
        estimation_methods=methods,
        min_observations=min_observations,
        dt=dt,
        **sections,
    )
