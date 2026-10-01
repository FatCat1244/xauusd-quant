"""Typed configuration for the research layer.

Mirrors :mod:`xauusd_quant.utils.config`: frozen dataclasses, unknown keys are
an error rather than a silent no-op, and the whole thing hashes to a stable
fingerprint that every report records.

The dataset itself is still described by ``config/data.yaml``. This file only
governs how that data is characterised, so a :class:`ResearchConfig` always
travels alongside a :class:`~xauusd_quant.utils.config.Config`.
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
    "ActivityConfig",
    "AnnualizationConfig",
    "AutocorrelationConfig",
    "BootstrapConfig",
    "ConditionalConfig",
    "DistributionConfig",
    "IntradayConfig",
    "NormalityConfig",
    "OutputConfig",
    "PlotsConfig",
    "ResearchConfig",
    "ReturnsConfig",
    "RollingConfig",
    "SessionWindow",
    "SessionsConfig",
    "SpreadConfig",
    "StabilityConfig",
    "StationarityConfig",
    "VolatilityConfig",
    "default_research_config_path",
    "load_research_config",
]

DEFAULT_RESEARCH_RELPATH = "config/research.yaml"
RESEARCH_CONFIG_ENV_VAR = "XAUUSD_RESEARCH_CONFIG"


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ReturnsConfig:
    price_column: str = "close"
    price_source: str = "mid"
    types: tuple[str, ...] = ("log", "simple")
    primary: str = "log"
    multi_period_horizons: tuple[int, ...] = (5, 10, 20)
    drop_session_gap_returns: bool = True
    session_gap_bar_multiple: float = 1.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "types", tuple(self.types))
        object.__setattr__(self, "multi_period_horizons", tuple(self.multi_period_horizons))
        _one_of(self.price_source, ("mid", "bid", "ask"), "returns.price_source")
        _one_of(self.primary, ("log", "simple"), "returns.primary")
        for t in self.types:
            _one_of(t, ("log", "simple"), "returns.types")
        if self.primary not in self.types:
            raise ConfigError(
                f"returns.primary={self.primary!r} is not in returns.types={list(self.types)}"
            )
        if any(h < 1 for h in self.multi_period_horizons):
            raise ConfigError("returns.multi_period_horizons: every horizon must be >= 1")


@dataclass(frozen=True, slots=True)
class DistributionConfig:
    quantiles: tuple[float, ...] = (
        0.001, 0.005, 0.01, 0.025, 0.05, 0.25, 0.50, 0.75, 0.95, 0.975, 0.99, 0.995, 0.999
    )
    gaussian_reference: bool = True
    tail_sigma_levels: tuple[float, ...] = (2, 3, 4, 5, 6)
    histogram_bins: int = 200

    def __post_init__(self) -> None:
        object.__setattr__(self, "quantiles", tuple(self.quantiles))
        object.__setattr__(self, "tail_sigma_levels", tuple(self.tail_sigma_levels))
        for q in self.quantiles:
            if not 0 < q < 1:
                raise ConfigError(f"distribution.quantiles: {q} is not in (0, 1)")


@dataclass(frozen=True, slots=True)
class NormalityConfig:
    jarque_bera: bool = True
    dagostino_k2: bool = True
    large_sample_warning_threshold: int = 100_000


@dataclass(frozen=True, slots=True)
class AutocorrelationConfig:
    max_lag: int = 100
    series: tuple[str, ...] = ("returns", "abs_returns", "squared_returns")
    confidence_level: float = 0.95
    ljung_box_lags: tuple[int, ...] = (5, 10, 20, 50)

    def __post_init__(self) -> None:
        object.__setattr__(self, "series", tuple(self.series))
        object.__setattr__(self, "ljung_box_lags", tuple(self.ljung_box_lags))
        if self.max_lag < 1:
            raise ConfigError("autocorrelation.max_lag: must be >= 1")
        for s in self.series:
            _one_of(s, ("returns", "abs_returns", "squared_returns"), "autocorrelation.series")
        if not 0 < self.confidence_level < 1:
            raise ConfigError("autocorrelation.confidence_level: must be in (0, 1)")


@dataclass(frozen=True, slots=True)
class AnnualizationConfig:
    """Annualisation is opt-in because the right factor is instrument-specific.

    XAUUSD trades roughly 23 h a day, five days a week. Applying the equity
    convention of 252 trading days x 6.5 h would overstate annualised
    volatility badly, so nothing is annualised until someone states the
    assumption deliberately.
    """

    enabled: bool = False
    trading_days_per_year: float = 260.0
    hours_per_trading_day: float = 23.0
    bars_per_year: float | None = None


@dataclass(frozen=True, slots=True)
class VolatilityConfig:
    rolling_windows: tuple[int, ...] = (20, 50, 100, 250)
    ewma_lambda: float = 0.94
    realized_vol_window: int = 20
    regime_quantiles: tuple[float, ...] = (0.25, 0.50, 0.75)
    annualization: AnnualizationConfig = field(default_factory=AnnualizationConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rolling_windows", tuple(self.rolling_windows))
        object.__setattr__(self, "regime_quantiles", tuple(self.regime_quantiles))
        if not 0 < self.ewma_lambda < 1:
            raise ConfigError("volatility.ewma_lambda: must be in (0, 1)")
        if any(w < 2 for w in self.rolling_windows):
            raise ConfigError("volatility.rolling_windows: every window must be >= 2")


@dataclass(frozen=True, slots=True)
class RollingConfig:
    windows: tuple[int, ...] = (20, 50, 100, 250)
    autocorr_lags: tuple[int, ...] = (1, 2, 5)
    min_periods_fraction: float = 0.9
    include_current_bar: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "windows", tuple(self.windows))
        object.__setattr__(self, "autocorr_lags", tuple(self.autocorr_lags))
        if not 0 < self.min_periods_fraction <= 1:
            raise ConfigError("rolling.min_periods_fraction: must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class RollingStationarityConfig:
    enabled: bool = False
    window_bars: int = 20_000
    step_bars: int = 10_000
    max_windows: int = 200


@dataclass(frozen=True, slots=True)
class SegmentedStationarityConfig:
    enabled: bool = True
    by: str = "year"
    min_observations: int = 5000
    rolling: RollingStationarityConfig = field(default_factory=RollingStationarityConfig)

    def __post_init__(self) -> None:
        _one_of(self.by, ("year", "quarter"), "stationarity.segmented.by")


@dataclass(frozen=True, slots=True)
class StationarityConfig:
    adf_enabled: bool = True
    kpss_enabled: bool = True
    series: tuple[str, ...] = ("price", "log_price", "log_returns")
    adf_regression: str = "c"
    adf_autolag: str | None = "AIC"
    kpss_regression: str = "c"
    kpss_nlags: str | int = "auto"
    max_observations: int = 500_000
    subsample_seed: int = 20260921
    segmented: SegmentedStationarityConfig = field(
        default_factory=SegmentedStationarityConfig
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "series", tuple(self.series))
        _one_of(self.adf_regression, ("c", "ct", "ctt", "n"), "stationarity.adf_regression")
        _one_of(self.kpss_regression, ("c", "ct"), "stationarity.kpss_regression")
        for s in self.series:
            _one_of(s, ("price", "log_price", "log_returns"), "stationarity.series")


@dataclass(frozen=True, slots=True)
class BootstrapConfig:
    enabled: bool = True
    iterations: int = 1000
    confidence_level: float = 0.95
    seed: int = 20260921
    max_samples: int = 200_000

    def __post_init__(self) -> None:
        if not 0 < self.confidence_level < 1:
            raise ConfigError("conditional.bootstrap.confidence_level: must be in (0, 1)")


@dataclass(frozen=True, slots=True)
class ConditionalConfig:
    lower_quantiles: tuple[float, ...] = (0.001, 0.005, 0.01, 0.025)
    upper_quantiles: tuple[float, ...] = (0.975, 0.99, 0.995, 0.999)
    forward_horizons: tuple[int, ...] = (1, 2, 3, 5, 10, 20)
    min_samples_warning: int = 200
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "lower_quantiles", tuple(self.lower_quantiles))
        object.__setattr__(self, "upper_quantiles", tuple(self.upper_quantiles))
        object.__setattr__(self, "forward_horizons", tuple(self.forward_horizons))
        for q in (*self.lower_quantiles, *self.upper_quantiles):
            if not 0 < q < 1:
                raise ConfigError(f"conditional quantile {q} is not in (0, 1)")
        if any(h < 1 for h in self.forward_horizons):
            raise ConfigError("conditional.forward_horizons: every horizon must be >= 1")


@dataclass(frozen=True, slots=True)
class SessionWindow:
    start: str = "00:00"
    end: str = "00:00"

    def __post_init__(self) -> None:
        for label, value in (("start", self.start), ("end", self.end)):
            try:
                hour, minute = (int(p) for p in value.split(":")[:2])
                assert 0 <= hour < 24 and 0 <= minute < 60
            except (ValueError, AssertionError) as exc:
                raise ConfigError(
                    f"intraday.sessions: {label}={value!r} is not a HH:MM time"
                ) from exc

    @property
    def start_minutes(self) -> int:
        hour, minute = (int(p) for p in self.start.split(":")[:2])
        return hour * 60 + minute

    @property
    def end_minutes(self) -> int:
        hour, minute = (int(p) for p in self.end.split(":")[:2])
        return hour * 60 + minute

    @property
    def wraps_midnight(self) -> bool:
        return self.end_minutes <= self.start_minutes


@dataclass(frozen=True, slots=True)
class SessionsConfig:
    enabled: bool = True
    definitions: dict[str, SessionWindow] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class IntradayConfig:
    enabled: bool = True
    time_basis: str = "timestamp"
    extreme_quantile: float = 0.99
    weekday_enabled: bool = True
    sessions: SessionsConfig = field(default_factory=SessionsConfig)

    def __post_init__(self) -> None:
        _one_of(self.time_basis, ("timestamp", "timestamp_utc"), "intraday.time_basis")
        if not 0.5 < self.extreme_quantile < 1:
            raise ConfigError("intraday.extreme_quantile: must be in (0.5, 1)")


@dataclass(frozen=True, slots=True)
class SpreadConfig:
    enabled: bool = True
    quantiles: tuple[float, ...] = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999)
    wide_quantile: float = 0.99
    correlate_with: tuple[str, ...] = ("volatility", "abs_return", "tick_count")

    def __post_init__(self) -> None:
        object.__setattr__(self, "quantiles", tuple(self.quantiles))
        object.__setattr__(self, "correlate_with", tuple(self.correlate_with))


@dataclass(frozen=True, slots=True)
class ActivityConfig:
    enabled: bool = True
    low_activity_quantile: float = 0.01
    quantiles: tuple[float, ...] = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)

    def __post_init__(self) -> None:
        object.__setattr__(self, "quantiles", tuple(self.quantiles))


@dataclass(frozen=True, slots=True)
class StabilityConfig:
    enabled: bool = True
    by: str = "year"
    min_observations: int = 1000
    metrics: tuple[str, ...] = (
        "mean", "std", "skew", "excess_kurtosis", "acf1",
        "extreme_frequency", "mean_spread", "mean_tick_count",
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", tuple(self.metrics))
        _one_of(self.by, ("year", "quarter"), "stability.by")


@dataclass(frozen=True, slots=True)
class PlotsConfig:
    enabled: bool = True
    format: str = "png"
    dpi: int = 130
    figsize: tuple[float, float] = (10.0, 6.0)
    style: str = "seaborn-v0_8-whitegrid"
    max_scatter_points: int = 50_000

    def __post_init__(self) -> None:
        object.__setattr__(self, "figsize", tuple(self.figsize))


@dataclass(frozen=True, slots=True)
class OutputConfig:
    write_parquet: bool = True
    write_csv: bool = True
    write_json: bool = True
    float_precision: int = 10


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ResearchConfig:
    """Fully-resolved research configuration."""

    project_root: Path
    config_path: Path | None
    schema_version: str
    results_path: Path
    timeframes: tuple[str, ...]

    returns: ReturnsConfig
    distribution: DistributionConfig
    normality: NormalityConfig
    autocorrelation: AutocorrelationConfig
    volatility: VolatilityConfig
    rolling: RollingConfig
    stationarity: StationarityConfig
    conditional: ConditionalConfig
    intraday: IntradayConfig
    spread: SpreadConfig
    activity: ActivityConfig
    stability: StabilityConfig
    plots: PlotsConfig
    output: OutputConfig

    def results_dir(self, timeframe: str) -> Path:
        return self.results_path / timeframe

    def plots_dir(self, timeframe: str) -> Path:
        return self.results_dir(timeframe) / "plots"

    def to_dict(self) -> dict[str, Any]:
        return _as_dict(self)

    def fingerprint(self) -> str:
        """Stable digest of the settings that affect research output."""
        payload = self.to_dict()
        for key in ("project_root", "config_path", "results_path"):
            payload.pop(key, None)
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.blake2b(blob, digest_size=8).hexdigest()


_SECTIONS: dict[str, type] = {
    "returns": ReturnsConfig,
    "distribution": DistributionConfig,
    "normality": NormalityConfig,
    "autocorrelation": AutocorrelationConfig,
    "rolling": RollingConfig,
    "spread": SpreadConfig,
    "activity": ActivityConfig,
    "stability": StabilityConfig,
    "plots": PlotsConfig,
    "output": OutputConfig,
}


def default_research_config_path(root: Path | None = None) -> Path:
    root = root or find_project_root()
    return resolve_path(
        os.environ.get(RESEARCH_CONFIG_ENV_VAR) or DEFAULT_RESEARCH_RELPATH, root
    )


def load_research_config(
    path: str | os.PathLike[str] | None = None,
    *,
    root: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> ResearchConfig:
    """Load, expand, validate and resolve ``config/research.yaml``."""
    root = root or find_project_root()
    cfg_path = (
        resolve_path(path, root) if path is not None else default_research_config_path(root)
    )
    if not cfg_path.exists():
        raise ConfigError(f"Research configuration file not found: {cfg_path}")

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

    # Sections with nested blocks need assembling by hand.
    vol_raw = dict(data.pop("volatility", {}) or {})
    annual = _build(
        AnnualizationConfig, vol_raw.pop("annualization", {}), "volatility.annualization"
    )
    volatility = _build(
        VolatilityConfig, {**vol_raw, "annualization": annual}, "volatility"
    )

    stat_raw = dict(data.pop("stationarity", {}) or {})
    seg_raw = dict(stat_raw.pop("segmented", {}) or {})
    seg_rolling = _build(
        RollingStationarityConfig, seg_raw.pop("rolling", {}), "stationarity.segmented.rolling"
    )
    segmented = _build(
        SegmentedStationarityConfig,
        {**seg_raw, "rolling": seg_rolling},
        "stationarity.segmented",
    )
    stationarity = _build(
        StationarityConfig, {**stat_raw, "segmented": segmented}, "stationarity"
    )

    cond_raw = dict(data.pop("conditional", {}) or {})
    bootstrap = _build(
        BootstrapConfig, cond_raw.pop("bootstrap", {}), "conditional.bootstrap"
    )
    conditional = _build(
        ConditionalConfig, {**cond_raw, "bootstrap": bootstrap}, "conditional"
    )

    intra_raw = dict(data.pop("intraday", {}) or {})
    sess_raw = dict(intra_raw.pop("sessions", {}) or {})
    definitions = {
        name: _build(SessionWindow, window, f"intraday.sessions.definitions.{name}")
        for name, window in (sess_raw.pop("definitions", {}) or {}).items()
    }
    sessions = _build(
        SessionsConfig, {**sess_raw, "definitions": definitions}, "intraday.sessions"
    )
    intraday = _build(IntradayConfig, {**intra_raw, "sessions": sessions}, "intraday")

    results_path = resolve_path(
        data.pop("results_path", "results/statistical_research"), root
    )
    schema_version = str(data.pop("schema_version", "1.0.0"))
    timeframes = tuple(data.pop("timeframes", ("1m", "5m", "15m", "30m", "1h")))

    if data:
        known = sorted({*_SECTIONS, "volatility", "stationarity", "conditional",
                        "intraday", "results_path", "schema_version", "timeframes"})
        raise ConfigError(
            f"{cfg_path}: unknown top-level option(s) {sorted(data)}. Known: {known}"
        )

    return ResearchConfig(
        project_root=root,
        config_path=cfg_path,
        schema_version=schema_version,
        results_path=results_path,
        timeframes=timeframes,
        volatility=volatility,
        stationarity=stationarity,
        conditional=conditional,
        intraday=intraday,
        **sections,
    )
