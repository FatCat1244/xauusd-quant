"""Typed configuration of the feature factory (Prompt #8, ``config/features.yaml``).

Same contract as every other layer: frozen dataclasses, unknown keys are an
error, and the settings hash to a fingerprint that is part of every feature
version. The factory reads the earlier layers' own configurations for what
it does not define itself (the regression store, the OU rules, the stored
FFT / wavelet feature sets, the regime store, the sessions).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..utils.config import ConfigError, _as_dict, _build, _expand
from ..utils.paths import find_project_root, resolve_path

__all__ = [
    "FAMILIES",
    "FeatureFactoryConfig",
    "InteractionDef",
    "default_features_config_path",
    "load_features_config",
]

DEFAULT_FEATURES_RELPATH = "config/features.yaml"
FEATURES_CONFIG_ENV_VAR = "XAUUSD_FEATURES_CONFIG"

#: The feature families, in the order the factory builds and the ablation grows them.
FAMILIES: tuple[str, ...] = ("returns", "volatility", "autocorrelation", "regression", "ou",
                             "fft", "wavelet", "regime", "microstructure", "time")
PRIOR_STATUSES: tuple[str, ...] = ("candidate", "weak", "failed_null_control", "redundant",
                                   "invalid", "non_causal", "retained_for_reference")


def _ints(values: Any, where: str, *, minimum: int = 1) -> tuple[int, ...]:
    try:
        out = tuple(int(v) for v in values)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{where}: expected a list of integers") from exc
    if any(v < minimum for v in out):
        raise ConfigError(f"{where}: every value must be >= {minimum}")
    return out


@dataclass(frozen=True, slots=True)
class ReturnsFamily:
    horizons: tuple[int, ...] = (1, 2, 5, 10, 20)
    z_horizons: tuple[int, ...] = (20, 64)
    moment_windows: tuple[int, ...] = (64, 256)
    range_windows: tuple[int, ...] = (64,)
    volatility_window: int = 50

    def __post_init__(self) -> None:
        for name in ("horizons", "z_horizons", "moment_windows", "range_windows"):
            object.__setattr__(self, name, _ints(getattr(self, name), f"returns.{name}"))
        if min(self.moment_windows, default=8) < 8:
            raise ConfigError("returns.moment_windows: need at least 8 bars")


@dataclass(frozen=True, slots=True)
class VolatilityFamily:
    rv_windows: tuple[int, ...] = (5, 20, 64, 256, 1024)
    ewma_lambdas: tuple[float, ...] = (0.94, 0.99)
    change_lag: int = 20
    ratio_pair: tuple[int, ...] = (20, 256)
    range_window: int = 20

    def __post_init__(self) -> None:
        object.__setattr__(self, "rv_windows", _ints(self.rv_windows, "volatility.rv_windows",
                                                     minimum=2))
        lambdas = tuple(float(v) for v in self.ewma_lambdas)
        if any(not 0.0 < v < 1.0 for v in lambdas):
            raise ConfigError("volatility.ewma_lambdas: each in (0, 1)")
        object.__setattr__(self, "ewma_lambdas", lambdas)
        pair = _ints(self.ratio_pair, "volatility.ratio_pair", minimum=2)
        if len(pair) != 2 or pair[0] >= pair[1] or not set(pair) <= set(self.rv_windows):
            raise ConfigError("volatility.ratio_pair: two rv_windows, short first")
        object.__setattr__(self, "ratio_pair", pair)


@dataclass(frozen=True, slots=True)
class AutocorrelationFamily:
    return_lags: tuple[int, ...] = (1, 2, 5)
    window: int = 256
    lag1_windows: tuple[int, ...] = (128, 512)

    def __post_init__(self) -> None:
        object.__setattr__(self, "return_lags", _ints(self.return_lags,
                                                      "autocorrelation.return_lags"))
        object.__setattr__(self, "lag1_windows", _ints(self.lag1_windows,
                                                       "autocorrelation.lag1_windows", minimum=16))
        if int(self.window) < 16:
            raise ConfigError("autocorrelation.window: need at least 16 bars")


@dataclass(frozen=True, slots=True)
class RegressionFamily:
    windows: tuple[int, ...] = (32, 64, 128, 256, 512)
    volatility_window: int = 50

    def __post_init__(self) -> None:
        object.__setattr__(self, "windows", _ints(self.windows, "regression.windows", minimum=8))


@dataclass(frozen=True, slots=True)
class OUFamily:
    regression_window: int = 128
    windows: tuple[int, ...] = (128, 256, 512)
    full_window: int = 256
    decay_horizon: int = 10

    def __post_init__(self) -> None:
        object.__setattr__(self, "windows", _ints(self.windows, "ou.windows", minimum=16))
        if self.full_window not in self.windows:
            raise ConfigError("ou.full_window must be one of ou.windows")


@dataclass(frozen=True, slots=True)
class FFTFamily:
    series: str = "log_return"
    windows: tuple[int, ...] = (128, 256, 512)
    full_window: int = 256
    residual_window: int = 256
    abs_innovation_window: int = 256

    def __post_init__(self) -> None:
        object.__setattr__(self, "windows", _ints(self.windows, "fft.windows", minimum=16))
        if self.full_window not in self.windows:
            raise ConfigError("fft.full_window must be one of fft.windows")


@dataclass(frozen=True, slots=True)
class WaveletFamily:
    series: str = "log_return"
    windows: tuple[int, ...] = (256, 512, 1024)
    full_window: int = 512

    def __post_init__(self) -> None:
        object.__setattr__(self, "windows", _ints(self.windows, "wavelet.windows", minimum=32))
        if self.full_window not in self.windows:
            raise ConfigError("wavelet.full_window must be one of wavelet.windows")


@dataclass(frozen=True, slots=True)
class RegimeFamily:
    model: str = "hmm"
    states: int = 3
    scheme: str = "expanding_quarterly"


@dataclass(frozen=True, slots=True)
class MicrostructureFamily:
    change_window: int = 20


@dataclass(frozen=True, slots=True)
class TimeFamily:
    session_indicators: bool = True


@dataclass(frozen=True, slots=True)
class FamiliesConfig:
    returns: ReturnsFamily = field(default_factory=ReturnsFamily)
    volatility: VolatilityFamily = field(default_factory=VolatilityFamily)
    autocorrelation: AutocorrelationFamily = field(default_factory=AutocorrelationFamily)
    regression: RegressionFamily = field(default_factory=RegressionFamily)
    ou: OUFamily = field(default_factory=OUFamily)
    fft: FFTFamily = field(default_factory=FFTFamily)
    wavelet: WaveletFamily = field(default_factory=WaveletFamily)
    regime: RegimeFamily = field(default_factory=RegimeFamily)
    microstructure: MicrostructureFamily = field(default_factory=MicrostructureFamily)
    time: TimeFamily = field(default_factory=TimeFamily)


@dataclass(frozen=True, slots=True)
class PriorRule:
    pattern: str
    status: str
    evidence: str

    def __post_init__(self) -> None:
        if self.status not in PRIOR_STATUSES:
            raise ConfigError(f"prior_status: unknown status {self.status!r}; "
                              f"expected one of {list(PRIOR_STATUSES)}")


@dataclass(frozen=True, slots=True)
class InvalidRule:
    pattern: str
    timeframes: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeframes", tuple(str(t) for t in self.timeframes))


@dataclass(frozen=True, slots=True)
class InteractionDef:
    a: str
    b: str
    why: str = ""

    @property
    def name(self) -> str:
        return f"ix_{self.a}__x__{self.b}"


@dataclass(frozen=True, slots=True)
class FeatureFactoryConfig:
    """Everything ``config/features.yaml`` defines, validated."""

    schema_version: str
    timeframes: tuple[str, ...]
    source_feed: str
    factory_path: Path
    results_path: Path
    percentile_days: int
    trading_hours_per_day: float
    families: FamiliesConfig
    prior_status: tuple[PriorRule, ...]
    invalid: tuple[InvalidRule, ...]
    interaction_standardisation_days: int
    interactions: tuple[InteractionDef, ...]
    ablation: dict[str, tuple[str, ...]]
    project_root: Path
    config_path: Path

    def bars_per_day(self, bar_seconds: float) -> int:
        return max(1, int(round(self.trading_hours_per_day * 3600.0 / bar_seconds)))

    def percentile_window(self, bar_seconds: float) -> int:
        return self.percentile_days * self.bars_per_day(bar_seconds)

    def prior(self, name: str) -> PriorRule:
        """The first prior-status rule whose pattern matches feature *name*."""
        for rule in self.prior_status:
            if fnmatch.fnmatchcase(name, rule.pattern):
                return rule
        return PriorRule("*", "candidate", "no earlier evidence")

    def invalid_reason(self, name: str, timeframe: str) -> str | None:
        for rule in self.invalid:
            if timeframe in rule.timeframes and fnmatch.fnmatchcase(name, rule.pattern):
                return rule.reason
        return None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = _as_dict(self)
        return out

    def fingerprint(self) -> str:
        payload = self.to_dict()
        for key in ("project_root", "config_path", "factory_path", "results_path"):
            payload.pop(key, None)
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        return hashlib.blake2b(blob, digest_size=8).hexdigest()


def default_features_config_path(root: Path | None = None) -> Path:
    root = root or find_project_root()
    return resolve_path(os.environ.get(FEATURES_CONFIG_ENV_VAR) or DEFAULT_FEATURES_RELPATH,
                        root)


_TOP = ("schema_version", "timeframes", "source_feed", "storage", "percentile_days",
        "trading_hours_per_day", "families", "prior_status", "invalid",
        "interaction_standardisation_days", "interactions", "ablation")


def load_features_config(path: str | os.PathLike[str] | None = None, *,
                         root: Path | None = None) -> FeatureFactoryConfig:
    """Load and validate ``config/features.yaml``."""
    root = root or find_project_root()
    cfg_path = resolve_path(path, root) if path is not None else default_features_config_path(root)
    if not cfg_path.exists():
        raise ConfigError(f"Feature configuration file not found: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{cfg_path}: top level must be a mapping")
    data: dict[str, Any] = _expand(raw)
    unknown = sorted(set(data) - set(_TOP))
    if unknown:
        raise ConfigError(f"{cfg_path}: unknown top-level key(s) {unknown}")
    fam_raw = data.get("families") or {}
    unknown_fam = sorted(set(fam_raw) - set(FAMILIES))
    if unknown_fam:
        raise ConfigError(f"families: unknown family(ies) {unknown_fam}")
    fam_classes = {"returns": ReturnsFamily, "volatility": VolatilityFamily,
                   "autocorrelation": AutocorrelationFamily, "regression": RegressionFamily,
                   "ou": OUFamily, "fft": FFTFamily, "wavelet": WaveletFamily,
                   "regime": RegimeFamily, "microstructure": MicrostructureFamily,
                   "time": TimeFamily}
    families = FamiliesConfig(**{name: _build(cls, fam_raw.get(name), f"families.{name}")
                                 for name, cls in fam_classes.items()})
    storage = data.get("storage") or {}
    ablation_raw = data.get("ablation") or {}
    ablation: dict[str, tuple[str, ...]] = {}
    for key, groups in ablation_raw.items():
        groups = tuple(str(g) for g in groups)
        bad = [g for g in groups if g not in FAMILIES]
        if bad:
            raise ConfigError(f"ablation.{key}: unknown family(ies) {bad}")
        ablation[str(key)] = groups
    timeframes = tuple(str(t) for t in data.get("timeframes") or ())
    if not timeframes:
        raise ConfigError("timeframes: at least one")
    return FeatureFactoryConfig(
        schema_version=str(data.get("schema_version", "1.0.0")),
        timeframes=timeframes,
        source_feed=str(data.get("source_feed", "historical_research_feed")),
        factory_path=resolve_path(storage.get("factory_path") or "data/features/factory", root),
        results_path=resolve_path(storage.get("results_path") or "results/feature_research",
                                  root),
        percentile_days=int(data.get("percentile_days", 20)),
        trading_hours_per_day=float(data.get("trading_hours_per_day", 23.0)),
        families=families,
        prior_status=tuple(_build(PriorRule, r, "prior_status[]")
                           for r in (data.get("prior_status") or [])),
        invalid=tuple(_build(InvalidRule, r, "invalid[]") for r in (data.get("invalid") or [])),
        interaction_standardisation_days=int(data.get("interaction_standardisation_days", 20)),
        interactions=tuple(_build(InteractionDef, r, "interactions[]")
                           for r in (data.get("interactions") or [])),
        ablation=ablation,
        project_root=root,
        config_path=cfg_path,
    )
