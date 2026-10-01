"""Typed configuration of the predictive feature research (``config/alpha_research.yaml``)."""

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

__all__ = ["AlphaConfig", "PreregisteredHypothesis", "default_alpha_config_path",
           "load_alpha_config"]

DEFAULT_ALPHA_RELPATH = "config/alpha_research.yaml"
ALPHA_CONFIG_ENV_VAR = "XAUUSD_ALPHA_CONFIG"


def _ints(values: Any, where: str, *, minimum: int = 1) -> tuple[int, ...]:
    out = tuple(int(v) for v in values)
    if any(v < minimum for v in out):
        raise ConfigError(f"{where}: every value must be >= {minimum}")
    return out


def _strs(values: Any) -> tuple[str, ...]:
    return tuple(str(v) for v in values)


@dataclass(frozen=True, slots=True)
class ICConfig:
    min_observations: int = 2000
    min_year_observations: int = 500
    min_quarter_observations: int = 250
    recent_start: str = "2021-01-01"
    eras: int = 3
    headline: str = "spearman"
    hac_lag_months: int = 2

    def __post_init__(self) -> None:
        if self.headline not in ("spearman", "pearson"):
            raise ConfigError("ic.headline: spearman or pearson")
        if int(self.eras) < 2:
            raise ConfigError("ic.eras: at least 2")
        if int(self.hac_lag_months) < 0:
            raise ConfigError("ic.hac_lag_months: >= 0")


@dataclass(frozen=True, slots=True)
class DecayConfig:
    marginal_lags: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 50)
    zero_z: float = 2.0
    min_fit_points: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(self, "marginal_lags", tuple(sorted(set(
            _ints(self.marginal_lags, "decay.marginal_lags")))))
        object.__setattr__(self, "zero_z", float(self.zero_z))


@dataclass(frozen=True, slots=True)
class RollingConfig:
    window_months: int = 24
    min_months: int = 12


@dataclass(frozen=True, slots=True)
class ConditioningConfig:
    volatility_feature: str = "log_rv_20"
    volatility_buckets: int = 4
    regime_thresholds: tuple[float, ...] = (0.5, 0.8)
    intraday: tuple[str, ...] = ("hour", "session", "weekday")
    target_families: tuple[str, ...] = ("return", "abs_return", "residual_reduction")
    horizons: tuple[int, ...] = (1, 5, 20)

    def __post_init__(self) -> None:
        object.__setattr__(self, "regime_thresholds",
                           tuple(float(v) for v in self.regime_thresholds))
        object.__setattr__(self, "intraday", _strs(self.intraday))
        object.__setattr__(self, "target_families", _strs(self.target_families))
        object.__setattr__(self, "horizons", _ints(self.horizons, "conditioning.horizons"))
        bad = [g for g in self.intraday if g not in ("hour", "session", "weekday")]
        if bad:
            raise ConfigError(f"conditioning.intraday: unknown {bad}")


@dataclass(frozen=True, slots=True)
class MIConfig:
    bins: int = 20
    target_families: tuple[str, ...] = ("return", "abs_return", "realized_vol",
                                        "residual_reduction")
    horizons: tuple[int, ...] = (1, 5, 20)
    null_shifts: int = 20
    null_permutations: int = 10
    conditional_on_volatility: bool = True
    feature_pairs_sample_rows: int = 100_000

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_families", _strs(self.target_families))
        object.__setattr__(self, "horizons", _ints(self.horizons, "mutual_information.horizons"))
        if not 4 <= int(self.bins) <= 64:
            raise ConfigError("mutual_information.bins: 4..64")


@dataclass(frozen=True, slots=True)
class RedundancyConfig:
    sample_rows: int = 200_000
    cluster_abs_corr: float = 0.9
    graph_abs_corr: float = 0.7
    linkage: str = "average"
    nonmonotone_nmi: float = 0.5

    def __post_init__(self) -> None:
        if not 0 < float(self.graph_abs_corr) <= float(self.cluster_abs_corr) < 1:
            raise ConfigError("redundancy: need 0 < graph_abs_corr <= cluster_abs_corr < 1")
        if self.linkage not in ("average", "complete", "single"):
            raise ConfigError("redundancy.linkage: average, complete or single")


@dataclass(frozen=True, slots=True)
class DecilesConfig:
    quantiles: int = 10
    target_families: tuple[str, ...] = ("return", "abs_return", "realized_vol",
                                        "residual_reduction", "residual_shrinks")
    horizons: tuple[int, ...] = (1, 5, 20)

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_families", _strs(self.target_families))
        object.__setattr__(self, "horizons", _ints(self.horizons, "deciles.horizons"))


@dataclass(frozen=True, slots=True)
class SyntheticConfig:
    white: int = 10
    ar1: tuple[float, ...] = (0.9, 0.99, 0.999)
    ar1_per_phi: int = 10
    matched_marginal: int = 10

    def __post_init__(self) -> None:
        phis = tuple(float(v) for v in self.ar1)
        if any(not 0.0 < v < 1.0 for v in phis):
            raise ConfigError("nulls.synthetic.ar1: each in (0, 1)")
        object.__setattr__(self, "ar1", phis)


@dataclass(frozen=True, slots=True)
class NullsConfig:
    circular_shifts: int = 100
    min_shift_fraction: float = 0.1
    permutations: int = 20
    null_quantile: float = 0.99
    time_shift_days: tuple[int, ...] = (5, 60)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    pipeline: tuple[str, ...] = ("random_walk", "shuffled_returns", "sign_flip",
                                 "block_bootstrap_1024")
    pipeline_families: tuple[str, ...] = ("returns", "volatility", "autocorrelation",
                                          "regression", "ou", "fft", "wavelet")
    pipeline_veto: tuple[str, ...] = ("random_walk", "shuffled_returns")
    pipeline_sign_veto: tuple[str, ...] = ("sign_flip",)
    veto_share: float = 0.5
    veto_null_z: float = 2.0
    slow_component_ratio: float = 0.5

    #: target kinds a sign-randomised null may veto (it keeps every magnitude effect)
    SIGN_VETO_KINDS = ("direction", "residual")

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_shift_days", _ints(self.time_shift_days,
                                                          "nulls.time_shift_days"))
        object.__setattr__(self, "pipeline", _strs(self.pipeline))
        object.__setattr__(self, "pipeline_families", _strs(self.pipeline_families))
        object.__setattr__(self, "pipeline_veto", _strs(self.pipeline_veto))
        object.__setattr__(self, "pipeline_sign_veto", _strs(self.pipeline_sign_veto))
        object.__setattr__(self, "veto_share", float(self.veto_share))
        object.__setattr__(self, "veto_null_z", float(self.veto_null_z))
        object.__setattr__(self, "slow_component_ratio", float(self.slow_component_ratio))
        if not 0.0 < float(self.min_shift_fraction) < 0.5:
            raise ConfigError("nulls.min_shift_fraction: in (0, 0.5)")
        if not 0.5 < float(self.null_quantile) < 1.0:
            raise ConfigError("nulls.null_quantile: in (0.5, 1)")
        if not 0.0 < self.veto_share <= 1.0:
            raise ConfigError("nulls.veto_share: in (0, 1]")
        vetoes = (*self.pipeline_veto, *self.pipeline_sign_veto)
        bad = [v for v in vetoes if v not in self.pipeline]
        if bad:
            raise ConfigError(f"nulls.pipeline_veto / pipeline_sign_veto: {bad} are not in "
                              "nulls.pipeline")
        if any(v.startswith("block_bootstrap") for v in vetoes):
            raise ConfigError("nulls.pipeline_veto: a block bootstrap keeps within-block "
                              "predictability and cannot be a veto null")

    def veto_nulls(self, kind: str) -> tuple[str, ...]:
        """The pipeline nulls that may veto a claim about outcomes of *kind*."""
        extra = self.pipeline_sign_veto if kind in self.SIGN_VETO_KINDS else ()
        return (*self.pipeline_veto, *extra)


@dataclass(frozen=True, slots=True)
class MultipleTestingConfig:
    fdr_q: float = 0.05
    bonferroni_alpha: float = 0.05


@dataclass(frozen=True, slots=True)
class InteractionsConfig:
    folds: int = 4
    ridge: float = 1.0
    max_rows: int = 400_000
    target_families: tuple[str, ...] = ("return", "abs_return", "residual_reduction")
    horizons: tuple[int, ...] = (1, 5, 20)

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_families", _strs(self.target_families))
        object.__setattr__(self, "horizons", _ints(self.horizons, "interactions.horizons"))


@dataclass(frozen=True, slots=True)
class ClassificationConfig:
    min_abs_rank_ic: float = 0.02
    strong_abs_rank_ic: float = 0.05
    candidate_sign_years: float = 0.65
    strong_sign_years: float = 0.8
    max_redundancy_rho: float = 0.9


@dataclass(frozen=True, slots=True)
class CostConfig:
    sample_bars: int = 20_000
    live_updates: int = 100
    cheap_ms: float = 0.5
    expensive_ms: float = 10.0


@dataclass(frozen=True, slots=True)
class PlotsConfig:
    enabled: bool = True
    top_features: int = 12


@dataclass(frozen=True, slots=True)
class PreregisteredHypothesis:
    """A claim fixed before any Prompt #8 statistic existed (from Prompts #2-#7)."""

    id: str
    claim: str
    features: tuple[str, ...]
    target_families: tuple[str, ...]
    expected: str
    horizons: tuple[int, ...] = ()               # empty = every configured horizon

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", _strs(self.features))
        object.__setattr__(self, "target_families", _strs(self.target_families))
        object.__setattr__(self, "horizons", _ints(self.horizons, f"{self.id}.horizons"))
        if not self.id.startswith("ALPHA-PR-"):
            raise ConfigError(f"preregistered id {self.id!r}: must start with ALPHA-PR-")

    def covers(self, feature: str, target_family: str, horizon: int) -> bool:
        return (target_family in self.target_families
                and (not self.horizons or horizon in self.horizons)
                and any(fnmatch.fnmatchcase(feature, p) for p in self.features))


@dataclass(frozen=True, slots=True)
class AlphaConfig:
    schema_version: str
    seed: int
    ic: ICConfig
    decay: DecayConfig
    rolling: RollingConfig
    conditioning: ConditioningConfig
    mutual_information: MIConfig
    redundancy: RedundancyConfig
    deciles: DecilesConfig
    nulls: NullsConfig
    multiple_testing: MultipleTestingConfig
    interactions: InteractionsConfig
    classification: ClassificationConfig
    cost: CostConfig
    plots: PlotsConfig
    project_root: Path
    config_path: Path
    preregistered: tuple[PreregisteredHypothesis, ...] = ()
    ledger_path: Path = Path("results/research_ledger.parquet")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = _as_dict(self)
        return out

    def fingerprint(self) -> str:
        """Settings that change a statistic (the preregistration text and plots do not)."""
        payload = self.to_dict()
        for key in ("project_root", "config_path", "plots", "preregistered", "ledger_path"):
            payload.pop(key, None)
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        return hashlib.blake2b(blob, digest_size=8).hexdigest()

    def preregistration(self, feature: str, target_family: str, horizon: int) -> str | None:
        """The id of the first preregistered hypothesis covering this test, or None."""
        for h in self.preregistered:
            if h.covers(feature, target_family, horizon):
                return h.id
        return None


def default_alpha_config_path(root: Path | None = None) -> Path:
    root = root or find_project_root()
    return resolve_path(os.environ.get(ALPHA_CONFIG_ENV_VAR) or DEFAULT_ALPHA_RELPATH, root)


_SECTIONS: dict[str, type] = {
    "ic": ICConfig, "decay": DecayConfig, "rolling": RollingConfig,
    "conditioning": ConditioningConfig, "mutual_information": MIConfig,
    "redundancy": RedundancyConfig, "deciles": DecilesConfig,
    "multiple_testing": MultipleTestingConfig, "interactions": InteractionsConfig,
    "classification": ClassificationConfig, "cost": CostConfig, "plots": PlotsConfig,
}


def load_alpha_config(path: str | os.PathLike[str] | None = None, *,
                      root: Path | None = None) -> AlphaConfig:
    root = root or find_project_root()
    cfg_path = resolve_path(path, root) if path is not None else default_alpha_config_path(root)
    if not cfg_path.exists():
        raise ConfigError(f"Alpha research configuration file not found: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{cfg_path}: top level must be a mapping")
    data: dict[str, Any] = _expand(raw)
    unknown = sorted(set(data) - {"schema_version", "seed", "nulls", "preregistered", "ledger",
                                  *_SECTIONS})
    if unknown:
        raise ConfigError(f"{cfg_path}: unknown top-level key(s) {unknown}")
    sections: dict[str, Any] = {name: _build(cls, data.get(name), name)
                                for name, cls in _SECTIONS.items()}
    nulls_raw = dict(data.get("nulls") or {})
    synthetic = _build(SyntheticConfig, nulls_raw.pop("synthetic", None), "nulls.synthetic")
    nulls = _build(NullsConfig, {**nulls_raw, "synthetic": synthetic}, "nulls")
    prereg = tuple(_build(PreregisteredHypothesis, item, "preregistered")
                   for item in data.get("preregistered") or ())
    ids = [h.id for h in prereg]
    if len(set(ids)) != len(ids):
        raise ConfigError("preregistered: duplicate ids")
    ledger = data.get("ledger") or {}
    if not isinstance(ledger, dict) or set(ledger) - {"path"}:
        raise ConfigError("ledger: a mapping with only `path`")
    return AlphaConfig(schema_version=str(data.get("schema_version", "1.0.0")),
                       seed=int(data.get("seed", 20260930)), nulls=nulls,
                       project_root=root, config_path=cfg_path, preregistered=prereg,
                       ledger_path=resolve_path(ledger.get("path")
                                                or "results/research_ledger.parquet", root),
                       **sections)
