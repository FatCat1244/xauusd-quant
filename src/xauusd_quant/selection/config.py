"""Typed configuration of the feature selection (``config/feature_selection.yaml``, Prompt #9)."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from ..utils.config import ConfigError, _as_dict, _build, _expand
from ..utils.paths import find_project_root, resolve_path

__all__ = [
    "FeatureSelectionConfig",
    "Period",
    "default_selection_config_path",
    "load_selection_config",
]

DEFAULT_SELECTION_RELPATH = "config/feature_selection.yaml"
SELECTION_CONFIG_ENV_VAR = "XAUUSD_FEATURE_SELECTION_CONFIG"
_KINDS = ("direction", "reversion", "volatility", "magnitude")


def _day(value: Any, where: str) -> date | None:
    if value is None:
        return None
    try:
        out = date.fromisoformat(str(value))
    except ValueError as exc:
        raise ConfigError(f"{where}: not an ISO date: {value!r}") from exc
    if out.day != 1 or out.month not in (1, 4, 7, 10):
        raise ConfigError(f"{where}: {out} is not the first day of a calendar quarter")
    return out


@dataclass(frozen=True, slots=True)
class Period:
    """A half-open calendar period ``[start, end)``; ``end = None`` is open-ended."""

    name: str
    start: date
    end: date | None

    def label(self) -> str:
        return f"{self.start.isoformat()} .. {self.end.isoformat() if self.end else 'end'}"


@dataclass(frozen=True, slots=True)
class Fold:
    validate_start: date
    validate_end: date


@dataclass(frozen=True, slots=True)
class PeriodsConfig:
    development: Period
    validation: Period
    reserved_test: Period
    recent_development: Period
    folds: tuple[Fold, ...]
    purge_bars: int
    embargo_bars: int


@dataclass(frozen=True, slots=True)
class UniverseConfig:
    live_safe_only: bool = True
    exclude_non_causal: bool = True
    exclude_invalid: bool = True
    exclude_failed_null: bool = True
    interactions: bool = True
    probes: int = 8


@dataclass(frozen=True, slots=True)
class FiltersConfig:
    max_missing_fraction: dict[str, float] = field(default_factory=lambda: {
        "development": 0.4, "validation": 0.05, "recent_development": 0.05})
    near_constant: dict[str, float] = field(default_factory=lambda: {
        "max_dominant_fraction": 0.995, "min_unique_values": 2, "binary_min_minority": 0.001})
    numerical: dict[str, float] = field(default_factory=lambda: {"max_abs_value": 1e6})
    drift: dict[str, float] = field(default_factory=lambda: {
        "moderate_psi": 0.1, "strong_psi": 0.25, "strong_median_shift_iqr": 1.0})


@dataclass(frozen=True, slots=True)
class NullScreenConfig:
    circular_shifts: int = 200
    null_quantile: float = 0.99
    min_shift_fraction: float = 0.1
    fdr_q: float = 0.05
    hac_lag_months: int = 2
    min_observations: int = 2000


@dataclass(frozen=True, slots=True)
class TargetsConfig:
    direction: str = "return"
    reversion: str = "residual_reduction"
    volatility: str = "realized_vol"
    magnitude: str = "abs_return"
    horizons: tuple[int, ...] = (1, 5, 20)
    binary: dict[str, str] = field(default_factory=lambda: {"direction": "sign"})

    def __post_init__(self) -> None:
        object.__setattr__(self, "horizons", tuple(int(h) for h in self.horizons))

    def family(self, kind: str) -> str:
        return str(getattr(self, kind))

    def columns(self) -> list[tuple[str, int, str]]:
        """(kind, horizon, target column) of every selection target, in a fixed order."""
        return [(k, h, f"target_{self.family(k)}_{h}") for k in _KINDS for h in self.horizons]


@dataclass(frozen=True, slots=True)
class RedundancyConfig:
    max_pairwise_correlation: float = 0.95
    cluster_abs_corr: float = 0.9
    sample_rows: int = 200_000
    nonmonotone_nmi: float = 0.5


@dataclass(frozen=True, slots=True)
class RepresentativeConfig:
    consistency_round: float = 0.1


@dataclass(frozen=True, slots=True)
class MRMRConfig:
    relevance: tuple[str, ...] = ("abs_rank_ic", "mutual_information")
    redundancy: str = "abs_spearman"
    lambdas: tuple[float, ...] = (0.5, 1.0)
    schemes: tuple[str, ...] = ("difference", "quotient")
    max_features: int = 75
    mi_bins: int = 20

    def __post_init__(self) -> None:
        object.__setattr__(self, "relevance", tuple(str(v) for v in self.relevance))
        object.__setattr__(self, "lambdas", tuple(float(v) for v in self.lambdas))
        object.__setattr__(self, "schemes", tuple(str(v) for v in self.schemes))
        bad = [s for s in self.schemes if s not in ("difference", "quotient")]
        if bad:
            raise ConfigError(f"mrmr.schemes: unknown {bad}")


@dataclass(frozen=True, slots=True)
class StabilityConfig:
    resamples: int = 50
    block: str = "quarter"
    fraction: float = 0.5
    year_subsets: tuple[str, ...] = ("odd_years", "even_years", "early", "middle", "late")
    selector_k: int = 20
    frequency_threshold: float = 0.6

    def __post_init__(self) -> None:
        object.__setattr__(self, "year_subsets", tuple(str(v) for v in self.year_subsets))
        if self.block != "quarter":
            raise ConfigError("stability_selection.block: only quarter blocks are supported")


@dataclass(frozen=True, slots=True)
class LinearConfig:
    ridge_alpha: float = 0.01
    elastic_net_l2: tuple[float, ...] = (0.0, 0.1)
    max_path_features: int = 75
    logistic_max_rows: int = 150_000
    logistic_lambdas: tuple[float, ...] = (0.01, 0.005, 0.002, 0.001, 0.0005)

    def __post_init__(self) -> None:
        object.__setattr__(self, "elastic_net_l2", tuple(float(v) for v in self.elastic_net_l2))
        object.__setattr__(self, "logistic_lambdas",
                           tuple(float(v) for v in self.logistic_lambdas))
        if any(v < 0 for v in self.elastic_net_l2):
            raise ConfigError("linear.elastic_net_l2: every value >= 0")


@dataclass(frozen=True, slots=True)
class PCAConfig:
    components: tuple[int, ...] = (5, 10, 20, 30)
    sparse_pca: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", tuple(int(v) for v in self.components))


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    candidate_set_sizes: tuple[int, ...] = (5, 10, 20, 30, 50, 75)
    permutation_repeats: int = 5
    permutation_block_bars: int = 288
    plateau: dict[str, float] = field(default_factory=lambda: {"minimal": 0.9,
                                                              "standard": 0.97})
    min_meaningful_ic: float = 0.02

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_set_sizes",
                           tuple(sorted(int(v) for v in self.candidate_set_sizes)))


@dataclass(frozen=True, slots=True)
class ManifestConfig:
    version: int = 1
    families_order: tuple[str, ...] = ("returns", "volatility", "autocorrelation", "regression",
                                       "ou", "fft", "wavelet", "regime", "microstructure",
                                       "time", "interaction")

    def __post_init__(self) -> None:
        object.__setattr__(self, "families_order", tuple(str(v) for v in self.families_order))


@dataclass(frozen=True, slots=True)
class FeatureSelectionConfig:
    schema_version: str
    seed: int
    timeframes: tuple[str, ...]
    periods: PeriodsConfig
    universe: UniverseConfig
    filters: FiltersConfig
    null_screen: NullScreenConfig
    targets: TargetsConfig
    redundancy: RedundancyConfig
    representative: RepresentativeConfig
    mrmr: MRMRConfig
    stability: StabilityConfig
    linear: LinearConfig
    pca: PCAConfig
    evaluation: EvaluationConfig
    drift_classes: tuple[str, ...]
    manifest: ManifestConfig
    ledger_path: Path
    results_path: Path
    project_root: Path
    config_path: Path

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = _as_dict(self)
        return out

    def fingerprint(self) -> str:
        payload = self.to_dict()
        for key in ("project_root", "config_path", "ledger_path", "results_path"):
            payload.pop(key, None)
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        return hashlib.blake2b(blob, digest_size=8).hexdigest()


def default_selection_config_path(root: Path | None = None) -> Path:
    root = root or find_project_root()
    return resolve_path(os.environ.get(SELECTION_CONFIG_ENV_VAR) or DEFAULT_SELECTION_RELPATH,
                        root)


_TOP = ("schema_version", "seed", "timeframes", "periods", "selection_universe", "filters",
        "null_screen", "targets", "redundancy", "representative", "mrmr",
        "stability_selection", "linear", "pca", "evaluation", "drift_classes", "manifest",
        "ledger", "storage")


def _periods(raw: dict[str, Any]) -> PeriodsConfig:
    def period(name: str, *, open_end: bool = False) -> Period:
        spec = raw.get(name) or {}
        start = _day(spec.get("start"), f"periods.{name}.start")
        end = _day(spec.get("end"), f"periods.{name}.end")
        if start is None or (end is None and not open_end):
            raise ConfigError(f"periods.{name}: needs start and end")
        if end is not None and end <= start:
            raise ConfigError(f"periods.{name}: end must follow start")
        return Period(name=name, start=start, end=end)

    dev = period("development")
    val = period("validation")
    test = period("reserved_test", open_end=True)
    recent = period("recent_development")
    if not (dev.end == val.start and val.end == test.start):
        raise ConfigError("periods: development, validation and reserved_test must be "
                          "consecutive and non-overlapping")
    if not (dev.start <= recent.start and recent.end is not None and dev.end is not None
            and recent.end <= dev.end):
        raise ConfigError("periods.recent_development: must lie inside development")
    folds = []
    for i, f in enumerate(raw.get("folds") or ()):
        vs = _day(f.get("validate_start"), f"periods.folds[{i}].validate_start")
        ve = _day(f.get("validate_end"), f"periods.folds[{i}].validate_end")
        if vs is None or ve is None or not (dev.start < vs < ve) or dev.end is None \
                or ve > dev.end:
            raise ConfigError(f"periods.folds[{i}]: must lie inside development, after its start")
        folds.append(Fold(validate_start=vs, validate_end=ve))
    return PeriodsConfig(development=dev, validation=val, reserved_test=test,
                         recent_development=recent, folds=tuple(folds),
                         purge_bars=int(raw.get("purge_bars", 50)),
                         embargo_bars=int(raw.get("embargo_bars", 0)))


def load_selection_config(path: str | os.PathLike[str] | None = None, *,
                          root: Path | None = None) -> FeatureSelectionConfig:
    root = root or find_project_root()
    cfg_path = (resolve_path(path, root) if path is not None
                else default_selection_config_path(root))
    if not cfg_path.exists():
        raise ConfigError(f"Feature selection configuration file not found: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{cfg_path}: top level must be a mapping")
    data: dict[str, Any] = _expand(raw)
    unknown = sorted(set(data) - set(_TOP))
    if unknown:
        raise ConfigError(f"{cfg_path}: unknown top-level key(s) {unknown}")
    periods = _periods(data.get("periods") or {})
    targets = _build(TargetsConfig, data.get("targets"), "targets")
    purge_needed = max(targets.horizons)
    if periods.purge_bars < purge_needed:
        raise ConfigError(f"periods.purge_bars ({periods.purge_bars}) must be at least the "
                          f"longest target horizon ({purge_needed})")
    ledger = data.get("ledger") or {}
    storage = data.get("storage") or {}
    return FeatureSelectionConfig(
        schema_version=str(data.get("schema_version", "1.0.0")),
        seed=int(data.get("seed", 20261001)),
        timeframes=tuple(str(t) for t in data.get("timeframes") or ("5m", "15m")),
        periods=periods,
        universe=_build(UniverseConfig, data.get("selection_universe"), "selection_universe"),
        filters=_build(FiltersConfig, data.get("filters"), "filters"),
        null_screen=_build(NullScreenConfig, data.get("null_screen"), "null_screen"),
        targets=targets,
        redundancy=_build(RedundancyConfig, data.get("redundancy"), "redundancy"),
        representative=_build(RepresentativeConfig, data.get("representative"),
                              "representative"),
        mrmr=_build(MRMRConfig, data.get("mrmr"), "mrmr"),
        stability=_build(StabilityConfig, data.get("stability_selection"),
                         "stability_selection"),
        linear=_build(LinearConfig, data.get("linear"), "linear"),
        pca=_build(PCAConfig, data.get("pca"), "pca"),
        evaluation=_build(EvaluationConfig, data.get("evaluation"), "evaluation"),
        drift_classes=tuple(str(v) for v in data.get("drift_classes") or ()),
        manifest=_build(ManifestConfig, data.get("manifest"), "manifest"),
        ledger_path=resolve_path(ledger.get("path") or "results/research_ledger.parquet", root),
        results_path=resolve_path(storage.get("results_path") or "results/feature_selection",
                                  root),
        project_root=root, config_path=cfg_path)
