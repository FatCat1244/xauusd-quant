"""Typed configuration for the regime-discovery layer (Prompt #7).

Same contract as the other research layers: frozen dataclasses, unknown keys
are an error, and the settings hash to a fingerprint recorded in every report,
every fitted model and every stored regime feature set.

``config/regimes.yaml`` governs only regime discovery. The inputs keep their
own definitions: the residual (``config/regression.yaml``), the OU fit
(``config/ou.yaml``), the stored FFT (``config/spectral.yaml``) and wavelet
(``config/wavelet.yaml``) feature sets, and sessions (``config/research.yaml``).
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
    "MODEL_NAMES",
    "OFFLINE_LABEL",
    "RegimeConfig",
    "default_regime_config_path",
    "load_regime_config",
]

DEFAULT_REGIME_RELPATH = "config/regimes.yaml"
REGIME_CONFIG_ENV_VAR = "XAUUSD_REGIME_CONFIG"

#: Stamped on every full-sample fit, smoothed probability and Viterbi path.
OFFLINE_LABEL = "OFFLINE / NON-CAUSAL RESEARCH ONLY"

#: The model family, as the CLI and the results tree name it.
MODEL_NAMES: tuple[str, ...] = ("kmeans", "gmm", "gmm_diag", "hmm")

_SCALINGS = ("robust", "standard")
_MISSING = ("marginalize", "exclude", "median_fill")
_COVARIANCES = ("full", "diag", "tied")
_SCHEMES = ("expanding", "rolling")
_REFITS = ("monthly", "quarterly")


def _ints(values: Any, where: str, *, minimum: int = 1) -> tuple[int, ...]:
    out = tuple(int(v) for v in values)
    if any(v < minimum for v in out):
        raise ConfigError(f"{where}: every value must be >= {minimum}")
    return out


def _states(values: Any, where: str) -> tuple[int, ...]:
    out = _ints(values, where, minimum=2)
    if any(v > 12 for v in out):
        raise ConfigError(f"{where}: at most 12 states (Prompt #7 studies K = 2..6)")
    return tuple(sorted(set(out)))


@dataclass(frozen=True, slots=True)
class InputsConfig:
    regression_window: int = 128
    ou_window: int = 256
    fft_series: str = "log_return"
    fft_window: int = 256
    wavelet_series: str = "log_return"
    wavelet_window: int = 512
    rv_window: int = 20
    return_horizon: int = 20
    abs_return_acf_window: int = 256
    percentile_days: int = 20
    trading_hours_per_day: float = 23.0

    def __post_init__(self) -> None:
        for name in ("regression_window", "ou_window", "fft_window", "wavelet_window",
                     "rv_window", "return_horizon", "abs_return_acf_window", "percentile_days"):
            if int(getattr(self, name)) < 2:
                raise ConfigError(f"inputs.{name}: must be >= 2")
        if not 1.0 <= float(self.trading_hours_per_day) <= 24.0:
            raise ConfigError("inputs.trading_hours_per_day: must be in [1, 24]")

    def bars_per_day(self, bar_seconds: float) -> int:
        return max(1, int(round(float(self.trading_hours_per_day) * 3600.0 / bar_seconds)))

    def percentile_window(self, bar_seconds: float) -> int:
        """Bars in the trailing percentile window (``percentile_days`` trading days)."""
        return self.percentile_days * self.bars_per_day(bar_seconds)


@dataclass(frozen=True, slots=True)
class RedundancyConfig:
    duplicate_abs_spearman: float = 0.98
    flag_abs_spearman: float = 0.90
    sample_rows: int = 200_000

    def __post_init__(self) -> None:
        if not 0 < self.flag_abs_spearman <= self.duplicate_abs_spearman <= 1:
            raise ConfigError("redundancy: need 0 < flag_abs_spearman <= "
                              "duplicate_abs_spearman <= 1")


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    scaling: str = "robust"
    compare_scaling: tuple[str, ...] = ("robust", "standard")
    clip: float | None = 10.0
    missing_policy: str = "marginalize"
    min_feature_coverage: float = 0.75
    required_features: tuple[str, ...] = ("log_rv_20",)

    def __post_init__(self) -> None:
        object.__setattr__(self, "compare_scaling", tuple(self.compare_scaling))
        object.__setattr__(self, "required_features", tuple(self.required_features))
        _one_of(self.scaling, _SCALINGS, "preprocessing.scaling")
        for name in self.compare_scaling:
            _one_of(name, _SCALINGS, "preprocessing.compare_scaling")
        _one_of(self.missing_policy, _MISSING, "preprocessing.missing_policy")
        if self.clip is not None and float(self.clip) <= 0:
            raise ConfigError("preprocessing.clip: must be > 0 or null")
        if not 0 < self.min_feature_coverage <= 1:
            raise ConfigError("preprocessing.min_feature_coverage: must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class KMeansConfig:
    enabled: bool = True
    clusters: tuple[int, ...] = (2, 3, 4, 5, 6)
    n_init: int = 5
    max_iter: int = 300
    tol: float = 1e-6

    def __post_init__(self) -> None:
        object.__setattr__(self, "tol", float(self.tol))
        object.__setattr__(self, "clusters", _states(self.clusters, "models.kmeans.clusters"))
        if self.n_init < 1 or self.max_iter < 1:
            raise ConfigError("models.kmeans: n_init and max_iter must be >= 1")


@dataclass(frozen=True, slots=True)
class GMMConfig:
    enabled: bool = True
    components: tuple[int, ...] = (2, 3, 4, 5, 6)
    covariance_types: tuple[str, ...] = ("full", "diag")
    n_init: int = 3
    max_iter: int = 300
    tol: float = 1e-5
    reg_covar: float = 1e-4

    def __post_init__(self) -> None:
        object.__setattr__(self, "tol", float(self.tol))
        object.__setattr__(self, "reg_covar", float(self.reg_covar))
        object.__setattr__(self, "components",
                           _states(self.components, "models.gmm.components"))
        object.__setattr__(self, "covariance_types", tuple(self.covariance_types))
        for name in self.covariance_types:
            _one_of(name, _COVARIANCES, "models.gmm.covariance_types")
        if self.n_init < 1 or self.max_iter < 1 or self.reg_covar <= 0:
            raise ConfigError("models.gmm: n_init, max_iter >= 1 and reg_covar > 0")


@dataclass(frozen=True, slots=True)
class HMMConfig:
    enabled: bool = True
    states: tuple[int, ...] = (2, 3, 4, 5, 6)
    covariance_type: str = "full"
    n_init: int = 3
    max_iter: int = 200
    tol: float = 1e-5
    reg_covar: float = 1e-4
    transition_pseudocount: float = 1.0
    block_length: int = 512

    def __post_init__(self) -> None:
        for name in ("tol", "reg_covar", "transition_pseudocount"):
            object.__setattr__(self, name, float(getattr(self, name)))
        object.__setattr__(self, "states", _states(self.states, "models.hmm.states"))
        _one_of(self.covariance_type, _COVARIANCES, "models.hmm.covariance_type")
        if self.n_init < 1 or self.max_iter < 1 or self.reg_covar <= 0:
            raise ConfigError("models.hmm: n_init, max_iter >= 1 and reg_covar > 0")
        if self.transition_pseudocount < 0:
            raise ConfigError("models.hmm.transition_pseudocount: must be >= 0")
        if self.block_length < 16:
            raise ConfigError("models.hmm.block_length: must be >= 16")


@dataclass(frozen=True, slots=True)
class ModelsConfig:
    kmeans: KMeansConfig = field(default_factory=KMeansConfig)
    gmm: GMMConfig = field(default_factory=GMMConfig)
    hmm: HMMConfig = field(default_factory=HMMConfig)


@dataclass(frozen=True, slots=True)
class OfflineConfig:
    enabled: bool = True
    silhouette_sample: int = 20_000
    profile_quantiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)
    timeline_slices: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_quantiles",
                           tuple(float(q) for q in self.profile_quantiles))
        if any(not 0 < q < 1 for q in self.profile_quantiles):
            raise ConfigError("offline.profile_quantiles: every value must be in (0, 1)")
        if self.silhouette_sample < 100:
            raise ConfigError("offline.silhouette_sample: must be >= 100")


@dataclass(frozen=True, slots=True)
class CausalConfig:
    schemes: tuple[str, ...] = ("expanding", "rolling")
    primary_scheme: str = "expanding"
    rolling_training_years: int = 5
    refit_frequencies: tuple[str, ...] = ("quarterly", "monthly")
    primary_refit: str = "quarterly"
    first_inference: str = "2008-01-01"
    min_training_observations: int = 20_000
    burn_in_bars: int = 2_000
    warm_start: bool = True
    fresh_inits_per_refit: int = 1
    fresh_init_every: int = 8
    refit_max_iter: int = 100
    refit_tol: float = 1e-4
    monthly_states: tuple[int, ...] = (3,)
    secondary_scheme_models: tuple[str, ...] = ("hmm",)

    def __post_init__(self) -> None:
        object.__setattr__(self, "refit_tol", float(self.refit_tol))
        object.__setattr__(self, "schemes", tuple(self.schemes))
        object.__setattr__(self, "refit_frequencies", tuple(self.refit_frequencies))
        object.__setattr__(self, "secondary_scheme_models", tuple(self.secondary_scheme_models))
        for name in self.secondary_scheme_models:
            _one_of(name, MODEL_NAMES, "causal.secondary_scheme_models")
        object.__setattr__(self, "monthly_states", _states(self.monthly_states,
                                                           "causal.monthly_states"))
        for name in self.schemes:
            _one_of(name, _SCHEMES, "causal.schemes")
        for name in self.refit_frequencies:
            _one_of(name, _REFITS, "causal.refit_frequencies")
        _one_of(self.primary_scheme, self.schemes, "causal.primary_scheme")
        _one_of(self.primary_refit, self.refit_frequencies, "causal.primary_refit")
        if self.rolling_training_years < 1:
            raise ConfigError("causal.rolling_training_years: must be >= 1")
        if self.burn_in_bars < 0 or self.min_training_observations < 100:
            raise ConfigError("causal: burn_in_bars >= 0, min_training_observations >= 100")
        if self.fresh_inits_per_refit < 0 or (self.fresh_inits_per_refit == 0
                                              and not self.warm_start):
            raise ConfigError("causal: a refit needs a warm start or a fresh initialisation")
        if self.fresh_init_every < 1 or self.refit_max_iter < 1 or self.refit_tol <= 0:
            raise ConfigError("causal: fresh_init_every, refit_max_iter >= 1, refit_tol > 0")


@dataclass(frozen=True, slots=True)
class BaselinesConfig:
    volatility_feature: str = "log_rv_20"
    random_label_seed: int = 20261002


@dataclass(frozen=True, slots=True)
class NullsConfig:
    enabled: bool = True
    states: tuple[int, ...] = (2, 3, 4, 5, 6)
    row_shuffle: bool = True
    row_block_days: tuple[int, ...] = (1, 5)
    pipeline: tuple[str, ...] = ("random_walk", "shuffled_returns", "block_bootstrap_1024")
    seed: int = 20260925
    max_rows: int = 400_000
    increment_models: tuple[str, ...] = ("hmm",)
    increment_states: tuple[int, ...] = (2, 3, 4, 5, 6)
    run_states: tuple[int, ...] | None = None     # K actually run (None = every K in states)

    def __post_init__(self) -> None:
        object.__setattr__(self, "states", _states(self.states, "nulls.states"))
        if self.run_states is not None:
            object.__setattr__(self, "run_states", _states(self.run_states, "nulls.run_states"))
        object.__setattr__(self, "increment_models", tuple(self.increment_models))
        object.__setattr__(self, "increment_states", _states(self.increment_states,
                                                             "nulls.increment_states"))
        for name in self.increment_models:
            _one_of(name, MODEL_NAMES, "nulls.increment_models")
        object.__setattr__(self, "row_block_days", _ints(self.row_block_days,
                                                         "nulls.row_block_days"))
        object.__setattr__(self, "pipeline", tuple(self.pipeline))
        for name in self.pipeline:
            if name not in ("random_walk", "shuffled_returns", "block_bootstrap") and not (
                    name.startswith("block_bootstrap_") and name[16:].isdigit()):
                raise ConfigError(f"nulls.pipeline: unknown control {name!r}")
        if self.max_rows < 10_000:
            raise ConfigError("nulls.max_rows: must be >= 10000")


@dataclass(frozen=True, slots=True)
class SyntheticConfig:
    bars: int = 60_000
    seed: int = 20261003

    def __post_init__(self) -> None:
        if self.bars < 5_000:
            raise ConfigError("synthetic.bars: must be >= 5000")


@dataclass(frozen=True, slots=True)
class OutcomesConfig:
    horizons: tuple[int, ...] = (1, 2, 3, 5, 10, 20)
    residual_extreme: float = 2.0
    ou_extreme: float = 2.0
    volatility_buckets: int = 3
    min_observations: int = 200

    def __post_init__(self) -> None:
        object.__setattr__(self, "horizons", _ints(self.horizons, "outcomes.horizons"))
        if self.volatility_buckets < 2:
            raise ConfigError("outcomes.volatility_buckets: must be >= 2")


@dataclass(frozen=True, slots=True)
class IncrementalConfig:
    enabled: bool = True
    targets: tuple[str, ...] = ("future_volatility", "future_abs_residual_reduction",
                                "future_return", "future_innovation_magnitude")
    binary_targets: tuple[str, ...] = ("residual_shrinks",)
    horizons: tuple[int, ...] = (1, 5, 20)
    test_blocks: tuple[tuple[int, int], ...] = ((2011, 2014), (2015, 2018), (2019, 2022),
                                                (2023, 2026))
    max_rows: int = 400_000
    ridge: float = 1.0
    binary_extreme: float = 2.0
    min_train_rows: int = 2_000
    volatility_windows: tuple[int, ...] = (5, 20, 64, 256, 1024)
    hours: bool = True

    def __post_init__(self) -> None:
        from ..research.spectral_ic import TARGETS

        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "binary_targets", tuple(self.binary_targets))
        object.__setattr__(self, "horizons", _ints(self.horizons, "incremental.horizons"))
        object.__setattr__(self, "volatility_windows",
                           _ints(self.volatility_windows, "incremental.volatility_windows",
                                 minimum=2))
        blocks = tuple((int(a), int(b)) for a, b in self.test_blocks)
        object.__setattr__(self, "test_blocks", blocks)
        for name in self.targets:
            _one_of(name, TARGETS, "incremental.targets")
        for name in self.binary_targets:
            _one_of(name, ("residual_shrinks",), "incremental.binary_targets")
        if any(a > b for a, b in blocks) or any(
                blocks[i][1] >= blocks[i + 1][0] for i in range(len(blocks) - 1)):
            raise ConfigError("incremental.test_blocks: increasing, non-overlapping "
                              "[first_year, last_year] pairs")
        if self.ridge < 0:
            raise ConfigError("incremental.ridge: must be >= 0")


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    bins: int = 10

    def __post_init__(self) -> None:
        if self.bins < 3:
            raise ConfigError("calibration.bins: must be >= 3")


@dataclass(frozen=True, slots=True)
class StabilityConfig:
    eras: int = 3
    change_quantiles: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)

    def __post_init__(self) -> None:
        object.__setattr__(self, "change_quantiles",
                           tuple(float(q) for q in self.change_quantiles))


@dataclass(frozen=True, slots=True)
class ChangepointsConfig:
    enabled: bool = True
    series: tuple[str, ...] = ("log_rv_20",)
    penalty_multiplier: float = 1.0
    min_segment_days: int = 20

    def __post_init__(self) -> None:
        object.__setattr__(self, "series", tuple(self.series))
        if self.penalty_multiplier <= 0 or self.min_segment_days < 2:
            raise ConfigError("changepoints: penalty_multiplier > 0, min_segment_days >= 2")


@dataclass(frozen=True, slots=True)
class CrossTimeframeConfig:
    pairs: tuple[tuple[str, str], ...] = (("5m", "1h"), ("15m", "1h"), ("5m", "15m"),
                                          ("30m", "1h"))

    def __post_init__(self) -> None:
        object.__setattr__(self, "pairs", tuple((str(a), str(b)) for a, b in self.pairs))


@dataclass(frozen=True, slots=True)
class DiagnosticsConfig:
    min_state_fraction: float = 0.01
    max_state_fraction: float = 0.90
    max_condition_number: float = 1e8
    duplicate_bhattacharyya: float = 0.05
    min_expected_duration: float = 2.0
    max_expected_duration: float = 100_000.0

    def __post_init__(self) -> None:
        for name in ("min_state_fraction", "max_state_fraction", "max_condition_number",
                     "duplicate_bhattacharyya", "min_expected_duration", "max_expected_duration"):
            object.__setattr__(self, name, float(getattr(self, name)))   # YAML reads 1e8 as text
        if not 0 <= self.min_state_fraction < self.max_state_fraction <= 1:
            raise ConfigError("diagnostics: need 0 <= min_state_fraction < max_state_fraction")


@dataclass(frozen=True, slots=True)
class PathConfig:
    path: str = ""


@dataclass(frozen=True, slots=True)
class FeaturesOutputConfig:
    path: str = "data/features/regime"
    float32: bool = True


@dataclass(frozen=True, slots=True)
class PlotsConfig:
    enabled: bool = True
    format: str = "png"
    dpi: int = 130
    figsize: tuple[float, float] = (11.0, 6.0)
    timeline_bars: int = 1500
    max_series_points: int = 20_000

    def __post_init__(self) -> None:
        object.__setattr__(self, "figsize", tuple(self.figsize))


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
class RegimeConfig:
    """Fully-resolved regime-discovery configuration."""

    project_root: Path
    config_path: Path | None
    schema_version: str
    results_path: Path
    inputs_path: Path
    features_path: Path
    models_path: Path
    ledger_path: Path
    timeframes: tuple[str, ...]
    seed: int
    feature_groups: dict[str, tuple[str, ...]]
    feature_sets: dict[str, tuple[str, ...]]
    primary_feature_set: str
    inputs: InputsConfig
    redundancy: RedundancyConfig
    preprocessing: PreprocessingConfig
    models: ModelsConfig
    offline: OfflineConfig
    causal: CausalConfig
    baselines: BaselinesConfig
    nulls: NullsConfig
    synthetic: SyntheticConfig
    outcomes: OutcomesConfig
    incremental: IncrementalConfig
    calibration: CalibrationConfig
    stability: StabilityConfig
    changepoints: ChangepointsConfig
    cross_timeframe: CrossTimeframeConfig
    diagnostics: DiagnosticsConfig
    features_output: FeaturesOutputConfig
    plots: PlotsConfig
    output: OutputConfig
    grid: GridConfig
    ledger: LedgerConfig

    def __post_init__(self) -> None:
        from .dataset import FEATURE_SPECS

        grouped = [f for members in self.feature_groups.values() for f in members]
        for name in grouped:
            if name not in FEATURE_SPECS:
                raise ConfigError(f"feature_groups: {name!r} is not a registered regime "
                                  f"feature; known: {sorted(FEATURE_SPECS)}")
        if len(set(grouped)) != len(grouped):
            raise ConfigError("feature_groups: a feature belongs to exactly one group")
        for set_name, members in self.feature_sets.items():
            if not members:
                raise ConfigError(f"feature_sets.{set_name}: empty")
            for name in members:
                if name not in FEATURE_SPECS:
                    raise ConfigError(f"feature_sets.{set_name}: {name!r} is not a registered "
                                      "regime feature")
            if len(set(members)) != len(members):
                raise ConfigError(f"feature_sets.{set_name}: duplicate feature")
        _one_of(self.primary_feature_set, tuple(self.feature_sets), "primary_feature_set")
        for name in self.preprocessing.required_features:
            if name not in self.feature_sets[self.primary_feature_set]:
                raise ConfigError(f"preprocessing.required_features: {name!r} is not in the "
                                  "primary feature set")
        if self.baselines.volatility_feature not in FEATURE_SPECS:
            raise ConfigError("baselines.volatility_feature: not a registered regime feature")

    # -- feature sets ---------------------------------------------------------
    def features(self, feature_set: str | None = None) -> tuple[str, ...]:
        name = feature_set or self.primary_feature_set
        if name not in self.feature_sets:
            raise KeyError(f"unknown feature set {name!r}; known: {sorted(self.feature_sets)}")
        return self.feature_sets[name]

    def group_of(self, feature: str) -> str | None:
        for group, members in self.feature_groups.items():
            if feature in members:
                return group
        return None

    # -- models ----------------------------------------------------------------
    def states_for(self, model: str) -> tuple[int, ...]:
        _one_of(model, MODEL_NAMES, "model")
        if model == "kmeans":
            return self.models.kmeans.clusters
        if model in ("gmm", "gmm_diag"):
            return self.models.gmm.components
        return self.models.hmm.states

    def enabled_models(self) -> list[str]:
        out = []
        if self.models.kmeans.enabled:
            out.append("kmeans")
        if self.models.gmm.enabled:
            out += ["gmm" if c == "full" else f"gmm_{c}"
                    for c in self.models.gmm.covariance_types]
        if self.models.hmm.enabled:
            out.append("hmm")
        return out

    # -- paths -----------------------------------------------------------------
    def results_dir(self, timeframe: str, model: str, states: int) -> Path:
        return self.results_path / timeframe / model / f"k{states}"

    def plots_dir(self, timeframe: str, model: str, states: int) -> Path:
        return self.results_dir(timeframe, model, states) / "plots"

    def timeframe_dir(self, timeframe: str) -> Path:
        return self.results_path / timeframe

    def to_dict(self) -> dict[str, Any]:
        return _as_dict(self)

    def fingerprint(self) -> str:
        """Stable digest of the settings that affect output.

        Options that only choose *which* runs happen (the monthly K, the models
        with secondary schemes, where pipeline-null increments are measured) are
        left out: they never change a stored value, so they never invalidate one.
        """
        payload = self.to_dict()
        for key in ("project_root", "config_path", "results_path", "inputs_path",
                    "features_path", "models_path", "ledger_path", "plots", "output", "grid",
                    "ledger"):
            payload.pop(key, None)
        for section, keys in (("causal", ("monthly_states", "secondary_scheme_models")),
                              ("nulls", ("increment_models", "increment_states",
                                         "run_states"))):
            for key in keys:
                payload.get(section, {}).pop(key, None)
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.blake2b(blob, digest_size=8).hexdigest()

    def inputs_fingerprint(self) -> str:
        """Digest of only what changes the per-bar input features (the input cache key)."""
        payload = {"schema_version": self.schema_version, "inputs": _as_dict(self.inputs)}
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.blake2b(blob, digest_size=8).hexdigest()


_SECTIONS: dict[str, type] = {
    "inputs": InputsConfig,
    "redundancy": RedundancyConfig,
    "preprocessing": PreprocessingConfig,
    "offline": OfflineConfig,
    "causal": CausalConfig,
    "baselines": BaselinesConfig,
    "nulls": NullsConfig,
    "synthetic": SyntheticConfig,
    "outcomes": OutcomesConfig,
    "incremental": IncrementalConfig,
    "calibration": CalibrationConfig,
    "stability": StabilityConfig,
    "changepoints": ChangepointsConfig,
    "cross_timeframe": CrossTimeframeConfig,
    "diagnostics": DiagnosticsConfig,
    "features_output": FeaturesOutputConfig,
    "plots": PlotsConfig,
    "output": OutputConfig,
    "grid": GridConfig,
    "ledger": LedgerConfig,
}
_PATH_SECTIONS = ("inputs_cache", "model_store")
_TOP_LEVEL = ("schema_version", "timeframes", "results_path", "seed", "feature_groups",
              "feature_sets", "primary_feature_set")


def default_regime_config_path(root: Path | None = None) -> Path:
    root = root or find_project_root()
    return resolve_path(os.environ.get(REGIME_CONFIG_ENV_VAR) or DEFAULT_REGIME_RELPATH, root)


def _feature_map(raw: Any, where: str) -> dict[str, tuple[str, ...]]:
    if not isinstance(raw, dict) or not raw:
        raise ConfigError(f"{where}: expected a non-empty mapping of name -> feature list")
    return {str(k): tuple(str(v) for v in (values or [])) for k, values in raw.items()}


def load_regime_config(
    path: str | os.PathLike[str] | None = None,
    *,
    root: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> RegimeConfig:
    """Load, expand, validate and resolve ``config/regimes.yaml``.

    *overrides* replace whole top-level keys or sections (a section given here
    is built from exactly what is passed, over the dataclass defaults).
    """
    root = root or find_project_root()
    cfg_path = (resolve_path(path, root) if path is not None
                else default_regime_config_path(root))
    if not cfg_path.exists():
        raise ConfigError(f"Regime configuration file not found: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{cfg_path}: top level must be a mapping")
    data: dict[str, Any] = _expand(raw)
    for key, value in (overrides or {}).items():
        if value is not None:
            data[key] = value

    models_raw = dict(data.pop("models", {}) or {})
    unknown_models = sorted(set(models_raw) - {"kmeans", "gmm", "hmm"})
    if unknown_models:
        raise ConfigError(f"models: unknown model(s) {unknown_models}")
    models = ModelsConfig(
        kmeans=_build(KMeansConfig, models_raw.get("kmeans", {}), "models.kmeans"),
        gmm=_build(GMMConfig, models_raw.get("gmm", {}), "models.gmm"),
        hmm=_build(HMMConfig, models_raw.get("hmm", {}), "models.hmm"),
    )
    sections: dict[str, Any] = {name: _build(cls, data.pop(name, {}), name)
                                for name, cls in _SECTIONS.items()}
    paths = {name: _build(PathConfig, data.pop(name, {}), name) for name in _PATH_SECTIONS}
    top = {key: data.pop(key) for key in list(data) if key in _TOP_LEVEL}
    if data:
        known = sorted({*_SECTIONS, *_TOP_LEVEL, *_PATH_SECTIONS, "models"})
        raise ConfigError(f"{cfg_path}: unknown top-level option(s) {sorted(data)}. Known: {known}")
    features_output: FeaturesOutputConfig = sections["features_output"]
    ledger: LedgerConfig = sections["ledger"]
    return RegimeConfig(
        project_root=root,
        config_path=cfg_path,
        schema_version=str(top.get("schema_version", "1.0.0")),
        results_path=resolve_path(top.get("results_path", "results/regime_research"), root),
        inputs_path=resolve_path(paths["inputs_cache"].path or "data/features/regime_inputs",
                                 root),
        features_path=resolve_path(features_output.path, root),
        models_path=resolve_path(paths["model_store"].path or "data/regime_models", root),
        ledger_path=resolve_path(ledger.path, root),
        timeframes=tuple(top.get("timeframes", ("5m", "15m", "30m", "1h"))),
        seed=int(top.get("seed", 20261001)),
        feature_groups=_feature_map(top.get("feature_groups"), "feature_groups"),
        feature_sets=_feature_map(top.get("feature_sets"), "feature_sets"),
        primary_feature_set=str(top.get("primary_feature_set", "core")),
        models=models,
        **sections,
    )
