"""Typed configuration of the ensemble research (``config/ensemble.yaml``, Prompt #11)."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..utils.config import ConfigError, _as_dict, _expand
from ..utils.paths import find_project_root, resolve_path

__all__ = ["EnsembleConfig", "default_ensemble_config_path", "load_ensemble_config"]

DEFAULT_ENSEMBLE_RELPATH = "config/ensemble.yaml"
ENSEMBLE_CONFIG_ENV_VAR = "XAUUSD_ENSEMBLE_CONFIG"

#: methods in the order of Step 84 (simplest first); research variants are not here
METHODS = ("best_individual", "simple_average", "median", "performance_weighted",
           "diversity_weighted", "stacking", "regime_conditioned", "volatility_conditioned",
           "dynamic")


@dataclass(frozen=True, slots=True)
class EnsembleConfig:
    schema_version: str
    random_seed: int
    timeframes: tuple[str, ...]
    primary_timeframe: str
    pairs: tuple[tuple[str, int], ...]
    calibrated_only_for_classification: bool
    eligibility: dict[str, Any]
    universe: dict[str, Any]
    evaluation_blocks: tuple[int, ...]
    embargo_bars: int
    recent_block: int
    methods: dict[str, bool]
    weighting: dict[str, Any]
    diversity: dict[str, Any]
    stacking: dict[str, Any]
    conditional: dict[str, Any]
    dynamic: dict[str, Any]
    calibration: dict[str, Any]
    size: dict[str, Any]
    ensembles_by_family: dict[str, tuple[str, ...]]
    feature_set_family: str
    family_groups: dict[str, tuple[str, ...]]
    cross_horizon: dict[str, dict[str, Any]]
    alpha_decay_horizons: tuple[int, ...]
    stability: dict[str, Any]
    failure: dict[str, Any]
    controls: dict[str, bool]
    freeze: dict[str, Any]
    inference: dict[str, Any]
    streaming: dict[str, Any]
    registry_version: int
    ledger_path: Path
    results_path: Path
    predictions_path: Path
    models_path: Path
    registry_path: Path
    project_root: Path
    config_path: Path

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = _as_dict(self)
        return out

    def fingerprint(self) -> str:
        payload = self.to_dict()
        for key in ("project_root", "config_path", "ledger_path", "results_path",
                    "predictions_path", "models_path", "registry_path"):
            payload.pop(key, None)
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        return hashlib.blake2b(blob, digest_size=8).hexdigest()

    def enabled(self, method: str) -> bool:
        return bool(self.methods.get(method, False))

    @property
    def minimum_model_count(self) -> int:
        return int(self.universe.get("minimum_model_count", 2))

    @property
    def maximum_model_count(self) -> int:
        return int(self.universe.get("maximum_model_count", 8))


_TOP = ("schema_version", "random_seed", "timeframes", "primary_timeframe", "pairs",
        "prediction_inputs", "eligibility", "universe", "meta_walk_forward", "methods",
        "weighting", "diversity", "stacking", "conditional", "dynamic", "calibration", "size",
        "ensembles_by_family", "feature_set_family", "family_groups", "cross_horizon",
        "alpha_decay_horizons", "stability", "failure", "controls", "freeze", "inference",
        "streaming", "registry", "ledger", "storage")


def default_ensemble_config_path(root: Path | None = None) -> Path:
    root = root or find_project_root()
    return resolve_path(os.environ.get(ENSEMBLE_CONFIG_ENV_VAR) or DEFAULT_ENSEMBLE_RELPATH, root)


def _shares(values: Any, where: str) -> tuple[float, ...]:
    out = tuple(float(v) for v in values or ())
    if any(not 0.0 <= v <= 1.0 for v in out):
        raise ConfigError(f"{where}: values must lie in [0, 1]")
    return out


def load_ensemble_config(path: str | os.PathLike[str] | None = None, *,
                         root: Path | None = None) -> EnsembleConfig:
    root = root or find_project_root()
    cfg_path = (resolve_path(path, root) if path is not None
                else default_ensemble_config_path(root))
    if not cfg_path.exists():
        raise ConfigError(f"ensemble configuration file not found: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{cfg_path}: top level must be a mapping")
    data: dict[str, Any] = _expand(raw)
    unknown = sorted(set(data) - set(_TOP))
    if unknown:
        raise ConfigError(f"{cfg_path}: unknown top-level key(s) {unknown}")
    pairs = []
    for i, pair in enumerate(data.get("pairs") or ()):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ConfigError(f"pairs[{i}]: expected [target, horizon]")
        pairs.append((str(pair[0]), int(pair[1])))
    if not pairs:
        raise ConfigError("pairs: at least one target x horizon is required")
    if len(set(pairs)) != len(pairs):
        raise ConfigError("pairs: duplicates")
    mwf = data.get("meta_walk_forward") or {}
    blocks = tuple(int(b) for b in mwf.get("evaluation_blocks") or (1, 2, 3, 4))
    if not blocks or min(blocks) < 1 or list(blocks) != sorted(set(blocks)):
        raise ConfigError("meta_walk_forward.evaluation_blocks: ascending, >= 1 (block 0 has no "
                          "earlier out-of-sample rows to fit on)")
    methods = {str(k): bool(v) for k, v in (data.get("methods") or {}).items()}
    weighting = dict(data.get("weighting") or {})
    _shares(weighting.get("shrink"), "weighting.shrink")
    diversity = dict(data.get("diversity") or {})
    _shares(diversity.get("lambdas"), "diversity.lambdas")
    freeze = dict(data.get("freeze") or {})
    order = [str(m) for m in freeze.get("order") or METHODS]
    bad = sorted(set(order) - set(METHODS))
    if bad:
        raise ConfigError(f"freeze.order: unknown methods {bad}")
    if order[0] != "best_individual":
        raise ConfigError("freeze.order must start with best_individual (Step 84)")
    freeze["order"] = order
    inference = dict(data.get("inference") or {})
    policy = str(inference.get("missing_constituent_policy", "fail_closed"))
    if policy != "fail_closed":
        raise ConfigError("inference.missing_constituent_policy: only fail_closed is validated "
                          "(Step 68)")
    storage = data.get("storage") or {}
    ledger = data.get("ledger") or {}
    registry = data.get("registry") or {}
    return EnsembleConfig(
        schema_version=str(data.get("schema_version", "1.0.0")),
        random_seed=int(data.get("random_seed", 20261011)),
        timeframes=tuple(str(t) for t in data.get("timeframes") or ("5m",)),
        primary_timeframe=str(data.get("primary_timeframe", "5m")),
        pairs=tuple(pairs),
        calibrated_only_for_classification=bool((data.get("prediction_inputs") or {}).get(
            "calibrated_only_for_classification", True)),
        eligibility=dict(data.get("eligibility") or {}),
        universe=dict(data.get("universe") or {}),
        evaluation_blocks=blocks,
        embargo_bars=int(mwf.get("embargo_bars", 12)),
        recent_block=int(mwf.get("recent_block", max(blocks))),
        methods=methods, weighting=weighting, diversity=diversity,
        stacking=dict(data.get("stacking") or {}),
        conditional=dict(data.get("conditional") or {}),
        dynamic=dict(data.get("dynamic") or {}),
        calibration=dict(data.get("calibration") or {}),
        size=dict(data.get("size") or {}),
        ensembles_by_family={str(k): tuple(str(m) for m in v)
                             for k, v in (data.get("ensembles_by_family") or {}).items()},
        feature_set_family=str(data.get("feature_set_family", "lightgbm")),
        family_groups={str(k): tuple(str(m) for m in v)
                       for k, v in (data.get("family_groups") or {}).items()},
        cross_horizon={str(k): dict(v or {}) for k, v in (data.get("cross_horizon") or {}).items()},
        alpha_decay_horizons=tuple(int(h) for h in data.get("alpha_decay_horizons")
                                   or (1, 5, 20)),
        stability=dict(data.get("stability") or {}),
        failure=dict(data.get("failure") or {}),
        controls={str(k): bool(v) for k, v in (data.get("controls") or {}).items()},
        freeze=freeze, inference=inference,
        streaming=dict(data.get("streaming") or {}),
        registry_version=int(registry.get("version", 1)),
        ledger_path=resolve_path(ledger.get("path") or "results/research_ledger.parquet", root),
        results_path=resolve_path(storage.get("results_path") or "results/ensemble_research",
                                  root),
        predictions_path=resolve_path(storage.get("predictions_path") or "results/ml_research",
                                      root),
        models_path=resolve_path(storage.get("models_path") or "data/models", root),
        registry_path=resolve_path(storage.get("registry_path")
                                   or "config/ensemble_registry.yaml", root),
        project_root=root, config_path=cfg_path)
