"""Typed configuration of the supervised predictive research (``config/ml.yaml``, Prompt #10)."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from ..utils.config import ConfigError, _as_dict, _expand
from ..utils.paths import find_project_root, resolve_path

__all__ = [
    "MLConfig",
    "TargetSpec",
    "WalkForwardConfig",
    "default_ml_config_path",
    "load_ml_config",
]

DEFAULT_ML_RELPATH = "config/ml.yaml"
ML_CONFIG_ENV_VAR = "XAUUSD_ML_CONFIG"

#: model families that exist in :mod:`.models`
MODEL_FAMILIES = ("constant", "logistic_l2", "logistic_l1", "logistic_en", "ols", "ridge",
                  "elastic_net", "random_forest", "xgboost", "lightgbm", "catboost")
CLASSIFIERS = ("constant", "logistic_l2", "logistic_l1", "logistic_en", "random_forest",
               "xgboost", "lightgbm", "catboost")
REGRESSORS = ("constant", "ols", "ridge", "elastic_net", "random_forest", "xgboost",
              "lightgbm", "catboost")
SOURCES = ("residual_shrinks", "residual_shrinks_c", "residual_reduction", "return",
           "up_beyond_cost", "down_beyond_cost", "realized_vol", "abs_return")
TRANSFORMS = ("none", "vol_scaled", "log")
KINDS = ("direction", "reversion", "volatility", "magnitude")


def _day(value: Any, where: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ConfigError(f"{where}: not an ISO date: {value!r}") from exc


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """One prediction problem: what is predicted, how, and at which horizons."""

    name: str
    source: str
    task: str                          # classification | regression
    kind: str                          # direction | reversion | volatility | magnitude
    horizons: tuple[int, ...]
    transform: str = "none"
    c: float | None = None             # residual_shrinks_c
    cost_multiple: float | None = None  # up/down_beyond_cost

    @property
    def is_classification(self) -> bool:
        return self.task == "classification"


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    scheme: str
    validation_blocks: tuple[tuple[date, date], ...]
    rolling_years: int
    embargo_bars: int
    inner_fraction: float
    recent_blocks: int


@dataclass(frozen=True, slots=True)
class MLConfig:
    schema_version: str
    random_seed: int
    timeframes: tuple[str, ...]
    primary_timeframe: str
    walk_forward: WalkForwardConfig
    feature_sets: tuple[str, ...]
    default_feature_set: str
    targets: dict[str, TargetSpec]
    log_floor: float
    primary: dict[str, tuple[int, ...]]
    tree_comparison: dict[str, int]
    focus: tuple[tuple[str, int], ...]
    models: dict[str, dict[str, Any]]
    threads: int
    simplicity_order: tuple[str, ...]
    linear_families: tuple[str, ...]
    tree_families: tuple[str, ...]
    preprocessing: dict[str, Any]
    calibration: dict[str, Any]
    search: dict[str, Any]
    weighting: dict[str, Any]
    learning_curve: dict[str, Any]
    retraining: dict[str, Any]
    decay: dict[str, Any]
    ablation: dict[str, Any]
    nulls: dict[str, Any]
    explain: dict[str, Any]
    streaming: dict[str, Any]
    freeze: dict[str, Any]
    registry_version: int
    ledger_path: Path
    results_path: Path
    models_path: Path
    registry_path: Path                # the generated registry of finalized models (YAML)
    project_root: Path
    config_path: Path

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = _as_dict(self)
        return out

    def fingerprint(self) -> str:
        payload = self.to_dict()
        for key in ("project_root", "config_path", "ledger_path", "results_path",
                    "models_path", "registry_path"):
            payload.pop(key, None)
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        return hashlib.blake2b(blob, digest_size=8).hexdigest()

    def model_params(self, family: str) -> dict[str, Any]:
        params = dict(self.models.get(family) or {})
        params.pop("enabled", None)
        return params

    def enabled_models(self, task: str) -> tuple[str, ...]:
        allowed = CLASSIFIERS if task == "classification" else REGRESSORS
        return tuple(m for m in allowed if (self.models.get(m) or {}).get("enabled", False))

    def primary_pairs(self) -> list[tuple[str, int]]:
        return [(t, h) for t, hs in self.primary.items() for h in hs]

    def comparison_pairs(self) -> list[tuple[str, int]]:
        return [(t, h) for t, h in self.tree_comparison.items()]

    def reference_linear(self, task: str) -> str:
        return "logistic_l2" if task == "classification" else "ridge"

    def all_pairs(self) -> list[tuple[str, int]]:
        return [(t, h) for t, spec in self.targets.items() for h in spec.horizons]


_TOP = ("schema_version", "random_seed", "timeframes", "primary_timeframe", "walk_forward",
        "feature_sets", "default_feature_set", "targets", "log_floor", "primary",
        "tree_comparison", "focus", "models", "threads", "simplicity_order", "linear_families",
        "tree_families", "preprocessing", "calibration", "search", "weighting",
        "learning_curve", "retraining", "decay", "ablation", "nulls", "explain", "streaming",
        "freeze", "registry", "ledger", "storage")


def default_ml_config_path(root: Path | None = None) -> Path:
    root = root or find_project_root()
    return resolve_path(os.environ.get(ML_CONFIG_ENV_VAR) or DEFAULT_ML_RELPATH, root)


def _walk_forward(raw: dict[str, Any]) -> WalkForwardConfig:
    blocks: list[tuple[date, date]] = []
    for i, pair in enumerate(raw.get("validation_blocks") or ()):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ConfigError(f"walk_forward.validation_blocks[{i}]: expected [start, end]")
        start, end = _day(pair[0], f"walk_forward.validation_blocks[{i}]"), \
            _day(pair[1], f"walk_forward.validation_blocks[{i}]")
        if end <= start:
            raise ConfigError(f"walk_forward.validation_blocks[{i}]: end must follow start")
        if blocks and start < blocks[-1][1]:
            raise ConfigError("walk_forward.validation_blocks: blocks must be chronological "
                              "and must not overlap")
        blocks.append((start, end))
    if not blocks:
        raise ConfigError("walk_forward.validation_blocks: at least one block is required")
    scheme = str(raw.get("scheme", "expanding"))
    if scheme not in ("expanding", "rolling"):
        raise ConfigError("walk_forward.scheme: expanding | rolling")
    inner = float(raw.get("inner_fraction", 0.1))
    if not 0.0 < inner < 0.5:
        raise ConfigError("walk_forward.inner_fraction must lie in (0, 0.5)")
    return WalkForwardConfig(scheme=scheme, validation_blocks=tuple(blocks),
                             rolling_years=int(raw.get("rolling_years", 5)),
                             embargo_bars=int(raw.get("embargo_bars", 12)),
                             inner_fraction=inner,
                             recent_blocks=int(raw.get("recent_blocks", 1)))


def _targets(raw: dict[str, Any]) -> dict[str, TargetSpec]:
    out = {}
    for name, spec in (raw or {}).items():
        spec = dict(spec or {})
        source = str(spec.get("source"))
        task = str(spec.get("task"))
        kind = str(spec.get("kind"))
        transform = str(spec.get("transform", "none"))
        if source not in SOURCES:
            raise ConfigError(f"targets.{name}.source: unknown {source!r}")
        if task not in ("classification", "regression"):
            raise ConfigError(f"targets.{name}.task: classification | regression")
        if kind not in KINDS:
            raise ConfigError(f"targets.{name}.kind: one of {KINDS}")
        if transform not in TRANSFORMS:
            raise ConfigError(f"targets.{name}.transform: one of {TRANSFORMS}")
        horizons = tuple(int(h) for h in spec.get("horizons") or ())
        if not horizons or any(h < 1 for h in horizons):
            raise ConfigError(f"targets.{name}.horizons: positive bar counts required")
        c = spec.get("c")
        if source == "residual_shrinks_c" and not (c is not None and 0.0 < float(c) < 1.0):
            raise ConfigError(f"targets.{name}.c must lie in (0, 1)")
        cost = spec.get("cost_multiple")
        if source in ("up_beyond_cost", "down_beyond_cost") and cost is None:
            raise ConfigError(f"targets.{name}.cost_multiple is required")
        out[name] = TargetSpec(name=name, source=source, task=task, kind=kind,
                               horizons=horizons, transform=transform,
                               c=None if c is None else float(c),
                               cost_multiple=None if cost is None else float(cost))
    if not out:
        raise ConfigError("targets: at least one target is required")
    return out


def load_ml_config(path: str | os.PathLike[str] | None = None, *,
                   root: Path | None = None) -> MLConfig:
    root = root or find_project_root()
    cfg_path = resolve_path(path, root) if path is not None else default_ml_config_path(root)
    if not cfg_path.exists():
        raise ConfigError(f"ML configuration file not found: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{cfg_path}: top level must be a mapping")
    data: dict[str, Any] = _expand(raw)
    unknown = sorted(set(data) - set(_TOP))
    if unknown:
        raise ConfigError(f"{cfg_path}: unknown top-level key(s) {unknown}")
    targets = _targets(data.get("targets") or {})
    primary = {str(k): tuple(int(h) for h in v) for k, v in (data.get("primary") or {}).items()}
    for name, hs in primary.items():
        if name not in targets or not set(hs) <= set(targets[name].horizons):
            raise ConfigError(f"primary.{name}: target or horizon not configured")
    comparison = {str(k): int(v) for k, v in (data.get("tree_comparison") or {}).items()}
    for name, h in comparison.items():
        if h not in primary.get(name, ()):
            raise ConfigError(f"tree_comparison.{name}: horizon {h} is not a primary horizon")
    focus = tuple((str(t), int(h)) for t, h in data.get("focus") or ())
    for name, h in focus:
        if h not in primary.get(name, ()):
            raise ConfigError(f"focus: ({name}, {h}) is not a primary pair")
    models = {str(k): dict(v or {}) for k, v in (data.get("models") or {}).items()}
    bad = sorted(set(models) - set(MODEL_FAMILIES))
    if bad:
        raise ConfigError(f"models: unknown families {bad}")
    ledger = data.get("ledger") or {}
    storage = data.get("storage") or {}
    registry = data.get("registry") or {}
    return MLConfig(
        schema_version=str(data.get("schema_version", "1.0.0")),
        random_seed=int(data.get("random_seed", 20261010)),
        timeframes=tuple(str(t) for t in data.get("timeframes") or ("5m",)),
        primary_timeframe=str(data.get("primary_timeframe", "5m")),
        walk_forward=_walk_forward(data.get("walk_forward") or {}),
        feature_sets=tuple(str(s) for s in data.get("feature_sets") or ("standard",)),
        default_feature_set=str(data.get("default_feature_set", "standard")),
        targets=targets,
        log_floor=float(data.get("log_floor", 1e-5)),
        primary=primary,
        tree_comparison=comparison,
        focus=focus,
        models=models,
        threads=int(data.get("threads", 4)),
        simplicity_order=tuple(str(m) for m in data.get("simplicity_order") or MODEL_FAMILIES),
        linear_families=tuple(str(m) for m in data.get("linear_families") or ()),
        tree_families=tuple(str(m) for m in data.get("tree_families") or ()),
        preprocessing=dict(data.get("preprocessing") or {}),
        calibration=dict(data.get("calibration") or {}),
        search=dict(data.get("search") or {}),
        weighting=dict(data.get("weighting") or {}),
        learning_curve=dict(data.get("learning_curve") or {}),
        retraining=dict(data.get("retraining") or {}),
        decay=dict(data.get("decay") or {}),
        ablation=dict(data.get("ablation") or {}),
        nulls=dict(data.get("nulls") or {}),
        explain=dict(data.get("explain") or {}),
        streaming=dict(data.get("streaming") or {}),
        freeze=dict(data.get("freeze") or {}),
        registry_version=int(registry.get("version", 1)),
        ledger_path=resolve_path(ledger.get("path") or "results/research_ledger.parquet", root),
        results_path=resolve_path(storage.get("results_path") or "results/ml_research", root),
        models_path=resolve_path(storage.get("models_path") or "data/models", root),
        registry_path=resolve_path(storage.get("registry_path")
                                   or "config/model_registry.yaml", root),
        project_root=root, config_path=cfg_path)
