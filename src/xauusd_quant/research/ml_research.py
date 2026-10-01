r"""Supervised predictive research, one timeframe at a time (Prompt #10).

The research is a plan of *units* (:class:`~..ml.training.UnitSpec`): target x
horizon x model family x feature set x variant x walk-forward fold. Every unit
writes its out-of-sample predictions (``<fold>.parquet``) and its metrics
(``<fold>.json``) under ``units/<target>/h<h>/<family>/<set>/<variant>/`` the
moment it finishes, stamped with the data versions, the unit spec and the code
of the modules that fit it; a rerun skips current units, so a stopped run loses
one unit.

Stages (Steps in brackets):

``grid``           constant + reference linear model at every target x horizon;
                   the other linear families at the tree-comparison pairs (5-25)
``trees``          LightGBM at every primary pair, all four tree families at the
                   tree-comparison pairs, with SHAP / permutation / ALE (19-23, 34-37)
``feature_sets``   Extended and target-specific manifests vs Standard (30-31)
``variants``       booster NaN handling vs imputation, class weights (16, 43)
``search``         small randomized searches, neighbourhoods, complexity (44-47)
``windows``        rolling vs expanding training, time-decay weights (41, 49)
``learning_curve`` the most recent 20-100 % of the history (48)
``decay``          one model frozen in 2011, scored on every later year (50)
``retraining``     refits every 1 / 3 / 6 / 12 months over 2018-2021 (51)
``ablation``       leave one family out and forward family addition (32-33)
``nulls``          shifted targets, noise columns, permuted features (67-69)
``pipeline_null``  residual targets on a random walk through the same pipeline
                   (invariant 9)
``pipeline_null_posthoc``  the same on the post-hoc nulls of ``nulls.pipeline_posthoc``
                   (``sign_flip``: the real bars in real order with random signs - a
                   martingale with the real volatility path), added 2026-10-01 after
                   the registered plan; recorded and labelled as post-hoc

The aggregation, comparisons, freeze and plots live in :mod:`.ml_reports`.
Nothing here outputs a trade.
"""

from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl

from ..features.config import RegressionConfig
from ..features.factory_config import FeatureFactoryConfig
from ..ml.config import MLConfig, TargetSpec
from ..ml.datasets import MLData, load_ml_data
from ..ml.hyperparameters import complexity_trials, neighbour_trials, search_trials
from ..ml.splits import Fold, first_row_at, refit_schedule, walk_forward_folds, with_fit_start
from ..ml.training import UnitSpec, run_unit
from ..selection.config import FeatureSelectionConfig
from ..targets.config import TargetConfig
from ..utils.clock import utc_now_iso
from ..utils.config import Config
from ..utils.logging import get_logger
from ..utils.paths import ensure_dir
from .feature_research import write_json
from .model_ablation import ablation_sets
from .study_io import clean_json, code_fingerprint

__all__ = ["STAGES", "MLConfigs", "MLContext", "default_explain", "load_context",
           "pending_units", "plan_configuration", "plan_stage", "run_ml_research", "run_units",
           "unit_paths"]

LOGGER = get_logger("research.ml")
STAGES: tuple[str, ...] = ("grid", "trees", "feature_sets", "variants", "search", "windows",
                           "learning_curve", "decay", "retraining", "ablation", "nulls",
                           "pipeline_null", "pipeline_null_posthoc")
PIPELINE_STAGES = ("pipeline_null", "pipeline_null_posthoc")
UNIT_SCHEMA = 1


@dataclass
class MLConfigs:
    ml: MLConfig
    selection: FeatureSelectionConfig
    features: FeatureFactoryConfig
    targets: TargetConfig
    config: Config
    regression: RegressionConfig
    ou: Any
    spectral: Any
    wavelet: Any
    research: Any


@dataclass
class MLContext:
    timeframe: str
    cfgs: MLConfigs
    data: MLData
    out_dir: Path
    progress: Callable[[str], None] | None = None
    null_data: MLData | None = None
    _code: str = ""
    _folds: dict[tuple[int, str, int], list[Fold]] = field(default_factory=dict)

    @property
    def cfg(self) -> MLConfig:
        return self.cfgs.ml

    def log(self, message: str) -> None:
        LOGGER.info("[%s] %s", self.timeframe, message)
        if self.progress is not None:
            self.progress(f"[{self.timeframe}] {message}")

    def code(self) -> str:
        if not self._code:
            base = Path(__file__).resolve().parent.parent / "ml"
            self._code = code_fingerprint(base / f"{m}.py" for m in (
                "datasets", "splits", "preprocessing", "models", "training", "calibration",
                "explainability"))
        return self._code

    def folds(self, horizon: int, scheme: str = "expanding") -> list[Fold]:
        key = (horizon, scheme, self.cfg.walk_forward.rolling_years)
        if key not in self._folds:
            self._folds[key] = walk_forward_folds(self.data.timestamps, self.cfg.walk_forward,
                                                  horizon=horizon, scheme=scheme)
        return self._folds[key]


# ---------------------------------------------------------------------------
# Unit storage
# ---------------------------------------------------------------------------
def variant_tag(variant: dict[str, Any]) -> str:
    if not variant:
        return "base"
    return "__".join(f"{k}-{variant[k]}" for k in sorted(variant)).replace(".", "p")


def unit_paths(out_dir: Path, spec: UnitSpec) -> tuple[Path, Path]:
    d = (out_dir / "units" / spec.target / f"h{spec.horizon}" / spec.family / spec.feature_set
         / variant_tag(spec.variant))
    return d / f"{spec.fold.name}.parquet", d / f"{spec.fold.name}.json"


def _stamp(ctx: MLContext, spec: UnitSpec, data: MLData) -> dict[str, Any]:
    """The unit's stamp in its stored (JSON) form: tuples become lists, NumPy scalars
    Python ones - so a stamp read back from disk compares equal to a fresh one."""
    cfg = ctx.cfg
    stamp = {"schema": UNIT_SCHEMA, "key": spec.key(), "params": spec.params,
             "features": list(spec.features), "seed": spec.seed,
             "fold": [spec.fold.fit, spec.fold.inner, spec.fold.validate],
             "versions": {k: data.versions.get(k) for k in ("factory_version", "target_version",
                                                            "tick_dataset_version")},
             "null_data": data.versions.get("pipeline_null"),
             "preprocessing": cfg.preprocessing, "calibration": cfg.calibration,
             "log_floor": cfg.log_floor, "explain": spec.explain,
             "explain_cfg": cfg.explain if spec.explain else None, "code": ctx.code()}
    out: dict[str, Any] = json.loads(json.dumps(clean_json(stamp), default=str))
    return out


def _current(json_path: Path, stamp: dict[str, Any]) -> bool:
    try:
        return json.loads(json_path.read_text(encoding="utf-8")).get("stamp") == stamp
    except (OSError, ValueError):
        return False


def pending_units(ctx: MLContext, units: list[UnitSpec], data: MLData | None = None
                  ) -> list[tuple[UnitSpec, dict[str, Any], Path, Path]]:
    """The units whose stored result is missing or stale (spec, data, config or code)."""
    data = data or ctx.data
    todo = []
    for spec in units:
        pq_path, js_path = unit_paths(ctx.out_dir, spec)
        stamp = _stamp(ctx, spec, data)
        if not (pq_path.exists() and _current(js_path, stamp)):
            todo.append((spec, stamp, pq_path, js_path))
    return todo


def run_units(ctx: MLContext, units: list[UnitSpec], *, stage: str,
              data: MLData | None = None) -> dict[str, int]:
    data = data or ctx.data
    todo = pending_units(ctx, units, data)
    ctx.log(f"{stage}: {len(units)} units, {len(units) - len(todo)} current, {len(todo)} to run")
    started = time.perf_counter()
    failed = 0
    targets_cache: dict[tuple[str, int], Any] = {}
    for i, (spec, stamp, pq_path, js_path) in enumerate(todo):
        tspec: TargetSpec = ctx.cfg.targets[spec.target]
        key = (spec.target, spec.horizon)
        if key not in targets_cache:
            targets_cache.clear()
            targets_cache[key] = data.target(tspec, spec.horizon, ctx.cfg.log_floor)
        try:
            result = run_unit(data, spec, tspec, ctx.cfg, target=targets_cache[key])
        except (ValueError, MemoryError) as exc:
            failed += 1
            LOGGER.warning("[%s] %s failed: %s", ctx.timeframe, spec.key(), exc)
            write_json(js_path.with_suffix(".failed.json"),
                       {"key": spec.key(), "error": f"{type(exc).__name__}: {exc}",
                        "utc": utc_now_iso()})
            continue
        ensure_dir(pq_path.parent)
        tmp = pq_path.with_name(pq_path.name + ".partial")
        result.predictions.write_parquet(tmp)
        tmp.replace(pq_path)
        write_json(js_path, {"stamp": stamp, "key": spec.key(), "target": spec.target,
                             "horizon": spec.horizon, "family": spec.family,
                             "feature_set": spec.feature_set, "variant": spec.variant,
                             "fold": {"name": spec.fold.name, "index": spec.fold.index,
                                      "fit": list(spec.fold.fit),
                                      "inner": list(spec.fold.inner),
                                      "validate": list(spec.fold.validate),
                                      "scheme": spec.fold.scheme},
                             "params": spec.params, "features": list(spec.features),
                             "metrics": result.metrics, "info": result.info,
                             "explain": result.explain, "utc": utc_now_iso()})
        del result
        if spec.feature_set not in data.manifests:
            # an ablation subset or a target-specific list is cached under its own key;
            # fifteen 60-column subsets at 5m would hold ~4.8 GB until the stage ends
            data.drop_designs()
        if (i + 1) % 5 == 0 or i + 1 == len(todo):
            gc.collect()
            ctx.log(f"{stage}: {i + 1}/{len(todo)} units ({time.perf_counter() - started:.0f}s)")
    return {"units": len(units), "ran": len(todo) - failed, "failed": failed}


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------
def _spec(ctx: MLContext, target: str, h: int, family: str, set_name: str, fold: Fold, *,
          variant: dict[str, Any] | None = None, params: dict[str, Any] | None = None,
          explain: bool = False, features: list[str] | None = None) -> UnitSpec:
    feats = features if features is not None else _set_features(ctx, target, set_name)
    return UnitSpec(target=target, horizon=h, family=family, feature_set=set_name,
                    features=feats, fold=fold,
                    params=params if params is not None else ctx.cfg.model_params(family),
                    variant=dict(variant or {}), seed=ctx.cfg.random_seed, explain=explain)


def _set_features(ctx: MLContext, target: str, set_name: str) -> list[str]:
    if set_name == "target":
        kind = ctx.cfg.targets[target].kind
        return ctx.data.feature_set(f"target_{kind}")
    return ctx.data.feature_set(set_name)


def _families(ctx: MLContext, task: str, which: str) -> list[str]:
    cfg = ctx.cfg
    enabled = set(cfg.enabled_models(task))
    if which == "linear":
        return [m for m in cfg.linear_families if m in enabled]
    return [m for m in cfg.tree_families if m in enabled]


def default_explain(cfg: MLConfig, target: str, h: int, family: str, set_name: str) -> bool:
    """Whether a base unit carries SHAP / permutation / ALE: the tree families and the
    reference linear model on the Standard set of a tree-comparison pair."""
    if set_name != cfg.default_feature_set or (target, h) not in set(cfg.comparison_pairs()):
        return False
    return family in cfg.tree_families or family == cfg.reference_linear(
        cfg.targets[target].task)


def plan_configuration(ctx: MLContext, target: str, h: int, families: list[str],
                       set_name: str, *, fold_indices: list[int] | None = None
                       ) -> list[UnitSpec]:
    """Base units of explicit families on one target x horizon x set (``xq ml-train`` /
    ``xq ml-walk-forward``); the same unit files the research stages write."""
    out = []
    for fold in ctx.folds(h):
        if fold_indices is not None and fold.index not in fold_indices:
            continue
        for fam in families:
            out.append(_spec(ctx, target, h, fam, set_name, fold,
                             explain=default_explain(ctx.cfg, target, h, fam, set_name)))
    return out


def _plan_grid(ctx: MLContext) -> list[UnitSpec]:
    cfg = ctx.cfg
    comparison = set(cfg.comparison_pairs())
    out = []
    for target, h in cfg.all_pairs():
        task = cfg.targets[target].task
        ref = cfg.reference_linear(task)
        fams = ["constant", ref]
        if (target, h) in comparison:
            fams += [m for m in _families(ctx, task, "linear") if m not in fams]
        for fold in ctx.folds(h):
            for fam in fams:
                out.append(_spec(ctx, target, h, fam, cfg.default_feature_set, fold,
                                 explain=default_explain(cfg, target, h, fam,
                                                         cfg.default_feature_set)))
    return out


def _plan_trees(ctx: MLContext) -> list[UnitSpec]:
    cfg = ctx.cfg
    comparison = set(cfg.comparison_pairs())
    out = []
    for target, h in cfg.primary_pairs():
        task = cfg.targets[target].task
        fams = _families(ctx, task, "trees") if (target, h) in comparison else ["lightgbm"]
        for fold in ctx.folds(h):
            for fam in fams:
                out.append(_spec(ctx, target, h, fam, cfg.default_feature_set, fold,
                                 explain=default_explain(cfg, target, h, fam,
                                                         cfg.default_feature_set)))
    return out


def _plan_feature_sets(ctx: MLContext) -> list[UnitSpec]:
    cfg = ctx.cfg
    out = []
    standard = ctx.data.feature_set(cfg.default_feature_set)
    for target, h in cfg.comparison_pairs():
        if target == "mean_reversion_half":
            continue
        task = cfg.targets[target].task
        for set_name in cfg.feature_sets:
            if set_name == cfg.default_feature_set:
                continue
            feats = _set_features(ctx, target, set_name)
            if feats == standard:
                continue                      # identical to Standard (Minimal at 5m)
            for fold in ctx.folds(h):
                for fam in (cfg.reference_linear(task), "lightgbm"):
                    out.append(_spec(ctx, target, h, fam, set_name, fold, features=feats))
    return out


def _plan_variants(ctx: MLContext) -> list[UnitSpec]:
    cfg = ctx.cfg
    out = []
    for fold in ctx.folds(5):
        out.append(_spec(ctx, "mean_reversion", 5, "lightgbm", cfg.default_feature_set, fold,
                         variant={"tree_missing": "imputed"}))
        for fam in ("logistic_l2", "lightgbm"):
            params = {**cfg.model_params(fam), "balanced": True}
            out.append(_spec(ctx, "direction_up_cost", 5, fam, cfg.default_feature_set, fold,
                             variant={"balanced": 1}, params=params))
    return out


def _plan_search(ctx: MLContext) -> list[UnitSpec]:
    cfg = ctx.cfg
    folds_used = [int(i) for i in cfg.search.get("folds", [3, 4])]
    out = []
    for target, h in (tuple(p) for p in cfg.search.get("targets", [])):
        folds = [f for f in ctx.folds(int(h)) if f.index in folds_used]
        for fam in ("lightgbm", "xgboost"):
            for t, trial in enumerate(search_trials(cfg, fam)):
                params = {**cfg.model_params(fam), **trial}
                for fold in folds:
                    out.append(_spec(ctx, target, int(h), fam, cfg.default_feature_set, fold,
                                     variant={"search": f"t{t:02d}"}, params=params))
        for leaves, params in complexity_trials(cfg):         # num_leaves alone
            for fold in folds:
                out.append(_spec(ctx, target, int(h), "lightgbm", cfg.default_feature_set, fold,
                                 variant={"leaves": leaves}, params=params))
    return out


def plan_neighbourhood(ctx: MLContext, best: dict[tuple[str, int], dict[str, Any]]
                       ) -> list[UnitSpec]:
    """LightGBM: one grid step down and up around each target's best search trial."""
    cfg = ctx.cfg
    grid = cfg.search.get("lightgbm") or {}
    folds_used = [int(i) for i in cfg.search.get("folds", [3, 4])]
    out = []
    for (target, h), params in best.items():
        folds = [f for f in ctx.folds(h) if f.index in folds_used]
        for label, trial in neighbour_trials(params, grid, list(cfg.search.get("neighborhood",
                                                                             []))):
            for fold in folds:
                out.append(_spec(ctx, target, h, "lightgbm", cfg.default_feature_set, fold,
                                 variant={"neighbour": label}, params=trial))
    return out


def _plan_windows(ctx: MLContext) -> list[UnitSpec]:
    cfg = ctx.cfg
    out = []
    for target, h in cfg.focus:
        task = cfg.targets[target].task
        fams = (cfg.reference_linear(task), "lightgbm")
        for fam in fams:
            for fold in ctx.folds(h, "rolling"):
                out.append(_spec(ctx, target, h, fam, cfg.default_feature_set, fold,
                                 variant={"window": f"rolling{cfg.walk_forward.rolling_years}"}))
            for hl in cfg.weighting.get("half_lives_years", []):
                for fold in ctx.folds(h):
                    out.append(_spec(ctx, target, h, fam, cfg.default_feature_set, fold,
                                     variant={"weighting": float(hl)}))
    return out


def _plan_learning_curve(ctx: MLContext) -> list[UnitSpec]:
    cfg = ctx.cfg
    out = []
    for target, h in cfg.focus:
        task = cfg.targets[target].task
        last = ctx.folds(h)[-1]
        for frac in cfg.learning_curve.get("fractions", []):
            if float(frac) >= 1.0:
                continue
            lo = int(last.fit[1] - float(frac) * (last.fit[1] - last.fit[0]))
            fold = with_fit_start(last, lo, f"{last.name}_recent{int(round(100 * frac))}")
            for fam in (cfg.reference_linear(task), "lightgbm"):
                out.append(_spec(ctx, target, h, fam, cfg.default_feature_set, fold,
                                 variant={"history": float(frac)}))
    return out


def decay_fold(ctx: MLContext, h: int) -> Fold:
    """Training as the first block's, scored on everything from its start to 2022."""
    first = ctx.folds(h)[0]
    return Fold(index=0, name="decay_from_2011", fit=first.fit, inner=first.inner,
                validate=(first.validate[0], ctx.data.n), horizon=h, embargo=first.embargo,
                scheme="expanding")


def _plan_decay(ctx: MLContext) -> list[UnitSpec]:
    cfg = ctx.cfg
    out = []
    for target, h in cfg.focus:
        task = cfg.targets[target].task
        for fam in (cfg.reference_linear(task), "lightgbm"):
            out.append(_spec(ctx, target, h, fam, cfg.default_feature_set, decay_fold(ctx, h),
                             variant={"decay": "2011"}))
    return out


def _plan_retraining(ctx: MLContext) -> list[UnitSpec]:
    cfg = ctx.cfg
    rt = cfg.retraining
    start, end = (date.fromisoformat(str(d)) for d in rt.get("window", ["2018-01-01",
                                                                       "2022-01-01"]))
    end = min(end, ctx.data.reserved_start)
    out = []
    for target, h in cfg.focus[:2]:
        task = cfg.targets[target].task
        plans = [(cfg.reference_linear(task), rt.get("every_months", [1, 3, 6, 12])),
                 ("lightgbm", rt.get("every_months_trees", [3, 6, 12]))]
        for fam, schedule in plans:
            for months in schedule:
                for fold in refit_schedule(ctx.data.timestamps, start, end, int(months),
                                           horizon=h, embargo=cfg.walk_forward.embargo_bars,
                                           inner_fraction=cfg.walk_forward.inner_fraction):
                    out.append(_spec(ctx, target, h, fam, cfg.default_feature_set, fold,
                                     variant={"refit": f"{int(months)}m"}))
    return out


def _plan_ablation(ctx: MLContext) -> list[UnitSpec]:
    cfg = ctx.cfg
    folds_used = [int(i) for i in cfg.ablation.get("folds", [3, 4])]
    out = []
    for target, h in cfg.focus:
        folds = [f for f in ctx.folds(h) if f.index in folds_used]
        sets = ablation_sets(_set_features(ctx, target, "extended"), ctx.data.registry)
        for label, feats in sets.items():
            for fold in folds:
                out.append(_spec(ctx, target, h, "lightgbm", f"ablation_{label}", fold,
                                 features=feats))
    return out


def _plan_nulls(ctx: MLContext) -> list[UnitSpec]:
    cfg = ctx.cfg
    nl = cfg.nulls
    out = []
    shuffled_folds = [int(i) for i in nl.get("shuffled_folds", [2, 3, 4])]
    pairs = [*cfg.focus, ("direction_up_cost", 5)]
    for target, h in pairs:
        task = cfg.targets[target].task
        folds = [f for f in ctx.folds(h) if f.index in shuffled_folds]
        for fam in (cfg.reference_linear(task), "lightgbm"):
            for rep in range(int(nl.get("shuffled_target_repeats", 2))):
                for fold in folds:
                    out.append(_spec(ctx, target, h, fam, cfg.default_feature_set, fold,
                                     variant={"null": "shift", "repeat": rep}))
    for target, h in cfg.focus:
        for fold in ctx.folds(h):
            out.append(_spec(ctx, target, h, "lightgbm", cfg.default_feature_set, fold,
                             variant={"noise": int(nl.get("noise_features", 5))}, explain=True))
            out.append(_spec(ctx, target, h, "lightgbm", cfg.default_feature_set, fold,
                             variant={"permute_features": 1}))
    return out


def pipeline_null_sets(ctx: MLContext) -> list[str]:
    """Feature sets read against the random-walk pipeline: Standard, and Extended when it
    differs (it carries the residual's own z-scores, the most mechanical inputs)."""
    default = ctx.cfg.default_feature_set
    sets = [default]
    if "extended" in ctx.data.manifests and \
            ctx.data.feature_set("extended") != ctx.data.feature_set(default):
        sets.append("extended")
    return sets


def pipeline_null_features(ctx: MLContext) -> list[str]:
    feats: list[str] = []
    for set_name in pipeline_null_sets(ctx):
        feats += [f for f in ctx.data.feature_set(set_name) if f not in feats]
    return feats


def pipeline_null_names(ctx: MLContext, stage: str) -> list[str]:
    """The registered pipeline null, or the post-hoc ones (``nulls.pipeline_posthoc``)."""
    nulls = ctx.cfg.nulls
    if stage == "pipeline_null":
        return [str(nulls.get("pipeline", "random_walk"))]
    return [str(n) for n in nulls.get("pipeline_posthoc") or []]


def null_version(ctx: MLContext, name: str) -> dict[str, Any]:
    """The null's entry in the null data's versions (part of every null unit's stamp)."""
    return {"null": name, "seed": ctx.cfg.random_seed}


def _plan_pipeline_null(ctx: MLContext, name: str | None = None) -> list[UnitSpec]:
    cfg = ctx.cfg
    variant = {"pipeline": name or pipeline_null_names(ctx, "pipeline_null")[0]}
    out = []
    for target, h in (("mean_reversion", 5), ("residual_reduction", 5),
                      ("mean_reversion_half", 5)):
        task = cfg.targets[target].task
        for fold in ctx.folds(h):
            for set_name in pipeline_null_sets(ctx):
                fams = (("constant", cfg.reference_linear(task), "lightgbm")
                        if set_name == cfg.default_feature_set
                        else (cfg.reference_linear(task), "lightgbm"))
                for fam in fams:
                    out.append(_spec(ctx, target, h, fam, set_name, fold, variant=variant))
    return out


def _plan_pipeline_null_posthoc(ctx: MLContext) -> list[UnitSpec]:
    return [u for name in pipeline_null_names(ctx, "pipeline_null_posthoc")
            for u in _plan_pipeline_null(ctx, name)]


_PLANNERS: dict[str, Callable[[MLContext], list[UnitSpec]]] = {
    "grid": _plan_grid, "trees": _plan_trees, "feature_sets": _plan_feature_sets,
    "variants": _plan_variants, "search": _plan_search, "windows": _plan_windows,
    "learning_curve": _plan_learning_curve, "decay": _plan_decay,
    "retraining": _plan_retraining, "ablation": _plan_ablation, "nulls": _plan_nulls,
    "pipeline_null": _plan_pipeline_null,
    "pipeline_null_posthoc": _plan_pipeline_null_posthoc}


def plan_stage(ctx: MLContext, stage: str) -> list[UnitSpec]:
    return _PLANNERS[stage](ctx)


# ---------------------------------------------------------------------------
# The random-walk pipeline null (invariant 9)
# ---------------------------------------------------------------------------
def build_pipeline_null(ctx: MLContext, features: list[str], name: str | None = None
                        ) -> MLData:
    """The features and residual targets computed on a pipeline null through the same
    engines (a Prompt #8 OHLC null: the registered random walk, or a post-hoc one), rows
    before the reserved start."""
    from ..features.factory import compute_families, load_bar_series, null_bar_series
    from ..features.joins import EngineProvider
    from ..ml.datasets import build_context
    from ..targets.alignment import TargetInputs, build_targets

    cfgs = ctx.cfgs
    name = name or pipeline_null_names(ctx, "pipeline_null")[0]
    real = load_bar_series(cfgs.config, ctx.timeframe)
    bars = null_bar_series(name, real, seed=ctx.cfg.random_seed)
    del real
    fams = {str(ctx.data.registry[f].get("family")) for f in features} - {"regime"}
    interactions = "interaction" in fams
    if interactions:                        # the components of every interaction, as live
        fams = (fams - {"interaction"}) | {"returns", "volatility", "autocorrelation",
                                           "regression", "ou", "fft", "wavelet",
                                           "microstructure", "time"}
    provider = EngineProvider(bars=bars, regression_config=cfgs.regression,
                              spectral_config=cfgs.spectral, wavelet_config=cfgs.wavelet)
    order = ("returns", "volatility", "autocorrelation", "regression", "ou", "fft", "wavelet",
             "microstructure", "time")         # OU before FFT / wavelets: they read |eta|
    values, extras = compute_families(bars, provider, cfgs.features, cfgs.ou, cfgs.research,
                                      families=tuple(f for f in order if f in fams), log=False,
                                      collect=False)
    if interactions:                        # regime components have no null: those stay NaN
        from ..features.interactions import interaction_columns

        values.update(interaction_columns(values, cfgs.features, bars.bar_seconds))
    reg = provider.regression(cfgs.targets.regression_window)
    tframe = build_targets(TargetInputs(bars=bars, residual=reg["residual"], sigma=reg["sigma"],
                                        ou_mu=None), cfgs.targets)
    n = first_row_at(bars.timestamps, ctx.data.reserved_start)
    stamps = bars.timestamps.head(n)
    if not stamps.equals(ctx.data.timestamps):
        raise RuntimeError("the pipeline null is not on the development bars")
    feats = {f: np.asarray(values[f][:n], dtype=np.float32) for f in features if f in values}
    for f in features:                          # regime features have no null: keep them missing
        feats.setdefault(f, np.full(n, np.nan, dtype=np.float32))
    for extra in ("spread_rel",):
        if extra in values:
            feats[extra] = np.asarray(values[extra][:n], dtype=np.float32)
    targets_raw = {}
    for c in ctx.data.targets_raw:
        if c in tframe.columns:
            v = np.array(tframe[c].cast(pl.Float64).fill_null(np.nan).to_numpy()[:n],
                         dtype=np.float64)
            h = int(c.rsplit("_", 1)[1])
            v[max(0, n - h):] = np.nan
            targets_raw[c] = v
    eps = np.asarray(reg["residual"][:n], dtype=np.float64)
    sigma = np.asarray(reg["sigma"][:n], dtype=np.float64)
    del values, extras, reg, tframe, provider
    gc.collect()
    null = MLData(timeframe=ctx.timeframe, timestamps=stamps,
                  reserved_start=ctx.data.reserved_start, features=feats,
                  manifests={k: v for k, v in ctx.data.manifests.items()
                             if set(v) <= set(feats)},
                  manifest_ids=ctx.data.manifest_ids, manifest_hashes=ctx.data.manifest_hashes,
                  registry=ctx.data.registry, targets_raw=targets_raw, epsilon=eps, sigma=sigma,
                  versions={**ctx.data.versions, "pipeline_null": null_version(ctx, name)})
    null.context = build_context(null)
    return null


# ---------------------------------------------------------------------------
# One timeframe
# ---------------------------------------------------------------------------
def load_context(cfgs: MLConfigs, timeframe: str, *, progress: Callable[[str], None] | None
                 = None) -> MLContext:
    data = load_ml_data(cfgs.ml, cfgs.selection, cfgs.features, cfgs.targets, cfgs.config,
                        cfgs.regression, timeframe, extra_horizons=(1, 2, 3, 5, 10, 20, 50))
    out = ensure_dir(cfgs.ml.results_path / timeframe)
    return MLContext(timeframe=timeframe, cfgs=cfgs, data=data, out_dir=out, progress=progress)


def run_ml_research(cfgs: MLConfigs, timeframe: str, *, stages: tuple[str, ...] = STAGES,
                    progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    unknown = sorted(set(stages) - set(STAGES) - {"neighbourhood"})
    if unknown:
        raise ValueError(f"unknown stage(s) {unknown}; expected from {list(STAGES)}")
    started = time.perf_counter()
    ctx = load_context(cfgs, timeframe, progress=progress)
    ctx.log(f"{ctx.data.n:,} development rows (< {ctx.data.reserved_start}), "
            f"{len(ctx.data.features)} features, sets {sorted(ctx.data.manifests)}")
    summary: dict[str, Any] = {"timeframe": timeframe, "stages": {}}
    for stage in STAGES:
        if stage not in stages:
            continue
        t0 = time.perf_counter()
        res: dict[str, Any]
        if stage in PIPELINE_STAGES:
            res = {"units": 0, "ran": 0, "failed": 0}
            for name in pipeline_null_names(ctx, stage):
                units = _plan_pipeline_null(ctx, name)
                # the stamp reads only the null data's versions: check before building it
                probe = SimpleNamespace(versions={**ctx.data.versions,
                                                  "pipeline_null": null_version(ctx, name)})
                if not pending_units(ctx, units, probe):  # type: ignore[arg-type]
                    ctx.log(f"{stage} {name}: {len(units)} units, all current")
                    res["units"] += len(units)
                    continue
                feats = pipeline_null_features(ctx)
                ctx.log(f"{stage} {name}: building {len(feats)} features and the residual "
                        "targets on the null")
                ctx.null_data = build_pipeline_null(ctx, feats, name)
                part = run_units(ctx, units, stage=f"{stage} {name}", data=ctx.null_data)
                res = {k: res[k] + part[k] for k in res}
                ctx.null_data = None
                gc.collect()
        else:
            res = run_units(ctx, plan_stage(ctx, stage), stage=stage)
        if stage == "search":
            from .ml_reports import best_search_trials

            best = best_search_trials(ctx)
            if best:
                res2 = run_units(ctx, plan_neighbourhood(ctx, best), stage="neighbourhood")
                res = {**res, "neighbourhood": res2}
        summary["stages"][stage] = {**res, "seconds": time.perf_counter() - t0}
        ctx.data.drop_designs()
        gc.collect()
    summary["seconds"] = time.perf_counter() - started
    summary["finished_utc"] = utc_now_iso()
    write_json(ctx.out_dir / f"run_{'_'.join(stages) if len(stages) < 4 else 'all'}.json",
               summary)
    return summary
