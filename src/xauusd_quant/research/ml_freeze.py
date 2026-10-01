r"""Freeze rules, final development models, streaming check and benchmark (Prompt #10,
Steps 75, 91-94, 100).

:func:`freeze_candidates` reads development results only and applies the rules
fixed in ``config/ml.yaml`` *before* any result existed, per tree-comparison
target:

1. eligible models beat the constant baseline on the primary metric in at
   least ``min_folds_beating_baseline`` of the five blocks, with a positive mean;
2. among them, the best mean; then the **simplest** model within one standard
   error of it (``simplicity_order``, then fewer features, Standard first);
3. training window: expanding, unless a rolling window or time-decay weights
   beat it by more than one SE, block by block (focus pairs only);
4. hyperparameters: the defaults, unless a search trial beats them by more than
   one SE and every one-step neighbour stays within two SE of it;
5. calibration: the simplest of none / platt / isotonic within one SE of the
   lowest log loss.

Each frozen candidate travels with the constant baseline and the reference
linear model of the same target, so the final test compares like with like.
:func:`finalize` then fits every frozen spec on the whole development span (no
reserved row exists in memory), saves the artifact, checks the reloaded model
reproduces its predictions, replays bars in streaming mode and times inference -
all before ``xq ml-final-test`` is allowed near the reserved period.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..ml.config import MLConfig
from ..ml.datasets import MLData, assert_no_leakage
from ..ml.evaluation import paired_blocks
from ..ml.registry import freeze_spec, load_artifact, load_frozen_spec, model_id, save_artifact
from ..ml.training import train_final
from ..utils.clock import utc_now_iso
from ..utils.paths import ensure_dir
from .feature_research import write_json
from .ml_reports import _metric_rows, primary_metric, summarise

__all__ = ["finalize", "freeze_candidates", "write_model_registry"]


def _per_block(view: pl.DataFrame, metric: str) -> dict[str, float | None]:
    return {r["fold"]: r.get(metric) for r in view.iter_rows(named=True)}


def _better_by_one_se(cmp: dict[str, Any]) -> bool:
    """A block-by-block difference (a - b) that exceeds its standard error.

    Needs two blocks at least; an SE of exactly zero (identical differences) counts
    as a consistent difference, not as a missing SE.
    """
    se = cmp.get("se")
    return int(cmp.get("blocks", 0)) >= 2 and se is not None and cmp["mean_diff"] > se


def _simplicity(cfg: MLConfig, family: str, n_features: int, set_name: str) -> tuple[int, int,
                                                                                     int]:
    order = list(cfg.simplicity_order)
    return (order.index(family) if family in order else len(order), n_features,
            0 if set_name == cfg.default_feature_set else 1)


def _spec_body(ctx: Any, target: str, h: int, family: str, set_name: str, features: list[str],
               params: dict[str, Any], calibration: str, policy: dict[str, Any], role: str,
               evidence: dict[str, Any]) -> dict[str, Any]:
    cfg: MLConfig = ctx.cfg
    data: MLData = ctx.data
    tspec = cfg.targets[target]
    from ..ml.models import preprocessing_kind

    key = set_name if set_name in data.manifests else f"target_{tspec.kind}"
    return {
        "spec_id": model_id(family, target, ctx.timeframe, h, cfg.registry_version),
        "role": role, "family": family, "target": target,
        "target_definition": {"source": tspec.source, "task": tspec.task, "kind": tspec.kind,
                              "transform": tspec.transform, "c": tspec.c,
                              "cost_multiple": tspec.cost_multiple},
        "horizon": int(h), "timeframe": ctx.timeframe, "feature_set": set_name,
        "feature_set_id": data.manifest_ids.get(key), "feature_set_hash":
            data.manifest_hashes.get(key), "features": list(features), "params": dict(params),
        "preprocessing": {"kind": preprocessing_kind(family, str(cfg.preprocessing.get(
            "tree_missing", "native"))), "clip": float(cfg.preprocessing.get("clip", 8.0))},
        "calibration": calibration if tspec.is_classification else "none",
        "training_policy": policy, "seed": cfg.random_seed,
        "dataset_versions": {k: data.versions.get(k) for k in (
            "tick_dataset_version", "bar_dataset_version", "factory_version", "target_version")},
        "development": evidence, "config_fingerprint": cfg.fingerprint(), "git_commit": None,
        "outputs": "probability" if tspec.is_classification else "expected value",
        "no_trading_output": True}


def freeze_candidates(ctx: Any, folds: pl.DataFrame) -> dict[str, Any]:
    cfg: MLConfig = ctx.cfg
    data: MLData = ctx.data
    view = _metric_rows(folds, cfg)
    summ = summarise(folds, cfg)
    frozen_dir = ensure_dir(ctx.out_dir / "frozen")
    decisions = []
    spec_paths = []
    fr = cfg.freeze
    min_folds = int(fr.get("min_folds_beating_baseline", 4))
    for target, h in cfg.comparison_pairs():
        tspec = cfg.targets[target]
        metric = primary_metric(tspec.task)
        # ablation subsets are research variants on two blocks, never candidates
        base = summ.filter((pl.col("target") == target) & (pl.col("horizon") == h)
                           & (pl.col("variant") == "base") & (pl.col("family") != "constant")
                           & ~pl.col("feature_set").str.starts_with("ablation_"))
        cands = base.filter((pl.col("blocks_beating_baseline") >= min_folds)
                            & (pl.col("mean") > 0))
        decision: dict[str, Any] = {"target": target, "horizon": h, "metric": metric,
                                    "configurations": base.height, "eligible": cands.height}
        if cands.is_empty():
            decision["frozen"] = None
            decision["reason"] = (f"no model beats the constant baseline in >= {min_folds} of "
                                  f"{len(cfg.walk_forward.validation_blocks)} blocks")
            decisions.append(decision)
            continue
        best = cands.sort("mean", descending=True).row(0, named=True)
        threshold = best["mean"] - (best["se"] or 0.0)
        near = [r for r in cands.iter_rows(named=True) if r["mean"] >= threshold]
        near.sort(key=lambda r: _simplicity(cfg, r["family"], len(_features(ctx, target,
                                                                           r["feature_set"])),
                                            r["feature_set"]))
        pick = near[0]
        fam, set_name = str(pick["family"]), str(pick["feature_set"])
        feats = _features(ctx, target, set_name)
        policy = {"scheme": "expanding", "rolling_years": cfg.walk_forward.rolling_years,
                  "weighting": None, "inner_fraction": cfg.walk_forward.inner_fraction,
                  "embargo_bars": cfg.walk_forward.embargo_bars, "purge_bars": int(h),
                  "train_end": str(data.reserved_start)}
        chosen_view = view.filter((pl.col("target") == target) & (pl.col("horizon") == h)
                                  & (pl.col("family") == fam))
        base_blocks = _per_block(chosen_view.filter((pl.col("variant") == "base")
                                                    & (pl.col("feature_set") == set_name)),
                                 metric)
        window_note = "expanding (default)"
        for var in ("window-rolling5", "weighting-2p0", "weighting-5p0"):
            alt = _per_block(chosen_view.filter((pl.col("variant") == var)
                                                & (pl.col("feature_set") == set_name)), metric)
            common = sorted(set(alt) & set(base_blocks))
            if not common:
                continue
            cmp = paired_blocks([alt[f] for f in common], [base_blocks[f] for f in common])
            if _better_by_one_se(cmp):
                if var.startswith("window"):
                    policy["scheme"] = "rolling"
                else:
                    policy["weighting"] = float(var.removeprefix("weighting-").replace("p", "."))
                window_note = f"{var}: +{cmp['mean_diff']:.4f} (SE {cmp['se']:.4f}) over expanding"
        params = cfg.model_params(fam)
        param_note = "configured defaults"
        if fam in ("lightgbm", "xgboost") and [target, h] in [list(p) for p in cfg.search.get(
                "targets", [])]:
            params, param_note = _search_choice(view, target, h, fam, metric, params,
                                                chosen_set=set_name,
                                                searched_set=cfg.default_feature_set)
        cal_note = "none (regression)"
        calibration = "none"
        if tspec.is_classification:
            calibration, cal_note = _calibration_choice(folds, target, h, fam, set_name)
        evidence = {"mean": pick["mean"], "se": pick["se"], "min": pick["min"],
                    **{k: v for k, v in pick.items() if k.startswith("mean_")},
                    "blocks_beating_baseline": pick["blocks_beating_baseline"],
                    "recent_block": pick["recent_block"], "recent_metric": pick["recent_metric"],
                    "per_block": base_blocks, "best_family": best["family"],
                    "best_feature_set": best["feature_set"], "best_mean": best["mean"],
                    "best_se": best["se"], "within_one_se": [(r["family"], r["feature_set"])
                                                             for r in near],
                    "window_rule": window_note, "hyperparameter_rule": param_note,
                    "calibration_rule": cal_note}
        chosen = _spec_body(ctx, target, h, fam, set_name, feats, params, calibration, policy,
                            "candidate", evidence)
        spec_paths.append(freeze_spec(chosen, frozen_dir))
        ref = cfg.reference_linear(tspec.task)
        companions = [("constant", cfg.model_params("constant"), "baseline")]
        if fam != ref:
            companions.append((ref, cfg.model_params(ref), "reference"))
        std = _features(ctx, target, cfg.default_feature_set)
        for cfam, cparams, role in companions:
            cview = view.filter((pl.col("target") == target) & (pl.col("horizon") == h)
                                & (pl.col("family") == cfam) & (pl.col("variant") == "base")
                                & (pl.col("feature_set") == cfg.default_feature_set))
            crow = base.filter((pl.col("family") == cfam)
                               & (pl.col("feature_set") == cfg.default_feature_set))
            if cfam == "constant":
                crow = summ.filter((pl.col("target") == target) & (pl.col("horizon") == h)
                                   & (pl.col("family") == "constant")
                                   & (pl.col("variant") == "base"))
            ev = {"per_block": _per_block(cview, metric), "role": role,
                  **({k: v for k, v in crow.row(0, named=True).items()
                      if k.startswith("mean") or k in ("se", "blocks_beating_baseline")}
                     if crow.height else {})}
            cal = "platt" if tspec.is_classification and cfam != "constant" else "none"
            body = _spec_body(ctx, target, h, cfam, cfg.default_feature_set, std, cparams, cal,
                              {**policy, "scheme": "expanding", "weighting": None}, role, ev)
            spec_paths.append(freeze_spec(body, frozen_dir))
        decision.update({"frozen": chosen["spec_id"], "family": fam, "feature_set": set_name,
                         "reason": f"simplest within one SE of the best ({best['family']}/"
                                   f"{best['feature_set']} {best['mean']:.4f} +- "
                                   f"{best['se'] or 0:.4f})",
                         "companions": [model_id(c, target, ctx.timeframe, h,
                                                 cfg.registry_version) for c, _, _ in companions],
                         **{k: evidence[k] for k in ("window_rule", "hyperparameter_rule",
                                                     "calibration_rule")}})
        decisions.append(decision)
    report = {"timeframe": ctx.timeframe, "rules": fr, "decisions": decisions,
              "specs": [p.name for p in spec_paths], "frozen_utc": utc_now_iso(),
              "MODEL_SPEC_FROZEN": True}
    write_json(frozen_dir / "freeze_report.json", report)
    return report


def _features(ctx: Any, target: str, set_name: str) -> list[str]:
    from .ml_research import _set_features

    if set_name.startswith("ablation_"):
        raise ValueError("ablation sets are research variants, never frozen")
    return _set_features(ctx, target, set_name)


def _search_choice(view: pl.DataFrame, target: str, h: int, fam: str, metric: str,
                   default: dict[str, Any], *, chosen_set: str, searched_set: str
                   ) -> tuple[dict[str, Any], str]:
    """Search trials against the defaults on the set the search ran on, block by block.

    Trials, neighbours and the default-parameter units compared with them must share
    one feature set: with base units of several sets on the same blocks, a per-block
    lookup would otherwise take whichever set's row came last. A model frozen on
    another set keeps the defaults - the search says nothing about that set.
    """
    if chosen_set != searched_set:
        return default, (f"configured defaults (the search ran on {searched_set}; the "
                         f"frozen set is {chosen_set})")
    part = view.filter((pl.col("target") == target) & (pl.col("horizon") == h)
                       & (pl.col("family") == fam) & (pl.col("feature_set") == searched_set))
    search_folds = set(part.filter(pl.col("variant").str.starts_with("search-"))["fold"]
                       .to_list())
    base = _per_block(part.filter((pl.col("variant") == "base")
                                  & pl.col("fold").is_in(list(search_folds))), metric)
    best_var, best_mean = None, -np.inf
    for (var,), p in part.filter(pl.col("variant").str.starts_with("search-")).group_by(
            ["variant"]):
        vals = [v for v in p[metric].to_list() if v is not None]
        if vals and np.mean(vals) > best_mean:
            best_var, best_mean = str(var), float(np.mean(vals))
    if best_var is None or not base:
        return default, "configured defaults (no search result)"
    trial = _per_block(part.filter(pl.col("variant") == best_var), metric)
    common = sorted(set(trial) & set(base))
    cmp = paired_blocks([trial[f] for f in common], [base[f] for f in common])
    js = json.loads(Path(str(part.filter(pl.col("variant") == best_var)["path"][0]))
                    .with_suffix(".json").read_text(encoding="utf-8"))
    if not _better_by_one_se(cmp):
        return default, (f"configured defaults: best trial {best_var} "
                         f"{cmp.get('mean_diff', 0):+.4f} vs defaults is within one SE")
    if fam == "lightgbm":
        neigh = part.filter(pl.col("variant").str.starts_with("neighbour-"))
        worst = min((float(np.mean([v for v in p[metric].to_list() if v is not None]))
                     for _, p in neigh.group_by(["variant"])), default=best_mean)
        if worst < best_mean - 2 * (cmp["se"] or 0.0):
            return default, (f"configured defaults: {best_var} is better but a one-step "
                             f"neighbour falls to {worst:.4f} (unstable)")
    return dict(js["params"]), f"search trial {best_var}: {cmp['mean_diff']:+.4f} over defaults"


def _calibration_choice(folds: pl.DataFrame, target: str, h: int, fam: str,
                        set_name: str) -> tuple[str, str]:
    part = folds.filter((pl.col("target") == target) & (pl.col("horizon") == h)
                        & (pl.col("family") == fam) & (pl.col("variant") == "base")
                        & (pl.col("feature_set") == set_name))
    means = {}
    blocks = {}
    for (calib,), p in part.group_by(["calibration"]):
        vals = [v for v in p["log_loss"].to_list() if v is not None]
        if vals:
            means[str(calib)] = float(np.mean(vals))
            blocks[str(calib)] = _per_block(p, "log_loss")
    if not means:
        return "platt", "platt (no calibration result)"
    best = min(means, key=means.get)  # type: ignore[arg-type]
    for method in ("raw", "platt", "isotonic"):             # simplest first
        if method not in means:
            continue
        if method != best:
            common = sorted(set(blocks[method]) & set(blocks[best]))
            # method - best: how much higher the method's loss is, block by block
            cmp = paired_blocks([blocks[method][f] for f in common],
                                [blocks[best][f] for f in common])
            se = cmp.get("se")
            if not (int(cmp.get("blocks", 0)) >= 2 and se is not None
                    and cmp["mean_diff"] <= se):
                continue
        chosen = "none" if method == "raw" else method
        return chosen, (f"{chosen}: mean log loss {means[method]:.5f}, within one SE of the "
                        f"lowest ({best} {means[best]:.5f})")
    return "platt", "platt"


# ---------------------------------------------------------------------------
# Final development models: artifacts, reload, streaming, benchmark
# ---------------------------------------------------------------------------
def finalize(ctx: Any, *, streaming: bool = True) -> dict[str, Any]:
    cfg: MLConfig = ctx.cfg
    data: MLData = ctx.data
    frozen_dir = ctx.out_dir / "frozen"
    specs = [load_frozen_spec(p) for p in sorted(frozen_dir.glob("MODEL_SPEC_*.json"))]
    out: dict[str, Any] = {"timeframe": ctx.timeframe, "models": []}
    bars = None
    for spec in specs:
        tspec = cfg.targets[spec["target"]]
        art = cfg.models_path / spec["spec_id"]
        entry: dict[str, Any] = {"model_id": spec["spec_id"], "role": spec.get("role"),
                                 "family": spec["family"]}
        if (art / "manifest.json").exists():
            model, pre, cal, man = load_artifact(art)
            if man.get("spec_hash") != spec["content_hash"]:
                raise RuntimeError(f"{spec['spec_id']}: the stored artifact was built from "
                                   "another spec")
            info = man.get("training", {})
            entry["artifact"] = "existing"
        else:
            t0 = time.perf_counter()
            model, pre, cal, info = train_final(data, spec, tspec, cfg)
            save_artifact(art, model=model, preprocessor=pre, calibrator=cal, spec=spec,
                          extra={"training": info, "training_policy": spec["training_policy"]})
            entry["train_seconds"] = time.perf_counter() - t0
            entry["artifact"] = "trained"
        feats = list(spec["features"])
        assert_no_leakage(feats, data.registry)                  # live compatibility (Step 92)
        entry["live_compatible"] = {
            "all_registered_live_safe": all(data.registry[f].get("live_safe") for f in feats),
            "regime_features_from_walk_forward_service":
                [f for f in feats if data.registry[f].get("family") == "regime"]}
        check = np.linspace(0, data.n - 1, 3000).astype(np.int64)
        xd = data.design(feats)[check]
        first = cal.apply(model.predict(pre.transform(xd, feats)))
        m2, p2, c2, man2 = load_artifact(art)
        again = c2.apply(m2.predict(p2.transform(xd, feats)))
        entry["reload_equal"] = bool(np.allclose(first, again, rtol=1e-7, atol=1e-9))
        entry["model_size_bytes"] = man2.get("model_size_bytes")
        one = xd[:1]
        t0 = time.perf_counter()
        for _ in range(300):
            cal.apply(model.predict(pre.transform(one, feats)))
        entry["latency_ms_per_row"] = (time.perf_counter() - t0) / 300 * 1000
        batch = data.design(feats)[:10_000]
        import tracemalloc

        tracemalloc.start()
        t0 = time.perf_counter()
        cal.apply(model.predict(pre.transform(batch, feats)))
        entry["latency_ms_per_10k_rows"] = (time.perf_counter() - t0) * 1000
        # Python / NumPy allocations only - a booster's native buffers are not traced
        entry["prediction_peak_mb_10k_rows"] = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
        entry["training"] = info
        if streaming and spec["family"] != "constant":
            from ..features.factory import load_bar_series
            from ..ml.streaming import stream_predictions
            from ..selection.live import required_buffer

            if bars is None:
                bars = load_bar_series(ctx.cfgs.config, ctx.timeframe)
            st = cfg.streaming
            steps, stride = int(st.get("steps", 40)), int(st.get("stride", 7))
            rows = [data.n - 1 - stride * k for k in range(steps)][::-1]
            pct = ctx.cfgs.features.percentile_window(bars.bar_seconds)
            buffer = required_buffer(data.registry, feats, pct)
            res = stream_predictions(model, pre, cal, feats, data.registry, bars,
                                     data.features, rows, buffer=buffer, fcfg=ctx.cfgs.features,
                                     ou=ctx.cfgs.ou, research=ctx.cfgs.research,
                                     regression=ctx.cfgs.regression, spectral=ctx.cfgs.spectral,
                                     wavelet=ctx.cfgs.wavelet, rtol=float(st.get("rtol", 1e-4)),
                                     atol=float(st.get("atol", 1e-6)))
            entry["streaming"] = {k: v for k, v in res.items() if k not in ("live", "batch")}
            entry["streaming_safe"] = bool(res["passed"])
        out["models"].append(entry)
        ctx.log(f"finalize {spec['spec_id']}: reload {entry['reload_equal']}, streaming "
                f"{entry.get('streaming_safe')}, {entry['latency_ms_per_row']:.3f} ms/row")
    out["finished_utc"] = utc_now_iso()
    write_json(frozen_dir / "finalize.json", out)
    write_model_registry(cfg, specs, out["models"], frozen_dir)
    return out


def write_model_registry(cfg: MLConfig, specs: list[dict[str, Any]],
                         models: list[dict[str, Any]], frozen_dir: Path) -> Path:
    """Merge the finalized models into ``config/model_registry.yaml`` (Step 74).

    One entry per model id: its spec and artifact (paths relative to the project,
    hashes), what it predicts, and whether it passed reload and streaming. The
    file is generated; entries of other timeframes are kept.
    """
    import yaml

    from ..utils.paths import atomic_write_text

    path = cfg.registry_path
    current: dict[str, Any] = {}
    if path.exists():
        current = dict((yaml.safe_load(path.read_text(encoding="utf-8")) or {})
                       .get("models") or {})
    root = cfg.project_root

    def rel(p: Path) -> str:
        try:
            return p.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            return p.as_posix()

    by_id = {m["model_id"]: m for m in models}
    for spec in specs:
        m = by_id.get(spec["spec_id"], {})
        current[spec["spec_id"]] = {
            "role": spec.get("role"), "family": spec["family"], "target": spec["target"],
            "horizon": int(spec["horizon"]), "timeframe": spec["timeframe"],
            "outputs": spec.get("outputs"), "feature_set_id": spec.get("feature_set_id"),
            "features": len(spec["features"]), "calibration": spec.get("calibration"),
            "training_policy": (spec.get("training_policy") or {}).get("scheme"),
            "spec": rel(frozen_dir / f"MODEL_SPEC_{spec['spec_id']}.json"),
            "spec_hash": spec["content_hash"],
            "artifact": rel(cfg.models_path / spec["spec_id"]),
            "reload_equal": m.get("reload_equal"), "streaming_safe": m.get("streaming_safe"),
            "latency_ms_per_row": m.get("latency_ms_per_row"),
            "model_size_bytes": m.get("model_size_bytes"),
            "no_trading_output": bool(spec.get("no_trading_output", False))}
    header = ("# Registry of finalized models - generated by `xq ml-finalize`; do not edit.\n"
              "# Each entry points to its frozen, content-hashed spec and its artifact;\n"
              "# models output probabilities / expected values only (Prompt #10).\n")
    body = yaml.safe_dump({"schema_version": 1, "updated_utc": utc_now_iso(),
                           "models": dict(sorted(current.items()))}, sort_keys=False,
                          allow_unicode=True)
    atomic_write_text(path, header + body)
    return path
