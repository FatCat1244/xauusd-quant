r"""Tables, comparisons, freeze rules and ledger rows of the supervised research
(Prompt #10, Steps 23-33, 41-66, 75, 95-96, 100).

Reads the unit files written by :mod:`.ml_research`:

* ``folds``: one row per unit and calibration (raw / platt / isotonic);
* ``pooled``: the folds' out-of-sample predictions concatenated (2011-2021)
  and scored again as one sample, per calendar year and on the most recent
  block;
* comparisons with the constant baseline and the reference linear model, block
  by block (mean difference, SE, wins, t);
* search / neighbourhood / complexity, windows, weights, learning curve, decay,
  retraining, ablation, nulls and the random-walk pipeline null;
* SHAP / permutation stability, prediction correlations, disagreement,
  distribution shift and the out-of-distribution score, failure analysis.

:func:`freeze_candidates` applies the pre-registered freeze rules of
``config/ml.yaml`` to development results only and writes the immutable model
specs; :func:`ledger_entries` gives every tried configuration a permanent
``ML-H`` id in the research ledger - failed ones included.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..alpha.information_coefficient import rank_scores
from ..ml.config import MLConfig
from ..ml.datasets import SESSION_LABELS, MLData
from ..ml.evaluation import (
    classification_metrics,
    decile_table,
    grouped_metrics,
    paired_blocks,
    regression_metrics,
)
from ..utils.paths import ensure_dir
from .feature_research import write_json, write_table

__all__ = [
    "best_search_trials",
    "build_ml_report",
    "collect_units",
    "export_oos_predictions",
    "ledger_entries",
    "primary_metric",
    "summarise",
]

_CAL = "platt"                      # the calibration read for classification comparisons


def primary_metric(task: str) -> str:
    return "log_loss_skill" if task == "classification" else "rank_ic"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------
def collect_units(out_dir: Path, *, with_explain: bool = False) -> tuple[pl.DataFrame,
                                                                          list[dict[str, Any]]]:
    """(one row per unit x calibration, the explain payloads when asked)."""
    rows = []
    explains = []
    for js in sorted((out_dir / "units").rglob("*.json")):
        if js.name.endswith(".failed.json"):
            continue
        d = _json(js)
        variant = d.get("variant") or {}
        tag = "__".join(f"{k}-{variant[k]}" for k in sorted(variant)) if variant else "base"
        common = {"target": d["target"], "horizon": int(d["horizon"]), "family": d["family"],
                  "feature_set": d["feature_set"], "variant": tag.replace(".", "p"),
                  "fold": d["fold"]["name"], "fold_index": int(d["fold"]["index"]),
                  "fit_seconds": d["info"].get("fit_seconds"),
                  "best_iteration": d["info"].get("best_iteration"),
                  "fit_rows": d["info"].get("fit_rows"),
                  "validation_rows": d["info"].get("validation_rows"),
                  "design_columns": d["info"].get("design_columns"),
                  "converged": d["info"].get("converged"),
                  "path": str(js.with_suffix(".parquet"))}
        for calib, met in (d.get("metrics") or {}).items():
            rows.append({**common, "calibration": calib,
                         **{k: v for k, v in met.items() if not isinstance(v, (list, dict))}})
        if with_explain and d.get("explain"):
            explains.append({**common, "explain": d["explain"]})
    frame = pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()
    return frame, explains


def _task(cfg: MLConfig, target: str) -> str:
    return cfg.targets[target].task


def _metric_rows(folds: pl.DataFrame, cfg: MLConfig) -> pl.DataFrame:
    """The comparison view: platt rows for classification, raw rows for regression.

    The constant is the baseline itself and is always read raw (the training mean):
    a Platt-calibrated constant is the inner slice's positive rate, another model.
    """
    if folds.is_empty():
        return folds
    cls = [t for t, s in cfg.targets.items() if s.is_classification]
    const = pl.col("family") == "constant"
    return folds.filter((pl.col("target").is_in(cls) & ~const & (pl.col("calibration") == _CAL))
                        | ((~pl.col("target").is_in(cls) | const)
                           & (pl.col("calibration") == "raw")))


def summarise(folds: pl.DataFrame, cfg: MLConfig) -> pl.DataFrame:
    """Per configuration: mean / SE of the primary metric over blocks, blocks beating the
    constant baseline, mean fit time."""
    view = _metric_rows(folds, cfg)
    if view.is_empty():
        return view
    base = (view.filter((pl.col("family") == "constant") & (pl.col("variant") == "base"))
            .select("target", "horizon", "fold", pl.col("log_loss").alias("base_log_loss")
                    if "log_loss" in view.columns else pl.lit(None).alias("base_log_loss")))
    out = []
    keys = ["target", "horizon", "family", "feature_set", "variant"]
    for key, part in view.group_by(keys, maintain_order=True):
        target, h, fam, fset, var = key
        task = _task(cfg, str(target))
        metric = primary_metric(task)
        vals = [v for v in part[metric].to_list() if v is not None] if metric in part.columns \
            else []
        skill = "log_loss_skill" if task == "classification" else "mse_skill"
        # above float noise: the constant scored against itself is 0 up to rounding
        beats = int(sum(1 for v in part[skill].to_list() if v is not None and v > 1e-12))
        arr = np.asarray(vals, dtype=np.float64)
        row = {"target": target, "horizon": int(h), "family": fam, "feature_set": fset,
               "variant": var, "task": task, "metric": metric, "blocks": int(part.height),
               "mean": float(arr.mean()) if arr.size else None,
               "se": float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else None,
               "min": float(arr.min()) if arr.size else None,
               "blocks_beating_baseline": beats,
               "fit_seconds": _mean_metric(part, "fit_seconds")}
        for extra in ("auc", "brier_skill", "ece", "rank_ic", "mse_skill", "r2", "pr_auc",
                      "rank_ic_raw", "log_loss_skill", "brier", "log_loss", "rmse", "mae"):
            if extra in part.columns:
                v = [x for x in part[extra].to_list() if x is not None]
                row[f"mean_{extra}"] = float(np.mean(v)) if v else None
        recent = part.filter(pl.col("fold_index") == part["fold_index"].max())
        row["recent_block"] = str(recent["fold"][0]) if recent.height else None
        row["recent_metric"] = (float(recent[metric][0]) if recent.height and metric in
                                recent.columns and recent[metric][0] is not None else None)
        out.append(row)
    del base
    return pl.DataFrame(out, infer_schema_length=None)


def paired_vs(folds: pl.DataFrame, cfg: MLConfig, *, reference: str) -> pl.DataFrame:
    """Every base configuration against a reference family (same target / horizon / blocks)."""
    view = _metric_rows(folds, cfg).filter(pl.col("variant") == "base")
    out = []
    for (target, h), part in view.group_by(["target", "horizon"], maintain_order=True):
        task = _task(cfg, str(target))
        metric = primary_metric(task)
        if reference == "constant":          # the skill that defines "beats the baseline"
            metric = "log_loss_skill" if task == "classification" else "mse_skill"
        ref_fam = cfg.reference_linear(task) if reference == "linear" else reference
        ref = part.filter((pl.col("family") == ref_fam)
                          & (pl.col("feature_set") == cfg.default_feature_set))
        ref_by = {r["fold"]: r.get(metric) for r in ref.iter_rows(named=True)}
        for (fam, fset), p in part.group_by(["family", "feature_set"], maintain_order=True):
            if fam == ref_fam and fset == cfg.default_feature_set:
                continue
            mine = {r["fold"]: r.get(metric) for r in p.iter_rows(named=True)}
            folds_common = sorted(set(mine) & set(ref_by))
            cmp = paired_blocks([mine[f] for f in folds_common], [ref_by[f] for f in folds_common])
            out.append({"target": target, "horizon": int(h), "family": fam, "feature_set": fset,
                        "reference": ref_fam, "metric": metric, **cmp})
    return pl.DataFrame(out, infer_schema_length=None) if out else pl.DataFrame()


# ---------------------------------------------------------------------------
# Pooled out-of-sample predictions
# ---------------------------------------------------------------------------
def load_predictions(paths: list[str]) -> pl.DataFrame:
    frames = []
    for p in paths:
        f = pl.read_parquet(p)
        frames.append(f.with_columns(pl.lit(Path(p).stem).alias("fold")))
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def pooled_tables(data: MLData, cfg: MLConfig, folds: pl.DataFrame
                  ) -> dict[str, pl.DataFrame]:
    """Pooled / yearly / grouped / decile / calibration / alpha-decay tables of base units."""
    base = folds.filter((pl.col("variant") == "base") & (pl.col("calibration") == "raw"))
    keys = ["target", "horizon", "family", "feature_set"]
    pooled_rows, year_rows, group_rows, decile_rows, calib_rows, decay_rows = \
        [], [], [], [], [], []
    curve_rows, hist_rows = [], []
    years = data.context["year"]
    horizons = sorted({int(c.rsplit("_", 1)[1]) for c in data.targets_raw})
    cache: dict[tuple[str, int], Any] = {}
    constant: dict[tuple[str, int], np.ndarray] = {}
    for (target, h), part in base.filter(pl.col("family") == "constant").group_by(
            ["target", "horizon"]):
        pred = load_predictions(sorted(set(part["path"].to_list())))
        arr = np.full(data.n, np.nan, dtype=np.float32)
        arr[pred["row"].to_numpy().astype(np.int64)] = pred["prediction"].to_numpy()
        constant[(str(target), int(h))] = arr
    for key, part in base.group_by(keys, maintain_order=True):
        target, h, fam, fset = str(key[0]), int(key[1]), str(key[2]), str(key[3])
        tspec = cfg.targets[target]
        if (target, h) not in cache:
            cache.clear()
            cache[(target, h)] = data.target(tspec, h, cfg.log_floor)
        ta = cache[(target, h)]
        pred = load_predictions(sorted(set(part["path"].to_list())))
        rows = pred["row"].to_numpy().astype(np.int64)
        y, y_raw = ta.y[rows], ta.y_raw[rows]
        carr = constant.get((target, h))
        base_pred = (carr[rows].astype(np.float64) if carr is not None
                     else np.full(rows.size, np.nan))
        cols = ["prediction"] + [c for c in ("cal_platt", "cal_isotonic") if c in pred.columns]
        ident = {"target": target, "horizon": h, "family": fam, "feature_set": fset}
        base_rate = float(np.nanmean(base_pred)) if np.isfinite(base_pred).any() else None
        for col in cols:
            p = pred[col].cast(pl.Float64).to_numpy()
            calib = {"prediction": "raw", "cal_platt": "platt", "cal_isotonic": "isotonic"}[col]
            if tspec.is_classification:
                met = classification_metrics(p, y, base_rate=base_rate)
                ok = np.isfinite(p) & np.isfinite(y) & np.isfinite(base_pred)
                if ok.sum() > 100:
                    bp = np.clip(base_pred[ok], 1e-6, 1 - 1e-6)
                    pc = np.clip(p[ok], 1e-6, 1 - 1e-6)
                    yy = y[ok]
                    ll_b = float(-np.mean(yy * np.log(bp) + (1 - yy) * np.log(1 - bp)))
                    ll_m = float(-np.mean(yy * np.log(pc) + (1 - yy) * np.log(1 - pc)))
                    met["log_loss_skill"] = 1.0 - ll_m / ll_b
                    met["brier_skill"] = 1.0 - float(np.mean((p[ok] - yy) ** 2)) / float(
                        np.mean((base_pred[ok] - yy) ** 2))
            else:
                met = regression_metrics(p, y, y_raw=y_raw)
                ok = np.isfinite(p) & np.isfinite(y) & np.isfinite(base_pred)
                if ok.sum() > 100:
                    met["mse_skill"] = 1.0 - float(np.mean((p[ok] - y[ok]) ** 2)) / float(
                        np.mean((base_pred[ok] - y[ok]) ** 2))
            pooled_rows.append({**ident, "calibration": calib, **met})
            if tspec.is_classification and calib in ("raw", _CAL, "isotonic"):
                from ..ml.calibration import reliability

                for r in reliability(p, y, bins=int(cfg.calibration.get("bins", 20))):
                    calib_rows.append({**ident, "calibration": calib, **r})
        score = pred[f"cal_{_CAL}" if f"cal_{_CAL}" in pred.columns else "prediction"].cast(
            pl.Float64).to_numpy() if tspec.is_classification else pred["prediction"].cast(
            pl.Float64).to_numpy()
        if tspec.is_classification:
            from sklearn.metrics import precision_recall_curve, roc_curve

            ok = np.isfinite(score) & np.isfinite(y)
            if ok.sum() > 1000 and 0 < y[ok].mean() < 1:
                fpr, tpr, _ = roc_curve(y[ok], score[ok])
                prec, rec, _ = precision_recall_curve(y[ok], score[ok])
                for kind, xs, ys in (("roc", fpr, tpr), ("pr", rec, prec)):
                    take = np.unique(np.linspace(0, xs.size - 1, min(xs.size, 200)).astype(int))
                    for i in take:
                        curve_rows.append({**ident, "curve": kind, "x": float(xs[i]),
                                           "y": float(ys[i])})
                counts, edges = np.histogram(score[ok], bins=40, range=(0.0, 1.0))
                for i, c in enumerate(counts):
                    hist_rows.append({**ident, "lo": float(edges[i]), "hi": float(edges[i + 1]),
                                      "share": float(c / max(1, counts.sum()))})
        for yr in np.unique(years[rows]):
            m = years[rows] == yr
            met = (classification_metrics(score[m], y[m], base_rate=base_rate)
                   if tspec.is_classification
                   else regression_metrics(score[m], y[m], y_raw=y_raw[m], base_value=base_rate))
            year_rows.append({**ident, "year": int(yr), **met})
        for gname, labels in (("regime_state", None), ("vol_quartile", None),
                              ("spread_quartile", None),
                              ("session", dict(enumerate(SESSION_LABELS)))):
            if gname not in data.context:
                continue
            for r in grouped_metrics(score, y, data.context[gname][rows], task=tspec.task,
                                     base=base_rate, y_raw=y_raw, labels=labels):
                group_rows.append({**ident, "grouping": gname, **r})
        dec = decile_table(score, y_raw if not tspec.is_classification else y)
        for r in dec["rows"]:
            decile_rows.append({**ident, **r, "spread": dec["spread"],
                                "monotonicity": dec["monotonicity"]})
        # model-level alpha decay: the score against the same family's outcome at other horizons
        for hh in horizons:
            if hh == h:
                other = y_raw
            else:
                try:
                    other = data.target(tspec, hh, cfg.log_floor).y_raw[rows]
                except KeyError:
                    continue
            ok = np.isfinite(score) & np.isfinite(other)
            if ok.sum() < 1000:
                continue
            ic = float(np.corrcoef(rank_scores(score[ok]), rank_scores(other[ok]))[0, 1])
            decay_rows.append({**ident, "outcome_horizon": hh, "rank_ic": ic,
                               "n": int(ok.sum())})
    return {"pooled": pl.DataFrame(pooled_rows, infer_schema_length=None),
            "yearly": pl.DataFrame(year_rows, infer_schema_length=None),
            "groups": pl.DataFrame(group_rows, infer_schema_length=None),
            "deciles": pl.DataFrame(decile_rows, infer_schema_length=None),
            "reliability": pl.DataFrame(calib_rows, infer_schema_length=None),
            "alpha_decay": pl.DataFrame(decay_rows, infer_schema_length=None),
            "roc_pr": pl.DataFrame(curve_rows, infer_schema_length=None),
            "probability_hist": pl.DataFrame(hist_rows, infer_schema_length=None)}


# ---------------------------------------------------------------------------
# Search, windows, learning curve, decay, retraining, ablation, nulls
# ---------------------------------------------------------------------------
def _mean_metric(view: pl.DataFrame, metric: str) -> float | None:
    v = [x for x in view[metric].to_list() if x is not None] if metric in view.columns else []
    return float(np.mean(v)) if v else None


def best_search_trials(ctx: Any) -> dict[tuple[str, int], dict[str, Any]]:
    """The best LightGBM search trial per searched pair (read from the unit files)."""
    cfg: MLConfig = ctx.cfg
    folds, _ = collect_units(ctx.out_dir)
    if folds.is_empty():
        return {}
    view = _metric_rows(folds, cfg).filter((pl.col("family") == "lightgbm")
                                          & pl.col("variant").str.starts_with("search-"))
    out = {}
    for (target, h), part in view.group_by(["target", "horizon"]):
        metric = primary_metric(_task(cfg, str(target)))
        best, best_val = None, -np.inf
        for (var,), p in part.group_by(["variant"]):
            val = _mean_metric(p, metric)
            if val is not None and val > best_val:
                best, best_val = str(var), val
        if best is None:
            continue
        js = _json(Path(str(part.filter(pl.col("variant") == best)["path"][0])).with_suffix(
            ".json"))
        out[(str(target), int(h))] = dict(js["params"])
    return out


def research_tables(folds: pl.DataFrame, cfg: MLConfig) -> dict[str, pl.DataFrame]:
    """Search, windows, weights, learning curve, retraining, ablation and null summaries."""
    view = _metric_rows(folds, cfg)
    out: dict[str, pl.DataFrame] = {}
    if view.is_empty():
        return out
    summ = summarise(folds, cfg)

    def pick(prefix: str) -> pl.DataFrame:
        return summ.filter(pl.col("variant").str.starts_with(prefix)
                           | pl.col("feature_set").str.starts_with(prefix))

    out["search"] = pick("search-")
    out["neighbourhood"] = pick("neighbour-")
    out["complexity"] = pick("leaves-")
    # windows and weights were fitted on the default set only: the expanding reference is
    # that set's base unit, never another set's (Extended, target sets, ablation subsets)
    out["windows"] = summ.filter((pl.col("variant").str.starts_with("window-")
                                  | pl.col("variant").str.starts_with("weighting-")
                                  | (pl.col("variant") == "base"))
                                 & (pl.col("feature_set") == cfg.default_feature_set))
    lc = pick("history-")
    if not lc.is_empty():                   # the full history is the last block's base unit
        last = folds.filter((pl.col("variant") == "base")
                            & (pl.col("feature_set") == cfg.default_feature_set))
        last = last.filter(pl.col("fold_index") == last["fold_index"].max())
        full = summarise(last, cfg).join(lc.select("target", "horizon", "family").unique(),
                                         on=["target", "horizon", "family"])
        lc = pl.concat([lc, full.with_columns(pl.lit("history-1p0").alias("variant"))],
                       how="diagonal_relaxed")
    out["learning_curve"] = lc
    out["retraining"] = pick("refit-")
    out["ablation"] = pick("ablation_")
    out["nulls"] = summ.filter(pl.col("variant").str.starts_with("null-")
                               | pl.col("variant").str.starts_with("noise-")
                               | pl.col("variant").str.starts_with("permute_features-")
                               | pl.col("variant").str.starts_with("pipeline-"))
    out["variants"] = summ.filter(pl.col("variant").str.starts_with("tree_missing-")
                                  | pl.col("variant").str.starts_with("balanced-"))
    return out


def skill_beyond_base_rate(folds: pl.DataFrame, cfg: MLConfig) -> pl.DataFrame:
    """Classification skill beyond the constant recalibrated on the inner slice.

    The baseline of every skill is the constant *training* positive rate. The same
    constant after Platt calibration on the inner slice is the recent base rate - a
    model that learns nothing from the features but follows a drifting base rate
    (for the one-spread direction labels it moves with spreads and volatility). Its
    skill is what calibration alone earns; a model's skill beyond it, block by
    block, is what the features add.
    """
    cls = [t for t, s in cfg.targets.items() if s.is_classification]
    if folds.is_empty():
        return pl.DataFrame()
    view = folds.filter(pl.col("target").is_in(cls) & (pl.col("variant") == "base")
                        & (pl.col("calibration") == _CAL))
    rows = []
    for (target, h), part in view.group_by(["target", "horizon"], maintain_order=True):
        const = part.filter(pl.col("family") == "constant")
        ref = {r["fold"]: r.get("log_loss_skill") for r in const.iter_rows(named=True)}
        if not ref:
            continue
        for (fam, fset), p in part.filter(pl.col("family") != "constant").group_by(
                ["family", "feature_set"], maintain_order=True):
            mine = {r["fold"]: r.get("log_loss_skill") for r in p.iter_rows(named=True)}
            common = [f for f in sorted(mine) if f in ref and mine[f] is not None
                      and ref[f] is not None]
            cmp = paired_blocks([mine[f] for f in common], [ref[f] for f in common])
            rows.append({"target": target, "horizon": int(h), "family": fam, "feature_set": fset,
                         "blocks": len(common),
                         "model_skill": float(np.mean([mine[f] for f in common]))
                         if common else None,
                         "recalibrated_constant_skill": float(np.mean([ref[f] for f in common]))
                         if common else None,
                         "skill_beyond_base_rate": cmp.get("mean_diff"), "se": cmp.get("se"),
                         "wins": cmp.get("wins"), "mean_auc": _mean_metric(p, "auc")})
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


_CONTROL_PREFIXES = ("null-", "noise-", "permute_features-", "pipeline-")


def null_comparison(folds: pl.DataFrame, cfg: MLConfig) -> pl.DataFrame:
    """Every control against the real model on the same blocks: real - control.

    Shifted targets, noise columns and permuted features share the real data; the
    random-walk pipeline null shares the bars and the blocks (invariant 9). A real
    model that does not beat its controls block by block has shown nothing.
    """
    view = _metric_rows(folds, cfg)
    if view.is_empty():
        return pl.DataFrame()
    base = view.filter(pl.col("variant") == "base")
    controls = view.filter(pl.any_horizontal([pl.col("variant").str.starts_with(p)
                                              for p in _CONTROL_PREFIXES]))
    rows = []
    for (target, h, fam, fset, var), part in controls.group_by(
            ["target", "horizon", "family", "feature_set", "variant"], maintain_order=True):
        metric = primary_metric(_task(cfg, str(target)))
        null_by = {r["fold"]: r.get(metric) for r in part.iter_rows(named=True)}
        real_by = {r["fold"]: r.get(metric) for r in base.filter(
            (pl.col("target") == target) & (pl.col("horizon") == h) & (pl.col("family") == fam)
            & (pl.col("feature_set") == fset)).iter_rows(named=True)}
        common = [f for f in sorted(null_by) if f in real_by and null_by[f] is not None
                  and real_by[f] is not None]
        cmp = paired_blocks([real_by[f] for f in common], [null_by[f] for f in common])
        kind = next(p for p in _CONTROL_PREFIXES if str(var).startswith(p)).rstrip("-")
        rows.append({"target": target, "horizon": int(h), "family": fam, "feature_set": fset,
                     "variant": var, "control": kind, "metric": metric, "blocks": len(common),
                     "real_mean": float(np.mean([real_by[f] for f in common])) if common
                     else None,
                     "control_mean": float(np.mean([null_by[f] for f in common])) if common
                     else None,
                     "real_minus_control": cmp.get("mean_diff"), "se": cmp.get("se"),
                     "wins": cmp.get("wins")})
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def decay_by_year(data: MLData, cfg: MLConfig, folds: pl.DataFrame) -> pl.DataFrame:
    """A model frozen at the start of 2011 scored on each later year (Step 50)."""
    part = folds.filter(pl.col("variant") == "decay-2011").filter(pl.col("calibration") == "raw")
    rows = []
    years = data.context["year"]
    for r in part.iter_rows(named=True):
        tspec = cfg.targets[r["target"]]
        ta = data.target(tspec, int(r["horizon"]), cfg.log_floor)
        pred = pl.read_parquet(r["path"])
        idx = pred["row"].to_numpy().astype(np.int64)
        col = f"cal_{_CAL}" if f"cal_{_CAL}" in pred.columns else "prediction"
        p = pred[col].cast(pl.Float64).to_numpy()
        base = float(np.nanmean(ta.y[:idx.min()]))
        for yr in np.unique(years[idx]):
            m = years[idx] == yr
            met = (classification_metrics(p[m], ta.y[idx][m], base_rate=base)
                   if tspec.is_classification else
                   regression_metrics(p[m], ta.y[idx][m], y_raw=ta.y_raw[idx][m],
                                      base_value=base))
            rows.append({"target": r["target"], "horizon": r["horizon"], "family": r["family"],
                         "year": int(yr), "years_after_training": int(yr) - 2011, **met})
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def retraining_pooled(data: MLData, cfg: MLConfig, folds: pl.DataFrame) -> pl.DataFrame:
    """Refit schedules scored on their concatenated 2018-2021 predictions (Step 51)."""
    part = folds.filter(pl.col("variant").str.starts_with("refit-")
                        & (pl.col("calibration") == "raw"))
    rows = []
    for (target, h, fam, var), p in part.group_by(["target", "horizon", "family", "variant"]):
        tspec = cfg.targets[str(target)]
        ta = data.target(tspec, int(h), cfg.log_floor)
        pred = load_predictions(sorted(set(p["path"].to_list())))
        idx = pred["row"].to_numpy().astype(np.int64)
        col = f"cal_{_CAL}" if f"cal_{_CAL}" in pred.columns else "prediction"
        score = pred[col].cast(pl.Float64).to_numpy()
        base = float(np.nanmean(ta.y[:idx.min()]))
        met = (classification_metrics(score, ta.y[idx], base_rate=base)
               if tspec.is_classification else
               regression_metrics(score, ta.y[idx], y_raw=ta.y_raw[idx], base_value=base))
        rows.append({"target": target, "horizon": int(h), "family": fam,
                     "refit_every": str(var).removeprefix("refit-"), "refits": int(p.height),
                     **met})
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


# ---------------------------------------------------------------------------
# Explanations: SHAP / permutation stability
# ---------------------------------------------------------------------------
def shap_stability(explains: list[dict[str, Any]]) -> dict[str, pl.DataFrame]:
    """Mean |SHAP| per feature per fold, its rank agreement across folds and years."""
    rows, year_rows, perm_rows, dir_rows, ale_rows = [], [], [], [], []
    for e in explains:
        ex = e["explain"]
        ident = {k: e[k] for k in ("target", "horizon", "family", "feature_set", "variant",
                                   "fold", "fold_index")}
        for f, v in (ex.get("shap_mean_abs") or {}).items():
            rows.append({**ident, "feature": f, "mean_abs_shap": float(v)})
        for yr, d in (ex.get("shap_by_year") or {}).items():
            for f, v in d.items():
                year_rows.append({**ident, "year": int(yr), "feature": f,
                                  "mean_abs_shap": float(v)})
        for f, v in (ex.get("permutation") or {}).items():
            perm_rows.append({**ident, "feature": f.removeprefix("missing__"),
                              "is_indicator": f.startswith("missing__"),
                              "importance": float(v)})
        for f, v in (ex.get("shap_direction") or {}).items():
            dir_rows.append({**ident, "feature": f, "direction": v})
        for f, curve in (ex.get("ale") or {}).items():
            edges = curve.get("edges_original") or curve.get("edges") or []
            for x, a in zip(edges, curve.get("ale") or [], strict=False):
                ale_rows.append({**ident, "feature": f, "x": float(x), "ale": float(a)})
    shap = pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()
    stab = []
    if not shap.is_empty():
        for key, part in shap.group_by(["target", "horizon", "family", "feature_set",
                                        "variant"]):
            wide = part.pivot(on="fold", index="feature", values="mean_abs_shap").fill_null(0.0)
            fold_cols = [c for c in wide.columns if c != "feature"]
            corrs = []
            tops = []
            for i in range(len(fold_cols)):
                a = wide[fold_cols[i]].to_numpy()
                tops.append(set(wide.sort(fold_cols[i], descending=True)["feature"]
                                .head(5).to_list()))
                for j in range(i + 1, len(fold_cols)):
                    b = wide[fold_cols[j]].to_numpy()
                    if a.std() > 0 and b.std() > 0:
                        corrs.append(float(np.corrcoef(rank_scores(a), rank_scores(b))[0, 1]))
            common = set.intersection(*tops) if tops else set()
            mean_share = wide.with_columns(
                pl.mean_horizontal(fold_cols).alias("mean")).sort("mean", descending=True)
            only_one = []
            for r in wide.iter_rows(named=True):
                vals = np.array([r[c] for c in fold_cols])
                if vals.max() > 0 and (vals >= 0.5 * vals.max()).sum() == 1 and \
                        r["feature"] in set(mean_share["feature"].head(8).to_list()):
                    only_one.append(r["feature"])
            stab.append({"target": key[0], "horizon": key[1], "family": key[2],
                         "feature_set": key[3], "variant": key[4], "folds": len(fold_cols),
                         "mean_rank_corr_across_folds": float(np.mean(corrs)) if corrs else None,
                         "top5_in_every_fold": sorted(common),
                         "top_features": mean_share["feature"].head(8).to_list(),
                         "flag_important_in_one_fold_only": only_one})
    return {"shap": shap,
            "shap_by_year": pl.DataFrame(year_rows, infer_schema_length=None)
            if year_rows else pl.DataFrame(),
            "shap_stability": pl.DataFrame(stab, infer_schema_length=None)
            if stab else pl.DataFrame(),
            "permutation": pl.DataFrame(perm_rows, infer_schema_length=None)
            if perm_rows else pl.DataFrame(),
            "shap_direction": pl.DataFrame(dir_rows, infer_schema_length=None)
            if dir_rows else pl.DataFrame(),
            "ale": pl.DataFrame(ale_rows, infer_schema_length=None)
            if ale_rows else pl.DataFrame()}


# ---------------------------------------------------------------------------
# Prediction correlations, disagreement, distribution shift, failures
# ---------------------------------------------------------------------------
def prediction_views(data: MLData, cfg: MLConfig, folds: pl.DataFrame
                     ) -> dict[str, pl.DataFrame]:
    """Across the families of each tree-comparison pair: correlations and disagreement."""
    from ..ml.diagnostics import disagreement

    base = folds.filter((pl.col("variant") == "base") & (pl.col("calibration") == "raw")
                        & (pl.col("feature_set") == cfg.default_feature_set))
    corr_rows, dis_rows, multi_rows = [], [], []
    shared: dict[str, pl.DataFrame] = {}
    for target, h in cfg.comparison_pairs():
        part = base.filter((pl.col("target") == target) & (pl.col("horizon") == h)
                           & (pl.col("family") != "constant"))
        if part.is_empty():
            continue
        tspec = cfg.targets[target]
        joined: pl.DataFrame | None = None
        for (fam,), p in part.group_by(["family"]):
            pred = load_predictions(sorted(set(p["path"].to_list())))
            col = f"cal_{_CAL}" if f"cal_{_CAL}" in pred.columns else "prediction"
            f = pred.select("row", pl.col(col).cast(pl.Float64).alias(str(fam)))
            joined = f if joined is None else joined.join(f, on="row", how="inner")
        if joined is None or joined.width < 3:
            continue
        fams = [c for c in joined.columns if c != "row"]
        mat = joined.select(fams).to_numpy()
        for i, a in enumerate(fams):
            for j, b in enumerate(fams):
                if j <= i:
                    continue
                corr_rows.append({"target": target, "horizon": h, "a": a, "b": b,
                                  "pearson": float(np.corrcoef(mat[:, i], mat[:, j])[0, 1]),
                                  "spearman": float(np.corrcoef(rank_scores(mat[:, i]),
                                                                rank_scores(mat[:, j]))[0, 1])})
        rows = joined["row"].to_numpy().astype(np.int64)
        dis = disagreement({f: mat[:, k] for k, f in enumerate(fams)})
        q = np.clip((rank_scores(dis) * 5).astype(np.int64), 0, 4)
        ta = data.target(tspec, h, cfg.log_floor)
        consensus = mat.mean(axis=1)
        for quint in range(5):
            m = q == quint
            entry: dict[str, Any] = {"target": target, "horizon": h,
                                     "disagreement_quintile": quint + 1, "n": int(m.sum()),
                                     "mean_disagreement": float(dis[m].mean())}
            for gname in ("vol_quartile", "spread_quartile", "regime_state"):
                if gname in data.context:
                    g = data.context[gname][rows][m]
                    g = g[g >= 0]
                    entry[f"mean_{gname}"] = float(g.mean()) if g.size else None
            if "regime_entropy" in data.features:
                ent = data.features["regime_entropy"][rows][m]
                entry["mean_regime_entropy"] = float(np.nanmean(ent)) if np.isfinite(
                    ent).any() else None
            y = ta.y[rows][m]
            c = consensus[m]
            if tspec.is_classification:
                entry.update({"consensus_confidence": float(np.mean(np.abs(c - 0.5))),
                              "consensus_auc": classification_metrics(c, y).get("auc")})
            else:
                entry["consensus_rank_ic"] = regression_metrics(c, y).get("rank_ic")
            dis_rows.append(entry)
        if "lightgbm" in fams:
            shared[f"{target}_h{h}"] = joined.select("row", pl.col("lightgbm").alias(
                f"{target}_h{h}"))
    if len(shared) >= 2:
        keys = list(shared)
        joined = shared[keys[0]]
        for k in keys[1:]:
            joined = joined.join(shared[k], on="row", how="inner")
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                x, y = joined[a].to_numpy(), joined[b].to_numpy()
                multi_rows.append({"a": a, "b": b, "n": int(joined.height),
                                   "spearman": float(np.corrcoef(rank_scores(x),
                                                                 rank_scores(y))[0, 1])})
    return {"prediction_correlation": pl.DataFrame(corr_rows, infer_schema_length=None),
            "disagreement": pl.DataFrame(dis_rows, infer_schema_length=None),
            "multitask_correlation": pl.DataFrame(multi_rows, infer_schema_length=None)}


def shift_views(data: MLData, cfg: MLConfig, folds: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Feature shift per block and performance by out-of-distribution decile (Steps 65-66)."""
    from ..ml.diagnostics import OODScorer, shift_table
    from ..ml.splits import walk_forward_folds

    names = data.feature_set(cfg.default_feature_set)
    x = data.design(names, key=cfg.default_feature_set)
    shift_rows, ood_rows = [], []
    fold_list = walk_forward_folds(data.timestamps, cfg.walk_forward, horizon=5)
    base = folds.filter((pl.col("variant") == "base") & (pl.col("calibration") == "raw")
                        & (pl.col("feature_set") == cfg.default_feature_set))
    for fold in fold_list:
        tr = x[fold.fit[0]:fold.fit[1]]
        va = x[fold.validate[0]:fold.validate[1]]
        for r in shift_table(tr, va, names):
            shift_rows.append({"fold": fold.name, **r})
        scorer = OODScorer.fit(tr)
        pct = scorer.percentile(va)
        min_rows = min(500, max(50, va.shape[0] // 20))     # 500 per decile on real blocks
        for (target, h, fam), p in base.filter(pl.col("fold") == fold.name).group_by(
                ["target", "horizon", "family"]):
            if (str(target), int(h)) not in set(cfg.focus) or fam == "constant":
                continue
            tspec = cfg.targets[str(target)]
            pred = pl.read_parquet(str(p["path"][0]))
            rows = pred["row"].to_numpy().astype(np.int64)
            col = f"cal_{_CAL}" if f"cal_{_CAL}" in pred.columns else "prediction"
            score = pred[col].cast(pl.Float64).to_numpy()
            ta = data.target(tspec, int(h), cfg.log_floor)
            local = pct[rows - fold.validate[0]]
            dec = np.clip((local * 10).astype(np.int64), 0, 9)
            base_rate = float(np.nanmean(ta.y[fold.fit[0]:fold.fit[1]]))
            for d in range(10):
                m = dec == d
                if m.sum() < min_rows:
                    continue
                met = (classification_metrics(score[m], ta.y[rows][m], base_rate=base_rate)
                       if tspec.is_classification else
                       regression_metrics(score[m], ta.y[rows][m], base_value=base_rate))
                ood_rows.append({"fold": fold.name, "target": target, "horizon": int(h),
                                 "family": fam, "ood_decile": d + 1, **met})
    return {"distribution_shift": pl.DataFrame(shift_rows, infer_schema_length=None),
            "ood_performance": pl.DataFrame(ood_rows, infer_schema_length=None)}


def failure_analysis(data: MLData, cfg: MLConfig, folds: pl.DataFrame) -> pl.DataFrame:
    """Mean-reversion model errors (confident reversion that did not happen) by condition."""
    part = folds.filter((pl.col("target") == "mean_reversion") & (pl.col("horizon") == 5)
                        & (pl.col("family") == "lightgbm") & (pl.col("variant") == "base")
                        & (pl.col("feature_set") == cfg.default_feature_set)
                        & (pl.col("calibration") == "raw"))
    if part.is_empty():
        return pl.DataFrame()
    pred = load_predictions(sorted(set(part["path"].to_list())))
    rows = pred["row"].to_numpy().astype(np.int64)
    col = f"cal_{_CAL}" if f"cal_{_CAL}" in pred.columns else "prediction"
    p = pred[col].cast(pl.Float64).to_numpy()
    ta = data.target(cfg.targets["mean_reversion"], 5, cfg.log_floor)
    y = ta.y[rows]
    confident = p >= np.quantile(p, 0.9)
    failed = confident & (y < 0.5)
    out = []
    conds: dict[str, np.ndarray] = {}
    for g in ("regime_state", "vol_quartile", "spread_quartile", "session"):
        if g in data.context:
            conds[g] = data.context[g][rows]
    feats = data.features
    for name, bins in (("reg_slope_vol_128", 5), ("reg_resid_z_128", 5), ("ou_valid_256", 2),
                       ("fft_entropy_256", 5), ("wav_entropy_512", 5)):
        if name in feats:
            v = feats[name][rows].astype(np.float64)
            code = np.full(v.size, -1)
            ok = np.isfinite(v)
            if bins == 2:
                code[ok] = (v[ok] > 0.5).astype(np.int64)
            else:
                code[ok] = np.clip((rank_scores(np.abs(v[ok]) if "z" in name else v[ok]) * bins)
                                   .astype(np.int64), 0, bins - 1)
            conds[name] = code
    for g, codes in conds.items():
        for c in np.unique(codes):
            if c < 0:
                continue
            m = codes == c
            conf = confident & m
            if conf.sum() < 200:
                continue
            out.append({"condition": g, "code": int(c), "rows": int(m.sum()),
                        "confident_rows": int(conf.sum()),
                        "failure_rate_when_confident": float(failed[m].sum() / conf.sum()),
                        "overall_failure_rate_when_confident": float(failed.sum()
                                                                     / confident.sum())})
    return pl.DataFrame(out, infer_schema_length=None)


def export_oos_predictions(ctx: Any, folds: pl.DataFrame) -> list[str]:
    """Walk-forward out-of-sample predictions of every base model of the tree-comparison
    pairs, one Parquet per target x horizon (Steps 61, 74; the input of ensemble research).

    Columns: ``timestamp``, ``row``, ``fold`` (the validation block), ``target``,
    ``horizon``, ``dataset_version``, ``label`` (the fitted scale) and ``label_raw``,
    then one column per model ``<family>|<feature set>`` - its raw prediction - and,
    for classification, ``<family>|<feature set>|platt`` (calibrated on the inner
    slice). The sidecar JSON gives each model column's family, set, feature-set id and
    blocks. Development rows only; probabilities and expected values, never a trade.
    """
    cfg: MLConfig = ctx.cfg
    data: MLData = ctx.data
    out = ensure_dir(ctx.out_dir / "predictions")
    base = folds.filter((pl.col("variant") == "base") & (pl.col("calibration") == "raw")
                        & ~pl.col("feature_set").str.starts_with("ablation_"))
    written = []
    for target, h in cfg.comparison_pairs():
        part = base.filter((pl.col("target") == target) & (pl.col("horizon") == h))
        if part.is_empty():
            continue
        tspec = cfg.targets[target]
        frames, blocks, models = [], [], {}
        for (fam, fset), p in part.group_by(["family", "feature_set"], maintain_order=True):
            pred = load_predictions(sorted(set(p["path"].to_list())))
            name = f"{fam}|{fset}"
            cols = [pl.col("row").cast(pl.Int64), pl.col("prediction").cast(pl.Float32)
                    .alias(name)]
            if tspec.is_classification and "cal_platt" in pred.columns:
                cols.append(pl.col("cal_platt").cast(pl.Float32).alias(f"{name}|platt"))
            frames.append(pred.select(cols))
            blocks.append(pred.select(pl.col("row").cast(pl.Int64), "fold"))
            key = str(fset) if fset in data.manifests else f"target_{tspec.kind}"
            models[name] = {"family": fam, "feature_set": fset,
                            "feature_set_id": data.manifest_ids.get(key),
                            "blocks": sorted(p["fold"].unique().to_list())}
        frame = pl.concat(blocks).unique(subset="row", keep="first")
        for f in frames:
            frame = frame.join(f, on="row", how="left")
        frame = frame.sort("row")
        rows = frame["row"].to_numpy()
        ta = data.target(tspec, h, cfg.log_floor)
        frame = frame.with_columns(
            pl.Series("timestamp", data.timestamps.gather(rows)),
            pl.lit(target).alias("target"), pl.lit(h, dtype=pl.Int16).alias("horizon"),
            pl.lit(data.versions.get("tick_dataset_version")).alias("dataset_version"),
            pl.Series("label", ta.y[rows].astype(np.float32)),
            pl.Series("label_raw", ta.y_raw[rows].astype(np.float32)))
        lead = ["timestamp", "row", "fold", "target", "horizon", "dataset_version", "label",
                "label_raw"]
        frame = frame.select([*lead, *[c for c in frame.columns if c not in lead]])
        path = out / f"{target}_h{h}.parquet"
        frame.write_parquet(path)
        write_json(path.with_suffix(".json"), {
            "timeframe": ctx.timeframe, "target": target, "horizon": h, "task": tspec.task,
            "rows": frame.height, "first": str(frame["timestamp"][0]),
            "last": str(frame["timestamp"][-1]), "models": models,
            "versions": {k: data.versions.get(k) for k in (
                "tick_dataset_version", "bar_dataset_version", "factory_version",
                "target_version")},
            "note": "walk-forward out-of-sample predictions on development rows "
                    "(before the reserved start); probabilities / expected values only"})
        written.append(path.name)
    return written


# ---------------------------------------------------------------------------
# Ledger, summary, one timeframe's report
# ---------------------------------------------------------------------------
def ledger_entries(timeframe: str, summ: pl.DataFrame, registry_path: Path,
                   versions: dict[str, Any], *, posthoc_variants: frozenset[str] = frozenset()
                   ) -> list[dict[str, Any]]:
    """One permanent ``ML-H`` id per tried configuration (Steps 95-96); the variants of
    post-hoc controls are marked as such in their definition and details."""
    from ..alpha.ranking import TestRegistry

    if summ.is_empty():
        return []
    registry = TestRegistry(registry_path, prefix="ML-H")
    rows = list(summ.iter_rows(named=True))
    keys = [f"{timeframe}|{r['target']}|h{r['horizon']}|{r['family']}|{r['feature_set']}|"
            f"{r['variant']}" for r in rows]
    fams = [f"{timeframe}|{r['target']}|h{r['horizon']}" for r in rows]
    ids = registry.assign(keys, fams, preregistered=False)
    out = []
    for tid, fam, r in zip(ids, fams, rows, strict=True):
        blocks = int(r["blocks"])
        beats = int(r["blocks_beating_baseline"])
        posthoc = r["variant"] in posthoc_variants
        out.append({
            "hypothesis_id": tid, "timeframe": timeframe, "input_series": "ml_research",
            "study": f"ml_research/{timeframe}", "test_family": fam, "preregistered": False,
            "definition": (("POST-HOC control: " if posthoc else "")
                           + f"{r['family']} on {r['feature_set']} ({r['variant']}) predicting "
                           f"{r['target']} at h={r['horizon']}, walk-forward "
                           f"{blocks} blocks"),
            "feature": f"{r['family']}/{r['feature_set']}/{r['variant']}",
            "target": r["target"], "horizon": int(r["horizon"]), "window": blocks,
            "metric": r["metric"], "value": r["mean"],
            "verdict": (f"beats_baseline_{beats}_of_{blocks}" if r["mean"] is not None
                        else "undefined"),
            "details": {**{k: r.get(k) for k in ("se", "min", "mean_auc", "mean_rank_ic",
                                                 "mean_mse_skill", "recent_metric",
                                                 "fit_seconds")}, "post_hoc": posthoc},
            "dataset_version": versions.get("factory_version"),
            "feature_version": versions.get("factory_version"),
            "source_feed": versions.get("source_feed")})
    return out


def write_target_views(out: Path, cfg: MLConfig, folds: pl.DataFrame,
                       summ: pl.DataFrame) -> list[str]:
    """``<tf>/<target>/h<h>/{comparisons, baselines, <family>, calibration, final_test}``
    (Step 99): per-target copies of the report tables, for browsing one problem."""
    import shutil

    written = []
    tables = out / "tables"
    reliability = (pl.read_parquet(tables / "reliability.parquet")
                   if (tables / "reliability.parquet").exists() else pl.DataFrame())
    linear = set(cfg.linear_families) | {"constant"}
    for (target, h), part in summ.group_by(["target", "horizon"], maintain_order=True):
        d = out / str(target) / f"h{int(h)}"
        write_table(part, d / "comparisons" / "configuration_summary")
        fp = folds.filter((pl.col("target") == target) & (pl.col("horizon") == h)
                          & (pl.col("variant") == "base")).drop("path")
        write_table(fp.filter(pl.col("family").is_in(sorted(linear))),
                    d / "baselines" / "fold_metrics")
        for (fam,), fpart in fp.filter(~pl.col("family").is_in(sorted(linear))).group_by(
                ["family"], maintain_order=True):
            write_table(fpart, d / str(fam) / "fold_metrics")
        if not reliability.is_empty():
            write_table(reliability.filter((pl.col("target") == target)
                                           & (pl.col("horizon") == h)),
                        d / "calibration" / "reliability")
        for js in sorted((out / "final_test").glob("*_V[0-9][0-9][0-9].json")):
            r = _json(js)
            if r.get("target") == target and int(r.get("horizon", -1)) == int(h):
                ensure_dir(d / "final_test")
                shutil.copyfile(js, d / "final_test" / js.name)
        written.append(f"{target}/h{int(h)}")
    return written


def build_ml_report(ctx: Any, *, ledger: bool = True, plots: bool = True,
                    heavy: bool = True) -> dict[str, Any]:
    """Every table of one timeframe, the ledger rows, summary.json and the figures."""
    from .research_ledger import ResearchLedger

    cfg: MLConfig = ctx.cfg
    data: MLData = ctx.data
    out = ctx.out_dir
    tables = ensure_dir(out / "tables")
    folds, explains = collect_units(out, with_explain=True)
    if folds.is_empty():
        return {"timeframe": ctx.timeframe, "units": 0}
    write_table(folds.drop("path"), tables / "fold_metrics")
    summ = summarise(folds, cfg)
    write_table(summ, tables / "configuration_summary")
    write_table(paired_vs(folds, cfg, reference="linear"), tables / "paired_vs_linear")
    write_table(paired_vs(folds, cfg, reference="constant"), tables / "paired_vs_constant")
    write_table(skill_beyond_base_rate(folds, cfg), tables / "skill_beyond_base_rate")
    for name, frame in research_tables(folds, cfg).items():
        if not frame.is_empty():
            write_table(frame, tables / name)
    from .model_ablation import ablation_table

    write_table(ablation_table(folds, cfg), tables / "ablation_deltas")
    write_table(null_comparison(folds, cfg), tables / "null_comparison")
    if heavy:
        for name, frame in pooled_tables(data, cfg, folds).items():
            if not frame.is_empty():
                write_table(frame, tables / name)
        for name, frame in shap_stability(explains).items():
            if not frame.is_empty():
                write_table(frame, tables / name)
        for name, frame in prediction_views(data, cfg, folds).items():
            if not frame.is_empty():
                write_table(frame, tables / name)
        for name, frame in shift_views(data, cfg, folds).items():
            if not frame.is_empty():
                write_table(frame, tables / name)
        for name, frame in (("decay_by_year", decay_by_year(data, cfg, folds)),
                            ("retraining_pooled", retraining_pooled(data, cfg, folds)),
                            ("failure_analysis", failure_analysis(data, cfg, folds))):
            if not frame.is_empty():
                write_table(frame, tables / name)
    predictions = export_oos_predictions(ctx, folds) if heavy else []
    summary: dict[str, Any] = {
        "timeframe": ctx.timeframe, "development_rows": data.n,
        "reserved_start": str(data.reserved_start), "units": int(folds["path"].n_unique()),
        "configurations": summ.height,
        "trials_by_family": {k: int(v) for k, v in summ.group_by("family").len().iter_rows()},
        "targets": sorted(summ["target"].unique().to_list()),
        "versions": {k: data.versions.get(k) for k in ("tick_dataset_version", "factory_version",
                                                       "target_version")},
        "predictions": predictions}
    if ledger:
        posthoc = frozenset(f"pipeline-{n}" for n in cfg.nulls.get("pipeline_posthoc") or [])
        entries = ledger_entries(ctx.timeframe, summ, cfg.results_path / "test_registry.parquet",
                                 data.versions, posthoc_variants=posthoc)
        summary["ledger_rows_upserted"] = len(entries)
        # one ML-H id per configuration: its block count (the ledger's `window`) may grow
        # between runs, and the row must be replaced, not duplicated
        summary["ledger_rows_after"] = ResearchLedger(cfg.ledger_path).upsert(
            entries, replace_by=("hypothesis_id",))
    if plots:
        from .ml_plots import write_ml_figures

        summary["figures"] = write_ml_figures(out, cfg)
    summary["target_views"] = write_target_views(out, cfg, folds, summ)
    write_json(out / "summary.json", summary)
    return summary
