r"""Summaries, the freeze rule and ledger rows of the ensemble research (Prompt #11,
Steps 54-65, 73-74, 84).

* :func:`series_summary`: every series of a pair over the evaluation blocks - mean,
  SE, worst block, spread across blocks, the recent block, mean ECE - and its
  block-by-block comparison with the simple average and with the best individual
  model chosen on earlier blocks.
* :func:`freeze_decision`: the pre-registered rule of ``config/ensemble.yaml``.
  Candidates in complexity order (best individual, simple average, median,
  conservative weights, diversity weights, stacking, regime / volatility
  conditioned, dynamic); a candidate replaces the one retained so far only when it
  is better on the primary metric block by block by more than one SE with
  ``min_wins`` blocks, and - for probabilities - does not raise the mean ECE by more
  than ``max_ece_increase``. If no ensemble earns its complexity the single model is
  retained and the simple average is still frozen as the *benchmark* ensemble.
* :func:`calibration_decision`: the final calibration of a probability average -
  the simplest of none / sigmoid / isotonic within one SE of the lowest log loss.
* :func:`ledger_entries`: one permanent ``ENS-H`` id per series of every pair (methods,
  variants, sizes, subsets, leave-outs, controls, calibration orders) - the number
  of ensemble experiments behind any result is in the ledger (Steps 73-74).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from ..ml.evaluation import paired_blocks
from .feature_research import write_json, write_table

if TYPE_CHECKING:
    from .ensemble_research import PairRun

__all__ = ["calibration_decision", "freeze_decision", "ledger_entries", "pair_report",
           "series_summary", "timeframe_report"]

_NAME_OF = {"best_individual": "best_individual", "simple_average": "simple_average",
            "median": "median", "stacking": "stacking",
            "regime_conditioned": "regime_conditioned",
            "volatility_conditioned": "volatility_conditioned", "dynamic": "dynamic"}


def _better_by_one_se(cmp: dict[str, Any]) -> bool:
    se = cmp.get("se")
    return int(cmp.get("blocks", 0)) >= 2 and se is not None and cmp["mean_diff"] > se


def series_summary(run: PairRun) -> pl.DataFrame:
    rows = []
    ks = run.eval_blocks
    recent = run.cfg.recent_block
    for name in run.series:
        vals = [run.primary(name, k) for k in ks]
        arr = np.array([v for v in vals if v is not None], dtype=np.float64)
        row: dict[str, Any] = {
            "series": name, "kind": run.kind[name], "members": len(run.members[name]),
            "member_list": ",".join(run.members[name]), "metric": run.metric,
            "blocks": int(arr.size), "mean": float(arr.mean()) if arr.size else None,
            "se": float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else None,
            "worst_block": float(arr.min()) if arr.size else None,
            "block_sd": float(arr.std(ddof=1)) if arr.size > 1 else None,
            "recent_block": (run.primary(name, recent) if recent in ks else None)}
        for extra in ("auc", "ece", "brier_skill", "rank_ic", "mse_skill", "r2", "sharpness"):
            got = [run.block_score(name, k).get(extra) for k in ks]
            v = [float(x) for x in got if x is not None and np.isfinite(x)]
            if v:
                row[f"mean_{extra}"] = float(np.mean(v))
        for ref in ("simple_average", "best_individual"):
            if ref in run.series and name != ref:
                cmp = paired_blocks(vals, [run.primary(ref, k) for k in ks])
                row[f"minus_{ref}"] = cmp.get("mean_diff")
                row[f"se_vs_{ref}"] = cmp.get("se")
                row[f"wins_vs_{ref}"] = cmp.get("wins")
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def _mean_metric(run: PairRun, name: str, metric: str) -> float | None:
    got = [run.block_score(name, k).get(metric) for k in run.eval_blocks]
    v = [float(x) for x in got if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else None


def freeze_decision(run: PairRun) -> dict[str, Any]:
    fr = run.cfg.freeze
    order = list(fr["order"])
    min_wins = int(fr.get("min_wins", 3))
    max_ece = float(fr.get("max_ece_increase", 0.005))
    names = {**_NAME_OF, "performance_weighted": run._candidate("performance_weighted"),
             "diversity_weighted": run._candidate("diversity_weighted")}
    retained = names["best_individual"]
    steps = []
    for method in order[1:]:
        s = names[method]
        if s not in run.series:
            steps.append({"method": method, "series": s, "decision": "not run"})
            continue
        cmp = paired_blocks([run.primary(s, k) for k in run.eval_blocks],
                            [run.primary(retained, k) for k in run.eval_blocks])
        better = _better_by_one_se(cmp) and int(cmp.get("wins", 0)) >= min_wins
        ece_note = None
        if run.pair.is_classification:
            a, b = _mean_metric(run, s, "ece"), _mean_metric(run, retained, "ece")
            if a is None or b is None:
                better = False
                ece_note = "ECE undefined"
            elif a - b > max_ece:
                better = False
                ece_note = f"ECE {a:.4f} vs {b:.4f}"
        steps.append({"method": method, "series": s, "versus": retained,
                      "mean_diff": cmp.get("mean_diff"), "se": cmp.get("se"),
                      "wins": cmp.get("wins"), "blocks": cmp.get("blocks"),
                      "ece_note": ece_note,
                      "decision": "replaces" if better else "does not earn its complexity"})
        if better:
            retained = s
    method_of = {v: k for k, v in names.items()}
    single = retained == names["best_individual"]
    # what is frozen is what the rule retained: the single model when no ensemble earned
    # its complexity (the simple average is evaluated beside it as the benchmark)
    frozen_method = method_of[retained]
    return {"retained": retained, "retained_method": method_of[retained],
            "single_model_retained": single, "frozen_method": frozen_method,
            "frozen_series": names[frozen_method],
            "role": ("single model retained (no ensemble earned its complexity in "
                     "development; the simple average is evaluated beside it)") if single
            else "candidate", "steps": steps, "rule": {k: fr.get(k) for k in (
                "order", "min_wins", "max_ece_increase")}}


def calibration_decision(run: PairRun) -> dict[str, Any]:
    if not run.pair.is_classification:
        return {"calibration": "none", "reason": "regression"}
    cands = [("none", "calibration:A_calibrate_then_average"),
             ("sigmoid", "calibration:C_both_sigmoid"),
             ("isotonic", "calibration:C_both_isotonic")]
    have = [(m, s) for m, s in cands if s in run.series]
    ll = {m: [run.block_score(s, k).get("log_loss") for k in run.eval_blocks] for m, s in have}
    means = {m: float(np.mean([x for x in v if x is not None])) for m, v in ll.items()
             if any(x is not None for x in v)}
    if not means:
        return {"calibration": "none", "reason": "no calibration result"}
    best = min(means, key=lambda m: means[m])
    for m, _ in have:                                   # simplest first
        if m not in means:
            continue
        if m == best:
            return {"calibration": m, "mean_log_loss": means, "reason": f"{m} has the lowest "
                    "mean log loss"}
        cmp = paired_blocks(ll[m], ll[best])           # how much higher m's loss is
        se = cmp.get("se")
        if int(cmp.get("blocks", 0)) >= 2 and se is not None and cmp["mean_diff"] <= se:
            return {"calibration": m, "mean_log_loss": means,
                    "reason": f"{m} within one SE of the lowest ({best})"}
    return {"calibration": best, "mean_log_loss": means, "reason": "lowest"}


def best_individual_for_test(run: PairRun) -> str | None:
    """The frozen best individual model: Prompt #10's frozen candidate when it is in the
    universe, otherwise the universe model with the best development mean."""
    frozen = run.tctx.ml_dir / "frozen" / "freeze_report.json"
    if frozen.exists():
        rep = json.loads(frozen.read_text(encoding="utf-8"))
        for d in rep.get("decisions", []):
            if d.get("target") == run.pair.target and int(d.get("horizon", -1)) == run.pair.horizon \
                    and d.get("frozen"):
                name = f"{d['family']}|{d['feature_set']}"
                if name in run.universe:
                    return name
    if not run.universe:
        return None
    means = {m: r["mean"] for m in run.universe for r in
             run.elig.filter(pl.col("model") == m).iter_rows(named=True)}
    return max(means, key=lambda m: means[m] if means[m] is not None else -np.inf)


def pair_report(run: PairRun, blocks: pl.DataFrame) -> dict[str, Any]:
    out = run.tctx.out_dir / run.pair.target / f"h{run.pair.horizon}"
    summ = series_summary(run)
    write_table(summ, out / "series_summary")
    decision = freeze_decision(run)
    decision["calibration"] = calibration_decision(run)
    decision["best_individual_for_final_test"] = best_individual_for_test(run)
    write_json(out / "freeze_decision.json", decision)
    head = {}
    for name in ("best_individual", "simple_average", "median", run._candidate(
            "performance_weighted"), run._candidate("diversity_weighted"), "stacking",
            "stacking_context", "regime_conditioned", "volatility_conditioned", "dynamic"):
        r = summ.filter(pl.col("series") == name)
        if r.height:
            row = r.row(0, named=True)
            head[name] = {k: row.get(k) for k in ("mean", "se", "worst_block", "block_sd",
                                                   "recent_block", "mean_ece",
                                                   "minus_best_individual",
                                                   "wins_vs_best_individual")}
    del blocks
    return {"freeze": decision, "headline": head}


# ---------------------------------------------------------------------------
# Ledger and the timeframe report
# ---------------------------------------------------------------------------
def ledger_entries(timeframe: str, pair_dir: Path, registry_path: Path,
                   versions: dict[str, Any]) -> list[dict[str, Any]]:
    """One ``ENS-H`` id per series of one pair (every ensemble experiment, Step 73)."""
    from ..alpha.ranking import TestRegistry

    p = pair_dir / "series_summary.parquet"
    js = pair_dir / "summary.json"
    if not p.exists() or not js.exists():
        return []
    summ = pl.read_parquet(p)
    meta = json.loads(js.read_text(encoding="utf-8"))
    target, h = meta["target"], int(meta["horizon"])
    keep = summ.filter(~pl.col("kind").is_in(["individual", "baseline"]))
    if keep.is_empty():
        return []
    rows = list(keep.iter_rows(named=True))
    registry = TestRegistry(registry_path, prefix="ENS-H")
    keys = [f"{timeframe}|{target}|h{h}|{r['series']}" for r in rows]
    fams = [f"{timeframe}|{target}|h{h}"] * len(rows)
    ids = registry.assign(keys, fams, preregistered=True)
    out = []
    for tid, fam, r in zip(ids, fams, rows, strict=True):
        diff, se, wins = r.get("minus_simple_average"), r.get("se_vs_simple_average"), \
            r.get("wins_vs_simple_average")
        verdict = ("simple_average" if r["series"] == "simple_average" else
                   "undefined" if diff is None else
                   f"vs_simple_average_{'better' if se is not None and diff > se else 'worse' if se is not None and diff < -se else 'within_1se'}_{wins}_of_{r['blocks']}")
        out.append({
            "hypothesis_id": tid, "timeframe": timeframe, "input_series": "ensemble_research",
            "study": f"ensemble_research/{timeframe}", "test_family": fam,
            "preregistered": r["kind"] == "method",
            "definition": (f"{r['kind']} ensemble '{r['series']}' of {r['members']} model(s) "
                           f"predicting {target} at h={h}; meta walk-forward over "
                           f"{r['blocks']} blocks, fitted on earlier out-of-sample blocks only"
                           + (" (control)" if r["kind"] == "control" else "")),
            "feature": r["series"], "target": target, "horizon": h, "window": int(r["blocks"]),
            "metric": r["metric"], "value": r["mean"],
            "control_value": None if diff is None or r["mean"] is None else r["mean"] - diff,
            "verdict": verdict,
            "details": {k: r.get(k) for k in ("se", "worst_block", "block_sd", "recent_block",
                                              "mean_ece", "minus_best_individual",
                                              "wins_vs_best_individual", "member_list")},
            "dataset_version": versions.get("factory_version"),
            "feature_version": versions.get("factory_version"), "n_obs": None})
    return out


def timeframe_report(out_dir: Path, *, timeframe: str, ledger_path: Path | None,
                     registry_path: Path, versions: dict[str, Any]) -> dict[str, Any]:
    """Cross-pair tables of one timeframe (and the ENS-H ledger rows)."""
    from .research_ledger import ResearchLedger

    tables = out_dir / "tables"
    pairs = sorted(p.parent for p in out_dir.glob("*/h*/summary.json"))
    elig: list[pl.DataFrame] = []
    summ: list[pl.DataFrame] = []
    nulls: list[pl.DataFrame] = []
    freeze: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    for d in pairs:
        meta = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        key = {"target": meta["target"], "horizon": int(meta["horizon"])}
        for name, bucket in (("model_eligibility", elig), ("series_summary", summ),
                             ("null_ensembles", nulls)):
            p = d / f"{name}.parquet"
            if p.exists():
                f = pl.read_parquet(p)
                if "target" not in f.columns:
                    f = f.with_columns(pl.lit(key["target"]).alias("target"),
                                       pl.lit(key["horizon"]).alias("horizon"))
                bucket.append(f)
        fd = meta.get("freeze") or {}
        freeze.append({**key, "universe": ",".join(meta.get("universe") or []),
                       "eligible": len(meta.get("eligible") or []),
                       "ensemble": meta.get("ensemble", True) is not None,
                       "reason": meta.get("reason"),
                       "retained": fd.get("retained"), "frozen_method": fd.get("frozen_method"),
                       "role": fd.get("role"),
                       "calibration": (fd.get("calibration") or {}).get("calibration"),
                       "best_individual_for_final_test": fd.get(
                           "best_individual_for_final_test")})
        if ledger_path is not None:
            entries += ledger_entries(timeframe, d, registry_path, versions)
    for name, bucket in (("eligibility", elig), ("series_summary", summ),
                         ("null_ensembles", nulls)):
        if bucket:
            write_table(pl.concat(bucket, how="diagonal_relaxed"), tables / name)
    write_table(pl.DataFrame(freeze, infer_schema_length=None), tables / "freeze_decisions")
    out: dict[str, Any] = {"timeframe": timeframe, "pairs": len(pairs),
                           "experiments": len(entries)}
    if ledger_path is not None and entries:
        out["ledger_rows_after"] = ResearchLedger(ledger_path).upsert(
            entries, replace_by=("hypothesis_id",))
    write_json(out_dir / "report_summary.json", out)
    return out


def joint_predictive_state(out_dir: Path, timeframe: str, ecfg: Any, *, log_floor: float
                           ) -> dict[str, Any]:
    """Step 89: the frozen methods' out-of-sample outputs of every target, joined by bar,
    as prediction-contract rows (development evaluation blocks; no decision field)."""
    from ..ensemble.contract import (
        FIELDS,
        SCHEMA,
        contract_frame,
        prediction_version,
        records_from_frame,
    )
    from ..ensemble.registry import ensemble_id

    statuses: dict[str, str] = {}
    sources: dict[str, dict[str, Any]] = {}
    horizons: dict[str, int] = {}
    frames = []
    for fname, (target, _) in FIELDS.items():
        hs = [h for t, h in ecfg.pairs if t == target]
        if not hs:
            statuses[fname] = "not_frozen"
            sources[fname] = {"ensemble_id": None, "status": "not_frozen"}
            continue
        h = int(hs[0])
        horizons[fname] = h
        d = out_dir / target / f"h{h}"
        p = d / "oos_ensemble_predictions.parquet"
        fd = (json.loads((d / "freeze_decision.json").read_text(encoding="utf-8"))
              if (d / "freeze_decision.json").exists() else {})
        if not p.exists() or not fd:
            statuses[fname] = "no_eligible_ensemble"
            sources[fname] = {"ensemble_id": None, "horizon_bars": h,
                              "status": "no_eligible_ensemble"}
            continue
        cols = ["timestamp", "frozen_method", "disagreement", "ood_score", "regime_entropy"]
        if fname == "reversion_probability":
            cols.append("frozen_entropy")
        if fname == "expected_return_vol_scaled":
            cols.append("sigma")
        f = pl.read_parquet(p, columns=cols).rename(
            {"frozen_method": fname, "disagreement": f"dis_{fname}",
             "ood_score": f"ood_{fname}", "regime_entropy": f"regent_{fname}"})
        frames.append(f)
        statuses[fname] = "ok"
        sources[fname] = {"ensemble_id": ensemble_id(target, timeframe, h,
                                                     ecfg.registry_version),
                          "horizon_bars": h, "method": fd.get("frozen_method"),
                          "role": fd.get("role"), "status": "ok"}
    if not frames:
        return {"rows": 0}
    joined = frames[0]
    for f in frames[1:]:
        joined = joined.join(f, on="timestamp", how="full", coalesce=True)
    joined = joined.sort("timestamp")
    n = joined.height

    def col(name: str) -> np.ndarray:
        return (joined[name].cast(pl.Float64).fill_null(np.nan).to_numpy()
                if name in joined.columns else np.full(n, np.nan))

    def first_of(prefix: str) -> np.ndarray:
        out = np.full(n, np.nan)
        for c in [c for c in joined.columns if c.startswith(prefix)]:
            v = col(c)
            out = np.where(np.isfinite(out), out, v)
        return out

    columns = {f: col(f) for f in FIELDS}
    columns["model_disagreement"] = col("dis_reversion_probability")
    columns["ood_score"] = first_of("ood_")
    columns["regime_entropy"] = first_of("regent_")
    columns["reversion_entropy"] = col("frozen_entropy")
    version = prediction_version(timeframe, {f: s.get("ensemble_id") for f, s in
                                             sources.items()})
    frame = contract_frame(joined["timestamp"], timeframe, version, columns, statuses,
                           col("sigma"), horizons, log_floor)
    for fld in FIELDS:
        frame = frame.with_columns(pl.Series(f"disagreement_{fld}", col(f"dis_{fld}")))
    write_table(frame, out_dir / "joint_predictive_state", csv=False)
    sample_rows = pl.concat([frame.head(3), frame.tail(3)])
    records = records_from_frame(sample_rows.drop([c for c in sample_rows.columns
                                                   if c.startswith("disagreement_")]),
                                 sources=sources, horizons=horizons, log_floor=log_floor)
    write_json(out_dir / "prediction_contract_sample.json",
               {"prediction_version": version, "sources": sources, "records": records})
    write_json(out_dir.parent / "prediction_contract_schema.json", SCHEMA)
    return {"rows": n, "prediction_version": version, "statuses": statuses}


_METHOD_FOLDERS = {
    "simple_average": ("simple_average", "median", "calibration:"),
    "weighted": ("performance_weighted_",),
    "diversity_weighted": ("diversity_weighted_",),
    "stacking": ("stacking",),
    "regime_conditioned": ("regime_conditioned", "volatility_conditioned", "dynamic")}


def method_views(pair_dir: Path) -> list[str]:
    """``<pair>/{simple_average, weighted, diversity_weighted, stacking, regime_conditioned}``:
    each family of methods' block metrics and weights, for browsing (Step 81)."""
    written: list[str] = []
    blocks = pair_dir / "block_metrics.parquet"
    if not blocks.exists():
        return written
    bm = pl.read_parquet(blocks)
    weights = (pl.read_parquet(pair_dir / "weights.parquet")
               if (pair_dir / "weights.parquet").exists() else pl.DataFrame())
    for folder, prefixes in _METHOD_FOLDERS.items():
        part = bm.filter(pl.any_horizontal([pl.col("series").str.starts_with(p)
                                            for p in prefixes]))
        if part.is_empty():
            continue
        write_table(part, pair_dir / folder / "block_metrics")
        if not weights.is_empty():
            w = weights.filter(pl.any_horizontal([pl.col("scheme").str.starts_with(p)
                                                  for p in prefixes]))
            write_table(w, pair_dir / folder / "weights")
        written.append(f"{pair_dir.parent.name}/{pair_dir.name}/{folder}")
    return written


def build_ensemble_report(cfgs: Any, timeframe: str, *, ledger: bool = True,
                          plots: bool = True) -> dict[str, Any]:
    """The timeframe tables, ``ENS-H`` ledger rows, figures and joint predictive state."""
    import shutil

    ecfg = cfgs.ensemble
    out_dir = ecfg.results_path / timeframe
    versions: dict[str, Any] = {}
    for p in sorted(out_dir.glob("*/h*/summary.json")):
        versions = (json.loads(p.read_text(encoding="utf-8")).get("versions") or {})
        if versions:
            break
    summary = timeframe_report(out_dir, timeframe=timeframe,
                               ledger_path=ecfg.ledger_path if ledger else None,
                               registry_path=ecfg.results_path / "test_registry.parquet",
                               versions=versions)
    summary["joint_predictive_state"] = joint_predictive_state(
        out_dir, timeframe, ecfg, log_floor=cfgs.ml.ml.log_floor)
    pair_dirs = sorted(p.parent for p in out_dir.glob("*/h*/summary.json"))
    summary["method_views"] = [v for d in pair_dirs for v in method_views(d)]
    if plots:
        from .ensemble_plots import write_ensemble_figures

        summary["figures"] = write_ensemble_figures(out_dir, pair_dirs)
        overview = out_dir / "plots" / "overview_methods.png"
        if overview.exists():                           # results/ensemble_research/plots/
            dest = ecfg.results_path / "plots" / f"overview_methods_{timeframe}.png"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(overview, dest)
    write_json(out_dir / "report_summary.json", summary)
    return summary
