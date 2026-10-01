r"""Summaries, multiple-testing accounting and figures of the feature selection (Prompt #9,
Steps 63-68).

Reads the stage tables of :mod:`.feature_selection_research` and writes
``summary.json`` / ``feature_selection_summary.md`` per timeframe, the trial
counts (candidate sets, regularisation strengths, PCA dimensions, mRMR
variants, feature counts, folds, target / horizon combinations) and one ledger
row per selection experiment, with a permanent ``SEL-H`` id. Failed and
unhelpful experiments are recorded like the rest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..alpha.ranking import TestRegistry
from ..selection.config import FeatureSelectionConfig
from ..utils.clock import utc_now_iso
from ..utils.paths import ensure_dir
from .feature_research import write_json, write_table
from .research_ledger import ResearchLedger

__all__ = ["build_selection_report", "step67_tables", "trial_counts",
           "write_selection_comparison"]

_LASSO_GRID = 60          # the penalties of one Lasso / elastic-net path (lasso_path n_alphas)


def _read(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path) if path.exists() else pl.DataFrame()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _kind(target: str) -> str:
    fam = target.removeprefix("target_").rsplit("_", 1)[0]
    return {"return": "direction", "residual_reduction": "reversion", "realized_vol": "volatility",
            "abs_return": "magnitude"}.get(fam, fam)


# ---------------------------------------------------------------------------
# Trials and ledger
# ---------------------------------------------------------------------------
def trial_counts(out: Path, cfg: FeatureSelectionConfig | None = None) -> dict[str, Any]:
    """Every selection experiment that was run, counted (Step 64)."""
    curve = _read(out / "feature_count_curve.parquet")
    pca = _read(out / "pca_predictive.parquet")
    stab = _read(out / "selection_stability.parquet")
    paths = _read(out / "lasso_elastic_net_selection.parquet")
    sets = _read(out / "set_curve.parquet")
    abl = _read(out / "family_ablation.parquet")
    lofo = _read(out / "leave_one_family_out.parquet")
    logit = _read(out / "logistic_l1.parquet")
    methods = sorted(curve["method"].unique().to_list()) if not curve.is_empty() else []
    counts: dict[str, Any] = {
        "nested_evaluations": curve.height, "pca_evaluations": pca.height,
        "stability_runs_target_selector": stab.height,
        "stability_resamples": (int(stab.select(pl.col("resamples").max()).item())
                                if not stab.is_empty() else 0),
        "candidate_set_evaluations": sets.height, "ablation_evaluations": abl.height,
        "leave_one_family_out_evaluations": lofo.height,
        "selection_methods": methods,
        "mrmr_variants": [m for m in methods if m.startswith("mrmr")],
        "regularisation_paths": (paths.select("split", "target", "method").unique().height
                                 if not paths.is_empty() else 0),
        "logistic_l1_targets": (logit["target"].n_unique() if not logit.is_empty() else 0),
        "feature_counts": sorted(curve["k"].unique().to_list()) if not curve.is_empty() else [],
        "pca_dimensions": sorted(pca["k"].unique().to_list()) if not pca.is_empty() else [],
        "folds": sorted(curve["split"].unique().to_list()) if not curve.is_empty() else [],
        "targets": sorted(curve["target"].unique().to_list()) if not curve.is_empty() else [],
    }
    if cfg is not None:
        counts["regularisation_strengths"] = {
            "ridge_alpha": [cfg.linear.ridge_alpha],
            "elastic_net_l2": list(cfg.linear.elastic_net_l2),
            "lasso_path_penalties_per_fit": _LASSO_GRID,
            "logistic_lambdas": list(cfg.linear.logistic_lambdas)}
        counts["candidate_set_sizes"] = list(cfg.evaluation.candidate_set_sizes)
    counts["total_evaluations"] = (curve.height + pca.height + sets.height + abl.height
                                   + lofo.height + stab.height)
    return counts


def _ledger_entries(timeframe: str, out: Path, registry: TestRegistry,
                    info: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[tuple[str, str, dict[str, Any]]] = []
    curve = _read(out / "feature_count_curve.parquet")
    for r in curve.iter_rows(named=True):
        key = f"{timeframe}|nested|{r['split']}|{r['target']}|{r['method']}|k={r['k']}"
        rows.append((key, f"{timeframe}|selection_method|{_kind(r['target'])}", {
            "definition": f"ridge on the first {r['k']} features of {r['method']} (selected on "
                          f"training rows) scored on {r['split']}",
            "feature": r["method"], "target": r["target"], "horizon": _h(r["target"]),
            "window": r["k"], "metric": "evaluation_rank_ic", "value": r.get("rank_ic"),
            "verdict": "evaluated" if r.get("rank_ic") is not None
            else "not_enough_candidates", "n_obs": r.get("n")}))
    for r in _read(out / "pca_predictive.parquet").iter_rows(named=True):
        key = f"{timeframe}|pca|{r['split']}|{r['target']}|k={r['k']}"
        rows.append((key, f"{timeframe}|pca|{_kind(r['target'])}", {
            "definition": f"ridge on {r['k']} training-fitted principal components, {r['split']}",
            "feature": "pca", "target": r["target"], "horizon": _h(r["target"]),
            "window": r["k"], "metric": "evaluation_rank_ic", "value": r.get("rank_ic"),
            "verdict": "evaluated", "n_obs": r.get("n")}))
    for r in _read(out / "selection_stability.parquet").iter_rows(named=True):
        key = f"{timeframe}|stability|{r['target']}|{r['selector']}"
        rows.append((key, f"{timeframe}|stability_selection|{r['kind']}", {
            "definition": f"{r['selector']} top-{r['k']} on {r['resamples']} quarter resamples "
                          "of development", "feature": r["selector"], "target": r["target"],
            "horizon": r["horizon"], "window": r["k"], "metric": "mean_pairwise_jaccard",
            "value": r.get("mean_pairwise_jaccard"), "verdict": "evaluated"}))
    for name, table in (("ablation", "family_ablation"), ("lofo", "leave_one_family_out")):
        for r in _read(out / f"{table}.parquet").iter_rows(named=True):
            step = r.get("step") or r.get("variant")
            key = f"{timeframe}|{name}|{r['split']}|{r['target']}|{step}"
            rows.append((key, f"{timeframe}|family_{name}|{_kind(r['target'])}", {
                "definition": f"{name} {step}, {r['split']}", "feature": str(step),
                "target": r["target"], "horizon": _h(r["target"]), "window": r.get("features"),
                "metric": "evaluation_rank_ic", "value": r.get("rank_ic"),
                "verdict": "evaluated", "n_obs": r.get("n")}))
    for r in _read(out / "set_curve.parquet").iter_rows(named=True):
        key = f"{timeframe}|candidate_set|k={r['k']}|{r['target']}"
        rows.append((key, f"{timeframe}|candidate_sets|{r['kind']}", {
            "definition": f"first {r['k']} features of the development stability ranking, "
                          "scored on validation", "feature": f"set_k{r['k']}",
            "target": r["target"], "horizon": r["horizon"], "window": r["k"],
            "metric": "validation_rank_ic", "value": r.get("rank_ic"), "verdict": "evaluated",
            "n_obs": r.get("n")}))
    keys = [k for k, _, _ in rows]
    families = [f for _, f, _ in rows]
    ids = registry.assign(keys, families, preregistered=False)
    entries = []
    for tid, (_, fam, r) in zip(ids, rows, strict=True):
        n_obs = r.get("n_obs")
        entries.append({"hypothesis_id": tid, "timeframe": timeframe,
                        "input_series": "feature_selection", "study": f"feature_selection/"
                        f"{timeframe}", "test_family": fam, "preregistered": False,
                        "dataset_version": info.get("factory_version"),
                        "feature_version": info.get("factory_version"),
                        "source_feed": info.get("source_feed"), **r,
                        "value": _num(r.get("value")),
                        "window": int(r["window"]) if r.get("window") is not None else None,
                        "n_obs": int(n_obs) if n_obs is not None else None})
    return entries


def _h(target: str) -> int | None:
    try:
        return int(target.rsplit("_", 1)[1])
    except (ValueError, IndexError):
        return None


def _num(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def _mrmr_vs_ic(orders: pl.DataFrame, k: int = 20) -> list[dict[str, Any]]:
    final = orders.filter(pl.col("split") == "development_to_validation")
    rows = []
    for (target,), part in final.group_by(["target"]):
        a = set(part.filter((pl.col("method") == "mrmr_ic_difference_1") & (pl.col("rank") <= k))
                ["feature"].to_list())
        b = set(part.filter((pl.col("method") == "ic_rank") & (pl.col("rank") <= k))
                ["feature"].to_list())
        union = a | b
        rows.append({"target": target, "k": k, "mrmr": len(a), "ic_rank": len(b),
                     "jaccard": len(a & b) / len(union) if union else None,
                     "only_mrmr": sorted(a - b), "only_ic_rank": sorted(b - a)})
    return sorted(rows, key=lambda r: r["target"])


def _l1_consistent(paths: pl.DataFrame, k: int = 20) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, list[str]]] = {}
    if paths.is_empty():
        return out
    splits = paths["split"].n_unique()
    top = paths.filter(pl.col("entry_rank") <= k)
    for (method, target), part in top.group_by(["method", "target"]):
        counts = part.group_by("feature").agg(pl.col("split").n_unique().alias("splits"),
                                              (pl.col("refit_coefficient") > 0).all()
                                              .alias("always_positive"),
                                              (pl.col("refit_coefficient") < 0).all()
                                              .alias("always_negative"))
        keep = counts.filter((pl.col("splits") >= max(1, splits - 1))
                             & (pl.col("always_positive") | pl.col("always_negative")))
        out.setdefault(str(method), {})[str(target)] = sorted(keep["feature"].to_list())
    return out


def step67_tables(out: Path) -> list[str]:
    """The per-method tables under the names of the Prompt #9 output layout (Step 67)."""
    written = []
    orders = _read(out / "method_orderings.parquet")
    if not orders.is_empty():
        write_table(orders.filter(pl.col("method").str.starts_with("mrmr")), out / "mrmr_results")
        written.append("mrmr_results")
    paths = _read(out / "lasso_elastic_net_selection.parquet")
    if not paths.is_empty():
        write_table(paths.filter(pl.col("method") == "lasso"), out / "lasso_selection")
        write_table(paths.filter(pl.col("method").str.starts_with("elastic_net")),
                    out / "elastic_net_selection")
        written += ["lasso_selection", "elastic_net_selection"]
    extended = ((_json(out / "sets.json").get("sets") or {}).get("extended") or [])
    if extended:
        from .feature_selection_plots import effective_rank_curve

        curve = effective_rank_curve(out / "cache", extended)
        if not curve.is_empty():
            write_table(curve, out / "effective_rank")
            written.append("effective_rank")
    return written


def _provenance(out: Path) -> dict[str, Any]:
    for path in sorted((out / "manifests").glob("*.json")):
        m = _json(path)
        return {k: m.get(k) for k in ("factory_version", "source_feed", "dataset_version",
                                      "partial_dataset")}
    return {}


def build_selection_report(results_path: Path, timeframe: str, *, ledger_path: Path | None,
                           cfg: FeatureSelectionConfig | None = None,
                           make_plots: bool = True) -> dict[str, Any]:
    out = results_path / timeframe
    universe = _json(out / "universe.json")
    filtering = _read(out / "filtering.parquet")
    clusters = _json(out / "redundancy_clusters.json")
    sets = _json(out / "sets.json")
    audit = _json(out / "leakage_audit.json")
    live = _json(out / "live_reconstruction.json")
    split = _json(out / "period_split.json")
    reasons: dict[str, int] = {}
    for r in filtering.iter_rows(named=True):
        for code in (r.get("reasons") or "").split(","):
            if code:
                reasons[code] = reasons.get(code, 0) + 1
    freq = _read(out / "selection_frequency.parquet")
    thr = cfg.stability.frequency_threshold if cfg is not None else 0.6
    stable: dict[str, list[str]] = {}
    unstable: list[str] = []
    if not freq.is_empty():
        # as the sets read it: mean over the two selectors per target, best target per kind
        per_target = freq.filter(~pl.col("is_probe")).group_by(["feature", "target", "kind"]) \
            .agg(pl.col("frequency").mean().alias("score"))
        per_kind = per_target.group_by(["feature", "kind"]).agg(pl.col("score").max()
                                                                .alias("best"))
        for (kind,), part in per_kind.filter(pl.col("best") >= thr).group_by(["kind"]):
            stable[str(kind)] = part.sort(["best", "feature"], descending=[True, False]) \
                ["feature"].to_list()
        best_any = per_kind.group_by("feature").agg(pl.col("best").max())
        unstable = sorted(best_any.filter((pl.col("best") > 0.2) & (pl.col("best") < thr))
                          ["feature"].to_list())
    stopping = _read(out / "selection_stability.parquet")
    stopping_rows = (stopping.select("target", "selector", "real_candidates", "mean_selected",
                                     "median_first_probe_position", "probe_first_share",
                                     "mean_pairwise_jaccard").sort(["target", "selector"])
                     .to_dicts() if "mean_selected" in stopping.columns else [])
    pca = _read(out / "pca_predictive.parquet")
    curve = _read(out / "feature_count_curve.parquet")
    pca_vs_raw = []
    if not pca.is_empty() and not curve.is_empty():
        final_pca = pca.filter(pl.col("split") == "development_to_validation")
        raw = curve.filter((pl.col("split") == "development_to_validation")
                           & (pl.col("method") == "mrmr_ic_difference_1"))
        pca_vs_raw = final_pca.join(raw.select("target", "k", pl.col("rank_ic").alias("raw_ic")),
                                    on=["target", "k"], how="left").select(
            "target", "k", "explained_variance", pl.col("rank_ic").alias("pca_ic"),
            "raw_ic").sort(["target", "k"]).to_dicts()
    lofo = _read(out / "leave_one_family_out.parquet")
    lofo_summary = (lofo.filter((pl.col("split") == "development_to_validation")
                                & (pl.col("variant") != "all"))
                    .group_by("variant").agg(pl.col("delta_rank_ic").mean().alias("mean_delta"),
                                             pl.col("delta_rank_ic").min().alias("min_delta"))
                    .sort("mean_delta").to_dicts() if not lofo.is_empty() else [])
    ablation = _read(out / "family_ablation.parquet")
    ablation_summary = (ablation.filter(pl.col("split") == "development_to_validation")
                        .group_by("step", maintain_order=True)
                        .agg(pl.col("rank_ic").mean().alias("mean_rank_ic")).to_dicts()
                        if not ablation.is_empty() else [])
    recent = _read(out / "recent_relevance.parquet")
    recent_counts = (recent.group_by("recent_relevance").len().to_dicts()
                     if not recent.is_empty() else [])
    regime = _read(out / "regime_robustness.parquet")
    regime_specific = (sorted(regime.filter(pl.col("regime_class") == "regime_specific")
                              ["feature"].unique().to_list()) if not regime.is_empty() else [])
    cost = _read(out / "cost.parquet").to_dicts() if (out / "cost.parquet").exists() else []
    coll = _read(out / "collinearity.parquet").to_dicts() if (out / "collinearity.parquet") \
        .exists() else []
    trials = trial_counts(out, cfg)
    run_info = _json(out / "run_info.json")
    provenance = _provenance(out)
    summary: dict[str, Any] = {
        "timeframe": timeframe, "generated_utc": utc_now_iso(),
        "provenance": provenance,
        "period_split": split,
        "registered_features": universe.get("registered"),
        "quality_universe": len(universe.get("quality_universe", [])),
        "selection_universe": len(universe.get("selection_universe", [])),
        "exclusion_reasons": reasons,
        "probes_passing_development_null": universe.get("probes_passing_development_null"),
        "largest_clusters": [{k: c[k] for k in ("size", "representative", "members")}
                             for c in (clusters.get("largest_clusters") or [])[:6]],
        "effective_rank": {"quality_universe": clusters.get("effective_rank_quality_universe"),
                           "selection_universe": clusters.get(
                               "effective_rank_selection_universe")},
        "stability_threshold": thr, "stable_by_kind": stable,
        "partially_stable_features": unstable, "probe_stopping": stopping_rows,
        "mrmr_vs_ic_ranking": _mrmr_vs_ic(_read(out / "method_orderings.parquet")),
        "l1_consistent": _l1_consistent(_read(out / "lasso_elastic_net_selection.parquet")),
        "pca_vs_raw": pca_vs_raw, "family_ablation": ablation_summary,
        "leave_one_family_out": lofo_summary, "recent_relevance": recent_counts,
        "regime_specific_features": regime_specific,
        "sets": {k: {"features": len(v), "justified": sets.get("justified", {}).get(k)}
                 for k, v in (sets.get("sets") or {}).items()},
        "plateau": sets.get("plateau"), "cost": cost, "collinearity": coll,
        "leakage_audit_all_verified": audit.get("all_verified"),
        "live_reconstruction": {k: live.get(k) for k in ("steps", "passed", "failed",
                                                         "seconds_per_bar", "buffer_bars")},
        "reserved_isolation": audit.get("reserved_isolation"),
        "trials": trials, "run_seconds": run_info.get("seconds"),
    }
    if ledger_path is not None:
        registry = TestRegistry(results_path / "test_registry.parquet", prefix="SEL-H")
        entries = _ledger_entries(timeframe, out, registry, provenance)
        summary["ledger_rows_upserted"] = len(entries)
        summary["ledger_rows_after"] = ResearchLedger(ledger_path).upsert(entries)
    summary["layout_tables"] = step67_tables(out)
    if make_plots:
        from .feature_selection_plots import write_selection_figures

        summary["figures"] = write_selection_figures(out, threshold=thr)
    write_json(out / "summary.json", summary)
    (out / "feature_selection_summary.md").write_text(_markdown(summary, sets), encoding="utf-8")
    return summary


def _markdown(s: dict[str, Any], sets: dict[str, Any]) -> str:
    p = s.get("period_split") or {}
    lines = [f"# Feature selection - {s['timeframe']}", "",
             f"Development {p.get('development', {}).get('period')}, validation "
             f"{p.get('validation', {}).get('period')}, reserved test "
             f"{p.get('reserved_test', {}).get('period')} (outcomes never loaded).", "",
             f"Registered {s['registered_features']} -> quality universe "
             f"{s['quality_universe']} -> development selection universe "
             f"{s['selection_universe']}.", "", "## Exclusions", ""]
    lines += [f"- {k}: {v}" for k, v in sorted(s["exclusion_reasons"].items())]
    lines += ["", "## Candidate sets", "", "| set | features | justified |", "|---|---:|---|"]
    for k, v in s["sets"].items():
        j = v.get("justified") or {}
        lines.append(f"| {k} | {v['features']} | {j.get('justified')} - {j.get('reason')} |")
    lines += ["", f"Plateau: {s.get('plateau')}", "", "## Stable features by target kind", ""]
    for kind, feats in (s.get("stable_by_kind") or {}).items():
        lines.append(f"- **{kind}**: {', '.join(feats[:25])}")
    lines += ["", "## Leave one family out (mean change in validation rank IC)", ""]
    lines += [f"- {r['variant']}: {r['mean_delta']:+.4f}" for r in s["leave_one_family_out"]
              if r.get("mean_delta") is not None]
    lines += ["", f"Leakage audit: all verified = {s['leakage_audit_all_verified']}; live "
                  f"reconstruction = {s['live_reconstruction']}; reserved isolation passed = "
                  f"{(s.get('reserved_isolation') or {}).get('passed')}.", "",
              "Descriptive research only: no model for trading, no PnL, no cost, no position.",
              ""]
    del sets
    return "\n".join(lines)


def write_selection_comparison(results_path: Path, timeframes: list[str]) -> dict[str, Any]:
    rows = []
    for tf in timeframes:
        s = _json(results_path / tf / "summary.json")
        if not s:
            continue
        rows.append({"timeframe": tf, "registered": s.get("registered_features"),
                     "quality_universe": s.get("quality_universe"),
                     "selection_universe": s.get("selection_universe"),
                     **{f"set_{k}": v.get("features") for k, v in (s.get("sets") or {}).items()},
                     "minimal_k": (s.get("plateau") or {}).get("minimal_k"),
                     "standard_k": (s.get("plateau") or {}).get("standard_k")})
    out = ensure_dir(results_path / "comparison")
    if rows:
        write_table(pl.DataFrame(rows, infer_schema_length=None), out / "timeframes")
    sets_rows = []
    for tf in timeframes:
        for path in sorted((results_path / tf / "manifests").glob("*.json")):
            m = json.loads(path.read_text(encoding="utf-8"))
            for f in m["features"]:
                sets_rows.append({"timeframe": tf, "set": m["set"], "feature": f["name"],
                                  "family": f["family"]})
    if sets_rows:
        frame = pl.DataFrame(sets_rows)
        write_table(frame, out / "manifest_features")
        common = (frame.filter(pl.col("set") == "standard").group_by("feature")
                  .agg(pl.col("timeframe").n_unique().alias("timeframes"))
                  .sort(["timeframes", "feature"], descending=[True, False]))
        write_table(common, out / "standard_features_across_timeframes")
    summary = {"timeframes": [r["timeframe"] for r in rows], "rows": rows}
    write_json(out / "comparison.json", summary)
    return summary
