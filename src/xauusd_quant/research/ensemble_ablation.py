r"""Which constituents earn their place? (Prompt #11, Steps 15-18, 20, 51-53)

All on the meta walk-forward of :mod:`.ensemble_research` (fitted on earlier blocks,
scored on the next), simple averages unless said otherwise:

* **size** (Step 15): the top-*s* universe models by their mean primary metric on
  the earlier blocks, *s* = 1 .. max_size; and a greedy forward selection on the
  history rows (start from the best model, add whichever most improves the
  history skill of the average) - every prefix scored (Step 16; every subset
  counted as a trial);
* **families** (Step 17): tree-only, linear + trees, compact diversified;
* **feature sets** (Step 18): one family across its feature sets;
* **weaker models** (question N): the universe plus the eligible-but-for-quality
  models (``weak`` / ``unstable`` / ``calibration_failed``; never a leakage, null
  or deprecated failure);
* **leave one model out / one family group out** (Steps 51-52) and each model's
  marginal contribution to the primary metric, calibration and diversity (Step 53);
* **cross-horizon** (Step 20): predictions of the same target at other horizons as
  extra inputs of a stacked model of the anchor horizon - complementary or not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from ..alpha.information_coefficient import rank_scores
from ..ensemble.averaging import weighted_mean
from ..ensemble.data import load_pair, load_unit_predictions
from ..ensemble.disagreement import disagreement_measures
from ..ensemble.meta_model import fit_meta_model
from ..ensemble.stability import primary_metric, score
from ..ensemble.weighting import skill
from ..ml.evaluation import paired_blocks
from ..utils.paths import ensure_dir
from .feature_research import write_json, write_table

if TYPE_CHECKING:
    from .ensemble_research import PairRun, TFContext

__all__ = ["cross_horizon", "run_ablations"]


def _subset_series(run: PairRun, name: str, members: list[str], kind: str) -> bool:
    if not members:
        return False
    v = run.empty()
    avg = weighted_mean(run.matrix(members))
    v[run.eval_mask] = avg[run.eval_mask]
    run.add(name, v, kind, members)
    return True


def _paired(run: PairRun, a: str, b: str) -> dict[str, Any]:
    ks = run.eval_blocks
    return paired_blocks([run.primary(a, k) for k in ks], [run.primary(b, k) for k in ks])


def _mean(run: PairRun, name: str, metric: str | None = None) -> float | None:
    got = [run.block_score(name, k).get(metric or run.metric) for k in run.eval_blocks]
    vals = [float(v) for v in got if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def size_research(run: PairRun) -> list[dict[str, Any]]:
    u = run.universe
    max_size = min(int(run.cfg.size.get("max_size", 8)), len(u))
    quality = {s: run.empty() for s in range(1, max_size + 1)}
    greedy = {s: run.empty() for s in range(1, len(u) + 1)}
    p = run.matrix()
    orders: dict[str, Any] = {}
    for k in run.eval_blocks:
        prior: dict[str, float] = {}
        for m in u:
            pm = run.prior_mean(f"model:{m}", k)
            prior[m] = pm if pm is not None else -np.inf
        ranked = sorted(u, key=lambda m: -prior[m])
        pos = run.pair.block_positions(k)
        for s in quality:
            cols = [u.index(m) for m in ranked[:s]]
            quality[s][pos] = p[pos][:, cols].mean(axis=1)
        hist = run.history(k)
        ph, yh, bh = p[hist], run.y[hist], run.base[hist]
        q = np.atleast_1d(skill(run.task, ph, yh, bh))
        if not np.isfinite(q).any():               # no defined skill: no greedy order here
            continue
        chosen = [int(np.nanargmax(q))]
        while len(chosen) < len(u):
            best_j, best_v = None, -np.inf
            for j in range(len(u)):
                if j in chosen:
                    continue
                v = float(skill(run.task, ph[:, [*chosen, j]].mean(axis=1), yh, bh))
                if v > best_v:
                    best_j, best_v = j, v
            if best_j is None:                        # undefined skill: stop the order here
                break
            chosen.append(best_j)
        for s in greedy:
            if s <= len(chosen):
                greedy[s][pos] = p[pos][:, chosen[:s]].mean(axis=1)
        orders[run.pair.block_names[k]] = {"quality": ranked, "greedy": [u[j] for j in chosen]}
    rows = []
    for tag, series in (("quality_top", quality), ("greedy", greedy)):
        for s, values in series.items():
            name = f"size:{tag}{s}"
            run.add(name, values, "size", u)
            cmp = _paired(run, name, "simple_average")
            rows.append({"selection": tag, "size": s, "series": name,
                         "mean": _mean(run, name), "worst_block": min(
                             (x for x in (run.primary(name, k) for k in run.eval_blocks)
                              if x is not None), default=None),
                         "minus_full_average": cmp.get("mean_diff"), "se": cmp.get("se"),
                         "wins_vs_full_average": cmp.get("wins")})
    run.notes["size_orders"] = orders
    return rows


def family_research(run: PairRun) -> list[dict[str, Any]]:
    cfg, ml = run.cfg, run.tctx.ml
    ref = ml.reference_linear(ml.targets[run.pair.target].task)
    eligible = set(run.elig.filter(pl.col("status") == "eligible")["model"].to_list())
    std = ml.default_feature_set
    rows = []
    for group, fams in cfg.ensembles_by_family.items():
        members = [f"{ref if f == 'linear' else f}|{std}" for f in fams]
        present = [m for m in members if m in eligible]
        name = f"family:{group}"
        if len(present) < 2:
            rows.append({"subset": name, "members": ",".join(present), "size": len(present),
                         "note": f"only {len(present)} eligible member(s) of {members}"})
            continue
        _subset_series(run, name, present, "subset")
        cmp = _paired(run, name, "simple_average")
        rows.append({"subset": name, "members": ",".join(present), "size": len(present),
                     "mean": _mean(run, name), "minus_full_average": cmp.get("mean_diff"),
                     "se": cmp.get("se"), "wins_vs_full_average": cmp.get("wins")})
    fam = cfg.feature_set_family
    sets = [m for m in run.pair.model_names if run.pair.family(m) == fam and m in eligible]
    name = f"featureset:{fam}"
    if len(sets) >= 2:
        _subset_series(run, name, sets, "subset")
        best_single = max(sets, key=lambda m: _mean(run, f"model:{m}") or -np.inf)
        cmp = _paired(run, name, f"model:{best_single}")
        rows.append({"subset": name, "members": ",".join(sets), "size": len(sets),
                     "mean": _mean(run, name), "versus": f"model:{best_single}",
                     "minus_best_member": cmp.get("mean_diff"), "se": cmp.get("se"),
                     "wins_vs_best_member": cmp.get("wins")})
    else:
        rows.append({"subset": name, "members": ",".join(sets), "size": len(sets),
                     "note": "fewer than two eligible feature sets"})
    weaker = run.elig.filter(pl.col("status").is_in(["weak", "unstable",
                                                     "calibration_failed"]))["model"].to_list()
    if weaker:
        name = "weak:with_weaker_models"
        _subset_series(run, name, [*run.universe, *weaker], "subset")
        cmp = _paired(run, name, "simple_average")
        rows.append({"subset": name, "members": ",".join([*run.universe, *weaker]),
                     "size": len(run.universe) + len(weaker), "added": ",".join(weaker),
                     "mean": _mean(run, name), "minus_full_average": cmp.get("mean_diff"),
                     "se": cmp.get("se"), "wins_vs_full_average": cmp.get("wins")})
    return rows


def _verdict(cmp: dict[str, Any]) -> str:
    """adds / hurts beyond one SE, no measurable effect - or untested when undefined."""
    diff, se = cmp.get("mean_diff"), cmp.get("se")
    if diff is None or se is None or not np.isfinite(diff) or not np.isfinite(se) \
            or int(cmp.get("blocks", 0)) < 2:
        return "untested"
    if diff > se:
        return "adds"
    if diff < -se:
        return "hurts"
    return "no measurable effect"


def leave_out(run: PairRun) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    u = run.universe
    lomo, contrib = [], []
    full = "simple_average"
    full_ece = _mean(run, full, "ece") if run.pair.is_classification else None
    dis_full = disagreement_measures(run.matrix())["prediction_std"]
    for m in u:
        rest = [x for x in u if x != m]
        name = f"lomo:-{m}"
        if not rest:
            continue
        _subset_series(run, name, rest, "lomo")
        cmp = _paired(run, full, name)                 # full - without m: what m adds
        lomo.append({"removed": m, "series": name, "mean_without": _mean(run, name),
                     "mean_full": _mean(run, full), "contribution": cmp.get("mean_diff"),
                     "se": cmp.get("se"), "blocks_where_it_helps": cmp.get("wins")})
        dis_rest = disagreement_measures(run.matrix(rest))["prediction_std"] if len(rest) > 1 \
            else np.zeros(run.pair.n)
        ev = run.eval_positions()
        rest_ece = _mean(run, name, "ece")
        contrib.append({
            "model": m, "individual_mean": _mean(run, f"model:{m}"),
            "primary_contribution": cmp.get("mean_diff"), "se": cmp.get("se"),
            "ece_contribution": (None if full_ece is None or rest_ece is None
                                 else float(rest_ece - full_ece)),
            "disagreement_contribution": float(np.nanmean(dis_full[ev]) - np.nanmean(dis_rest[ev])),
            "verdict": _verdict(cmp)})
    groups = run.cfg.family_groups
    for group, fams in groups.items():
        rest = [x for x in u if run.pair.family(x) not in fams]
        removed = [x for x in u if run.pair.family(x) in fams]
        if not removed or not rest:
            continue
        name = f"lofo:-{group}"
        _subset_series(run, name, rest, "lofo")
        cmp = _paired(run, full, name)
        lomo.append({"removed": f"family group {group} ({','.join(removed)})", "series": name,
                     "mean_without": _mean(run, name), "mean_full": _mean(run, full),
                     "contribution": cmp.get("mean_diff"), "se": cmp.get("se"),
                     "blocks_where_it_helps": cmp.get("wins")})
    return lomo, contrib


def run_ablations(run: PairRun, out: Any) -> dict[str, Any]:
    size_rows = size_research(run)
    write_table(pl.DataFrame(size_rows, infer_schema_length=None), out / "ablation" / "size")
    fam_rows = family_research(run)
    write_table(pl.DataFrame(fam_rows, infer_schema_length=None), out / "ablation" / "subsets")
    lomo, contrib = leave_out(run)
    write_table(pl.DataFrame(lomo, infer_schema_length=None), out / "ablation" / "leave_out")
    write_table(pl.DataFrame(contrib, infer_schema_length=None),
                out / "ablation" / "contribution")
    best_size = max((r for r in size_rows if r["mean"] is not None),
                    key=lambda r: r["mean"], default=None)
    return {"size_rows": len(size_rows), "best_size": None if best_size is None else {
        k: best_size[k] for k in ("selection", "size", "mean")},
        "subsets": len(fam_rows), "leave_out": len(lomo)}


# ---------------------------------------------------------------------------
# Step 20: cross-horizon information
# ---------------------------------------------------------------------------
def cross_horizon(tctx: TFContext, target: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Do predictions of the same target at other horizons add information about the
    anchor horizon? Stacked (meta walk-forward) with and without them, per family."""
    ml, cfg = tctx.ml, tctx.cfg
    tspec = ml.targets[target]
    anchor = int(spec.get("anchor", 5))
    horizons = list(tspec.horizons)
    fams = [str(f) for f in spec.get("families") or []]
    out = ensure_dir(tctx.out_dir / "cross_horizon" / target)
    pair = load_pair(tctx.ml_dir / "predictions" / f"{target}_h{anchor}.parquet",
                     timeframe=tctx.timeframe, target=target, horizon=anchor, task=tspec.task,
                     calibrated_inputs=cfg.calibrated_only_for_classification,
                     reserved_start=tctx.data.reserved_start)
    units = tctx.ml_dir / "units"
    pos_of = {int(r): i for i, r in enumerate(pair.rows)}
    cols: dict[str, np.ndarray] = {}
    for fam in fams:
        for h in horizons:
            f = load_unit_predictions(units, target, h, fam, ml.default_feature_set,
                                      development_rows=tctx.data.n)
            if f.is_empty():
                continue
            v = np.full(pair.n, np.nan)
            col = "cal_platt" if (tspec.is_classification and "cal_platt" in f.columns) \
                else "prediction"
            idx = np.array([pos_of.get(int(r), -1) for r in f["row"].to_numpy()])
            ok = idx >= 0
            v[idx[ok]] = f[col].cast(pl.Float64).to_numpy()[ok]
            cols[f"{fam}|h{h}"] = v
    names = sorted(cols)
    corr_rows = []
    mat = np.column_stack([cols[n] for n in names]) if names else np.empty((pair.n, 0))
    ok_all = np.isfinite(mat).all(axis=1) & np.isfinite(pair.label)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            j = names.index(b)
            sel = np.isfinite(mat[:, i]) & np.isfinite(mat[:, j])
            if sel.sum() < 1000:
                continue
            corr_rows.append({"a": a, "b": b, "rows": int(sel.sum()),
                              "spearman": float(np.corrcoef(rank_scores(mat[sel, i]),
                                                            rank_scores(mat[sel, j]))[0, 1])})
    write_table(pl.DataFrame(corr_rows, infer_schema_length=None), out / "horizon_correlation")
    metric = primary_metric(tspec.task)
    variants: dict[str, list[str]] = {}
    for fam in fams:
        own = [n for n in names if n.startswith(f"{fam}|")]
        if f"{fam}|h{anchor}" in own:
            variants[f"{fam}:anchor_only"] = [f"{fam}|h{anchor}"]
            if len(own) > 1:
                variants[f"{fam}:all_horizons"] = own
    if len(names) > 1:
        variants["all_families_all_horizons"] = names
    rows = []
    for vname, inputs in variants.items():
        cols_idx = [names.index(n) for n in inputs]
        for k in cfg.evaluation_blocks:
            if k >= len(pair.block_names):
                continue
            hist = pair.history(k, cfg.embargo_bars)
            hist = hist[ok_all[hist]]
            pos = pair.block_positions(k)
            pos = pos[ok_all[pos]]
            if hist.size < 1000 or pos.size < 1000:
                continue
            meta = fit_meta_model(mat[hist][:, cols_idx], pair.label[hist], inputs,
                                  constituents=len(inputs), task=tspec.task,
                                  logistic_c=float(cfg.stacking.get("logistic_C", 1.0)),
                                  ridge_alpha=float(cfg.stacking.get("ridge_alpha", 1.0)),
                                  max_rows=cfg.stacking.get("max_rows"), seed=cfg.random_seed)
            pred = meta.predict(mat[pos][:, cols_idx])
            met = score(tspec.task, pred, pair.label[pos], pair.base[pos], pair.label_raw[pos])
            rows.append({"variant": vname, "inputs": ",".join(inputs),
                         "block": pair.block_names[k], "rows": int(pos.size),
                         "metric": metric, "value": met.get(metric), "auc": met.get("auc"),
                         "rank_ic": met.get("rank_ic"),
                         "coefficients": json_coef(meta.inputs, meta.coef)})
    frame = pl.DataFrame(rows, infer_schema_length=None)
    write_table(frame, out / "incremental")
    summary: dict[str, Any] = {"anchor": anchor, "inputs": names, "variants": list(variants)}
    if not frame.is_empty():
        for fam in fams:
            fa = frame.filter(pl.col("variant") == f"{fam}:anchor_only")
            fb = frame.filter(pl.col("variant") == f"{fam}:all_horizons")
            if fa.height and fb.height:
                common = sorted(set(fa["block"]) & set(fb["block"]))
                av = {r["block"]: r["value"] for r in fa.iter_rows(named=True)}
                bv = {r["block"]: r["value"] for r in fb.iter_rows(named=True)}
                summary[f"{fam}_all_minus_anchor"] = paired_blocks(
                    [bv[c] for c in common], [av[c] for c in common])
    write_json(out / "summary.json", summary)
    return summary


def json_coef(names: list[str], coef: np.ndarray) -> str:
    return ", ".join(f"{n}={c:+.3f}" for n, c in zip(names, coef, strict=True))
