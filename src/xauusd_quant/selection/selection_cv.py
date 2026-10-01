r"""Nested chronological selection and evaluation (Prompt #9, Steps 20-27, 35-40).

For every split (the nested development folds, then development ->
validation) the whole selection procedure runs on the **training rows only**:

1. training CDFs of every candidate feature and target (the rank-linear design
   of :mod:`.linear`), frozen and applied to the evaluation rows;
2. in-split evidence: rank IC with Newey-West errors and BH q per target,
   yearly sign consistency, the median yearly IC, recent relevance (the last
   three years of the span), the studentized max-T circular-shift screen and
   the pipeline-null veto re-read on the training quarters;
3. candidates for a target = features whose kind beats the max-T null, with
   q < level, not reproduced by a veto null (residual targets *require* the
   pipeline control - invariant 9);
4. selection methods, each an ordering of the candidates: IC ranking,
   effect stability, mRMR (relevance |rank IC| or MI, difference / quotient,
   lambda), cluster representatives, Lasso and elastic-net entry order;
5. for each ``k``: a ridge fit on the training moments of the first ``k``
   features, scored on the evaluation rows (rank IC of the predictions, R^2 of
   the CDF-mapped target); PCA of the quality universe (fitted on training)
   with the same ridge on ``k`` components.

Training rows are purged by the longest target horizon and embargoed before the
evaluation span; nothing is shuffled; the evaluation rows' outcomes are read
only to score, never to select.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from ..alpha.config import AlphaConfig
from ..targets.alignment import target_kind
from .clustering import EvidenceKey, cluster_members, representative_order
from .config import FeatureSelectionConfig
from .data import SelectionData
from .linear import Moments, evaluate, lasso_path, ridge
from .mrmr import effect_stability_score, ic_ranking, mrmr
from .pca import fit_pca, pca_ridge
from .periods import SplitRows
from .redundancy import correlation_matrices
from .relevance import mutual_information, null_screen, pipeline_veto, quarter_labels

__all__ = ["SplitWork", "evaluate_split", "prepare_split", "split_evidence"]

CHUNK = 65_536


@dataclass
class SplitWork:
    """One split's frozen training transforms, design matrices and moments."""

    split: SplitRows
    train: tuple[int, int]
    evaluate: tuple[int, int]
    columns: list[int]                 # data.names indices (universe + probes)
    x_train: np.ndarray                # (n_train, p) float32 centred CDF scores
    x_eval: np.ndarray                 # (n_eval, p)
    y_train: np.ndarray                # (n_train, T) centred CDF scores
    y_eval: np.ndarray                 # (n_eval, T) CDF-mapped with the training CDFs
    y_eval_raw: np.ndarray             # (n_eval, T) raw targets (for the rank IC)
    moments: Moments
    abs_spearman: np.ndarray           # (p, p) on a training sample
    extra: dict[str, Any] = field(default_factory=dict)


def _cdf_scores(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(training rows' centred CDF scores, the sorted training values)."""
    s = np.sort(train[np.isfinite(train)])
    return _apply(train, s), s


def _apply(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    out = np.zeros(x.size, dtype=np.float32)
    ok = np.isfinite(x)
    if s.size:
        lo = np.searchsorted(s, x[ok], side="left")
        hi = np.searchsorted(s, x[ok], side="right")
        out[ok] = ((lo + hi) / (2.0 * s.size) - 0.5).astype(np.float32)
    return out


def _moments(x: np.ndarray, y: np.ndarray) -> Moments:
    ok = np.isfinite(y).all(axis=1)
    p, t = x.shape[1], y.shape[1]
    total = Moments(0.0, np.zeros(p), np.zeros((p, p)), np.zeros(t), np.zeros(t),
                    np.zeros((p, t)))
    for lo in range(0, x.shape[0], CHUNK):
        sel = ok[lo:lo + CHUNK]
        xs = x[lo:lo + CHUNK][sel].astype(np.float64)
        ys = y[lo:lo + CHUNK][sel].astype(np.float64)
        total = total + Moments(float(sel.sum()), xs.sum(axis=0), xs.T @ xs, ys.sum(axis=0),
                                (ys * ys).sum(axis=0), xs.T @ ys)
    return total


def prepare_split(data: SelectionData, split: SplitRows, columns: list[int],
                  cfg: FeatureSelectionConfig) -> SplitWork:
    horizon = max(cfg.targets.horizons)
    train = split.train_rows(horizon, cfg.periods.embargo_bars)
    lo, hi = train
    e_lo, e_hi = split.evaluate
    data.guard.check_rows(slice(e_lo, e_hi), f"split {split.name} evaluation")
    p = len(columns)
    x_train = np.empty((hi - lo, p), dtype=np.float32)
    x_eval = np.empty((e_hi - e_lo, p), dtype=np.float32)
    for i, c in enumerate(columns):
        x_train[:, i], s = _cdf_scores(data.features[lo:hi, c])
        x_eval[:, i] = _apply(data.features[e_lo:e_hi, c], s)
    t = data.targets.shape[1]
    y_train = np.empty((hi - lo, t), dtype=np.float32)
    y_eval = np.empty((e_hi - e_lo, t), dtype=np.float32)
    for j in range(t):
        tr = data.targets[lo:hi, j]
        y_train[:, j], s = _cdf_scores(tr)
        y_train[~np.isfinite(tr), j] = np.nan
        ev = data.targets[e_lo:e_hi, j]
        y_eval[:, j] = _apply(ev, s)
        y_eval[~np.isfinite(ev), j] = np.nan
    moments = _moments(x_train, y_train)
    sample = min(hi - lo, cfg.redundancy.sample_rows)
    _, spear, _ = correlation_matrices(x_train, 0, hi - lo, sample_rows=sample)
    return SplitWork(split=split, train=train, evaluate=(e_lo, e_hi), columns=columns,
                     x_train=x_train, x_eval=x_eval, y_train=y_train, y_eval=y_eval,
                     y_eval_raw=data.targets[e_lo:e_hi].astype(np.float32),
                     moments=moments, abs_spearman=np.abs(np.nan_to_num(spear, nan=0.0)))


def split_evidence(work: SplitWork, data: SelectionData, cfg: FeatureSelectionConfig,
                   alpha: AlphaConfig, *, with_mi: bool = True) -> pl.DataFrame:
    """Per feature x target evidence inside the training rows of the split."""
    from ..alpha.information_coefficient import ic_with_errors, month_index, month_moments
    from ..alpha.ranking import benjamini_hochberg
    from ..alpha.stability import consistency, grouped_ics
    from ..research.feature_reports import pipeline_verdicts

    lo, hi = work.train
    names = [data.names[c] for c in work.columns]
    stamps = data.timestamps.slice(lo, hi - lo)
    months = month_index(stamps)
    mom = np.zeros((len(names), work.y_train.shape[1], months.size, 6))
    ys = [work.y_train[:, j] for j in range(work.y_train.shape[1])]
    for b0 in range(0, len(names), 32):
        xs = [work.x_train[:, i] for i in range(b0, min(len(names), b0 + 32))]
        mom[b0:b0 + len(xs)] = month_moments(xs, ys, months.starts)
    m = np.moveaxis(mom, 2, 0)
    stats = ic_with_errors(m, hac_lags=cfg.null_screen.hac_lag_months)
    _, yic, _ = grouped_ics(m, months.years, min_obs=500)
    last_years = sorted(set(months.years.tolist()))[-3:]
    recent_mask = np.isin(months.years, last_years)
    recent = ic_with_errors(m, select=recent_mask, hac_lags=cfg.null_screen.hac_lag_months)
    cons = consistency(yic, stats["ic"], recent["ic"])
    q = np.column_stack([benjamini_hochberg(stats["p"][:, j]) for j in range(stats["p"].shape[1])])
    mi = (mutual_information(names, data.target_meta, work.x_train, work.y_train, 0, hi - lo,
                             bins=cfg.mrmr.mi_bins) if with_mi
          else np.full(stats["ic"].shape, np.nan))
    rows = []
    for i, f in enumerate(names):
        for j, (kind, h, col) in enumerate(data.target_meta):
            rows.append({"feature": f, "target": col, "kind": kind, "horizon": h,
                         "value": _f(stats["ic"][i, j]), "se": _f(stats["se"][i, j]),
                         "q": _f(q[i, j]), "n": float(stats["n"][i, j]),
                         "median_yearly_ic": _f(cons["median_ic"][i, j]),
                         "sign_consistency": _f(cons["sign_consistency"][i, j]),
                         "recent_ic": _f(recent["ic"][i, j]), "mi": _f(mi[i, j])})
    ev = pl.DataFrame(rows, infer_schema_length=None)
    screen = null_screen(names, data.target_meta, work.x_train, work.y_train, 0, hi - lo, cfg,
                         seed=cfg.seed + 31)
    ev = ev.join(screen.select("feature", "kind", "beyond_max_t", "max_t_p"),
                 on=["feature", "kind"], how="left")
    quarters = quarter_labels(data.timestamps, lo, hi)
    veto = (pipeline_veto(data.research_dir, quarters, cfg, alpha)
            if data.research_dir is not None else pl.DataFrame())
    if not veto.is_empty():
        wide = veto.pivot(on="null", index=["feature", "target"], values=["ic", "se"])
        rename = {}
        for c in wide.columns:
            if c.startswith("ic_"):
                rename[c] = f"pipeline_{c[3:]}_ic"
            elif c.startswith("se_"):
                rename[c] = f"pipeline_{c[3:]}_se"
        ev = ev.join(wide.rename(rename), on=["feature", "target"], how="left")
    # the veto reads Prompt #8's target kinds ("residual" gets the sign-flip veto too)
    selection_kind = ev["kind"]
    ev = pipeline_verdicts(ev.with_columns(
        pl.col("target").map_elements(target_kind, return_dtype=pl.Utf8).alias("kind")), alpha)
    ev = ev.with_columns(selection_kind.alias("kind"))
    fdr = cfg.null_screen.fdr_q
    pipeline_ok = (pl.when(pl.col("kind") == "reversion")
                   .then(pl.col("beyond_pipeline_null").fill_null(False))
                   .otherwise(pl.col("beyond_pipeline_null").fill_null(True)))
    return ev.with_columns(
        (pl.col("beyond_max_t").fill_null(False) & (pl.col("q") < fdr).fill_null(False)
         & pipeline_ok).alias("passes"),
        pl.lit(work.split.name).alias("split"))


def _f(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def method_orderings(work: SplitWork, ev: pl.DataFrame, data: SelectionData, target: str,
                     cfg: FeatureSelectionConfig) -> dict[str, list[int]]:
    """Every selection method's ordering (indices into work.columns) for one target."""
    names = [data.names[c] for c in work.columns]
    part = ev.filter(pl.col("target") == target)
    by = {r["feature"]: r for r in part.iter_rows(named=True)}

    def get(name: str, key: str) -> float:
        v = by.get(name, {}).get(key)
        return float(v) if v is not None else np.nan

    cand = [i for i, n in enumerate(names) if by.get(n, {}).get("passes")]
    rel_ic = np.nan_to_num(np.abs(np.array([get(n, "value") for n in names])), nan=0.0)
    rel_mi = np.nan_to_num(np.clip(np.array([get(n, "mi") for n in names]), 0.0, None), nan=0.0)
    kmax = cfg.mrmr.max_features
    hard = cfg.redundancy.max_pairwise_correlation
    out: dict[str, list[int]] = {"ic_rank": ic_ranking(rel_ic, cand, k=kmax)}
    score = effect_stability_score(np.array([get(n, "median_yearly_ic") for n in names]),
                                   np.array([get(n, "sign_consistency") for n in names]))
    out["effect_stability"] = [i for i in sorted(cand, key=lambda i: (-score[i], i))
                               if score[i] > 0][:kmax]
    for rel_name, rel in (("ic", rel_ic), ("mi", rel_mi)):
        if rel_name == "mi" and "mutual_information" not in cfg.mrmr.relevance:
            continue
        for scheme in cfg.mrmr.schemes:
            lams = cfg.mrmr.lambdas if scheme == "difference" else (1.0,)
            for lam in lams:
                steps = mrmr(rel, work.abs_spearman, cand, k=kmax, lam=lam, scheme=scheme,
                             hard_limit=hard)
                key = f"mrmr_{rel_name}_{scheme}" + (f"_{lam:g}" if scheme == "difference"
                                                      else "")
                out[key] = [s.index for s in steps]
    out["cluster_representatives"] = _cluster_order(work, names, by, cand, cfg, data)
    j = data.target_columns().index(target)
    for l2 in cfg.linear.elastic_net_l2:
        if not cand:
            out["lasso" if l2 == 0 else f"elastic_net_{l2:g}"] = []
            continue
        q, c, _, _, _ = work.moments.standardized(cand)
        path = lasso_path(q, c[:, j], l2=l2, max_features=cfg.linear.max_path_features)
        out["lasso" if l2 == 0 else f"elastic_net_{l2:g}"] = [cand[k] for k in path["order"]]
    return out


def _cluster_order(work: SplitWork, names: list[str], by: dict[str, dict[str, Any]],
                   cand: list[int], cfg: FeatureSelectionConfig, data: SelectionData
                   ) -> list[int]:
    if not cand:
        return []
    sub = work.abs_spearman[np.ix_(cand, cand)]
    clusters, _ = cluster_members(sub, [names[i] for i in cand],
                                  threshold=cfg.redundancy.cluster_abs_corr)
    keys = {}
    for i in cand:
        r = by.get(names[i], {})
        spec = data.registry.get(names[i], {})
        recent = r.get("recent_ic")
        pooled = r.get("value")
        rel = abs(recent) if (recent is not None and pooled is not None
                              and np.sign(recent) == np.sign(pooled)) else 0.0
        keys[names[i]] = EvidenceKey(
            name=names[i], passes_null=bool(r.get("passes")),
            sign_consistency=r.get("sign_consistency"),
            median_yearly_ic=r.get("median_yearly_ic"), recent_relevance=rel,
            missing_development=0.0, cost=spec.get("cost"),
            min_history=int(spec.get("min_history") or 0))
    reps = []
    for members in clusters.values():
        best = representative_order([keys[m] for m in members], cfg.representative
                                    .consistency_round)[0]
        reps.append(names.index(best.name))
    rel = {i: abs(by.get(names[i], {}).get("value") or 0.0) for i in reps}
    return sorted(reps, key=lambda i: (-rel[i], i))


def evaluate_split(work: SplitWork, orderings: dict[str, list[int]], target: str,
                   data: SelectionData, cfg: FeatureSelectionConfig) -> list[dict[str, Any]]:
    """Validation metrics of every method's first-k features (ridge, frozen training fit)."""
    j = data.target_columns().index(target)
    alpha = cfg.linear.ridge_alpha
    rows = []
    for method, order in orderings.items():
        for k in cfg.evaluation.candidate_set_sizes:
            if k > len(order):
                rows.append({"split": work.split.name, "target": target, "method": method,
                             "k": k, "available": len(order), "rank_ic": None, "r2": None,
                             "n": 0})
                continue
            cols = order[:k]
            q, c, mx, sd, my = work.moments.standardized(cols)
            beta = ridge(q, c[:, [j]], alpha)[:, 0]
            pred = ((work.x_eval[:, cols] - mx) / sd) @ beta + my[j]
            metrics = evaluate(pred, work.y_eval_raw[:, j], work.y_eval[:, j], float(my[j]))
            rows.append({"split": work.split.name, "target": target, "method": method, "k": k,
                         "available": len(order), **metrics})
    return rows


def pca_evaluation(work: SplitWork, universe: list[int], target: str, data: SelectionData,
                   cfg: FeatureSelectionConfig) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Ridge on the first k training-fitted components of the quality universe."""
    j = data.target_columns().index(target)
    model = fit_pca(work.moments, universe)
    _, _, _, _, my = work.moments.standardized(universe)
    rows = []
    for k in cfg.pca.components:
        if k > len(universe):
            continue
        beta = pca_ridge(model, work.moments, k, cfg.linear.ridge_alpha)[:, j]
        scores = model.transform(work.x_eval.astype(np.float64), k)
        pred = scores @ beta + my[j]
        metrics = evaluate(pred, work.y_eval_raw[:, j], work.y_eval[:, j], float(my[j]))
        rows.append({"split": work.split.name, "target": target, "method": "pca", "k": k,
                     "explained_variance": float(model.explained()[:k].sum()), **metrics})
    return rows, model.explained()
