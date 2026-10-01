r"""Model explanations on validation samples (Prompt #10, Steps 34-38).

* **SHAP**: the boosters' own TreeSHAP (LightGBM ``pred_contrib``, XGBoost
  ``pred_contribs``, CatBoost ``ShapValues``), ``shap.TreeExplainer`` for the
  random forest, and the exact linear decomposition
  :math:`\phi_j = \beta_j (x_j - \bar x_j)` for linear models - on a sample of
  the validation block only, never 23 years at once. Summaries: mean
  :math:`|\phi_j|` overall and per year, and the directionality
  :math:`\mathrm{corr}_S(x_j, \phi_j)` (positive: a larger value pushes the
  prediction up).
* **Permutation importance**: the rise in log loss / MSE when one feature is
  permuted in blocks of about one trading day (the block keeps its
  autocorrelation). Correlated features share credit, so importance is read as
  descriptive, never causal.
* **ALE**: first-order accumulated local effects of the top features
  (centred), which, unlike partial dependence, do not average over impossible
  combinations of correlated features.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..alpha.information_coefficient import rank_scores
from .config import MLConfig, TargetSpec
from .datasets import MLData
from .models import Model
from .preprocessing import Preprocessor

__all__ = ["ale_curve", "explain_unit", "permutation_importance", "shap_values"]


def shap_values(model: Model, x: np.ndarray) -> np.ndarray | None:
    """(n, columns) SHAP values in the model's raw output space (log-odds for boosters)."""
    fam = model.family
    est = model._est
    if fam == "constant":
        return np.zeros_like(x, dtype=np.float64)
    if fam == "lightgbm":
        return np.asarray(est.predict(x, pred_contrib=True))[:, :-1]
    if fam == "xgboost":
        import xgboost as xgb

        booster = est.get_booster()
        best = getattr(est, "best_iteration", None)
        rng = (0, int(best) + 1) if best is not None else (0, 0)
        contrib = booster.predict(xgb.DMatrix(x), pred_contribs=True, iteration_range=rng)
        return np.asarray(contrib)[:, :-1]
    if fam == "catboost":
        from catboost import Pool

        return np.asarray(est.get_feature_importance(Pool(x), type="ShapValues"))[:, :-1]
    if fam == "random_forest":
        import shap

        vals = shap.TreeExplainer(est).shap_values(x, check_additivity=False)
        arr = np.asarray(vals)
        if arr.ndim == 3:                       # (n, p, classes)
            arr = arr[:, :, 1] if arr.shape[2] == 2 else arr[:, :, 0]
        return arr
    coef = model.coefficients()
    if coef is None:
        return None
    return (x - x.mean(axis=0)) * coef[None, :]


def _loss(model: Model, x: np.ndarray, y: np.ndarray, task: str) -> float:
    p = model.predict(x)
    if task == "classification":
        pc = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))
    return float(np.mean((p - y) ** 2))


def permutation_importance(model: Model, x: np.ndarray, y: np.ndarray, names: list[str],
                           task: str, *, block: int, seed: int, repeats: int = 2
                           ) -> dict[str, float]:
    """Mean rise in loss when each column is permuted in contiguous blocks."""
    rng = np.random.default_rng(seed)
    ref = _loss(model, x, y, task)
    n = x.shape[0]
    starts = np.arange(0, n, block)
    out = {}
    for j, name in enumerate(names):
        drops = []
        for _ in range(repeats):
            order = rng.permutation(starts.size)
            idx = np.concatenate([np.arange(s, min(n, s + block)) for s in starts[order]])[:n]
            xp = x.copy()
            xp[:, j] = x[idx, j]
            drops.append(_loss(model, xp, y, task) - ref)
        out[name] = float(np.mean(drops))
    return out


def ale_curve(model: Model, x: np.ndarray, j: int, *, bins: int = 20) -> dict[str, Any]:
    """Centred first-order ALE of column *j* (in design units)."""
    col = x[:, j]
    ok = np.isfinite(col)
    if ok.sum() < 200:
        return {"edges": [], "ale": []}
    edges = np.unique(np.quantile(col[ok], np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        return {"edges": [float(e) for e in edges], "ale": [0.0] * edges.size}
    idx = np.clip(np.searchsorted(edges, col, side="right") - 1, 0, edges.size - 2)
    effects = np.zeros(edges.size - 1)
    counts = np.zeros(edges.size - 1)
    for b in range(edges.size - 1):
        m = ok & (idx == b)
        if not m.any():
            continue
        lo_x, hi_x = x[m].copy(), x[m].copy()
        lo_x[:, j] = edges[b]
        hi_x[:, j] = edges[b + 1]
        effects[b] = float(np.mean(model.predict(hi_x) - model.predict(lo_x)))
        counts[b] = m.sum()
    ale = np.concatenate([[0.0], np.cumsum(effects)])
    centre = np.sum(counts * (ale[:-1] + ale[1:]) / 2.0) / max(1.0, counts.sum())
    return {"edges": [float(e) for e in edges], "ale": [float(a - centre) for a in ale],
            "counts": [int(c) for c in counts]}


def explain_unit(model: Model, pre: Preprocessor, xv: np.ndarray, y_val: np.ndarray,
                 val_rows: np.ndarray, names: list[str], data: MLData, tspec: TargetSpec,
                 cfg: MLConfig, *, seed: int) -> dict[str, Any]:
    ex = cfg.explain
    if list(pre.features) != list(names):
        raise ValueError("explain_unit: the preprocessor was fitted on another feature order")
    n = xv.shape[0]
    size = int(ex.get("shap_rows_per_fold", 20000))
    if model.family == "random_forest":
        size = min(size, 5000)
    take = np.unique(np.linspace(0, n - 1, min(n, size)).astype(np.int64))
    xs = xv[take]
    out_names = pre.output_names()
    phi = shap_values(model, xs)
    result: dict[str, Any] = {"sample_rows": int(take.size), "columns": out_names}
    base_names = [nm.removeprefix("missing__") for nm in out_names]
    if phi is not None:
        mean_abs = np.abs(phi).mean(axis=0)
        agg: dict[str, float] = {}
        for nm, v in zip(base_names, mean_abs, strict=True):
            agg[nm] = agg.get(nm, 0.0) + float(v)
        result["shap_mean_abs"] = agg
        direction = {}
        for j, nm in enumerate(out_names):
            if nm.startswith("missing__"):
                continue
            xj = xs[:, j]
            ok = np.isfinite(xj)
            if ok.sum() > 100 and np.std(xj[ok]) > 0 and np.std(phi[ok, j]) > 0:
                r = np.corrcoef(rank_scores(xj[ok]), rank_scores(phi[ok, j]))[0, 1]
                direction[nm] = float(r) if np.isfinite(r) else None
        result["shap_direction"] = direction
        years = data.timestamps.gather(val_rows[take]).dt.year().to_numpy()
        by_year = {}
        for yr in np.unique(years):
            m = years == yr
            if m.sum() < 200:
                continue
            ma = np.abs(phi[m]).mean(axis=0)
            yagg: dict[str, float] = {}
            for nm, v in zip(base_names, ma, strict=True):
                yagg[nm] = yagg.get(nm, 0.0) + float(v)
            by_year[int(yr)] = yagg
        result["shap_by_year"] = by_year
    result["gain"] = model.feature_importance(out_names)
    bars_day = max(1, int(round(data.n / max(1, data.timestamps.dt.date().n_unique()))))
    result["permutation"] = permutation_importance(model, xs, y_val[take], out_names, tspec.task,
                                                   block=bars_day, seed=seed)
    top = sorted((k for k in result.get("shap_mean_abs", {}) if not k.startswith("noise_")),
                 key=lambda k: -result["shap_mean_abs"][k])[:int(ex.get("top_features", 6))]
    curves = {}
    for nm in top:
        j = out_names.index(nm)
        curve = ale_curve(model, xs, j, bins=int(ex.get("ale_bins", 20)))
        if pre.kind == "linear" and pre.centers is not None and pre.scales is not None:
            k = pre.features.index(nm)
            curve["edges_original"] = [float(e * pre.scales[k] + pre.centers[k])
                                       for e in curve["edges"]]
        curves[nm] = curve
    result["ale"] = curves
    return result
