r"""Stability of ensembles across time and states (Prompt #11, Steps 39-45, 55-57).

* :func:`score`: the development metrics of one prediction series on given rows
  (log-loss / Brier skill against the per-row constant, AUC, PR AUC, ECE, rank IC,
  sharpness for probabilities; MAE, RMSE, R^2, Pearson, rank IC, bias, MSE skill for
  expected values) - the Prompt #10 definitions (:mod:`xauusd_quant.ml.evaluation`).
* :func:`period_table`: the same per calendar year / quarter / any grouping, each
  row flagged ``few_rows`` below the configured minimum (Step 40: short periods
  are noisy; the flag travels with the number).
* :func:`weight_stability`: how much each constituent's weight moves across the
  evaluation blocks (range, standard deviation), the weight entropy and effective
  number of models per block, and a flag when a weight moves more than
  ``weight_range_flag`` (Step 55: 0.8 -> 0.1 -> 0.7 is an unstable scheme).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..ml.evaluation import classification_metrics, regression_metrics
from .weighting import weight_entropy

__all__ = ["period_table", "primary_metric", "score", "weight_stability"]


def primary_metric(task: str) -> str:
    return "log_loss_skill" if task == "classification" else "rank_ic"


def score(task: str, pred: np.ndarray, y: np.ndarray, base: np.ndarray,
          y_raw: np.ndarray | None = None) -> dict[str, Any]:
    """Metrics of *pred* on these rows; skills against the per-row constant *base*."""
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    ok = np.isfinite(pred) & np.isfinite(y) & np.isfinite(base)
    if task == "classification":
        out = classification_metrics(pred[ok], y[ok])
        if out.get("n", 0) >= 100:
            pc = np.clip(pred[ok], 1e-6, 1 - 1e-6)
            bc = np.clip(base[ok], 1e-6, 1 - 1e-6)
            yo = y[ok]
            ll = -np.mean(yo * np.log(pc) + (1 - yo) * np.log(1 - pc))
            llb = -np.mean(yo * np.log(bc) + (1 - yo) * np.log(1 - bc))
            br = np.mean((pred[ok] - yo) ** 2)
            brb = np.mean((base[ok] - yo) ** 2)
            out.update({"log_loss_base": float(llb), "brier_base": float(brb),
                        "log_loss_skill": float(1 - ll / llb) if llb > 0 else None,
                        "brier_skill": float(1 - br / brb) if brb > 0 else None,
                        "sharpness": out.get("p_std")})
        return out
    out = regression_metrics(pred[ok], y[ok], y_raw=None if y_raw is None else
                             np.asarray(y_raw, dtype=np.float64)[ok])
    if out.get("n", 0) >= 100:
        err = (pred[ok] - y[ok]) ** 2
        errb = (base[ok] - y[ok]) ** 2
        out["mse_base"] = float(errb.mean())
        out["mse_skill"] = float(1 - err.mean() / errb.mean()) if errb.mean() > 0 else None
    return out


def period_table(task: str, preds: dict[str, np.ndarray], y: np.ndarray, base: np.ndarray,
                 periods: np.ndarray, *, labels: dict[int, str] | None = None,
                 min_rows: int = 0, y_raw: np.ndarray | None = None
                 ) -> list[dict[str, Any]]:
    """Metrics of every series in *preds* per period code (negative codes skipped)."""
    out = []
    periods = np.asarray(periods)
    for g in np.unique(periods):
        if g < 0:
            continue
        m = periods == g
        for name, pred in preds.items():
            met = score(task, pred[m], y[m], base[m], None if y_raw is None else y_raw[m])
            out.append({"period": int(g), "label": (labels or {}).get(int(g), str(int(g))),
                        "series": name, "few_rows": bool(int(m.sum()) < min_rows),
                        **{k: v for k, v in met.items() if not isinstance(v, (list, dict))}})
    return out


def weight_stability(weights: dict[str, np.ndarray], names: list[str], *,
                     range_flag: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """(per block x model rows, summary) for one weighting scheme's per-block weights."""
    rows = []
    blocks = list(weights)
    mat = np.array([weights[b] for b in blocks], dtype=np.float64)
    for bi, b in enumerate(blocks):
        h, eff = weight_entropy(mat[bi])
        for j, name in enumerate(names):
            rows.append({"block": b, "model": name, "weight": float(mat[bi, j]),
                         "entropy": h, "effective_models": eff})
    if mat.shape[0] >= 2:
        span = mat.max(axis=0) - mat.min(axis=0)
        sd = mat.std(axis=0, ddof=1)
    else:
        span = np.zeros(mat.shape[1])
        sd = np.zeros(mat.shape[1])
    entropies = [weight_entropy(w)[0] for w in mat]
    summary = {"blocks": len(blocks), "max_weight_range": float(span.max()) if span.size else 0.0,
               "mean_weight_sd": float(sd.mean()) if sd.size else 0.0,
               "model_with_largest_range": names[int(np.argmax(span))] if span.size else None,
               "mean_entropy": float(np.mean(entropies)) if entropies else None,
               "max_entropy": float(np.log(len(names))) if names else None,
               "unstable": bool(span.size and span.max() > range_flag)}
    return rows, summary
