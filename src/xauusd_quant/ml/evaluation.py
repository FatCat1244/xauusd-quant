r"""Out-of-sample metrics (Prompt #10, Steps 24-25, 28, 52-59).

Classification (probability ``p`` of a 0/1 label): ROC AUC, PR AUC, log loss,
Brier score, their *skill* against the constant baseline (the training
positive rate) - :math:`1 - \mathrm{LL}/\mathrm{LL}_{base}`, positive = better
than the baseline - accuracy / precision / recall at 0.5 (secondary), the
expected calibration error and the rank IC of the score with the label.

Regression (prediction ``f`` of ``y`` on the fitted scale): MAE, RMSE, the MSE
skill against the training mean, :math:`R^2` against the block's own mean,
Pearson and Spearman correlation, the rank IC against the untransformed outcome
and the bias. Small correlations are meaningful only when they are stable, so
every comparison is also read block by block (:func:`paired_blocks`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..alpha.information_coefficient import rank_scores
from .calibration import expected_calibration_error

__all__ = [
    "classification_metrics",
    "decile_table",
    "grouped_metrics",
    "paired_blocks",
    "regression_metrics",
]

_EPS = 1e-6


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 3:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        r = float(np.corrcoef(a, b)[0, 1])
    return r if np.isfinite(r) else None


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 3:
        return None
    return _corr(rank_scores(a), rank_scores(b))


def classification_metrics(p: np.ndarray, y: np.ndarray, *, base_rate: float | None = None,
                           bins: int = 20) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    n = int(p.size)
    out: dict[str, Any] = {"n": n}
    if n < 100:
        return out
    pc = np.clip(p, _EPS, 1.0 - _EPS)
    ll = float(-np.mean(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc)))
    brier = float(np.mean((p - y) ** 2))
    pos = float(y.mean())
    two = 0.0 < pos < 1.0
    out.update({
        "positive_rate": pos, "mean_p": float(p.mean()),
        "auc": float(roc_auc_score(y, p)) if two else None,
        "pr_auc": float(average_precision_score(y, p)) if two else None,
        "log_loss": ll, "brier": brier,
        "accuracy": float(np.mean((p >= 0.5) == (y > 0.5))),
        "ece": expected_calibration_error(p, y, bins=bins),
        "rank_ic": _spearman(p, y),
        "share_p_above_0.6": float(np.mean(p > 0.6)),
        "share_p_above_0.7": float(np.mean(p > 0.7)),
        "share_p_below_0.4": float(np.mean(p < 0.4)),
        "p_std": float(p.std())})
    pred = p >= 0.5
    tp = float(np.sum(pred & (y > 0.5)))
    out["precision"] = tp / float(pred.sum()) if pred.any() else None
    out["recall"] = tp / float((y > 0.5).sum()) if (y > 0.5).any() else None
    if base_rate is not None and 0.0 < base_rate < 1.0:
        b = float(base_rate)
        ll_base = float(-np.mean(y * np.log(b) + (1.0 - y) * np.log(1.0 - b)))
        brier_base = float(np.mean((b - y) ** 2))
        out["log_loss_base"] = ll_base
        out["brier_base"] = brier_base
        out["log_loss_skill"] = 1.0 - ll / ll_base if ll_base > 0 else None
        out["brier_skill"] = 1.0 - brier / brier_base if brier_base > 0 else None
    return out


def regression_metrics(f: np.ndarray, y: np.ndarray, *, y_raw: np.ndarray | None = None,
                       base_value: float | None = None) -> dict[str, Any]:
    f = np.asarray(f, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(f) & np.isfinite(y)
    if y_raw is not None:
        ok &= np.isfinite(y_raw)
    fo, yo = f[ok], y[ok]
    n = int(fo.size)
    out: dict[str, Any] = {"n": n}
    if n < 100:
        return out
    err = fo - yo
    mse = float(np.mean(err ** 2))
    sst = float(np.mean((yo - yo.mean()) ** 2))
    out.update({"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(mse)),
                "r2": 1.0 - mse / sst if sst > 0 else None, "bias": float(err.mean()),
                "pearson": _corr(fo, yo), "spearman": _spearman(fo, yo),
                "rank_ic": _spearman(fo, yo),
                "pred_std": float(fo.std()), "target_std": float(yo.std())})
    if y_raw is not None:
        out["rank_ic_raw"] = _spearman(fo, np.asarray(y_raw, dtype=np.float64)[ok])
    if base_value is not None:
        mse_base = float(np.mean((yo - base_value) ** 2))
        out["mse_base"] = mse_base
        out["mse_skill"] = 1.0 - mse / mse_base if mse_base > 0 else None
    return out


def grouped_metrics(pred: np.ndarray, y: np.ndarray, groups: np.ndarray, *, task: str,
                    base: float | None = None, y_raw: np.ndarray | None = None,
                    labels: dict[int, str] | None = None, min_rows: int = 2000
                    ) -> list[dict[str, Any]]:
    """The metrics inside every group code (negative codes = undetermined, skipped)."""
    rows = []
    for g in np.unique(groups):
        if g < 0:
            continue
        m = groups == g
        if m.sum() < min_rows:
            continue
        if task == "classification":
            met = classification_metrics(pred[m], y[m], base_rate=base)
        else:
            met = regression_metrics(pred[m], y[m], y_raw=None if y_raw is None else y_raw[m],
                                     base_value=base)
        rows.append({"group": int(g), "label": (labels or {}).get(int(g), str(int(g))), **met})
    return rows


def decile_table(score: np.ndarray, outcome: np.ndarray, *, q: int = 10) -> dict[str, Any]:
    """Outcome by score decile, the top-minus-bottom spread and the monotonicity (Steps 57-58)."""
    s = np.asarray(score, dtype=np.float64)
    o = np.asarray(outcome, dtype=np.float64)
    ok = np.isfinite(s) & np.isfinite(o)
    s, o = s[ok], o[ok]
    if s.size < 10 * q:
        return {"rows": [], "spread": None, "monotonicity": None}
    ranks = rank_scores(s)
    bucket = np.clip((ranks * q).astype(np.int64), 0, q - 1)
    rows = []
    for b in range(q):
        m = bucket == b
        if not m.any():                  # tied scores (a constant model) leave buckets empty
            continue
        rows.append({"decile": b + 1, "mean_score": float(s[m].mean()),
                     "mean_outcome": float(o[m].mean()), "n": int(m.sum())})
    if len(rows) < 2:
        return {"rows": rows, "spread": None, "monotonicity": None}
    means = np.array([r["mean_outcome"] for r in rows])
    mono = _spearman(np.array([r["decile"] for r in rows], dtype=np.float64), means)
    return {"rows": rows, "spread": float(means[-1] - means[0]), "monotonicity": mono}


def paired_blocks(a: list[float | None], b: list[float | None]) -> dict[str, Any]:
    """Block-by-block comparison of two models' metrics (a - b): mean, SE, wins, t."""
    d = np.array([x - y for x, y in zip(a, b, strict=True) if x is not None and y is not None],
                 dtype=np.float64)
    if d.size == 0:
        return {"blocks": 0}
    se = float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else None
    return {"blocks": int(d.size), "mean_diff": float(d.mean()), "se": se,
            "wins": int((d > 0).sum()),
            "t": float(d.mean() / se) if se not in (None, 0.0) else None}
