r"""How different are the constituents, really? (Prompt #11, Steps 6-7, 58, 77)

Different library names do not mean different information (Rule 1), so
diversity is read several ways, pair of models by pair of models, on the same
out-of-sample rows:

* prediction correlation, Pearson and Spearman (Step 6);
* error correlation :math:`\mathrm{corr}(e_i, e_j)`, :math:`e = y - \hat y` (Step 7):
  similar outputs with different errors may still diversify, different outputs
  with the same failures do not;
* the error covariance and the mean absolute disagreement :math:`E|p_i - p_j|`;
* classification only: the correlation of the *calibration residuals*
  :math:`\bar y_{bin(p_{i,t})} - p_{i,t}` (whether two models are mis-calibrated in
  the same places) and Yule's Q-statistic of their correctness at 0.5,
  :math:`Q = (N^{11}N^{00} - N^{01}N^{10}) / (N^{11}N^{00} + N^{01}N^{10})`.

A duplicated model shows correlation 1 on every view (Step 77).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..alpha.information_coefficient import rank_scores

__all__ = ["calibration_residuals", "diversity_table", "q_statistic"]


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        r = float(np.corrcoef(a[ok], b[ok])[0, 1])
    return r if np.isfinite(r) else None


def calibration_residuals(p: np.ndarray, y: np.ndarray, *, bins: int = 20) -> np.ndarray:
    """Observed frequency of the row's quantile bin minus its probability."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    out = np.full(p.size, np.nan)
    ok = np.isfinite(p) & np.isfinite(y)
    if ok.sum() < bins:
        return out
    edges = np.unique(np.quantile(p[ok], np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        return out
    idx = np.clip(np.searchsorted(edges, p[ok], side="right") - 1, 0, edges.size - 2)
    freq = np.bincount(idx, weights=y[ok], minlength=edges.size - 1) / np.maximum(
        np.bincount(idx, minlength=edges.size - 1), 1)
    out[ok] = freq[idx] - p[ok]
    return out


def q_statistic(p_a: np.ndarray, p_b: np.ndarray, y: np.ndarray) -> float | None:
    ok = np.isfinite(p_a) & np.isfinite(p_b) & np.isfinite(y)
    if ok.sum() < 10:
        return None
    ca = (p_a[ok] >= 0.5) == (y[ok] > 0.5)
    cb = (p_b[ok] >= 0.5) == (y[ok] > 0.5)
    n11 = float(np.sum(ca & cb))
    n00 = float(np.sum(~ca & ~cb))
    n10 = float(np.sum(ca & ~cb))
    n01 = float(np.sum(~ca & cb))
    den = n11 * n00 + n01 * n10
    return (n11 * n00 - n01 * n10) / den if den > 0 else None


def diversity_table(p: np.ndarray, y: np.ndarray, names: list[str], *, task: str,
                    max_rows: int | None = 400_000) -> list[dict[str, Any]]:
    """One row per pair of constituents with every diversity view (evenly spaced rows
    beyond *max_rows* - the figures are then estimates, flagged ``sampled``)."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(p).all(axis=1) & np.isfinite(y)
    p, y = p[ok], y[ok]
    sampled = False
    if max_rows and p.shape[0] > max_rows:
        sel = np.unique(np.linspace(0, p.shape[0] - 1, max_rows).astype(np.int64))
        p, y = p[sel], y[sel]
        sampled = True
    ranks = np.column_stack([rank_scores(p[:, j]) for j in range(p.shape[1])])
    err = y[:, None] - p
    calres = (np.column_stack([calibration_residuals(p[:, j], y) for j in range(p.shape[1])])
              if task == "classification" else None)
    rows = []
    for i in range(p.shape[1]):
        for j in range(i + 1, p.shape[1]):
            row: dict[str, Any] = {
                "a": names[i], "b": names[j], "rows": int(p.shape[0]), "sampled": sampled,
                "pearson": _corr(p[:, i], p[:, j]), "spearman": _corr(ranks[:, i], ranks[:, j]),
                "error_correlation": _corr(err[:, i], err[:, j]),
                "error_covariance": float(np.cov(err[:, i], err[:, j])[0, 1]),
                "mean_abs_disagreement": float(np.mean(np.abs(p[:, i] - p[:, j])))}
            if calres is not None:
                row["calibration_residual_correlation"] = _corr(calres[:, i], calres[:, j])
                row["q_statistic"] = q_statistic(p[:, i], p[:, j], y)
            rows.append(row)
    return rows
