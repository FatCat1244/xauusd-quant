r"""Redundancy and collinearity of feature sets (Prompt #9, Steps 9, 47-50).

Correlations are measured on evenly spaced *development* rows (feature values
only - no outcome), pairwise-complete, Pearson and Spearman. For a set of
features:

* **condition number** of its Spearman correlation matrix,
* **variance inflation factors** ``VIF_i = [R^{-1}]_{ii}`` (diagnostic only:
  heavy tails and nonlinear dependence make VIF a rough guide),
* **effective rank** ``exp(H)`` with ``H = -sum p_i ln p_i`` over the
  normalised eigenvalues ``p_i = lambda_i / sum lambda`` - 50 features that
  span 5 directions have an effective rank near 5,
* the largest pairwise |Spearman| (the hard limit is
  ``max_pairwise_correlation``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..features.redundancy import evenly_spaced_rows, pairwise_correlation

__all__ = [
    "correlation_matrices",
    "effective_rank",
    "set_collinearity",
]


def correlation_matrices(features: np.ndarray, lo: int, hi: int, *, sample_rows: int
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(Pearson, Spearman, pair counts) over evenly spaced rows of [lo, hi) (estimates)."""
    rows = lo + evenly_spaced_rows(hi - lo, sample_rows)
    sample = features[rows].astype(np.float64)
    pearson, counts = pairwise_correlation(sample)
    spearman, _ = pairwise_correlation(sample, rank=True)
    return pearson, spearman, counts


def _clean(corr: np.ndarray) -> np.ndarray:
    c = np.nan_to_num(np.asarray(corr, dtype=np.float64), nan=0.0)
    c = (c + c.T) / 2.0
    np.fill_diagonal(c, 1.0)
    return c


def effective_rank(corr: np.ndarray) -> float:
    """exp(entropy of the normalised eigenvalues) of a correlation matrix."""
    if corr.shape[0] == 0:
        return 0.0
    lam = np.clip(np.linalg.eigvalsh(_clean(corr)), 0.0, None)
    total = lam.sum()
    if total <= 0:
        return 0.0
    p = lam[lam > 0] / total
    return float(np.exp(-(p * np.log(p)).sum()))


def set_collinearity(corr: np.ndarray, idx: list[int], names: list[str]) -> dict[str, Any]:
    """Condition number, VIFs, effective rank and the largest |rho| of the features *idx*."""
    if not idx:
        return {"features": 0}
    sub = _clean(corr[np.ix_(idx, idx)])
    lam = np.linalg.eigvalsh(sub)
    cond = float(lam.max() / max(lam.min(), 1e-12))
    try:
        vif = np.diag(np.linalg.inv(sub))
    except np.linalg.LinAlgError:
        vif = np.full(len(idx), np.inf)
    off = np.abs(sub - np.eye(len(idx)))
    i, j = np.unravel_index(np.argmax(off), off.shape) if len(idx) > 1 else (0, 0)
    return {"features": len(idx), "condition_number": cond,
            "effective_rank": effective_rank(sub),
            "max_vif": float(np.max(vif)), "median_vif": float(np.median(vif)),
            "max_abs_spearman": float(off.max()) if len(idx) > 1 else 0.0,
            "max_pair": [names[idx[i]], names[idx[j]]] if len(idx) > 1 else [],
            "vif": {names[k]: float(v) for k, v in zip(idx, vif, strict=True)}}
