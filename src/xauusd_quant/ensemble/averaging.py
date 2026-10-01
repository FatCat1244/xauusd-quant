r"""Fixed-weight combinations of constituent predictions (Prompt #11, Steps 10-11).

* :func:`weighted_mean`: :math:`\hat y = \sum_i w_i \hat y_i` with
  :math:`\sum_i w_i = 1`; equal weights are the simple average, the benchmark
  every other combination has to beat. Probabilities are averaged as
  probabilities (Step 10), so an average of calibrated probabilities is a
  probability - not necessarily a calibrated one (Step 33).
* :func:`median_combination`: the row-wise median, robust to one extreme model.

Rows where any constituent is missing are NaN: nothing is renormalised over the
models that happen to be present (that is a fallback policy, Step 68, and the
default is to refuse).
"""

from __future__ import annotations

import numpy as np

__all__ = ["equal_weights", "normalise", "median_combination", "weighted_mean"]


def equal_weights(m: int) -> np.ndarray:
    if m < 1:
        raise ValueError("an ensemble needs at least one constituent")
    return np.full(m, 1.0 / m)


def normalise(w: np.ndarray) -> np.ndarray:
    """Non-negative weights summing to one (equal weights when all are zero)."""
    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 1 or w.size == 0:
        raise ValueError("weights must be a non-empty vector")
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError(f"weights must be finite and non-negative: {w}")
    total = float(w.sum())
    return equal_weights(w.size) if total <= 0 else w / total


def weighted_mean(x: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Row-wise weighted mean of the (n, M) predictions; NaN where any input is NaN.

    *w* is one weight vector (M,) or one per row (n, M); each must sum to one.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("predictions must be an (n, M) matrix")
    m = x.shape[1]
    if w is None:
        w = equal_weights(m)
    w = np.asarray(w, dtype=np.float64)
    if w.ndim == 1:
        if w.size != m:
            raise ValueError(f"{w.size} weights for {m} constituents")
        if not np.isclose(w.sum(), 1.0, atol=1e-9):
            raise ValueError(f"weights sum to {w.sum()}, not 1")
        return x @ w                                  # NaN propagates: never renormalised
    if w.shape != x.shape:
        raise ValueError(f"row weights {w.shape} do not match predictions {x.shape}")
    if not np.allclose(w.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("every row's weights must sum to 1")
    return np.einsum("ij,ij->i", x, w)


def median_combination(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.median(x, axis=1)                        # NaN in any column -> NaN
    return np.where(np.isfinite(x).all(axis=1), out, np.nan)
