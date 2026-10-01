r"""Structural breaks by PELT (Steps 49-50) - research only, not a regime model.

A change point is a *once-only* break in a series' level: the segment before
and after have different means. That is a different concept from a latent
state that the market enters and leaves repeatedly, and the two answer
different questions: an HMM describes switching on the scale of hours to
weeks; change points describe eras. Comparing them tells whether the regime
states are really eras in disguise (a state that occupies one long stretch of
years and never returns) or genuinely recurring.

PELT (Killick, Fearnhead & Eckley 2012) finds the exact minimiser of
:math:`\sum_{segments} C(y_{a:b}) + \beta \cdot (\text{segments} - 1)` with the
Gaussian mean-change cost :math:`C = \sum (y - \bar y)^2 / \hat\sigma^2`, pruning
candidates that can never be optimal. :math:`\hat\sigma^2` is estimated
robustly from first differences (MAD), and the penalty is the BIC-type
:math:`\beta = 2 \ln n` times ``penalty_multiplier``. It runs on daily medians
(a few thousand points), never on bars.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["pelt_mean", "segments_table"]


def _robust_variance(y: np.ndarray) -> float:
    diff = np.diff(y)
    mad = np.median(np.abs(diff - np.median(diff)))
    sigma = 1.4826 * mad / np.sqrt(2.0)
    return float(sigma ** 2) if sigma > 0 else float(np.var(y) or 1.0)


def pelt_mean(values: np.ndarray, *, penalty: float | None = None, min_size: int = 2,
              penalty_multiplier: float = 1.0) -> tuple[list[int], dict[str, Any]]:
    """Change points (indices where a new segment starts) of *values* by PELT."""
    y = np.asarray(values, dtype=np.float64)
    y = y[np.isfinite(y)]
    n = y.size
    if n < 2 * min_size:
        return [], {"n": n, "penalty": None, "variance": None}
    var = _robust_variance(y)
    beta = penalty if penalty is not None else penalty_multiplier * 2.0 * np.log(n)
    s1 = np.concatenate(([0.0], np.cumsum(y)))
    s2 = np.concatenate(([0.0], np.cumsum(y * y)))

    def cost(a: np.ndarray, b: int) -> np.ndarray:
        length = b - a
        mean_sum = s1[b] - s1[a]
        return ((s2[b] - s2[a]) - mean_sum * mean_sum / length) / var

    best = np.full(n + 1, np.inf)          # F(t): optimal penalised cost of y[0:t]
    best[0] = -beta
    last = np.zeros(n + 1, dtype=np.int64)
    candidates = np.array([0], dtype=np.int64)  # possible starts of the last segment
    for t in range(1, n + 1):
        ready = t - candidates >= min_size
        usable = candidates[ready]
        if usable.size:
            fit = best[usable] + cost(usable, t)
            total = fit + beta
            j = int(np.argmin(total))
            best[t] = total[j]
            last[t] = usable[j]
            # PELT pruning: a start that cannot beat F(t) now never will.
            candidates = np.concatenate([usable[fit <= best[t]], candidates[~ready]])
        candidates = np.append(candidates, t)
    points = []
    t = n
    while t > 0:
        t = int(last[t])
        if t > 0:
            points.append(t)
    return sorted(points), {"n": n, "penalty": float(beta), "variance": var}


def segments_table(values: np.ndarray, points: list[int]) -> list[dict[str, Any]]:
    """Start, end, length and mean of each segment between change points."""
    y = np.asarray(values, dtype=np.float64)
    edges = [0, *points, y.size]
    return [{"segment": i, "start": edges[i], "end": edges[i + 1],
             "length": edges[i + 1] - edges[i], "mean": float(np.nanmean(y[edges[i]:edges[i + 1]]))}
            for i in range(len(edges) - 1)]
