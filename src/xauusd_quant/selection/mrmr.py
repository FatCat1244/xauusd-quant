r"""Minimum-redundancy maximum-relevance selection and the simpler rankings it is compared with
(Prompt #9, Steps 13, 41).

Greedy mRMR (Peng, Long & Ding 2005): start from the most relevant candidate,
then repeatedly add the candidate ``j`` maximising

.. math::

    \text{difference (MID):}\quad \tilde r_j - \lambda\,\overline{|\rho|}_{j,S}
    \qquad\text{or}\qquad
    \text{quotient (MIQ):}\quad \tilde r_j / (\epsilon + \overline{|\rho|}_{j,S})

where :math:`\tilde r_j` is the relevance divided by the largest candidate
relevance (so :math:`\lambda` has one meaning for every target), relevance is
|rank IC| or copula mutual information with the target, and redundancy is the
mean |Spearman| with the features already selected. A candidate whose |rho|
with any selected feature reaches ``hard_limit`` is never added (the
``max_pairwise_correlation`` rule). Ties break on the candidate order, which
callers make deterministic (registry order).

The simpler views it is compared with: **IC ranking** (relevance alone, no
redundancy term) and the **effect-stability ranking** (Step 41) with the
documented score ``|median yearly rank IC| x max(0, 2 c - 1)`` for yearly sign
consistency ``c`` - a feature whose yearly sign is a coin flip scores 0,
whatever its pooled IC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["MRMRStep", "effect_stability_score", "ic_ranking", "mrmr"]


@dataclass(frozen=True)
class MRMRStep:
    index: int
    score: float
    relevance: float
    redundancy: float


def mrmr(relevance: np.ndarray, abs_corr: np.ndarray, candidates: list[int], *, k: int,
         lam: float = 1.0, scheme: str = "difference", hard_limit: float = 0.95,
         eps: float = 1e-3) -> list[MRMRStep]:
    """Greedy mRMR over *candidates* (indices into *relevance* / *abs_corr*)."""
    rel = np.asarray(relevance, dtype=np.float64)
    pool = [c for c in candidates if np.isfinite(rel[c]) and rel[c] > 0]
    if not pool or k <= 0:
        return []
    top = max(rel[c] for c in pool)
    norm = {c: rel[c] / top for c in pool}
    corr = np.nan_to_num(np.abs(np.asarray(abs_corr, dtype=np.float64)), nan=0.0)
    first = max(pool, key=lambda c: (norm[c], -pool.index(c)))
    steps = [MRMRStep(first, norm[first], float(rel[first]), 0.0)]
    chosen = [first]
    red_sum = corr[first].copy()
    red_max = corr[first].copy()
    remaining = [c for c in pool if c != first]
    while remaining and len(chosen) < k:
        best, best_score, best_red = None, -np.inf, 0.0
        for c in remaining:
            if red_max[c] >= hard_limit:
                continue
            red = red_sum[c] / len(chosen)
            score = norm[c] - lam * red if scheme == "difference" else norm[c] / (eps + red)
            if score > best_score:
                best, best_score, best_red = c, score, red
        if best is None:
            break
        steps.append(MRMRStep(best, float(best_score), float(rel[best]), float(best_red)))
        chosen.append(best)
        remaining.remove(best)
        red_sum += corr[best]
        red_max = np.maximum(red_max, corr[best])
    return steps


def ic_ranking(relevance: np.ndarray, candidates: list[int], *, k: int) -> list[int]:
    """Top-k by relevance alone (no redundancy control): the naive comparison."""
    rel = np.asarray(relevance, dtype=np.float64)
    pool = [c for c in candidates if np.isfinite(rel[c]) and rel[c] > 0]
    return sorted(pool, key=lambda c: (-rel[c], pool.index(c)))[:k]


def effect_stability_score(median_yearly_ic: np.ndarray, sign_consistency: np.ndarray
                           ) -> np.ndarray:
    """|median yearly IC| x max(0, 2 c - 1); NaN inputs score 0."""
    med = np.nan_to_num(np.abs(np.asarray(median_yearly_ic, dtype=np.float64)), nan=0.0)
    c = np.nan_to_num(np.asarray(sign_consistency, dtype=np.float64), nan=0.5)
    return med * np.maximum(0.0, 2.0 * c - 1.0)
