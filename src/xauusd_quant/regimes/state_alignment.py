r"""State alignment: the label-switching problem (Step 23).

An unsupervised fit numbers its states arbitrarily - refit the same data from
another start and ``state_0`` may come back as ``state_2``. Every comparison
across fits (refits, schemes, nulls) therefore first matches states.

Matching is a linear assignment (Hungarian algorithm, ``scipy.optimize``) on
a cost matrix of **Bhattacharyya distances** between the two fits' Gaussian
states, computed in raw feature units. The Bhattacharyya distance is
invariant under any common affine map, so two models fitted with different
scalers are compared exactly; it sees means *and* covariances. Transition
behaviour can be added as a second cost (the difference in expected log
duration). K-Means centres, which have no covariance, are compared by squared
distance in the reference scaler's units.

The first model of a walk-forward (and every offline fit) is put in a
canonical order - ascending mean of one reference feature - purely so that
runs are reproducible and tables line up. The order carries no meaning and
the IDs stay numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .emissions import bhattacharyya

__all__ = ["Alignment", "align_states", "canonical_order", "cost_matrix", "permutation_inverse"]


@dataclass(frozen=True)
class Alignment:
    """``order[i]`` is the new model's state that becomes label ``i``."""

    order: np.ndarray
    costs: np.ndarray            # matched cost per label
    total_cost: float
    ambiguity: np.ndarray        # matched cost / second-best cost in the row (lower = clearer)

    @property
    def identity(self) -> bool:
        return bool(np.array_equal(self.order, np.arange(self.order.size)))


def cost_matrix(ref_means: np.ndarray, ref_covs: np.ndarray | None, new_means: np.ndarray,
                new_covs: np.ndarray | None, *, ref_durations: np.ndarray | None = None,
                new_durations: np.ndarray | None = None, duration_weight: float = 0.0,
                scale: np.ndarray | None = None) -> np.ndarray:
    """Pairwise dissimilarity (reference states x new states)."""
    k_ref, k_new = ref_means.shape[0], new_means.shape[0]
    cost = np.zeros((k_ref, k_new))
    for i in range(k_ref):
        for j in range(k_new):
            if ref_covs is not None and new_covs is not None:
                cost[i, j] = bhattacharyya(ref_means[i], ref_covs[i], new_means[j], new_covs[j])
            else:
                s = np.ones(ref_means.shape[1]) if scale is None else scale
                cost[i, j] = float((((ref_means[i] - new_means[j]) / s) ** 2).sum())
    if duration_weight > 0 and ref_durations is not None and new_durations is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.log(np.clip(ref_durations, 1.0, 1e9))
            ln = np.log(np.clip(new_durations, 1.0, 1e9))
        cost = cost + duration_weight * np.abs(lr[:, None] - ln[None, :])
    return cost


def align_states(cost: np.ndarray) -> Alignment:
    """Minimum-cost one-to-one matching (square cost matrix)."""
    if cost.shape[0] != cost.shape[1]:
        raise ValueError("alignment needs the same number of states in both models")
    rows, cols = linear_sum_assignment(cost)
    order = np.empty(cost.shape[0], dtype=np.int64)
    order[rows] = cols
    matched = cost[np.arange(cost.shape[0]), order]
    ambiguity = np.full(cost.shape[0], np.nan)
    if cost.shape[1] > 1:
        for i in range(cost.shape[0]):
            others = np.delete(cost[i], order[i])
            best_other = others.min()
            ambiguity[i] = matched[i] / best_other if best_other > 0 else np.inf
    return Alignment(order=order, costs=matched, total_cost=float(matched.sum()),
                     ambiguity=ambiguity)


def canonical_order(means_raw: np.ndarray, feature_index: int) -> np.ndarray:
    """States by ascending mean of one reference feature (ties by the next features)."""
    keys = [means_raw[:, j] for j in reversed(range(means_raw.shape[1]))]
    keys.append(means_raw[:, feature_index])
    return np.lexsort(keys)


def permutation_inverse(order: np.ndarray) -> np.ndarray:
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    return inv
