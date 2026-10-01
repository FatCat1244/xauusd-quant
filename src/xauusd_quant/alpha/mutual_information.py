r"""Mutual information between a feature and a target, and between features
(Prompt #8, Steps 35-37, 41).

Estimator: the **copula histogram**. Each variable is replaced by its rank
score :math:`u \in (0, 1)` (so the marginals are uniform and the estimate is
invariant to any monotone transform), the unit square is cut into ``B x B``
equal cells, and

.. math:: \hat I = \sum_{ij} \hat p_{ij} \ln \frac{\hat p_{ij}}{\hat p_{i\cdot} \hat p_{\cdot j}}
          - \frac{(B_{xy} - B_x - B_y + 1)}{2n}

with the Miller-Madow correction for the plug-in bias (``B_*`` the occupied
cells). With hundreds of thousands of pairs per estimate and ``B = 20`` the
remaining bias is ~1e-4 nats - still not zero, which is why every MI is read
against the same estimator on a shifted or permuted target, never alone.
It sees non-monotone dependence (a V, a U) that a rank correlation misses, at
a resolution of 1/B.

Conditional MI :math:`I(X; Y \mid V) = \sum_v p(v) I(X; Y \mid V = v)` uses the
same estimator within the (causal) volatility quartiles.

For the many repeated estimates of the research (every feature against every
target, under 20 shifts and 10 permutations each) :func:`sentinel_codes` and
:func:`joint_counts` give the same tables without masking: a missing value
goes to an extra cell that is dropped afterwards, and a circular shift is two
slices rather than a copy.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "binned_codes",
    "conditional_mi",
    "conditional_mi_from_codes",
    "copula_mi",
    "joint_counts",
    "mi_from_joint",
    "normalized_mi",
    "sentinel_codes",
]


def binned_codes(u: np.ndarray, bins: int) -> np.ndarray:
    """Cell index 0..bins-1 of rank scores in (0, 1); -1 where missing."""
    out = np.full(u.size, -1, dtype=np.int64)
    ok = np.isfinite(u)
    out[ok] = np.minimum((u[ok] * bins).astype(np.int64), bins - 1)
    return out


def mi_from_joint(joint: np.ndarray) -> tuple[float, int]:
    """Miller-Madow corrected MI (nats) of a (bins, bins) count table, and its total."""
    n = int(joint.sum())
    if n < 10:
        return float("nan"), n
    p = joint / n
    px = p.sum(axis=1)
    py = p.sum(axis=0)
    nz = p > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        plug = float((p[nz] * np.log(p[nz] / (px[:, None] * py[None, :])[nz])).sum())
    correction = (int(nz.sum()) - int((px > 0).sum()) - int((py > 0).sum()) + 1) / (2.0 * n)
    return plug - correction, n


def _mi_from_codes(bx: np.ndarray, by: np.ndarray, bins: int) -> tuple[float, int]:
    ok = (bx >= 0) & (by >= 0)
    joint = np.bincount(bx[ok] * bins + by[ok], minlength=bins * bins).reshape(bins, bins)
    return mi_from_joint(joint)


def sentinel_codes(codes: np.ndarray, bins: int, *, scale: int = 1) -> np.ndarray:
    """int32 cell codes with missing (-1) sent to the extra cell *bins*, times *scale*."""
    c = np.where(codes >= 0, codes, bins).astype(np.int32)
    return c * np.int32(scale) if scale != 1 else c


def joint_counts(cx: np.ndarray, cy: np.ndarray, bins: int, *, shift: int = 0,
                 order: np.ndarray | None = None) -> np.ndarray:
    """(bins, bins) counts of (x_t, y_(t+shift)) - circularly - over pairs with both present.

    *cx* must be ``sentinel_codes(bx, bins, scale=bins + 1)`` and *cy*
    ``sentinel_codes(by, bins)``; *order* permutes *cx* (a permutation null).
    """
    k = bins + 1
    x = cx if order is None else cx[order]
    n = x.size
    s = shift % n
    if s == 0:
        counts = np.bincount(x + cy, minlength=k * k)
    else:
        counts = (np.bincount(x[: n - s] + cy[s:], minlength=k * k)
                  + np.bincount(x[n - s:] + cy[:s], minlength=k * k))
    return counts.reshape(k, k)[:bins, :bins]


def conditional_mi_from_codes(cx: np.ndarray, cy: np.ndarray, groups: np.ndarray, n_groups: int,
                              bins: int) -> tuple[float, dict[int, float]]:
    """:func:`conditional_mi` from sentinel codes; *groups* 0..n_groups-1, or -1 (left out)."""
    k = bins + 1
    g = np.where(groups >= 0, groups, n_groups).astype(np.int64)
    counts = np.bincount(g * (k * k) + cx + cy, minlength=(n_groups + 1) * k * k)
    tables = counts.reshape(n_groups + 1, k, k)[:n_groups, :bins, :bins]
    total = int(tables.sum())
    if total == 0:
        return float("nan"), {}
    parts: dict[int, float] = {}
    acc = 0.0
    for j in range(n_groups):
        mi, n = mi_from_joint(tables[j])
        if np.isfinite(mi):
            parts[j] = mi
            acc += (n / total) * mi
    return acc, parts


def copula_mi(bx: np.ndarray, by: np.ndarray, bins: int) -> tuple[float, int]:
    """Bias-corrected copula MI (nats) of two binned rank-score arrays, and the pair count."""
    return _mi_from_codes(bx, by, bins)


def conditional_mi(bx: np.ndarray, by: np.ndarray, groups: np.ndarray, bins: int
                   ) -> tuple[float, dict[int, float]]:
    """sum_v p(v) I(X; Y | V = v) over the groups >= 0, and the per-group MI."""
    ok = (bx >= 0) & (by >= 0) & (groups >= 0)
    total = int(ok.sum())
    if total == 0:
        return float("nan"), {}
    parts: dict[int, float] = {}
    acc = 0.0
    for g in np.unique(groups[ok]):
        sel = ok & (groups == g)
        mi, n = _mi_from_codes(bx[sel], by[sel], bins)
        if np.isfinite(mi):
            parts[int(g)] = mi
            acc += (n / total) * mi
    return acc, parts


def normalized_mi(bx: np.ndarray, by: np.ndarray, bins: int) -> float:
    """MI / min(H(X), H(Y)) in [0, 1] - near 1: one variable nearly determines the other."""
    ok = (bx >= 0) & (by >= 0)
    n = int(ok.sum())
    if n < 10:
        return float("nan")
    mi, _ = _mi_from_codes(bx, by, bins)

    def entropy(codes: np.ndarray) -> float:
        counts = np.bincount(codes, minlength=bins) / n
        c = counts[counts > 0]
        return float(-(c * np.log(c)).sum())

    h = min(entropy(bx[ok]), entropy(by[ok]))
    return float(mi / h) if h > 0 else float("nan")
