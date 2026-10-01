r"""Model diagnostics: degeneracy, separation, agreement and feature contribution.

**Degenerate states** (Step 20). A model is flagged - never silently accepted
because its likelihood is high - when a state holds almost no bars
(``tiny_state``) or almost all of them (``dominant_state``), a covariance
collapses (condition number above the cap: ``covariance_collapse``), two
states are near-duplicates (Bhattacharyya distance below the threshold:
``duplicate_states``), or the transition matrix is pathological: a state whose
expected duration is under ``min_expected_duration`` bars flickers
(``flickering_state``), one above ``max_expected_duration`` is effectively
absorbing (``absorbing_state``).

**Agreement between two labelings** is measured without matching labels:
adjusted Rand index, normalised mutual information and Cramer's V, all
invariant to renumbering. **Feature contribution** (Step 62) is the ANOVA
effect size :math:`\eta^2` (between-state share of a feature's variance), the
mutual information between the feature's deciles and the state, and the
standardised state means.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from .emissions import bhattacharyya

__all__ = [
    "adjusted_rand",
    "contingency",
    "cramers_v",
    "degenerate_flags",
    "eta_squared",
    "feature_contribution",
    "normalized_mutual_info",
    "state_separation",
]


def contingency(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Counts of (a, b) label pairs over rows where both are >= 0."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    ok = (a >= 0) & (b >= 0)
    a, b = a[ok], b[ok]
    if a.size == 0:
        return np.zeros((0, 0), dtype=np.int64)
    table = np.zeros((int(a.max()) + 1, int(b.max()) + 1), dtype=np.int64)
    np.add.at(table, (a, b), 1)
    return table


def _comb2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x * (x - 1) / 2.0


def adjusted_rand(a: np.ndarray, b: np.ndarray) -> float | None:
    """Adjusted Rand index of two labelings (1 identical up to renaming, ~0 chance)."""
    table = contingency(a, b)
    n = table.sum()
    if n < 2:
        return None
    sum_ij = _comb2(table).sum()
    sum_a = _comb2(table.sum(axis=1)).sum()
    sum_b = _comb2(table.sum(axis=0)).sum()
    expected = sum_a * sum_b / _comb2(np.array([n]))[0]
    maximum = 0.5 * (sum_a + sum_b)
    if maximum == expected:
        return 1.0 if sum_ij == expected else 0.0
    return float((sum_ij - expected) / (maximum - expected))


def normalized_mutual_info(a: np.ndarray, b: np.ndarray) -> float | None:
    """Mutual information / sqrt(H(a) H(b)) (1 = one determines the other)."""
    table = contingency(a, b).astype(np.float64)
    n = table.sum()
    if n == 0:
        return None
    p = table / n
    pa, pb = p.sum(axis=1), p.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = np.nansum(np.where(p > 0, p * np.log(p / (pa[:, None] * pb[None, :])), 0.0))
        ha = -np.nansum(np.where(pa > 0, pa * np.log(pa), 0.0))
        hb = -np.nansum(np.where(pb > 0, pb * np.log(pb), 0.0))
    if ha <= 0 or hb <= 0:
        return 0.0
    return float(mi / np.sqrt(ha * hb))


def cramers_v(table: np.ndarray) -> float | None:
    """Cramer's V of a contingency table (0 independent, 1 perfectly associated)."""
    t = np.asarray(table, dtype=np.float64)
    t = t[t.sum(axis=1) > 0][:, t.sum(axis=0) > 0]
    n = t.sum()
    if n == 0 or min(t.shape) < 2:
        return None
    expected = t.sum(axis=1, keepdims=True) * t.sum(axis=0, keepdims=True) / n
    chi2 = float(((t - expected) ** 2 / expected).sum())
    return float(np.sqrt(chi2 / (n * (min(t.shape) - 1))))


def eta_squared(values: np.ndarray, labels: np.ndarray) -> float | None:
    """Between-state share of the variance of *values* (finite values, labels >= 0)."""
    x = np.asarray(values, dtype=np.float64)
    lab = np.asarray(labels, dtype=np.int64)
    ok = np.isfinite(x) & (lab >= 0)
    x, lab = x[ok], lab[ok]
    if x.size < 3:
        return None
    total = float(((x - x.mean()) ** 2).sum())
    if total <= 0:
        return None
    counts = np.bincount(lab)
    sums = np.bincount(lab, weights=x)
    present = counts > 0
    means = sums[present] / counts[present]
    between = float((counts[present] * (means - x.mean()) ** 2).sum())
    return between / total


def _mutual_info_deciles(values: np.ndarray, labels: np.ndarray) -> float | None:
    x = np.asarray(values, dtype=np.float64)
    lab = np.asarray(labels, dtype=np.int64)
    ok = np.isfinite(x) & (lab >= 0)
    if ok.sum() < 100:
        return None
    x, lab = x[ok], lab[ok]
    edges = np.unique(np.quantile(x, np.linspace(0, 1, 11))[1:-1])
    bins = np.searchsorted(edges, x, side="right")
    table = contingency(bins, lab).astype(np.float64)
    p = table / table.sum()
    pa, pb = p.sum(axis=1), p.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.nansum(np.where(p > 0, p * np.log(p / (pa[:, None] * pb[None, :])),
                                        0.0)))


def feature_contribution(frame: pl.DataFrame, labels: np.ndarray,
                         features: list[str] | tuple[str, ...]) -> pl.DataFrame:
    """Per feature: eta^2, mutual information with the state (deciles, nats)."""
    rows = []
    for name in features:
        if name not in frame.columns:
            continue
        values = frame[name].cast(pl.Float64).fill_null(np.nan).to_numpy()
        rows.append({"feature": name, "eta_squared": eta_squared(values, labels),
                     "mutual_information_nats": _mutual_info_deciles(values, labels)})
    return pl.DataFrame(rows, infer_schema_length=None)


def state_separation(means_raw: np.ndarray, covs_raw: np.ndarray | None) -> np.ndarray:
    """Pairwise Bhattacharyya distances between a model's states (K x K)."""
    k = means_raw.shape[0]
    out = np.zeros((k, k))
    if covs_raw is None:
        return out
    for i in range(k):
        for j in range(i + 1, k):
            out[i, j] = out[j, i] = bhattacharyya(means_raw[i], covs_raw[i], means_raw[j],
                                                  covs_raw[j])
    return out


def degenerate_flags(occupancy: np.ndarray, *, covariances: np.ndarray | None,
                     separation: np.ndarray | None, transition: np.ndarray | None,
                     min_state_fraction: float, max_state_fraction: float,
                     max_condition_number: float, duplicate_bhattacharyya: float,
                     min_expected_duration: float, max_expected_duration: float
                     ) -> dict[str, Any]:
    """Every degeneracy check on one fitted model; ``degenerate`` if any fires."""
    occ = np.asarray(occupancy, dtype=np.float64)
    share = occ / occ.sum() if occ.sum() > 0 else occ
    flags: dict[str, Any] = {
        "tiny_state": bool((share < min_state_fraction).any()),
        "dominant_state": bool((share > max_state_fraction).any()),
        "min_state_share": float(share.min()) if share.size else None,
        "max_state_share": float(share.max()) if share.size else None,
    }
    if covariances is not None:
        conditions = []
        for c in covariances:
            eig = np.linalg.eigvalsh(0.5 * (c + c.T))
            conditions.append(float(eig.max() / eig.min()) if eig.min() > 0 else np.inf)
        flags["max_condition_number"] = float(max(conditions))
        flags["covariance_collapse"] = bool(max(conditions) > max_condition_number)
    if separation is not None and separation.shape[0] > 1:
        off = separation[np.triu_indices(separation.shape[0], 1)]
        flags["min_bhattacharyya"] = float(off.min())
        flags["duplicate_states"] = bool(off.min() < duplicate_bhattacharyya)
    if transition is not None:
        diag = np.diag(transition)
        with np.errstate(divide="ignore"):
            durations = np.where(diag < 1, 1.0 / (1.0 - diag), np.inf)
        flags["min_expected_duration"] = float(durations.min())
        flags["max_expected_duration"] = float(durations.max())
        flags["flickering_state"] = bool((durations < min_expected_duration).any())
        flags["absorbing_state"] = bool((durations > max_expected_duration).any())
    names = ("tiny_state", "dominant_state", "covariance_collapse", "duplicate_states",
             "flickering_state", "absorbing_state")
    fired = [name for name in names if flags.get(name)]
    flags["flags"] = ",".join(fired)
    flags["degenerate"] = bool(fired)
    return flags
