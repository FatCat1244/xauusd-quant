r"""Which features carry the same information (Prompt #8, Steps 38-42).

On an evenly spaced sample of bars (labelled as estimates):

* Pearson and Spearman correlation matrices, pairwise-complete (each pair on
  the bars where both are defined), from masked matrix products;
* hierarchical clustering on :math:`d_{ij} = 1 - |\rho^S_{ij}|` (average
  linkage), cut at ``1 - cluster_abs_corr``: a cluster is one phenomenon;
* a redundancy graph with an edge wherever :math:`|\rho^S| \ge` ``graph_abs_corr``
  - its connected components are the machine-readable redundancy groups;
* the normalised copula MI of every pair, which flags *non-monotone*
  near-duplicates that a rank correlation cannot see (Prompt #7's R^2 vs
  |slope / volatility| was one: Spearman 0.13, the V explaining 86 %).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import squareform

from ..alpha.information_coefficient import rank_scores
from ..alpha.mutual_information import binned_codes, normalized_mi

__all__ = [
    "cluster_features",
    "evenly_spaced_rows",
    "feature_nmi_matrix",
    "nonmonotone_pairs",
    "pairwise_correlation",
    "redundancy_groups",
]


def evenly_spaced_rows(n: int, size: int) -> np.ndarray:
    if n <= size:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, size).astype(np.int64))


def pairwise_correlation(x: np.ndarray, *, rank: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(p x p) pairwise-complete correlation and pair counts of the columns of *x*;
    Spearman if *rank*."""
    z = np.column_stack([rank_scores(x[:, j]) for j in range(x.shape[1])]) if rank else \
        np.asarray(x, dtype=np.float64)
    present = np.isfinite(z).astype(np.float64)
    z0 = np.where(present > 0, z, 0.0)
    n = present.T @ present
    sx = z0.T @ present             # sum of x_i over rows where x_j is also present
    sxx = (z0 * z0).T @ present
    sxy = z0.T @ z0
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy - sx * sx.T / n
        vx = sxx - sx ** 2 / n
        vy = vx.T
        r = cov / np.sqrt(vx * vy)
    r = np.where((n > 2) & (vx > 0) & (vy > 0), r, np.nan)
    np.fill_diagonal(r, 1.0)
    return np.clip(r, -1.0, 1.0), n


def cluster_features(abs_corr: np.ndarray, *, threshold: float, method: str = "average"
                     ) -> tuple[np.ndarray, np.ndarray]:
    """(cluster label per feature, linkage matrix) cutting at 1 - |rho| <= 1 - threshold."""
    d = 1.0 - np.nan_to_num(abs_corr, nan=0.0)
    d = (d + d.T) / 2.0
    np.fill_diagonal(d, 0.0)
    d = np.clip(d, 0.0, 1.0)
    z = linkage(squareform(d, checks=False), method=method)
    labels = fcluster(z, t=1.0 - threshold, criterion="distance")
    return labels.astype(np.int64), z


def redundancy_groups(abs_corr: np.ndarray, names: list[str], *, threshold: float
                      ) -> tuple[list[list[str]], list[dict[str, Any]]]:
    """Connected components of the |rho| >= threshold graph, and its edges."""
    adj = np.nan_to_num(abs_corr, nan=0.0) >= threshold
    np.fill_diagonal(adj, False)
    count, labels = connected_components(csr_matrix(adj.astype(np.int8)), directed=False)
    groups = [[names[i] for i in np.flatnonzero(labels == g)] for g in range(count)]
    groups = sorted((g for g in groups if len(g) > 1), key=len, reverse=True)
    edges = [{"a": names[i], "b": names[j], "abs_rho": float(abs_corr[i, j])}
             for i, j in zip(*np.nonzero(np.triu(adj, 1)), strict=True)]
    return groups, edges


def feature_nmi_matrix(x: np.ndarray, *, bins: int) -> np.ndarray:
    """(p x p) normalised copula MI of every pair of columns of *x* (diagonal 1)."""
    codes = [binned_codes(rank_scores(x[:, j]), bins) for j in range(x.shape[1])]
    p = x.shape[1]
    out = np.full((p, p), np.nan)
    np.fill_diagonal(out, 1.0)
    for i in range(p):
        for j in range(i + 1, p):
            out[i, j] = out[j, i] = normalized_mi(codes[i], codes[j], bins)
    return out


def nonmonotone_pairs(nmi: np.ndarray, names: list[str], abs_corr: np.ndarray, *,
                      nmi_threshold: float, corr_ceiling: float) -> list[dict[str, Any]]:
    """Pairs whose normalised copula MI is high although |rho| is below *corr_ceiling*."""
    out = []
    p = nmi.shape[0]
    for i in range(p):
        for j in range(i + 1, p):
            if np.isfinite(abs_corr[i, j]) and abs_corr[i, j] >= corr_ceiling:
                continue
            if np.isfinite(nmi[i, j]) and nmi[i, j] >= nmi_threshold:
                out.append({"a": names[i], "b": names[j], "normalized_mi": float(nmi[i, j]),
                            "abs_spearman": float(abs_corr[i, j])})
    return sorted(out, key=lambda r: -r["normalized_mi"])
