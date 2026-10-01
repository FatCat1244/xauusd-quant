r"""K-Means: the geometric baseline (Step 10).

Lloyd's algorithm with k-means++ seeding, several restarts, the lowest inertia
kept. K-Means has no notion of time or of uncertainty - every bar goes to its
nearest centre - so it is only a baseline for the probabilistic models: if a
GMM or HMM found nothing K-Means does not, the extra machinery bought nothing.

Clusters are ``cluster_0 .. cluster_{K-1}`` - numbers, never names.

Bars with missing features are assigned on their observed coordinates, the
squared distance rescaled by ``D / |observed|`` so that partially observed
bars are not systematically "closer". The silhouette is an O(n^2) statistic
and is computed on an evenly spaced sample (an estimate, labelled as such).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["KMeansModel", "fit_kmeans", "kmeans_plus_plus", "silhouette_estimate"]

_CHUNK = 65_536


def _sq_distances(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances (rows x centres), complete rows only.

    The last chunk is padded to the full chunk size, so every matrix product
    has one shape and a row's distances never depend on how many rows exist.
    """
    n = x.shape[0]
    out = np.empty((n, centers.shape[0]))
    c2 = (centers ** 2).sum(axis=1)
    block = np.zeros((_CHUNK, x.shape[1]))
    for start in range(0, n, _CHUNK):
        size = min(_CHUNK, n - start)
        block[:size] = x[start:start + size]
        block[size:] = 0.0
        d = (block ** 2).sum(axis=1)[:, None] - 2.0 * block @ centers.T + c2[None, :]
        out[start:start + size] = np.maximum(d[:size], 0.0)
    return out


def kmeans_plus_plus(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding: each new centre drawn with probability proportional to D^2."""
    n = x.shape[0]
    centers = np.empty((k, x.shape[1]))
    centers[0] = x[rng.integers(n)]
    closest = _sq_distances(x, centers[:1])[:, 0]
    for j in range(1, k):
        total = closest.sum()
        if total <= 0:
            centers[j] = x[rng.integers(n)]
        else:
            centers[j] = x[rng.choice(n, p=closest / total)]
        closest = np.minimum(closest, _sq_distances(x, centers[j:j + 1])[:, 0])
    return centers


@dataclass
class KMeansModel:
    """Fitted cluster centres (in scaled units) and fit diagnostics."""

    centers: np.ndarray
    inertia: float
    n_observations: int
    n_iter: int
    converged: bool
    empty_cluster_resets: int = 0

    @property
    def n_states(self) -> int:
        return int(self.centers.shape[0])

    def predict(self, values: np.ndarray, *, scorable: np.ndarray | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
        """(labels, squared distance to the chosen centre); -1 / NaN where unscored."""
        x = np.asarray(values, dtype=np.float64)
        n, d = x.shape
        labels = np.full(n, -1, dtype=np.int64)
        dist = np.full(n, np.nan)
        finite = np.isfinite(x)
        use = finite.any(axis=1) if scorable is None else (np.asarray(scorable, bool)
                                                           & finite.any(axis=1))
        complete = use & finite.all(axis=1)
        rows = np.flatnonzero(complete)
        if rows.size:
            sq = _sq_distances(x[rows], self.centers)
            labels[rows] = np.argmin(sq, axis=1)
            dist[rows] = sq[np.arange(rows.size), labels[rows]]
        partial = np.flatnonzero(use & ~finite.all(axis=1))
        for i in partial:                      # few rows: marginal-coordinate distances
            obs = finite[i]
            sq = ((x[i, obs][None, :] - self.centers[:, obs]) ** 2).sum(axis=1) * d / obs.sum()
            labels[i] = int(np.argmin(sq))
            dist[i] = float(sq[labels[i]])
        return labels, dist

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "kmeans", "centers": self.centers.tolist(), "inertia": self.inertia,
                "n_observations": self.n_observations, "n_iter": self.n_iter,
                "converged": self.converged, "empty_cluster_resets": self.empty_cluster_resets}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KMeansModel:
        return cls(centers=np.asarray(data["centers"], dtype=np.float64),
                   inertia=float(data["inertia"]), n_observations=int(data["n_observations"]),
                   n_iter=int(data["n_iter"]), converged=bool(data["converged"]),
                   empty_cluster_resets=int(data.get("empty_cluster_resets", 0)))

    def permuted(self, order: np.ndarray | list[int]) -> KMeansModel:
        idx = np.asarray(order, dtype=np.int64)
        return KMeansModel(self.centers[idx].copy(), self.inertia, self.n_observations,
                           self.n_iter, self.converged, self.empty_cluster_resets)


def _lloyd(x: np.ndarray, centers: np.ndarray, *, max_iter: int,
           tol: float) -> tuple[np.ndarray, float, int, bool, int]:
    k = centers.shape[0]
    scale = float(np.mean(np.var(x, axis=0))) or 1.0
    resets = 0
    inertia = np.inf
    for it in range(1, max_iter + 1):
        sq = _sq_distances(x, centers)
        labels = np.argmin(sq, axis=1)
        inertia = float(sq[np.arange(x.shape[0]), labels].sum())
        counts = np.bincount(labels, minlength=k)
        sums = np.zeros_like(centers)
        np.add.at(sums, labels, x)
        new = centers.copy()
        filled = counts > 0
        new[filled] = sums[filled] / counts[filled, None]
        for j in np.flatnonzero(~filled):      # an empty cluster restarts at the worst point
            new[j] = x[int(np.argmax(sq[np.arange(x.shape[0]), labels]))]
            resets += 1
        shift = float(((new - centers) ** 2).sum(axis=1).max())
        centers = new
        if shift <= tol * scale:
            return centers, inertia, it, True, resets
    return centers, inertia, max_iter, False, resets


def fit_kmeans(values: np.ndarray, k: int, *, n_init: int = 5, max_iter: int = 300,
               tol: float = 1e-6, seed: int = 0, seed_sample: int = 50_000) -> KMeansModel:
    """Best of *n_init* k-means++ restarts on complete rows (lowest inertia)."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x).all(axis=1)]
    if x.shape[0] < k:
        raise ValueError(f"K-Means with {k} clusters needs at least {k} complete rows")
    best: KMeansModel | None = None
    for init in range(n_init):
        rng = np.random.default_rng([seed, init])
        sample = x
        if x.shape[0] > seed_sample:
            sample = x[np.sort(rng.choice(x.shape[0], seed_sample, replace=False))]
        centers = kmeans_plus_plus(sample, k, rng)
        centers, inertia, n_iter, converged, resets = _lloyd(x, centers, max_iter=max_iter,
                                                             tol=tol)
        model = KMeansModel(centers=centers, inertia=inertia, n_observations=int(x.shape[0]),
                            n_iter=n_iter, converged=converged, empty_cluster_resets=resets)
        if best is None or model.inertia < best.inertia:
            best = model
    assert best is not None
    return best


def silhouette_estimate(values: np.ndarray, labels: np.ndarray, *,
                        sample: int) -> float | None:
    """Mean silhouette on an evenly spaced sample of complete, labelled rows (estimate)."""
    x = np.asarray(values, dtype=np.float64)
    lab = np.asarray(labels)
    rows = np.flatnonzero((lab >= 0) & np.isfinite(x).all(axis=1))
    if rows.size > sample:
        rows = rows[np.linspace(0, rows.size - 1, sample).round().astype(np.int64)]
    x, lab = x[rows], lab[rows]
    clusters = np.unique(lab)
    if clusters.size < 2 or rows.size < 10:
        return None
    n = x.shape[0]
    mean_dist = np.zeros((n, clusters.size))
    counts = np.array([(lab == c).sum() for c in clusters], dtype=np.float64)
    for start in range(0, n, 2048):
        block = x[start:start + 2048]
        sq = ((block ** 2).sum(axis=1)[:, None] - 2 * block @ x.T + (x ** 2).sum(axis=1)[None, :])
        dist = np.sqrt(np.maximum(sq, 0.0))
        for ci, c in enumerate(clusters):
            mean_dist[start:start + 2048, ci] = dist[:, lab == c].sum(axis=1)
    own = np.searchsorted(clusters, lab)
    own_count = counts[own]
    a = np.where(own_count > 1, mean_dist[np.arange(n), own] / np.maximum(own_count - 1, 1), 0.0)
    other = mean_dist / counts[None, :]
    other[np.arange(n), own] = np.inf
    b = other.min(axis=1)
    s = np.where(own_count > 1, (b - a) / np.maximum(np.maximum(a, b), 1e-300), 0.0)
    return float(np.mean(s))
