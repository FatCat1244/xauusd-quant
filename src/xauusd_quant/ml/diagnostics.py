r"""Distribution shift, out-of-distribution scores and model disagreement (Prompt #10,
Steps 62-66).

* :func:`shift_table`: per feature, the population stability index (10
  training-quantile bins), the Kolmogorov-Smirnov statistic and the Wasserstein
  distance in training standard deviations, between a fold's fitting rows and
  its validation block.
* :class:`OODScorer`: a robust Mahalanobis distance (median / IQR scaling, a
  ridge-regularised covariance fitted on training rows), reported as the
  percentile of each row's distance among the training rows' own distances -
  diagnostic only.
* :func:`disagreement`: the cross-model standard deviation of predictions on the
  same rows (an uncertainty proxy, never a rule).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..features.quality import _proportions, ks_and_wasserstein

__all__ = ["OODScorer", "disagreement", "shift_table"]


def _sample(v: np.ndarray, size: int) -> np.ndarray:
    v = v[np.isfinite(v)]
    if v.size > size:
        v = v[np.linspace(0, v.size - 1, size).astype(np.int64)]
    return v


def shift_table(x_train: np.ndarray, x_val: np.ndarray, names: list[str], *,
                size: int = 50_000) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for j, name in enumerate(names):
        a = np.sort(_sample(x_train[:, j].astype(np.float64), size))
        b = np.sort(_sample(x_val[:, j].astype(np.float64), size))
        if a.size < 100 or b.size < 100:
            rows.append({"feature": name, "psi": None, "ks": None, "wasserstein_sd": None})
            continue
        edges = np.unique(np.quantile(a, np.linspace(0, 1, 11)))
        psi = None
        if edges.size >= 3:
            edges[0], edges[-1] = -np.inf, np.inf
            pa, pb = _proportions(a, edges), _proportions(b, edges)
            psi = float(((pb - pa) * np.log(pb / pa)).sum())
        ks, w1 = ks_and_wasserstein(b, a)
        sd = float(a.std()) or 1.0
        rows.append({"feature": name, "psi": psi, "ks": ks, "wasserstein_sd": w1 / sd,
                     "median_shift_sd": float((np.median(b) - np.median(a)) / sd)})
    return rows


@dataclass
class OODScorer:
    center: np.ndarray
    scale: np.ndarray
    medians: np.ndarray
    precision: np.ndarray
    train_distances: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, *, ridge: float = 1e-2, size: int = 100_000) -> OODScorer:
        xs = np.asarray(x, dtype=np.float64)
        if xs.shape[0] > size:
            xs = xs[np.linspace(0, xs.shape[0] - 1, size).astype(np.int64)]
        finite = np.isfinite(xs)
        medians = np.array([np.median(xs[finite[:, j], j]) if finite[:, j].any() else 0.0
                            for j in range(xs.shape[1])])
        filled = np.where(finite, xs, medians[None, :])
        center = np.median(filled, axis=0)
        q75, q25 = np.percentile(filled, [75, 25], axis=0)
        scale = np.where(q75 - q25 > 1e-12, q75 - q25, np.maximum(filled.std(axis=0), 1e-12))
        z = (filled - center) / scale
        cov = np.cov(z, rowvar=False) + ridge * np.eye(z.shape[1])
        precision = np.linalg.inv(cov)
        scorer = cls(center=center, scale=scale, medians=medians, precision=precision,
                     train_distances=np.zeros(0))
        scorer.train_distances = np.sort(scorer.distance(xs))
        return scorer

    def distance(self, x: np.ndarray) -> np.ndarray:
        xs = np.asarray(x, dtype=np.float64)
        filled = np.where(np.isfinite(xs), xs, self.medians[None, :])
        z = (filled - self.center) / self.scale
        return np.sqrt(np.einsum("ij,jk,ik->i", z, self.precision, z))

    def percentile(self, x: np.ndarray) -> np.ndarray:
        d = self.distance(x)
        return np.searchsorted(self.train_distances, d, side="right") / max(
            1, self.train_distances.size)


def disagreement(predictions: dict[str, np.ndarray]) -> np.ndarray:
    """Row-wise standard deviation across models' predictions."""
    stack = np.column_stack(list(predictions.values())).astype(np.float64)
    return np.nanstd(stack, axis=1)
