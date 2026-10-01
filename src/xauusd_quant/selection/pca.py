r"""PCA as a research comparison, fitted on training rows only (Prompt #9, Steps 24-28).

PCA is unsupervised: it keeps the directions of largest variance, which need
not be the directions that predict. The question is whether compression
*preserves forward information*, answered by comparing a ridge model on the
first ``k`` components with one on raw selected features, both fitted on the
training span and scored on the evaluation span.

Causality: the component loadings, the standardisation and the training CDFs
of the rank-linear design are all *fitted on training rows* and frozen
(:class:`PCAModel`); transforming later rows reads them and nothing else, so
appending future observations can never alter a historical projection. A PCA
fitted on 2003-2026 and used on 2005 would place 2005 in coordinates chosen
with 2025's distribution - it is never done.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .linear import Moments

__all__ = ["PCAModel", "fit_pca"]


@dataclass(frozen=True)
class PCAModel:
    """Frozen training PCA of standardised (rank-linear) features."""

    columns: tuple[int, ...]           # design columns it was fitted on
    mean: np.ndarray                   # (p,) training means
    sd: np.ndarray                     # (p,) training standard deviations
    eigenvalues: np.ndarray            # (p,) descending
    loadings: np.ndarray               # (p, p), column i = component i

    def explained(self) -> np.ndarray:
        total = self.eigenvalues.sum()
        return self.eigenvalues / total if total > 0 else self.eigenvalues

    def transform(self, x: np.ndarray, k: int) -> np.ndarray:
        """Scores of rows of the design *x* (all design columns) on the first *k* components."""
        z = (x[:, list(self.columns)] - self.mean) / self.sd
        return z @ self.loadings[:, :k]


def fit_pca(train: Moments, columns: list[int]) -> PCAModel:
    """Eigen-decomposition of the training correlation matrix of *columns*."""
    q, _, mx, sd, _ = train.standardized(columns)
    lam, vec = np.linalg.eigh((q + q.T) / 2.0)
    order = np.argsort(lam)[::-1]
    lam = np.clip(lam[order], 0.0, None)
    vec = vec[:, order]
    # a deterministic sign: the largest-|loading| entry of each component is positive
    flip = np.sign(vec[np.argmax(np.abs(vec), axis=0), np.arange(vec.shape[1])])
    vec = vec * np.where(flip == 0, 1.0, flip)
    return PCAModel(columns=tuple(columns), mean=mx, sd=sd, eigenvalues=lam, loadings=vec)


def pca_ridge(model: PCAModel, train: Moments, k: int, alpha: float) -> np.ndarray:
    """Ridge coefficients on the first *k* training component scores (closed form)."""
    _, c, _, _, _ = train.standardized(list(model.columns))
    proj = model.loadings[:, :k].T @ c                 # (k, T) cross-moments of the scores
    return proj / (model.eigenvalues[:k, None] + alpha)
