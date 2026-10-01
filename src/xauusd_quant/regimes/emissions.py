r"""Gaussian state densities, shared by the mixture and the hidden Markov model.

Every regime model here describes state ``k`` by a Gaussian
:math:`\mathcal N(\mu_k, \Sigma_k)` over the scaled features. This module
computes :math:`\ln \mathcal N(x_t \mid \mu_k, \Sigma_k)` for every bar and
state, with two guarantees the causal layer relies on:

**Exact marginalisation of missing features.** A bar observed on the
coordinates ``O`` is scored by the marginal density
:math:`\mathcal N(x_O \mid \mu_{k,O}, \Sigma_{k,OO})` - what the model says
about the features that exist, with nothing imputed. Bars are grouped by
missingness pattern and each pattern gets its own sub-covariance Cholesky
factor. Bars that may not be scored (too little coverage) get
``log-density = 0`` in every state: no evidence, so a filter keeps its prior.

**Row-by-row determinism.** Rows are processed in fixed-size chunks and the
last chunk of each pattern is padded to the full size, so every linear-algebra
call sees the same shapes whatever the series length. A bar's log-densities
are therefore bit-for-bit identical whether or not later bars exist - the
property the prefix-invariance tests pin down.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "CHUNK_ROWS",
    "GaussianStates",
    "bhattacharyya",
    "log_densities",
    "parameter_count",
    "regularise",
]

#: Rows per linear-algebra call. Fixed: changing it changes last-bit rounding.
CHUNK_ROWS = 16_384
_LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class GaussianStates:
    """``K`` Gaussian states over ``D`` scaled features."""

    means: np.ndarray            # (K, D)
    covariances: np.ndarray      # (K, D, D), full matrices whatever the covariance type
    covariance_type: str = "full"

    @property
    def n_states(self) -> int:
        return int(self.means.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.means.shape[1])

    def permuted(self, order: np.ndarray | list[int]) -> GaussianStates:
        idx = np.asarray(order, dtype=np.int64)
        return GaussianStates(self.means[idx].copy(), self.covariances[idx].copy(),
                              self.covariance_type)


def regularise(covariances: np.ndarray, reg: float, covariance_type: str) -> np.ndarray:
    """Covariances in their constrained form (diag / tied) with ``reg`` on the diagonal."""
    cov = np.array(covariances, dtype=np.float64, copy=True)
    k, d, _ = cov.shape
    if covariance_type == "diag":
        diag = np.einsum("kii->ki", cov)
        cov = np.zeros_like(cov)
        cov[:, np.arange(d), np.arange(d)] = diag
    elif covariance_type == "tied":
        pooled = cov.mean(axis=0)
        cov = np.broadcast_to(pooled, cov.shape).copy()
    elif covariance_type != "full":
        raise ValueError(f"unknown covariance type {covariance_type!r}")
    cov = 0.5 * (cov + np.transpose(cov, (0, 2, 1)))
    cov[:, np.arange(d), np.arange(d)] += reg
    return cov


def parameter_count(n_states: int, dimension: int, covariance_type: str) -> int:
    """Free Gaussian parameters of ``n_states`` states (means + covariances)."""
    k, d = n_states, dimension
    means = k * d
    if covariance_type == "full":
        cov = k * d * (d + 1) // 2
    elif covariance_type == "diag":
        cov = k * d
    elif covariance_type == "tied":
        cov = d * (d + 1) // 2
    else:
        raise ValueError(f"unknown covariance type {covariance_type!r}")
    return means + cov


def _patterns(finite: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unique missingness patterns (as bool rows) and each row's pattern index."""
    d = finite.shape[1]
    weights = (1 << np.arange(d, dtype=np.int64))
    codes = finite.astype(np.int64) @ weights
    unique, inverse = np.unique(codes, return_inverse=True)
    patterns = ((unique[:, None] >> np.arange(d)[None, :]) & 1).astype(bool)
    return patterns, inverse.reshape(-1)


def _chunks(rows: np.ndarray) -> list[np.ndarray]:
    return [rows[i:i + CHUNK_ROWS] for i in range(0, rows.size, CHUNK_ROWS)]


def log_densities(values: np.ndarray, states: GaussianStates, *,
                  scorable: np.ndarray | None = None) -> np.ndarray:
    """``ln N(x_t | mu_k, Sigma_k)`` for every row and state, (rows, K).

    NaN coordinates are marginalised out. Rows outside *scorable*, and rows
    with no finite coordinate, get 0 in every state (no evidence).
    """
    x = np.asarray(values, dtype=np.float64)
    n, d = x.shape
    k = states.n_states
    if d != states.dimension:
        raise ValueError(f"values have {d} features, the states {states.dimension}")
    out = np.zeros((n, k), dtype=np.float64)
    finite = np.isfinite(x)
    use = finite.any(axis=1)
    if scorable is not None:
        use &= np.asarray(scorable, dtype=bool)
    rows_all = np.flatnonzero(use)
    if rows_all.size == 0:
        return out
    patterns, inverse = _patterns(finite[rows_all])
    for p, pattern in enumerate(patterns):
        dims = np.flatnonzero(pattern)
        if dims.size == 0:
            continue
        rows = rows_all[inverse == p]
        m = dims.size
        sub_means = states.means[:, dims]                                   # (K, m)
        sub_cov = states.covariances[:, dims][:, :, dims]                   # (K, m, m)
        chol = np.linalg.cholesky(sub_cov)
        inv_t = np.stack([np.linalg.inv(chol[j]).T for j in range(k)])     # (K, m, m)
        log_det = 2.0 * np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1)
        constant = -0.5 * (m * _LOG_2PI + log_det)                          # (K,)
        block = np.zeros((CHUNK_ROWS, m))
        for chunk in _chunks(rows):
            size = chunk.size
            block[:size] = x[np.ix_(chunk, dims)]
            block[size:] = 0.0
            for j in range(k):
                z = (block - sub_means[j]) @ inv_t[j]
                out[chunk, j] = constant[j] - 0.5 * np.einsum("ij,ij->i", z, z)[:size]
    return out


def bhattacharyya(mean_a: np.ndarray, cov_a: np.ndarray, mean_b: np.ndarray,
                  cov_b: np.ndarray) -> float:
    r"""Bhattacharyya distance between two Gaussians.

    :math:`\tfrac18 \Delta^\top \Sigma^{-1} \Delta + \tfrac12 \ln\frac{|\Sigma|}
    {\sqrt{|\Sigma_a||\Sigma_b|}}`, :math:`\Sigma = (\Sigma_a + \Sigma_b)/2`.
    Invariant under any common invertible affine map, so two models can be
    compared in raw feature units whatever their scalers were.
    """
    delta = np.asarray(mean_a, dtype=np.float64) - np.asarray(mean_b, dtype=np.float64)
    cov = 0.5 * (np.asarray(cov_a, dtype=np.float64) + np.asarray(cov_b, dtype=np.float64))
    _, logdet = np.linalg.slogdet(cov)
    _, logdet_a = np.linalg.slogdet(cov_a)
    _, logdet_b = np.linalg.slogdet(cov_b)
    maha = float(delta @ np.linalg.solve(cov, delta))
    return 0.125 * maha + 0.5 * (logdet - 0.5 * (logdet_a + logdet_b))
