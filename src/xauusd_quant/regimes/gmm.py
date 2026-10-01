r"""Gaussian mixture model (Step 11): soft states without time.

.. math::

    p(X_t) = \sum_{k=1}^{K} \pi_k\, \mathcal N(X_t \mid \mu_k, \Sigma_k)

fitted by expectation-maximisation on complete rows, with ``full``, ``diag``
or ``tied`` covariances (Step 12) and ``reg_covar`` on every diagonal so a
component cannot collapse onto a point. The best of several initialisations
(k-means++ seeded) is kept by training log-likelihood.

The posterior :math:`P(S_t = k \mid X_t)` uses only the bar's own features, so
with a frozen scaler and frozen parameters it is causal - but it has no
memory: consecutive bars are assigned independently, which is what the hidden
Markov model adds. Per bar a GMM gives ``gmm_p<k>``, the most likely state,
its probability (confidence) and the entropy of the probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import logsumexp

from .clustering import _lloyd, kmeans_plus_plus
from .emissions import GaussianStates, log_densities, parameter_count, regularise

__all__ = ["GMMModel", "fit_gmm", "m_step", "state_entropy"]

_ROWS = 262_144
_EPS = 10 * np.finfo(np.float64).eps


def state_entropy(probs: np.ndarray) -> np.ndarray:
    r""":math:`H = -\sum_k p_k \ln p_k` per row (NaN rows stay NaN)."""
    p = np.asarray(probs, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * np.log(p), 0.0)
    h = -terms.sum(axis=1)
    return np.where(np.isnan(p).any(axis=1), np.nan, h)


def m_step(x: np.ndarray, resp: np.ndarray, *, covariance_type: str,
           reg_covar: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weights, means and covariances from responsibilities (rows x states)."""
    n, d = x.shape
    k = resp.shape[1]
    nk = resp.sum(axis=0) + _EPS
    weights = nk / nk.sum()
    means = (resp.T @ x) / nk[:, None]
    cov = np.zeros((k, d, d))
    for start in range(0, n, _ROWS):
        block = x[start:start + _ROWS]
        r = resp[start:start + _ROWS]
        for j in range(k):
            diff = block - means[j]
            if covariance_type == "diag":
                cov[j, np.arange(d), np.arange(d)] += (r[:, j, None] * diff * diff).sum(axis=0)
            else:
                cov[j] += (diff * r[:, j, None]).T @ diff
    if covariance_type == "tied":
        pooled = cov.sum(axis=0) / nk.sum()
        cov = np.broadcast_to(pooled, cov.shape).copy()
    else:
        cov /= nk[:, None, None]
    return weights, means, regularise(cov, reg_covar, covariance_type)


@dataclass
class GMMModel:
    """Fitted mixture (in scaled units) with its training diagnostics."""

    weights: np.ndarray
    states: GaussianStates
    covariance_type: str
    reg_covar: float
    log_likelihood: float = float("nan")
    n_observations: int = 0
    n_iter: int = 0
    converged: bool = False
    reseeds: int = 0

    @property
    def n_states(self) -> int:
        return self.states.n_states

    @property
    def dimension(self) -> int:
        return self.states.dimension

    @property
    def n_parameters(self) -> int:
        return (self.n_states - 1) + parameter_count(self.n_states, self.dimension,
                                                     self.covariance_type)

    def bic(self) -> float:
        return -2.0 * self.log_likelihood + self.n_parameters * np.log(self.n_observations)

    def aic(self) -> float:
        return -2.0 * self.log_likelihood + 2.0 * self.n_parameters

    def posterior(self, values: np.ndarray, *, scorable: np.ndarray | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
        """(P(S_t=k | X_t), ln p(X_t)); NaN rows where a bar is not scored."""
        x = np.asarray(values, dtype=np.float64)
        used = np.isfinite(x).any(axis=1)
        if scorable is not None:
            used &= np.asarray(scorable, dtype=bool)
        joint = log_densities(x, self.states, scorable=used) + np.log(self.weights)[None, :]
        marginal = logsumexp(joint, axis=1)
        probs = np.exp(joint - marginal[:, None])
        probs[~used] = np.nan
        marginal = np.where(used, marginal, np.nan)
        return probs, marginal

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "gmm", "weights": self.weights.tolist(),
                "means": self.states.means.tolist(),
                "covariances": self.states.covariances.tolist(),
                "covariance_type": self.covariance_type, "reg_covar": self.reg_covar,
                "log_likelihood": self.log_likelihood, "n_observations": self.n_observations,
                "n_iter": self.n_iter, "converged": self.converged, "reseeds": self.reseeds}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GMMModel:
        states = GaussianStates(np.asarray(data["means"], dtype=np.float64),
                                np.asarray(data["covariances"], dtype=np.float64),
                                str(data["covariance_type"]))
        return cls(weights=np.asarray(data["weights"], dtype=np.float64), states=states,
                   covariance_type=str(data["covariance_type"]),
                   reg_covar=float(data["reg_covar"]),
                   log_likelihood=float(data["log_likelihood"]),
                   n_observations=int(data["n_observations"]), n_iter=int(data["n_iter"]),
                   converged=bool(data["converged"]), reseeds=int(data.get("reseeds", 0)))

    def permuted(self, order: np.ndarray | list[int]) -> GMMModel:
        idx = np.asarray(order, dtype=np.int64)
        return GMMModel(self.weights[idx].copy(), self.states.permuted(idx),
                        self.covariance_type, self.reg_covar, self.log_likelihood,
                        self.n_observations, self.n_iter, self.converged, self.reseeds)


def _initial(x: np.ndarray, k: int, rng: np.random.Generator, *, covariance_type: str,
             reg_covar: float, sample: int = 50_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """k-means++ on a sample, a few Lloyd steps, then one hard M-step on every row."""
    sub = x
    if x.shape[0] > sample:
        sub = x[np.sort(rng.choice(x.shape[0], sample, replace=False))]
    centers = kmeans_plus_plus(sub, k, rng)
    centers, _, _, _, _ = _lloyd(sub, centers, max_iter=25, tol=1e-4)
    labels = np.empty(x.shape[0], dtype=np.int64)
    c2 = (centers ** 2).sum(axis=1)
    for start in range(0, x.shape[0], _ROWS):
        block = x[start:start + _ROWS]
        labels[start:start + _ROWS] = np.argmin(c2[None, :] - 2.0 * block @ centers.T, axis=1)
    resp = np.zeros((x.shape[0], k))
    resp[np.arange(x.shape[0]), labels] = 1.0
    return m_step(x, resp, covariance_type=covariance_type, reg_covar=reg_covar)


def _em(x: np.ndarray, weights: np.ndarray, means: np.ndarray, cov: np.ndarray, *,
        covariance_type: str, reg_covar: float, max_iter: int, tol: float,
        rng: np.random.Generator) -> GMMModel:
    n = x.shape[0]
    previous = -np.inf
    reseeds = 0
    ll = -np.inf
    converged = False
    iterations = 0
    for _ in range(max_iter):
        iterations += 1
        states = GaussianStates(means, cov, covariance_type)
        joint = log_densities(x, states) + np.log(weights)[None, :]
        marginal = logsumexp(joint, axis=1)
        ll = float(marginal.sum())
        resp = np.exp(joint - marginal[:, None])
        del joint
        if abs(ll - previous) / n < tol:
            converged = True
            break
        previous = ll
        weights, means, cov = m_step(x, resp, covariance_type=covariance_type,
                                     reg_covar=reg_covar)
        empty = np.flatnonzero(weights * n < 2.0)
        if empty.size:                       # restart a vanished component at a random bar
            overall = np.cov(x[rng.choice(n, min(n, 20_000), replace=False)].T)
            for j in empty:
                means[j] = x[rng.integers(n)]
                cov[j] = regularise(overall[None], reg_covar, covariance_type)[0]
                weights[j] = 1.0 / n
                reseeds += 1
            weights = weights / weights.sum()
            previous = -np.inf
    if not converged:                        # the last M-step moved the parameters
        states = GaussianStates(means, cov, covariance_type)
        ll = float(logsumexp(log_densities(x, states) + np.log(weights)[None, :], axis=1).sum())
    return GMMModel(weights=weights, states=GaussianStates(means, cov, covariance_type),
                    covariance_type=covariance_type, reg_covar=reg_covar, log_likelihood=ll,
                    n_observations=n, n_iter=iterations, converged=converged, reseeds=reseeds)


def fit_gmm(values: np.ndarray, k: int, *, covariance_type: str = "full", n_init: int = 3,
            max_iter: int = 300, tol: float = 1e-5, reg_covar: float = 1e-4, seed: int = 0,
            init: GMMModel | None = None) -> GMMModel:
    """Best of *n_init* EM runs on the complete rows of *values* (plus *init*, if given)."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x).all(axis=1)]
    if x.shape[0] < 10 * k:
        raise ValueError(f"a {k}-component mixture needs at least {10 * k} complete rows")
    candidates: list[GMMModel] = []
    if init is not None:
        rng = np.random.default_rng([seed, 999])
        candidates.append(_em(x, init.weights.copy(), init.states.means.copy(),
                              regularise(init.states.covariances, 0.0, covariance_type),
                              covariance_type=covariance_type, reg_covar=reg_covar,
                              max_iter=max_iter, tol=tol, rng=rng))
    for run in range(n_init):
        rng = np.random.default_rng([seed, run])
        w, m, c = _initial(x, k, rng, covariance_type=covariance_type, reg_covar=reg_covar)
        candidates.append(_em(x, w, m, c, covariance_type=covariance_type, reg_covar=reg_covar,
                              max_iter=max_iter, tol=tol, rng=rng))
    if not candidates:
        raise ValueError("fit_gmm needs n_init >= 1 or an initial model")
    return max(candidates, key=lambda model: model.log_likelihood)
