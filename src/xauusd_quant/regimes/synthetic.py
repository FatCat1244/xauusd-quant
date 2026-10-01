r"""Synthetic controls with known answers (Steps 44-46).

``known_hmm``          three states - low volatility, high volatility, and one with
                       a shifted mean and a different correlation - switching by a
                       known transition matrix. A correct implementation recovers the
                       number of states, the state distributions and the transition
                       behaviour, up to a relabelling.
``single_regime``      one stationary distribution (Gaussian, or Student-t with the
                       same covariance for heavy tails). There is nothing to find:
                       any "states" a model reports here are either duplicates,
                       flickering, or - for heavy tails - components that model the
                       tails, which is the classic way a mixture invents regimes.
``smooth_continuum``   the volatility changes continuously (a slow AR(1) in log
                       volatility), with no discrete states at all. An HMM will still
                       cut it into ordered bands; the diagnostics show how that looks
                       (neighbouring-state transitions, durations set by the speed
                       of the drift) so the same pattern can be recognised in real
                       data. Markets may not have discrete regimes.

Everything is seeded and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["SyntheticSeries", "known_hmm", "single_regime", "smooth_continuum"]


@dataclass(frozen=True)
class SyntheticSeries:
    values: np.ndarray                 # (n, d)
    states: np.ndarray | None          # true state per row (None when there are none)
    transition: np.ndarray | None
    means: np.ndarray | None
    covariances: np.ndarray | None
    description: str


def _chain(transition: np.ndarray, n: int, rng: np.random.Generator,
           start: int = 0) -> np.ndarray:
    k = transition.shape[0]
    cumulative = np.cumsum(transition, axis=1)
    u = rng.random(n)
    states = np.empty(n, dtype=np.int64)
    s = start
    for t in range(n):
        states[t] = s
        s = int(min(np.searchsorted(cumulative[s], u[t], side="right"), k - 1))
    return states


def known_hmm(n: int, *, seed: int, dimension: int = 4) -> SyntheticSeries:
    """Three persistent Gaussian states (expected durations 100, 50 and 40 bars)."""
    rng = np.random.default_rng(seed)
    d = dimension
    transition = np.array([[0.990, 0.006, 0.004],
                           [0.012, 0.980, 0.008],
                           [0.010, 0.015, 0.975]])
    means = np.zeros((3, d))
    means[1, 0] = 1.5                                   # high volatility: shifted level
    means[2, 1] = -1.2                                  # a third state: other means ...
    means[2, 2] = 1.0
    cov = np.zeros((3, d, d))
    cov[0] = 0.25 * np.eye(d)                           # low volatility
    cov[1] = 2.25 * np.eye(d)                           # high volatility
    rho = 0.7                                           # ... and a correlation structure
    cov[2] = np.eye(d)
    cov[2, 0, 1] = cov[2, 1, 0] = rho
    states = _chain(transition, n, rng)
    values = np.empty((n, d))
    for k in range(3):
        rows = states == k
        values[rows] = rng.multivariate_normal(means[k], cov[k], size=int(rows.sum()))
    return SyntheticSeries(values, states, transition, means, cov,
                           "3-state Gaussian HMM: low vol, high vol, shifted mean + correlation")


def single_regime(n: int, *, seed: int, dimension: int = 4,
                  heavy_tails: bool = False) -> SyntheticSeries:
    """One stationary distribution: i.i.d. Gaussian or Student-t (5 d.o.f.), correlated."""
    rng = np.random.default_rng(seed)
    d = dimension
    cov = 0.5 * np.eye(d) + 0.5
    chol = np.linalg.cholesky(cov)
    z = rng.standard_normal((n, d))
    if heavy_tails:
        dof = 5.0
        z = z / np.sqrt(rng.chisquare(dof, size=(n, 1)) / dof) * np.sqrt((dof - 2) / dof)
    values = z @ chol.T
    label = "Student-t(5), one regime" if heavy_tails else "Gaussian, one regime"
    return SyntheticSeries(values, None, None, np.zeros((1, d)), cov[None], label)


def smooth_continuum(n: int, *, seed: int, dimension: int = 4,
                     persistence: float = 0.999) -> SyntheticSeries:
    """Log volatility follows a slow AR(1): the state varies continuously, never jumps.

    Features: the (log) realised level itself plus a few variables whose scale
    follows it - what volatility-driven market features look like without regimes.
    """
    rng = np.random.default_rng(seed)
    d = dimension
    sd = 1.0 * np.sqrt(1 - persistence ** 2)
    log_vol = np.empty(n)
    log_vol[0] = 0.0
    shocks = rng.normal(0.0, sd, n)
    for t in range(1, n):
        log_vol[t] = persistence * log_vol[t - 1] + shocks[t]
    values = np.empty((n, d))
    values[:, 0] = log_vol + rng.normal(0, 0.1, n)      # a noisy volatility measure
    scale = np.exp(log_vol)
    values[:, 1:] = rng.standard_normal((n, d - 1)) * scale[:, None]
    return SyntheticSeries(values, None, None, None, None,
                           f"continuous log-volatility AR(1), phi = {persistence}")
