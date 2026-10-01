r"""Gaussian hidden Markov model (Steps 13-18): the primary temporal regime model.

A latent state :math:`S_t \in \{0..K-1\}` follows a Markov chain,
:math:`A_{ij} = P(S_t = j \mid S_{t-1} = i)`, and the features are Gaussian
given the state, :math:`X_t \mid S_t = k \sim \mathcal N(\mu_k, \Sigma_k)`.
Parameters are estimated by Baum-Welch (EM) on one long bar sequence in trading
time; bars without complete features carry no evidence during fitting.

Three different state estimates exist and they are **not** interchangeable:

``filter``   :math:`P(S_t \mid X_{\le t})` - the forward recursion. With frozen
             parameters it uses only bars up to ``t``: **LIVE SAFE**, the only
             state estimate that may ever become a feature.
``smooth``   :math:`P(S_t \mid X_{1:T})` - forward-backward. Uses every later
             bar: **OFFLINE / NON-CAUSAL**. Appending data changes it (the
             tests show it does).
``viterbi``  the jointly most likely path given all bars: **OFFLINE**.

Numerics
--------
The forward and backward recursions are linear once each step is rescaled, so
the sequence is cut into blocks of ``block_length`` bars anchored at its first
bar. Per block the product of the step operators :math:`D_t A` (``D_t`` the
diagonal of emission densities) is accumulated with every row renormalised
and its log-scale kept; a short sequential pass over the blocks gives each
block's incoming state distribution, and the within-block recursions then run
for all blocks at once. The result equals the textbook scaled recursion (the
tests compare them), costs a few vectorised passes instead of one Python step
per bar, and is exact in log-space whatever the sequence length.

The live filter (``exact=True``) does its small matrix products elementwise in
a fixed order, so the probabilities at bar ``t`` are bit-for-bit identical
whether or not later bars exist. Fitting uses the same algorithm with BLAS
products (faster, equal to rounding).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .emissions import GaussianStates, log_densities, parameter_count, regularise
from .gmm import GMMModel, fit_gmm, m_step

__all__ = [
    "FilterResult",
    "HMMModel",
    "fit_hmm",
    "forward_backward",
    "forward_filter",
    "stationary_distribution",
    "viterbi_path",
]


# ---------------------------------------------------------------------------
# Recursions
# ---------------------------------------------------------------------------
def _scaled_emissions(log_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``b = exp(log_b - max)`` per row (max 1) and the row maxima."""
    m = log_b.max(axis=1)
    return np.exp(log_b - m[:, None]), m


def _blocks(b: np.ndarray, length: int) -> np.ndarray:
    """Rows padded with no-evidence rows (all ones) to whole blocks: (blocks, length, K)."""
    n, k = b.shape
    count = max(1, -(-n // length))
    out = np.ones((count * length, k))
    out[:n] = b
    return out.reshape(count, length, k)


#: Blocks per matrix-product call in the exact (live) filter: a fixed shape, so a
#: block's arithmetic never depends on how many blocks the sequence has.
_GROUP = 128
#: Steps between row renormalisations. A row keeps at least min(A) of its mass
#: per step (every state is reachable), so four steps cannot underflow.
_NORMALISE_EVERY = 4


def _block_operators(b_blk: np.ndarray, transition: np.ndarray, *,
                     exact: bool) -> tuple[np.ndarray, np.ndarray]:
    r"""Per block, :math:`\prod_t D_t A` with rows renormalised, and the row log-scales.

    ``exact`` multiplies in groups of :data:`_GROUP` blocks (the last group
    padded), so every BLAS call has the same shape whatever the series length
    and a block's product is bit-for-bit reproducible; otherwise one call per
    step covers every block.
    """
    nb, length, k = b_blk.shape
    total = -(-nb // _GROUP) * _GROUP if exact else nb
    if total != nb:
        b_blk = np.concatenate([b_blk, np.ones((total - nb, length, k))])
    prod = np.zeros((total, k, k))
    prod[:, np.arange(k), np.arange(k)] = 1.0
    work = np.empty_like(prod)
    scale = np.zeros((total, k))
    a = np.ascontiguousarray(transition, dtype=np.float64)
    for m in range(length):
        np.multiply(prod, b_blk[:, m, None, :], out=work)    # work[j,i,s] = prod[j,i,s] b[s]
        if exact:
            for g in range(0, total, _GROUP):
                np.matmul(work[g:g + _GROUP].reshape(_GROUP * k, k), a,
                          out=prod[g:g + _GROUP].reshape(_GROUP * k, k))
        else:
            np.matmul(work.reshape(total * k, k), a, out=prod.reshape(total * k, k))
        if (m + 1) % _NORMALISE_EVERY == 0 or m == length - 1:
            q = prod.sum(axis=2)
            zero = ~(q > 0)
            safe = np.where(zero, 1.0, q)
            prod /= safe[:, :, None]
            with np.errstate(divide="ignore"):
                scale = np.where(zero, -np.inf, scale + np.log(safe))
    return prod[:nb], scale[:nb]


def _forward_carries(prod: np.ndarray, scale: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """The predictive state distribution entering each block."""
    nb, k, _ = prod.shape
    pred = np.empty((nb, k))
    cur = prior / prior.sum()
    for j in range(nb):
        pred[j] = cur
        with np.errstate(divide="ignore"):
            w = np.log(cur) + scale[j]
        top = w.max()
        if np.isfinite(top):
            v = np.exp(w - top) @ prod[j]
            total = v.sum()
            if total > 0:
                cur = v / total
    return pred


def _forward_pass(b_blk: np.ndarray, pred: np.ndarray, transition: np.ndarray, *,
                  exact: bool) -> tuple[np.ndarray, np.ndarray]:
    """Filtered probabilities and log normalisers, all blocks at once."""
    nb, length, k = b_blk.shape
    total = -(-nb // _GROUP) * _GROUP if exact else nb
    if total != nb:                          # fixed-shape products, as in _block_operators
        b_blk = np.concatenate([b_blk, np.ones((total - nb, length, k))])
        pred = np.concatenate([pred, np.full((total - nb, k), 1.0 / k)])
    alpha = np.empty((total, length, k))
    logc = np.empty((total, length))
    cur = np.array(pred, dtype=np.float64)
    a = np.ascontiguousarray(transition, dtype=np.float64)
    for m in range(length):
        u = cur * b_blk[:, m, :]
        s = u.sum(axis=1)
        bad = ~(s > 0)
        safe = np.where(bad, 1.0, s)
        now = u / safe[:, None]
        if bad.any():
            now[bad] = cur[bad]
        alpha[:, m] = now
        with np.errstate(divide="ignore"):
            logc[:, m] = np.where(bad, -np.inf, np.log(safe))
        if exact:
            for g in range(0, total, _GROUP):
                np.matmul(now[g:g + _GROUP], a, out=cur[g:g + _GROUP])
        else:
            cur = now @ a
    return alpha[:nb], logc[:nb]


def forward_filter(log_b: np.ndarray, transition: np.ndarray, prior: np.ndarray, *,
                   block_length: int = 512, exact: bool = True
                   ) -> tuple[np.ndarray, np.ndarray]:
    r"""Filtered :math:`P(S_t \mid X_{\le t})` and :math:`\ln p(X_t \mid X_{<t})` per bar.

    *log_b* are the per-bar state log-densities (0 in every state where a bar
    carries no evidence); *prior* is the state distribution before the first
    bar. LIVE SAFE: row ``t`` depends on rows ``<= t`` only.
    """
    lb = np.asarray(log_b, dtype=np.float64)
    n, k = lb.shape
    if n == 0:
        return np.zeros((0, k)), np.zeros(0)
    b, m = _scaled_emissions(lb)
    b_blk = _blocks(b, block_length)
    prod, scale = _block_operators(b_blk, transition, exact=exact)
    pred = _forward_carries(prod, scale, np.asarray(prior, dtype=np.float64))
    alpha, logc = _forward_pass(b_blk, pred, transition, exact=exact)
    return alpha.reshape(-1, k)[:n], logc.reshape(-1)[:n] + m


def _backward_carries(prod: np.ndarray, scale: np.ndarray) -> np.ndarray:
    nb, k, _ = prod.shape
    carry = np.empty((nb + 1, k))
    carry[nb] = 1.0 / k
    for j in range(nb - 1, -1, -1):
        v = prod[j] @ carry[j + 1]
        with np.errstate(divide="ignore"):
            w = scale[j] + np.log(v)
        top = w.max()
        if not np.isfinite(top):
            carry[j] = 1.0 / k
            continue
        e = np.exp(w - top)
        carry[j] = e / e.sum()
    return carry


def _backward_pass(b_blk: np.ndarray, carry: np.ndarray, transition: np.ndarray) -> np.ndarray:
    nb, length, k = b_blk.shape
    beta = np.empty((nb, length, k))
    at = transition.T
    cur = carry[1:] @ at
    cur /= cur.sum(axis=1, keepdims=True)
    beta[:, length - 1] = cur
    for m in range(length - 2, -1, -1):
        cur = (b_blk[:, m + 1, :] * cur) @ at
        total = cur.sum(axis=1, keepdims=True)
        cur = cur / np.where(total > 0, total, 1.0)
        beta[:, m] = cur
    return beta


@dataclass
class _Posterior:
    alpha: np.ndarray        # (n, K) filtered
    gamma: np.ndarray        # (n, K) smoothed
    xi_sum: np.ndarray       # (K, K) expected transition counts
    log_likelihood: float


def forward_backward(log_b: np.ndarray, transition: np.ndarray, prior: np.ndarray, *,
                     block_length: int = 512, want_xi: bool = True) -> _Posterior:
    r"""Smoothed :math:`P(S_t \mid X_{1:T})`, filtered probabilities and expected
    transition counts. OFFLINE / NON-CAUSAL: every row depends on every bar."""
    lb = np.asarray(log_b, dtype=np.float64)
    n, k = lb.shape
    b, m = _scaled_emissions(lb)
    b_blk = _blocks(b, block_length)
    prod, scale = _block_operators(b_blk, transition, exact=False)
    pred = _forward_carries(prod, scale, np.asarray(prior, dtype=np.float64))
    alpha_blk, logc = _forward_pass(b_blk, pred, transition, exact=False)
    carry = _backward_carries(prod, scale)
    beta_blk = _backward_pass(b_blk, carry, transition)
    alpha = alpha_blk.reshape(-1, k)[:n]
    beta = beta_blk.reshape(-1, k)[:n]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    ll = float((logc.reshape(-1)[:n] + m).sum())
    xi_sum = np.zeros((k, k))
    if want_xi and n > 1:
        w = b[1:] * beta[1:]
        z = ((alpha[:-1] @ transition) * w).sum(axis=1)
        z = np.where(z > 0, z, 1.0)
        xi_sum = transition * ((alpha[:-1] / z[:, None]).T @ w)
    return _Posterior(alpha=alpha, gamma=gamma, xi_sum=xi_sum, log_likelihood=ll)


def viterbi_path(log_b: np.ndarray, transition: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Most likely state path given every bar. OFFLINE / NON-CAUSAL."""
    lb = np.asarray(log_b, dtype=np.float64)
    n, k = lb.shape
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    with np.errstate(divide="ignore"):
        log_a = np.log(transition)
        delta = np.log(prior / prior.sum()) + lb[0]
    back = np.empty((n, k), dtype=np.int8 if k < 127 else np.int16)
    cols = np.arange(k)
    for t in range(1, n):
        cand = delta[:, None] + log_a
        best = np.argmax(cand, axis=0)
        back[t] = best
        delta = cand[best, cols] + lb[t]
        delta -= delta.max()                 # keep the scale bounded; the argmax is unchanged
    path = np.empty(n, dtype=np.int64)
    path[-1] = int(np.argmax(delta))
    for t in range(n - 1, 0, -1):
        path[t - 1] = int(back[t, path[t]])
    return path


def stationary_distribution(transition: np.ndarray) -> np.ndarray:
    r"""The :math:`\pi` with :math:`\pi A = \pi`, :math:`\sum \pi = 1`."""
    a = np.asarray(transition, dtype=np.float64)
    k = a.shape[0]
    system = np.vstack([a.T - np.eye(k), np.ones((1, k))])
    target = np.zeros(k + 1)
    target[-1] = 1.0
    pi, *_ = np.linalg.lstsq(system, target, rcond=None)
    pi = np.clip(pi, 0.0, None)
    return pi / pi.sum() if pi.sum() > 0 else np.full(k, 1.0 / k)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
@dataclass
class FilterResult:
    """Live-safe forward-filter output for a span of bars."""

    probs: np.ndarray            # (n, K) P(S_t | X_<=t)
    log_predictive: np.ndarray   # (n,) ln p(X_t | X_<t); 0 where a bar was not scored
    scored: np.ndarray           # (n,) the bar carried evidence


@dataclass
class HMMModel:
    """Fitted Gaussian HMM (in scaled units) with its training diagnostics."""

    start: np.ndarray
    transition: np.ndarray
    states: GaussianStates
    covariance_type: str
    reg_covar: float
    pseudocount: float
    block_length: int = 512
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
        k = self.n_states
        return (k - 1) + k * (k - 1) + parameter_count(k, self.dimension, self.covariance_type)

    def bic(self) -> float:
        return -2.0 * self.log_likelihood + self.n_parameters * np.log(self.n_observations)

    def aic(self) -> float:
        return -2.0 * self.log_likelihood + 2.0 * self.n_parameters

    def stationary(self) -> np.ndarray:
        return stationary_distribution(self.transition)

    def expected_durations(self) -> np.ndarray:
        r""":math:`D_i = 1 / (1 - A_{ii})` bars (geometric durations)."""
        diag = np.diag(self.transition)
        with np.errstate(divide="ignore"):
            return np.where(diag < 1.0, 1.0 / (1.0 - diag), np.inf)

    def emission_log(self, values: np.ndarray, *, scorable: np.ndarray | None = None
                     ) -> tuple[np.ndarray, np.ndarray]:
        """(log-densities with 0 where not scored, scored mask)."""
        x = np.asarray(values, dtype=np.float64)
        used = np.isfinite(x).any(axis=1)
        if scorable is not None:
            used &= np.asarray(scorable, dtype=bool)
        return log_densities(x, self.states, scorable=used), used

    def _prior(self, prior: str | np.ndarray) -> np.ndarray:
        if isinstance(prior, np.ndarray):
            return prior
        if prior == "stationary":
            return self.stationary()
        if prior == "start":
            return self.start
        raise ValueError(f"unknown prior {prior!r}")

    def filter(self, values: np.ndarray, *, scorable: np.ndarray | None = None,
               prior: str | np.ndarray = "stationary", exact: bool = True) -> FilterResult:
        r"""LIVE SAFE :math:`P(S_t \mid X_{\le t})` over *values* (rows in time order)."""
        log_b, used = self.emission_log(values, scorable=scorable)
        probs, log_pred = forward_filter(log_b, self.transition, self._prior(prior),
                                         block_length=self.block_length, exact=exact)
        return FilterResult(probs=probs, log_predictive=np.where(used, log_pred, 0.0),
                            scored=used)

    def smooth(self, values: np.ndarray, *, scorable: np.ndarray | None = None,
               prior: str | np.ndarray = "stationary") -> np.ndarray:
        r"""OFFLINE / NON-CAUSAL :math:`P(S_t \mid X_{1:T})`."""
        log_b, _ = self.emission_log(values, scorable=scorable)
        return forward_backward(log_b, self.transition, self._prior(prior),
                                block_length=self.block_length, want_xi=False).gamma

    def viterbi(self, values: np.ndarray, *, scorable: np.ndarray | None = None,
                prior: str | np.ndarray = "stationary") -> np.ndarray:
        """OFFLINE / NON-CAUSAL most likely path."""
        log_b, _ = self.emission_log(values, scorable=scorable)
        return viterbi_path(log_b, self.transition, self._prior(prior))

    def predict_next(self, probs: np.ndarray) -> np.ndarray:
        r""":math:`P(S_{t+1} = j \mid X_{\le t}) = \sum_i P(S_t = i \mid X_{\le t}) A_{ij}`.

        Summed elementwise in a fixed order (K is small), so a row's value
        never depends on how many rows are passed (prefix invariance).
        """
        p = np.asarray(probs, dtype=np.float64)
        out = p[:, 0, None] * self.transition[0][None, :]
        for i in range(1, self.n_states):
            out = out + p[:, i, None] * self.transition[i][None, :]
        return out

    def leave_probability(self, probs: np.ndarray) -> np.ndarray:
        r""":math:`1 - \sum_i P(S_t = i \mid X_{\le t}) A_{ii}`, elementwise (prefix invariant)."""
        p = np.asarray(probs, dtype=np.float64)
        stay = p[:, 0] * self.transition[0, 0]
        for i in range(1, self.n_states):
            stay = stay + p[:, i] * self.transition[i, i]
        return 1.0 - stay

    def permuted(self, order: np.ndarray | list[int]) -> HMMModel:
        idx = np.asarray(order, dtype=np.int64)
        return HMMModel(start=self.start[idx].copy(),
                        transition=self.transition[np.ix_(idx, idx)].copy(),
                        states=self.states.permuted(idx), covariance_type=self.covariance_type,
                        reg_covar=self.reg_covar, pseudocount=self.pseudocount,
                        block_length=self.block_length, log_likelihood=self.log_likelihood,
                        n_observations=self.n_observations, n_iter=self.n_iter,
                        converged=self.converged, reseeds=self.reseeds)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "hmm", "start": self.start.tolist(),
                "transition": self.transition.tolist(), "means": self.states.means.tolist(),
                "covariances": self.states.covariances.tolist(),
                "covariance_type": self.covariance_type, "reg_covar": self.reg_covar,
                "pseudocount": self.pseudocount, "block_length": self.block_length,
                "log_likelihood": self.log_likelihood, "n_observations": self.n_observations,
                "n_iter": self.n_iter, "converged": self.converged, "reseeds": self.reseeds}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HMMModel:
        states = GaussianStates(np.asarray(data["means"], dtype=np.float64),
                                np.asarray(data["covariances"], dtype=np.float64),
                                str(data["covariance_type"]))
        return cls(start=np.asarray(data["start"], dtype=np.float64),
                   transition=np.asarray(data["transition"], dtype=np.float64), states=states,
                   covariance_type=str(data["covariance_type"]),
                   reg_covar=float(data["reg_covar"]), pseudocount=float(data["pseudocount"]),
                   block_length=int(data["block_length"]),
                   log_likelihood=float(data["log_likelihood"]),
                   n_observations=int(data["n_observations"]), n_iter=int(data["n_iter"]),
                   converged=bool(data["converged"]), reseeds=int(data.get("reseeds", 0)))


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def _transition_from_labels(labels: np.ndarray, valid: np.ndarray, k: int,
                            pseudocount: float) -> np.ndarray:
    """Transition counts between consecutive scored bars' labels, plus a pseudo-count."""
    counts = np.full((k, k), max(pseudocount, 1e-3))
    idx = np.flatnonzero(valid)
    if idx.size > 1:
        a, b = labels[idx[:-1]], labels[idx[1:]]
        adjacent = idx[1:] == idx[:-1] + 1
        np.add.at(counts, (a[adjacent], b[adjacent]), 1.0)
    return counts / counts.sum(axis=1, keepdims=True)


def _from_gmm(gmm: GMMModel, x: np.ndarray, train: np.ndarray, *, reg_covar: float,
              covariance_type: str, pseudocount: float, block_length: int) -> HMMModel:
    probs, _ = gmm.posterior(x, scorable=train)
    labels = np.where(train, np.argmax(np.nan_to_num(probs, nan=-1.0), axis=1), -1)
    transition = _transition_from_labels(labels, train, gmm.n_states, pseudocount)
    return HMMModel(start=gmm.weights.copy(), transition=transition,
                    states=GaussianStates(gmm.states.means.copy(),
                                          regularise(gmm.states.covariances, 0.0,
                                                     covariance_type), covariance_type),
                    covariance_type=covariance_type, reg_covar=reg_covar,
                    pseudocount=pseudocount, block_length=block_length)


def _baum_welch(model: HMMModel, x: np.ndarray, train: np.ndarray, *, max_iter: int,
                tol: float, rng: np.random.Generator) -> HMMModel:
    """EM from *model*; *train* marks the bars that carry evidence (complete rows)."""
    n_obs = int(train.sum())
    xs = x[train]
    previous = -np.inf
    current = model
    ll = -np.inf
    converged = False
    reseeds = 0
    iterations = 0
    for _ in range(max_iter):
        iterations += 1
        log_b = log_densities(x, current.states, scorable=train)
        post = forward_backward(log_b, current.transition, current.start,
                                block_length=current.block_length)
        del log_b
        ll = post.log_likelihood
        if abs(ll - previous) / max(n_obs, 1) < tol:
            converged = True
            break
        previous = ll
        gamma = post.gamma
        start = gamma[0] + 1e-3
        start /= start.sum()
        counts = post.xi_sum + current.pseudocount
        transition = counts / counts.sum(axis=1, keepdims=True)
        _, means, cov = m_step(xs, gamma[train], covariance_type=current.covariance_type,
                               reg_covar=current.reg_covar)
        weight = gamma[train].sum(axis=0)
        for j in np.flatnonzero(weight < 2.0):  # a state that lost every bar restarts
            means[j] = xs[rng.integers(xs.shape[0])]
            cov[j] = regularise(np.cov(xs[rng.choice(xs.shape[0], min(xs.shape[0], 20_000),
                                                     replace=False)].T)[None],
                                current.reg_covar, current.covariance_type)[0]
            reseeds += 1
            previous = -np.inf
        current = HMMModel(start=start, transition=transition,
                           states=GaussianStates(means, cov, current.covariance_type),
                           covariance_type=current.covariance_type, reg_covar=current.reg_covar,
                           pseudocount=current.pseudocount, block_length=current.block_length)
        del post, gamma
    if not converged:                        # score the parameters actually returned
        log_b = log_densities(x, current.states, scorable=train)
        _, log_pred = forward_filter(log_b, current.transition, current.start,
                                     block_length=current.block_length, exact=False)
        ll = float(log_pred.sum())
    current.log_likelihood = ll
    current.n_observations = n_obs
    current.n_iter = iterations
    current.converged = converged
    current.reseeds = reseeds
    return current


def fit_hmm(values: np.ndarray, k: int, *, train: np.ndarray | None = None,
            covariance_type: str = "full", n_init: int = 3, max_iter: int = 200,
            tol: float = 1e-5, reg_covar: float = 1e-4, pseudocount: float = 1.0,
            block_length: int = 512, seed: int = 0, init: HMMModel | None = None,
            gmm_max_iter: int = 100, init_rows: int = 200_000) -> HMMModel:
    """Best of *n_init* Baum-Welch runs (each started from a GMM fit) and *init*.

    *values* are rows in time order (scaled). *train* marks the bars used as
    evidence; by default the complete rows. Other bars keep their place in the
    sequence, so the chain runs through them without evidence. The starting
    mixture of each run is fitted on at most *init_rows* evenly spaced
    evidence rows; Baum-Welch then uses every bar.
    """
    x = np.asarray(values, dtype=np.float64)
    evidence = np.isfinite(x).all(axis=1)
    if train is not None:
        evidence &= np.asarray(train, dtype=bool)
    if int(evidence.sum()) < 10 * k:
        raise ValueError(f"a {k}-state HMM needs at least {10 * k} complete rows")
    candidates: list[HMMModel] = []
    if init is not None:
        start_model = HMMModel(start=init.start.copy(), transition=init.transition.copy(),
                               states=init.states, covariance_type=covariance_type,
                               reg_covar=reg_covar, pseudocount=pseudocount,
                               block_length=block_length)
        candidates.append(_baum_welch(start_model, x, evidence, max_iter=max_iter, tol=tol,
                                      rng=np.random.default_rng([seed, 999])))
    rows = np.flatnonzero(evidence)
    if rows.size > init_rows:                # the starting mixture needs no more than this
        rows = rows[np.linspace(0, rows.size - 1, init_rows).round().astype(np.int64)]
    for run in range(n_init):
        gmm = fit_gmm(x[rows], k, covariance_type=covariance_type, n_init=1,
                      max_iter=gmm_max_iter, tol=max(tol, 1e-4), reg_covar=reg_covar,
                      seed=seed + 7919 * (run + 1))
        start_model = _from_gmm(gmm, x, evidence, reg_covar=reg_covar,
                                covariance_type=covariance_type, pseudocount=pseudocount,
                                block_length=block_length)
        candidates.append(_baum_welch(start_model, x, evidence, max_iter=max_iter, tol=tol,
                                      rng=np.random.default_rng([seed, run])))
    if not candidates:
        raise ValueError("fit_hmm needs n_init >= 1 or an initial model")
    return max(candidates, key=lambda model: model.log_likelihood)
