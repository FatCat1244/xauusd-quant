"""One interface over the three model families: fit, align, infer.

A :class:`FittedRegimeModel` bundles a fitted K-Means, GMM or HMM with the
frozen scaler it was trained under and the feature list, so that inference on
new bars always goes raw features -> the *training* scaler -> the model. The
walk-forward engine (causal) and the offline studies (full sample, labelled
non-causal) both fit through :func:`fit_regime_model`.

``infer`` returns only live-safe quantities (the HMM forward filter, the
GMM's per-bar posterior, K-Means' nearest centre). The smoothed probabilities
and the Viterbi path are separate methods, named for what they are.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .clustering import KMeansModel, _lloyd, fit_kmeans
from .config import RegimeConfig
from .emissions import GaussianStates, regularise
from .gmm import GMMModel, fit_gmm, state_entropy
from .hmm import HMMModel, fit_hmm
from .preprocessing import FittedScaler, MissingPolicy, coverage

__all__ = ["FittedRegimeModel", "InferenceResult", "fit_regime_model", "missing_policy",
           "warm_start"]

_COVARIANCE = {"gmm": "full", "gmm_diag": "diag", "gmm_tied": "tied"}


def missing_policy(regime: RegimeConfig) -> MissingPolicy:
    pre = regime.preprocessing
    return MissingPolicy(policy=pre.missing_policy, min_coverage=pre.min_feature_coverage,
                         required=pre.required_features)


@dataclass
class InferenceResult:
    """Live-safe per-bar output of one model over a span of bars."""

    probs: np.ndarray | None          # (n, K) state probabilities (None for K-Means)
    state: np.ndarray                 # (n,) most likely state, -1 where no estimate
    confidence: np.ndarray            # (n,) probability of that state
    entropy: np.ndarray               # (n,) entropy of the probabilities
    log_score: np.ndarray             # (n,) HMM: ln p(x_t | x_<t); GMM: ln p(x_t); K-Means: -d^2
    scored: np.ndarray                # (n,) the bar carried evidence
    complete: np.ndarray              # (n,) every feature observed
    coverage: np.ndarray              # (n,) share of features observed
    next_probs: np.ndarray | None = None   # HMM: P(S_{t+1} = j | X_<=t)
    leave: np.ndarray | None = None        # HMM: P(S_{t+1} != S_t | X_<=t)

    def tail(self, start: int) -> InferenceResult:
        cut = slice(start, None)

        def part(a: np.ndarray | None) -> np.ndarray | None:
            return None if a is None else a[cut]

        return InferenceResult(part(self.probs), self.state[cut], self.confidence[cut],
                               self.entropy[cut], self.log_score[cut], self.scored[cut],
                               self.complete[cut], self.coverage[cut], part(self.next_probs),
                               part(self.leave))


@dataclass
class FittedRegimeModel:
    """A fitted model, the frozen scaler it expects, and the feature order."""

    family: str
    model: KMeansModel | GMMModel | HMMModel
    scaler: FittedScaler

    @property
    def features(self) -> tuple[str, ...]:
        return self.scaler.features

    @property
    def n_states(self) -> int:
        return self.model.n_states

    def raw_parameters(self) -> tuple[np.ndarray, np.ndarray | None]:
        """State means (and covariances) in raw feature units."""
        if isinstance(self.model, KMeansModel):
            return self.scaler.means_to_raw(self.model.centers), None
        states = self.model.states
        return (self.scaler.means_to_raw(states.means),
                self.scaler.covariances_to_raw(states.covariances))

    def permuted(self, order: np.ndarray) -> FittedRegimeModel:
        return FittedRegimeModel(self.family, self.model.permuted(order), self.scaler)

    def transition(self) -> np.ndarray | None:
        return self.model.transition if isinstance(self.model, HMMModel) else None

    def train_log_likelihood(self) -> float | None:
        if isinstance(self.model, KMeansModel):
            return None
        return float(self.model.log_likelihood)

    def information_criteria(self) -> dict[str, float | None]:
        if isinstance(self.model, KMeansModel):
            return {"n_parameters": None, "bic": None, "aic": None}
        return {"n_parameters": float(self.model.n_parameters), "bic": self.model.bic(),
                "aic": self.model.aic()}

    def _prepare(self, values: np.ndarray, policy: MissingPolicy
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw = np.asarray(values, dtype=np.float64)
        cov = coverage(raw)
        z, scorable = policy.prepare(self.scaler.transform(raw), self.features, self.scaler)
        complete = np.isfinite(raw).all(axis=1)
        return z, scorable, complete, cov

    def infer(self, values: np.ndarray, policy: MissingPolicy, *,
              exact: bool = True) -> InferenceResult:
        """LIVE SAFE estimates for rows in time order (an HMM filters through them)."""
        z, scorable, complete, cov = self._prepare(values, policy)
        n = z.shape[0]
        if isinstance(self.model, HMMModel):
            result = self.model.filter(z, scorable=scorable, exact=exact)
            probs = result.probs
            nxt = self.model.predict_next(probs)
            leave = self.model.leave_probability(probs)
            state = np.argmax(probs, axis=1).astype(np.int64)
            return InferenceResult(probs=probs, state=state, confidence=probs.max(axis=1),
                                   entropy=state_entropy(probs),
                                   log_score=np.where(result.scored, result.log_predictive,
                                                      np.nan),
                                   scored=result.scored, complete=complete & result.scored,
                                   coverage=cov, next_probs=nxt, leave=leave)
        if isinstance(self.model, GMMModel):
            probs, marginal = self.model.posterior(z, scorable=scorable)
            scored = np.isfinite(marginal)
            state = np.where(scored, np.argmax(np.nan_to_num(probs, nan=-1.0), axis=1), -1)
            confidence = np.where(scored, np.nanmax(np.where(scored[:, None], probs, 0.0),
                                                     axis=1), np.nan)
            return InferenceResult(probs=probs, state=state.astype(np.int64),
                                   confidence=confidence, entropy=state_entropy(probs),
                                   log_score=marginal, scored=scored, complete=complete & scored,
                                   coverage=cov)
        labels, dist = self.model.predict(z, scorable=scorable)
        scored = labels >= 0
        return InferenceResult(probs=None, state=labels, confidence=np.full(n, np.nan),
                               entropy=np.full(n, np.nan), log_score=-dist, scored=scored,
                               complete=complete & scored, coverage=cov)

    def smoothed(self, values: np.ndarray, policy: MissingPolicy) -> np.ndarray:
        """OFFLINE / NON-CAUSAL smoothed probabilities (HMM only)."""
        if not isinstance(self.model, HMMModel):
            raise TypeError("only a hidden Markov model has smoothed probabilities")
        z, scorable, _, _ = self._prepare(values, policy)
        return self.model.smooth(z, scorable=scorable)

    def viterbi(self, values: np.ndarray, policy: MissingPolicy) -> np.ndarray:
        """OFFLINE / NON-CAUSAL most likely path (HMM only)."""
        if not isinstance(self.model, HMMModel):
            raise TypeError("only a hidden Markov model has a Viterbi path")
        z, scorable, _, _ = self._prepare(values, policy)
        return self.model.viterbi(z, scorable=scorable)

    def to_dict(self) -> dict[str, Any]:
        return {"family": self.family, "parameters": self.model.to_dict(),
                "scaler": self.scaler.to_dict()}


def warm_start(previous: FittedRegimeModel, scaler: FittedScaler
               ) -> KMeansModel | GMMModel | HMMModel:
    """The previous period's model re-expressed in the new scaler's units."""
    model = previous.model
    if isinstance(model, KMeansModel):
        raw = previous.scaler.means_to_raw(model.centers)
        centers = (raw - scaler.center) / scaler.scale
        return KMeansModel(centers, model.inertia, model.n_observations, 0, False)
    means, cov = previous.scaler.map_to(scaler, model.states.means, model.states.covariances)
    states = GaussianStates(means, cov, model.states.covariance_type)
    if isinstance(model, GMMModel):
        return GMMModel(model.weights.copy(), states, model.covariance_type, model.reg_covar)
    return HMMModel(start=model.start.copy(), transition=model.transition.copy(), states=states,
                    covariance_type=model.covariance_type, reg_covar=model.reg_covar,
                    pseudocount=model.pseudocount, block_length=model.block_length)


def fit_regime_model(family: str, scaled: np.ndarray, k: int, regime: RegimeConfig, *,
                     seed: int, n_init: int | None = None, max_iter: int | None = None,
                     tol: float | None = None,
                     init: KMeansModel | GMMModel | HMMModel | None = None
                     ) -> KMeansModel | GMMModel | HMMModel:
    """Fit one model on scaled training rows in time order (incomplete rows kept for the HMM)."""
    m = regime.models
    if family == "kmeans":
        cfg = m.kmeans
        runs = cfg.n_init if n_init is None else n_init
        x = scaled[np.isfinite(scaled).all(axis=1)]
        best: KMeansModel | None = None
        if runs > 0:
            best = fit_kmeans(x, k, n_init=runs, max_iter=max_iter or cfg.max_iter,
                              tol=tol or cfg.tol, seed=seed)
        if isinstance(init, KMeansModel):
            centers, inertia, it, conv, resets = _lloyd(x, init.centers.copy(),
                                                        max_iter=max_iter or cfg.max_iter,
                                                        tol=tol or cfg.tol)
            warm = KMeansModel(centers, inertia, int(x.shape[0]), it, conv, resets)
            if best is None or warm.inertia < best.inertia:
                best = warm
        if best is None:
            raise ValueError("K-Means refit needs n_init >= 1 or a warm start")
        return best
    if family in _COVARIANCE:
        cfg_g = m.gmm
        return fit_gmm(scaled, k, covariance_type=_COVARIANCE[family],
                       n_init=cfg_g.n_init if n_init is None else n_init,
                       max_iter=max_iter or cfg_g.max_iter, tol=tol or cfg_g.tol,
                       reg_covar=cfg_g.reg_covar, seed=seed,
                       init=init if isinstance(init, GMMModel) else None)
    if family == "hmm":
        cfg_h = m.hmm
        hmm_init: HMMModel | None = None
        if isinstance(init, HMMModel):
            hmm_init = HMMModel(start=init.start, transition=init.transition,
                                states=GaussianStates(init.states.means,
                                                      regularise(init.states.covariances, 0.0,
                                                                 cfg_h.covariance_type),
                                                      cfg_h.covariance_type),
                                covariance_type=cfg_h.covariance_type,
                                reg_covar=cfg_h.reg_covar,
                                pseudocount=cfg_h.transition_pseudocount,
                                block_length=cfg_h.block_length)
        return fit_hmm(scaled, k, covariance_type=cfg_h.covariance_type,
                       n_init=cfg_h.n_init if n_init is None else n_init,
                       max_iter=max_iter or cfg_h.max_iter, tol=tol or cfg_h.tol,
                       reg_covar=cfg_h.reg_covar, pseudocount=cfg_h.transition_pseudocount,
                       block_length=cfg_h.block_length, seed=seed, init=hmm_init)
    raise ValueError(f"unknown model family {family!r}")
