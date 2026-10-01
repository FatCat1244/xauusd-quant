r"""Null and synthetic controls for regime discovery (Steps 43-48).

**Why nulls.** Every regime input is a trailing-window statistic, so consecutive
bars share most of their data and the features move smoothly - an HMM will
report "persistent states" in almost anything built this way. Persistence is
therefore read against controls that keep that mechanism and remove the rest:

``row_shuffle``            the real feature vectors in random order: every
                           marginal kept, every temporal link gone. Persistence
                           must collapse (Step 47) - a check that the persistence
                           comes from the ordering at all.
``row_block_<d>d``         circular blocks of ``d`` trading days of real feature
                           rows: local dependence kept, anything longer broken
                           (Step 48). Real durations far beyond the block length
                           would be long-range regime structure.
``random_walk`` ...        pipeline nulls (Prompts #5/#6): a Gaussian random walk,
                           the real returns shuffled, or block-bootstrapped in
                           1024-bar blocks, pushed through the *same* regression, OU,
                           FFT and wavelet engines. They keep the rolling-window
                           smoothness and (for the bootstrap) volatility clustering
                           but no price regimes. Spread and activity are the real
                           bars' (they are not functions of the price path).

For each source and K the same Gaussian HMM (and a GMM with the same K) is
fitted on the same span of bars; the table reports self-transitions,
expected and observed (smoothed-path) durations, the persistence excess
:math:`A_{ii} - \pi_i` (0 for no temporal structure), the likelihood the
transition matrix adds over the time-blind mixture, and state separation.

**Synthetic controls** (Steps 44-46) run the same code on series with known
answers: a 3-state HMM (recovery), one regime (Gaussian and heavy-tailed), and
a continuously drifting volatility (what discretising a continuum looks like).
"""

from __future__ import annotations

import gc
import time
from typing import Any

import numpy as np
import polars as pl

from ..regimes.config import RegimeConfig
from ..regimes.dataset import feature_matrix
from ..regimes.diagnostics import degenerate_flags, state_separation
from ..regimes.emissions import bhattacharyya, log_densities
from ..regimes.gmm import GMMModel, fit_gmm
from ..regimes.hmm import HMMModel, fit_hmm
from ..regimes.preprocessing import fit_scaler
from ..regimes.state_alignment import align_states, canonical_order
from ..regimes.synthetic import known_hmm, single_regime, smooth_continuum
from ..regimes.transitions import run_lengths
from .regime_analysis import single_gaussian

__all__ = [
    "adjacency_share",
    "block_bootstrap_rows",
    "null_persistence",
    "persistence_row",
    "shuffle_rows",
    "synthetic_controls",
]


def shuffle_rows(x: np.ndarray, seed: int) -> np.ndarray:
    return x[np.random.default_rng(seed).permutation(x.shape[0])]


def block_bootstrap_rows(x: np.ndarray, block: int, seed: int) -> np.ndarray:
    """Circular blocks of *block* consecutive rows, drawn with replacement."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
    return x[idx]


def adjacency_share(transition: np.ndarray, order: np.ndarray) -> float | None:
    """Share of off-diagonal transition mass going to a neighbouring state in *order*.

    States ordered by a feature (volatility): a discretised continuum moves
    only between neighbours (share ~1); discrete regimes can jump anywhere.
    Undefined for K < 3.
    """
    k = transition.shape[0]
    if k < 3:
        return None
    rank = np.empty(k, dtype=np.int64)
    rank[order] = np.arange(k)
    off = transition.copy()
    np.fill_diagonal(off, 0.0)
    adjacent = np.abs(rank[:, None] - rank[None, :]) == 1
    total = off.sum()
    return float(off[adjacent].sum() / total) if total > 0 else None


def persistence_row(hmm: HMMModel, gmm: GMMModel | None, z: np.ndarray, *, vol_index: int,
                    regime: RegimeConfig) -> dict[str, Any]:
    """Persistence, temporal information and separation of one fitted HMM."""
    k = hmm.n_states
    pi = hmm.stationary()
    diag = np.diag(hmm.transition)
    durations = hmm.expected_durations()
    probs = hmm.smooth(z)
    path = np.argmax(probs, axis=1)
    runs = run_lengths(path)
    full = runs.filter(~pl.col("censored"))["length"].to_numpy()
    means = hmm.states.means
    order = canonical_order(means, vol_index)
    separation = state_separation(means, hmm.states.covariances)
    flags = degenerate_flags(pi, covariances=hmm.states.covariances, separation=separation,
                             transition=hmm.transition,
                             min_state_fraction=regime.diagnostics.min_state_fraction,
                             max_state_fraction=regime.diagnostics.max_state_fraction,
                             max_condition_number=regime.diagnostics.max_condition_number,
                             duplicate_bhattacharyya=regime.diagnostics.duplicate_bhattacharyya,
                             min_expected_duration=regime.diagnostics.min_expected_duration,
                             max_expected_duration=regime.diagnostics.max_expected_duration)
    n = hmm.n_observations
    row: dict[str, Any] = {
        "states": k, "bars": int(z.shape[0]),
        "mean_self_transition": float(diag.mean()),
        "min_self_transition": float(diag.min()),
        "median_expected_duration": float(np.median(durations)),
        "min_expected_duration": float(durations.min()),
        "median_observed_duration": float(np.median(full)) if full.size else None,
        "mean_observed_duration": float(full.mean()) if full.size else None,
        "persistence_excess": float(np.mean(diag - pi)),
        "switch_rate": float(np.mean(path[1:] != path[:-1])),
        "adjacency_share": adjacency_share(hmm.transition, order),
        "min_bhattacharyya": flags.get("min_bhattacharyya"),
        "degenerate": flags["degenerate"], "flags": flags["flags"],
        "hmm_ll_per_obs": hmm.log_likelihood / n if n else None,
    }
    if gmm is not None and gmm.n_observations:
        row["gmm_ll_per_obs"] = gmm.log_likelihood / gmm.n_observations
        row["temporal_gain_per_obs"] = row["hmm_ll_per_obs"] - row["gmm_ll_per_obs"]
    return row


def _fit_pair(x: np.ndarray, features: tuple[str, ...], k: int, regime: RegimeConfig, *,
              seed: int, n_init: int) -> tuple[HMMModel, GMMModel, np.ndarray]:
    pre = regime.preprocessing
    scaler = fit_scaler(x, features, method=pre.scaling, clip=pre.clip)
    z = scaler.transform(x)
    h = regime.models.hmm
    hmm = fit_hmm(z, k, covariance_type=h.covariance_type, n_init=n_init, max_iter=h.max_iter,
                  tol=h.tol, reg_covar=h.reg_covar, pseudocount=h.transition_pseudocount,
                  block_length=h.block_length, seed=seed)
    gmm = fit_gmm(z, k, covariance_type=h.covariance_type, n_init=n_init,
                  max_iter=regime.models.gmm.max_iter, tol=regime.models.gmm.tol,
                  reg_covar=regime.models.gmm.reg_covar, seed=seed)
    return hmm, gmm, z


def null_persistence(sources: dict[str, np.ndarray], features: tuple[str, ...],
                     regime: RegimeConfig, *, states: tuple[int, ...], seed: int,
                     n_init: int = 1, progress: Any = None) -> pl.DataFrame:
    """HMM persistence and temporal information for every source and K (same code, same span)."""
    vol_index = (list(features).index(regime.baselines.volatility_feature)
                 if regime.baselines.volatility_feature in features else 0)
    rows = []
    for name, x in sources.items():
        for k in states:
            t0 = time.perf_counter()
            try:
                hmm, gmm, z = _fit_pair(x, features, k, regime, seed=seed + k, n_init=n_init)
            except ValueError as exc:
                rows.append({"source": name, "states": k, "error": str(exc)})
                continue
            row = persistence_row(hmm, gmm, z, vol_index=vol_index, regime=regime)
            row.update({"source": name, "fit_seconds": time.perf_counter() - t0})
            rows.append(row)
            del hmm, gmm, z
            gc.collect()
            if progress is not None:
                progress(f"null {name} K={k}: {time.perf_counter() - t0:.0f}s")
    return pl.DataFrame(rows, infer_schema_length=None)


# ---------------------------------------------------------------------------
# Synthetic controls (Steps 44-46)
# ---------------------------------------------------------------------------
def _holdout_ll(model: GMMModel | HMMModel, z: np.ndarray, cut: int) -> float:
    if isinstance(model, HMMModel):
        result = model.filter(z, exact=False)
        return float(result.log_predictive[cut:].mean())
    probs, marginal = model.posterior(z[cut:])
    return float(np.nanmean(marginal))


def synthetic_controls(regime: RegimeConfig, *, states: tuple[int, ...] = (1, 2, 3, 4, 5)
                       ) -> dict[str, pl.DataFrame]:
    """Model selection, recovery and persistence on series with known answers."""
    cfg = regime.synthetic
    h = regime.models.hmm
    n = cfg.bars
    cases = {
        "known_3_state_hmm": known_hmm(n, seed=cfg.seed),
        "single_regime_gaussian": single_regime(n, seed=cfg.seed + 1),
        "single_regime_student_t": single_regime(n, seed=cfg.seed + 2, heavy_tails=True),
        "smooth_continuum": smooth_continuum(n, seed=cfg.seed + 3),
    }
    shuffled = cases["known_3_state_hmm"]
    rng = np.random.default_rng(cfg.seed + 4)
    cases["known_hmm_rows_shuffled"] = type(shuffled)(
        shuffled.values[rng.permutation(n)], None, None, None, None,
        "the 3-state HMM's rows in random order (temporal structure removed)")
    selection_rows: list[dict[str, Any]] = []
    persistence_rows: list[dict[str, Any]] = []
    recovery: dict[str, Any] = {}
    cut = int(0.7 * n)
    for name, series in cases.items():
        x = series.values
        d = x.shape[1]
        features = tuple(f"x{j}" for j in range(d))
        scaler = fit_scaler(x[:cut], features, method="standard", clip=None)
        z = scaler.transform(x)
        for k in states:
            if k == 1:
                one = single_gaussian(z[:cut], h.reg_covar)
                selection_rows.append({"case": name, "model": "single", "states": 1,
                                       "bic": -2 * one.log_likelihood + (d + d * (d + 1) // 2)
                                       * np.log(one.n_observations),
                                       "test_ll_per_obs": float(log_densities(
                                           z[cut:], one.states)[:, 0].mean())})
                continue
            gmm = fit_gmm(z[:cut], k, covariance_type="full", n_init=2, seed=cfg.seed + k,
                          reg_covar=h.reg_covar)
            hmm = fit_hmm(z[:cut], k, covariance_type="full", n_init=2, seed=cfg.seed + k,
                          reg_covar=h.reg_covar, pseudocount=h.transition_pseudocount,
                          block_length=h.block_length)
            for family, model in (("gmm", gmm), ("hmm", hmm)):
                selection_rows.append({"case": name, "model": family, "states": k,
                                       "bic": model.bic(), "aic": model.aic(),
                                       "train_ll_per_obs": model.log_likelihood
                                       / model.n_observations,
                                       "test_ll_per_obs": _holdout_ll(model, z, cut)})
            row = persistence_row(hmm, gmm, z[:cut], vol_index=0, regime=regime)
            row["case"] = name
            persistence_rows.append(row)
            if name == "known_3_state_hmm" and k == 3 and series.states is not None:
                recovery = _recovery(hmm, series, scaler, z)
    return {"selection": pl.DataFrame(selection_rows, infer_schema_length=None),
            "persistence": pl.DataFrame(persistence_rows, infer_schema_length=None),
            "recovery": pl.DataFrame([recovery]) if recovery else pl.DataFrame()}


def _recovery(hmm: HMMModel, series: Any, scaler: Any, z: np.ndarray) -> dict[str, Any]:
    """Fitted vs true 3-state HMM after aligning labels (Hungarian on Bhattacharyya)."""
    true_means = (series.means - scaler.center) / scaler.scale
    true_cov = series.covariances / (scaler.scale[:, None] * scaler.scale[None, :])
    cost = np.array([[bhattacharyya(true_means[i], true_cov[i], hmm.states.means[j],
                                    hmm.states.covariances[j]) for j in range(3)]
                     for i in range(3)])
    alignment = align_states(cost)
    fitted = hmm.permuted(alignment.order)
    result = fitted.filter(z, exact=False)
    predicted = np.argmax(result.probs, axis=1)
    truth = series.states
    confidence = result.probs.max(axis=1)
    correct = (predicted == truth).astype(np.float64)
    edges = np.quantile(confidence, np.linspace(0, 1, 11))
    bins = np.clip(np.searchsorted(edges, confidence, side="right") - 1, 0, 9)
    gap = max(abs(confidence[bins == b].mean() - correct[bins == b].mean())
              for b in range(10) if (bins == b).any())
    return {"max_transition_error": float(np.abs(fitted.transition - series.transition).max()),
            "true_self_transitions": np.diag(series.transition).tolist(),
            "fitted_self_transitions": np.diag(fitted.transition).tolist(),
            "max_mean_error_scaled": float(np.abs(fitted.states.means - true_means).max()),
            "matched_bhattacharyya": alignment.costs.tolist(),
            "filtered_state_accuracy": float(correct.mean()),
            "smoothed_state_accuracy": float(np.mean(np.argmax(fitted.smooth(z), axis=1)
                                                     == truth)),
            "filtered_confidence_calibration_max_gap": float(gap)}


def real_span(frame: pl.DataFrame, max_rows: int) -> tuple[int, int]:
    """The last *max_rows* bars: the span every null and the real series share."""
    start = max(0, frame.height - max_rows)
    return start, frame.height


def span_matrix(frame: pl.DataFrame, features: tuple[str, ...], span: tuple[int, int]
                ) -> np.ndarray:
    return feature_matrix(frame.slice(span[0], span[1] - span[0]), features)
