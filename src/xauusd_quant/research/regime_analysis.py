r"""Offline model comparison, baselines and state descriptions (Steps 9-12, 19-22, 38-39, 43).

**Offline fits** use the whole 2003-2026 sample and a full-sample scaler. They
answer "what does the data look like?" - information criteria, separation,
profiles, smoothed durations - and are labelled
``OFFLINE / NON-CAUSAL RESEARCH ONLY``: nothing fitted here becomes a feature.
The **chronological hold-out** refits every model on bars before
``holdout_split`` and scores the bars after it (the HMM by its forward
filter), which is the offline study's out-of-sample likelihood.

**Baselines** every model is read against:

``single_state``        one Gaussian, K = 1: no regimes at all.
``volatility_buckets``  K quantile buckets of the volatility feature, edges from the
                        training data. As a *density* the buckets are a mixture whose
                        components are fitted to each bucket's bars (hard assignment),
                        so its likelihood is directly comparable with a GMM's.
``random_labels``       states drawn from a Markov chain with the model's own transition
                        matrix (or state frequencies), independent of the data: what
                        "differentiation" chance produces.

States are numbered, never named; :func:`describe_states` produces factual
descriptions ("higher volatility, wider spread ...") from the profiles.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp

from ..regimes.clustering import KMeansModel, silhouette_estimate
from ..regimes.config import OFFLINE_LABEL, RegimeConfig
from ..regimes.diagnostics import (
    adjusted_rand,
    contingency,
    cramers_v,
    degenerate_flags,
    normalized_mutual_info,
    state_separation,
)
from ..regimes.emissions import GaussianStates, log_densities
from ..regimes.fitting import FittedRegimeModel, fit_regime_model, missing_policy
from ..regimes.gmm import GMMModel, m_step
from ..regimes.hmm import HMMModel
from ..regimes.preprocessing import clip_fractions, fit_scaler
from ..regimes.state_alignment import canonical_order
from ..regimes.transitions import duration_summary, run_lengths
from .config import ResearchConfig
from .intraday import assign_sessions
from .ou_estimation import num

__all__ = [
    "PROFILE_COLUMNS",
    "agreement",
    "bucket_labels",
    "bucket_mixture",
    "describe_states",
    "frequency_table",
    "holdout_scores",
    "intraday_frequency",
    "offline_fit",
    "offline_model_row",
    "random_labels",
    "scaling_comparison",
    "single_gaussian",
    "state_profiles",
    "standardized_means",
]

#: What a state profile describes (Step 21), inputs and descriptive columns alike.
PROFILE_COLUMNS: tuple[str, ...] = (
    "log_return", "return_z_20", "log_rv_20", "rv_percentile", "abs_return_acf1",
    "regression_slope", "slope_over_volatility", "residual_zscore", "r_squared", "ou_theta",
    "ou_half_life_bars", "log_ou_half_life", "ou_valid", "spectral_entropy",
    "fft_high_low_log_ratio", "fft_top3_power_share", "wavelet_entropy",
    "wavelet_fast_slow_log_ratio", "wavelet_top3_scale_share", "median_spread",
    "spread_percentile", "tick_count", "activity_percentile",
)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def offline_fit(x: np.ndarray, features: tuple[str, ...], regime: RegimeConfig, *,
                family: str, k: int, seed: int, scaling: str | None = None,
                rows: slice | None = None, n_init: int | None = None
                ) -> tuple[FittedRegimeModel, float]:
    """Full-sample (or *rows*) fit with a scaler from the same rows; canonical state order."""
    pre = regime.preprocessing
    train = x if rows is None else x[rows]
    started = time.perf_counter()
    scaler = fit_scaler(train, features, method=scaling or pre.scaling, clip=pre.clip)
    model = fit_regime_model(family, scaler.transform(train), k, regime, seed=seed,
                             n_init=n_init)
    fitted = FittedRegimeModel(family, model, scaler)
    vol = (list(features).index(regime.baselines.volatility_feature)
           if regime.baselines.volatility_feature in features else 0)
    means_raw, _ = fitted.raw_parameters()
    return fitted.permuted(canonical_order(means_raw, vol)), time.perf_counter() - started


def single_gaussian(z_train: np.ndarray, reg: float) -> GMMModel:
    """The K = 1 baseline: one Gaussian over the complete training rows."""
    x = z_train[np.isfinite(z_train).all(axis=1)]
    resp = np.ones((x.shape[0], 1))
    weights, means, cov = m_step(x, resp, covariance_type="full", reg_covar=reg)
    model = GMMModel(weights, GaussianStates(means, cov, "full"), "full", reg,
                     n_observations=int(x.shape[0]), n_iter=1, converged=True)
    model.log_likelihood = float(log_densities(x, model.states)[:, 0].sum())
    return model


def bucket_labels(values: np.ndarray, k: int, *, edges: np.ndarray | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Quantile buckets 0..k-1 of *values* (-1 where missing); edges from *values* unless given."""
    v = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(v)
    if edges is None:
        edges = np.quantile(v[finite], np.linspace(0, 1, k + 1)[1:-1]) if finite.any() else \
            np.zeros(k - 1)
    labels = np.where(finite, np.searchsorted(edges, v, side="right"), -1)
    return labels.astype(np.int64), np.asarray(edges)


def bucket_mixture(z_train: np.ndarray, labels_train: np.ndarray, k: int, reg: float) -> GMMModel:
    """A GMM whose components are the volatility buckets (hard assignment, then Gaussians)."""
    ok = np.isfinite(z_train).all(axis=1) & (labels_train >= 0)
    x, lab = z_train[ok], labels_train[ok]
    resp = np.zeros((x.shape[0], k))
    resp[np.arange(x.shape[0]), lab] = 1.0
    weights, means, cov = m_step(x, resp, covariance_type="full", reg_covar=reg)
    model = GMMModel(weights, GaussianStates(means, cov, "full"), "full", reg,
                     n_observations=int(x.shape[0]), n_iter=1, converged=True)
    joint = log_densities(x, model.states) + np.log(model.weights)[None, :]
    model.log_likelihood = float(logsumexp(joint, axis=1).sum())
    return model


def random_labels(n: int, k: int, *, seed: int, transition: np.ndarray | None = None,
                  frequencies: np.ndarray | None = None) -> np.ndarray:
    """Data-independent states: a Markov chain with *transition*, else i.i.d. *frequencies*.

    The chain is simulated run by run - a geometric stay with parameter
    ``1 - A_ii``, then a jump drawn from row ``i`` without its diagonal - which
    is the same process as bar-by-bar sampling, in one step per run.
    """
    rng = np.random.default_rng(seed)
    if transition is not None:
        a = np.asarray(transition, dtype=np.float64)
        stay = np.clip(np.diag(a), 0.0, 1.0 - 1e-12)
        jumps = a.copy()
        np.fill_diagonal(jumps, 0.0)
        totals = jumps.sum(axis=1, keepdims=True)
        jumps = np.where(totals > 0, jumps / np.where(totals > 0, totals, 1.0), 1.0 / max(k - 1,
                                                                                          1))
        np.fill_diagonal(jumps, 0.0 if k > 1 else 1.0)
        out = np.empty(n, dtype=np.int64)
        s = int(rng.integers(k))
        t = 0
        while t < n:
            length = int(rng.geometric(1.0 - stay[s]))
            out[t:t + length] = s
            t += length
            s = int(rng.choice(k, p=jumps[s])) if k > 1 else s
        return out
    p = np.full(k, 1.0 / k) if frequencies is None else np.asarray(frequencies) / np.sum(
        frequencies)
    return rng.choice(k, size=n, p=p).astype(np.int64)


# ---------------------------------------------------------------------------
# Model-selection rows
# ---------------------------------------------------------------------------
def _labels_and_probs(fitted: FittedRegimeModel, x: np.ndarray, regime: RegimeConfig, *,
                      smoothed: bool) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Hard labels (and probabilities) of a fit over *x*; HMM offline: smoothed (labelled)."""
    policy = missing_policy(regime)
    result = fitted.infer(x, policy, exact=False)
    if smoothed and isinstance(fitted.model, HMMModel):
        probs = fitted.smoothed(x, policy)
        labels = np.argmax(probs, axis=1)
        return labels, probs, result.scored
    return result.state, result.probs, result.scored


def offline_model_row(fitted: FittedRegimeModel, x: np.ndarray, regime: RegimeConfig, *,
                      timeframe: str, seconds: float) -> tuple[dict[str, Any], np.ndarray]:
    """BIC/AIC, likelihood, silhouette, balance, separation, durations and degeneracy."""
    k = fitted.n_states
    labels, _, scored = _labels_and_probs(fitted, x, regime, smoothed=True)
    labels = np.where(scored, labels, -1)
    z = fitted.scaler.transform(x)
    complete = np.isfinite(z).all(axis=1)
    counts = np.bincount(labels[labels >= 0], minlength=k).astype(np.float64)
    share = counts / max(counts.sum(), 1.0)
    means_raw, covs_raw = fitted.raw_parameters()
    separation = state_separation(means_raw, covs_raw) if covs_raw is not None else None
    model = fitted.model
    flags = degenerate_flags(
        share, covariances=None if isinstance(model, KMeansModel) else model.states.covariances,
        separation=separation, transition=fitted.transition(),
        min_state_fraction=regime.diagnostics.min_state_fraction,
        max_state_fraction=regime.diagnostics.max_state_fraction,
        max_condition_number=regime.diagnostics.max_condition_number,
        duplicate_bhattacharyya=regime.diagnostics.duplicate_bhattacharyya,
        min_expected_duration=regime.diagnostics.min_expected_duration,
        max_expected_duration=regime.diagnostics.max_expected_duration)
    runs = run_lengths(labels)
    durations = duration_summary(runs, k)
    ll = fitted.train_log_likelihood()
    n_obs = getattr(model, "n_observations", int(complete.sum()))
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy_balance = float(-(share[share > 0] * np.log(share[share > 0])).sum() / np.log(k))
    row: dict[str, Any] = {
        "timeframe": timeframe, "model": fitted.family, "states": k,
        "label": OFFLINE_LABEL, "observations": int(n_obs), "fit_seconds": seconds,
        "log_likelihood": ll, "ll_per_obs": (ll / n_obs) if ll is not None else None,
        **fitted.information_criteria(),
        "silhouette": silhouette_estimate(z, labels, sample=regime.offline.silhouette_sample),
        "min_state_share": float(share.min()), "max_state_share": float(share.max()),
        "occupancy_entropy": entropy_balance,
        "median_run_bars": (num(runs.filter(~pl.col("censored"))["length"].median())
                            if runs.height else None),
        "switch_rate": float(np.mean(labels[1:] != labels[:-1])) if labels.size > 1 else None,
        "n_iter": getattr(model, "n_iter", None), "converged": getattr(model, "converged", None),
        "min_bhattacharyya": flags.get("min_bhattacharyya"),
        "max_condition_number": flags.get("max_condition_number"),
        "degenerate": flags["degenerate"], "flags": flags["flags"],
        "state_shares": [float(v) for v in share],
    }
    if isinstance(model, KMeansModel):
        row["inertia"] = model.inertia
    if isinstance(model, HMMModel):
        row["expected_durations"] = [float(v) for v in model.expected_durations()]
        row["median_expected_duration"] = float(np.median(model.expected_durations()))
        row["self_transition_min"] = float(np.diag(model.transition).min())
    if not durations.is_empty() and "median_duration" in durations.columns:
        row["median_state_durations"] = durations["median_duration"].to_list()
    return row, labels


def holdout_scores(x: np.ndarray, times: np.ndarray, features: tuple[str, ...],
                   regime: RegimeConfig, *, family: str, k: int, split: np.datetime64,
                   seed: int) -> dict[str, Any]:
    """Fit on bars before *split*, score the complete bars after it (per observation).

    GMM: :math:`\\ln p(x_t)`; HMM: the forward filter's :math:`\\ln p(x_t \\mid x_{<t})`
    run through the training bars first; K = 1 and the volatility-bucket mixture
    are scored the same way as a GMM. K-Means has no likelihood.
    """
    cut = int(np.searchsorted(times, split, side="left"))
    fitted, seconds = offline_fit(x, features, regime, family=family, k=k, seed=seed,
                                  rows=slice(0, cut), n_init=1)
    policy = missing_policy(regime)
    test_complete = np.isfinite(x[cut:]).all(axis=1)
    out: dict[str, Any] = {"model": family, "states": k, "train_bars": cut,
                           "test_bars": int(x.shape[0] - cut), "fit_seconds": seconds}
    if isinstance(fitted.model, KMeansModel):
        result = fitted.infer(x[cut:], policy)
        out["test_mean_sq_distance"] = float(np.nanmean(-result.log_score[result.complete]))
        return out
    if isinstance(fitted.model, HMMModel):
        result = fitted.infer(x, policy, exact=False).tail(cut)
    else:
        result = fitted.infer(x[cut:], policy)
    score = result.log_score[test_complete & np.isfinite(result.log_score)]
    out["test_ll_per_obs"] = float(score.mean()) if score.size else None
    out["test_complete_bars"] = int(score.size)
    train_ll = fitted.train_log_likelihood()
    out["train_ll_per_obs"] = (train_ll / fitted.model.n_observations
                               if train_ll is not None else None)
    return out


def baseline_holdout(x: np.ndarray, times: np.ndarray, features: tuple[str, ...],
                     regime: RegimeConfig, *, k: int, split: np.datetime64) -> dict[str, Any]:
    """Hold-out likelihood of K = 1 and of the K-bucket volatility mixture."""
    cut = int(np.searchsorted(times, split, side="left"))
    pre = regime.preprocessing
    scaler = fit_scaler(x[:cut], features, method=pre.scaling, clip=pre.clip)
    z_train, z_test = scaler.transform(x[:cut]), scaler.transform(x[cut:])
    test = z_test[np.isfinite(z_test).all(axis=1)]
    reg = regime.models.gmm.reg_covar
    one = single_gaussian(z_train, reg)
    vol = list(features).index(regime.baselines.volatility_feature)
    labels, _ = bucket_labels(x[:cut, vol], k)
    buckets = bucket_mixture(z_train, labels, k, reg)
    joint = log_densities(test, buckets.states) + np.log(buckets.weights)[None, :]
    return {"states": k,
            "single_state_test_ll_per_obs": float(log_densities(test, one.states)[:, 0].mean()),
            "single_state_train_ll_per_obs": one.log_likelihood / one.n_observations,
            "volatility_buckets_test_ll_per_obs": float(logsumexp(joint, axis=1).mean()),
            "volatility_buckets_train_ll_per_obs": buckets.log_likelihood
            / buckets.n_observations}


# ---------------------------------------------------------------------------
# Scaling comparison (Step 7)
# ---------------------------------------------------------------------------
def scaling_comparison(x: np.ndarray, features: tuple[str, ...], regime: RegimeConfig, *,
                       family: str, k: int, seed: int) -> list[dict[str, Any]]:
    """Robust vs standard scaling: stability, outlier sensitivity, numerical behaviour.

    Stability is the agreement (ARI) between fits on the first and the second
    half of the history, both applied to the whole history; outlier
    sensitivity is the smallest state's share and the share of clipped values;
    numerics are the worst covariance condition number and EM convergence.
    Nothing is chosen by an outcome.
    """
    rows = []
    half = x.shape[0] // 2
    policy = missing_policy(regime)
    for method in regime.preprocessing.compare_scaling:
        first, s1 = offline_fit(x, features, regime, family=family, k=k, seed=seed,
                                scaling=method, rows=slice(0, half), n_init=1)
        second, s2 = offline_fit(x, features, regime, family=family, k=k, seed=seed,
                                 scaling=method, rows=slice(half, None), n_init=1)
        a = first.infer(x, policy, exact=False).state
        b = second.infer(x, policy, exact=False).state
        both = (a >= 0) & (b >= 0)
        unclipped = fit_scaler(x, features, method=method, clip=None).transform(x, clip=False)
        clipped = clip_fractions(unclipped, regime.preprocessing.clip)
        shares = np.bincount(a[a >= 0], minlength=k) / max(int((a >= 0).sum()), 1)
        conditions = []
        for fit in (first, second):
            if not isinstance(fit.model, KMeansModel):
                for c in fit.model.states.covariances:
                    eig = np.linalg.eigvalsh(c)
                    conditions.append(float(eig.max() / eig.min()))
        rows.append({
            "scaling": method, "model": family, "states": k,
            "half_to_half_ari": adjusted_rand(a[both], b[both]),
            "smallest_state_share": float(shares.min()),
            "max_clipped_share": float(np.nanmax(clipped)) if clipped.size else None,
            "clipped_share_by_feature": dict(zip(features, [float(c) for c in clipped],
                                                 strict=True)),
            "max_abs_scaled": float(np.nanmax(np.abs(unclipped))),
            "max_condition_number": max(conditions) if conditions else None,
            "converged": bool(getattr(first.model, "converged", True)
                              and getattr(second.model, "converged", True)),
            "n_iter": [getattr(first.model, "n_iter", None), getattr(second.model, "n_iter", None)],
            "fit_seconds": s1 + s2,
        })
    return rows


# ---------------------------------------------------------------------------
# Describing states
# ---------------------------------------------------------------------------
def state_profiles(frame: pl.DataFrame, labels: np.ndarray, *, quantiles: tuple[float, ...],
                   columns: tuple[str, ...] = PROFILE_COLUMNS, label: str) -> pl.DataFrame:
    """Median and quantiles of every profile column by state (Step 21)."""
    cols = [c for c in columns if c in frame.columns]
    data = frame.select(cols).with_columns(pl.Series("state", labels)).filter(
        pl.col("state") >= 0).with_columns(pl.col(pl.Float64).fill_nan(None))
    exprs: list[pl.Expr] = [pl.len().alias("bars")]
    for c in cols:
        exprs += [pl.col(c).mean().alias(f"{c}__mean")]
        exprs += [pl.col(c).quantile(q).alias(f"{c}__p{int(round(q * 100))}") for q in quantiles]
    wide = data.group_by("state").agg(exprs).sort("state")
    rows = []
    for record in wide.iter_rows(named=True):
        for c in cols:
            rows.append({"state": record["state"], "bars": record["bars"], "feature": c,
                         "mean": record[f"{c}__mean"],
                         **{f"p{int(round(q * 100))}": record[f"{c}__p{int(round(q * 100))}"]
                            for q in quantiles}})
    return pl.DataFrame(rows, infer_schema_length=None).with_columns(pl.lit(label).alias("basis"))


def standardized_means(x: np.ndarray, labels: np.ndarray,
                       features: tuple[str, ...]) -> pl.DataFrame:
    """State medians of each feature in units of its overall IQR around the median (Step 62)."""
    rows = []
    for j, name in enumerate(features):
        v = x[:, j]
        ok = np.isfinite(v) & (labels >= 0)
        if not ok.any():
            continue
        q25, q50, q75 = np.quantile(v[ok], [0.25, 0.5, 0.75])
        scale = (q75 - q25) or 1.0
        for k in np.unique(labels[ok]):
            sel = ok & (labels == k)
            rows.append({"feature": name, "state": int(k),
                         "standardized_median": float((np.median(v[sel]) - q50) / scale),
                         "standardized_mean": float((np.mean(v[sel]) - q50) / scale)})
    return pl.DataFrame(rows, infer_schema_length=None)


_WORDS = {
    "log_rv_20": ("lower volatility", "higher volatility"),
    "rv_percentile": ("volatility low for its recent history", "volatility high for its recent "
                      "history"),
    "spread_percentile": ("narrower spread than recently", "wider spread than recently"),
    "activity_percentile": ("fewer quotes than recently", "more quotes than recently"),
    "r_squared": ("less linear path", "more linear path"),
    "slope_over_volatility": ("downward drift", "upward drift"),
    "return_z_20": ("recent decline", "recent rise"),
    "wavelet_fast_slow_log_ratio": ("energy at slower scales", "energy at faster scales"),
    "wavelet_entropy": ("concentrated wavelet energy", "spread-out wavelet energy"),
    "spectral_entropy": ("concentrated spectrum", "flatter spectrum"),
    "abs_return_acf1": ("weaker volatility clustering", "stronger volatility clustering"),
    "log_ou_half_life": ("faster residual reversion", "slower residual reversion"),
    "fft_high_low_log_ratio": ("more low-frequency power", "more high-frequency power"),
}


def describe_states(standardized: pl.DataFrame, *, threshold: float = 0.5,
                    top: int = 4) -> dict[int, str]:
    """Factual one-line descriptions from standardised medians (no semantic names)."""
    out: dict[int, str] = {}
    if standardized.is_empty():
        return out
    for (state,), part in standardized.group_by("state", maintain_order=True):
        strong = part.filter(pl.col("standardized_median").abs() >= threshold).sort(
            pl.col("standardized_median").abs(), descending=True).head(top)
        words = []
        for r in strong.iter_rows(named=True):
            low, high = _WORDS.get(r["feature"], (f"lower {r['feature']}", f"higher {r['feature']}"))
            words.append(f"{high if r['standardized_median'] > 0 else low} "
                         f"({r['standardized_median']:+.1f} IQR)")
        out[int(state)] = ", ".join(words) if words else "close to the overall median everywhere"
    return dict(sorted(out.items()))


# ---------------------------------------------------------------------------
# Frequencies and agreement
# ---------------------------------------------------------------------------
def frequency_table(timestamps: pl.Series, labels: np.ndarray, *, by: str, k: int) -> pl.DataFrame:
    """Share of bars in each state per year / quarter / month (Step 39)."""
    t = pl.col("timestamp")
    key = {"year": t.dt.year().cast(pl.Utf8),
           "quarter": t.dt.year().cast(pl.Utf8) + "Q" + t.dt.quarter().cast(pl.Utf8),
           "month": t.dt.strftime("%Y-%m")}[by]
    frame = pl.DataFrame({"timestamp": timestamps, "state": labels}).filter(pl.col("state") >= 0)
    table = (frame.with_columns(key.alias("period")).group_by("period", "state").len()
             .pivot(on="state", index="period", values="len").sort("period").fill_null(0))
    names = [str(j) for j in range(k)]
    for name in names:
        if name not in table.columns:
            table = table.with_columns(pl.lit(0).alias(name))
    total = pl.sum_horizontal([pl.col(n) for n in names])
    return table.select("period", total.alias("bars"),
                        *[(pl.col(n) / total).alias(f"state_{n}_share") for n in names])


def intraday_frequency(timestamps: pl.Series, labels: np.ndarray, research: ResearchConfig, *,
                       k: int) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """State shares by hour, session and weekday, and Cramer's V of each (Step 38)."""
    frame = pl.DataFrame({"timestamp": timestamps, "state": labels}).filter(pl.col("state") >= 0)
    frame = frame.with_columns(
        pl.col("timestamp").dt.hour().cast(pl.Int32).cast(pl.Utf8).str.zfill(2).alias("hour"),
        assign_sessions(pl.col("timestamp"), research).alias("session"),
        pl.col("timestamp").dt.weekday().cast(pl.Utf8).alias("weekday"))
    tables = []
    stats = []
    for grouping in ("hour", "session", "weekday"):
        counts = frame.group_by(grouping, "state").len()
        wide = counts.pivot(on="state", index=grouping, values="len").fill_null(0).sort(grouping)
        names = [str(j) for j in range(k) if str(j) in wide.columns]
        total = pl.sum_horizontal([pl.col(n) for n in names])
        tables.append(wide.select(pl.lit(grouping).alias("grouping"),
                                  pl.col(grouping).cast(pl.Utf8).alias("group"),
                                  total.alias("bars"),
                                  *[(pl.col(n) / total).alias(f"state_{n}_share")
                                    for n in names]))
        matrix = wide.select(names).to_numpy()
        stats.append({"grouping": grouping, "cramers_v": cramers_v(matrix)})
    return pl.concat(tables, how="diagonal_relaxed"), stats


def agreement(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Label-invariant agreement of two labelings, plus the best one-to-one match rate."""
    ok = (np.asarray(a) >= 0) & (np.asarray(b) >= 0)
    if ok.sum() < 10:
        return {"rows": int(ok.sum())}
    table = contingency(np.asarray(a)[ok], np.asarray(b)[ok])
    rows, cols = linear_sum_assignment(-table)
    return {"rows": int(ok.sum()), "ari": adjusted_rand(np.asarray(a)[ok], np.asarray(b)[ok]),
            "nmi": normalized_mutual_info(np.asarray(a)[ok], np.asarray(b)[ok]),
            "cramers_v": cramers_v(table),
            "matched_agreement": float(table[rows, cols].sum() / table.sum())}
