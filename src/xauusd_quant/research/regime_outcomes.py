r"""What happens after a state, and what a state adds (Steps 31-37, 58-63).

Everything here reads **causal** walk-forward states (the forward filter of a
model fitted on earlier data) against **later** outcomes. Outcomes are never
features; every table sets the regime labels beside two references with the
same number of groups - volatility buckets (the baseline) and random labels
(chance) - so "the states differ" is always read against "volatility differs"
and "any partition differs by this much".

Descriptive vs forward (Step 63). A state defined partly by volatility has
high volatility *by construction*: the profiles (``state_profiles``) are
descriptions, not findings. Findings are forward relations - the next bars'
return, residual, volatility change - and increments over the continuous
features in chronological out-of-sample models.

``incremental_information`` fits, on chronological folds with an embargo:

====================  ============================================================
A                     volatility (ln RV over 5..1024 bars, percentile, |r| ACF,
                      mean |eta|), hour of day, regression, OU, FFT, wavelet, spread
                      and activity - every continuous feature we have
A + soft              A + the filtered state probabilities
A + hard              A + one-hot most-likely state
A + volatility        A + one-hot volatility buckets (edges from the training fold)
A + random            A + one-hot random states (a Markov chain with the model's
                      transition matrix, independent of the data): the chance increment
====================  ============================================================

Research models only: fixed ridge penalty, standardised on each training fold.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy.special import expit

from ..regimes.transitions import transition_calibration
from .ou_estimation import num
from .regime_analysis import bucket_labels
from .spectral_ic import sample_rows, target_columns
from .spectral_nulls import SourceData
from .wavelet_predictiveness import (
    _binary_metrics,
    _design,
    _select,
    chronological_folds,
    fit_linear_models,
    fit_logistic,
    trailing_mean,
)

__all__ = [
    "INCREMENTAL_MODELS",
    "calibration_tables",
    "conditional_outcomes",
    "entropy_analysis",
    "future_volatility_change",
    "incremental_information",
    "incremental_summary",
    "mean_reversion_by_state",
    "mean_reversion_increment",
    "ou_by_state",
    "state_medians",
]

#: Nested incremental models: name -> blocks.
BASE_BLOCKS: tuple[str, ...] = ("volatility", "time", "regression", "ou", "fourier", "wavelet",
                                "microstructure")
INCREMENTAL_MODELS: dict[str, tuple[str, ...]] = {
    "A": BASE_BLOCKS,
    "A+soft": (*BASE_BLOCKS, "regime_soft"),
    "A+hard": (*BASE_BLOCKS, "regime_hard"),
    "A+volatility": (*BASE_BLOCKS, "volatility_buckets"),
    "A+random": (*BASE_BLOCKS, "random_labels"),
}
_ORDER: tuple[str, ...] = (*BASE_BLOCKS, "regime_soft", "regime_hard", "volatility_buckets",
                           "random_labels")


# ---------------------------------------------------------------------------
# Forward outcomes by state (Steps 32-33)
# ---------------------------------------------------------------------------
def _forward(values: np.ndarray, h: int) -> np.ndarray:
    out = np.full(values.size, np.nan)
    if h < values.size:
        out[:-h] = values[h:]
    return out


def _span_return(returns: np.ndarray, h: int) -> np.ndarray:
    """r_{t+1} + ... + r_{t+h} (NaN past the end or across a missing return)."""
    r = np.asarray(returns, dtype=np.float64)
    finite = np.isfinite(r)
    total = np.concatenate(([0.0], np.cumsum(np.where(finite, r, 0.0))))
    bad = np.concatenate(([0], np.cumsum(~finite, dtype=np.int64)))
    n = r.size
    out = np.full(n, np.nan)
    idx = np.arange(n - h)
    ok = (bad[idx + h + 1] - bad[idx + 1]) == 0
    out[idx] = np.where(ok, total[idx + h + 1] - total[idx + 1], np.nan)
    return out


def _group_stats(values: np.ndarray, labels: np.ndarray, k: int) -> list[dict[str, Any]]:
    rows = []
    for s in range(k):
        v = values[(labels == s) & np.isfinite(values)]
        rows.append({"group": s, "observations": int(v.size),
                     "mean": float(v.mean()) if v.size else None,
                     "variance": float(v.var(ddof=1)) if v.size > 1 else None,
                     "std_error": float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else None,
                     "median": float(np.median(v)) if v.size else None})
    return rows


def conditional_outcomes(inputs: pl.DataFrame, labelings: dict[str, np.ndarray], *, k: int,
                         horizons: tuple[int, ...]) -> pl.DataFrame:
    r""":math:`E[r_{t+h}\mid S_t]`, :math:`Var(r_{t+h}\mid S_t)` and
    :math:`E[|\epsilon_{t+h}| - |\epsilon_t| \mid S_t]` for every labeling."""
    r = inputs["log_return"].to_numpy()
    eps = np.abs(inputs["regression_residual"].to_numpy())
    rows: list[dict[str, Any]] = []
    for h in horizons:
        future_r = _span_return(r, h)
        change = _forward(eps, h) - eps
        for name, labels in labelings.items():
            for outcome, values in (("future_return", future_r),
                                    ("abs_residual_change", change)):
                for row in _group_stats(values, labels, k):
                    rows.append({"labeling": name, "outcome": outcome, "horizon": h, **row})
    return pl.DataFrame(rows, infer_schema_length=None)


def mean_reversion_by_state(inputs: pl.DataFrame, labelings: dict[str, np.ndarray], *, k: int,
                            horizons: tuple[int, ...], extreme: float) -> pl.DataFrame:
    r""":math:`P(|\epsilon_{t+h}| < |\epsilon_t| \mid |Z_t| > z, S_t = k)` (Step 33), with SEs.

    The same probability within each volatility tercile is added for the
    regime labeling (``within_volatility``), so a state effect can be told
    apart from a volatility effect.
    """
    eps = np.abs(inputs["regression_residual"].to_numpy())
    z = np.abs(inputs["residual_zscore"].to_numpy())
    vol = inputs["log_rv_20"].to_numpy()
    tercile, _ = bucket_labels(vol, 3)
    rows: list[dict[str, Any]] = []
    extreme_rows = np.isfinite(z) & (z > extreme) & np.isfinite(eps)
    for h in horizons:
        later = _forward(eps, h)
        shrink = np.where(np.isfinite(later), (later < eps).astype(np.float64), np.nan)
        for name, labels in labelings.items():
            for s in range(k):
                sel = extreme_rows & (labels == s) & np.isfinite(shrink)
                n = int(sel.sum())
                p = float(shrink[sel].mean()) if n else None
                rows.append({"labeling": name, "horizon": h, "group": s, "within": "all",
                             "observations": n, "prob_shrinks": p,
                             "std_error": float(np.sqrt(p * (1 - p) / n)) if n and p is not None
                             else None})
                if name != "regime":
                    continue
                for v in range(3):
                    sel_v = sel & (tercile == v)
                    n_v = int(sel_v.sum())
                    p_v = float(shrink[sel_v].mean()) if n_v else None
                    rows.append({"labeling": name, "horizon": h, "group": s,
                                 "within": f"volatility_tercile_{v}", "observations": n_v,
                                 "prob_shrinks": p_v,
                                 "std_error": float(np.sqrt(p_v * (1 - p_v) / n_v))
                                 if n_v and p_v is not None else None})
    return pl.DataFrame(rows, infer_schema_length=None)


def ou_by_state(inputs: pl.DataFrame, labels: np.ndarray, *, k: int, extreme: float,
                horizons: tuple[int, ...] = (5, 10)) -> pl.DataFrame:
    """OU parameters, validity, realised decay and OU forecast error by state (Step 34).

    Realised decay: median of :math:`|\\epsilon_{t+h}|/|\\epsilon_t|` after
    :math:`|Z| > z`, beside the fit's implied :math:`e^{-\\theta h}`. Forecast
    error: mean squared error of the OU forecast :math:`\\epsilon_t e^{-\\theta h}`
    relative to "no change" (below 1 = the fit helps).
    """
    eps = inputs["regression_residual"].to_numpy()
    z = np.abs(inputs["residual_zscore"].to_numpy())
    theta = inputs["ou_theta"].to_numpy()
    half_life = inputs["ou_half_life_bars"].to_numpy()
    valid = inputs["ou_valid"].to_numpy() > 0
    rows = []
    for s in range(k):
        sel = labels == s
        row: dict[str, Any] = {"state": s, "bars": int(sel.sum()),
                               "valid_ou_share": float(valid[sel].mean()) if sel.any() else None}
        th = theta[sel & np.isfinite(theta)]
        hl = half_life[sel & np.isfinite(half_life)]
        row.update({"median_theta": float(np.median(th)) if th.size else None,
                    "theta_p25": float(np.quantile(th, 0.25)) if th.size else None,
                    "theta_p75": float(np.quantile(th, 0.75)) if th.size else None,
                    "median_half_life_bars": float(np.median(hl)) if hl.size else None})
        for h in horizons:
            later = _forward(eps, h)
            ext = sel & np.isfinite(z) & (z > extreme) & np.isfinite(later) & np.isfinite(theta)
            if ext.sum() >= 20:
                ratio = np.abs(later[ext]) / np.abs(eps[ext])
                implied = np.exp(-theta[ext] * h)
                forecast = eps[ext] * implied
                mse_ou = float(np.mean((later[ext] - forecast) ** 2))
                mse_flat = float(np.mean((later[ext] - eps[ext]) ** 2))
                row.update({f"realised_decay_h{h}": float(np.median(ratio)),
                            f"implied_decay_h{h}": float(np.median(implied)),
                            f"ou_mse_ratio_h{h}": mse_ou / mse_flat if mse_flat > 0 else None,
                            f"extremes_h{h}": int(ext.sum())})
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def state_medians(inputs: pl.DataFrame, labels: np.ndarray, *, k: int,
                  columns: tuple[str, ...], lag_persistence: dict[str, int] | None = None
                  ) -> pl.DataFrame:
    """Median, p25, p75 of *columns* by state; optional same-value persistence at a lag.

    ``lag_persistence`` maps a column to a lag (the FFT dominant period at
    ``t`` equal to its value ``N`` bars earlier: dominant-period persistence).
    """
    rows = []
    for s in range(k):
        sel = labels == s
        row: dict[str, Any] = {"state": s, "bars": int(sel.sum())}
        for c in columns:
            if c not in inputs.columns:
                continue
            v = inputs[c].cast(pl.Float64).fill_null(np.nan).to_numpy()[sel]
            v = v[np.isfinite(v)]
            row[f"{c}__median"] = float(np.median(v)) if v.size else None
            row[f"{c}__p25"] = float(np.quantile(v, 0.25)) if v.size else None
            row[f"{c}__p75"] = float(np.quantile(v, 0.75)) if v.size else None
        for c, lag in (lag_persistence or {}).items():
            if c not in inputs.columns:
                continue
            v = inputs[c].cast(pl.Float64).fill_null(np.nan).to_numpy()
            before = np.full(v.size, np.nan)
            before[lag:] = v[:-lag]
            ok = sel & np.isfinite(v) & np.isfinite(before)
            row[f"{c}__same_as_lag_{lag}"] = float(np.mean(v[ok] == before[ok])) if ok.any() \
                else None
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def future_volatility_change(inputs: pl.DataFrame, labelings: dict[str, np.ndarray], *, k: int,
                             horizons: tuple[int, ...]) -> pl.DataFrame:
    r"""Forward volatility beyond the input (Step 63): :math:`\ln RV_{t+1..t+h} - \ln RV^{20}_t`.

    Also within volatility deciles (``within_decile_mean``: the state's mean
    minus the decile's mean, averaged over deciles), so a state is credited
    only with what it says beyond the current volatility.
    """
    r = inputs["log_return"].to_numpy()
    now = inputs["log_rv_20"].to_numpy()
    decile, _ = bucket_labels(now, 10)
    rows: list[dict[str, Any]] = []
    for h in horizons:
        future_sq = _span_return(r ** 2, h) / h
        with np.errstate(divide="ignore", invalid="ignore"):
            future = np.where(future_sq > 0, 0.5 * np.log(future_sq), np.nan)
        change = future - now
        for name, labels in labelings.items():
            for row in _group_stats(change, labels, k):
                s = row["group"]
                diffs, weights = [], []
                for d in range(10):
                    in_d = (decile == d) & np.isfinite(change)
                    sel = in_d & (labels == s)
                    if sel.sum() >= 20 and in_d.sum() > sel.sum():
                        diffs.append(change[sel].mean() - change[in_d].mean())
                        weights.append(sel.sum())
                row["within_decile_mean"] = (float(np.average(diffs, weights=weights))
                                             if diffs else None)
                rows.append({"labeling": name, "horizon": h, **row})
    return pl.DataFrame(rows, infer_schema_length=None)


# ---------------------------------------------------------------------------
# Calibration and uncertainty (Steps 31, 60, 61)
# ---------------------------------------------------------------------------
def calibration_tables(frame: pl.DataFrame, family: str, k: int, *, bins: int
                       ) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Predicted P(leave) vs the realised next-bar change of the filtered state, and the
    next-state probability of the realised next state (multiclass Brier score)."""
    state = frame[f"{family}_state"].to_numpy()
    changed = np.full(state.size, np.nan)
    changed[:-1] = (state[1:] != state[:-1]).astype(np.float64)
    tables = []
    summary: dict[str, Any] = {}
    if "leave_probability" in frame.columns:
        leave = frame["leave_probability"].to_numpy()
        table = transition_calibration(leave, changed, bins=bins)
        tables.append(table.with_columns(pl.lit("leave_probability").alias("forecast")))
        ok = np.isfinite(leave) & np.isfinite(changed)
        if ok.any():
            summary["leave_brier"] = float(np.mean((leave[ok] - changed[ok]) ** 2))
            summary["leave_mean_predicted"] = float(leave[ok].mean())
            summary["leave_observed_rate"] = float(changed[ok].mean())
            design = np.column_stack([np.ones(ok.sum()),
                                      np.log(np.clip(leave[ok], 1e-9, 1 - 1e-9)
                                             / (1 - np.clip(leave[ok], 1e-9, 1 - 1e-9)))])
            try:
                summary["leave_calibration_slope"] = float(fit_logistic(design, changed[ok],
                                                                        ridge=0.0)[1])
            except np.linalg.LinAlgError:
                summary["leave_calibration_slope"] = None
            if not table.is_empty():
                summary["leave_max_reliability_gap"] = num(
                    (table["mean_predicted"] - table["observed_frequency"]).abs().max())
    nxt_cols = [f"next_state_p{j}" for j in range(k)]
    if all(c in frame.columns for c in nxt_cols):
        nxt = frame.select(nxt_cols).to_numpy()
        target = np.zeros_like(nxt)
        ok = np.zeros(state.size, dtype=bool)
        ok[:-1] = state[1:] >= 0
        idx = np.flatnonzero(ok)
        target[idx, state[idx + 1]] = 1.0
        summary["next_state_brier"] = float(np.mean(((nxt[idx] - target[idx]) ** 2).sum(axis=1)))
        summary["next_state_accuracy"] = float(np.mean(np.argmax(nxt[idx], axis=1)
                                                       == state[idx + 1]))
    conf_col = f"{family}_state_confidence"
    if conf_col in frame.columns:
        conf = frame[conf_col].to_numpy()
        conf = conf[np.isfinite(conf)]
        if conf.size:
            summary["confidence_quantiles"] = {f"p{q}": float(np.quantile(conf, q / 100))
                                               for q in (5, 25, 50, 75, 95)}
            summary["share_confidence_above_0.99"] = float(np.mean(conf > 0.99))
    table_all = pl.concat(tables, how="diagonal_relaxed") if tables else pl.DataFrame()
    return table_all, summary


def entropy_analysis(frame: pl.DataFrame, inputs: pl.DataFrame, family: str, *,
                     quantile: float = 0.9) -> pl.DataFrame:
    """High-entropy bars vs the rest: near a switch? unusual volatility? model misfit?"""
    ent_col = f"{family}_entropy"
    if ent_col not in frame.columns:
        return pl.DataFrame()
    entropy = frame[ent_col].to_numpy()
    state = frame[f"{family}_state"].to_numpy()
    switch = np.zeros(state.size, dtype=bool)
    switch[1:] = state[1:] != state[:-1]
    idx = np.flatnonzero(switch)
    positions = np.arange(state.size)
    if idx.size:
        nxt = np.searchsorted(idx, positions)
        after = np.where(nxt < idx.size, idx[np.minimum(nxt, idx.size - 1)] - positions, np.inf)
        before = np.where(nxt > 0, positions - idx[np.maximum(nxt - 1, 0)], np.inf)
        distance = np.minimum(after, before)
    else:
        distance = np.full(state.size, np.inf)
    vol = inputs["log_rv_20"].to_numpy()
    score = frame["log_score"].to_numpy()
    complete = frame["complete"].to_numpy()
    finite = np.isfinite(entropy)
    edge = np.quantile(entropy[finite], quantile) if finite.any() else np.nan
    rows = []
    for name, sel in (("high_entropy", finite & (entropy >= edge)),
                      ("other", finite & (entropy < edge))):
        vol_rank = np.full(vol.size, np.nan)
        vf = np.isfinite(vol)
        vol_rank[vf] = (np.argsort(np.argsort(vol[vf])) + 0.5) / vf.sum()
        ok_score = sel & complete & np.isfinite(score)
        rows.append({"group": name, "bars": int(sel.sum()),
                     "entropy_threshold": float(edge),
                     "median_bars_to_nearest_switch": float(np.median(distance[sel]))
                     if sel.any() else None,
                     "share_within_5_bars_of_switch": float(np.mean(distance[sel] <= 5))
                     if sel.any() else None,
                     "median_volatility_rank": float(np.nanmedian(vol_rank[sel]))
                     if sel.any() else None,
                     "mean_log_score": float(score[ok_score].mean()) if ok_score.any() else None})
    return pl.DataFrame(rows, infer_schema_length=None)


# ---------------------------------------------------------------------------
# Incremental information (Steps 58-59)
# ---------------------------------------------------------------------------
def _baseline_blocks(inputs: pl.DataFrame, rows: np.ndarray, *, windows: tuple[int, ...],
                     hours: bool) -> dict[str, dict[str, np.ndarray]]:
    def col(name: str) -> np.ndarray:
        return inputs[name].cast(pl.Float64).fill_null(np.nan).to_numpy()

    r = col("log_return")
    eta = np.abs(col("ou_innovation"))
    with np.errstate(divide="ignore", invalid="ignore"):
        volatility = {f"log_rv_{w}": 0.5 * np.log(trailing_mean(r ** 2, w)[rows])
                      for w in windows}
        volatility["rv_percentile"] = col("rv_percentile")[rows]
        volatility["abs_return_acf1"] = col("abs_return_acf1")[rows]
        for w in (20, 256):
            volatility[f"log_mean_abs_innovation_{w}"] = np.log(trailing_mean(eta, w)[rows])
    for block in volatility.values():
        block[~np.isfinite(block)] = np.nan
    time: dict[str, np.ndarray] = {}
    if hours:
        hour = inputs["timestamp"].gather(pl.Series(rows)).dt.hour().to_numpy()
        time = {f"hour_{h:02d}": (hour == h).astype(np.float64) for h in range(1, 24)}
    rz = col("residual_zscore")[rows]
    oz = col("ou_zscore")[rows]
    return {
        "volatility": volatility, "time": time,
        "regression": {"residual_zscore": rz, "abs_residual_zscore": np.abs(rz),
                       "slope_over_volatility": col("slope_over_volatility")[rows],
                       "r_squared": col("r_squared")[rows],
                       "return_z_20": col("return_z_20")[rows]},
        "ou": {"ou_zscore": oz, "abs_ou_zscore": np.abs(oz),
               "log_ou_half_life": col("log_ou_half_life")[rows],
               "ou_valid": col("ou_valid")[rows]},
        "fourier": {"spectral_entropy": col("spectral_entropy")[rows],
                    "fft_high_low_log_ratio": col("fft_high_low_log_ratio")[rows],
                    "fft_top3_power_share": col("fft_top3_power_share")[rows]},
        "wavelet": {"wavelet_entropy": col("wavelet_entropy")[rows],
                    "wavelet_fast_slow_log_ratio": col("wavelet_fast_slow_log_ratio")[rows],
                    "wavelet_top3_scale_share": col("wavelet_top3_scale_share")[rows]},
        "microstructure": {"spread_percentile": col("spread_percentile")[rows],
                           "activity_percentile": col("activity_percentile")[rows]},
    }


def _one_hot(labels: np.ndarray, k: int, prefix: str) -> dict[str, np.ndarray]:
    return {f"{prefix}_{j}": np.where(labels >= 0, (labels == j).astype(np.float64), np.nan)
            for j in range(k)}


def _binary_residual(source: SourceData, rows: np.ndarray, horizons: tuple[int, ...],
                     extreme: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    n = source.size
    values = source.columns["regression_residual"]
    z = source.columns["residual_zscore"]
    out = {}
    scope = np.abs(np.nan_to_num(z[rows])) > extreme
    for h in horizons:
        ahead = rows + h
        inside = ahead < n
        later = np.where(inside, values[np.where(inside, ahead, 0)], np.nan)
        now = values[rows]
        with np.errstate(invalid="ignore"):
            y = np.where(np.isfinite(later) & np.isfinite(now),
                         (np.abs(later) < np.abs(now)).astype(np.float64), np.nan)
        out[f"residual_shrinks_h{h}"] = (y, scope & np.isfinite(y))
    return out


def incremental_information(inputs: pl.DataFrame, regimes: pl.DataFrame, source: SourceData, *,
                            family: str, k: int, transition: np.ndarray | None, cfg: Any,
                            seed: int, volatility_feature: str = "log_rv_20"
                            ) -> dict[str, pl.DataFrame]:
    """Models A, A+soft, A+hard, A+volatility, A+random on chronological folds."""
    stamps = inputs["timestamp"]
    joined = pl.DataFrame({"timestamp": stamps}).with_row_index("__row").join(
        regimes, on="timestamp", how="inner")
    covered = joined["__row"].to_numpy().astype(np.int64)
    mask = np.zeros(inputs.height, dtype=bool)
    mask[covered] = True
    rows = sample_rows(mask, cfg.max_rows)
    pos = np.searchsorted(covered, rows)
    state = joined[f"{family}_state"].to_numpy()[pos]
    blocks = _baseline_blocks(inputs, rows, windows=cfg.volatility_windows, hours=cfg.hours)
    prob_cols = [f"{family}_p{j}" for j in range(k)]
    if all(c in joined.columns for c in prob_cols):
        blocks["regime_soft"] = {c: joined[c].cast(pl.Float64).to_numpy()[pos] for c in prob_cols}
    else:                                      # K-Means has no probabilities
        blocks["regime_soft"] = _one_hot(state, k, "state")
    blocks["regime_hard"] = _one_hot(state, k, "state")
    random = np.random.default_rng(seed)
    if transition is not None:
        from .regime_analysis import random_labels

        chance = random_labels(inputs.height, k, seed=int(random.integers(2**31)),
                               transition=transition)[rows]
    else:
        freq = np.bincount(state[state >= 0], minlength=k)
        chance = random.choice(k, size=rows.size, p=freq / freq.sum())
    blocks["random_labels"] = _one_hot(chance, k, "random")
    vol_now = inputs[volatility_feature].cast(pl.Float64).fill_null(np.nan).to_numpy()[rows]
    embargo = max((*cfg.horizons, 1))
    folds = chronological_folds(stamps.gather(pl.Series(rows)), rows, cfg.test_blocks, embargo)
    continuous = target_columns(source, cfg.targets, cfg.horizons, rows)
    complete = np.all([np.isfinite(v) for v in continuous.values()], axis=0)
    binary = _binary_residual(source, rows, cfg.horizons, cfg.binary_extreme)
    linear_rows: list[dict[str, Any]] = []
    binary_rows: list[dict[str, Any]] = []
    for fold, train, test in folds:
        labels, _ = bucket_labels(vol_now, k, edges=np.quantile(
            vol_now[train][np.isfinite(vol_now[train])], np.linspace(0, 1, k + 1)[1:-1]))
        blocks["volatility_buckets"] = _one_hot(labels, k, "volatility_bucket")
        train_c, test_c = train[complete[train]], test[complete[test]]
        if train_c.size >= cfg.min_train_rows and test_c.size >= 100:
            design = _design(blocks, train_c, _ORDER)
            subsets = {name: _select(design, members)
                       for name, members in INCREMENTAL_MODELS.items()}
            for record in fit_linear_models(design, continuous, train_c, test_c,
                                            ridge=cfg.ridge, subsets=subsets):
                target, horizon = record["target"].rsplit("_h", 1)
                record.update({"fold": fold, "target": target, "horizon": int(horizon)})
                linear_rows.append(record)
        for key, (y, scope) in binary.items():
            tr, te = train[scope[train]], test[scope[test]]
            if tr.size < cfg.min_train_rows // 4 or te.size < 50:
                continue
            design = _design(blocks, tr, _ORDER)
            target, horizon = key.rsplit("_h", 1)
            for name, members in INCREMENTAL_MODELS.items():
                cols = _select(design, members)
                try:
                    beta = fit_logistic(design.matrix[tr][:, cols], y[tr], ridge=cfg.ridge)
                except np.linalg.LinAlgError:
                    continue
                p = expit(design.matrix[te][:, cols] @ beta)
                binary_rows.append({"fold": fold, "target": target, "horizon": int(horizon),
                                    "model": name, "train_rows": int(tr.size),
                                    "test_rows": int(te.size), **_binary_metrics(y[te], p)})
    return {"linear": pl.DataFrame(linear_rows, infer_schema_length=None),
            "binary": pl.DataFrame(binary_rows, infer_schema_length=None)}


def incremental_summary(tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Fold-mean scores and each variant's increment over A (R^2 gain / log-loss reduction)."""
    rows: list[dict[str, Any]] = []
    linear, binary = tables.get("linear"), tables.get("binary")
    variants = [m for m in INCREMENTAL_MODELS if m != "A"]
    if isinstance(linear, pl.DataFrame) and not linear.is_empty():
        wide = linear.pivot(on="model", index=["target", "horizon", "fold"], values="oos_r2")
        for (target, horizon), part in wide.group_by("target", "horizon", maintain_order=True):
            rec: dict[str, Any] = {"target": target, "horizon": horizon, "kind": "continuous",
                                   "metric": "oos_r2", "folds": part.height,
                                   "A": num(part["A"].mean())}
            for m in variants:
                if m in part.columns:
                    delta = part[m] - part["A"]
                    rec[f"delta_{m}"] = num(delta.mean())
                    rec[f"folds_better_{m}"] = int((delta > 0).sum())
            if {"A+soft", "A+hard"} <= set(part.columns):
                rec["folds_soft_beats_hard"] = int((part["A+soft"] > part["A+hard"]).sum())
            rows.append(rec)
    if isinstance(binary, pl.DataFrame) and not binary.is_empty():
        for metric in ("log_loss", "auc"):
            wide = binary.pivot(on="model", index=["target", "horizon", "fold"], values=metric)
            sign = -1.0 if metric == "log_loss" else 1.0
            for (target, horizon), part in wide.group_by("target", "horizon",
                                                         maintain_order=True):
                if "A" not in part.columns:
                    continue
                rec = {"target": target, "horizon": horizon, "kind": "binary",
                       "metric": metric, "folds": part.height, "A": num(part["A"].mean())}
                for m in variants:
                    if m in part.columns:
                        delta = sign * (part[m] - part["A"])
                        rec[f"delta_{m}"] = num(delta.mean())
                        rec[f"folds_better_{m}"] = int((delta > 0).sum())
                if {"A+soft", "A+hard"} <= set(part.columns):
                    rec["folds_soft_beats_hard"] = int(
                        ((sign * (part["A+soft"] - part["A+hard"])) > 0).sum())
                rows.append(rec)
    return pl.DataFrame(rows, infer_schema_length=None)


def mean_reversion_increment(inputs: pl.DataFrame, frame: pl.DataFrame, source: Any,
                              model: str, k: int, regime: Any,
                              seed: int) -> dict[str, Any]:
    """REG-H-001: |Z|, Z and volatility quartiles, then + states / + random states."""

    cfg = regime.incremental
    joined = pl.DataFrame({"timestamp": inputs["timestamp"]}).with_row_index("__row").join(
        frame, on="timestamp", how="inner")
    covered = joined["__row"].to_numpy().astype(np.int64)
    mask = np.zeros(inputs.height, dtype=bool)
    mask[covered] = True
    z = inputs["residual_zscore"].to_numpy()
    mask &= np.abs(np.nan_to_num(z)) > cfg.binary_extreme
    rows = sample_rows(mask, cfg.max_rows)
    if rows.size < cfg.min_train_rows:
        return {"rows": []}
    pos = np.searchsorted(covered, rows)
    state = joined[f"{model}_state"].to_numpy()[pos]
    prob_cols = [f"{model}_p{j}" for j in range(k)]
    soft = ({c: joined[c].cast(pl.Float64).to_numpy()[pos] for c in prob_cols}
            if all(c in joined.columns for c in prob_cols) else _one_hot(state, k, "state"))
    freq = np.bincount(state[state >= 0], minlength=k)
    chance = np.random.default_rng(seed + 5).choice(k, size=rows.size, p=freq / freq.sum())
    vol = inputs[regime.baselines.volatility_feature].to_numpy()[rows]
    blocks: dict[str, dict[str, np.ndarray]] = {
        "extreme": {"residual_zscore": z[rows], "abs_residual_zscore": np.abs(z[rows])},
        "regime_soft": soft, "random_labels": _one_hot(chance, k, "random")}
    models = {"V": ("extreme", "volatility_quartiles"),
              "V+soft": ("extreme", "volatility_quartiles", "regime_soft"),
              "V+random": ("extreme", "volatility_quartiles", "random_labels")}
    order = ("extreme", "volatility_quartiles", "regime_soft", "random_labels")
    embargo = max((*cfg.horizons, 1))
    folds = chronological_folds(inputs["timestamp"].gather(pl.Series(rows)), rows,
                                cfg.test_blocks, embargo)
    binary = _binary_residual(source, rows, cfg.horizons, cfg.binary_extreme)
    out_rows = []
    for fold, train, test in folds:
        edges = np.quantile(vol[train][np.isfinite(vol[train])], [0.25, 0.5, 0.75])
        quart, _ = bucket_labels(vol, 4, edges=edges)
        blocks["volatility_quartiles"] = _one_hot(quart, 4, "volatility_quartile")
        for key, (y, scope) in binary.items():
            tr, te = train[scope[train]], test[scope[test]]
            if tr.size < cfg.min_train_rows // 4 or te.size < 50:
                continue
            design = _design(blocks, tr, order)
            horizon = int(key.rsplit("_h", 1)[1])
            for name, members in models.items():
                cols = _select(design, members)
                try:
                    beta = fit_logistic(design.matrix[tr][:, cols], y[tr], ridge=cfg.ridge)
                except np.linalg.LinAlgError:
                    continue
                p = expit(design.matrix[te][:, cols] @ beta)
                out_rows.append({"fold": fold, "horizon": horizon, "model": name,
                                 "train_rows": int(tr.size), "test_rows": int(te.size),
                                 **_binary_metrics(y[te], p)})
    table = pl.DataFrame(out_rows, infer_schema_length=None)
    summary: dict[str, Any] = {"rows": out_rows}
    if not table.is_empty():
        wide = table.pivot(on="model", index=["horizon", "fold"], values="log_loss")
        for (horizon,), part in wide.group_by("horizon", maintain_order=True):
            if not {"V", "V+soft", "V+random"} <= set(part.columns):
                continue
            gain = part["V"] - part["V+soft"]
            gain_random = part["V"] - part["V+random"]
            summary[f"h{horizon}"] = {"log_loss_reduction_soft": num(gain.mean()),
                                      "log_loss_reduction_random": num(gain_random.mean()),
                                      "folds_soft_better": int((gain > 0).sum()),
                                      "folds": part.height}
    return summary
