r"""Does estimated OU decay resemble actual residual decay?

This is the question the whole OU layer exists to answer, so it is asked in
several independent ways, all of them *outcome* analysis: the forward values
used here never feed back into an estimate, a threshold or a conditioning
variable. Every start is chosen by what is known at ``t`` (a valid fit and its
Z-score); every quantity after ``t`` is only measured.

* :func:`forecast_evaluation` - the model expectation
  :math:`E[X_{t+h}\mid X_t] = \mu_t + (X_t-\mu_t)b_t^h` against the realised
  :math:`X_{t+h}`, next to two naive forecasts: persistence (:math:`X_t`, no
  decay at all) and the equilibrium (:math:`\mu_t`, instant decay).
* :func:`extreme_events` / :func:`extreme_summary` - what happens after
  :math:`|Z_{OU}|` exceeds 1, 2, 3.
* :func:`realized_decay_comparison` - the first time the deviation from
  :math:`\mu_t` halves, against the half-life estimated at ``t``.
* :func:`first_passage_summary` - first passage to 75 / 50 / 25 % of the
  starting deviation, and to the equilibrium itself.

On reading these numbers
------------------------
Consecutive starts overlap heavily (a single excursion contributes many
starts), so effective sample sizes are far below the counts shown; the
"episode start" rows keep only the first bar of each excursion. And even a
*perfectly specified* OU does not halve its deviation in exactly one
half-life - noise makes the first passage random and, from a large start,
usually earlier. That is why the same statistics are always computed on an
exact OU simulation with the fitted parameters: the reference, not the
number 1.0, is what the realised/estimated ratio should be compared with.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from ..models.config import OUConfig
from .ou_estimation import num

__all__ = [
    "extreme_events",
    "extreme_summary",
    "first_passage_summary",
    "forecast_evaluation",
    "realized_decay_comparison",
]


def _arrays(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    def col(name: str) -> np.ndarray:
        return frame[name].fill_null(np.nan).to_numpy().astype(np.float64)

    return {
        "x": col("residual"),
        "mu": col("ou_mu"),
        "b": col("ou_b"),
        "theta": col("ou_theta"),
        "stat": col("ou_stationary_std"),
        "hl": col("ou_half_life_bars"),
        "z": col("ou_zscore"),
        "valid": frame["ou_valid"].to_numpy().astype(bool),
    }


# ---------------------------------------------------------------------------
# Forecast versus reality
# ---------------------------------------------------------------------------
def forecast_evaluation(frame: pl.DataFrame, *, source: str, config: OUConfig) -> pl.DataFrame:
    r"""MAE, RMSE and bias of the OU expectation, beside naive forecasts.

    ``skill_vs_persistence`` is :math:`1 - MSE_{OU}/MSE_{persistence}`: above 0
    the fitted decay beats assuming no decay. A detrended random walk also
    "beats" persistence - detrending pulls every series toward zero - so
    compare with the control's skill, not with 0.
    """
    arr = _arrays(frame)
    x, mu, b, stat = arr["x"], arr["mu"], arr["b"], arr["stat"]
    n = x.size
    level = config.extremes.primary_abs_z
    usable = arr["valid"] & np.isfinite(arr["z"])
    rows: list[dict[str, Any]] = []
    for h in config.expected_path.horizons:
        if h >= n:
            continue
        start = np.flatnonzero(usable[: n - h])
        future = x[start + h]
        ok = np.isfinite(future)
        start, future = start[ok], future[ok]
        now = x[start]
        expected = mu[start] + (now - mu[start]) * np.power(b[start], h)
        z = arr["z"][start]
        for subset, mask in (
            ("all", np.ones(start.size, dtype=bool)),
            (f"|Z|>{level:g}", np.abs(z) > level),
            (f"Z>{level:g}", z > level),
            (f"Z<-{level:g}", z < -level),
        ):
            if not mask.any():
                continue
            err = future[mask] - expected[mask]
            err_persist = future[mask] - now[mask]
            err_mu = future[mask] - mu[start][mask]
            mse = float(np.mean(err ** 2))
            mse_p = float(np.mean(err_persist ** 2))
            rows.append({
                "source": source, "horizon": h, "subset": subset,
                "observations": int(mask.sum()),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(mse)),
                "bias": float(np.mean(err)),
                "rmse_in_stationary_std": float(np.sqrt(np.mean((err / stat[start][mask]) ** 2))),
                "mae_persistence": float(np.mean(np.abs(err_persist))),
                "rmse_persistence": float(np.sqrt(mse_p)),
                "mae_equilibrium": float(np.mean(np.abs(err_mu))),
                "rmse_equilibrium": float(np.sqrt(np.mean(err_mu ** 2))),
                "skill_vs_persistence": 1.0 - mse / mse_p if mse_p > 0 else None,
                "skill_vs_equilibrium": (
                    1.0 - mse / float(np.mean(err_mu ** 2)) if np.any(err_mu) else None),
            })
    return pl.DataFrame(rows, infer_schema_length=None)


# ---------------------------------------------------------------------------
# Paths after extreme deviations
# ---------------------------------------------------------------------------
def extreme_events(frame: pl.DataFrame, *, config: OUConfig) -> pl.DataFrame:
    r"""One row per start with :math:`|Z_{OU,t}|` above the lowest level.

    Starts need a valid fit at ``t`` and a full ``max_horizon`` of future
    bars, so the end of the sample is never mistaken for censoring. From each
    start, with :math:`D_0 = |X_t-\mu_t|` and :math:`\mu_t` held fixed:

    * ``fp_<f>`` - first ``h`` with :math:`|X_{t+h}-\mu_t| \le f D_0`
      (0 = not within the horizon: censored);
    * ``cross`` - first ``h`` on the other side of (or on) :math:`\mu_t`;
    * ``max_adverse_z`` - largest further move away from :math:`\mu_t` within
      ``adverse_horizon``, in units of the stationary std at ``t``;
    * ``remaining_h<h>`` - :math:`(X_{t+h}-\mu_t)/(X_t-\mu_t)`: 1 = no
      progress, 0 = at equilibrium, negative = overshot to the other side;
    * ``toward_h<h>`` - whether :math:`|X_{t+h}-\mu_t| < D_0`.

    A missing future value never counts as a hit. The loop visits only
    events still waiting for an answer, so cost falls quickly with ``h``.
    """
    cfg = config.extremes
    arr = _arrays(frame)
    x, mu, stat, z = arr["x"], arr["mu"], arr["stat"], arr["z"]
    n = x.size
    horizon = cfg.max_horizon
    lowest = min(cfg.abs_z_levels)
    abs_z = np.abs(z)
    candidates = arr["valid"] & np.isfinite(z) & (abs_z > lowest)
    starts = np.flatnonzero(candidates[: max(n - horizon, 0)])
    if starts.size == 0:
        return pl.DataFrame()

    side = np.sign(x[starts] - mu[starts])
    d0 = np.abs(x[starts] - mu[starts])
    fractions = cfg.first_passage_fractions
    hits = {f: np.zeros(starts.size, dtype=np.int32) for f in fractions}
    cross = np.zeros(starts.size, dtype=np.int32)
    adverse = np.zeros(starts.size)
    forward = sorted(set(cfg.forward_horizons))
    remaining = {h: np.full(starts.size, np.nan) for h in forward}
    toward = {h: np.zeros(starts.size, dtype=bool) for h in forward}
    must_run = max(cfg.adverse_horizon, max(forward))
    active = np.arange(starts.size)
    for h in range(1, horizon + 1):
        if active.size == 0:
            break
        idx = starts[active] + h
        value = x[idx]
        dev = (value - mu[starts[active]]) * side[active]   # > 0: same side as the start
        absdev = np.abs(value - mu[starts[active]])
        finite = np.isfinite(value)
        for f in fractions:
            pending = hits[f][active] == 0
            hit = pending & finite & (absdev <= f * d0[active])
            hits[f][active[hit]] = h
        pending = cross[active] == 0
        crossed = pending & finite & (dev <= 0)
        cross[active[crossed]] = h
        if h <= cfg.adverse_horizon:
            further = np.where(finite, dev - d0[active], 0.0)
            adverse[active] = np.maximum(adverse[active], further)
        if h in remaining:
            with np.errstate(invalid="ignore", divide="ignore"):
                remaining[h][active] = np.where(finite, dev / d0[active], np.nan)
            toward[h][active] = finite & (absdev < d0[active])
        if h >= must_run:
            done = cross[active] > 0
            for f in fractions:
                done &= hits[f][active] > 0
            active = active[~done]

    previous_abs_z = np.concatenate(([np.nan], abs_z[:-1]))[starts]
    columns: dict[str, Any] = {
        "index": starts,
        "abs_z": abs_z[starts],
        "previous_abs_z": previous_abs_z,
        "side": np.where(side > 0, "positive", "negative"),
        "d0": d0,
        "d0_in_stationary_std": d0 / stat[starts],
        "half_life_estimated": arr["hl"][starts],
        "theta": arr["theta"][starts],
        "b": arr["b"][starts],
        "cross": cross,
        "max_adverse_z": adverse / stat[starts],
    }
    for f in fractions:
        columns[f"fp_{int(round(f * 100))}"] = hits[f]
    for h in forward:
        columns[f"remaining_h{h}"] = remaining[h]
        columns[f"toward_h{h}"] = toward[h]
    if "timestamp" in frame.columns:
        columns["timestamp"] = frame["timestamp"].gather(starts)
    return pl.DataFrame(columns)


def _level_subsets(events: pl.DataFrame, config: OUConfig) -> list[tuple[float, str, pl.DataFrame]]:
    """(level, subset name, rows) for every level, all starts and episode starts."""
    out: list[tuple[float, str, pl.DataFrame]] = []
    for level in config.extremes.abs_z_levels:
        above = events.filter(pl.col("abs_z") > level)
        episode = above.filter(
            pl.col("previous_abs_z").is_null() | (pl.col("previous_abs_z") <= level)
        )
        out.append((level, "all starts", above))
        out.append((level, "episode starts", episode))
    return out


def extreme_summary(events: pl.DataFrame, *, source: str, config: OUConfig) -> pl.DataFrame:
    """Behaviour after |Z_OU| > each level, by side, against the model's own expectation."""
    if events.is_empty():
        return pl.DataFrame()
    cfg = config.extremes
    rows: list[dict[str, Any]] = []
    for level, subset, rows_level in _level_subsets(events, config):
        for side in ("either", "positive", "negative"):
            part = rows_level if side == "either" else rows_level.filter(pl.col("side") == side)
            if part.is_empty():
                continue
            cross = part["cross"].to_numpy()
            censored = cross == 0
            row: dict[str, Any] = {
                "source": source, "abs_z_level": level, "subset": subset, "side": side,
                "observations": part.height,
                "thin_sample": part.height < cfg.min_samples_warning,
                "prob_cross_within_horizon": float(np.mean(~censored)),
                "median_bars_to_cross": (
                    float(np.median(cross[~censored])) if censored.mean() < 0.5 else None),
                "max_adverse_z_median": num(part["max_adverse_z"].median()),
                "max_adverse_z_p90": num(part["max_adverse_z"].quantile(0.9)),
                "prob_no_adverse_move": num((part["max_adverse_z"] <= 0).mean()),
            }
            b = part["b"].to_numpy()
            for h in sorted(set(cfg.forward_horizons)):
                remaining = part[f"remaining_h{h}"].drop_nulls().drop_nans()
                row[f"prob_toward_h{h}"] = num(part[f"toward_h{h}"].mean())
                row[f"remaining_mean_h{h}"] = num(remaining.mean()) if remaining.len() else None
                row[f"remaining_median_h{h}"] = (
                    num(remaining.median()) if remaining.len() else None)
                row[f"model_remaining_mean_h{h}"] = float(np.mean(np.power(b, h)))
            rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def realized_decay_comparison(
    events: pl.DataFrame, *, source: str, config: OUConfig
) -> pl.DataFrame:
    r"""Realised half-decay time against the half-life estimated at the start.

    ``realized_half_decay_time`` is the first ``h`` with
    :math:`|X_{t+h}-\mu_t| \le 0.5\,D_0`. Censored starts (no halving within
    the horizon) are kept: they rank above every observed time in the rank
    correlation, and ratio quantiles that fall among them are reported as
    null. The Pearson correlation uses uncensored starts only and says so.

    A near-zero correlation is *expected* even for a correct OU with constant
    parameters, because then every estimated half-life differs only by
    estimation noise. Read the median ratio against the OU reference.
    """
    from scipy import stats

    if events.is_empty():
        return pl.DataFrame()
    horizon = config.extremes.max_horizon
    rows: list[dict[str, Any]] = []
    for level, subset, part in _level_subsets(events, config):
        if part.height < 3:
            continue
        realized = part["fp_50"].to_numpy().astype(np.float64)
        estimated = part["half_life_estimated"].to_numpy()
        censored = realized == 0
        ranked = np.where(censored, horizon + 1.0, realized)
        ratio = np.where(censored, np.inf, realized / estimated)
        row: dict[str, Any] = {
            "source": source, "abs_z_level": level, "subset": subset,
            "observations": part.height,
            "censored_fraction": float(censored.mean()),
            # Undefined (not zero) when either side is constant.
            "spearman_realized_vs_estimated": (
                float(stats.spearmanr(ranked, estimated).statistic)
                if np.ptp(ranked) > 0 and np.ptp(estimated) > 0 else None),
            "pearson_log_uncensored": (
                float(np.corrcoef(np.log(realized[~censored]), np.log(estimated[~censored]))[0, 1])
                if (~censored).sum() > 2 and np.ptp(realized[~censored]) > 0
                and np.ptp(estimated[~censored]) > 0 else None
            ),
            "median_estimated_half_life": float(np.median(estimated)),
            "median_realized_half_decay": (
                float(np.median(realized[~censored])) if censored.mean() < 0.5 else None),
            "fraction_realized_faster": float(np.mean(ratio < 1.0)),
        }
        for q in (0.10, 0.25, 0.50, 0.75, 0.90):
            with np.errstate(invalid="ignore"):
                value = float(np.quantile(ratio, q))
            row[f"ratio_p{int(q * 100)}"] = value if np.isfinite(value) else None
        row["median_ratio"] = row["ratio_p50"]
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def first_passage_summary(events: pl.DataFrame, *, source: str, config: OUConfig) -> pl.DataFrame:
    r"""Time to 75 / 50 / 25 % of the starting deviation, and to equilibrium.

    Beside each realised distribution is the model's *deterministic* time,
    :math:`\ln(1/f)/\theta_t`, the time the expected path takes. The expected
    path never reaches the equilibrium itself, so crossing has no model time.
    """
    if events.is_empty():
        return pl.DataFrame()
    rows: list[dict[str, Any]] = []
    theta_all = events["theta"].to_numpy()
    for level, subset, part in _level_subsets(events, config):
        if part.is_empty():
            continue
        theta = part["theta"].to_numpy()
        targets: list[tuple[str, str, float | None]] = [
            (f"{int(round(f * 100))}% of D0", f"fp_{int(round(f * 100))}", f)
            for f in config.extremes.first_passage_fractions
        ]
        targets.append(("equilibrium crossing", "cross", None))
        for label, column, fraction in targets:
            times = part[column].to_numpy().astype(np.float64)
            censored = times == 0
            observed = times[~censored]
            row: dict[str, Any] = {
                "source": source, "abs_z_level": level, "subset": subset, "target": label,
                "observations": part.height,
                "censored_fraction": float(censored.mean()),
                "mean_uncensored": float(observed.mean()) if observed.size else None,
            }
            pooled = np.where(censored, np.inf, times)
            for q in (0.25, 0.50, 0.75, 0.90):
                with np.errstate(invalid="ignore"):
                    value = float(np.quantile(pooled, q))
                row[f"p{int(q * 100)}"] = value if np.isfinite(value) else None
            if fraction is not None:
                model = np.log(1.0 / fraction) / theta
                row["model_time_median"] = float(np.median(model))
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(censored, np.inf, times / model)
                with np.errstate(invalid="ignore"):
                    median = float(np.quantile(ratio, 0.5))
                row["median_ratio_to_model"] = median if np.isfinite(median) else None
            rows.append(row)
    del theta_all
    return pl.DataFrame(rows, infer_schema_length=None)
