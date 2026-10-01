r"""Ornstein-Uhlenbeck estimation of the rolling-regression residual.

What this module produces
-------------------------
* :func:`ou_parameter_frame` - the per-bar table of rolling OU estimates. Every
  column at ``t`` is a function of the residuals :math:`X_{t-M+1..t}` only
  (and the one-step innovation of :math:`X_t` given the fit at ``t-1``), so it
  is a *feature* in the sense of CLAUDE.md invariant 8.
* :func:`static_fit` / :func:`segment_fits` - whole-sample and per-calendar
  segment AR(1)/OU fits, with OLS and Newey-West errors.
* distribution tables: parameters, half-life (with censoring made explicit),
  the classification of ``b``, the equilibrium, the two Z-scores, expected
  decay.
* :func:`control_series` - the residuals the same pipeline is run on for
  comparison: a detrended random walk (no reversion at all), an exact OU with
  the whole-sample parameters (what a constant OU would look like), and
  optionally shuffled returns.

Why the controls are not optional
---------------------------------
Detrending any series - a random walk included - leaves something bounded and
zero-centred, so an AR(1) fit on a rolling-regression residual returns
``0 < b < 1`` and a finite "half-life" *mechanically*. A half-life reported
without the same number for a detrended random walk beside it cannot be read.
And a constant-parameter OU, estimated in the same small windows, shows how
much of the observed variation in the estimates is pure estimation noise and
small-window bias. Both references travel with every headline number.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl

from ..features.rolling_regression import rolling_ols
from ..models.config import OUConfig
from ..models.ornstein_uhlenbeck import (
    OU_STATES,
    AR1Fit,
    fit_ar1,
    rolling_ou,
    simulate_ou,
    simulate_random_walk,
)

__all__ = [
    "CONTROL_BUILDERS",
    "control_step_std",
    "num",
    "random_walk_features",
    "PARAMETER_COLUMNS",
    "b_classification",
    "control_series",
    "equilibrium_summary",
    "expected_decay_table",
    "half_life_summary",
    "ou_parameter_frame",
    "parameter_distribution",
    "segment_fits",
    "static_fit",
    "zscore_comparison",
]

def num(value: Any, default: float = float("nan")) -> float:
    """A Polars scalar (median, mean, quantile...) as a plain float.

    Polars types those results as a broad union; anything that is not a
    number - None, an empty-series null - becomes *default*.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


#: Columns of the per-bar OU table, in order.
PARAMETER_COLUMNS: tuple[str, ...] = (
    "timestamp", "residual", "ou_a", "ou_b", "ou_b_se", "ou_r_squared", "ou_innovation_std",
    "ou_theta", "ou_mu", "ou_sigma", "ou_stationary_std", "ou_half_life_bars",
    "ou_mr_per_bar", "ou_window_mean", "ou_window_std", "ou_std_ratio", "ou_zscore",
    "ou_innovation", "ou_innovation_standardized", "ou_state", "ou_valid",
    "ou_near_unit_root", "ou_numerical_warning", "ou_n_pairs",
)
_STATE_ENUM = pl.Enum(list(OU_STATES))


# ---------------------------------------------------------------------------
# Per-bar table
# ---------------------------------------------------------------------------
def ou_parameter_frame(
    residual: np.ndarray,
    timestamps: pl.Series | None,
    *,
    ou_window: int,
    config: OUConfig,
) -> pl.DataFrame:
    r"""Rolling OU estimates for every bar, missing values as null (never NaN).

    Polars treats NaN and null differently, and ``is_not_null()`` keeps NaN,
    so the warm-up rows would otherwise survive every downstream filter and
    drag every quantile upward - the exact failure CLAUDE.md records for the
    regression features. Every float column is therefore built with NaN as null.

    Columns are converted one at a time, each NumPy array released as soon as
    Polars holds it. At 23 years of 1-minute bars a column is ~63 MB, and
    holding the NumPy set, a Polars copy and a NaN-free copy of every column
    at once made this the memory peak of the whole study.
    """
    values = np.asarray(residual, dtype=np.float64)
    result = rolling_ou(
        values, ou_window, rules=config.validity_rules(),
        chunk_rows=config.numerical.chunk_rows,
    )
    fits, mapping = result.fits, result.mapping
    floats: dict[str, np.ndarray] = {
        "residual": values,
        "ou_a": fits.a,
        "ou_b": fits.b,
        "ou_b_se": fits.se_b,
        "ou_r_squared": fits.r_squared,
        "ou_innovation_std": np.sqrt(fits.innovation_var),
        "ou_theta": mapping.theta,
        "ou_mu": mapping.mu,
        "ou_sigma": mapping.sigma,
        "ou_stationary_std": mapping.stationary_std,
        "ou_half_life_bars": mapping.half_life_bars,
        "ou_mr_per_bar": mapping.mr_per_bar,
        "ou_window_mean": fits.window_mean,
        "ou_window_std": fits.window_std,
        "ou_std_ratio": result.std_ratio,
        "ou_zscore": result.zscore,
        "ou_innovation": result.innovation,
        "ou_innovation_standardized": result.innovation_standardized,
    }
    flags = {"ou_valid": mapping.valid, "ou_near_unit_root": mapping.near_unit_root,
             "ou_numerical_warning": mapping.numerical_warning}
    state_code = mapping.state_code
    n_pairs = np.where(fits.complete, ou_window - 1, 0).astype(np.int32)
    del result, fits, mapping           # the dicts now hold the only references

    columns: list[pl.Series] = [] if timestamps is None else [timestamps.alias("timestamp")]
    for name in list(floats):
        columns.append(pl.Series(name, floats.pop(name), nan_to_null=True))
    columns.append(pl.Series("ou_state", OU_STATES, dtype=_STATE_ENUM).gather(state_code))
    columns += [pl.Series(name, flag) for name, flag in flags.items()]
    columns.append(pl.Series("ou_n_pairs", n_pairs))
    return pl.DataFrame(columns)


# ---------------------------------------------------------------------------
# Static fits
# ---------------------------------------------------------------------------
def static_fit(residual: np.ndarray, config: OUConfig) -> AR1Fit:
    """Whole-series AR(1)/OU fit with Newey-West errors (lags from the config)."""
    values = np.asarray(residual, dtype=np.float64)
    pairs = int(np.count_nonzero(np.isfinite(values[:-1]) & np.isfinite(values[1:])))
    return fit_ar1(
        values,
        rules=config.validity_rules(),
        hac_lags=config.equilibrium.resolve_lags(pairs),
        confidence_level=config.equilibrium.confidence_level,
    )


def fit_row(fit: AR1Fit, **labels: Any) -> dict[str, Any]:
    """Flatten a static fit into one table row."""
    ou = fit.ou
    return {
        **labels,
        "pairs": fit.n_pairs,
        "a": fit.a,
        "b": fit.b,
        "b_se_ols": fit.se_b,
        "b_se_newey_west": fit.hac_se_b,
        "df_statistic": fit.df_statistic,
        "r_squared": fit.r_squared,
        "innovation_std": fit.innovation_std,
        "state": ou.state,
        "ou_valid": ou.valid,
        "theta": ou.theta,
        "mu": ou.mu,
        "mu_ci_lower": fit.mu_ci_lower,
        "mu_ci_upper": fit.mu_ci_upper,
        "sigma": ou.sigma,
        "stationary_std": ou.stationary_std,
        "half_life_bars": ou.half_life_bars,
        "mr_per_bar": ou.mr_per_bar,
        "sample_std": fit.sample_std,
        "std_ratio": fit.std_ratio,
        "near_unit_root": ou.near_unit_root,
        "numerical_warning": ou.numerical_warning,
    }


def segment_fits(
    frame: pl.DataFrame, *, by: str, config: OUConfig, min_pairs: int | None = None
) -> pl.DataFrame:
    """One static fit per calendar year or quarter, pairs kept within a segment."""
    minimum = min_pairs or config.segmented_estimation.min_observations
    if by == "year":
        label = pl.col("timestamp").dt.year().cast(pl.Utf8)
    elif by == "quarter":
        label = (pl.col("timestamp").dt.year().cast(pl.Utf8) + "Q"
                 + pl.col("timestamp").dt.quarter().cast(pl.Utf8))
    else:
        raise ValueError(f"by must be 'year' or 'quarter', got {by!r}")
    tagged = frame.select("timestamp", "residual").with_columns(label.alias("segment"))
    rows: list[dict[str, Any]] = []
    for segment in sorted(tagged["segment"].unique().drop_nulls().to_list()):
        values = (
            tagged.filter(pl.col("segment") == segment)["residual"]
            .fill_null(np.nan).to_numpy().astype(np.float64)
        )
        pairs = int(np.count_nonzero(np.isfinite(values[:-1]) & np.isfinite(values[1:])))
        if pairs < minimum:
            rows.append({"segment": segment, "pairs": pairs, "sufficient_data": False})
            continue
        fit = fit_ar1(values, rules=config.validity_rules(),
                      hac_lags=config.equilibrium.resolve_lags(pairs),
                      confidence_level=config.equilibrium.confidence_level)
        rows.append({**fit_row(fit, segment=segment), "sufficient_data": True})
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------
_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def _describe(values: np.ndarray) -> dict[str, float | int | None]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    q = np.quantile(values, _QUANTILES)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else None,
        "min": float(values.min()),
        "p5": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
        "p75": float(q[3]), "p95": float(q[4]),
        "max": float(values.max()),
    }


def parameter_distribution(frame: pl.DataFrame, *, source: str) -> pl.DataFrame:
    """Distribution of every rolling estimate, over the rows where it is defined.

    ``b``, ``r_squared`` and ``1-b`` are defined for every complete window;
    the OU parameters only for ``valid`` ones. The row count of each line says
    which population it describes.
    """
    rows: list[dict[str, Any]] = []
    for column, population in (
        ("ou_b", "complete windows"),
        ("ou_mr_per_bar", "complete windows"),
        ("ou_r_squared", "complete windows"),
        ("ou_innovation_std", "complete windows"),
        ("ou_theta", "valid OU windows"),
        ("ou_mu", "valid OU windows"),
        ("ou_sigma", "valid OU windows"),
        ("ou_stationary_std", "valid OU windows"),
        ("ou_half_life_bars", "valid OU windows"),
        ("ou_std_ratio", "valid OU windows"),
        ("ou_zscore", "valid OU windows"),
    ):
        if column not in frame.columns:
            continue
        values = frame[column].drop_nulls().to_numpy().astype(np.float64)
        rows.append({"source": source, "parameter": column.removeprefix("ou_"),
                     "population": population, **_describe(values)})
    return pl.DataFrame(rows, infer_schema_length=None)


def b_classification(frame: pl.DataFrame, *, source: str, config: OUConfig) -> dict[str, Any]:
    """Share of windows in each state, and above each near-unit-root line."""
    # Select before filtering: a filter copies every column it keeps, and the
    # full frame is ~30 columns (about 2 GB at 23 years of 1-minute bars).
    complete = frame.select("ou_n_pairs", "ou_state", "ou_b", "ou_numerical_warning").filter(
        pl.col("ou_n_pairs") > 0)
    total = complete.height
    out: dict[str, Any] = {"source": source, "windows_estimated": total}
    if total == 0:
        return out
    counts = dict(complete.group_by("ou_state").len().iter_rows())
    for state in OU_STATES:
        if state == "insufficient_data":
            continue
        out[f"fraction_{state}"] = counts.get(state, 0) / total
    b = complete["ou_b"].drop_nulls().to_numpy()
    out["valid_ou_fraction"] = counts.get("valid", 0) / total
    out["fraction_b_in_unit_interval"] = float(np.mean((b > 0) & (b < 1))) if b.size else None
    out["fraction_b_ge_1"] = float(np.mean(b >= 1)) if b.size else None
    out["fraction_b_le_0"] = float(np.mean(b <= 0)) if b.size else None
    for threshold in config.near_unit_root.thresholds:
        key = f"fraction_b_gt_{threshold:g}".replace(".", "_")
        out[key] = float(np.mean(b > threshold)) if b.size else None
    out["near_unit_root_fraction"] = float(np.mean(b > config.near_unit_root.flag_threshold)) if (
        b.size) else None
    out["numerical_warning_fraction"] = num(complete["ou_numerical_warning"].mean(), 0.0)
    return out


def half_life_summary(frame: pl.DataFrame, *, source: str, config: OUConfig) -> dict[str, Any]:
    r"""Half-life distribution with censoring stated, not hidden.

    Windows with :math:`0 < b < 1` but a half-life above the cap are
    *censored*: their half-life is only known to be large. Quantiles are taken
    over valid and censored windows together, with the censored ones ranked
    above every reportable value; a quantile that lands among them is
    returned as null with ``p.._above_cap = True`` rather than as a number.
    Mean and standard deviation are over the reportable windows only and are
    labelled so.
    """
    cap = float(config.half_life.max_reportable_bars)
    complete = frame.select("ou_n_pairs", "ou_half_life_bars", "ou_state").filter(
        pl.col("ou_n_pairs") > 0)
    reportable = complete["ou_half_life_bars"].drop_nulls().to_numpy()
    censored = int((complete["ou_state"] == "near_unit_root").sum())
    total = complete.height
    out: dict[str, Any] = {
        "source": source,
        "windows_estimated": total,
        "valid_windows": int(reportable.size),
        "valid_fraction": reportable.size / total if total else None,
        "censored_above_cap": censored,
        "cap_bars": cap,
    }
    if reportable.size:
        pooled = np.concatenate([reportable, np.full(censored, np.inf)])
        for q in config.half_life.quantiles:
            # Interpolating between two censored (+inf) values gives NaN: either
            # way the quantile lies above the cap, which is what is reported.
            with np.errstate(invalid="ignore"):
                value = float(np.quantile(pooled, q))
            key = f"p{q * 100:g}"
            out[key] = value if np.isfinite(value) else None
            out[f"{key}_above_cap"] = not np.isfinite(value)
        out["median"] = out.get("p50")
        out["mean_reportable"] = float(reportable.mean())
        out["std_reportable"] = float(reportable.std(ddof=1)) if reportable.size > 1 else None
        out["max_reportable"] = float(reportable.max())
        out["min"] = float(reportable.min())
    return out


def equilibrium_summary(
    frame: pl.DataFrame, fit: AR1Fit, *, config: OUConfig
) -> dict[str, Any]:
    r"""Is :math:`\mu` distinguishable from zero, and is it material?

    The residual is zero-mean *within each fitting window* by OLS
    construction, but the OU equilibrium of the residual series need not be.
    It is estimated, never imposed, and compared with zero in units of the
    residual's own standard deviation.
    """
    material = config.equilibrium.material_fraction_of_std
    out: dict[str, Any] = {
        "static_mu": fit.ou.mu,
        "static_mu_se_newey_west": fit.mu_se,
        "static_mu_ci_lower": fit.mu_ci_lower,
        "static_mu_ci_upper": fit.mu_ci_upper,
        "residual_std": fit.sample_std,
        "confidence_level": config.equilibrium.confidence_level,
    }
    if fit.ou.valid and fit.sample_std and np.isfinite(fit.sample_std):
        ratio = abs(fit.ou.mu) / fit.sample_std
        excludes_zero = bool(fit.mu_ci_lower > 0 or fit.mu_ci_upper < 0)
        out.update({
            "static_abs_mu_over_residual_std": ratio,
            "static_ci_excludes_zero": excludes_zero,
            "static_flag_for_investigation": bool(excludes_zero and ratio > material),
        })
    valid = frame.select("ou_valid", "ou_mu", "ou_window_std").filter(pl.col("ou_valid"))
    if valid.height:
        mu = valid["ou_mu"].to_numpy()
        scale = valid["ou_window_std"].to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            relative = np.abs(mu) / scale
        relative = relative[np.isfinite(relative)]
        out.update({
            "rolling_mu_median": float(np.median(mu)),
            "rolling_mu_p5": float(np.quantile(mu, 0.05)),
            "rolling_mu_p95": float(np.quantile(mu, 0.95)),
            "rolling_abs_mu_over_window_std_median": (
                float(np.median(relative)) if relative.size else None),
            "rolling_fraction_abs_mu_over_std_gt_material": (
                float(np.mean(relative > material)) if relative.size else None),
        })
    out["material_fraction_of_std"] = material
    return out


def zscore_comparison(frame: pl.DataFrame, *, source: str) -> pl.DataFrame:
    """The OU Z-score beside the Prompt #3 rolling residual Z-score.

    They answer different questions - distance from the *estimated OU
    equilibrium* in units of the *model's* stationary std, versus distance
    from the recent residual mean in units of the recent residual std - so
    both are kept and compared, never substituted.
    """
    columns = [c for c in ("ou_zscore", "residual_zscore_rolling") if c in frame.columns]
    rows: list[dict[str, Any]] = []
    for column in columns:
        values = frame[column].drop_nulls().to_numpy().astype(np.float64)
        row: dict[str, Any] = {"source": source, "series": column, **_describe(values)}
        if values.size:
            for level in (1, 2, 3):
                row[f"abs_gt_{level}_frequency"] = float(np.mean(np.abs(values) > level))
        rows.append(row)
    if len(columns) == 2:
        both = frame.select(columns).drop_nulls()
        if both.height > 2:
            a, b = both[columns[0]].to_numpy(), both[columns[1]].to_numpy()
            from scipy import stats

            rows.append({
                "source": source, "series": "correlation", "count": both.height,
                "pearson": float(np.corrcoef(a, b)[0, 1]),
                "spearman": float(stats.spearmanr(a, b).statistic),
                "sign_agreement": float(np.mean(np.sign(a) == np.sign(b))),
            })
    return pl.DataFrame(rows, infer_schema_length=None)


def expected_decay_table(frame: pl.DataFrame, *, source: str, config: OUConfig) -> pl.DataFrame:
    r"""Model-implied share of a deviation remaining, :math:`e^{-\theta h} = b^h`.

    Per horizon in bars, the distribution over valid windows; per multiple of
    the window's own half-life the answer is exactly :math:`0.5^k` by
    definition, which is listed with the number of bars it corresponds to.
    """
    valid = frame.select("ou_valid", "ou_theta", "ou_half_life_bars").filter(pl.col("ou_valid"))
    rows: list[dict[str, Any]] = []
    if valid.is_empty():
        return pl.DataFrame()
    theta = valid["ou_theta"].to_numpy()
    half_life = valid["ou_half_life_bars"].to_numpy()
    for h in config.expected_path.decay_bars:
        remaining = np.exp(-theta * h)
        rows.append({"source": source, "measure": f"after {h} bar(s)", "horizon_bars": float(h),
                     **{f"remaining_{k}": v for k, v in _describe(remaining).items()}})
    for k in config.expected_path.decay_half_lives:
        rows.append({
            "source": source, "measure": f"after {k:g} half-life(s)",
            "horizon_bars": float(np.median(half_life) * k),
            "remaining_count": int(half_life.size),
            "remaining_p50": 0.5 ** k,
            "remaining_mean": 0.5 ** k,
            "horizon_bars_p25": float(np.quantile(half_life, 0.25) * k),
            "horizon_bars_p75": float(np.quantile(half_life, 0.75) * k),
        })
    return pl.DataFrame(rows, infer_schema_length=None)


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
def _random_walk(
    n: int, *, regression_window: int, step_std: float, seed: int, **_: Any
) -> np.ndarray:
    """Residual of the same rolling regression on a Gaussian random walk."""
    walk = simulate_random_walk(n, step_std, x0=float(np.log(1000.0)), seed=seed)
    return rolling_ols(walk, regression_window).residual


def random_walk_features(
    n: int, *, regression_window: int, step_std: float, seed: int, volatility_window: int
) -> pl.DataFrame:
    """The random-walk control with the conditioning variables a real residual has.

    The same walk and rolling regression as ``_random_walk`` - whose residual
    this returns unchanged - plus that regression's slope and R^2 and the
    walk's log return and trailing volatility, defined exactly as
    ``rolling_regression_features`` defines them. Conditioning the control on
    them shows which dependence of the half-life on trend, fit quality or
    volatility the rolling detrending produces by itself.
    """
    walk = simulate_random_walk(n, step_std, x0=float(np.log(1000.0)), seed=seed)
    fit = rolling_ols(walk, regression_window)
    frame = pl.DataFrame({
        "residual": fit.residual, "regression_slope": fit.slope,
        "r_squared": fit.r_squared, "log_price": walk,
    }).with_columns(
        pl.col("log_price").diff().alias("log_return")
    ).with_columns(
        pl.col("log_return")
        .rolling_std(window_size=volatility_window, min_samples=volatility_window, ddof=1)
        .alias("trailing_volatility")
    ).drop("log_price")
    return frame.with_columns(pl.col(frame.columns).fill_nan(None))


def _shuffled_returns(
    n: int, *, regression_window: int, log_returns: np.ndarray, seed: int, **_: Any
) -> np.ndarray:
    """Residual of the regression on a path built from the data's own returns, shuffled.

    Matches the marginal distribution of returns exactly (fat tails
    included) while destroying every serial dependence. Registered for
    later use; off unless ``controls`` asks for it.
    """
    returns = log_returns[np.isfinite(log_returns)]
    rng = np.random.default_rng(seed)
    path = np.log(1000.0) + np.concatenate(([0.0], np.cumsum(rng.choice(returns, n - 1))))
    return rolling_ols(path, regression_window).residual


def _ou_reference(n: int, *, reference_fit: AR1Fit | None, seed: int, **_: Any) -> np.ndarray | None:
    """An exact OU with the whole-sample parameters of the real residual."""
    if reference_fit is None or not reference_fit.ou.valid:
        return None
    ou = reference_fit.ou
    return simulate_ou(ou.theta, ou.mu, ou.sigma, n, seed=seed)


#: Name -> builder. Adding a null control (block bootstrap, volatility-matched
#: walk, ...) means adding one function here; the report runs whatever is on.
CONTROL_BUILDERS: dict[str, Callable[..., np.ndarray | None]] = {
    "control_random_walk": _random_walk,
    "reference_ou": _ou_reference,
    "control_shuffled_returns": _shuffled_returns,
}


def control_series(
    name: str,
    n: int,
    *,
    regression_window: int,
    log_returns: np.ndarray,
    reference_fit: AR1Fit | None,
    seed: int,
) -> np.ndarray | None:
    """Build one comparison series of length ``n`` (NaN where undefined)."""
    return CONTROL_BUILDERS[name](
        n, regression_window=regression_window, step_std=control_step_std(log_returns),
        log_returns=log_returns, reference_fit=reference_fit, seed=seed,
    )


def control_step_std(log_returns: np.ndarray) -> float:
    """Step std of the random-walk control: the data's own bar-return std."""
    finite = log_returns[np.isfinite(log_returns)]
    return float(np.std(finite, ddof=1)) if finite.size > 1 else 1e-3
