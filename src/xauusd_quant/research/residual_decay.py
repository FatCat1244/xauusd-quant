r"""Does the residual decay toward zero?

Three separate pieces of evidence:

1. **The reversion coefficient** — regress
   :math:`\Delta\epsilon_{t+1} = a + b\,\epsilon_t + \eta_t`.
   :math:`b < 0` is consistent with local mean reversion. It is **not** an OU
   estimate and **not** a half-life; Prompt #4 handles that properly.
2. **Conditional decay paths** — group by the Z-score at ``t`` and track
   :math:`E[\epsilon_{t+h}]`, :math:`E[Z_{t+h}]` and :math:`E[Z_{t+h} - Z_t]`.
3. **Movement toward zero** —
   :math:`P(|\epsilon_{t+h}| < |\epsilon_t| \mid Z_t \in \text{bin})`.

Everything forward-looking here is an **outcome**. Future residuals never enter
a feature, a threshold or a conditioning variable; the bin a bar falls into
depends only on its own Z-score at ``t``.

On standard errors
------------------
Consecutive residuals come from windows sharing ``N-1`` bars, so the
observations are strongly dependent and plain OLS standard errors are far too
small. Newey-West errors are reported beside them, and even those should be
read as indicative rather than exact. :math:`b` is also biased toward negative
values purely by construction — see :func:`reversion_coefficient` — which is
why a random-walk control is computed alongside.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import polars as pl

from ..features.config import RegressionConfig
from ..features.rolling_regression import rolling_ols

__all__ = [
    "DecayAnalysis",
    "ReversionCoefficient",
    "bootstrap_proportion_ci",
    "conditional_decay",
    "move_toward_zero",
    "reversion_coefficient",
]


# ---------------------------------------------------------------------------
# Reversion coefficient
# ---------------------------------------------------------------------------
@dataclass
class ReversionCoefficient:
    r"""OLS of :math:`\Delta\epsilon_{t+1}` on :math:`\epsilon_t`."""

    timeframe: str
    window: int
    segment: str = "all"
    observations: int = 0
    coefficient: float = float("nan")
    intercept: float = float("nan")
    standard_error: float = float("nan")
    t_statistic: float = float("nan")
    newey_west_se: float = float("nan")
    newey_west_t: float = float("nan")
    r_squared: float = float("nan")
    control_coefficient: float = float("nan")
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _newey_west_se(x: np.ndarray, residuals: np.ndarray, lags: int) -> float:
    """Newey-West standard error of the slope in a simple regression.

    Bartlett-weighted HAC variance of the score ``x_centred * residual``. It
    collapses to roughly the OLS standard error when the score is close to
    white, and inflates it when the score is serially correlated -- both
    behaviours are pinned down in ``tests/test_residual_decay.py``.
    """
    n = x.size
    x_centred = x - x.mean()
    sxx = float(x_centred @ x_centred)
    if sxx <= 0:
        return float("nan")

    u = x_centred * residuals
    variance = float(u @ u)
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        variance += 2.0 * weight * float(u[lag:] @ u[:-lag])
    if variance < 0:
        return float("nan")
    return float(np.sqrt(variance) / sxx)


def reversion_coefficient(
    residual: np.ndarray,
    *,
    timeframe: str,
    window: int,
    config: RegressionConfig,
    segment: str = "all",
    with_control: bool = True,
) -> ReversionCoefficient:
    r"""Estimate :math:`b` in :math:`\Delta\epsilon_{t+1} = a + b\epsilon_t + \eta_t`.

    A negative :math:`b` means large residuals tend to be followed by movement
    back toward zero. Two caveats travel with it:

    * Any bounded, zero-mean series gives :math:`b < 0` mechanically, so a
      random-walk control detrended the same way is fitted too. The difference
      between the two is the informative quantity, not :math:`b` itself.
    * The OLS t-statistic assumes independent errors, so a Newey-West standard
      error is reported beside it. Measured on this data the two come out close
      (the score autocorrelation of the one-step regression is only about 0.02
      at lag 1), which is a property of the sample rather than a guarantee --
      the overlap bites much harder in the forward-horizon statistics than it
      does here. Either t-statistic is evidence about a coefficient, never
      about profitability.
    """
    values = np.asarray(residual, dtype=np.float64)
    values = values[np.isfinite(values)]
    result = ReversionCoefficient(timeframe=timeframe, window=window, segment=segment)
    if values.size < config.decay.min_observations:
        result.notes.append(
            f"Only {values.size} residuals, below the configured minimum of "
            f"{config.decay.min_observations}; not estimated."
        )
        return result

    b, a, se, t_stat, r2, nw_se, n = _fit_reversion(values, config)
    result.observations = n
    result.coefficient = b
    result.intercept = a
    result.standard_error = se
    result.t_statistic = t_stat
    result.r_squared = r2
    result.newey_west_se = nw_se
    result.newey_west_t = b / nw_se if nw_se and np.isfinite(nw_se) and nw_se > 0 else float("nan")

    if with_control:
        control = _control_residual(values.size + window, window)
        if control.size >= config.decay.min_observations:
            result.control_coefficient = _fit_reversion(control, config)[0]

    result.notes += [
        "b < 0 is consistent with the residual decaying toward zero. It is NOT an "
        "OU speed and NOT a half-life.",
        "It also does not say the PRICE reverts. eps = y - yhat and yhat is "
        "refitted every bar, so the residual can collapse to zero purely because "
        "the trend line rotates to meet a price that has not moved. Measured on "
        "this data at h=20, 77-106% of the decay toward zero comes from the line "
        "moving rather than the price, and the price component is sometimes "
        "negative. Decompose before reading b as a price effect.",
        "control_coefficient is the same regression on a detrended random walk. Any "
        "bounded zero-mean series gives b < 0, so compare against it.",
        "newey_west_t relaxes the independent-errors assumption behind "
        "t_statistic. On this data the two are usually close; that is measured, "
        "not assumed. Neither is evidence of profitability.",
    ]
    return result


def _fit_reversion(
    values: np.ndarray, config: RegressionConfig
) -> tuple[float, float, float, float, float, float, int]:
    """Shared OLS core, returned as a plain tuple."""
    x = values[:-1]
    dy = np.diff(values)
    n = x.size
    x_mean, y_mean = x.mean(), dy.mean()
    x_centred, y_centred = x - x_mean, dy - y_mean
    sxx = float(x_centred @ x_centred)
    if sxx <= config.numerical.epsilon:
        return (float("nan"),) * 6 + (n,)

    b = float(x_centred @ y_centred) / sxx
    a = float(y_mean - b * x_mean)
    fitted = a + b * x
    resid = dy - fitted
    sse = float(resid @ resid)
    sst = float(y_centred @ y_centred)
    sigma2 = sse / (n - 2) if n > 2 else float("nan")
    se = float(np.sqrt(sigma2 / sxx)) if np.isfinite(sigma2) and sigma2 >= 0 else float("nan")
    t_stat = b / se if se and np.isfinite(se) and se > 0 else float("nan")
    r2 = float(1.0 - sse / sst) if sst > config.numerical.epsilon else float("nan")

    lags = config.decay.newey_west_lags
    if lags == "auto":
        lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    nw_se = _newey_west_se(x, resid, int(lags))
    return b, a, se, t_stat, r2, nw_se, n


def _control_residual(n: int, window: int, *, seed: int = 20260924) -> np.ndarray:
    """Residual of the same rolling regression applied to a random walk."""
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.normal(0.0, 0.001, n)) + np.log(400.0)
    residual = rolling_ols(walk, window).residual
    return residual[np.isfinite(residual)]


# ---------------------------------------------------------------------------
# Conditional decay
# ---------------------------------------------------------------------------
@dataclass
class DecayAnalysis:
    """Forward residual and Z paths, grouped by the starting Z bin."""

    timeframe: str
    window: int
    table: pl.DataFrame = field(default_factory=pl.DataFrame)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeframe": self.timeframe, "window": self.window,
            "rows": self.table.height, "notes": self.notes, "warnings": self.warnings,
        }


def assign_z_bins(frame: pl.DataFrame, *, column: str, config: RegressionConfig) -> pl.DataFrame:
    """Label each row with its Z bin. Uses only the value at ``t``."""
    edges = list(config.extremes.zscore_bins)
    labels = config.extremes.bin_labels()
    expr = pl.lit(None, dtype=pl.Utf8)
    for (lo, hi), label in zip(zip(edges[:-1], edges[1:], strict=True), labels, strict=True):
        expr = (
            pl.when((pl.col(column) >= lo) & (pl.col(column) < hi))
            .then(pl.lit(label))
            .otherwise(expr)
        )
    return frame.with_columns(expr.alias("z_bin"))


def conditional_decay(
    frame: pl.DataFrame,
    *,
    timeframe: str,
    window: int,
    config: RegressionConfig,
    zscore_column: str = "residual_zscore_fit",
) -> DecayAnalysis:
    r"""Average forward residual and Z path for each starting Z bin.

    Reports :math:`E[\epsilon_{t+h}]`, :math:`E[Z_{t+h}]` and
    :math:`E[Z_{t+h} - Z_t]` together with the probability that the residual
    moved toward zero.
    """
    result = DecayAnalysis(timeframe=timeframe, window=window)
    if zscore_column not in frame.columns:
        result.warnings.append(f"No {zscore_column} column; decay analysis skipped.")
        return result

    usable = frame.filter(
        pl.col("residual").is_not_null() & pl.col(zscore_column).is_not_null()
    )
    if usable.height < 100:
        result.warnings.append(f"Only {usable.height} usable rows; skipped.")
        return result

    residual = usable["residual"].to_numpy().astype(np.float64)
    zscore = usable[zscore_column].to_numpy().astype(np.float64)
    binned = assign_z_bins(usable, column=zscore_column, config=config)
    bins = binned["z_bin"].to_numpy()

    horizons = config.extremes.forward_horizons
    boot = config.extremes.bootstrap
    rows: list[dict[str, Any]] = []

    for label in config.extremes.bin_labels():
        mask = bins == label
        starts = np.flatnonzero(mask)
        if starts.size == 0:
            continue
        for horizon in horizons:
            valid = starts[starts + horizon < residual.size]
            if valid.size == 0:
                continue
            eps_now, eps_future = residual[valid], residual[valid + horizon]
            z_now, z_future = zscore[valid], zscore[valid + horizon]
            ok = np.isfinite(eps_future) & np.isfinite(z_future)
            if ok.sum() == 0:
                continue
            eps_now, eps_future = eps_now[ok], eps_future[ok]
            z_now, z_future = z_now[ok], z_future[ok]

            toward_zero = np.abs(eps_future) < np.abs(eps_now)
            n = int(toward_zero.size)
            lo = hi = None
            if boot.enabled:
                lo, hi = bootstrap_proportion_ci(
                    toward_zero, iterations=boot.iterations,
                    confidence_level=boot.confidence_level,
                    seed=boot.seed + horizon, max_samples=boot.max_samples,
                )
            rows.append({
                "z_bin": label,
                "horizon": horizon,
                "observations": n,
                "mean_residual_now": float(np.mean(eps_now)),
                "mean_residual_future": float(np.mean(eps_future)),
                "mean_z_now": float(np.mean(z_now)),
                "mean_z_future": float(np.mean(z_future)),
                "mean_z_change": float(np.mean(z_future - z_now)),
                "median_z_future": float(np.median(z_future)),
                "prob_move_toward_zero": float(np.mean(toward_zero)),
                "move_toward_zero_ci_lower": lo,
                "move_toward_zero_ci_upper": hi,
                "thin_sample": n < config.extremes.min_samples_warning,
            })

    if not rows:
        result.warnings.append("No bin/horizon combination had usable observations.")
        return result

    result.table = pl.DataFrame(rows, infer_schema_length=None).sort("z_bin", "horizon")
    thin = result.table.filter(pl.col("thin_sample"))
    if thin.height:
        result.warnings.append(
            f"{thin.height} bin/horizon cells have fewer than "
            f"{config.extremes.min_samples_warning} observations and are flagged."
        )
    result.notes += [
        "Forward residuals are OUTCOMES. The bin depends only on Z at t.",
        "prob_move_toward_zero = P(|eps_{t+h}| < |eps_t|). A value near 0.5 means no "
        "tendency either way; the baseline is not zero.",
        "Overlapping forward windows make consecutive observations dependent, so the "
        "bootstrap intervals are narrower than the truth.",
        "No transaction cost is applied anywhere. Nothing here is a trading signal.",
    ]
    return result


def move_toward_zero(
    frame: pl.DataFrame,
    *,
    config: RegressionConfig,
    zscore_column: str = "residual_zscore_fit",
    group_column: str | None = None,
    horizons: tuple[int, ...] | None = None,
) -> pl.DataFrame:
    r"""P(residual moved toward zero) for extreme starts, optionally per group.

    Used for the volatility, slope, :math:`R^2`, hour and year conditioning
    studies: pass the grouping column and every cell is computed the same way.
    """
    threshold = config.conditioning.extreme_abs_z
    horizons = horizons or config.extremes.forward_horizons
    usable = frame.filter(
        pl.col("residual").is_not_null() & pl.col(zscore_column).is_not_null()
    )
    if usable.is_empty():
        return pl.DataFrame()

    residual = usable["residual"].to_numpy().astype(np.float64)
    zscore = usable[zscore_column].to_numpy().astype(np.float64)
    groups = (
        usable[group_column].to_numpy()
        if group_column and group_column in usable.columns
        else np.full(usable.height, "all", dtype=object)
    )

    extreme = np.abs(zscore) > threshold
    rows: list[dict[str, Any]] = []
    for group in sorted({g for g in groups.tolist() if g is not None}, key=str):
        for side, side_mask in (
            ("negative", zscore < -threshold),
            ("positive", zscore > threshold),
            ("either", extreme),
        ):
            starts = np.flatnonzero((groups == group) & side_mask)
            if starts.size == 0:
                continue
            for horizon in horizons:
                valid = starts[starts + horizon < residual.size]
                if valid.size == 0:
                    continue
                now, future = residual[valid], residual[valid + horizon]
                ok = np.isfinite(future)
                if ok.sum() == 0:
                    continue
                toward = np.abs(future[ok]) < np.abs(now[ok])
                rows.append({
                    "group": str(group),
                    "side": side,
                    "horizon": horizon,
                    "observations": int(toward.size),
                    "prob_move_toward_zero": float(np.mean(toward)),
                    "mean_residual_change": float(
                        np.mean(np.abs(future[ok]) - np.abs(now[ok]))
                    ),
                    "thin_sample": int(toward.size) < config.extremes.min_samples_warning,
                })
    if not rows:
        return pl.DataFrame()
    frame_out = pl.DataFrame(rows, infer_schema_length=None)
    if group_column:
        frame_out = frame_out.rename({"group": group_column})
    return frame_out.sort(frame_out.columns[0], "side", "horizon")


def bootstrap_proportion_ci(
    flags: np.ndarray,
    *,
    iterations: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 20260924,
    max_samples: int = 200_000,
) -> tuple[float | None, float | None]:
    """Percentile bootstrap interval for a proportion.

    For a 0/1 vector the IID bootstrap mean is *exactly* ``Binomial(n, p)/n``,
    so the resampling loop is replaced by direct binomial draws. That is an
    identity, not an approximation, and it is several thousand times faster on
    the sample sizes this project produces -- which is what lets the interval
    be computed for every cell instead of being skipped on the large ones.

    Anything that is not a 0/1 vector falls back to ordinary resampling, and
    ``max_samples`` caps that path alone.

    Returns ``(None, None)`` when the sample is too small to bootstrap, so a
    caller can say "not computed" rather than print a fabricated interval.

    The resampling is IID, which the overlapping forward windows violate: the
    interval is therefore narrower than the truth in either form.
    """
    values = np.asarray(flags, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = values.size
    if n < 30:
        return None, None

    alpha = (1.0 - confidence_level) / 2.0
    rng = np.random.default_rng(seed)

    if np.all((values == 0.0) | (values == 1.0)):
        means = rng.binomial(n, float(values.mean()), iterations) / n
    else:
        if n > max_samples:
            return None, None
        means = np.empty(iterations, dtype=np.float64)
        for i in range(iterations):
            means[i] = values[rng.integers(0, n, n)].mean()

    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))
