r"""What happens after an extreme residual.

Three studies, all outcome analysis:

**Zero crossing** — after :math:`|Z_t| > \theta`, how many bars until
:math:`\epsilon` changes sign? Observations that never cross inside the search
horizon are reported as *censored* rather than dropped, because dropping them
would bias every crossing statistic toward speed.

**Persistence** — starting from :math:`|Z_t| > \theta`, does the residual reach
a *more* extreme level before returning toward zero? A process can revert on
average and still make large excursions first, and that distinction matters
far more than the average.

**Maximum adverse residual excursion (MARE)** — how much further from zero the
residual travels within a fixed horizon. This is in residual and Z units, not
currency: it is **not** trading MAE, there is no position and no PnL.

None of this is a trading rule. Zero crossing is not an exit, the persistence
levels are not stops, and no cost is applied anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from ..features.config import RegressionConfig
from .residual_decay import bootstrap_proportion_ci

__all__ = [
    "ExtremeAnalysis",
    "adverse_excursion",
    "extreme_persistence",
    "zero_crossing_analysis",
]


@dataclass
class ExtremeAnalysis:
    """Zero-crossing, persistence and excursion tables for one model."""

    timeframe: str
    window: int
    zero_crossing: pl.DataFrame = field(default_factory=pl.DataFrame)
    persistence: pl.DataFrame = field(default_factory=pl.DataFrame)
    excursion: pl.DataFrame = field(default_factory=pl.DataFrame)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "window": self.window,
            "zero_crossing_rows": self.zero_crossing.height,
            "persistence_rows": self.persistence.height,
            "excursion_rows": self.excursion.height,
            "notes": self.notes,
            "warnings": self.warnings,
        }


def _forward_matrix(values: np.ndarray, starts: np.ndarray, horizon: int) -> np.ndarray:
    """``(len(starts), horizon)`` matrix of the values following each start.

    Positions running past the end of the series are NaN, which keeps the
    final observations from silently behaving as if they had no future.
    """
    offsets = np.arange(1, horizon + 1)
    index = starts[:, None] + offsets[None, :]
    out = np.full(index.shape, np.nan, dtype=np.float64)
    inside = index < values.size
    out[inside] = values[index[inside]]
    return out


def _first_true(matrix: np.ndarray) -> np.ndarray:
    """1-based index of the first True in each row, or 0 when none."""
    any_true = matrix.any(axis=1)
    first = np.argmax(matrix, axis=1) + 1
    return np.where(any_true, first, 0)


def zero_crossing_analysis(
    frame: pl.DataFrame,
    *,
    timeframe: str,
    window: int,
    config: RegressionConfig,
    zscore_column: str = "residual_zscore_fit",
) -> pl.DataFrame:
    """Time until the residual changes sign, after each class of extreme start.

    Censoring is explicit: ``censored_count`` and ``censored_fraction`` record
    the starts that never crossed inside ``max_crossing_horizon``, and the
    median is reported as NaN when more than half of them are censored, since
    it would otherwise be unknowable.
    """
    usable = frame.filter(
        pl.col("residual").is_not_null() & pl.col(zscore_column).is_not_null()
    )
    if usable.height < 100:
        return pl.DataFrame()

    residual = usable["residual"].to_numpy().astype(np.float64)
    zscore = usable[zscore_column].to_numpy().astype(np.float64)
    horizon = config.extremes.max_crossing_horizon
    checkpoints = config.extremes.crossing_checkpoints
    threshold = config.conditioning.extreme_abs_z

    conditions = {
        f"Z<-{threshold:g}": zscore < -threshold,
        f"Z>{threshold:g}": zscore > threshold,
        f"|Z|>{threshold:g}": np.abs(zscore) > threshold,
        "|Z|>3": np.abs(zscore) > 3.0,
        f"|Z|<={threshold:g}": np.abs(zscore) <= threshold,
    }

    rows: list[dict[str, Any]] = []
    for label, mask in conditions.items():
        starts = np.flatnonzero(mask & np.isfinite(residual))
        # A start must have room for the full search horizon, otherwise it
        # would be recorded as censored purely for sitting near the end.
        starts = starts[starts + horizon < residual.size]
        if starts.size == 0:
            continue

        future = _forward_matrix(residual, starts, horizon)
        sign_now = np.sign(residual[starts])[:, None]
        crossed = np.sign(future) * sign_now < 0
        first = _first_true(np.nan_to_num(crossed, nan=0.0).astype(bool))

        censored = first == 0
        crossed_at = first[~censored].astype(np.float64)
        row: dict[str, Any] = {
            "timeframe": timeframe,
            "window": window,
            "condition": label,
            "observations": int(starts.size),
            "censored_count": int(censored.sum()),
            "censored_fraction": float(censored.mean()),
            "mean_bars_to_cross": (
                float(crossed_at.mean()) if crossed_at.size else float("nan")
            ),
            "median_bars_to_cross": (
                float(np.median(crossed_at))
                if crossed_at.size and censored.mean() < 0.5
                else float("nan")
            ),
        }
        for checkpoint in checkpoints:
            within = (first > 0) & (first <= checkpoint)
            row[f"crossed_within_{checkpoint}"] = float(within.mean())
        row["thin_sample"] = starts.size < config.extremes.min_samples_warning
        rows.append(row)

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows, infer_schema_length=None)


def extreme_persistence(
    frame: pl.DataFrame,
    *,
    timeframe: str,
    window: int,
    config: RegressionConfig,
    zscore_column: str = "residual_zscore_fit",
) -> pl.DataFrame:
    r"""Does an extreme get worse before it gets better?

    From a start with :math:`|Z_t| > \theta`, reports the probability that
    :math:`|Z|` reaches each configured level *before* falling back to
    ``persistence_revert_to``. High values mean the residual routinely
    overshoots on its way to reverting - which is exactly the behaviour that an
    "it mean-reverts" summary hides.
    """
    usable = frame.filter(pl.col(zscore_column).is_not_null())
    if usable.height < 100:
        return pl.DataFrame()

    zscore = usable[zscore_column].to_numpy().astype(np.float64)
    horizon = config.extremes.max_crossing_horizon
    revert_to = config.extremes.persistence_revert_to
    threshold = config.conditioning.extreme_abs_z

    rows: list[dict[str, Any]] = []
    for side, mask in (
        ("negative", zscore < -threshold),
        ("positive", zscore > threshold),
    ):
        starts = np.flatnonzero(mask)
        starts = starts[starts + horizon < zscore.size]
        if starts.size == 0:
            continue
        future = _forward_matrix(zscore, starts, horizon)
        sign = -1.0 if side == "negative" else 1.0
        signed = future * sign                       # positive = more extreme

        reverted = _first_true(np.nan_to_num(signed <= revert_to, nan=0.0).astype(bool))
        for level in config.extremes.persistence_levels:
            worse = _first_true(np.nan_to_num(signed >= level, nan=0.0).astype(bool))
            # "Before reverting": reached the level, and either never reverted
            # inside the horizon or reached it first.
            hit_first = (worse > 0) & ((reverted == 0) | (worse < reverted))
            lower, upper = bootstrap_proportion_ci(
                hit_first.astype(np.float64),
                iterations=config.extremes.bootstrap.iterations,
                confidence_level=config.extremes.bootstrap.confidence_level,
                seed=config.extremes.bootstrap.seed,
                max_samples=config.extremes.bootstrap.max_samples,
            ) if config.extremes.bootstrap.enabled else (None, None)
            rows.append({
                "timeframe": timeframe,
                "window": window,
                "side": side,
                "entry_abs_z": threshold,
                "reached_abs_z": level,
                "revert_to_abs_z": revert_to,
                "observations": int(starts.size),
                "prob_reached_before_reverting": float(hit_first.mean()),
                "ci_lower": lower,
                "ci_upper": upper,
                "prob_reverted_within_horizon": float((reverted > 0).mean()),
                "thin_sample": starts.size < config.extremes.min_samples_warning,
            })
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows, infer_schema_length=None)


def adverse_excursion(
    frame: pl.DataFrame,
    *,
    timeframe: str,
    window: int,
    config: RegressionConfig,
    zscore_column: str = "residual_zscore_fit",
) -> pl.DataFrame:
    r"""Maximum adverse residual excursion after an extreme start.

    MARE is how much *further from zero* the residual travelled within
    ``mare_horizon`` bars, in residual units and in Z units. It is not trading
    MAE: no position, no entry price, no currency.
    """
    usable = frame.filter(
        pl.col("residual").is_not_null() & pl.col(zscore_column).is_not_null()
    )
    if usable.height < 100:
        return pl.DataFrame()

    residual = usable["residual"].to_numpy().astype(np.float64)
    zscore = usable[zscore_column].to_numpy().astype(np.float64)
    horizon = config.extremes.mare_horizon
    threshold = config.conditioning.extreme_abs_z

    rows: list[dict[str, Any]] = []
    for side, mask in (
        ("negative", zscore < -threshold),
        ("positive", zscore > threshold),
    ):
        starts = np.flatnonzero(mask)
        starts = starts[starts + horizon < residual.size]
        if starts.size == 0:
            continue
        sign = -1.0 if side == "negative" else 1.0
        eps_future = _forward_matrix(residual, starts, horizon) * sign
        z_future = _forward_matrix(zscore, starts, horizon) * sign
        eps_now = residual[starts] * sign
        z_now = zscore[starts] * sign

        with np.errstate(invalid="ignore"):
            worst_eps = np.nanmax(eps_future, axis=1)
            worst_z = np.nanmax(z_future, axis=1)
        mare_eps = np.maximum(worst_eps - eps_now, 0.0)
        mare_z = np.maximum(worst_z - z_now, 0.0)
        finite = np.isfinite(mare_eps) & np.isfinite(mare_z)
        if finite.sum() == 0:
            continue
        mare_eps, mare_z = mare_eps[finite], mare_z[finite]

        row: dict[str, Any] = {
            "timeframe": timeframe,
            "window": window,
            "side": side,
            "entry_abs_z": threshold,
            "horizon": horizon,
            "observations": int(mare_eps.size),
            "mare_residual_mean": float(np.mean(mare_eps)),
            "mare_residual_median": float(np.median(mare_eps)),
            "mare_z_mean": float(np.mean(mare_z)),
            "mare_z_median": float(np.median(mare_z)),
            "prob_no_adverse_move": float(np.mean(mare_z <= 0)),
            "thin_sample": int(mare_eps.size) < config.extremes.min_samples_warning,
        }
        for q in (0.5, 0.75, 0.9, 0.95, 0.99):
            key = f"p{q * 100:g}".replace(".", "_")
            row[f"mare_z_{key}"] = float(np.quantile(mare_z, q))
        rows.append(row)

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows, infer_schema_length=None)


def analyse_extremes(
    frame: pl.DataFrame,
    *,
    timeframe: str,
    window: int,
    config: RegressionConfig,
    zscore_column: str = "residual_zscore_fit",
) -> ExtremeAnalysis:
    """Run all three extreme studies and collect their caveats."""
    result = ExtremeAnalysis(timeframe=timeframe, window=window)
    result.zero_crossing = zero_crossing_analysis(
        frame, timeframe=timeframe, window=window, config=config,
        zscore_column=zscore_column,
    )
    result.persistence = extreme_persistence(
        frame, timeframe=timeframe, window=window, config=config,
        zscore_column=zscore_column,
    )
    result.excursion = adverse_excursion(
        frame, timeframe=timeframe, window=window, config=config,
        zscore_column=zscore_column,
    )
    result.notes += [
        "Zero crossing is a sign change of the residual, not a trade exit.",
        "Starts without room for the full search horizon are excluded, so they are "
        "not miscounted as censored.",
        f"Observations that never crossed within {config.extremes.max_crossing_horizon} "
        "bars are reported as censored; the median is NaN when most are censored, "
        "because it is then unknowable rather than large.",
        "MARE is in residual and Z units. It is NOT trading MAE - there is no "
        "position, no entry price and no cost model anywhere in this module.",
        "Persistence levels are reference levels for measurement, not stop-losses.",
    ]
    for table, name in (
        (result.zero_crossing, "zero-crossing"),
        (result.persistence, "persistence"),
        (result.excursion, "excursion"),
    ):
        if not table.is_empty() and "thin_sample" in table.columns:
            thin = int(table["thin_sample"].sum())
            if thin:
                result.warnings.append(f"{thin} {name} row(s) are thin samples.")
    return result
