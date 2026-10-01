r"""Are the OU parameters stable through time?

With 23 years of data the question is answered in four ways, none of which
lets the result pick its own periods:

* :func:`half_life_stability` - how the rolling half-life moves from bar to
  bar and between *disjoint* windows. Adjacent windows share ``M-1`` of ``M``
  observations, so bar-to-bar changes are small by construction; comparing
  estimates ``M`` bars apart uses windows with no observation in common, which
  is the honest stability test. The OU reference supplies the dispersion that
  pure estimation noise produces under genuinely constant parameters.
* :func:`segment_table` - rolling-estimate summaries and a static fit for
  every calendar year and quarter, with flags for sign changes, large shifts,
  near-unit-root and invalid segments.
* :func:`era_table` - the full span cut into equal-length chronological
  blocks (early / middle / recent by default). Boundaries are arithmetic,
  never chosen by looking at results or at market events.
* :func:`recent_versus_history` - the most recent twelve months against all
  earlier data, with effect sizes. Distribution tests are shown for scale
  only: consecutive estimates are strongly dependent, so their p-values are
  far too small.

Pooling all 23 years into one estimate assumes a stable market structure. The
whole-sample fit is therefore reported as a reference point only.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl

from ..models.config import OUConfig
from ..models.ornstein_uhlenbeck import fit_ar1
from .ou_estimation import fit_row, num

__all__ = [
    "era_table",
    "half_life_stability",
    "recent_versus_history",
    "rolling_half_life_profile",
    "segment_table",
]


def half_life_stability(
    frame: pl.DataFrame, *, ou_window: int, source: str, config: OUConfig
) -> dict[str, Any]:
    """Bar-to-bar and disjoint-window stability of the rolling half-life."""
    hl = frame["ou_half_life_bars"].fill_null(np.nan).to_numpy().astype(np.float64)
    # Enum codes, not strings: equal exactly when the states are equal.
    state = frame["ou_state"].to_physical().to_numpy()
    complete = frame["ou_n_pairs"].to_numpy() > 0
    out: dict[str, Any] = {"source": source, "windows_estimated": int(complete.sum())}
    valid = np.isfinite(hl)
    if valid.sum() < 10:
        return out
    log_hl = np.log(np.where(valid, hl, np.nan))

    adjacent = valid[1:] & valid[:-1]
    step = np.abs(np.diff(log_hl))[adjacent]
    out.update({
        "adjacent_pairs": int(adjacent.sum()),
        "adjacent_median_abs_change_bars": float(np.median(np.abs(np.diff(hl))[adjacent])),
        "adjacent_median_abs_log_change": float(np.median(step)),
        "adjacent_p90_abs_log_change": float(np.quantile(step, 0.9)),
        "adjacent_jump_fraction": float(np.mean(step > np.log(config.stability.jump_factor))),
    })
    lag = ou_window
    if hl.size > lag:
        pair = valid[lag:] & valid[:-lag]
        if pair.sum() > 10:
            later, earlier = log_hl[lag:][pair], log_hl[:-lag][pair]
            change = np.abs(later - earlier)
            out.update({
                "disjoint_lag_bars": lag,
                "disjoint_pairs": int(pair.sum()),
                "disjoint_median_abs_log_change": float(np.median(change)),
                "disjoint_p90_abs_log_change": float(np.quantile(change, 0.9)),
                "disjoint_fraction_changed_over_factor_2": float(np.mean(change > np.log(2.0))),
                "disjoint_log_correlation": float(np.corrcoef(later, earlier)[0, 1]),
            })

    both = complete[1:] & complete[:-1]
    out["state_change_fraction"] = float(np.mean(state[1:][both] != state[:-1][both]))
    runs = _run_lengths(valid & complete)
    out["valid_run_mean_bars"] = float(np.mean(runs)) if runs.size else None
    out["valid_run_median_bars"] = float(np.median(runs)) if runs.size else None
    out["invalid_episodes"] = int(_run_lengths(complete & ~valid).size)
    values = hl[valid]
    out["cv_overall"] = float(np.std(values, ddof=1) / np.mean(values))
    window = config.stability.rolling_window
    if hl.size >= window:
        series = pl.Series(np.where(valid, hl, np.nan)).fill_nan(None)
        cv = (
            series.rolling_std(window_size=window, min_samples=window // 2)
            / series.rolling_mean(window_size=window, min_samples=window // 2)
        ).drop_nulls().drop_nans()
        out["rolling_cv_median"] = num(cv.median()) if cv.len() else None
    return out


def _run_lengths(mask: np.ndarray) -> np.ndarray:
    """Lengths of consecutive True runs."""
    if not mask.any():
        return np.empty(0, dtype=np.int64)
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return edges[1::2] - edges[::2]


def rolling_half_life_profile(frame: pl.DataFrame, *, config: OUConfig) -> pl.DataFrame:
    """Rolling median and IQR of the half-life through time (for plotting)."""
    window = config.stability.rolling_window
    minimum = max(10, window // 4)
    hl = pl.col("ou_half_life_bars")
    return frame.select(
        "timestamp",
        hl.rolling_median(window_size=window, min_samples=minimum).alias("rolling_median"),
        hl.rolling_quantile(0.25, window_size=window, min_samples=minimum).alias("rolling_p25"),
        hl.rolling_quantile(0.75, window_size=window, min_samples=minimum).alias("rolling_p75"),
        pl.col("ou_valid").cast(pl.Float64)
        .rolling_mean(window_size=window, min_samples=minimum).alias("rolling_valid_fraction"),
    )


_SEGMENT_COLUMNS: tuple[str, ...] = (
    "timestamp", "residual", "ou_n_pairs", "ou_valid", "ou_b", "ou_r_squared", "ou_theta",
    "ou_half_life_bars", "ou_sigma", "ou_mu", "ou_stationary_std",
)


def _segment_rows(
    frame: pl.DataFrame, label: pl.Expr, *, config: OUConfig, min_pairs: int
) -> list[dict[str, Any]]:
    # Only what a segment row reads: each segment below is a filtered copy.
    tagged = frame.select(_SEGMENT_COLUMNS).with_columns(label.alias("segment"))
    rows: list[dict[str, Any]] = []
    for segment in sorted(tagged["segment"].unique().drop_nulls().to_list()):
        part = tagged.filter(pl.col("segment") == segment)
        complete = part.filter(pl.col("ou_n_pairs") > 0)
        valid = complete.filter(pl.col("ou_valid"))
        row: dict[str, Any] = {
            "segment": segment,
            "bars": part.height,
            "first_timestamp": str(part["timestamp"].min()),
            "last_timestamp": str(part["timestamp"].max()),
            "windows_estimated": complete.height,
        }
        if complete.height:
            b = complete["ou_b"].drop_nulls()
            row.update({
                "valid_ou_fraction": valid.height / complete.height,
                "fraction_b_ge_1": num((b >= 1).mean(), 0.0),
                "fraction_b_le_0": num((b <= 0).mean(), 0.0),
                "near_unit_root_fraction": num(
                    (b > config.near_unit_root.flag_threshold).mean(), 0.0),
                "rolling_median_b": num(b.median()),
                "rolling_median_r_squared": num(complete["ou_r_squared"].median()),
            })
        if valid.height:
            hl = valid["ou_half_life_bars"]
            row.update({
                "rolling_median_theta": num(valid["ou_theta"].median()),
                "rolling_median_half_life": num(hl.median()),
                "rolling_half_life_p25": num(hl.quantile(0.25)),
                "rolling_half_life_p75": num(hl.quantile(0.75)),
                "rolling_median_sigma": num(valid["ou_sigma"].median()),
                "rolling_median_mu": num(valid["ou_mu"].median()),
                "rolling_median_stationary_std": num(
                    valid["ou_stationary_std"].median()),
            })
        values = part["residual"].fill_null(np.nan).to_numpy().astype(np.float64)
        pairs = int(np.count_nonzero(np.isfinite(values[:-1]) & np.isfinite(values[1:])))
        row["sufficient_data"] = pairs >= min_pairs
        if pairs >= min_pairs:
            fit = fit_ar1(values, rules=config.validity_rules(),
                          hac_lags=config.equilibrium.resolve_lags(pairs),
                          confidence_level=config.equilibrium.confidence_level)
            static = fit_row(fit)
            row.update({f"static_{k}": v for k, v in static.items()})
            resid = fit.residuals
            if resid.size > 20:
                row["static_innovation_acf1"] = float(np.corrcoef(resid[1:], resid[:-1])[0, 1])
                sq = resid ** 2
                row["static_innovation_sq_acf1"] = float(np.corrcoef(sq[1:], sq[:-1])[0, 1])
                from scipy import stats

                row["static_innovation_excess_kurtosis"] = float(
                    stats.kurtosis(resid, fisher=True, bias=False))
        rows.append(row)
    return rows


def segment_table(frame: pl.DataFrame, *, by: str, config: OUConfig) -> pl.DataFrame:
    """Per-year or per-quarter stability, with shift flags against the previous segment."""
    if by == "year":
        label = pl.col("timestamp").dt.year().cast(pl.Utf8)
    elif by == "quarter":
        label = (pl.col("timestamp").dt.year().cast(pl.Utf8) + "Q"
                 + pl.col("timestamp").dt.quarter().cast(pl.Utf8))
    else:
        raise ValueError(f"by must be 'year' or 'quarter', got {by!r}")
    rows = _segment_rows(frame, label, config=config,
                         min_pairs=config.segmented_estimation.min_observations)
    factor = config.stability.segment_shift_factor
    previous: dict[str, Any] | None = None
    for row in rows:
        row["flag_invalid_ou"] = bool(row.get("sufficient_data") and not row.get("static_ou_valid"))
        row["flag_near_unit_root"] = bool(
            (row.get("static_b") or 0) > config.near_unit_root.flag_threshold)
        row["flag_mu_sign_change"] = False
        row["flag_half_life_shift"] = False
        row["flag_theta_shift"] = False
        if previous is not None and previous.get("sufficient_data") and row.get("sufficient_data"):
            mu_now = num(row.get("static_mu"))
            mu_before = num(previous.get("static_mu"))
            if np.isfinite(mu_now) and np.isfinite(mu_before):
                row["flag_mu_sign_change"] = bool(np.sign(mu_now) != np.sign(mu_before))
            for key, flag in (("rolling_median_half_life", "flag_half_life_shift"),
                              ("rolling_median_theta", "flag_theta_shift")):
                now, before = num(row.get(key)), num(previous.get(key))
                if np.isfinite(now) and np.isfinite(before) and now > 0 and before > 0:
                    row[flag] = bool(max(now / before, before / now) > factor)
        previous = row
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def era_table(frame: pl.DataFrame, *, config: OUConfig, blocks: int = 3) -> pl.DataFrame:
    """Equal-length chronological blocks - boundaries by arithmetic, not by choice."""
    first, last = frame["timestamp"].min(), frame["timestamp"].max()
    if not isinstance(first, datetime) or not isinstance(last, datetime):
        return pl.DataFrame()
    span = (last - first) / blocks
    names = ["early", "middle", "recent"] if blocks == 3 else [f"block_{i + 1}" for i in range(blocks)]
    edges = [first + span * i for i in range(blocks)] + [last + timedelta(microseconds=1)]
    label = pl.lit(None, dtype=pl.Utf8)
    for i in reversed(range(blocks)):
        label = pl.when(
            (pl.col("timestamp") >= edges[i]) & (pl.col("timestamp") < edges[i + 1])
        ).then(pl.lit(f"{i + 1}_{names[i]}")).otherwise(label)
    rows = _segment_rows(frame, label, config=config,
                         min_pairs=config.segmented_estimation.min_observations)
    for row, (lo, hi) in zip(rows, zip(edges[:-1], edges[1:], strict=True), strict=False):
        row["block_start"] = str(lo)
        row["block_end"] = str(hi)
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def recent_versus_history(frame: pl.DataFrame, *, months: int = 12) -> dict[str, Any]:
    """The last *months* of data against everything before, with effect sizes."""
    from scipy import stats

    last = frame["timestamp"].max()
    if not isinstance(last, datetime):
        return {}
    cut = last - timedelta(days=int(months * 30.44))
    compared = ("ou_b", "ou_theta", "ou_half_life_bars", "ou_sigma", "ou_r_squared")
    slim = frame.select("timestamp", "ou_valid", *compared)     # a filter copies what it keeps
    recent = slim.filter((pl.col("timestamp") >= cut) & pl.col("ou_valid"))
    older = slim.filter((pl.col("timestamp") < cut) & pl.col("ou_valid"))
    out: dict[str, Any] = {"recent_from": str(cut), "recent_to": str(last),
                           "recent_windows": recent.height, "older_windows": older.height}
    if recent.height < 30 or older.height < 30:
        out["sufficient_data"] = False
        return out
    out["sufficient_data"] = True
    for column in compared:
        a = recent[column].drop_nulls().to_numpy()
        b = older[column].drop_nulls().to_numpy()
        name = column.removeprefix("ou_")
        out[f"{name}_recent_median"] = float(np.median(a))
        out[f"{name}_older_median"] = float(np.median(b))
        out[f"{name}_median_ratio"] = (
            float(np.median(a) / np.median(b)) if np.median(b) != 0 else None)
        out[f"{name}_ks_statistic"] = float(stats.ks_2samp(a, b).statistic)
    out["note"] = (
        "KS statistics are effect sizes here. Consecutive rolling estimates share "
        "almost all their data, so a KS p-value would be wildly overconfident and is "
        "deliberately not reported."
    )
    return out
