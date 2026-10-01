r"""Does OU behaviour depend on volatility, trend, fit quality or time of day?

Every conditioning variable is known at ``t``: the volatility of log returns
over the OU window (and the Prompt #2 trailing volatility), the rolling
regression's slope and :math:`R^2`, the hour and the session. Quantile edges
come from the whole sample, which is fine for describing history and would be
look-ahead in a live rule; nothing here is a rule.

Two cautions travel with these tables:

* A rolling estimate at ``t`` describes the whole window ending at ``t``, so
  bucketing it by a value at ``t`` smears the relationship. The window
  volatility is therefore computed over the same ``M`` bars as the estimate.
* An ``M``-bar window spans many hours, so the hour-of-day median of a
  rolling estimate mostly reflects neighbouring hours. The intraday table
  therefore also fits a *pooled* one-step AR(1) on only the pairs that start
  in each hour - a genuinely local estimate - and reports the one-step
  innovation std by hour, which is where intraday structure actually shows.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from ..models.config import OUConfig
from ..models.ornstein_uhlenbeck import fit_ar1_pairs
from .config import ResearchConfig
from .intraday import assign_sessions
from .ou_estimation import num

__all__ = [
    "attach_conditioning",
    "intraday_conditioning",
    "r2_conditioning",
    "trend_conditioning",
    "volatility_conditioning",
]

TREND_LABELS_5 = ("strongly_negative", "weak_negative", "near_flat", "weak_positive",
                  "strongly_positive")


def attach_conditioning(
    ou_frame: pl.DataFrame, features: pl.DataFrame, *, ou_window: int
) -> pl.DataFrame:
    """Join the regression's conditioning columns and the OU-window volatility."""
    wanted = [c for c in ("regression_slope", "r_squared", "trailing_volatility", "log_return")
              if c in features.columns]
    joined = ou_frame.with_columns(features.select(wanted))
    if "log_return" in joined.columns:
        joined = joined.with_columns(
            pl.col("log_return").rolling_std(window_size=ou_window, min_samples=ou_window)
            .alias("ou_window_volatility")
        )
        if "regression_slope" in joined.columns:
            joined = joined.with_columns(
                pl.when(pl.col("ou_window_volatility") > 0)
                .then(pl.col("regression_slope") / pl.col("ou_window_volatility"))
                .otherwise(None).alias("slope_over_volatility")
            )
    return joined


def _bucket(frame: pl.DataFrame, column: str, buckets: int,
            labels: tuple[str, ...] | None = None) -> tuple[pl.DataFrame, list[float]]:
    """Quantile buckets Q1..Qk (Q1 lowest), built explicitly, lowest edge first."""
    values = frame.get_column(column)
    usable = values.filter(values.is_not_null() & values.is_not_nan())    # one column only
    if usable.is_empty():
        return frame.with_columns(pl.lit(None, dtype=pl.Utf8).alias("bucket")), []
    edges = [num(usable.quantile(i / buckets)) for i in range(1, buckets)]
    names = list(labels) if labels else [f"Q{i + 1}" for i in range(buckets)]
    expr = pl.lit(names[-1])
    for i in reversed(range(buckets - 1)):
        expr = pl.when(pl.col(column) <= edges[i]).then(pl.lit(names[i])).otherwise(expr)
    return frame.with_columns(
        pl.when(pl.col(column).is_null() | pl.col(column).is_nan()).then(None)
        .otherwise(expr).alias("bucket")
    ), edges


#: What a group summary reads from the per-bar OU frame.
_SUMMARY_COLUMNS: tuple[str, ...] = (
    "ou_n_pairs", "ou_valid", "ou_b", "ou_theta", "ou_half_life_bars", "ou_mu", "ou_sigma",
    "ou_stationary_std", "ou_r_squared",
)


def _group_summary(frame: pl.DataFrame, group: str, *, config: OUConfig) -> pl.DataFrame:
    """Rolling-estimate medians per group; rows without a group are left out."""
    # Select before filtering: a filter copies every column it keeps, and the
    # full frame is ~30 columns (about 2 GB at 23 years of 1-minute bars).
    complete = frame.select(group, *_SUMMARY_COLUMNS).filter(
        (pl.col("ou_n_pairs") > 0) & pl.col(group).is_not_null())
    if complete.is_empty():
        return pl.DataFrame()
    valid = pl.col("ou_valid")
    return (
        complete.group_by(group)
        .agg(
            pl.len().alias("windows"),
            valid.mean().alias("valid_ou_fraction"),
            (pl.col("ou_b") >= 1).mean().alias("fraction_b_ge_1"),
            (pl.col("ou_b") > config.near_unit_root.flag_threshold).mean()
            .alias("near_unit_root_fraction"),
            pl.col("ou_b").median().alias("median_b"),
            pl.col("ou_theta").filter(valid).median().alias("median_theta"),
            pl.col("ou_half_life_bars").filter(valid).median().alias("median_half_life"),
            pl.col("ou_half_life_bars").filter(valid).quantile(0.25).alias("half_life_p25"),
            pl.col("ou_half_life_bars").filter(valid).quantile(0.75).alias("half_life_p75"),
            pl.col("ou_mu").filter(valid).median().alias("median_mu"),
            pl.col("ou_sigma").filter(valid).median().alias("median_sigma"),
            pl.col("ou_stationary_std").filter(valid).median().alias("median_stationary_std"),
            pl.col("ou_r_squared").median().alias("median_ou_r_squared"),
        )
        .with_columns((pl.col("windows") < config.conditioning.min_samples_warning)
                      .alias("thin_sample"))
        .sort(group)
    )


def _conditioned(frame: pl.DataFrame, variables: list[tuple[str, int, tuple[str, ...] | None]],
                 *, source: str, config: OUConfig) -> pl.DataFrame:
    tables: list[pl.DataFrame] = []
    for column, buckets, labels in variables:
        if column not in frame.columns:
            continue
        bucketed, edges = _bucket(frame, column, buckets, labels)
        summary = _group_summary(bucketed, "bucket", config=config)
        if summary.is_empty():
            continue
        order = list(labels) if labels else [f"Q{i + 1}" for i in range(buckets)]
        bounds = [None, *edges]
        uppers = [*edges, None]
        lookup = {name: (lo, hi) for name, lo, hi in zip(order, bounds, uppers, strict=True)}
        tables.append(summary.with_columns(
            pl.lit(source).alias("source"),
            pl.lit(column).alias("variable"),
            pl.col("bucket").replace_strict(
                {k: v[0] for k, v in lookup.items()}, default=None, return_dtype=pl.Float64
            ).alias("lower_edge"),
            pl.col("bucket").replace_strict(
                {k: v[1] for k, v in lookup.items()}, default=None, return_dtype=pl.Float64
            ).alias("upper_edge"),
            pl.col("bucket").replace_strict(
                {name: i for i, name in enumerate(order)}, return_dtype=pl.Int32
            ).alias("bucket_rank"),
        ).sort("bucket_rank"))
    if not tables:
        return pl.DataFrame()
    return pl.concat(tables, how="diagonal_relaxed").select(
        "source", "variable", "bucket", "bucket_rank", "lower_edge", "upper_edge",
        pl.exclude("source", "variable", "bucket", "bucket_rank", "lower_edge", "upper_edge"),
    )


def volatility_conditioning(frame: pl.DataFrame, *, source: str, config: OUConfig) -> pl.DataFrame:
    """OU behaviour by volatility quartile (window-matched, and Prompt #2 trailing)."""
    k = config.conditioning.volatility_buckets
    return _conditioned(frame, [("ou_window_volatility", k, None),
                                ("trailing_volatility", k, None)], source=source, config=config)


def trend_conditioning(frame: pl.DataFrame, *, source: str, config: OUConfig) -> pl.DataFrame:
    """OU behaviour by trend strength: the raw slope and slope per unit volatility."""
    k = config.conditioning.trend_buckets
    labels = TREND_LABELS_5 if k == 5 else None
    return _conditioned(frame, [("regression_slope", k, labels),
                                ("slope_over_volatility", k, labels)],
                        source=source, config=config)


def r2_conditioning(frame: pl.DataFrame, *, source: str, config: OUConfig) -> pl.DataFrame:
    """OU behaviour by the rolling regression's R-squared quartile."""
    return _conditioned(frame, [("r_squared", config.conditioning.r_squared_buckets, None)],
                        source=source, config=config)


def intraday_conditioning(
    frame: pl.DataFrame, *, source: str, config: OUConfig, research: ResearchConfig
) -> pl.DataFrame:
    """By hour and by session: rolling medians, a pooled local fit, innovation std."""
    basis = config.conditioning.time_basis if config.conditioning.time_basis in frame.columns \
        else "timestamp"
    tagged = frame.with_columns(pl.col(basis).dt.hour().cast(pl.Int32).alias("hour"))
    groupings: list[tuple[str, pl.DataFrame]] = []
    if config.conditioning.by_hour:
        groupings.append(("hour", tagged.with_columns(pl.col("hour").cast(pl.Utf8)
                                                      .str.zfill(2).alias("group"))))
    if config.conditioning.by_session and research.intraday.sessions.definitions:
        groupings.append(("session", tagged.with_columns(
            assign_sessions(pl.col(basis), research).alias("group"))))

    x = frame["residual"].fill_null(np.nan).to_numpy().astype(np.float64)
    nxt = np.concatenate((x[1:], [np.nan]))
    innovation = frame["ou_innovation"].fill_null(np.nan).to_numpy().astype(np.float64)
    tables: list[pl.DataFrame] = []
    for kind, grouped in groupings:
        summary = _group_summary(grouped, "group", config=config)
        labels = grouped["group"]
        extra: list[dict[str, Any]] = []
        for group in summary["group"].to_list() if not summary.is_empty() else []:
            # A boolean mask per group, never an array of Python strings.
            mask = (labels == group).fill_null(False).to_numpy()
            ok = mask & np.isfinite(x) & np.isfinite(nxt)
            row: dict[str, Any] = {"group": group, "pooled_pairs": int(ok.sum())}
            if ok.sum() >= config.min_observations:
                fit = fit_ar1_pairs(x[ok], nxt[ok], rules=config.validity_rules(),
                                    keep_residuals=False)
                row.update({
                    "pooled_b": fit.b, "pooled_state": fit.ou.state,
                    "pooled_theta": fit.ou.theta, "pooled_half_life": fit.ou.half_life_bars,
                    "pooled_sigma": fit.ou.sigma, "pooled_r_squared": fit.r_squared,
                })
            eta = innovation[mask]
            eta = eta[np.isfinite(eta)]
            row["innovation_std"] = float(np.std(eta, ddof=1)) if eta.size > 1 else None
            row["mean_abs_innovation"] = float(np.mean(np.abs(eta))) if eta.size else None
            extra.append(row)
        if summary.is_empty():
            continue
        merged = summary.join(pl.DataFrame(extra, infer_schema_length=None), on="group",
                              how="left")
        tables.append(merged.with_columns(pl.lit(source).alias("source"),
                                          pl.lit(kind).alias("grouping")))
    if not tables:
        return pl.DataFrame()
    return pl.concat(tables, how="diagonal_relaxed").select(
        "source", "grouping", "group", pl.exclude("source", "grouping", "group")
    )
