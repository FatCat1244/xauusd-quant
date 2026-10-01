"""Time-of-day, weekday and session behaviour.

Every grouping here is done on an explicitly-named timestamp basis, and that
name is carried into the output. Nothing assumes UTC, London, Bangkok or
anything else: by default this is the broker-local clock established in
``config/data.yaml`` (America/New_York + 7 h), and switching
``intraday.time_basis`` to ``timestamp_utc`` regroups everything without
touching the code.

Session boundaries are configuration, not fact. They approximate liquidity
centres and are **not** daylight-saving adjusted: the broker clock tracks US
DST, but London and Tokyo switch on different dates, so for a few weeks each
year the edges drift by an hour. :func:`session_analysis` states this in its
own output rather than leaving it to a reader to remember.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from .config import ResearchConfig

__all__ = [
    "IntradayAnalysis",
    "assign_sessions",
    "hourly_analysis",
    "session_analysis",
    "weekday_analysis",
]

WEEKDAY_NAMES = {
    1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday",
    5: "Friday", 6: "Saturday", 7: "Sunday",
}


@dataclass
class IntradayAnalysis:
    """One grouped view of intraday behaviour, with its basis recorded."""

    name: str
    time_basis: str
    timezone_description: str
    table: pl.DataFrame = field(default_factory=pl.DataFrame)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "time_basis": self.time_basis,
            "timezone_description": self.timezone_description,
            "rows": self.table.height,
            "notes": self.notes,
            "warnings": self.warnings,
        }


def _group_aggregations(return_column: str, extreme_threshold: float | None) -> list[pl.Expr]:
    """The statistics computed for every intraday grouping."""
    aggs = [
        pl.len().alias("bars"),
        pl.col(return_column).mean().alias("mean_return"),
        pl.col(return_column).median().alias("median_return"),
        pl.col(return_column).std().alias("return_std"),
        pl.col(return_column).abs().mean().alias("mean_abs_return"),
        pl.col(return_column).min().alias("min_return"),
        pl.col(return_column).max().alias("max_return"),
    ]
    if extreme_threshold is not None:
        aggs.append(
            (pl.col(return_column).abs() > extreme_threshold)
            .mean()
            .alias("extreme_return_frequency")
        )
    return aggs


def _optional_aggregations(columns: set[str]) -> list[pl.Expr]:
    """Spread and activity aggregations, for whichever columns exist."""
    out: list[pl.Expr] = []
    if "mean_spread" in columns:
        out += [
            pl.col("mean_spread").mean().alias("mean_spread"),
            pl.col("mean_spread").median().alias("median_spread"),
            pl.col("mean_spread").quantile(0.95).alias("p95_spread"),
        ]
    if "tick_count" in columns:
        out += [
            pl.col("tick_count").mean().alias("mean_tick_count"),
            pl.col("tick_count").median().alias("median_tick_count"),
            pl.col("tick_count").sum().alias("total_ticks"),
        ]
    if "volume" in columns:
        out.append(pl.col("volume").mean().alias("mean_volume"))
    return out


def hourly_analysis(
    frame: pl.DataFrame, *, return_column: str, config: ResearchConfig,
    timezone_description: str,
) -> IntradayAnalysis:
    """Statistics grouped by hour of day on the configured clock."""
    basis = _resolve_basis(frame, config)
    usable = frame.filter(pl.col(return_column).is_not_null())
    result = IntradayAnalysis(
        name="hourly", time_basis=basis, timezone_description=timezone_description
    )
    if usable.is_empty():
        result.warnings.append("No bars with returns; hourly analysis skipped.")
        return result

    threshold = _extreme_threshold(usable, return_column, config)
    columns = set(usable.columns)
    result.table = (
        usable.with_columns(pl.col(basis).dt.hour().alias("hour"))
        .group_by("hour")
        .agg(_group_aggregations(return_column, threshold) + _optional_aggregations(columns))
        .sort("hour")
    )
    result.notes += [
        f"Hours are on the `{basis}` clock: {timezone_description}.",
        f"extreme_return_frequency = share of bars with |r| above the "
        f"{config.intraday.extreme_quantile:.1%} quantile of |r| "
        f"({threshold:.6g}) computed over the whole period.",
    ]
    return result


def weekday_analysis(
    frame: pl.DataFrame, *, return_column: str, config: ResearchConfig,
    timezone_description: str,
) -> IntradayAnalysis:
    """Statistics grouped by day of week.

    Saturday and Sunday bars are reported if present rather than dropped:
    their existence is itself information about the export, since XAUUSD is
    normally closed at the weekend.
    """
    basis = _resolve_basis(frame, config)
    usable = frame.filter(pl.col(return_column).is_not_null())
    result = IntradayAnalysis(
        name="weekday", time_basis=basis, timezone_description=timezone_description
    )
    if usable.is_empty():
        result.warnings.append("No bars with returns; weekday analysis skipped.")
        return result

    threshold = _extreme_threshold(usable, return_column, config)
    columns = set(usable.columns)
    table = (
        usable.with_columns(pl.col(basis).dt.weekday().alias("weekday_number"))
        .group_by("weekday_number")
        .agg(_group_aggregations(return_column, threshold) + _optional_aggregations(columns))
        .sort("weekday_number")
        .with_columns(
            pl.col("weekday_number")
            .replace_strict(WEEKDAY_NAMES, default="Unknown")
            .alias("weekday")
        )
    )
    result.table = table.select(["weekday_number", "weekday", *[
        c for c in table.columns if c not in ("weekday_number", "weekday")
    ]])

    weekend = table.filter(pl.col("weekday_number") > 5)
    if not weekend.is_empty():
        total = int(weekend["bars"].sum())
        result.warnings.append(
            f"{total:,} bars fall on Saturday or Sunday on the `{basis}` clock. "
            "XAUUSD is normally closed then, so this is reported rather than "
            "silently dropped - it may indicate session-boundary effects or an "
            "export quirk worth checking."
        )
    result.notes.append(f"Weekdays are on the `{basis}` clock: {timezone_description}.")
    return result


def assign_sessions(timestamps: pl.Expr, config: ResearchConfig) -> pl.Expr:
    """Label each bar with the first matching configured session.

    Windows are ``[start, end)`` on the configured clock and may wrap midnight.
    They are allowed to overlap - ``london_ny_overlap`` deliberately does - so
    the first match in configuration order wins, and a bar matching nothing is
    labelled ``unclassified``.
    """
    # dt.hour() and dt.minute() are Int8: hour * 60 overflows above hour 2
    # (3 * 60 = 180 wraps to -76), which silently sends every bar into a
    # midnight-wrapping window. Widen before doing any arithmetic.
    minutes = (
        timestamps.dt.hour().cast(pl.Int32) * 60
        + timestamps.dt.minute().cast(pl.Int32)
    )
    expr = pl.lit("unclassified")
    for name, window in reversed(list(config.intraday.sessions.definitions.items())):
        start, end = window.start_minutes, window.end_minutes
        inside = (
            (minutes >= start) | (minutes < end)
            if window.wraps_midnight
            else (minutes >= start) & (minutes < end)
        )
        expr = pl.when(inside).then(pl.lit(name)).otherwise(expr)
    return expr


def session_analysis(
    frame: pl.DataFrame, *, return_column: str, config: ResearchConfig,
    timezone_description: str,
) -> IntradayAnalysis:
    """Statistics grouped by configured trading session."""
    basis = _resolve_basis(frame, config)
    result = IntradayAnalysis(
        name="session", time_basis=basis, timezone_description=timezone_description
    )
    sessions = config.intraday.sessions
    if not sessions.enabled or not sessions.definitions:
        result.warnings.append("Session analysis is disabled or has no definitions.")
        return result

    usable = frame.filter(pl.col(return_column).is_not_null())
    if usable.is_empty():
        result.warnings.append("No bars with returns; session analysis skipped.")
        return result

    threshold = _extreme_threshold(usable, return_column, config)
    columns = set(usable.columns)
    result.table = (
        usable.with_columns(assign_sessions(pl.col(basis), config).alias("session"))
        .group_by("session")
        .agg(_group_aggregations(return_column, threshold) + _optional_aggregations(columns))
        .sort("session")
    )

    definitions = {
        name: f"{w.start}-{w.end}" for name, w in sessions.definitions.items()
    }
    result.notes += [
        f"Sessions are defined on the `{basis}` clock: {timezone_description}.",
        f"Boundaries used: {definitions}.",
        "Windows are [start, end) and may overlap; the first match in "
        "configuration order wins, so `london_ny_overlap` claims its bars before "
        "the wider London and New York windows see them.",
    ]
    result.warnings.append(
        "Session boundaries are NOT daylight-saving adjusted. The broker clock "
        "follows US DST, but London and Tokyo change on different dates, so for "
        "a few weeks each spring and autumn these edges are off by an hour. "
        "Treat session comparisons near those transitions with care."
    )
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_basis(frame: pl.DataFrame, config: ResearchConfig) -> str:
    """The configured time basis, falling back to `timestamp` if absent."""
    basis = config.intraday.time_basis
    if basis in frame.columns:
        return basis
    if "timestamp" in frame.columns:
        return "timestamp"
    raise KeyError(f"Frame has no timestamp column; columns: {frame.columns}")


def _extreme_threshold(frame: pl.DataFrame, return_column: str, config: ResearchConfig) -> float | None:
    """|r| quantile above which a bar counts as an extreme move."""
    try:
        value = frame.select(
            pl.col(return_column).abs().quantile(config.intraday.extreme_quantile)
        ).item()
    except Exception:  # noqa: BLE001 - a missing threshold must not kill the grouping
        return None
    return float(value) if value is not None else None
