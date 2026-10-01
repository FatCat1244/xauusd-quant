"""Data-quality diagnostics for the generated bar datasets.

Missing bars are *never* filled. The job here is the opposite: find every hole
and describe it precisely enough that a researcher can tell a real market
closure from missing data.

Gaps are classified against the configured trading schedule:

``weekend``
    The gap spans the ordinary weekly close-to-open window.
``holiday_weekend``
    The gap ends at the weekly open and covers a Saturday, but started before
    the ordinary weekly close - an extended closure such as Good Friday or
    Christmas. Expected, but worth keeping distinct from a normal weekend.
``daily_break``
    The gap matches the instrument's daily maintenance break.
``holiday_or_unknown``
    Everything else - a mid-session early close, a feed outage, a minute with
    no ticks at all, or genuinely absent data. These are the ones worth
    looking at, and the largest are listed explicitly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from ..utils.clock import utc_now_iso
from ..utils.config import Config, SessionConfig
from ..utils.logging import get_logger
from ..utils.paths import atomic_write_text, ensure_dir
from .resampler import parse_timeframe

__all__ = ["BarDiagnostics", "diagnose_all", "diagnose_timeframe"]

LOGGER = get_logger("data.diagnostics")

DIAGNOSTICS_NAME = "bar_diagnostics.json"


@dataclass
class BarDiagnostics:
    """Quality report for one bar timeframe."""

    timeframe: str
    bars: int = 0
    first_timestamp: str | None = None
    last_timestamp: str | None = None

    expected_slots_in_span: int = 0
    missing_slots: int = 0
    coverage_ratio: float = 0.0

    gap_count: int = 0
    gaps_by_category: dict[str, int] = field(default_factory=dict)
    missing_slots_by_category: dict[str, int] = field(default_factory=dict)
    largest_unexplained_gaps: list[dict[str, Any]] = field(default_factory=list)

    mean_ticks_per_bar: float | None = None
    median_ticks_per_bar: float | None = None
    min_ticks_per_bar: int | None = None
    low_tick_count_threshold: float | None = None
    low_tick_count_bars: int = 0

    spread_distribution: dict[str, float] = field(default_factory=dict)
    zero_range_bars: int = 0
    bars_per_year: dict[str, int] = field(default_factory=dict)

    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    generated_utc: str = field(
        default_factory=utc_now_iso
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def diagnose_timeframe(config: Config, timeframe: str) -> BarDiagnostics:
    """Profile the bars of one timeframe, reading only the columns needed."""
    directory = config.bars_dir(timeframe)
    files = sorted(directory.rglob("*.parquet"))
    report = BarDiagnostics(timeframe=timeframe)
    if not files:
        report.warnings.append(f"No bar files under {directory}. Run `xq build-bars`.")
        return report

    wanted = ["timestamp", "open", "high", "low", "close", "tick_count",
              "mean_spread", "median_spread", "min_spread", "max_spread"]
    lazy = pl.scan_parquet(files)
    available = set(lazy.collect_schema().names())
    bars = lazy.select([c for c in wanted if c in available]).collect().sort("timestamp")

    report.bars = bars.height
    if bars.is_empty():
        report.warnings.append("Bar files exist but contain no rows.")
        return report

    step = parse_timeframe(timeframe)
    _span(report, bars)
    _tick_counts(report, bars, config.diagnostics.low_tick_count_quantile)
    _spreads(report, bars)
    _ranges(report, bars)
    _gaps(report, bars, step, config)
    _per_year(report, bars)

    report.notes.append(
        "Missing bars are reported, never filled. A missing slot means no tick "
        "was recorded in that interval."
    )
    report.notes.append(
        f"Bar timestamps use the '{config.resampling.label}' convention on the "
        f"{config.resampling.time_basis} basis ({config.timezone.describe()})."
    )
    return report


def _span(report: BarDiagnostics, bars: pl.DataFrame) -> None:
    lo, hi = bars.select(
        pl.col("timestamp").min().alias("lo"), pl.col("timestamp").max().alias("hi")
    ).row(0)
    report.first_timestamp = lo.isoformat(sep=" ")
    report.last_timestamp = hi.isoformat(sep=" ")


def _tick_counts(report: BarDiagnostics, bars: pl.DataFrame, quantile: float) -> None:
    if "tick_count" not in bars.columns:
        return
    stats = bars.select(
        pl.col("tick_count").mean().alias("mean"),
        pl.col("tick_count").median().alias("median"),
        pl.col("tick_count").min().alias("min"),
        pl.col("tick_count").quantile(quantile, interpolation="lower").alias("low"),
    ).row(0)
    report.mean_ticks_per_bar = float(stats[0]) if stats[0] is not None else None
    report.median_ticks_per_bar = float(stats[1]) if stats[1] is not None else None
    report.min_ticks_per_bar = int(stats[2]) if stats[2] is not None else None
    report.low_tick_count_threshold = float(stats[3]) if stats[3] is not None else None
    if stats[3] is not None:
        report.low_tick_count_bars = int(
            bars.select((pl.col("tick_count") <= stats[3]).sum()).item()
        )


def _spreads(report: BarDiagnostics, bars: pl.DataFrame) -> None:
    if "mean_spread" not in bars.columns:
        return
    column = pl.col("mean_spread")
    stats = bars.select(
        column.min().alias("min"),
        column.quantile(0.01).alias("p01"),
        column.quantile(0.25).alias("p25"),
        column.median().alias("p50"),
        column.quantile(0.75).alias("p75"),
        column.quantile(0.95).alias("p95"),
        column.quantile(0.99).alias("p99"),
        column.max().alias("max"),
        column.mean().alias("mean"),
    ).row(0, named=True)
    report.spread_distribution = {
        k: float(v) for k, v in stats.items() if v is not None
    }


def _ranges(report: BarDiagnostics, bars: pl.DataFrame) -> None:
    """Bars whose high equals their low - a single price for the whole interval."""
    if {"high", "low"} <= set(bars.columns):
        report.zero_range_bars = int(bars.select((pl.col("high") == pl.col("low")).sum()).item())


def _per_year(report: BarDiagnostics, bars: pl.DataFrame) -> None:
    counts = (
        bars.select(pl.col("timestamp").dt.year().alias("year"))
        .group_by("year").len().sort("year")
    )
    report.bars_per_year = {str(r["year"]): int(r["len"]) for r in counts.iter_rows(named=True)}


def _gaps(
    report: BarDiagnostics, bars: pl.DataFrame, step: timedelta, config: Config
) -> None:
    """Find and classify every interval longer than one bar step."""
    diag = config.diagnostics
    session = diag.session
    step_us = int(step.total_seconds() * 1_000_000)

    gaps = (
        bars.select(
            pl.col("timestamp").shift(1).alias("prev"),
            pl.col("timestamp").alias("curr"),
        )
        .drop_nulls()
        .with_columns(
            ((pl.col("curr") - pl.col("prev")).dt.total_microseconds() // step_us - 1)
            .alias("missing")
        )
        .filter(pl.col("missing") > 0)
    )

    expected = int(
        (
            bars.select(
                (pl.col("timestamp").max() - pl.col("timestamp").min())
                .dt.total_microseconds()
            ).item()
            // step_us
        )
        + 1
    )
    report.expected_slots_in_span = expected
    report.missing_slots = int(gaps.select(pl.col("missing").sum()).item() or 0)
    report.coverage_ratio = report.bars / expected if expected else 0.0
    report.gap_count = gaps.height

    by_category: dict[str, int] = {}
    slots_by_category: dict[str, int] = {}
    unexplained: list[dict[str, Any]] = []
    for row in gaps.iter_rows(named=True):
        category = _classify(row["prev"], row["curr"], session)
        by_category[category] = by_category.get(category, 0) + 1
        slots_by_category[category] = slots_by_category.get(category, 0) + int(row["missing"])
        if category == "holiday_or_unknown":
            unexplained.append({
                "from": row["prev"].isoformat(sep=" "),
                "to": row["curr"].isoformat(sep=" "),
                "missing_bars": int(row["missing"]),
                "duration_hours": round(
                    (row["curr"] - row["prev"]).total_seconds() / 3600.0, 3
                ),
            })

    report.gaps_by_category = dict(sorted(by_category.items()))
    report.missing_slots_by_category = dict(sorted(slots_by_category.items()))
    unexplained.sort(key=lambda g: g["missing_bars"], reverse=True)
    report.largest_unexplained_gaps = unexplained[: diag.max_reported_gaps]

    if unexplained:
        report.warnings.append(
            f"{len(unexplained):,} gap(s) match neither the weekly schedule nor the "
            f"daily break ({sum(g['missing_bars'] for g in unexplained):,} missing bars). "
            "These are mid-session early closes, thin intervals with no ticks, or real "
            "data loss - inspect before modelling."
        )


def _classify(prev: datetime, curr: datetime, session: SessionConfig) -> str:
    """Label a gap against the configured trading schedule."""
    if _is_weekend_gap(prev, curr, session):
        return "weekend"
    if _is_holiday_weekend(prev, curr, session):
        return "holiday_weekend"
    if _is_daily_break(prev, curr, session):
        return "daily_break"
    return "holiday_or_unknown"


def _is_holiday_weekend(prev: datetime, curr: datetime, session: SessionConfig) -> bool:
    """A closure that ends at the weekly open and swallowed a whole weekend.

    Kept separate from ``weekend`` so an extended holiday break is not mistaken
    for a normal one, and separate from ``holiday_or_unknown`` so it does not
    bury the gaps that actually need investigating.
    """
    if curr.isoweekday() != session.week_open_weekday:
        return False
    if (curr - prev) > timedelta(days=7):
        return False
    return _contains_saturday(prev, curr)


def _contains_saturday(prev: datetime, curr: datetime) -> bool:
    """True when at least one Saturday falls inside the gap."""
    day = prev.date()
    while day <= curr.date():
        if day.isoweekday() == 6:
            return True
        day += timedelta(days=1)
    return False


def _is_weekend_gap(prev: datetime, curr: datetime, session: SessionConfig) -> bool:
    """A gap from the weekly close to the next weekly open."""
    if prev.isoweekday() != session.week_close_weekday:
        return False
    if curr.isoweekday() != session.week_open_weekday:
        return False
    return (curr - prev) < timedelta(days=4)


def _is_daily_break(prev: datetime, curr: datetime, session: SessionConfig) -> bool:
    """A gap that starts at/after the daily close and ends at/before the reopen."""
    if not (session.daily_break_start and session.daily_break_end):
        return False
    start, end = _parse_time(session.daily_break_start), _parse_time(session.daily_break_end)
    if start is None or end is None:
        return False
    if (curr - prev) > timedelta(hours=24):
        return False
    # The break may straddle midnight; compare on the wall clock only.
    after_close = prev.time() >= _minus_epsilon(start) or start == time(0, 0)
    before_open = curr.time() <= end
    return after_close and before_open and curr.date() >= prev.date()


def _parse_time(text: str) -> time | None:
    try:
        hour, minute = (int(part) for part in text.split(":")[:2])
        return time(hour, minute)
    except (ValueError, TypeError):
        return None


def _minus_epsilon(value: time) -> time:
    """One minute before *value*, clamped at midnight."""
    total = value.hour * 60 + value.minute - 1
    return time(0, 0) if total < 0 else time(total // 60, total % 60)


def diagnose_all(
    config: Config, timeframes: list[str] | None = None
) -> dict[str, BarDiagnostics]:
    """Diagnose every timeframe that has bars on disk."""
    selected = timeframes or list(config.resampling.timeframes)
    results = {}
    for timeframe in selected:
        LOGGER.info("Diagnosing %s bars", timeframe)
        results[timeframe] = diagnose_timeframe(config, timeframe)
    return results


def write_diagnostics(
    config: Config, results: dict[str, BarDiagnostics], name: str = DIAGNOSTICS_NAME
) -> Path:
    """Persist a combined diagnostics document under ``metadata_path``."""
    payload = {
        "generated_utc": utc_now_iso(),
        "instrument": config.instrument,
        "config_fingerprint": config.fingerprint(),
        "bar_timestamp_convention": config.resampling.label,
        "timezone": config.timezone.describe(),
        "timeframes": {tf: r.to_dict() for tf, r in results.items()},
    }
    path = ensure_dir(config.metadata_path) / name
    atomic_write_text(path, json.dumps(payload, indent=2, default=str) + "\n")
    LOGGER.info("Bar diagnostics written: %s", path)
    return path
