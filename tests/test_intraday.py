"""Time-of-day, weekday and session grouping."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from xauusd_quant.research.intraday import (
    assign_sessions,
    hourly_analysis,
    session_analysis,
    weekday_analysis,
)


def frame_at(stamps: list[datetime], returns: list[float] | None = None) -> pl.DataFrame:
    n = len(stamps)
    return pl.DataFrame({
        "timestamp": stamps,
        "ret_log": returns if returns is not None else [0.001] * n,
        "mean_spread": [0.30] * n,
        "tick_count": [100] * n,
    })


# ---------------------------------------------------------------------------
# Hour assignment
# ---------------------------------------------------------------------------
def test_each_bar_lands_in_the_hour_of_its_timestamp(research_config):
    stamps = [datetime(2024, 5, 1, hour, 30) for hour in range(24)]
    result = hourly_analysis(
        frame_at(stamps), return_column="ret_log", config=research_config,
        timezone_description="broker time",
    )
    assert result.table["hour"].to_list() == list(range(24))
    assert result.table["bars"].to_list() == [1] * 24


def test_hour_boundaries_are_assigned_correctly(research_config):
    """59:59.999 belongs to the hour, 00:00.000 to the next."""
    stamps = [
        datetime(2024, 5, 1, 9, 59, 59, 999000),
        datetime(2024, 5, 1, 10, 0, 0),
        datetime(2024, 5, 1, 10, 59, 59, 999000),
        datetime(2024, 5, 1, 11, 0, 0),
    ]
    result = hourly_analysis(
        frame_at(stamps), return_column="ret_log", config=research_config,
        timezone_description="broker time",
    )
    counts = dict(zip(result.table["hour"].to_list(),
                      result.table["bars"].to_list(), strict=True))
    assert counts == {9: 1, 10: 2, 11: 1}


def test_hourly_output_records_its_time_basis(research_config):
    result = hourly_analysis(
        frame_at([datetime(2024, 5, 1, 3, 0)]), return_column="ret_log",
        config=research_config, timezone_description="America/New_York +7h",
    )
    assert result.time_basis == research_config.intraday.time_basis
    assert "America/New_York +7h" in result.timezone_description
    assert any("clock" in note for note in result.notes)


def test_hourly_statistics_are_computed_per_hour(research_config):
    stamps, returns = [], []
    for hour, value in ((1, 0.01), (2, -0.02)):
        for minute in range(0, 60, 5):
            stamps.append(datetime(2024, 5, 1, hour, minute))
            returns.append(value)
    result = hourly_analysis(
        frame_at(stamps, returns), return_column="ret_log", config=research_config,
        timezone_description="broker time",
    )
    table = result.table.sort("hour")
    assert table["mean_return"].to_list() == pytest.approx([0.01, -0.02])
    assert table["mean_abs_return"].to_list() == pytest.approx([0.01, 0.02])
    assert "mean_spread" in table.columns
    assert "mean_tick_count" in table.columns


# ---------------------------------------------------------------------------
# Weekday
# ---------------------------------------------------------------------------
def test_weekdays_are_named_and_numbered_correctly(research_config):
    # 2024-05-06 is a Monday.
    stamps = [datetime(2024, 5, 6) + timedelta(days=i) for i in range(7)]
    result = weekday_analysis(
        frame_at(stamps), return_column="ret_log", config=research_config,
        timezone_description="broker time",
    )
    pairs = list(zip(result.table["weekday_number"].to_list(),
                     result.table["weekday"].to_list(), strict=True))
    assert pairs == [
        (1, "Monday"), (2, "Tuesday"), (3, "Wednesday"), (4, "Thursday"),
        (5, "Friday"), (6, "Saturday"), (7, "Sunday"),
    ]


def test_weekend_bars_are_reported_not_dropped(research_config):
    """XAUUSD is normally closed at the weekend, so their presence is a finding."""
    stamps = [datetime(2024, 5, 6), datetime(2024, 5, 11), datetime(2024, 5, 12)]
    result = weekday_analysis(
        frame_at(stamps), return_column="ret_log", config=research_config,
        timezone_description="broker time",
    )
    assert 6 in result.table["weekday_number"].to_list()
    assert any("Saturday or Sunday" in w for w in result.warnings)


def test_weekday_analysis_without_weekend_data_gives_no_warning(research_config):
    stamps = [datetime(2024, 5, 6) + timedelta(days=i) for i in range(5)]
    result = weekday_analysis(
        frame_at(stamps), return_column="ret_log", config=research_config,
        timezone_description="broker time",
    )
    assert not any("Saturday" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("hour", "minute", "expected"),
    [
        (3, 0, "asia"),
        (9, 59, "asia"),
        (10, 0, "london"),
        (15, 0, "london"),
        (15, 30, "london_ny_overlap"),   # overlap wins over london and new_york
        (18, 0, "london_ny_overlap"),
        (18, 30, "new_york"),
        (22, 59, "new_york"),
        (23, 0, "off_hours"),
        (0, 30, "off_hours"),            # the window wraps midnight
        (1, 59, "off_hours"),
        (2, 0, "asia"),
    ],
)
def test_session_boundaries(research_config, hour, minute, expected):
    frame = pl.DataFrame({"timestamp": [datetime(2024, 5, 1, hour, minute)]})
    label = frame.select(
        assign_sessions(pl.col("timestamp"), research_config).alias("s")
    )["s"][0]
    assert label == expected


def test_overlap_session_takes_precedence(research_config):
    """london_ny_overlap deliberately overlaps both parents; first match wins."""
    frame = pl.DataFrame({"timestamp": [datetime(2024, 5, 1, 16, 0)]})
    assert frame.select(
        assign_sessions(pl.col("timestamp"), research_config).alias("s")
    )["s"][0] == "london_ny_overlap"


def test_session_analysis_states_its_boundaries_and_dst_limitation(research_config):
    stamps = [datetime(2024, 5, 1, h, 0) for h in range(24)]
    result = session_analysis(
        frame_at(stamps), return_column="ret_log", config=research_config,
        timezone_description="broker time",
    )
    assert not result.table.is_empty()
    assert any("Boundaries used" in note for note in result.notes)
    assert any("daylight-saving" in w.lower() for w in result.warnings)


def test_every_bar_gets_a_session_label(research_config):
    stamps = [datetime(2024, 5, 1) + timedelta(minutes=15 * i) for i in range(96)]
    frame = pl.DataFrame({"timestamp": stamps}).with_columns(
        assign_sessions(pl.col("timestamp"), research_config).alias("session")
    )
    assert frame["session"].null_count() == 0
    assert "unclassified" not in frame["session"].to_list(), (
        "the configured sessions should tile the whole day"
    )


def test_session_analysis_can_be_disabled(research_config_factory):
    config = research_config_factory(intraday={"sessions": {"enabled": False}})
    result = session_analysis(
        frame_at([datetime(2024, 5, 1, 3, 0)]), return_column="ret_log",
        config=config, timezone_description="broker time",
    )
    assert result.table.is_empty()
    assert any("disabled" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Time basis
# ---------------------------------------------------------------------------
def test_switching_the_time_basis_regroups_the_data(research_config_factory):
    """Grouping must follow the configured clock, not a hard-coded one."""
    config = research_config_factory(intraday={"time_basis": "timestamp_utc"})
    stamps = [datetime(2024, 5, 1, 10, 0)]
    frame = pl.DataFrame({
        "timestamp": stamps,
        "timestamp_utc": [datetime(2024, 5, 1, 7, 0)],   # broker time minus 3h
        "ret_log": [0.001],
    })
    result = hourly_analysis(
        frame, return_column="ret_log", config=config,
        timezone_description="UTC",
    )
    assert result.time_basis == "timestamp_utc"
    assert result.table["hour"].to_list() == [7], "must group on UTC, not broker time"


def test_missing_basis_falls_back_to_timestamp(research_config_factory):
    config = research_config_factory(intraday={"time_basis": "timestamp_utc"})
    result = hourly_analysis(
        frame_at([datetime(2024, 5, 1, 10, 0)]), return_column="ret_log",
        config=config, timezone_description="broker time",
    )
    assert result.time_basis == "timestamp"


def test_sessions_partition_the_day_exactly_once(research_config):
    """Every minute of the day belongs to exactly one configured session."""
    stamps = [datetime(2024, 5, 1) + timedelta(minutes=i) for i in range(24 * 60)]
    frame = pl.DataFrame({"timestamp": stamps}).with_columns(
        assign_sessions(pl.col("timestamp"), research_config).alias("session")
    )
    counts = frame.group_by("session").len().sort("session")
    assert counts["len"].sum() == 24 * 60
    assert "unclassified" not in counts["session"].to_list()
    # The overlap must not have been swallowed by its two parent windows.
    assert "london_ny_overlap" in counts["session"].to_list()
    overlap = counts.filter(pl.col("session") == "london_ny_overlap")["len"][0]
    assert overlap == 180, "15:30-18:30 is three hours"
