"""Timezone policy.

The anchored-DST cases encode the conclusion reached from the real file:
broker time = America/New_York wall clock + 7 h, i.e. UTC+2 in US winter and
UTC+3 in US summer, switching on US dates. If a future change breaks that, these
tests say so loudly.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from xauusd_quant.data.timezones import describe_policy, to_utc_expr, utc_offset_at
from xauusd_quant.utils.config import TimezoneConfig

BROKER = TimezoneConfig(
    source_label="broker_server_time", mode="anchored_dst",
    anchor_tz="America/New_York", offset_hours=7, emit_utc_column=True,
)


def convert(cfg: TimezoneConfig, stamps: list[datetime]) -> list[datetime]:
    frame = pl.DataFrame({"timestamp": pl.Series(stamps, dtype=pl.Datetime("us"))})
    utc = frame.select(to_utc_expr(cfg))["timestamp_utc"]
    return [None if v is None else v.replace(tzinfo=None) for v in utc]


# ---------------------------------------------------------------------------
# anchored_dst
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("local", "expected_utc", "why"),
    [
        (datetime(2021, 1, 4, 1, 0, 0), datetime(2021, 1, 3, 23, 0, 0),
         "US winter -> UTC+2; this is the real file's first tick"),
        (datetime(2024, 7, 8, 15, 30, 0), datetime(2024, 7, 8, 12, 30, 0),
         "US summer -> UTC+3; 08:30 New York data release"),
        (datetime(2024, 1, 8, 15, 30, 0), datetime(2024, 1, 8, 13, 30, 0),
         "US winter -> UTC+2; same release, other half of the year"),
        (datetime(2021, 11, 3, 15, 30, 0), datetime(2021, 11, 3, 12, 30, 0),
         "EU already off DST, US still on -> must follow the US calendar"),
        (datetime(2022, 3, 15, 15, 30, 0), datetime(2022, 3, 15, 12, 30, 0),
         "US on DST, EU not yet -> must follow the US calendar"),
    ],
)
def test_anchored_dst_conversion(local, expected_utc, why):
    assert convert(BROKER, [local]) == [expected_utc], why


def test_us_release_lands_at_the_same_local_time_year_round():
    """The property that identified this timezone in the first place."""
    winter, summer = convert(
        BROKER, [datetime(2024, 1, 8, 15, 30), datetime(2024, 7, 8, 15, 30)]
    )
    assert winter.hour == 13 and summer.hour == 12  # 08:30 EST and 08:30 EDT


def test_dst_switch_happens_on_us_dates_only():
    """Between the EU and US switch dates the offset must still be the US one."""
    before_eu = convert(BROKER, [datetime(2021, 10, 25, 12, 0)])[0]
    between = convert(BROKER, [datetime(2021, 11, 2, 12, 0)])[0]
    after_us = convert(BROKER, [datetime(2021, 11, 9, 12, 0)])[0]
    assert (datetime(2021, 10, 25, 12, 0) - before_eu).total_seconds() / 3600 == 3
    assert (datetime(2021, 11, 2, 12, 0) - between).total_seconds() / 3600 == 3
    assert (datetime(2021, 11, 9, 12, 0) - after_us).total_seconds() / 3600 == 2


def test_utc_offset_helper_agrees_with_the_expression():
    for moment in (datetime(2021, 1, 4, 1, 0), datetime(2024, 7, 8, 15, 30)):
        converted = convert(BROKER, [moment])[0]
        assert utc_offset_at(BROKER, moment) == pytest.approx(
            (moment - converted).total_seconds() / 3600
        )


# ---------------------------------------------------------------------------
# Other modes
# ---------------------------------------------------------------------------
def test_fixed_offset_has_no_dst():
    cfg = TimezoneConfig(mode="fixed_offset", fixed_offset_hours=2, emit_utc_column=True)
    winter, summer = convert(cfg, [datetime(2024, 1, 8, 12, 0), datetime(2024, 7, 8, 12, 0)])
    assert winter == datetime(2024, 1, 8, 10, 0)
    assert summer == datetime(2024, 7, 8, 10, 0)


def test_iana_mode_follows_that_zones_calendar():
    cfg = TimezoneConfig(mode="iana", iana_tz="Europe/London", emit_utc_column=True)
    winter, summer = convert(cfg, [datetime(2024, 1, 8, 12, 0), datetime(2024, 7, 8, 12, 0)])
    assert winter == datetime(2024, 1, 8, 12, 0)   # GMT
    assert summer == datetime(2024, 7, 8, 11, 0)   # BST


def test_naive_mode_refuses_to_invent_a_zone():
    cfg = TimezoneConfig(mode="naive")
    with pytest.raises(ValueError, match="naive"):
        to_utc_expr(cfg)
    with pytest.raises(ValueError, match="naive"):
        utc_offset_at(cfg, datetime(2021, 1, 4))


def test_fractional_offsets_are_supported():
    cfg = TimezoneConfig(mode="fixed_offset", fixed_offset_hours=5.5, emit_utc_column=True)
    assert convert(cfg, [datetime(2024, 1, 8, 12, 0)]) == [datetime(2024, 1, 8, 6, 30)]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def test_policy_description_is_recorded_for_reports():
    policy = describe_policy(BROKER, evidence="measured from data-release spikes")
    payload = policy.to_dict()
    assert payload["mode"] == "anchored_dst"
    assert payload["determined"] is True
    assert "America/New_York" in payload["description"]
    assert payload["evidence"]


def test_naive_policy_is_reported_as_undetermined():
    payload = describe_policy(TimezoneConfig(mode="naive")).to_dict()
    assert payload["determined"] is False
    assert payload["emits_utc_column"] is False


def test_conversion_preserves_sub_second_precision():
    local = datetime(2021, 1, 4, 1, 0, 0, 413000)
    assert convert(BROKER, [local])[0] == datetime(2021, 1, 3, 23, 0, 0, 413000)
