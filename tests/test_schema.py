"""Canonical schema, column detection and derived-column correctness."""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from xauusd_quant.data.schema import (
    ASK,
    BID,
    MID,
    SPREAD,
    TIMESTAMP,
    canonical_schema,
    derive_expressions,
    detect_columns,
    detect_timestamp_format,
)
from xauusd_quant.utils.config import CanonicalConfig, InputConfig


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------
def test_detects_the_real_dataset_header():
    detection = detect_columns(
        ["DateTime", "Bid", "Ask", "Volume"], ["20210104 01:00:00.413"]
    )
    assert detection.mapping == {
        "timestamp": "DateTime", "bid": "Bid", "ask": "Ask", "volume": "Volume"
    }
    assert detection.timestamp_format == "%Y%m%d %H:%M:%S%.f"
    assert detection.timestamp_lexicographic is True
    assert detection.is_usable
    assert detection.missing() == ()


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        (["timestamp", "bid", "ask"], {"timestamp", "bid", "ask"}),
        (["Time", "BidPrice", "AskPrice", "TickVolume"],
         {"timestamp", "bid", "ask", "volume"}),
        (["gmt time", "Bid", "Offer", "Spread"], {"timestamp", "bid", "ask", "spread"}),
        (["DATE_TIME", "BID", "ASK"], {"timestamp", "bid", "ask"}),
    ],
)
def test_detection_handles_header_variants(headers, expected):
    assert set(detect_columns(headers).mapping) == expected


def test_detection_is_case_insensitive_and_strips_bom():
    detection = detect_columns(["﻿DateTime", " bid ", "ASK"])
    assert detection.mapping["timestamp"] == "DateTime"
    assert detection.mapping["bid"] == "bid"


def test_detection_reports_unmapped_and_missing_columns():
    detection = detect_columns(["when", "price", "flags"])
    assert not detection.is_usable
    assert set(detection.missing()) == {"timestamp", "bid", "ask"}
    assert "flags" in detection.unmapped_headers


def test_each_source_column_is_claimed_once():
    """'Bid' must not be consumed by both the bid and ask patterns."""
    detection = detect_columns(["DateTime", "Bid", "Ask"])
    sources = [c.source for c in detection.columns]
    assert len(sources) == len(set(sources))


def test_config_fragment_round_trips_into_input_config():
    detection = detect_columns(["DateTime", "Bid", "Ask", "Volume"],
                               ["20210104 01:00:00.413"])
    fragment = detection.to_config_fragment()
    inp = InputConfig(**{k: v for k, v in fragment.items() if v is not None})
    assert inp.timestamp_column == "DateTime"
    assert inp.source_columns == {
        "timestamp": "DateTime", "bid": "Bid", "ask": "Ask", "volume": "Volume"
    }


# ---------------------------------------------------------------------------
# Timestamp format detection
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("sample", "expected"),
    [
        ("20210104 01:00:00.413", "%Y%m%d %H:%M:%S%.f"),
        ("20210104 01:00:00", "%Y%m%d %H:%M:%S"),
        ("2021-01-04 01:00:00.413", "%Y-%m-%d %H:%M:%S%.f"),
        ("2021-01-04T01:00:00", "%Y-%m-%dT%H:%M:%S"),
        ("2021.01.04 01:00:00", "%Y.%m.%d %H:%M:%S"),
        ("04/01/2021 01:00:00", "%d/%m/%Y %H:%M:%S"),
        ("nonsense", None),
    ],
)
def test_timestamp_format_detection(sample, expected):
    assert detect_timestamp_format([sample]) == expected


def test_non_lexicographic_format_is_flagged():
    """A day-first format cannot be byte-offset seeked, so resume must not use it."""
    detection = detect_columns(["DateTime", "Bid", "Ask"], ["04/01/2021 01:00:00"])
    assert detection.timestamp_format == "%d/%m/%Y %H:%M:%S"
    assert detection.timestamp_lexicographic is False


def test_mixed_samples_reject_a_format():
    assert detect_timestamp_format(["20210104 01:00:00", "2021-01-04 01:00:00"]) is None


# ---------------------------------------------------------------------------
# Canonical schema
# ---------------------------------------------------------------------------
def test_canonical_schema_contains_derived_columns():
    schema = canonical_schema(InputConfig(), CanonicalConfig(), with_utc=True)
    assert list(schema) == [
        "timestamp", "timestamp_utc", "bid", "ask", "mid", "spread", "volume"
    ]
    assert schema["timestamp"] == pl.Datetime("us")
    assert schema["timestamp_utc"].time_zone == "UTC"


def test_absent_source_columns_are_not_invented():
    """A feed with no volume must not gain a fabricated volume column."""
    inp = InputConfig(volume_column=None)
    schema = canonical_schema(inp, CanonicalConfig(), with_utc=False)
    assert "volume" not in schema
    assert "timestamp_utc" not in schema
    assert {"bid", "ask", "mid", "spread"} <= set(schema)


def test_trade_only_feed_gets_no_mid_or_spread():
    inp = InputConfig(bid_column=None, ask_column=None, volume_column="Volume")
    schema = canonical_schema(inp, CanonicalConfig(), with_utc=False)
    assert set(schema) == {"timestamp", "volume"}
    assert derive_expressions(inp, CanonicalConfig()) == []


def test_source_spread_column_is_kept_separately():
    """A source spread must never overwrite the derived one."""
    inp = InputConfig(spread_column="Spread")
    schema = canonical_schema(inp, CanonicalConfig(), with_utc=False)
    assert "spread" in schema and "spread_source" in schema


# ---------------------------------------------------------------------------
# Derived values
# ---------------------------------------------------------------------------
def test_mid_and_spread_are_exact():
    frame = pl.DataFrame({
        BID: [1904.998, 2000.0, 1500.5],
        ASK: [1905.366, 2000.5, 1500.5],
    }).with_columns(derive_expressions(InputConfig(), CanonicalConfig()))
    assert frame[MID].to_list() == pytest.approx([1905.182, 2000.25, 1500.5])
    assert frame[SPREAD].to_list() == pytest.approx([0.368, 0.5, 0.0])


def test_derivation_never_mutates_bid_or_ask():
    original = pl.DataFrame({BID: [1904.998], ASK: [1905.366]})
    derived = original.with_columns(derive_expressions(InputConfig(), CanonicalConfig()))
    assert derived[BID].to_list() == original[BID].to_list()
    assert derived[ASK].to_list() == original[ASK].to_list()


def test_spread_is_negative_when_the_quote_is_crossed():
    """Crossed quotes must surface as negative spread, not be silently clamped."""
    frame = pl.DataFrame({BID: [2000.5], ASK: [2000.1]}).with_columns(
        derive_expressions(InputConfig(), CanonicalConfig())
    )
    assert frame[SPREAD][0] == pytest.approx(-0.4)


def test_timestamp_parsing_matches_the_source_format():
    frame = pl.DataFrame({"raw": ["20210104 01:00:00.413", "20260911 02:59:59.698"]})
    parsed = frame.select(
        pl.col("raw").str.strptime(pl.Datetime("us"), format="%Y%m%d %H:%M:%S%.f")
        .alias(TIMESTAMP)
    )
    assert parsed[TIMESTAMP].to_list() == [
        datetime(2021, 1, 4, 1, 0, 0, 413000),
        datetime(2026, 9, 11, 2, 59, 59, 698000),
    ]


def test_unparseable_timestamps_become_null_not_an_exception():
    frame = pl.DataFrame({"raw": ["20210104 01:00:00.413", "garbage", ""]})
    parsed = frame.select(
        pl.col("raw").str.strptime(
            pl.Datetime("us"), format="%Y%m%d %H:%M:%S%.f", strict=False
        ).alias(TIMESTAMP)
    )
    assert parsed[TIMESTAMP].null_count() == 2
