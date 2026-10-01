"""Bar construction: OHLC correctness, boundaries and no look-ahead."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from conftest import canonical_ticks
from xauusd_quant.data.converter import TickConverter
from xauusd_quant.data.resampler import (
    BAR_COLUMNS,
    BarResampler,
    parse_timeframe,
    resample_ticks,
    validate_timeframes,
)

BASE = datetime(2021, 1, 4, 10, 0, 0)


# ---------------------------------------------------------------------------
# Timeframe parsing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "seconds"),
    [("1m", 60), ("5m", 300), ("15m", 900), ("30m", 1800), ("1h", 3600),
     ("30s", 30), ("1d", 86400)],
)
def test_parse_timeframe(text, seconds):
    assert parse_timeframe(text).total_seconds() == seconds


@pytest.mark.parametrize("text", ["", "m", "0m", "-5m", "5x", "1 m"])
def test_invalid_timeframes_are_rejected(text):
    with pytest.raises(ValueError):
        parse_timeframe(text)


def test_timeframes_that_do_not_divide_a_day_are_rejected():
    """Such a bar could straddle a month partition and be silently split."""
    validate_timeframes(["1m", "5m", "15m", "30m", "1h"])
    with pytest.raises(ValueError, match="does not divide a day evenly"):
        validate_timeframes(["7m"])
    with pytest.raises(ValueError, match="does not divide a day evenly"):
        validate_timeframes(["2d"])


# ---------------------------------------------------------------------------
# OHLC correctness
# ---------------------------------------------------------------------------
def test_ohlc_matches_the_ticks_in_the_bar():
    stamps = [BASE + timedelta(seconds=s) for s in (0, 10, 20, 30, 40)]
    bids = [100.0, 104.0, 98.0, 102.0, 101.0]
    asks = [b + 0.4 for b in bids]
    bars = resample_ticks(canonical_ticks(stamps, bids, asks), "1m")

    assert bars.height == 1
    row = bars.row(0, named=True)
    mids = [(b + a) / 2 for b, a in zip(bids, asks, strict=True)]
    assert row["open"] == pytest.approx(mids[0])
    assert row["high"] == pytest.approx(max(mids))
    assert row["low"] == pytest.approx(min(mids))
    assert row["close"] == pytest.approx(mids[-1])
    assert row["tick_count"] == 5


def test_open_is_first_and_close_is_last_in_time_order_not_file_order():
    """Unsorted input must still yield a chronologically correct bar."""
    stamps = [BASE + timedelta(seconds=s) for s in (30, 0, 50, 10)]
    bids = [103.0, 100.0, 105.0, 101.0]
    bars = resample_ticks(canonical_ticks(stamps, bids, [b + 0.4 for b in bids]), "1m")
    assert bars["open"][0] == pytest.approx(100.2)   # the tick at +0s
    assert bars["close"][0] == pytest.approx(105.2)  # the tick at +50s


def test_high_is_never_below_low_or_outside_open_close():
    stamps = [BASE + timedelta(seconds=s) for s in range(0, 300, 7)]
    bids = [100.0 + (i * 37 % 23) * 0.1 for i in range(len(stamps))]
    bars = resample_ticks(canonical_ticks(stamps, bids, [b + 0.4 for b in bids]), "1m")
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["high"] >= bars["open"]).all()
    assert (bars["high"] >= bars["close"]).all()
    assert (bars["low"] <= bars["open"]).all()
    assert (bars["low"] <= bars["close"]).all()


def test_ohlc_uses_the_configured_price_source():
    stamps = [BASE, BASE + timedelta(seconds=30)]
    frame = canonical_ticks(stamps, [100.0, 102.0], [100.4, 102.4])
    assert resample_ticks(frame, "1m", price_source="bid")["open"][0] == pytest.approx(100.0)
    assert resample_ticks(frame, "1m", price_source="ask")["open"][0] == pytest.approx(100.4)
    assert resample_ticks(frame, "1m", price_source="mid")["open"][0] == pytest.approx(100.2)


# ---------------------------------------------------------------------------
# Bar boundaries and look-ahead
# ---------------------------------------------------------------------------
def test_boundary_tick_belongs_to_the_next_bar():
    """A tick at exactly 10:01:00.000 must not be counted in the 10:00 bar."""
    stamps = [BASE, BASE + timedelta(seconds=59, milliseconds=999),
              BASE + timedelta(minutes=1)]
    bars = resample_ticks(
        canonical_ticks(stamps, [100.0, 101.0, 999.0], [100.4, 101.4, 999.4]), "1m"
    )
    assert bars.height == 2
    first = bars.row(0, named=True)
    assert first["tick_count"] == 2
    assert first["high"] == pytest.approx(101.2)
    assert first["close"] == pytest.approx(101.2), "the 10:01 tick must not leak backwards"
    assert bars.row(1, named=True)["open"] == pytest.approx(999.2)


def test_no_bar_contains_a_tick_outside_its_own_interval():
    stamps = [BASE + timedelta(seconds=s) for s in range(0, 900, 11)]
    bids = [100.0 + i * 0.01 for i in range(len(stamps))]
    bars = resample_ticks(canonical_ticks(stamps, bids, [b + 0.4 for b in bids]), "5m")
    step = timedelta(minutes=5)
    for row in bars.iter_rows(named=True):
        assert row["timestamp"] <= row["first_tick_timestamp"]
        assert row["last_tick_timestamp"] < row["timestamp"] + step


def test_bar_timestamps_sit_on_the_grid():
    stamps = [BASE + timedelta(seconds=s) for s in (17, 123, 456, 899)]
    bars = resample_ticks(
        canonical_ticks(stamps, [100.0] * 4, [100.4] * 4), "5m"
    )
    for stamp in bars["timestamp"]:
        assert stamp.second == 0 and stamp.microsecond == 0
        assert stamp.minute % 5 == 0


def test_label_close_shifts_stamps_without_changing_membership():
    stamps = [BASE, BASE + timedelta(seconds=30), BASE + timedelta(minutes=1)]
    frame = canonical_ticks(stamps, [100.0, 101.0, 102.0], [100.4, 101.4, 102.4])
    opened = resample_ticks(frame, "1m", label="open")
    closed = resample_ticks(frame, "1m", label="close")

    assert opened["tick_count"].to_list() == closed["tick_count"].to_list()
    assert opened["open"].to_list() == pytest.approx(closed["open"].to_list())
    assert closed["timestamp"][0] == opened["timestamp"][0] + timedelta(minutes=1)


def test_5m_bars_aggregate_exactly_five_1m_bars():
    stamps = [BASE + timedelta(seconds=s) for s in range(0, 300, 5)]
    bids = [100.0 + (i % 17) * 0.1 for i in range(len(stamps))]
    frame = canonical_ticks(stamps, bids, [b + 0.4 for b in bids])

    minute = resample_ticks(frame, "1m")
    five = resample_ticks(frame, "5m")

    assert five.height == 1
    row = five.row(0, named=True)
    assert row["open"] == pytest.approx(minute["open"][0])
    assert row["close"] == pytest.approx(minute["close"][-1])
    assert row["high"] == pytest.approx(minute["high"].max())
    assert row["low"] == pytest.approx(minute["low"].min())
    assert row["tick_count"] == minute["tick_count"].sum()


# ---------------------------------------------------------------------------
# Gaps are not filled
# ---------------------------------------------------------------------------
def test_empty_intervals_produce_no_bar():
    stamps = [BASE, BASE + timedelta(minutes=10)]
    bars = resample_ticks(canonical_ticks(stamps, [100.0, 105.0], [100.4, 105.4]), "1m")
    assert bars.height == 2, "the nine empty minutes must not be materialised"
    assert bars["timestamp"].to_list() == [BASE, BASE + timedelta(minutes=10)]


def test_single_tick_bar_is_valid_and_has_zero_range():
    bars = resample_ticks(canonical_ticks([BASE], [100.0], [100.4]), "1m")
    row = bars.row(0, named=True)
    assert row["tick_count"] == 1
    assert row["open"] == row["high"] == row["low"] == row["close"]


def test_min_ticks_per_bar_filters_thin_bars():
    stamps = [BASE, BASE + timedelta(seconds=5), BASE + timedelta(minutes=5)]
    frame = canonical_ticks(stamps, [100.0, 101.0, 102.0], [100.4, 101.4, 102.4])
    assert resample_ticks(frame, "1m", min_ticks_per_bar=2).height == 1


# ---------------------------------------------------------------------------
# Spread, volume and bid/ask edges
# ---------------------------------------------------------------------------
def test_spread_statistics_are_aggregated_per_bar():
    stamps = [BASE + timedelta(seconds=s) for s in (0, 10, 20, 30)]
    bids = [100.0] * 4
    asks = [100.2, 100.4, 100.6, 101.0]
    row = resample_ticks(canonical_ticks(stamps, bids, asks), "1m").row(0, named=True)
    assert row["min_spread"] == pytest.approx(0.2)
    assert row["max_spread"] == pytest.approx(1.0)
    assert row["mean_spread"] == pytest.approx(0.55)
    assert row["median_spread"] == pytest.approx(0.5)


def test_bid_ask_edges_are_preserved_for_realistic_fills():
    """The backtester will need entry/exit sides, so edges must survive."""
    stamps = [BASE, BASE + timedelta(seconds=30)]
    row = resample_ticks(
        canonical_ticks(stamps, [100.0, 102.0], [100.4, 102.8]), "1m"
    ).row(0, named=True)
    assert row["first_bid"] == pytest.approx(100.0)
    assert row["first_ask"] == pytest.approx(100.4)
    assert row["last_bid"] == pytest.approx(102.0)
    assert row["last_ask"] == pytest.approx(102.8)


def test_volume_is_summed_within_the_bar():
    stamps = [BASE, BASE + timedelta(seconds=30)]
    row = resample_ticks(
        canonical_ticks(stamps, [100.0, 101.0], [100.4, 101.4], [120.0, 380.0]), "1m"
    ).row(0, named=True)
    assert row["volume"] == pytest.approx(500.0)


def test_bars_expose_the_canonical_column_set():
    bars = resample_ticks(canonical_ticks([BASE], [100.0], [100.4]), "1m")
    assert bars.columns == [c for c in BAR_COLUMNS if c in bars.columns]
    assert set(bars.columns) == set(BAR_COLUMNS)


def test_feed_without_volume_still_builds_bars():
    frame = canonical_ticks([BASE], [100.0], [100.4]).drop("volume")
    bars = resample_ticks(frame, "1m")
    assert "volume" not in bars.columns
    assert bars.height == 1


def test_missing_price_source_raises_a_clear_error():
    frame = canonical_ticks([BASE], [100.0], [100.4]).drop("mid")
    with pytest.raises(KeyError, match="price_source"):
        resample_ticks(frame, "1m", price_source="mid")


# ---------------------------------------------------------------------------
# Full pipeline: CSV -> Parquet -> bars
# ---------------------------------------------------------------------------
def test_bars_built_from_converted_parquet(clean_config):
    TickConverter(clean_config).run()
    resampler = BarResampler(clean_config)
    result = resampler.build("1m")

    assert result.bars > 0
    assert result.ticks_read == 600
    files = sorted(clean_config.bars_dir("1m").rglob("*.parquet"))
    assert files and files[0].parent.name == "year=2021"

    bars = pl.read_parquet(files)
    assert bars["timestamp"].is_sorted()
    assert bars["tick_count"].sum() == 600


def test_build_all_covers_every_configured_timeframe(clean_config):
    TickConverter(clean_config).run()
    results = BarResampler(clean_config).build_all()
    assert set(results) == {"1m", "5m"}
    assert results["1m"].bars >= results["5m"].bars
    assert results["1m"].ticks_read == results["5m"].ticks_read == 600


def test_bar_files_record_their_conventions(clean_config):
    import pyarrow.parquet as pq

    TickConverter(clean_config).run()
    BarResampler(clean_config).build("5m")
    path = next(clean_config.bars_dir("5m").rglob("*.parquet"))
    metadata = pq.read_schema(path).metadata
    assert metadata[b"xq_timeframe"].decode() == "5m"
    assert metadata[b"xq_bar_label"].decode() == "open"
    assert metadata[b"xq_bar_interval"].decode() == "[t, t+D)"
    assert metadata[b"xq_price_source"].decode() == "mid"


def test_month_boundary_bars_are_not_split(two_month_csv, config_factory):
    """Building per partition must not chop a bar in half at a month edge."""
    config = config_factory(two_month_csv)
    TickConverter(config).run()
    BarResampler(config).build("1m")
    bars = pl.read_parquet(sorted(config.bars_dir("1m").rglob("*.parquet"))).sort("timestamp")
    assert bars["timestamp"].n_unique() == bars.height, "duplicate bar timestamps"


def test_building_bars_without_ticks_raises(clean_config):
    with pytest.raises(FileNotFoundError, match="Run `xq convert` first"):
        BarResampler(clean_config).build("1m")
