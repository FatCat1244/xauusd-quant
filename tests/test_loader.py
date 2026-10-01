"""DuckDB query layer, dataset metadata and bar diagnostics."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from xauusd_quant.data.converter import TickConverter
from xauusd_quant.data.diagnostics import diagnose_all, diagnose_timeframe, write_diagnostics
from xauusd_quant.data.loader import DataStore, TimeRangeError, TooManyRowsError
from xauusd_quant.data.metadata import build_dataset_metadata, write_dataset_metadata
from xauusd_quant.data.resampler import BarResampler


@pytest.fixture
def built(clean_config):
    """A fully converted dataset with 1m and 5m bars built."""
    TickConverter(clean_config).run()
    BarResampler(clean_config).build_all()
    return clean_config


# ---------------------------------------------------------------------------
# Range queries
# ---------------------------------------------------------------------------
def test_load_ticks_returns_only_the_requested_range(built):
    with DataStore(built) as store:
        frame = store.load_ticks("2021-01-04 01:00:00", "2021-01-04 01:00:10")
    assert frame.height == 100          # 10 s at one tick per 100 ms
    assert frame["timestamp"].min() >= datetime(2021, 1, 4, 1, 0, 0)
    assert frame["timestamp"].max() < datetime(2021, 1, 4, 1, 0, 10)


def test_the_range_is_half_open_so_chunks_tile_exactly(built):
    with DataStore(built) as store:
        first = store.load_ticks("2021-01-04 01:00:00", "2021-01-04 01:00:30")
        second = store.load_ticks("2021-01-04 01:00:30", "2021-01-04 01:01:00")
        whole = store.load_ticks("2021-01-04 01:00:00", "2021-01-04 01:01:00")
    assert first.height + second.height == whole.height
    assert set(first["timestamp"]).isdisjoint(set(second["timestamp"]))


def test_column_projection(built):
    with DataStore(built) as store:
        frame = store.load_ticks("2021-01-04", "2021-01-05", ["timestamp", "bid"])
    assert frame.columns == ["timestamp", "bid"]


def test_unsafe_column_names_are_refused(built):
    with DataStore(built) as store, pytest.raises(ValueError, match="unsafe column"):
        store.load_ticks("2021-01-04", "2021-01-05", ["bid; DROP TABLE x"])


def test_load_bars_for_a_timeframe(built):
    with DataStore(built) as store:
        bars = store.load_bars("1m", "2021-01-04", "2021-01-05")
    assert bars.height > 0
    assert "open" in bars.columns and "tick_count" in bars.columns
    assert bars["timestamp"].is_sorted()


def test_unknown_timeframe_lists_what_is_available(built):
    with DataStore(built) as store, pytest.raises(FileNotFoundError, match="Available"):
        store.load_bars("4h", "2021-01-04", "2021-01-05")


def test_inverted_or_unparseable_ranges_are_rejected(built):
    with DataStore(built) as store:
        with pytest.raises(TimeRangeError, match="strictly after"):
            store.load_ticks("2021-01-05", "2021-01-04")
        with pytest.raises(TimeRangeError, match="not a recognised"):
            store.load_ticks("yesterday", "today")


def test_row_guard_blocks_an_oversized_query(built):
    store = DataStore(built)
    object.__setattr__(store.config.duckdb, "max_rows_without_override", 10)
    with pytest.raises(TooManyRowsError, match="guard"):
        store.load_ticks("2021-01-01", "2022-01-01")
    # The opt-in path still works.
    assert store.load_ticks("2021-01-01", "2022-01-01", allow_large=True).height == 600
    store.close()


def test_limit_below_the_guard_is_allowed(built):
    store = DataStore(built)
    object.__setattr__(store.config.duckdb, "max_rows_without_override", 10)
    assert store.load_ticks("2021-01-01", "2022-01-01", limit=5).height == 5
    store.close()


def test_iter_ticks_streams_chunks_covering_everything(built):
    with DataStore(built) as store:
        chunks = list(store.iter_ticks("2021-01-04", "2021-01-06", chunk="1d"))
    assert sum(c.height for c in chunks) == 600
    assert all(c["timestamp"].is_sorted() for c in chunks)


def test_scan_ticks_is_lazy(built):
    import polars as pl

    with DataStore(built) as store:
        lazy = store.scan_ticks("2021-01-04", "2021-01-05")
        assert isinstance(lazy, pl.LazyFrame)
        assert lazy.collect().height == 600


def test_sql_views_are_registered(built):
    with DataStore(built) as store:
        assert store.sql("SELECT count(*) AS n FROM ticks")["n"][0] == 600
        assert store.sql("SELECT count(*) AS n FROM bars_1m")["n"][0] > 0


def test_discovery_helpers(built):
    with DataStore(built) as store:
        assert store.has_ticks()
        assert set(store.available_timeframes()) == {"1m", "5m"}
        lo, hi = store.tick_span()
        assert lo == datetime(2021, 1, 4, 1, 0, 0)
        assert hi < datetime(2021, 1, 4, 1, 1, 0)


# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------
def test_dataset_metadata_reports_exact_full_dataset_figures(built):
    meta = build_dataset_metadata(built)
    assert meta.total_rows == 600
    assert meta.start_timestamp.startswith("2021-01-04 01:00:00")
    assert meta.trading_days == 1
    assert meta.files == 1 and meta.partitions == 1
    assert meta.storage_bytes > 0
    assert meta.price_min is not None and meta.price_max is not None
    assert meta.coverage["row_stats_exact"] is True


def test_metadata_folds_in_the_conversion_reports(built):
    meta = build_dataset_metadata(built)
    assert meta.cleaning["rows_input"] == 600
    assert meta.cleaning["rows_dropped"] == 0
    assert meta.coverage["validation_counts_cover_full_raw_pass"] is True
    assert not meta.warnings


def test_metadata_warns_when_the_dataset_is_incomplete(two_month_csv, config_factory,
                                                      monkeypatch):
    """An interrupted conversion leaves an incomplete dataset; metadata must say so."""
    config = config_factory(two_month_csv)
    original = TickConverter._convert_partition

    def fail_on_march(self, key, *args, **kwargs):
        if key == "2021-03":
            raise KeyboardInterrupt
        return original(self, key, *args, **kwargs)

    monkeypatch.setattr(TickConverter, "_convert_partition", fail_on_march)
    with pytest.raises(KeyboardInterrupt):
        TickConverter(config).run()
    monkeypatch.undo()

    meta = build_dataset_metadata(config)
    assert meta.dataset_status == "incomplete"
    assert meta.coverage["validation_counts_cover_full_raw_pass"] is False
    assert any("not complete" in w and "2021-03" in w for w in meta.warnings)
    assert any("only part of the raw input" in w for w in meta.warnings)


def test_metadata_labels_the_quantile_method(built):
    meta = build_dataset_metadata(built)
    assert "approx" in meta.quantile_method
    assert meta.spread_percentiles["p50"] == pytest.approx(0.30, abs=1e-6)


def test_metadata_is_written_as_json(built):
    meta = build_dataset_metadata(built)
    path = write_dataset_metadata(built, meta)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["total_rows"] == 600
    assert payload["timezone"]["mode"] == "anchored_dst"
    assert payload["timezone"]["evidence"]


def test_metadata_without_a_converted_dataset_raises(clean_config):
    with pytest.raises(FileNotFoundError, match="Run `xq convert`"):
        build_dataset_metadata(clean_config)


# ---------------------------------------------------------------------------
# Bar diagnostics
# ---------------------------------------------------------------------------
def test_diagnostics_describe_a_contiguous_dataset(built):
    report = diagnose_timeframe(built, "1m")
    assert report.bars == 1                       # 600 ticks * 100 ms = 60 s
    assert report.gap_count == 0
    assert report.missing_slots == 0
    assert report.mean_ticks_per_bar == pytest.approx(600.0)
    assert report.spread_distribution["mean"] == pytest.approx(0.30, abs=1e-9)


def test_diagnostics_find_and_classify_gaps(two_month_csv, config_factory):
    config = config_factory(two_month_csv, resampling={"timeframes": ["1m"]})
    TickConverter(config).run()
    BarResampler(config).build("1m")
    report = diagnose_timeframe(config, "1m")

    assert report.gap_count >= 2
    assert report.missing_slots > 0
    assert sum(report.gaps_by_category.values()) == report.gap_count
    assert report.coverage_ratio < 1.0


def test_diagnostics_never_fill_missing_bars(two_month_csv, config_factory):
    import polars as pl

    config = config_factory(two_month_csv, resampling={"timeframes": ["1m"]})
    TickConverter(config).run()
    BarResampler(config).build("1m")
    before = pl.read_parquet(sorted(config.bars_dir("1m").rglob("*.parquet"))).height
    diagnose_timeframe(config, "1m")
    after = pl.read_parquet(sorted(config.bars_dir("1m").rglob("*.parquet"))).height
    assert before == after
    assert any("never filled" in note for note in diagnose_timeframe(config, "1m").notes)


def test_diagnostics_without_bars_warns_instead_of_raising(clean_config):
    report = diagnose_timeframe(clean_config, "1m")
    assert report.bars == 0
    assert any("Run `xq build-bars`" in w for w in report.warnings)


def test_diagnostics_are_written_as_json(built):
    results = diagnose_all(built, ["1m", "5m"])
    path = write_diagnostics(built, results)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload["timeframes"]) == {"1m", "5m"}
    assert payload["bar_timestamp_convention"] == "open"


# ---------------------------------------------------------------------------
# Gap classification
# ---------------------------------------------------------------------------
def test_gap_categories_separate_weekends_holidays_and_anomalies():
    """Each kind of closure must land in its own bucket."""
    from datetime import datetime as dt

    from xauusd_quant.data.diagnostics import _classify
    from xauusd_quant.utils.config import SessionConfig

    session = SessionConfig(
        daily_break_start="00:00", daily_break_end="01:00",
        week_open_weekday=1, week_open_time="01:00",
        week_close_weekday=5, week_close_time="23:59",
    )
    cases = {
        # Friday 23:59 -> Monday 01:00 : the ordinary weekly close
        "weekend": (dt(2021, 11, 5, 23, 59), dt(2021, 11, 8, 1, 0)),
        # Thursday 23:58 -> Monday 01:00 : Christmas, an extended closure
        "holiday_weekend": (dt(2021, 12, 23, 23, 58), dt(2021, 12, 27, 1, 0)),
        # Tuesday 23:59 -> Wednesday 01:00 : the nightly maintenance break
        "daily_break": (dt(2021, 11, 2, 23, 59), dt(2021, 11, 3, 1, 0)),
        # Monday 14:00 -> Monday 17:00 : a hole in the middle of a session
        "holiday_or_unknown": (dt(2021, 11, 8, 14, 0), dt(2021, 11, 8, 17, 0)),
    }
    for expected, (start, end) in cases.items():
        assert _classify(start, end, session) == expected, f"{start} -> {end}"


def test_holiday_weekends_do_not_hide_intraday_gaps(two_month_csv, config_factory):
    """Long expected closures must not crowd out the gaps worth investigating."""
    config = config_factory(two_month_csv, resampling={"timeframes": ["1h"]})
    TickConverter(config).run()
    BarResampler(config).build("1h")
    report = diagnose_timeframe(config, "1h")
    assert sum(report.gaps_by_category.values()) == report.gap_count
    for gap in report.largest_unexplained_gaps:
        assert gap["missing_bars"] > 0


# ---------------------------------------------------------------------------
# Result shape and timezone rendering
# ---------------------------------------------------------------------------
def test_loaders_return_the_canonical_columns_without_hive_artifacts(built):
    """`year`/`month` exist only as directory names; they are not tick data."""
    with DataStore(built) as store:
        ticks = store.load_ticks("2021-01-04", "2021-01-05")
        bars = store.load_bars("1m", "2021-01-04", "2021-01-05")
    assert ticks.columns == [
        "timestamp", "timestamp_utc", "bid", "ask", "mid", "spread", "volume"
    ]
    assert "year" not in bars.columns and "month" not in bars.columns
    assert bars.columns[0] == "timestamp"


def test_hive_columns_are_still_selectable_by_name(built):
    with DataStore(built) as store:
        frame = store.load_ticks("2021-01-04", "2021-01-05", ["timestamp", "year", "month"])
    assert frame.columns == ["timestamp", "year", "month"]
    assert frame["year"].unique().to_list() == [2021]


def test_utc_column_renders_as_utc_not_the_machine_timezone(built):
    """Otherwise the same query prints different times on different machines."""
    with DataStore(built) as store:
        frame = store.load_ticks("2021-01-04", "2021-01-05", limit=1)
    dtype = frame.schema["timestamp_utc"]
    assert dtype.time_zone == "UTC", f"rendered in {dtype.time_zone}"
    # January is UTC+2 under the anchored-DST policy.
    row = frame.row(0, named=True)
    offset = row["timestamp"] - row["timestamp_utc"].replace(tzinfo=None)
    assert offset.total_seconds() / 3600 == 2


def test_metadata_reports_the_canonical_schema_without_hive_columns(built):
    """Directory names are not data; they must not appear as columns or nulls."""
    meta = build_dataset_metadata(built)
    assert meta.columns == [
        "timestamp", "timestamp_utc", "bid", "ask", "mid", "spread", "volume"
    ]
    assert set(meta.null_counts) == set(meta.columns)
    assert all(v == 0 for v in meta.null_counts.values())
