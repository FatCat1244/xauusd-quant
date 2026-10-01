"""Independent verification of a converted dataset.

`verify_tick_dataset` must find problems without trusting the converter: a
tampered file, a file the manifest does not know, a month that the source has
but the output lacks. It must also report genuine gaps in history without
calling them pipeline failures.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from conftest import make_clean_rows, write_csv
from xauusd_quant.data.converter import TickConverter
from xauusd_quant.data.dataset_validation import verify_tick_dataset, write_verification


def months(tmp_path, *spec):
    rows = []
    for year, month in spec:
        rows += make_clean_rows(datetime(year, month, 5, 1, 0, 0), 200, bid0=1900.0 + month)
    return write_csv(tmp_path / "m.csv", rows)


def test_a_clean_conversion_passes(tmp_path, config_factory):
    config = config_factory(months(tmp_path, (2021, 1), (2021, 2), (2021, 3)))
    TickConverter(config).run()
    result = verify_tick_dataset(config)
    assert result.passed, result.failures
    assert result.total_rows == 600 and result.partition_count == 3
    assert result.non_monotonic_rows == 0
    assert result.source_reconciliation["balanced"]
    assert result.dataset_version == result.dataset_version_recomputed
    assert [row["months_present"] for row in result.coverage] == [3]


def test_missing_history_in_the_source_is_reported_not_failed(tmp_path, config_factory):
    config = config_factory(months(tmp_path, (2021, 1), (2021, 4)))
    TickConverter(config).run()
    result = verify_tick_dataset(config)
    assert result.passed
    assert result.months_missing_from_source == ["2021-02", "2021-03"]
    assert result.months_missing_from_output == []
    assert "missing 2" in result.coverage[0]["status"]


def test_a_tampered_partition_fails_the_digest(tmp_path, config_factory):
    config = config_factory(months(tmp_path, (2021, 1), (2021, 2)))
    TickConverter(config).run()
    path = next(config.processed_data_path.rglob("*2021-02.parquet"))
    table = pq.read_table(path)
    bid = table.column("bid").to_pylist()
    bid[5] += 0.001
    pq.write_table(table.set_column(table.schema.get_field_index("bid"), "bid",
                                    pa.array(bid, type=pa.float64())), path)
    result = verify_tick_dataset(config)
    assert not result.passed
    assert result.digest_mismatches == ["2021-02"]


def test_an_unknown_file_and_a_lost_month_are_both_caught(tmp_path, config_factory):
    config = config_factory(months(tmp_path, (2021, 1), (2021, 2)))
    TickConverter(config).run()
    stray = config.processed_data_path / "year=2021" / "stray.parquet"
    pl.DataFrame({"timestamp": [datetime(2021, 1, 9)]}).write_parquet(stray)
    next(config.processed_data_path.rglob("*2021-02.parquet")).unlink()
    result = verify_tick_dataset(config)
    assert not result.passed
    assert any("not in the manifest" in d for d in result.manifest_disagreements)
    assert any("missing on disk" in d for d in result.manifest_disagreements)


def test_gaps_are_classified_including_across_months(tmp_path, config_factory):
    config = config_factory(months(tmp_path, (2021, 1), (2021, 2)))
    TickConverter(config).run()
    result = verify_tick_dataset(config)
    # One gap from 2021-01-05 to 2021-02-05, spanning the partition boundary.
    assert sum(result.gaps_by_category.values()) == 1
    assert result.largest_unexplained_gaps[0]["from"].startswith("2021-01-05")


def test_the_reports_are_written(tmp_path, config_factory):
    config = config_factory(months(tmp_path, (2021, 1)))
    TickConverter(config).run()
    report, coverage = write_verification(config, verify_tick_dataset(config))
    assert report.exists() and coverage.exists()
    assert pl.read_csv(coverage)["tick_rows"].to_list() == [200]


def test_tick_level_daily_breaks_are_not_called_unexplained():
    """Ticks straddle the 00:00-01:00 break by seconds; bars land on it exactly."""
    from xauusd_quant.data.dataset_validation import _classify_tick_gap
    from xauusd_quant.utils.config import load_config

    session = load_config().diagnostics.session
    assert _classify_tick_gap(datetime(2015, 3, 3, 23, 59, 55),
                              datetime(2015, 3, 4, 1, 0, 3), session) == "daily_break"
    assert _classify_tick_gap(datetime(2015, 3, 3, 21, 0, 0),
                              datetime(2015, 3, 4, 1, 0, 3), session) == "holiday_or_unknown"
    assert _classify_tick_gap(datetime(2015, 3, 6, 23, 58, 50),
                              datetime(2015, 3, 9, 1, 0, 3), session) == "weekend"
