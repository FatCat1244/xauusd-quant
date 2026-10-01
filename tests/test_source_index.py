"""The source index: every raw line mapped to exactly one partition.

The converter trusts the index to say where each month's lines are. If the
index misplaces, loses or double-counts a line, a month is silently wrong, so
these tests pin the accounting down on awkward inputs: tiny scan blocks that
split lines, rows that arrive after the stream has moved to a later month,
lines with no usable timestamp, CRLF endings, a missing final newline, a
timestamp that is not the first column, and quoted fields.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pytest

from conftest import make_clean_rows, write_config, write_csv
from xauusd_quant.data.source_index import (
    SourceIndexer,
    month_key_layout,
    ordering_report,
    read_partition_bytes,
)
from xauusd_quant.utils.config import load_config


def build(tmp_path: Path, rows: list[str], *, block_bytes: int = 1 << 20,
          header: str = "DateTime,Bid,Ask,Volume", **overrides):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = write_csv(tmp_path / "source.csv", rows, header=header)
    config = load_config(write_config(tmp_path, source, **overrides), root=tmp_path)
    indexer = SourceIndexer(config, block_bytes=block_bytes)
    names = header.split(",")
    indexer.resolve_columns(names)
    return indexer.build([source]), source, config


def spliced_rows() -> list[str]:
    """January -> February, with a January row repeated after February began."""
    rows = make_clean_rows(datetime(2021, 1, 31, 23, 59, 59), 20, step_ms=100)
    rows += make_clean_rows(datetime(2021, 2, 1, 0, 0, 5), 30, step_ms=100)
    rows.insert(45, rows[3])          # a verbatim January row, arriving in February
    return rows


# ---------------------------------------------------------------------------
# Timestamp layout
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("fmt", "expected"),
    [
        ("%Y%m%d %H:%M:%S%.f", (0, 4)),
        ("%Y-%m-%d %H:%M:%S", (0, 5)),
        ("%d/%m/%Y %H:%M", (6, 3)),
        ("%Y.%m.%d", (0, 5)),
    ],
)
def test_year_and_month_offsets_come_from_the_format(fmt, expected):
    assert month_key_layout(fmt) == expected


def test_a_format_without_fixed_positions_is_refused_clearly():
    with pytest.raises(ValueError, match="fixed character positions"):
        month_key_layout("%b %d %Y")


# ---------------------------------------------------------------------------
# Accounting
# ---------------------------------------------------------------------------
def test_every_line_is_in_exactly_one_partition_or_unassigned(tmp_path):
    rows = spliced_rows()
    rows.insert(10, "garbage,line")
    rows += make_clean_rows(datetime(2021, 3, 1, 1, 0, 0), 5)
    index, _, _ = build(tmp_path, rows)

    assert index.total_lines == len(rows)
    assert index.unassigned_lines == 1
    assert index.accounting_ok()
    assert {k: p.lines for k, p in index.partitions.items()} == {
        "2021-01": 11, "2021-02": 40, "2021-03": 5,
    }
    assert index.unassigned_examples[0]["text"] == "garbage,line"


def test_a_late_row_from_an_earlier_month_gets_its_own_run_in_that_month(tmp_path):
    index, _, _ = build(tmp_path, spliced_rows())
    january = index.partitions["2021-01"]
    assert len(january.runs) == 2, "the straggler must be a separate run of January"
    assert january.first_raw_timestamp == "20210131 23:59:59.000"
    assert january.last_raw_timestamp == "20210131 23:59:59.900"


def test_tiny_scan_blocks_give_exactly_the_same_index(tmp_path):
    """Lines split across blocks must never be lost, duplicated or misassigned."""
    rows = spliced_rows() + make_clean_rows(datetime(2021, 3, 1, 1, 0, 0), 40)
    big, _, _ = build(tmp_path / "big", rows, block_bytes=1 << 20)
    tiny, _, _ = build(tmp_path / "tiny", rows, block_bytes=97)
    assert {k: (p.lines, p.runs, p.content_hash) for k, p in big.partitions.items()} == {
        k: (p.lines, p.runs, p.content_hash) for k, p in tiny.partitions.items()
    }
    big_order = {k: v for k, v in big.ordering.items() if k != "events"}
    tiny_order = {k: v for k, v in tiny.ordering.items() if k != "events"}
    assert big_order == tiny_order
    assert big.files[0].blake2b == tiny.files[0].blake2b


def test_partition_bytes_reproduce_the_content_hash(tmp_path):
    index, source, _ = build(tmp_path, spliced_rows())
    for key, part in index.partitions.items():
        data, digest = read_partition_bytes(index, key, [source])
        assert digest == part.content_hash
        assert data.count(b"\n") == part.lines


# ---------------------------------------------------------------------------
# Ordering diagnostics
# ---------------------------------------------------------------------------
def test_ordering_statistics_are_exact(tmp_path):
    index, _, _ = build(tmp_path, spliced_rows())
    order = index.ordering
    assert order["input_sorted"] is False
    assert order["backward_jumps"] == 1
    assert order["late_lines"] == 1
    assert order["late_lines_in_earlier_partition"] == 1
    event = order["events"][0]
    assert event["crosses_partition"] is True
    assert event["timestamp"] == "20210131 23:59:59.300"


def test_a_sorted_source_reports_itself_sorted(tmp_path):
    rows = make_clean_rows(datetime(2021, 1, 4, 1, 0, 0), 500)
    index, _, _ = build(tmp_path, rows, block_bytes=333)
    assert index.ordering["input_sorted"] is True
    assert index.ordering["backward_jumps"] == 0
    assert index.ordering["late_lines"] == 0


def test_a_repeated_block_is_measured_line_for_line(tmp_path):
    """A block of earlier rows repeated verbatim, strictly behind the maximum."""
    first = make_clean_rows(datetime(2021, 1, 4, 1, 0, 0), 300, step_ms=1000)
    rows = first + first[:120] + make_clean_rows(datetime(2021, 1, 4, 1, 5, 0), 50)
    index, source, _ = build(tmp_path, rows, block_bytes=500)
    order = index.ordering
    assert order["backward_jumps"] == 1
    assert order["late_lines"] == 120
    assert order["late_lines_in_earlier_partition"] == 0

    report = ordering_report(index, [source])
    event = report["events"][0]
    assert event["late_lines_repeated_verbatim"] == event["late_run_lines"] == 120
    assert event["late_lines_not_seen_before"] == 0
    assert report["summary"]["late_lines_not_seen_before"] == 0


def test_a_repeat_that_reaches_the_maximum_leaves_one_tie(tmp_path):
    """The weekly-splice shape of the real export.

    The whole opening hour is repeated, so the repeat's LAST row equals the
    running maximum: it is a duplicate but not "late". That is why the real
    export has 2,683,507 late rows but 2,683,808 duplicates - one tie per splice.
    """
    first = make_clean_rows(datetime(2021, 1, 4, 1, 0, 0), 300, step_ms=1000)
    rows = first + first + make_clean_rows(datetime(2021, 1, 4, 1, 5, 0), 50)
    index, _, _ = build(tmp_path, rows, block_bytes=700)
    assert index.ordering["backward_jumps"] == 1
    assert index.ordering["late_lines"] == 299
    assert index.ordering["equal_adjacent_timestamps"] == 0


def test_a_genuinely_new_late_row_is_not_called_a_repeat(tmp_path):
    rows = make_clean_rows(datetime(2021, 1, 4, 1, 0, 0), 50, step_ms=1000)
    rows.append("20210104 01:00:10.500,2000.000,2000.300,120")   # new, 40 s behind
    rows += make_clean_rows(datetime(2021, 1, 4, 1, 1, 0), 10)
    index, source, _ = build(tmp_path, rows)
    report = ordering_report(index, [source])
    assert report["summary"]["late_lines_not_seen_before"] == 1
    assert report["events"][0]["backward_seconds"] == pytest.approx(38.5)


# ---------------------------------------------------------------------------
# Awkward files
# ---------------------------------------------------------------------------
def test_crlf_and_a_missing_final_newline_are_handled(tmp_path):
    rows = make_clean_rows(datetime(2021, 1, 31, 23, 59, 59), 20)
    source = tmp_path / "crlf.csv"
    source.write_bytes(("DateTime,Bid,Ask,Volume\r\n" + "\r\n".join(rows)).encode())
    config = load_config(write_config(tmp_path, source), root=tmp_path)
    indexer = SourceIndexer(config, block_bytes=64)
    indexer.resolve_columns(["DateTime", "Bid", "Ask", "Volume"])
    index = indexer.build([source])
    assert index.total_lines == 20
    assert index.accounting_ok()
    assert index.partitions["2021-02"].last_raw_timestamp == "20210201 00:00:00.900"
    data, digest = read_partition_bytes(index, "2021-02", [source])
    assert digest == index.partitions["2021-02"].content_hash
    assert data.endswith(b"\n"), "a run without a final newline gets a separator"


def test_timestamp_in_a_later_column_and_quoted(tmp_path):
    rows = [
        f'2000.{i:03d},2000.300,"20210131 23:59:{50 + i:02d}.000",120' for i in range(5)
    ] + [f'2001.000,2001.300,"20210201 00:00:0{i}.000",120' for i in range(3)]
    header = "Bid,Ask,DateTime,Volume"
    index, _, _ = build(tmp_path, rows, header=header, block_bytes=50)
    assert {k: p.lines for k, p in index.partitions.items()} == {"2021-01": 5, "2021-02": 3}
    assert index.unassigned_lines == 0


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
def test_a_cached_index_is_reused_until_the_source_changes(tmp_path):
    rows = make_clean_rows(datetime(2021, 1, 4, 1, 0, 0), 100)
    source = write_csv(tmp_path / "source.csv", rows)
    config = load_config(write_config(tmp_path, source), root=tmp_path)
    indexer = SourceIndexer(config)
    indexer.resolve_columns(["DateTime", "Bid", "Ask", "Volume"])

    first = indexer.load_or_build([source])
    assert indexer.path.exists()
    again = indexer.load_or_build([source])
    assert again.created_utc == first.created_utc, "an unchanged source reuses the index"

    with source.open("a", encoding="utf-8") as handle:
        handle.write("20210104 01:00:10.000,2000.000,2000.300,120\n")
    rebuilt = indexer.load_or_build([source])
    assert rebuilt.total_lines == 101
    assert json.loads(indexer.path.read_text(encoding="utf-8"))["total_lines"] == 101


def test_an_in_place_edit_with_the_same_size_and_mtime_is_caught_by_the_hash(tmp_path):
    """Size and mtime are only the cheap check; the partition hash is the real one."""
    rows = make_clean_rows(datetime(2021, 1, 4, 1, 0, 0), 100)
    source = write_csv(tmp_path / "source.csv", rows)
    config = load_config(write_config(tmp_path, source), root=tmp_path)
    indexer = SourceIndexer(config)
    indexer.resolve_columns(["DateTime", "Bid", "Ask", "Volume"])
    index = indexer.load_or_build([source])

    stat = source.stat()
    text = source.read_bytes().replace(b"2000.300,120", b"2000.400,120", 1)
    source.write_bytes(text)
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert indexer.load_or_build([source]).created_utc == index.created_utc

    _, digest = read_partition_bytes(index, "2021-01", [source])
    assert digest != index.partitions["2021-01"].content_hash
