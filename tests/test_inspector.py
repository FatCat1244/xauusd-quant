"""Large-file inspection: sampling, seeking and honest labelling."""

from __future__ import annotations

from datetime import datetime

import pytest

from conftest import fmt, make_clean_rows
from xauusd_quant.data.inspector import (
    find_offset_for_prefix,
    inspect_file,
    inspect_sources,
    iter_block_samples,
    read_head_lines,
    read_tail_lines,
    resolve_sources,
    sniff_dialect,
)


# ---------------------------------------------------------------------------
# Head / tail
# ---------------------------------------------------------------------------
def test_head_and_tail_are_exact(clean_csv):
    head = read_head_lines(clean_csv, 3)
    tail = read_tail_lines(clean_csv, 2)
    all_lines = clean_csv.read_text(encoding="utf-8").splitlines()
    assert head == all_lines[:3]
    assert tail == all_lines[-2:]


def test_tail_works_when_the_window_must_expand(tmp_path):
    rows = make_clean_rows(datetime(2021, 1, 4, 1, 0, 0), 5000)
    path = tmp_path / "big.csv"
    path.write_text("DateTime,Bid,Ask,Volume\n" + "\n".join(rows) + "\n", encoding="utf-8")
    assert read_tail_lines(path, 3, window=64) == rows[-3:]


def test_tail_of_a_single_line_file(tmp_path):
    path = tmp_path / "one.csv"
    path.write_text("DateTime,Bid,Ask,Volume\n", encoding="utf-8")
    assert read_tail_lines(path, 5) == ["DateTime,Bid,Ask,Volume"]


# ---------------------------------------------------------------------------
# Byte-offset search
# ---------------------------------------------------------------------------
def test_finds_the_byte_offset_of_a_date(two_month_csv):
    offset = find_offset_for_prefix(two_month_csv, b"20210201")
    with two_month_csv.open("rb") as handle:
        handle.seek(offset)
        line = handle.readline().decode()
    assert line.startswith("20210201")

    # The line immediately before must belong to the previous month.
    with two_month_csv.open("rb") as handle:
        handle.seek(max(offset - 200, 0))
        previous = [ln for ln in handle.read(200).split(b"\n") if ln][:-1]
    assert previous and previous[-1].decode().startswith("202101")


def test_offset_search_lands_on_a_line_boundary(two_month_csv):
    for prefix in (b"20210101", b"20210201", b"20210301", b"20210401"):
        offset = find_offset_for_prefix(two_month_csv, prefix)
        if offset >= two_month_csv.stat().st_size:
            continue
        with two_month_csv.open("rb") as handle:
            handle.seek(offset)
            assert handle.readline().count(b",") == 3


def test_offset_search_past_the_end_returns_the_size(two_month_csv):
    size = two_month_csv.stat().st_size
    assert find_offset_for_prefix(two_month_csv, b"29990101") >= size


def test_offset_search_respects_data_start(clean_csv):
    """The header must never be returned as a data row."""
    with clean_csv.open("rb") as handle:
        handle.readline()
        data_start = handle.tell()
    assert find_offset_for_prefix(clean_csv, b"20210104", data_start=data_start) >= data_start


# ---------------------------------------------------------------------------
# Block sampling
# ---------------------------------------------------------------------------
def test_samples_never_contain_partial_lines(two_month_csv):
    with two_month_csv.open("rb") as handle:
        handle.readline()
        data_start = handle.tell()
    for sample in iter_block_samples(two_month_csv, blocks=6, block_bytes=256,
                                     data_start=data_start):
        for line in sample.lines:
            assert line.count(b",") == 3, line
            assert line.split(b",")[0].isdigit() or b" " in line.split(b",")[0]


def test_small_files_yield_one_full_block(clean_csv):
    samples = list(iter_block_samples(clean_csv, blocks=40, block_bytes=1 << 20))
    assert len(samples) == 1


def test_empty_file_yields_no_samples(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_bytes(b"")
    assert list(iter_block_samples(path)) == []


# ---------------------------------------------------------------------------
# Dialect sniffing
# ---------------------------------------------------------------------------
def test_sniffs_the_real_dataset_dialect(clean_csv):
    dialect = sniff_dialect(clean_csv)
    assert dialect.delimiter == ","
    assert dialect.has_header is True
    assert dialect.encoding == "utf-8"
    assert dialect.has_bom is False
    assert dialect.line_terminator == "\n"


def test_sniffs_semicolons_and_crlf(tmp_path):
    path = tmp_path / "semi.csv"
    path.write_text("DateTime;Bid;Ask\r\n20210104 01:00:00.413;1.0;1.1\r\n", encoding="utf-8")
    dialect = sniff_dialect(path)
    assert dialect.delimiter == ";"
    assert dialect.line_terminator == "\r\n"


def test_detects_a_byte_order_mark(tmp_path):
    path = tmp_path / "bom.csv"
    path.write_bytes("﻿DateTime,Bid,Ask\n20210104 01:00:00.413,1.0,1.1\n".encode())
    dialect = sniff_dialect(path)
    assert dialect.has_bom is True
    assert dialect.encoding == "utf-8-sig"


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------
def test_inspection_reports_the_expected_facts(clean_csv):
    report = inspect_file(clean_csv, blocks=8, block_bytes=4096,
                          timestamp_format="%Y%m%d %H:%M:%S%.f")
    assert report.exists
    assert report.header == ["DateTime", "Bid", "Ask", "Volume"]
    assert report.size_bytes == clean_csv.stat().st_size
    assert report.sampled_rows > 0
    assert report.estimated_total_rows is not None
    assert report.detection["mapping"]["bid"] == "Bid"
    assert report.spread_summary["median"] == pytest.approx(0.30, abs=1e-9)


def test_first_and_last_timestamps_are_exact_not_sampled(clean_csv):
    report = inspect_file(clean_csv, blocks=2, block_bytes=128)
    assert report.first_timestamp == fmt(datetime(2021, 1, 4, 1, 0, 0))
    assert report.last_timestamp == fmt(datetime(2021, 1, 4, 1, 0, 59, 900000))
    assert any("exact" in note for note in report.notes)


def test_row_count_estimate_is_labelled_as_an_estimate(clean_csv):
    report = inspect_file(clean_csv)
    assert report.row_count_is_estimate is True
    assert report.estimated_total_rows == pytest.approx(600, rel=0.05)


def test_column_profiles_include_dtype_and_range(clean_csv):
    report = inspect_file(clean_csv, blocks=4, block_bytes=8192)
    by_name = {c["name"]: c for c in report.columns}
    assert by_name["Bid"]["inferred_dtype"] == "float64"
    assert by_name["DateTime"]["inferred_dtype"] == "timestamp(string)"
    assert by_name["Bid"]["min_value"] == pytest.approx(2000.0)
    assert by_name["Bid"]["null_rate"] == 0.0


def test_defects_are_surfaced_as_warnings(dirty_csv):
    report = inspect_file(dirty_csv, blocks=1, block_bytes=1 << 20,
                          timestamp_format="%Y%m%d %H:%M:%S%.f")
    assert report.field_count_mismatches >= 1
    assert report.sampled_ask_below_bid >= 1
    assert report.sampled_duplicate_rows >= 1
    assert report.sampled_non_monotonic >= 1
    assert any("did not have" in w for w in report.warnings)
    assert any("ask < bid" in w for w in report.warnings)


def test_missing_file_is_reported_not_raised(tmp_path):
    report = inspect_file(tmp_path / "absent.csv")
    assert report.exists is False
    assert report.warnings


def test_unrecognised_header_produces_a_clear_warning(tmp_path):
    path = tmp_path / "odd.csv"
    path.write_text("alpha,beta,gamma\n1,2,3\n", encoding="utf-8")
    report = inspect_file(path)
    assert any("Could not map required column" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Source resolution
# ---------------------------------------------------------------------------
def test_resolves_a_single_file_a_directory_and_a_glob(tmp_path, clean_csv):
    assert resolve_sources(clean_csv) == [clean_csv]

    folder = tmp_path / "many"
    folder.mkdir()
    for name in ("b.csv", "a.csv"):
        (folder / name).write_text("DateTime,Bid,Ask\n", encoding="utf-8")
    assert [p.name for p in resolve_sources(folder)] == ["a.csv", "b.csv"]
    assert [p.name for p in resolve_sources(folder / "*.csv")] == ["a.csv", "b.csv"]


def test_resolution_of_a_missing_path_is_empty(tmp_path):
    assert resolve_sources(tmp_path / "nothing.csv") == []


def test_multi_file_report_aggregates_totals(clean_csv, two_month_csv):
    report = inspect_sources([clean_csv, two_month_csv], blocks=2, block_bytes=4096)
    assert report["file_count"] == 2
    assert report["total_size_bytes"] == (
        clean_csv.stat().st_size + two_month_csv.stat().st_size
    )
    assert len(report["files"]) == 2
