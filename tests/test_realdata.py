"""End-to-end checks against the real XAUUSD dataset.

Every test here is skipped automatically when the converted dataset is absent,
so the suite still runs on a clean checkout. Run them after a full
``xq convert`` + ``xq build-bars --all`` to verify the pipeline against the
actual 2003-2026 export (~732 million source rows) rather than a synthetic
fixture.

    pytest -m realdata
"""

from __future__ import annotations

import random

import polars as pl
import pytest

from xauusd_quant import load_config
from xauusd_quant.data import DataStore
from xauusd_quant.data.converter import load_dataset_manifest
from xauusd_quant.data.resampler import parse_timeframe

pytestmark = pytest.mark.realdata


@pytest.fixture(scope="module")
def config():
    """The live configuration, if the dataset on disk is actually usable.

    A partially-converted dataset makes every test in this module meaningless -
    they would fail for the wrong reason and bury the real problem. So the
    manifest is checked against what is on disk first, and the whole module
    skips with a message naming the inconsistency.
    """
    cfg = load_config()
    files = sorted(cfg.processed_data_path.rglob("*.parquet"))
    if not files:
        pytest.skip("no converted dataset; run `xq convert` first")

    manifest = load_dataset_manifest(cfg.processed_data_path)
    if manifest is None:
        pytest.skip("the dataset has no manifest; re-run `xq convert`")
    if manifest.get("status") != "complete":
        pytest.skip(
            f"dataset is {manifest.get('status')!r}: {len(manifest.get('missing_partitions') or [])} "
            "partition(s) missing. Re-run `xq convert` before trusting these tests."
        )
    expected = manifest.get("partition_count") or 0
    if len(files) != expected:
        pytest.skip(
            f"manifest records {expected} partitions but {len(files)} files are on disk"
        )
    if manifest.get("tick_fingerprint") != cfg.tick_fingerprint():
        pytest.skip("manifest was written under a different tick configuration")
    return cfg


@pytest.fixture(scope="module")
def store(config):
    with DataStore(config) as handle:
        yield handle


def test_dataset_is_sorted_and_free_of_duplicate_timestamps(store):
    row = store.connect().execute(
        f"""
        SELECT count(*), count(DISTINCT timestamp), min(timestamp), max(timestamp)
        FROM read_parquet('{store.tick_glob()}')
        """  # noqa: S608
    ).fetchone()
    total, distinct, lo, hi = row
    assert total == distinct, f"{total - distinct:,} duplicate timestamps"
    assert lo < hi


def test_every_partition_is_internally_sorted(config):
    for path in sorted(config.processed_data_path.rglob("*.parquet")):
        stamps = pl.read_parquet(path, columns=["timestamp"])["timestamp"]
        assert stamps.is_sorted(), path.name


def test_partitions_hold_only_their_own_month(config):
    for path in sorted(config.processed_data_path.rglob("*.parquet")):
        year = int(path.parent.parent.name.split("=")[1])
        month = int(path.parent.name.split("=")[1])
        stamps = pl.read_parquet(path, columns=["timestamp"])["timestamp"]
        assert stamps.dt.year().unique().to_list() == [year], path
        assert stamps.dt.month().unique().to_list() == [month], path


def test_derived_columns_hold_across_a_sampled_month(config):
    path = next(iter(sorted(config.processed_data_path.rglob("*.parquet"))))
    frame = pl.read_parquet(path)
    assert (frame["mid"] - (frame["bid"] + frame["ask"]) / 2).abs().max() < 1e-9
    assert (frame["spread"] - (frame["ask"] - frame["bid"])).abs().max() < 1e-9
    assert (frame["bid"] > 0).all() and (frame["ask"] > 0).all()


@pytest.mark.parametrize("timeframe", ["1m", "5m", "1h"])
def test_bars_reproduce_exactly_from_ticks_with_no_look_ahead(config, timeframe):
    """Recompute real bars from the raw ticks and compare, field by field."""
    tick_files = sorted(config.processed_data_path.rglob("*.parquet"))
    bar_files = sorted(config.bars_dir(timeframe).rglob("*.parquet"))
    if not bar_files:
        pytest.skip(f"no {timeframe} bars; run `xq build-bars --all`")

    # A single mid-dataset partition keeps this fast and deterministic.
    tick_path = tick_files[len(tick_files) // 2]
    ticks = pl.read_parquet(tick_path)
    bars = pl.read_parquet(bar_files).filter(
        (pl.col("timestamp") >= ticks["timestamp"].min())
        & (pl.col("timestamp") <= ticks["timestamp"].max())
    )
    assert not bars.is_empty()

    step = parse_timeframe(timeframe)
    sample = random.Random(7).sample(
        list(bars.iter_rows(named=True)), min(200, bars.height)
    )
    checked = 0
    for row in sample:
        start = row["timestamp"]
        window = ticks.filter(
            (pl.col("timestamp") >= start) & (pl.col("timestamp") < start + step)
        )
        if window.is_empty():
            continue
        checked += 1
        mid = window["mid"]
        assert row["tick_count"] == window.height, start
        assert row["open"] == pytest.approx(mid[0]), start
        assert row["close"] == pytest.approx(mid[-1]), start
        assert row["high"] == pytest.approx(mid.max()), start
        assert row["low"] == pytest.approx(mid.min()), start
        assert row["first_bid"] == pytest.approx(window["bid"][0]), start
        assert row["first_ask"] == pytest.approx(window["ask"][0]), start
        assert row["last_bid"] == pytest.approx(window["bid"][-1]), start
        assert row["last_ask"] == pytest.approx(window["ask"][-1]), start
        assert row["mean_spread"] == pytest.approx(window["spread"].mean()), start
        # No look-ahead: every tick used must lie inside this bar's own window.
        assert start <= row["first_tick_timestamp"], start
        assert row["last_tick_timestamp"] < start + step, start

    assert checked >= 50, f"only {checked} bars could be checked"


def test_utc_column_matches_the_documented_offsets(store):
    """UTC+2 in US winter, UTC+3 in US summer - the whole timezone conclusion."""
    winter = store.load_ticks("2024-01-08 15:30:00", "2024-01-08 15:30:05", limit=1)
    summer = store.load_ticks("2024-07-08 15:30:00", "2024-07-08 15:30:05", limit=1)
    for frame, expected_hours in ((winter, 2), (summer, 3)):
        if frame.is_empty():
            pytest.skip("expected window has no ticks")
        row = frame.row(0, named=True)
        offset = (row["timestamp"] - row["timestamp_utc"].replace(tzinfo=None))
        assert offset.total_seconds() / 3600 == expected_hours


def test_manifest_agrees_with_what_is_on_disk(config, store):
    manifest = load_dataset_manifest(config.processed_data_path)
    assert manifest is not None
    on_disk = store.connect().execute(
        f"SELECT count(*) FROM read_parquet('{store.tick_glob()}')"  # noqa: S608
    ).fetchone()
    assert manifest["total_rows"] == on_disk[0]
    assert manifest["tick_fingerprint"] == config.tick_fingerprint()
    assert manifest["status"] == "complete"
    assert manifest["output_sorted"] is True
    assert len(manifest["partitions"]) == len(
        list(config.processed_data_path.rglob("*.parquet"))
    )
    assert all(p["digest"] for p in manifest["partitions"])
    assert manifest["dataset_version"].startswith("ticks-")


def test_every_source_line_is_accounted_for(config):
    """rows written + rows dropped + malformed = every line of the raw file."""
    manifest = load_dataset_manifest(config.processed_data_path)
    assert manifest is not None
    index = manifest["source_index"]
    written = manifest["total_rows"]
    unassigned = manifest.get("unassigned") or {}
    dropped = manifest["rows_dropped"] + int(unassigned.get("rows_dropped") or 0)
    malformed = sum(int(e.get("malformed_rows") or 0) for e in manifest["partitions"]) + int(
        unassigned.get("malformed_rows") or 0
    )
    assert written + dropped + malformed == index["total_lines"]


# ---------------------------------------------------------------------------
# Properties specific to the 2003-2026 export
# ---------------------------------------------------------------------------
def test_the_only_disorder_is_the_weekly_splices_and_they_are_verbatim_repeats(config):
    """The source's one ordering problem, diagnosed and resolved end to end.

    From 2021 the export repeats the opening hour or two of every trading week
    verbatim. Each repeat starts with one backward jump and ends on a row that
    ties the running maximum, so the exact-duplicate drops must equal the late
    rows plus one per jump - and no repeated row may be new information.
    """
    import json

    report_path = config.metadata_path / "source_ordering_report.json"
    if not report_path.exists():
        pytest.skip("run `xq source-index` to produce the ordering report")
    summary = json.loads(report_path.read_text(encoding="utf-8"))["summary"]
    manifest = load_dataset_manifest(config.processed_data_path)
    assert manifest is not None
    assert summary["late_lines_in_earlier_partition"] == 0
    assert summary["late_lines_not_seen_before"] == 0
    assert summary["first_event_timestamp"] >= "20210101"
    duplicates_dropped = sum(
        int((e.get("dropped_by_reason") or {}).get("exact_duplicate_rows", 0))
        for e in manifest["partitions"]
    )
    assert duplicates_dropped == summary["late_lines"] + summary["backward_jumps"]
def test_source_spread_is_preserved_and_matches_the_derived_one(config, store):
    """The source ships a Spread column; we keep it *and* derive our own.

    They agree in this export, but the pipeline must never assume that - the
    point of keeping both is to be able to check.
    """
    columns = set(pl.read_parquet(
        next(iter(sorted(config.processed_data_path.rglob("*.parquet")))), n_rows=1
    ).columns)
    if "spread_source" not in columns:
        pytest.skip("this export has no source spread column")

    row = store.connect().execute(
        f"""
        SELECT count(*),
               max(abs(spread - spread_source)),
               sum(CASE WHEN spread_source IS NULL THEN 1 ELSE 0 END)
        FROM read_parquet('{store.tick_glob()}')
        """  # noqa: S608
    ).fetchone()
    total, max_delta, nulls = row
    assert total > 0
    assert nulls == 0, f"{nulls:,} rows have a null source spread"
    assert max_delta is not None and max_delta < 1e-9, (
        f"source spread and (ask - bid) disagree by up to {max_delta}"
    )


def test_timezone_rule_holds_in_the_earliest_era_too(store):
    """The +7h offset was re-derived for 2003-2010, so check it there.

    May 2003 is US summer, so the file's clock should be UTC+3 then - the same
    rule that gives UTC+2 in winter twenty years later.
    """
    early = store.load_ticks("2003-05-05", "2003-05-06", limit=1)
    if early.is_empty():
        pytest.skip("no 2003 data in this dataset")
    row = early.row(0, named=True)
    offset = (row["timestamp"] - row["timestamp_utc"].replace(tzinfo=None))
    assert offset.total_seconds() / 3600 == 3, "May 2003 should be UTC+3 (US summer)"

    winter = store.load_ticks("2004-01-15", "2004-01-16", limit=1)
    if not winter.is_empty():
        row = winter.row(0, named=True)
        offset = (row["timestamp"] - row["timestamp_utc"].replace(tzinfo=None))
        assert offset.total_seconds() / 3600 == 2, "Jan 2004 should be UTC+2 (US winter)"


def test_prices_are_plausible_across_two_decades(store):
    """Gold traded near $340 in 2003 and above $4000 in 2026."""
    row = store.connect().execute(
        f"""
        SELECT min(bid), max(ask),
               min(CASE WHEN year(timestamp) = 2003 THEN bid END),
               max(CASE WHEN year(timestamp) = 2003 THEN ask END)
        FROM read_parquet('{store.tick_glob()}')
        """  # noqa: S608
    ).fetchone()
    overall_min, overall_max, y2003_min, y2003_max = row
    assert 200 < overall_min < 500, f"implausible all-time low {overall_min}"
    if y2003_min is not None:
        assert 300 < y2003_min < 420, f"2003 low {y2003_min} is not gold-like"
        assert 300 < y2003_max < 450, f"2003 high {y2003_max} is not gold-like"

    # Only assert the modern high when the dataset actually reaches those years.
    span = store.connect().execute(
        f"SELECT max(year(timestamp)) FROM read_parquet('{store.tick_glob()}')"  # noqa: S608
    ).fetchone()[0]
    if span and span >= 2020:
        assert overall_max > 3000, f"implausible all-time high {overall_max}"


def test_the_2012_export_change_is_visible_but_the_clock_is_not(store):
    """Before 2012 the 00:00-01:00 hour is exported; from 2012 it is not.

    This is a change in what the provider ships, not a change of timezone, so
    it must show up as tick counts in hour 0 and nothing else.
    """
    row = store.connect().execute(
        f"""
        SELECT
          sum(CASE WHEN year(timestamp) BETWEEN 2005 AND 2011
                    AND hour(timestamp) = 0 THEN 1 ELSE 0 END),
          sum(CASE WHEN year(timestamp) BETWEEN 2005 AND 2011 THEN 1 ELSE 0 END),
          sum(CASE WHEN year(timestamp) BETWEEN 2013 AND 2019
                    AND hour(timestamp) = 0 THEN 1 ELSE 0 END),
          sum(CASE WHEN year(timestamp) BETWEEN 2013 AND 2019 THEN 1 ELSE 0 END)
        FROM read_parquet('{store.tick_glob()}')
        """  # noqa: S608
    ).fetchone()
    early_h0, early_all, late_h0, late_all = row
    if not early_all or not late_all:
        pytest.skip("dataset does not span both eras")
    early_share = early_h0 / early_all
    late_share = late_h0 / late_all
    assert early_share > 0.005, f"pre-2012 hour 0 should be populated, got {early_share:.4%}"
    assert late_share < 0.001, f"post-2012 hour 0 should be near-empty, got {late_share:.4%}"
