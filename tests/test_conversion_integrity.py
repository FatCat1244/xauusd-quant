"""Conversion integrity: disorder, crashes, resume, overwrite and the swap.

These tests exist because of a real failure. A full conversion of the 23-year
export completed, then an ``--overwrite`` deleted it *before* the replacement
was built, and the replacement stopped after 19 of 281 months. Separately, the
old converter would have replaced a finished month with a single row had any
row of that month arrived after the stream moved on.

Each test drives the converter into one of those situations and checks that
no data is lost, nothing is duplicated, the live dataset is never left
half-replaced, and a rerun finishes the job rather than starting over.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from conftest import make_clean_rows, write_csv
from xauusd_quant.data import converter as converter_module
from xauusd_quant.data.converter import (
    DATASET_MANIFEST,
    InsufficientSpaceError,
    TickConverter,
    load_dataset_manifest,
)
from xauusd_quant.data.loader import DataStore


def parquet_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.parquet"))


def read_all(root: Path) -> pl.DataFrame:
    return pl.read_parquet(parquet_files(root))


def three_months(tmp_path: Path) -> Path:
    rows = (
        make_clean_rows(datetime(2021, 1, 31, 22, 0, 0), 300, bid0=1900.0)
        + make_clean_rows(datetime(2021, 2, 1, 1, 0, 0), 300, bid0=1910.0)
        + make_clean_rows(datetime(2021, 3, 1, 1, 0, 0), 300, bid0=1920.0)
    )
    return write_csv(tmp_path / "three.csv", rows)


def reference(config_factory, source: Path, tmp_path: Path, name: str = "ref", **overrides):
    config = config_factory(
        source,
        processed_data_path=str(tmp_path / f"pq_{name}").replace("\\", "/"),
        metadata_path=str(tmp_path / f"meta_{name}").replace("\\", "/"),
        **overrides,
    )
    TickConverter(config).run()
    return read_all(config.processed_data_path).sort("timestamp")


def fail_on(key: str, exc: type[BaseException] = KeyboardInterrupt):
    original = TickConverter._convert_partition

    def wrapper(self, partition, *args, **kwargs):
        if partition == key:
            raise exc(f"simulated failure while converting {partition}")
        return original(self, partition, *args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Disorder in the source
# ---------------------------------------------------------------------------
def test_out_of_order_rows_are_sorted_into_place_not_dropped(tmp_path, config_factory):
    rows = make_clean_rows(datetime(2021, 1, 4, 1, 0, 0), 100, step_ms=1000)
    late = "20210104 01:00:10.500,2000.000,2000.300,120"      # new, not a repeat
    rows.insert(60, late)
    config = config_factory(write_csv(tmp_path / "late.csv", rows))

    result = TickConverter(config).run()
    frame = read_all(config.processed_data_path)
    assert frame.height == 101, "an out-of-order row is an observation, not an error"
    assert frame["timestamp"].is_sorted()
    assert result.partitions[0].source_backward_steps == 1
    assert result.input_sorted is False and result.output_sorted is True


def test_a_row_arriving_after_its_month_closed_does_not_clobber_the_month(
    tmp_path, config_factory
):
    """Regression: the old converter recreated January's file for the late row."""
    january = make_clean_rows(datetime(2021, 1, 31, 23, 0, 0), 200, step_ms=1000)
    february = make_clean_rows(datetime(2021, 2, 1, 1, 0, 0), 200, step_ms=1000)
    straggler = "20210131 23:59:59.500,2000.000,2000.300,120"
    config = config_factory(write_csv(tmp_path / "s.csv", january + february + [straggler]))

    TickConverter(config).run()
    frame = read_all(config.processed_data_path)
    jan = frame.filter(pl.col("timestamp").dt.month() == 1)
    assert jan.height == 201, "every January row, including the straggler, survives"
    assert jan["timestamp"].is_sorted()
    assert jan["timestamp"][-1] == datetime(2021, 1, 31, 23, 59, 59, 500000)
    assert frame.height == 401


@pytest.mark.parametrize("drop", [True, False])
def test_a_repeated_block_follows_the_explicit_duplicate_policy(tmp_path, config_factory,
                                                                 drop):
    """The real export repeats the first hour of each week, verbatim."""
    hour = make_clean_rows(datetime(2021, 1, 4, 1, 0, 0), 300, step_ms=1000)
    rows = hour + hour + make_clean_rows(datetime(2021, 1, 4, 1, 10, 0), 100)
    config = config_factory(
        write_csv(tmp_path / "splice.csv", rows),
        cleaning={"drop_exact_duplicate_rows": drop},
    )
    result = TickConverter(config).run()
    frame = read_all(config.processed_data_path)
    record = result.partitions[0]
    assert record.validation_counts["exact_duplicate_rows"] == 300
    assert frame["timestamp"].is_sorted()
    if drop:
        assert frame.height == 400
        assert record.dropped_by_reason == {"exact_duplicate_rows": 300}
        assert frame["timestamp"].n_unique() == 400
    else:
        assert frame.height == 700
        assert record.rows_dropped == 0
        # Kept copies sit next to their originals once the month is sorted.
        assert frame["timestamp"].n_unique() == 400


# ---------------------------------------------------------------------------
# Crashes and resume
# ---------------------------------------------------------------------------
def test_an_interrupted_conversion_resumes_at_the_partition_level(
    tmp_path, config_factory, monkeypatch
):
    source = three_months(tmp_path)
    expected = reference(config_factory, source, tmp_path)
    config = config_factory(source)

    monkeypatch.setattr(TickConverter, "_convert_partition", fail_on("2021-02"))
    with pytest.raises(KeyboardInterrupt):
        TickConverter(config).run()
    monkeypatch.undo()

    manifest = load_dataset_manifest(config.processed_data_path)
    assert manifest is not None
    assert manifest["status"] == "incomplete"
    assert [e["key"] for e in manifest["partitions"] if e["complete"]] == ["2021-01"]

    result = TickConverter(config).run()
    assert result.skipped_partitions == ["2021-01"]
    assert [p.key for p in result.partitions] == ["2021-02", "2021-03"]
    assert result.status == "complete"
    assert read_all(config.processed_data_path).sort("timestamp").equals(expected)


def test_a_crash_while_writing_leaves_no_partition_behind(tmp_path, config_factory,
                                                           monkeypatch):
    """A half-written file must never carry a final name or a completion mark."""
    source = three_months(tmp_path)
    config = config_factory(source)
    real_write = pl.DataFrame.write_parquet
    calls = {"n": 0}

    def flaky(self, file, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:                       # dies while writing February,
            Path(file).write_bytes(b"PAR1 half a file")   # leaving a torn file behind
            raise OSError("disk vanished")
        return real_write(self, file, *args, **kwargs)

    monkeypatch.setattr(pl.DataFrame, "write_parquet", flaky)
    with pytest.raises(OSError, match="disk vanished"):
        TickConverter(config).run()
    monkeypatch.undo()

    names = [p.name for p in parquet_files(config.processed_data_path)]
    assert names == ["xauusd-ticks-2021-01.parquet"]
    manifest = load_dataset_manifest(config.processed_data_path)
    assert manifest is not None
    assert "2021-02" not in {e["key"] for e in manifest["partitions"]}

    result = TickConverter(config).run()
    assert result.status == "complete"
    assert not list(config.processed_data_path.rglob("*.partial"))
    assert read_all(config.processed_data_path).height == 900


# ---------------------------------------------------------------------------
# --overwrite is transactional
# ---------------------------------------------------------------------------
def test_overwrite_keeps_the_old_dataset_until_the_new_one_is_complete(
    tmp_path, config_factory, monkeypatch
):
    source = three_months(tmp_path)
    config = config_factory(source)
    TickConverter(config).run()
    before = read_all(config.processed_data_path).sort("timestamp")

    monkeypatch.setattr(TickConverter, "_convert_partition", fail_on("2021-03"))
    with pytest.raises(KeyboardInterrupt):
        TickConverter(config).run(overwrite=True)
    monkeypatch.undo()

    # The live dataset is untouched, complete and readable.
    live = load_dataset_manifest(config.processed_data_path)
    assert live is not None and live["status"] == "complete"
    assert read_all(config.processed_data_path).sort("timestamp").equals(before)
    with DataStore(config) as store:
        assert store.load_ticks("2021-01-01", "2021-04-01").height == 900
    building = config.processed_data_path.with_name("parquet.building")
    assert building.exists()

    # A plain rerun continues the rebuild instead of starting it over, then swaps.
    result = TickConverter(config).run()
    assert result.mode == "rebuild"
    assert set(result.skipped_partitions) == {"2021-01", "2021-02"}
    assert [p.key for p in result.partitions] == ["2021-03"]
    assert result.swapped
    assert not building.exists()
    assert not config.processed_data_path.with_name("parquet.previous").exists()
    assert read_all(config.processed_data_path).sort("timestamp").equals(before)


def test_an_interrupted_swap_is_completed_on_the_next_run(tmp_path, config_factory,
                                                          monkeypatch):
    source = three_months(tmp_path)
    config = config_factory(source)
    TickConverter(config).run()
    canonical = config.processed_data_path

    def half_swap(self, building, canonical_dir, *, keep_previous):
        canonical_dir.replace(canonical_dir.with_name(canonical_dir.name + ".previous"))
        raise KeyboardInterrupt("stopped between the two renames")

    monkeypatch.setattr(TickConverter, "_swap_in", half_swap)
    with pytest.raises(KeyboardInterrupt):
        TickConverter(config).run(overwrite=True)
    monkeypatch.undo()
    assert not canonical.exists()

    result = TickConverter(config).run()
    assert canonical.exists()
    assert not canonical.with_name("parquet.previous").exists()
    assert not canonical.with_name("parquet.building").exists()
    assert result.status == "complete"
    assert read_all(canonical).height == 900


def test_an_interrupted_swap_without_a_finished_build_rolls_back(tmp_path, config_factory):
    source = three_months(tmp_path)
    config = config_factory(source)
    TickConverter(config).run()
    canonical = config.processed_data_path
    canonical.replace(canonical.with_name("parquet.previous"))   # crash mid-swap

    result = TickConverter(config).run()
    assert result.skipped_partitions == ["2021-01", "2021-02", "2021-03"]
    assert read_all(canonical).height == 900


def test_keep_previous_leaves_a_timestamped_backup(tmp_path, config_factory):
    source = three_months(tmp_path)
    config = config_factory(source)
    TickConverter(config).run()
    TickConverter(config).run(overwrite=True, keep_previous=True)
    backups = [p for p in tmp_path.iterdir() if p.name.startswith("parquet.backup-")]
    assert len(backups) == 1
    assert len(parquet_files(backups[0])) == 3
    assert read_all(config.processed_data_path).height == 900


def test_files_the_pipeline_did_not_create_survive_the_swap(tmp_path, config_factory):
    source = three_months(tmp_path)
    config = config_factory(source)
    TickConverter(config).run()
    notes = config.processed_data_path / "NOTES.md"
    notes.write_text("hand-written", encoding="utf-8")
    TickConverter(config).run(overwrite=True)
    assert notes.read_text(encoding="utf-8") == "hand-written"


# ---------------------------------------------------------------------------
# Guards and housekeeping
# ---------------------------------------------------------------------------
def test_insufficient_space_is_refused_before_anything_is_written(tmp_path, config_factory):
    source = three_months(tmp_path)
    config = config_factory(source, conversion={"min_free_gb": 1_000_000.0})
    with pytest.raises(InsufficientSpaceError, match="reserve"):
        TickConverter(config).run()
    assert not parquet_files(config.processed_data_path)


def test_a_month_removed_from_the_source_is_removed_from_the_dataset(
    tmp_path, config_factory
):
    source = three_months(tmp_path)
    config = config_factory(source)
    TickConverter(config).run()
    lines = source.read_text(encoding="utf-8").splitlines()
    source.write_text("\n".join(ln for ln in lines if not ln.startswith("202103")) + "\n",
                      encoding="utf-8")

    result = TickConverter(config).run()
    assert result.removed_partitions == ["2021-03"]
    assert result.status == "complete"
    assert read_all(config.processed_data_path).height == 600


def test_operational_settings_never_invalidate_the_dataset(tmp_path, config_factory):
    source = three_months(tmp_path)
    TickConverter(config_factory(source)).run()
    tuned = config_factory(
        source, conversion={"read_block_mb": 2, "progress": True, "min_free_gb": 0.5},
    )
    result = TickConverter(tuned).run()
    assert result.skipped_partitions == ["2021-01", "2021-02", "2021-03"]
    assert result.partitions == []


def test_a_tick_setting_change_rebuilds_beside_the_live_dataset(tmp_path, config_factory):
    source = three_months(tmp_path)
    TickConverter(config_factory(source)).run()
    changed = config_factory(source, validation={"max_spread": 1.0})
    result = TickConverter(changed).run()
    assert result.mode == "rebuild"
    assert result.swapped
    assert result.skipped_partitions == []


def test_a_dataset_without_a_manifest_is_rebuilt_transactionally(tmp_path, config_factory):
    source = three_months(tmp_path)
    config = config_factory(source)
    TickConverter(config).run()
    (config.processed_data_path / DATASET_MANIFEST).unlink()

    result = TickConverter(config).run()
    assert result.mode == "rebuild"
    assert read_all(config.processed_data_path).height == 900


def test_partial_files_are_invisible_to_readers_and_cleaned_up(tmp_path, config_factory):
    source = three_months(tmp_path)
    config = config_factory(source)
    TickConverter(config).run()
    stray = (config.processed_data_path / "year=2021" / "month=01"
             / "xauusd-ticks-2021-01.parquet.partial")
    stray.write_bytes(b"not parquet")
    with DataStore(config) as store:
        assert store.load_ticks("2021-01-01", "2021-04-01").height == 900
    TickConverter(config).run()
    assert not stray.exists()


def test_every_source_line_is_accounted_for(tmp_path, dirty_csv, config_factory):
    config = config_factory(dirty_csv)
    TickConverter(config).run()
    manifest = load_dataset_manifest(config.processed_data_path)
    assert manifest is not None and manifest["status"] == "complete"
    source_lines = len(dirty_csv.read_text(encoding="utf-8").splitlines()) - 1
    unassigned = manifest["unassigned"]
    assert manifest["source_lines_accounted"] == source_lines
    written = manifest["total_rows"]
    dropped = manifest["rows_dropped"] + unassigned["rows_dropped"]
    malformed = sum(e["malformed_rows"] for e in manifest["partitions"]) + unassigned[
        "malformed_rows"]
    assert written + dropped + malformed == source_lines


def test_dataset_version_is_content_addressed(tmp_path, config_factory):
    source = three_months(tmp_path)
    config = config_factory(source)
    first = TickConverter(config).run().dataset_version
    again = TickConverter(config).run(overwrite=True).dataset_version
    assert first == again and first.startswith("ticks-")

    text = source.read_bytes().replace(b"20210301 01:00:00.000,1920.000",
                                       b"20210301 01:00:00.000,1920.250")
    source.write_bytes(text)
    changed = TickConverter(config).run().dataset_version
    assert changed != first


def test_the_manifest_copy_in_metadata_matches_the_dataset(tmp_path, config_factory):
    source = three_months(tmp_path)
    config = config_factory(source)
    TickConverter(config).run()
    inside = load_dataset_manifest(config.processed_data_path)
    copy = json.loads(
        (config.metadata_path / converter_module.MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert inside is not None
    assert copy["dataset_version"] == inside["dataset_version"]
    assert copy["total_rows"] == inside["total_rows"] == 900
