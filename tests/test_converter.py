"""End-to-end conversion: partitioning, resume, determinism, reports."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest

from conftest import make_clean_rows
from xauusd_quant.data.converter import MANIFEST_NAME, TickConverter
from xauusd_quant.data.schema import SCHEMA_VERSION


def parquet_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.parquet"))


def read_all(root: Path) -> pl.DataFrame:
    return pl.read_parquet(parquet_files(root)).sort("timestamp")


# ---------------------------------------------------------------------------
# Basic conversion
# ---------------------------------------------------------------------------
def test_converts_every_clean_row(clean_config):
    result = TickConverter(clean_config).run()
    assert result.rows_read == 600
    assert result.rows_written == 600
    assert result.rows_dropped == 0
    assert result.malformed_rows == 0
    assert read_all(clean_config.processed_data_path).height == 600


def test_output_has_the_canonical_schema(clean_config):
    TickConverter(clean_config).run()
    frame = read_all(clean_config.processed_data_path)
    assert frame.columns == [
        "timestamp", "timestamp_utc", "bid", "ask", "mid", "spread", "volume"
    ]
    assert frame.schema["timestamp"] == pl.Datetime("us")
    assert frame.schema["timestamp_utc"].time_zone == "UTC"


def test_derived_columns_are_correct_in_the_output(clean_config):
    TickConverter(clean_config).run()
    frame = read_all(clean_config.processed_data_path)
    assert frame["mid"].to_list() == pytest.approx(
        ((frame["bid"] + frame["ask"]) / 2).to_list()
    )
    assert frame["spread"].to_list() == pytest.approx(
        (frame["ask"] - frame["bid"]).to_list()
    )


def test_bid_and_ask_survive_the_round_trip_unchanged(clean_config, clean_csv):
    TickConverter(clean_config).run()
    frame = read_all(clean_config.processed_data_path)
    raw = [
        line.split(",") for line in
        clean_csv.read_text(encoding="utf-8").strip().splitlines()[1:]
    ]
    assert frame["bid"].to_list() == pytest.approx([float(r[1]) for r in raw])
    assert frame["ask"].to_list() == pytest.approx([float(r[2]) for r in raw])


def test_output_is_sorted_within_every_partition(clean_config):
    TickConverter(clean_config).run()
    for path in parquet_files(clean_config.processed_data_path):
        assert pl.read_parquet(path)["timestamp"].is_sorted()


def test_provenance_is_embedded_in_the_parquet_metadata(clean_config):
    TickConverter(clean_config).run()
    metadata = pq.read_schema(parquet_files(clean_config.processed_data_path)[0]).metadata
    assert metadata[b"xq_schema_version"].decode() == SCHEMA_VERSION
    assert metadata[b"xq_timezone_mode"].decode() == "anchored_dst"
    assert b"never rewritten" in metadata[b"xq_timestamp_column"]


def test_raw_source_is_never_modified(clean_config, clean_csv):
    before = (clean_csv.stat().st_size, clean_csv.read_bytes())
    TickConverter(clean_config).run()
    assert (clean_csv.stat().st_size, clean_csv.read_bytes()) == before


# ---------------------------------------------------------------------------
# Timezone
# ---------------------------------------------------------------------------
def test_utc_column_uses_the_configured_policy(clean_config):
    """January ticks are UTC+2 under the anchored-DST rule."""
    TickConverter(clean_config).run()
    row = read_all(clean_config.processed_data_path).row(0, named=True)
    assert row["timestamp"] == datetime(2021, 1, 4, 1, 0, 0)
    assert row["timestamp_utc"].replace(tzinfo=None) == datetime(2021, 1, 3, 23, 0, 0)


def test_naive_mode_emits_no_utc_column(clean_csv, config_factory):
    config = config_factory(
        clean_csv, timezone={"mode": "naive", "emit_utc_column": False,
                             "source_label": "unknown"}
    )
    TickConverter(config).run()
    assert "timestamp_utc" not in read_all(config.processed_data_path).columns


def test_local_timestamps_are_identical_whatever_the_timezone_policy(
    clean_csv, config_factory, tmp_path
):
    """The canonical timestamp must never move when the tz policy changes."""
    anchored = config_factory(clean_csv)
    TickConverter(anchored).run()
    local_a = read_all(anchored.processed_data_path)["timestamp"].to_list()

    fixed = config_factory(
        clean_csv,
        timezone={"mode": "fixed_offset", "fixed_offset_hours": 5,
                  "emit_utc_column": True, "source_label": "test"},
        processed_data_path=str(tmp_path / "parquet2").replace("\\", "/"),
    )
    TickConverter(fixed).run()
    assert read_all(fixed.processed_data_path)["timestamp"].to_list() == local_a


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------
def test_rows_land_in_year_month_partitions(two_month_csv, config_factory):
    config = config_factory(two_month_csv)
    result = TickConverter(config).run()
    names = {p.relative_to(config.processed_data_path).as_posix()
             for p in parquet_files(config.processed_data_path)}
    assert names == {
        "year=2021/month=01/xauusd-ticks-2021-01.parquet",
        "year=2021/month=02/xauusd-ticks-2021-02.parquet",
        "year=2021/month=03/xauusd-ticks-2021-03.parquet",
    }
    assert {p.key for p in result.partitions} == {"2021-01", "2021-02", "2021-03"}


def test_each_partition_holds_only_its_own_month(two_month_csv, config_factory):
    config = config_factory(two_month_csv)
    TickConverter(config).run()
    for path in parquet_files(config.processed_data_path):
        year = int(path.parent.parent.name.split("=")[1])
        month = int(path.parent.name.split("=")[1])
        stamps = pl.read_parquet(path)["timestamp"]
        assert stamps.dt.year().unique().to_list() == [year]
        assert stamps.dt.month().unique().to_list() == [month]


def test_partition_row_counts_sum_to_the_total(two_month_csv, config_factory):
    config = config_factory(two_month_csv)
    result = TickConverter(config).run()
    assert sum(p.rows for p in result.partitions) == result.rows_written == 900


def test_block_and_row_group_sizes_do_not_change_the_data(two_month_csv, config_factory,
                                                          tmp_path):
    """Parse-block and row-group boundaries must not affect content."""
    big = config_factory(two_month_csv, parquet={"row_group_rows": 100_000})
    TickConverter(big).run()
    reference = read_all(big.processed_data_path)

    small = config_factory(
        two_month_csv,
        conversion={"read_block_mb": 1, "index_block_mb": 1},
        parquet={"row_group_rows": 100},
        processed_data_path=str(tmp_path / "parquet_small").replace("\\", "/"),
        metadata_path=str(tmp_path / "meta_small").replace("\\", "/"),
    )
    TickConverter(small).run()
    assert read_all(small.processed_data_path).equals(reference)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_reconversion_produces_identical_data_and_digests(two_month_csv, config_factory):
    config = config_factory(two_month_csv)
    first = TickConverter(config).run()
    frame_a = read_all(config.processed_data_path)
    digests_a = {p.key: p.digest for p in first.partitions}

    second = TickConverter(config).run(resume=False, overwrite=True)
    frame_b = read_all(config.processed_data_path)
    digests_b = {p.key: p.digest for p in second.partitions}

    assert frame_a.equals(frame_b)
    assert digests_a == digests_b
    assert all(d for d in digests_a.values())


# ---------------------------------------------------------------------------
# Cleaning integration
# ---------------------------------------------------------------------------
def test_defective_rows_are_dropped_counted_and_reported(dirty_csv, config_factory):
    config = config_factory(dirty_csv)
    result = TickConverter(config).run()

    assert result.malformed_rows == 1
    assert result.rows_dropped == 4  # 2 unusable timestamps, 1 missing bid, 1 zero bid
    assert result.rows_read - result.rows_dropped == result.rows_written

    cleaning = json.loads(
        (config.metadata_path / "cleaning_report.json").read_text(encoding="utf-8")
    )
    assert cleaning["consistency_check"]["input_equals_output_plus_dropped"]
    assert cleaning["dropped_by_reason"]["missing_bid_or_ask"] == 1
    # Kept-but-flagged rows are still in the output.
    frame = read_all(config.processed_data_path)
    assert (frame["ask"] < frame["bid"]).sum() == 1


def test_reports_are_written(clean_config):
    TickConverter(clean_config).run()
    for name in (MANIFEST_NAME, "validation_report.json", "cleaning_report.json"):
        path = clean_config.metadata_path / name
        assert path.exists(), name
        assert json.loads(path.read_text(encoding="utf-8"))


def test_reports_carry_the_config_fingerprint(clean_config):
    TickConverter(clean_config).run()
    manifest = json.loads(
        (clean_config.metadata_path / MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert manifest["config_fingerprint"] == clean_config.fingerprint()
    assert manifest["schema_version"] == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------
def test_resume_skips_completed_partitions(two_month_csv, config_factory):
    config = config_factory(two_month_csv)
    TickConverter(config).run()

    again = TickConverter(config).run(resume=True)
    assert set(again.skipped_partitions) == {"2021-01", "2021-02", "2021-03"}
    assert again.rows_read == 0
    assert read_all(config.processed_data_path).height == 900


def test_resume_rebuilds_a_partition_deleted_from_disk(two_month_csv, config_factory):
    config = config_factory(two_month_csv)
    TickConverter(config).run()
    target = config.processed_data_path / "year=2021" / "month=03"
    for path in target.glob("*.parquet"):
        path.unlink()

    TickConverter(config).run(resume=True)
    assert read_all(config.processed_data_path).height == 900


def test_resume_is_refused_when_the_configuration_changed(two_month_csv, config_factory):
    config = config_factory(two_month_csv)
    TickConverter(config).run()

    changed = config_factory(two_month_csv, validation={"max_spread": 1.0})
    assert changed.fingerprint() != config.fingerprint()
    result = TickConverter(changed).run(resume=True)
    assert result.skipped_partitions == []
    assert result.rows_read == 900


def test_appending_to_the_source_converts_only_the_new_month(two_month_csv, config_factory):
    """A changed source is detected, but unchanged months are proven unchanged.

    Every existing partition is re-verified against the *content hash* of its
    own source lines, so appending April rebuilds April alone - the manifest
    is never trusted blindly and nothing is reconverted without cause.
    """
    config = config_factory(two_month_csv)
    TickConverter(config).run()

    extra = make_clean_rows(datetime(2021, 4, 1, 1, 0, 0), 50, bid0=1930.0)
    with two_month_csv.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(extra) + "\n")

    result = TickConverter(config).run(resume=True)
    assert set(result.skipped_partitions) == {"2021-01", "2021-02", "2021-03"}
    assert [p.key for p in result.partitions] == ["2021-04"]
    assert result.rows_read == 50
    assert result.status == "complete"
    assert read_all(config.processed_data_path).height == 950


def test_editing_one_month_of_the_source_rebuilds_only_that_month(two_month_csv,
                                                                   config_factory):
    config = config_factory(two_month_csv)
    TickConverter(config).run()

    text = two_month_csv.read_bytes()
    # Same length, different price: only February's source bytes change.
    two_month_csv.write_bytes(text.replace(b"20210201 01:00:00.000,1910.000",
                                           b"20210201 01:00:00.000,1910.500"))

    result = TickConverter(config).run(resume=True)
    assert [p.key for p in result.partitions] == ["2021-02"]
    assert result.rebuild_reasons["2021-02"] == "its source lines changed"
    assert set(result.skipped_partitions) == {"2021-01", "2021-03"}
    frame = read_all(config.processed_data_path)
    assert 1910.5 in frame["bid"].to_list()


def test_resumed_output_matches_a_single_pass(two_month_csv, config_factory, tmp_path):
    """Byte-offset resume must reconstruct exactly the same dataset."""
    reference = config_factory(
        two_month_csv,
        processed_data_path=str(tmp_path / "parquet_ref").replace("\\", "/"),
        metadata_path=str(tmp_path / "meta_ref").replace("\\", "/"),
    )
    TickConverter(reference).run()
    expected = read_all(reference.processed_data_path)

    config = config_factory(two_month_csv)
    TickConverter(config).run()
    # Force the last partition to be rebuilt through the seek path.
    for path in (config.processed_data_path / "year=2021" / "month=03").glob("*.parquet"):
        path.unlink()
    TickConverter(config).run(resume=True)

    assert read_all(config.processed_data_path).equals(expected)


# ---------------------------------------------------------------------------
# Limits and safety
# ---------------------------------------------------------------------------
def test_limit_rows_writes_a_smoke_dataset_and_never_touches_the_canonical_one(
    clean_config,
):
    """A smoke test once replaced a finished dataset. It must not be able to."""
    TickConverter(clean_config).run()
    canonical = read_all(clean_config.processed_data_path)
    manifest_before = (clean_config.metadata_path / MANIFEST_NAME).read_text(encoding="utf-8")

    result = TickConverter(clean_config).run(limit_rows=100, resume=False)
    assert result.truncated_by_limit
    assert result.rows_read == 100
    assert result.mode == "smoke"
    assert result.to_dict()["compression_ratio"] is None

    smoke = Path(result.dataset_dir)
    assert smoke != clean_config.processed_data_path
    assert read_all(smoke).height == 100
    smoke_manifest = json.loads((smoke / "_manifest.json").read_text(encoding="utf-8"))
    assert smoke_manifest["status"] == "partial"
    assert smoke_manifest["truncated_by_limit"] is True

    # The canonical dataset, its manifest and its reports are exactly as before.
    assert read_all(clean_config.processed_data_path).equals(canonical)
    assert (clean_config.metadata_path / MANIFEST_NAME).read_text(
        encoding="utf-8") == manifest_before


def test_limit_rows_refuses_to_target_the_canonical_directory(clean_config):
    with pytest.raises(ValueError, match="never writes to the canonical dataset"):
        TickConverter(clean_config).run(
            limit_rows=10, output_dir=clean_config.processed_data_path
        )


def test_missing_source_raises_a_clear_error(clean_csv, config_factory):
    config = config_factory(clean_csv, raw_data_path="does/not/exist.csv")
    with pytest.raises(FileNotFoundError, match="No raw data found"):
        TickConverter(config).run()


def test_misconfigured_column_names_raise_a_clear_error(clean_csv, config_factory):
    config = config_factory(clean_csv, bid_column="NotAColumn")
    with pytest.raises(KeyError, match="not in the header"):
        TickConverter(config).run()


# ---------------------------------------------------------------------------
# The digest must fingerprint content, not physical layout
# ---------------------------------------------------------------------------
@pytest.fixture
def many_rows_csv(tmp_path: Path) -> Path:
    """Two months large enough to arrive in several CSV blocks, with nulls.

    Both properties matter:

    * A 1 MiB block already holds ~24k of these rows, so a smaller month would
      arrive in a single batch and every configuration would produce exactly
      one flush - making a chunking test pass vacuously.
    * Without nulls, a digest that mixes a column's values with its null mask
      still looks chunk-invariant, because an all-false mask is
      indistinguishable however it is split.
    """
    from conftest import write_csv

    rows = (
        make_clean_rows(datetime(2021, 1, 10, 1, 0, 0), 60_000, step_ms=50, bid0=1900.0)
        + make_clean_rows(datetime(2021, 2, 10, 1, 0, 0), 60_000, step_ms=50, bid0=1910.0)
    )
    # Scatter empty volume cells so `volume` really carries nulls.
    for i in range(37, len(rows), 211):
        rows[i] = rows[i].rsplit(",", 1)[0] + ","
    return write_csv(tmp_path / "many.csv", rows)


def _row_groups(root: Path) -> int:
    return sum(pq.read_metadata(f).num_row_groups for f in parquet_files(root))


def test_digest_is_invariant_to_chunking(many_rows_csv, config_factory, tmp_path):
    """The digest must fingerprint content, never the way it was chunked.

    The digest is fed one row group at a time, so `row_group_rows` sets how
    many separate updates each column's hashers receive. Comparing row-group
    counts is a direct measurement that the two runs really did chunk
    differently - the condition the digest has to be invariant to - and the
    parse block size differs as well.
    """
    coarse = config_factory(
        many_rows_csv,
        conversion={"read_block_mb": 64},
        parquet={"row_group_rows": 10_000_000},                       # one chunk/month
    )
    a = {p.key: p.digest for p in TickConverter(coarse).run().partitions}

    fine = config_factory(
        many_rows_csv,
        conversion={"read_block_mb": 1},
        parquet={"row_group_rows": 1_000},                            # 60 chunks/month
        processed_data_path=str(tmp_path / "parquet_fine").replace("\\", "/"),
        metadata_path=str(tmp_path / "meta_fine").replace("\\", "/"),
    )
    b = {p.key: p.digest for p in TickConverter(fine).run().partitions}

    # Guard the guard: prove the two runs really chunked differently.
    coarse_flushes, fine_flushes = _row_groups(coarse.processed_data_path), _row_groups(
        fine.processed_data_path
    )
    assert fine_flushes > coarse_flushes, (
        f"both runs chunked identically ({coarse_flushes}); test is vacuous"
    )
    nulls = read_all(coarse.processed_data_path)["volume"].null_count()
    assert nulls > 0, "no nulls present; the null-mask stream would not be exercised"

    assert a == b, "digest changed with chunking, so it fingerprints layout not content"
    assert all(v for v in a.values())
    # And the data itself really is identical.
    assert read_all(fine.processed_data_path).equals(read_all(coarse.processed_data_path))


def test_digest_changes_when_the_data_changes(two_month_csv, config_factory, tmp_path):
    before = {p.key: p.digest for p in TickConverter(config_factory(two_month_csv)).run().partitions}

    edited = tmp_path / "edited.csv"
    lines = two_month_csv.read_bytes().split(b"\n")
    lines[1] = lines[1].replace(b"1900.000", b"1900.500")
    edited.write_bytes(b"\n".join(lines))

    after_config = config_factory(
        edited, processed_data_path=str(tmp_path / "parquet_edited").replace("\\", "/"),
        metadata_path=str(tmp_path / "meta_edited").replace("\\", "/"),
    )
    after = {p.key: p.digest for p in TickConverter(after_config).run().partitions}
    assert before["2021-01"] != after["2021-01"]
    assert before["2021-03"] == after["2021-03"], "untouched months must be unchanged"


def test_resumed_partition_digest_matches_a_single_pass(two_month_csv, config_factory,
                                                        tmp_path):
    reference = config_factory(
        two_month_csv,
        processed_data_path=str(tmp_path / "pq_ref2").replace("\\", "/"),
        metadata_path=str(tmp_path / "meta_ref2").replace("\\", "/"),
    )
    expected = {p.key: p.digest for p in TickConverter(reference).run().partitions}

    config = config_factory(two_month_csv, conversion={"read_block_mb": 1})
    TickConverter(config).run()
    for stale in (config.processed_data_path / "year=2021" / "month=03").glob("*.parquet"):
        stale.unlink()
    rebuilt = {p.key: p.digest for p in TickConverter(config).run(resume=True).partitions}

    assert rebuilt["2021-03"] == expected["2021-03"]


def test_resume_rebuilds_a_partition_deleted_from_the_middle(two_month_csv, config_factory):
    """Each partition is independent, so a hole is filled without touching its neighbours."""
    config = config_factory(two_month_csv)
    TickConverter(config).run()
    for stale in (config.processed_data_path / "year=2021" / "month=02").glob("*.parquet"):
        stale.unlink()

    result = TickConverter(config).run(resume=True)

    assert set(result.skipped_partitions) == {"2021-01", "2021-03"}
    assert [p.key for p in result.partitions] == ["2021-02"]
    assert result.rebuild_reasons["2021-02"] == "file missing"
    assert read_all(config.processed_data_path).height == 900
    assert len(parquet_files(config.processed_data_path)) == 3


def test_resume_after_a_middle_gap_matches_a_single_pass(two_month_csv, config_factory,
                                                         tmp_path):
    reference = config_factory(
        two_month_csv,
        processed_data_path=str(tmp_path / "pq_mid").replace("\\", "/"),
        metadata_path=str(tmp_path / "meta_mid").replace("\\", "/"),
    )
    expected = read_all(TickConverter(reference).run() and reference.processed_data_path)

    config = config_factory(two_month_csv)
    TickConverter(config).run()
    for stale in (config.processed_data_path / "year=2021" / "month=02").glob("*.parquet"):
        stale.unlink()
    TickConverter(config).run(resume=True)

    assert read_all(config.processed_data_path).equals(expected)


# ---------------------------------------------------------------------------
# A source that ships its own spread column
# ---------------------------------------------------------------------------
@pytest.fixture
def csv_with_spread(tmp_path: Path) -> Path:
    """Rows carrying a Spread column that deliberately disagrees in places.

    Real exports usually have spread == ask - bid, but the pipeline must not
    depend on that: the source value is preserved so it can be *checked*.
    """
    from conftest import fmt

    base = datetime(2021, 1, 4, 1, 0, 0)
    rows = []
    for i in range(20):
        moment = base + timedelta(milliseconds=i * 100)
        bid = round(2000.0 + i * 0.1, 3)
        ask = round(bid + 0.30, 3)
        # Rows 5 and 11 carry a source spread that does NOT match ask - bid.
        source_spread = 9.99 if i in (5, 11) else round(ask - bid, 3)
        rows.append(f"{fmt(moment)},{bid:.3f},{ask:.3f},120,{source_spread}")
    path = tmp_path / "with_spread.csv"
    path.write_bytes(
        ("DateTime,Bid,Ask,Volume,Spread\n" + "\n".join(rows) + "\n").encode()
    )
    return path


def test_source_spread_is_kept_and_never_overwrites_the_derived_one(
    csv_with_spread, config_factory
):
    config = config_factory(csv_with_spread, spread_column="Spread")
    TickConverter(config).run()
    frame = read_all(config.processed_data_path)

    assert "spread" in frame.columns and "spread_source" in frame.columns
    # `spread` is always derived from bid/ask, regardless of the source column.
    assert frame["spread"].to_list() == pytest.approx(
        (frame["ask"] - frame["bid"]).to_list()
    )
    # The source values survive untouched, including the two that disagree.
    disagreeing = frame.filter(
        (pl.col("spread") - pl.col("spread_source")).abs() > 1e-9
    )
    assert disagreeing.height == 2
    assert disagreeing["spread_source"].to_list() == pytest.approx([9.99, 9.99])


def test_a_disagreeing_source_spread_does_not_trip_the_cleaner(
    csv_with_spread, config_factory
):
    """Nothing should be dropped just because the two spreads differ."""
    config = config_factory(csv_with_spread, spread_column="Spread")
    result = TickConverter(config).run()
    assert result.rows_read == 20
    assert result.rows_dropped == 0


def test_validator_uses_the_derived_spread_not_the_source_one(
    csv_with_spread, config_factory
):
    """An absurd source spread must not be mistaken for an extreme market spread."""
    config = config_factory(
        csv_with_spread, spread_column="Spread", validation={"max_spread": 5.0}
    )
    TickConverter(config).run()
    report = json.loads(
        (config.metadata_path / "validation_report.json").read_text(encoding="utf-8")
    )
    assert report["counts"]["extreme_spread"] == 0, (
        "the 9.99 source values should not be read as real spreads"
    )
