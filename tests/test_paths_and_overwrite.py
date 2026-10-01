"""Path utilities and the surgical `--overwrite` behaviour."""

from __future__ import annotations

import pytest

from xauusd_quant.data.converter import TickConverter
from xauusd_quant.utils.paths import (
    atomic_write_text,
    dir_size_bytes,
    ensure_dir,
    find_project_root,
    human_bytes,
    resolve_path,
)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def test_project_root_is_the_directory_holding_pyproject():
    root = find_project_root()
    assert (root / "pyproject.toml").exists()


def test_relative_paths_resolve_against_the_given_root(tmp_path):
    assert resolve_path("data/parquet", tmp_path) == (tmp_path / "data" / "parquet").resolve()


def test_absolute_paths_pass_through(tmp_path):
    assert resolve_path(str(tmp_path / "x"), tmp_path) == tmp_path / "x"


def test_environment_variables_expand_in_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("XQ_PATH_TEST", "somewhere")
    assert resolve_path("${XQ_PATH_TEST}/x", tmp_path).parts[-2:] == ("somewhere", "x")


def test_ensure_dir_is_idempotent(tmp_path):
    target = tmp_path / "a" / "b"
    assert ensure_dir(target) == ensure_dir(target) == target
    assert target.is_dir()


def test_atomic_write_leaves_no_temporary_file(tmp_path):
    target = tmp_path / "report.json"
    atomic_write_text(target, '{"ok": true}')
    assert target.read_text(encoding="utf-8") == '{"ok": true}'
    assert [p.name for p in tmp_path.iterdir()] == ["report.json"]


def test_atomic_write_replaces_existing_content(tmp_path):
    target = tmp_path / "report.json"
    atomic_write_text(target, "first")
    atomic_write_text(target, "second")
    assert target.read_text(encoding="utf-8") == "second"


def test_atomic_write_creates_missing_parents(tmp_path):
    target = tmp_path / "deep" / "nested" / "x.json"
    atomic_write_text(target, "{}")
    assert target.exists()


def test_dir_size_counts_only_matching_files(tmp_path):
    (tmp_path / "a.parquet").write_bytes(b"x" * 100)
    (tmp_path / "b.txt").write_bytes(b"y" * 50)
    assert dir_size_bytes(tmp_path, "**/*.parquet") == 100
    assert dir_size_bytes(tmp_path) == 150
    assert dir_size_bytes(tmp_path / "absent") == 0


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, "0 B"), (512, "512 B"), (1024, "1.00 KiB"), (1536, "1.50 KiB"),
     (1048576, "1.00 MiB"), (14_776_440_068, "13.76 GiB")],
)
def test_human_bytes(value, expected):
    assert human_bytes(value) == expected


# ---------------------------------------------------------------------------
# --overwrite must not destroy files the pipeline did not create
# ---------------------------------------------------------------------------
def test_overwrite_removes_parquet_but_keeps_other_files(clean_config):
    TickConverter(clean_config).run()
    out = clean_config.processed_data_path
    keep = out / ".gitkeep"
    keep.write_text("", encoding="utf-8")
    notes = out / "NOTES.md"
    notes.write_text("hand-written", encoding="utf-8")
    assert list(out.rglob("*.parquet"))

    TickConverter(clean_config).run(resume=False, overwrite=True)

    assert keep.exists(), ".gitkeep must survive an --overwrite"
    assert notes.read_text(encoding="utf-8") == "hand-written"
    assert len(list(out.rglob("*.parquet"))) > 0  # rebuilt


def test_overwrite_leaves_no_stale_partitions(two_month_csv, clean_csv, config_factory):
    """Months absent from the new source must not linger from the old run."""
    config = config_factory(two_month_csv)
    TickConverter(config).run()
    assert len(list(config.processed_data_path.rglob("*.parquet"))) == 3

    narrower = config_factory(clean_csv)  # January only
    TickConverter(narrower).run(resume=False, overwrite=True)
    remaining = sorted(
        p.relative_to(narrower.processed_data_path).as_posix()
        for p in narrower.processed_data_path.rglob("*.parquet")
    )
    assert remaining == ["year=2021/month=01/xauusd-ticks-2021-01.parquet"]
