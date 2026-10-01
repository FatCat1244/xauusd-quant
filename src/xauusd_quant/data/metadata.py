"""Dataset-level metadata computed from the converted Parquet dataset.

Everything here is measured over the *whole* Parquet dataset via DuckDB, not
from a sample, and each figure is labelled ``exact`` or ``approximate`` so a
reader never has to guess. Quantiles default to DuckDB's t-digest
(``approx_quantile``) because exact quantiles over 300M+ rows are expensive;
set ``metadata.exact_quantiles: true`` to pay for the exact version.

Validation and cleaning counts are *not* recomputed here. They come from the
reports the conversion pass wrote, since those are the only ones that saw the
raw rows before anything was dropped.
"""

from __future__ import annotations

import json
import time as _time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..utils.clock import utc_now_iso
from ..utils.config import Config
from ..utils.logging import format_duration, get_logger
from ..utils.paths import atomic_write_text, dir_size_bytes, ensure_dir, human_bytes
from .converter import (
    CLEANING_REPORT_NAME,
    MANIFEST_NAME,
    VALIDATION_REPORT_NAME,
    load_dataset_manifest,
)
from .loader import DataStore
from .schema import SCHEMA_VERSION
from .timezones import describe_policy

__all__ = ["DatasetMetadata", "build_dataset_metadata", "write_dataset_metadata"]

LOGGER = get_logger("data.metadata")

METADATA_NAME = "dataset_metadata.json"

TIMEZONE_EVIDENCE = (
    "Derived, not assumed, across the whole 2003-2026 export: US 08:30 "
    "America/New_York releases appear at 15:30 in this file in both DST halves of "
    "the year (clear from 2011 on, including the weeks when the EU and US DST "
    "calendars disagree), and the CME settlement lull (17:00-18:00 New York) sits "
    "at local 00:00-01:00 in every era and both seasons, which is what settles "
    "2003-2010. Re-derive it for any new export; see config/data.yaml."
)


@dataclass
class DatasetMetadata:
    """Summary of the converted tick dataset."""

    instrument: str = ""
    schema_version: str = SCHEMA_VERSION
    config_fingerprint: str = ""

    total_rows: int = 0
    start_timestamp: str | None = None
    end_timestamp: str | None = None
    trading_days: int = 0
    calendar_days_spanned: int = 0
    distinct_months: int = 0

    partitions: int = 0
    files: int = 0
    storage_bytes: int = 0
    storage_human: str = ""
    source_bytes: int = 0
    compression_ratio: float | None = None

    columns: list[str] = field(default_factory=list)
    price_min: float | None = None
    price_max: float | None = None
    bid_min: float | None = None
    ask_max: float | None = None

    spread_mean: float | None = None
    spread_median: float | None = None
    spread_min: float | None = None
    spread_max: float | None = None
    spread_percentiles: dict[str, float] = field(default_factory=dict)
    quantile_method: str = "approx_quantile (t-digest)"

    volume_sum: float | None = None
    volume_mean: float | None = None

    null_counts: dict[str, int] = field(default_factory=dict)
    duplicate_timestamps: int | None = None
    validation_counts: dict[str, int] = field(default_factory=dict)
    cleaning: dict[str, Any] = field(default_factory=dict)

    timezone: dict[str, Any] = field(default_factory=dict)
    bar_timeframes: dict[str, Any] = field(default_factory=dict)

    coverage: dict[str, Any] = field(default_factory=dict)
    dataset_version: str | None = None
    dataset_status: str | None = None
    warnings: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    generated_utc: str = field(
        default_factory=utc_now_iso
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_dataset_metadata(config: Config) -> DatasetMetadata:
    """Compute dataset metadata over the full Parquet dataset."""
    started = _time.perf_counter()
    meta = DatasetMetadata(
        instrument=config.instrument,
        config_fingerprint=config.fingerprint(),
        timezone=describe_policy(config.timezone, TIMEZONE_EVIDENCE).to_dict(),
    )

    root = config.processed_data_path
    files = sorted(root.rglob("*.parquet")) if root.exists() else []
    if not files:
        raise FileNotFoundError(
            f"No Parquet files under {root}. Run `xq convert` before `xq metadata`."
        )

    meta.files = len(files)
    meta.partitions = len({f.parent for f in files})
    meta.storage_bytes = dir_size_bytes(root, "**/*.parquet")
    meta.storage_human = human_bytes(meta.storage_bytes)

    with DataStore(config) as store:
        _core_stats(store, meta, config)
        _spread_stats(store, meta, config)
        _null_counts(store, meta)
        _bar_summary(store, meta, config)

    _attach_pass_reports(config, meta)
    meta.duration_seconds = _time.perf_counter() - started
    LOGGER.info(
        "Dataset metadata computed in %s: %s rows, %s, %d partitions",
        format_duration(meta.duration_seconds), f"{meta.total_rows:,}",
        meta.storage_human, meta.partitions,
    )
    return meta


def _core_stats(store: DataStore, meta: DatasetMetadata, config: Config) -> None:
    """Row count, span, trading days and price extremes - all exact."""
    glob = store.tick_glob()
    conn = store.connect()
    # DuckDB surfaces the Hive partition columns; they are directory names, not
    # tick data, so keep them out of the reported schema and null counts - the
    # same contract `DataStore.load_ticks` presents.
    all_columns = list(
        conn.execute(f"SELECT * FROM read_parquet('{glob}') LIMIT 0").pl().columns
    )
    meta.columns = [c for c in all_columns if c not in DataStore.HIVE_COLUMNS]

    has_bid = "bid" in meta.columns
    has_ask = "ask" in meta.columns
    price_expr = (
        "min(bid), max(ask)" if has_bid and has_ask
        else "min(mid), max(mid)" if "mid" in meta.columns
        else "NULL, NULL"
    )
    row = conn.execute(
        f"""
        SELECT count(*), min(timestamp), max(timestamp),
               count(DISTINCT CAST(timestamp AS DATE)),
               count(DISTINCT date_trunc('month', timestamp)),
               {price_expr}
        FROM read_parquet('{glob}')
        """  # noqa: S608
    ).fetchone()
    assert row is not None
    meta.total_rows = int(row[0])
    meta.start_timestamp = row[1].isoformat(sep=" ") if row[1] else None
    meta.end_timestamp = row[2].isoformat(sep=" ") if row[2] else None
    meta.trading_days = int(row[3])
    meta.distinct_months = int(row[4])
    meta.bid_min = float(row[5]) if row[5] is not None else None
    meta.ask_max = float(row[6]) if row[6] is not None else None
    meta.price_min, meta.price_max = meta.bid_min, meta.ask_max
    if row[1] and row[2]:
        meta.calendar_days_spanned = (row[2].date() - row[1].date()).days + 1

    if "volume" in meta.columns:
        vol = conn.execute(
            f"SELECT sum(volume), avg(volume) FROM read_parquet('{glob}')"  # noqa: S608
        ).fetchone()
        if vol:
            meta.volume_sum = float(vol[0]) if vol[0] is not None else None
            meta.volume_mean = float(vol[1]) if vol[1] is not None else None

    dupes = conn.execute(
        f"""
        SELECT count(*) FROM (
            SELECT timestamp FROM read_parquet('{glob}')
            GROUP BY timestamp HAVING count(*) > 1
        )
        """  # noqa: S608
    ).fetchone()
    meta.duplicate_timestamps = int(dupes[0]) if dupes else None

    sources = config.raw_data_path
    if sources.exists() and sources.is_file():
        meta.source_bytes = sources.stat().st_size
    if meta.source_bytes and meta.storage_bytes:
        meta.compression_ratio = meta.source_bytes / meta.storage_bytes


def _spread_stats(store: DataStore, meta: DatasetMetadata, config: Config) -> None:
    """Spread moments and percentiles over every tick."""
    if "spread" not in meta.columns:
        return
    glob = store.tick_glob()
    conn = store.connect()
    exact = config.metadata.exact_quantiles
    meta.quantile_method = "quantile_cont (exact)" if exact else "approx_quantile (t-digest)"

    row = conn.execute(
        f"SELECT avg(spread), min(spread), max(spread) FROM read_parquet('{glob}')"  # noqa: S608
    ).fetchone()
    assert row is not None
    meta.spread_mean = float(row[0]) if row[0] is not None else None
    meta.spread_min = float(row[1]) if row[1] is not None else None
    meta.spread_max = float(row[2]) if row[2] is not None else None

    func = "quantile_cont" if exact else "approx_quantile"
    percentiles = list(config.metadata.spread_percentiles)
    selects = ", ".join(
        f"{func}(spread, {p / 100.0}) AS p{str(p).replace('.', '_')}" for p in percentiles
    )
    values = conn.execute(
        f"SELECT {selects} FROM read_parquet('{glob}')"  # noqa: S608
    ).fetchone()
    if values:
        meta.spread_percentiles = {
            f"p{p}": float(v) for p, v in zip(percentiles, values, strict=True) if v is not None
        }
        meta.spread_median = meta.spread_percentiles.get("p50")


def _null_counts(store: DataStore, meta: DatasetMetadata) -> None:
    """Exact null count per column - what 'missing data' means post-conversion."""
    glob = store.tick_glob()
    selects = ", ".join(
        f'sum(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END) AS "{c}"' for c in meta.columns
    )
    row = store.connect().execute(
        f"SELECT {selects} FROM read_parquet('{glob}')"  # noqa: S608
    ).fetchone()
    if row:
        meta.null_counts = {
            c: int(v or 0) for c, v in zip(meta.columns, row, strict=True)
        }


def _bar_summary(store: DataStore, meta: DatasetMetadata, config: Config) -> None:
    """Row counts and spans for whichever bar timeframes exist."""
    for timeframe in store.available_timeframes():
        directory = config.bars_dir(timeframe)
        row = store.connect().execute(
            f"SELECT count(*), min(timestamp), max(timestamp) "  # noqa: S608
            f"FROM read_parquet('{store.bar_glob(timeframe)}')"
        ).fetchone()
        if not row:
            continue
        meta.bar_timeframes[timeframe] = {
            "bars": int(row[0]),
            "first_timestamp": row[1].isoformat(sep=" ") if row[1] else None,
            "last_timestamp": row[2].isoformat(sep=" ") if row[2] else None,
            "storage_bytes": dir_size_bytes(directory, "**/*.parquet"),
            "files": len(list(directory.rglob("*.parquet"))),
        }


def _attach_pass_reports(config: Config, meta: DatasetMetadata) -> None:
    """Fold in the conversion-pass reports and state exactly what is covered."""
    validation = _read_json(config.metadata_path / VALIDATION_REPORT_NAME)
    cleaning = _read_json(config.metadata_path / CLEANING_REPORT_NAME)
    # The manifest inside the dataset directory is authoritative; the copy
    # under metadata_path is only a convenience and may lag an interrupted run.
    manifest = load_dataset_manifest(config.processed_data_path) or _read_json(
        config.metadata_path / MANIFEST_NAME
    )
    status = manifest.get("status")
    meta.dataset_version = manifest.get("dataset_version")
    meta.dataset_status = status
    if status is not None and status != "complete":
        missing = manifest.get("missing_partitions") or []
        meta.warnings.append(
            f"The tick dataset is {status!r}, not complete: "
            f"{len(missing)} partition(s) missing"
            + (f" (first {', '.join(missing[:5])})" if missing else "")
            + ". Every figure here describes only what has been converted. Re-run "
            "`xq convert` to finish it."
        )

    meta.validation_counts = {
        k: v for k, v in (validation.get("counts") or {}).items() if v
    }
    meta.cleaning = {
        "rows_input": cleaning.get("rows_input"),
        "rows_output": cleaning.get("rows_output"),
        "rows_dropped": cleaning.get("rows_dropped"),
        "dropped_by_reason": cleaning.get("dropped_by_reason", {}),
        "flagged_by_check": cleaning.get("flagged_by_check", {}),
        "enabled_drop_rules": cleaning.get("enabled_drop_rules", []),
    }

    complete = bool(validation.get("complete_pass")) and status in (None, "complete")
    meta.coverage = {
        "scope": "entire Parquet dataset",
        "row_stats_exact": True,
        "quantiles_exact": config.metadata.exact_quantiles,
        "validation_counts_cover_full_raw_pass": complete,
        "validation_rows_checked": validation.get("rows_checked"),
        "manifest_total_rows": manifest.get("total_rows"),
        "manifest_partitions": manifest.get("partition_count"),
    }

    if not validation:
        meta.warnings.append(
            "No validation_report.json found: raw-input validation counts are unavailable. "
            "They can only be produced by a conversion pass."
        )
    elif not complete:
        meta.warnings.append(
            "validation_report.json covers only part of the raw input (the dataset is "
            "not complete). Its counts describe that portion, not the whole raw file. "
            "Re-run `xq convert` to finish the dataset."
        )
    manifest_rows = manifest.get("total_rows")
    if manifest_rows and manifest_rows != meta.total_rows:
        meta.warnings.append(
            f"Manifest records {manifest_rows:,} rows but the Parquet dataset holds "
            f"{meta.total_rows:,}. The output may have been modified outside the pipeline."
        )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        LOGGER.warning("Could not read %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def write_dataset_metadata(
    config: Config, meta: DatasetMetadata, name: str = METADATA_NAME
) -> Path:
    """Persist dataset metadata under ``metadata_path``."""
    path = ensure_dir(config.metadata_path) / name
    atomic_write_text(path, json.dumps(meta.to_dict(), indent=2, default=str) + "\n")
    LOGGER.info("Dataset metadata written: %s", path)
    return path
