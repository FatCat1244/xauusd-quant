"""Independent verification of the converted tick dataset.

The converter already checks each partition before renaming it into place.
This module does not take its word for anything: it re-reads every partition
from disk and recomputes, from the data alone,

* ordering (the output must have no backward step at all), month purity and
  the content digest recorded in the manifest;
* duplicates, crossed quotes, invalid prices, nulls and derived-column drift;
* the schema of every file against the first;
* coverage - earliest and latest tick, months present per year, months missing
  from the output that the source has, and months missing from the source
  itself (genuine gaps in history, reported but never filled);
* every gap longer than ``validation.large_gap_seconds``, classified against
  the configured trading schedule, including gaps that span a month boundary;
* agreement between the files on disk, the manifest, and the source index.

It writes ``dataset_validation.json`` and a small ``tick_coverage.csv``. A
dataset "passes" only when every *pipeline* check is clean; missing history in
the raw source is described, not treated as a pipeline failure.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq

from ..utils.clock import utc_now_iso
from ..utils.config import Config
from ..utils.logging import format_duration, get_logger
from ..utils.paths import atomic_write_text, ensure_dir, human_bytes
from .converter import _PartitionDigest, dataset_version, load_dataset_manifest
from .diagnostics import _classify
from .schema import TIMESTAMP
from .source_index import INDEX_NAME, SourceIndex

__all__ = ["DatasetVerification", "verify_tick_dataset", "write_verification"]

LOGGER = get_logger("data.dataset_validation")

VERIFICATION_NAME = "dataset_validation.json"
COVERAGE_NAME = "tick_coverage.csv"
_MAX_LISTED_GAPS = 50


@dataclass
class DatasetVerification:
    """Everything re-derived from the files on disk."""

    dataset_dir: str = ""
    generated_utc: str = field(default_factory=utc_now_iso)
    deep: bool = True
    passed: bool = False
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    dataset_version: str | None = None
    dataset_version_recomputed: str | None = None
    manifest_status: str | None = None
    raw_fingerprint: str | None = None

    earliest_timestamp: str | None = None
    latest_timestamp: str | None = None
    total_rows: int = 0
    partition_count: int = 0
    storage_bytes: int = 0
    partitions_by_year: dict[str, int] = field(default_factory=dict)
    months_by_year: dict[str, list[int]] = field(default_factory=dict)
    months_missing_from_output: list[str] = field(default_factory=list)
    months_missing_from_source: list[str] = field(default_factory=list)

    non_monotonic_rows: int = 0
    impure_rows: int = 0
    duplicate_timestamps: int = 0
    exact_duplicate_rows: int = 0
    ask_below_bid: int = 0
    non_positive_prices: int = 0
    non_finite_prices: int = 0
    prices_out_of_range: int = 0
    null_counts: dict[str, int] = field(default_factory=dict)
    max_mid_error: float = 0.0
    max_spread_error: float = 0.0
    digest_mismatches: list[str] = field(default_factory=list)
    schema_mismatches: list[str] = field(default_factory=list)
    manifest_disagreements: list[str] = field(default_factory=list)

    gap_threshold_seconds: float = 0.0
    gaps_by_category: dict[str, int] = field(default_factory=dict)
    gaps_by_year: dict[str, dict[str, int]] = field(default_factory=dict)
    largest_unexplained_gaps: list[dict[str, Any]] = field(default_factory=list)

    source_reconciliation: dict[str, Any] = field(default_factory=dict)
    coverage: list[dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["storage_human"] = human_bytes(self.storage_bytes)
        return payload


def verify_tick_dataset(
    config: Config, *, dataset_dir: Path | None = None, deep: bool = True
) -> DatasetVerification:
    """Re-derive every integrity and coverage figure from the files on disk."""
    started = time.perf_counter()
    root = Path(dataset_dir) if dataset_dir else config.processed_data_path
    result = DatasetVerification(dataset_dir=str(root), deep=deep)
    manifest = load_dataset_manifest(root)
    if manifest is None:
        result.failures.append(f"No manifest in {root}; the dataset cannot be verified.")
        return result
    result.manifest_status = manifest.get("status")
    result.dataset_version = manifest.get("dataset_version")
    result.raw_fingerprint = manifest.get("raw_fingerprint")
    if result.manifest_status != "complete":
        result.failures.append(f"Manifest status is {result.manifest_status!r}, not complete.")

    entries = {e["key"]: e for e in manifest.get("partitions", [])}
    files = sorted(root.rglob("*.parquet"))
    recorded = {str((root / e["path"]).resolve()) for e in entries.values() if e.get("path")}
    on_disk = {str(f.resolve()) for f in files}
    for extra in sorted(on_disk - recorded):
        result.manifest_disagreements.append(f"file not in the manifest: {extra}")
    for missing in sorted(recorded - on_disk):
        result.manifest_disagreements.append(f"manifest file missing on disk: {missing}")

    threshold = float(config.validation.large_gap_seconds)
    result.gap_threshold_seconds = threshold
    session = config.diagnostics.session
    reference_schema = None
    previous_last: datetime | None = None
    unexplained: list[dict[str, Any]] = []
    null_totals: dict[str, int] = {}
    columns = None if deep else [TIMESTAMP, "bid", "ask", "volume"]

    for key in sorted(entries):
        entry = entries[key]
        if not entry.get("path"):
            continue
        path = root / entry["path"]
        if not path.exists():
            continue
        schema = pq.read_schema(path).remove_metadata()
        if reference_schema is None:
            reference_schema = schema
        elif not schema.equals(reference_schema):
            result.schema_mismatches.append(key)
        frame = pl.read_parquet(path, columns=columns)
        year, month = int(key[:4]), int(key[5:7])
        _check_partition(frame, key, year, month, entry, config, result, null_totals, deep)

        stamps = frame[TIMESTAMP]
        first, last = stamps[0], stamps[-1]
        result.total_rows += frame.height
        result.storage_bytes += path.stat().st_size
        result.partition_count += 1
        result.partitions_by_year[key[:4]] = result.partitions_by_year.get(key[:4], 0) + 1
        result.months_by_year.setdefault(key[:4], []).append(month)
        if result.earliest_timestamp is None:
            result.earliest_timestamp = first.isoformat(sep=" ")
        result.latest_timestamp = last.isoformat(sep=" ")

        gaps = _gap_rows(frame, threshold)
        if previous_last is not None and (first - previous_last).total_seconds() > threshold:
            gaps.insert(0, (previous_last, first))
        for start, end in gaps:
            category = _classify_tick_gap(start, end, session)
            year_key = str(start.year)
            result.gaps_by_category[category] = result.gaps_by_category.get(category, 0) + 1
            bucket = result.gaps_by_year.setdefault(year_key, {})
            bucket[category] = bucket.get(category, 0) + 1
            if category == "holiday_or_unknown":
                unexplained.append({
                    "from": start.isoformat(sep=" "), "to": end.isoformat(sep=" "),
                    "hours": round((end - start).total_seconds() / 3600.0, 3),
                })
        previous_last = last
        del frame

    unexplained.sort(key=lambda g: g["hours"], reverse=True)
    result.largest_unexplained_gaps = unexplained[:_MAX_LISTED_GAPS]
    result.null_counts = {k: v for k, v in null_totals.items() if v}
    _coverage(result, manifest, config)
    _reconcile(result, manifest, config)

    recomputed = dataset_version(manifest)
    result.dataset_version_recomputed = recomputed
    if manifest.get("dataset_version") and manifest["dataset_version"] != recomputed:
        result.failures.append("dataset_version does not match the manifest contents")
    if result.total_rows != int(manifest.get("total_rows") or 0):
        result.manifest_disagreements.append(
            f"rows on disk {result.total_rows:,} != manifest {manifest.get('total_rows'):,}"
        )

    for name, value in (
        ("non-monotonic output rows", result.non_monotonic_rows),
        ("rows outside their partition's month", result.impure_rows),
    ):
        if value:
            result.failures.append(f"{value:,} {name}")
    if result.digest_mismatches:
        result.failures.append(f"content digest mismatch in {result.digest_mismatches}")
    if result.schema_mismatches:
        result.failures.append(f"schema differs in {result.schema_mismatches}")
    if result.manifest_disagreements:
        result.failures.append(f"{len(result.manifest_disagreements)} manifest disagreement(s)")
    if result.months_missing_from_output:
        result.failures.append(
            f"months present in the source but missing from the output: "
            f"{result.months_missing_from_output}"
        )
    if not result.source_reconciliation.get("balanced", False):
        result.failures.append("source lines do not reconcile with rows written/dropped")
    result.passed = not result.failures

    result.notes += [
        "Missing months in `months_missing_from_source` are absent from the RAW export "
        "itself: genuine gaps in history. They are reported, never filled.",
        "Duplicate and crossed-quote counts describe what the cleaning policy KEPT; "
        "they are information, not pipeline failures.",
        "Gap categories come from diagnostics.session in config/data.yaml; "
        "`holiday_or_unknown` is simply what that schedule cannot explain.",
    ]
    result.duration_seconds = time.perf_counter() - started
    LOGGER.info(
        "Verified %d partitions, %s rows in %s: %s",
        result.partition_count, f"{result.total_rows:,}",
        format_duration(result.duration_seconds), "PASSED" if result.passed else
        f"FAILED ({'; '.join(result.failures)})",
    )
    return result


def _check_partition(
    frame: pl.DataFrame, key: str, year: int, month: int, entry: dict[str, Any],
    config: Config, result: DatasetVerification, nulls: dict[str, int], deep: bool,
) -> None:
    stamps = frame[TIMESTAMP]
    diffs = stamps.diff().dt.total_microseconds().drop_nulls()
    result.non_monotonic_rows += int((diffs < 0).sum())
    result.duplicate_timestamps += int((diffs == 0).sum())
    result.impure_rows += int(
        ((stamps.dt.year() != year) | (stamps.dt.month() != month)).sum()
    )
    keys = [c for c in (TIMESTAMP, "bid", "ask", "volume") if c in frame.columns]
    result.exact_duplicate_rows += frame.height - frame.select(
        pl.struct(keys).n_unique()
    ).item()
    bid, ask = frame["bid"], frame["ask"]
    result.ask_below_bid += int((ask < bid).sum())
    result.non_positive_prices += int(((bid <= 0) | (ask <= 0)).sum())
    result.non_finite_prices += int(
        (bid.is_nan() | ask.is_nan() | bid.is_infinite() | ask.is_infinite()).sum()
    )
    cfg = config.validation
    result.prices_out_of_range += int(((bid < cfg.min_price) | (ask > cfg.max_price)).sum())
    for name, count in zip(frame.columns, frame.null_count().row(0), strict=True):
        nulls[name] = nulls.get(name, 0) + int(count)
    if frame.height != int(entry.get("rows") or 0):
        result.manifest_disagreements.append(
            f"{key}: {frame.height:,} rows on disk, manifest says {entry.get('rows')}"
        )
    if not deep:
        return
    if "mid" in frame.columns:
        error = frame.select(((pl.col("bid") + pl.col("ask")) / 2 - pl.col("mid")).abs().max())
        result.max_mid_error = max(result.max_mid_error, float(error.item() or 0.0))
    if "spread" in frame.columns:
        error = frame.select((pl.col("ask") - pl.col("bid") - pl.col("spread")).abs().max())
        result.max_spread_error = max(result.max_spread_error, float(error.item() or 0.0))
    digest = _PartitionDigest()
    digest.update(frame)
    recomputed = digest.hexdigest(frame.columns)
    if entry.get("digest") and recomputed != entry["digest"]:
        result.digest_mismatches.append(key)


_TICK_TOLERANCE = timedelta(minutes=5)


def _classify_tick_gap(start: datetime, end: datetime, session: Any) -> str:
    """Classify a gap between two TICKS against the trading schedule.

    The bar-level classifier expects the first bar after the daily break to be
    stamped exactly at the reopen. Ticks never are: the last tick before the
    break lands at 23:59:5x and the first after it at 01:00:0x, so the gap
    overhangs the configured window by seconds on each side. A gap whose two
    edges each lie within five minutes of the break window's edges is the
    daily break; everything else keeps the bar-level classification.
    """
    category = _classify(start, end, session)
    if category != "holiday_or_unknown":
        return category
    opening = _parse_clock(session.daily_break_start)
    closing = _parse_clock(session.daily_break_end)
    if opening is None or closing is None:
        return category
    for day in {start.date(), end.date()}:
        window_start = datetime.combine(day, opening)
        window_end = datetime.combine(day, closing)
        if window_end <= window_start:
            window_end += timedelta(days=1)
        if (abs(start - window_start) <= _TICK_TOLERANCE
                and abs(end - window_end) <= _TICK_TOLERANCE):
            return "daily_break"
    return category


def _parse_clock(text: str | None) -> Any:
    if not text:
        return None
    try:
        hour, minute = (int(part) for part in text.split(":")[:2])
    except ValueError:
        return None
    from datetime import time as clock

    return clock(hour, minute)


def _gap_rows(frame: pl.DataFrame, threshold: float) -> list[tuple[datetime, datetime]]:
    stamps = frame.select(
        pl.col(TIMESTAMP).shift(1).alias("start"), pl.col(TIMESTAMP).alias("end"),
        (pl.col(TIMESTAMP) - pl.col(TIMESTAMP).shift(1)).dt.total_milliseconds().alias("ms"),
    ).filter(pl.col("ms") > threshold * 1000.0)
    return list(zip(stamps["start"].to_list(), stamps["end"].to_list(), strict=True))


def _coverage(result: DatasetVerification, manifest: dict[str, Any], config: Config) -> None:
    """Months per year, what is missing, and why."""
    index = _load_index(config)
    source_months = set(index.partitions) if index else set()
    output_months = {f"{y}-{m:02d}" for y, months in result.months_by_year.items() for m in months}
    if index:
        result.months_missing_from_output = sorted(source_months - output_months)
    if not output_months:
        return
    first, last = min(output_months), max(output_months)
    calendar = _month_range(first, last)
    result.months_missing_from_source = sorted(
        m for m in calendar if m not in source_months and m not in output_months
    )
    rows_by_key = {e["key"]: int(e.get("rows") or 0) for e in manifest.get("partitions", [])}
    first_by_key = {e["key"]: e.get("first_timestamp") for e in manifest.get("partitions", [])}
    last_by_key = {e["key"]: e.get("last_timestamp") for e in manifest.get("partitions", [])}
    for year in sorted({m[:4] for m in calendar}):
        expected = [m for m in calendar if m.startswith(year)]
        present = [m for m in expected if m in output_months]
        missing = [m for m in expected if m not in output_months]
        status = "complete" if not missing else f"missing {len(missing)}: {', '.join(missing)}"
        if year == first[:4] and first[5:] != "01":
            status += f" (history starts {first})"
        if year == last[:4] and last[5:] != "12":
            status += f" (history ends {last})"
        result.coverage.append({
            "year": int(year),
            "months_present": len(present),
            "months_expected": len(expected),
            "tick_rows": sum(rows_by_key.get(m, 0) for m in present),
            "start": first_by_key.get(present[0]) if present else None,
            "end": last_by_key.get(present[-1]) if present else None,
            "status": status,
        })


def _reconcile(result: DatasetVerification, manifest: dict[str, Any], config: Config) -> None:
    """Source lines = rows written + rows dropped + malformed, with nothing left over."""
    entries = manifest.get("partitions", [])
    unassigned = manifest.get("unassigned") or {}
    written = sum(int(e.get("rows") or 0) for e in entries)
    dropped = sum(int(e.get("rows_dropped") or 0) for e in entries) + int(
        unassigned.get("rows_dropped") or 0
    )
    malformed = sum(int(e.get("malformed_rows") or 0) for e in entries) + int(
        unassigned.get("malformed_rows") or 0
    )
    index = _load_index(config)
    lines = index.total_lines if index else None
    by_reason: dict[str, int] = {}
    for e in [*entries, unassigned]:
        for reason, count in (e.get("dropped_by_reason") or {}).items():
            by_reason[reason] = by_reason.get(reason, 0) + int(count)
    result.source_reconciliation = {
        "source_lines": lines,
        "rows_written": written,
        "rows_dropped": dropped,
        "dropped_by_reason": by_reason,
        "malformed_rows": malformed,
        "unassigned_lines": unassigned.get("source_lines", 0),
        "balanced": lines is not None and written + dropped + malformed == lines,
        "raw_fingerprint_matches_index": bool(
            index and index.raw_fingerprint == manifest.get("raw_fingerprint")
        ),
    }


def _load_index(config: Config) -> SourceIndex | None:
    path = config.metadata_path / INDEX_NAME
    if not path.exists():
        return None
    try:
        return SourceIndex.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError, TypeError, KeyError):
        return None


def _month_range(first: str, last: str) -> list[str]:
    year, month = int(first[:4]), int(first[5:7])
    end = (int(last[:4]), int(last[5:7]))
    out: list[str] = []
    while (year, month) <= end:
        out.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def write_verification(config: Config, result: DatasetVerification) -> tuple[Path, Path]:
    """Persist the verification report and the per-year coverage table."""
    directory = ensure_dir(config.metadata_path)
    report = directory / VERIFICATION_NAME
    atomic_write_text(report, json.dumps(result.to_dict(), indent=2, default=str) + "\n")
    coverage = directory / COVERAGE_NAME
    if result.coverage:
        pl.DataFrame(result.coverage).write_csv(coverage)
    return report, coverage
