"""Partition-safe CSV to Parquet conversion.

Architecture
------------
::

    raw files --(SourceIndex: one read-only pass)--> byte runs per year-month
    for each partition, independently:
        read exactly its runs -> check their content hash against the index
        -> parse -> canonicalise -> validate the whole month in source order
        -> clean (explicit policy) -> stable sort by timestamp -> check order
        -> write <file>.parquet.partial -> re-open and verify
        -> atomic rename -> record in the manifest

Nothing here depends on the source being sorted. A row that arrives late -
even one that belongs to a month the stream has already left - is read from its
own byte run and sorted into place with the rest of its month. Out-of-order
rows are never dropped for being out of order; only the configured cleaning
policy removes anything, and every removal is counted per partition.

The previous design streamed the file once, closed a month as soon as a later
one appeared and *recreated* a closed month's file if another row for it
turned up, which would have silently replaced the month with that one row. It
also refused to resume at all once any row was out of order.

Directories
-----------
``data/parquet/``              the current dataset; its manifest is ``_manifest.json``
``data/parquet.building/``     a complete rebuild in progress (``--overwrite``)
``data/parquet.previous/``     the replaced dataset, only while a swap is running
``data/parquet.smoke/``        ``--limit-rows`` output; never the canonical dataset

A partition is only ever visible under its final name after it has been
written, re-opened and verified; until then it is ``*.parquet.partial``, which
no reader globs. The manifest lives inside the dataset directory, so a dataset
and its manifest always move together.

Modes
-----
``incremental`` (default)
    Verify every partition of the current dataset against its manifest entry
    and the source index; rebuild only the missing, damaged or changed ones,
    each atomically, in place. A month appended to the source converts just
    that month. When the tick configuration or schema changed, every
    partition is stale, so this becomes a ``rebuild`` instead of rewriting the
    live dataset piece by piece.
``rebuild`` (``--overwrite`` / ``--no-resume``)
    Build a complete new dataset in the building directory - continuing it if
    an earlier rebuild was interrupted - validate it, and only then swap it in.
    The previous dataset stays intact and readable until the new one is
    proven complete.
``smoke`` (``--limit-rows``)
    Converts the first N source rows into a separate directory. It never
    touches the canonical dataset, its manifest or its reports.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from ..utils.clock import utc_now_iso
from ..utils.config import Config
from ..utils.logging import format_duration, get_logger
from ..utils.paths import atomic_write_text, dir_size_bytes, ensure_dir, human_bytes
from .cleaner import Cleaner, CleaningSummary
from .inspector import resolve_sources
from .schema import (
    ASK,
    BID,
    SCHEMA_VERSION,
    TIMESTAMP,
    canonical_schema,
    derive_expressions,
    price_dtype,
    timestamp_dtype,
    volume_dtype,
)
from .source_index import (
    PartitionSource,
    SourceIndex,
    SourceIndexer,
    read_partition_bytes,
    read_runs,
)
from .timezones import describe_policy, to_utc_expr
from .validator import CHECKS, RAW_TIMESTAMP, StreamValidator

__all__ = [
    "BUILDING_SUFFIX",
    "DATASET_MANIFEST",
    "MANIFEST_NAME",
    "PARTIAL_SUFFIX",
    "PREVIOUS_SUFFIX",
    "SMOKE_SUFFIX",
    "ConversionResult",
    "InsufficientSpaceError",
    "PartitionRecord",
    "SourceChangedError",
    "TickConverter",
    "dataset_version",
    "load_dataset_manifest",
]

LOGGER = get_logger("data.converter")

#: Copy of the dataset manifest kept beside the other reports.
MANIFEST_NAME = "conversion_manifest.json"
#: The authoritative manifest, inside the dataset directory.
DATASET_MANIFEST = "_manifest.json"
VALIDATION_REPORT_NAME = "validation_report.json"
CLEANING_REPORT_NAME = "cleaning_report.json"
MANIFEST_VERSION = 2

BUILDING_SUFFIX = ".building"
PREVIOUS_SUFFIX = ".previous"
SMOKE_SUFFIX = ".smoke"
PARTIAL_SUFFIX = ".partial"

#: Output/source size ratio assumed before any partition has been written.
#: Measured on this export: 9.35 GiB of Parquet from 33.75 GiB of CSV (0.28).
DEFAULT_SIZE_RATIO = 0.30
_MAX_EXAMPLES_PER_CHECK = 3
_MAX_GAPS_PER_PARTITION = 10


class SourceChangedError(RuntimeError):
    """A partition's source bytes no longer match the source index."""


class InsufficientSpaceError(RuntimeError):
    """The planned writes would leave less free space than configured."""


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
@dataclass
class PartitionRecord:
    """One converted partition, exactly as recorded in the manifest."""

    year: int
    month: int
    path: str | None = None          # relative to the dataset directory
    status: str = "pending"          # complete | partial
    validated: bool = False
    rows: int = 0
    bytes: int = 0
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    digest: str | None = None
    source_hash: str = ""
    source_lines: int = 0
    source_bytes: int = 0
    malformed_rows: int = 0
    rows_input: int = 0
    rows_dropped: int = 0
    dropped_by_reason: dict[str, int] = field(default_factory=dict)
    flagged_by_check: dict[str, int] = field(default_factory=dict)
    flagged_kept: dict[str, int] = field(default_factory=dict)
    validation_counts: dict[str, int] = field(default_factory=dict)
    rows_flagged: int = 0
    examples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    gap_count: int = 0
    largest_gap_seconds: float = 0.0
    gaps: list[dict[str, Any]] = field(default_factory=list)
    source_backward_steps: int = 0
    tick_fingerprint: str = ""
    config_fingerprint: str = ""
    schema_version: str = SCHEMA_VERSION
    converted_utc: str = ""
    duration_seconds: float = 0.0

    @property
    def key(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"

    @property
    def complete(self) -> bool:
        return self.status == "complete" and self.validated

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "key": self.key, "complete": self.complete}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PartitionRecord:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass
class ConversionResult:
    """Everything one conversion run did."""

    mode: str = "incremental"
    status: str = "incomplete"
    dataset_dir: str = ""
    source_files: list[str] = field(default_factory=list)
    source_bytes: int = 0
    raw_fingerprint: str = ""
    rows_read: int = 0
    rows_written: int = 0
    rows_dropped: int = 0
    malformed_rows: int = 0
    partitions: list[PartitionRecord] = field(default_factory=list)
    skipped_partitions: list[str] = field(default_factory=list)
    removed_partitions: list[str] = field(default_factory=list)
    rebuild_reasons: dict[str, str] = field(default_factory=dict)
    total_partitions: int = 0
    total_rows: int = 0
    output_bytes: int = 0
    duration_seconds: float = 0.0
    input_sorted: bool = True
    output_sorted: bool = True
    schema_version: str = SCHEMA_VERSION
    config_fingerprint: str = ""
    tick_fingerprint: str = ""
    dataset_version: str = ""
    timezone: dict[str, Any] = field(default_factory=dict)
    truncated_by_limit: bool = False
    space_check: dict[str, Any] = field(default_factory=dict)
    swapped: bool = False
    generated_utc: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["partitions"] = [p.to_dict() for p in self.partitions]
        payload["output_bytes_human"] = human_bytes(self.output_bytes)
        payload["source_bytes_human"] = human_bytes(self.source_bytes)
        complete = self.status == "complete" and not self.truncated_by_limit
        payload["compression_ratio"] = (
            self.source_bytes / self.output_bytes if self.output_bytes and complete else None
        )
        payload["rows_per_second"] = (
            self.rows_read / self.duration_seconds if self.duration_seconds else None
        )
        return payload


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------
def load_dataset_manifest(directory: Path) -> dict[str, Any] | None:
    """The manifest inside a dataset directory, or None if absent or unreadable."""
    path = Path(directory) / DATASET_MANIFEST
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        LOGGER.warning("Ignoring unreadable manifest %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


def dataset_version(manifest: dict[str, Any]) -> str:
    """Content identity of a tick dataset: changes iff its data or settings do."""
    hasher = hashlib.blake2b(digest_size=8)
    for key in ("schema_version", "tick_fingerprint", "raw_fingerprint"):
        hasher.update(str(manifest.get(key, "")).encode())
    for entry in sorted(manifest.get("partitions", []), key=lambda e: e["key"]):
        hasher.update(f"{entry['key']}|{entry.get('digest')}|{entry.get('rows')}".encode())
    return f"ticks-{hasher.hexdigest()}"


def _sibling(path: Path, suffix: str) -> Path:
    return path.with_name(path.name + suffix)


class _PartitionDigest:
    """Content digest: per column, one hasher for values and one for the null mask.

    Every stream is fed only its own bytes, in row order, so the result is a
    function of the rows alone - not of how they were chunked into row groups.
    """

    def __init__(self) -> None:
        self._streams: dict[str, tuple[Any, Any]] = {}

    def update(self, frame: pl.DataFrame) -> None:
        for name in frame.columns:
            series = frame[name]
            values = np.ascontiguousarray(series.to_numpy())
            if values.dtype == object:
                raise TypeError(
                    f"Column {name!r} has object dtype; it cannot be digested "
                    "deterministically. Disable conversion.digest or cast it."
                )
            pair = self._streams.get(name)
            if pair is None:
                pair = self._streams[name] = (
                    hashlib.blake2b(digest_size=16),
                    hashlib.blake2b(digest_size=16),
                )
            pair[0].update(values.tobytes())
            pair[1].update(np.ascontiguousarray(series.is_null().to_numpy()).tobytes())

    def hexdigest(self, column_order: list[str]) -> str | None:
        if not self._streams:
            return None
        combined = hashlib.blake2b(digest_size=16)
        for name in column_order:
            pair = self._streams.get(name)
            if pair is None:
                continue
            combined.update(name.encode())
            for hasher in pair:
                combined.update(hasher.digest())
        return combined.hexdigest()


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------
class TickConverter:
    """Converts raw tick files into the canonical partitioned Parquet dataset."""

    def __init__(
        self,
        config: Config,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """*progress_callback* receives ``(rows_read_total, rows_in_this_partition)``."""
        self.config = config
        self._progress = progress_callback
        self._stream_validator = StreamValidator(config)
        self._with_utc = config.timezone.emit_utc_column and config.timezone.mode != "naive"
        self._pl_schema = canonical_schema(config.input, config.canonical, with_utc=self._with_utc)
        self._arrow_schema = self._build_arrow_schema()
        self._indexer = SourceIndexer(config, block_bytes=config.conversion.index_block_bytes)
        self._last_validation: dict[str, Any] = {}
        self._last_cleaning: dict[str, Any] = {}
        self._partition_malformed = 0
        if not config.cleaning.drop_unparseable_timestamp:
            raise ValueError(
                "cleaning.drop_unparseable_timestamp is false, but a row without a "
                "usable timestamp cannot be placed in a time partition. Set it to true."
            )

    # -- public API --------------------------------------------------------
    @property
    def validation_report(self) -> dict[str, Any]:
        """Validation counts of the dataset written by the last :meth:`run`."""
        return self._last_validation

    @property
    def cleaning_summary(self) -> dict[str, Any]:
        """Cleaning audit of the dataset written by the last :meth:`run`."""
        return self._last_cleaning

    @property
    def malformed_rows(self) -> int:
        """Rows the CSV parser rejected during :meth:`iter_canonical_batches`.

        Malformed rows never reach a batch, so a caller driving that iterator
        with its own validator must fold this count in explicitly.
        """
        return self._stream_validator.report().malformed_rows

    def run(
        self,
        *,
        limit_rows: int | None = None,
        resume: bool | None = None,
        overwrite: bool = False,
        output_dir: Path | None = None,
        keep_previous: bool | None = None,
        rebuild_index: bool = False,
        check_space: bool = True,
    ) -> ConversionResult:
        """Bring the tick dataset up to date with the source.

        Parameters
        ----------
        limit_rows:
            Smoke test: convert only the first N source rows, into a separate
            directory (``--out`` or ``<processed>.smoke``). The canonical
            dataset is never touched.
        resume:
            Overrides ``conversion.resume``. ``False`` means a full rebuild.
        overwrite:
            Full transactional rebuild: the new dataset is built beside the
            old one and swapped in only once it is complete and validated.
        output_dir:
            Overrides ``processed_data_path`` for this run.
        keep_previous:
            After a swap, keep the replaced dataset as a timestamped backup
            instead of deleting it. Overrides ``conversion.keep_previous``.
        rebuild_index:
            Re-scan the source even if the cached index looks current.
        check_space:
            Refuse to start when the estimated output would leave less than
            ``conversion.min_free_gb`` free.
        """
        started = time.perf_counter()
        sources = resolve_sources(self.config.raw_data_path)
        if not sources:
            raise FileNotFoundError(
                f"No raw data found at {self.config.raw_data_path}. "
                "Set `raw_data_path` in config/data.yaml or pass --raw-path."
            )
        self._check_headers(sources)
        self._indexer.resolve_columns(self.source_column_names(sources[0]))
        index = self._indexer.load_or_build(sources, rebuild=rebuild_index)

        canonical = Path(output_dir) if output_dir else self.config.processed_data_path
        keep = self.config.conversion.keep_previous if keep_previous is None else keep_previous
        resume = self.config.conversion.resume if resume is None else resume

        result = ConversionResult(
            source_files=[str(s) for s in sources],
            source_bytes=sum(s.stat().st_size for s in sources),
            raw_fingerprint=index.raw_fingerprint,
            config_fingerprint=self.config.fingerprint(),
            tick_fingerprint=self.config.tick_fingerprint(),
            timezone=describe_policy(self.config.timezone).to_dict(),
            input_sorted=bool(index.ordering.get("input_sorted", True)),
        )

        if limit_rows is not None:
            if output_dir is not None and _same_path(Path(output_dir),
                                                     self.config.processed_data_path):
                raise ValueError(
                    "--limit-rows never writes to the canonical dataset. Pass --out "
                    "pointing somewhere else, or omit it to use "
                    f"{_sibling(self.config.processed_data_path, SMOKE_SUFFIX)}."
                )
            target = Path(output_dir) if output_dir else _sibling(canonical, SMOKE_SUFFIX)
            result.mode = "smoke"
            _remove_pipeline_output(target)
            manifest = self._new_manifest(index, status="partial")
        else:
            self._recover_interrupted_swap(canonical)
            result.mode, target, manifest = self._choose_mode(
                canonical, index, resume=resume, overwrite=overwrite
            )
        ensure_dir(target)
        result.dataset_dir = str(target)
        LOGGER.info(
            "Converting %d source file(s), %s -> %s (mode: %s)",
            len(sources), human_bytes(result.source_bytes), target, result.mode,
        )
        LOGGER.info("Timezone policy: %s", self.config.timezone.describe())
        if not result.input_sorted:
            LOGGER.info(
                "Source is not time-sorted (%s backward jumps, %s late rows, %s in an "
                "earlier partition). Each partition is sorted as a whole, so this is "
                "handled; see source_ordering_report.json for the diagnosis.",
                f"{index.ordering.get('backward_jumps', 0):,}",
                f"{index.ordering.get('late_lines', 0):,}",
                f"{index.ordering.get('late_lines_in_earlier_partition', 0):,}",
            )

        entries: dict[str, dict[str, Any]] = {
            e["key"]: e for e in manifest.get("partitions", [])
        }
        plan = self._plan(index, entries, target, result, limit_rows)
        if check_space and plan:
            result.space_check = self._check_disk_space(
                target, [index.partitions[k] for k in plan], entries
            )

        remaining = limit_rows
        try:
            for key in plan:
                part = index.partitions[key]
                budget = None
                if remaining is not None:
                    if remaining <= 0:
                        break
                    budget = min(remaining, part.lines)
                record = self._convert_partition(key, index, sources, target, max_lines=budget)
                entries[key] = record.to_dict()
                result.partitions.append(record)
                result.rows_read += record.rows_input
                result.rows_written += record.rows
                result.rows_dropped += record.rows_dropped
                result.malformed_rows += record.malformed_rows
                if remaining is not None:
                    remaining -= record.source_lines
                self._write_manifest(target, manifest, entries, index, final=False)
                if self._progress is not None:
                    self._progress(result.rows_read, record.rows_input)
        except BaseException:
            # Everything finished so far is durable and recorded; say so.
            snapshot = self._write_manifest(target, manifest, entries, index, final=False)
            if result.mode == "incremental":
                # The live dataset changed, so its reports must not keep describing
                # the old state. A rebuild leaves the live dataset and reports alone.
                try:
                    self._write_reports(self.config.metadata_path, snapshot, index)
                except Exception:  # noqa: BLE001 - never mask the original error
                    LOGGER.exception("Could not refresh reports after the interruption")
            LOGGER.error(
                "Conversion interrupted after %d partition(s) this run. Completed "
                "partitions are recorded in %s; re-run the same command to continue.",
                len(result.partitions), target / DATASET_MANIFEST,
            )
            raise

        if result.mode != "smoke":
            result.removed_partitions = self._remove_orphans(target, entries, index)
            previous = manifest.get("unassigned")
            unassigned = self._process_unassigned(index, sources, previous)
            manifest["unassigned"] = unassigned
            if unassigned is not previous and unassigned.get("source_lines"):
                result.rows_read += int(unassigned.get("rows_input") or 0)
                result.rows_dropped += int(unassigned.get("rows_dropped") or 0)
                result.malformed_rows += int(unassigned.get("malformed_rows") or 0)
        if limit_rows is not None:
            result.truncated_by_limit = limit_rows < index.assigned_lines
        manifest = self._write_manifest(
            target, manifest, entries, index, final=True, smoke=result.mode == "smoke",
            truncated=result.truncated_by_limit,
        )
        result.status = manifest["status"]
        result.dataset_version = manifest.get("dataset_version", "")
        result.total_partitions = manifest["partition_count"]
        result.total_rows = manifest["total_rows"]
        result.output_sorted = all(
            e.get("validated") for e in entries.values() if e.get("rows")
        )

        if result.mode == "rebuild":
            if manifest["status"] != "complete":
                raise RuntimeError(
                    f"Rebuild finished but the new dataset is {manifest['status']!r}; "
                    f"the current dataset at {canonical} was left untouched. "
                    f"Inspect {target / DATASET_MANIFEST}."
                )
            self._swap_in(target, canonical, keep_previous=keep)
            result.swapped = True
            result.dataset_dir = str(canonical)
            target = canonical

        reports_dir = target if result.mode == "smoke" else self.config.metadata_path
        self._write_reports(reports_dir, manifest, index)
        if result.mode != "smoke":
            ensure_dir(self.config.metadata_path)
            atomic_write_text(
                self.config.metadata_path / MANIFEST_NAME,
                json.dumps(manifest, indent=2, default=str) + "\n",
            )
        result.output_bytes = dir_size_bytes(target, "**/*.parquet")
        result.duration_seconds = time.perf_counter() - started
        result.partitions.sort(key=lambda p: (p.year, p.month))
        self._log_result(result)
        return result

    # -- source reading for `xq validate` ------------------------------------
    def iter_canonical_batches(
        self, source: Path, *, start_offset: int | None = None
    ) -> Iterator[pl.DataFrame]:
        """Yield canonical batches from *source*, in source order, writing nothing.

        This is the parse/derive half of a conversion, exposed so that the
        standalone ``xq validate`` pass shares exactly the same parsing.
        """
        data_start = self.data_start_offset(source)
        offset = data_start if start_offset is None else start_offset
        for batch in self._iter_source_batches(source, offset):
            yield self._to_canonical(batch)

    def source_column_names(self, source: Path) -> list[str]:
        """Full ordered column list for *source*.

        Taken from the header row when there is one. Headerless files get
        positional ``column_0``, ``column_1``... names, which is also what
        :mod:`xauusd_quant.data.inspector` reports, so the two agree.
        """
        inp = self.config.input
        with source.open("rb") as handle:
            first = handle.readline().decode(inp.encoding, errors="replace")
        if not first.strip():
            raise ValueError(f"{source} is empty or has no readable first line.")
        cells = [
            cell.strip().lstrip("﻿")
            for cell in first.rstrip("\r\n").split(inp.delimiter)
        ]
        if not inp.has_header:
            return [f"column_{i}" for i in range(len(cells))]

        missing = [c for c in inp.source_columns.values() if c not in cells]
        if missing:
            raise KeyError(
                f"Configured column(s) {missing} are not in the header of {source.name} "
                f"({cells}). Run `xq detect-schema --save` or fix config/data.yaml."
            )
        return cells

    def data_start_offset(self, source: Path) -> int:
        """Byte offset of the first data row, skipping a header if present."""
        if not self.config.input.has_header:
            return 0
        with source.open("rb") as handle:
            handle.readline()
            return handle.tell()

    # -- mode selection ------------------------------------------------------
    def _choose_mode(
        self,
        canonical: Path,
        index: SourceIndex,
        *,
        resume: bool,
        overwrite: bool,
    ) -> tuple[str, Path, dict[str, Any]]:
        building = _sibling(canonical, BUILDING_SUFFIX)
        current = load_dataset_manifest(canonical)
        in_progress = load_dataset_manifest(building) if building.exists() else None

        if in_progress is not None and self._compatible(in_progress):
            LOGGER.info("Continuing an interrupted rebuild in %s", building)
            return "rebuild", building, in_progress
        if building.exists():
            LOGGER.warning(
                "Discarding an incompatible or unreadable rebuild directory %s "
                "(generated data only)", building,
            )
            _remove_pipeline_output(building)

        if overwrite or not resume:
            return "rebuild", building, self._new_manifest(index, status="building")

        if current is None:
            if _has_pipeline_files(canonical):
                LOGGER.warning(
                    "%s holds Parquet files but no verifiable manifest; building a new "
                    "dataset beside it and swapping it in when complete.", canonical,
                )
                return "rebuild", building, self._new_manifest(index, "building")
            return "incremental", canonical, self._new_manifest(index, "incomplete")

        if not self._compatible(current):
            LOGGER.warning(
                "The current dataset was written under a different tick configuration, "
                "schema or manifest version, so every partition is stale. Rebuilding "
                "beside it; it stays readable until the new dataset is complete."
            )
            return "rebuild", building, self._new_manifest(index, "building")
        return "incremental", canonical, current

    def _compatible(self, manifest: dict[str, Any]) -> bool:
        return (
            manifest.get("manifest_version") == MANIFEST_VERSION
            and manifest.get("tick_fingerprint") == self.config.tick_fingerprint()
            and manifest.get("schema_version") == SCHEMA_VERSION
        )

    def _new_manifest(self, index: SourceIndex, status: str) -> dict[str, Any]:
        return {
            "manifest_version": MANIFEST_VERSION,
            "status": status,
            "schema_version": SCHEMA_VERSION,
            "tick_fingerprint": self.config.tick_fingerprint(),
            "config_fingerprint": self.config.fingerprint(),
            "instrument": self.config.instrument,
            "raw_fingerprint": index.raw_fingerprint,
            "source_signature": index.signature(),
            "partitions": [],
        }

    def _plan(
        self,
        index: SourceIndex,
        entries: dict[str, dict[str, Any]],
        target: Path,
        result: ConversionResult,
        limit_rows: int | None,
    ) -> list[str]:
        """Partitions to (re)build, in chronological order; the rest are verified."""
        plan: list[str] = []
        for key in index.keys:
            entry = entries.get(key)
            if limit_rows is None and entry is not None:
                ok, why = self._verify_existing(entry, target, index.partitions[key])
                if ok:
                    result.skipped_partitions.append(key)
                    continue
                result.rebuild_reasons[key] = why
            else:
                result.rebuild_reasons[key] = "not converted yet"
            plan.append(key)
        if result.skipped_partitions:
            LOGGER.info(
                "%d partition(s) verified and reused; %d to build",
                len(result.skipped_partitions), len(plan),
            )
        for key in plan:
            reason = result.rebuild_reasons.get(key, "")
            if reason and reason != "not converted yet":
                LOGGER.info("Partition %s will be rebuilt: %s", key, reason)
        return plan

    def _verify_existing(
        self, entry: dict[str, Any], dataset_dir: Path, part: PartitionSource
    ) -> tuple[bool, str]:
        """Is this recorded partition still exactly what the source would give?"""
        if entry.get("status") != "complete" or not entry.get("validated"):
            return False, "not recorded as complete and validated"
        if entry.get("tick_fingerprint") != self.config.tick_fingerprint():
            return False, "tick configuration changed"
        if entry.get("schema_version") != SCHEMA_VERSION:
            return False, "schema version changed"
        if entry.get("source_hash") != part.content_hash or entry.get("source_lines") != part.lines:
            return False, "its source lines changed"
        rows = int(entry.get("rows") or 0)
        relative = entry.get("path")
        if rows == 0:
            return True, "verified (no rows)"
        if not relative:
            return False, "no file recorded"
        path = dataset_dir / relative
        if not path.exists():
            return False, "file missing"
        if path.stat().st_size != entry.get("bytes"):
            return False, "file size changed"
        try:
            meta = pq.read_metadata(path)
            schema = pq.read_schema(path)
        except Exception as exc:  # noqa: BLE001 - any unreadable footer means rebuild
            return False, f"unreadable Parquet footer ({type(exc).__name__})"
        if meta.num_rows != rows:
            return False, f"row count {meta.num_rows} != recorded {rows}"
        if not schema.remove_metadata().equals(self._arrow_schema.remove_metadata()):
            return False, "schema differs"
        return True, "verified"

    # -- one partition ---------------------------------------------------------
    def _convert_partition(
        self,
        key: str,
        index: SourceIndex,
        sources: list[Path],
        dataset_dir: Path,
        *,
        max_lines: int | None = None,
    ) -> PartitionRecord:
        started = time.perf_counter()
        part = index.partitions[key]
        record = PartitionRecord(
            year=part.year, month=part.month, source_hash=part.content_hash,
            source_lines=part.lines, source_bytes=part.source_bytes,
            tick_fingerprint=self.config.tick_fingerprint(),
            config_fingerprint=self.config.fingerprint(),
        )

        data, content_hash = read_partition_bytes(index, key, sources)
        if content_hash != part.content_hash:
            raise SourceChangedError(
                f"Partition {key}: its source bytes no longer match the source index "
                "(the raw file changed after indexing). Re-run with --rebuild-index."
            )
        lines = part.lines
        if max_lines is not None and max_lines < lines:
            data = _first_lines(data, max_lines)
            lines = max_lines
            record.status = "partial"
            record.source_lines = lines
            record.source_hash = ""

        names = self.source_column_names(sources[part.runs[0][0]])
        clean, audit = self._parse_validate_clean(data, lines, names, label=key)
        del data
        for name, value in audit.items():
            setattr(record, name, value)
        record.source_backward_steps = int(record.validation_counts.get("non_monotonic", 0))

        if clean.height:
            if clean[TIMESTAMP].null_count():  # pragma: no cover - guarded in __init__
                raise ValueError(f"Partition {key}: rows without a timestamp survived cleaning")
            foreign = clean.filter(
                (pl.col(TIMESTAMP).dt.year() != part.year)
                | (pl.col(TIMESTAMP).dt.month() != part.month)
            ).height
            if foreign:
                raise RuntimeError(  # pragma: no cover - index/parser disagreement
                    f"Partition {key}: {foreign} row(s) parse to a different month than "
                    "their raw text; the source index and the parser disagree."
                )

        # A stable sort: ties keep source order, so the first copy of a repeat
        # is the one kept and the result does not depend on how it was read.
        ordered = clean.sort(TIMESTAMP, maintain_order=True)
        del clean
        if not ordered[TIMESTAMP].is_sorted():  # pragma: no cover - sort guarantees it
            raise RuntimeError(f"Partition {key}: output is not sorted after sorting")

        record.gap_count, record.largest_gap_seconds, record.gaps = _gaps(
            ordered, self.config.validation.large_gap_seconds
        )
        record.rows = ordered.height
        if ordered.height:
            record.first_timestamp = ordered[TIMESTAMP][0].isoformat(sep=" ")
            record.last_timestamp = ordered[TIMESTAMP][-1].isoformat(sep=" ")
            record.path = self._relative_path(part.year, part.month)
            record.bytes, record.digest = self._write_verified(ordered, dataset_dir / record.path)
        else:
            stale = dataset_dir / self._relative_path(part.year, part.month)
            stale.unlink(missing_ok=True)
        if record.status != "partial":
            record.status = "complete"
        record.validated = True
        record.converted_utc = utc_now_iso()
        record.duration_seconds = time.perf_counter() - started
        LOGGER.info(
            "Partition %s %s: %s rows (%s source, %s dropped%s), %s, %s",
            key, record.status, f"{record.rows:,}", f"{record.source_lines:,}",
            f"{record.rows_dropped:,}",
            f", {record.source_backward_steps:,} out-of-order sorted into place"
            if record.source_backward_steps else "",
            human_bytes(record.bytes), format_duration(record.duration_seconds),
        )
        return record

    def _parse_validate_clean(
        self, data: bytes, lines: int, names: list[str], *, label: str
    ) -> tuple[pl.DataFrame, dict[str, Any]]:
        """Parse raw lines, validate them in source order, apply the drop policy.

        Returns the surviving rows (still in source order) and the audit trail
        for the record: malformed count, per-check validation counts, a few
        examples, and the cleaner's per-reason counts.
        """
        self._partition_malformed = 0
        # Parse, validate and clean block by block, in source order: only one
        # block's raw strings and flag columns exist at a time, and each block
        # is reduced to its surviving canonical rows immediately. The validator
        # is scoped to this month with a repeat window covering every line of
        # it, so duplicate detection is still exact across the whole month.
        validator = StreamValidator(self.config, duplicate_window=max(lines, 1))
        cleaner = Cleaner(self.config.cleaning)
        kept: list[pl.DataFrame] = []
        parsed = 0
        for batch in self._iter_bytes(data, names):
            parsed += batch.height
            flagged = validator.validate(self._to_canonical(batch))
            kept.append(cleaner.clean(flagged).drop([RAW_TIMESTAMP], strict=False))
            del batch, flagged
        malformed = self._partition_malformed
        if parsed + malformed != lines:
            raise RuntimeError(  # pragma: no cover - would mean the parser lost rows
                f"{label}: parsed {parsed:,} + malformed {malformed:,} != "
                f"{lines:,} indexed lines"
            )
        validator.note_malformed(malformed)
        if kept:
            clean = pl.concat(kept, how="vertical", rechunk=True)
        else:
            clean = pl.DataFrame(schema=self._pl_schema)
        kept.clear()

        report = validator.report()
        summary = cleaner.summary
        audit = {
            "malformed_rows": malformed,
            "validation_counts": {k: int(v) for k, v in report.counts.items() if v},
            "rows_flagged": int(report.rows_flagged),
            "examples": {
                check: rows[:_MAX_EXAMPLES_PER_CHECK]
                for check, rows in report.examples.items() if rows
            },
            "rows_input": summary.rows_input,
            "rows_dropped": summary.rows_dropped,
            "dropped_by_reason": {k: v for k, v in summary.dropped_by_reason.items() if v},
            "flagged_by_check": {k: v for k, v in summary.flagged_by_check.items() if v},
            "flagged_kept": {k: v for k, v in summary.flagged_kept.items() if v},
        }
        return clean, audit

    def _process_unassigned(
        self, index: SourceIndex, sources: list[Path], previous: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Account for source lines that carry no readable year-month.

        They cannot belong to any partition, but they are still parsed,
        validated and cleaned exactly like partition rows, so each one is
        attributed to a precise reason (malformed, missing or unparseable
        timestamp) instead of silently vanishing.
        """
        base = {
            "source_lines": index.unassigned_lines,
            "source_hash": index.unassigned_hash,
            "tick_fingerprint": self.config.tick_fingerprint(),
        }
        if not index.unassigned_lines:
            return {**base, "malformed_rows": 0, "rows_input": 0, "rows_dropped": 0,
                    "rows_flagged": 0, "validation_counts": {}, "dropped_by_reason": {},
                    "flagged_by_check": {}, "flagged_kept": {}, "examples": {}}
        if previous and all(previous.get(k) == v for k, v in base.items()):
            return previous
        data, content_hash = read_runs(index.unassigned_runs, sources)
        if content_hash != index.unassigned_hash:
            raise SourceChangedError(
                "Unassigned source lines changed after indexing. Re-run with --rebuild-index."
            )
        names = self.source_column_names(sources[index.unassigned_runs[0][0]])
        clean, audit = self._parse_validate_clean(
            data, index.unassigned_lines, names, label="unassigned lines"
        )
        if clean.height:
            raise RuntimeError(  # pragma: no cover - index/parser disagreement
                f"{clean.height} line(s) without a readable year-month nevertheless "
                "parsed to a valid timestamp; the source index and the parser disagree."
            )
        LOGGER.warning(
            "%s source line(s) have no readable year-month; all were dropped and "
            "attributed: %s (malformed: %s)",
            f"{index.unassigned_lines:,}", audit["dropped_by_reason"],
            f"{audit['malformed_rows']:,}",
        )
        return {**base, **audit}

    def _iter_bytes(self, data: bytes, names: list[str]) -> Iterator[pl.DataFrame]:
        """Stream one partition's raw lines through the CSV parser, block by block."""
        if not data:
            return
        wanted = list(self.config.input.source_columns.values())
        reader = pacsv.open_csv(
            pa.BufferReader(pa.py_buffer(data)),
            read_options=pacsv.ReadOptions(
                block_size=self.config.conversion.read_block_bytes,
                use_threads=True,
                column_names=names,
                skip_rows=0,
            ),
            parse_options=pacsv.ParseOptions(
                delimiter=self.config.input.delimiter,
                quote_char=self.config.input.quote_char or False,
                newlines_in_values=False,
                invalid_row_handler=self._on_partition_invalid_row,
            ),
            convert_options=pacsv.ConvertOptions(
                column_types=self._source_arrow_types(),
                strings_can_be_null=True,
                include_columns=wanted,
            ),
        )
        for record_batch in reader:
            if record_batch.num_rows:
                frame = pl.from_arrow(record_batch)
                assert isinstance(frame, pl.DataFrame)
                yield frame

    def _on_partition_invalid_row(self, _row: Any) -> str:
        self._partition_malformed += 1
        return "skip"

    def _relative_path(self, year: int, month: int) -> str:
        parts = self.config.parquet.partitioning
        pieces: list[str] = []
        if "year" in parts:
            pieces.append(f"year={year:04d}")
        if "month" in parts:
            pieces.append(f"month={month:02d}")
        pieces.append(
            self.config.parquet.filename_template.format(year=f"{year:04d}", month=f"{month:02d}")
        )
        return "/".join(pieces)

    def _write_verified(self, frame: pl.DataFrame, path: Path) -> tuple[int, str | None]:
        """Write to ``*.partial``, re-open and check it, then rename into place."""
        pq_cfg = self.config.parquet
        tmp = path.with_name(path.name + PARTIAL_SUFFIX)
        ensure_dir(path.parent)
        tmp.unlink(missing_ok=True)
        # Polars' writer compresses column chunks in parallel; on this data it
        # is ~7x faster than pyarrow's single-threaded writer at the same zstd
        # level, with identical content and schema. Passing the Arrow schema
        # embeds the provenance metadata where pyarrow's read_schema finds it.
        # Encodings (e.g. dictionary pages) are the writer's choice: they
        # change bytes, never content, which the digest verifies.
        compression = "uncompressed" if pq_cfg.compression == "none" else pq_cfg.compression
        leveled = compression in ("zstd", "gzip", "brotli")
        frame.write_parquet(
            tmp,
            compression=compression,  # type: ignore[arg-type]
            compression_level=pq_cfg.compression_level if leveled else None,
            statistics=pq_cfg.write_statistics,
            row_group_size=pq_cfg.row_group_rows,
            arrow_schema=self._arrow_schema,
        )

        digest: str | None = None
        if self.config.conversion.digest:
            hasher = _PartitionDigest()
            step = pq_cfg.row_group_rows
            for offset in range(0, frame.height, step):
                hasher.update(frame.slice(offset, step))
            digest = hasher.hexdigest(self._arrow_schema.names)

        meta = pq.read_metadata(tmp)
        schema = pq.read_schema(tmp)
        problems: list[str] = []
        if meta.num_rows != frame.height:
            problems.append(f"row count {meta.num_rows} != {frame.height}")
        if not schema.remove_metadata().equals(self._arrow_schema.remove_metadata()):
            problems.append("schema differs from the canonical schema")
        if problems:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"{path.name}: verification failed: {'; '.join(problems)}")
        tmp.replace(path)
        return path.stat().st_size, digest

    # -- manifest ------------------------------------------------------------
    def _write_manifest(
        self,
        target: Path,
        manifest: dict[str, Any],
        entries: dict[str, dict[str, Any]],
        index: SourceIndex,
        *,
        final: bool,
        smoke: bool = False,
        truncated: bool = False,
    ) -> dict[str, Any]:
        """Persist the manifest inside *target*. Called after every partition."""
        ordered = [entries[k] for k in sorted(entries)]
        complete_keys = {e["key"] for e in ordered if e.get("complete")}
        missing = [k for k in index.keys if k not in complete_keys]
        unassigned = manifest.get("unassigned") or {}
        # Every source line must be accounted for exactly once: in a partition
        # or among the unassigned lines, and then as written, dropped or malformed.
        accounted = sum(int(e.get("source_lines") or 0) for e in ordered) + int(
            unassigned.get("source_lines") or 0
        )
        consumed = sum(
            int(e.get("rows_input") or 0) + int(e.get("malformed_rows") or 0)
            for e in [*ordered, unassigned]
        )
        unassigned_ok = unassigned.get("source_hash") == index.unassigned_hash
        status = manifest.get("status", "incomplete")
        if smoke:
            status = "partial"
        elif final:
            status = (
                "complete"
                if not missing and unassigned_ok and accounted == index.total_lines
                and consumed == accounted
                else "incomplete"
            )
        elif status == "complete":
            status = "incomplete"
        payload = {
            **{k: v for k, v in manifest.items() if k != "partitions"},
            "manifest_version": MANIFEST_VERSION,
            "generated_utc": utc_now_iso(),
            "status": status,
            "schema_version": SCHEMA_VERSION,
            "tick_fingerprint": self.config.tick_fingerprint(),
            "config_fingerprint": self.config.fingerprint(),
            "instrument": self.config.instrument,
            "raw_fingerprint": index.raw_fingerprint,
            "source_signature": index.signature(),
            "source_files": [f.path for f in index.files],
            "source_index": {
                "created_utc": index.created_utc,
                "total_lines": index.total_lines,
                "assigned_lines": index.assigned_lines,
                "unassigned_lines": index.unassigned_lines,
                "partition_count": len(index.partitions),
            },
            "input_sorted": bool(index.ordering.get("input_sorted", True)),
            "source_ordering": {
                k: v for k, v in index.ordering.items() if k not in ("events", "definitions")
            },
            "output_sorted": all(e.get("validated") for e in ordered if e.get("rows")),
            "truncated_by_limit": truncated,
            "timezone": describe_policy(self.config.timezone).to_dict(),
            "missing_partitions": missing,
            "source_lines_accounted": accounted,
            "partition_count": len(ordered),
            "total_rows": sum(int(e.get("rows") or 0) for e in ordered),
            "total_bytes": sum(int(e.get("bytes") or 0) for e in ordered),
            "rows_dropped": sum(int(e.get("rows_dropped") or 0) for e in ordered),
            "first_timestamp": next(
                (e["first_timestamp"] for e in ordered if e.get("first_timestamp")), None
            ),
            "last_timestamp": next(
                (e["last_timestamp"] for e in reversed(ordered) if e.get("last_timestamp")),
                None,
            ),
            "partitions": ordered,
        }
        payload["dataset_version"] = (
            dataset_version(payload) if status == "complete" else None
        )
        manifest.clear()
        manifest.update(payload)
        ensure_dir(target)
        atomic_write_text(
            target / DATASET_MANIFEST, json.dumps(payload, indent=1, default=str) + "\n"
        )
        return payload

    def _remove_orphans(
        self, target: Path, entries: dict[str, dict[str, Any]], index: SourceIndex
    ) -> list[str]:
        """Drop recorded partitions whose month no longer exists in the source."""
        removed: list[str] = []
        for key in sorted(set(entries) - set(index.partitions)):
            entry = entries.pop(key)
            relative = entry.get("path")
            if relative:
                path = target / relative
                LOGGER.warning(
                    "Partition %s is no longer in the source; removing %s (%s)",
                    key, path, human_bytes(path.stat().st_size) if path.exists() else "missing",
                )
                path.unlink(missing_ok=True)
            removed.append(key)
        for stray in target.rglob(f"*{PARTIAL_SUFFIX}"):
            LOGGER.warning("Removing an unfinished partial file %s", stray)
            stray.unlink(missing_ok=True)
        _remove_empty_dirs(target)
        return removed

    # -- transactional swap ----------------------------------------------------
    def _swap_in(self, building: Path, canonical: Path, *, keep_previous: bool) -> None:
        """Replace *canonical* with the validated *building* directory.

        Two directory renames: the live dataset moves aside, the new one takes
        its name. Should the process stop between them, the next run finds
        ``.previous`` without a live dataset and completes or rolls back the
        swap (:meth:`_recover_interrupted_swap`).
        """
        previous = _sibling(canonical, PREVIOUS_SUFFIX)
        if previous.exists():
            raise RuntimeError(
                f"{previous} already exists; an earlier swap was not cleaned up. "
                "Re-run `xq convert` to let it recover, or inspect it by hand."
            )
        had_current = canonical.exists()
        if had_current:
            canonical.replace(previous)
        try:
            building.replace(canonical)
        except OSError:
            if had_current:
                previous.replace(canonical)
            raise
        LOGGER.info("New dataset swapped into %s", canonical)
        if had_current:
            self._dispose_previous(previous, canonical, keep=keep_previous)

    def _dispose_previous(self, previous: Path, canonical: Path, *, keep: bool) -> None:
        # Files the pipeline did not create (.gitkeep, notes) move to the new dataset.
        for item in previous.iterdir():
            if not _is_pipeline_path(item) and not (canonical / item.name).exists():
                shutil.move(str(item), str(canonical / item.name))
        if keep:
            backup = _sibling(canonical, ".backup-" + utc_now_iso().replace(":", "").replace("-", ""))
            previous.replace(backup)
            LOGGER.info("Previous dataset kept as %s", backup)
            return
        size = dir_size_bytes(previous, "**/*")
        _remove_pipeline_output(previous)
        if previous.exists() and not any(previous.iterdir()):
            previous.rmdir()
        if previous.exists():
            LOGGER.warning(
                "%s still holds files the pipeline did not create; left in place.", previous
            )
        else:
            LOGGER.info("Previous dataset removed (%s reclaimed)", human_bytes(size))

    def _recover_interrupted_swap(self, canonical: Path) -> None:
        previous = _sibling(canonical, PREVIOUS_SUFFIX)
        if not previous.exists():
            return
        building = _sibling(canonical, BUILDING_SUFFIX)
        if not canonical.exists():
            ready = load_dataset_manifest(building) if building.exists() else None
            if ready is not None and ready.get("status") == "complete":
                LOGGER.warning("Completing an interrupted swap: %s -> %s", building, canonical)
                building.replace(canonical)
            else:
                LOGGER.warning("Rolling back an interrupted swap: %s -> %s", previous, canonical)
                previous.replace(canonical)
                return
        current = load_dataset_manifest(canonical)
        if current is not None and current.get("status") == "complete":
            LOGGER.warning("Finishing the clean-up of %s from an interrupted swap", previous)
            self._dispose_previous(previous, canonical, keep=False)
        else:
            LOGGER.warning(
                "Both %s and %s exist and the live one is not a complete dataset; "
                "leaving both for inspection.", canonical, previous,
            )

    # -- disk space ------------------------------------------------------------
    def _check_disk_space(
        self, target: Path, parts: list[PartitionSource], entries: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        source_total = sum(p.source_bytes for p in parts)
        written = [e for e in entries.values() if e.get("complete") and e.get("source_bytes")]
        ratio = (
            sum(int(e.get("bytes") or 0) for e in written)
            / sum(int(e["source_bytes"]) for e in written)
            if written else DEFAULT_SIZE_RATIO
        )
        needed = int(source_total * max(ratio, 0.05) * 1.10)
        probe = target
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        free = shutil.disk_usage(probe).free
        reserve = int(self.config.conversion.min_free_gb * (1 << 30))
        check = {
            "partitions_planned": len(parts),
            "source_bytes_planned": source_total,
            "size_ratio_assumed": ratio,
            "estimated_output_bytes": needed,
            "free_bytes": free,
            "reserve_bytes": reserve,
            "free_after_estimate": free - needed,
        }
        LOGGER.info(
            "Space check: ~%s to write, %s free, %s reserve",
            human_bytes(needed), human_bytes(free), human_bytes(reserve),
        )
        if free - needed < reserve:
            raise InsufficientSpaceError(
                f"Converting {len(parts)} partition(s) needs about {human_bytes(needed)} "
                f"(at {ratio:.2f}x the {human_bytes(source_total)} of source), which would "
                f"leave {human_bytes(max(free - needed, 0))} free on {probe.anchor} - below "
                f"the {human_bytes(reserve)} reserve (conversion.min_free_gb). Free "
                f"{human_bytes(reserve - (free - needed))} more, lower the reserve, or "
                "point processed_data_path at a larger volume."
            )
        return check

    # -- reports -----------------------------------------------------------------
    def _write_reports(
        self, directory: Path, manifest: dict[str, Any], index: SourceIndex
    ) -> None:
        """Dataset-wide validation and cleaning reports, aggregated per partition."""
        entries = list(manifest.get("partitions", []))
        if manifest.get("unassigned"):
            entries.append(manifest["unassigned"])
        counts = dict.fromkeys(CHECKS, 0)
        examples: dict[str, list[dict[str, Any]]] = {}
        gaps: list[dict[str, Any]] = []
        limit = self.config.validation.max_reported_examples
        for entry in entries:
            for check, value in (entry.get("validation_counts") or {}).items():
                counts[check] = counts.get(check, 0) + int(value)
            for check, rows in (entry.get("examples") or {}).items():
                bucket = examples.setdefault(check, [])
                bucket.extend(rows[: max(limit - len(bucket), 0)])
            gaps.extend(entry.get("gaps") or [])
        gaps.sort(key=lambda g: g["seconds"], reverse=True)
        complete = manifest.get("status") == "complete"
        validation = {
            "generated_utc": utc_now_iso(),
            "scope": "every partition of the dataset, validated in source order",
            "complete_pass": complete,
            "rows_checked": sum(int(e.get("rows_input") or 0) for e in entries),
            "rows_flagged": sum(int(e.get("rows_flagged") or 0) for e in entries),
            "malformed_rows": sum(int(e.get("malformed_rows") or 0) for e in entries),
            "unassigned_source_lines": index.unassigned_lines,
            "unassigned_examples": index.unassigned_examples[:limit],
            "counts": counts,
            "examples": examples,
            "gap_count": sum(int(e.get("gap_count") or 0) for e in entries),
            "largest_gap_seconds": max(
                (float(e.get("largest_gap_seconds") or 0.0) for e in entries), default=0.0
            ),
            "gaps": gaps[: self.config.validation.max_reported_gaps],
            "gap_note": (
                "Gaps are measured on the SORTED output within each month; gaps across "
                "month boundaries are reported by `xq verify-dataset`."
            ),
            "first_timestamp": manifest.get("first_timestamp"),
            "last_timestamp": manifest.get("last_timestamp"),
            "source_ordering": manifest.get("source_ordering", {}),
            "non_monotonic_note": (
                "`non_monotonic` counts rows that were earlier than the row before them "
                "IN THE SOURCE. Each partition is sorted before it is written, so the "
                "output has none; `verify-dataset` checks that independently."
            ),
            "thresholds": {
                "min_price": self.config.validation.min_price,
                "max_price": self.config.validation.max_price,
                "max_spread": self.config.validation.max_spread,
                "max_volume": self.config.validation.max_volume,
                "large_gap_seconds": self.config.validation.large_gap_seconds,
            },
            "config_fingerprint": self.config.fingerprint(),
            "source_files": manifest.get("source_files", []),
            "dataset_version": manifest.get("dataset_version"),
        }
        validation["clean_rows_estimate"] = validation["rows_checked"] - validation["rows_flagged"]

        summary = CleaningSummary(
            dropped_by_reason=dict.fromkeys(CHECKS, 0),
            flagged_by_check=dict.fromkeys(CHECKS, 0),
            flagged_kept=dict.fromkeys(CHECKS, 0),
            enabled_drop_rules=list(Cleaner(self.config.cleaning).summary.enabled_drop_rules),
            policy=asdict(self.config.cleaning),
        )
        for entry in entries:
            summary.rows_input += int(entry.get("rows_input") or 0)
            summary.rows_output += int(entry.get("rows") or 0)
            for name, target in (
                ("dropped_by_reason", summary.dropped_by_reason),
                ("flagged_by_check", summary.flagged_by_check),
                ("flagged_kept", summary.flagged_kept),
            ):
                for check, value in (entry.get(name) or {}).items():
                    target[check] = target.get(check, 0) + int(value)
        summary.rows_dropped = summary.rows_input - summary.rows_output
        cleaning = summary.to_dict()
        cleaning["config_fingerprint"] = self.config.fingerprint()
        cleaning["complete_pass"] = complete
        cleaning["dataset_version"] = manifest.get("dataset_version")

        ensure_dir(directory)
        atomic_write_text(
            directory / VALIDATION_REPORT_NAME,
            json.dumps(validation, indent=2, default=str) + "\n",
        )
        atomic_write_text(
            directory / CLEANING_REPORT_NAME, json.dumps(cleaning, indent=2, default=str) + "\n"
        )
        self._last_validation = validation
        self._last_cleaning = cleaning

    # -- streaming parse (xq validate) -------------------------------------------
    def _iter_source_batches(
        self, source: Path, start_offset: int
    ) -> Iterator[pl.DataFrame]:
        """Yield raw record batches as Polars frames, bounded by block size."""
        wanted = list(self.config.input.source_columns.values())
        read_options = pacsv.ReadOptions(
            block_size=self.config.conversion.read_block_bytes,
            use_threads=True,
            column_names=self.source_column_names(source),
            skip_rows=0,
        )
        parse_options = pacsv.ParseOptions(
            delimiter=self.config.input.delimiter,
            quote_char=self.config.input.quote_char or False,
            newlines_in_values=False,
            invalid_row_handler=self._on_invalid_row,
        )
        convert_options = pacsv.ConvertOptions(
            column_types=self._source_arrow_types(),
            strings_can_be_null=True,
            include_columns=wanted,
        )

        with source.open("rb") as handle:
            handle.seek(start_offset)
            reader = pacsv.open_csv(
                handle,
                read_options=read_options,
                parse_options=parse_options,
                convert_options=convert_options,
            )
            for record_batch in reader:
                if record_batch.num_rows:
                    yield pl.from_arrow(record_batch)  # type: ignore[misc]

    def _on_invalid_row(self, _row: Any) -> str:
        """Count a malformed CSV row and skip it rather than aborting the run."""
        self._stream_validator.note_malformed(1)
        return "skip"

    # -- canonicalisation ----------------------------------------------------------
    def _to_canonical(self, batch: pl.DataFrame) -> pl.DataFrame:
        """Rename, parse, derive and order columns into the canonical schema."""
        inp, canon = self.config.input, self.config.canonical
        rename = {src: name for name, src in inp.source_columns.items()}
        frame = batch.rename({k: v for k, v in rename.items() if k in batch.columns})

        # Keep the raw string so the validator can tell "cell was empty" from
        # "cell could not be parsed".
        frame = frame.with_columns(pl.col(TIMESTAMP).alias(RAW_TIMESTAMP)).with_columns(
            pl.col(TIMESTAMP)
            .str.strptime(
                timestamp_dtype(canon), format=inp.timestamp_format, strict=False, exact=True
            )
            .alias(TIMESTAMP)
        )

        casts: list[pl.Expr] = []
        price = price_dtype(canon)
        for column in (BID, ASK):
            if column in frame.columns:
                casts.append(pl.col(column).cast(price, strict=False).alias(column))
        if "volume" in frame.columns:
            casts.append(pl.col("volume").cast(volume_dtype(canon), strict=False).alias("volume"))
        if "spread_source" in frame.columns:
            casts.append(
                pl.col("spread_source").cast(price, strict=False).alias("spread_source")
            )
        if casts:
            frame = frame.with_columns(casts)

        if "spread_source" in frame.columns and canon.keep_source_spread_as != "spread_source":
            frame = frame.rename({"spread_source": canon.keep_source_spread_as})

        derived = derive_expressions(inp, canon)
        if derived:
            frame = frame.with_columns(derived)
        if self._with_utc:
            frame = frame.with_columns(to_utc_expr(self.config.timezone, TIMESTAMP))

        # Order by the canonical schema so a renamed source-spread column
        # (canonical.keep_source_spread_as) is carried through correctly.
        ordered = [c for c in self._pl_schema if c in frame.columns]
        return frame.select([*ordered, RAW_TIMESTAMP])

    # -- schema --------------------------------------------------------------------
    def _check_headers(self, sources: list[Path]) -> None:
        """Every source file must share one layout, or rows would be misread."""
        first = self.source_column_names(sources[0])
        for other in sources[1:]:
            names = self.source_column_names(other)
            if names != first:
                raise ValueError(
                    f"{other.name} has columns {names} but {sources[0].name} has {first}; "
                    "all source files must share one layout."
                )

    def _source_arrow_types(self) -> dict[str, pa.DataType]:
        """Force the CSV parser's types: timestamps stay strings for Polars."""
        types: dict[str, pa.DataType] = {self.config.input.timestamp_column: pa.string()}
        numeric = (
            self.config.input.bid_column,
            self.config.input.ask_column,
            self.config.input.volume_column,
            self.config.input.spread_column,
        )
        for column in numeric:
            if column:
                types[column] = pa.float64()
        return types

    def _build_arrow_schema(self) -> pa.Schema:
        """Arrow schema for the output files, with provenance in the metadata."""
        unit = self.config.canonical.timestamp_precision
        fields: list[pa.Field] = []
        for name, dtype in self._pl_schema.items():
            if isinstance(dtype, pl.Datetime):
                arrow_type = pa.timestamp(unit, tz="UTC" if dtype.time_zone else None)
            elif dtype == pl.Float32():
                arrow_type = pa.float32()
            elif dtype == pl.Int64():
                arrow_type = pa.int64()
            else:
                arrow_type = pa.float64()
            fields.append(pa.field(name, arrow_type, nullable=True))
        metadata = {
            b"xq_schema_version": SCHEMA_VERSION.encode(),
            b"xq_instrument": self.config.instrument.encode(),
            b"xq_timezone_mode": self.config.timezone.mode.encode(),
            b"xq_timezone_description": self.config.timezone.describe().encode(),
            b"xq_timestamp_column": b"timestamp is broker-local wall clock, never rewritten",
            b"xq_config_fingerprint": self.config.fingerprint().encode(),
            b"xq_tick_fingerprint": self.config.tick_fingerprint().encode(),
            b"xq_sorted_by": b"timestamp (stable; ties keep source order)",
        }
        return pa.schema(fields, metadata=metadata)

    # -- logging ---------------------------------------------------------------------
    def _log_result(self, result: ConversionResult) -> None:
        LOGGER.info(
            "Conversion %s in %s (%s): %d partition(s) built, %d verified and reused, "
            "%d removed; this run read %s rows, wrote %s, dropped %s. Dataset: %d "
            "partitions, %s rows, %s%s",
            result.status, format_duration(result.duration_seconds), result.mode,
            len(result.partitions), len(result.skipped_partitions),
            len(result.removed_partitions), f"{result.rows_read:,}",
            f"{result.rows_written:,}", f"{result.rows_dropped:,}",
            result.total_partitions, f"{result.total_rows:,}",
            human_bytes(result.output_bytes),
            f", version {result.dataset_version}" if result.dataset_version else "",
        )
        if result.truncated_by_limit:
            LOGGER.warning(
                "OUTPUT IS PARTIAL: --limit-rows was used. It was written to %s and the "
                "canonical dataset was not touched.", result.dataset_dir,
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _first_lines(data: bytes, n: int) -> bytes:
    """The first *n* lines of *data*, newline included."""
    if n <= 0:
        return b""
    newlines = np.flatnonzero(np.frombuffer(data, dtype=np.uint8) == 10)
    if newlines.size < n:
        return data
    return data[: int(newlines[n - 1]) + 1]


def _gaps(
    frame: pl.DataFrame, threshold_seconds: float
) -> tuple[int, float, list[dict[str, Any]]]:
    """Gaps longer than the threshold between consecutive rows of a sorted month."""
    if frame.height < 2:
        return 0, 0.0, []
    stamps = frame.select(
        pl.col(TIMESTAMP).shift(1).alias("start"),
        pl.col(TIMESTAMP).alias("end"),
        (pl.col(TIMESTAMP) - pl.col(TIMESTAMP).shift(1))
        .dt.total_milliseconds().alias("ms"),
    ).drop_nulls()
    big = stamps.filter(pl.col("ms") > threshold_seconds * 1000.0).sort("ms", descending=True)
    if big.is_empty():
        return 0, 0.0, []
    listed = [
        {
            "start": row["start"].isoformat(sep=" "),
            "end": row["end"].isoformat(sep=" "),
            "seconds": row["ms"] / 1000.0,
        }
        for row in big.head(_MAX_GAPS_PER_PARTITION).iter_rows(named=True)
    ]
    return big.height, float(big["ms"][0]) / 1000.0, listed


def _same_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:  # pragma: no cover
        return str(a) == str(b)


def _is_pipeline_path(path: Path) -> bool:
    """Files and directories this pipeline creates inside a dataset directory."""
    name = path.name
    return (
        name.endswith(".parquet")
        or name.endswith(PARTIAL_SUFFIX)
        or name == DATASET_MANIFEST
        or name.startswith("year=")
        or name.startswith("month=")
    )


def _has_pipeline_files(root: Path) -> bool:
    return root.exists() and any(root.rglob("*.parquet"))


def _remove_empty_dirs(root: Path) -> None:
    if not root.exists():
        return
    for directory in sorted(
        (d for d in root.rglob("*") if d.is_dir()), key=lambda d: len(d.parts), reverse=True
    ):
        if not any(directory.iterdir()):
            directory.rmdir()


def _remove_pipeline_output(root: Path) -> None:
    """Delete this pipeline's output under *root*, and nothing else.

    Deliberately surgical rather than ``rmtree``: the directory may also hold
    files the pipeline did not create (``.gitkeep``, notes, a README), and they
    are left alone. Only ``*.parquet``, ``*.parquet.partial`` and the manifest
    are removed, then any directories that became empty.
    """
    if not root.exists():
        return
    for pattern in ("*.parquet", f"*{PARTIAL_SUFFIX}", DATASET_MANIFEST):
        for path in root.rglob(pattern):
            path.unlink(missing_ok=True)
    _remove_empty_dirs(root)
