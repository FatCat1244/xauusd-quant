"""Inspection of very large delimited text files.

Nothing in this module ever reads a file end-to-end. Everything is built from:

* the first and last few kilobytes (exact head/tail),
* a stratified set of fixed-size blocks read at evenly-spaced byte offsets,
* ``O(log n)`` binary search over line boundaries when a specific date is
  needed.

That keeps a 34 GB file inspectable in a couple of seconds, but it also means
every count derived from sampling is an *estimate*. The report labels them as
such, and :mod:`xauusd_quant.data.validator` is what produces exact figures
during the full conversion pass.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Any

import polars as pl

from ..utils.clock import utc_from_timestamp_iso, utc_now_iso
from ..utils.logging import get_logger
from ..utils.paths import atomic_write_text, human_bytes
from .schema import ColumnDetection, detect_columns

__all__ = [
    "BlockSample",
    "FileInspection",
    "find_offset_for_prefix",
    "inspect_file",
    "iter_block_samples",
    "read_head_lines",
    "read_tail_lines",
    "resolve_sources",
]

LOGGER = get_logger("data.inspector")

DEFAULT_BLOCKS = 40
DEFAULT_BLOCK_BYTES = 1 << 20  # 1 MiB
_NUMERIC_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


# ---------------------------------------------------------------------------
# Source resolution
# ---------------------------------------------------------------------------
def resolve_sources(path: Path) -> list[Path]:
    """Expand a file, directory or glob into a deterministic list of files."""
    if any(ch in str(path) for ch in "*?["):
        matches = sorted(Path(p) for p in __import__("glob").glob(str(path), recursive=True))
        return [m for m in matches if m.is_file()]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.is_file() and not p.name.startswith("."))
    return [path] if path.is_file() else []


# ---------------------------------------------------------------------------
# Byte-level primitives
# ---------------------------------------------------------------------------
def _next_line_start(handle: IO[bytes], offset: int, floor: int = 0) -> int:
    """Byte offset of the first line boundary at or after *offset*."""
    if offset <= floor:
        return floor
    handle.seek(offset - 1)
    handle.readline()  # discard the partial line
    return handle.tell()


def find_offset_for_prefix(
    path: Path, prefix: bytes, *, data_start: int = 0, size: int | None = None
) -> int:
    """Line-start offset of the first row whose raw bytes sort >= *prefix*.

    Only valid for files sorted by a lexicographically-ordered leading field
    (see ``input.timestamp_lexicographic``). Returns the file size when every
    row sorts below *prefix*.
    """
    size = size if size is not None else path.stat().st_size
    lo, hi = data_start, size
    with path.open("rb") as handle:
        while lo < hi:
            mid = (lo + hi) // 2
            start = _next_line_start(handle, mid, data_start)
            if start >= hi:
                hi = mid
                continue
            handle.seek(start)
            line = handle.readline()
            if not line:
                hi = mid
                continue
            if line[: len(prefix)] < prefix:
                lo = start + len(line)
            else:
                hi = start
        return _next_line_start(handle, lo, data_start)


def read_head_lines(path: Path, n: int, *, encoding: str = "utf-8") -> list[str]:
    """First *n* lines, decoded leniently so a bad byte cannot abort inspection."""
    out: list[str] = []
    with path.open("rb") as handle:
        for _ in range(n):
            line = handle.readline()
            if not line:
                break
            out.append(line.decode(encoding, errors="replace").rstrip("\r\n"))
    return out


def read_tail_lines(
    path: Path, n: int, *, encoding: str = "utf-8", window: int = 1 << 16
) -> list[str]:
    """Last *n* complete lines, read by seeking backwards from EOF.

    Cost is independent of file size, which is the whole point: a 34 GB file's
    final rows are available without touching the rest of it.
    """
    size = path.stat().st_size
    if size == 0:
        return []
    with path.open("rb") as handle:
        chunks: list[bytes] = []
        pos, found = size, 0
        while pos > 0 and found <= n:
            step = min(window, pos)
            pos -= step
            handle.seek(pos)
            chunk = handle.read(step)
            chunks.insert(0, chunk)
            found += chunk.count(b"\n")
        blob = b"".join(chunks)
    lines = blob.split(b"\n")
    if pos > 0:
        lines = lines[1:]  # first element is a partial line
    text = [ln.decode(encoding, errors="replace").rstrip("\r") for ln in lines]
    while text and not text[-1]:
        text.pop()
    return text[-n:]


@dataclass(frozen=True, slots=True)
class BlockSample:
    """One fixed-size block read at a byte offset, trimmed to whole lines."""

    offset: int
    lines: tuple[bytes, ...]


def iter_block_samples(
    path: Path,
    *,
    blocks: int = DEFAULT_BLOCKS,
    block_bytes: int = DEFAULT_BLOCK_BYTES,
    data_start: int = 0,
) -> Iterator[BlockSample]:
    """Yield *blocks* evenly-spaced samples, each trimmed to complete lines.

    Partial first/last lines are dropped, so a caller never parses a truncated
    row and mistakes it for corruption in the source.
    """
    size = path.stat().st_size
    usable = max(size - data_start, 0)
    if usable == 0:
        return
    if usable <= block_bytes * blocks:
        blocks, block_bytes = 1, usable

    with path.open("rb") as handle:
        span = max(usable - block_bytes, 0)
        for i in range(blocks):
            offset = data_start + (0 if blocks == 1 else round(i * span / (blocks - 1)))
            handle.seek(offset)
            raw = handle.read(block_bytes)
            if not raw:
                continue
            parts = raw.split(b"\n")
            # Drop the leading partial line unless we started exactly at a
            # known line start, and always drop the trailing partial line.
            head_partial = offset != data_start
            lines = parts[1:-1] if head_partial else parts[:-1]
            clean = tuple(ln.rstrip(b"\r") for ln in lines if ln.strip())
            if clean:
                yield BlockSample(offset=offset, lines=clean)


# ---------------------------------------------------------------------------
# Dialect sniffing
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Dialect:
    delimiter: str
    has_header: bool
    encoding: str
    has_bom: bool
    line_terminator: str
    quote_char: str | None


def sniff_dialect(path: Path, *, sample_bytes: int = 64 << 10) -> Dialect:
    """Infer delimiter, header presence, encoding and line terminator."""
    with path.open("rb") as handle:
        raw = handle.read(sample_bytes)

    has_bom = raw.startswith(b"\xef\xbb\xbf")
    encoding = "utf-8-sig" if has_bom else "utf-8"
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        encoding = "latin-1"

    terminator = "\r\n" if b"\r\n" in raw[: 4 << 10] else "\n"
    text = raw.decode(encoding, errors="replace")
    # Keep whole lines only, so the sniffer never sees a truncated record.
    text = text[: text.rfind("\n") + 1] or text

    delimiter: str = ","
    quote_char: str | None = '"'
    has_header = True
    try:
        sniffed = csv.Sniffer().sniff(text[: 8 << 10], delimiters=",;\t|")
        delimiter = sniffed.delimiter
        quote_char = sniffed.quotechar
    except csv.Error:
        counts = {d: text.count(d) for d in ",;\t|"}
        delimiter = max(counts, key=lambda d: counts[d]) if any(counts.values()) else ","
    try:
        has_header = csv.Sniffer().has_header(text[: 8 << 10])
    except csv.Error:
        first = text.split("\n", 1)[0].split(delimiter)
        has_header = not all(_NUMERIC_RE.match(f.strip()) for f in first if f.strip())

    return Dialect(
        delimiter=delimiter,
        has_header=has_header,
        encoding=encoding,
        has_bom=has_bom,
        line_terminator=terminator,
        quote_char=quote_char,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
@dataclass
class ColumnProfile:
    """Per-column statistics accumulated over the sampled blocks."""

    name: str
    index: int
    inferred_dtype: str
    non_null: int = 0
    nulls: int = 0
    numeric: int = 0
    min_value: float | None = None
    max_value: float | None = None
    examples: list[str] = field(default_factory=list)

    @property
    def null_rate(self) -> float:
        total = self.non_null + self.nulls
        return self.nulls / total if total else 0.0


@dataclass
class FileInspection:
    """Everything learned about one raw file without a full read."""

    path: str
    exists: bool
    size_bytes: int
    size_human: str
    modified_utc: str | None

    dialect: dict[str, Any] = field(default_factory=dict)
    header: list[str] = field(default_factory=list)
    head_lines: list[str] = field(default_factory=list)
    tail_lines: list[str] = field(default_factory=list)

    sampled_blocks: int = 0
    sampled_rows: int = 0
    mean_bytes_per_line: float = 0.0
    estimated_total_rows: int | None = None
    row_count_is_estimate: bool = True

    columns: list[dict[str, Any]] = field(default_factory=list)
    detection: dict[str, Any] = field(default_factory=dict)

    first_timestamp: str | None = None
    last_timestamp: str | None = None
    timestamp_samples: list[str] = field(default_factory=list)
    quote_samples: list[dict[str, Any]] = field(default_factory=list)

    field_count_mismatches: int = 0
    unparseable_numeric_cells: int = 0
    sampled_duplicate_rows: int = 0
    sampled_duplicate_timestamps: int = 0
    sampled_non_monotonic: int = 0
    sampled_ask_below_bid: int = 0
    spread_summary: dict[str, float] = field(default_factory=dict)
    observed_gaps: list[dict[str, Any]] = field(default_factory=list)

    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_file(
    path: Path,
    *,
    blocks: int = DEFAULT_BLOCKS,
    block_bytes: int = DEFAULT_BLOCK_BYTES,
    head: int = 5,
    tail: int = 5,
    timestamp_format: str | None = None,
    large_gap_seconds: float = 3600.0,
    max_gaps: int = 50,
) -> FileInspection:
    """Profile one large delimited file from head, tail and sampled blocks."""
    report = FileInspection(
        path=str(path),
        exists=path.exists(),
        size_bytes=0,
        size_human="0 B",
        modified_utc=None,
    )
    if not path.is_file():
        report.warnings.append(f"Not a readable file: {path}")
        return report

    stat = path.stat()
    report.size_bytes = stat.st_size
    report.size_human = human_bytes(stat.st_size)
    report.modified_utc = utc_from_timestamp_iso(stat.st_mtime)

    dialect = sniff_dialect(path)
    report.dialect = asdict(dialect)

    report.head_lines = read_head_lines(path, head + (1 if dialect.has_header else 0),
                                        encoding=dialect.encoding)
    report.tail_lines = read_tail_lines(path, tail, encoding=dialect.encoding)

    data_start = 0
    if dialect.has_header and report.head_lines:
        report.header = [h.strip() for h in report.head_lines[0].split(dialect.delimiter)]
        report.head_lines = report.head_lines[1:]
        with path.open("rb") as handle:
            handle.readline()
            data_start = handle.tell()
    else:
        ncols = len(report.head_lines[0].split(dialect.delimiter)) if report.head_lines else 0
        report.header = [f"column_{i}" for i in range(ncols)]
        report.notes.append("No header detected; columns are positional placeholders.")

    _profile_blocks(
        path, report, dialect, data_start,
        blocks=blocks, block_bytes=block_bytes,
        timestamp_format=timestamp_format,
        large_gap_seconds=large_gap_seconds, max_gaps=max_gaps,
    )

    detection = detect_columns(report.header, report.timestamp_samples)
    report.detection = detection.to_dict()
    _apply_exact_endpoints(report, detection, dialect)
    _add_warnings(report, detection)
    return report


def _profile_blocks(
    path: Path,
    report: FileInspection,
    dialect: Dialect,
    data_start: int,
    *,
    blocks: int,
    block_bytes: int,
    timestamp_format: str | None,
    large_gap_seconds: float,
    max_gaps: int,
) -> None:
    """Accumulate per-column stats and integrity counters over sampled blocks."""
    delim = dialect.delimiter.encode()
    ncols = len(report.header)
    profiles = [
        ColumnProfile(name=name, index=i, inferred_dtype="unknown")
        for i, name in enumerate(report.header)
    ]
    numeric_hits = [0] * ncols
    total_bytes = 0
    detection = detect_columns(report.header)
    idx = {c.canonical: c.index for c in detection.columns}
    ts_i, bid_i, ask_i = idx.get("timestamp"), idx.get("bid"), idx.get("ask")

    spreads: list[float] = []
    ts_values: list[str] = []
    prev_line: bytes | None = None
    prev_ts: bytes | None = None

    for sample in iter_block_samples(
        path, blocks=blocks, block_bytes=block_bytes, data_start=data_start
    ):
        report.sampled_blocks += 1
        # Ordering checks are only meaningful *inside* a block, since
        # consecutive blocks are far apart in the file.
        prev_line = prev_ts = None
        for raw in sample.lines:
            report.sampled_rows += 1
            total_bytes += len(raw) + 1
            cells = raw.split(delim)
            if len(cells) != ncols:
                report.field_count_mismatches += 1
                continue

            if prev_line == raw:
                report.sampled_duplicate_rows += 1
            prev_line = raw

            for i, cell in enumerate(cells):
                text = cell.strip()
                prof = profiles[i]
                if not text:
                    prof.nulls += 1
                    continue
                prof.non_null += 1
                decoded = text.decode(dialect.encoding, errors="replace")
                if len(prof.examples) < 5:
                    prof.examples.append(decoded)
                if _NUMERIC_RE.match(decoded):
                    numeric_hits[i] += 1
                    prof.numeric += 1
                    value = float(decoded)
                    prof.min_value = value if prof.min_value is None else min(prof.min_value, value)
                    prof.max_value = value if prof.max_value is None else max(prof.max_value, value)
                elif i in (bid_i, ask_i):
                    report.unparseable_numeric_cells += 1

            if ts_i is not None:
                ts_raw = cells[ts_i].strip()
                if len(ts_values) < 200:
                    ts_values.append(ts_raw.decode(dialect.encoding, errors="replace"))
                if prev_ts is not None:
                    if ts_raw == prev_ts:
                        report.sampled_duplicate_timestamps += 1
                    elif ts_raw < prev_ts:
                        report.sampled_non_monotonic += 1
                prev_ts = ts_raw

            if bid_i is not None and ask_i is not None:
                try:
                    bid = float(cells[bid_i])
                    ask = float(cells[ask_i])
                except ValueError:
                    continue
                if ask < bid:
                    report.sampled_ask_below_bid += 1
                if len(spreads) < 2_000_000:
                    spreads.append(ask - bid)

    for i, prof in enumerate(profiles):
        prof.inferred_dtype = _infer_dtype(prof, numeric_hits[i], ts_i, i)
    report.columns = [
        {**asdict(p), "null_rate": round(p.null_rate, 6)} for p in profiles
    ]

    if report.sampled_rows:
        report.mean_bytes_per_line = total_bytes / report.sampled_rows
        usable = max(report.size_bytes - data_start, 0)
        report.estimated_total_rows = int(round(usable / report.mean_bytes_per_line))

    report.timestamp_samples = ts_values[:10]
    if spreads:
        report.spread_summary = _summarise(spreads)
    _collect_quote_samples(report, dialect, idx)
    if ts_i is not None and timestamp_format:
        report.observed_gaps = _sampled_gaps(
            path, ts_i, dialect, data_start, timestamp_format, large_gap_seconds, max_gaps
        )


def _infer_dtype(
    prof: ColumnProfile, numeric: int, ts_index: int | None, index: int
) -> str:
    """Classify a column from its sampled values."""
    if prof.non_null == 0:
        return "empty"
    if index == ts_index:
        return "timestamp(string)"
    if numeric == prof.non_null:
        ints = prof.min_value is not None and prof.max_value is not None and all(
            float(v).is_integer() for v in (prof.min_value, prof.max_value)
        )
        return "int64" if ints and not any("." in e for e in prof.examples) else "float64"
    return "string"


def _summarise(values: list[float]) -> dict[str, float]:
    """Min/percentiles/max/mean over a sampled list."""
    ordered = sorted(values)
    n = len(ordered)

    def pct(p: float) -> float:
        return ordered[min(n - 1, max(0, int(round(p / 100 * (n - 1)))))]

    return {
        "sampled_count": float(n),
        "min": ordered[0],
        "p01": pct(1),
        "p25": pct(25),
        "median": pct(50),
        "p75": pct(75),
        "p99": pct(99),
        "max": ordered[-1],
        "mean": math.fsum(ordered) / n,
    }


def _collect_quote_samples(
    report: FileInspection, dialect: Dialect, idx: dict[str, int]
) -> None:
    """A handful of decoded bid/ask rows, for eyeballing in the CLI output."""
    bid_i, ask_i, ts_i = idx.get("bid"), idx.get("ask"), idx.get("timestamp")
    if bid_i is None or ask_i is None:
        return
    for line in (report.head_lines[:3] + report.tail_lines[-2:]):
        cells = [c.strip() for c in line.split(dialect.delimiter)]
        if max(bid_i, ask_i) >= len(cells):
            continue
        try:
            bid, ask = float(cells[bid_i]), float(cells[ask_i])
        except ValueError:
            continue
        report.quote_samples.append({
            "timestamp": cells[ts_i] if ts_i is not None and ts_i < len(cells) else None,
            "bid": bid,
            "ask": ask,
            "mid": (bid + ask) / 2,
            "spread": ask - bid,
        })


def _sampled_gaps(
    path: Path,
    ts_index: int,
    dialect: Dialect,
    data_start: int,
    timestamp_format: str,
    large_gap_seconds: float,
    max_gaps: int,
) -> list[dict[str, Any]]:
    """Large gaps observed *within* sampled blocks.

    These are illustrative only: gaps straddling two blocks are invisible here.
    Exact gap accounting happens during conversion.
    """
    gaps: list[dict[str, Any]] = []
    delim = dialect.delimiter
    for sample in iter_block_samples(path, blocks=12, block_bytes=1 << 20, data_start=data_start):
        text = b"\n".join(sample.lines).decode(dialect.encoding, errors="replace")
        try:
            frame = pl.read_csv(
                io.StringIO(text), has_header=False, separator=delim,
                infer_schema_length=0, truncate_ragged_lines=True,
            )
        except Exception:  # noqa: BLE001 - a bad sample must not abort inspection
            continue
        col = frame.columns[ts_index] if ts_index < frame.width else None
        if col is None:
            continue
        stamps = (
            frame.select(
                pl.col(col).str.strptime(pl.Datetime("us"), format=timestamp_format, strict=False)
                .alias("ts")
            )
            .drop_nulls()
            .with_columns(pl.col("ts").diff().dt.total_milliseconds().alias("delta_ms"))
            .filter(pl.col("delta_ms") > large_gap_seconds * 1000)
        )
        for row in stamps.iter_rows(named=True):
            gaps.append({
                "ends_at": row["ts"].isoformat(sep=" "),
                "gap_seconds": row["delta_ms"] / 1000.0,
                "block_offset": sample.offset,
            })
            if len(gaps) >= max_gaps:
                return gaps
    return gaps


def _apply_exact_endpoints(
    report: FileInspection, detection: ColumnDetection, dialect: Dialect
) -> None:
    """Take first/last timestamps from the exact head/tail, not from samples."""
    ts_index = next((c.index for c in detection.columns if c.canonical == "timestamp"), None)
    if ts_index is None:
        return
    if report.head_lines:
        cells = report.head_lines[0].split(dialect.delimiter)
        if ts_index < len(cells):
            report.first_timestamp = cells[ts_index].strip()
    if report.tail_lines:
        cells = report.tail_lines[-1].split(dialect.delimiter)
        if ts_index < len(cells):
            report.last_timestamp = cells[ts_index].strip()
    report.notes.append(
        "first_timestamp/last_timestamp are exact (read from the file's head and tail); "
        "all sampled_* counters and estimated_total_rows are estimates from sampled blocks."
    )


def _add_warnings(report: FileInspection, detection: ColumnDetection) -> None:
    """Surface anything that would block or distort the conversion."""
    if missing := detection.missing():
        report.warnings.append(
            f"Could not map required column(s) {list(missing)} from header {report.header}. "
            "Set them explicitly in config/data.yaml."
        )
    if detection.timestamp_format is None and report.timestamp_samples:
        report.warnings.append(
            f"Timestamp format not recognised from samples {report.timestamp_samples[:3]}. "
            "Set `timestamp_format` in config/data.yaml."
        )
    if report.field_count_mismatches:
        report.warnings.append(
            f"{report.field_count_mismatches:,} sampled row(s) did not have "
            f"{len(report.header)} fields."
        )
    if report.sampled_non_monotonic:
        report.warnings.append(
            f"{report.sampled_non_monotonic:,} sampled row(s) are out of timestamp order; "
            "byte-offset resume will be disabled."
        )
    if report.sampled_ask_below_bid:
        report.warnings.append(
            f"{report.sampled_ask_below_bid:,} sampled row(s) have ask < bid (crossed quotes)."
        )
    if not detection.timestamp_lexicographic and detection.timestamp_format:
        report.notes.append(
            "Timestamp format does not sort lexicographically; "
            "`convert --resume` will fall back to a full re-scan."
        )


# ---------------------------------------------------------------------------
# Multi-file report
# ---------------------------------------------------------------------------
def inspect_sources(
    paths: Sequence[Path], **kwargs: Any
) -> dict[str, Any]:
    """Inspect every source file and wrap the results in one report document."""
    files = [inspect_file(p, **kwargs) for p in paths]
    return {
        "generated_utc": utc_now_iso(),
        "file_count": len(files),
        "total_size_bytes": sum(f.size_bytes for f in files),
        "total_size_human": human_bytes(sum(f.size_bytes for f in files)),
        "estimated_total_rows": sum(f.estimated_total_rows or 0 for f in files) or None,
        "files": [f.to_dict() for f in files],
    }


def write_report(report: dict[str, Any], path: Path) -> Path:
    """Write *report* as pretty JSON, atomically."""
    atomic_write_text(path, json.dumps(report, indent=2, default=str) + os.linesep)
    LOGGER.info("Inspection report written: %s", path)
    return path
