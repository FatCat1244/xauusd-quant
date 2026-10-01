"""Byte-level index of the raw source: which lines belong to which partition.

Why this exists
---------------
The converter used to assume the source was sorted by time. It located a month
by binary search, closed a partition the moment a later month appeared, and
refused to resume if even one row was out of order. The real export is not
perfectly sorted, and under that design a row belonging to an already-closed
month would have *replaced* that month's file rather than joining it.

The index removes the assumption instead of the safety check. One read-only
pass over every source file records, for each ``year-month`` partition:

* the exact byte ranges ("runs") holding its lines, wherever they are;
* how many lines it has, and its earliest and latest raw timestamp;
* a content hash of exactly those lines, in source order.

A partition can then be converted, verified and resumed entirely on its own,
whatever order the source is in: its lines are read from its runs and nothing
else, and the hash proves the bytes are the ones the index saw.

The same pass measures the source's ordering precisely - every backward time
jump, how far back it goes, how many lines stay behind the running maximum,
and whether any of them belong to an *earlier partition* than the one the
stream had reached - and hashes every file, which gives the raw dataset its
fingerprint.

Partition keys come from the raw timestamp *text*: the year and month digits
at fixed positions derived from ``timestamp_format``. That is the broker-local
wall clock exactly as recorded, which is what partitioning has always used.
Every line is accounted for exactly once: it either belongs to one partition
or is counted as *unassigned* (no recognisable year-month), and the header is
counted separately.

The raw files are opened read-only and never written.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.clock import utc_now_iso
from ..utils.config import Config
from ..utils.logging import format_duration, get_logger
from ..utils.paths import atomic_write_text, ensure_dir, human_bytes

__all__ = [
    "INDEX_FORMAT_VERSION",
    "INDEX_NAME",
    "ORDERING_REPORT_NAME",
    "PartitionSource",
    "SourceFileInfo",
    "SourceIndex",
    "SourceIndexer",
    "month_key_layout",
    "ordering_report",
    "read_partition_bytes",
    "read_runs",
]

LOGGER = get_logger("data.source_index")

INDEX_NAME = "source_index.json"
ORDERING_REPORT_NAME = "source_ordering_report.json"
INDEX_FORMAT_VERSION = 2

_BOM = b"\xef\xbb\xbf"
_MAX_FIELD = 32          # bytes of the timestamp field kept for keys and ordering
_MAX_EVENTS = 100_000    # backward-jump events kept in full; counts stay exact
_MAX_UNASSIGNED_EXAMPLES = 50

# Fixed-width strftime/chrono directives that may precede %Y / %m.
_FIXED_WIDTH = {"Y": 4, "m": 2, "d": 2, "H": 2, "M": 2, "S": 2, "y": 2}


# ---------------------------------------------------------------------------
# Timestamp layout
# ---------------------------------------------------------------------------
def month_key_layout(timestamp_format: str) -> tuple[int, int]:
    """Character offsets of the year and month digits in a rendered timestamp.

    Only the part of the format *before* ``%Y`` and ``%m`` has to be fixed
    width, because only those offsets are needed. ``%Y%m%d %H:%M:%S%.f`` gives
    ``(0, 4)``; ``%d/%m/%Y`` gives ``(6, 3)``.
    """
    positions: dict[str, int] = {}
    offset = 0
    i = 0
    while i < len(timestamp_format) and len(positions) < 2:
        char = timestamp_format[i]
        if char != "%":
            offset += 1
            i += 1
            continue
        if i + 1 >= len(timestamp_format):
            break
        directive = timestamp_format[i + 1]
        if directive in ("Y", "m"):
            positions[directive] = offset
        if directive == "%":
            offset += 1
        elif directive in _FIXED_WIDTH:
            offset += _FIXED_WIDTH[directive]
        else:
            break
        i += 2
    if set(positions) != {"Y", "m"}:
        raise ValueError(
            f"timestamp_format {timestamp_format!r}: the source index needs %Y and %m at "
            "fixed character positions, i.e. only fixed-width directives (%Y %m %d %H "
            "%M %S %y) and literal characters before them."
        )
    return positions["Y"], positions["m"]


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
@dataclass
class SourceFileInfo:
    """Identity and line accounting for one raw file."""

    path: str
    size: int
    mtime_ns: int
    blake2b: str = ""
    header: str | None = None
    data_start: int = 0
    lines: int = 0
    unassigned_lines: int = 0

    def signature(self) -> dict[str, Any]:
        return {"path": self.path, "size": self.size, "mtime_ns": self.mtime_ns}


@dataclass
class PartitionSource:
    """Where one partition's lines live in the source, and what they hash to."""

    key: str
    runs: list[list[int]] = field(default_factory=list)   # [file_index, start, end)
    lines: int = 0
    source_bytes: int = 0
    first_raw_timestamp: str = ""
    last_raw_timestamp: str = ""
    content_hash: str = ""

    @property
    def year(self) -> int:
        return int(self.key[:4])

    @property
    def month(self) -> int:
        return int(self.key[5:7])


@dataclass
class SourceIndex:
    """The complete line-to-partition map of the raw source."""

    format_version: int = INDEX_FORMAT_VERSION
    created_utc: str = field(default_factory=utc_now_iso)
    layout: dict[str, Any] = field(default_factory=dict)
    files: list[SourceFileInfo] = field(default_factory=list)
    partitions: dict[str, PartitionSource] = field(default_factory=dict)
    total_lines: int = 0
    unassigned_lines: int = 0
    unassigned_examples: list[dict[str, Any]] = field(default_factory=list)
    # Lines with no readable year-month are kept track of exactly like a
    # partition, so the converter can still parse them and attribute each one
    # to a precise reason (malformed, missing or unparseable timestamp).
    unassigned_runs: list[list[int]] = field(default_factory=list)
    unassigned_hash: str = ""
    ordering: dict[str, Any] = field(default_factory=dict)
    raw_fingerprint: str = ""
    duration_seconds: float = 0.0

    # -- derived views -----------------------------------------------------
    @property
    def keys(self) -> list[str]:
        return sorted(self.partitions)

    @property
    def assigned_lines(self) -> int:
        return sum(p.lines for p in self.partitions.values())

    def signature(self) -> list[dict[str, Any]]:
        return [f.signature() for f in self.files]

    def accounting_ok(self) -> bool:
        """Every data line is in exactly one partition or counted as unassigned."""
        return self.assigned_lines + self.unassigned_lines == self.total_lines

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["partitions"] = {k: asdict(self.partitions[k]) for k in self.keys}
        payload["partition_count"] = len(self.partitions)
        payload["assigned_lines"] = self.assigned_lines
        payload["accounting_ok"] = self.accounting_ok()
        payload["source_bytes"] = sum(f.size for f in self.files)
        payload["source_bytes_human"] = human_bytes(payload["source_bytes"])
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SourceIndex:
        files = [SourceFileInfo(**f) for f in payload.get("files", [])]
        partitions = {
            k: PartitionSource(**v) for k, v in payload.get("partitions", {}).items()
        }
        known = set(cls.__dataclass_fields__) - {"files", "partitions"}
        scalars = {k: v for k, v in payload.items() if k in known}
        return cls(files=files, partitions=partitions, **scalars)


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------
class _PartitionAccumulator:
    __slots__ = ("hasher", "lo", "record")

    def __init__(self, key: str) -> None:
        self.record = PartitionSource(key=key)
        self.hasher = hashlib.blake2b(digest_size=16)
        self.lo: bytes | None = None


class SourceIndexer:
    """Builds, caches and validates the :class:`SourceIndex` for a config.

    ``load_or_build`` reuses the cached index only while every source file
    keeps its size and modification time and the layout settings are
    unchanged. Converting a partition re-hashes its bytes against the index,
    so a file edited in place without changing size or mtime is still caught
    at the partition that changed.
    """

    def __init__(self, config: Config, *, block_bytes: int = 64 << 20) -> None:
        self.config = config
        self.block_bytes = block_bytes
        inp = config.input
        self._delimiter = inp.delimiter.encode()[0]
        self._quote = (inp.quote_char or "").encode()[:1]
        self._year_at, self._month_at = month_key_layout(inp.timestamp_format or "")
        self._lexicographic = bool(inp.timestamp_lexicographic)
        self._ts_col: int | None = None
        self._unassigned_hasher = hashlib.blake2b(digest_size=16)

    # -- cache ---------------------------------------------------------------
    @property
    def path(self) -> Path:
        return self.config.metadata_path / INDEX_NAME

    def layout(self) -> dict[str, Any]:
        """Settings that change how lines are assigned; part of the cache key."""
        inp = self.config.input
        return {
            "delimiter": inp.delimiter,
            "quote_char": inp.quote_char,
            "has_header": inp.has_header,
            "timestamp_column": inp.timestamp_column,
            "timestamp_format": inp.timestamp_format,
            "timestamp_lexicographic": inp.timestamp_lexicographic,
            "year_offset": self._year_at,
            "month_offset": self._month_at,
        }

    def load(self) -> SourceIndex | None:
        if not self.path.exists():
            return None
        try:
            return SourceIndex.from_dict(json.loads(self.path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, TypeError, KeyError) as exc:
            LOGGER.warning("Ignoring unreadable source index %s: %s", self.path, exc)
            return None

    def stale_reasons(self, index: SourceIndex, sources: list[Path]) -> list[str]:
        """Why a cached index may not be used, or an empty list if it may."""
        reasons: list[str] = []
        if index.format_version != INDEX_FORMAT_VERSION:
            reasons.append("index format changed")
        if index.layout != self.layout():
            reasons.append("layout settings changed")
        current = [_signature(s) for s in sources]
        if index.signature() != current:
            reasons.append("source files changed (path, size or mtime)")
        return reasons

    def load_or_build(self, sources: list[Path], *, rebuild: bool = False) -> SourceIndex:
        if not rebuild:
            cached = self.load()
            if cached is not None:
                reasons = self.stale_reasons(cached, sources)
                if not reasons:
                    return cached
                LOGGER.info("Source index is stale (%s); rebuilding", "; ".join(reasons))
        index = self.build(sources)
        self.save(index)
        return index

    def save(self, index: SourceIndex) -> Path:
        ensure_dir(self.path.parent)
        atomic_write_text(self.path, json.dumps(index.to_dict(), indent=1) + "\n")
        return self.path

    # -- build ---------------------------------------------------------------
    def build(self, sources: list[Path]) -> SourceIndex:
        """One read-only pass over every source file."""
        started = time.perf_counter()
        index = SourceIndex(layout=self.layout())
        state = _ScanState(lexicographic=self._lexicographic)
        accumulators: dict[str, _PartitionAccumulator] = {}
        self._unassigned_hasher = hashlib.blake2b(digest_size=16)
        total = sum(s.stat().st_size for s in sources)
        LOGGER.info("Indexing %d source file(s), %s", len(sources), human_bytes(total))
        for file_index, source in enumerate(sources):
            info = self._scan_file(file_index, source, accumulators, state, index)
            index.files.append(info)
        for key in sorted(accumulators):
            acc = accumulators[key]
            acc.record.content_hash = acc.hasher.hexdigest()
            index.partitions[key] = acc.record
        index.unassigned_hash = self._unassigned_hasher.hexdigest()
        index.total_lines = sum(f.lines for f in index.files)
        index.unassigned_lines = sum(f.unassigned_lines for f in index.files)
        index.ordering = state.summary()
        combined = hashlib.blake2b(digest_size=16)
        for info in index.files:
            combined.update(Path(info.path).name.encode())
            combined.update(info.blake2b.encode())
        index.raw_fingerprint = combined.hexdigest()
        index.duration_seconds = time.perf_counter() - started
        if not index.accounting_ok():  # pragma: no cover - a defect in this module
            raise RuntimeError("source index accounting does not balance")
        LOGGER.info(
            "Indexed %s lines into %d partitions in %s (%s unassigned, %s backward "
            "jumps, %s lines behind the running maximum, %s of them in an earlier "
            "partition)",
            f"{index.total_lines:,}", len(index.partitions),
            format_duration(index.duration_seconds), f"{index.unassigned_lines:,}",
            f"{index.ordering.get('backward_jumps', 0):,}",
            f"{index.ordering.get('late_lines', 0):,}",
            f"{index.ordering.get('late_lines_in_earlier_partition', 0):,}",
        )
        return index

    def _scan_file(
        self,
        file_index: int,
        source: Path,
        accumulators: dict[str, _PartitionAccumulator],
        state: _ScanState,
        index: SourceIndex,
    ) -> SourceFileInfo:
        stat = source.stat()
        info = SourceFileInfo(path=str(source), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        hasher = hashlib.blake2b(digest_size=32)
        with source.open("rb") as handle:
            data_start = 0
            if self.config.input.has_header:
                header = handle.readline()
                hasher.update(header)
                data_start = len(header)
                info.header = header.decode(self.config.input.encoding, errors="replace").rstrip(
                    "\r\n"
                ).lstrip("﻿")
            else:
                head = handle.read(len(_BOM))
                if head == _BOM:
                    hasher.update(head)
                    data_start = len(_BOM)
                else:
                    handle.seek(0)
            info.data_start = data_start
            first_line_number = 2 if self.config.input.has_header else 1

            position = data_start
            pending = b""
            last_log = time.perf_counter()
            while True:
                if time.perf_counter() - last_log >= 30:
                    last_log = time.perf_counter()
                    LOGGER.info(
                        "  ... indexed %s of %s (%.0f%%)", human_bytes(position),
                        human_bytes(info.size), 100.0 * position / max(info.size, 1),
                    )
                chunk = handle.read(self.block_bytes)
                if chunk:
                    hasher.update(chunk)
                buffer = pending + chunk if pending else chunk
                if not buffer:
                    break
                if chunk:
                    cut = buffer.rfind(b"\n")
                    if cut < 0:
                        pending = buffer
                        continue
                    body, pending = buffer[: cut + 1], buffer[cut + 1:]
                else:
                    body, pending = buffer, b""
                lines = self._scan_block(
                    body, position, file_index, first_line_number + info.lines,
                    accumulators, state, index, info,
                )
                info.lines += lines
                position += len(body)
                if not chunk:
                    break
        info.blake2b = hasher.hexdigest()
        return info

    def _scan_block(
        self,
        body: bytes,
        base: int,
        file_index: int,
        first_line_number: int,
        accumulators: dict[str, _PartitionAccumulator],
        state: _ScanState,
        index: SourceIndex,
        info: SourceFileInfo,
    ) -> int:
        """Assign every line of *body* to a partition and update the ordering state."""
        arr = np.frombuffer(body, dtype=np.uint8)
        newlines = np.flatnonzero(arr == 10)
        if body.endswith(b"\n"):
            ends = newlines + 1
            content_ends = newlines.copy()
        else:  # the last line of a file with no trailing newline
            ends = np.append(newlines + 1, len(body))
            content_ends = np.append(newlines, len(body))
        n = int(ends.size)
        starts = np.empty(n, dtype=np.int64)
        starts[0] = 0
        starts[1:] = ends[:-1]
        # Ignore a trailing \r so CRLF files behave like LF files.
        has_cr = (content_ends > starts) & (arr[np.maximum(content_ends - 1, 0)] == 13)
        content_ends = content_ends - has_cr.astype(np.int64)

        fields = self._field_matrix(arr, starts, content_ends)
        keys, valid = self._month_keys(fields)

        # -- partition runs and content hashes ---------------------------------
        view = memoryview(body)
        change = np.flatnonzero(keys[1:] != keys[:-1]) + 1
        seg_lo = np.concatenate(([0], change))
        seg_hi = np.concatenate((change, [n]))
        stamps = (
            _BlockStamps(fields, valid, self._year_at, self._month_at) if valid.any() else None
        )
        for lo, hi in zip(seg_lo.tolist(), seg_hi.tolist(), strict=True):
            key = int(keys[lo])
            byte_lo, byte_hi = int(starts[lo]), int(ends[hi - 1])
            if key < 0:
                info.unassigned_lines += hi - lo
                self._unassigned_hasher.update(view[byte_lo:byte_hi])
                start, end = base + byte_lo, base + byte_hi
                runs = index.unassigned_runs
                if runs and runs[-1][0] == file_index and runs[-1][2] == start:
                    runs[-1][2] = end
                else:
                    runs.append([file_index, start, end])
                if len(index.unassigned_examples) < _MAX_UNASSIGNED_EXAMPLES:
                    for i in range(lo, min(hi, lo + _MAX_UNASSIGNED_EXAMPLES)):
                        index.unassigned_examples.append({
                            "file_index": file_index,
                            "line": first_line_number + i,
                            "offset": base + int(starts[i]),
                            "text": bytes(arr[starts[i]:content_ends[i]]).decode(
                                "latin-1")[:120],
                        })
                        if len(index.unassigned_examples) >= _MAX_UNASSIGNED_EXAMPLES:
                            break
                continue
            name = f"{key // 100:04d}-{key % 100:02d}"
            acc = accumulators.get(name)
            if acc is None:
                acc = accumulators[name] = _PartitionAccumulator(name)
            acc.hasher.update(view[byte_lo:byte_hi])
            record = acc.record
            start, end = base + byte_lo, base + byte_hi
            if record.runs and record.runs[-1][0] == file_index and record.runs[-1][2] == start:
                record.runs[-1][2] = end
            else:
                record.runs.append([file_index, start, end])
            record.lines += hi - lo
            record.source_bytes += byte_hi - byte_lo
            if stamps is not None:
                seg = stamps.segment_extremes(lo, hi)
                if seg is not None:
                    low, high = seg
                    if not record.first_raw_timestamp or low < record.first_raw_timestamp:
                        record.first_raw_timestamp = low
                    if high > record.last_raw_timestamp:
                        record.last_raw_timestamp = high

        if stamps is not None and self._lexicographic:
            state.update(stamps, keys, starts, base, file_index, first_line_number)
        return n

    # -- field extraction ------------------------------------------------------
    def _field_matrix(
        self, arr: np.ndarray, starts: np.ndarray, content_ends: np.ndarray
    ) -> np.ndarray:
        """The timestamp field of every line as a zero-padded ``(n, W)`` byte matrix.

        One strided row gather copies the first ``W`` bytes from each field
        start; everything from the field's end (the next delimiter, the end of
        the line, or ``W``) onward is zeroed. Month keys and the ordering key
        are both read from this one matrix, which is what keeps the scan fast.
        """
        width = _MAX_FIELD
        column = self._timestamp_column_index
        if column == 0:
            fstart = starts
        else:
            delims = np.flatnonzero(arr == self._delimiter)
            if delims.size == 0:
                fstart = content_ends.copy()
            else:
                # After the column-th delimiter at or after the line start; a
                # line with too few fields gets an empty field.
                idx = np.searchsorted(delims, starts) + (column - 1)
                found = idx < delims.size
                fstart = np.where(
                    found, delims[np.minimum(idx, delims.size - 1)] + 1, content_ends
                )
                fstart = np.minimum(fstart, content_ends)
        padded = np.zeros(arr.size + width, dtype=np.uint8)
        padded[: arr.size] = arr
        window = np.lib.stride_tricks.as_strided(
            padded, shape=(arr.size + 1, width), strides=(1, 1), writeable=False
        )
        matrix = np.array(window[fstart])
        room = np.maximum(content_ends - fstart, 0)
        is_delim = matrix == self._delimiter
        first = is_delim.argmax(axis=1)
        found = is_delim[np.arange(first.size), first]
        length = np.minimum(np.where(found, first, width), room)
        if self._quote:
            q = self._quote[0]
            opens = (length > 0) & (matrix[:, 0] == q)
            if opens.any():
                rows = np.flatnonzero(opens)
                matrix[rows, :-1] = matrix[rows, 1:]
                matrix[rows, -1] = 0
                length[rows] -= 1
                last = matrix[rows, np.maximum(length[rows] - 1, 0)]
                length[rows[(length[rows] > 0) & (last == q)]] -= 1
        uniform = int(length[0]) if length.size and (length == length[0]).all() else -1
        if uniform >= 0:
            # Fixed-width timestamps, the usual case: one slice instead of a mask.
            matrix[:, uniform:] = 0
        else:
            matrix[np.arange(width)[None, :] >= length[:, None]] = 0
        return matrix

    @property
    def _timestamp_column_index(self) -> int:
        if self._ts_col is None:
            raise RuntimeError(
                "Call resolve_columns() with the source column names before indexing."
            )
        return self._ts_col

    def resolve_columns(self, names: list[str]) -> None:
        """Locate the timestamp column among the source's column names."""
        column = self.config.input.timestamp_column
        if column not in names:
            raise KeyError(
                f"Timestamp column {column!r} is not in {names}. Run "
                "`xq detect-schema --save` or fix config/data.yaml."
            )
        self._ts_col = names.index(column)

    def _month_keys(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``year*100 + month`` per line, -1 where no year-month can be read."""
        # uint8 arithmetic: a byte below '0' wraps round to >= 208, so a single
        # `<= 9` test rejects every non-digit.
        year_digits = matrix[:, self._year_at:self._year_at + 4] - np.uint8(48)
        month_digits = matrix[:, self._month_at:self._month_at + 2] - np.uint8(48)
        valid = (year_digits <= 9).all(axis=1) & (month_digits <= 9).all(axis=1)
        year = (
            year_digits[:, 0].astype(np.int32) * 1000 + year_digits[:, 1].astype(np.int32) * 100
            + year_digits[:, 2].astype(np.int32) * 10 + year_digits[:, 3].astype(np.int32)
        )
        month = month_digits[:, 0].astype(np.int32) * 10 + month_digits[:, 1].astype(np.int32)
        valid &= (month >= 1) & (month <= 12)
        return np.where(valid, year * 100 + month, -1).astype(np.int64), valid


# ---------------------------------------------------------------------------
# Timestamp text of one block, as fixed-width byte strings
# ---------------------------------------------------------------------------
class _BlockStamps:
    """The timestamp field of every *valid* line as a zero-padded ``S`` array.

    Zero padding keeps lexicographic order correct for fields of different
    lengths (``...:03`` sorts before ``...:03.5``), so a lexicographic
    timestamp format can be ordered without being parsed.
    """

    def __init__(
        self, matrix: np.ndarray, valid: np.ndarray, year_at: int, month_at: int,
    ) -> None:
        self.valid_index = np.flatnonzero(valid)
        self.width = matrix.shape[1]
        rows = matrix if self.valid_index.size == matrix.shape[0] else matrix[self.valid_index]
        self.values = np.ascontiguousarray(rows).view(f"S{self.width}").ravel()
        self.year_at, self.month_at = year_at, month_at
        # Position of each line in `values`, -1 for invalid lines.
        self.position = np.full(matrix.shape[0], -1, dtype=np.int64)
        self.position[self.valid_index] = np.arange(self.valid_index.size)
        self.back = (
            np.flatnonzero(self.values[1:] < self.values[:-1]) + 1
            if self.values.size > 1 else np.empty(0, dtype=np.int64)
        )

    def segment_extremes(self, lo: int, hi: int) -> tuple[str, str] | None:
        """Min and max timestamp text among valid lines in line range [lo, hi)."""
        pos = self.position[lo:hi]
        pos = pos[pos >= 0]
        if pos.size == 0:
            return None
        a, b = int(pos[0]), int(pos[-1]) + 1
        # Within a run of non-decreasing values the first is the minimum and the
        # last the maximum, so only run edges are candidates.
        inner = self.back[(self.back > a) & (self.back < b)]
        lows = [self.values[a], *self.values[inner]]
        highs = [self.values[b - 1], *self.values[inner - 1]]
        return (
            min(lows).decode("latin-1"),
            max(highs).decode("latin-1"),
        )


# ---------------------------------------------------------------------------
# Ordering state carried across blocks and files
# ---------------------------------------------------------------------------
class _ScanState:
    """Exact ordering statistics for the whole stream, in source order."""

    def __init__(self, *, lexicographic: bool) -> None:
        self.lexicographic = lexicographic
        self.prev: bytes | None = None
        self.running_max: bytes | None = None
        self.backward_jumps = 0
        self.equal_adjacent = 0
        self.late_lines = 0
        self.late_lines_in_earlier_partition = 0
        self.events: list[dict[str, Any]] = []
        self._open_event: dict[str, Any] | None = None

    @staticmethod
    def _key_of(value: bytes, year_at: int, month_at: int) -> int:
        return int(value[year_at:year_at + 4]) * 100 + int(value[month_at:month_at + 2])

    def update(
        self, stamps: _BlockStamps, keys: np.ndarray, starts: np.ndarray, base: int,
        file_index: int, first_line_number: int,
    ) -> None:
        values = stamps.values
        if values.size == 0:
            return
        line_of = stamps.valid_index
        valid_keys = keys[line_of]
        # Adjacent equality, including across the block boundary.
        self.equal_adjacent += int(np.count_nonzero(values[1:] == values[:-1]))
        first = bytes(values[0])
        jumps = stamps.back.tolist()
        if self.prev is not None:
            if first == self.prev:
                self.equal_adjacent += 1
            if first < self.prev:
                jumps = [0, *jumps]
        # Split into non-decreasing runs; inside each only a prefix can be late.
        edges = [0, *[j for j in jumps if j > 0], values.size]
        jump_set = set(jumps)
        for a, b in zip(edges[:-1], edges[1:], strict=True):
            if a == b:
                continue
            segment = values[a:b]
            rm = self.running_max
            late = 0
            earlier = 0
            if rm is not None:
                late = int(np.searchsorted(segment, np.bytes_(rm), side="left"))
                if late:
                    rm_key = self._key_of(rm, stamps.year_at, stamps.month_at)
                    earlier = int(np.count_nonzero(valid_keys[a:a + late] < rm_key))
            self.late_lines += late
            self.late_lines_in_earlier_partition += earlier

            if self._open_event is not None:
                # A late run that reached the end of the previous block continues here.
                if a == 0 and 0 not in jump_set:
                    self._open_event["late_run_lines"] += late
                    self._open_event["late_lines_in_earlier_partition"] += earlier
                    if late < b - a:
                        self._open_event = None
                else:
                    self._open_event = None

            if a in jump_set:
                previous = bytes(values[a - 1]) if a > 0 else (self.prev or b"")
                current = bytes(values[a])
                self.backward_jumps += 1
                event = {
                    "file_index": file_index,
                    "line": first_line_number + int(line_of[a]),
                    "offset": base + int(starts[line_of[a]]),
                    "previous_timestamp": previous.decode("latin-1"),
                    "timestamp": current.decode("latin-1"),
                    "running_max": (rm or previous).decode("latin-1"),
                    "late_run_lines": late,
                    "late_lines_in_earlier_partition": earlier,
                    "crosses_partition": bool(
                        rm is not None
                        and self._key_of(current, stamps.year_at, stamps.month_at)
                        < self._key_of(rm, stamps.year_at, stamps.month_at)
                    ),
                }
                if len(self.events) < _MAX_EVENTS:
                    self.events.append(event)
                self._open_event = event if (late == b - a and b == values.size) else None
            last = bytes(segment[-1])
            if rm is None or last > rm:
                self.running_max = last
        self.prev = bytes(values[-1])

    def summary(self) -> dict[str, Any]:
        if not self.lexicographic:
            return {
                "computed": False,
                "reason": "timestamp_lexicographic is false; ordering is checked per "
                          "partition by the validator instead",
            }
        return {
            "computed": True,
            "input_sorted": self.backward_jumps == 0,
            "backward_jumps": self.backward_jumps,
            "late_lines": self.late_lines,
            "late_lines_in_earlier_partition": self.late_lines_in_earlier_partition,
            "equal_adjacent_timestamps": self.equal_adjacent,
            "events_recorded": len(self.events),
            "events_truncated": self.backward_jumps > len(self.events),
            "events": self.events,
            "definitions": {
                "backward_jump": "a line whose timestamp is earlier than the line before it",
                "late_line": "a line whose timestamp is earlier than the maximum seen so far",
                "late_lines_in_earlier_partition": (
                    "late lines whose year-month is earlier than the year-month of the "
                    "running maximum: rows that arrive after the stream has already "
                    "moved into a later partition"
                ),
            },
        }


def _signature(source: Path) -> dict[str, Any]:
    stat = source.stat()
    return {"path": str(source), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


# ---------------------------------------------------------------------------
# Reading one partition back
# ---------------------------------------------------------------------------
def read_partition_bytes(
    index: SourceIndex, key: str, sources: list[Path]
) -> tuple[bytes, str]:
    """The raw lines of one partition, in source order, and their content hash.

    The caller compares the hash with ``index.partitions[key].content_hash``;
    a mismatch means the source changed after it was indexed.
    """
    return read_runs(index.partitions[key].runs, sources)


def read_runs(runs: list[list[int]], sources: list[Path]) -> tuple[bytes, str]:
    """Concatenate byte runs ``[file_index, start, end)`` and hash them.

    A run that does not end in a newline (the last line of a file) gets one
    appended so it cannot fuse with the next run; the separator is not hashed.
    """
    hasher = hashlib.blake2b(digest_size=16)
    chunks: list[bytes] = []
    handles: dict[int, Any] = {}
    try:
        for file_index, start, end in runs:
            handle = handles.get(file_index)
            if handle is None:
                handle = handles[file_index] = sources[file_index].open("rb")
            handle.seek(start)
            data = handle.read(end - start)
            if len(data) != end - start:
                raise OSError(
                    f"{sources[file_index]}: expected {end - start} bytes at {start}, "
                    f"read {len(data)} - the file is shorter than when it was indexed"
                )
            hasher.update(data)
            chunks.append(data)
            if not data.endswith(b"\n"):
                chunks.append(b"\n")
    finally:
        for handle in handles.values():
            handle.close()
    return b"".join(chunks), hasher.hexdigest()


# ---------------------------------------------------------------------------
# Ordering diagnosis
# ---------------------------------------------------------------------------
_SPLIT = re.compile(rb"\r?\n")


def ordering_report(
    index: SourceIndex,
    sources: list[Path],
    *,
    context_lines: int = 3,
    lookback_bytes: int = 4 << 20,
    max_events: int = 2000,
) -> dict[str, Any]:
    """Explain every backward jump, with the surrounding lines.

    For each event the lines that arrive behind the running maximum (the
    "late run") are compared, byte for byte, with the lines that preceded the
    jump. A late line found verbatim among them is a *repeat*; one that is not
    is a genuinely out-of-order observation that must be kept and moved into
    place. Nothing is deleted on the strength of this report: it describes the
    source, and the cleaning policy is configured separately.
    """
    ordering = index.ordering
    events = ordering.get("events", []) if ordering.get("computed") else []
    rows: list[dict[str, Any]] = []
    for event in events[:max_events]:
        source = sources[event["file_index"]]
        offset = int(event["offset"])
        lo = max(index.files[event["file_index"]].data_start, offset - lookback_bytes)
        with source.open("rb") as handle:
            handle.seek(lo)
            before_blob = handle.read(offset - lo)
            late_n = int(event["late_run_lines"])
            # Read enough to cover the late run plus the context after it.
            after_blob = b""
            while after_blob.count(b"\n") < late_n + context_lines + 1:
                more = handle.read(1 << 20)
                if not more:
                    break
                after_blob += more
        before = [ln for ln in _SPLIT.split(before_blob) if ln]
        if lo > index.files[event["file_index"]].data_start and before:
            before = before[1:]  # the first line may be cut in half
        after = _SPLIT.split(after_blob)
        late_lines = after[:late_n]
        earlier = set(before)
        repeats = sum(1 for ln in late_lines if ln in earlier)
        first_late = late_lines[0].split(b",")[0].decode("latin-1") if late_lines else ""
        last_late = late_lines[-1].split(b",")[0].decode("latin-1") if late_lines else ""
        rows.append({
            **{k: event[k] for k in (
                "file_index", "line", "offset", "previous_timestamp", "timestamp",
                "running_max", "late_run_lines", "crosses_partition",
                "late_lines_in_earlier_partition",
            )},
            "backward_seconds": _seconds_between(event["timestamp"], event["running_max"]),
            "late_span": [first_late, last_late],
            "late_lines_repeated_verbatim": repeats,
            "late_lines_not_seen_before": len(late_lines) - repeats,
            "lookback_complete": len(before_blob) < lookback_bytes or lo == 0,
            "context_before": [ln.decode("latin-1") for ln in before[-context_lines:]],
            "context_after": [ln.decode("latin-1") for ln in after[:context_lines]],
        })

    backward = [r["backward_seconds"] for r in rows if r["backward_seconds"] is not None]
    total_late = sum(r["late_run_lines"] for r in rows)
    total_repeat = sum(r["late_lines_repeated_verbatim"] for r in rows)
    by_month: dict[str, int] = {}
    for r in rows:
        by_month[r["timestamp"][:6]] = by_month.get(r["timestamp"][:6], 0) + 1
    return {
        "generated_utc": utc_now_iso(),
        "source_files": [f.path for f in index.files],
        "raw_fingerprint": index.raw_fingerprint,
        "summary": {
            **{k: v for k, v in ordering.items() if k not in ("events", "definitions")},
            "events_explained": len(rows),
            "late_lines_in_explained_events": total_late,
            "late_lines_repeated_verbatim": total_repeat,
            "late_lines_not_seen_before": total_late - total_repeat,
            "backward_seconds_min": min(backward) if backward else None,
            "backward_seconds_median": float(np.median(backward)) if backward else None,
            "backward_seconds_max": max(backward) if backward else None,
            "events_crossing_partition": sum(1 for r in rows if r["crosses_partition"]),
            "events_by_year_month": dict(sorted(by_month.items())),
            "first_event_timestamp": rows[0]["timestamp"] if rows else None,
            "last_event_timestamp": rows[-1]["timestamp"] if rows else None,
        },
        "definitions": ordering.get("definitions", {}),
        "events": rows,
    }


def _seconds_between(earlier: str, later: str) -> float | None:
    """Seconds from *earlier* to *later* for ``YYYYMMDD HH:MM:SS[.fff]`` text."""
    from datetime import datetime

    def parse(text: str) -> datetime | None:
        for fmt in ("%Y%m%d %H:%M:%S.%f", "%Y%m%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f",
                    "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(text.strip(), fmt)
            except ValueError:
                continue
        return None

    a, b = parse(earlier), parse(later)
    return (b - a).total_seconds() if a is not None and b is not None else None
