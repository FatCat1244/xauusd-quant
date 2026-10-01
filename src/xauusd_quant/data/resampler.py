"""Tick-to-bar resampling.

No-look-ahead guarantees
------------------------
A bar aggregates ticks from a half-open interval ``[t, t + Δ)`` (``closed:
left``). A tick landing exactly on the boundary belongs to the *next* bar, so
no bar can contain information that did not exist by its own close. Every
aggregate - OHLC, tick count, spread statistics, bid/ask edges - is computed
from that interval alone. Nothing is shifted, filled or carried across bars.

Bars with no ticks are simply absent. Gaps are real information about the
market and about the data, and :mod:`xauusd_quant.data.diagnostics` reports
them rather than papering over them.

Memory
------
Bars are built one Parquet partition (one month) at a time, so a full
23-year 1-minute build never holds more than a month of ticks. This is only valid
because every supported timeframe divides a day evenly - a bar can therefore
never straddle a month boundary. :func:`validate_timeframes` enforces that.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, cast

import polars as pl
import pyarrow.parquet as pq

from ..utils.clock import utc_now_iso
from ..utils.config import Config
from ..utils.logging import format_duration, get_logger
from ..utils.paths import atomic_write_text, dir_size_bytes, ensure_dir, human_bytes
from .converter import load_dataset_manifest
from .schema import ASK, BID, MID, SCHEMA_VERSION, SPREAD, TIMESTAMP, VOLUME

__all__ = [
    "BAR_COLUMNS",
    "BAR_MANIFEST",
    "BarBuildResult",
    "BarResampler",
    "bar_dataset_version",
    "load_bar_manifest",
    "parse_timeframe",
    "resample_ticks",
    "validate_timeframes",
]

LOGGER = get_logger("data.resampler")

_TIMEFRAME_RE = re.compile(r"^(\d+)(ms|s|m|h|d)$", re.IGNORECASE)
_UNIT_SECONDS = {"ms": 0.001, "s": 1, "m": 60, "h": 3600, "d": 86400}

#: Canonical bar column order.
BAR_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "open", "high", "low", "close",
    "tick_count", "volume",
    "mean_spread", "median_spread", "min_spread", "max_spread",
    "first_bid", "first_ask", "last_bid", "last_ask",
    "first_tick_timestamp", "last_tick_timestamp",
)


def parse_timeframe(timeframe: str) -> timedelta:
    """Parse ``"5m"``/``"1h"``/``"30s"`` into a :class:`~datetime.timedelta`."""
    match = _TIMEFRAME_RE.match(timeframe.strip())
    if not match:
        raise ValueError(
            f"Unrecognised timeframe {timeframe!r}. Use forms like '1m', '5m', '1h', '1d'."
        )
    amount, unit = int(match.group(1)), match.group(2).lower()
    if amount <= 0:
        raise ValueError(f"Timeframe {timeframe!r} must be positive.")
    return timedelta(seconds=amount * _UNIT_SECONDS[unit])


def validate_timeframes(timeframes: tuple[str, ...] | list[str]) -> None:
    """Reject timeframes that would make per-month building incorrect.

    A timeframe that does not divide a day evenly produces bars that can
    straddle a month boundary, which the partition-at-a-time build would split
    into two partial bars.
    """
    day = 86400
    for timeframe in timeframes:
        seconds = parse_timeframe(timeframe).total_seconds()
        if seconds > day or day % seconds != 0:
            raise ValueError(
                f"Timeframe {timeframe!r} does not divide a day evenly. Bars would "
                "straddle month partitions and be split. Supported timeframes must "
                "satisfy 86400 % seconds == 0."
            )


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------
def resample_ticks(
    ticks: pl.DataFrame,
    timeframe: str,
    *,
    price_source: str = "mid",
    label: str = "open",
    closed: str = "left",
    time_basis: str = TIMESTAMP,
    min_ticks_per_bar: int = 1,
    include_bid_ask_edges: bool = True,
) -> pl.DataFrame:
    """Aggregate canonical ticks into OHLC bars for one timeframe.

    Parameters
    ----------
    ticks:
        Canonical tick frame. Must contain *time_basis* and the price column
        implied by *price_source*.
    label:
        ``"open"`` stamps each bar with the start of its interval (the
        default), ``"close"`` with the end.
    closed:
        ``"left"`` makes intervals ``[t, t+Δ)``. This is what prevents a tick
        on the boundary from leaking backwards into the previous bar.

    Returns
    -------
    A frame with :data:`BAR_COLUMNS`, sorted by timestamp, containing only
    intervals that actually had ticks.
    """
    if time_basis not in ticks.columns:
        raise KeyError(f"Tick frame has no column {time_basis!r}; columns: {ticks.columns}")
    price_column = {"mid": MID, "bid": BID, "ask": ASK}[price_source]
    if price_column not in ticks.columns:
        raise KeyError(
            f"price_source={price_source!r} needs column {price_column!r}; "
            f"available: {ticks.columns}"
        )

    every = _to_polars_interval(parse_timeframe(timeframe))
    # Validated by ResamplingConfig; narrowed here for the Polars signature.
    closed_side = cast(Literal["left", "right", "both", "none"], closed)
    ordered = ticks.sort(time_basis, maintain_order=True)

    aggregations: list[pl.Expr] = [
        pl.col(price_column).first().alias("open"),
        pl.col(price_column).max().alias("high"),
        pl.col(price_column).min().alias("low"),
        pl.col(price_column).last().alias("close"),
        pl.len().alias("tick_count"),
        pl.col(time_basis).first().alias("first_tick_timestamp"),
        pl.col(time_basis).last().alias("last_tick_timestamp"),
    ]
    if VOLUME in ordered.columns:
        aggregations.append(pl.col(VOLUME).sum().alias("volume"))
    if SPREAD in ordered.columns:
        aggregations += [
            pl.col(SPREAD).mean().alias("mean_spread"),
            pl.col(SPREAD).median().alias("median_spread"),
            pl.col(SPREAD).min().alias("min_spread"),
            pl.col(SPREAD).max().alias("max_spread"),
        ]
    if include_bid_ask_edges and {BID, ASK} <= set(ordered.columns):
        aggregations += [
            pl.col(BID).first().alias("first_bid"),
            pl.col(ASK).first().alias("first_ask"),
            pl.col(BID).last().alias("last_bid"),
            pl.col(ASK).last().alias("last_ask"),
        ]

    bars = (
        ordered.group_by_dynamic(
            time_basis,
            every=every,
            period=every,
            closed=closed_side,
            label="left",          # always aggregate on interval start...
            start_by="window",
            include_boundaries=False,
        )
        .agg(aggregations)
    )
    if time_basis != "timestamp":
        bars = bars.rename({time_basis: "timestamp"})

    if min_ticks_per_bar > 1:
        bars = bars.filter(pl.col("tick_count") >= min_ticks_per_bar)

    if label == "close":
        # ...then restamp, so the convention change cannot alter membership.
        bars = bars.with_columns(
            (pl.col("timestamp") + pl.duration(microseconds=_micros(parse_timeframe(timeframe))))
            .alias("timestamp")
        )

    present = [c for c in BAR_COLUMNS if c in bars.columns]
    return bars.select(present).sort("timestamp")


def _to_polars_interval(delta: timedelta) -> str:
    """Render a timedelta as a Polars duration string."""
    micros = _micros(delta)
    if micros % 1_000_000 == 0:
        return f"{micros // 1_000_000}s"
    return f"{micros}us"


def _micros(delta: timedelta) -> int:
    return int(round(delta.total_seconds() * 1_000_000))


# ---------------------------------------------------------------------------
# Partition-at-a-time builder
# ---------------------------------------------------------------------------
@dataclass
class BarBuildResult:
    """Outcome of building one timeframe across the whole dataset."""

    timeframe: str
    bars: int = 0
    ticks_read: int = 0
    files_written: list[str] = field(default_factory=list)
    partitions_read: int = 0
    output_bytes: int = 0
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    duration_seconds: float = 0.0
    label: str = "open"
    closed: str = "left"
    price_source: str = "mid"
    time_basis: str = TIMESTAMP
    schema_version: str = SCHEMA_VERSION
    config_fingerprint: str = ""
    tick_dataset_version: str | None = None
    bar_dataset_version: str | None = None
    skipped: bool = False
    duplicate_timestamps: int = 0
    bars_per_year: dict[str, int] = field(default_factory=dict)
    generated_utc: str = field(
        default_factory=utc_now_iso
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "output_bytes_human": human_bytes(self.output_bytes),
            "mean_ticks_per_bar": self.ticks_read / self.bars if self.bars else None,
            "bar_timestamp_convention": (
                f"{self.label} of interval, intervals are "
                f"{'[t, t+D)' if self.closed == 'left' else '(t, t+D]'}"
            ),
        }


class BarResampler:
    """Builds every configured timeframe from the partitioned tick dataset."""

    def __init__(self, config: Config) -> None:
        self.config = config
        validate_timeframes(config.resampling.timeframes)

    def tick_partitions(self) -> list[Path]:
        """Tick files recorded in the dataset manifest, in chronological order.

        Only files the manifest vouches for are read, so a stray or half-written
        file in the tick directory can never leak into the bars. A tick
        directory without a manifest (written by hand or by an old converter)
        falls back to every ``*.parquet`` file, in path order.
        """
        root = self.config.processed_data_path
        if not root.exists():
            return []
        manifest = load_dataset_manifest(root)
        if manifest is None:
            return sorted(root.rglob("*.parquet"))
        return [
            root / entry["path"]
            for entry in sorted(manifest.get("partitions", []), key=lambda e: e["key"])
            if entry.get("path") and entry.get("rows")
        ]

    def build(
        self, timeframe: str, *, overwrite: bool = True, allow_incomplete: bool = False
    ) -> BarBuildResult:
        """Build one timeframe, one tick partition at a time, transactionally.

        The bars are written to ``<timeframe>.building``, verified (sorted,
        unique timestamps, row counts, a content digest per year file) and
        only then swapped into place, so an interrupted build never leaves a
        half-written timeframe where a complete one used to be.

        ``overwrite=False`` skips the build when the existing bars were made
        from the current tick dataset with the current settings.
        ``allow_incomplete`` permits bars from a tick dataset whose manifest is
        not complete - for development only; such bars describe part of history.
        """
        started = time.perf_counter()
        rs = self.config.resampling
        result = BarBuildResult(
            timeframe=timeframe,
            label=rs.label,
            closed=rs.closed,
            price_source=rs.price_source,
            time_basis=rs.time_basis,
            config_fingerprint=self.config.fingerprint(),
        )

        partitions = self.tick_partitions()
        if not partitions:
            raise FileNotFoundError(
                f"No tick Parquet found under {self.config.processed_data_path}. "
                "Run `xq convert` first."
            )
        tick_manifest = load_dataset_manifest(self.config.processed_data_path)
        status = tick_manifest.get("status") if tick_manifest else None
        if status != "complete" and not allow_incomplete:
            raise RuntimeError(
                f"The tick dataset is {status or 'unverified (no manifest)'}, not complete. "
                "Bars built from it would describe only part of history. Run "
                "`xq convert` (and `xq verify-dataset`) first."
            )
        result.tick_dataset_version = (tick_manifest or {}).get("dataset_version")

        out_dir = self.config.bars_dir(timeframe)
        if not overwrite:
            current = load_bar_manifest(out_dir)
            if (
                current is not None
                and current.get("status") == "complete"
                and current.get("tick_dataset_version") == result.tick_dataset_version
                and current.get("bar_fingerprint") == self.config.bar_fingerprint()
            ):
                LOGGER.info("%s bars are up to date (%s); skipped", timeframe,
                            current.get("bar_dataset_version"))
                result.skipped = True
                result.bar_dataset_version = current.get("bar_dataset_version")
                result.bars = int(current.get("rows") or 0)
                return result

        building = out_dir.with_name(out_dir.name + ".building")
        _remove_bar_output(building)
        ensure_dir(building)
        LOGGER.info(
            "Building %s bars from %d tick partition(s) -> %s",
            timeframe, len(partitions), out_dir,
        )

        writers: dict[int, pq.ParquetWriter] = {}
        needed = self._required_columns(partitions[0])
        try:
            for partition in partitions:
                frame = pl.read_parquet(partition, columns=needed)
                if frame.is_empty():
                    continue
                result.partitions_read += 1
                result.ticks_read += frame.height

                bars = resample_ticks(
                    frame,
                    timeframe,
                    price_source=rs.price_source,
                    label=rs.label,
                    closed=rs.closed,
                    time_basis=rs.time_basis,
                    min_ticks_per_bar=rs.min_ticks_per_bar,
                    include_bid_ask_edges=rs.include_bid_ask_edges,
                )
                del frame
                if bars.is_empty():
                    continue

                result.bars += bars.height
                span = bars.select(
                    pl.col("timestamp").min().alias("lo"), pl.col("timestamp").max().alias("hi")
                ).row(0)
                if result.first_timestamp is None:
                    result.first_timestamp = span[0].isoformat(sep=" ")
                result.last_timestamp = span[1].isoformat(sep=" ")

                self._write_bars(bars, timeframe, building, writers, result)
        finally:
            for writer in writers.values():
                writer.close()

        manifest = self._verify_and_describe(building, timeframe, result)
        _swap_directory(building, out_dir)
        result.files_written = [
            str(out_dir / Path(f).relative_to(building)) for f in result.files_written
        ]
        result.output_bytes = dir_size_bytes(out_dir, "**/*.parquet")
        result.duration_seconds = time.perf_counter() - started
        LOGGER.info(
            "%s bars: %s bars from %s ticks in %s (%s), version %s",
            timeframe, f"{result.bars:,}", f"{result.ticks_read:,}",
            format_duration(result.duration_seconds), human_bytes(result.output_bytes),
            manifest["bar_dataset_version"],
        )
        return result

    def _verify_and_describe(
        self, directory: Path, timeframe: str, result: BarBuildResult
    ) -> dict[str, Any]:
        """Check what was written, then record it in ``_manifest.json``."""
        from .converter import _PartitionDigest

        years: list[dict[str, Any]] = []
        previous_last = None
        duplicates = 0
        total = 0
        for path in sorted(directory.rglob("*.parquet")):
            frame = pl.read_parquet(path)
            stamps = frame["timestamp"]
            if not stamps.is_sorted():
                raise RuntimeError(f"{path.name}: bars are not sorted by timestamp")
            duplicates += frame.height - stamps.n_unique()
            if previous_last is not None and stamps[0] <= previous_last:
                raise RuntimeError(f"{path.name}: overlaps the previous year's bars")
            previous_last = stamps[-1]
            digest = _PartitionDigest()
            digest.update(frame)
            year = int(stamps[0].year)
            total += frame.height
            result.bars_per_year[str(year)] = frame.height
            years.append({
                "year": year,
                "path": path.relative_to(directory).as_posix(),
                "rows": frame.height,
                "first_timestamp": stamps[0].isoformat(sep=" "),
                "last_timestamp": stamps[-1].isoformat(sep=" "),
                "bytes": path.stat().st_size,
                "digest": digest.hexdigest(frame.columns),
            })
        if total != result.bars:
            raise RuntimeError(f"{timeframe}: wrote {result.bars:,} bars but read back {total:,}")
        if duplicates:
            raise RuntimeError(f"{timeframe}: {duplicates} duplicate bar timestamp(s)")
        result.duplicate_timestamps = duplicates

        hasher = hashlib.blake2b(digest_size=8)
        hasher.update(f"{timeframe}|{result.tick_dataset_version}|".encode())
        hasher.update(self.config.bar_fingerprint().encode())
        for entry in years:
            hasher.update(f"{entry['year']}|{entry['rows']}|{entry['digest']}".encode())
        result.bar_dataset_version = f"bars-{timeframe}-{hasher.hexdigest()}"
        manifest = {
            "timeframe": timeframe,
            "status": "complete",
            "bar_dataset_version": result.bar_dataset_version,
            "tick_dataset_version": result.tick_dataset_version,
            "bar_fingerprint": self.config.bar_fingerprint(),
            "config_fingerprint": self.config.fingerprint(),
            "schema_version": SCHEMA_VERSION,
            "rows": total,
            "ticks_read": result.ticks_read,
            "partitions_read": result.partitions_read,
            "first_timestamp": result.first_timestamp,
            "last_timestamp": result.last_timestamp,
            "checks": {"sorted": True, "duplicate_timestamps": 0, "years_overlap": False},
            "convention": {
                "label": result.label, "closed": result.closed,
                "price_source": result.price_source, "time_basis": result.time_basis,
            },
            "years": years,
            "generated_utc": utc_now_iso(),
        }
        atomic_write_text(directory / BAR_MANIFEST, json.dumps(manifest, indent=1) + "\n")
        return manifest

    def build_all(self, timeframes: list[str] | None = None) -> dict[str, BarBuildResult]:
        """Build every configured timeframe."""
        selected = timeframes or list(self.config.resampling.timeframes)
        validate_timeframes(selected)
        return {tf: self.build(tf) for tf in selected}

    # -- internals ---------------------------------------------------------
    def _required_columns(self, sample: Path) -> list[str]:
        """Only read the tick columns the bar aggregation actually needs."""
        rs = self.config.resampling
        wanted = {rs.time_basis, {"mid": MID, "bid": BID, "ask": ASK}[rs.price_source]}
        wanted |= {SPREAD, VOLUME}
        if rs.include_bid_ask_edges:
            wanted |= {BID, ASK}
        available = set(pq.read_schema(sample).names)
        missing = wanted - available - {SPREAD, VOLUME}
        if missing:
            raise KeyError(
                f"Tick dataset {sample} lacks required column(s) {sorted(missing)}; "
                "re-run `xq convert` with a matching configuration."
            )
        return [c for c in (TIMESTAMP, "timestamp_utc", BID, ASK, MID, SPREAD, VOLUME)
                if c in wanted and c in available]

    def _write_bars(
        self,
        bars: pl.DataFrame,
        timeframe: str,
        out_dir: Path,
        writers: dict[int, pq.ParquetWriter],
        result: BarBuildResult,
    ) -> None:
        """Append bars to the per-year file, opening writers on demand."""
        rs = self.config.resampling
        pq_cfg = self.config.parquet
        tagged = bars.with_columns(pl.col("timestamp").dt.year().alias("__year"))
        for (raw_year,), chunk in tagged.group_by(["__year"], maintain_order=True):
            year = int(raw_year)
            table = chunk.drop("__year").to_arrow()
            writer = writers.get(year)
            if writer is None:
                directory = ensure_dir(out_dir / f"year={year:04d}")
                filename = rs.filename_template.format(timeframe=timeframe, year=f"{year:04d}")
                path = directory / filename
                writer = writers[year] = pq.ParquetWriter(
                    path,
                    table.schema.with_metadata(self._bar_metadata(timeframe)),
                    compression=pq_cfg.compression,
                    compression_level=pq_cfg.compression_level
                    if pq_cfg.compression == "zstd"
                    else None,
                    write_statistics=pq_cfg.write_statistics,
                )
                result.files_written.append(str(path))
            writer.write_table(table.cast(writer.schema), row_group_size=pq_cfg.row_group_rows)

    def _bar_metadata(self, timeframe: str) -> dict[bytes, bytes]:
        """Provenance stored in each bar file, so conventions travel with data."""
        rs = self.config.resampling
        interval = "[t, t+D)" if rs.closed == "left" else "(t, t+D]"
        return {
            b"xq_schema_version": SCHEMA_VERSION.encode(),
            b"xq_instrument": self.config.instrument.encode(),
            b"xq_timeframe": timeframe.encode(),
            b"xq_price_source": rs.price_source.encode(),
            b"xq_bar_label": rs.label.encode(),
            b"xq_bar_interval": interval.encode(),
            b"xq_time_basis": rs.time_basis.encode(),
            b"xq_timezone_mode": self.config.timezone.mode.encode(),
            b"xq_timezone_description": self.config.timezone.describe().encode(),
            b"xq_config_fingerprint": self.config.fingerprint().encode(),
            b"xq_bar_fingerprint": self.config.bar_fingerprint().encode(),
            b"xq_tick_dataset_version": str(self._tick_version()).encode(),
        }

    def _tick_version(self) -> str | None:
        manifest = load_dataset_manifest(self.config.processed_data_path)
        return manifest.get("dataset_version") if manifest else None


# ---------------------------------------------------------------------------
# Bar manifests and the transactional swap
# ---------------------------------------------------------------------------
BAR_MANIFEST = "_manifest.json"


def load_bar_manifest(directory: Path) -> dict[str, Any] | None:
    """The manifest of one timeframe's bars, or None."""
    path = Path(directory) / BAR_MANIFEST
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def bar_dataset_version(config: Config, timeframe: str) -> str | None:
    """Version of the bars currently on disk for *timeframe*, if recorded."""
    manifest = load_bar_manifest(config.bars_dir(timeframe))
    return manifest.get("bar_dataset_version") if manifest else None


def _remove_bar_output(root: Path) -> None:
    """Delete bar files and the bar manifest under *root*, then empty directories."""
    if not root.exists():
        return
    for pattern in ("*.parquet", BAR_MANIFEST):
        for path in root.rglob(pattern):
            path.unlink(missing_ok=True)
    for directory in sorted(
        (d for d in root.rglob("*") if d.is_dir()), key=lambda d: len(d.parts), reverse=True
    ):
        if not any(directory.iterdir()):
            directory.rmdir()
    if root.exists() and not any(root.iterdir()):
        root.rmdir()


def _swap_directory(building: Path, target: Path) -> None:
    """Move a verified build into place; the old one is removed only afterwards."""
    previous = target.with_name(target.name + ".previous")
    _remove_bar_output(previous)
    had_target = target.exists()
    if had_target:
        target.replace(previous)
    try:
        building.replace(target)
    except OSError:
        if had_target:
            previous.replace(target)
        raise
    if had_target:
        for item in list(previous.iterdir()):
            keep = not (item.name.endswith(".parquet") or item.name == BAR_MANIFEST
                        or item.name.startswith("year="))
            if keep and not (target / item.name).exists():
                item.replace(target / item.name)
        _remove_bar_output(previous)
