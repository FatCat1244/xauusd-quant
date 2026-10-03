"""Bounded column reads, timestamp provenance checks and incremental audit records."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TextIO, cast
from zoneinfo import ZoneInfo

import polars as pl
import pyarrow.parquet as pq

from ..data.timezones import to_utc_expr
from ..utils.config import Config
from .config import ExecutionConfig
from .engine import BarClose, Quote
from .policy import Forecast, utc_time


def local_time(stamp: datetime, config: Config) -> datetime:
    tz = config.timezone
    stamp = utc_time(stamp)
    if tz.mode == "anchored_dst":
        return stamp.astimezone(ZoneInfo(tz.anchor_tz)).replace(tzinfo=None) + timedelta(
            hours=tz.offset_hours
        )
    if tz.mode == "fixed_offset":
        return stamp.replace(tzinfo=None) + timedelta(hours=tz.fixed_offset_hours)
    if tz.mode == "iana" and tz.iana_tz:
        return stamp.astimezone(ZoneInfo(tz.iana_tz)).replace(tzinfo=None)
    raise ValueError("execution requires a documented, determined source timezone")


def validate_interval(config: Config, start: datetime, end: datetime) -> tuple[datetime, datetime]:
    a, b = utc_time(start), utc_time(end)
    if b <= a:
        raise ValueError("execution interval must have start < end")
    # No new reserved-period access path: Stage 12 consumes development forecasts.
    if local_time(b, config) > datetime(2022, 1, 1):
        raise ValueError("Stage 12 development diagnostic refuses 2022+; no new final-test access")
    return local_time(a, config), local_time(b, config)


def tick_quotes(
    config: Config, execution: ExecutionConfig, start: datetime, end: datetime
) -> Iterator[Quote]:
    lower, upper = validate_interval(config, start, end)
    manifest = json.loads(
        (config.processed_data_path / "_manifest.json").read_text(encoding="utf-8")
    )
    sequence = 0
    for entry in sorted(manifest["partitions"], key=lambda p: (p["year"], p["month"])):
        if (
            datetime.fromisoformat(entry["last_timestamp"]) < lower
            or datetime.fromisoformat(entry["first_timestamp"]) >= upper
        ):
            continue
        path = config.processed_data_path / entry["path"]
        if not path.resolve().is_relative_to(config.processed_data_path.resolve()):
            raise ValueError("tick path escapes dataset")
        columns = ["timestamp", "timestamp_utc", "bid", "ask"]
        parquet = pq.ParquetFile(path)
        if not set(columns) <= set(parquet.schema_arrow.names):
            raise ValueError("tick data lack source timestamp, derived UTC, Bid or Ask")
        column_index = parquet.schema_arrow.names.index("timestamp")
        row_groups = []
        for group in range(parquet.num_row_groups):
            stats = parquet.metadata.row_group(group).column(column_index).statistics
            if stats is None or not stats.has_min_max or (stats.max >= lower and stats.min < upper):
                row_groups.append(group)
        for batch in parquet.iter_batches(
            batch_size=execution.batch_rows, columns=columns, row_groups=row_groups
        ):
            frame = cast(pl.DataFrame, pl.from_arrow(batch))
            expected = frame.select(to_utc_expr(config.timezone))["timestamp_utc"]
            if expected.null_count() or not frame["timestamp_utc"].equals(expected):
                raise ValueError("stored UTC differs from configured broker timestamp provenance")
            # Tick values aren't regression features; the bounded partition batch is intentional.
            for local, stamp, bid, ask in frame.iter_rows():
                if start <= stamp < end:
                    yield Quote(
                        stamp,
                        local,
                        float(bid) if bid is not None else math.nan,
                        float(ask) if ask is not None else math.nan,
                        sequence,
                    )
                    sequence += 1


def bar_opens(
    config: Config, execution: ExecutionConfig, start: datetime, end: datetime
) -> Iterator[tuple[datetime, datetime, int]]:
    """Timestamp-only per-year scans; no future bar prices or labels are materialized."""
    lower, upper = validate_interval(config, start, end)
    lower -= timedelta(seconds=execution.bar_seconds, milliseconds=execution.computation_delay_ms)
    directory = config.bars_dir(execution.timeframe)
    manifest = json.loads((directory / "_manifest.json").read_text(encoding="utf-8"))
    tick_manifest = json.loads(
        (config.processed_data_path / "_manifest.json").read_text(encoding="utf-8")
    )
    if (
        manifest.get("tick_dataset_version") != tick_manifest.get("dataset_version")
        or manifest.get("status") != "complete"
        or manifest.get("convention")
        != {"label": "open", "closed": "left", "price_source": "mid", "time_basis": "timestamp"}
    ):
        raise ValueError("bar lineage or opening-time convention is invalid")
    index = 0
    for file in sorted(directory.glob("year=*/*.parquet")):
        year = int(file.parent.name.split("=")[1])
        if year >= 2022 or year < lower.year or year > upper.year:
            continue
        frame = (
            pl.scan_parquet(file)
            .select("timestamp")
            .filter((pl.col("timestamp") >= lower) & (pl.col("timestamp") < upper))
            .collect()
        )
        frame = frame.with_columns(to_utc_expr(config.timezone))
        for local, stamp in frame.iter_rows():
            if stamp is None:
                raise ValueError("unknown bar UTC availability")
            yield local, stamp, index
            index += 1


def bar_closes(
    config: Config, execution: ExecutionConfig, start: datetime, end: datetime
) -> Iterator[BarClose]:
    for _, stamp, index in bar_opens(config, execution, start, end):
        close = stamp + timedelta(seconds=execution.bar_seconds)
        if close <= end:
            yield BarClose(close, index)


def forecast_records(
    path: Path,
    metadata: dict[str, Any],
    config: Config,
    execution: ExecutionConfig,
    start: datetime,
    end: datetime,
) -> Iterator[Forecast]:
    lower, upper = validate_interval(config, start, end)
    lower -= timedelta(seconds=execution.bar_seconds, milliseconds=execution.computation_delay_ms)
    wanted = ["timestamp", execution.policy_field, "status_expected_return_vol_scaled"]
    schema = pq.read_schema(path)
    if not set(wanted) <= set(schema.names):
        raise ValueError(f"forecast table must contain {wanted}")
    bar_iter = iter(bar_opens(config, execution, start, end))
    bar = next(bar_iter, None)
    # Filter BEFORE collecting forecasts; only the separate expected-return contract is read.
    frame = (
        pl.scan_parquet(path)
        .select(wanted)
        .filter(
            (pl.col("timestamp") >= lower)
            & (pl.col("timestamp") < upper)
            & (pl.col("timestamp") < datetime(2022, 1, 1))
        )
        .collect()
    )
    if not frame["timestamp"].is_sorted() or frame["timestamp"].n_unique() != frame.height:
        raise ValueError("forecast timestamps must be ordered and unique")
    for local, value, status in frame.iter_rows():
        while bar is not None and bar[0] < local:
            bar = next(bar_iter, None)
        if bar is None or bar[0] != local:
            raise ValueError("forecast bar not found in verified bar timeline")
        f = Forecast.from_bar(
            forecast_id=f"{metadata['forecast_id']}:{local.isoformat()}",
            bar_open_utc=bar[1],
            bar_index=bar[2],
            value=value,
            config=execution,
            status=status,
            provenance_id=metadata["forecast_id"],
        )
        if start <= f.available_at_utc < end:
            yield f


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def write_new_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n")


class DiskRecorder:
    """Append structured rows as events occur; one interrupted event is the maximum loss."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.stack = ExitStack()
        self.streams: dict[str, TextIO] = {}
        self.periods: dict[str, dict[str, float | int]] = {}

    def __enter__(self) -> DiskRecorder:
        return self

    def __exit__(self, *args: Any) -> None:
        self.stack.close()

    def __call__(self, table: str, row: dict[str, Any]) -> None:
        if table == "trades":
            period = row["exit_utc"].strftime("%Y-%m")
            totals = self.periods.setdefault(
                period,
                {"closed_positions": 0, "gross_price_pnl_account": 0.0, "net_pnl_account": 0.0},
            )
            totals["closed_positions"] += 1
            totals["gross_price_pnl_account"] += row["gross_price_pnl_account"]
            totals["net_pnl_account"] += row["net_pnl_account"]
        if table not in self.streams:
            self.streams[table] = self.stack.enter_context(
                (self.directory / f"{table}.jsonl").open("x", encoding="utf-8", newline="\n")
            )
        stream = self.streams[table]
        stream.write(json.dumps(_json_safe(row), allow_nan=False) + "\n")
        stream.flush()


def parse_utc(text: str) -> datetime:
    return utc_time(datetime.fromisoformat(text.replace("Z", "+00:00"))).astimezone(UTC)
