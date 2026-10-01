"""DuckDB-backed access to the Parquet datasets.

The point of this layer is to make range queries the easy path and full-dataset
loads the hard one:

* ``start`` and ``end`` are **required** on every loader, so no call can
  accidentally pull ~730 million ticks into RAM.
* A row-count guard (``duckdb.max_rows_without_override``) is checked before
  materialising anything; exceeding it raises unless the caller opts in.
* ``scan_ticks``/``iter_ticks`` return lazy or chunked results for genuinely
  large ranges.

Hive partition pruning is explicit: the generated SQL constrains ``year`` and
``month`` as well as ``timestamp``, so DuckDB skips whole files rather than
reading and filtering them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from typing import Any

import duckdb
import polars as pl

from ..utils.config import Config
from ..utils.logging import get_logger
from .resampler import parse_timeframe

__all__ = ["DataStore", "TimeRangeError", "TooManyRowsError"]

LOGGER = get_logger("data.loader")

TimeLike = str | date | datetime


class TimeRangeError(ValueError):
    """Raised when a requested range is missing, malformed or inverted."""


class TooManyRowsError(RuntimeError):
    """Raised when a query would materialise more rows than the configured cap."""


def _coerce(value: TimeLike, *, what: str) -> datetime:
    """Accept ISO strings, dates and datetimes; reject anything else clearly."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, str):
        text = value.strip().replace("/", "-")
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
                    "%Y-%m-%d", "%Y-%m", "%Y"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
    raise TimeRangeError(
        f"{what}={value!r} is not a recognised date/time. "
        "Use 'YYYY-MM-DD', 'YYYY-MM-DD HH:MM:SS' or a datetime object."
    )


class DataStore:
    """Read-only query interface over the tick and bar Parquet datasets."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._conn: duckdb.DuckDBPyConnection | None = None

    # -- connection --------------------------------------------------------
    def connect(self) -> duckdb.DuckDBPyConnection:
        """Return the shared in-process DuckDB connection, creating it lazily."""
        if self._conn is None:
            conn = duckdb.connect(database=":memory:")
            cfg = self.config.duckdb
            if cfg.threads:
                conn.execute(f"SET threads = {int(cfg.threads)}")
            if cfg.memory_limit:
                conn.execute(f"SET memory_limit = '{cfg.memory_limit}'")
            # `timestamp_utc` is stored as TIMESTAMP WITH TIME ZONE. Without
            # this, DuckDB renders it in the machine's local zone, so the same
            # query would print different-looking times on different machines.
            conn.execute("SET TimeZone = 'UTC'")
            self._conn = conn
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> DataStore:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- dataset discovery -------------------------------------------------
    def tick_glob(self) -> str:
        return (self.config.processed_data_path / "**" / "*.parquet").as_posix()

    def bar_glob(self, timeframe: str) -> str:
        return (self.config.bars_dir(timeframe) / "**" / "*.parquet").as_posix()

    def available_timeframes(self) -> list[str]:
        """Timeframes that actually have bar files on disk."""
        root = self.config.bars_path
        if not root.exists():
            return []
        return sorted(
            d.name for d in root.iterdir() if d.is_dir() and any(d.rglob("*.parquet"))
        )

    def has_ticks(self) -> bool:
        root = self.config.processed_data_path
        return root.exists() and any(root.rglob("*.parquet"))

    def tick_span(self) -> tuple[datetime | None, datetime | None]:
        """Min/max tick timestamp, answered from Parquet statistics alone."""
        if not self.has_ticks():
            return None, None
        row = self.connect().execute(
            f"SELECT min(timestamp), max(timestamp) FROM read_parquet('{self.tick_glob()}')"
        ).fetchone()
        return (row[0], row[1]) if row else (None, None)

    # -- loaders -----------------------------------------------------------
    def load_ticks(
        self,
        start: TimeLike,
        end: TimeLike,
        columns: Sequence[str] | None = None,
        *,
        limit: int | None = None,
        allow_large: bool = False,
    ) -> pl.DataFrame:
        """Ticks with ``start <= timestamp < end``.

        The range is half-open so consecutive calls tile without overlap.
        """
        lo, hi = self._range(start, end)
        sql, params = self._tick_query(lo, hi, columns, limit)
        self._guard(lambda: self._count_ticks(lo, hi), limit, allow_large=allow_large)
        return self.connect().execute(sql, params).pl()

    def load_bars(
        self,
        timeframe: str,
        start: TimeLike,
        end: TimeLike,
        columns: Sequence[str] | None = None,
        *,
        limit: int | None = None,
        allow_large: bool = False,
    ) -> pl.DataFrame:
        """Bars of *timeframe* with ``start <= timestamp < end``."""
        parse_timeframe(timeframe)  # fail fast on a typo
        glob = self.bar_glob(timeframe)
        if not any(self.config.bars_dir(timeframe).rglob("*.parquet")):
            available = self.available_timeframes()
            raise FileNotFoundError(
                f"No bars found for timeframe {timeframe!r}. "
                f"Available: {available or 'none - run `xq build-bars --all`'}"
            )
        lo, hi = self._range(start, end)
        projection = "* EXCLUDE (year)" if not columns else self._projection(columns)
        self._guard(lambda: self._count_bars(glob, lo, hi), limit, allow_large=allow_large)
        sql = (
            f"SELECT {projection} FROM read_parquet('{glob}', hive_partitioning=1) "  # noqa: S608
            "WHERE timestamp >= ? AND timestamp < ? ORDER BY timestamp"
        )
        if limit:
            sql += f" LIMIT {int(limit)}"
        return self.connect().execute(sql, [lo, hi]).pl()

    def iter_ticks(
        self,
        start: TimeLike,
        end: TimeLike,
        columns: Sequence[str] | None = None,
        *,
        chunk: timedelta | str = "1d",
    ) -> Iterator[pl.DataFrame]:
        """Yield ticks in time-ordered chunks, for ranges too big to hold at once."""
        lo, hi = self._range(start, end)
        step = parse_timeframe(chunk) if isinstance(chunk, str) else chunk
        if step <= timedelta(0):
            raise TimeRangeError("chunk must be a positive duration")
        cursor = lo
        while cursor < hi:
            upper = min(cursor + step, hi)
            frame = self.load_ticks(cursor, upper, columns, allow_large=True)
            if not frame.is_empty():
                yield frame
            cursor = upper

    def scan_ticks(
        self, start: TimeLike, end: TimeLike, columns: Sequence[str] | None = None
    ) -> pl.LazyFrame:
        """A Polars LazyFrame over the range - nothing is read until collected."""
        lo, hi = self._range(start, end)
        lazy = pl.scan_parquet(
            self.tick_glob(), hive_partitioning=True
        ).filter((pl.col("timestamp") >= lo) & (pl.col("timestamp") < hi))
        return lazy.select(list(columns)) if columns else lazy

    def sql(self, query: str, params: Sequence[Any] | None = None) -> pl.DataFrame:
        """Escape hatch for ad-hoc DuckDB SQL.

        ``ticks`` and each ``bars_<timeframe>`` are registered as views, so a
        query can read ``SELECT * FROM ticks WHERE ...`` directly.
        """
        conn = self.connect()
        if self.has_ticks():
            conn.execute(
                f"CREATE OR REPLACE VIEW ticks AS "  # noqa: S608
                f"SELECT * FROM read_parquet('{self.tick_glob()}', hive_partitioning=1)"
            )
        for timeframe in self.available_timeframes():
            view = f"bars_{timeframe}"
            conn.execute(
                f'CREATE OR REPLACE VIEW "{view}" AS '  # noqa: S608
                f"SELECT * FROM read_parquet('{self.bar_glob(timeframe)}', hive_partitioning=1)"
            )
        return conn.execute(query, list(params or [])).pl()

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _range(start: TimeLike, end: TimeLike) -> tuple[datetime, datetime]:
        lo, hi = _coerce(start, what="start"), _coerce(end, what="end")
        if hi <= lo:
            raise TimeRangeError(f"end ({hi}) must be strictly after start ({lo})")
        return lo, hi

    HIVE_COLUMNS = ("year", "month")

    @classmethod
    def _projection(cls, columns: Sequence[str] | None) -> str:
        """SQL projection, hiding the Hive partition columns by default.

        ``year``/``month`` exist only because of the directory layout. Leaving
        them in would mean ``load_ticks`` returned a different column set from
        the canonical tick schema. They are still selectable by name.
        """
        if not columns:
            excluded = ", ".join(cls.HIVE_COLUMNS)
            return f"* EXCLUDE ({excluded})"
        for name in columns:
            if not name.replace("_", "").isalnum():
                raise ValueError(f"Refusing unsafe column name {name!r}")
        return ", ".join(f'"{c}"' for c in columns)

    def _tick_query(
        self,
        lo: datetime,
        hi: datetime,
        columns: Sequence[str] | None,
        limit: int | None,
    ) -> tuple[str, list[Any]]:
        sql = (
            f"SELECT {self._projection(columns)} "  # noqa: S608
            f"FROM read_parquet('{self.tick_glob()}', hive_partitioning=1) "
            f"WHERE {self._partition_predicate(lo, hi)} "
            "AND timestamp >= ? AND timestamp < ? ORDER BY timestamp"
        )
        if limit:
            sql += f" LIMIT {int(limit)}"
        return sql, [lo, hi]

    @staticmethod
    def _partition_predicate(lo: datetime, hi: datetime) -> str:
        """Constrain the Hive columns so DuckDB prunes whole files.

        The timestamp filter alone would already be correct; this makes the
        pruning explicit and independent of statistics quality.
        """
        months = []
        cursor = datetime(lo.year, lo.month, 1)
        # `hi` is exclusive, but a tick at exactly `hi` would live in hi's
        # month, so include it and let the timestamp predicate do the rest.
        while cursor <= datetime(hi.year, hi.month, 1):
            months.append((cursor.year, cursor.month))
            cursor = (
                datetime(cursor.year + 1, 1, 1)
                if cursor.month == 12
                else datetime(cursor.year, cursor.month + 1, 1)
            )
        # Hive values are typed from their text, so `month=01` arrives as
        # VARCHAR while `year=2021` arrives as BIGINT. Cast both so the
        # comparison cannot silently match nothing.
        if len(months) > 64:  # a long range: year-level pruning is enough
            years = ", ".join(str(y) for y in sorted({y for y, _ in months}))
            return f"CAST(year AS INTEGER) IN ({years})"
        pairs = ", ".join(f"({y}, {m})" for y, m in months)
        return f"(CAST(year AS INTEGER), CAST(month AS INTEGER)) IN ({pairs})"

    def _count_ticks(self, lo: datetime, hi: datetime) -> int:
        row = self.connect().execute(
            f"SELECT count(*) FROM read_parquet('{self.tick_glob()}', hive_partitioning=1) "  # noqa: S608
            f"WHERE {self._partition_predicate(lo, hi)} AND timestamp >= ? AND timestamp < ?",
            [lo, hi],
        ).fetchone()
        return int(row[0]) if row else 0

    def _count_bars(self, glob: str, lo: datetime, hi: datetime) -> int:
        row = self.connect().execute(
            f"SELECT count(*) FROM read_parquet('{glob}', hive_partitioning=1) "  # noqa: S608
            "WHERE timestamp >= ? AND timestamp < ?",
            [lo, hi],
        ).fetchone()
        return int(row[0]) if row else 0

    def _guard(
        self, count: Callable[[], int], limit: int | None, *, allow_large: bool
    ) -> None:
        """Refuse queries that would materialise more rows than the cap.

        The row count is only computed when it can actually change the
        outcome, so the guard costs nothing on opted-in or limited queries.
        """
        cap = self.config.duckdb.max_rows_without_override
        if allow_large or (limit is not None and limit <= cap):
            return
        rows = count()
        if rows <= cap:
            return
        raise TooManyRowsError(
            f"This range holds {rows:,} rows, above the configured guard of {cap:,}. "
            "Narrow the range, pass limit=, iterate with iter_ticks(), use "
            "scan_ticks() for a lazy query, or pass allow_large=True if you really "
            "want it all in memory."
        )


@contextmanager
def open_store(config: Config) -> Iterator[DataStore]:
    """Context manager yielding a connected :class:`DataStore`."""
    store = DataStore(config)
    try:
        store.connect()
        yield store
    finally:
        store.close()
