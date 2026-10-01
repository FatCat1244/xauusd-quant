"""Tick validation.

The validator *flags*, it never deletes. It attaches one boolean column per
check to a batch and accumulates exact counters across the whole stream;
:mod:`xauusd_quant.data.cleaner` is the only place that decides what those
flags mean for row retention.

Checks that need to look at the previous row (duplicates, ordering, gaps) carry
state between batches, so results are identical whether the file arrives in one
batch or ten thousand.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Final

import polars as pl

from ..utils.clock import utc_now_iso
from ..utils.config import Config, ValidationConfig
from .schema import ASK, BID, SPREAD, TIMESTAMP, VOLUME

__all__ = [
    "CHECKS",
    "FLAG_PREFIX",
    "GapRecord",
    "StreamValidator",
    "ValidationReport",
    "flag_column",
]

FLAG_PREFIX: Final[str] = "__flag_"

#: Every check, in the order used for drop attribution. Structural problems
#: (a row that cannot be interpreted at all) come first so that a row dropped
#: for being unparseable is not also counted as, say, an extreme spread.
CHECKS: Final[tuple[str, ...]] = (
    "missing_timestamp",
    "unparseable_timestamp",
    "missing_bid_or_ask",
    "non_finite",
    "non_positive_price",
    "price_out_of_range",
    "ask_below_bid",
    "negative_spread",
    "extreme_spread",
    "negative_volume",
    "invalid_volume",
    "exact_duplicate_rows",
    "duplicate_timestamps",
    "non_monotonic",
)

RAW_TIMESTAMP: Final[str] = "__raw_timestamp"


def flag_column(check: str) -> str:
    """Column name holding the boolean result of *check*."""
    return f"{FLAG_PREFIX}{check}"


@dataclass(frozen=True, slots=True)
class GapRecord:
    """A period with no ticks at all, longer than the configured threshold."""

    start: str
    end: str
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    """Exact counters for a validated stream."""

    rows_checked: int = 0
    rows_flagged: int = 0
    malformed_rows: int = 0
    counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(CHECKS, 0))
    examples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    gap_count: int = 0
    largest_gap_seconds: float = 0.0
    gaps: list[GapRecord] = field(default_factory=list)

    first_timestamp: str | None = None
    last_timestamp: str | None = None

    thresholds: dict[str, Any] = field(default_factory=dict)
    generated_utc: str = field(
        default_factory=utc_now_iso
    )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gaps"] = [g.to_dict() if isinstance(g, GapRecord) else g for g in self.gaps]
        payload["clean_rows_estimate"] = self.rows_checked - self.rows_flagged
        return payload

    def summary_line(self) -> str:
        """One-line digest suitable for a log record."""
        problems = {k: v for k, v in self.counts.items() if v}
        return (
            f"{self.rows_checked:,} rows checked, "
            f"{self.malformed_rows:,} malformed, "
            f"{self.gap_count:,} gaps, "
            f"issues={problems or 'none'}"
        )


class StreamValidator:
    """Applies every check to a stream of batches, carrying state between them.

    Usage::

        validator = StreamValidator(config)
        for batch in batches:
            flagged = validator.validate(batch)
        report = validator.report()
    """

    def __init__(
        self,
        config: Config,
        *,
        collect_examples: bool = True,
        duplicate_window: int | None = None,
    ) -> None:
        """*duplicate_window* overrides ``validation.duplicate_window_rows``.

        The converter validates one month at a time and passes the month's
        line count, which makes the repeat check exact within the month.
        """
        self._cfg: ValidationConfig = config.validation
        self._duplicate_window = duplicate_window
        self._has_volume = bool(config.input.volume_column)
        self._has_quotes = bool(config.input.bid_column and config.input.ask_column)
        self._collect_examples = collect_examples
        self._report = ValidationReport(
            thresholds={
                "min_price": self._cfg.min_price,
                "max_price": self._cfg.max_price,
                "max_spread": self._cfg.max_spread,
                "max_volume": self._cfg.max_volume,
                "large_gap_seconds": self._cfg.large_gap_seconds,
            }
        )
        # Carry-over state for cross-batch checks.
        self._last_ts: datetime | None = None
        self._last_row: dict[str, Any] | None = None
        # Rows carried forward so repeated blocks spanning a batch
        # boundary are still detected.
        self._dup_tail: pl.DataFrame | None = None

    # -- public API --------------------------------------------------------
    @property
    def last_timestamp(self) -> datetime | None:
        return self._last_ts

    def note_malformed(self, n: int) -> None:
        """Record rows the CSV parser rejected before they reached a batch."""
        self._report.malformed_rows += n

    def validate(self, batch: pl.DataFrame) -> pl.DataFrame:
        """Return *batch* with one boolean flag column per check appended."""
        if batch.is_empty():
            return batch

        flagged = batch.with_columns(self._flag_expressions(batch))
        flagged = flagged.with_columns(*self._duplicate_flags(batch))
        self._accumulate(flagged)
        self._remember_tail(batch)
        self._update_carry_over(batch)
        return flagged

    def report(self) -> ValidationReport:
        """Finalised counters. Safe to call repeatedly."""
        self._report.gaps.sort(key=lambda g: g.seconds, reverse=True)
        return self._report

    # -- checks ------------------------------------------------------------
    def _flag_expressions(self, batch: pl.DataFrame) -> list[pl.Expr]:
        cfg = self._cfg
        cols = set(batch.columns)
        exprs: list[pl.Expr] = []
        false = pl.lit(False)  # placeholder flag for checks this source cannot support

        # --- timestamp -----------------------------------------------------
        if RAW_TIMESTAMP in cols:
            missing_ts = pl.col(RAW_TIMESTAMP).is_null() | (
                pl.col(RAW_TIMESTAMP).str.strip_chars().str.len_chars() == 0
            )
        else:
            missing_ts = pl.col(TIMESTAMP).is_null()
        exprs.append(missing_ts.alias(flag_column("missing_timestamp")))
        exprs.append(
            (pl.col(TIMESTAMP).is_null() & ~missing_ts).alias(
                flag_column("unparseable_timestamp")
            )
        )

        # --- quotes --------------------------------------------------------
        if self._has_quotes and {BID, ASK} <= cols:
            bid, ask = pl.col(BID), pl.col(ASK)
            missing_quote = bid.is_null() | ask.is_null()
            non_finite = (
                bid.is_infinite() | ask.is_infinite() | bid.is_nan() | ask.is_nan()
            ).fill_null(False)
            exprs += [
                missing_quote.alias(flag_column("missing_bid_or_ask")),
                non_finite.alias(flag_column("non_finite")),
                ((bid <= 0) | (ask <= 0)).fill_null(False).alias(
                    flag_column("non_positive_price")
                ),
                (
                    (bid < cfg.min_price) | (ask > cfg.max_price)
                ).fill_null(False).alias(flag_column("price_out_of_range")),
                (ask < bid).fill_null(False).alias(flag_column("ask_below_bid")),
            ]
            spread = pl.col(SPREAD) if SPREAD in cols else (ask - bid)
            exprs += [
                (spread < 0).fill_null(False).alias(
                    flag_column("negative_spread")
                ),
                (spread > cfg.max_spread).fill_null(False).alias(
                    flag_column("extreme_spread")
                ),
            ]
        else:
            for check in ("missing_bid_or_ask", "non_finite", "non_positive_price",
                          "price_out_of_range", "ask_below_bid", "negative_spread",
                          "extreme_spread"):
                exprs.append(false.alias(flag_column(check)))

        # --- volume --------------------------------------------------------
        if self._has_volume and VOLUME in cols:
            vol = pl.col(VOLUME)
            exprs += [
                (vol < 0).fill_null(False).alias(flag_column("negative_volume")),
                (
                    vol.is_nan() | vol.is_infinite() | (vol > cfg.max_volume)
                ).fill_null(False).alias(flag_column("invalid_volume")),
            ]
        else:
            exprs += [
                false.alias(flag_column("negative_volume")),
                false.alias(flag_column("invalid_volume")),
            ]

        # --- ordering and duplicates (cross-batch aware) -------------------
        # Ordering is a property of adjacent rows, so it stays an expression.
        # The duplicate checks are not - see `_duplicate_flags`.
        prev_ts = self._with_carry(pl.col(TIMESTAMP).shift(1), TIMESTAMP, batch)
        exprs.append(
            (pl.col(TIMESTAMP) < prev_ts).fill_null(False).alias(
                flag_column("non_monotonic")
            )
        )

        return exprs

    def _duplicate_flags(self, batch: pl.DataFrame) -> list[pl.Series]:
        """Mark rows whose timestamp, or whose whole value tuple, was seen before.

        Comparing against the previous row alone is not enough: this export
        repeats a whole hour of ticks at each weekly splice, so the two copies
        sit thousands of rows apart. A window of earlier rows is carried
        forward, which also catches a repeat that straddles a batch boundary.

        Only the *later* copy is flagged, so the first occurrence always
        survives cleaning and the outcome does not depend on batch size.
        """
        keys = [c for c in (TIMESTAMP, BID, ASK, VOLUME) if c in batch.columns]
        if not keys:
            empty = [False] * batch.height
            return [
                pl.Series(flag_column("duplicate_timestamps"), empty),
                pl.Series(flag_column("exact_duplicate_rows"), empty),
            ]

        window = self._dup_tail
        if window is None or window.is_empty():
            combined, offset = batch.select(keys), 0
        else:
            combined = pl.concat([window, batch.select(keys)], how="vertical")
            offset = window.height

        # `is_first_distinct` is True for the first occurrence of each value,
        # so its negation marks every later copy - and only the later copies.
        seen = combined.select(
            (~pl.col(TIMESTAMP).is_first_distinct()).alias("dup_ts"),
            (~pl.struct(keys).is_first_distinct()).alias("dup_row"),
        )
        return [
            seen["dup_ts"].slice(offset, batch.height)
            .rename(flag_column("duplicate_timestamps")),
            seen["dup_row"].slice(offset, batch.height)
            .rename(flag_column("exact_duplicate_rows")),
        ]

    def _remember_tail(self, batch: pl.DataFrame) -> None:
        """Keep the last `duplicate_window_rows` rows for the next batch."""
        keys = [c for c in (TIMESTAMP, BID, ASK, VOLUME) if c in batch.columns]
        if not keys:
            return
        limit = (
            self._duplicate_window if self._duplicate_window is not None
            else self._cfg.duplicate_window_rows
        )
        if limit <= 0:
            self._dup_tail = None
            return
        window = self._dup_tail
        combined = (
            batch.select(keys)
            if window is None or window.is_empty()
            else pl.concat([window, batch.select(keys)], how="vertical")
        )
        self._dup_tail = combined.tail(limit)

    def _with_carry(self, shifted: pl.Expr, column: str, batch: pl.DataFrame) -> pl.Expr:
        """Replace the batch's first shifted value with the carried-over row.

        Without this, the first row of every batch would silently escape the
        duplicate and ordering checks.
        """
        if self._last_row is None or column not in self._last_row:
            return shifted
        carried = pl.lit(self._last_row[column], dtype=batch.schema[column])
        return pl.when(pl.int_range(pl.len(), dtype=pl.UInt32) == 0).then(carried).otherwise(shifted)

    # -- accumulation ------------------------------------------------------
    def _accumulate(self, flagged: pl.DataFrame) -> None:
        report = self._report
        report.rows_checked += flagged.height

        flag_cols = [flag_column(c) for c in CHECKS if flag_column(c) in flagged.columns]
        totals = flagged.select([pl.col(c).sum().alias(c) for c in flag_cols]).row(0)
        for name, total in zip(flag_cols, totals, strict=True):
            report.counts[name.removeprefix(FLAG_PREFIX)] += int(total or 0)

        any_flag = pl.any_horizontal([pl.col(c) for c in flag_cols])
        report.rows_flagged += int(flagged.select(any_flag.sum()).item() or 0)

        if self._collect_examples:
            self._collect(flagged, flag_cols)
        self._collect_gaps(flagged)
        self._update_span(flagged)

    def _collect(self, flagged: pl.DataFrame, flag_cols: list[str]) -> None:
        """Keep a bounded number of concrete bad rows per check, for the report."""
        limit = self._cfg.max_reported_examples
        value_cols = [c for c in flagged.columns if not c.startswith(FLAG_PREFIX)]
        for column in flag_cols:
            check = column.removeprefix(FLAG_PREFIX)
            bucket = self._report.examples.setdefault(check, [])
            if len(bucket) >= limit or not self._report.counts[check]:
                continue
            hits = flagged.filter(pl.col(column)).select(value_cols).head(limit - len(bucket))
            for row in hits.iter_rows(named=True):
                bucket.append({k: _jsonable(v) for k, v in row.items()})

    def _collect_gaps(self, flagged: pl.DataFrame) -> None:
        """Record inter-tick gaps above the threshold, including across batches."""
        threshold_ms = self._cfg.large_gap_seconds * 1000.0
        prev = self._with_carry(pl.col(TIMESTAMP).shift(1), TIMESTAMP, flagged)
        gaps = (
            flagged.select(
                prev.alias("start"),
                pl.col(TIMESTAMP).alias("end"),
                (pl.col(TIMESTAMP) - prev).dt.total_milliseconds().alias("ms"),
            )
            .drop_nulls()
            .filter(pl.col("ms") > threshold_ms)
        )
        if gaps.is_empty():
            return
        self._report.gap_count += gaps.height
        self._report.largest_gap_seconds = max(
            self._report.largest_gap_seconds, float(gaps.select(pl.col("ms").max()).item()) / 1000.0
        )
        room = self._cfg.max_reported_gaps - len(self._report.gaps)
        if room <= 0:
            return
        for row in gaps.head(room).iter_rows(named=True):
            self._report.gaps.append(
                GapRecord(
                    start=row["start"].isoformat(sep=" "),
                    end=row["end"].isoformat(sep=" "),
                    seconds=row["ms"] / 1000.0,
                )
            )

    def _update_span(self, flagged: pl.DataFrame) -> None:
        stamps = flagged.select(
            pl.col(TIMESTAMP).min().alias("lo"), pl.col(TIMESTAMP).max().alias("hi")
        ).row(0)
        if stamps[0] is not None and self._report.first_timestamp is None:
            self._report.first_timestamp = stamps[0].isoformat(sep=" ")
        if stamps[1] is not None:
            self._report.last_timestamp = stamps[1].isoformat(sep=" ")

    def _update_carry_over(self, batch: pl.DataFrame) -> None:
        tail = batch.tail(1)
        if tail.is_empty():
            return
        self._last_row = tail.row(0, named=True)
        self._last_ts = self._last_row.get(TIMESTAMP)


def _jsonable(value: Any) -> Any:
    return value.isoformat(sep=" ") if isinstance(value, datetime) else value
