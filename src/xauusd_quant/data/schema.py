"""Canonical tick schema and source-column detection.

The canonical schema is the single contract every downstream stage relies on.
Two rules are non-negotiable:

1. ``bid`` and ``ask`` are carried through untouched, because the eventual
   backtester must fill longs at the ask and shorts at the bid.
2. ``mid`` and ``spread`` are *derived* columns. If the source happens to ship
   its own spread column it is preserved separately (as ``spread_source``) and
   never allowed to overwrite the derived value.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Final

import polars as pl

from ..utils.config import CanonicalConfig, InputConfig

__all__ = [
    "CANONICAL_ORDER",
    "SCHEMA_VERSION",
    "ColumnDetection",
    "DetectedColumn",
    "TIMESTAMP",
    "BID",
    "ASK",
    "MID",
    "SPREAD",
    "VOLUME",
    "TIMESTAMP_UTC",
    "canonical_schema",
    "derive_expressions",
    "detect_columns",
    "price_dtype",
    "timestamp_dtype",
    "volume_dtype",
]

# Bump the MINOR component when columns are added, the MAJOR component when an
# existing column changes meaning or dtype. Written into every Parquet file's
# key-value metadata and into every report.
SCHEMA_VERSION: Final[str] = "1.0.0"

TIMESTAMP: Final[str] = "timestamp"
TIMESTAMP_UTC: Final[str] = "timestamp_utc"
BID: Final[str] = "bid"
ASK: Final[str] = "ask"
MID: Final[str] = "mid"
SPREAD: Final[str] = "spread"
VOLUME: Final[str] = "volume"
SPREAD_SOURCE: Final[str] = "spread_source"

#: Canonical column order. Columns absent from the source are simply skipped.
CANONICAL_ORDER: Final[tuple[str, ...]] = (
    TIMESTAMP, TIMESTAMP_UTC, BID, ASK, MID, SPREAD, VOLUME, SPREAD_SOURCE,
)


# ---------------------------------------------------------------------------
# dtypes
# ---------------------------------------------------------------------------
def timestamp_dtype(canonical: CanonicalConfig) -> pl.Datetime:
    return pl.Datetime(time_unit=canonical.timestamp_precision)  # type: ignore[arg-type]


def price_dtype(canonical: CanonicalConfig) -> pl.DataType:
    return pl.Float32() if canonical.price_dtype == "float32" else pl.Float64()


def volume_dtype(canonical: CanonicalConfig) -> pl.DataType:
    return {"float32": pl.Float32(), "float64": pl.Float64(), "int64": pl.Int64()}[
        canonical.volume_dtype
    ]


def canonical_schema(
    inp: InputConfig,
    canonical: CanonicalConfig,
    *,
    with_utc: bool,
) -> dict[str, pl.DataType]:
    """Ordered ``{column: dtype}`` for the canonical tick table.

    Only columns that the source can actually supply (plus the derived ones)
    are included, honouring the requirement that absent columns must not be
    invented.
    """
    price = price_dtype(canonical)
    schema: dict[str, pl.DataType] = {TIMESTAMP: timestamp_dtype(canonical)}
    if with_utc:
        schema[TIMESTAMP_UTC] = pl.Datetime(
            time_unit=canonical.timestamp_precision, time_zone="UTC"  # type: ignore[arg-type]
        )
    has_quotes = bool(inp.bid_column and inp.ask_column)
    if inp.bid_column:
        schema[BID] = price
    if inp.ask_column:
        schema[ASK] = price
    if has_quotes and canonical.derive_mid:
        schema[MID] = price
    if has_quotes and canonical.derive_spread:
        schema[SPREAD] = price
    if inp.volume_column:
        schema[VOLUME] = volume_dtype(canonical)
    if inp.spread_column:
        schema[canonical.keep_source_spread_as] = price
    return schema


def derive_expressions(
    inp: InputConfig, canonical: CanonicalConfig
) -> list[pl.Expr]:
    """Expressions that build ``mid`` and ``spread`` from ``bid``/``ask``.

    Returns an empty list when the source has no bid/ask pair, so callers do
    not need to special-case trade-only feeds.
    """
    if not (inp.bid_column and inp.ask_column):
        return []
    price = price_dtype(canonical)
    exprs: list[pl.Expr] = []
    if canonical.derive_mid:
        exprs.append((((pl.col(BID) + pl.col(ASK)) / 2.0).cast(price)).alias(MID))
    if canonical.derive_spread:
        exprs.append(((pl.col(ASK) - pl.col(BID)).cast(price)).alias(SPREAD))
    return exprs


# ---------------------------------------------------------------------------
# Source-column detection
# ---------------------------------------------------------------------------
#: Canonical field -> regexes matched (case-insensitively) against headers.
#: Ordered most-specific first; the first hit wins.
_PATTERNS: Final[dict[str, tuple[str, ...]]] = {
    "timestamp": (r"^date_?time$", r"^timestamp$", r"^time$", r"^date$", r"^ts$",
                  r"^gmt_?time$", r"^utc_?time$", r"^local_?time$", r".*date.*time.*"),
    "bid": (r"^bid$", r"^bid_?price$", r"^bidprice$", r"^b$", r".*\bbid\b.*"),
    "ask": (r"^ask$", r"^ask_?price$", r"^askprice$", r"^offer$", r"^a$", r".*\bask\b.*"),
    "volume": (r"^volume$", r"^vol$", r"^qty$", r"^quantity$", r"^size$",
               r"^tick_?volume$", r"^real_?volume$", r".*volume.*"),
    "spread": (r"^spread$", r"^spr$", r".*spread.*"),
    "mid": (r"^mid$", r"^mid_?price$", r"^midprice$"),
}

#: Timestamp formats tried against sample values, most specific first.
#: Each entry is (polars/chrono format, regex the sample must match).
_TIMESTAMP_FORMATS: Final[tuple[tuple[str, str], ...]] = (
    (r"%Y%m%d %H:%M:%S%.f", r"^\d{8} \d{2}:\d{2}:\d{2}\.\d+$"),
    (r"%Y%m%d %H:%M:%S", r"^\d{8} \d{2}:\d{2}:\d{2}$"),
    (r"%Y%m%d %H:%M", r"^\d{8} \d{2}:\d{2}$"),
    (r"%Y-%m-%d %H:%M:%S%.f", r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+$"),
    (r"%Y-%m-%dT%H:%M:%S%.f", r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+$"),
    (r"%Y-%m-%d %H:%M:%S", r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"),
    (r"%Y-%m-%dT%H:%M:%S", r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$"),
    (r"%Y.%m.%d %H:%M:%S", r"^\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2}$"),
    (r"%Y/%m/%d %H:%M:%S", r"^\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}$"),
    (r"%d/%m/%Y %H:%M:%S", r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}$"),
    (r"%d.%m.%Y %H:%M:%S", r"^\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2}$"),
)

#: Formats whose lexical order matches chronological order. Only these can be
#: byte-offset seeked, which is what makes `convert --resume` cheap.
_LEXICOGRAPHIC_FORMATS: Final[frozenset[str]] = frozenset({
    r"%Y%m%d %H:%M:%S%.f", r"%Y%m%d %H:%M:%S", r"%Y%m%d %H:%M",
    r"%Y-%m-%d %H:%M:%S%.f", r"%Y-%m-%dT%H:%M:%S%.f",
    r"%Y-%m-%d %H:%M:%S", r"%Y-%m-%dT%H:%M:%S",
    r"%Y.%m.%d %H:%M:%S", r"%Y/%m/%d %H:%M:%S",
})


@dataclass(frozen=True, slots=True)
class DetectedColumn:
    """One canonical field mapped onto a source header."""

    canonical: str
    source: str
    index: int
    matched_pattern: str


@dataclass(frozen=True, slots=True)
class ColumnDetection:
    """Result of inferring the source schema from a header row plus samples."""

    headers: tuple[str, ...]
    columns: tuple[DetectedColumn, ...]
    timestamp_format: str | None
    timestamp_lexicographic: bool
    timestamp_samples: tuple[str, ...]
    unmapped_headers: tuple[str, ...]

    @property
    def mapping(self) -> dict[str, str]:
        """Canonical field -> source column name."""
        return {c.canonical: c.source for c in self.columns}

    @property
    def is_usable(self) -> bool:
        """True when at least a timestamp and a bid/ask pair were found."""
        m = self.mapping
        return "timestamp" in m and "bid" in m and "ask" in m

    def missing(self) -> tuple[str, ...]:
        """Required canonical fields that could not be mapped."""
        m = self.mapping
        return tuple(f for f in ("timestamp", "bid", "ask") if f not in m)

    def to_config_fragment(self) -> dict[str, object]:
        """YAML fragment that can be merged into ``config/data.yaml``."""
        m = self.mapping
        return {
            "timestamp_column": m.get("timestamp"),
            "bid_column": m.get("bid"),
            "ask_column": m.get("ask"),
            "volume_column": m.get("volume"),
            "spread_column": m.get("spread"),
            "mid_column": m.get("mid"),
            "timestamp_format": self.timestamp_format,
            "timestamp_lexicographic": self.timestamp_lexicographic,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "headers": list(self.headers),
            "mapping": self.mapping,
            "columns": [asdict(c) for c in self.columns],
            "timestamp_format": self.timestamp_format,
            "timestamp_lexicographic": self.timestamp_lexicographic,
            "timestamp_samples": list(self.timestamp_samples),
            "unmapped_headers": list(self.unmapped_headers),
            "usable": self.is_usable,
            "missing_required": list(self.missing()),
        }


def detect_columns(
    headers: Sequence[str], timestamp_samples: Sequence[str] = ()
) -> ColumnDetection:
    """Map source headers onto canonical fields and infer the timestamp format.

    Detection is a *suggestion*: the CLI prints it and can write it into the
    config, but nothing in the pipeline silently relies on it. Explicit
    configuration always wins.
    """
    cleaned = [h.strip().lstrip("﻿") for h in headers]
    # Match against a normalised form so "gmt time", "gmt_time" and "gmttime"
    # all behave the same, while the reported source name stays verbatim.
    normalised = [re.sub(r"\s+", "_", h).strip("_") for h in cleaned]
    taken: set[int] = set()
    found: list[DetectedColumn] = []

    for canonical, patterns in _PATTERNS.items():
        for pattern in patterns:
            hit = next(
                (i for i, h in enumerate(normalised)
                 if i not in taken and re.fullmatch(pattern, h, re.IGNORECASE)),
                None,
            )
            if hit is not None:
                taken.add(hit)
                found.append(DetectedColumn(canonical, cleaned[hit], hit, pattern))
                break

    fmt = detect_timestamp_format(timestamp_samples)
    return ColumnDetection(
        headers=tuple(cleaned),
        columns=tuple(found),
        timestamp_format=fmt,
        timestamp_lexicographic=fmt in _LEXICOGRAPHIC_FORMATS if fmt else False,
        timestamp_samples=tuple(timestamp_samples[:5]),
        unmapped_headers=tuple(h for i, h in enumerate(cleaned) if i not in taken),
    )


def detect_timestamp_format(samples: Sequence[str]) -> str | None:
    """Return the first known format matching *every* non-empty sample."""
    values = [s.strip() for s in samples if s and s.strip()]
    if not values:
        return None
    for fmt, pattern in _TIMESTAMP_FORMATS:
        if all(re.fullmatch(pattern, v) for v in values):
            return fmt
    return None
