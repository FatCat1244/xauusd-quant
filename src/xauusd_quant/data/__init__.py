"""Data pipeline: raw files -> validated ticks -> Parquet -> bars.

Stage order, and the module that owns each stage::

    raw CSV        inspector.py   profile a large file without reading it all
      |            schema.py      canonical column contract + detection
      v            timezones.py   how naive timestamps are interpreted
    validated      validator.py   flag every defect, delete nothing
      |            cleaner.py     apply the drop policy, keep an audit trail
      v
    Parquet        converter.py   streaming, partitioned, resumable
      |            metadata.py    dataset-level statistics
      v
    bars           resampler.py   tick -> OHLC, no look-ahead
                   diagnostics.py bar coverage and gap analysis
                   loader.py      DuckDB range queries
"""

from __future__ import annotations

from .cleaner import Cleaner, CleaningSummary
from .converter import ConversionResult, TickConverter
from .diagnostics import BarDiagnostics, diagnose_all, diagnose_timeframe
from .inspector import FileInspection, inspect_file, inspect_sources
from .loader import DataStore, TooManyRowsError
from .metadata import DatasetMetadata, build_dataset_metadata
from .resampler import BarResampler, parse_timeframe, resample_ticks
from .schema import SCHEMA_VERSION, ColumnDetection, canonical_schema, detect_columns
from .timezones import describe_policy, to_utc_expr
from .validator import StreamValidator, ValidationReport

__all__ = [
    "SCHEMA_VERSION",
    "BarDiagnostics",
    "BarResampler",
    "Cleaner",
    "CleaningSummary",
    "ColumnDetection",
    "ConversionResult",
    "DataStore",
    "DatasetMetadata",
    "FileInspection",
    "StreamValidator",
    "TickConverter",
    "TooManyRowsError",
    "ValidationReport",
    "build_dataset_metadata",
    "canonical_schema",
    "describe_policy",
    "detect_columns",
    "diagnose_all",
    "diagnose_timeframe",
    "inspect_file",
    "inspect_sources",
    "parse_timeframe",
    "resample_ticks",
    "to_utc_expr",
]
