"""Typed configuration loaded from ``config/data.yaml``.

Design rules:

* Every section is a frozen dataclass, so configuration is immutable once
  loaded and typos surface as an explicit :class:`ConfigError` instead of a
  silently-ignored key.
* Nothing about the XAUUSD dataset is hard-coded here. Column names, the
  timestamp format and the timezone policy all come from YAML.
* :meth:`Config.fingerprint` returns a stable hash of the effective
  configuration. Every report written by the pipeline embeds it, so a result
  file can always be tied back to the settings that produced it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, ClassVar

import yaml

from .paths import find_project_root, resolve_path

__all__ = [
    "CanonicalConfig",
    "CleaningConfig",
    "Config",
    "ConfigError",
    "ConversionConfig",
    "DiagnosticsConfig",
    "DuckDBConfig",
    "InputConfig",
    "MetadataConfig",
    "ParquetConfig",
    "ResamplingConfig",
    "SessionConfig",
    "TimezoneConfig",
    "ValidationConfig",
    "default_config_path",
    "load_config",
]

DEFAULT_CONFIG_RELPATH = "config/data.yaml"
CONFIG_ENV_VAR = "XAUUSD_QUANT_CONFIG"

_ENV_PATTERN = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}")


class ConfigError(ValueError):
    """Raised when the configuration file is malformed or internally inconsistent."""


# ---------------------------------------------------------------------------
# Generic dataclass construction with strict key checking
# ---------------------------------------------------------------------------

def _expand(value: Any) -> Any:
    """Recursively apply ``${VAR}`` / ``${VAR:-default}`` expansion to strings."""
    if isinstance(value, str):
        return _ENV_PATTERN.sub(
            lambda m: os.environ.get(m.group("name")) or (m.group("default") or ""), value
        )
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _build[T](cls: type[T], data: Any, where: str) -> T:
    """Instantiate dataclass *cls* from mapping *data*, rejecting unknown keys."""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    assert is_dataclass(cls)
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(
            f"{where}: unknown option(s) {unknown}. Valid options: {sorted(known)}"
        )
    return cls(**data)


def _one_of(value: Any, allowed: tuple[Any, ...], where: str) -> None:
    if value not in allowed:
        raise ConfigError(f"{where}: expected one of {list(allowed)}, got {value!r}")


def _as_dict(obj: Any) -> Any:
    """JSON-safe recursive conversion used for reports and fingerprints."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _as_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, dict):
        return {k: _as_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_as_dict(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class InputConfig:
    """How to read the raw source file."""

    input_format: str = "csv"
    delimiter: str = ","
    encoding: str = "utf-8"
    has_header: bool = True
    quote_char: str | None = '"'
    comment_prefix: str | None = None

    timestamp_column: str = "DateTime"
    bid_column: str | None = "Bid"
    ask_column: str | None = "Ask"
    volume_column: str | None = "Volume"
    spread_column: str | None = None
    mid_column: str | None = None

    timestamp_format: str | None = "%Y%m%d %H:%M:%S%.f"
    timestamp_lexicographic: bool = True
    decimal_separator: str = "."
    thousands_separator: str | None = None

    def __post_init__(self) -> None:
        _one_of(self.input_format, ("csv", "tsv", "txt"), "input_format")
        if len(self.delimiter) != 1:
            raise ConfigError(f"delimiter: must be exactly one character, got {self.delimiter!r}")
        if not self.timestamp_column:
            raise ConfigError("timestamp_column: required")
        if self.decimal_separator != ".":
            raise ConfigError(
                "decimal_separator: only '.' is supported by the streaming CSV reader; "
                "pre-convert the file or open an issue"
            )

    @property
    def source_columns(self) -> dict[str, str]:
        """Canonical name -> source column name, for columns that exist."""
        pairs = {
            "timestamp": self.timestamp_column,
            "bid": self.bid_column,
            "ask": self.ask_column,
            "volume": self.volume_column,
            "spread_source": self.spread_column,
            "mid_source": self.mid_column,
        }
        return {k: v for k, v in pairs.items() if v}


@dataclass(frozen=True, slots=True)
class TimezoneConfig:
    """Policy for interpreting the naive timestamps in the raw file.

    See the extensive comment block in ``config/data.yaml`` for why this
    dataset needs ``anchored_dst`` rather than a plain IANA zone.
    """

    source_label: str = "unknown"
    mode: str = "naive"
    anchor_tz: str = "America/New_York"
    offset_hours: float = 0.0
    fixed_offset_hours: float = 0.0
    iana_tz: str | None = None
    emit_utc_column: bool = False
    on_ambiguous: str = "earliest"
    on_non_existent: str = "null"

    MODES = ("naive", "fixed_offset", "iana", "anchored_dst")

    def __post_init__(self) -> None:
        _one_of(self.mode, self.MODES, "timezone.mode")
        _one_of(self.on_ambiguous, ("earliest", "latest", "null", "raise"), "timezone.on_ambiguous")
        _one_of(self.on_non_existent, ("null", "raise"), "timezone.on_non_existent")
        if self.mode == "iana" and not self.iana_tz:
            raise ConfigError("timezone.iana_tz: required when timezone.mode == 'iana'")
        if self.mode == "naive" and self.emit_utc_column:
            raise ConfigError(
                "timezone.emit_utc_column: cannot emit UTC in 'naive' mode - the file's "
                "zone is unknown. Choose a mode that states the offset, or set this to false."
            )

    def describe(self) -> str:
        """Human-readable one-liner used in logs and reports."""
        if self.mode == "naive":
            return "naive wall clock; zone unknown and not assumed"
        if self.mode == "fixed_offset":
            return f"fixed UTC{self.fixed_offset_hours:+g} (no DST)"
        if self.mode == "iana":
            return f"IANA zone {self.iana_tz}"
        return (
            f"{self.anchor_tz} wall clock {self.offset_hours:+g}h "
            f"(follows the {self.anchor_tz} DST calendar)"
        )


@dataclass(frozen=True, slots=True)
class CanonicalConfig:
    timestamp_precision: str = "us"
    price_dtype: str = "float64"
    volume_dtype: str = "float64"
    derive_mid: bool = True
    derive_spread: bool = True
    keep_source_spread_as: str = "spread_source"

    def __post_init__(self) -> None:
        _one_of(self.timestamp_precision, ("ms", "us", "ns"), "canonical.timestamp_precision")
        _one_of(self.price_dtype, ("float32", "float64"), "canonical.price_dtype")
        _one_of(self.volume_dtype, ("float32", "float64", "int64"), "canonical.volume_dtype")


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    min_price: float = 0.0
    max_price: float = 1e12
    max_spread: float = 1e9
    max_volume: float = 1e12
    large_gap_seconds: float = 3600.0
    max_reported_gaps: int = 200
    max_reported_examples: int = 20
    # How many rows of the previous batch are carried forward when
    # looking for repeated rows. Duplicates further apart than this are
    # not detected by the streaming pass; `xq metadata` re-checks the
    # whole dataset exactly and will report any that escaped.
    duplicate_window_rows: int = 200_000


@dataclass(frozen=True, slots=True)
class CleaningConfig:
    drop_unparseable_timestamp: bool = True
    drop_missing_bid_or_ask: bool = True
    drop_non_positive_price: bool = True
    drop_non_finite: bool = True
    drop_ask_below_bid: bool = False
    drop_negative_spread: bool = False
    drop_extreme_spread: bool = False
    drop_price_out_of_range: bool = False
    drop_negative_volume: bool = False
    drop_exact_duplicate_rows: bool = False
    drop_duplicate_timestamps: bool = False
    drop_non_monotonic: bool = False
    duplicate_keep: str = "first"

    def __post_init__(self) -> None:
        _one_of(self.duplicate_keep, ("first", "last"), "cleaning.duplicate_keep")

    def enabled_drops(self) -> tuple[str, ...]:
        """Names of the checks currently configured to remove rows."""
        return tuple(
            f.name.removeprefix("drop_")
            for f in fields(self)
            if f.name.startswith("drop_") and getattr(self, f.name)
        )


@dataclass(frozen=True, slots=True)
class ConversionConfig:
    """How conversion runs. None of these settings changes the converted data.

    They are therefore excluded from :meth:`Config.tick_fingerprint`, so
    tuning them never invalidates a finished dataset.
    """

    read_block_mb: int = 64
    index_block_mb: int = 64
    resume: bool = True
    progress: bool = True
    digest: bool = True
    min_free_gb: float = 5.0
    keep_previous: bool = False

    def __post_init__(self) -> None:
        if self.read_block_mb < 1:
            raise ConfigError("conversion.read_block_mb: must be >= 1")
        if self.index_block_mb < 1:
            raise ConfigError("conversion.index_block_mb: must be >= 1")
        if self.min_free_gb < 0:
            raise ConfigError("conversion.min_free_gb: cannot be negative")

    @property
    def read_block_bytes(self) -> int:
        return self.read_block_mb * 1024 * 1024

    @property
    def index_block_bytes(self) -> int:
        return self.index_block_mb * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ParquetConfig:
    compression: str = "zstd"
    compression_level: int | None = 7
    partitioning: tuple[str, ...] = ("year", "month")
    row_group_rows: int = 500_000
    write_statistics: bool = True
    use_dictionary: bool = False
    filename_template: str = "xauusd-ticks-{year}-{month}.parquet"

    def __post_init__(self) -> None:
        _one_of(self.compression, ("zstd", "snappy", "gzip", "lz4", "brotli", "none"),
                "parquet.compression")
        object.__setattr__(self, "partitioning", tuple(self.partitioning))
        allowed = {("year",), ("year", "month"), ()}
        if tuple(self.partitioning) not in allowed:
            raise ConfigError(
                f"parquet.partitioning: expected one of {sorted(allowed)}, "
                f"got {list(self.partitioning)}"
            )


@dataclass(frozen=True, slots=True)
class MetadataConfig:
    exact_quantiles: bool = False
    spread_percentiles: tuple[float, ...] = (1, 5, 25, 50, 75, 90, 95, 99, 99.9)

    def __post_init__(self) -> None:
        object.__setattr__(self, "spread_percentiles", tuple(self.spread_percentiles))
        for p in self.spread_percentiles:
            if not 0 < p < 100:
                raise ConfigError(f"metadata.spread_percentiles: {p} is not in (0, 100)")


@dataclass(frozen=True, slots=True)
class ResamplingConfig:
    timeframes: tuple[str, ...] = ("1m", "5m", "15m", "30m", "1h")
    price_source: str = "mid"
    label: str = "open"
    closed: str = "left"
    time_basis: str = "timestamp"
    min_ticks_per_bar: int = 1
    include_bid_ask_edges: bool = True
    partition_by: str = "year"
    filename_template: str = "xauusd-bars-{timeframe}-{year}.parquet"

    def __post_init__(self) -> None:
        object.__setattr__(self, "timeframes", tuple(self.timeframes))
        _one_of(self.price_source, ("mid", "bid", "ask"), "resampling.price_source")
        _one_of(self.label, ("open", "close"), "resampling.label")
        _one_of(self.closed, ("left", "right"), "resampling.closed")
        _one_of(self.time_basis, ("timestamp", "timestamp_utc"), "resampling.time_basis")
        _one_of(self.partition_by, ("year", "month"), "resampling.partition_by")
        if self.min_ticks_per_bar < 1:
            raise ConfigError("resampling.min_ticks_per_bar: must be >= 1")


@dataclass(frozen=True, slots=True)
class SessionConfig:
    """Known trading schedule, used only to classify gaps in diagnostics."""

    daily_break_start: str | None = None
    daily_break_end: str | None = None
    week_open_weekday: int = 1
    week_open_time: str = "00:00"
    week_close_weekday: int = 5
    week_close_time: str = "23:59"


@dataclass(frozen=True, slots=True)
class DiagnosticsConfig:
    low_tick_count_quantile: float = 0.01
    max_reported_gaps: int = 100
    session: SessionConfig = field(default_factory=SessionConfig)

    def __post_init__(self) -> None:
        if not 0 < self.low_tick_count_quantile < 1:
            raise ConfigError("diagnostics.low_tick_count_quantile: must be in (0, 1)")


@dataclass(frozen=True, slots=True)
class DuckDBConfig:
    threads: int | None = None
    memory_limit: str | None = "4GB"
    max_rows_without_override: int = 50_000_000


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Config:
    """Fully-resolved pipeline configuration."""

    project_root: Path
    config_path: Path | None

    schema_version: str
    instrument: str

    raw_data_path: Path
    processed_data_path: Path
    bars_path: Path
    metadata_path: Path

    input: InputConfig
    timezone: TimezoneConfig
    canonical: CanonicalConfig
    validation: ValidationConfig
    cleaning: CleaningConfig
    conversion: ConversionConfig
    parquet: ParquetConfig
    metadata: MetadataConfig
    resampling: ResamplingConfig
    diagnostics: DiagnosticsConfig
    duckdb: DuckDBConfig

    # -- derived helpers ---------------------------------------------------
    def bars_dir(self, timeframe: str) -> Path:
        return self.bars_path / timeframe

    def metadata_file(self, name: str) -> Path:
        return self.metadata_path / name

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view, suitable for embedding in reports."""
        return _as_dict(self)

    def fingerprint(self) -> str:
        """Stable 16-hex-char digest of the settings that affect output.

        Paths and ``config_path`` are excluded so that moving the repository
        does not invalidate previously-written partitions.
        """
        payload = self.to_dict()
        for key in ("project_root", "config_path", "raw_data_path",
                    "processed_data_path", "bars_path", "metadata_path"):
            payload.pop(key, None)
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.blake2b(blob, digest_size=8).hexdigest()

    #: Sections whose settings determine the content of the tick dataset.
    TICK_SECTIONS: ClassVar[tuple[str, ...]] = (
        "instrument", "schema_version", "input", "timezone", "canonical",
        "validation", "cleaning", "parquet",
    )

    def tick_fingerprint(self) -> str:
        """Digest of only the settings that change the converted tick data.

        A partition is reusable exactly when this matches. Operational knobs -
        block sizes, progress display, disk headroom, the query layer, bars,
        diagnostics - are left out, so changing them never forces a rebuild.
        """
        payload = self.to_dict()
        chosen = {key: payload.get(key) for key in self.TICK_SECTIONS}
        blob = json.dumps(chosen, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.blake2b(blob, digest_size=8).hexdigest()

    def bar_fingerprint(self) -> str:
        """Digest of the settings that determine bar content (ticks + resampling)."""
        payload = {"ticks": self.tick_fingerprint(), "resampling": self.to_dict()["resampling"]}
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.blake2b(blob, digest_size=8).hexdigest()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def default_config_path(root: Path | None = None) -> Path:
    """Config path from ``$XAUUSD_QUANT_CONFIG`` or the repository default."""
    root = root or find_project_root()
    return resolve_path(os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG_RELPATH, root)


_TOP_LEVEL_SECTIONS = {
    "timezone": TimezoneConfig,
    "canonical": CanonicalConfig,
    "validation": ValidationConfig,
    "cleaning": CleaningConfig,
    "conversion": ConversionConfig,
    "parquet": ParquetConfig,
    "metadata": MetadataConfig,
    "resampling": ResamplingConfig,
    "duckdb": DuckDBConfig,
}

# Keys that live at the top level of the YAML but belong to InputConfig.
_INPUT_KEYS = {f.name for f in fields(InputConfig)}

_PATH_KEYS = ("raw_data_path", "processed_data_path", "bars_path", "metadata_path")
_SCALAR_KEYS = ("schema_version", "instrument")


def load_config(
    path: str | os.PathLike[str] | None = None,
    *,
    root: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Load, expand, validate and resolve the YAML configuration.

    Parameters
    ----------
    path:
        Config file to read. Defaults to ``$XAUUSD_QUANT_CONFIG`` or
        ``config/data.yaml``.
    root:
        Repository root used to anchor relative paths. Auto-detected by
        default.
    overrides:
        Shallow top-level overrides applied after the file is read. Used by
        the CLI for flags such as ``--raw-path``.
    """
    root = root or find_project_root()
    cfg_path = resolve_path(path, root) if path is not None else default_config_path(root)

    if not cfg_path.exists():
        raise ConfigError(f"Configuration file not found: {cfg_path}")

    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{cfg_path}: top level must be a mapping")

    data: dict[str, Any] = _expand(raw)
    for key, value in (overrides or {}).items():
        if value is not None:
            data[key] = value

    # Split the flat top level into its typed sections.
    input_data = {k: data.pop(k) for k in list(data) if k in _INPUT_KEYS}
    sections: dict[str, Any] = {
        name: _build(cls, data.pop(name, {}), name)
        for name, cls in _TOP_LEVEL_SECTIONS.items()
    }

    diag_raw = dict(data.pop("diagnostics", {}) or {})
    session = _build(SessionConfig, diag_raw.pop("session", {}), "diagnostics.session")
    diagnostics = _build(DiagnosticsConfig, {**diag_raw, "session": session}, "diagnostics")

    paths = {k: resolve_path(data.pop(k, f"data/{k}"), root) for k in _PATH_KEYS}
    scalars = {k: data.pop(k, "") for k in _SCALAR_KEYS}

    if data:
        raise ConfigError(
            f"{cfg_path}: unknown top-level option(s) {sorted(data)}. "
            f"Known sections: {sorted({*_TOP_LEVEL_SECTIONS, 'diagnostics', *_PATH_KEYS, *_SCALAR_KEYS, *_INPUT_KEYS})}"
        )

    return Config(
        project_root=root,
        config_path=cfg_path,
        schema_version=str(scalars["schema_version"] or "1.0.0"),
        instrument=str(scalars["instrument"] or "UNKNOWN"),
        raw_data_path=paths["raw_data_path"],
        processed_data_path=paths["processed_data_path"],
        bars_path=paths["bars_path"],
        metadata_path=paths["metadata_path"],
        input=_build(InputConfig, input_data, "input"),
        timezone=sections["timezone"],
        canonical=sections["canonical"],
        validation=sections["validation"],
        cleaning=sections["cleaning"],
        conversion=sections["conversion"],
        parquet=sections["parquet"],
        metadata=sections["metadata"],
        resampling=sections["resampling"],
        diagnostics=diagnostics,
        duckdb=sections["duckdb"],
    )
