"""Shared fixtures.

Every fixture builds small synthetic datasets. Nothing here reads the real
34 GB file, so the suite runs in seconds and is safe on any machine.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import yaml

from xauusd_quant.utils.config import Config, load_config

TS_FORMAT = "%Y%m%d %H:%M:%S.%f"


def fmt(moment: datetime) -> str:
    """Render a datetime the way the raw XAUUSD file does (millisecond precision)."""
    return moment.strftime("%Y%m%d %H:%M:%S.") + f"{moment.microsecond // 1000:03d}"


# ---------------------------------------------------------------------------
# Synthetic raw data
# ---------------------------------------------------------------------------
def write_csv(path: Path, rows: list[str], header: str = "DateTime,Bid,Ask,Volume") -> Path:
    """Write a CSV with LF endings, matching the real dataset exactly."""
    payload = header + "\n" + "\n".join(rows) + "\n"
    path.write_bytes(payload.encode("utf-8"))
    return path


def make_clean_rows(
    start: datetime,
    count: int,
    *,
    step_ms: int = 100,
    bid0: float = 2000.0,
    spread: float = 0.30,
    volume: float = 120,
) -> list[str]:
    """A run of well-formed rows with a deterministic sawtooth price path."""
    rows = []
    for i in range(count):
        moment = start + timedelta(milliseconds=i * step_ms)
        bid = round(bid0 + (i % 10) * 0.1, 3)
        rows.append(f"{fmt(moment)},{bid:.3f},{bid + spread:.3f},{volume:g}")
    return rows


@pytest.fixture
def clean_csv(tmp_path: Path) -> Path:
    """600 clean ticks starting 2021-01-04 01:00:00, one every 100 ms."""
    rows = make_clean_rows(datetime(2021, 1, 4, 1, 0, 0), 600)
    return write_csv(tmp_path / "clean.csv", rows)


@pytest.fixture
def dirty_csv(tmp_path: Path) -> Path:
    """A file containing one instance of every defect the validator checks for.

    Layout (20 data lines):
      0-4    clean
      5      empty timestamp cell        -> missing_timestamp
      6      unparseable timestamp       -> unparseable_timestamp
      7      empty bid                   -> missing_bid_or_ask
      8      bid = 0                     -> non_positive_price
      9      ask < bid                   -> ask_below_bid + negative_spread
      10     spread = 80                 -> extreme_spread
      11     negative volume             -> negative_volume
      12     exact duplicate of line 11  -> exact_duplicate_rows
      13     same timestamp as line 12   -> duplicate_timestamps
      14     timestamp goes backwards    -> non_monotonic
      15     only three fields           -> malformed (rejected by the parser)
      16     price far out of range      -> price_out_of_range
      17-19  clean, after a 2-hour gap   -> one large gap
    """
    base = datetime(2021, 1, 4, 1, 0, 0)
    rows = make_clean_rows(base, 5)
    rows.append(",2000.000,2000.300,120")
    rows.append("not-a-timestamp,2000.000,2000.300,120")
    rows.append(f"{fmt(base + timedelta(milliseconds=700))},,2000.300,120")
    rows.append(f"{fmt(base + timedelta(milliseconds=800))},0.000,2000.300,120")
    rows.append(f"{fmt(base + timedelta(milliseconds=900))},2000.500,2000.100,120")
    rows.append(f"{fmt(base + timedelta(milliseconds=1000))},2000.000,2080.000,120")
    dup = f"{fmt(base + timedelta(milliseconds=1100))},2000.000,2000.300,-5"
    rows.append(dup)
    rows.append(dup)
    rows.append(f"{fmt(base + timedelta(milliseconds=1100))},2001.000,2001.300,120")
    rows.append(f"{fmt(base + timedelta(milliseconds=50))},2002.000,2002.300,120")
    rows.append(f"{fmt(base + timedelta(milliseconds=1300))},2003.000,120")
    rows.append(f"{fmt(base + timedelta(milliseconds=1400))},99.000,99.300,120")
    rows += make_clean_rows(base + timedelta(hours=2), 3)

    return write_csv(tmp_path / "dirty.csv", rows)


@pytest.fixture
def two_month_csv(tmp_path: Path) -> Path:
    """Ticks spanning a month boundary, for partitioning and resume tests."""
    rows = (
        make_clean_rows(datetime(2021, 1, 31, 22, 0, 0), 300, bid0=1900.0)
        + make_clean_rows(datetime(2021, 2, 1, 1, 0, 0), 300, bid0=1910.0)
        + make_clean_rows(datetime(2021, 3, 1, 1, 0, 0), 300, bid0=1920.0)
    )
    return write_csv(tmp_path / "two_months.csv", rows)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_CONFIG: dict[str, Any] = {
    "schema_version": "1.0.0",
    "instrument": "XAUUSD",
    "input_format": "csv",
    "delimiter": ",",
    "encoding": "utf-8",
    "has_header": True,
    "timestamp_column": "DateTime",
    "bid_column": "Bid",
    "ask_column": "Ask",
    "volume_column": "Volume",
    "spread_column": None,
    "timestamp_format": "%Y%m%d %H:%M:%S%.f",
    "timestamp_lexicographic": True,
    "timezone": {
        "source_label": "broker_server_time",
        "mode": "anchored_dst",
        "anchor_tz": "America/New_York",
        "offset_hours": 7,
        "emit_utc_column": True,
    },
    "canonical": {"timestamp_precision": "us", "derive_mid": True, "derive_spread": True},
    "validation": {
        "min_price": 100.0,
        "max_price": 100000.0,
        "max_spread": 50.0,
        "large_gap_seconds": 3600,
    },
    "cleaning": {
        "drop_unparseable_timestamp": True,
        "drop_missing_bid_or_ask": True,
        "drop_non_positive_price": True,
        "drop_non_finite": True,
    },
    "conversion": {"read_block_mb": 1, "index_block_mb": 1, "progress": False,
                   "min_free_gb": 0.0},
    "parquet": {"compression": "zstd", "compression_level": 3, "row_group_rows": 1000},
    "resampling": {"timeframes": ["1m", "5m"], "label": "open", "closed": "left"},
    "diagnostics": {
        "session": {"daily_break_start": "00:00", "daily_break_end": "01:00"},
    },
    "duckdb": {"memory_limit": "1GB", "max_rows_without_override": 1_000_000},
}


def write_config(tmp_path: Path, raw: Path, **overrides: Any) -> Path:
    """Write a YAML config pointing at *raw*, merged with *overrides*."""
    data = {
        **BASE_CONFIG,
        "raw_data_path": str(raw).replace("\\", "/"),
        "processed_data_path": str(tmp_path / "parquet").replace("\\", "/"),
        "bars_path": str(tmp_path / "bars").replace("\\", "/"),
        "metadata_path": str(tmp_path / "metadata").replace("\\", "/"),
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def config_factory(tmp_path: Path):
    """Build a :class:`Config` for a given raw file and optional overrides."""

    def build(raw: Path, **overrides: Any) -> Config:
        return load_config(write_config(tmp_path, raw, **overrides), root=tmp_path)

    return build


@pytest.fixture
def clean_config(clean_csv: Path, config_factory) -> Config:
    return config_factory(clean_csv)


@pytest.fixture
def world(tmp_path: Path, config_factory, monkeypatch):
    """Ticks -> verified 1m bars, with the feature store pointed at tmp_path."""
    import numpy as np

    from xauusd_quant.data.converter import TickConverter
    from xauusd_quant.data.resampler import BarResampler
    from xauusd_quant.features import load_regression_config

    rng = np.random.default_rng(5)
    rows = []
    start = datetime(2021, 1, 4, 1, 0, 0)
    price = 2000.0
    for minute in range(900):
        price *= float(np.exp(rng.normal(0, 0.0005)))
        rows += make_clean_rows(start.replace(hour=1 + minute // 60, minute=minute % 60),
                                3, step_ms=10_000, bid0=round(price, 3))
    source = write_csv(tmp_path / "ticks.csv", rows)
    config = config_factory(source, resampling={"timeframes": ["1m"]})
    TickConverter(config).run()
    BarResampler(config).build("1m")
    monkeypatch.setenv("XAUUSD_FEATURES_PATH", str(tmp_path / "features").replace("\\", "/"))
    regression = load_regression_config()
    return config, regression


# ---------------------------------------------------------------------------
# Synthetic canonical ticks (for resampler tests, skipping the CSV stage)
# ---------------------------------------------------------------------------
def canonical_ticks(
    stamps: list[datetime],
    bids: list[float],
    asks: list[float],
    volumes: list[float] | None = None,
) -> pl.DataFrame:
    """Build a canonical tick frame directly from explicit values."""
    bid = pl.Series("bid", bids, dtype=pl.Float64)
    ask = pl.Series("ask", asks, dtype=pl.Float64)
    return pl.DataFrame({
        "timestamp": pl.Series("timestamp", stamps, dtype=pl.Datetime("us")),
        "bid": bid,
        "ask": ask,
        "mid": (bid + ask) / 2,
        "spread": ask - bid,
        "volume": pl.Series("volume", volumes or [100.0] * len(stamps), dtype=pl.Float64),
    })


# ---------------------------------------------------------------------------
# Research fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def research_config():
    """The repository's real research configuration.

    Used rather than a synthetic one so the tests exercise the settings that
    actually ship, and a bad edit to research.yaml fails the suite.
    """
    from xauusd_quant.research.config import load_research_config

    return load_research_config()


@pytest.fixture
def research_config_factory(tmp_path: Path):
    """Build a ResearchConfig with selected sections overridden."""
    import yaml as _yaml

    from xauusd_quant.research.config import load_research_config

    def build(**overrides: Any):
        base = _yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "config" / "research.yaml")
            .read_text(encoding="utf-8")
        )
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key] = {**base[key], **value}
            else:
                base[key] = value
        base["results_path"] = str(tmp_path / "results").replace("\\", "/")
        path = tmp_path / "research.yaml"
        path.write_text(_yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
        return load_research_config(path, root=tmp_path)

    return build


def make_bars(
    n: int = 500,
    *,
    start: datetime | None = None,
    step_minutes: int = 5,
    seed: int = 7,
    drift: float = 0.0,
    vol: float = 0.0008,
) -> pl.DataFrame:
    """Deterministic synthetic bar frame with the full canonical bar schema."""
    import numpy as np

    start = start or datetime(2024, 1, 2, 1, 0)
    rng = np.random.default_rng(seed)
    shocks = rng.normal(drift, vol, n)
    prices = 2000.0 * np.exp(np.cumsum(shocks))
    return pl.DataFrame({
        "timestamp": [start + timedelta(minutes=step_minutes * i) for i in range(n)],
        "open": prices,
        "high": prices * 1.0003,
        "low": prices * 0.9997,
        "close": prices,
        "tick_count": (50 + rng.integers(0, 200, n)).astype("int64"),
        "volume": (1000.0 + rng.integers(0, 500, n)).astype("float64"),
        "mean_spread": 0.30 + 0.05 * rng.random(n),
        "median_spread": 0.30 + 0.05 * rng.random(n),
        "min_spread": np.full(n, 0.20),
        "max_spread": np.full(n, 0.90),
        "first_bid": prices - 0.15,
        "first_ask": prices + 0.15,
        "last_bid": prices - 0.15,
        "last_ask": prices + 0.15,
    })
