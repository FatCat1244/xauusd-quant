"""Configuration loading, validation and fingerprinting."""

from __future__ import annotations

import pytest
import yaml

from conftest import BASE_CONFIG, write_config
from xauusd_quant.utils.config import (
    CleaningConfig,
    ConfigError,
    ParquetConfig,
    ResamplingConfig,
    TimezoneConfig,
    load_config,
)


def test_loads_the_shipped_repository_config():
    """The config committed to the repo must always be valid."""
    config = load_config()
    assert config.instrument == "XAUUSD"
    assert config.input.timestamp_column == "DateTime"
    assert config.timezone.mode == "anchored_dst"
    assert config.resampling.timeframes == ("1m", "5m", "15m", "30m", "1h")


def test_relative_paths_anchor_at_the_project_root(clean_csv, tmp_path):
    path = write_config(tmp_path, clean_csv, processed_data_path="data/parquet")
    config = load_config(path, root=tmp_path)
    assert config.processed_data_path == (tmp_path / "data" / "parquet").resolve()


def test_environment_variables_are_expanded(clean_csv, tmp_path, monkeypatch):
    monkeypatch.setenv("XQ_TEST_OUT", str(tmp_path / "from_env").replace("\\", "/"))
    path = write_config(tmp_path, clean_csv, processed_data_path="${XQ_TEST_OUT}")
    assert load_config(path, root=tmp_path).processed_data_path.name == "from_env"


def test_environment_default_is_used_when_unset(clean_csv, tmp_path, monkeypatch):
    monkeypatch.delenv("XQ_UNSET_VAR", raising=False)
    path = write_config(tmp_path, clean_csv,
                        processed_data_path="${XQ_UNSET_VAR:-data/fallback}")
    assert load_config(path, root=tmp_path).processed_data_path.name == "fallback"


def test_cli_override_wins_over_the_file(clean_csv, tmp_path):
    path = write_config(tmp_path, clean_csv)
    other = tmp_path / "other.csv"
    other.write_text("DateTime,Bid,Ask,Volume\n", encoding="utf-8")
    config = load_config(path, root=tmp_path, overrides={"raw_data_path": str(other)})
    assert config.raw_data_path == other.resolve()


# ---------------------------------------------------------------------------
# Strictness
# ---------------------------------------------------------------------------
def test_unknown_top_level_key_is_rejected(clean_csv, tmp_path):
    path = write_config(tmp_path, clean_csv, mystery_option=1)
    with pytest.raises(ConfigError, match="unknown top-level option"):
        load_config(path, root=tmp_path)


def test_unknown_section_key_is_rejected(clean_csv, tmp_path):
    path = write_config(tmp_path, clean_csv, parquet={"compresion": "zstd"})
    with pytest.raises(ConfigError, match="unknown option"):
        load_config(path, root=tmp_path)


def test_missing_config_file_is_reported_clearly(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml", root=tmp_path)


@pytest.mark.parametrize(
    ("section", "payload", "message"),
    [
        ("timezone", {"mode": "guesswork"}, "timezone.mode"),
        ("timezone", {"mode": "iana", "iana_tz": None}, "iana_tz"),
        ("timezone", {"mode": "naive", "emit_utc_column": True}, "cannot emit UTC"),
        ("resampling", {"label": "middle"}, "resampling.label"),
        ("resampling", {"closed": "both"}, "resampling.closed"),
        ("parquet", {"compression": "rar"}, "parquet.compression"),
        ("parquet", {"partitioning": ["day"]}, "parquet.partitioning"),
        ("conversion", {"read_block_mb": 0}, "read_block_mb"),
        ("conversion", {"min_free_gb": -1}, "min_free_gb"),
        ("conversion", {"flush_rows": 1000}, "unknown option"),
        ("diagnostics", {"low_tick_count_quantile": 2.0}, "low_tick_count_quantile"),
    ],
)
def test_invalid_values_are_rejected(clean_csv, tmp_path, section, payload, message):
    path = write_config(tmp_path, clean_csv, **{section: payload})
    with pytest.raises(ConfigError, match=message):
        load_config(path, root=tmp_path)


def test_multi_character_delimiter_is_rejected(clean_csv, tmp_path):
    path = write_config(tmp_path, clean_csv, delimiter="||")
    with pytest.raises(ConfigError, match="one character"):
        load_config(path, root=tmp_path)


def test_non_dot_decimal_separator_is_rejected(clean_csv, tmp_path):
    path = write_config(tmp_path, clean_csv, decimal_separator=",")
    with pytest.raises(ConfigError, match="decimal_separator"):
        load_config(path, root=tmp_path)


# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------
def test_source_columns_skips_absent_columns():
    from xauusd_quant.utils.config import InputConfig

    inp = InputConfig(volume_column=None, spread_column=None)
    assert set(inp.source_columns) == {"timestamp", "bid", "ask"}


def test_enabled_drops_lists_only_active_rules():
    assert CleaningConfig().enabled_drops() == (
        "unparseable_timestamp", "missing_bid_or_ask", "non_positive_price", "non_finite"
    )
    assert CleaningConfig(
        drop_unparseable_timestamp=False, drop_missing_bid_or_ask=False,
        drop_non_positive_price=False, drop_non_finite=False,
    ).enabled_drops() == ()


def test_timezone_descriptions_are_human_readable():
    assert "unknown" in TimezoneConfig(mode="naive").describe()
    assert "UTC+2" in TimezoneConfig(mode="fixed_offset", fixed_offset_hours=2).describe()
    assert "America/New_York" in TimezoneConfig(
        mode="anchored_dst", offset_hours=7).describe()


def test_partitioning_and_timeframes_become_tuples():
    assert ParquetConfig(partitioning=["year", "month"]).partitioning == ("year", "month")
    assert ResamplingConfig(timeframes=["1m"]).timeframes == ("1m",)


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------
def test_fingerprint_is_stable_across_identical_loads(clean_csv, tmp_path):
    path = write_config(tmp_path, clean_csv)
    assert load_config(path, root=tmp_path).fingerprint() == \
        load_config(path, root=tmp_path).fingerprint()


def test_fingerprint_changes_when_processing_settings_change(clean_csv, tmp_path):
    base = load_config(write_config(tmp_path, clean_csv), root=tmp_path)
    other = tmp_path / "b.yaml"
    data = {**BASE_CONFIG, "raw_data_path": str(clean_csv).replace("\\", "/"),
            "validation": {**BASE_CONFIG["validation"], "max_spread": 3.0}}
    other.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert load_config(other, root=tmp_path).fingerprint() != base.fingerprint()


def test_fingerprint_ignores_paths(clean_csv, tmp_path):
    """Moving the repository must not invalidate already-written partitions."""
    a = load_config(write_config(tmp_path, clean_csv), root=tmp_path)
    b = load_config(
        write_config(tmp_path, clean_csv,
                     processed_data_path=str(tmp_path / "elsewhere").replace("\\", "/")),
        root=tmp_path,
    )
    assert a.fingerprint() == b.fingerprint()


def test_to_dict_is_json_safe(clean_config):
    import json

    payload = clean_config.to_dict()
    assert isinstance(json.dumps(payload), str)
    assert isinstance(payload["raw_data_path"], str)
