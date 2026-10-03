"""Actual converter/bar path guards values at the scan, before feature or label materialization."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from conftest import make_clean_rows, write_csv
from strategy_synth import fixture
from xauusd_quant.data.converter import TickConverter
from xauusd_quant.data.resampler import BarResampler
from xauusd_quant.strategy_validation.source import BarDevelopmentSource


def test_development_bar_values_are_cut_before_materialization(
    config_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    start = datetime(2021, 1, 4, 12)
    rows = []
    for i in range(32):
        rows += make_clean_rows(start + timedelta(minutes=5 * i, seconds=1), 1, bid0=2000 + i)
    path = write_csv(tmp_path / "ticks.csv", rows)
    config = config_factory(
        path,
        timezone={"mode": "fixed_offset", "fixed_offset_hours": 2, "emit_utc_column": True},
        resampling={"timeframes": ["5m"]},
    )
    TickConverter(config).run()
    BarResampler(config).build("5m")
    plan, c, _, _ = fixture()
    source = BarDevelopmentSource(config, c, plan)
    beginning = datetime(2021, 1, 4, 10, tzinfo=UTC)
    cutoff = beginning + timedelta(minutes=75)
    original = pl.LazyFrame.collect
    counts = []

    def guarded(lazy: pl.LazyFrame, *args: Any, **kwargs: Any) -> pl.DataFrame:
        frame = original(lazy, *args, **kwargs)
        if "close" in frame.columns:
            counts.append(frame.height)
            assert frame.height == 14, "bar prices at or after cutoff were materialized"
            assert all(t + timedelta(minutes=5) < cutoff for t in frame["timestamp_utc"])
        return frame

    monkeypatch.setattr(pl.LazyFrame, "collect", guarded)
    training = source.training(beginning, cutoff)
    assert counts == [14]
    assert training and all(r.label_available_at_utc < cutoff for r in training)
    # A long horizon uses observed-bar information ends and admits fewer labels.
    longer = BarDevelopmentSource(config, replace(c, horizon_bars=3), replace(plan, horizon_bars=3))
    assert len(longer.training(beginning, cutoff)) == len(training) - 2


def test_registered_row_cap_limits_collection_before_rejection(
    config_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    start = datetime(2021, 1, 4, 12)
    rows = []
    for i in range(32):
        rows += make_clean_rows(start + timedelta(minutes=5 * i, seconds=1), 1, bid0=2000 + i)
    config = config_factory(
        write_csv(tmp_path / "ticks.csv", rows),
        timezone={"mode": "fixed_offset", "fixed_offset_hours": 2, "emit_utc_column": True},
        resampling={"timeframes": ["5m"]},
    )
    TickConverter(config).run()
    BarResampler(config).build("5m")
    plan, c, _, _ = fixture()
    source = BarDevelopmentSource(config, c, replace(plan, max_training_rows=10))
    original = pl.LazyFrame.collect

    def guarded(lazy: pl.LazyFrame, *args: Any, **kwargs: Any) -> pl.DataFrame:
        frame = original(lazy, *args, **kwargs)
        assert frame.height <= 11
        return frame

    monkeypatch.setattr(pl.LazyFrame, "collect", guarded)
    with pytest.raises(ValueError, match="memory row cap"):
        source.training(datetime(2021, 1, 4, 10, tzinfo=UTC), datetime(2021, 1, 4, 14, tzinfo=UTC))
