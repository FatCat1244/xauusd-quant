"""Real converter/public guarded reader path, before price-value collection."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from conftest import make_clean_rows, write_csv
from econometric_synth import fixture
from xauusd_quant.data.converter import TickConverter
from xauusd_quant.data.resampler import BarResampler
from xauusd_quant.econometrics.data import GridData
from xauusd_quant.strategy_validation.source import BarDevelopmentSource


def test_econometric_public_reader_excludes_future_values_before_collection(
    config_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    start = datetime(2021, 5, 3, 12)
    ticks = []
    for i in range(32):
        ticks += make_clean_rows(start + timedelta(minutes=5 * i, seconds=1), 1, bid0=2000 + i)
    config = config_factory(
        write_csv(tmp_path / "ticks.csv", ticks),
        timezone={"mode": "fixed_offset", "fixed_offset_hours": 2, "emit_utc_column": True},
        resampling={"timeframes": ["5m"]},
    )
    TickConverter(config).run()
    BarResampler(config).build("5m")
    plan, execution, _ = fixture()
    source = BarDevelopmentSource(config, execution, plan.validation_plan)
    first = datetime(2021, 5, 3, 10, tzinfo=UTC)
    cutoff = first + timedelta(minutes=90)
    collect = pl.LazyFrame.collect
    observed = []

    def guard(lazy: pl.LazyFrame, *args: Any, **kwargs: Any) -> pl.DataFrame:
        frame = collect(lazy, *args, **kwargs)
        if "close" in frame.columns:
            observed.append(frame.height)
            assert frame.height == 17, "future econometric prices materialized"
            assert all(t + timedelta(minutes=5) < cutoff for t in frame["timestamp_utc"])
        return frame

    monkeypatch.setattr(pl.LazyFrame, "collect", guard)
    opens, prices = source.completed_bars(first, cutoff)
    data = GridData(opens, prices, 300)
    samples = data.samples(first, cutoff, 1, 0)
    assert observed == [17] and len(samples) == 4
    assert all(s.outcome.matured_utc < cutoff for s in samples)
    with pytest.raises(ValueError, match="2022"):
        source.completed_bars(first, datetime(2022, 1, 2, tzinfo=UTC))
    assert observed == [17]
