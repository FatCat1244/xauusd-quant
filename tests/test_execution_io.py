"""Tick source sequence, broker DST, month boundaries and the actual CLI path."""

from __future__ import annotations

import heapq
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import yaml

from conftest import make_clean_rows, write_csv
from xauusd_quant.data.converter import TickConverter
from xauusd_quant.data.resampler import BarResampler
from xauusd_quant.data.timezones import to_utc_expr
from xauusd_quant.execution.config import ExecutionConfig
from xauusd_quant.execution.engine import ExecutionEngine, MemoryRecorder
from xauusd_quant.execution.io import bar_closes, forecast_records, local_time, tick_quotes
from xauusd_quant.execution.readiness import inventory, sha256
from xauusd_quant.execution.runs import _event_key, _preflight, diagnostic_or_backtest
from xauusd_quant.utils.config import Config, load_config


@pytest.mark.parametrize(
    "source,expected",
    [
        (datetime(2021, 1, 15, 12), datetime(2021, 1, 15, 10, tzinfo=UTC)),
        (datetime(2021, 7, 15, 12), datetime(2021, 7, 15, 9, tzinfo=UTC)),
        # US has switched DST while Europe has not: broker time follows US rules.
        (datetime(2021, 3, 15, 12), datetime(2021, 3, 15, 9, tzinfo=UTC)),
        (datetime(2021, 3, 12, 12), datetime(2021, 3, 12, 10, tzinfo=UTC)),
    ],
)
def test_stored_broker_clock_is_not_reinterpreted_as_utc(
    source: datetime, expected: datetime
) -> None:
    config = load_config()
    derived = pl.DataFrame({"timestamp": [source]}).select(to_utc_expr(config.timezone))[
        "timestamp_utc"
    ][0]
    assert derived == expected
    assert local_time(expected, config) == source


def fixture(
    config_factory: Any, tmp_path: Path
) -> tuple[Config, ExecutionConfig, datetime, datetime, Path, Path]:
    begin = datetime(2021, 1, 31, 23, 58)
    rows = []
    for i in range(4):
        rows += make_clean_rows(
            begin + timedelta(minutes=i, seconds=1), 3, step_ms=1000, bid0=2000 + i
        )
    source = write_csv(tmp_path / "ticks.csv", rows)
    config = config_factory(
        source,
        timezone={"mode": "fixed_offset", "fixed_offset_hours": 2, "emit_utc_column": True},
        resampling={"timeframes": ["1m"]},
    )
    config = replace(config, project_root=tmp_path)
    TickConverter(config).run()
    BarResampler(config).build("1m")
    c = replace(ExecutionConfig(), timeframe="1m", batch_rows=2, max_gap_ms=120_000)
    start = datetime(2021, 1, 31, 21, 58, tzinfo=UTC)
    end = start + timedelta(minutes=4)
    path = tmp_path / "forecasts.parquet"
    pl.DataFrame(
        {
            "timestamp": [begin],
            "expected_return": [0.001],
            "status_expected_return_vol_scaled": ["ok"],
        }
    ).write_parquet(path)
    tick_manifest = json.loads((config.processed_data_path / "_manifest.json").read_text())
    bar_manifest = json.loads((config.bars_dir("1m") / "_manifest.json").read_text())
    metadata = {
        "forecast_id": "synthetic-month-boundary",
        "forecast_sha256": sha256(path),
        "timeframe": "1m",
        "target": "future_return",
        "units": "log_mid_return",
        "horizon_bars": 1,
        "timestamp_convention": "broker_local_bar_open",
        "dataset_version": tick_manifest["dataset_version"],
        "bar_dataset_version": bar_manifest["bar_dataset_version"],
        "feature_set_ids": [],
        "model_provenance": "synthetic fixture only",
        "feature_provenance": "no fitted features",
        "evaluation_history_classification": "historical_inspected",
    }
    sidecar = tmp_path / "forecasts.json"
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    return config, c, start, end, path, sidecar


def test_chunked_tick_reader_and_state_cross_months(config_factory: Any, tmp_path: Path) -> None:
    config, c, start, end, path, sidecar = fixture(config_factory, tmp_path)
    quotes = list(tick_quotes(config, c, start, end))
    full = list(tick_quotes(config, replace(c, batch_rows=1000), start, end))
    assert quotes == full
    assert quotes[0].timestamp_local.month == 1 and quotes[-1].timestamp_local.month == 2
    metadata = json.loads(sidecar.read_text())
    record = MemoryRecorder()
    engine = ExecutionEngine(c, forecast_records(path, metadata, config, c, start, end), record)
    events = list(heapq.merge(quotes, bar_closes(config, c, start, end), key=_event_key))
    # The entry and subsequent-bar exit are on opposite sides of the source partition boundary.
    for event in events:
        engine.consume([event])
    summary = engine.finish(end)
    assert summary["counts"]["closed_positions"] == 1
    assert record.tables["trades"][0]["entry_local"] == datetime(2021, 1, 31, 23, 59, 1)
    assert record.tables["trades"][0]["exit_local"] == datetime(2021, 2, 1, 0, 0, 1)
    facts = inventory(config)
    assert facts["data_metadata_status"] == "passed"
    assert facts["partitions"] == 2 and facts["actual_footer_rows"] == 12


def test_end_to_end_historical_scenarios_are_frozen_before_execution(
    config_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xauusd_quant.execution import runs

    config, c, start, end, path, sidecar = fixture(config_factory, tmp_path)
    scenarios = tmp_path / "scenarios.yaml"
    scenarios.write_text(
        yaml.safe_dump(
            [
                {"scenario_id": "BASE_V001"},
                {
                    "scenario_id": "COST_CONTROL_V001",
                    "commission_account_per_lot_per_leg": 0,
                    "slippage_usd_per_ounce_per_leg": 0,
                },
                {"scenario_id": "DELAY_V001", "latency_ms": 500},
            ]
        ),
        encoding="utf-8",
    )
    original_ticks, original_forecasts = runs.tick_quotes, runs.forecast_records

    def registered() -> None:
        manifest_path = tmp_path / "results/execution/FIXTURE_BACKTEST_V001/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert len(manifest["all_evaluated_variants"]) == 3

    def guarded_ticks(*args: Any) -> Any:
        registered()
        return original_ticks(*args)

    def guarded_forecasts(*args: Any) -> Any:
        registered()
        return original_forecasts(*args)

    monkeypatch.setattr(runs, "tick_quotes", guarded_ticks)
    monkeypatch.setattr(runs, "forecast_records", guarded_forecasts)
    result = diagnostic_or_backtest(
        config,
        c,
        "FIXTURE_BACKTEST_V001",
        start,
        end,
        forecast_path=path,
        metadata_path=sidecar,
        historical=True,
        scenario_set=scenarios,
    )
    assert result["promotion_status"] == "historical_diagnostic_only"
    assert result["scenarios"][0]["counts"]["fills"] == 2
    assert len(result["scenarios"]) == 3
    assert all(row["counts"]["fills"] == 2 for row in result["scenarios"])
    manifest = json.loads((Path(result["output"]) / "manifest.json").read_text())
    assert (
        manifest["all_evaluated_variants"][0]["config"]
        == replace(c, scenario_id="BASE_V001").resolved()
    )
    assert manifest["readiness_gates"]["untouched_evaluation"] == "failed"


def test_readiness_refusal_precedes_any_quote_or_forecast_values(
    config_factory: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xauusd_quant.execution import runs

    config, c, start, end, path, sidecar = fixture(config_factory, tmp_path)

    def forbidden(*args: Any) -> None:
        pytest.fail("unready real-data run opened executable values")

    monkeypatch.setattr(runs, "tick_quotes", forbidden)
    monkeypatch.setattr(runs, "forecast_records", forbidden)
    with pytest.raises(ValueError, match="readiness unresolved"):
        diagnostic_or_backtest(
            config, c, "REFUSED_V001", start, end, forecast_path=path, metadata_path=sidecar
        )


def test_unknown_forecast_availability_and_hash_fail_preflight(
    config_factory: Any, tmp_path: Path
) -> None:
    config, c, start, end, path, sidecar = fixture(config_factory, tmp_path)
    metadata = json.loads(sidecar.read_text())
    metadata["timestamp_convention"] = "bar_open_is_signal_time"
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="availability convention"):
        diagnostic_or_backtest(
            config,
            c,
            "BAD_AVAILABILITY_V001",
            start,
            end,
            forecast_path=path,
            metadata_path=sidecar,
            historical=True,
        )
    metadata["timestamp_convention"] = "broker_local_bar_open"
    metadata["forecast_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="forecast file hash"):
        diagnostic_or_backtest(
            config,
            c,
            "BAD_HASH_V001",
            start,
            end,
            forecast_path=path,
            metadata_path=sidecar,
            historical=True,
        )


def test_unrelated_hashed_file_does_not_verify_scientific_and_specification_evidence(
    config_factory: Any, tmp_path: Path
) -> None:
    config, c, start, end, path, sidecar = fixture(config_factory, tmp_path)
    metadata = json.loads(sidecar.read_text())
    unrelated = tmp_path / "notes.txt"
    unrelated.write_text("No scientific audit or broker specification", encoding="utf-8")
    metadata["evidence_files"] = {"notes.txt": sha256(unrelated)}
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    _, gates = _preflight(config, c, path, sidecar, start, end, inventory(config))
    assert gates["evidence_identity"] == "unknown"
