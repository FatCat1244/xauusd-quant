"""Scenario causality, matched cash flows and synthetic engine reuse."""

from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from xauusd_quant.execution.engine import ExecutionEngine, MemoryRecorder, Quote
from xauusd_quant.execution.io import DiskRecorder
from xauusd_quant.robustness.reports import read_records
from xauusd_quant.robustness.stress import (
    OperationalScenario,
    matched_cost_check,
    monte_carlo_scenarios,
)
from xauusd_quant.robustness.studies import fit_predictions, synthetic_fixture
from xauusd_quant.strategy_validation.plan import Scenario
from xauusd_quant.strategy_validation.runs import TrialLedger, execute_trial


def test_engine_stress_costs_and_first_eligible_quote_ordering(tmp_path: Path) -> None:
    plan, base, source, quotes = synthetic_fixture(2, 0.5)
    with DiskRecorder(tmp_path) as sink:
        forecasts, closes, _ = fit_predictions(plan, source, TrialLedger(sink, 20))
    results = []
    for i, scenario in enumerate(
        (
            OperationalScenario("BASE"),
            OperationalScenario("COST", commission_multiplier=2, extra_slippage=0.02),
        )
    ):
        config = scenario.configuration(base)
        directory = tmp_path / f"replay{i}"
        execute_trial(
            plan, config, Scenario("BASE"), forecasts, closes, quotes, 0.0, "forecast", directory
        )
        results.append((read_records(directory / "trades.jsonl", 2000), config))
        fills = read_records(directory / "fills.jsonl", 2000)
        orders = {
            r["order_id"]: r
            for r in read_records(directory / "orders.jsonl", 2000)
            if r["status"] == "submitted"
        }
        for fill in fills:
            order = orders[fill["order_id"]]
            assert order["arrival_utc"] < fill["timestamp_utc"] < order["expires_utc"]
    checked = matched_cost_check(results[0][0], results[1][0], results[0][1], results[1][1])
    assert checked["status"] == "verified" and checked["maximum_error_account"] < 1e-7
    assert sum(t["net_pnl_account"] for t in results[1][0]) < sum(
        t["net_pnl_account"] for t in results[0][0]
    )
    assert matched_cost_check(results[0][0], [], base, base)["status"] == "unavailable"


def test_feed_interruption_never_creates_quotes_or_favorable_fills() -> None:
    plan, base, _, quotes = synthetic_fixture(4, 0.5)
    start, end = plan.start + timedelta(seconds=300), plan.start + timedelta(seconds=330)
    scenario = OperationalScenario("INTERRUPTION", interruption_start=start, interruption_end=end)
    original = list(quotes(base))
    altered = list(scenario.quotes(original))
    assert len(altered) < len(original)
    assert all(not start <= q.timestamp_utc < end for q in altered)
    assert [q for q in original if not start <= q.timestamp_utc < end] == altered
    record = MemoryRecorder()
    engine = ExecutionEngine(base, [], record)
    engine.consume(altered)
    assert engine.finish(plan.end)["counts"].get("fills", 0) == 0


def test_spread_sides_invalid_quotes_and_scenario_reproducibility() -> None:
    plan, base, _, quotes = synthetic_fixture(4, 0.0)
    original = list(quotes(base))[:2]
    scenario = OperationalScenario("SPREAD", spread_multiplier=2)
    wider = list(scenario.quotes(original))
    assert all(a.bid < b.bid <= b.ask < a.ask for a, b in zip(wider, original, strict=True))
    assert all(
        a.timestamp_utc == b.timestamp_utc and a.sequence == b.sequence
        for a, b in zip(wider, original, strict=True)
    )
    bad = Quote(plan.start, plan.start.replace(tzinfo=None), 2.0, 1.0, 0)
    assert list(scenario.quotes([bad])) == [bad]
    with pytest.raises(ValueError, match="invalid executable"):
        list(
            OperationalScenario("EXTREME", spread_multiplier=100).quotes(
                [replace(bad, bid=0.1, ask=1)]
            )
        )
    assert monte_carlo_scenarios(4, 5) == monte_carlo_scenarios(4, 5)
    with pytest.raises(ValueError):
        monte_carlo_scenarios(999, 5)
