"""Hand-calculated fold-boundary execution through the Stage 12 engine."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from strategy_synth import at, fixture
from xauusd_quant.execution.engine import BarClose, Quote
from xauusd_quant.execution.policy import Forecast
from xauusd_quant.strategy_validation.metrics import aggregate_folds
from xauusd_quant.strategy_validation.plan import Scenario
from xauusd_quant.strategy_validation.policies import stress_quotes
from xauusd_quant.strategy_validation.runs import execute_trial


def quote(minutes: float, bid: float, ask: float, sequence: int) -> Quote:
    stamp = at(minutes)
    return Quote(stamp, stamp.replace(tzinfo=None), bid, ask, sequence)


def test_fold_boundary_carries_position_costs_and_rejects_duplicate_exposure(
    tmp_path: Path,
) -> None:
    plan, c, _, _ = fixture()
    plan = replace(plan, horizon_bars=2)
    c = replace(c, horizon_bars=2, latency_ms=0)
    closes = [BarClose(at(900 + 5 * i), i) for i in range(25)]
    f = Forecast.from_bar(
        forecast_id="F1-entry", bar_open_utc=at(950), bar_index=11, value=0.001, config=c
    )
    overlap = Forecast.from_bar(
        forecast_id="F2-overlap", bar_open_utc=at(955), bar_index=12, value=-0.001, config=c
    )
    events = [
        quote(900, 100, 102, 0),
        quote(955 + 1 / 60, 100, 102, 1),
        quote(965 + 1 / 60, 105, 107, 2),
    ]
    result = execute_trial(
        plan,
        c,
        Scenario("BASE"),
        [f, overlap],
        closes,
        lambda _: events,
        0,
        "forecast",
        tmp_path / "trial",
    )
    assert result["execution"]["counts"]["fills"] == 2
    assert result["execution"]["counts"]["decisions_rejected"] == 1
    assert result["folds"][0]["end_mark"]["open_quantity_lots"] == 0.01
    assert result["folds"][1]["start_mark"] == result["folds"][0]["end_mark"]
    expected = 3 - 0.04 - 0.06 - 2 * 0.01 * 600 / 86400
    assert result["execution"]["closed_net_pnl_account"] == pytest.approx(expected)
    assert result["aggregate"]["marked_equity_change_account"] == pytest.approx(expected)
    assert sum(r["cash_change_account"] for r in result["folds"]) == pytest.approx(expected)
    fills = [json.loads(line) for line in (tmp_path / "trial/fills.jsonl").read_text().splitlines()]
    assert len(fills) == 2
    assert fills[0]["timestamp_utc"] > f.available_at_utc.isoformat()
    reset = copy.deepcopy(result["folds"])
    reset[1]["start_mark"] = dict(reset[1]["start_mark"])
    reset[1]["start_mark"]["cash_account"] = c.initial_cash_account
    with pytest.raises(ValueError, match="conceal a reset"):
        aggregate_folds(reset, c.initial_cash_account)


def test_spread_stress_preserves_clocks_sequence_and_invalid_quotes() -> None:
    quotes = [quote(901, 100, 102, 7), quote(902, 103, 102, 8)]
    stressed = list(stress_quotes(quotes, Scenario("STRESS", spread_multiplier=2)))
    assert stressed[0].bid == 99 and stressed[0].ask == 103
    assert stressed[0].timestamp_utc == quotes[0].timestamp_utc
    assert stressed[0].sequence == 7
    assert stressed[1] == quotes[1]


def test_invalid_availability_is_rejected_before_policy_and_execution(tmp_path: Path) -> None:
    plan, c, source, quotes = fixture()
    rows, closes = source.evaluation(plan.start, plan.end)
    row = rows[0]
    f = Forecast.from_bar(
        forecast_id="leak",
        bar_open_utc=row.bar_open_utc,
        bar_index=row.bar_index,
        value=0.001,
        config=c,
    )
    f = replace(f, available_at_utc=f.bar_open_utc)
    # Choose an input within the run but before its actual bar close.
    row = next(r for r in rows if r.bar_open_utc > plan.start)
    f = replace(
        f, bar_open_utc=row.bar_open_utc, available_at_utc=row.bar_open_utc, bar_index=row.bar_index
    )
    result = execute_trial(
        plan, c, Scenario("BASE"), [f], closes, quotes, 0, "forecast", tmp_path / "bad"
    )
    assert result["execution"]["counts"]["decisions_rejected"] == 1
    assert result["execution"]["counts"].get("fills", 0) == 0
