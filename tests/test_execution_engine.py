"""Hand-calculated Stage 12 prices, causal timing and state carried across chunks."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from xauusd_quant.execution.config import ExecutionConfig
from xauusd_quant.execution.engine import BarClose, ExecutionEngine, MemoryRecorder, Quote
from xauusd_quant.execution.policy import Forecast

T = datetime(2021, 1, 4, 10, tzinfo=UTC)


def cfg(**changes: Any) -> ExecutionConfig:
    return replace(
        ExecutionConfig(),
        timeframe="1m",
        quantity_lots=0.02,
        contract_ounces_per_lot=10,
        latency_ms=0,
        slippage_usd_per_ounce_per_leg=0,
        financing_long_account_per_lot_per_day=0,
        financing_short_account_per_lot_per_day=0,
        **changes,
    )


def q(
    seconds: float, bid: float = 100, ask: float = 102, sequence: int = 0, age: float = 0
) -> Quote:
    stamp = T + timedelta(seconds=seconds)
    return Quote(
        stamp,
        (stamp + timedelta(hours=2)).replace(tzinfo=None),
        bid,
        ask,
        sequence,
        stamp - timedelta(seconds=age),
    )


def f(c: ExecutionConfig, value: float | None = 0.001, index: int = 0) -> Forecast:
    return Forecast.from_bar(
        forecast_id=f"fixture-{index}",
        bar_open_utc=T + timedelta(seconds=index * 60),
        bar_index=index,
        value=value,
        config=c,
    )


def run(
    c: ExecutionConfig,
    forecasts: list[Forecast],
    events: list[Quote | BarClose],
    cutoff: float = 121,
    cuts: list[int] | None = None,
) -> tuple[dict[str, Any], MemoryRecorder]:
    record = MemoryRecorder()
    engine = ExecutionEngine(c, forecasts, record)
    indexes = [0, *(cuts or []), len(events)]
    for a, b in zip(indexes[:-1], indexes[1:], strict=True):
        engine.consume(events[a:b])
    return engine.finish(T + timedelta(seconds=cutoff)), record


def stream(exit_bid: float = 105, exit_ask: float = 107) -> list[Quote | BarClose]:
    return [
        q(0),
        BarClose(T + timedelta(seconds=60), 0),
        q(61),
        BarClose(T + timedelta(seconds=120), 1),
        q(121, exit_bid, exit_ask),
    ]


@pytest.mark.parametrize(
    ("value", "exit_bid", "exit_ask", "gross"), [(0.001, 105, 107, 0.6), (-0.001, 97, 99, 0.2)]
)
def test_long_and_short_quote_side_pnl_contract_and_commission(
    value: float, exit_bid: float, exit_ask: float, gross: float
) -> None:
    c = cfg()
    summary, record = run(c, [f(c, value)], stream(exit_bid, exit_ask))
    trade = record.tables["trades"][0]
    assert trade["gross_price_pnl_account"] == pytest.approx(gross)
    assert trade["commission_account"] == pytest.approx(0.12)
    assert trade["net_pnl_account"] == pytest.approx(gross - 0.12)
    # The executable spread is present exactly once, not subtracted from gross again.
    assert trade["matched_midpoint_pnl_account"] - trade["spread_cost_account"] == pytest.approx(
        gross
    )
    assert sum(r["amount_account"] for r in record.tables["cash_flows"]) == pytest.approx(
        gross - 0.12
    )
    assert summary["final_mark"]["equity_account"] == pytest.approx(
        c.initial_cash_account + gross - 0.12
    )
    assert summary["final_mark"]["reconciliation_error_account"] == pytest.approx(0, abs=1e-8)


def test_adverse_slippage_and_currency_conversion_are_not_double_counted() -> None:
    c = replace(
        cfg(),
        slippage_usd_per_ounce_per_leg=0.5,
        account_currency="THB",
        account_currency_per_usd=35,
    )
    _, record = run(c, [f(c)], stream())
    trade = record.tables["trades"][0]
    assert trade["entry_price"] == 102.5 and trade["exit_price"] == 104.5
    assert trade["gross_price_pnl_account"] == pytest.approx(14)
    assert trade["spread_cost_account"] == pytest.approx(14)
    assert trade["slippage_account"] == pytest.approx(7)
    assert trade["net_pnl_account"] == pytest.approx(13.88)


def test_latency_never_fills_an_earlier_or_same_arrival_quote() -> None:
    c = replace(cfg(), latency_ms=1000)
    events = [
        q(0),
        q(60, 1, 2, 1),
        BarClose(T + timedelta(seconds=60), 0),
        q(60.5, 2, 3, 2),
        q(61, 3, 4, 3),
        q(61, 4, 5, 4),
        q(61.001, 100, 102, 5),
    ]
    _, record = run(c, [f(c)], events, cutoff=62)
    fills = record.tables["fills"]
    assert len(fills) == 1
    assert fills[0]["sequence"] == 5 and fills[0]["price_usd_per_ounce"] == 102


def test_bar_close_and_computation_delay_determine_availability() -> None:
    c = replace(cfg(), computation_delay_ms=500)
    forecast = f(c)
    assert forecast.available_at_utc == T + timedelta(seconds=60.5)
    events = [
        q(0),
        BarClose(T + timedelta(seconds=60), 0),
        q(60.4, 2, 3, 1),
        q(60.5, 3, 4, 2),
        q(60.6, 100, 102, 3),
    ]
    _, record = run(c, [forecast], events, cutoff=61)
    assert record.tables["fills"][0]["sequence"] == 3


def test_same_time_quotes_preserve_original_sequence_without_favorable_selection() -> None:
    c = cfg()
    events = [q(0), BarClose(T + timedelta(seconds=60), 0), q(61, 100, 110, 1), q(61, 100, 101, 2)]
    _, record = run(c, [f(c)], events, cutoff=61)
    assert record.tables["fills"][0]["price_usd_per_ounce"] == 110
    with pytest.raises(ValueError, match="out of order"):
        run(c, [f(c)], [*events[:-2], events[-1], events[-2]], cutoff=61)


def test_expiry_is_exclusive_and_invalid_stale_quotes_cannot_fill() -> None:
    c = replace(cfg(), order_ttl_ms=1000)
    events = [
        q(0),
        BarClose(T + timedelta(seconds=60), 0),
        q(60.2, 102, 100, 1),
        q(60.3, 100, 102, 2, age=2),
        q(60.4, float("nan"), 102, 3),
        q(61, 100, 102, 4),
    ]
    summary, record = run(c, [f(c)], events, cutoff=61)
    assert summary["counts"]["expired"] == 1
    assert summary["counts"]["invalid_quotes"] == 2
    assert summary["counts"]["stale_quotes"] == 1
    assert "fills" not in record.tables
    assert record.tables["orders"][-1]["reason"] == "no_eligible_quote_before_expiry"


def test_session_gap_cancels_pending_order_and_retains_exposure() -> None:
    c = replace(cfg(), max_gap_ms=1000, order_ttl_ms=120_000)
    summary, record = run(c, [f(c)], stream(), cutoff=121)
    assert summary["counts"]["unfilled"] == 1
    assert "fills" not in record.tables
    assert record.tables["orders"][-1]["reason"] == "session_or_data_gap"
    c = replace(c, max_gap_ms=100_000)
    events = stream()[:-1] + [q(1000, 105, 107)]
    summary, record = run(c, [f(c)], events, cutoff=1000)
    assert summary["open_position"] is not None
    assert len(record.tables["fills"]) == 1
    assert summary["counts"]["expired"] == 1


def test_horizon_counts_observed_bars_instead_of_interpolating_across_closures() -> None:
    c = replace(cfg(), max_gap_ms=3 * 86400 * 1000)
    events = [
        q(0),
        BarClose(T + timedelta(seconds=60), 0),
        q(61),
        q(200),
        BarClose(T + timedelta(days=1), 1),
        q(86401, 105, 107),
    ]
    _, record = run(c, [f(c)], events, cutoff=86401)
    assert record.tables["trades"][0]["exit_utc"] == T + timedelta(seconds=86401)


@pytest.mark.parametrize("value", [0.001, -0.001])
def test_open_position_liquidation_side_mark_and_stale_mark_unknown(value: float) -> None:
    c = cfg()
    summary, _ = run(c, [f(c, value)], stream()[:3], cutoff=61)
    assert summary["final_mark"]["unrealized_price_pnl_account"] == pytest.approx(-0.4)
    assert summary["final_mark"]["equity_account"] == pytest.approx(c.initial_cash_account - 0.46)
    stale, _ = run(c, [f(c, value)], stream()[:3], cutoff=63)
    assert stale["final_mark"]["equity_account"] is None
    assert stale["open_position"] is not None


def test_financing_is_charged_overnight_including_a_session_gap() -> None:
    c = replace(
        cfg(),
        quantity_lots=1,
        contract_ounces_per_lot=1,
        financing_long_account_per_lot_per_day=2,
        financing_short_account_per_lot_per_day=-1,
        max_gap_ms=2 * 86400 * 1000,
    )
    events = [
        q(0),
        BarClose(T + timedelta(seconds=60), 0),
        q(61),
        BarClose(T + timedelta(seconds=86460), 1),
        q(86461, 105, 107),
    ]
    _, record = run(c, [f(c)], events, cutoff=86461)
    trade = record.tables["trades"][0]
    assert trade["financing_account"] == pytest.approx(2)
    assert trade["net_pnl_account"] == pytest.approx(3 - 6 - 2)
    _, record = run(c, [f(c, -0.001)], events, cutoff=86461)
    assert record.tables["trades"][0]["financing_account"] == pytest.approx(-1)


def test_chunk_and_month_boundaries_do_not_reset_any_state() -> None:
    c = cfg()
    events = stream()
    whole, records = run(c, [f(c)], events)
    chunked, chunk_records = run(c, [f(c)], events, cuts=[1, 2, 3, 4])
    assert whole == chunked
    assert records.tables == chunk_records.tables


def test_future_append_preserves_earlier_decisions_and_fills() -> None:
    c = cfg()
    events = stream()
    _, prefix = run(c, [f(c)], events)
    _, extended = run(
        c,
        [f(c), f(c, -0.001, index=2)],
        [
            *events,
            BarClose(T + timedelta(seconds=180), 2),
            q(181, 1000, 1002),
            BarClose(T + timedelta(seconds=240), 3),
            q(241, 10, 12),
        ],
        cutoff=241,
    )
    assert prefix.tables["decisions"] == extended.tables["decisions"][:1]
    assert prefix.tables["fills"] == extended.tables["fills"][:2]


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"value": float("nan")}, "nonfinite_forecast"),
        ({"value": None}, "nonfinite_forecast"),
        ({"units": "log_volatility"}, "unsupported_target_or_units"),
        ({"horizon_bars": 5}, "incompatible_horizon_or_timeframe"),
        ({"status": "ENSEMBLE_INVALID"}, "invalid_forecast_status_or_provenance"),
        ({"available_at_utc": T}, "forecast_before_bar_close"),
    ],
)
def test_invalid_forecasts_are_rejected_explicitly(change: dict[str, Any], reason: str) -> None:
    c = cfg()
    summary, record = run(c, [replace(f(c), **change)], stream())
    assert "fills" not in record.tables
    assert summary["counts"]["decisions_rejected"] == 1
    assert record.tables["decisions"][0]["reason"] == reason


def test_neutral_overlap_and_end_orders_are_not_silently_dropped() -> None:
    c = cfg()
    _, record = run(c, [f(c, 0)], stream())
    assert record.tables["decisions"][0]["status"] == "neutral"
    summary, record = run(c, [f(c), replace(f(c), forecast_id="overlap")], stream())
    assert summary["counts"]["decisions_rejected"] == 1
    assert record.tables["decisions"][1]["reason"] == "position_or_order_already_active"
    summary, record = run(c, [f(c)], stream()[:2], cutoff=60)
    assert summary["counts"]["unfilled"] == 1
    assert record.tables["orders"][-1]["reason"] == "end_of_run"


def test_cost_sensitivity_on_matched_trades_has_exact_arithmetic() -> None:
    base = cfg()
    costly = replace(base, commission_account_per_lot_per_leg=5, slippage_usd_per_ounce_per_leg=0.1)
    _, left = run(base, [f(base)], stream())
    _, right = run(costly, [f(costly)], stream())
    a, b = left.tables["trades"][0], right.tables["trades"][0]
    assert a["entry_utc"] == b["entry_utc"] and a["exit_utc"] == b["exit_utc"]
    assert a["net_pnl_account"] - b["net_pnl_account"] == pytest.approx(
        2 * 0.02 * 2 + 2 * 0.1 * 0.2
    )


def test_drawdown_turnover_and_unannualized_equity_return_have_declared_units() -> None:
    c = replace(cfg(), initial_cash_account=100)
    events = stream()
    events.insert(3, q(100, 94, 96))
    summary, _ = run(c, [f(c)], events)
    assert summary["max_drawdown_account"] == pytest.approx(1.66)
    assert summary["turnover_account"] == pytest.approx((102 + 105) * 0.2)
    assert summary["exposure_lot_seconds"] == pytest.approx(0.02 * 60)
    assert summary["cutoff_equity_return"] == pytest.approx(0.48 / 100)


def test_small_per_tick_financing_does_not_disappear_against_large_cash() -> None:
    c = replace(
        cfg(),
        quantity_lots=1,
        contract_ounces_per_lot=1,
        financing_long_account_per_lot_per_day=2,
        initial_cash_account=1e9,
        max_gap_ms=2 * 86400 * 1000,
    )
    events = [q(0), BarClose(T + timedelta(seconds=60), 0), q(61)]
    events += [q(61 + i / 1000) for i in range(1, 1001)]
    events += [BarClose(T + timedelta(seconds=86460), 1), q(86461, 105, 107)]
    summary, record = run(c, [f(c)], events, cutoff=86461)
    assert record.tables["trades"][0]["financing_account"] == pytest.approx(2)
    assert summary["final_mark"]["cash_account"] == pytest.approx(1e9 - 5, abs=1e-7, rel=0)
