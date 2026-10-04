"""Hand-check shared execution, netting, attribution and restart invariants."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from xauusd_quant.alpha_portfolio.allocation import Allocation
from xauusd_quant.alpha_portfolio.intents import Intent
from xauusd_quant.alpha_portfolio.portfolio import Portfolio, WeightUpdate, event_key
from xauusd_quant.execution.config import ExecutionConfig
from xauusd_quant.execution.engine import MemoryRecorder, Quote

T = datetime(2021, 1, 4, tzinfo=UTC)
CONFIG = ExecutionConfig(
    quantity_lots=0.02,
    financing_long_account_per_lot_per_day=0,
    financing_short_account_per_lot_per_day=0,
)


def quote(second: float, mid: float = 1800, sequence: int = 0) -> Quote:
    at = T + timedelta(seconds=second)
    return Quote(at, at.replace(tzinfo=None), mid - 0.1, mid + 0.1, sequence)


def intent(
    alpha: str, second: float, target: float, until: float = 30, state: str | None = None
) -> Intent:
    at = T + timedelta(seconds=second)
    return Intent(
        alpha,
        f"{alpha}_V001",
        at,
        at,
        T + timedelta(seconds=until),
        target,
        state or ("active" if target else "flat"),
        "synthetic fixture",
        "5m",
        1,
    )


def make(
    *alphas: str, config: ExecutionConfig = CONFIG
) -> tuple[Portfolio, MemoryRecorder, WeightUpdate]:
    sink = MemoryRecorder()
    portfolio = Portfolio(config, {a: f"{a}_V001" for a in alphas}, 0.02, sink)
    allocation = Allocation(
        "EQUAL_V001", T, T - timedelta(seconds=1), {a: 1 / len(alphas) for a in alphas}, "equal", {}
    )
    return portfolio, sink, WeightUpdate(allocation)


def test_empty_universe_inactive() -> None:
    p, sink, w = make()
    p.consume([w, quote(0), quote(2)])
    result = p.finish(T + timedelta(seconds=3))
    assert result["result"] == "NO_ELIGIBLE_ALPHAS"
    assert result["execution"]["final_mark"]["cash_account"] == 10000
    assert "fills" not in sink.tables


def test_opposing_intents_net_before_any_fill() -> None:
    p, sink, w = make("A", "B")
    p.consume([w, quote(0), intent("A", 1, 1), intent("B", 1, -1), quote(2)])
    assert p.target_lots == 0
    assert not sink.tables.get("fills")
    last = sink.tables["portfolio_decisions"][-1]
    assert last["gross_intended_lots"] == 0.02
    assert last["cancelled_intent_lots"] == 0.02


def test_agreement_one_shared_fill_and_hand_cash_costs() -> None:
    p, sink, w = make("A", "B")
    p.consume(
        [
            w,
            quote(0),
            intent("A", 1, 1),
            intent("B", 1, 1),
            quote(2),
            intent("A", 3, 0),
            intent("B", 3, 0),
            quote(4, 1802),
        ]
    )
    result = p.finish(T + timedelta(seconds=4))
    assert len(sink.tables["fills"]) == 2
    assert sink.tables["fills"][0]["quantity_lots"] == 0.02
    assert result["execution"]["closed_gross_price_pnl_account"] == pytest.approx(3.52)
    assert result["execution"]["closed_net_pnl_account"] == pytest.approx(3.4)
    assert result["execution"]["total_commission_account"] == pytest.approx(0.12)
    assert result["execution"]["final_mark"]["cash_account"] == pytest.approx(10003.4)
    for metric, total in result["attribution_totals"].items():
        assert sum(v.get(metric, 0) for v in result["attribution"].values()) == pytest.approx(total)
    assert result["attribution"]["A"]["net_pnl_account"] == pytest.approx(1.7)


def test_variable_target_resize_and_reversal_full_close_then_later_reentry() -> None:
    p, sink, w = make("A", "B")
    p.consume(
        [
            w,
            quote(0),
            intent("A", 1, 1),
            quote(2),
            intent("B", 3, 1),
            quote(4),
            quote(5),
            intent("A", 6, -1),
            intent("B", 6, -1),
            quote(7),
            quote(8),
        ]
    )
    fills = sink.tables["fills"]
    assert [f["purpose"] for f in fills] == ["entry", "exit", "entry", "exit", "entry"]
    assert [f["quantity_lots"] for f in fills] == [0.01, 0.01, 0.02, 0.02, 0.02]
    assert p.engine.position is not None and p.engine.position.direction == -1
    assert fills[2]["timestamp_utc"] > fills[1]["timestamp_utc"]


def test_pending_target_cancelled_and_no_same_arrival_fill() -> None:
    p, sink, w = make("A", config=replace(CONFIG, latency_ms=1000))
    p.consume(
        [w, quote(0), intent("A", 1, 1), quote(2), intent("A", 2.5, -1), quote(3.5), quote(4)]
    )
    assert len(sink.tables["fills"]) == 1
    assert sink.tables["fills"][0]["direction"] == -1
    assert any(o["status"] == "cancelled" for o in sink.tables["orders"])


def test_expiry_at_same_timestamp_quote_cancels_entry() -> None:
    p, sink, w = make("A")
    p.consume([w, quote(0), intent("A", 1, 1, until=2), quote(2)])
    assert not sink.tables.get("fills")
    assert p.target_lots == 0


def test_invalid_withdrawal_requests_exit_and_retains_actual_until_fill() -> None:
    p, sink, w = make("A")
    p.consume([w, quote(0), intent("A", 1, 1), quote(2), intent("A", 3, 0, state="invalid")])
    assert p.target_lots == 0 and p.engine.position is not None
    assert p.engine.pending is not None and p.engine.pending.purpose == "exit"
    p.consume([quote(4)])
    assert p.engine.position is None
    assert len(sink.tables["fills"]) == 2


def test_rejected_invalid_quote_expired_order_retried_without_invented_fill() -> None:
    p, sink, w = make("A", config=replace(CONFIG, order_ttl_ms=500))
    p.consume([w, quote(0), intent("A", 1, 1), replace(quote(1.2), bid=1801, ask=1800), quote(2)])
    assert not sink.tables.get("fills")
    assert p.engine.pending is not None
    p.consume([quote(2.2)])
    assert len(sink.tables["fills"]) == 1
    assert p.engine.counts["invalid_quotes"] == 1
    assert p.engine.counts["expired"] == 1


@pytest.mark.parametrize("split", [3, 5, 7])
def test_state_reload_chunks_same_records_and_funding(split: int) -> None:
    config = replace(CONFIG, financing_long_account_per_lot_per_day=2)
    p, sink, w = make("A", "B", config=config)
    events = [
        w,
        quote(0),
        intent("A", 1, 1),
        intent("B", 1, 1),
        quote(2),
        intent("A", 3, 0),
        intent("B", 3, 0),
        quote(4, 1802),
    ]
    p.consume(events)
    expected = p.finish(T + timedelta(seconds=5))
    q, other, _ = make("A", "B", config=config)
    q.consume(events[:split])
    q = Portfolio.restore(config, q.specifications, 0.02, other, q.state())
    q.consume(events[split:])
    assert q.finish(T + timedelta(seconds=5)) == expected
    assert other.tables == sink.tables
    with pytest.raises(ValueError, match="configuration"):
        Portfolio.restore(config, q.specifications, 0.01, other, q.state())


def test_future_append_does_not_change_earlier_decisions_weights_fills() -> None:
    p, sink, w = make("A")
    prefix = [w, quote(0), intent("A", 1, 1), quote(2)]
    p.consume(prefix)
    q, other, _ = make("A")
    q.consume([*prefix, intent("A", 3, -1), quote(4), quote(5)])
    for table in ("portfolio_decisions", "allocation_updates", "fills"):
        assert other.tables[table][: len(sink.tables[table])] == sink.tables[table]


def test_deterministic_ties_and_wrong_specification_refused() -> None:
    p, _, w = make("A", "B")
    events = [intent("B", 1, 1), quote(0), w, intent("A", 1, 1)]
    p.consume(sorted(events, key=event_key))
    with pytest.raises(ValueError, match="order"):
        p.consume([quote(0)])
    with pytest.raises(ValueError, match="specification"):
        p.consume([replace(intent("A", 2, 1), specification_id="WRONG_V001")])


def test_unrealized_owner_pnl_and_financing_reconcile() -> None:
    p, sink, w = make("A", "B", config=replace(CONFIG, financing_long_account_per_lot_per_day=5))
    p.consume([w, quote(0), intent("A", 1, 1), intent("B", 1, 1), quote(2), quote(3, 1801)])
    result = p.finish(T + timedelta(seconds=3))
    attribution = sink.tables["sleeve_equity_attribution"][-1]
    assert sum(attribution["pnl"].values()) == pytest.approx(
        result["execution"]["final_mark"]["equity_account"] - 10000
    )
