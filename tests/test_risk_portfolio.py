"""Causal capability boundary, shared cash reconciliation and prefix invariance."""

from dataclasses import replace
from datetime import timedelta

import pytest

from risk_synth import T, configuration, health
from xauusd_quant.alpha_portfolio.allocation import Allocation
from xauusd_quant.alpha_portfolio.intents import Intent
from xauusd_quant.alpha_portfolio.portfolio import Portfolio, WeightUpdate
from xauusd_quant.execution.config import ExecutionConfig
from xauusd_quant.execution.engine import ExecutionEngine, MemoryRecorder, Quote
from xauusd_quant.risk.portfolio import RiskPortfolio


def quote(second: float, mid: float = 1800, spread: float = .2) -> Quote:
    at = T + timedelta(seconds=second)
    return Quote(at, at.replace(tzinfo=None), mid - spread / 2, mid + spread / 2, 0)


def intent(alpha: str, second: float, target: float, expiry: float = 30) -> Intent:
    at = T + timedelta(seconds=second)
    return Intent(alpha, f"{alpha}_V001", at, at, T + timedelta(seconds=expiry),
                  target, "active" if target else "flat", "synthetic", "5m", 1)


def make(*alphas: str, **changes: object) -> tuple[RiskPortfolio, MemoryRecorder, WeightUpdate]:
    sink = MemoryRecorder()
    cfg = configuration(*alphas, **changes)
    alphas = alphas or ("A", "B")
    p = RiskPortfolio(ExecutionConfig(quantity_lots=.02), {a: f"{a}_V001" for a in alphas}, .02, sink, cfg)
    weights = WeightUpdate(Allocation("EQUAL_V001", T, T - timedelta(seconds=1),
                                      {a: 1 / len(alphas) for a in alphas}, "equal", {}))
    for a in alphas:
        p.health_update(health(a))
    return p, sink, weights


def feed(p: RiskPortfolio, events: list[Quote | Intent | WeightUpdate]) -> None:
    for e in events:
        at = e.available_utc if isinstance(e, Intent) else e.timestamp_utc
        for a in p.specifications:
            p.health_update(health(a, (at - T).total_seconds()))
        p.consume([e])


def test_raw_strategy_and_private_submit_cannot_bypass_governed_account() -> None:
    p, sink, w = make("A")
    p.consume([w, quote(0)])
    with pytest.raises(PermissionError):
        p.engine.request_target(T, .03, "RAW")
    with pytest.raises(PermissionError):
        p.engine._submit(T, "RAW", "entry", 1, 100, .03)
    assert not sink.tables.get("orders")
    saved = p.engine.target_state()
    with pytest.raises(ValueError):
        ExecutionEngine.restore_target_state(p.config, saved, sink)


def test_raw_call_cannot_reuse_current_authorized_context() -> None:
    from risk_synth import market, request

    p, _, w = make("A")
    p.consume([w, quote(0)])
    p._active_decision = p.risk.evaluate(request(), market(), {"A": health()})
    try:
        with pytest.raises(PermissionError):
            p.engine.request_target(T, .02, "RAW_WITH_MATCHING_SIZE")
        with pytest.raises(PermissionError):
            p.engine.request_target(T, .03, "ENLARGED", risk_authority=p._authority)
    finally:
        p._active_decision = None
    assert p.engine.pending is None


def test_submission_guard_works_without_secondary_portfolio_checks() -> None:
    sink = MemoryRecorder()
    engine = ExecutionEngine(ExecutionConfig(), [], sink,
                             risk_authority=object(), risk_boundary=lambda *args: False)
    with pytest.raises(PermissionError):
        engine._submit(T, "BYPASS", "entry", 1, 100, .01)
    assert engine.pending is None and not sink.tables


def test_agreement_one_shared_account_and_reconciled_costs() -> None:
    p, sink, w = make()
    feed(p, [w, quote(0), intent("A", 1, 1), intent("B", 1, 1), quote(2),
             intent("A", 3, 0), intent("B", 3, 0), quote(4, 1802)])
    report = p.finish(T + timedelta(seconds=4))
    assert len(sink.tables["fills"]) == 2
    assert report["execution"]["closed_net_pnl_account"] == pytest.approx(3.4 - .0000009259259259259259)
    assert p.risk.state.account["signed_lots"] == 0
    assert not p.risk.state.reservations
    cash = sum(r["amount_account"] for r in sink.tables["cash_flows"])
    assert p.engine.cash - p.config.initial_cash_account == pytest.approx(cash)
    assert sum(a["cash_change_account"] for a in p.attribution.values()) == pytest.approx(cash)


def test_opposite_same_time_intents_net_and_cancel_confirmed_offline() -> None:
    p, sink, w = make()
    feed(p, [w, quote(0), intent("A", 1, 1), intent("B", 1, -1), quote(2)])
    assert not sink.tables.get("fills")
    assert p.engine.position is None and not p.risk.state.reservations
    assert any(r["status"] == "cancelled" for r in sink.tables["orders"])
    assert any(r["decision"] == "REQUEST_CANCEL" for r in sink.tables["risk_decisions"])


def test_pending_target_change_never_uses_previous_approval_for_enlargement() -> None:
    p, sink, w = make()
    feed(p, [w, quote(0), intent("A", 1, 1), intent("B", 1, 1), quote(2)])
    assert sink.tables["fills"][0]["quantity_lots"] == .02
    # The A-only pending approval (.01) was cancelled when B enlarged the target;
    # new .02 reservation was created before the later fill.
    entry = sink.tables["fills"][0]
    identity = p.order_intents[entry["order_id"]]
    assert p.risk.state.decisions[identity]["approved_change_lots"] == .02


def test_quote_gap_and_bad_health_prevent_pending_fill() -> None:
    p, sink, w = make("A")
    p.consume([w, quote(0), intent("A", 1, 1)])
    p.health_update(health("A", 2, forecast_value=float("nan")))
    p.consume([quote(2)])
    assert not sink.tables.get("fills")
    p, sink, w = make("A")
    feed(p, [w, quote(0), intent("A", 1, 1), quote(2, 1810)])
    assert not sink.tables.get("fills")
    assert any("FILL_EXCEEDS_RESERVED_NOTIONAL" in r["rules"] for r in sink.tables["risk_fill_rejections"])


def test_loss_halt_requests_close_but_failed_quote_does_not_mark_flat() -> None:
    p, sink, w = make("A", daily_loss_limit=1, drawdown_limit=2)
    feed(p, [w, quote(0), intent("A", 1, 1), quote(2), quote(3, 1700)])
    assert p.risk.state.halts and p.engine.position is not None
    assert p.engine.pending is not None and p.engine.pending.purpose == "exit"
    # A close request is not a fill; invalid next quote cannot make it flat.
    p.consume([quote(4, spread=-1)])
    assert p.engine.position is not None
    assert p.risk.summary()["unresolved_actions"]


def test_future_prefix_and_chunk_equality() -> None:
    events = [quote(0), intent("A", 1, 1), quote(2), intent("A", 3, 0), quote(4)]
    p, sink, w = make("A")
    feed(p, [w, *events[:3]])
    prefix = {k: list(v) for k, v in sink.tables.items()}
    feed(p, events[3:])
    for k, v in prefix.items():
        assert sink.tables[k][:len(v)] == v
    q, other, weight = make("A")
    feed(q, [weight, *events])
    assert q.risk.checkpoint() == p.risk.checkpoint()
    assert other.tables == sink.tables


def test_restart_requires_reconciliation_and_preserves_halt() -> None:
    p, sink, w = make("A")
    feed(p, [w, quote(0), intent("A", 1, 1), quote(2)])
    p.risk.kill(T + timedelta(seconds=2), "offline restart fixture")
    state = p.risk_state()
    q = RiskPortfolio.restore_risk(p.config, p.specifications, p.budget_lots, MemoryRecorder(),
                                  p.risk.configuration, state)
    assert q.risk.state.needs_reconciliation and q.risk.state.halts
    assert q.engine.position.quantity_lots == p.engine.position.quantity_lots
    q._sync_account(T + timedelta(seconds=2))
    assert not q.risk.state.needs_reconciliation and q.risk.state.state != "READY"
    with pytest.raises(PermissionError):
        q.engine.request_target(T + timedelta(seconds=2), .03, "RAW")
    with pytest.raises(ValueError):
        RiskPortfolio.restore_risk(p.config, p.specifications, p.budget_lots, sink,
                                   replace(p.risk.configuration, configuration_id="CHANGED_V002"), state)


def test_mismatched_execution_cost_or_currency_rejected() -> None:
    p, sink, _ = make("A")
    with pytest.raises(ValueError):
        RiskPortfolio(replace(p.config, commission_account_per_lot_per_leg=100), p.specifications,
                      p.budget_lots, sink, p.risk.configuration)
    assert isinstance(p, Portfolio)


def test_empty_universe_governed_but_unconfigured_is_inactive() -> None:
    from risk_synth import ROOT
    from xauusd_quant.risk.policy import load_configuration

    sink = MemoryRecorder()
    p = RiskPortfolio.from_alphas(ExecutionConfig(), (), T, .02, sink,
                                 risk_configuration=load_configuration(ROOT / "config/risk.yaml"))
    p.consume([quote(0), quote(1)])
    result = p.finish(T + timedelta(seconds=1))
    assert result["result"] == "NO_ELIGIBLE_ALPHAS" and p.risk.state.state == "UNCONFIGURED"
    assert not sink.tables.get("orders") and not sink.tables.get("fills")
    with pytest.raises(ValueError):
        RiskPortfolio.from_alphas(ExecutionConfig(), (), T, .02, sink)


def test_governed_scientific_entry_refuses_future_or_failed_eligibility() -> None:
    from test_alpha_portfolio_causality import candidate

    cfg = configuration("A")
    for alpha in (replace(candidate(), evidence_available_utc=T.isoformat()),
                  replace(candidate(), stage14_verdict="BLOCKED")):
        with pytest.raises(ValueError):
            RiskPortfolio.from_alphas(ExecutionConfig(), (alpha,), T, .02, MemoryRecorder(),
                                      risk_configuration=cfg)
