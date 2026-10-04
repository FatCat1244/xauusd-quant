"""Reservations, uncertain outcomes, persistent losses and explicit offline rearm."""

from copy import deepcopy
from dataclasses import replace
from datetime import timedelta

import pytest

from risk_synth import T, account, configuration, health, market, request
from test_risk_decisions import ready
from xauusd_quant.execution.config import content_hash
from xauusd_quant.risk.contracts import ExecutionUpdate
from xauusd_quant.risk.engine import RiskEngine


def update(status: str, filled: float = 0, second: float = 1,
           identity: str = "INTENT_1", event_id: str | None = None) -> ExecutionUpdate:
    at = T + timedelta(seconds=second)
    return ExecutionUpdate(event_id or f"EVENT_{status}_{second}", identity, at, at, status, filled, "ORDER_1")


def test_idempotent_intents_and_acknowledgements() -> None:
    e = ready()
    first = e.evaluate(request(), market(), {"A": health()})
    assert e.evaluate(request(), market(), {"A": health()}) == first
    assert len(e.state.reservations) == 1 and len(e.state.order_times) == 1
    ack = update("acknowledged")
    e.execution_update(ack)
    count = e.state.sequence
    e.execution_update(ack)
    assert e.state.sequence == count
    with pytest.raises(ValueError):
        e.evaluate(request(target=.01, second=1), market(1), {"A": health(second=1)})
    assert "INTENT_ID_COLLISION" in e.state.halts


def test_opposing_pending_is_not_netting_credit() -> None:
    e = ready()
    e.evaluate(request(), market(), {"A": health()})
    e.execution_update(update("acknowledged"))
    d = e.evaluate(request("SECOND", -.02, 1), market(1), {"A": health(second=1)})
    assert d["decision"] == "REQUEST_CANCEL" and d["approved_change_lots"] == 0
    assert d["measurements"]["reserved_absolute_lots"] == .02
    assert len(e.state.reservations) == 1


def test_cancel_request_retains_risk_and_late_fill_requires_account() -> None:
    e = ready()
    e.evaluate(request(), market(), {"A": health()})
    e.execution_update(update("acknowledged"))
    e.execution_update(update("cancel_requested", second=2))
    assert e.summary()["reserved_lots"] == .02
    e.execution_update(update("filled", .02, 3))
    assert e.summary()["reserved_lots"] == .02
    assert not e.observe_account(account(3, signed_lots=0), reconcile=True)
    assert "POSITION_RECONCILIATION_FAILED" in e.state.halts
    assert e.observe_account(account(4, signed_lots=.02, open_estimated_loss=37), reconcile=True)
    assert not e.state.reservations
    # A retained severe halt needs explicit rearm, even after successful sync.
    assert e.state.state != "READY"


def test_terminal_cancel_requires_confirmation_and_reconciliation() -> None:
    e = ready()
    e.evaluate(request(), market(), {"A": health()})
    e.execution_update(update("cancel_requested"))
    assert e.state.reservations
    e.execution_update(update("cancelled", second=2))
    assert e.state.reservations
    assert e.observe_account(account(2), reconcile=True)
    assert not e.state.reservations
    with pytest.raises(ValueError):
        e.execution_update(update("filled", .02, 3))
    assert e.state.state == "RECONCILIATION_REQUIRED"


def test_unknown_and_partial_outcomes_block_without_guessing_flat() -> None:
    for status, filled in (("unknown", 0), ("partial", .01)):
        e = ready()
        e.evaluate(request(), market(), {"A": health()})
        e.execution_update(update(status, filled))
        assert not e.observe_account(account(1, signed_lots=filled), reconcile=True)
        assert e.state.reservations and e.state.needs_reconciliation
        d = e.evaluate(request("NEXT", 0, 1), market(1), {})
        assert not d["approved_change_lots"]


def test_partial_then_cancelled_reconciles_residual_position() -> None:
    e = ready()
    e.evaluate(request(), market(), {"A": health()})
    e.execution_update(update("partial", .01))
    e.execution_update(update("cancelled", .01, 2))
    assert e.observe_account(account(2, signed_lots=.01), reconcile=True)
    assert not e.state.reservations and e.state.account["signed_lots"] == .01
    assert "PARTIAL_EXECUTION" in e.state.halts


def test_loss_costs_external_flows_daily_reset_and_hwm_persist() -> None:
    e = ready()
    assert e.observe_account(account(1, balance=9980, equity=9975), reconcile=True)
    assert e.state.daily_loss == 25 and "DAILY_LOSS" in e.state.halts
    assert e.observe_account(account(2, balance=10980, equity=10975, external_flow_total=1000), reconcile=True)
    assert e.state.daily_loss == 25 and e.state.drawdown == 25
    assert e.observe_account(account(3, balance=10960, equity=10960, external_flow_total=1000), reconcile=True)
    assert "EQUITY_DRAWDOWN" in e.state.halts
    restored = RiskEngine.restore(e.configuration, e.record, e.checkpoint())
    assert restored.state.needs_reconciliation and restored.state.high_water == 10000
    assert set(restored.state.halts) == {"DAILY_LOSS", "EQUITY_DRAWDOWN"}
    assert restored.observe_account(account(86401, balance=10960, equity=10960, external_flow_total=1000), reconcile=True)
    assert restored.state.daily_loss == 0 and restored.state.drawdown == 40
    assert "EQUITY_DRAWDOWN" in restored.state.halts and restored.state.state != "READY"
    with pytest.raises(ValueError):
        restored.rearm(T + timedelta(seconds=86401), operator="offline tester", reason="new day")


def test_explicit_rearm_requires_fresh_recovered_account_and_identity() -> None:
    e = ready()
    e.kill(T, "manual fixture")
    with pytest.raises(ValueError):
        e.rearm(T, operator="", reason="")
    e.rearm(T, operator="offline tester", reason="fixture issue resolved")
    assert e.state.state == "READY" and not e.state.halts
    e.kill(T, "again")
    with pytest.raises(ValueError):
        e.rearm(T + timedelta(seconds=6), operator="tester", reason="stale")


def test_restart_never_ready_and_incompatible_or_corrupt_state_refused() -> None:
    e = ready()
    e.evaluate(request(), market(), {"A": health()})
    cp = e.checkpoint()
    frozen = deepcopy(cp)
    r = RiskEngine.restore(e.configuration, e.record, cp)
    assert r.state.state == "RECONCILIATION_REQUIRED" and r.state.reservations
    assert r.evaluate(request("NEW", .02, 1), market(1), {"A": health(second=1)})["approved_change_lots"] == 0
    assert cp == frozen
    bad = deepcopy(cp)
    bad["body"]["drawdown"] = 0.5
    with pytest.raises(ValueError):
        RiskEngine.restore(e.configuration, e.record, bad)
    bad["content_sha256"] = content_hash(bad["body"])
    bad["body"]["reservations"]["INTENT_1"]["signed_change"] = 0
    bad["content_sha256"] = content_hash(bad["body"])
    with pytest.raises(ValueError):
        RiskEngine.restore(e.configuration, e.record, bad)
    cfg = configuration(drawdown_limit=41)
    with pytest.raises(ValueError):
        RiskEngine.restore(cfg, e.record, cp)


def test_reduction_does_not_reverse_duplicate_or_guess_unknown_account() -> None:
    e = ready()
    e.observe_account(account(.1, signed_lots=.02, open_estimated_loss=37), reconcile=True)
    e.kill(T + timedelta(seconds=.1), "test")
    d = e.evaluate(request(second=.1, target=.01), market(.1), {})
    assert d["decision"] == "REQUEST_REDUCE_OR_FLATTEN" and d["approved_change_lots"] == -.02
    duplicate = e.evaluate(request("REDUCE_2", .0, .1), market(.1), {})
    assert duplicate["approved_change_lots"] == 0
    e = ready()
    e.observe_account(account(.1, signed_lots=.02, open_estimated_loss=37), reconcile=True)
    d = e.evaluate(request("REVERSE", -.02, .1), market(.1), {"A": health(second=.1)})
    assert d["approved_target_lots"] == 0 and d["approved_change_lots"] == -.02
    e = ready()
    e.observe_account(account(.1, signed_lots=.02, open_estimated_loss=37), reconcile=True)
    d = e.evaluate(request("STALE", -.02, 6), market(6), {})
    assert d["approved_change_lots"] == 0


def test_stale_reduction_and_failed_execution_remain_unresolved() -> None:
    e = ready()
    e.observe_account(account(.1, signed_lots=.02, open_estimated_loss=37), reconcile=True)
    e.kill(T + timedelta(seconds=.1), "emergency")
    d = e.evaluate(request(target=0, second=.1), market(-10), {})
    assert d["approved_change_lots"] == 0 and e.summary()["unresolved_actions"]
    assert e.state.account["signed_lots"] == .02
    e.observe_account(account(1, signed_lots=.02, execution_available=False), reconcile=True)
    d = e.evaluate(request("FAILED", 0, 1), market(1), {})
    assert d["approved_change_lots"] == 0 and "EXECUTION_UNAVAILABLE" in d["rules"]


def test_out_of_order_and_invalid_lifecycle_fail_closed() -> None:
    e = ready()
    e.evaluate(request(), market(), {"A": health()})
    e.execution_update(update("acknowledged", second=2))
    with pytest.raises(ValueError):
        e.execution_update(update("filled", .02, 1))
    assert "OUT_OF_ORDER_RECEIPT" in e.state.halts
    e = ready()
    e.evaluate(request(), market(), {"A": health()})
    with pytest.raises(ValueError):
        e.execution_update(update("partial", .03))
    assert "INVALID_EXECUTION_UPDATE" in e.state.halts
    e = ready()
    assert not e.observe_account(account(1, event_utc=T - timedelta(seconds=1)), reconcile=True)


def test_changed_configuration_cannot_erase_halt() -> None:
    e = ready()
    e.kill(T, "severe")
    cfg = replace(e.configuration, configuration_id="RECONFIGURED_V002")
    with pytest.raises(ValueError):
        RiskEngine.restore(cfg, e.record, e.checkpoint())


def test_mutating_nested_settings_and_decision_cannot_enlarge_approval() -> None:
    e = ready()
    d = e.evaluate(request(), market(), {"A": health()})
    d["sleeve_lots"]["A"] = 10
    d["approved_change_lots"] = 10
    assert e.state.decisions["INTENT_1"]["approved_change_lots"] == .02
    assert e.state.decisions["INTENT_1"]["sleeve_lots"]["A"] == .02
    e.configuration.policy.alpha_contracts["A"]["model"] = "BYPASS"
    with pytest.raises(ValueError):
        e.evaluate(request("NEW", .03), market(), {"A": health()})
    assert e.state.state == "UNCONFIGURED"


def test_reduction_rate_limit_retains_unresolved_exposure() -> None:
    e = ready(max_orders_per_window=1)
    e.observe_account(account(.1, signed_lots=.02), reconcile=True)
    e.state.order_times = [T.isoformat()]
    d = e.evaluate(request(target=0, second=.1), market(.1), {})
    assert d["approved_change_lots"] == 0 and "ORDER_RATE_LIMIT" in d["rules"]
    assert e.state.account["signed_lots"] == .02 and e.state.pending_actions
