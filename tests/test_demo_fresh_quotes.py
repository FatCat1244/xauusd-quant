"""Quote movement is rerisked; hard guards, reservation and full history survive."""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from scripts import run_unsubmitted_demo_smoke as operator_tool

from demo_synth import DemoNative, create, propose
from test_demo_operator_rearm import quote_abort
from xauusd_quant.demo import broker as broker_module
from xauusd_quant.demo.broker import NativeDemoBroker, Permit, approval_state
from xauusd_quant.demo.coordinator import Coordinator
from xauusd_quant.demo.journal import Journal
from xauusd_quant.demo.revalidation import evaluate_quote
from xauusd_quant.execution.config import content_hash


def test_moving_quotes_complete_fake_lifecycle_with_reserved_envelope(tmp_path: Path) -> None:
    c, b, j = create(tmp_path)
    try:
        c.fresh_quotes = True
        entry = propose(c, b)
        r = c.risk.state.reservations[entry["intent_id"]]
        assert r.estimated_loss == 40 and r.margin == 9950
        original = deepcopy(c.risk.state.decisions)
        b.changed_snapshot = True
        c.submit(entry["intent_id"], b.snapshot(), b.now)
        assert c.risk.state.decisions == original
        assert c.reconcile(b.snapshot(), b.now)["reconciled"]
        assert c.risk.state.account["open_estimated_loss"] == 40
        close = propose(c, b, close=True)
        c.submit(close["intent_id"], b.snapshot(), b.now)
        assert c.reconcile(b.snapshot(), b.now)["verified_flat"]
        assert len(b.sent) == 2 and not c.risk.state.reservations
        assert sum(row["event"] == "quote_revalidation" for row in j.rows) == 2
    finally:
        j.close()


@pytest.mark.parametrize("fault", ["spread", "stale", "future", "budget", "nan", "cash", "permission", "instrument", "expired"])
def test_fresh_quote_does_not_bypass_limits(tmp_path: Path, fault: str) -> None:
    c, b, j = create(tmp_path)
    try:
        c.fresh_quotes = True
        entry = propose(c, b)
        snapshot = b.snapshot()
        if fault == "spread":
            snapshot["quote"]["ask"] += 1
        elif fault == "stale":
            snapshot["quote"]["time_msc"] -= 60000
        elif fault == "future":
            snapshot["quote"]["time_msc"] += 10000
        elif fault == "budget":
            snapshot["quote"].update(bid=2100., ask=2100.1)
        elif fault == "nan":
            snapshot["quote"]["ask"] = float("nan")
        elif fault == "cash":
            snapshot["account"]["balance"] += 1
        elif fault == "permission":
            snapshot["permissions"] = False
        elif fault == "instrument":
            snapshot["symbol"]["trade_contract_size"] = 10
        else:
            b.now += timedelta(seconds=5)
            snapshot["received_utc"] = b.now.isoformat()
        with pytest.raises((ValueError, RuntimeError)):
            c.submit(entry["intent_id"], snapshot, b.now)
        assert not b.sent and entry["intent_id"] in c.risk.state.reservations
    finally:
        j.close()


@pytest.mark.parametrize("fault", ["resize", "unknown", "filled", "halt", "other_pending", "small_envelope", "missing", "configuration", "turnover"])
def test_full_stage16_and_reservation_required(tmp_path: Path, fault: str) -> None:
    c, b, j = create(tmp_path)
    try:
        c.fresh_quotes = True
        entry = propose(c, b)
        payload = c._quote_payload(entry)
        assert payload is not None
        checkpoint = payload["checkpoint"]
        body = checkpoint["body"]
        r = body["reservations"][entry["intent_id"]]
        if fault == "resize":
            payload["signed_lots"] *= 2
        elif fault == "unknown":
            r["status"] = "unknown"
        elif fault == "filled":
            r["cumulative_filled"] = .01
        elif fault == "halt":
            body["halts"].append("DAILY_LOSS")
        elif fault == "other_pending":
            body["reservations"]["OTHER"] = {**r, "intent_id": "OTHER"}
            body["decisions"]["OTHER"] = body["decisions"][entry["intent_id"]]
        elif fault == "small_envelope":
            r["estimated_loss"] = .01
        elif fault == "missing":
            body["reservations"].clear()
        elif fault == "configuration":
            payload["configuration"] = None
        else:
            body["turnover"] = [[b.now.isoformat(), .20]]
        checkpoint["content_sha256"] = content_hash(body)
        with pytest.raises((ValueError, RuntimeError)):
            evaluate_quote(payload, b.snapshot()["quote"], b.snapshot()["account"], b.now)
        assert c.risk.state.reservations[entry["intent_id"]].status == "approved"
        assert not b.sent
    finally:
        j.close()


def test_native_rechecks_full_risk_on_latest_quote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c, b, j = create(tmp_path)
    try:
        c.fresh_quotes = True
        entry = propose(c, b)
        native = DemoNative(Path(str(c.terminal.terminal_path)))
        native.rows[-1].update(b.snapshot()["quote"])
        native.order_calc_margin = lambda side, symbol, volume, price: volume * price * 5
        native.order_calc_profit = lambda side, symbol, volume, start, end: (end - start) * volume * 100 * (1 if side == 0 else -1)

        class Clock(datetime):
            @classmethod
            def now(cls, tz: object = None) -> datetime:
                return b.now

        monkeypatch.setattr(broker_module, "datetime", Clock)
        native_broker = NativeDemoBroker(c.config, c.terminal, native=native)
        authority = object()
        native_broker.bind(authority)
        native_broker.connect()
        request = entry["request"]
        permit = Permit(authority, content_hash(request), b.now + timedelta(seconds=5),
            approval_state(b.snapshot(), closing=False), 5, c._quote_payload(entry))
        native.rows[-1]["bid"] += .1
        native.rows[-1]["ask"] += .1
        checked = native_broker.check(request, permit)
        assert checked["risk_revalidation"]["decision"] == "APPROVE"
        native.rows[-1]["bid"] += .1
        native.rows[-1]["ask"] += .1
        sent = native_broker.send(request, permit)
        assert sent["risk_revalidation"]["approved_change_lots"] == .02
        assert native.calls.count("order_send") == 1
        original_margin = native.order_calc_margin

        def slow_margin(*args: object) -> float:
            b.now += timedelta(milliseconds=400)
            return original_margin(*args)

        native.order_calc_margin = slow_margin
        with pytest.raises(ValueError, match="quote"):
            native_broker.check(request, replace(permit, max_quote_age_seconds=.2))
        native.order_calc_margin = original_margin
        assert native.calls.count("order_check") == 1
        native.rows[-1]["ask"] += 2
        with pytest.raises(ValueError, match="risk"):
            native_broker.check(request, permit)
        assert native.calls.count("order_check") == 1
        native.account.balance += 1
        with pytest.raises(ValueError, match="state"):
            native_broker.send(request, permit)
        assert native.calls.count("order_send") == 1
        native_broker.shutdown()
    finally:
        j.close()


def test_known_unsubmitted_code_migration_keeps_history(tmp_path: Path) -> None:
    c, b, j = create(tmp_path, code=operator_tool.LEGACY_CODE)
    try:
        quote_abort(c, b)
        c.halt("EXPLICIT_REARM_SMOKE_ABORTED", b.now)
        c.fresh_quotes = True
        old = c.risk.state.high_water, c.risk.state.daily_start, c.risk.state.order_times.copy()
        assert operator_tool.resume_code(j, "NEW_CODE", fresh_quotes=True) == operator_tool.LEGACY_CODE
        operator_tool.rearm_unsubmitted(c, b.snapshot(), b.now, operator="OPERATOR", new_code="NEW_CODE")
        assert c.code == "NEW_CODE" and not c.halts and not b.sent
        assert old == (c.risk.state.high_water, c.risk.state.daily_start, c.risk.state.order_times)
        assert any(row["event"] == "execution_code_migration" for row in j.rows)
    finally:
        j.close()


@pytest.mark.parametrize("fault", ["source", "submission", "response", "tail", "halt", "position"])
def test_checkpoint_migration_never_erases_unsafe_state(tmp_path: Path, fault: str) -> None:
    c, b, j = create(tmp_path, code=operator_tool.LEGACY_CODE)
    try:
        quote_abort(c, b)
        c.fresh_quotes = True
        if fault == "source":
            c.code = "UNKNOWN_SOURCE"
        elif fault == "submission":
            next(iter(c.intents.values()))["submission_utc"] = b.now.isoformat()
        elif fault == "response":
            c._record("broker_response", {})
        elif fault == "halt":
            c.halt("DAILY_LOSS", b.now)
        c.persist()
        if fault == "tail":
            c._record("incomplete", {})
        if fault in {"source", "submission", "response", "tail"}:
            with pytest.raises(ValueError):
                operator_tool.resume_code(j, "NEW_CODE", fresh_quotes=True)
        else:
            snapshot = b.snapshot()
            if fault == "position":
                snapshot["positions"] = [{"volume": .01, "type": 0}]
            with pytest.raises((ValueError, RuntimeError)):
                operator_tool.rearm_unsubmitted(c, snapshot, b.now, operator="OPERATOR", new_code="NEW_CODE")
        assert c.code != "NEW_CODE" and c.halts and not b.sent
    finally:
        j.close()


def test_restart_preserves_envelope_without_arming(tmp_path: Path) -> None:
    c, b, j = create(tmp_path)
    c.fresh_quotes = True
    entry = propose(c, b)
    expected = c.risk.state.reservations[entry["intent_id"]].estimated_loss
    j.close()
    b.authority = None
    with_journal = Journal(tmp_path / "runtime")
    try:
        restored = Coordinator(c.config, c.terminal, c.risk.configuration, b, with_journal,
            "RESTART_V001", c.code, offline_synthetic=True, fresh_quotes=True)
        assert restored.risk.state.reservations[entry["intent_id"]].estimated_loss == expected
        assert not restored.armed and restored.risk.state.needs_reconciliation
        with pytest.raises(ValueError, match="arming"):
            restored.initialize(b.snapshot(), b.now, arm=True)
        assert not b.sent
    finally:
        with_journal.close()


def test_disabled_fresh_risk_guard_is_detected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Coordinator, "_fresh_risk", lambda *args: None)
    with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
        test_fresh_quote_does_not_bypass_limits(tmp_path, "budget")


def test_operator_fresh_workflow_migrates_and_closes_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    c, b, j = create(tmp_path, code=operator_tool.LEGACY_CODE)
    quote_abort(c, b)
    c.halt("EXPLICIT_REARM_SMOKE_ABORTED", b.now)
    config, terminal, risk = c.config, c.terminal, c.risk.configuration
    j.close()
    b.authority = None
    b.changed_snapshot = True
    monkeypatch.setattr(operator_tool, "load_demo_config", lambda path: config)
    monkeypatch.setattr(operator_tool, "load_shadow_config", lambda path: terminal)
    monkeypatch.setattr(operator_tool, "load_configuration", lambda path: risk)
    monkeypatch.setattr(operator_tool, "readiness", lambda *args: {"blocking_reasons": []})
    monkeypatch.setattr(operator_tool, "account_directory", lambda *args: tmp_path / "runtime")
    monkeypatch.setattr(operator_tool, "DemoProcessBroker", lambda *args: b)

    def synthetic_coordinator(*args: object, **kwargs: object) -> Coordinator:
        return Coordinator(*args, offline_synthetic=True, **kwargs)

    monkeypatch.setattr(operator_tool, "Coordinator", synthetic_coordinator)
    result = operator_tool.run(tmp_path, tmp_path / "unused.yaml", "FRESH_SMOKE_V001",
                               "OPERATOR", fresh_quotes=True)
    assert result["status"] == "VERIFIED_DEMO_LIFECYCLE"
    assert result["entry_submissions"] == result["close_submissions"] == 1
    assert result["verified_flat"] and not result["reservations"]
    assert b.shutdown_count == 1 and len(b.sent) == 2


def test_quote_receipt_clock_is_sampled_after_api_retrieval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    c, b, j = create(tmp_path)
    try:
        entry = propose(c, b)
        native = DemoNative(Path(str(c.terminal.terminal_path)))
        native.rows[-1].update(b.snapshot()["quote"])

        class Clock(datetime):
            @classmethod
            def now(cls, tz: object = None) -> datetime:
                return b.now

        original_tick = native.symbol_info_tick

        def arriving_tick(symbol: str) -> object:
            b.now += timedelta(milliseconds=100)
            native.rows[-1]["time_msc"] = int(b.now.timestamp() * 1000)
            return original_tick(symbol)

        monkeypatch.setattr(broker_module, "datetime", Clock)
        monkeypatch.setattr(native, "symbol_info_tick", arriving_tick)
        native_broker = NativeDemoBroker(c.config, c.terminal, native=native)
        authority = object()
        native_broker.bind(authority)
        native_broker.connect()
        request = entry["request"]
        permit = Permit(authority, content_hash(request), b.now + timedelta(seconds=5),
                        approval_state(b.snapshot(), closing=False), 5)
        assert native_broker.check(request, permit) == {"retcode": 0}
        assert native.calls.count("order_check") == 1 and "order_send" not in native.calls
        native_broker.shutdown()
    finally:
        j.close()
