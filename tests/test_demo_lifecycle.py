"""Broker-confirmed lifecycle, accounting and uncertainty; vendor package unused."""
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import pytest

from demo_synth import create, propose
from xauusd_quant.demo.broker import Permit
from xauusd_quant.demo.coordinator import Coordinator
from xauusd_quant.demo.journal import Journal


def test_reservation_and_attempt_are_durable_before_broker_call(tmp_path: Path) -> None:
    c, b, j = create(tmp_path)
    try:
        entry = propose(c, b)
        original = b.send

        def checked_send(request: dict[str, object], permit: Permit) -> object:
            saved = j.rows[-1]
            assert saved["event"] == "checkpoint"
            intent = saved["data"]["intents"][entry["intent_id"]]
            assert intent["state"] == "SUBMISSION_ATTEMPTED"
            assert intent["submission_utc"]
            assert saved["data"]["risk"]["body"]["reservations"]
            return original(request, permit)

        b.send = checked_send  # type: ignore[method-assign]
        c.submit(entry["intent_id"], b.snapshot(), b.now)
        assert len(b.sent) == 1
    finally:
        j.close()


def test_recovery_after_original_runtime_can_close_but_never_reenter(tmp_path: Path) -> None:
    c, b, j = create(tmp_path)
    entry = propose(c, b)
    c.submit(entry["intent_id"], b.snapshot(), b.now)
    c.reconcile(b.snapshot(), b.now)
    c.halt("FAILED_SHUTDOWN", b.now)
    cfg, terminal, risk = c.config, c.terminal, c.risk.configuration
    j.close()
    b.now += timedelta(seconds=120)
    b.authority = None
    reopened = Journal(tmp_path / "runtime")
    try:
        restored = Coordinator(cfg, terminal, risk, b, reopened, "RECOVERY_V001", "SYNTHETIC_CODE", offline_synthetic=True)
        restored.initialize(b.snapshot(), b.now, arm=False)
        assert not restored.armed
        close = propose(restored, b, close=True, name="RECOVER_CLOSE_V001")
        restored.submit(close["intent_id"], b.snapshot(), b.now)
        assert restored.reconcile(b.snapshot(), b.now)["verified_flat"]
        with pytest.raises(ValueError):
            restored.initialize(b.snapshot(), b.now, arm=True)
        assert restored.risk.state.halts
        assert len(b.sent) == 2
    finally:
        reopened.close()


def test_successful_check_only_restores_history_before_explicit_smoke_arm(tmp_path: Path) -> None:
    c, b, j = create(tmp_path)
    entry = propose(c, b)
    c.submit(entry["intent_id"], b.snapshot(), b.now, precheck_only=True)
    assert c.reconcile(b.snapshot(), b.now)["verified_flat"]
    original_hwm = c.risk.state.high_water
    cfg, terminal, risk = c.config, c.terminal, c.risk.configuration
    j.close()
    b.authority = None
    reopened = Journal(tmp_path / "runtime")
    try:
        restored = Coordinator(cfg, terminal, risk, b, reopened, "SMOKE_AFTER_CHECK_V001", "SYNTHETIC_CODE", offline_synthetic=True)
        assert not restored.armed
        restored.initialize(b.snapshot(), b.now, arm=True)
        assert restored.armed and restored.risk.state.high_water == original_hwm
        new = propose(restored, b, name="ENTRY_AFTER_CHECK_V001")
        restored.submit(new["intent_id"], b.snapshot(), b.now)
        assert len(b.sent) == 1
    finally:
        reopened.close()


def test_hand_checked_entry_closure_cash_and_distinct_tickets(tmp_path: Path) -> None:
    c, b, j = create(tmp_path)
    try:
        entry = propose(c, b)
        assert entry["state"] == "RISK_APPROVED"
        assert c.risk.state.reservations
        c.submit(entry["intent_id"], b.snapshot(), b.now)
        assert c.intents[entry["intent_id"]]["state"] == "ACCEPTED_OR_PENDING"
        assert c.risk.state.reservations  # A DONE response is not a fill/account mark.
        c.reconcile(b.snapshot(), b.now)
        assert c.intents[entry["intent_id"]]["state"] == "FILLED"
        assert not c.risk.state.reservations
        assert b.positions[0]["ticket"] != b.deals[0]["order"] != b.deals[0]["ticket"]
        assert b.balance == pytest.approx(10000 - .02 * 3)
        b.now += timedelta(seconds=1)
        b.bid += 2
        b.ask += 2
        close = propose(c, b, close=True)
        assert close["request"]["position"] == 9001
        assert close["position_identifier"] == 8001
        assert close["request"]["type"] == 1
        c.submit(close["intent_id"], b.snapshot(), b.now)
        result = c.reconcile(b.snapshot(), b.now)
        assert result["verified_flat"]
        assert b.balance == pytest.approx(10000 + (1801.9 - 1800.1) * .02 * 100 - 2 * .02 * 3)
        assert c.summary()["broker_reported_cash"] == pytest.approx(3.48)
        assert c.summary()["verified_entry_deals"] == c.summary()["verified_close_deals"] == 1
        assert c.summary()["verified_flat"]
    finally:
        j.close()


@pytest.mark.parametrize("mode", ["timeout_after_fill", "exception_after_fill"])
def test_timeout_later_deal_discovery_no_duplicate(tmp_path: Path, mode: str) -> None:
    c, b, j = create(tmp_path, mode=mode)
    try:
        entry = propose(c, b)
        c.submit(entry["intent_id"], b.snapshot(), b.now)
        assert c.intents[entry["intent_id"]]["state"] == "UNKNOWN"
        assert c.risk.state.reservations[entry["intent_id"]].status == "unknown"
        assert not c.summary()["verified_flat"]
        c.submit(entry["intent_id"], b.snapshot(), b.now)
        assert len(b.sent) == 1
        c.reconcile(b.snapshot(), b.now)
        assert c.intents[entry["intent_id"]]["state"] == "FILLED"
        assert not c.risk.state.reservations
        assert c.risk.state.halts  # Discovery does not erase the halt.
        b.mode = "full"
        close = propose(c, b, close=True)
        c.submit(close["intent_id"], b.snapshot(), b.now)
        assert c.reconcile(b.snapshot(), b.now)["verified_flat"]
    finally:
        j.close()


def test_accepted_without_fill_and_delayed_history_retain_reservation(tmp_path: Path) -> None:
    c, b, j = create(tmp_path, mode="accepted")
    try:
        entry = propose(c, b)
        c.submit(entry["intent_id"], b.snapshot(), b.now)
        c.reconcile(b.snapshot(), b.now)
        assert c.intents[entry["intent_id"]]["state"] == "ACCEPTED_OR_PENDING"
        assert c.risk.state.reservations
        assert c.summary()["verified_deals"] == 0
        assert not c.summary()["verified_flat"]
        b.orders = []
        b.execute(entry["request"])
        b.show_history = False
        c.reconcile(b.snapshot(), b.now)
        assert c.risk.state.reservations
        b.show_history = True
        c.reconcile(b.snapshot(), b.now)
        assert c.summary()["verified_entry_deals"] == 1
        assert not c.risk.state.reservations
    finally:
        j.close()


def test_terminal_partial_fill_only_releases_after_account_reconciliation(tmp_path: Path) -> None:
    c, b, j = create(tmp_path, mode="partial")
    try:
        entry = propose(c, b)
        c.submit(entry["intent_id"], b.snapshot(), b.now)
        assert c.risk.state.reservations
        c.reconcile(b.snapshot(), b.now)
        assert c.intents[entry["intent_id"]]["state"] == "CLOSED_OR_CANCELLED"
        assert c.intents[entry["intent_id"]]["filled_lots"] == .01
        assert not c.risk.state.reservations
        assert c.risk.state.halts
        b.mode = "full"
        close = propose(c, b, close=True)
        assert close["request"]["volume"] == .01
        c.submit(close["intent_id"], b.snapshot(), b.now)
        assert c.reconcile(b.snapshot(), b.now)["verified_flat"]
    finally:
        j.close()


@pytest.mark.parametrize("code", [None, 10009, 10019, 10027])
def test_check_success_code_distinct_and_failed_check_never_sends(tmp_path: Path, code: int | None) -> None:
    c, b, j = create(tmp_path)
    try:
        b.check_code = code
        entry = propose(c, b)
        assert c.submit(entry["intent_id"], b.snapshot(), b.now)["state"] == "REJECTED"
        assert not b.sent
        assert c.reconcile(b.snapshot(), b.now)["verified_flat"]
    finally:
        j.close()


def test_precheck_success_can_be_followed_by_rejection(tmp_path: Path) -> None:
    c, b, j = create(tmp_path, mode="reject")
    try:
        entry = propose(c, b)
        c.submit(entry["intent_id"], b.snapshot(), b.now)
        assert len(b.sent) == 1
        assert c.summary()["verified_deals"] == 0
        assert c.reconcile(b.snapshot(), b.now)["verified_flat"]
    finally:
        j.close()


@pytest.mark.parametrize("checkpoint", ["before_submit", "before_call", "after_call"])
def test_restart_reconciles_without_resubmission(tmp_path: Path, checkpoint: str) -> None:
    c, b, j = create(tmp_path)
    entry = propose(c, b)
    if checkpoint != "before_submit":
        c.intents[entry["intent_id"]].update(state="SUBMISSION_ATTEMPTED", submission_utc=b.now.isoformat())
        c.persist()
    if checkpoint == "after_call":
        b.execute(entry["request"])
    cfg, terminal, risk = c.config, c.terminal, c.risk.configuration
    j.close()
    b.authority = None
    reopened = Journal(tmp_path / "runtime")
    try:
        restored = Coordinator(cfg, terminal, risk, b, reopened, "RESTART_V001", "SYNTHETIC_CODE", offline_synthetic=True)
        restored.initialize(b.snapshot(), b.now, arm=False)
        assert not restored.armed
        assert not b.sent
        if checkpoint == "before_call":
            assert restored.intents[entry["intent_id"]]["state"] == "UNKNOWN"
            assert restored.risk.state.reservations
        elif checkpoint == "after_call":
            assert restored.intents[entry["intent_id"]]["state"] == "FILLED"
            assert restored.summary()["verified_entry_deals"] == 1
        else:
            assert restored.intents[entry["intent_id"]]["state"] == "REJECTED"
            assert not restored.risk.state.reservations
        with pytest.raises(ValueError, match="arming"):
            restored.initialize(b.snapshot(), b.now, arm=True)
    finally:
        reopened.close()


def test_external_activity_and_failed_exit_never_report_flat(tmp_path: Path) -> None:
    c, b, j = create(tmp_path)
    try:
        entry = propose(c, b)
        c.submit(entry["intent_id"], b.snapshot(), b.now)
        c.reconcile(b.snapshot(), b.now)
        b.mode = "none_without_fill"
        close = propose(c, b, close=True)
        c.submit(close["intent_id"], b.snapshot(), b.now)
        c.reconcile(b.snapshot(), b.now)
        assert b.positions and c.risk.state.reservations
        assert not c.summary()["verified_flat"]
        assert c.summary()["unresolved_intents"]
        foreign = deepcopy(b.history[0])
        foreign.update(ticket=9999, magic=0, comment="manual")
        b.history.append(foreign)
        assert not c.reconcile(b.snapshot(), b.now)["reconciled"]
        assert "EXTERNAL_OR_UNVERIFIED_ACTIVITY" in c.halts
        assert len(b.sent) == 2
    finally:
        j.close()


def test_external_cash_transfer_excluded_from_risk_profit(tmp_path: Path) -> None:
    c, b, j = create(tmp_path)
    try:
        b.balance += 100
        b.now += timedelta(seconds=1)
        b.deals.append({"ticket": 4001, "order": 0, "position_id": 0, "symbol": "", "type": 2, "entry": 0,
            "volume": 0, "price": 0, "profit": 100., "commission": 0., "swap": 0., "fee": 0.,
            "time_msc": int(b.now.timestamp() * 1000), "magic": 0, "comment": "synthetic deposit"})
        assert c.reconcile(b.snapshot(), b.now)["reconciled"]
        assert c.risk.state.high_water == 10000
        assert c.risk.state.daily_loss == 0
    finally:
        j.close()
