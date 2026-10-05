"""Explicit operator recovery preserves economic history and every submission guard."""
from datetime import timedelta
from pathlib import Path

import pytest
from scripts import run_unsubmitted_demo_smoke as operator_tool

from demo_synth import FakeBroker, create, propose
from xauusd_quant.demo.coordinator import Coordinator
from xauusd_quant.demo.journal import Journal
from xauusd_quant.shadow.runs import code_identity


def quote_abort(c: Coordinator, b: FakeBroker) -> None:
    entry = propose(c, b)
    b.bid += .01
    b.ask += .01
    with pytest.raises(ValueError, match="material state"):
        c.submit(entry["intent_id"], b.snapshot(), b.now)
    assert c.reconcile(b.snapshot(), b.now)["verified_flat"]
    c.run_id = "EXPLICIT_OPERATOR_V001"


def test_explicit_rearm_retains_history_and_requires_fresh_approval(tmp_path: Path) -> None:
    c, b, j = create(tmp_path)
    try:
        quote_abort(c, b)
        before = (c.risk.state.high_water, c.risk.state.daily_start,
                  c.risk.state.daily_loss, c.risk.state.drawdown,
                  c.risk.state.order_times.copy(), c.risk.state.turnover.copy())
        old = c.intents.copy()
        operator_tool.rearm_unsubmitted(c, b.snapshot(), b.now, operator="SYNTHETIC_OPERATOR")
        assert not c.armed and not b.sent
        assert c.risk.state.state == "READY" and not c.halts
        assert before == (c.risk.state.high_water, c.risk.state.daily_start,
                          c.risk.state.daily_loss, c.risk.state.drawdown,
                          c.risk.state.order_times, c.risk.state.turnover)
        assert c.intents == old
        assert any(row["event"] == "risk_rearms" for row in j.rows)
        assert any(row["event"] == "execution_rearm" for row in j.rows)
        c.initialize(b.snapshot(), b.now, arm=True)
        fresh = propose(c, b, name="NEW_APPROVAL_V001")
        c.submit(fresh["intent_id"], b.snapshot(), b.now)
        assert len(b.sent) == 1
    finally:
        j.close()


@pytest.mark.parametrize("state", ["FILLED", "REJECTED", "UNKNOWN"])
def test_any_previous_submission_prohibits_rearm(tmp_path: Path, state: str) -> None:
    c, b, j = create(tmp_path)
    try:
        quote_abort(c, b)
        next(iter(c.intents.values())).update(state=state, submission_utc=b.now.isoformat())
        c.persist()
        with pytest.raises(ValueError, match="submission"):
            operator_tool.rearm_unsubmitted(c, b.snapshot(), b.now, operator="OPERATOR")
        assert c.halts and not b.sent
    finally:
        j.close()


def test_full_journal_not_only_latest_intents_checks_submission(tmp_path: Path) -> None:
    c, b, j = create(tmp_path)
    try:
        quote_abort(c, b)
        intent = next(iter(c.intents.values()))
        intent["submission_utc"] = b.now.isoformat()
        c.persist()
        intent.pop("submission_utc")  # Simulates incomplete current metadata, not legitimate repair.
        with pytest.raises(ValueError, match="submission"):
            operator_tool.rearm_unsubmitted(c, b.snapshot(), b.now, operator="OPERATOR")
        assert c.halts
    finally:
        j.close()


@pytest.mark.parametrize("fault", ["permission", "stale", "spread", "position", "identity", "halt", "operator"])
def test_invalid_rearm_inputs_keep_halts(tmp_path: Path, fault: str) -> None:
    c, b, j = create(tmp_path)
    try:
        quote_abort(c, b)
        snapshot = b.snapshot()
        name = "OPERATOR"
        if fault == "permission":
            snapshot["permissions"] = False
        elif fault == "stale":
            snapshot["quote"]["time_msc"] -= 100000
        elif fault == "spread":
            snapshot["quote"]["ask"] += 100
        elif fault == "position":
            snapshot["positions"] = [{"volume": .01, "type": 0}]
        elif fault == "identity":
            snapshot["identity_verified"] = False
        elif fault == "halt":
            c.halt("DAILY_LOSS", b.now)
        else:
            name = ""
        with pytest.raises((ValueError, RuntimeError)):
            operator_tool.rearm_unsubmitted(c, snapshot, b.now, operator=name)
        assert c.halts and not b.sent
    finally:
        j.close()


def test_restart_does_not_inherit_same_process_operator_permission(tmp_path: Path) -> None:
    c, b, j = create(tmp_path)
    quote_abort(c, b)
    operator_tool.rearm_unsubmitted(c, b.snapshot(), b.now, operator="OPERATOR")
    config, terminal, risk = c.config, c.terminal, c.risk.configuration
    j.close()
    b.authority = None
    reopened = Journal(tmp_path / "runtime")
    try:
        restored = Coordinator(config, terminal, risk, b, reopened, "RESTORED_V001",
                               "SYNTHETIC_CODE", offline_synthetic=True)
        with pytest.raises(ValueError, match="arming"):
            restored.initialize(b.snapshot(), b.now, arm=True)
        assert not restored.armed and not b.sent
    finally:
        reopened.close()


def test_new_quote_change_still_aborts_without_broker_submission(tmp_path: Path) -> None:
    c, b, j = create(tmp_path)
    try:
        quote_abort(c, b)
        operator_tool.rearm_unsubmitted(c, b.snapshot(), b.now, operator="OPERATOR")
        c.initialize(b.snapshot(), b.now, arm=True)
        new = propose(c, b, name="NEW_ENTRY_V001")
        b.now += timedelta(seconds=.1)
        b.ask += .01
        with pytest.raises(ValueError, match="material state"):
            c.submit(new["intent_id"], b.snapshot(), b.now)
        assert not b.sent and "MATERIAL_STATE_CHANGED_REVALIDATION_REQUIRED" in c.halts
    finally:
        j.close()


@pytest.mark.parametrize("failed_close", [False, True])
def test_bounded_workflow_and_failed_exit_keep_actual_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_close: bool,
) -> None:
    c, b, j = create(tmp_path, code=code_identity(tmp_path))
    quote_abort(c, b)
    config, terminal, risk = c.config, c.terminal, c.risk.configuration
    j.close()
    b.authority = None
    if failed_close:
        original_send = b.send

        def fail_exit(request: dict, permit: object) -> object:
            if "position" in request:
                b.mode = "none_without_fill"
            return original_send(request, permit)

        monkeypatch.setattr(b, "send", fail_exit)
    monkeypatch.setattr(operator_tool, "load_demo_config", lambda path: config)
    monkeypatch.setattr(operator_tool, "load_shadow_config", lambda path: terminal)
    monkeypatch.setattr(operator_tool, "load_configuration", lambda path: risk)
    monkeypatch.setattr(operator_tool, "readiness", lambda *args: {"blocking_reasons": []})
    monkeypatch.setattr(operator_tool, "account_directory", lambda *args: tmp_path / "runtime")
    monkeypatch.setattr(operator_tool, "DemoProcessBroker", lambda *args: b)

    def synthetic_coordinator(*args: object) -> Coordinator:
        return Coordinator(*args, offline_synthetic=True)

    monkeypatch.setattr(operator_tool, "Coordinator", synthetic_coordinator)
    result = operator_tool.run(tmp_path, tmp_path / "unused.yaml", "EXPLICIT_SMOKE_V001", "OPERATOR")
    assert b.shutdown_count == 1 and len(b.sent) == 2
    assert result["verified_entry_deals"] == 1
    if failed_close:
        assert result["status"] == "NO_VERIFIED_LIFECYCLE"
        assert b.positions and result["reservations"] == 1
        assert not result["verified_flat"] and result["unresolved_intents"]
    else:
        assert result["status"] == "VERIFIED_DEMO_LIFECYCLE"
        assert result["verified_close_deals"] == 1 and result["verified_flat"]
        assert not b.positions and result["reservations"] == 0


def test_disabled_submission_guard_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inspect

    original = inspect.getsource(operator_tool.rearm_unsubmitted)
    guard = 'if any(v.get("submission_utc") for v in [*prior_intents, *c.intents.values()]):'
    assert guard in original
    namespace = operator_tool.__dict__.copy()
    exec(compile(original.replace(guard, "if False:"), "<isolated-disabled-guard>", "exec"), namespace)
    monkeypatch.setattr(operator_tool, "rearm_unsubmitted", namespace["rearm_unsubmitted"])
    with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
        test_full_journal_not_only_latest_intents_checks_submission(tmp_path)
