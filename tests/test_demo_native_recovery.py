"""Explicit native-abort recovery binds evidence; precheck diagnostics never send."""
from datetime import timedelta
from pathlib import Path

import pytest
from scripts import run_unsubmitted_demo_smoke as tool

from demo_synth import FakeBroker, create, propose
from xauusd_quant.alpha_portfolio.plan import freeze
from xauusd_quant.demo.coordinator import Coordinator
from xauusd_quant.demo.journal import Journal


def native_abort(root: Path) -> tuple[Coordinator, FakeBroker, Journal]:
    c, b, j = create(root, code=tool.FAILED_NATIVE_CODE)
    c.run_id = "DEMO_FRESH_QUOTE_SMOKE_V001"
    c.fresh_quotes = True
    entry = propose(c, b)
    original_check = b.check

    def rejected_boundary(*args: object) -> object:
        raise ValueError("synthetic native validation failure")

    b.check = rejected_boundary  # type: ignore[method-assign]
    with pytest.raises(ValueError):
        c.submit(entry["intent_id"], b.snapshot(), b.now)
    b.check = original_check  # type: ignore[method-assign]
    c.halt("EXPLICIT_REARM_SMOKE_ABORTED", b.now)
    assert c.reconcile(b.snapshot(), b.now)["verified_flat"]
    directory = root / "results/demo/runs/DEMO_FRESH_QUOTE_SMOKE_V001"
    freeze(directory / "verdict.json", {**c.summary(), "status": "NO_VERIFIED_LIFECYCLE",
        "failure": "ValueError", "code_identity": c.code})
    freeze(directory / "provenance.json", {"code_identity": c.code,
        "execution_identity": c.config.identity, "risk_identity": c.risk.configuration.identity})
    c.run_id = "RECOVERY_V001"
    return c, b, j


def test_explicit_native_abort_recovery_preserves_history(tmp_path: Path) -> None:
    c, b, j = native_abort(tmp_path)
    try:
        old = c.risk.state.high_water, c.risk.state.daily_start, c.risk.state.order_times.copy()
        assert tool.resume_code(j, "NEW_CODE", fresh_quotes=True, recover_validation_abort=True) == tool.FAILED_NATIVE_CODE
        tool.rearm_unsubmitted(c, b.snapshot(), b.now, operator="OPERATOR", new_code="NEW_CODE",
                               recovery_root=tmp_path)
        assert c.code == "NEW_CODE" and not c.halts and not c.armed
        assert old == (c.risk.state.high_water, c.risk.state.daily_start, c.risk.state.order_times)
        assert any(r["event"] == "execution_rearm" and r["data"].get("recovery_evidence") for r in j.rows)
        assert not b.sent
    finally:
        j.close()


@pytest.mark.parametrize("fault", ["default", "submitted", "severe", "missing", "wrong_risk", "expired_quote", "book", "no_operator"])
def test_bad_recovery_never_clears_halt(tmp_path: Path, fault: str) -> None:
    c, b, j = native_abort(tmp_path)
    try:
        snapshot = b.snapshot()
        root = tmp_path
        operator = "OPERATOR"
        if fault == "default":
            root = None
        elif fault == "submitted":
            next(iter(c.intents.values()))["submission_utc"] = b.now.isoformat()
            c.persist()
        elif fault == "severe":
            c.halt("DAILY_LOSS", b.now)
        elif fault == "missing":
            root = tmp_path / "ABSENT"
        elif fault == "wrong_risk":
            c.config = c.config.__class__(**{**c.config.__dict__, "configuration_id": "OTHER_V001"})
        elif fault == "expired_quote":
            snapshot["quote"]["time_msc"] -= 60000
        elif fault == "book":
            snapshot["positions"] = [{"volume": .01, "type": 0}]
        else:
            operator = ""
        with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
            tool.rearm_unsubmitted(c, snapshot, b.now, operator=operator, recovery_root=root)
        assert c.halts and not c.armed and not b.sent
    finally:
        j.close()


def test_precheck_only_full_workflow_sends_no_orders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c, b, j = native_abort(tmp_path)
    config, terminal, risk = c.config, c.terminal, c.risk.configuration
    j.close()
    b.authority = None
    monkeypatch.setattr(tool, "load_demo_config", lambda path: config)
    monkeypatch.setattr(tool, "load_shadow_config", lambda path: terminal)
    monkeypatch.setattr(tool, "load_configuration", lambda path: risk)
    monkeypatch.setattr(tool, "readiness", lambda *args: {"blocking_reasons": []})
    monkeypatch.setattr(tool, "account_directory", lambda *args: tmp_path / "runtime")
    monkeypatch.setattr(tool, "DemoProcessBroker", lambda *args: b)

    def synthetic(*args: object, **kwargs: object) -> Coordinator:
        return Coordinator(*args, offline_synthetic=True, **kwargs)

    monkeypatch.setattr(tool, "Coordinator", synthetic)
    actual_send = b.send
    monkeypatch.setattr(b, "send", lambda *args: pytest.fail("diagnostic attempted order_send"))
    report = tool.run(tmp_path, tmp_path / "unused", "DIAGNOSTIC_V001", "OPERATOR",
                      fresh_quotes=True, recover_validation_abort=True, precheck_only=True)
    assert report["status"] == "BROKER_PRECHECK_PASSED_NO_ORDERS"
    assert report["submissions"] == 0 and report["verified_flat"] and not report["reservations"]
    assert b.shutdown_count == 1 and not b.sent
    # Explicit subsequent smoke, after approval counters expire naturally.
    b.authority = None
    b.now += timedelta(seconds=301)
    monkeypatch.setattr(b, "send", actual_send)
    smoke = tool.run(tmp_path, tmp_path / "unused", "AFTER_DIAGNOSTIC_SMOKE_V001", "OPERATOR",
                     fresh_quotes=True, recover_validation_abort=True)
    assert smoke["status"] == "VERIFIED_DEMO_LIFECYCLE"
    assert smoke["submissions"] == 2 and smoke["verified_flat"] and not smoke["reservations"]


def test_guard_disabled_submission_is_detected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import inspect

    original = inspect.getsource(tool.rearm_unsubmitted)
    guard = 'if any(v.get("submission_utc") for v in [*prior_intents, *c.intents.values()]):'
    namespace = tool.__dict__.copy()
    exec(compile(original.replace(guard, "if False:"), "<isolated-recovery-guard>", "exec"), namespace)
    monkeypatch.setattr(tool, "rearm_unsubmitted", namespace["rearm_unsubmitted"])
    with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
        test_bad_recovery_never_clears_halt(tmp_path, "submitted")


def rejected_session(root: Path) -> tuple[Coordinator, FakeBroker, Journal]:
    c, b, j = native_abort(root)
    tool.rearm_unsubmitted(c, b.snapshot(), b.now, operator="OPERATOR", new_code="NEW_CODE",
                           recovery_root=root)
    c.run_id = "SESSION_DIAGNOSTIC_V001"
    c.initialize(b.snapshot(), b.now, arm=True)
    # Keep configuration identical; only advance the synthetic clock outside its session.
    b.now = (b.now + timedelta(days=5)).replace(hour=1)  # Saturday, session closed.
    c.run_started_utc = b.now
    entry = propose(c, b, name=f"{c.run_id}:entry")
    assert entry["state"] == "REJECTED" and set(entry["decision"]["rules"]) <= tool.SESSION_RULES
    directory = root / "results/demo/runs" / c.run_id
    freeze(directory / "plan.json", tool.DIAGNOSTIC_PLAN)
    freeze(directory / "provenance.json", {"code_identity": c.code,
        "execution_identity": c.config.identity, "risk_identity": c.risk.configuration.identity,
        "script_sha256": "SYNTHETIC_SCRIPT"})
    freeze(directory / "verdict.json", {**c.summary(), "failure": None, "code_identity": c.code,
        "script_sha256": "SYNTHETIC_SCRIPT"})
    c.armed = False
    return c, b, j


@pytest.mark.parametrize("fault", [None, "other_rule", "approval", "check"])
def test_session_rejection_allows_only_explicit_diagnostic_recovery(tmp_path: Path, fault: str | None) -> None:
    c, b, j = rejected_session(tmp_path)
    try:
        entry = c.intents[f"{c.run_id}:entry"]
        if fault == "other_rule":
            entry["decision"]["rules"].append("DAILY_LOSS")
        elif fault == "approval":
            entry["decision"]["approved_change_lots"] = .01
        elif fault == "check":
            entry["check"] = {"retcode": 0}
        assert tool.session_rejected_diagnostic(c, tmp_path) is (fault is None)
        assert not tool.passed_diagnostic(c)
        if fault is None:
            b.now = (b.now + timedelta(days=2)).replace(hour=12)
            tool.rearm_unsubmitted(c, b.snapshot(), b.now, operator="OPERATOR", recovery_root=tmp_path)
            assert not c.halts and not c.armed and not b.sent
        else:
            with pytest.raises(ValueError):
                tool.rearm_unsubmitted(c, b.snapshot(), b.now, operator="OPERATOR", recovery_root=tmp_path)
    finally:
        j.close()


@pytest.mark.parametrize("fault", ["marker_only", "wrong_code", "bad_check", "later_run"])
def test_diagnostic_marker_requires_durable_completed_current_precheck(tmp_path: Path, fault: str) -> None:
    c, b, j = create(tmp_path)
    try:
        entry = propose(c, b)
        c.submit(entry["intent_id"], b.snapshot(), b.now, precheck_only=True)
        c._record("native_diagnostic_passed", {"code_identity": c.code, "intent_id": entry["intent_id"],
            "plan_id": tool.DIAGNOSTIC_PLAN["plan_id"], "broker_submissions": 0})
        c.persist()
        assert tool.passed_diagnostic(c)
        if fault == "marker_only":
            c.intents.clear()
        elif fault == "wrong_code":
            c.code = "OTHER_CODE"
        elif fault == "bad_check":
            c.intents[entry["intent_id"]]["check"] = {"retcode": 10009}
        else:
            c.run_id = "LATER_FAILED_DIAGNOSTIC_V001"
        c.persist()
        assert not tool.passed_diagnostic(c) and not b.sent
    finally:
        j.close()


def test_recovery_smoke_requires_diagnostic_before_connecting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c, b, j = native_abort(tmp_path)
    config, terminal, risk = c.config, c.terminal, c.risk.configuration
    j.close()
    b.authority = None
    monkeypatch.setattr(tool, "load_demo_config", lambda path: config)
    monkeypatch.setattr(tool, "load_shadow_config", lambda path: terminal)
    monkeypatch.setattr(tool, "load_configuration", lambda path: risk)
    monkeypatch.setattr(tool, "readiness", lambda *args: {"blocking_reasons": []})
    monkeypatch.setattr(tool, "account_directory", lambda *args: tmp_path / "runtime")
    monkeypatch.setattr(tool, "DemoProcessBroker", lambda *args: b)

    def synthetic(*args: object, **kwargs: object) -> Coordinator:
        return Coordinator(*args, offline_synthetic=True, **kwargs)

    monkeypatch.setattr(tool, "Coordinator", synthetic)
    monkeypatch.setattr(b, "connect", lambda: pytest.fail("smoke connected without diagnostic"))
    report = tool.run(tmp_path, tmp_path / "unused", "PREMATURE_SMOKE_V001", "OPERATOR",
                      fresh_quotes=True, recover_validation_abort=True)
    assert report["failure"] == "ValueError" and report["last_phase"] == "RESTORE"
    assert report["submissions"] == 0 and not b.sent


def test_persisted_diagnostic_attempt_cannot_be_repeated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c, b, j = native_abort(tmp_path)
    config, terminal, risk = c.config, c.terminal, c.risk.configuration
    c._record("native_diagnostic_attempted", {"plan_id": tool.DIAGNOSTIC_PLAN["plan_id"]})
    c.persist()
    j.close()
    b.authority = None
    monkeypatch.setattr(tool, "load_demo_config", lambda path: config)
    monkeypatch.setattr(tool, "load_shadow_config", lambda path: terminal)
    monkeypatch.setattr(tool, "load_configuration", lambda path: risk)
    monkeypatch.setattr(tool, "readiness", lambda *args: {"blocking_reasons": []})
    monkeypatch.setattr(tool, "account_directory", lambda *args: tmp_path / "runtime")
    monkeypatch.setattr(tool, "DemoProcessBroker", lambda *args: b)

    def synthetic(*args: object, **kwargs: object) -> Coordinator:
        return Coordinator(*args, offline_synthetic=True, **kwargs)

    monkeypatch.setattr(tool, "Coordinator", synthetic)
    monkeypatch.setattr(b, "connect", lambda: pytest.fail("exhausted diagnostic connected"))
    report = tool.run(tmp_path, tmp_path / "unused", "REPEATED_DIAGNOSTIC_V001", "OPERATOR",
                      fresh_quotes=True, recover_validation_abort=True, precheck_only=True)
    assert report["failure"] == "ValueError" and report["last_phase"] == "RESTORE"
    assert not b.sent


def test_session_blocked_workflow_reports_rules_without_broker_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c, b, j = rejected_session(tmp_path)
    config, terminal, risk = c.config, c.terminal, c.risk.configuration
    j.close()
    b.authority = None
    b.now += timedelta(seconds=60)
    monkeypatch.setattr(tool, "load_demo_config", lambda path: config)
    monkeypatch.setattr(tool, "load_shadow_config", lambda path: terminal)
    monkeypatch.setattr(tool, "load_configuration", lambda path: risk)
    monkeypatch.setattr(tool, "code_identity", lambda root: "NEW_CODE")
    monkeypatch.setattr(tool, "readiness", lambda *args: {"blocking_reasons": []})
    monkeypatch.setattr(tool, "account_directory", lambda *args: tmp_path / "runtime")
    monkeypatch.setattr(tool, "DemoProcessBroker", lambda *args: b)

    def synthetic(*args: object, **kwargs: object) -> Coordinator:
        return Coordinator(*args, offline_synthetic=True, **kwargs)

    monkeypatch.setattr(tool, "Coordinator", synthetic)
    monkeypatch.setattr(b, "check", lambda *args: pytest.fail("closed session reached order_check"))
    monkeypatch.setattr(b, "send", lambda *args: pytest.fail("closed session reached order_send"))
    report = tool.run(tmp_path, tmp_path / "unused", "STILL_CLOSED_DIAGNOSTIC_V001", "OPERATOR",
                      fresh_quotes=True, recover_validation_abort=True, precheck_only=True)
    assert report["status"] == "RISK_BLOCKED_NO_ORDERS" and report["entry_risk_decision"] == "REJECT"
    assert set(report["entry_risk_rules"]) <= tool.SESSION_RULES and report["entry_risk_rules"]
    assert not report["prechecks"] and not report["submissions"] and report["verified_flat"]


def test_disabled_diagnostic_gate_is_detected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import inspect

    original = inspect.getsource(tool.run)
    guard = "if recover_validation_abort and not precheck_only and not passed_diagnostic(c):"
    namespace = tool.__dict__.copy()
    exec(compile(original.replace(guard, "if False:"), "<isolated-precheck-guard>", "exec"), namespace)
    broken = namespace["run"]

    def invoke(*args: object, **kwargs: object) -> object:
        namespace.update({name: value for name, value in tool.__dict__.items() if name != "run"})
        return broken(*args, **kwargs)

    monkeypatch.setattr(tool, "run", invoke)
    with pytest.raises(pytest.fail.Exception, match="smoke connected without diagnostic"):
        test_recovery_smoke_requires_diagnostic_before_connecting(tmp_path, monkeypatch)
