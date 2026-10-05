"""Native API semantics against a fake vendor; no order-changing integration tests."""
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from demo_synth import DemoNative, smoke_config
from shadow_synth import local_config
from xauusd_quant.demo.broker import NativeDemoBroker, Permit, approval_state, classify, filling
from xauusd_quant.execution.config import content_hash
from xauusd_quant.shadow.adapter import IdentityFailure, MT5ReadOnly


def request() -> dict[str, object]:
    return {"action": 1, "symbol": "SYNTHETIC_GOLD", "volume": .02, "type": 0,
            "magic": 42018, "comment": "XQTEST:synthetic", "deviation": 0, "type_filling": 1, "type_time": 0}


def approved(authority: object, r: dict[str, object], native: DemoNative) -> Permit:
    snapshot = {"account": vars(native.account), "symbol": vars(native.symbol), "quote": native.rows[-1],
                "positions": [vars(p) for p in native.positions]}
    if any(not hasattr(p, "symbol") for p in native.positions):
        snapshot["positions"] = []  # Malformed/unrelated fixture still must fail the book guard.
    return Permit(authority, content_hash(r), datetime.now(UTC) + timedelta(seconds=5),
                  approval_state(snapshot, closing="position" in r), 30)


@pytest.mark.parametrize("mode", [1, 2, 99])
def test_real_contest_unknown_accounts_rejected(tmp_path: Path, mode: int) -> None:
    cfg = local_config(tmp_path / "terminal64.exe")
    native = DemoNative(Path(str(cfg.terminal_path)))
    native.account.trade_mode = mode
    broker = NativeDemoBroker(smoke_config(), cfg, native=native)
    with pytest.raises(IdentityFailure):
        broker.connect()
    assert "order_send" not in native.calls


@pytest.mark.parametrize("field,value", [("login", 9), ("server", "WRONG"), ("company", "OTHER"), ("trade_mode", 2), ("margin_mode", 2), ("currency", "USD_CENT")])
def test_identity_change_at_submission_disarms(tmp_path: Path, field: str, value: object) -> None:
    cfg = local_config(tmp_path / "terminal64.exe")
    native = DemoNative(Path(str(cfg.terminal_path)))
    broker = NativeDemoBroker(smoke_config(), cfg, native=native)
    authority = object()
    broker.bind(authority)
    broker.connect()
    r = request()
    permit = approved(authority, r, native)
    assert broker.check(r, permit) == {"retcode": 0}
    setattr(native.account, field, value)
    with pytest.raises(IdentityFailure):
        broker.send(r, permit)
    assert "order_send" not in native.calls


@pytest.mark.parametrize("kind,field,value", [("terminal", "tradeapi_disabled", True), ("terminal", "trade_allowed", False), ("account", "trade_expert", False), ("account", "trade_allowed", False)])
def test_execution_permissions_required(tmp_path: Path, kind: str, field: str, value: object) -> None:
    cfg = local_config(tmp_path / "terminal64.exe")
    native = DemoNative(Path(str(cfg.terminal_path)))
    broker = NativeDemoBroker(smoke_config(), cfg, native=native)
    authority = object()
    broker.bind(authority)
    broker.connect()
    setattr(getattr(native, kind), field, value)
    r = request()
    with pytest.raises(IdentityFailure):
        broker.send(r, approved(authority, r, native))
    assert "order_send" not in native.calls


@pytest.mark.parametrize("volume", [0, -.01, .005, .025, .04, float("nan"), float("inf")])
def test_invalid_volume_never_rounded_up(tmp_path: Path, volume: float) -> None:
    cfg = local_config(tmp_path / "terminal64.exe")
    native = DemoNative(Path(str(cfg.terminal_path)))
    broker = NativeDemoBroker(smoke_config(), cfg, native=native)
    authority = object()
    broker.bind(authority)
    broker.connect()
    r = request() | {"volume": volume}
    with pytest.raises(ValueError):
        broker.send(r, approved(authority, r, native))
    assert "order_send" not in native.calls


def test_shadow_read_only_and_raw_strategy_cannot_invoke_demo(tmp_path: Path) -> None:
    cfg = local_config(tmp_path / "terminal64.exe")
    native = DemoNative(Path(str(cfg.terminal_path)))
    shadow = MT5ReadOnly(cfg, native=native)
    assert not hasattr(shadow, "send") and not hasattr(shadow, "order_send")
    broker = NativeDemoBroker(smoke_config(), cfg, native=native)
    broker.bind(object())
    broker.connect()
    r = request()
    with pytest.raises(ValueError, match="capability"):
        broker.send(r, approved(object(), r, native))
    assert "order_send" not in native.calls


def test_final_identity_read_rejects_racing_real_account(tmp_path: Path) -> None:
    """Exercise the last demo-enum guard independently of the read-only verifier."""
    cfg = local_config(tmp_path / "terminal64.exe")
    native = DemoNative(Path(str(cfg.terminal_path)))
    broker = NativeDemoBroker(smoke_config(), cfg, native=native)
    authority = object()
    broker.bind(authority)
    broker.connect()
    original = native.account_info
    count = 0

    def changing_account() -> object:
        nonlocal count
        count += 1
        # _identity reads once through MT5ReadOnly.verify, then directly again.
        if count == 2:
            native.account.trade_mode = 2
        return original()

    native.account_info = changing_account  # type: ignore[method-assign]
    r = request()
    with pytest.raises(IdentityFailure):
        broker.send(r, approved(authority, r, native))
    assert "order_send" not in native.calls


def test_identity_change_after_send_is_unknown_not_safe_retry(tmp_path: Path) -> None:
    cfg = local_config(tmp_path / "terminal64.exe")
    native = DemoNative(Path(str(cfg.terminal_path)))
    broker = NativeDemoBroker(smoke_config(), cfg, native=native)
    authority = object()
    broker.bind(authority)
    broker.connect()

    def changed_send(r: dict[str, object]) -> object:
        native.calls.append("order_send")
        native.account.trade_mode = 2
        return SimpleNamespace(retcode=10009)

    native.order_send = changed_send  # type: ignore[method-assign]
    r = request()
    with pytest.raises(IdentityFailure):
        broker.send(r, approved(authority, r, native))
    with pytest.raises(IdentityFailure):
        broker.send(r, approved(authority, r, native))
    assert native.calls.count("order_send") == 1


@pytest.mark.parametrize("change", ["expired", "missing", "quote", "cash", "capabilities", "stale_quote"])
def test_native_handoff_revalidates_expiry_and_economic_state(tmp_path: Path, change: str) -> None:
    cfg = local_config(tmp_path / "terminal64.exe")
    native = DemoNative(Path(str(cfg.terminal_path)))
    broker = NativeDemoBroker(smoke_config(), cfg, native=native)
    authority = object()
    broker.bind(authority)
    broker.connect()
    r = request()
    p = approved(authority, r, native)
    if change == "expired":
        p = replace(p, expires_utc=datetime.now(UTC) - timedelta(seconds=1))
    elif change == "missing":
        p = replace(p, expires_utc=None)
    elif change == "quote":
        native.rows[-1]["bid"] += .01
    elif change == "cash":
        native.account.balance += 1
    elif change == "capabilities":
        native.symbol.trade_contract_size = 10
    elif change == "stale_quote":
        native.rows[-1]["time_msc"] -= 60000
    with pytest.raises(ValueError):
        broker.send(r, p)
    assert "order_send" not in native.calls


def test_hedging_close_uses_actual_position_ticket_and_no_reversal(tmp_path: Path) -> None:
    cfg = local_config(tmp_path / "terminal64.exe")
    native = DemoNative(Path(str(cfg.terminal_path)))
    native.account.margin_mode = 2
    native.positions = [SimpleNamespace(ticket=9001, identifier=8001, symbol="SYNTHETIC_GOLD",
        magic=42018, type=0, volume=.02)]
    broker = NativeDemoBroker(replace(smoke_config(), expected_account_mode="HEDGING"), cfg, native=native)
    authority = object()
    broker.bind(authority)
    broker.connect()
    close = request() | {"type": 1, "position": 9001}
    broker.send(close, approved(authority, close, native))
    assert "order_send" in native.calls
    for r in (close | {"position": 1001}, close | {"type": 0}, close | {"volume": .03}):
        with pytest.raises(ValueError):
            broker.send(r, approved(authority, r, native))


def test_pending_and_unrelated_positions_block_native_entry(tmp_path: Path) -> None:
    cfg = local_config(tmp_path / "terminal64.exe")
    native = DemoNative(Path(str(cfg.terminal_path)))
    broker = NativeDemoBroker(smoke_config(), cfg, native=native)
    authority = object()
    broker.bind(authority)
    broker.connect()
    r = request()
    native.pending = [SimpleNamespace(ticket=888)]
    with pytest.raises(ValueError, match="pending"):
        broker.send(r, approved(authority, r, native))
    native.pending = []
    native.positions = [SimpleNamespace(ticket=777)]
    with pytest.raises(ValueError, match="flat"):
        broker.send(r, approved(authority, r, native))
    assert "order_send" not in native.calls


@pytest.mark.parametrize("policy,execution,flags,enum", [("FOK", 2, 1, 0), ("IOC", 2, 2, 1), ("FOK", 0, 0, 0), ("IOC", 1, 0, 1)])
def test_filling_flags_are_mapped_not_copied(policy: str, execution: int, flags: int, enum: int) -> None:
    assert filling(policy, execution, flags) == enum


@pytest.mark.parametrize("policy,execution,flags", [("RETURN", 2, 3), ("FOK", 2, 2), ("IOC", 2, 1), ("BOC", 2, 4), ("FOK", 3, 1)])
def test_unsupported_filling_rejects(policy: str, execution: int, flags: int) -> None:
    with pytest.raises(ValueError):
        filling(policy, execution, flags)


@pytest.mark.parametrize("code,state", [(10008, "ACCEPTED_OR_PENDING"), (10009, "COMPLETION_REPORTED"), (10010, "PARTIAL_REPORTED"), (10006, "REJECTION_REPORTED"), (10018, "REJECTION_REPORTED"), (10012, "UNKNOWN"), (10031, "UNKNOWN"), (0, "UNKNOWN"), (99999, "UNKNOWN")])
def test_retcode_classification_not_fill(code: int, state: str) -> None:
    assert classify({"retcode": code}) == state
    assert classify(None) == "UNKNOWN"
