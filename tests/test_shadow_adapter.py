"""Identity verification, optional import and trapping all native order operations."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from shadow_synth import Native, T, local_config
from xauusd_quant.shadow.adapter import IdentityFailure, MT5ReadOnly, ReadFailure
from xauusd_quant.shadow.config import ShadowConfig, load_shadow_config


def test_unconfigured_never_initializes() -> None:
    native = Native(Path("synthetic.exe"))
    with pytest.raises(IdentityFailure):
        MT5ReadOnly(ShadowConfig(), native=native).connect()
    assert not native.calls


@pytest.mark.parametrize("field,value", [("login", 999), ("server", "WRONG"),
    ("company", "OTHER"), ("trade_mode", 2)])
def test_wrong_or_real_identity_blocks(tmp_path: Path, field: str, value: object) -> None:
    cfg = local_config(tmp_path / "terminal64.exe")
    native = Native(Path(str(cfg.terminal_path)))
    setattr(native.account, field, value)
    with pytest.raises(IdentityFailure):
        MT5ReadOnly(cfg, native=native).connect()
    assert native.calls[-1] == "shutdown"


def test_read_only_calls_and_privacy(tmp_path: Path) -> None:
    cfg = local_config(tmp_path / "terminal64.exe")
    native = Native(Path(str(cfg.terminal_path)))
    adapter = MT5ReadOnly(cfg, native=native)
    public = adapter.connect()
    assert public["demo_mode_verified"]
    assert adapter.ticks(T, 10)[0]["bid"] == 1800
    assert adapter.latest() is not None
    adapter.shutdown()
    assert set(native.calls) <= {"initialize", "copy_ticks_from", "shutdown"}
    serialized = json.dumps(public) + json.dumps(cfg.public())
    for private in (str(cfg.expected_login), str(cfg.expected_server), str(cfg.terminal_path)):
        assert private not in serialized
    for forbidden in ("order_send", "order_check", "orders_cancel", "close", "symbol_select", "login"):
        assert not hasattr(adapter, forbidden)


def test_reconnect_rechecks_identity_and_symbol(tmp_path: Path) -> None:
    cfg = local_config(tmp_path / "terminal64.exe")
    native = Native(Path(str(cfg.terminal_path)))
    adapter = MT5ReadOnly(cfg, native=native)
    adapter.connect()
    native.account.trade_mode = 2
    with pytest.raises(IdentityFailure):
        adapter.latest()
    adapter.shutdown()
    with pytest.raises(IdentityFailure):
        adapter.connect()
    native.account.trade_mode = 0
    native.symbol.visible = False
    with pytest.raises(IdentityFailure):
        adapter.connect()
    native.symbol.visible = True
    native.symbol.trade_tick_size = float("nan")
    with pytest.raises(IdentityFailure):
        adapter.connect()


def test_terminal_path_and_unavailable_metadata(tmp_path: Path) -> None:
    cfg = local_config(tmp_path / "terminal64.exe")
    native = Native(Path(str(cfg.terminal_path)))
    native.terminal.path = str(tmp_path / "other")
    with pytest.raises(IdentityFailure):
        MT5ReadOnly(cfg, native=native).connect()
    native.terminal.path = str(tmp_path)
    native.terminal.connected = False
    with pytest.raises(ReadFailure):
        MT5ReadOnly(cfg, native=native).connect()


@pytest.mark.parametrize("bid", [-1., float("nan"), 1800.123])
def test_invalid_precision_and_quote_sides_block(tmp_path: Path, bid: float) -> None:
    cfg = local_config(tmp_path / "terminal64.exe")
    native = Native(Path(str(cfg.terminal_path)))
    adapter = MT5ReadOnly(cfg, native=native)
    adapter.connect()
    native.rows[0]["bid"] = bid
    with pytest.raises(ReadFailure):
        adapter.ticks(T, 10)
    with pytest.raises(ReadFailure):
        adapter.latest()


@pytest.mark.parametrize("changes", [{"duration_seconds": float("nan")},
    {"max_events": True}, {"poll_seconds": .01}, {"timeframes": ("5m", "5m")},
    {"expected_login": True}, {"batch_size": 1000000}, {"symbol": ""}])
def test_bounds_and_explicit_units(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(ShadowConfig(), **changes)


def test_no_trading_config_switch(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("enable_trading: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        load_shadow_config(path)
