"""Bounded native-worker failure paths and fixed verbs; no MT5 initialization."""
from pathlib import Path
from typing import Any

import pytest

from demo_synth import smoke_config
from shadow_synth import local_config
from xauusd_quant.demo.broker import Permit
from xauusd_quant.demo.worker import DemoProcessBroker, serve
from xauusd_quant.execution.config import content_hash
from xauusd_quant.shadow.adapter import ReadFailure


class Pipe:
    def __init__(self) -> None:
        self.closed = False
        self.sent: list[Any] = []

    def send(self, value: Any) -> None:
        self.sent.append(value)

    def poll(self, timeout: float) -> bool:
        return False

    def close(self) -> None:
        self.closed = True


class Process:
    def __init__(self) -> None:
        self.alive = True
        self.terminated = False

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float) -> None:
        pass

    def terminate(self) -> None:
        self.terminated = True
        self.alive = False


def test_native_timeout_stops_owned_worker_and_never_resends(tmp_path: Path) -> None:
    terminal = local_config(tmp_path / "terminal64.exe")
    broker = DemoProcessBroker(smoke_config(), terminal)
    authority = object()
    broker.bind(authority)
    pipe, process = Pipe(), Process()
    broker._DemoProcessBroker__pipe = pipe  # type: ignore[attr-defined]
    broker._DemoProcessBroker__process = process  # type: ignore[attr-defined]
    request = {"action": 1}
    with pytest.raises(ReadFailure, match="unknown"):
        broker.send(request, Permit(authority, content_hash(request)))
    assert len(pipe.sent) == 1 and pipe.sent[0][0] == "send"
    assert pipe.closed and process.terminated
    with pytest.raises(ReadFailure, match="unavailable"):
        broker.send(request, Permit(authority, content_hash(request)))
    assert len(pipe.sent) == 1


def test_raw_strategy_cannot_access_native_ipc(tmp_path: Path) -> None:
    broker = DemoProcessBroker(smoke_config(), local_config(tmp_path / "terminal64.exe"))
    broker.bind(object())
    request = {"action": 1}
    with pytest.raises(ValueError, match="capability"):
        broker.send(request, Permit(object(), content_hash(request)))


def test_child_has_fixed_demo_verbs_without_generic_vendor_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    terminal = local_config(tmp_path / "terminal64.exe")
    calls: list[str] = []

    class FakeNative:
        def __init__(self, *args: Any) -> None:
            pass

        def bind(self, authority: object) -> None:
            pass

        def shutdown(self) -> None:
            calls.append("shutdown")

    class Commands(Pipe):
        verbs = iter([("order_send", ()), ("shutdown", ())])

        def recv(self) -> Any:
            return next(self.verbs)

    monkeypatch.setattr("xauusd_quant.demo.worker.NativeDemoBroker", FakeNative)
    pipe = Commands()
    serve(smoke_config(), terminal, pipe)
    assert pipe.sent == [("validation_failure", None), ("ok", None)]
    assert calls == ["shutdown", "shutdown"] and pipe.closed
