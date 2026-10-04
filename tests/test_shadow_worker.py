"""Native call isolation and bounded shutdown, using fake pipes/processes only."""

from typing import Any

import pytest

from xauusd_quant.shadow.adapter import IdentityFailure, ReadFailure
from xauusd_quant.shadow.config import ShadowConfig
from xauusd_quant.shadow.worker import MT5ProcessFeed, serve


class Pipe:
    def __init__(self, response: tuple = ("ok", None), *, hung: bool = False) -> None:
        self.response, self.hung = response, hung
        self.sent: list[Any] = []
        self.closed = False

    def send(self, value: Any) -> None:
        self.sent.append(value)

    def poll(self, seconds: float) -> bool:
        return not self.hung

    def recv(self) -> Any:
        return self.response

    def close(self) -> None:
        self.closed = True


class Process:
    def __init__(self) -> None:
        self.running = True
        self.terminated = False
        self.joins: list[float] = []

    def is_alive(self) -> bool:
        return self.running

    def join(self, seconds: float) -> None:
        self.joins.append(seconds)

    def terminate(self) -> None:
        self.running, self.terminated = False, True


def test_hung_vendor_read_stops_only_owned_worker() -> None:
    adapter = MT5ProcessFeed(ShadowConfig(), timeout_seconds=.01)
    pipe, process = Pipe(hung=True), Process()
    adapter._MT5ProcessFeed__connection = pipe
    adapter._MT5ProcessFeed__process = process
    with pytest.raises(ReadFailure, match="deadline"):
        adapter.latest()
    assert process.terminated and pipe.closed
    assert pipe.sent == [("latest", ())]
    assert not process.running


def test_worker_identity_failure_is_explicit() -> None:
    adapter = MT5ProcessFeed(ShadowConfig())
    adapter._MT5ProcessFeed__connection = Pipe(("identity_failure", None))
    with pytest.raises(IdentityFailure):
        adapter.verify()


def test_worker_dispatch_never_accepts_order_verb() -> None:
    class Commands(Pipe):
        def recv(self) -> tuple:
            if self.sent:
                raise EOFError
            return "order_send", ()
    pipe = Commands()
    serve(ShadowConfig(), pipe)
    assert pipe.sent == [("identity_failure", None)] and pipe.closed


def test_unconfigured_worker_never_spawns() -> None:
    adapter = MT5ProcessFeed(ShadowConfig())
    with pytest.raises(IdentityFailure):
        adapter.connect()
