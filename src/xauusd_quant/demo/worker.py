"""Bounded native demo worker; timeout is uncertainty, never permission to resend."""
from __future__ import annotations

import multiprocessing
from datetime import datetime
from typing import Any

from ..execution.config import content_hash
from ..shadow.adapter import IdentityFailure, ReadFailure
from ..shadow.config import ShadowConfig
from .broker import NativeDemoBroker, Permit
from .config import DemoConfig


def serve(config: DemoConfig, terminal: ShadowConfig, pipe: Any) -> None:
    broker = NativeDemoBroker(config, terminal)
    authority = object()
    broker.bind(authority)
    try:
        while True:
            verb, args = pipe.recv()
            try:
                if verb == "connect":
                    result: Any = broker.connect()
                elif verb == "snapshot":
                    result = broker.snapshot(*args)
                elif verb == "calculations":
                    result = broker.calculations(*args)
                elif verb in ("check", "send"):
                    request = args[0]
                    permit = Permit(authority, content_hash(request), *args[1:])
                    result = broker.check(request, permit) if verb == "check" else broker.send(request, permit)
                elif verb == "shutdown":
                    broker.shutdown()
                    pipe.send(("ok", None))
                    return
                else:
                    raise ValueError("unsupported demo verb")
                pipe.send(("ok", result))
            except IdentityFailure:
                pipe.send(("identity_failure", None))
            except ValueError:
                pipe.send(("validation_failure", None))
            except Exception:
                pipe.send(("read_or_submission_failure", None))
    except (EOFError, OSError):
        pass
    finally:
        broker.shutdown()
        pipe.close()


class DemoProcessBroker:
    def __init__(self, config: DemoConfig, terminal: ShadowConfig) -> None:
        self.config, self.terminal = config, terminal
        self.__authority: object | None = None
        self.__process: Any = None
        self.__pipe: Any = None

    def bind(self, authority: object) -> None:
        if self.__authority is not None:
            raise ValueError("broker authority already bound")
        self.__authority = authority

    def _call(self, verb: str, *args: Any) -> Any:
        if self.__pipe is None:
            raise ReadFailure("demo worker unavailable; reconcile before recovery")
        try:
            self.__pipe.send((verb, args))
            if not self.__pipe.poll(10):
                self.shutdown(force=True)
                raise ReadFailure("native deadline exceeded; submission outcome may be unknown")
            status, result = self.__pipe.recv()
        except (OSError, EOFError):
            self.shutdown(force=True)
            raise ReadFailure("native IPC unavailable; reconcile before retry") from None
        if status == "identity_failure":
            raise IdentityFailure("demo identity or permissions could not be verified")
        if status == "validation_failure":
            raise ValueError("native request boundary rejected invalid state/request")
        if status != "ok":
            raise ReadFailure("native call failed; do not blindly retry")
        return result

    def connect(self) -> dict[str, Any]:
        self.shutdown()
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        self.__pipe = parent
        self.__process = context.Process(target=serve, args=(self.config, self.terminal, child), daemon=True)
        self.__process.start()
        child.close()
        return dict(self._call("connect"))

    def snapshot(self, since: datetime) -> dict[str, Any]:
        return dict(self._call("snapshot", since))

    def calculations(self, side: str, volume: float, entry: float, adverse: float) -> dict[str, float]:
        return dict(self._call("calculations", side, volume, entry, adverse))

    def _permit(self, request: dict[str, Any], permit: Permit) -> None:
        if self.__authority is None or permit.authority is not self.__authority or permit.request_sha256 != content_hash(request):
            raise ValueError("bound coordinator capability required before native IPC")

    def check(self, request: dict[str, Any], permit: Permit) -> dict[str, Any] | None:
        self._permit(request, permit)
        return self._call("check", request, permit.expires_utc, permit.expected_state, permit.max_quote_age_seconds)

    def send(self, request: dict[str, Any], permit: Permit) -> dict[str, Any] | None:
        self._permit(request, permit)
        return self._call("send", request, permit.expires_utc, permit.expected_state, permit.max_quote_age_seconds)

    def shutdown(self, *, force: bool = False) -> None:
        process, pipe = self.__process, self.__pipe
        self.__process, self.__pipe = None, None
        if process is None:
            return
        if pipe is not None and process.is_alive() and not force:
            try:
                pipe.send(("shutdown", ()))
                if pipe.poll(.5):
                    pipe.recv()
            except (OSError, EOFError):
                pass
        if pipe is not None:
            pipe.close()
        process.join(.5)
        if process.is_alive():
            process.terminate()
            process.join(2)
        if process.is_alive():
            process.kill()
            process.join(2)
        if process.is_alive():
            raise ReadFailure("owned demo worker failed to stop")
