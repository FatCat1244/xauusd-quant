"""Owned native-read worker: a stalled vendor call cannot leave an unbounded run.

Only five fixed read/lifecycle verbs cross the pipe. The child has no broker
order route. Forceful timeout stops this Python worker, never the user's terminal.
"""

from __future__ import annotations

import multiprocessing
from datetime import datetime
from typing import Any

from .adapter import IdentityFailure, MT5ReadOnly, ReadFailure
from .config import ShadowConfig


def serve(config: ShadowConfig, connection: Any) -> None:
    adapter = MT5ReadOnly(config)
    try:
        while True:
            verb, arguments = connection.recv()
            try:
                result: Any
                if verb == "connect":
                    result = adapter.connect()
                elif verb == "verify":
                    result = adapter.verify()
                elif verb == "ticks":
                    result = adapter.ticks(*arguments)
                elif verb == "latest":
                    result = adapter.latest()
                elif verb == "shutdown":
                    adapter.shutdown()
                    connection.send(("ok", None))
                    return
                else:
                    raise IdentityFailure("unsupported read-only verb")
                connection.send(("ok", result))
            except IdentityFailure:
                connection.send(("identity_failure", None))
            except Exception:
                connection.send(("read_failure", None))
    except (EOFError, OSError):
        pass
    finally:
        adapter.shutdown()
        connection.close()


class MT5ProcessFeed:
    def __init__(self, config: ShadowConfig, *, timeout_seconds: float = 10) -> None:
        self.config = config
        self.timeout_seconds = timeout_seconds
        self.__process: Any = None
        self.__connection: Any = None

    def _call(self, verb: str, *arguments: Any) -> Any:
        if self.__connection is None:
            raise ReadFailure("native worker unavailable")
        try:
            self.__connection.send((verb, arguments))
            if not self.__connection.poll(self.timeout_seconds):
                self.shutdown(force=True)
                raise ReadFailure("native read deadline exceeded; worker stopped")
            status, result = self.__connection.recv()
        except (EOFError, OSError):
            self.shutdown(force=True)
            raise ReadFailure("native worker connection failed") from None
        if status == "identity_failure":
            raise IdentityFailure("configured demo identity verification failed")
        if status != "ok":
            raise ReadFailure("native read failed")
        return result

    def connect(self) -> dict[str, Any]:
        if not self.config.configured:
            raise IdentityFailure("explicit terminal and demo identity required")
        self.shutdown()
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        self.__connection = parent
        self.__process = context.Process(target=serve, args=(self.config, child), daemon=True)
        self.__process.start()
        child.close()
        return dict(self._call("connect"))

    def verify(self) -> dict[str, Any]:
        return dict(self._call("verify"))

    def ticks(self, start: datetime, count: int) -> list[dict[str, Any]]:
        return list(self._call("ticks", start, count))

    def latest(self) -> dict[str, Any] | None:
        return self._call("latest")

    def shutdown(self, *, force: bool = False) -> None:
        process, connection = self.__process, self.__connection
        self.__process, self.__connection = None, None
        if process is None:
            return
        if connection is not None and process.is_alive() and not force:
            try:
                connection.send(("shutdown", ()))
                if connection.poll(.5):
                    connection.recv()
            except (OSError, EOFError):
                pass
        if connection is not None:
            connection.close()
        process.join(.5)
        if process.is_alive():
            process.terminate()
            process.join(2)
        if process.is_alive():
            process.kill()
            process.join(2)
        if process.is_alive():
            raise ReadFailure("owned native worker failed to stop")
