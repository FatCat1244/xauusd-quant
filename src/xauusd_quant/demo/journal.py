"""Single writer and fsynced hash chain survive uncertainty around broker submission."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any, BinaryIO

from ..execution.config import content_hash
from ..execution.io import _json_safe


class Journal:
    """Hold an OS lock for the account, not just one run or process-local intent.

    A crash releases the lock but never the durable reservation. An interrupted
    line is fatal and requires investigation; it is never truncated silently.
    Files remain private under ignored runtime/demo.
    """

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.lock: BinaryIO = (directory / "writer.lock").open("a+b")
        try:
            if os.fstat(self.lock.fileno()).st_size == 0:
                self.lock.write(b"0")
                self.lock.flush()
            self.lock.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl: Any = importlib.import_module("fcntl")
                fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            raise RuntimeError("account execution already has a writer") from None
        self.path = directory / "journal.jsonl"
        self.rows: list[dict[str, Any]] = []
        self.sequence = 0
        self.digest = "0" * 64
        self.stream: BinaryIO | None = None
        try:
            if self.path.exists():
                if self.path.stat().st_size > 16 * 1024 * 1024:
                    raise ValueError("bounded journal capacity exceeded")
                with self.path.open("rb") as prior:
                    for line in prior:
                        row = json.loads(line)
                        if (not line.endswith(b"\n") or row["sequence"] != self.sequence + 1
                            or row["previous_sha256"] != self.digest
                            or row["content_sha256"] != content_hash(row["body"])):
                            raise ValueError("journal chain is invalid; reconciliation required")
                        self.sequence += 1
                        self.digest = hashlib.sha256(line).hexdigest()
                        self.rows.append(row["body"])
                        if len(self.rows) > 10000:
                            raise ValueError("bounded journal event capacity exceeded")
            self.stream = self.path.open("ab")
        except Exception:
            self.close()
            raise

    def append(self, event: str, data: dict[str, Any]) -> None:
        if self.stream is None:
            raise OSError("journal is closed")
        body = _json_safe({"event": event, "data": data})
        row = {"sequence": self.sequence + 1, "previous_sha256": self.digest,
               "content_sha256": content_hash(body), "body": body}
        line = (json.dumps(row, allow_nan=False) + "\n").encode("utf-8")
        if len(self.rows) >= 10000 or self.stream.tell() + len(line) > 16 * 1024 * 1024:
            raise OSError("bounded journal capacity exhausted")
        self.stream.write(line)
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.sequence += 1
        self.digest = hashlib.sha256(line).hexdigest()
        self.rows.append(body)

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()
            self.stream = None
        if not self.lock.closed:
            # Closing the handle releases the OS lock, including on process crash.
            self.lock.close()
