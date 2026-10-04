"""Bounded synthetic capture, durable replay, recovery/shutdown and storage faults."""

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from shadow_synth import ROOT, T, local_config, row
from xauusd_quant.execution.engine import MemoryRecorder
from xauusd_quant.features.factory_config import load_features_config
from xauusd_quant.shadow.adapter import IdentityFailure, ReadFailure
from xauusd_quant.shadow.config import ShadowConfig
from xauusd_quant.shadow.pipeline import ShadowPipeline
from xauusd_quant.shadow.runs import capture, code_identity, compare, replay, restore_checkpoint


class Clock:
    def __init__(self) -> None:
        self.elapsed = 0.

    def wall(self) -> Any:
        return T + timedelta(seconds=self.elapsed)

    def monotonic(self) -> float:
        return self.elapsed

    def sleep(self, seconds: float) -> None:
        self.elapsed += seconds


class Feed:
    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.shutdown_count = 0
        self.connect_count = 0
        self.fail_next = False
        self.wrong_reconnect = False

    def connect(self) -> dict[str, Any]:
        self.connect_count += 1
        if self.connect_count > 1 and self.wrong_reconnect:
            raise IdentityFailure("synthetic wrong reconnect")
        return self.verify()

    def verify(self) -> dict[str, Any]:
        return {"identity_verified": True, "source": "synthetic fixture"}

    def ticks(self, start: Any, count: int) -> list[dict[str, Any]]:
        if self.fail_next:
            self.fail_next = False
            raise ReadFailure("synthetic disconnected")
        return [row(i * 5, 1800 + .01 * (i // 12))
                for i in range(int(self.clock.elapsed // 5) + 1)
                if (T + timedelta(seconds=i * 5)) >= start][:count]

    def latest(self) -> dict[str, Any]:
        return row(int(self.clock.elapsed // 5) * 5)

    def shutdown(self) -> None:
        self.shutdown_count += 1


def run(tmp_path: Path, *, clock: Clock | None = None, adapter: Feed | None = None) -> tuple:
    clock = clock or Clock()
    adapter = adapter or Feed(clock)
    config = replace(local_config(tmp_path / "terminal64.exe"), duration_seconds=180)
    summary = capture(ROOT, config, tmp_path / "CAPTURE_V001", adapter=adapter,
        wall=clock.wall, monotonic=clock.monotonic, sleep=clock.sleep)
    return config, adapter, summary


def test_capture_replay_equality_and_broker_state_separation(tmp_path: Path) -> None:
    config, adapter, summary = run(tmp_path)
    assert summary["status"] == "COMPLETED" and summary["accepted_committed_ticks"] == 36
    assert summary["bars"] == 2 and summary["broker_orders"] == 0
    assert not summary["native_connectivity_completed"]
    assert not summary["live_observation_completed"] and not summary["stage18_ready"]
    assert adapter.shutdown_count == 1
    replay(ROOT, config, tmp_path / "CAPTURE_V001", tmp_path / "REPLAY_V001")
    comparison = compare(tmp_path / "CAPTURE_V001", tmp_path / "REPLAY_V001")
    assert comparison["equal"] and comparison["observations"]["bars"] == 2
    assert not comparison["full_model_equality"]
    with pytest.raises(FileExistsError):
        replay(ROOT, config, tmp_path / "CAPTURE_V001", tmp_path / "REPLAY_V001")


def test_reconnection_checks_identity_and_never_clears_halt(tmp_path: Path) -> None:
    clock = Clock()
    adapter = Feed(clock)
    adapter.fail_next = True
    _, adapter, summary = run(tmp_path, clock=clock, adapter=adapter)
    assert adapter.connect_count == 2 and adapter.shutdown_count == 2
    assert "DISCONNECTION_CONTINUITY_UNVERIFIED" in summary["halts"]
    assert not summary["shadow_strategy_active"]


def test_wrong_identity_after_reconnect_stops(tmp_path: Path) -> None:
    clock = Clock()
    adapter = Feed(clock)
    adapter.fail_next, adapter.wrong_reconnect = True, True
    _, _, summary = run(tmp_path, clock=clock, adapter=adapter)
    assert summary["status"] == "HALTED" and summary["accepted_committed_ticks"] == 0
    assert adapter.shutdown_count == 2


def test_storage_failure_stops_and_closes_adapter(tmp_path: Path, monkeypatch: Any) -> None:
    from xauusd_quant.shadow.runs import Store
    original = Store.__call__
    def fail(self: Store, table: str, payload: dict[str, Any]) -> None:
        if table == "ticks":
            raise OSError("synthetic disk full")
        original(self, table, payload)
    monkeypatch.setattr(Store, "__call__", fail)
    _, adapter, summary = run(tmp_path)
    assert summary["status"] == "HALTED" and summary["accepted_committed_ticks"] == 0
    assert adapter.shutdown_count == 1
    assert not list((tmp_path / "CAPTURE_V001").glob("checkpoint_*.json"))


def test_corrupt_checkpoint_stops_before_terminal_connect(tmp_path: Path) -> None:
    config = local_config(tmp_path / "terminal64.exe")
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema": "bad", "body": {}}), encoding="utf-8")
    clock, adapter = Clock(), Feed(Clock())
    with pytest.raises((ValueError, KeyError)):
        capture(ROOT, config, tmp_path / "BROKEN_V001", adapter=adapter, resume=path,
            wall=clock.wall, monotonic=clock.monotonic, sleep=clock.sleep)
    assert adapter.connect_count == 0 and adapter.shutdown_count == 1


def test_replay_incomplete_tail_is_not_equality(tmp_path: Path) -> None:
    config, _, _ = run(tmp_path)
    path = tmp_path / "CAPTURE_V001/records.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    final = next(i for i in range(len(lines) - 1, -1, -1) if json.loads(lines[i])["table"] == "batch_commit")
    path.write_text("\n".join(lines[:final]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest|uncommitted"):
        replay(ROOT, config, tmp_path / "CAPTURE_V001", tmp_path / "BAD_REPLAY_V001")


def test_missing_local_identity_never_connects(tmp_path: Path) -> None:
    adapter = Feed(Clock())
    with pytest.raises(IdentityFailure):
        capture(ROOT, ShadowConfig(), tmp_path / "NONE_V001", adapter=adapter)
    assert adapter.connect_count == 0 and not (tmp_path / "NONE_V001").exists()


def test_checkpoint_reconciles_audit_and_resumed_stream_halts(tmp_path: Path) -> None:
    config, _, _ = run(tmp_path)
    checkpoints = sorted((tmp_path / "CAPTURE_V001").glob("checkpoint_*.json"))
    pipeline = ShadowPipeline(config, load_features_config(root=ROOT), MemoryRecorder())
    cursor = restore_checkpoint(config, pipeline, code_identity(ROOT), checkpoints[-1])
    assert cursor.sequence == pipeline.bars.last_sequence == 36
    clock = Clock()
    clock.elapsed = 180
    adapter = Feed(clock)
    result = capture(ROOT, config, tmp_path / "RESUME_V001", adapter=adapter,
        resume=checkpoints[-1], wall=clock.wall, monotonic=clock.monotonic, sleep=clock.sleep)
    assert "RESTART_CONTINUITY_UNVERIFIED" in result["halts"]
    assert adapter.shutdown_count == 1
    assert result["accepted_committed_ticks"] == 36


def test_changed_audit_prefix_blocks_restore(tmp_path: Path) -> None:
    config, _, _ = run(tmp_path)
    checkpoints = sorted((tmp_path / "CAPTURE_V001").glob("checkpoint_*.json"))
    records = tmp_path / "CAPTURE_V001/records.jsonl"
    data = records.read_bytes()
    records.write_bytes(data.replace(b"1800.0", b"1799.0", 1))
    pipeline = ShadowPipeline(config, load_features_config(root=ROOT), MemoryRecorder())
    with pytest.raises(ValueError, match="prefix"):
        restore_checkpoint(config, pipeline, code_identity(ROOT), checkpoints[-1])
