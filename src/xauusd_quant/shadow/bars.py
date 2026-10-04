"""Bounded event-time bars using the canonical Stage1 left/open resampler.

Only a later source watermark completes an interval. No wall-clock flush,
flat bars, interpolation, or revision of an emitted decision is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import polars as pl

from ..data.resampler import resample_ticks
from ..execution.io import parse_utc
from .config import ShadowConfig
from .ingestion import ContinuityFailure, Tick


@dataclass(frozen=True)
class CompletedBar:
    timeframe: str
    values: dict[str, Any]
    available_utc: datetime
    backfill: bool
    valid: bool
    reasons: tuple[str, ...]

    @property
    def open_utc(self) -> datetime:
        return self.values["timestamp"]


class Bars:
    def __init__(self, config: ShadowConfig) -> None:
        self.config = config
        self.buffers: dict[str, list[Tick]] = {t: [] for t in config.timeframes}
        self.emitted: dict[str, int] = dict.fromkeys(config.timeframes, -1)
        self.first: int | None = None
        self.watermark: int | None = None
        self.last_receipt: datetime | None = None
        self.bad_intervals: dict[str, set[int]] = {t: set() for t in config.timeframes}
        self.halts: set[str] = set()
        self.last_sequence = 0

    def consume(self, tick: Tick) -> list[CompletedBar]:
        if tick.sequence <= self.last_sequence:
            raise ContinuityFailure("duplicate or out-of-order ingestion sequence")
        self.last_sequence = tick.sequence
        if self.last_receipt is not None and tick.received_utc < self.last_receipt:
            raise ContinuityFailure("receipt clock moved backwards")
        self.last_receipt = tick.received_utc
        if self.first is None:
            self.first = tick.time_msc
        if self.watermark is not None and tick.time_msc - self.watermark > self.config.max_gap_seconds * 1000:
            self.halts.add("GAP_CONTINUITY_UNVERIFIED")
            for tf in self.buffers:
                seconds = int(tf[:-1]) * 60
                self.bad_intervals[tf].add(self.watermark // (seconds * 1000) * seconds)
                self.bad_intervals[tf].add(tick.time_msc // (seconds * 1000) * seconds)
        self.watermark = max(self.watermark or tick.time_msc, tick.time_msc)
        out = []
        for tf, buffer in self.buffers.items():
            seconds = int(tf[:-1]) * 60
            opening = tick.time_msc // (seconds * 1000) * seconds
            if opening <= self.emitted[tf]:
                self.halts.add("LATE_TICK_AFTER_EMISSION")
                raise ContinuityFailure("late correction recorded; emitted bars stay immutable")
            if len(buffer) >= self.config.max_buffer_ticks:
                self.halts.add("QUEUE_OVERLOAD")
                raise ContinuityFailure("bounded bar buffer overloaded; no silent tick dropping")
            buffer.append(tick)
            cutoff = (self.watermark - int(self.config.lateness_seconds * 1000)) / 1000
            ready = [t for t in buffer if (t.time_msc // (seconds * 1000) + 1) * seconds <= cutoff]
            if not ready:
                continue
            frame = pl.DataFrame({"timestamp": [t.event_utc for t in ready],
                                  "bid": [t.bid for t in ready], "ask": [t.ask for t in ready],
                                  "mid": [(t.bid + t.ask) / 2 for t in ready],
                                  "spread": [t.ask - t.bid for t in ready]})
            canonical = resample_ticks(frame, tf, time_basis="timestamp", label="open", closed="left")
            for row in canonical.iter_rows(named=True):
                start = int(row["timestamp"].timestamp())
                members = [t for t in ready if t.time_msc // (seconds * 1000) * seconds == start]
                reasons = []
                if self.first > start * 1000:
                    reasons.append("PARTIAL_STARTUP_BAR")
                if start in self.bad_intervals[tf]:
                    reasons.append("GAP_AFFECTED_BAR")
                available = max(tick.received_utc, row["timestamp"] + timedelta(seconds=seconds))
                out.append(CompletedBar(tf, row, available,
                                        tick.backfill or any(t.backfill for t in members),
                                        not reasons, tuple(reasons)))
                self.emitted[tf] = start
                self.bad_intervals[tf].discard(start)
            ready_sequences = {t.sequence for t in ready}
            self.buffers[tf] = [t for t in buffer if t.sequence not in ready_sequences]
        return sorted(out, key=lambda b: (b.open_utc + timedelta(minutes=int(b.timeframe[:-1])), b.timeframe))

    def state(self) -> dict[str, Any]:
        return {"buffers": {tf: [t.resolved() for t in ticks] for tf, ticks in self.buffers.items()},
                "emitted": self.emitted, "first": self.first, "watermark": self.watermark,
                "last_receipt": self.last_receipt.isoformat() if self.last_receipt else None,
                "bad_intervals": {tf: sorted(v) for tf, v in self.bad_intervals.items()},
                "halts": sorted(self.halts), "last_sequence": self.last_sequence}

    @classmethod
    def restore(cls, config: ShadowConfig, body: dict[str, Any]) -> Bars:
        obj = cls(config)
        if set(body["buffers"]) != set(config.timeframes) or set(body["emitted"]) != set(config.timeframes):
            raise ValueError("bar checkpoint timeframes mismatch")
        for tf, rows in body["buffers"].items():
            if len(rows) > config.max_buffer_ticks:
                raise ValueError("oversized checkpoint buffer")
            obj.buffers[tf] = [Tick.restore(r) for r in rows]
        obj.emitted = body["emitted"]
        obj.first, obj.watermark = body["first"], body["watermark"]
        obj.last_receipt = parse_utc(body["last_receipt"]) if body["last_receipt"] else None
        obj.bad_intervals = {tf: set(v) for tf, v in body["bad_intervals"].items()}
        obj.halts = set(body["halts"])
        obj.last_sequence = body["last_sequence"]
        if type(obj.last_sequence) is not int or obj.last_sequence < 0:
            raise ValueError("invalid restored bar sequence")
        if obj.watermark is not None and any(t.time_msc > obj.watermark for rows in obj.buffers.values() for t in rows):
            raise ValueError("bar state exceeds recorded watermark")
        return obj


def serialize_bar(bar: CompletedBar) -> dict[str, Any]:
    return {"timeframe": bar.timeframe, "available_utc": bar.available_utc.isoformat(),
            "backfill": bar.backfill, "valid": bar.valid, "reasons": list(bar.reasons),
            "values": {k: v.isoformat() if isinstance(v, datetime) else v for k, v in bar.values.items()}}


def restore_values(values: dict[str, Any]) -> dict[str, Any]:
    return {k: parse_utc(v) if "timestamp" in k else v for k, v in values.items()}
