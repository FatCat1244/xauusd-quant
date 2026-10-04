"""Occurrence-preserving second-overlap cursor; ambiguous history halts decisions."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..execution.config import content_hash
from ..execution.io import parse_utc


class ContinuityFailure(RuntimeError):  # noqa: N818 - explicit domain failure
    """Missing, reordered, overloaded or ambiguous observations; never silently skip."""


@dataclass(frozen=True)
class Tick:
    time_msc: int
    bid: float
    ask: float
    source_hash: str
    sequence: int
    received_utc: datetime
    backfill: bool

    def __post_init__(self) -> None:
        if (type(self.time_msc) is not int or self.time_msc < 0 or type(self.sequence) is not int or self.sequence < 1
            or self.received_utc.tzinfo is None or type(self.backfill) is not bool
            or not isinstance(self.source_hash, str) or len(self.source_hash) != 64
            or not all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) and v > 0 for v in (self.bid, self.ask))
            or self.ask < self.bid):
            raise ValueError("invalid recorded tick units, identity or clocks")

    @property
    def event_utc(self) -> datetime:
        return datetime.fromtimestamp(self.time_msc / 1000, UTC)

    def resolved(self) -> dict[str, Any]:
        return asdict(self) | {"received_utc": self.received_utc.isoformat()}

    @classmethod
    def restore(cls, body: dict[str, Any]) -> Tick:
        return cls(**(body | {"received_utc": parse_utc(body["received_utc"])}))


@dataclass
class Cursor:
    start_second: int
    last_msc: int | None = None
    sequence: int = 0
    boundary_hashes: list[str] = field(default_factory=list)

    @property
    def retrieval_start(self) -> datetime:
        second = self.start_second if self.last_msc is None else self.last_msc // 1000
        return datetime.fromtimestamp(second, UTC)

    def ingest(self, rows: list[dict[str, Any]], received: datetime, *,
               batch_limit: int, live_start: datetime, max_future_seconds: float) -> list[Tick]:
        if received.tzinfo is None or live_start.tzinfo is None or len(rows) > batch_limit:
            raise ContinuityFailure("invalid receipt time or oversized retrieval")
        if not rows:
            return []  # Empty poll is not proof of continuity or of market activity.
        raw_times = [r.get("time_msc") for r in rows]
        if any(type(t) is not int or t < 0 for t in raw_times):
            raise ContinuityFailure("invalid source milliseconds")
        times: list[int] = [r["time_msc"] for r in rows]
        if any(b < a for a, b in zip(times, times[1:], strict=False)):
            raise ContinuityFailure("source order is not chronological")
        if times[0] < int(self.retrieval_start.timestamp()) * 1000:
            raise ContinuityFailure("retrieval precedes declared overlap")
        try:
            hashes = [content_hash(r) for r in rows]
        except (ValueError, TypeError):
            raise ContinuityFailure("nonfinite or unsupported source tick fields") from None
        skip = 0
        if self.last_msc is not None:
            boundary = self.last_msc // 1000
            overlap = [h for h, t in zip(hashes, times, strict=True) if t // 1000 == boundary]
            if overlap[:len(self.boundary_hashes)] != self.boundary_hashes:
                raise ContinuityFailure("overlap occurrence prefix missing or changed")
            skip = len(self.boundary_hashes)
        if len(rows) == batch_limit and times[0] // 1000 == times[-1] // 1000:
            raise ContinuityFailure("saturated single-second batch: ordering/completeness ambiguous")
        out: list[Tick] = []
        for row, signature in zip(rows[skip:], hashes[skip:], strict=True):
            msc = row["time_msc"]
            prices = (row.get("bid"), row.get("ask"))
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in prices):
                raise ContinuityFailure("invalid executable Bid/Ask")
            bid, ask = float(row["bid"]), float(row["ask"])
            if ask < bid:
                raise ContinuityFailure("invalid executable Bid/Ask")
            if msc / 1000 > received.timestamp() + max_future_seconds:
                raise ContinuityFailure("future source time: clock health unresolved")
            out.append(Tick(msc, bid, ask, signature, self.sequence + len(out) + 1,
                            received, msc / 1000 < live_start.timestamp()))
        # Validate whole batch before mutating checkpoint state.
        if out:
            boundary = times[-1] // 1000
            self.boundary_hashes = [h for h, t in zip(hashes, times, strict=True) if t // 1000 == boundary]
            self.last_msc = times[-1]
            self.sequence = out[-1].sequence
        return out

    @classmethod
    def restore(cls, body: dict[str, Any], limit: int) -> Cursor:
        obj = cls(**body)
        if type(obj.start_second) is not int or obj.start_second < 0 or type(obj.sequence) is not int or obj.sequence < 0:
            raise ValueError("invalid persisted cursor")
        if obj.last_msc is not None and (type(obj.last_msc) is not int or obj.last_msc < obj.start_second * 1000):
            raise ValueError("invalid persisted source boundary")
        if len(obj.boundary_hashes) >= limit or any(not isinstance(h, str) or len(h) != 64 for h in obj.boundary_hashes):
            raise ValueError("invalid or saturated overlap checkpoint")
        if (obj.last_msc is None) != (not obj.boundary_hashes):
            raise ValueError("incomplete cursor checkpoint")
        return obj
