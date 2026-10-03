"""Two-sided Page CUSUM on matured standardized forecast errors; health outputs only."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any


class ErrorCUSUM:
    def __init__(self, scale: float, *, drift: float = 0.5, threshold: float = 8.0) -> None:
        if not all(math.isfinite(v) and v > 0 for v in (scale, drift, threshold)):
            raise ValueError("positive fixed training scale/drift/threshold required")
        self.scale, self.drift, self.threshold = scale, drift, threshold
        self.positive = self.negative = 0.0
        self.last_maturity: datetime | None = None
        self.updates = self.alarms = 0

    def update(self, error: float, matured: datetime, now: datetime) -> dict[str, Any]:
        if matured >= now:
            raise ValueError("change monitor cannot read an unmatured error")
        if not math.isfinite(error) or (
            self.last_maturity is not None and matured <= self.last_maturity
        ):
            raise ValueError("invalid/duplicate/out-of-order monitored error")
        z = error / self.scale
        self.positive = max(0.0, self.positive + z - self.drift)
        self.negative = max(0.0, self.negative - z - self.drift)
        alarm = max(self.positive, self.negative) >= self.threshold
        self.updates += 1
        self.alarms += int(alarm)
        self.last_maturity = matured
        record = {
            "matured_utc": matured,
            "observed_utc": now,
            "standardized_error": z,
            "positive_cusum": self.positive,
            "negative_cusum": self.negative,
            "alarm": alarm,
            "updates": self.updates,
            "alarms": self.alarms,
            "drift": self.drift,
            "threshold": self.threshold,
            "fixed_scale": self.scale,
            "action": "health diagnostic only; reset accumulator after alarm",
        }
        if alarm:
            self.positive = self.negative = 0.0
        return record

    def resolved(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "drift": self.drift,
            "threshold": self.threshold,
            "positive": self.positive,
            "negative": self.negative,
            "last_maturity": self.last_maturity.isoformat() if self.last_maturity else None,
            "updates": self.updates,
            "alarms": self.alarms,
        }

    @classmethod
    def restore(cls, body: dict[str, Any]) -> ErrorCUSUM:
        state = cls(body["scale"], drift=body["drift"], threshold=body["threshold"])
        if not all(math.isfinite(body[k]) and body[k] >= 0 for k in ("positive", "negative")):
            raise ValueError("invalid persisted monitor")
        state.positive, state.negative, state.updates, state.alarms = (
            body["positive"],
            body["negative"],
            body["updates"],
            body["alarms"],
        )
        state.last_maturity = (
            datetime.fromisoformat(body["last_maturity"]) if body["last_maturity"] else None
        )
        return state
