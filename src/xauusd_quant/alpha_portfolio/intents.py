"""Comparable policy exposure, retaining each forecast's own units and horizon."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

from ..execution.config import ExecutionConfig, content_hash
from ..execution.io import parse_utc
from ..execution.policy import Forecast, decision, utc_time


@dataclass(frozen=True)
class Intent:
    alpha_id: str
    specification_id: str
    decision_utc: datetime
    available_utc: datetime
    valid_until_utc: datetime
    target: float
    state: str
    rationale: str
    timeframe: str
    horizon_bars: int
    instrument: str = "XAUUSD"
    units: str = "signed_budget_fraction"

    def __post_init__(self) -> None:
        if (
            not utc_time(self.decision_utc)
            <= utc_time(self.available_utc)
            < utc_time(self.valid_until_utc)
        ):
            raise ValueError("intent publication/expiry clock invalid")
        if not math.isfinite(self.target) or abs(self.target) > 1:
            raise ValueError("bounded finite normalized exposure required")
        if self.state not in ("active", "flat", "invalid") or not self.rationale:
            raise ValueError("explicit state/rationale required")
        if self.state != "active" and self.target != 0:
            raise ValueError("invalid/flat signals must withdraw exposure")
        if (self.instrument, self.units) != ("XAUUSD", "signed_budget_fraction"):
            raise ValueError("comparable XAUUSD intent contract required")
        if (
            not self.alpha_id
            or not self.specification_id
            or type(self.horizon_bars) is not int
            or self.horizon_bars < 1
            or self.timeframe not in ("1m", "5m", "15m", "30m", "1h")
        ):
            raise ValueError("alpha/specification/horizon identities required")

    def resolved(self) -> dict[str, Any]:
        return asdict(self)


class ObservedBarPolicy:
    """Existing signed-return interpretation, with causal observed-bar exit events.

    This produces *sleeve intent*, not standalone fills. Overlapping directional
    signals are ignored. Invalid evidence clears intent immediately. Maximum
    wall-clock staleness is a separate safety/abstention convention, not a change
    to the forecast horizon or assumed knowledge of future observed bar times.
    """

    def __init__(
        self,
        alpha_id: str,
        specification_id: str,
        config: ExecutionConfig,
        maximum_age_seconds: int,
        threshold: float = 0,
    ) -> None:
        if maximum_age_seconds < 1 or not math.isfinite(threshold) or threshold < 0:
            raise ValueError("explicit age and fixed threshold required")
        self.alpha_id, self.specification_id, self.config = alpha_id, specification_id, config
        self.maximum_age_seconds, self.threshold = maximum_age_seconds, threshold
        self.completed_bar = -1
        self.exit_bar: int | None = None
        self.active: Intent | None = None
        self.clock: datetime | None = None

    def _at(self, at: datetime) -> None:
        if self.clock is not None and at < self.clock:
            raise ValueError("policy events must be chronological")
        self.clock = at
        if self.active is not None and self.active.valid_until_utc <= at:
            self.active, self.exit_bar = None, None

    def _intent(self, at: datetime, target: float, state: str, reason: str) -> Intent:
        return Intent(
            self.alpha_id,
            self.specification_id,
            at,
            at,
            at + timedelta(seconds=self.maximum_age_seconds),
            target,
            state,
            reason,
            self.config.timeframe,
            self.config.horizon_bars,
        )

    def on_bar(self, index: int, at_utc: datetime) -> Intent | None:
        at = utc_time(at_utc)
        self._at(at)
        if index != self.completed_bar + 1:
            raise ValueError("per-alpha observed bars contiguous")
        self.completed_bar = index
        if self.exit_bar is not None and index >= self.exit_bar:
            self.active, self.exit_bar = None, None
            return self._intent(at, 0, "flat", "observed_holding_end")
        return None

    def on_forecast(self, forecast: Forecast, at_utc: datetime) -> Intent | None:
        at = utc_time(at_utc)
        if utc_time(forecast.available_at_utc) > at:
            raise ValueError("forecast not yet available")
        self._at(at)
        direction, reason = decision(forecast, self.config)
        if (
            forecast.bar_index > self.completed_bar
            or forecast.bar_index + forecast.horizon_bars <= self.completed_bar
        ):
            direction, reason = None, "forecast_bar_or_horizon_unavailable"
        if direction is None:
            self.active, self.exit_bar = None, None
            return self._intent(at, 0, "invalid", reason)
        if at >= utc_time(forecast.available_at_utc) + timedelta(seconds=self.maximum_age_seconds):
            self.active, self.exit_bar = None, None
            return self._intent(at, 0, "invalid", "stale_forecast")
        if self.active is not None:
            return None
        value = forecast.value
        target = float(direction) if value is not None and abs(value) > self.threshold else 0.0
        self.active = self._intent(at, target, "active" if target else "flat", "fixed_sign_policy")
        self.active = Intent(
            **{
                **self.active.resolved(),
                "valid_until_utc": utc_time(forecast.available_at_utc)
                + timedelta(seconds=self.maximum_age_seconds),
            }
        )
        self.exit_bar = forecast.bar_index + forecast.horizon_bars if target else None
        result = self.active
        if not target:
            self.active = None
        return result

    def state(self) -> dict[str, Any]:
        return {
            "alpha_id": self.alpha_id,
            "specification_id": self.specification_id,
            "configuration_sha256": content_hash(
                {
                    "execution": self.config.resolved(),
                    "age": self.maximum_age_seconds,
                    "threshold": self.threshold,
                }
            ),
            "completed_bar": self.completed_bar,
            "exit_bar": self.exit_bar,
            "clock": self.clock.isoformat() if self.clock else None,
            "active": {
                **self.active.resolved(),
                **{
                    name: getattr(self.active, name).isoformat()
                    for name in ("decision_utc", "available_utc", "valid_until_utc")
                },
            }
            if self.active
            else None,
        }

    def restore(self, state: dict[str, Any]) -> None:
        if (state["alpha_id"], state["specification_id"]) != (self.alpha_id, self.specification_id):
            raise ValueError("policy state identity mismatch")
        if state.get("configuration_sha256") != self.state()["configuration_sha256"]:
            raise ValueError("policy state configuration mismatch")
        self.completed_bar, self.exit_bar = state["completed_bar"], state["exit_bar"]
        self.clock = parse_utc(state["clock"]) if state["clock"] else None
        body = state["active"]
        self.active = (
            Intent(
                **{
                    **body,
                    **{
                        name: parse_utc(body[name])
                        for name in ("decision_utc", "available_utc", "valid_until_utc")
                    },
                }
            )
            if body
            else None
        )
