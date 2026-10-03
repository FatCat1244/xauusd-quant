"""Bounded delayed residual outcome intervals; empirical dependent-data diagnostics only."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from .data import Outcome


@dataclass(frozen=True)
class PendingInterval:
    row_id: str
    prediction: float
    issued_utc: datetime
    end_utc: datetime
    matured_utc: datetime
    lower: float | None
    upper: float | None


class DelayedIntervals:
    """Absolute residual quantiles, optionally bounded adaptive conformal alpha updates.

    alpha <- clip(alpha + gamma*(target_alpha - miss), .01, .5).
    Clipping and financial dependence preclude an exchangeability coverage guarantee.
    Calibration errors enter only strictly after target publication, never by label position.
    """

    def __init__(
        self,
        *,
        window: int = 128,
        minimum: int = 32,
        alpha: float = 0.2,
        gamma: float = 0.0,
        horizon: int = 1,
        seconds: int = 300,
    ) -> None:
        if (
            not 1 <= minimum <= window
            or not 0.01 < alpha < 0.5
            or not math.isfinite(gamma)
            or gamma < 0
        ):
            raise ValueError("invalid calibration window/coverage/update")
        if not 1 <= horizon <= 12 or seconds not in (300, 900):
            raise ValueError("invalid uncertainty target grid/horizon")
        self.window, self.minimum, self.target_alpha, self.gamma, self.horizon, self.seconds = (
            window,
            minimum,
            alpha,
            gamma,
            horizon,
            seconds,
        )
        self.alpha = alpha
        self.errors: deque[float] = deque(maxlen=window)
        self.pending: dict[str, PendingInterval] = {}
        self.last_maturity: datetime | None = None
        self.last_decision: datetime | None = None
        self.last_update_utc: datetime | None = None
        self.updates = 0

    def issue(
        self, row_id: str, prediction: float, issued: datetime, end: datetime, matured: datetime
    ) -> dict[str, Any]:
        if not math.isfinite(prediction) or not issued < end <= matured:
            raise ValueError("invalid forecast/outcome availability for interval")
        if self.last_decision is not None and issued <= self.last_decision:
            raise ValueError("interval decisions must be strictly chronological")
        if self.last_update_utc is not None and issued < self.last_update_utc:
            raise ValueError("cannot backdate a decision behind the calibration update clock")
        if self.last_maturity is not None and self.last_maturity >= issued:
            raise ValueError("calibration contains outcome not preceding decision")
        if row_id in self.pending or len(self.pending) >= 64:
            raise ValueError(
                "duplicate/unbounded pending forecasts; expire missing outcomes explicitly"
            )
        self.last_decision = issued
        n = len(self.errors)
        rank = math.ceil((n + 1) * (1 - self.alpha))
        q = sorted(self.errors)[rank - 1] if n >= self.minimum and rank <= n else None
        lower, upper = (prediction - q, prediction + q) if q is not None else (None, None)
        self.pending[row_id] = PendingInterval(
            row_id, prediction, issued, end, matured, lower, upper
        )
        return {
            "row_id": row_id,
            "issued_utc": issued,
            "target_end_utc": end,
            "outcome_matured_utc": matured,
            "prediction": prediction,
            "lower": lower,
            "upper": upper,
            "width": 2 * q if q is not None else None,
            "status": "available" if q is not None else "uncalibrated",
            "type": "future_outcome_prediction_interval",
            "target": "future_return",
            "units": "log_mid_return",
            "horizon_bars": self.horizon,
            "grid_seconds": self.seconds,
            "calibration_observations": n,
            "calibration_window": self.window,
            "calibration_as_of_utc": self.last_maturity,
            "nominal_coverage": 1 - self.target_alpha,
            "current_alpha": self.alpha,
            "adaptive_gamma": self.gamma,
            "guarantee": "none asserted for dependent data; not conditional-mean or parameter confidence",
        }

    def deliver(self, outcome: Outcome, now: datetime) -> dict[str, Any] | None:
        if outcome.matured_utc >= now:
            raise ValueError("calibration outcome has not strictly matured before update")
        if (self.last_decision is not None and now < self.last_decision) or (
            self.last_update_utc is not None and now < self.last_update_utc
        ):
            raise ValueError("calibration update clock moved backward")
        if self.last_maturity is not None and outcome.matured_utc < self.last_maturity:
            raise ValueError("calibration updates not in outcome-availability order")
        item = self.pending.get(outcome.row_id)
        if item is None:
            return None
        if (outcome.horizon_bars, outcome.seconds) != (self.horizon, self.seconds) or (
            item.end_utc != outcome.end_utc or item.matured_utc != outcome.matured_utc
        ):
            raise ValueError("uncertainty target/horizon/publication mismatch")
        if not math.isfinite(outcome.log_return):
            raise ValueError("invalid matured outcome")
        del self.pending[outcome.row_id]
        error = outcome.log_return - item.prediction
        if not math.isfinite(error):
            raise ValueError("nonfinite residual")
        self.errors.append(abs(error))
        self.last_maturity = outcome.matured_utc
        self.last_update_utc = now
        self.updates += 1
        covered = (
            item.lower <= outcome.log_return <= item.upper
            if item.lower is not None and item.upper is not None
            else None
        )
        if self.gamma and covered is not None:
            self.alpha = min(
                0.5, max(0.01, self.alpha + self.gamma * (self.target_alpha - int(not covered)))
            )
        return {
            "row_id": outcome.row_id,
            "issued_utc": item.issued_utc,
            "observed_utc": now,
            "matured_utc": outcome.matured_utc,
            "error": error,
            "covered": covered,
            "width": item.upper - item.lower
            if item.upper is not None and item.lower is not None
            else None,
            "alpha_after": self.alpha,
            "updates": self.updates,
        }

    def expire(self, before: datetime) -> list[str]:
        ids = [key for key, value in self.pending.items() if value.matured_utc < before]
        for key in ids:
            del self.pending[key]
        return ids

    def resolved(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "minimum": self.minimum,
            "target_alpha": self.target_alpha,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "horizon": self.horizon,
            "seconds": self.seconds,
            "errors": list(self.errors),
            "pending": {
                k: {
                    **asdict(v),
                    "issued_utc": v.issued_utc.isoformat(),
                    "end_utc": v.end_utc.isoformat(),
                    "matured_utc": v.matured_utc.isoformat(),
                }
                for k, v in sorted(self.pending.items())
            },
            "last_maturity": self.last_maturity.isoformat() if self.last_maturity else None,
            "last_decision": self.last_decision.isoformat() if self.last_decision else None,
            "last_update_utc": self.last_update_utc.isoformat() if self.last_update_utc else None,
            "updates": self.updates,
        }

    @classmethod
    def restore(cls, body: dict[str, Any]) -> DelayedIntervals:
        state = cls(
            window=body["window"],
            minimum=body["minimum"],
            alpha=body["target_alpha"],
            gamma=body["gamma"],
            horizon=body["horizon"],
            seconds=body["seconds"],
        )
        if (
            len(body["errors"]) > state.window
            or len(body["pending"]) > 64
            or not 0.01 <= body["alpha"] <= 0.5
            or not all(math.isfinite(x) and x >= 0 for x in body["errors"])
        ):
            raise ValueError("invalid persisted calibration state")
        state.alpha, state.updates = body["alpha"], body["updates"]
        state.errors.extend(body["errors"])
        state.pending = {
            k: PendingInterval(
                **{
                    **v,
                    **{
                        t: datetime.fromisoformat(v[t])
                        for t in ("issued_utc", "end_utc", "matured_utc")
                    },
                }
            )
            for k, v in body["pending"].items()
        }
        state.last_maturity = (
            datetime.fromisoformat(body["last_maturity"]) if body["last_maturity"] else None
        )
        state.last_decision = (
            datetime.fromisoformat(body["last_decision"]) if body["last_decision"] else None
        )
        state.last_update_utc = (
            datetime.fromisoformat(body["last_update_utc"]) if body["last_update_utc"] else None
        )
        return state
