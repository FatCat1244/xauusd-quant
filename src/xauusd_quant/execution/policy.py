"""Separate prespecified decision policy: signed expected log return, no fitted choices.

The forecast refers to mid-price closes, not to liquidation profit. Sign only supplies
direction. Quantity is fixed, exits follow the target's observed-bar horizon. There is
no cost-aware tuning, volatility direction inference or residual-to-price substitution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from .config import ExecutionConfig

TARGET_MEANINGS = {
    "mean_reversion": ("probability residual magnitude shrinks", "probability"),
    "mean_reversion_half": ("probability residual magnitude halves", "probability"),
    "residual_reduction": ("residual reduction / trailing volatility", "volatility_scaled"),
    "future_return": ("expected future log mid-close return", "log_mid_return"),
    "direction_up_cost": ("probability log return exceeds a trailing spread proxy", "probability"),
    "direction_down_cost": ("probability log return falls below a spread proxy", "probability"),
    "future_volatility": ("expected ln(realized volatility + floor)", "log_volatility"),
    "future_abs_move": ("expected ln(abs(log return) + floor)", "log_absolute_return"),
}


def utc_time(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("execution clocks require explicit UTC provenance, not naive broker time")
    return value.astimezone(UTC)


@dataclass(frozen=True)
class Forecast:
    """Adapter record, outside the ML contract; bar_index anchors observed-bar exits."""

    forecast_id: str
    bar_open_utc: datetime
    bar_index: int
    available_at_utc: datetime
    value: float | None
    timeframe: str
    horizon_bars: int
    units: str = "log_mid_return"
    target: str = "future_return"
    status: str = "ok"
    provenance_id: str = "synthetic"

    @classmethod
    def from_bar(
        cls,
        *,
        forecast_id: str,
        bar_open_utc: datetime,
        bar_index: int,
        value: float | None,
        config: ExecutionConfig,
        **kwargs: Any,
    ) -> Forecast:
        stamp = utc_time(bar_open_utc)
        available = stamp + timedelta(
            seconds=config.bar_seconds, milliseconds=config.computation_delay_ms
        )
        return cls(
            forecast_id,
            stamp,
            bar_index,
            available,
            value,
            config.timeframe,
            config.horizon_bars,
            **kwargs,
        )


def decision(forecast: Forecast, config: ExecutionConfig) -> tuple[int | None, str]:
    """+1/-1 direction, zero neutral, or explicit rejection reason."""
    earliest = utc_time(forecast.bar_open_utc) + timedelta(
        seconds=config.bar_seconds, milliseconds=config.computation_delay_ms
    )
    if utc_time(forecast.available_at_utc) < earliest:
        return None, "forecast_before_bar_close"
    if forecast.timeframe != config.timeframe or forecast.horizon_bars != config.horizon_bars:
        return None, "incompatible_horizon_or_timeframe"
    if forecast.units != config.policy_units or forecast.target != "future_return":
        return None, "unsupported_target_or_units"
    if forecast.status != "ok" or not forecast.provenance_id:
        return None, "invalid_forecast_status_or_provenance"
    if forecast.value is None or not math.isfinite(forecast.value):
        return None, "nonfinite_forecast"
    return (1 if forecast.value > 0 else -1 if forecast.value < 0 else 0), "reference_sign"
