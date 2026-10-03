"""Separate strategy consumers; preserve raw predictions and exercise the Stage 12 engine."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import replace

import numpy as np

from ..execution.config import ExecutionConfig
from ..execution.engine import Quote, Recorder
from ..execution.policy import Forecast, decision
from .plan import ExperimentPlan, Scenario


def policy_spec(
    threshold: float, config: ExecutionConfig, kind: str = "forecast"
) -> dict[str, object]:
    return {
        "family": kind,
        "target": "future_return",
        "units": "log_mid_return",
        "threshold_abs_log_return": threshold,
        "entry": "sign above threshold; otherwise abstain",
        "exit": "h subsequent observed-bar closes",
        "horizon_bars": config.horizon_bars,
        "overlap": "reject while order/position active",
        "mode": "single net position",
        "quantity_lots": config.quantity_lots,
        "sessions": "gaps cancel pending; no interpolation",
        "overnight": "allowed; declared continuous long/short funding",
        "required_costs": "Bid/Ask, commission, slippage, latency, financing, FX",
    }


def consume_forecasts(
    forecasts: Iterable[Forecast],
    config: ExecutionConfig,
    plan: ExperimentPlan,
    threshold: float,
    kind: str,
    record: Recorder,
) -> Iterator[Forecast]:
    rng = np.random.default_rng(plan.seed)
    block_direction = 1
    for index, forecast in enumerate(forecasts):
        valid_direction, reason = decision(forecast, config)
        value = forecast.value
        if valid_direction is not None:
            if kind == "no_trading":
                value = 0.0
            elif kind == "random_direction":
                if index % plan.random_direction_block_opportunities == 0:
                    block_direction = int(rng.choice([-1, 1]))
                value = float(block_direction) * 1e-4
            elif value is not None and abs(value) <= threshold:
                value = 0.0
        record(
            "policy_consumption",
            {
                "forecast_id": forecast.forecast_id,
                "available_at_utc": forecast.available_at_utc,
                "raw_expected_return": forecast.value,
                "policy_value": value,
                "kind": kind,
                "threshold": threshold,
                "validation_reason": reason,
                "status": "rejected"
                if valid_direction is None
                else "abstain"
                if value == 0
                else "directional",
            },
        )
        yield replace(forecast, value=value)


def stress_quotes(quotes: Iterable[Quote], scenario: Scenario) -> Iterator[Quote]:
    """Counterfactual spread widening preserves UTC/sequence and original invalid quotes."""
    for quote in quotes:
        if scenario.spread_multiplier == 1 or not (
            math.isfinite(quote.bid) and math.isfinite(quote.ask) and 0 < quote.bid <= quote.ask
        ):
            yield quote
        else:
            half = (quote.ask - quote.bid) * scenario.spread_multiplier / 2
            yield replace(quote, bid=quote.midpoint - half, ask=quote.midpoint + half)
