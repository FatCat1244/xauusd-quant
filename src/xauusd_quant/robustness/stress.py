"""Hypothetical operational scenarios replay decisions and fills through Stage12."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from typing import Any

import numpy as np

from ..execution.config import ExecutionConfig
from ..execution.engine import Quote
from ..execution.policy import Forecast


@dataclass(frozen=True)
class OperationalScenario:
    scenario_id: str
    commission_multiplier: float = 1.0
    extra_slippage: float = 0.0
    latency_ms: int = 100
    spread_multiplier: float = 1.0
    financing_extra: float = 0.0
    miss_probability: float = 0.0
    interruption_start: datetime | None = None
    interruption_end: datetime | None = None
    seed: int = 140014

    def __post_init__(self) -> None:
        if not self.scenario_id or any(
            not math.isfinite(x)
            for x in (
                self.commission_multiplier,
                self.extra_slippage,
                self.spread_multiplier,
                self.financing_extra,
                self.miss_probability,
            )
        ):
            raise ValueError("finite identified scenario required")
        if (
            self.commission_multiplier < 1
            or self.extra_slippage < 0
            or self.spread_multiplier < 1
            or self.financing_extra < 0
            or not 0 <= self.miss_probability <= 1
            or type(self.latency_ms) is not int
            or self.latency_ms < 0
        ):
            raise ValueError("adverse scenario bounds required")
        if (self.interruption_start is None) != (self.interruption_end is None):
            raise ValueError("both interruption boundaries required")
        if (
            self.interruption_start is not None
            and self.interruption_end is not None
            and (
                self.interruption_start.tzinfo is None
                or self.interruption_end.tzinfo is None
                or self.interruption_end <= self.interruption_start
            )
        ):
            raise ValueError("aware chronological interruption required")

    def configuration(self, base: ExecutionConfig) -> ExecutionConfig:
        if self.latency_ms < base.latency_ms:
            raise ValueError("stress cannot reduce declared base latency")
        return replace(
            base,
            scenario_id=self.scenario_id,
            commission_account_per_lot_per_leg=base.commission_account_per_lot_per_leg
            * self.commission_multiplier,
            slippage_usd_per_ounce_per_leg=base.slippage_usd_per_ounce_per_leg
            + self.extra_slippage,
            latency_ms=self.latency_ms,
            financing_long_account_per_lot_per_day=base.financing_long_account_per_lot_per_day
            + self.financing_extra,
            financing_short_account_per_lot_per_day=base.financing_short_account_per_lot_per_day
            + self.financing_extra,
        )

    def quotes(self, quotes: Iterable[Quote]) -> Iterator[Quote]:
        for q in quotes:
            if (
                self.interruption_start is not None
                and self.interruption_end is not None
                and self.interruption_start <= q.timestamp_utc < self.interruption_end
            ):
                continue  # Missing feed never fabricates a fill; engine timers advance on next event.
            if (
                self.spread_multiplier != 1
                and math.isfinite(q.bid)
                and math.isfinite(q.ask)
                and 0 < q.bid <= q.ask
            ):
                half = (q.ask - q.bid) * self.spread_multiplier / 2
                bid, ask = q.midpoint - half, q.midpoint + half
                if bid <= 0 or not math.isfinite(ask):
                    raise ValueError("stress would create invalid executable quote")
                yield replace(q, bid=bid, ask=ask)
            else:
                yield q  # Original invalid quotes stay invalid for engine rejection.

    def forecasts(self, forecasts: list[Forecast]) -> list[Forecast]:
        rng = np.random.default_rng(self.seed)
        return [
            replace(f, value=None, status="hypothetical_missed_opportunity")
            if rng.random() < self.miss_probability
            else f
            for f in forecasts
        ]

    def resolved(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "provenance": "hypothetical scenario, not measured Exness/broker distributions",
        }


def monte_carlo_scenarios(paths: int, seed: int) -> list[OperationalScenario]:
    if type(paths) is not int or not 1 <= paths <= 32:
        raise ValueError("bounded path count required")
    rng = np.random.default_rng(seed)
    return [
        OperationalScenario(
            f"MC_{i:03}",
            commission_multiplier=float(rng.uniform(1, 2)),
            extra_slippage=float(rng.uniform(0, 0.04)),
            latency_ms=int(rng.integers(100, 2001)),
            spread_multiplier=float(rng.uniform(1, 2)),
            financing_extra=float(rng.uniform(0, 4)),
            miss_probability=0.1,
            seed=seed + i + 1,
        )
        for i in range(paths)
    ]


def matched_cost_check(
    base: list[dict[str, Any]],
    stressed: list[dict[str, Any]],
    base_config: ExecutionConfig,
    stressed_config: ExecutionConfig,
) -> dict[str, Any]:
    """Cost break-even only on identical signed quantities, timestamps and quote populations."""
    fields = (
        "forecast_id",
        "direction",
        "quantity_lots",
        "entry_utc",
        "exit_utc",
        "entry_midpoint",
        "matched_midpoint_pnl_account",
    )
    if len(base) != len(stressed) or any(
        any(a[f] != b[f] for f in fields) for a, b in zip(base, stressed, strict=True)
    ):
        return {
            "status": "unavailable",
            "reason": "changed trade population/timing; no monotonic aggregate inference",
        }
    if not base:
        return {"status": "unavailable", "reason": "no matched closed trades"}
    # Spread and slippage are already in price PnL: never subtract spread twice.
    errors = []
    for a, b in zip(base, stressed, strict=True):
        quantity = a["quantity_lots"]
        fee = (
            2
            * quantity
            * (
                stressed_config.commission_account_per_lot_per_leg
                - base_config.commission_account_per_lot_per_leg
            )
        )
        slip = (
            2
            * quantity
            * base_config.contract_ounces_per_lot
            * base_config.account_currency_per_usd
            * (
                stressed_config.slippage_usd_per_ounce_per_leg
                - base_config.slippage_usd_per_ounce_per_leg
            )
        )
        funding = b["financing_account"] - a["financing_account"]
        spread = b["spread_cost_account"] - a["spread_cost_account"]
        errors.append((a["net_pnl_account"] - b["net_pnl_account"]) - fee - slip - funding - spread)
    if max(abs(e) for e in errors) > 1e-7:
        raise AssertionError("matched execution cost arithmetic failed")
    lots = sum(t["quantity_lots"] for t in base)
    return {
        "status": "verified",
        "maximum_error_account": max(abs(e) for e in errors),
        "additional_commission_break_even_account_per_lot_per_leg": sum(
            t["net_pnl_account"] for t in base
        )
        / (2 * lots),
        "interpretation": "signed additional per-leg commission that sets closed-population net to zero; negative means already losing; fixed timing/quantity only; excludes open positions",
    }
