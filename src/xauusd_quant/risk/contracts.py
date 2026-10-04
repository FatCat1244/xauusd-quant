"""Explicit ledger denomination, CFD units and time-stamped offline interfaces."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from ..execution.config import ExecutionConfig, content_hash
from ..execution.io import _json_safe
from ..execution.policy import utc_time


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def positive(value: float, name: str, *, zero: bool = False) -> None:
    if not finite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(f"{name}: finite {'nonnegative' if zero else 'positive'} value required")


def resolved(value: Any) -> dict[str, Any]:
    return dict(_json_safe(asdict(value)))


def clocks(value: Any) -> None:
    for key, timestamp in asdict(value).items():
        if key.endswith("_utc"):
            utc_time(timestamp)


@dataclass(frozen=True)
class AccountSpec:
    specification_id: str
    currency: str
    denomination: str
    ledger_units_per_major: float
    major_currency_per_usd: float
    provenance: str
    status: str

    def __post_init__(self) -> None:
        if not self.specification_id or not self.currency or not self.provenance:
            raise ValueError("identified account units and provenance required")
        if self.denomination not in ("major", "cent") or self.ledger_units_per_major != (
            100 if self.denomination == "cent" else 1
        ):
            raise ValueError("explicit major=1 or cent=100 denomination required")
        positive(self.major_currency_per_usd, "currency conversion")
        if self.currency == "USD" and self.major_currency_per_usd != 1:
            raise ValueError("USD major-currency conversion must be one")
        if self.status not in ("synthetic", "verified_supplied"):
            raise ValueError("unknown account metadata cannot enable risk")

    @property
    def ledger_currency(self) -> str:
        return self.currency if self.denomination == "major" else f"{self.currency}_CENT"

    @property
    def ledger_per_usd(self) -> float:
        return self.ledger_units_per_major * self.major_currency_per_usd


@dataclass(frozen=True)
class InstrumentSpec:
    specification_id: str
    symbol: str
    price_units: str
    quantity_units: str
    ounces_per_lot: float
    lot_step: float
    min_lots: float
    max_lots: float
    tick_size: float
    price_decimals: int
    margin_notional_fraction: float
    commission_ledger_per_lot_leg: float
    slippage_usd_per_ounce_leg: float
    financing_long_ledger_per_lot_day: float
    financing_short_ledger_per_lot_day: float
    provenance: str
    status: str
    pnl_convention: str = "linear_quote_side_cfd"
    margin_convention: str = "absolute_notional_fraction_no_netting_credit"

    def __post_init__(self) -> None:
        if (
            not self.specification_id or self.symbol != "XAUUSD" or not self.provenance
            or self.price_units != "USD_per_troy_ounce" or self.quantity_units != "lots"
            or self.status not in ("synthetic", "verified_supplied")
            or self.pnl_convention != "linear_quote_side_cfd"
            or self.margin_convention != "absolute_notional_fraction_no_netting_credit"
        ):
            raise ValueError("identified supported instrument/unit/calculation convention required")
        for field in ("ounces_per_lot", "lot_step", "min_lots", "max_lots", "tick_size"):
            positive(getattr(self, field), field)
        for field in ("commission_ledger_per_lot_leg", "slippage_usd_per_ounce_leg"):
            positive(getattr(self, field), field, zero=True)
        if (
            type(self.price_decimals) is not int or not 0 <= self.price_decimals <= 8
            or not self.min_lots <= self.max_lots
            or not 0 < self.margin_notional_fraction <= 1
            or not math.isfinite(self.financing_long_ledger_per_lot_day)
            or not math.isfinite(self.financing_short_ledger_per_lot_day)
        ):
            raise ValueError("invalid precision, bounds, margin or financing")
        for value in (self.min_lots, self.max_lots):
            if not math.isclose(value / self.lot_step, round(value / self.lot_step), abs_tol=1e-8):
                raise ValueError("quantity bounds must align to increment")
        if not math.isclose(
            self.tick_size * 10**self.price_decimals,
            round(self.tick_size * 10**self.price_decimals), abs_tol=1e-8,
        ):
            raise ValueError("tick size incompatible with precision")

    def match_execution(self, account: AccountSpec, execution: ExecutionConfig) -> None:
        if self.status == "verified_supplied" and execution.specification_status != "verified_supplied":
            raise ValueError("supplied risk terms require supplied execution terms")
        if self.status == "synthetic" and execution.specification_status != "hypothetical":
            raise ValueError("synthetic risk terms require hypothetical execution")
        pairs = (
            (account.ledger_currency, execution.account_currency),
            (account.ledger_per_usd, execution.account_currency_per_usd),
            (self.ounces_per_lot, execution.contract_ounces_per_lot),
            (self.lot_step, execution.quantity_increment_lots),
            (self.min_lots, execution.min_quantity_lots),
            (self.max_lots, execution.max_quantity_lots),
            (self.commission_ledger_per_lot_leg, execution.commission_account_per_lot_per_leg),
            (self.slippage_usd_per_ounce_leg, execution.slippage_usd_per_ounce_per_leg),
            (self.financing_long_ledger_per_lot_day, execution.financing_long_account_per_lot_per_day),
            (self.financing_short_ledger_per_lot_day, execution.financing_short_account_per_lot_per_day),
        )
        if any(left != right for left, right in pairs):
            raise ValueError("risk/execution unit and cost specifications must match exactly")


@dataclass(frozen=True)
class MarketSnapshot:
    event_utc: datetime
    received_utc: datetime
    bid: float
    ask: float

    def __post_init__(self) -> None:
        clocks(self)


@dataclass(frozen=True)
class AccountSnapshot:
    snapshot_id: str
    event_utc: datetime
    received_utc: datetime
    account_specification_id: str
    balance: float
    equity: float | None
    free_margin: float
    signed_lots: float
    verified: bool
    execution_available: bool
    external_flow_total: float = 0.0
    open_estimated_loss: float | None = None

    def __post_init__(self) -> None:
        clocks(self)


@dataclass(frozen=True)
class HealthSnapshot:
    alpha_id: str
    policy_specification_id: str
    model_id: str
    feature_id: str
    event_utc: datetime
    received_utc: datetime
    forecast_available_utc: datetime
    forecast_expiry_utc: datetime
    forecast_value: float
    forecast_units: str
    horizon_seconds: int
    eligible: bool
    warm: bool
    features_ready: bool
    data_healthy: bool
    calibration_healthy: bool
    diagnostic_healthy: bool | None
    volatility: float | None
    uncertainty: float | None

    def __post_init__(self) -> None:
        clocks(self)


@dataclass(frozen=True)
class RiskRequest:
    intent_id: str
    portfolio_id: str
    allocation_id: str
    decision_utc: datetime
    received_utc: datetime
    target_lots: float
    sleeve_lots: dict[str, float]
    sizing_method: str
    stop_price: float | None = None

    def __post_init__(self) -> None:
        utc_time(self.decision_utc)
        utc_time(self.received_utc)
        if not self.intent_id or not self.portfolio_id or not self.allocation_id:
            raise ValueError("identified portfolio/intent/allocation required")
        if self.decision_utc > self.received_utc or not math.isfinite(self.target_lots):
            raise ValueError("causal finite target required")
        if any(not a or not math.isfinite(v) for a, v in self.sleeve_lots.items()):
            raise ValueError("finite identified sleeve exposures required")
        if not math.isclose(sum(self.sleeve_lots.values()), self.target_lots, abs_tol=1e-8):
            raise ValueError("sleeve intents must reconcile to unrounded target")

    @property
    def identity(self) -> str:
        return content_hash(resolved(self))


@dataclass(frozen=True)
class ExecutionUpdate:
    event_id: str
    intent_id: str
    event_utc: datetime
    received_utc: datetime
    status: str
    cumulative_filled_lots: float
    order_id: str | None = None

    def __post_init__(self) -> None:
        if not self.event_id or not self.intent_id or self.status not in (
            "acknowledged", "partial", "cancel_requested", "cancelled", "filled",
            "rejected", "expired", "unknown",
        ):
            raise ValueError("identified supported execution lifecycle required")
        positive(self.cumulative_filled_lots, "cumulative fill", zero=True)
        utc_time(self.event_utc)
        utc_time(self.received_utc)
        if self.event_utc > self.received_utc:
            raise ValueError("execution event cannot arrive before event time")
