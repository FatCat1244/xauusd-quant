"""Strict versioned settings; actual configuration has no implicit trading values."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

from ..alpha_portfolio.plan import versioned
from ..execution.config import content_hash
from .contracts import AccountSpec, InstrumentSpec, positive, resolved


@dataclass(frozen=True)
class RiskPolicy:
    policy_id: str
    status: str
    account_loss_budget: float
    position_loss_budget: float
    alpha_loss_budget: float
    max_net_lots: float
    max_gross_intended_lots: float
    max_positions: int
    max_pending_orders: int
    max_orders_per_window: int
    order_window_seconds: int
    max_turnover_lots_per_window: float
    turnover_window_seconds: int
    daily_loss_limit: float
    drawdown_limit: float
    daily_timezone: str
    max_spread_usd_per_ounce: float
    max_quote_age_seconds: float
    max_account_age_seconds: float
    max_health_age_seconds: float
    max_forecast_abs_log_return: float
    max_volatility: float
    max_uncertainty: float
    require_diagnostic: bool
    minimum_free_margin: float
    sizing_method: str
    horizon_stress_fraction: float | None
    adverse_exit_usd_per_ounce: float
    max_holding_seconds: int
    session_timezone: str
    session_start_hour: int
    session_end_hour: int
    session_weekdays: tuple[int, ...]
    allow_overnight: bool
    limit_response: str
    emergency_allow_wide_spread: bool
    warmup_snapshots: int
    expected_portfolio_id: str
    allowed_allocation_ids: tuple[str, ...]
    alpha_contracts: dict[str, dict[str, str]]

    def __post_init__(self) -> None:
        versioned(self.policy_id)
        if self.status not in ("synthetic", "verified_supplied"):
            raise ValueError("risk policy must explicitly be synthetic or verified_supplied")
        for field in (
            "account_loss_budget", "position_loss_budget", "alpha_loss_budget", "max_net_lots",
            "max_gross_intended_lots", "max_turnover_lots_per_window", "daily_loss_limit",
            "drawdown_limit", "max_spread_usd_per_ounce", "max_quote_age_seconds",
            "max_account_age_seconds", "max_health_age_seconds", "max_forecast_abs_log_return",
            "max_volatility", "max_uncertainty",
        ):
            positive(getattr(self, field), field)
        for field in ("minimum_free_margin", "adverse_exit_usd_per_ounce"):
            positive(getattr(self, field), field, zero=True)
        for field in (
            "max_positions", "max_pending_orders", "max_orders_per_window", "order_window_seconds",
            "turnover_window_seconds", "max_holding_seconds", "warmup_snapshots",
        ):
            if type(getattr(self, field)) is not int or getattr(self, field) < 1:
                raise ValueError(f"positive integer {field} required")
        if self.max_positions != 1:
            raise ValueError("Stage12 supports exactly one net position")
        if not self.alpha_loss_budget <= self.position_loss_budget <= self.account_loss_budget:
            raise ValueError("alpha <= position <= account loss budgets required")
        if self.max_net_lots > self.max_gross_intended_lots:
            raise ValueError("net cap must not exceed gross-intent cap")
        if self.sizing_method not in ("stop_distance", "horizon_stress"):
            raise ValueError("explicit supported sizing method required")
        if self.sizing_method == "horizon_stress":
            if self.horizon_stress_fraction is None:
                raise ValueError("horizon policy requires declared hypothetical stress fraction")
            positive(self.horizon_stress_fraction, "horizon_stress_fraction")
            if self.horizon_stress_fraction > 1:
                raise ValueError("bounded horizon stress fraction required")
        if self.limit_response not in ("block", "cancel", "flatten"):
            raise ValueError("declared limit response required")
        for field in ("require_diagnostic", "allow_overnight", "emergency_allow_wide_spread"):
            if type(getattr(self, field)) is not bool:
                raise ValueError(f"explicit boolean {field} required")
        for name in (self.daily_timezone, self.session_timezone):
            ZoneInfo(name)
        if (
            type(self.session_start_hour) is not int or type(self.session_end_hour) is not int
            or not 0 <= self.session_start_hour < self.session_end_hour <= 24
            or not self.session_weekdays
            or any(type(d) is not int or d not in range(7) for d in self.session_weekdays)
        ):
            raise ValueError("explicit non-wrapping session and weekdays required")
        if not self.expected_portfolio_id or not self.allowed_allocation_ids:
            raise ValueError("identified portfolio and permitted allocations required")
        if any(
            not a or set(c) != {"policy", "model", "features"} or not all(c.values())
            for a, c in self.alpha_contracts.items()
        ):
            raise ValueError("complete alpha/policy/model/feature identities required")

    @property
    def identity(self) -> str:
        return content_hash(resolved(self))


@dataclass(frozen=True)
class RiskConfiguration:
    configuration_id: str
    policy: RiskPolicy | None
    account: AccountSpec | None
    instrument: InstrumentSpec | None

    def __post_init__(self) -> None:
        versioned(self.configuration_id)
        if self.policy is not None and self.account is not None and self.instrument is not None and len({self.policy.status, self.account.status, self.instrument.status}) != 1:
            raise ValueError("synthetic and supplied specifications cannot be mixed")

    @property
    def configured(self) -> bool:
        return all(x is not None for x in (self.policy, self.account, self.instrument))

    @property
    def identity(self) -> str:
        return content_hash(resolved(self))


def load_configuration(path: Path) -> RiskConfiguration:
    body: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict) or set(body) != {
        "configuration_id", "policy", "account", "instrument"
    }:
        raise ValueError("exact risk configuration schema required")
    if body["policy"] is not None:
        body["policy"]["session_weekdays"] = tuple(body["policy"]["session_weekdays"])
        body["policy"]["allowed_allocation_ids"] = tuple(body["policy"]["allowed_allocation_ids"])
        body["policy"] = RiskPolicy(**body["policy"])
    if body["account"] is not None:
        body["account"] = AccountSpec(**body["account"])
    if body["instrument"] is not None:
        body["instrument"] = InstrumentSpec(**body["instrument"])
    return RiskConfiguration(**body)
