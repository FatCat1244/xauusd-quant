"""Explicit hypothetical execution units and immutable reference-policy settings."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


def content_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class ExecutionConfig:
    """Amounts in account currency; quantity in lots; prices USD per troy ounce.

    Conversion is a declared constant account-currency/USD scenario, not a measured
    FX series. Financing accrues continuously per elapsed UTC day, including gaps.
    Positive financing rates are charges; negative rates are credits. No margin model.
    """

    scenario_id: str = "SYNTHETIC_USD_V001"
    specification_status: str = "hypothetical"
    specification_source: str = "fully declared synthetic fixture, not broker terms"
    account_currency: str = "USD"
    account_currency_per_usd: float = 1.0
    contract_ounces_per_lot: float = 100.0
    quantity_lots: float = 0.01
    quantity_increment_lots: float = 0.01
    min_quantity_lots: float = 0.01
    max_quantity_lots: float = 1.0
    commission_account_per_lot_per_leg: float = 3.0
    slippage_usd_per_ounce_per_leg: float = 0.02
    financing_long_account_per_lot_per_day: float = 2.0
    financing_short_account_per_lot_per_day: float = 2.0
    financing_model: str = "continuous_elapsed_utc_days"
    latency_ms: int = 100
    computation_delay_ms: int = 0
    order_ttl_ms: int = 30_000
    max_quote_age_ms: int = 1000
    max_gap_ms: int = 300_000
    initial_cash_account: float = 10_000.0
    end_policy: str = "retain_and_mark"
    equity_sample_ms: int = 1000
    batch_rows: int = 16_384
    policy_id: str = "SIGNED_EXPECTED_RETURN_V001"
    policy_field: str = "expected_return"
    policy_units: str = "log_mid_return"
    timeframe: str = "5m"
    horizon_bars: int = 1

    def __post_init__(self) -> None:
        if self.specification_status not in ("hypothetical", "unknown", "verified_supplied"):
            raise ValueError(
                "specification_status must be hypothetical, unknown or verified_supplied"
            )
        if not self.scenario_id or not self.specification_source or not self.account_currency:
            raise ValueError("scenario identity, currency and specification source are required")
        for name in (
            "account_currency_per_usd",
            "contract_ounces_per_lot",
            "quantity_lots",
            "quantity_increment_lots",
            "min_quantity_lots",
            "max_quantity_lots",
            "initial_cash_account",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name}: finite positive value required")
        if self.account_currency == "USD" and self.account_currency_per_usd != 1.0:
            raise ValueError("USD account conversion must be 1")
        if not self.min_quantity_lots <= self.quantity_lots <= self.max_quantity_lots:
            raise ValueError("quantity outside permitted bounds")
        steps = self.quantity_lots / self.quantity_increment_lots
        if not math.isclose(steps, round(steps), abs_tol=1e-9):
            raise ValueError("quantity must be a multiple of quantity_increment_lots")
        for name in ("commission_account_per_lot_per_leg", "slippage_usd_per_ounce_per_leg"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name}: finite nonnegative value required")
        for name in (
            "financing_long_account_per_lot_per_day",
            "financing_short_account_per_lot_per_day",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name}: financing must be explicitly declared and finite")
        for name in (
            "latency_ms",
            "computation_delay_ms",
            "order_ttl_ms",
            "max_quote_age_ms",
            "max_gap_ms",
            "equity_sample_ms",
            "batch_rows",
            "horizon_bars",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < (
                0 if name in ("latency_ms", "computation_delay_ms") else 1
            ):
                raise ValueError(f"{name}: invalid integer")
        if self.financing_model != "continuous_elapsed_utc_days":
            raise ValueError("declare continuous_elapsed_utc_days financing; no silent zero swap")
        if self.end_policy != "retain_and_mark":
            raise ValueError("only retain_and_mark is supported; no retrospective end fill")
        if (self.policy_id, self.policy_field, self.policy_units) != (
            "SIGNED_EXPECTED_RETURN_V001",
            "expected_return",
            "log_mid_return",
        ):
            raise ValueError("only the fixed signed log-mid-return reference policy is supported")
        if self.timeframe not in ("1m", "5m", "15m", "30m", "1h"):
            raise ValueError("unsupported timeframe")

    @property
    def bar_seconds(self) -> int:
        return {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600}[self.timeframe]

    def resolved(self) -> dict[str, Any]:
        return asdict(self)


def load_execution_config(path: Path) -> ExecutionConfig:
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ValueError("execution config must be a mapping")
    unknown = set(body) - ExecutionConfig.__dataclass_fields__.keys()
    if unknown:
        raise ValueError(f"unknown execution config fields: {sorted(unknown)}")
    required = {
        "scenario_id",
        "specification_status",
        "specification_source",
        "account_currency",
        "account_currency_per_usd",
        "contract_ounces_per_lot",
        "quantity_lots",
        "quantity_increment_lots",
        "min_quantity_lots",
        "max_quantity_lots",
        "commission_account_per_lot_per_leg",
        "slippage_usd_per_ounce_per_leg",
        "financing_long_account_per_lot_per_day",
        "financing_short_account_per_lot_per_day",
        "financing_model",
    }
    if required - body.keys():
        raise ValueError(
            f"explicit execution assumptions required: {sorted(required - body.keys())}"
        )
    return ExecutionConfig(**body)
