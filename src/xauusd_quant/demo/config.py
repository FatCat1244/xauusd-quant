"""Strict demo run configuration; missing limits never acquire inferred defaults."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from ..alpha_portfolio.plan import versioned
from ..execution.config import content_hash
from ..risk.contracts import positive


@dataclass(frozen=True)
class DemoConfig:
    configuration_id: str = "DEMO_UNCONFIGURED_V001"
    mode: str = "DEMO"
    terminal_config_path: str | None = None
    risk_config_path: str | None = None
    risk_policy_id: str | None = None
    run_type: str | None = None
    specification_id: str | None = None
    expected_account_mode: str | None = None
    expected_account_currency: str | None = None
    dedicated_account_confirmed: bool | None = None
    project_tag: str | None = None
    magic: int | None = None
    side: str | None = None
    quantity_lots: float | None = None
    max_quantity_lots: float | None = None
    max_net_lots: float | None = None
    max_gross_lots: float | None = None
    entry_budget: int | None = None
    entry_request_budget: int | None = None
    cleanup_request_budget: int | None = None
    deviation_points: int | None = None
    filling_policy: str | None = None
    max_duration_seconds: float | None = None
    cleanup_seconds: float | None = None
    hold_seconds: float | None = None
    intent_ttl_seconds: float | None = None
    shutdown_policy: str | None = None
    protection: str | None = None

    def __post_init__(self) -> None:
        versioned(self.configuration_id)
        for identity in (self.risk_policy_id, self.specification_id):
            if identity is not None:
                versioned(identity)
        if self.mode != "DEMO":
            raise ValueError("only DEMO mode exists")
        for name, choices in (
            ("run_type", {"SMOKE", "STRATEGY"}),
            ("expected_account_mode", {"NETTING", "HEDGING"}),
            ("side", {"BUY", "SELL"}), ("filling_policy", {"FOK", "IOC"}),
            ("shutdown_policy", {"CLOSE_OWNED", "REPORT_UNRESOLVED"}),
            ("protection", {"PROCESS_EXIT", "BROKER_STOP"}),
        ):
            value = getattr(self, name)
            if value is not None and value not in choices:
                raise ValueError(f"unsupported {name}")
        if self.dedicated_account_confirmed is not None and type(self.dedicated_account_confirmed) is not bool:
            raise ValueError("explicit dedicated-account declaration required")
        for name in ("terminal_config_path", "risk_config_path", "risk_policy_id", "specification_id",
                     "expected_account_currency", "project_tag"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"nonempty {name} required")
        if self.project_tag is not None and (not self.project_tag.isascii() or not self.project_tag.isalnum() or len(self.project_tag) > 8):
            raise ValueError("project tag: <=8 ASCII alphanumeric characters")
        for name, minimum, maximum in (("magic", 1, 2147483647), ("entry_budget", 1, 3),
                ("entry_request_budget", 1, 3), ("cleanup_request_budget", 1, 3),
                ("deviation_points", 0, 10000)):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or not minimum <= value <= maximum):
                raise ValueError(f"bounded integer {name} required")
        for name in ("quantity_lots", "max_quantity_lots", "max_net_lots", "max_gross_lots",
                     "max_duration_seconds", "cleanup_seconds", "intent_ttl_seconds", "hold_seconds"):
            value = getattr(self, name)
            if value is not None:
                positive(value, name, zero=name == "hold_seconds")
        if self.max_duration_seconds is not None and self.max_duration_seconds > 300:
            raise ValueError("bounded run duration <=300 seconds required")
        if self.intent_ttl_seconds is not None and self.intent_ttl_seconds > 60:
            raise ValueError("bounded intent validity <=60 seconds required")
        if self.cleanup_seconds is not None and not 5 <= self.cleanup_seconds <= 120:
            raise ValueError("cleanup budget must be 5..120 seconds")
        if (all(v is not None for v in (self.max_duration_seconds, self.cleanup_seconds, self.hold_seconds))
            and float(self.cleanup_seconds or 0) + float(self.hold_seconds or 0) >= float(self.max_duration_seconds or 0)):
            raise ValueError("duration must leave entry time and reserved cleanup time")
        if self.run_type == "SMOKE" and self.entry_budget is not None and (self.entry_budget != 1 or self.entry_request_budget != 1):
            raise ValueError("smoke specification permits one entry attempt only")

    @property
    def missing(self) -> list[str]:
        return [name for name, value in asdict(self).items() if value is None]

    @property
    def configured(self) -> bool:
        return not self.missing and self.dedicated_account_confirmed is True

    @property
    def identity(self) -> str:
        return content_hash(asdict(self))

    def public(self) -> dict[str, Any]:
        hidden = {"terminal_config_path", "risk_config_path", "expected_account_currency", "project_tag", "magic"}
        return {k: v for k, v in asdict(self).items() if k not in hidden} | {"configured": self.configured}


def load_demo_config(path: Path) -> DemoConfig:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("demo configuration must be a mapping")
        return DemoConfig(**value)
    except (yaml.YAMLError, TypeError):
        raise ValueError("invalid or unknown demo configuration fields") from None
