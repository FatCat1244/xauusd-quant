"""Explicit local demo identity and bounded UTC ingestion; no trading switch."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from ..execution.config import content_hash


@dataclass(frozen=True)
class ShadowConfig:
    configuration_id: str = "SHADOW_CAPTURE_V001"
    terminal_path: str | None = None
    expected_login: int | None = None
    expected_server: str | None = None
    expected_company: str | None = None
    symbol: str | None = None
    timeframes: tuple[str, ...] = ("5m", "15m")
    batch_size: int = 2000
    poll_seconds: float = 1.0
    duration_seconds: float = 60.0
    max_events: int = 10000
    max_buffer_ticks: int = 50000
    backfill_seconds: int = 900
    lateness_seconds: float = 0.0
    max_gap_seconds: float = 30.0
    max_quote_age_seconds: float = 5.0
    max_future_seconds: float = 2.0
    reconnect_attempts: int = 2
    initialize_timeout_ms: int = 5000

    def __post_init__(self) -> None:
        if not self.configuration_id.endswith("_V001"):
            raise ValueError("supported versioned shadow configuration required")
        for name, lo, hi in (("batch_size", 2, 10000), ("max_events", 1, 100000),
                             ("max_buffer_ticks", 2, 100000), ("backfill_seconds", 0, 3600),
                             ("reconnect_attempts", 0, 3), ("initialize_timeout_ms", 1000, 10000)):
            value = getattr(self, name)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f"bounded integer {name} required")
        for name, lower, upper in (("poll_seconds", .25, 5), ("duration_seconds", .1, 300),
                             ("lateness_seconds", 0, 5), ("max_gap_seconds", 1, 300),
                             ("max_quote_age_seconds", .1, 60), ("max_future_seconds", 0, 5)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"bounded finite {name} required")
        if not self.timeframes or len(set(self.timeframes)) != len(self.timeframes) or any(t not in ("1m", "5m", "15m") for t in self.timeframes):
            raise ValueError("unique supported timeframes required")
        if self.expected_login is not None and (type(self.expected_login) is not int or self.expected_login <= 0):
            raise ValueError("explicit positive demo login required")
        for name in ("terminal_path", "expected_server", "expected_company", "symbol"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"explicit nonempty {name} required")

    @property
    def configured(self) -> bool:
        return all(getattr(self, n) is not None for n in (
            "terminal_path", "expected_login", "expected_server", "expected_company", "symbol"))

    @property
    def identity(self) -> str:
        # Local checkpoint identity only; do not publish this low-entropy identity hash.
        return content_hash(asdict(self))

    def public(self) -> dict[str, Any]:
        private = {"terminal_path", "expected_login", "expected_server", "expected_company", "symbol"}
        return {k: v for k, v in asdict(self).items() if k not in private} | {
            "configured": self.configured, "timezone": "UTC", "broker_execution": "unavailable"}


def load_shadow_config(path: Path) -> ShadowConfig:
    try:
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        raise ValueError("invalid local shadow YAML syntax") from None
    if not isinstance(body, dict):
        raise ValueError("shadow configuration must be a mapping")
    if "timeframes" in body:
        body["timeframes"] = tuple(body["timeframes"])
    try:
        return ShadowConfig(**body)
    except TypeError as exc:
        raise ValueError("unknown or invalid shadow configuration fields") from exc
