"""Deterministic synthetic broker with distinct order/deal/position identifiers."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from risk_synth import configuration
from shadow_synth import Native, local_config
from xauusd_quant.demo.broker import Permit
from xauusd_quant.demo.config import DemoConfig
from xauusd_quant.demo.coordinator import Coordinator
from xauusd_quant.demo.journal import Journal
from xauusd_quant.execution.config import content_hash
from xauusd_quant.risk.contracts import RiskRequest
from xauusd_quant.risk.policy import RiskConfiguration
from xauusd_quant.shadow.adapter import ReadFailure

T = datetime(2021, 1, 4, 12, tzinfo=UTC)


def smoke_risk() -> RiskConfiguration:
    original = configuration()
    assert original.policy is not None
    return replace(original, configuration_id="RISK_DEMO_SYNTHETIC_V001",
        policy=replace(original.policy, policy_id="RISK_DEMO_SMOKE_SYNTHETIC_V001",
            expected_portfolio_id="EXECUTION_SMOKE_V001", allowed_allocation_ids=("SMOKE_V001",),
            alpha_contracts={}, require_diagnostic=False))


def smoke_config() -> DemoConfig:
    return DemoConfig(configuration_id="DEMO_SYNTHETIC_V001", terminal_config_path="SYNTHETIC_TERMINAL.yaml",
        risk_config_path="SYNTHETIC_RISK.yaml", risk_policy_id="RISK_DEMO_SMOKE_SYNTHETIC_V001",
        run_type="SMOKE", specification_id="EXECUTION_SMOKE_SPEC_SYNTHETIC_V001",
        expected_account_mode="NETTING", expected_account_currency="USD", dedicated_account_confirmed=True,
        project_tag="XQTEST", magic=42018, side="BUY", quantity_lots=.02, max_quantity_lots=.03,
        max_net_lots=.03, max_gross_lots=.04, entry_budget=1, entry_request_budget=1,
        cleanup_request_budget=2, deviation_points=0, filling_policy="IOC", max_duration_seconds=60,
        cleanup_seconds=20, hold_seconds=0, intent_ttl_seconds=5, shutdown_policy="CLOSE_OWNED",
        protection="PROCESS_EXIT")


class FakeBroker:
    def __init__(self, config: DemoConfig | None = None) -> None:
        self.config = config or smoke_config()
        self.now = T
        self.bid, self.ask = 1799.9, 1800.1
        self.balance = 10000.0
        self.positions: list[dict[str, Any]] = []
        self.orders: list[dict[str, Any]] = []
        self.history: list[dict[str, Any]] = []
        self.deals: list[dict[str, Any]] = []
        self.sent: list[dict[str, Any]] = []
        self.authority: object | None = None
        self.mode = "full"
        self.check_code: int | None = 0
        self.show_history = True
        self.permissions = True
        self.account_mode = 0
        self.currency = "USD"
        self.shutdown_count = 0
        self.changed_snapshot = False
        self.calculation_factor = 1.0

    def bind(self, authority: object) -> None:
        if self.authority is not None:
            raise ValueError("authority already bound")
        self.authority = authority

    def connect(self) -> dict[str, Any]:
        return {"demo_mode_verified": True}

    def snapshot(self, since: datetime = T) -> dict[str, Any]:
        if self.changed_snapshot:
            self.bid += .01
            self.ask += .01
        unreal = sum((self.bid - p["price_open"]) * p["volume"] * 100 if p["type"] == 0
                     else (p["price_open"] - self.ask) * p["volume"] * 100 for p in self.positions)
        return deepcopy({"received_utc": self.now.isoformat(), "identity_verified": True,
            "permissions": self.permissions,
            "quote": {"time_msc": int(self.now.timestamp() * 1000), "bid": round(self.bid, 2), "ask": round(self.ask, 2)},
            "account": {"balance": self.balance, "equity": self.balance + unreal, "margin_free": self.balance - (180 if self.positions else 0),
                        "currency": self.currency, "currency_digits": 2, "margin_mode": self.account_mode, "credit": 0},
            "symbol": {"trade_contract_size": 100., "trade_tick_size": .01, "point": .01,
                "volume_min": .01, "volume_step": .01, "volume_max": 1., "digits": 2,
                "trade_exemode": 2, "filling_mode": 3, "trade_mode": 4, "order_mode": 1,
                "trade_stops_level": 0, "trade_freeze_level": 0},
            "positions": self.positions, "orders": self.orders,
            "history_orders": self.history if self.show_history else [],
            "deals": self.deals if self.show_history else []})

    def calculations(self, side: str, volume: float, entry: float, adverse: float) -> dict[str, float]:
        return {"margin": volume * entry * 100 * .05,
                "profit": (adverse - entry) * volume * 100 * (1 if side == "BUY" else -1) * self.calculation_factor}

    def check(self, request: dict[str, Any], permit: Permit) -> dict[str, Any] | None:
        assert permit.authority is self.authority and permit.request_sha256 == content_hash(request)
        return {"retcode": self.check_code} if self.check_code is not None else None

    def execute(self, request: dict[str, Any], *, partial: bool = False) -> dict[str, Any]:
        order_id = 1001 + len(self.history)
        volume = request["volume"] * (.5 if partial else 1)
        close = "position" in request
        price = self.ask if request["type"] == 0 else self.bid
        profit = 0.0
        if close:
            assert len(self.positions) == 1 and request["position"] == self.positions[0]["ticket"]
            position = self.positions[0]
            assert request["type"] != position["type"] and volume <= position["volume"] + 1e-9
            profit = (price - position["price_open"]) * volume * 100 * (1 if position["type"] == 0 else -1)
            identifier = position["identifier"]
            position["volume"] -= volume
            if position["volume"] < 1e-9:
                self.positions = []
        else:
            identifier = 8001
            self.positions = [{"ticket": 9001, "identifier": identifier, "symbol": "SYNTHETIC_GOLD",
                "type": request["type"], "volume": volume, "price_open": price, "price_current": price,
                "profit": 0, "swap": 0, "sl": 0, "tp": 0, "magic": self.config.magic}]
        fee = -3 * volume
        self.balance += profit + fee
        self.history.append({"ticket": order_id, "position_id": identifier, "symbol": request["symbol"],
            "type": request["type"], "volume_initial": request["volume"], "volume_current": request["volume"] - volume,
            "state": 2 if partial else 4, "time_setup_msc": int(self.now.timestamp() * 1000),
            "magic": request["magic"], "comment": request["comment"]})
        self.deals.append({"ticket": 2001 + len(self.deals), "order": order_id, "position_id": identifier,
            "symbol": request["symbol"], "type": request["type"], "entry": 1 if close else 0,
            "volume": volume, "price": price, "profit": profit, "commission": fee, "swap": 0, "fee": 0,
            "time_msc": int(self.now.timestamp() * 1000), "magic": request["magic"], "comment": request["comment"]})
        return {"retcode": 10010 if partial else 10009, "deal": self.deals[-1]["ticket"],
                "order": order_id, "volume": volume, "price": price, "request_id": len(self.sent), "retcode_external": 0}

    def send(self, request: dict[str, Any], permit: Permit) -> dict[str, Any] | None:
        assert permit.authority is self.authority and permit.request_sha256 == content_hash(request)
        self.sent.append(deepcopy(request))
        if self.mode == "reject":
            return {"retcode": 10006, "order": 0, "deal": 0}
        if self.mode == "accepted":
            self.orders = [{"ticket": 1001, "position_id": 0, "symbol": request["symbol"],
                "type": request["type"], "volume_initial": request["volume"], "volume_current": request["volume"],
                "state": 1, "time_setup_msc": int(self.now.timestamp() * 1000), "magic": request["magic"], "comment": request["comment"]}]
            return {"retcode": 10008, "order": 1001, "deal": 0}
        if self.mode == "none_without_fill":
            return None
        result = self.execute(request, partial=self.mode == "partial")
        if self.mode == "exception_after_fill":
            raise ReadFailure("synthetic timeout after actual fake fill")
        return None if self.mode == "timeout_after_fill" else result

    def shutdown(self) -> None:
        self.shutdown_count += 1


def create(path: Path, *, mode: str = "full", code: str = "SYNTHETIC_CODE") -> tuple[Coordinator, FakeBroker, Journal]:
    cfg, risk = smoke_config(), smoke_risk()
    terminal = local_config(path / "terminal64.exe")
    broker = FakeBroker(cfg)
    broker.mode = mode
    journal = Journal(path / "runtime")
    coordinator = Coordinator(cfg, terminal, risk, broker, journal, "SYNTHETIC_DEMO_V001", code, offline_synthetic=True)
    coordinator.initialize(broker.snapshot(), broker.now, arm=True)
    return coordinator, broker, journal


def propose(coordinator: Coordinator, broker: FakeBroker, *, close: bool = False, name: str | None = None) -> dict[str, Any]:
    target = 0 if close else .02
    request = RiskRequest(name or ("CLOSE_V001" if close else "ENTRY_V001"), "EXECUTION_SMOKE_V001", "SMOKE_V001",
        broker.now, broker.now, target, {"EXECUTION_SMOKE": target}, "horizon_stress")
    return coordinator.propose(request, broker.snapshot(), broker.now, expires_utc=broker.now + timedelta(seconds=5))


class DemoNative(Native):
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    TRADE_ACTION_DEAL = 1
    ORDER_TIME_GTC = 0

    def __init__(self, terminal: Path) -> None:
        super().__init__(terminal)
        self.terminal.trade_allowed = True
        self.terminal.tradeapi_disabled = False
        self.account.trade_allowed = self.account.trade_expert = True
        self.account.margin_mode = 0
        self.account.currency = "USD"
        self.account.balance = self.account.equity = self.account.margin_free = 10000.
        self.account.currency_digits = 2
        self.account.credit = 0
        self.rows[0]["time_msc"] = int(datetime.now(UTC).timestamp() * 1000)
        self.symbol.trade_mode = 4
        self.symbol.trade_exemode = 2
        self.symbol.filling_mode = 3
        self.symbol.order_mode = 1
        self.symbol.trade_stops_level = self.symbol.trade_freeze_level = 0
        self.positions: list[Any] = []
        self.pending: list[Any] = []

    def positions_get(self, **kwargs: Any) -> Any:
        return self.positions

    def orders_get(self) -> Any:
        return self.pending

    def order_check(self, request: dict[str, Any]) -> Any:
        self.calls.append("order_check")
        return SimpleNamespace(retcode=0)

    def order_send(self, request: dict[str, Any]) -> Any:
        self.calls.append("order_send")
        return SimpleNamespace(retcode=10006, deal=0, order=0, volume=0, price=0, request_id=1, retcode_external=0)
