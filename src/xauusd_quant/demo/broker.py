"""Official native reads/check/send, isolated from the shadow adapter.

    Identity and permissions are verified at every changing boundary. Requests
    require the bound coordinator's opaque capability. This is an architectural
    boundary, not a security sandbox against malicious code in this interpreter.
"""
from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from ..execution.config import content_hash
from ..risk.policy import RiskConfiguration
from ..shadow.adapter import IdentityFailure, MT5ReadOnly, ReadFailure, validate_quote
from ..shadow.config import ShadowConfig
from .config import DemoConfig
from .revalidation import evaluate_quote


@dataclass(frozen=True)
class Permit:
    authority: object
    request_sha256: str
    expires_utc: datetime | None = None
    expected_state: dict[str, Any] | None = None
    max_quote_age_seconds: float | None = None
    revalidation: dict[str, Any] | None = None


def approval_state(snapshot: dict[str, Any], *, closing: bool) -> dict[str, Any]:
    """Bind approval to economic state; market reductions tolerate price changes."""
    account_keys = ("currency", "margin_mode", "credit") if closing else (
        "balance", "equity", "margin_free", "currency", "currency_digits", "margin_mode", "credit")
    return {"account": {k: snapshot["account"][k] for k in account_keys},
        "quote": None if closing else {k: snapshot["quote"][k] for k in ("bid", "ask")},
        "positions": [{k: p[k] for k in ("ticket", "identifier", "symbol", "type", "volume", "magic")}
                      for p in snapshot["positions"]],
        "symbol": {k: snapshot["symbol"][k] for k in ("trade_contract_size", "trade_tick_size",
            "point", "volume_min", "volume_step", "volume_max", "digits", "trade_exemode",
            "filling_mode", "trade_mode", "order_mode", "trade_stops_level", "trade_freeze_level")}}


class Broker(Protocol):
    def bind(self, authority: object) -> None: ...
    def connect(self) -> dict[str, Any]: ...
    def snapshot(self, since: datetime) -> dict[str, Any]: ...
    def calculations(self, side: str, volume: float, entry: float, adverse: float) -> dict[str, float]: ...
    def check(self, request: dict[str, Any], permit: Permit) -> dict[str, Any] | None: ...
    def send(self, request: dict[str, Any], permit: Permit) -> dict[str, Any] | None: ...
    def shutdown(self) -> None: ...


def classify(response: dict[str, Any] | None) -> str:
    if response is None:
        return "UNKNOWN"
    code = response.get("retcode")
    if code == 10008:
        return "ACCEPTED_OR_PENDING"
    if code == 10009:
        return "COMPLETION_REPORTED"
    if code == 10010:
        return "PARTIAL_REPORTED"
    if code in {10004, 10006, 10007, 10013, 10014, 10015, 10016, 10017, 10018,
                10019, 10020, 10021, 10022, 10024, 10026, 10027, 10029, 10030,
                10032, 10033, 10034, 10035, 10036, 10038, 10039, 10040, 10042,
                10043, 10044, 10045, 10046}:
        return "REJECTION_REPORTED"
    return "UNKNOWN"


def filling(policy: str, execution: int, flags: int) -> int:
    # SYMBOL flags FOK=1/IOC=2 differ from ORDER enums FOK=0/IOC=1.
    if policy not in {"FOK", "IOC"} or execution not in (0, 1, 2):
        raise ValueError("only declared FOK/IOC request/instant/market execution supported")
    flag = 1 if policy == "FOK" else 2
    if execution == 2 and not flags & flag:
        raise ValueError("declared filling policy unsupported by symbol")
    return 0 if policy == "FOK" else 1


class NativeDemoBroker:
    def __init__(self, config: DemoConfig, terminal: ShadowConfig, *, native: Any = None) -> None:
        self.config, self.terminal = config, terminal
        self.__native = native
        self.__reader: MT5ReadOnly | None = None
        self.__authority: object | None = None
        self.entries = self.cleanup = 0
        self.validation_stage = "NOT_STARTED"

    def bind(self, authority: object) -> None:
        if self.__authority is not None:
            raise ValueError("broker authority already bound")
        self.__authority = authority

    def connect(self) -> dict[str, Any]:
        if self.__native is None:
            self.__native = importlib.import_module("MetaTrader5")
        self.__reader = MT5ReadOnly(self.terminal, native=self.__native)
        metadata = self.__reader.connect()
        return metadata | {"execution_permissions_verified": self._identity(False)["permissions"]}

    def _identity(self, changing: bool) -> dict[str, Any]:
        if self.__reader is None:
            raise IdentityFailure("demo adapter disconnected")
        info = self.__reader.verify()
        account, terminal = self.__native.account_info(), self.__native.terminal_info()
        if (account is None or terminal is None
            or account.login != self.terminal.expected_login
            or account.server != self.terminal.expected_server
            or account.company != self.terminal.expected_company
            or account.trade_mode != self.__native.ACCOUNT_TRADE_MODE_DEMO):
            raise IdentityFailure("demo identity changed; execution disarmed")
        permission_checks = {"terminal_trade_allowed": bool(terminal.trade_allowed),
            "python_trade_api_enabled": not bool(terminal.tradeapi_disabled),
            "account_trade_allowed": bool(account.trade_allowed),
            "account_expert_allowed": bool(account.trade_expert)}
        permissions = all(permission_checks.values())
        mode = {0: "NETTING", 2: "HEDGING"}.get(account.margin_mode)
        if changing and (mode != self.config.expected_account_mode or account.currency != self.config.expected_account_currency):
            raise IdentityFailure("account mode/currency changed; execution disarmed")
        if changing and not permissions:
            raise IdentityFailure("terminal/account execution permissions unavailable")
        return {"metadata": info, "account": account, "permissions": permissions,
                "permission_checks": permission_checks}

    def snapshot(self, since: datetime) -> dict[str, Any]:
        identity = self._identity(False)
        a = identity["account"]
        symbol = self.__native.symbol_info(self.terminal.symbol)
        tick = self.__native.symbol_info_tick(self.terminal.symbol)
        positions = self.__native.positions_get()
        orders = self.__native.orders_get()
        now = datetime.now(UTC)
        history = self.__native.history_orders_get(since, now)
        deals = self.__native.history_deals_get(since, now)
        if any(v is None for v in (symbol, tick, positions, orders, history, deals)):
            raise ReadFailure("account/order/deal history unavailable")
        if sum(len(v) for v in (positions, orders, history, deals)) > 10000:
            raise ReadFailure("bounded reconciliation history exceeded")
        quote = {k: getattr(tick, k) for k in ("time_msc", "bid", "ask")}
        # Receipt time follows retrieval. A legitimate tick arriving during
        # earlier API reads must not be compared to an earlier local clock sample.
        now = datetime.now(UTC)
        validate_quote(quote, identity["metadata"]["symbol_metadata"])

        def rows(values: Any, keys: tuple[str, ...]) -> list[dict[str, Any]]:
            return [{k: getattr(v, k) for k in keys} for v in values]

        return {"received_utc": now.isoformat(), "identity_verified": True,
            "permissions": identity["permissions"], "quote": quote,
            "permission_checks": identity["permission_checks"],
            "account": {k: getattr(a, k) for k in ("balance", "equity", "margin_free", "currency", "currency_digits", "margin_mode", "credit")},
            "symbol": identity["metadata"]["symbol_metadata"] | {k: getattr(symbol, k) for k in (
                "trade_exemode", "filling_mode", "trade_mode", "order_mode", "trade_stops_level", "trade_freeze_level")},
            "positions": rows(positions, ("ticket", "identifier", "symbol", "type", "volume",
                "price_open", "price_current", "profit", "swap", "sl", "tp", "magic")),
            "orders": rows(orders, ("ticket", "position_id", "symbol", "type", "volume_initial",
                "volume_current", "state", "time_setup_msc", "magic", "comment")),
            "history_orders": rows(history, ("ticket", "position_id", "symbol", "type", "volume_initial",
                "volume_current", "state", "time_setup_msc", "magic", "comment")),
            "deals": rows(deals, ("ticket", "order", "position_id", "symbol", "type", "entry", "volume",
                "price", "profit", "commission", "swap", "fee", "time_msc", "magic", "comment"))}

    def calculations(self, side: str, volume: float, entry: float, adverse: float) -> dict[str, float]:
        self._identity(False)
        if side not in {"BUY", "SELL"} or not all(math.isfinite(v) and v > 0 for v in (volume, entry, adverse)):
            raise ValueError("explicit finite side/quantity/calculation prices required")
        kind = self.__native.ORDER_TYPE_BUY if side == "BUY" else self.__native.ORDER_TYPE_SELL
        margin = self.__native.order_calc_margin(kind, self.terminal.symbol, volume, entry)
        profit = self.__native.order_calc_profit(kind, self.terminal.symbol, volume, entry, adverse)
        if margin is None or profit is None or not math.isfinite(margin) or not math.isfinite(profit):
            raise ReadFailure("broker margin/profit calculation unavailable")
        return {"margin": margin, "profit": profit}

    def _boundary(self, request: dict[str, Any], permit: Permit) -> dict[str, Any] | None:
        if self.__authority is None or permit.authority is not self.__authority or permit.request_sha256 != content_hash(request):
            raise ValueError("bound risk-coordinator capability required")
        identity = self._identity(True)
        now = datetime.now(UTC)
        if (permit.expires_utc is None or permit.expires_utc.tzinfo is None
            or now >= permit.expires_utc or permit.expected_state is None
            or permit.max_quote_age_seconds is None or not math.isfinite(permit.max_quote_age_seconds)
            or permit.max_quote_age_seconds <= 0):
            raise ValueError("unexpired state-bound approval required at native boundary")
        if self.config.run_type != "SMOKE":
            raise ValueError("native strategy binding and scientific readiness not established")
        if not self.config.configured or request.get("symbol") != self.terminal.symbol or request.get("magic") != self.config.magic:
            raise ValueError("explicit configured demo request required")
        if set(request) - {"action", "symbol", "volume", "type", "price", "deviation", "magic", "comment", "type_filling", "type_time", "position"}:
            raise ValueError("unsupported request fields/order type/protection")
        if request.get("action") != self.__native.TRADE_ACTION_DEAL or request.get("type") not in (0, 1):
            raise ValueError("market entry/position closure only")
        symbol = self.__native.symbol_info(self.terminal.symbol)
        volume = request.get("volume")
        if (not isinstance(volume, (int, float)) or isinstance(volume, bool) or not math.isfinite(volume)
            or not symbol.volume_min <= volume <= min(symbol.volume_max, float(self.config.max_quantity_lots or 0))
            or not math.isclose(volume / symbol.volume_step, round(volume / symbol.volume_step), abs_tol=1e-8)):
            raise ValueError("quantity exceeds supplied limits or native increment")
        if request.get("type_filling") != filling(str(self.config.filling_policy), symbol.trade_exemode, symbol.filling_mode):
            raise ValueError("filling policy changed or unsupported")
        if request.get("deviation") != self.config.deviation_points or request.get("type_time") != self.__native.ORDER_TIME_GTC:
            raise ValueError("declared point deviation/time semantics required")
        if symbol.trade_exemode == 2 and "price" in request:
            raise ValueError("market execution omits requested price")
        if symbol.trade_exemode != 2:
            price = request.get("price")
            if not isinstance(price, (float, int)) or not math.isfinite(price) or price <= 0 or not math.isclose(price / symbol.trade_tick_size, round(price / symbol.trade_tick_size), abs_tol=1e-6):
                raise ValueError("aligned executable request price required")
        if symbol.trade_mode == 0 or not symbol.order_mode & 1:
            raise ValueError("symbol market trading unavailable")
        all_positions, all_orders = self.__native.positions_get(), self.__native.orders_get()
        if all_positions is None or all_orders is None or all_orders:
            raise ValueError("current execution book is unavailable or has pending orders")
        tick = self.__native.symbol_info_tick(self.terminal.symbol)
        if tick is None:
            raise ValueError("native executable quote unavailable")
        quote = {k: getattr(tick, k) for k in ("time_msc", "bid", "ask")}
        now = datetime.now(UTC)  # Quote receipt follows API retrieval, not earlier identity reads.
        validate_quote(quote, identity["metadata"]["symbol_metadata"])
        if not 0 <= now.timestamp() - quote["time_msc"] / 1000 <= permit.max_quote_age_seconds:
            raise ValueError("native quote is stale or from the future")
        if "position" in request:
            positions = [p for p in all_positions if p.ticket == request["position"]]
            if (positions is None or len(positions) != 1 or positions[0].symbol != self.terminal.symbol
                or len(all_positions) != 1
                or positions[0].magic != self.config.magic or request["type"] == positions[0].type
                or volume > positions[0].volume + 1e-9):
                raise ValueError("verified bounded owned-position closure required")
        elif all_positions or volume > min(float(self.config.max_net_lots or 0), float(self.config.max_gross_lots or 0)):
            raise ValueError("flat dedicated account and exposure limits required")
        elif symbol.trade_mode == 3 or (symbol.trade_mode == 1 and request["type"] != 0) or (symbol.trade_mode == 2 and request["type"] != 1):
            raise ValueError("symbol direction/close-only restrictions")
        current_state = approval_state({
            "account": {k: getattr(identity["account"], k) for k in permit.expected_state["account"]},
            "quote": quote,
            "positions": [{k: getattr(p, k) for k in ("ticket", "identifier", "symbol", "type", "volume", "magic")}
                          for p in all_positions],
            "symbol": {k: getattr(symbol, k) for k in permit.expected_state["symbol"]},
        }, closing="position" in request)
        expected_state = permit.expected_state.copy()
        proof = None
        cfg: RiskConfiguration | None = None
        if permit.revalidation is not None:
            if self.config.run_type != "SMOKE" or symbol.trade_exemode != 2 or "position" in request:
                raise ValueError("fresh quote revalidation supports MARKET entry only")
            cfg = permit.revalidation.get("configuration")
            if not isinstance(cfg, RiskConfiguration) or cfg.policy is None or cfg.policy.policy_id != self.config.risk_policy_id:
                raise ValueError("matching authoritative risk policy required")
            signed = request["volume"] * (1 if request["type"] == 0 else -1)
            if not math.isclose(signed, permit.revalidation["signed_lots"], abs_tol=1e-9):
                raise ValueError("quote revalidation cannot enlarge or reverse approved quantity")
            # All non-quote state still must match. A quote change is accepted
            # only after another full Stage16 evaluation at this actual quote.
            expected_state["quote"] = current_state["quote"]
        if current_state != expected_state:
            raise ValueError("native economic state changed; risk revalidation required")
        if permit.revalidation is not None:
            assert cfg is not None
            now = datetime.now(UTC)
            proof = evaluate_quote(permit.revalidation, quote, current_state["account"], now)
            p, i, a = cfg.policy, cfg.instrument, cfg.account
            assert p is not None and i is not None and a is not None
            entry = quote["ask"] if request["type"] == 0 else quote["bid"]
            adverse = entry * (1 - float(p.horizon_stress_fraction or 0) * (1 if signed > 0 else -1))
            calculated = self.calculations("BUY" if signed > 0 else "SELL", request["volume"], entry, adverse)
            expected_profit = (adverse - entry) * signed * i.ounces_per_lot * a.ledger_per_usd
            if (not math.isclose(calculated["profit"], expected_profit, rel_tol=1e-6,
                                 abs_tol=.5 * 10 ** -identity["account"].currency_digits)
                or not 0 <= calculated["margin"] <= proof["measurements"]["margin_required"] + 1e-6):
                raise ValueError("native fresh margin/profit incompatible with risk units")
            # Reverify account/permissions after calculations. Quote observation
            # and broker execution are inherently not an atomic transaction.
            latest = self._identity(True)
            latest_symbol = self.__native.symbol_info(self.terminal.symbol)
            latest_positions, latest_orders = self.__native.positions_get(), self.__native.orders_get()
            if (any(getattr(latest["account"], k) != v for k, v in current_state["account"].items())
                or latest_symbol is None or any(getattr(latest_symbol, k) != v for k, v in current_state["symbol"].items())
                or latest_positions is None or latest_orders is None or latest_positions or latest_orders):
                raise ValueError("account/book/capabilities changed during fresh risk validation")
        if symbol.trade_exemode != 2 and request["price"] != quote["ask" if request["type"] == 0 else "bid"]:
            raise ValueError("native executable request price changed")
        final_now = datetime.now(UTC)
        if not 0 <= final_now.timestamp() - quote["time_msc"] / 1000 <= permit.max_quote_age_seconds:
            raise ValueError("native quote is stale or from the future")
        if final_now >= permit.expires_utc:
            raise ValueError("approval expired during native validation")
        return proof

    def check(self, request: dict[str, Any], permit: Permit) -> dict[str, Any] | None:
        self.validation_stage = "CHECK_BOUNDARY"
        proof = self._boundary(request, permit)
        self.validation_stage = "ORDER_CHECK_CALL"
        result = self.__native.order_check(request)
        self.validation_stage = "ORDER_CHECK_RETURNED"
        return None if result is None else {"retcode": result.retcode} | (
            {"risk_revalidation": proof} if proof is not None else {})

    def send(self, request: dict[str, Any], permit: Permit) -> dict[str, Any] | None:
        self.validation_stage = "SEND_BOUNDARY"
        proof = self._boundary(request, permit)
        if "position" in request:
            if self.cleanup >= int(self.config.cleanup_request_budget or 0):
                raise ValueError("cleanup request budget exhausted")
            self.cleanup += 1
        else:
            if self.entries >= min(int(self.config.entry_request_budget or 0), int(self.config.entry_budget or 0)):
                raise ValueError("entry request budget exhausted")
            self.entries += 1
        self.validation_stage = "ORDER_SEND_CALL"
        result = self.__native.order_send(request)
        self.validation_stage = "ORDER_SEND_RETURNED"
        self._identity(False)  # A changed account cannot inherit this response or cleanup.
        return None if result is None else {k: getattr(result, k) for k in (
            "retcode", "deal", "order", "volume", "price", "request_id", "retcode_external")} | (
            {"risk_revalidation": proof} if proof is not None else {})

    def shutdown(self) -> None:
        if self.__reader is not None:
            self.__reader.shutdown()
