"""Allowlisted official MT5 reads only; vendor import is optional and lazy.

Never select a terminal/account implicitly, change Market Watch, or expose a
generic native-method dispatcher. Native exceptions and account fields are
not included in reports: they can contain private identity information.
"""

from __future__ import annotations

import importlib
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .config import ShadowConfig


class ReadFailure(RuntimeError):  # noqa: N818 - consistent named domain failures
    """Sanitized bounded read failure; caller halts or reconnects explicitly."""


class IdentityFailure(ReadFailure):
    """Never retry a mismatched terminal/account as a transient data error."""


class ReadOnlyFeed(Protocol):
    def connect(self) -> dict[str, Any]: ...
    def verify(self) -> dict[str, Any]: ...
    def ticks(self, start: datetime, count: int) -> list[dict[str, Any]]: ...
    def latest(self) -> dict[str, Any] | None: ...
    def shutdown(self) -> None: ...


class MT5ReadOnly:
    def __init__(self, config: ShadowConfig, *, native: Any = None) -> None:
        self.config = config
        self.__native = native
        self.connected = False

    def connect(self) -> dict[str, Any]:
        if not self.config.configured:
            raise IdentityFailure("explicit terminal, demo identity, company and symbol required")
        if not Path(str(self.config.terminal_path)).is_file():
            raise IdentityFailure("configured terminal executable is unavailable")
        try:
            if self.__native is None:
                self.__native = importlib.import_module("MetaTrader5")
            # No login/password/server arguments: never switch the authenticated account.
            ok = self.__native.initialize(self.config.terminal_path,
                                          timeout=self.config.initialize_timeout_ms)
        except Exception:
            self.shutdown()
            raise ReadFailure("native MT5 initialization unavailable") from None
        if not ok:
            self.shutdown()
            raise ReadFailure("native MT5 initialization failed")
        self.connected = True
        try:
            return self.verify()
        except Exception:
            self.shutdown()
            raise

    def verify(self) -> dict[str, Any]:
        if not self.connected:
            raise ReadFailure("adapter disconnected")
        try:
            terminal = self.__native.terminal_info()
            account = self.__native.account_info()
            symbol = self.__native.symbol_info(self.config.symbol)
        except Exception:
            raise ReadFailure("identity metadata read failed") from None
        if terminal is None or not terminal.connected:
            raise ReadFailure("terminal disconnected")
        expected_parent = os.path.normcase(str(Path(str(self.config.terminal_path)).parent.resolve()))
        actual_parent = os.path.normcase(str(Path(terminal.path).resolve()))
        if actual_parent != expected_parent:
            raise IdentityFailure("terminal identity mismatch")
        if account is None or (
            account.login != self.config.expected_login
            or account.server != self.config.expected_server
            or account.company != self.config.expected_company
            or "exness" not in str(account.company).casefold()
            or account.trade_mode != self.__native.ACCOUNT_TRADE_MODE_DEMO
        ):
            raise IdentityFailure("configured demo account identity could not be verified")
        if symbol is None or symbol.name != self.config.symbol or not symbol.visible:
            raise IdentityFailure("exact configured symbol must already be visible")
        values = {name: getattr(symbol, name, None) for name in (
            "trade_contract_size", "trade_tick_size", "point", "volume_min", "volume_step", "volume_max")}
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in values.values()):
            raise IdentityFailure("invalid instrument units")
        if type(symbol.digits) is not int or not 0 <= symbol.digits <= 10 or symbol.volume_min > symbol.volume_max:
            raise IdentityFailure("invalid price precision or quantity bounds")
        return {"identity_verified": True, "demo_mode_verified": True,
                "symbol_metadata": values | {"digits": symbol.digits,
                    "currency_profit": symbol.currency_profit,
                    "currency_margin": symbol.currency_margin,
                    "volume_units": "MT5 lots; physical contract units require supplied verification",
                    "price_units": "native quote units; not assumed historical USD/ounce units"},
                "provenance": "configured MT5 symbol metadata, not verified costs or feed transfer",
                "observed_utc": datetime.now(UTC).isoformat(),
                "broker_execution": "unavailable", "account_values_logged": False}

    def ticks(self, start: datetime, count: int) -> list[dict[str, Any]]:
        if start.tzinfo is None or start.utcoffset() is None or not 0 < count <= self.config.batch_size:
            raise ValueError("aware UTC boundary and bounded tick count required")
        metadata = self.verify()  # Identity checked on every poll and after reconnection.
        try:
            rows = self.__native.copy_ticks_from(self.config.symbol, start.astimezone(UTC),
                                                 count, self.__native.COPY_TICKS_ALL)
            if rows is None:
                raise ReadFailure("MT5 tick history unavailable")
            if len(rows) > count:
                raise ReadFailure("tick batch exceeded requested bound")
            result = [{name: row[name].item() for name in rows.dtype.names} for row in rows]
            for row in result:
                validate_quote(row, metadata["symbol_metadata"])
            return result
        except ReadFailure:
            raise
        except Exception:
            raise ReadFailure("MT5 tick retrieval failed") from None

    def latest(self) -> dict[str, Any] | None:
        metadata = self.verify()
        try:
            tick = self.__native.symbol_info_tick(self.config.symbol)
            if tick is None:
                return None
            result = {name: getattr(tick, name) for name in ("time_msc", "bid", "ask")}
            validate_quote(result, metadata["symbol_metadata"])
            return result
        except ReadFailure:
            raise
        except Exception:
            raise ReadFailure("current quote unavailable") from None

    def shutdown(self) -> None:
        try:
            if self.__native is not None:
                self.__native.shutdown()
        except Exception:
            # Fail-safe caller has already halted; never mask the original failure.
            pass
        finally:
            self.connected = False


def validate_quote(row: dict[str, Any], metadata: dict[str, Any]) -> None:
    if type(row.get("time_msc")) is not int or row["time_msc"] < 0:
        raise ReadFailure("invalid quote milliseconds")
    prices = [row.get("bid"), row.get("ask")]
    if any(not isinstance(p, (float, int)) or not math.isfinite(p) or p <= 0 for p in prices):
        raise ReadFailure("invalid executable quote sides")
    if row["ask"] < row["bid"]:
        raise ReadFailure("crossed executable quote")
    for p in (row["bid"], row["ask"]):
        if abs(p - round(p, metadata["digits"])) > max(1e-9, abs(p) * 1e-12):
            raise ReadFailure("quote incompatible with supplied native price precision")
        units = p / metadata["trade_tick_size"]
        if abs(units - round(units)) > 1e-6:
            raise ReadFailure("quote incompatible with native price increment")
