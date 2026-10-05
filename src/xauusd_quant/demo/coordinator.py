"""Risk-authorized durable execution and broker-history reconciliation.

No response is a fill. No timeout permits a retry. Flatness and cash movements
are reconciled to the dedicated account, not inferred from sent close requests.
"""
from __future__ import annotations

import math
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from typing import Any

from ..execution.config import content_hash
from ..risk.contracts import (
    AccountSnapshot,
    ExecutionUpdate,
    HealthSnapshot,
    MarketSnapshot,
    RiskRequest,
    finite,
    resolved,
)
from ..risk.engine import RiskEngine
from ..risk.policy import RiskConfiguration
from ..shadow.adapter import IdentityFailure, ReadFailure, validate_quote
from ..shadow.config import ShadowConfig
from .broker import Broker, Permit, approval_state, classify, filling
from .config import DemoConfig
from .journal import Journal

STATES = {"CREATED", "RISK_APPROVED", "CHECKED", "SUBMISSION_ATTEMPTED", "ACCEPTED_OR_PENDING",
          "PARTIALLY_FILLED", "FILLED", "REJECTED", "UNKNOWN", "RECONCILIATION_REQUIRED", "CLOSED_OR_CANCELLED"}
TERMINAL = {"FILLED", "REJECTED", "CLOSED_OR_CANCELLED"}


def settings(config: DemoConfig, risk: RiskConfiguration, *, offline_synthetic: bool) -> None:
    if not config.configured or not risk.configured:
        raise ValueError("complete explicit execution and risk settings required")
    p, a, i = risk.policy, risk.account, risk.instrument
    assert p is not None and a is not None and i is not None
    if not offline_synthetic and any(v.status != "verified_supplied" for v in (p, a, i)):
        raise ValueError("synthetic metadata/limits cannot enable native demo execution")
    if config.risk_policy_id != p.policy_id or config.protection != "PROCESS_EXIT" or p.sizing_method != "horizon_stress":
        raise ValueError("matching risk policy and supported process-exit horizon policy required; broker stops unavailable")
    if config.run_type == "SMOKE" and (p.expected_portfolio_id != "EXECUTION_SMOKE_V001"
        or p.allowed_allocation_ids != ("SMOKE_V001",) or p.alpha_contracts or p.require_diagnostic):
        raise ValueError("dedicated no-model smoke risk configuration required")
    if config.run_type == "SMOKE" and config.shutdown_policy != "CLOSE_OWNED":
        raise ValueError("smoke lifecycle requires declared close-owned shutdown")
    if (float(config.quantity_lots or 0) > min(float(config.max_quantity_lots or 0), float(config.max_net_lots or 0), float(config.max_gross_lots or 0))
        or float(config.max_net_lots or 0) > p.max_net_lots
        or float(config.max_gross_lots or 0) > p.max_gross_intended_lots
        or float(config.max_duration_seconds or 0) > p.max_holding_seconds
        or p.max_orders_per_window < int(config.entry_request_budget or 0) + int(config.cleanup_request_budget or 0)
        or p.max_turnover_lots_per_window < 2 * float(config.quantity_lots or 0) * int(config.entry_budget or 0)):
        raise ValueError("exposure/holding/order/turnover settings must reserve cleanup capacity")


class Coordinator:
    def __init__(self, config: DemoConfig, terminal: ShadowConfig, risk: RiskConfiguration,
                 broker: Broker, journal: Journal, run_id: str, code: str, *,
                 offline_synthetic: bool = False) -> None:
        settings(config, risk, offline_synthetic=offline_synthetic)
        if not terminal.configured:
            raise ValueError("explicit demo terminal identity required")
        self.config, self.terminal, self.broker, self.journal = config, terminal, broker, journal
        self.run_id, self.code = run_id, code
        self.__authority = object()
        broker.bind(self.__authority)
        self.intents: dict[str, dict[str, Any]] = {}
        self.baseline: dict[str, Any] | None = None
        self.last_snapshot: dict[str, Any] | None = None
        self.last_reconciled = False
        self.armed = False
        self.recovered = bool(journal.rows)
        self.halts: list[str] = []
        self.opened_utc: datetime | None = None
        # A recovery run gets a new bounded cleanup window, while the original
        # history boundary and all losses/reservations remain persisted.
        self.run_started_utc: datetime | None = None
        checkpoints = [r["data"] for r in journal.rows if r["event"] == "checkpoint"]
        if checkpoints:
            if journal.rows[-1]["event"] != "checkpoint":
                raise ValueError("uncheckpointed journal tail; reconcile without resetting risk history")
            saved = checkpoints[-1]
            if saved["config_sha256"] != config.identity or saved["terminal_sha256"] != terminal.identity or saved["code_identity"] != code:
                raise ValueError("incompatible persisted execution state; retain journal and reconcile")
            self.risk = RiskEngine.restore(risk, self._record, saved["risk"])
            self.intents, self.baseline = deepcopy(saved["intents"]), deepcopy(saved["baseline"])
            self.halts = list(saved["halts"])
            self.opened_utc = datetime.fromisoformat(saved["opened_utc"]) if saved["opened_utc"] else None
            if any(row["state"] not in STATES for row in self.intents.values()):
                raise ValueError("invalid persisted execution lifecycle")
        else:
            if journal.rows:
                raise ValueError("incomplete journal without checkpoint; do not initialize fresh risk")
            self.risk = RiskEngine(risk, self._record)

    def _record(self, table: str, row: dict[str, Any]) -> None:
        self.journal.append(table, {"run_id": self.run_id, **row})

    def persist(self) -> None:
        self.journal.append("checkpoint", {"run_id": self.run_id, "code_identity": self.code,
            "config_sha256": self.config.identity, "terminal_sha256": self.terminal.identity,
            "risk": self.risk.checkpoint(), "intents": deepcopy(self.intents),
            "baseline": deepcopy(self.baseline), "halts": self.halts.copy(),
            "opened_utc": self.opened_utc.isoformat() if self.opened_utc else None})

    def halt(self, reason: str, now: datetime) -> None:
        self.armed = False
        if reason not in self.halts:
            self.halts.append(reason)
        self.risk.kill(now, reason)
        self.persist()

    def _validate(self, snapshot: dict[str, Any], now: datetime) -> None:
        p, a, i = self.risk.configuration.policy, self.risk.configuration.account, self.risk.configuration.instrument
        assert p is not None and a is not None and i is not None
        receipt = datetime.fromisoformat(snapshot["received_utc"])
        if (snapshot.get("identity_verified") is not True or receipt.tzinfo is None
            or not 0 <= (now - receipt).total_seconds() <= p.max_account_age_seconds):
            raise IdentityFailure("fresh verified account required")
        account, native = snapshot["account"], snapshot["symbol"]
        if ({0: "NETTING", 2: "HEDGING"}.get(account["margin_mode"]) != self.config.expected_account_mode
            or account["currency"] != self.config.expected_account_currency
            or type(account["currency_digits"]) is not int or not 0 <= account["currency_digits"] <= 8
            or not all(finite(account[k]) for k in ("balance", "equity", "margin_free", "credit"))
            or account["credit"] != 0 or account["margin_free"] < 0):
            raise ValueError("account mode/currency/credit/balances unsupported or changed")
        units = (("trade_contract_size", i.ounces_per_lot), ("trade_tick_size", i.tick_size),
                 ("volume_min", i.min_lots), ("volume_step", i.lot_step), ("volume_max", i.max_lots),
                 ("digits", i.price_decimals))
        if any(native[k] != value for k, value in units):
            raise ValueError("supplied instrument terms differ from native capabilities")
        validate_quote(snapshot["quote"], native)
        filling(str(self.config.filling_policy), native["trade_exemode"], native["filling_mode"])
        if any(not finite(pos["volume"]) or pos["volume"] <= 0 or pos["type"] not in (0, 1)
               for pos in snapshot["positions"]):
            raise ValueError("invalid broker position units")
        if len(snapshot["positions"]) > 1:
            raise ValueError("one dedicated shared position only; multiple hedged positions unsupported")
        if len({d["ticket"] for d in snapshot["deals"]}) != len(snapshot["deals"]):
            raise ValueError("duplicate broker deal tickets cannot inflate cash or quantity")
        for d in snapshot["deals"]:
            if (type(d["ticket"]) is not int or d["ticket"] <= 0 or type(d["time_msc"]) is not int
                or not all(finite(d[k]) for k in ("volume", "price", "profit", "commission", "swap", "fee"))
                or d["volume"] < 0 or (d["type"] in (0, 1) and (d["volume"] <= 0 or d["price"] <= 0))):
                raise ValueError("invalid broker deal quantities/cash flows")

    def initialize(self, snapshot: dict[str, Any], now: datetime, *, arm: bool) -> None:
        if self.run_started_utc is None:
            self.run_started_utc = now
        self._validate(snapshot, now)
        if self.baseline is None:
            if snapshot["positions"] or snapshot["orders"]:
                raise ValueError("dedicated flat account required; never adopt unrelated exposure")
            self.baseline = {"balance": snapshot["account"]["balance"], "received_utc": snapshot["received_utc"],
                "deal_tickets": [d["ticket"] for d in snapshot["deals"]],
                "order_tickets": [o["ticket"] for o in snapshot["history_orders"]]}
            self.opened_utc = now
            self.persist()
        self.reconcile(snapshot, now)
        if arm:
            # A successful check-only run did not execute. Reusing it requires
            # fresh reconciliation and explicit arming; submissions never qualify.
            pure_precheck = bool(self.intents) and all(
                v["state"] == "REJECTED" and not v.get("submission_utc")
                and (v.get("check") or {}).get("retcode") == 0 for v in self.intents.values()
            )
            if (self.recovered and not pure_precheck) or self.halts or self.risk.state.state != "READY" or not snapshot["permissions"]:
                raise ValueError("arming requires fresh smoke/strategy session, verified permissions and READY risk")
            self.armed = True
            self._record("arming", {"mode": "DEMO", "run_type": self.config.run_type, "received_utc": now.isoformat()})

    def _account(self, snapshot: dict[str, Any], now: datetime, *, reconcile: bool) -> None:
        a, i = self.risk.configuration.account, self.risk.configuration.instrument
        assert a is not None and i is not None
        positions = snapshot["positions"]
        signed = sum(p["volume"] * (1 if p["type"] == 0 else -1) for p in positions)
        losses = [v["decision"]["measurements"].get("loss_per_lot", 0)
                  for v in self.intents.values() if not v.get("closing") and v.get("decision")]
        # Preserve the supplied conditional loss convention; this is not a loss ceiling.
        estimate = max(losses, default=0) * abs(signed)
        baseline = self.baseline or {}
        flows = sum(d["profit"] + d["commission"] + d["swap"] + d["fee"] for d in snapshot["deals"]
                    if d["ticket"] not in baseline.get("deal_tickets", []) and d["type"] == 2)
        account = snapshot["account"]
        mark_id = content_hash({"snapshot": snapshot, "verified_risk_sequence": self.risk.state.sequence})
        self.risk.observe_account(AccountSnapshot(mark_id, now, now, a.specification_id,
            account["balance"], account["equity"], account["margin_free"], signed, True,
            snapshot["permissions"], flows, estimate), reconcile=reconcile)

    def _execution(self, intent: dict[str, Any], status: str, cumulative: float, now: datetime) -> None:
        identity = intent["intent_id"]
        if identity not in self.risk.state.reservations:
            return
        reservation = self.risk.state.reservations[identity]
        if reservation.status == status and math.isclose(reservation.cumulative_filled, cumulative, abs_tol=1e-9):
            return
        self.risk.execution_update(ExecutionUpdate(
            content_hash({"intent": identity, "status": status, "cumulative": cumulative}), identity,
            now, now, status, cumulative, str(intent["order_ticket"]) if intent.get("order_ticket") else None))

    def reconcile(self, snapshot: dict[str, Any], now: datetime, *, retire_unsubmitted: bool = True) -> dict[str, Any]:
        self.last_reconciled = False
        self._validate(snapshot, now)
        assert self.baseline is not None
        baseline = self.baseline
        orders = {o["ticket"]: o for o in [*snapshot["history_orders"], *snapshot["orders"]]}
        owned_order_ids: set[int] = set()
        pending = {o["ticket"] for o in snapshot["orders"]}
        for intent in self.intents.values():
            if not intent.get("submission_utc"):
                if retire_unsubmitted and intent["state"] not in TERMINAL:
                    intent["state"] = "REJECTED"
                    self._execution(intent, "rejected", 0, now)
                continue
            request = intent["request"]
            start = datetime.fromisoformat(intent["submission_utc"]).timestamp() - 2
            matches = [o for o in orders.values() if o["symbol"] == self.terminal.symbol
                and o["magic"] == self.config.magic and o["comment"] == request["comment"]
                and o["type"] == request["type"] and math.isclose(o["volume_initial"], request["volume"], abs_tol=1e-9)
                and start <= o["time_setup_msc"] / 1000 <= now.timestamp() + 2
                and (not intent.get("order_ticket") or o["ticket"] == intent["order_ticket"])]
            if len(matches) > 1:
                self.halt("AMBIGUOUS_ORDER_OWNERSHIP", now)
                return {"reconciled": False}
            if not matches:
                if intent["state"] == "REJECTED":
                    self._execution(intent, "rejected", 0, now)
                elif intent["state"] not in TERMINAL:
                    intent["state"] = "UNKNOWN"
                    self._execution(intent, "unknown", intent.get("filled_lots", 0), now)
                continue
            order = matches[0]
            intent["order_ticket"] = order["ticket"]
            owned_order_ids.add(order["ticket"])
            deals = [d for d in snapshot["deals"] if d["order"] == order["ticket"] and d["type"] in (0, 1)]
            previous_deals = {d["ticket"]: d for d in intent.get("deals", [])}
            if any(d["ticket"] in previous_deals and content_hash(d) != content_hash(previous_deals[d["ticket"]]) for d in deals):
                self.halt("BROKER_DEAL_CORRECTION_REQUIRES_RECONCILIATION", now)
                return {"reconciled": False}
            if any(d["symbol"] != self.terminal.symbol or d["magic"] != self.config.magic
                   or d["type"] != request["type"] or d["entry"] not in ((1,) if intent["closing"] else (0,)) for d in deals):
                self.halt("DEAL_OWNERSHIP_OR_REVERSAL_MISMATCH", now)
                return {"reconciled": False}
            if intent["closing"] and any(d["position_id"] != intent["position_identifier"] for d in deals):
                self.halt("CLOSE_POSITION_IDENTITY_MISMATCH", now)
                return {"reconciled": False}
            cumulative = sum(d["volume"] for d in deals)
            if cumulative > request["volume"] + 1e-9 or cumulative < intent.get("filled_lots", 0) - 1e-9:
                self.halt("FILL_QUANTITY_OR_HISTORY_INCOMPLETE", now)
                return {"reconciled": False}
            identifiers = {d["position_id"] for d in deals}
            if len(identifiers) > 1:
                self.halt("MULTIPLE_POSITION_IDENTITIES", now)
                return {"reconciled": False}
            intent["filled_lots"] = cumulative
            if identifiers:
                intent["position_identifier"] = next(iter(identifiers))
            intent["deals"] = deepcopy(deals)
            terminal = order["state"] in (2, 4, 5, 6) and order["ticket"] not in pending
            if math.isclose(cumulative, request["volume"], abs_tol=1e-9) and terminal:
                intent["state"] = "FILLED"
                self._execution(intent, "filled", cumulative, now)
            elif cumulative:
                intent["state"] = "PARTIALLY_FILLED"
                self._execution(intent, "partial", cumulative, now)
                if terminal:
                    intent["state"] = "CLOSED_OR_CANCELLED"
                    self._execution(intent, "cancelled", cumulative, now)
            elif terminal and order["state"] in (2, 5, 6):
                intent["state"] = "REJECTED"
                self._execution(intent, "rejected", 0, now)
            else:
                intent["state"] = "ACCEPTED_OR_PENDING"
                self._execution(intent, "acknowledged", 0, now)
        owned_positions = {v.get("position_identifier") for v in self.intents.values()
                           if v.get("position_identifier") and not v["closing"]}
        foreign_orders = [o for o in orders.values() if o["ticket"] not in baseline["order_tickets"] and o["ticket"] not in owned_order_ids]
        new_deals = [d for d in snapshot["deals"] if d["ticket"] not in baseline["deal_tickets"]]
        if (foreign_orders or any(p["identifier"] not in owned_positions or p["symbol"] != self.terminal.symbol
                                  or p["magic"] != self.config.magic for p in snapshot["positions"])
            or any(d["type"] in (0, 1) and d["order"] not in owned_order_ids for d in new_deals)
            or any(d["type"] not in (0, 1, 2) for d in new_deals)):
            self.halt("EXTERNAL_OR_UNVERIFIED_ACTIVITY", now)
            return {"reconciled": False}
        expected = sum(d["volume"] * (1 if d["type"] == 0 else -1) for d in new_deals if d["type"] in (0, 1))
        actual = sum(p["volume"] * (1 if p["type"] == 0 else -1) for p in snapshot["positions"])
        cash = sum(d["profit"] + d["commission"] + d["swap"] + d["fee"] for d in new_deals)
        balance_difference = snapshot["account"]["balance"] - baseline["balance"] - cash
        cash_tolerance = max(1e-6, .5 * 10 ** -snapshot["account"]["currency_digits"])
        if not math.isclose(expected, actual, abs_tol=1e-8) or not math.isclose(balance_difference, 0, abs_tol=cash_tolerance):
            self.halt("POSITION_OR_CASH_RECONCILIATION_INCOMPLETE", now)
            return {"reconciled": False, "balance_discrepancy": balance_difference}
        self._account(snapshot, now, reconcile=True)
        self.last_snapshot = deepcopy(snapshot)
        self.last_reconciled = not self.risk.state.needs_reconciliation
        row = {"received_utc": now.isoformat(), "reconciled": not self.risk.state.needs_reconciliation,
               "signed_lots": actual, "broker_cash_change": cash, "balance_discrepancy": balance_difference,
               "verified_flat": not snapshot["positions"] and not snapshot["orders"]
                    and not self.risk.state.reservations and all(v["state"] in TERMINAL for v in self.intents.values())}
        self._record("broker_snapshot", snapshot)
        self._record("reconciliation", row)
        self.persist()
        return row

    def propose(self, request: RiskRequest, snapshot: dict[str, Any], now: datetime,
                health: dict[str, HealthSnapshot] | None = None, *, expires_utc: datetime) -> dict[str, Any]:
        if request.intent_id in self.intents:
            old = self.intents[request.intent_id]
            if old["risk_request_sha256"] != request.identity:
                raise ValueError("durable intent identity collision")
            return deepcopy(old)
        self._validate(snapshot, now)
        if request.received_utc != now or request.decision_utc > now:
            raise ValueError("current causal decision/receipt timestamps required")
        if len(self.intents) >= 512 or not self.opened_utc:
            raise ValueError("bounded initialized intent state required")
        close = request.target_lots == 0 and bool(snapshot["positions"])
        duration = (now - (self.run_started_utc or self.opened_utc)).total_seconds()
        if (not self.armed and not close) or duration >= float(self.config.max_duration_seconds or 0):
            raise ValueError("execution disarmed or run deadline reached")
        if not close and duration >= float(self.config.max_duration_seconds or 0) - float(self.config.cleanup_seconds or 0):
            raise ValueError("entry cutoff preserves cleanup capacity")
        if expires_utc.tzinfo is None or not 0 < (expires_utc - now).total_seconds() <= float(self.config.intent_ttl_seconds or 0):
            raise ValueError("unexpired bounded intent required")
        reconciliation = self.reconcile(snapshot, now)
        if not reconciliation.get("reconciled") or snapshot["orders"] or self.risk.state.reservations:
            raise ValueError("reconcile all pending/uncertain activity before proposing changes")
        position = snapshot["positions"][0] if close else None
        if close:
            assert position is not None
            if not any(v.get("position_identifier") == position["identifier"] and not v["closing"] for v in self.intents.values()):
                raise ValueError("durable owned position proof required for close")
        entry_count = sum(not v["closing"] for v in self.intents.values() if v["run_id"] == self.run_id)
        close_count = sum(v["closing"] for v in self.intents.values() if v["run_id"] == self.run_id)
        if (close and close_count >= int(self.config.cleanup_request_budget or 0)) or (not close and entry_count >= int(self.config.entry_budget or 0)):
            raise ValueError("declared entry/cleanup intent budget exhausted")
        intent = {"intent_id": request.intent_id, "run_id": self.run_id, "state": "CREATED", "closing": close,
            "created_utc": now.isoformat(), "expires_utc": expires_utc.isoformat(),
            "risk_request_sha256": request.identity, "risk_request": resolved(request), "filled_lots": 0.0}
        if self.config.run_type == "STRATEGY" and health:
            p = self.risk.configuration.policy
            assert p is not None
            intent["health_valid_until"] = min(min(h.event_utc + timedelta(seconds=p.max_health_age_seconds), h.forecast_expiry_utc)
                                               for h in health.values()).isoformat()
        self.intents[request.intent_id] = intent
        self.persist()
        q = snapshot["quote"]
        market = MarketSnapshot(datetime.fromtimestamp(q["time_msc"] / 1000, UTC), now, q["bid"], q["ask"])
        decision = self.risk.evaluate_smoke(request, market) if self.config.run_type == "SMOKE" else self.risk.evaluate(request, market, health or {})
        intent["decision"] = decision
        change = decision["approved_change_lots"]
        if not change or request.intent_id not in self.risk.state.reservations:
            intent["state"] = "REJECTED"
            self.persist()
            return deepcopy(intent)
        kind = 0 if change > 0 else 1
        volume = abs(change)
        if volume > float(self.config.max_quantity_lots or 0) or not close and volume > min(float(self.config.max_net_lots or 0), float(self.config.max_gross_lots or 0)):
            self._execution(intent, "rejected", 0, now)
            intent["state"] = "REJECTED"
            self.persist()
            return deepcopy(intent)
        native = snapshot["symbol"]
        comment = f"{self.config.project_tag}:{content_hash({'run': self.run_id, 'intent': request.intent_id})[:20]}"
        broker_request = {"action": 1, "symbol": self.terminal.symbol, "volume": volume,
            "type": kind, "magic": self.config.magic, "comment": comment, "deviation": self.config.deviation_points,
            "type_filling": filling(str(self.config.filling_policy), native["trade_exemode"], native["filling_mode"]), "type_time": 0}
        if native["trade_exemode"] != 2:
            broker_request["price"] = q["ask"] if kind == 0 else q["bid"]
        if close:
            assert position is not None
            broker_request["position"] = position["ticket"]
            intent["position_identifier"] = position["identifier"]
        intent.update({"state": "RISK_APPROVED", "request": broker_request, "approval_snapshot": deepcopy(snapshot)})
        self.persist()
        return deepcopy(intent)

    def submit(self, intent_id: str, current: dict[str, Any], now: datetime, *, precheck_only: bool = False) -> dict[str, Any]:
        intent = self.intents[intent_id]
        if intent["state"] != "RISK_APPROVED" or intent.get("submission_utc"):
            return deepcopy(intent)
        self._validate(current, now)
        if not self.reconcile(current, now, retire_unsubmitted=False).get("reconciled"):
            raise ValueError("current account/deal state must reconcile before check/send")
        reservation = self.risk.state.reservations.get(intent_id)
        if (reservation is None or reservation.status != "approved"
            or self.risk.state.decisions[intent_id] != intent["decision"]
            or not math.isclose(abs(reservation.signed_change), intent["request"]["volume"], abs_tol=1e-9)
            or now >= datetime.fromisoformat(intent["expires_utc"])
            or intent.get("health_valid_until") and now >= datetime.fromisoformat(intent["health_valid_until"])):
            raise ValueError("current unexpired risk approval/reservation required")
        previous = intent["approval_snapshot"]
        # Conservative declared fallback: any changed economic quote/account/book
        # abstains; do not silently reuse cached risk approval or chase the price.
        if not intent["closing"] and (current["quote"]["bid"] != previous["quote"]["bid"] or current["quote"]["ask"] != previous["quote"]["ask"]
            or current["account"] != previous["account"] or current["positions"] != previous["positions"]
            or current["orders"] != previous["orders"]):
            self.halt("MATERIAL_STATE_CHANGED_REVALIDATION_REQUIRED", now)
            raise ValueError("material state changed; abandon this unsubmitted intent and reconcile")
        p, a, i = self.risk.configuration.policy, self.risk.configuration.account, self.risk.configuration.instrument
        assert p is not None and a is not None and i is not None
        quote = current["quote"]
        market = MarketSnapshot(datetime.fromtimestamp(quote["time_msc"] / 1000, UTC), now, quote["bid"], quote["ask"])
        if self.risk._market_reasons(market, now, reducing=intent["closing"]):
            raise ValueError("fresh executable quote still required at submission")
        request = intent["request"]
        side = "BUY" if request["type"] == 0 else "SELL"
        entry = quote["ask"] if side == "BUY" else quote["bid"]
        adverse = entry * (1 - float(p.horizon_stress_fraction or 0) * (1 if side == "BUY" else -1))
        calculation = self.broker.calculations(side, request["volume"], entry, adverse)
        expected = (adverse - entry) * (1 if side == "BUY" else -1) * request["volume"] * i.ounces_per_lot * a.ledger_per_usd
        profit_tolerance = max(1e-6, .5 * 10 ** -current["account"]["currency_digits"])
        if not math.isclose(calculation["profit"], expected, rel_tol=1e-6, abs_tol=profit_tolerance):
            raise ValueError("native profit calculation incompatible with supplied CFD/conversion units")
        if not intent["closing"] and (calculation["margin"] < 0
            or calculation["margin"] > reservation.margin + 1e-6
            or calculation["margin"] + p.minimum_free_margin > current["account"]["margin_free"]):
            raise ValueError("native margin exceeds risk reservation or funds")
        permit = Permit(self.__authority, content_hash(request),
            datetime.fromisoformat(intent["expires_utc"]),
            approval_state(current, closing=intent["closing"]), p.max_quote_age_seconds)
        check = self.broker.check(request, permit)
        intent["check"] = check
        self._record("broker_precheck", {"intent_id": intent_id, "result": check, "calculations": calculation,
                                        "received_utc": now.isoformat()})
        if check is None or check.get("retcode") != 0:  # order_check success != order_send DONE.
            intent["state"] = "REJECTED"
            self._execution(intent, "rejected", 0, now)
            self.persist()
            return deepcopy(intent)
        intent["state"] = "CHECKED"
        self.persist()
        if precheck_only:
            intent["state"] = "REJECTED"
            self._execution(intent, "rejected", 0, now)
            self.persist()
            return deepcopy(intent)
        # Refresh after precheck, then the native boundary rechecks identity again.
        refreshed = self.broker.snapshot(self.opened_utc or now)
        fresh_now = datetime.fromisoformat(refreshed["received_utc"])
        self._validate(refreshed, fresh_now)
        if not self.reconcile(refreshed, fresh_now, retire_unsubmitted=False).get("reconciled"):
            raise ValueError("post-check broker state must reconcile before submission")
        if (fresh_now >= datetime.fromisoformat(intent["expires_utc"])
            or intent.get("health_valid_until") and fresh_now >= datetime.fromisoformat(intent["health_valid_until"])
            or fresh_now >= (self.run_started_utc or now) + timedelta(seconds=float(self.config.max_duration_seconds or 0))
            or not intent["closing"] and (not self.armed or self.risk.state.halts)
            or refreshed["orders"] != current["orders"]
            or (not intent["closing"] and (refreshed["account"] != current["account"] or refreshed["positions"] != current["positions"]
                or any(refreshed["quote"][k] != current["quote"][k] for k in ("bid", "ask"))))
            or (intent["closing"] and (refreshed["symbol"]["trade_exemode"] != 2
                and any(refreshed["quote"][k] != current["quote"][k] for k in ("bid", "ask"))))):
            self.halt("POST_CHECK_STATE_CHANGED", fresh_now)
            raise ValueError("precheck does not permit stale state or expired approval")
        fresh_market = MarketSnapshot(datetime.fromtimestamp(refreshed["quote"]["time_msc"] / 1000, UTC), fresh_now,
                                     refreshed["quote"]["bid"], refreshed["quote"]["ask"])
        if self.risk._market_reasons(fresh_market, fresh_now, reducing=intent["closing"]):
            raise ValueError("post-check quote became stale/invalid")
        if intent["closing"]:
            position = refreshed["positions"][0] if len(refreshed["positions"]) == 1 else None
            if (position is None or position["ticket"] != request["position"]
                or position["identifier"] != intent["position_identifier"]
                or position["type"] == request["type"] or position["volume"] < request["volume"] - 1e-9
                or len(self.risk.state.reservations) != 1):
                raise ValueError("close quantity/identity changed; prevent reversal")
        intent["state"], intent["submission_utc"] = "SUBMISSION_ATTEMPTED", fresh_now.isoformat()
        self.persist()  # Crash here is uncertain even if native call never started.
        permit = Permit(self.__authority, content_hash(request),
            datetime.fromisoformat(intent["expires_utc"]),
            approval_state(refreshed, closing=intent["closing"]), p.max_quote_age_seconds)
        try:
            response = self.broker.send(request, permit)
        except (IdentityFailure, ReadFailure, ValueError, RuntimeError, OSError):
            response = None
            self.armed = False
        intent["response"] = response
        result = classify(response)
        intent["state"] = "REJECTED" if result == "REJECTION_REPORTED" else "UNKNOWN" if result == "UNKNOWN" else "ACCEPTED_OR_PENDING"
        if response is not None and response.get("order"):
            intent["order_ticket"] = response["order"]
        self._record("broker_response", {"intent_id": intent_id, "classification": result, "response": response,
                                       "request": request, "submission_utc": intent["submission_utc"]})
        if result == "UNKNOWN":
            self._execution(intent, "unknown", 0, fresh_now)
        self.persist()
        return deepcopy(intent)

    def summary(self) -> dict[str, Any]:
        attempted = [v for v in self.intents.values() if v.get("submission_utc")]
        deals = {d["ticket"]: d for v in self.intents.values() for d in v.get("deals", [])}
        unresolved = [v["intent_id"] for v in self.intents.values() if v["state"] not in TERMINAL]
        snapshot = self.last_snapshot
        return {"run_type": self.config.run_type, "submissions": len(attempted),
            "prechecks": sum("check" in v for v in self.intents.values()),
            "entry_submissions": sum(not v["closing"] for v in attempted),
            "close_submissions": sum(v["closing"] for v in attempted), "verified_deals": len(deals),
            "verified_entry_deals": sum(d["entry"] == 0 for d in deals.values()),
            "verified_close_deals": sum(d["entry"] == 1 for d in deals.values()),
            "owned_positions_remaining": len(snapshot["positions"]) if snapshot and self.last_reconciled else None,
            "orders_remaining": len(snapshot["orders"]) if snapshot and self.last_reconciled else None,
            "unresolved_intents": unresolved, "halts": self.halts.copy(),
            "risk_state": self.risk.state.state, "reservations": len(self.risk.state.reservations),
            "verified_flat": bool(self.last_reconciled and snapshot is not None and not snapshot["positions"] and not snapshot["orders"]
                                  and not unresolved and not self.risk.state.reservations),
            "broker_reported_cash": sum(d["profit"] + d["commission"] + d["swap"] + d["fee"] for d in deals.values()),
            "strategy_validity_claim": None, "real_money_readiness": False, "stage19_started": False}
