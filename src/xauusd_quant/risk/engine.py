"""Deterministic risk state, conservative reservations and auditable offline decisions.

Budgets describe conditional loss scenarios, never guaranteed stop-loss ceilings.
Terminal execution reports retain reservations until an account reconciliation.
"""

from __future__ import annotations

import math
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from ..execution.config import content_hash
from ..execution.engine import Recorder
from ..execution.io import _json_safe
from ..execution.policy import utc_time
from .contracts import (
    AccountSnapshot,
    ExecutionUpdate,
    HealthSnapshot,
    MarketSnapshot,
    RiskRequest,
    finite,
    resolved,
)
from .policy import RiskConfiguration

TERMINAL = {"filled", "cancelled", "rejected", "expired"}
STATES = {"UNCONFIGURED", "WARMING_UP", "READY", "HALTED", "REDUCING", "RECONCILIATION_REQUIRED"}
TRANSITIONS = {
    "UNCONFIGURED": {"UNCONFIGURED", "HALTED", "RECONCILIATION_REQUIRED"},
    "WARMING_UP": {"WARMING_UP", "READY", "HALTED", "REDUCING", "RECONCILIATION_REQUIRED"},
    "READY": {"READY", "HALTED", "REDUCING", "RECONCILIATION_REQUIRED"},
    "HALTED": {"HALTED", "REDUCING", "RECONCILIATION_REQUIRED", "READY", "WARMING_UP"},
    "REDUCING": {"REDUCING", "READY", "HALTED", "RECONCILIATION_REQUIRED"},
    "RECONCILIATION_REQUIRED": {"RECONCILIATION_REQUIRED", "HALTED", "REDUCING", "READY", "WARMING_UP"},
}


@dataclass
class Reservation:
    intent_id: str
    signed_change: float
    estimated_loss: float
    margin: float
    created_utc: str
    reducing: bool
    status: str = "approved"
    cumulative_filled: float = 0.0
    last_event_utc: str | None = None
    order_id: str | None = None


@dataclass
class RiskState:
    state: str
    sequence: int = 0
    halts: list[str] = field(default_factory=list)
    reservations: dict[str, Reservation] = field(default_factory=dict)
    decisions: dict[str, dict[str, Any]] = field(default_factory=dict)
    acknowledgements: dict[str, str] = field(default_factory=dict)
    terminal: dict[str, dict[str, Any]] = field(default_factory=dict)
    account: dict[str, Any] | None = None
    last_receipt_utc: str | None = None
    adjusted_equity: float | None = None
    high_water: float | None = None
    daily_start: float | None = None
    daily_key: str | None = None
    daily_loss: float = 0.0
    drawdown: float = 0.0
    warm_snapshots: int = 0
    needs_reconciliation: bool = True
    order_times: list[str] = field(default_factory=list)
    turnover: list[list[Any]] = field(default_factory=list)
    pending_actions: list[dict[str, Any]] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)


class RiskEngine:
    def __init__(self, configuration: RiskConfiguration, record: Recorder) -> None:
        self.configuration, self.record = configuration, record
        self._configuration_identity = configuration.identity
        self.state = RiskState("WARMING_UP" if configuration.configured else "UNCONFIGURED")

    def _audit(self, table: str, body: dict[str, Any]) -> None:
        self.state.sequence += 1
        self.record(table, _json_safe({"sequence": self.state.sequence, **body}))

    def _transition(self, state: str, reason: str, now: datetime) -> None:
        previous = self.state.state
        if state not in TRANSITIONS[previous]:
            raise ValueError("unsupported risk state transition")
        self.state.state = state
        if previous != state:
            self._audit("risk_transitions", {
                "previous": previous, "state": state, "reason": reason, "received_utc": now,
            })

    def _receipt(self, now: datetime) -> None:
        if self.configuration.identity != self._configuration_identity:
            self.state.state = "UNCONFIGURED"
            raise ValueError("risk configuration mutated; restore only with original immutable settings")
        now = utc_time(now)
        if self.state.last_receipt_utc and now < datetime.fromisoformat(self.state.last_receipt_utc):
            self._halt("OUT_OF_ORDER_RECEIPT", now, reconciliation=True)
            raise ValueError("out-of-order receipt; risk remains halted")
        self.state.last_receipt_utc = now.isoformat()

    def _halt(self, reason: str, now: datetime, *, reconciliation: bool = False) -> None:
        if reason not in self.state.halts:
            self.state.halts.append(reason)
            self._audit("risk_halts", {"reason": reason, "received_utc": now})
        if reconciliation:
            self.state.needs_reconciliation = True
        self._transition("RECONCILIATION_REQUIRED" if reconciliation else "HALTED", reason, now)
        p = self.configuration.policy
        if p is not None and p.limit_response in ("cancel", "flatten"):
            for reservation in self.state.reservations.values():
                if not reservation.reducing and reservation.status not in TERMINAL:
                    self._action("REQUEST_CANCEL", reservation.intent_id, now)
        if p is not None and p.limit_response == "flatten" and (
            self.state.account is None or self.state.account["signed_lots"] != 0 or self.state.reservations
        ):
            self._action("REQUEST_REDUCE_OR_FLATTEN", "shared-account", now)

    def _action(self, action: str, identity: str, now: datetime) -> None:
        if any(a["action"] == action and a["identity"] == identity and a["status"] == "unresolved"
               for a in self.state.pending_actions):
            return
        row = {"action": action, "identity": identity, "requested_utc": now.isoformat(),
               "status": "unresolved"}
        self.state.pending_actions.append(row)
        self._audit("risk_actions", row)

    def kill(self, now: datetime, reason: str) -> None:
        self._receipt(now)
        if not reason:
            raise ValueError("explicit kill reason required")
        self._halt(f"KILL:{reason}", now)

    def observe_account(self, snapshot: AccountSnapshot, *, reconcile: bool = False) -> bool:
        """Fresh verified marks include cash costs and quote-side unrealized P&L.

        External flow is a verified cumulative ledger amount. Subtract it from
        equity before computing daily loss and HWM; no restart or midnight rearm.
        Reconciliation only releases terminal reservations whose fills agree with
        the account and whose last execution event precedes the verified snapshot.
        """
        now = utc_time(snapshot.received_utc)
        self._receipt(now)
        p, a = self.configuration.policy, self.configuration.account
        if p is None or a is None:
            return False
        numeric = (snapshot.balance, snapshot.free_margin, snapshot.signed_lots,
                   snapshot.external_flow_total)
        previous = self.state.account
        if previous and snapshot.snapshot_id == previous["snapshot_id"]:
            if resolved(snapshot) != previous:
                self._halt("ACCOUNT_ID_COLLISION", now, reconciliation=True)
                return False
            return True
        age = (now - utc_time(snapshot.event_utc)).total_seconds()
        invalid = (
            not snapshot.snapshot_id or snapshot.account_specification_id != a.specification_id
            or snapshot.verified is not True or type(snapshot.execution_available) is not bool or snapshot.equity is None
            or not all(finite(v) for v in (*numeric, snapshot.equity))
            or snapshot.free_margin < 0 or not 0 <= age <= p.max_account_age_seconds
            or (previous is not None and snapshot.event_utc < datetime.fromisoformat(previous["event_utc"]))
            or (snapshot.open_estimated_loss is not None and (
                not finite(snapshot.open_estimated_loss) or snapshot.open_estimated_loss < 0
            ))
        )
        if invalid:
            self._halt("INVALID_ACCOUNT_SNAPSHOT", now, reconciliation=True)
            return False
        if reconcile:
            # Filled/partial quantities cannot disappear merely on an ACK.
            expected = (previous["signed_lots"] if previous else 0.0) + sum(
                math.copysign(r.cumulative_filled, r.signed_change)
                for r in self.state.reservations.values()
            )
            has_reports = any(r.cumulative_filled for r in self.state.reservations.values())
            if has_reports and not math.isclose(expected, snapshot.signed_lots, abs_tol=1e-8):
                self._halt("POSITION_RECONCILIATION_FAILED", now, reconciliation=True)
                return False
            if any(r.status == "unknown" for r in self.state.reservations.values()):
                self._halt("UNKNOWN_EXECUTION_OUTCOME", now, reconciliation=True)
                return False
            for identity, r in list(self.state.reservations.items()):
                if r.last_event_utc and datetime.fromisoformat(r.last_event_utc) > snapshot.event_utc:
                    self._halt("ACCOUNT_PRECEDES_EXECUTION", now, reconciliation=True)
                    return False
                if r.status in TERMINAL:
                    self.state.terminal[identity] = asdict(r)
                    del self.state.reservations[identity]
                elif r.cumulative_filled:
                    # Partial reports need a terminal status before release. No
                    # inferred netting credit; retain full original reservation.
                    self._halt("PARTIAL_FILL_RECONCILIATION_REQUIRED", now, reconciliation=True)
                    return False
            self.state.needs_reconciliation = False
            for action in self.state.pending_actions:
                if action["action"] == "REQUEST_CANCEL" and action["identity"] in self.state.terminal:
                    action["status"] = "verified_terminal"
                if action["action"] == "REQUEST_REDUCE_OR_FLATTEN" and snapshot.signed_lots == 0 and not self.state.reservations:
                    action["status"] = "verified_flat"
        elif previous and snapshot.signed_lots != previous["signed_lots"]:
            self._halt("UNRECONCILED_POSITION_CHANGE", now, reconciliation=True)
            return False
        if self.state.needs_reconciliation:
            self._transition("RECONCILIATION_REQUIRED", "verified reconciliation required", now)
        assert snapshot.equity is not None
        adjusted = snapshot.equity - snapshot.external_flow_total
        day = snapshot.event_utc.astimezone(ZoneInfo(p.daily_timezone)).date().isoformat()
        if self.state.daily_key != day:
            # Baseline is the last verified adjusted mark, not the first new-day
            # post-loss mark. An absent midnight mark is explicitly approximate.
            self.state.daily_start = self.state.adjusted_equity if self.state.adjusted_equity is not None else adjusted
            self.state.daily_key = day
        self.state.adjusted_equity = adjusted
        self.state.high_water = max(self.state.high_water if self.state.high_water is not None else adjusted, adjusted)
        self.state.daily_loss = max(0.0, (self.state.daily_start or 0) - adjusted)
        self.state.drawdown = max(0.0, self.state.high_water - adjusted)
        self.state.account = resolved(snapshot)
        if previous is None or snapshot.event_utc > datetime.fromisoformat(previous["event_utc"]):
            self.state.warm_snapshots += 1
        if self.state.daily_loss >= p.daily_loss_limit:
            self._halt("DAILY_LOSS", now)
        if self.state.drawdown >= p.drawdown_limit:
            self._halt("EQUITY_DRAWDOWN", now)
        if abs(snapshot.signed_lots) > p.max_net_lots + 1e-9:
            self._halt("ACTUAL_EXPOSURE_LIMIT", now)
        if snapshot.signed_lots and snapshot.free_margin < p.minimum_free_margin:
            self._halt("FREE_MARGIN_LIMIT", now)
        if snapshot.open_estimated_loss is not None and snapshot.open_estimated_loss > p.account_loss_budget:
            self._halt("ACTUAL_LOSS_BUDGET", now)
        local = now.astimezone(ZoneInfo(p.session_timezone))
        if snapshot.signed_lots and (local.weekday() not in p.session_weekdays or not p.session_start_hour <= local.hour < p.session_end_hour):
            self._halt("SESSION_EXPOSURE", now)
        if not snapshot.execution_available:
            self._halt("EXECUTION_UNAVAILABLE", now, reconciliation=True)
        if not self.state.halts and not self.state.needs_reconciliation:
            next_state = "READY" if self.state.warm_snapshots >= p.warmup_snapshots else "WARMING_UP"
            if any(r.reducing for r in self.state.reservations.values()):
                next_state = "REDUCING"
            self._transition(next_state,
                             "verified account/warmup", now)
        self._audit("risk_account_marks", {**resolved(snapshot), "adjusted_equity": adjusted,
                    "daily_loss": self.state.daily_loss, "drawdown": self.state.drawdown,
                    "daily_key": day, "reconciled": reconcile})
        return True

    def execution_update(self, update: ExecutionUpdate) -> None:
        now = utc_time(update.received_utc)
        self._receipt(now)
        digest = content_hash(resolved(update))
        if update.event_id in self.state.acknowledgements:
            if self.state.acknowledgements[update.event_id] != digest:
                self._halt("ACK_ID_COLLISION", now, reconciliation=True)
                raise ValueError("acknowledgement identity collision")
            return
        r = self.state.reservations.get(update.intent_id)
        if r is None:
            self._halt("LATE_OR_UNKNOWN_ORDER_EVENT", now, reconciliation=True)
            raise ValueError("event for unknown or reconciled terminal reservation")
        if (
            update.cumulative_filled_lots < r.cumulative_filled
            or update.cumulative_filled_lots > abs(r.signed_change) + 1e-9
            or (r.last_event_utc and update.event_utc < datetime.fromisoformat(r.last_event_utc))
            or update.event_utc < datetime.fromisoformat(r.created_utc)
            or (r.status in TERMINAL and update.status != r.status)
            or (update.status == "filled" and not math.isclose(update.cumulative_filled_lots, abs(r.signed_change), abs_tol=1e-9))
            or (r.order_id is not None and update.order_id is not None and r.order_id != update.order_id)
        ):
            self._halt("INVALID_EXECUTION_UPDATE", now, reconciliation=True)
            raise ValueError("invalid fill quantity, lifecycle, order identity or event ordering")
        r.cumulative_filled, r.status = update.cumulative_filled_lots, update.status
        r.last_event_utc, r.order_id = update.event_utc.isoformat(), update.order_id or r.order_id
        self.state.acknowledgements[update.event_id] = digest
        self._audit("risk_execution_updates", resolved(update))
        if update.status in ("unknown", "partial"):
            self._halt("UNKNOWN_EXECUTION_OUTCOME" if update.status == "unknown" else "PARTIAL_EXECUTION",
                       now, reconciliation=True)
        # Cancellation request never releases the reservation. Even a terminal
        # ACK requires a verified account reconciliation before release.

    def _market_reasons(self, market: MarketSnapshot | None, now: datetime, *, reducing: bool) -> list[str]:
        p, instrument = self.configuration.policy, self.configuration.instrument
        if p is None or instrument is None or market is None:
            return ["MISSING_QUOTE"]
        reasons: list[str] = []
        age = (now - utc_time(market.event_utc)).total_seconds()
        if market.received_utc > now or market.event_utc > market.received_utc or not 0 <= age <= p.max_quote_age_seconds:
            reasons.append("STALE_OR_FUTURE_QUOTE")
        if not all(finite(v) for v in (market.bid, market.ask)) or not 0 < market.bid <= market.ask:
            reasons.append("INVALID_BID_ASK")
        elif not all(math.isclose(v / instrument.tick_size, round(v / instrument.tick_size), abs_tol=1e-6)
                     for v in (market.bid, market.ask)):
            reasons.append("QUOTE_TICK_ALIGNMENT")
        if (not reducing or not p.emergency_allow_wide_spread) and market.ask - market.bid > p.max_spread_usd_per_ounce:
            reasons.append("SPREAD_LIMIT")
        return reasons

    def _health_reasons(self, request: RiskRequest, health: dict[str, HealthSnapshot], now: datetime) -> list[str]:
        p = self.configuration.policy
        assert p is not None
        reasons: list[str] = []
        for alpha, lots in request.sleeve_lots.items():
            if lots == 0:
                continue
            h, contract = health.get(alpha), p.alpha_contracts.get(alpha)
            if h is None or contract is None:
                reasons.append(f"MISSING_HEALTH:{alpha}")
                continue
            if (h.alpha_id != alpha or h.policy_specification_id != contract["policy"]
                or h.model_id != contract["model"] or h.feature_id != contract["features"]):
                reasons.append(f"SPECIFICATION_MISMATCH:{alpha}")
            age = (now - utc_time(h.event_utc)).total_seconds()
            if (h.received_utc > now or h.event_utc > h.received_utc or not 0 <= age <= p.max_health_age_seconds
                or h.forecast_available_utc > now or h.forecast_expiry_utc <= now
                or h.forecast_available_utc > h.forecast_expiry_utc
                or type(h.horizon_seconds) is not int or not 0 < h.horizon_seconds <= p.max_holding_seconds):
                reasons.append(f"STALE_OR_INVALID_FORECAST_TIME:{alpha}")
            if (h.forecast_units != "log_mid_return" or not finite(h.forecast_value)
                or abs(h.forecast_value) > p.max_forecast_abs_log_return):
                reasons.append(f"INVALID_FORECAST:{alpha}")
            if not all(v is True for v in (h.eligible, h.warm, h.features_ready, h.data_healthy, h.calibration_healthy)):
                reasons.append(f"HEALTH_OR_ELIGIBILITY:{alpha}")
            if (p.require_diagnostic and h.diagnostic_healthy is not True) or h.diagnostic_healthy is False:
                reasons.append(f"DIAGNOSTIC_UNHEALTHY_OR_MISSING:{alpha}")
            for name, cap in (("volatility", p.max_volatility), ("uncertainty", p.max_uncertainty)):
                v = getattr(h, name)
                if not finite(v) or v < 0 or v > cap:
                    reasons.append(f"INVALID_{name.upper()}:{alpha}")
        return reasons

    def evaluate(self, request: RiskRequest, market: MarketSnapshot | None,
                 health: dict[str, HealthSnapshot]) -> dict[str, Any]:
        return self._evaluate(request, market, health, execution_smoke=False)

    def evaluate_smoke(self, request: RiskRequest, market: MarketSnapshot | None) -> dict[str, Any]:
        """Explicit lifecycle probe, with the same monetary/exposure/loss guards.

        This is not a forecast or eligible alpha. Only a dedicated smoke policy
        with no alpha contracts can omit predictive-health inputs. Native demo
        routing additionally requires SMOKE mode and its frozen specification.
        """
        p = self.configuration.policy
        if (p is None or p.expected_portfolio_id != "EXECUTION_SMOKE_V001"
            or p.allowed_allocation_ids != ("SMOKE_V001",) or p.alpha_contracts
            or p.require_diagnostic or p.sizing_method != "horizon_stress"
            or request.portfolio_id != p.expected_portfolio_id
            or request.allocation_id != "SMOKE_V001"
            or set(request.sleeve_lots) != {"EXECUTION_SMOKE"}):
            raise ValueError("dedicated no-model smoke policy required; strategy health is unchanged")
        return self._evaluate(request, market, {}, execution_smoke=True)

    def _evaluate(self, request: RiskRequest, market: MarketSnapshot | None,
                  health: dict[str, HealthSnapshot], *, execution_smoke: bool) -> dict[str, Any]:
        now = utc_time(request.received_utc)
        self._receipt(now)
        old = self.state.decisions.get(request.intent_id)
        if old is not None:
            if (old["request_sha256"] != request.identity
                or old.get("evaluation_kind", "STRATEGY") != ("EXECUTION_SMOKE" if execution_smoke else "STRATEGY")):
                self._halt("INTENT_ID_COLLISION", now, reconciliation=True)
                raise ValueError("intent identity collision")
            return deepcopy(old)
        if len(self.state.decisions) >= 10000:
            self._halt("BOUNDED_STATE_CAPACITY", now)
            raise RuntimeError("bounded decision state exhausted; checkpoint and stop")
        previous = self.state.state
        p, a, i = self.configuration.policy, self.configuration.account, self.configuration.instrument
        account = self.state.account
        current = float(account["signed_lots"]) if account else 0.0
        target = request.target_lots
        # Stage12 resizes by full close then a subsequent newly approved entry.
        reducing = account is not None and current != 0 and (target == 0 or current * target > 0 and abs(target) < abs(current))
        reasons: list[str] = []
        decision, approved, change = "REJECT", current, 0.0
        measurements: dict[str, Any] = {"actual_signed_lots": current,
            "reserved_absolute_lots": sum(abs(r.signed_change) for r in self.state.reservations.values()),
            "gross_intended_lots": sum(abs(v) for v in request.sleeve_lots.values()),
            "daily_loss": self.state.daily_loss, "drawdown": self.state.drawdown}
        if not self.configuration.configured or p is None or a is None or i is None:
            reasons.append("UNCONFIGURED")
        else:
            self.state.order_times = [t for t in self.state.order_times if (now - datetime.fromisoformat(t)).total_seconds() < p.order_window_seconds]
            self.state.turnover = [v for v in self.state.turnover if (now - datetime.fromisoformat(v[0])).total_seconds() < p.turnover_window_seconds]
            if account is None:
                reasons.append("MISSING_ACCOUNT")
            else:
                age = (now - datetime.fromisoformat(account["event_utc"])).total_seconds()
                if not 0 <= age <= p.max_account_age_seconds or not account["verified"]:
                    reasons.append("STALE_OR_UNVERIFIED_ACCOUNT")
                if not account["execution_available"]:
                    reasons.append("EXECUTION_UNAVAILABLE")
            if self.state.needs_reconciliation:
                reasons.append("RECONCILIATION_REQUIRED")
            reasons.extend(self._market_reasons(market, now, reducing=reducing))
            if self.state.reservations:
                # Do not assume opposite orders cancel; no new target until prior
                # order reaches terminal status AND verified reconciliation.
                reservation = next(iter(self.state.reservations.values()))
                prior = self.state.decisions[reservation.intent_id]
                retain = (
                    len(self.state.reservations) == 1 and not reservation.reducing
                    and reservation.status in ("approved", "acknowledged")
                    and request.target_lots == prior["proposed_lots"]
                    and request.sleeve_lots == prior["sleeve_lots"]
                    and request.portfolio_id == prior["portfolio_id"]
                    and request.allocation_id == prior["allocation_id"]
                    and not self.state.halts and not reasons
                    and (execution_smoke or not self._health_reasons(request, health, now))
                )
                if retain:
                    decision, approved = "APPROVE", prior["approved_target_lots"]
                    reasons.append("PENDING_APPROVAL_RETAINED")
                else:
                    reasons.append("PENDING_OR_UNCERTAIN_EXPOSURE")
                    for r in self.state.reservations.values():
                        if not r.reducing:
                            self._action("REQUEST_CANCEL", r.intent_id, now)
                    decision = "REQUEST_REDUCE_OR_FLATTEN" if reservation.reducing else "REQUEST_CANCEL"
            if reducing:
                if not reasons:
                    decision, approved, change = "REQUEST_REDUCE_OR_FLATTEN", 0.0, -current
                    self._transition("REDUCING", "verified close-only", now)
            elif target == current and not reasons:
                if measurements["gross_intended_lots"] > p.max_gross_intended_lots:
                    reasons.append("GROSS_INTENT_LIMIT")
                else:
                    decision, approved = "APPROVE", current
            elif target == 0 and current == 0 and not reasons:
                decision, approved = "APPROVE", 0.0
            else:
                if current != 0 and current * target <= 0:
                    reasons.append("CLOSE_BEFORE_REVERSAL")
                    self._action("REQUEST_REDUCE_OR_FLATTEN", "shared-account", now)
                    decision = "REQUEST_REDUCE_OR_FLATTEN"
                    if set(reasons) == {"CLOSE_BEFORE_REVERSAL"}:
                        approved, change = 0.0, -current
                else:
                    if self.state.state != "READY" or self.state.halts:
                        reasons.append("NOT_READY_OR_HALTED")
                        decision = "HALT_NEW_EXPOSURE"
                    if request.portfolio_id != p.expected_portfolio_id or request.allocation_id not in p.allowed_allocation_ids:
                        reasons.append("PORTFOLIO_ALLOCATION_MISMATCH")
                    if not execution_smoke:
                        reasons.extend(self._health_reasons(request, health, now))
                    local = now.astimezone(ZoneInfo(p.session_timezone))
                    if local.weekday() not in p.session_weekdays or not p.session_start_hour <= local.hour < p.session_end_hour:
                        reasons.append("SESSION_CLOSED")
                    if not p.allow_overnight and (
                        local.hour * 3600 + local.minute * 60 + local.second + p.max_holding_seconds >= p.session_end_hour * 3600
                    ):
                        reasons.append("OVERNIGHT_RESTRICTION")
                    if request.sizing_method != p.sizing_method:
                        reasons.append("SIZING_METHOD_MISMATCH")
                    if request.sizing_method == "stop_distance" and request.stop_price is None:
                        reasons.append("MISSING_STOP")
                    self.state.order_times = [t for t in self.state.order_times if (now - datetime.fromisoformat(t)).total_seconds() < p.order_window_seconds]
                    self.state.turnover = [v for v in self.state.turnover if (now - datetime.fromisoformat(v[0])).total_seconds() < p.turnover_window_seconds]
                    if len(self.state.order_times) >= p.max_orders_per_window:
                        reasons.append("ORDER_RATE_LIMIT")
                    if len(self.state.reservations) >= p.max_pending_orders:
                        reasons.append("PENDING_ORDER_LIMIT")
                    if not reasons and market is not None and account is not None:
                        entry = market.ask if target > 0 else market.bid
                        conversion = i.ounces_per_lot * a.ledger_per_usd
                        holding_days = p.max_holding_seconds / 86400
                        financing = max(0, i.financing_long_ledger_per_lot_day if target > 0 else i.financing_short_ledger_per_lot_day) * holding_days
                        if p.sizing_method == "stop_distance":
                            stop = request.stop_price
                            if stop is None or not math.isfinite(stop) or stop <= 0 or (entry - stop) * target <= 0 or not math.isclose(stop / i.tick_size, round(stop / i.tick_size), abs_tol=1e-6):
                                reasons.append("INVALID_EXECUTABLE_STOP")
                                loss_per_lot = 0.0
                            else:
                                loss_per_lot = abs(entry - stop) * conversion
                        else:
                            assert p.horizon_stress_fraction is not None
                            loss_per_lot = (entry * p.horizon_stress_fraction + market.ask - market.bid) * conversion
                        loss_per_lot += (2 * i.slippage_usd_per_ounce_leg + p.adverse_exit_usd_per_ounce) * conversion + 2 * i.commission_ledger_per_lot_leg + financing
                        open_loss = account["open_estimated_loss"]
                        if current and open_loss is None:
                            reasons.append("MISSING_OPEN_POSITION_RISK")
                        open_loss = float(open_loss or 0)
                        # Free margin after entry includes the immediate liquidation
                        # mark and commission. Use Ask for conservative short margin.
                        entry_cash_loss_per_lot = (market.ask - market.bid + i.slippage_usd_per_ounce_leg) * conversion + i.commission_ledger_per_lot_leg
                        margin_per_lot = market.ask * conversion * i.margin_notional_fraction + entry_cash_loss_per_lot
                        turnover_remaining = p.max_turnover_lots_per_window - sum(v[1] for v in self.state.turnover)
                        gross = measurements["gross_intended_lots"]
                        caps = [abs(target), i.max_lots, p.max_net_lots,
                                p.position_loss_budget / loss_per_lot,
                                max(0, p.account_loss_budget - open_loss) / loss_per_lot,
                                max(0, account["free_margin"] - p.minimum_free_margin) / margin_per_lot,
                                max(0, turnover_remaining)]
                        if gross:
                            caps.append(abs(target) * min(1, p.max_gross_intended_lots / gross))
                        for lots in request.sleeve_lots.values():
                            if lots:
                                caps.append(abs(target) * min(1, p.alpha_loss_budget / (abs(lots) * loss_per_lot)))
                        quantity = math.floor(max(0, min(caps)) / i.lot_step + 1e-10) * i.lot_step
                        if quantity < i.min_lots:
                            reasons.append("BELOW_MINIMUM_WITHIN_BUDGET")
                        if current and quantity > abs(current):
                            # Stage12 cannot add to a position; close first and
                            # separately approve any replacement from a flat mark.
                            reasons.append("CLOSE_BEFORE_RESIZE")
                            decision, approved, change = "REQUEST_REDUCE_OR_FLATTEN", 0.0, -current
                        elif not reasons:
                            approved = math.copysign(quantity, target)
                            change = approved if current == 0 else 0.0
                            if current and not math.isclose(approved, current, abs_tol=1e-9):
                                decision, approved, change = "REQUEST_REDUCE_OR_FLATTEN", 0.0, -current
                            else:
                                decision = "APPROVE" if math.isclose(approved, target, abs_tol=1e-9) else "RESIZE"
                        measurements.update({"loss_per_lot": loss_per_lot, "estimated_loss": quantity * loss_per_lot,
                            "margin_required": quantity * margin_per_lot, "quantity_rounded_lots": quantity,
                            "daily_loss_limit": p.daily_loss_limit, "drawdown_limit": p.drawdown_limit,
                            "net_limit_lots": p.max_net_lots, "gross_limit_lots": p.max_gross_intended_lots,
                            "post_round_limits_valid": quantity <= min(caps) + 1e-9})
                        if quantity > min(caps) + 1e-9:
                            reasons.append("POST_ROUND_LIMIT_FAILURE")
                            decision, approved, change = "REJECT", current, 0.0
                        if decision == "RESIZE":
                            reasons.append("SIZING_EXPOSURE_MARGIN_OR_TURNOVER_CAP")
            if change:
                if len(self.state.order_times) >= p.max_orders_per_window:
                    reasons.append("ORDER_RATE_LIMIT")
                    decision, approved, change = "REJECT", current, 0.0
                elif sum(v[1] for v in self.state.turnover) + abs(change) > p.max_turnover_lots_per_window + 1e-9:
                    reasons.append("TURNOVER_LIMIT")
                    decision, approved, change = "REJECT", current, 0.0
                if not change and reducing:
                    self._action("REQUEST_REDUCE_OR_FLATTEN", "shared-account", now)
            if change:
                r = Reservation(request.intent_id, change,
                    0 if change * current < 0 else measurements.get("estimated_loss", 0),
                    0 if change * current < 0 else measurements.get("margin_required", 0),
                    now.isoformat(), change * current < 0)
                self.state.reservations[request.intent_id] = r
                self.state.order_times.append(now.isoformat())
                self.state.turnover.append([now.isoformat(), abs(change)])
        row = {"intent_id": request.intent_id, "request_sha256": request.identity,
            "evaluation_kind": "EXECUTION_SMOKE" if execution_smoke else "STRATEGY",
            "alpha_ids": sorted(request.sleeve_lots), "portfolio_id": request.portfolio_id,
            "allocation_id": request.allocation_id, "decision_utc": request.decision_utc,
            "received_utc": now, "policy_id": p.policy_id if p else None,
            "configuration_sha256": self.configuration.identity,
            "input_snapshot_sha256": content_hash({"account": account,
                "market": resolved(market) if market else None,
                "health": {k: resolved(v) for k, v in health.items()}}),
            "account_snapshot_id": account["snapshot_id"] if account else None,
            "proposed_lots": target, "decision": decision, "approved_target_lots": approved,
            "sleeve_lots": request.sleeve_lots.copy(),
            "sizing_method": request.sizing_method, "stop_price": request.stop_price,
            "approved_change_lots": change, "rules": list(dict.fromkeys(reasons)),
            "measurements": measurements, "previous_state": previous, "state": self.state.state,
            "pending_actions": [a.copy() for a in self.state.pending_actions if a["status"] == "unresolved"]}
        row = _json_safe(row)
        self.state.decisions[request.intent_id] = row
        self.state.counts[decision] = self.state.counts.get(decision, 0) + 1
        self._audit("risk_decisions", row)
        return deepcopy(row)

    def rearm(self, now: datetime, *, operator: str, reason: str) -> None:
        """Explicit offline operator action; no reset of HWM, loss or reservations."""
        self._receipt(now)
        p, account = self.configuration.policy, self.state.account
        if (not operator or not reason or p is None or account is None
            or self.state.needs_reconciliation or self.state.reservations
            or self.state.daily_loss >= p.daily_loss_limit or self.state.drawdown >= p.drawdown_limit
            or not account["execution_available"] or not account["verified"]
            or not 0 <= (now - datetime.fromisoformat(account["event_utc"])).total_seconds() <= p.max_account_age_seconds):
            raise ValueError("rearm requires explicit operator/reason, fresh reconciliation and recovered limits")
        prior = self.state.halts.copy()
        self.state.halts.clear()
        self._transition("READY" if self.state.warm_snapshots >= p.warmup_snapshots else "WARMING_UP", "explicit rearm", now)
        self._audit("risk_rearms", {"operator": operator, "reason": reason, "prior_halts": prior, "received_utc": now})

    def checkpoint(self) -> dict[str, Any]:
        body = _json_safe(asdict(self.state))
        return {"schema": "RISK_STATE_V001", "configuration_sha256": self.configuration.identity,
                "body": body, "content_sha256": content_hash(body)}

    @classmethod
    def restore(cls, configuration: RiskConfiguration, record: Recorder,
                checkpoint: dict[str, Any]) -> RiskEngine:
        engine = cls(configuration, record)
        body = checkpoint.get("body")
        if (checkpoint.get("schema") != "RISK_STATE_V001"
            or checkpoint.get("configuration_sha256") != configuration.identity
            or not isinstance(body, dict) or checkpoint.get("content_sha256") != content_hash(body)
            or set(body) != set(RiskState.__dataclass_fields__)):
            raise ValueError("invalid or incompatible risk checkpoint; do not initialize READY")
        # Restored runtime dictionaries must not mutate the frozen input checkpoint.
        body = deepcopy(body)
        restored = RiskState(**body)
        if (restored.state not in STATES or restored.sequence < 0 or restored.warm_snapshots < 0
            or any(not math.isfinite(v) or v < 0 for v in (restored.daily_loss, restored.drawdown))
            or any(v is not None and not math.isfinite(v) for v in (restored.high_water, restored.adjusted_equity, restored.daily_start))):
            raise ValueError("invalid persisted risk measurements")
        reservations: dict[str, Reservation] = {}
        for k, v in body["reservations"].items():
            r = Reservation(**v)
            if (k != r.intent_id or not math.isfinite(r.signed_change) or r.signed_change == 0
                or not 0 <= r.cumulative_filled <= abs(r.signed_change)
                or not all(math.isfinite(x) and x >= 0 for x in (r.margin, r.estimated_loss))
                or r.status not in TERMINAL | {"approved", "acknowledged", "partial", "cancel_requested", "unknown"}):
                raise ValueError("invalid persisted exposure reservation")
            utc_time(datetime.fromisoformat(r.created_utc))
            reservations[k] = r
        restored.reservations = reservations
        if restored.high_water is not None and restored.adjusted_equity is not None and (
            restored.high_water < restored.adjusted_equity
            or not math.isclose(restored.drawdown, max(0, restored.high_water - restored.adjusted_equity), abs_tol=1e-8)
        ):
            raise ValueError("persisted drawdown does not reconcile with high-water mark")
        for timestamp in (restored.last_receipt_utc, *(v[0] for v in restored.turnover), *restored.order_times):
            if timestamp is not None:
                utc_time(datetime.fromisoformat(timestamp))
        if any(k not in restored.decisions or not math.isclose(
            v.signed_change, restored.decisions[k]["approved_change_lots"], abs_tol=1e-9
        ) for k, v in restored.reservations.items()):
            raise ValueError("reservation does not match persisted approval")
        if not all(isinstance(v, str) and v for v in restored.halts):
            raise ValueError("invalid persisted halt reasons")
        # Persistence never establishes current account truth; reconcile first,
        # then explicitly rearm any preexisting severe halt.
        restored.needs_reconciliation = True
        restored.state = "RECONCILIATION_REQUIRED" if configuration.configured else "UNCONFIGURED"
        engine.state = restored
        return engine

    def summary(self) -> dict[str, Any]:
        p = self.configuration.policy
        actual = abs(self.state.account["signed_lots"]) if self.state.account else None
        return {"state": self.state.state, "counts": dict(Counter(self.state.counts)),
                "halts": self.state.halts.copy(), "reserved_lots": sum(abs(r.signed_change) for r in self.state.reservations.values()),
                "daily_loss": self.state.daily_loss, "drawdown": self.state.drawdown,
                "daily_loss_utilization": self.state.daily_loss / p.daily_loss_limit if p else None,
                "drawdown_utilization": self.state.drawdown / p.drawdown_limit if p else None,
                "net_exposure_utilization": actual / p.max_net_lots if p and actual is not None else None,
                "pending_orders": len(self.state.reservations),
                "estimated_loss_reserved": sum(r.estimated_loss for r in self.state.reservations.values()),
                "margin_reserved": sum(r.margin for r in self.state.reservations.values()),
                "audit_sequence": self.state.sequence,
                "unresolved_actions": [a.copy() for a in self.state.pending_actions if a["status"] == "unresolved"],
                "actual_account": self.state.account, "configuration": resolved(self.configuration)}
