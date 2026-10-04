"""Stage15 intentions must cross risk before the single Stage12 account can act.

The capability is process-local protection against accidental application bypass,
not a security boundary against malicious Python code modifying private attributes.
Offline cancellations are immediate verified simulator outcomes, never broker ACKs.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from ..alpha_portfolio.portfolio import Portfolio
from ..alpha_portfolio.registry import Alpha
from ..execution.config import ExecutionConfig, content_hash
from ..execution.engine import ExecutionEngine, Quote, Recorder
from ..execution.io import parse_utc
from .contracts import AccountSnapshot, ExecutionUpdate, HealthSnapshot, MarketSnapshot, RiskRequest
from .engine import RiskEngine
from .policy import RiskConfiguration


class RiskPortfolio(Portfolio):
    def __init__(self, config: ExecutionConfig, specifications: dict[str, str], budget_lots: float,
                 record: Recorder, risk_configuration: RiskConfiguration,
                 contracts: dict[str, tuple[str, int]] | None = None) -> None:
        super().__init__(config, specifications, budget_lots, record, contracts)
        self.risk = RiskEngine(risk_configuration, record)
        a, i, p = risk_configuration.account, risk_configuration.instrument, risk_configuration.policy
        if a is not None and i is not None:
            i.match_execution(a, config)
        if p is not None and set(p.alpha_contracts) != set(specifications):
            raise ValueError("risk alpha universe must exactly match portfolio")
        if p is not None and any(p.alpha_contracts[k]["policy"] != v for k, v in specifications.items()):
            raise ValueError("risk/portfolio policy specifications must match")
        self._authority = object()
        self._active_decision: dict[str, Any] | None = None
        self.health: dict[str, HealthSnapshot] = {}
        self.market: MarketSnapshot | None = None
        self.request_number = 0
        self.snapshot_number = 0
        self.order_intents: dict[int, str] = {}
        self.risk_checkpoint_required = False
        self.engine = ExecutionEngine(config, [], self._record,
                                      risk_authority=self._authority, risk_boundary=self._boundary)

    @classmethod
    def from_alphas(cls, config: ExecutionConfig, alphas: tuple[Alpha, ...], cutoff: datetime,
                    budget_lots: float, record: Recorder,
                    *, risk_configuration: RiskConfiguration | None = None) -> RiskPortfolio:
        if risk_configuration is None:
            raise ValueError("governed portfolio requires explicit risk configuration")
        if any(not a.eligible_at(cutoff) for a in alphas) or len({a.alpha_id for a in alphas}) != len(alphas):
            raise ValueError("ineligible, future-known or duplicate alpha universe")
        record("fold_eligibility", {"cutoff_utc": cutoff, "eligible_alphas": [a.resolved() for a in alphas],
                                    "classification": "declared historical eligibility, not deployment approval"})
        return cls(config, {a.alpha_id: a.specification_id for a in alphas}, budget_lots,
                   record, risk_configuration, {a.alpha_id: (a.timeframe, a.horizon_bars) for a in alphas})

    def health_update(self, snapshot: HealthSnapshot) -> None:
        old = self.health.get(snapshot.alpha_id)
        if old and (snapshot.event_utc < old.event_utc or snapshot.received_utc < old.received_utc):
            self.risk.kill(snapshot.received_utc, "OUT_OF_ORDER_HEALTH")
            raise ValueError("out-of-order health event")
        self.health[snapshot.alpha_id] = snapshot
        self.record("risk_health_inputs", {
            "alpha_id": snapshot.alpha_id, "event_utc": snapshot.event_utc,
            "received_utc": snapshot.received_utc,
            "source": "explicit adapter health, not inferred from registry",
        })

    def _record(self, table: str, row: dict[str, Any]) -> None:
        if table == "orders" and hasattr(self, "risk"):
            order = int(row["order_id"])
            if row["status"] == "submitted":
                if self._active_decision is None:
                    raise PermissionError("unapproved order submission")
                self.order_intents[order] = self._active_decision["intent_id"]
            identity = self.order_intents.get(order)
            if identity is None:
                raise RuntimeError("execution order lacks risk identity")
            status = {"submitted": "acknowledged", "unfilled": "expired"}.get(row["status"], row["status"])
            quantity = float(row["quantity_lots"] or self.config.quantity_lots) if status == "filled" else 0.0
            self.risk.execution_update(ExecutionUpdate(
                f"ORDER_{order}_{status}", identity, row["event_utc"], row["event_utc"],
                status, quantity, str(order),
            ))
        super()._record(table, row)

    def _boundary(self, phase: str, at: datetime, signed: float, purpose: str,
                  quote: Quote | None) -> bool:
        decision = self._active_decision
        if phase == "target":
            return decision is not None and math.isclose(signed, decision["approved_target_lots"], abs_tol=1e-9) and decision["received_utc"] == at.isoformat()
        if phase == "submit":
            if decision is None or not decision["approved_change_lots"]:
                return False
            r = self.risk.state.reservations.get(decision["intent_id"])
            if r is None or r.status != "approved":
                return False
            expected = -r.signed_change if purpose == "exit" else r.signed_change
            return math.isclose(signed, expected, abs_tol=1e-9) and (purpose == "exit") == r.reducing
        order = self.engine.pending
        identity = self.order_intents.get(order.order_id) if order else None
        r = self.risk.state.reservations.get(identity or "")
        if r is None or quote is None or r.status not in ("acknowledged", "cancel_requested"):
            return False
        self._sync_account(at, quote)
        reasons = self.risk._market_reasons(self.market, at, reducing=r.reducing)
        p, i, a = self.risk.configuration.policy, self.risk.configuration.instrument, self.risk.configuration.account
        if p is None or i is None or a is None or self.risk.state.needs_reconciliation:
            return False
        if r.reducing:
            position = self.engine.position
            return not reasons and position is not None and math.isclose(signed, position.direction * position.quantity_lots, abs_tol=1e-9)
        if self.risk.state.halts or self.risk.state.state != "READY":
            reasons.append("FILL_WHILE_HALTED")
        original = self.risk.state.decisions[r.intent_id]
        request = RiskRequest(
            "FILL_CHECK", p.expected_portfolio_id, original["allocation_id"], at, at,
            signed, {alpha: signed / len(original["alpha_ids"]) for alpha in original["alpha_ids"]},
            p.sizing_method,
        )
        reasons.extend(self.risk._health_reasons(request, self.health, at))
        if self.engine.position is not None:
            reasons.append("POSITION_EXISTS_AT_ENTRY_FILL")
        # A pending market order is rechecked at its actual executable quote.
        # Hypothetical fixed loss/margin reservation may not cover a price jump.
        price = quote.ask if signed > 0 else quote.bid
        account = self.risk.state.account
        required_margin = abs(signed) * (quote.ask * i.ounces_per_lot * a.ledger_per_usd * i.margin_notional_fraction + (quote.ask - quote.bid + i.slippage_usd_per_ounce_leg) * i.ounces_per_lot * a.ledger_per_usd + i.commission_ledger_per_lot_leg)
        if account is None or required_margin > account["free_margin"] - p.minimum_free_margin + 1e-9:
            reasons.append("FILL_MARGIN_LIMIT")
        # Sizing caps are frozen; allow only prices at or better than the risk
        # snapshot's entry notional. Adverse jumps require a new approval.
        if required_margin > r.margin + 1e-9:
            reasons.append("FILL_EXCEEDS_RESERVED_NOTIONAL")
        conversion = i.ounces_per_lot * a.ledger_per_usd
        if p.sizing_method == "stop_distance":
            stop = original["stop_price"]
            loss_per_lot = abs(price - stop) * conversion if stop is not None else float("inf")
        else:
            assert p.horizon_stress_fraction is not None
            loss_per_lot = (price * p.horizon_stress_fraction + quote.ask - quote.bid) * conversion
        loss_per_lot += (2 * i.slippage_usd_per_ounce_leg + p.adverse_exit_usd_per_ounce) * conversion + 2 * i.commission_ledger_per_lot_leg + max(0, i.financing_long_ledger_per_lot_day if signed > 0 else i.financing_short_ledger_per_lot_day) * p.max_holding_seconds / 86400
        if abs(signed) * loss_per_lot > r.estimated_loss + 1e-8:
            reasons.append("FILL_EXCEEDS_RESERVED_LOSS")
        if reasons:
            self.record("risk_fill_rejections", {"intent_id": r.intent_id, "event_utc": at, "rules": reasons})
        return not reasons

    def _sync_account(self, at: datetime, quote: Quote | None = None) -> None:
        a, i, p = self.risk.configuration.account, self.risk.configuration.instrument, self.risk.configuration.policy
        if a is None or i is None or p is None:
            return
        q = quote or self.engine.last_quote
        position = self.engine.position
        current = position.direction * position.quantity_lots if position else 0.0
        equity: float | None = self.engine.cash
        fresh = q is not None and 0 <= (at - (q.quoted_at_utc or q.timestamp_utc)).total_seconds() <= p.max_quote_age_seconds
        if position:
            equity = self.engine.cash + position.direction * ((q.bid if position.direction > 0 else q.ask) - position.entry_price) * position.quantity_lots * i.ounces_per_lot * a.ledger_per_usd if fresh and q is not None and math.isfinite(q.bid) and math.isfinite(q.ask) and 0 < q.bid <= q.ask else None
        margin = abs(current) * (q.ask if q is not None and math.isfinite(q.ask) and q.ask > 0 else 0) * i.ounces_per_lot * a.ledger_per_usd * i.margin_notional_fraction
        self.snapshot_number += 1
        # Position sizing risk is a conservative declared horizon stress; not a
        # broker margin balance and not a guaranteed maximum loss.
        open_loss = None
        if current and p.sizing_method == "horizon_stress" and q is not None and p.horizon_stress_fraction is not None:
            open_loss = abs(current) * ((q.ask * p.horizon_stress_fraction + q.ask - q.bid + 2 * i.slippage_usd_per_ounce_leg + p.adverse_exit_usd_per_ounce) * i.ounces_per_lot * a.ledger_per_usd + 2 * i.commission_ledger_per_lot_leg + max(0, i.financing_long_ledger_per_lot_day, i.financing_short_ledger_per_lot_day) * p.max_holding_seconds / 86400)
        snapshot = AccountSnapshot(
            f"SIM_ACCOUNT_{self.snapshot_number}", at, at, a.specification_id, self.engine.cash,
            equity, max(0, (equity or 0) - margin), current, True, True,
            open_estimated_loss=open_loss,
        )
        self.risk.observe_account(snapshot, reconcile=True)

    def _route_target(self, at: datetime, _lots: float, identity: str) -> None:
        self.engine.checkpoint(at)
        self._sync_account(at)
        self.request_number += 1
        # The portfolio's unrounded intent sum is authoritative for gross/net
        # intent measurements. Risk owns all final rounding and approved size.
        sleeves = {a: v * self.budget_lots for a, v in self.contributions.items()}
        proposal = sum(sleeves.values())
        request = RiskRequest(f"PORTFOLIO_RISK_{self.request_number}",
            self.risk.configuration.policy.expected_portfolio_id if self.risk.configuration.policy else "UNCONFIGURED_PORTFOLIO",
            identity, at, at, proposal, sleeves,
            self.risk.configuration.policy.sizing_method if self.risk.configuration.policy else "unconfigured")
        decision = self.risk.evaluate(request, self.market, self.health)
        if decision["approved_change_lots"]:
            self._active_decision = decision
            try:
                self.engine.request_target(at, decision["approved_target_lots"], identity,
                                           risk_authority=self._authority)
            finally:
                self._active_decision = None
        elif decision["decision"] == "REQUEST_CANCEL":
            order = self.engine.pending
            if order is not None and order.purpose == "entry":
                # Explicit synchronous simulator terminal confirmation. Stage17
                # must supply asynchronous ACK/reconciliation; no assumed success.
                self.engine._terminal_order("cancelled", "risk_requested_offline_cancel", at)
                self._sync_account(at)
                # One bounded retry only after synchronous simulator terminal
                # confirmation AND account reconciliation. No async assumption.
                if not self.risk.state.reservations:
                    self._route_target(at, _lots, identity)
                    return
        self._sync_account(at)

    def _consume_quote(self, event: Quote) -> None:
        self.market = MarketSnapshot(event.quoted_at_utc or event.timestamp_utc,
                                     event.timestamp_utc, event.bid, event.ask)
        self.engine.consume([event])
        self._sync_account(event.timestamp_utc, event)
        # Configured emergency actions are requested after limits trigger. They
        # still require a verified position and a valid fresh executable quote.
        p = self.risk.configuration.policy
        if self.risk.state.halts and p is not None:
            if p.limit_response in ("cancel", "flatten") and self.engine.pending and self.engine.pending.purpose == "entry":
                self.engine._terminal_order("cancelled", "offline_halt_cancel_confirmed", event.timestamp_utc)
                self._sync_account(event.timestamp_utc, event)
            if p.limit_response == "flatten" and self.engine.position is not None:
                self._emergency_close(event.timestamp_utc)

    def _emergency_close(self, at: datetime) -> None:
        self.request_number += 1
        p = self.risk.configuration.policy
        assert p is not None
        request = RiskRequest(f"EMERGENCY_{self.request_number}", p.expected_portfolio_id,
                              p.allowed_allocation_ids[0], at, at, 0, {}, p.sizing_method)
        decision = self.risk.evaluate(request, self.market, self.health)
        if decision["approved_change_lots"]:
            self._active_decision = decision
            try:
                self.engine.request_target(at, 0, p.allowed_allocation_ids[0], risk_authority=self._authority)
            finally:
                self._active_decision = None

    def risk_state(self) -> dict[str, Any]:
        body = {"portfolio": self.state(), "risk": self.risk.checkpoint(),
                "request_number": self.request_number, "snapshot_number": self.snapshot_number,
                "order_intents": self.order_intents,
                "health": {k: as_health(v) for k, v in self.health.items()},
                "market": as_market(self.market)}
        return {"schema": "GOVERNED_PORTFOLIO_V001", "body": body,
                "content_sha256": content_hash(body)}

    def finish(self, end: datetime) -> dict[str, Any]:
        summary = super().finish(end)
        self._sync_account(end)
        summary["risk"] = self.risk.summary()
        return summary

    @classmethod
    def restore_risk(cls, config: ExecutionConfig, specifications: dict[str, str], budget_lots: float,
                     record: Recorder, risk_configuration: RiskConfiguration, checkpoint: dict[str, Any],
                     contracts: dict[str, tuple[str, int]] | None = None) -> RiskPortfolio:
        if checkpoint.get("schema") != "GOVERNED_PORTFOLIO_V001" or content_hash(checkpoint["body"]) != checkpoint.get("content_sha256"):
            raise ValueError("invalid governed portfolio checkpoint")
        body = checkpoint["body"]
        result = cls(config, specifications, budget_lots, record, risk_configuration, contracts)
        result.risk = RiskEngine.restore(risk_configuration, record, body["risk"])
        state = body["portfolio"]
        if state["configuration_sha256"] != result.state()["configuration_sha256"]:
            raise ValueError("portfolio configuration changed at restart")
        result.engine = ExecutionEngine.restore_target_state(config, state["engine"], result._record,
                             risk_authority=result._authority, risk_boundary=result._boundary)
        # Reuse Stage15 state parsing while preserving the mandatory boundary.
        parsed = Portfolio.restore(config, specifications, budget_lots, record, state, contracts,
                                   risk_authority=result._authority, risk_boundary=result._boundary)
        for name in ("intents", "allocation", "last_key", "target_lots", "contributions", "owners", "attribution", "account_totals", "counts"):
            setattr(result, name, getattr(parsed, name))
        result.request_number, result.snapshot_number = body["request_number"], body["snapshot_number"]
        if type(result.request_number) is not int or type(result.snapshot_number) is not int or min(result.request_number, result.snapshot_number) < 0:
            raise ValueError("invalid persisted adapter counters")
        result.order_intents = {int(k): v for k, v in body["order_intents"].items()}
        for k, v in body["health"].items():
            parsed_health: dict[str, Any] = {name: parse_utc(value) if name.endswith("_utc") else value for name, value in v.items()}
            result.health[k] = HealthSnapshot(**parsed_health)
        if body["market"]:
            parsed_market: dict[str, Any] = {name: parse_utc(value) if name.endswith("_utc") else value for name, value in body["market"].items()}
            result.market = MarketSnapshot(**parsed_market)
        return result


def as_health(snapshot: HealthSnapshot) -> dict[str, Any]:
    from .contracts import resolved

    return resolved(snapshot)


def as_market(snapshot: MarketSnapshot | None) -> dict[str, Any] | None:
    from .contracts import resolved

    return resolved(snapshot) if snapshot else None
