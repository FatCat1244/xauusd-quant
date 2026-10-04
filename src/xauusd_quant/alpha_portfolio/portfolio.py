"""Streaming multi-frequency intentions, one authoritative Stage12 account."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..execution.config import ExecutionConfig, content_hash
from ..execution.engine import ExecutionEngine, Quote, Recorder
from ..execution.io import parse_utc
from ..execution.policy import utc_time
from .allocation import Allocation
from .intents import Intent
from .registry import Alpha


@dataclass(frozen=True)
class WeightUpdate:
    allocation: Allocation

    @property
    def timestamp_utc(self) -> datetime:
        return self.allocation.effective_utc


def event_key(event: Quote | Intent | WeightUpdate) -> tuple[datetime, int, str]:
    """At ties: weight versions, quote sequence, then signals by stable alpha identity.

    Expiries are applied before each timestamp's quote; simultaneous signal
    changes cannot fill until a strictly later quote. Final same-time target wins.
    """
    if isinstance(event, WeightUpdate):
        return (utc_time(event.timestamp_utc), 0, event.allocation.allocation_id)
    if isinstance(event, Quote):
        return (utc_time(event.timestamp_utc), 1, f"{event.sequence:020d}")
    return (utc_time(event.available_utc), 2, event.alpha_id)


class Portfolio:
    def __init__(
        self,
        config: ExecutionConfig,
        specifications: dict[str, str],
        budget_lots: float,
        record: Recorder,
        contracts: dict[str, tuple[str, int]] | None = None,
    ) -> None:
        if not math.isfinite(budget_lots) or not 0 < budget_lots <= config.max_quantity_lots:
            raise ValueError("bounded research exposure budget required")
        if len(specifications) > 3 or any(not k or not v for k, v in specifications.items()):
            raise ValueError("bounded identified universe required")
        self.config, self.specifications, self.budget_lots = (
            config,
            specifications.copy(),
            budget_lots,
        )
        self.record = record
        self.contracts = contracts or {}
        if self.contracts and set(self.contracts) != set(specifications):
            raise ValueError("complete per-alpha timeframe/horizon contracts required")
        self.intents: dict[str, Intent] = {}
        self.allocation: Allocation | None = None
        self.last_key: tuple[datetime, int, str] | None = None
        self.target_lots = 0.0
        self.contributions: dict[str, float] = {}
        self.owners: dict[str, float] = {}
        self.attribution: dict[str, dict[str, float]] = {alpha: {} for alpha in specifications}
        self.account_totals: dict[str, float] = {}
        self.engine = ExecutionEngine(config, [], self._record)
        self.counts: Counter[str] = Counter()

    @classmethod
    def from_alphas(
        cls,
        config: ExecutionConfig,
        alphas: tuple[Alpha, ...],
        cutoff: datetime,
        budget_lots: float,
        record: Recorder,
    ) -> Portfolio:
        """Scientific entry: never reinterpret a later Stage14 universe as prior known."""
        if any(not alpha.eligible_at(cutoff) for alpha in alphas):
            raise ValueError("universe contains ineligible or future-known alpha")
        if len({a.alpha_id for a in alphas}) != len(alphas):
            raise ValueError("unique alpha identities required")
        record(
            "fold_eligibility",
            {
                "cutoff_utc": utc_time(cutoff),
                "eligible_alphas": [a.resolved() for a in alphas],
                "classification": "declared historical evidence, not prospective approval",
            },
        )
        return cls(
            config,
            {a.alpha_id: a.specification_id for a in alphas},
            budget_lots,
            record,
            {a.alpha_id: (a.timeframe, a.horizon_bars) for a in alphas},
        )

    def _allocate(self, metric: str, value: float, stamp: datetime) -> None:
        if not self.owners or not math.isclose(sum(self.owners.values()), 1):
            raise RuntimeError("cash flow without reconciled entry attribution owners")
        self.account_totals[metric] = self.account_totals.get(metric, 0) + value
        for alpha, share in self.owners.items():
            totals = self.attribution[alpha]
            amount = share * value
            totals[metric] = totals.get(metric, 0) + amount
            self.record(
                "sleeve_cash_attribution",
                {
                    "alpha_id": alpha,
                    "timestamp_utc": stamp,
                    "metric": metric,
                    "amount_account": amount,
                    "share": share,
                    "convention": "owners frozen at entry fill",
                },
            )

    def _record(self, table: str, row: dict[str, Any]) -> None:
        if table == "fills":
            if row["purpose"] == "entry":
                sign = row["direction"]
                supporters = {
                    alpha: abs(value)
                    for alpha, value in self.contributions.items()
                    if value * sign > 0
                }
                total = sum(supporters.values())
                if total <= 0:
                    raise RuntimeError("entry fill has no valid supporting intent")
                self.owners = {alpha: value / total for alpha, value in supporters.items()}
                self.record(
                    "entry_ownership",
                    {
                        "order_id": row["order_id"],
                        "timestamp_utc": row["timestamp_utc"],
                        "owners": self.owners.copy(),
                        "excluded_opposing_intents": [
                            a for a, v in self.contributions.items() if v * sign < 0
                        ],
                    },
                )
            for field in ("commission_account", "spread_cost_account", "slippage_account"):
                self._allocate(field, row[field], row["timestamp_utc"])
            turnover = (
                row["price_usd_per_ounce"]
                * row["quantity_lots"]
                * self.config.contract_ounces_per_lot
                * self.config.account_currency_per_usd
            )
            self._allocate("turnover_account", turnover, row["timestamp_utc"])
        elif table == "cash_flows":
            self._allocate("cash_change_account", row["amount_account"], row["timestamp_utc"])
            if row["kind"] == "financing":
                self._allocate("financing_account", -row["amount_account"], row["timestamp_utc"])
        elif table == "trades":
            for metric in ("net_pnl_account", "gross_price_pnl_account"):
                self._allocate(metric, row[metric], row["exit_utc"])
        elif table == "equity":
            unrealized = row["unrealized_price_pnl_account"]
            self.record(
                "sleeve_equity_attribution",
                {
                    "timestamp_utc": row["timestamp_utc"],
                    "pnl": {
                        alpha: (
                            values.get("cash_change_account", 0)
                            + self.owners.get(alpha, 0) * unrealized
                        )
                        if unrealized is not None
                        else None
                        for alpha, values in self.attribution.items()
                    },
                    "portfolio_equity_account": row["equity_account"],
                    "convention": "cash changes plus entry-owner unrealized; initial capital not sleeve assets",
                },
            )
        self.record(table, row)

    def _combine(self, at: datetime, reason: str) -> None:
        expired = [alpha for alpha, intent in self.intents.items() if intent.valid_until_utc <= at]
        for alpha in expired:
            self.record(
                "intent_expiry",
                {"alpha_id": alpha, "timestamp_utc": self.intents[alpha].valid_until_utc},
            )
            del self.intents[alpha]
            self.counts["expired_intents"] += 1
        weights = self.allocation.weights if self.allocation else {}
        contributions = {
            alpha: weights.get(alpha, 0) * intent.target for alpha, intent in self.intents.items()
        }
        net = sum(contributions.values()) * self.budget_lots
        step = self.config.quantity_increment_lots
        # Float tolerance only for exact representable increment boundaries.
        lots = math.copysign(math.floor(abs(net) / step + 1e-10) * step, net) if net else 0.0
        if abs(lots) < self.config.min_quantity_lots:
            lots = 0.0
        changed = (
            contributions != self.contributions or lots != self.target_lots or reason != "quote"
        )
        self.contributions, self.target_lots = contributions, lots
        if changed:
            p = self.engine.position
            self.record(
                "portfolio_decisions",
                {
                    "timestamp_utc": at,
                    "reason": reason,
                    "allocation_id": self.allocation.allocation_id if self.allocation else None,
                    "sleeve_intents": {a: i.resolved() for a, i in self.intents.items()},
                    "weighted_intents": contributions.copy(),
                    "net_before_rounding_lots": net,
                    "target_lots": lots,
                    "gross_intended_lots": sum(abs(v) for v in contributions.values())
                    * self.budget_lots,
                    "cancelled_intent_lots": (
                        sum(abs(v) for v in contributions.values())
                        - abs(sum(contributions.values()))
                    )
                    * self.budget_lots,
                    "actual_signed_lots": p.direction * p.quantity_lots if p else 0,
                    "pending_order": self.engine.pending.order_id if self.engine.pending else None,
                },
            )
        self.engine.request_target(
            at, lots, self.allocation.allocation_id if self.allocation else "INACTIVE_V001"
        )

    def _expire_until(self, at: datetime) -> None:
        # Run timers at their exact expiry even when no quote arrives.
        while self.intents:
            expiry = min(intent.valid_until_utc for intent in self.intents.values())
            if expiry > at:
                break
            self._combine(expiry, "expiry")

    def consume(self, events: Iterable[Quote | Intent | WeightUpdate]) -> None:
        for event in events:
            key = event_key(event)
            at = key[0]
            if self.last_key is not None and key <= self.last_key:
                raise ValueError("strict deterministic event order required across chunks")
            self._expire_until(at)
            self.last_key = key
            if isinstance(event, WeightUpdate):
                allocation = event.allocation
                if set(allocation.weights) != set(self.specifications):
                    raise ValueError("allocation must match frozen universe")
                self.allocation = allocation
                self.record("allocation_updates", allocation.resolved())
                self._combine(at, "weight_update")
            elif isinstance(event, Intent):
                if event.specification_id != self.specifications.get(event.alpha_id):
                    raise ValueError("intent alpha/specification not in frozen universe")
                if (
                    self.contracts
                    and (event.timeframe, event.horizon_bars) != self.contracts[event.alpha_id]
                ):
                    raise ValueError("intent timeframe/horizon contract mismatch")
                if event.available_utc > at:
                    raise ValueError("intent unavailable at decision")
                self.intents[event.alpha_id] = event
                self.record("exposure_intents", event.resolved())
                self._combine(at, "intent")
            else:
                self._combine(at, "quote")
                self.engine.consume([event])
                # After close: submit replacement, which cannot use this same quote.
                self.engine.request_target(
                    at,
                    self.target_lots,
                    self.allocation.allocation_id if self.allocation else "INACTIVE_V001",
                )
                p = self.engine.position
                self.record(
                    "exposure",
                    {
                        "timestamp_utc": at,
                        "target_lots": self.target_lots,
                        "actual_signed_lots": p.direction * p.quantity_lots if p else 0,
                        "gross_intended_lots": sum(abs(v) for v in self.contributions.values())
                        * self.budget_lots,
                    },
                )

    def finish(self, end: datetime) -> dict[str, Any]:
        end = utc_time(end)
        if self.last_key and end < self.last_key[0]:
            raise ValueError("cutoff precedes consumed event")
        self._expire_until(end)
        result = self.engine.finish(end)
        for metric, total in self.account_totals.items():
            allocated = sum(values.get(metric, 0) for values in self.attribution.values())
            if not math.isclose(total, allocated, abs_tol=1e-7):
                raise RuntimeError("sleeve attribution does not reconcile")
        cash_delta = self.engine.cash - self.config.initial_cash_account
        if not math.isclose(
            cash_delta, self.account_totals.get("cash_change_account", 0), abs_tol=1e-7
        ):
            raise RuntimeError("attributed cash differs from authoritative account")
        return {
            "execution": result,
            "attribution": self.attribution,
            "attribution_totals": self.account_totals,
            "counts": dict(self.counts),
            "result": "NO_ELIGIBLE_ALPHAS"
            if not self.specifications
            else "SYNTHETIC_OR_DECLARED_RESEARCH_ONLY",
        }

    def state(self) -> dict[str, Any]:
        body = {
            "schema": "PORTFOLIO_STATE_V001",
            "configuration_sha256": content_hash(
                {
                    "execution": self.config.resolved(),
                    "specifications": self.specifications,
                    "budget": self.budget_lots,
                    "contracts": self.contracts,
                }
            ),
            "engine": self.engine.target_state(),
            "intents": {a: i.resolved() for a, i in self.intents.items()},
            "allocation": self.allocation.resolved() if self.allocation else None,
            "last_key": [self.last_key[0].isoformat(), *self.last_key[1:]]
            if self.last_key
            else None,
            "target_lots": self.target_lots,
            "contributions": self.contributions,
            "owners": self.owners,
            "attribution": self.attribution,
            "account_totals": self.account_totals,
            "counts": dict(self.counts),
        }
        return json.loads(json.dumps(body, default=lambda t: t.isoformat(), allow_nan=False))

    @classmethod
    def restore(
        cls,
        config: ExecutionConfig,
        specifications: dict[str, str],
        budget_lots: float,
        record: Recorder,
        state: dict[str, Any],
        contracts: dict[str, tuple[str, int]] | None = None,
    ) -> Portfolio:
        portfolio = cls(config, specifications, budget_lots, record, contracts)
        if (
            state.get("schema") != "PORTFOLIO_STATE_V001"
            or state.get("configuration_sha256") != portfolio.state()["configuration_sha256"]
        ):
            raise ValueError("portfolio state configuration mismatch")
        portfolio.engine = ExecutionEngine.restore_target_state(
            config, state["engine"], portfolio._record
        )
        for alpha, body in state["intents"].items():
            portfolio.intents[alpha] = Intent(
                **{
                    **body,
                    **{
                        name: parse_utc(body[name])
                        for name in ("decision_utc", "available_utc", "valid_until_utc")
                    },
                }
            )
        body = state["allocation"]
        if body is not None:
            portfolio.allocation = Allocation(
                **{
                    **body,
                    **{
                        name: parse_utc(body[name])
                        for name in ("effective_utc", "information_as_of_utc")
                    },
                }
            )
        key = state["last_key"]
        portfolio.last_key = (parse_utc(key[0]), key[1], key[2]) if key else None
        for name in ("target_lots", "contributions", "owners", "attribution", "account_totals"):
            setattr(portfolio, name, state[name])
        portfolio.counts = Counter(state["counts"])
        return portfolio
