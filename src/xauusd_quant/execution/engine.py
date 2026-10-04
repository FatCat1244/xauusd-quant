"""Streaming single-position CFD accounting using causal quote-side market fills.

At equal timestamps quotes precede bar-close events in the input stream. Orders
require a strictly subsequent timestamp, so no quote at decision/arrival time can
fill. Original sequence resolves quote ties. Timers survive chunks and partitions.
Only one net position is allowed, without pyramiding or partial fills. The Stage15
target API closes fully before any resize/reversal; fills/accounting stay here.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

from .config import ExecutionConfig, content_hash
from .policy import Forecast, decision, utc_time

Recorder = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class Quote:
    timestamp_utc: datetime
    timestamp_local: datetime
    bid: float
    ask: float
    sequence: int
    quoted_at_utc: datetime | None = None

    @property
    def midpoint(self) -> float:
        return (self.bid + self.ask) / 2


@dataclass(frozen=True)
class BarClose:
    timestamp_utc: datetime
    bar_index: int


@dataclass
class Order:
    order_id: int
    forecast_id: str
    purpose: str
    direction: int
    created_utc: datetime
    arrival_utc: datetime
    expires_utc: datetime
    close_bar_index: int
    immediate_midpoint: float | None
    quantity_lots: float | None = None


@dataclass
class Position:
    position_id: int
    forecast_id: str
    direction: int
    quantity_lots: float
    entry_utc: datetime
    entry_local: datetime
    entry_price: float
    entry_midpoint: float
    entry_spread_cost: float
    entry_commission: float
    close_bar_index: int
    immediate_entry_midpoint: float | None
    financing: float = 0.0
    exit_requested: bool = False


class MemoryRecorder:
    """Small-fixture recorder only; production uses the incremental disk recorder."""

    def __init__(self) -> None:
        self.tables: dict[str, list[dict[str, Any]]] = {}

    def __call__(self, table: str, row: dict[str, Any]) -> None:
        self.tables.setdefault(table, []).append(row.copy())


class ExecutionEngine:
    def __init__(
        self, config: ExecutionConfig, forecasts: Iterable[Forecast], record: Recorder
    ) -> None:
        self.config = config
        self.record = record
        self.forecasts: Iterator[Forecast] = iter(forecasts)
        self.next_forecast = next(self.forecasts, None)
        self.last_forecast_time: datetime | None = None
        self.pending: Order | None = None
        self.position: Position | None = None
        self.last_quote: Quote | None = None
        self.previous_quote: Quote | None = None
        self.clock: datetime | None = None
        self.first_clock: datetime | None = None
        self.last_key: tuple[datetime, int, int] | None = None
        self.completed_bar_index = -1
        self.cash = config.initial_cash_account
        self.cash_correction = 0.0
        self.realized = 0.0
        self.counts: Counter[str] = Counter()
        self.total_commission = self.total_financing = self.total_gross = 0.0
        self.total_spread = self.total_slippage = self.midpoint_gross = 0.0
        self.turnover = self.lot_seconds = self.position_seconds = 0.0
        self.peak_equity = self.cash
        self.max_drawdown = 0.0
        self.last_equity_record: datetime | None = None
        self.order_number = self.position_number = 0

    @property
    def price_multiplier(self) -> float:
        c = self.config
        return c.quantity_lots * c.contract_ounces_per_lot * c.account_currency_per_usd

    def _cash_flow(self, stamp: datetime, amount: float, kind: str) -> None:
        # Compensated summation preserves small funding charges against large cash.
        adjusted = amount - self.cash_correction
        updated = self.cash + adjusted
        self.cash_correction = (updated - self.cash) - adjusted
        self.cash = updated
        self.record(
            "cash_flows",
            {
                "timestamp_utc": stamp,
                "kind": kind,
                "amount_account": amount,
                "cash_account": self.cash,
                "position_id": self.position_number,
                "currency": self.config.account_currency,
            },
        )

    def _accrue(self, stamp: datetime) -> None:
        if self.clock is not None and stamp < self.clock:
            raise ValueError("events must be chronological")
        p = self.position
        if self.clock is not None and p is not None:
            seconds = (stamp - self.clock).total_seconds()
            rate = (
                self.config.financing_long_account_per_lot_per_day
                if p.direction == 1
                else self.config.financing_short_account_per_lot_per_day
            )
            charge = rate * p.quantity_lots * seconds / 86400
            self.lot_seconds += seconds * p.quantity_lots
            self.position_seconds += seconds
            if charge:
                p.financing += charge
                self.total_financing += charge
                self._cash_flow(stamp, -charge, "financing")
        self.clock = stamp
        if self.first_clock is None:
            self.first_clock = stamp

    def _fresh(self, quote: Quote | None, stamp: datetime) -> bool:
        if quote is None:
            return False
        age = (stamp - utc_time(quote.quoted_at_utc or quote.timestamp_utc)).total_seconds()
        return 0 <= age * 1000 <= self.config.max_quote_age_ms

    def _terminal_order(self, status: str, reason: str, stamp: datetime) -> None:
        if self.pending is not None:
            self.record(
                "orders",
                {**asdict(self.pending), "status": status, "reason": reason, "event_utc": stamp},
            )
            self.counts[status] += 1
            self.pending = None

    def _submit(
        self,
        stamp: datetime,
        forecast_id: str,
        purpose: str,
        direction: int,
        close_bar_index: int,
        quantity_lots: float | None = None,
    ) -> None:
        self.order_number += 1
        snapshot = (
            self.last_quote.midpoint
            if self.last_quote is not None and self._fresh(self.last_quote, stamp)
            else None
        )
        arrival = stamp + timedelta(milliseconds=self.config.latency_ms)
        order = Order(
            self.order_number,
            forecast_id,
            purpose,
            direction,
            stamp,
            arrival,
            arrival + timedelta(milliseconds=self.config.order_ttl_ms),
            close_bar_index,
            snapshot,
            quantity_lots,
        )
        self.pending = order
        self.counts["submitted"] += 1
        self.record("orders", {**asdict(order), "status": "submitted", "event_utc": stamp})

    def request_target(self, at_utc: datetime, target_lots: float, identity: str) -> None:
        """Reconcile one shared target, using actual state and the same fill machinery.

        Zero cancels pending entry and requests liquidation. Resize/reversal closes
        the entire position, then a later call may reopen. Existing pending exits
        are retained. The caller owns target persistence and retry policy; no fill
        or fee is synthesized here. This API cannot be mixed with forecast entries.
        """
        at = utc_time(at_utc)
        c = self.config
        quantity = abs(target_lots)
        if not math.isfinite(target_lots) or not identity:
            raise ValueError("finite target and specification identity required")
        if quantity and (
            not c.min_quantity_lots <= quantity <= c.max_quantity_lots
            or not math.isclose(
                quantity / c.quantity_increment_lots,
                round(quantity / c.quantity_increment_lots),
                abs_tol=1e-9,
            )
        ):
            raise ValueError("target quantity outside declared bounds/increments")
        if self.next_forecast is not None or self.last_forecast_time is not None:
            raise ValueError("shared target account cannot also consume standalone forecasts")
        self._advance(at, inclusive=False)
        direction = 1 if target_lots > 0 else -1 if target_lots < 0 else 0
        order = self.pending
        if order is not None and order.purpose == "entry":
            pending_quantity = order.quantity_lots or c.quantity_lots
            if direction != order.direction or not math.isclose(quantity, pending_quantity):
                self._terminal_order("cancelled", "portfolio_target_changed", at)
        if self.pending is not None:
            return
        p = self.position
        if p is not None:
            if p.direction != direction or not math.isclose(p.quantity_lots, quantity):
                p.exit_requested = True
                self._submit(at, identity, "exit", p.direction, p.close_bar_index, p.quantity_lots)
        elif direction:
            # Portfolio owns observed-bar holding and intent expiry, not this timer.
            self._submit(at, identity, "entry", direction, 2**63 - 1, quantity)

    def _decide(self, f: Forecast) -> None:
        stamp = utc_time(f.available_at_utc)
        if self.last_forecast_time is not None and stamp < self.last_forecast_time:
            raise ValueError("forecast availability must be chronological")
        self.last_forecast_time = stamp
        self._accrue(stamp)
        direction, reason = decision(f, self.config)
        outcome = "accepted"
        if direction is None:
            outcome = "rejected"
        elif direction == 0:
            outcome, reason = "neutral", "zero_expected_return"
        elif self.position is not None or self.pending is not None:
            outcome, reason = "rejected", "position_or_order_already_active"
        elif f.bar_index > self.completed_bar_index:
            outcome, reason = "rejected", "forecast_bar_not_complete"
        elif f.bar_index + f.horizon_bars <= self.completed_bar_index:
            outcome, reason = "rejected", "forecast_horizon_already_complete"
        self.counts[f"decisions_{outcome}"] += 1
        self.record(
            "decisions",
            {
                **asdict(f),
                "decision_utc": stamp,
                "status": outcome,
                "reason": reason,
                "direction": direction,
                "policy_id": self.config.policy_id,
            },
        )
        if outcome == "accepted" and direction is not None:
            self._submit(stamp, f.forecast_id, "entry", direction, f.bar_index + f.horizon_bars)

    def _advance(self, stamp: datetime, *, inclusive: bool) -> None:
        # A close-time forecast is processed after its BarClose, before later quotes.
        while self.next_forecast is not None:
            at = utc_time(self.next_forecast.available_at_utc)
            if at > stamp or (at == stamp and not inclusive):
                break
            if self.pending is not None and self.pending.expires_utc <= at:
                self._accrue(self.pending.expires_utc)
                self._terminal_order("expired", "no_eligible_quote_before_expiry", self.clock or at)
            self._decide(self.next_forecast)
            self.next_forecast = next(self.forecasts, None)
        if self.pending is not None and self.pending.expires_utc <= stamp:
            self._accrue(self.pending.expires_utc)
            self._terminal_order("expired", "no_eligible_quote_before_expiry", self.clock or stamp)
        self._accrue(stamp)

    def _fill(self, q: Quote) -> None:
        o = self.pending
        assert o is not None
        c = self.config
        quantity = o.quantity_lots if o.quantity_lots is not None else c.quantity_lots
        mult = quantity * c.contract_ounces_per_lot * c.account_currency_per_usd
        direction = o.direction if o.purpose == "entry" else -o.direction
        side = q.ask if direction == 1 else q.bid
        price = side + direction * c.slippage_usd_per_ounce_per_leg
        if price <= 0:
            self._terminal_order("rejected", "slippage_produces_invalid_price", q.timestamp_utc)
            return
        commission = c.commission_account_per_lot_per_leg * quantity
        spread_cost = abs(side - q.midpoint) * mult
        self.total_commission += commission
        self.turnover += price * mult
        self.counts["fills"] += 1
        fill = {
            "order_id": o.order_id,
            "forecast_id": o.forecast_id,
            "purpose": o.purpose,
            "timestamp_utc": q.timestamp_utc,
            "timestamp_local": q.timestamp_local,
            "sequence": q.sequence,
            "direction": direction,
            "price_usd_per_ounce": price,
            "bid": q.bid,
            "ask": q.ask,
            "midpoint": q.midpoint,
            "quantity_lots": quantity,
            "contract_ounces_per_lot": c.contract_ounces_per_lot,
            "commission_account": commission,
            "spread_cost_account": spread_cost,
            "slippage_account": c.slippage_usd_per_ounce_per_leg * mult,
        }
        self.record("fills", fill)
        self._terminal_order("filled", "first_eligible_subsequent_quote", q.timestamp_utc)
        if o.purpose == "entry":
            self.position_number += 1
            self.position = Position(
                self.position_number,
                o.forecast_id,
                o.direction,
                quantity,
                q.timestamp_utc,
                q.timestamp_local,
                price,
                q.midpoint,
                spread_cost,
                commission,
                o.close_bar_index,
                o.immediate_midpoint,
            )
            self._cash_flow(q.timestamp_utc, -commission, "entry_commission")
            self.record("positions", {**asdict(self.position), "status": "opened"})
        else:
            p = self.position
            assert p is not None
            gross = p.direction * (price - p.entry_price) * mult
            midpoint = p.direction * (q.midpoint - p.entry_midpoint) * mult
            spread = p.entry_spread_cost + spread_cost
            slip = 2 * c.slippage_usd_per_ounce_per_leg * mult
            net = gross - p.entry_commission - commission - p.financing
            immediate = (
                p.direction * (o.immediate_midpoint - p.immediate_entry_midpoint) * mult
                if o.immediate_midpoint is not None and p.immediate_entry_midpoint is not None
                else None
            )
            if not math.isclose(midpoint - spread - slip, gross, abs_tol=1e-7):
                raise RuntimeError("spread/slippage cash-flow reconciliation failed")
            self._cash_flow(q.timestamp_utc, gross, "closed_price_pnl")
            self._cash_flow(q.timestamp_utc, -commission, "exit_commission")
            self.realized += net
            self.total_gross += gross
            self.midpoint_gross += midpoint
            self.total_spread += spread
            self.total_slippage += slip
            self.counts["closed_positions"] += 1
            self.record(
                "trades",
                {
                    **asdict(p),
                    "exit_utc": q.timestamp_utc,
                    "exit_local": q.timestamp_local,
                    "exit_price": price,
                    "gross_price_pnl_account": gross,
                    "commission_account": p.entry_commission + commission,
                    "financing_account": p.financing,
                    "net_pnl_account": net,
                    "matched_midpoint_pnl_account": midpoint,
                    "spread_cost_account": spread,
                    "slippage_account": slip,
                    "immediate_midpoint_counterfactual_account": immediate,
                    "midpoint_delay_difference_account": (
                        midpoint - immediate if immediate is not None else None
                    ),
                },
            )
            self.record(
                "positions",
                {
                    "position_id": p.position_id,
                    "status": "closed",
                    "timestamp_utc": q.timestamp_utc,
                },
            )
            self.position = None

    def mark(self, stamp: datetime) -> dict[str, Any]:
        p = self.position
        fresh = self._fresh(self.last_quote, stamp)
        unrealized: float | None = 0.0
        if p is not None:
            q = self.last_quote
            unrealized = (
                p.direction
                * ((q.bid if p.direction == 1 else q.ask) - p.entry_price)
                * p.quantity_lots
                * self.config.contract_ounces_per_lot
                * self.config.account_currency_per_usd
                if fresh and q is not None
                else None
            )
        equity = self.cash + unrealized if unrealized is not None else None
        if equity is not None:
            self.peak_equity = max(self.peak_equity, equity)
            self.max_drawdown = max(self.max_drawdown, self.peak_equity - equity)
        open_paid = (p.entry_commission + p.financing) if p is not None else 0.0
        error = self.cash - (self.config.initial_cash_account + self.realized - open_paid)
        tolerance = max(1e-7, 4 * math.ulp(self.cash))
        if not math.isclose(error, 0, abs_tol=tolerance):
            raise RuntimeError("cash / realized / open costs reconciliation failed")
        return {
            "timestamp_utc": stamp,
            "cash_account": self.cash,
            "realized_net_pnl_account": self.realized,
            "unrealized_price_pnl_account": unrealized,
            "equity_account": equity,
            "quote_fresh": fresh,
            "open_quantity_lots": p.quantity_lots if p else 0.0,
            "open_direction": p.direction if p else 0,
            "reconciliation_error_account": error,
            "reconciliation_tolerance_account": tolerance,
        }

    def consume(self, events: Iterable[Quote | BarClose]) -> None:
        """Incremental calls are equivalent to one call; never finalize a chunk."""
        for event in events:
            stamp = utc_time(event.timestamp_utc)
            key = (
                (stamp, 0, event.sequence)
                if isinstance(event, Quote)
                else (stamp, 1, event.bar_index)
            )
            if self.last_key is not None and key < self.last_key:
                raise ValueError("input events out of order; preserve original equal-time sequence")
            self.last_key = key
            self._advance(stamp, inclusive=False)
            if isinstance(event, BarClose):
                if event.bar_index != self.completed_bar_index + 1:
                    raise ValueError("bar indices must be contiguous across partitions")
                self.completed_bar_index = event.bar_index
                if (
                    self.pending is not None
                    and self.pending.purpose == "entry"
                    and self.pending.close_bar_index <= event.bar_index
                ):
                    self._terminal_order("unfilled", "horizon_complete_before_entry", stamp)
                p = self.position
                if p is not None and p.close_bar_index <= event.bar_index and not p.exit_requested:
                    p.exit_requested = True
                    self._submit(stamp, p.forecast_id, "exit", p.direction, p.close_bar_index)
                self._advance(stamp, inclusive=True)
                continue
            self.counts["quotes_observed"] += 1
            # Gaps invalidate pending orders; they do not synthesize closing prices.
            if (
                self.previous_quote is not None
                and (stamp - self.previous_quote.timestamp_utc).total_seconds() * 1000
                > self.config.max_gap_ms
            ):
                self.counts["session_or_data_gaps"] += 1
                if self.pending is not None:
                    self._terminal_order("unfilled", "session_or_data_gap", stamp)
            self.previous_quote = event
            valid = (
                math.isfinite(event.bid) and math.isfinite(event.ask) and 0 < event.bid <= event.ask
            )
            if not valid or not self._fresh(event, stamp):
                self.counts["invalid_quotes" if not valid else "stale_quotes"] += 1
                self.record(
                    "quote_rejections",
                    {
                        "timestamp_utc": stamp,
                        "sequence": event.sequence,
                        "reason": "invalid_quote" if not valid else "stale_quote",
                    },
                )
            else:
                self.last_quote = event
                o = self.pending
                if o is not None and o.arrival_utc < stamp < o.expires_utc:
                    self._fill(event)
            mark = self.mark(stamp)
            if mark["equity_account"] is None:
                self.counts["unknown_equity_marks"] += 1
            if (
                self.last_equity_record is None
                or (stamp - self.last_equity_record).total_seconds() * 1000
                >= self.config.equity_sample_ms
            ):
                self.record("equity", mark)
                self.last_equity_record = stamp

    def checkpoint(self, at_utc: datetime) -> dict[str, Any]:
        """Left-boundary observation: advance clocks without ending/resetting the run.

        Events at the boundary belong to the following interval. Forecasts exactly
        at it wait for their bar-close/input events; positions and pending orders carry.
        """
        at = utc_time(at_utc)
        self._advance(at, inclusive=False)
        mark = self.mark(at)
        self.record("equity", mark)
        return mark

    def target_state(self) -> dict[str, Any]:
        """Bounded restart state for the target-only account; no future forecast iterator."""
        if self.next_forecast is not None or self.last_forecast_time is not None:
            raise ValueError("serialization supported for shared target accounts only")
        names = (
            "completed_bar_index",
            "cash",
            "cash_correction",
            "realized",
            "total_commission",
            "total_financing",
            "total_gross",
            "total_spread",
            "total_slippage",
            "midpoint_gross",
            "turnover",
            "lot_seconds",
            "position_seconds",
            "peak_equity",
            "max_drawdown",
            "order_number",
            "position_number",
        )
        clocks = ("clock", "first_clock", "last_equity_record")
        return {
            "schema": "TARGET_ACCOUNT_STATE_V001",
            "config_sha256": content_hash(self.config.resolved()),
            "numbers": {name: getattr(self, name) for name in names},
            "clocks": {
                name: getattr(self, name).isoformat() if getattr(self, name) else None
                for name in clocks
            },
            "counts": dict(self.counts),
            "last_key": [self.last_key[0].isoformat(), *self.last_key[1:]]
            if self.last_key
            else None,
            **{
                name: asdict(getattr(self, name)) if getattr(self, name) else None
                for name in ("pending", "position", "last_quote", "previous_quote")
            },
        }

    @classmethod
    def restore_target_state(
        cls, config: ExecutionConfig, state: dict[str, Any], record: Recorder
    ) -> ExecutionEngine:
        if state.get("schema") != "TARGET_ACCOUNT_STATE_V001" or state.get(
            "config_sha256"
        ) != content_hash(config.resolved()):
            raise ValueError("target state schema/configuration mismatch")
        engine = cls(config, [], record)
        expected = set(engine.target_state()["numbers"])
        if set(state["numbers"]) != expected:
            raise ValueError("incomplete account state")
        for name, value in state["numbers"].items():
            setattr(engine, name, value)
        for name, value in state["clocks"].items():
            setattr(engine, name, datetime.fromisoformat(value) if value else None)
        engine.counts = Counter(state["counts"])
        key = state["last_key"]
        engine.last_key = (datetime.fromisoformat(key[0]), key[1], key[2]) if key else None
        for name, kind in (
            ("pending", Order),
            ("position", Position),
            ("last_quote", Quote),
            ("previous_quote", Quote),
        ):
            body = state[name]
            if body is not None:
                body = body.copy()
                for field, value in body.items():
                    if (
                        value is not None
                        and (field.endswith("_utc") or field.endswith("_local"))
                        and isinstance(value, str)
                    ):
                        body[field] = datetime.fromisoformat(value)
                setattr(engine, name, kind(**body))
        return engine

    def finish(self, end_utc: datetime) -> dict[str, Any]:
        """Declared cutoff: no future quote is read and no earlier quote becomes a fill."""
        end = utc_time(end_utc)
        self._advance(end, inclusive=True)
        if self.pending is not None:
            self._terminal_order("unfilled", "end_of_run", end)
        mark = self.mark(end)
        self.record("equity", mark)
        if self.position is not None:
            self.record("positions", {**asdict(self.position), "status": "open_at_end"})
        duration = (end - self.first_clock).total_seconds() if self.first_clock else 0.0
        pnl = (
            mark["equity_account"] - self.config.initial_cash_account
            if mark["equity_account"] is not None
            else None
        )
        return {
            "counts": dict(self.counts),
            "final_mark": mark,
            "closed_gross_price_pnl_account": self.total_gross,
            "closed_net_pnl_account": self.realized,
            "total_commission_account": self.total_commission,
            "total_financing_account": self.total_financing,
            "closed_matched_midpoint_pnl_account": self.midpoint_gross,
            "closed_spread_cost_account": self.total_spread,
            "closed_slippage_account": self.total_slippage,
            "turnover_account": self.turnover,
            "exposure_lot_seconds": self.lot_seconds,
            "time_exposure_fraction": self.position_seconds / duration if duration > 0 else None,
            "max_drawdown_account": self.max_drawdown,
            "cutoff_equity_return": pnl / self.config.initial_cash_account
            if pnl is not None
            else None,
            "return_definition": "cutoff marked-equity change / initial cash; no leverage or annualization",
            "drawdown_definition": "absolute account-currency decline from running marked-equity peak",
            "drawdown_coverage": "known quote marks only; stale unpriced intervals are unknown",
            "mark_convention": "liquidation-side quote, excluding prospective exit fee/slippage",
            "end_policy": self.config.end_policy,
            "open_position": asdict(self.position) if self.position else None,
        }
