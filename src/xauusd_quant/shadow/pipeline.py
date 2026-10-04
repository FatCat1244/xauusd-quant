"""UTC features and forecasts -> alpha policies -> authoritative risk -> local fills.

Broker state never initializes the simulator. Actual operation is inactive
without eligible, feed-compatible frozen inputs and supplied risk/execution
terms. Synthetic bindings are accepted only by an offline fixture constructor.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from ..alpha_portfolio.intents import Intent, ObservedBarPolicy
from ..alpha_portfolio.portfolio import event_key
from ..alpha_portfolio.registry import Alpha
from ..execution.config import content_hash
from ..execution.engine import Quote, Recorder
from ..execution.io import parse_utc
from ..execution.policy import Forecast
from ..features.factory_config import FeatureFactoryConfig
from ..risk.contracts import HealthSnapshot
from ..risk.portfolio import RiskPortfolio
from .bars import Bars, serialize_bar
from .config import ShadowConfig
from .features import Features
from .ingestion import ContinuityFailure, Tick
from .models import Predictor


class Diagnostics(Protocol):
    identity: str
    def evaluate(self, available: datetime, values: dict[str, Any]) -> dict[str, Any]: ...
    def state(self) -> dict[str, Any]: ...
    def restore(self, body: dict[str, Any]) -> None: ...


@dataclass
class Binding:
    alpha: Alpha
    predictor: Predictor
    policy: ObservedBarPolicy
    feed_evidence_sha256: str | None
    diagnostics: Diagnostics | None = None


class ShadowPipeline:
    def __init__(self, config: ShadowConfig, feature_config: FeatureFactoryConfig,
                 record: Recorder, *, bindings: tuple[Binding, ...] = (),
                 portfolio: RiskPortfolio | None = None, cutoff: datetime | None = None,
                 offline_synthetic: bool = False,
                 blocked: tuple[str, ...] = ("NO_ELIGIBLE_ALPHAS", "RISK_UNCONFIGURED", "FEED_TRANSFER_UNASSESSED")) -> None:
        self.config, self.record, self.bindings, self.portfolio = config, record, bindings, portfolio
        self.blocked = blocked if not bindings else ()
        if bindings and (portfolio is None or cutoff is None):
            raise ValueError("risk-governed portfolio and prior eligibility cutoff required")
        ids = {b.alpha.alpha_id for b in bindings}
        if len(ids) != len(bindings) or len(bindings) > 3:
            raise ValueError("bounded unique shadow universe required")
        if portfolio is not None and set(portfolio.specifications) != ids:
            raise ValueError("portfolio must match shadow bindings exactly")
        for b in bindings:
            if (b.policy.alpha_id, b.policy.specification_id) != (b.alpha.alpha_id, b.alpha.specification_id):
                raise ValueError("policy identity differs from alpha")
            if not (offline_synthetic and b.alpha.synthetic) and (cutoff is None or not b.alpha.eligible_at(cutoff) or b.feed_evidence_sha256 is None):
                raise ValueError("historical eligibility and explicit Exness feed evidence required")
            if (b.predictor.target, b.predictor.units, b.predictor.timeframe, b.predictor.horizon) != ("future_return", "log_mid_return", b.alpha.timeframe, b.alpha.horizon_bars):
                raise ValueError("directional shadow alpha requires compatible return forecast")
            if portfolio is None or portfolio.risk.configuration.policy is None:
                raise ValueError("configured authoritative risk policy required")
            contract = portfolio.risk.configuration.policy.alpha_contracts[b.alpha.alpha_id]
            if contract != {"policy": b.alpha.specification_id, "model": b.predictor.model_id,
                             "features": b.predictor.feature_id}:
                raise ValueError("model/feature/policy identities differ from risk contract")
            if not offline_synthetic and portfolio.risk.configuration.policy.status != "verified_supplied":
                raise ValueError("synthetic risk settings cannot govern actual feed decisions")
        self.bars = Bars(config)
        self.features = {tf: Features(feature_config, tf, tuple(dict.fromkeys(
            f for b in bindings if b.alpha.timeframe == tf for f in b.predictor.features)) or ("ret_1",))
                         for tf in config.timeframes}
        if any(b.alpha.timeframe not in self.features for b in bindings):
            raise ValueError("binding timeframe not captured")
        self.identity = content_hash({"features": {t: f.identity for t, f in self.features.items()},
            "bindings": [{"alpha": b.alpha.resolved(), "model": b.predictor.identity,
                          "feed_evidence": b.feed_evidence_sha256,
                          "diagnostics": b.diagnostics.identity if b.diagnostics else None} for b in bindings],
            "risk": portfolio.risk.configuration.identity if portfolio else None,
            "execution": portfolio.config.resolved() if portfolio else None,
            "blocked": self.blocked})
        self.counts: Counter[str] = Counter()
        self.bar_indices: dict[str, int] = dict.fromkeys(config.timeframes, -1)
        self.halts: set[str] = set()
        self.last_decision: datetime | None = None

    def halt(self, reason: str, now: datetime) -> None:
        self.halts.add(reason)
        if self.portfolio is not None:
            self.portfolio.risk.kill(now, reason)
        self.record("health", {"reason": reason, "receipt_utc": now.isoformat(), "new_actions": "halted"})

    def consume(self, ticks: list[Tick], clock: Callable[[], datetime] | None = None) -> list[str]:
        try:
            return self._consume(ticks, clock)
        except (RuntimeError, ValueError, TypeError, OSError):
            self.halts.add("STREAM_OR_STORAGE_FAILURE")
            if self.portfolio is not None and ticks:
                at = ticks[-1].received_utc
                last = self.portfolio.risk.state.last_receipt_utc
                if last is not None:
                    at = max(at, parse_utc(last))
                # Preserve the original failure; caller stops even if storage
                # also prevents writing the halt audit.
                with suppress(ValueError, OSError):
                    self.portfolio.risk.kill(at, "STREAM_OR_STORAGE_FAILURE")
            raise

    def _consume(self, ticks: list[Tick], clock: Callable[[], datetime] | None = None) -> list[str]:
        """Batch quotes precede publications; use recorded completion clocks on replay."""
        clock = clock or (lambda: datetime.now(UTC))
        completed = []
        old_decisions = set(self.portfolio.risk.state.decisions) if self.portfolio else set()
        for tick in ticks:
            if self.last_decision is not None and tick.received_utc <= self.last_decision:
                raise ContinuityFailure("new retrieval receipt precedes completed decision")
            completed.extend(self.bars.consume(tick))
            self.counts["ticks"] += 1
        if self.bars.halts:
            now = ticks[-1].received_utc
            for reason in sorted(self.bars.halts - self.halts):
                self.halt(reason, now)
        latest_indices = {tf: self.bar_indices[tf] + sum(b.timeframe == tf for b in completed) for tf in self.features}
        # Do not expose retroactively recovered quotes to the local fill engine.
        if self.portfolio is not None:
            for tick in ticks:
                if not tick.backfill:
                    if tick.event_utc > tick.received_utc:
                        self.halt("CLOCK_HEALTH_UNRESOLVED", tick.received_utc)
                        continue
                    self.portfolio.consume([Quote(tick.received_utc,
                        tick.received_utc.replace(tzinfo=None), tick.bid, tick.ask,
                        tick.sequence, tick.event_utc)])
        clocks = []
        publications: list[Intent] = []
        for bar in completed:
            features = self.features[bar.timeframe].update(bar)
            self.bar_indices[bar.timeframe] += 1
            index = self.bar_indices[bar.timeframe]
            values = features["values"]
            predicted = {}
            for binding in self.bindings:
                if binding.alpha.timeframe == bar.timeframe and features["ready"] and bar.valid:
                    predicted[binding.alpha.alpha_id] = binding.predictor.predict(values)
            available = clock()  # After feature/model computations, not at historical bar close.
            if available.tzinfo is None or available < bar.available_utc or (ticks and available < ticks[-1].received_utc):
                raise ContinuityFailure("publication precedes input availability")
            if self.last_decision is not None and available < self.last_decision:
                raise ContinuityFailure("publication clock moved backwards")
            clocks.append(available.isoformat())
            self.last_decision = available
            self.counts["bars"] += 1
            self.record("bars", serialize_bar(bar))
            self.record("features", features | {"timeframe": bar.timeframe, "bar_open_utc": bar.open_utc.isoformat(),
                "available_utc": available.isoformat(), "initialization": "declared bounded UTC feed buffer"})
            reasons = list(self.blocked) + sorted(self.halts)
            if bar.backfill:
                reasons.append("BACKFILL_NO_RETROSPECTIVE_ACTION")
            if not bar.valid or not features["ready"]:
                reasons.append("WARMUP_OR_INVALID_BAR")
            close = bar.open_utc + timedelta(minutes=int(bar.timeframe[:-1]))
            if (available - close).total_seconds() > self.config.max_quote_age_seconds:
                reasons.append("DELAYED_BAR_PUBLICATION")
            for binding in self.bindings:
                if binding.alpha.timeframe != bar.timeframe:
                    continue
                value = predicted.get(binding.alpha.alpha_id)
                if value is not None and not math.isfinite(value):
                    value = None
                self.record("predictions", {"alpha_id": binding.alpha.alpha_id,
                    "model_id": binding.predictor.model_id, "model_identity": binding.predictor.identity,
                    "bar_open_utc": bar.open_utc.isoformat(), "available_utc": available.isoformat(),
                    "value": value, "units": "log_mid_return", "status": "ok" if value is not None else "blocked"})
                assert self.portfolio is not None
                horizon = binding.alpha.horizon_bars * int(bar.timeframe[:-1]) * 60
                binding_reasons = reasons.copy()
                if index + binding.alpha.horizon_bars <= latest_indices[bar.timeframe]:
                    binding_reasons.append("MATURED_HORIZON_NO_ACTION")
                diagnostics = binding.diagnostics.evaluate(available, values) if binding.diagnostics else {}
                diagnostic_time = diagnostics.get("available_utc")
                if diagnostics and (not isinstance(diagnostic_time, datetime) or diagnostic_time > available):
                    raise ContinuityFailure("health diagnostics not available at publication")
                health = HealthSnapshot(binding.alpha.alpha_id, binding.alpha.specification_id,
                    binding.predictor.model_id, binding.predictor.feature_id,
                    diagnostic_time if isinstance(diagnostic_time, datetime) else available, available,
                    available, available + timedelta(seconds=horizon), value if value is not None else float("nan"),
                    "log_mid_return", horizon, True, features["ready"], features["ready"],
                    not binding_reasons, diagnostics.get("calibration_healthy") is True,
                    diagnostics.get("diagnostic_healthy"), diagnostics.get("volatility"),
                    diagnostics.get("uncertainty"))
                self.record("uncertainty", {"alpha_id": binding.alpha.alpha_id,
                    "available_utc": available.isoformat(), "diagnostic_identity": binding.diagnostics.identity if binding.diagnostics else None,
                    "status": "available" if diagnostics else "unavailable", "values": diagnostics})
                # Unavailable volatility/uncertainty/health remain explicit and risk can reject.
                self.portfolio.health_update(health)
                if bar.backfill:
                    continue
                exit_intent = binding.policy.on_bar(index, available)
                candidate: Intent | None
                if binding_reasons:
                    if binding.policy.active is not None:
                        binding.policy.active, binding.policy.exit_bar = None, None
                    candidate = Intent(binding.alpha.alpha_id, binding.alpha.specification_id,
                        available, available, available + timedelta(seconds=1), 0, "invalid",
                        ";".join(binding_reasons), bar.timeframe, binding.alpha.horizon_bars)
                else:
                    forecast = Forecast(f"SHADOW_{binding.alpha.alpha_id}_{index}", bar.open_utc,
                        index, available, value, bar.timeframe, binding.alpha.horizon_bars,
                        provenance_id=binding.predictor.identity)
                    candidate = binding.policy.on_forecast(forecast, available) or exit_intent
                if candidate is not None:
                    publications.append(candidate)
            self.record("shadow_actions", {"bar_open_utc": bar.open_utc.isoformat(),
                "available_utc": available.isoformat(), "timeframe": bar.timeframe,
                "action": "NO_ACTION" if reasons or not self.bindings else "RISK_EVALUATION_REQUIRED",
                "reasons": reasons, "broker_request": False})
        if self.portfolio is not None and publications:
            # At a simultaneous event each sleeve has exactly one final target.
            unique = {(i.available_utc, i.alpha_id): i for i in publications}
            self.portfolio.consume(sorted(unique.values(), key=event_key))
        if self.portfolio is not None:
            for identity, decision in self.portfolio.risk.state.decisions.items():
                if identity in old_decisions:
                    continue
                change = decision["approved_change_lots"]
                action = "NO_ACTION"
                if change:
                    current = decision["measurements"]["actual_signed_lots"]
                    action = "WOULD_REDUCE" if current * change < 0 else ("WOULD_BUY" if change > 0 else "WOULD_SELL")
                self.record("approved_shadow_actions", {"intent_id": identity, "action": action,
                    "approved_change_lots": change, "risk_decision": decision["decision"],
                    "receipt_utc": decision["received_utc"], "broker_request": False})
        return clocks

    def state(self) -> dict[str, Any]:
        return {"identity": self.identity, "bars": self.bars.state(),
                "features": {t: f.state() for t, f in self.features.items()},
                "counts": dict(self.counts), "bar_indices": self.bar_indices,
                "halts": sorted(self.halts),
                "last_decision": self.last_decision.isoformat() if self.last_decision else None,
                "policies": {b.alpha.alpha_id: b.policy.state() for b in self.bindings},
                "diagnostics": {b.alpha.alpha_id: b.diagnostics.state() if b.diagnostics else None for b in self.bindings},
                "governed_account": self.portfolio.risk_state() if self.portfolio else None}

    def restore(self, body: dict[str, Any]) -> None:
        if body["identity"] != self.identity or set(body["features"]) != set(self.features):
            raise ValueError("model/feed/feature/risk checkpoint incompatible")
        self.bars = Bars.restore(self.config, body["bars"])
        for tf, state in body["features"].items():
            self.features[tf].restore(state)
        self.counts, self.bar_indices = Counter(body["counts"]), body["bar_indices"]
        self.halts = set(body["halts"])
        self.last_decision = parse_utc(body["last_decision"]) if body["last_decision"] else None
        for binding in self.bindings:
            binding.policy.restore(body["policies"][binding.alpha.alpha_id])
            if binding.diagnostics is not None:
                binding.diagnostics.restore(body["diagnostics"][binding.alpha.alpha_id])
        if self.portfolio is not None:
            old = self.portfolio
            self.portfolio = RiskPortfolio.restore_risk(old.config, old.specifications,
                old.budget_lots, old.record, old.risk.configuration, body["governed_account"], old.contracts)
            # Existing risk restore forces reconciliation before exposure increases.
