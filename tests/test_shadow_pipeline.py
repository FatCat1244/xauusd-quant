"""Synthetic forecasts reach shared execution only through authoritative risk."""

from dataclasses import replace
from datetime import timedelta

import pytest

from risk_synth import configuration
from shadow_synth import ROOT, FixtureDiagnostics, FixtureModel, T, feed, governed, trend
from xauusd_quant.alpha_portfolio.allocation import Allocation
from xauusd_quant.alpha_portfolio.intents import ObservedBarPolicy
from xauusd_quant.alpha_portfolio.portfolio import WeightUpdate
from xauusd_quant.alpha_portfolio.registry import Alpha
from xauusd_quant.execution.config import ExecutionConfig
from xauusd_quant.execution.engine import MemoryRecorder
from xauusd_quant.features.factory_config import load_features_config
from xauusd_quant.risk.portfolio import RiskPortfolio
from xauusd_quant.shadow.config import ShadowConfig
from xauusd_quant.shadow.pipeline import Binding, ShadowPipeline


def test_governed_synthetic_path_has_hand_checkable_prediction_and_local_fills() -> None:
    pipeline, sink = governed()
    feed(pipeline, trend())
    assert sink.tables["predictions"][0]["value"] is None  # first completed bar warms
    prediction = next(r for r in sink.tables["predictions"] if r["value"] is not None)
    features = next(r for r in sink.tables["features"] if r["ready"])
    assert prediction["value"] == pytest.approx(features["values"]["ret_1"] * .2)
    assert sink.tables["risk_decisions"]
    assert sink.tables["fills"]
    assert all(r["broker_request"] is False for r in sink.tables["approved_shadow_actions"])
    assert any(r["action"] == "WOULD_BUY" for r in sink.tables["approved_shadow_actions"])
    assert pipeline.portfolio is not None
    with pytest.raises(PermissionError):
        pipeline.portfolio.engine.request_target(T + timedelta(seconds=300), .1, "BYPASS")
    totals = pipeline.portfolio.account_totals
    for metric, total in totals.items():
        assert sum(v.get(metric, 0) for v in pipeline.portfolio.attribution.values()) == pytest.approx(total)


def test_missing_health_does_not_become_green() -> None:
    pipeline, sink = governed(diagnostics=False)
    feed(pipeline, trend())
    assert not sink.tables.get("fills")
    assert any("INVALID_VOLATILITY:A" in r["rules"] for r in sink.tables["risk_decisions"])
    assert any("INVALID_UNCERTAINTY:A" in r["rules"] for r in sink.tables["risk_decisions"])


def test_synthetic_binding_cannot_enter_actual_constructor() -> None:
    pipeline, _ = governed()
    with pytest.raises(ValueError, match="eligibility"):
        ShadowPipeline(pipeline.config, load_features_config(root=ROOT), MemoryRecorder(),
            bindings=pipeline.bindings, portfolio=pipeline.portfolio, cutoff=T)


def test_volatility_scaled_return_is_not_a_raw_return() -> None:
    pipeline, _ = governed()
    pipeline.bindings[0].predictor.units = "volatility_scaled_return"
    with pytest.raises(ValueError, match="compatible return"):
        ShadowPipeline(pipeline.config, load_features_config(root=ROOT), MemoryRecorder(),
            bindings=pipeline.bindings, portfolio=pipeline.portfolio, cutoff=T,
            offline_synthetic=True)


def test_backfill_never_places_even_local_orders() -> None:
    pipeline, sink = governed()
    initial_orders = len(sink.tables.get("orders", []))
    feed(pipeline, [replace(t, backfill=True) for t in trend()])
    assert len(sink.tables.get("orders", [])) == initial_orders
    assert not sink.tables.get("fills")


def test_delayed_batch_does_not_trade_matured_forecasts() -> None:
    pipeline, sink = governed()
    inputs = [replace(t, received_utc=T + timedelta(seconds=250.2)) for t in trend()]
    pipeline.consume(inputs, lambda: T + timedelta(seconds=250.3))
    assert not sink.tables.get("orders") and not sink.tables.get("fills")
    intents = sink.tables["exposure_intents"]
    assert all(i["target"] == 0 for i in intents)
    assert any("DELAYED_BAR_PUBLICATION" in i["rationale"] for i in intents)


def test_restart_preserves_halt_and_requires_reconciliation() -> None:
    pipeline, _ = governed()
    feed(pipeline, trend(10))
    pipeline.halt("SYNTHETIC_KILL", T + timedelta(seconds=91))
    state = pipeline.state()
    new, _ = governed()
    new.restore(state)
    assert new.portfolio is not None
    assert new.portfolio.risk.state.needs_reconciliation
    assert new.portfolio.risk.state.halts
    assert new.halts == {"SYNTHETIC_KILL"}


def test_future_health_provider_is_rejected() -> None:
    pipeline, _ = governed()
    provider = pipeline.bindings[0].diagnostics
    assert provider is not None
    old = provider.evaluate
    provider.evaluate = lambda available, values: old(available + timedelta(seconds=1), values)  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="diagnostics"):
        feed(pipeline, trend())


def test_two_frequencies_opposition_nets_and_future_data_keeps_prefix() -> None:
    def make() -> tuple[ShadowPipeline, MemoryRecorder]:
        sink = MemoryRecorder()
        execution = ExecutionConfig(quantity_lots=.02)
        account = RiskPortfolio(execution, {"A": "A_V001", "B": "B_V001"}, .02,
            sink, configuration("A", "B", max_health_age_seconds=1800))
        bindings = []
        for alpha_id, tf, direction in (("A", "5m", 1), ("B", "15m", -1)):
            model = FixtureModel()
            model.model_id, model.feature_id = f"MODEL_{alpha_id}_V001", f"FEATURES_{alpha_id}_V001"
            model.timeframe = tf
            # Fixed synthetic directional fixture, declared independently of outcomes.
            model.predict = lambda values, direction=direction: .001 * direction
            alpha = Alpha(alpha_id, f"{alpha_id}_V001", tf, 1, {}, {}, "BLOCKED", "BLOCKED",
                None, ("synthetic multi-frequency fixture",), synthetic=True)
            policy = ObservedBarPolicy(alpha_id, alpha.specification_id,
                replace(execution, timeframe=tf), 3600)
            bindings.append(Binding(alpha, model, policy, None, FixtureDiagnostics()))
        pipeline = ShadowPipeline(ShadowConfig(timeframes=("5m", "15m")),
            load_features_config(root=ROOT), sink, bindings=tuple(bindings),
            portfolio=account, cutoff=T, offline_synthetic=True)
        account.consume([WeightUpdate(Allocation("EQUAL_V001", T, T - timedelta(seconds=1),
            {"A": .5, "B": .5}, "equal", {}))])
        return pipeline, sink
    full, results = make()
    prefix, earlier = make()
    inputs = trend(182)
    feed(full, inputs)
    feed(prefix, inputs[:100])
    for table, rows in earlier.tables.items():
        assert results.tables[table][:len(rows)] == rows
    at = T + timedelta(seconds=1800.21)
    decisions = [r for r in results.tables["portfolio_decisions"] if r["timestamp_utc"] == at]
    assert decisions[-1]["target_lots"] == 0
    assert decisions[-1]["gross_intended_lots"] == pytest.approx(.02)
    assert decisions[-1]["cancelled_intent_lots"] == pytest.approx(.02)
    assert {r["alpha_id"] for r in results.tables["predictions"]} == {"A", "B"}
