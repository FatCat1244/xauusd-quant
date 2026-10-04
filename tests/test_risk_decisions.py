"""Known numerical budgets, units and hostile input decisions."""

from dataclasses import replace
from datetime import timedelta

import pytest

from risk_synth import ROOT, T, account, configuration, health, market, request
from xauusd_quant.execution.engine import MemoryRecorder
from xauusd_quant.risk.contracts import AccountSpec, InstrumentSpec
from xauusd_quant.risk.engine import RiskEngine
from xauusd_quant.risk.policy import RiskConfiguration, load_configuration


def ready(**changes: object) -> RiskEngine:
    engine = RiskEngine(configuration(**changes), MemoryRecorder())
    assert engine.observe_account(account(), reconcile=True)
    return engine


def test_unconfigured_never_approves() -> None:
    engine = RiskEngine(load_configuration(ROOT / "config/risk.yaml"), MemoryRecorder())
    d = engine.evaluate(request(), market(), {"A": health()})
    assert d["decision"] == "REJECT" and d["rules"] == ["UNCONFIGURED"]
    assert not engine.state.reservations and engine.state.state == "UNCONFIGURED"


def test_known_horizon_loss_rounding_and_caps() -> None:
    engine = ready(position_loss_budget=40, alpha_loss_budget=40)
    d = engine.evaluate(request(target=.037), market(), {"A": health()})
    # Per lot: (1800.1*.01 + .20 + .04 + .10)*100 + 6 + 2*900/86400.
    loss = 1840.1 + .020833333333333332
    assert d["measurements"]["loss_per_lot"] == pytest.approx(loss)
    assert d["decision"] == "RESIZE" and d["approved_target_lots"] == .02
    assert d["measurements"]["estimated_loss"] == pytest.approx(.02 * loss)
    assert d["measurements"]["post_round_limits_valid"]
    assert len(engine.state.reservations) == 1


def test_minimum_exceeds_budget_never_round_up() -> None:
    engine = ready(account_loss_budget=1, position_loss_budget=1, alpha_loss_budget=1)
    d = engine.evaluate(request(target=.01), market(), {"A": health()})
    assert d["decision"] == "REJECT" and "BELOW_MINIMUM_WITHIN_BUDGET" in d["rules"]
    assert not engine.state.reservations


def test_margin_and_gross_sleeves_resize() -> None:
    engine = ready()
    engine.observe_account(account(.1, free_margin=145), reconcile=True)
    d = engine.evaluate(request(second=.1), market(.1), {"A": health(second=.1)})
    assert d["approved_target_lots"] == .01  # 95 free after floor; .02 needs 180.
    engine = ready(max_gross_intended_lots=.04)
    d = engine.evaluate(request(target=.02, sleeve_lots={"A": .08, "B": -.06}),
                        market(), {"A": health(), "B": health("B")})
    assert "BELOW_MINIMUM_WITHIN_BUDGET" in d["rules"]


def test_margin_cap_reserves_immediate_costs_and_quote_side_mark() -> None:
    e = ready()
    # .01 lot nominal margin is 90.005; post-entry spread/slip/commission
    # consume .25 more, so exactly nominal-margin headroom cannot enable it.
    e.observe_account(account(.1, free_margin=140.005), reconcile=True)
    d = e.evaluate(request(second=.1, target=.01), market(.1), {"A": health(second=.1)})
    assert d["approved_change_lots"] == 0 and "BELOW_MINIMUM_WITHIN_BUDGET" in d["rules"]


def test_stop_uses_quote_sides_and_explicit_costs() -> None:
    engine = ready(sizing_method="stop_distance", horizon_stress_fraction=None)
    req = request(target=.03, sizing_method="stop_distance", stop_price=1790.1)
    d = engine.evaluate(req, market(), {"A": health()})
    assert d["measurements"]["loss_per_lot"] == pytest.approx(1020.0208333333333)
    assert d["approved_target_lots"] == .03
    for stop in (None, 1900, float("nan"), 0, 1790.105):
        e = ready(sizing_method="stop_distance", horizon_stress_fraction=None)
        out = e.evaluate(replace(req, stop_price=stop), market(), {"A": health()})
        assert out["approved_change_lots"] == 0


@pytest.mark.parametrize("changes,reason", [
    ({"forecast_value": float("nan")}, "INVALID_FORECAST:A"),
    ({"forecast_value": None}, "INVALID_FORECAST:A"),
    ({"forecast_units": "volatility"}, "INVALID_FORECAST:A"),
    ({"model_id": "UNKNOWN"}, "SPECIFICATION_MISMATCH:A"),
    ({"feature_id": "UNKNOWN"}, "SPECIFICATION_MISMATCH:A"),
    ({"eligible": False}, "HEALTH_OR_ELIGIBILITY:A"),
    ({"eligible": "yes"}, "HEALTH_OR_ELIGIBILITY:A"),
    ({"warm": False}, "HEALTH_OR_ELIGIBILITY:A"),
    ({"features_ready": False}, "HEALTH_OR_ELIGIBILITY:A"),
    ({"calibration_healthy": False}, "HEALTH_OR_ELIGIBILITY:A"),
    ({"diagnostic_healthy": None}, "DIAGNOSTIC_UNHEALTHY_OR_MISSING:A"),
    ({"uncertainty": None}, "INVALID_UNCERTAINTY:A"),
    ({"volatility": .11}, "INVALID_VOLATILITY:A"),
    ({"forecast_available_utc": T + timedelta(seconds=1)}, "STALE_OR_INVALID_FORECAST_TIME:A"),
    ({"forecast_expiry_utc": T}, "STALE_OR_INVALID_FORECAST_TIME:A"),
])
def test_health_rejects_explicitly(changes: dict[str, object], reason: str) -> None:
    e = ready()
    d = e.evaluate(request(), market(), {"A": health(**changes)})
    assert reason in d["rules"] and d["approved_change_lots"] == 0


@pytest.mark.parametrize("quote,reason", [
    (None, "MISSING_QUOTE"),
    (market(bid=1801), "INVALID_BID_ASK"),
    (market(bid=float("nan")), "INVALID_BID_ASK"),
    (market(ask=1801), "SPREAD_LIMIT"),
    (market(bid=1799.905), "QUOTE_TICK_ALIGNMENT"),
    (market(-10), "STALE_OR_FUTURE_QUOTE"),
    (market(1), "STALE_OR_FUTURE_QUOTE"),
])
def test_market_rejections(quote: object, reason: str) -> None:
    d = ready().evaluate(request(), quote, {"A": health()})
    assert reason in d["rules"] and d["approved_change_lots"] == 0


def test_stale_account_and_warmup_block() -> None:
    e = ready()
    d = e.evaluate(request(second=6), market(6), {"A": health(second=6)})
    assert "STALE_OR_UNVERIFIED_ACCOUNT" in d["rules"] and d["approved_change_lots"] == 0
    e = ready(warmup_snapshots=2)
    assert e.evaluate(request(), market(), {"A": health()})["decision"] == "HALT_NEW_EXPOSURE"
    e.observe_account(account(1), reconcile=True)
    assert e.state.state == "READY"


def test_missing_conversion_and_incompatible_units_block_construction() -> None:
    cfg = configuration()
    assert cfg.account and cfg.instrument
    with pytest.raises(ValueError):
        replace(cfg.account, major_currency_per_usd=float("nan"))
    with pytest.raises(ValueError):
        replace(cfg.instrument, quantity_units="ounces")
    with pytest.raises(ValueError):
        replace(cfg.account, denomination="cent", ledger_units_per_major=1)
    with pytest.raises(ValueError):
        replace(cfg.instrument, price_units="points")
    for missing in ("account", "instrument", "policy"):
        e = RiskEngine(replace(cfg, **{missing: None}), MemoryRecorder())
        assert e.evaluate(request(), market(), {"A": health()})["rules"] == ["UNCONFIGURED"]


def test_cent_denomination_explicit_scaled_loss_budget() -> None:
    cfg = configuration()
    assert cfg.account and cfg.instrument and cfg.policy
    a = replace(cfg.account, denomination="cent", ledger_units_per_major=100)
    i = replace(cfg.instrument, commission_ledger_per_lot_leg=300,
                financing_long_ledger_per_lot_day=200, financing_short_ledger_per_lot_day=200)
    p = replace(cfg.policy, account_loss_budget=10000, position_loss_budget=6000,
                alpha_loss_budget=4000, minimum_free_margin=5000)
    e = RiskEngine(RiskConfiguration("CENT_SYNTH_V001", p, a, i), MemoryRecorder())
    assert e.observe_account(account(balance=1000000, equity=1000000, free_margin=1000000), reconcile=True)
    d = e.evaluate(request(), market(), {"A": health()})
    reference = ready().evaluate(request(), market(), {"A": health()})
    assert d["approved_target_lots"] == reference["approved_target_lots"]
    assert d["measurements"]["estimated_loss"] == pytest.approx(reference["measurements"]["estimated_loss"] * 100)


def test_configuration_validation() -> None:
    cfg = configuration()
    assert cfg.policy and cfg.instrument and cfg.account
    for changes in ({"max_net_lots": float("nan")}, {"warmup_snapshots": 0},
                    {"allow_overnight": "false"}, {"session_timezone": "Mars"},
                    {"horizon_stress_fraction": None}, {"max_positions": 2}):
        with pytest.raises((ValueError, KeyError)):
            replace(cfg.policy, **changes)
    assert isinstance(cfg.account, AccountSpec) and isinstance(cfg.instrument, InstrumentSpec)


def test_order_and_turnover_limits_and_session() -> None:
    e = ready(max_orders_per_window=1)
    e.state.order_times = [T.isoformat()]
    assert "ORDER_RATE_LIMIT" in e.evaluate(request(), market(), {"A": health()})["rules"]
    e = ready(max_turnover_lots_per_window=.005)
    assert "BELOW_MINIMUM_WITHIN_BUDGET" in e.evaluate(request(), market(), {"A": health()})["rules"]
    e = ready(session_start_hour=13, session_end_hour=14)
    assert "SESSION_CLOSED" in e.evaluate(request(), market(), {"A": health()})["rules"]
    e = ready(session_end_hour=12)
    assert "OVERNIGHT_RESTRICTION" in e.evaluate(request(), market(), {"A": health()})["rules"]
