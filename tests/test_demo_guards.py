"""Submission guards must block uncertainty, unsafe rounding and stale approvals."""
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from demo_synth import create, propose, smoke_config, smoke_risk
from xauusd_quant.demo.coordinator import settings
from xauusd_quant.risk.contracts import AccountSnapshot, MarketSnapshot, RiskRequest


@pytest.mark.parametrize("change", ["expiry", "missing", "quantity", "quote", "account", "post_check"])
def test_missing_expired_or_changed_approval_blocks_submission(tmp_path: Path, change: str) -> None:
    c, b, j = create(tmp_path)
    try:
        entry = propose(c, b)
        if change == "expiry":
            b.now += timedelta(seconds=6)
        elif change == "missing":
            c.risk.state.reservations.clear()
        elif change == "quantity":
            c.risk.state.reservations[entry["intent_id"]].signed_change += .01
        elif change == "quote":
            b.bid += .01
        elif change == "account":
            b.balance += 1
        elif change == "post_check":
            b.changed_snapshot = True
        # Isolate post-check change: pass the unchanged snapshot to submit.
        snapshot = deepcopy(entry["approval_snapshot"]) if change == "post_check" else b.snapshot()
        with pytest.raises(ValueError):
            c.submit(entry["intent_id"], snapshot, b.now)
        assert not b.sent
        assert not c.summary()["verified_flat"]
    finally:
        j.close()


def test_duplicate_intent_and_submission_do_not_allocate_twice(tmp_path: Path) -> None:
    c, b, j = create(tmp_path)
    try:
        first = propose(c, b)
        repeated = propose(c, b)
        assert first == repeated
        assert len(c.risk.state.reservations) == 1
        c.submit(first["intent_id"], b.snapshot(), b.now)
        c.submit(first["intent_id"], b.snapshot(), b.now)
        assert len(b.sent) == 1
        with pytest.raises(ValueError, match="collision"):
            changed = RiskRequest("ENTRY_V001", "EXECUTION_SMOKE_V001", "SMOKE_V001", b.now, b.now,
                                  .01, {"EXECUTION_SMOKE": .01}, "horizon_stress")
            c.propose(changed, b.snapshot(), b.now, expires_utc=b.now + timedelta(seconds=5))
    finally:
        j.close()


@pytest.mark.parametrize("field,value", [("tick_size", .1), ("ounces_per_lot", 10), ("lot_step", .02)])
def test_incompatible_units_block(tmp_path: Path, field: str, value: float) -> None:
    c, b, j = create(tmp_path)
    try:
        native_field = {"tick_size": "trade_tick_size", "ounces_per_lot": "trade_contract_size", "lot_step": "volume_step"}[field]
        snapshot = b.snapshot()
        snapshot["symbol"][native_field] = value
        with pytest.raises(ValueError, match="instrument"):
            c.reconcile(snapshot, b.now)
        assert not b.sent
    finally:
        j.close()


def test_smoke_cannot_disable_strategy_health(tmp_path: Path) -> None:
    c, b, j = create(tmp_path)
    try:
        request = RiskRequest("RAW_STRATEGY_V001", "EXECUTION_SMOKE_V001", "SMOKE_V001", b.now, b.now,
                              .02, {"EXECUTION_SMOKE": .02}, "horizon_stress")
        account = c.risk.configuration.account
        assert account is not None
        c.risk.observe_account(AccountSnapshot(
            "MARK_V001", b.now, b.now, account.specification_id, 10000, 10000, 10000, 0, True, True), reconcile=True)
        result = c.risk.evaluate(request, MarketSnapshot(b.now, b.now, b.bid, b.ask), {})
        assert result["decision"] == "REJECT"
        assert "MISSING_HEALTH:EXECUTION_SMOKE" in result["rules"]
        standard = smoke_risk()
        assert standard.policy is not None
        c.risk.configuration = replace(standard, policy=replace(standard.policy, expected_portfolio_id="ACTUAL_PORTFOLIO_V001"))
        with pytest.raises(ValueError, match="dedicated"):
            c.risk.evaluate_smoke(request, MarketSnapshot(b.now, b.now, b.bid, b.ask))
    finally:
        j.close()


def test_synthetic_terms_never_native_and_cleanup_capacity_required() -> None:
    with pytest.raises(ValueError, match="synthetic"):
        settings(smoke_config(), smoke_risk(), offline_synthetic=False)
    cfg = smoke_config()
    with pytest.raises(ValueError, match="cleanup"):
        settings(replace(cfg, max_net_lots=.1), smoke_risk(), offline_synthetic=True)


def test_broker_profit_conversion_missing_blocks_before_check(tmp_path: Path) -> None:
    c, b, j = create(tmp_path)
    try:
        entry = propose(c, b)
        b.calculation_factor = 100  # A cents/major mismatch must not pass unnoticed.
        with pytest.raises(ValueError, match="conversion"):
            c.submit(entry["intent_id"], b.snapshot(), b.now)
        assert not b.sent
    finally:
        j.close()


def test_precheck_only_never_sends(tmp_path: Path) -> None:
    c, b, j = create(tmp_path)
    try:
        entry = propose(c, b)
        c.submit(entry["intent_id"], b.snapshot(), b.now, precheck_only=True)
        assert not b.sent
        assert c.reconcile(b.snapshot(), b.now)["verified_flat"]
    finally:
        j.close()
