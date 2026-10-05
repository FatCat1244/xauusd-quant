"""Adversarial broker history, ownership and reduction measurements."""
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import pytest

from demo_synth import FakeBroker, create, propose
from xauusd_quant.demo.coordinator import Coordinator
from xauusd_quant.demo.journal import Journal


def filled(tmp_path: Path) -> tuple[Coordinator, FakeBroker, Journal]:
    c, b, j = create(tmp_path)
    entry = propose(c, b)
    c.submit(entry["intent_id"], b.snapshot(), b.now)
    c.reconcile(b.snapshot(), b.now)
    return c, b, j


def test_partial_close_then_verified_remainder_never_reverses(tmp_path: Path) -> None:
    c, b, j = filled(tmp_path)
    try:
        b.mode = "partial"
        close = propose(c, b, close=True)
        c.submit(close["intent_id"], b.snapshot(), b.now)
        c.reconcile(b.snapshot(), b.now)
        assert b.positions[0]["volume"] == pytest.approx(.01)
        assert not c.risk.state.reservations
        b.mode = "full"
        remainder = propose(c, b, close=True, name="CLOSE_REMAINDER_V001")
        assert remainder["request"]["volume"] == pytest.approx(.01)
        assert remainder["request"]["position"] == b.positions[0]["ticket"]
        c.submit(remainder["intent_id"], b.snapshot(), b.now)
        assert c.reconcile(b.snapshot(), b.now)["verified_flat"]
        assert c.summary()["verified_close_deals"] == 2
    finally:
        j.close()


def test_changed_market_price_can_still_permit_verified_risk_reduction(tmp_path: Path) -> None:
    c, b, j = filled(tmp_path)
    try:
        close = propose(c, b, close=True)
        b.changed_snapshot = True
        c.submit(close["intent_id"], deepcopy(close["approval_snapshot"]), b.now)
        b.changed_snapshot = False
        assert c.reconcile(b.snapshot(), b.now)["verified_flat"]
    finally:
        j.close()


@pytest.mark.parametrize("change", ["price", "ticket_duplicate", "position_identifier", "deal_side", "quantity", "cash"])
def test_history_corruption_or_unexplained_activity_cannot_pass(tmp_path: Path, change: str) -> None:
    c, b, j = filled(tmp_path)
    try:
        if change == "price":
            b.deals[0]["price"] += 1
        elif change == "ticket_duplicate":
            b.deals.append(deepcopy(b.deals[0]))
        elif change == "position_identifier":
            b.positions[0]["identifier"] += 1
        elif change == "deal_side":
            b.deals[0]["type"] = 1
        elif change == "quantity":
            b.positions[0]["volume"] += .01
        else:
            b.balance += 100
        try:
            result = c.reconcile(b.snapshot(), b.now)
            assert not result["reconciled"]
        except ValueError:
            pass
        assert not c.summary()["verified_flat"]
        assert len(b.sent) == 1
    finally:
        j.close()


def test_costs_breach_daily_limit_halt_and_permit_only_verified_close(tmp_path: Path) -> None:
    c, b, j = filled(tmp_path)
    try:
        b.bid -= 20
        b.ask -= 20
        b.now += timedelta(seconds=1)
        c.reconcile(b.snapshot(), b.now)
        assert "DAILY_LOSS" in c.risk.state.halts
        assert c.risk.state.state == "HALTED"
        close = propose(c, b, close=True)
        assert close["state"] == "RISK_APPROVED"
        c.submit(close["intent_id"], b.snapshot(), b.now)
        assert c.reconcile(b.snapshot(), b.now)["verified_flat"]
        assert c.risk.state.daily_loss > 25  # No claim that configured loss is guaranteed.
    finally:
        j.close()


def test_ambiguous_identical_comments_do_not_establish_ownership(tmp_path: Path) -> None:
    c, b, j = create(tmp_path, mode="timeout_after_fill")
    try:
        entry = propose(c, b)
        c.submit(entry["intent_id"], b.snapshot(), b.now)
        other = deepcopy(b.history[0])
        other["ticket"] += 100
        b.history.append(other)
        assert not c.reconcile(b.snapshot(), b.now)["reconciled"]
        assert "AMBIGUOUS_ORDER_OWNERSHIP" in c.halts
        assert c.risk.state.reservations
    finally:
        j.close()
