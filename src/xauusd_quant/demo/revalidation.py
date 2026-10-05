"""Causal quote-only revalidation through Stage16, without releasing real reservations."""
from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

from ..execution.config import content_hash
from ..execution.engine import MemoryRecorder
from ..risk.contracts import AccountSnapshot, MarketSnapshot, RiskRequest
from ..risk.engine import RiskEngine
from ..risk.policy import RiskConfiguration


def evaluate_quote(
    payload: dict[str, Any], quote: dict[str, Any], account: dict[str, Any], now: datetime,
) -> dict[str, Any]:
    """Fresh same-quantity smoke decision in a clone; actual pending risk stays reserved.

    Order-rate and turnover history are retained even in the clone, conservatively
    counting this check as another approval. Nothing is reset in the actual engine.
    """
    cfg = payload.get("configuration")
    if not isinstance(cfg, RiskConfiguration) or not cfg.configured:
        raise ValueError("complete risk configuration required for quote revalidation")
    p, a = cfg.policy, cfg.account
    assert p is not None and a is not None
    if (p.expected_portfolio_id != "EXECUTION_SMOKE_V001"
        or p.allowed_allocation_ids != ("SMOKE_V001",) or p.alpha_contracts
        or p.require_diagnostic or p.sizing_method != "horizon_stress"):
        raise ValueError("quote revalidation is dedicated mechanical smoke only")
    clone = RiskEngine.restore(cfg, MemoryRecorder(), payload["checkpoint"])
    original_id = payload["intent_id"]
    reservation = clone.state.reservations.get(original_id)
    if (reservation is None or reservation.status != "approved" or reservation.reducing
        or reservation.cumulative_filled != 0 or clone.state.halts
        or len(clone.state.reservations) != 1
        or (clone.state.account or {}).get("signed_lots", 0) != 0
        or not math.isclose(reservation.signed_change, payload["signed_lots"], abs_tol=1e-9)):
        raise ValueError("unsubmitted same-quantity reserved approval required")
    # Replacement-risk calculation only: the actual reservation is untouched.
    del clone.state.reservations[original_id]
    external_flow = (clone.state.account or {}).get("external_flow_total", 0.0)
    mark = AccountSnapshot(content_hash({"quote": quote, "account": account, "now": now}),
        now, now, a.specification_id, account["balance"], account["equity"],
        account["margin_free"], 0.0, True, True, external_flow, 0.0)
    if not clone.observe_account(mark, reconcile=True):
        raise ValueError("fresh account state failed risk validation")
    market = MarketSnapshot(datetime.fromtimestamp(quote["time_msc"] / 1000, UTC), now,
                            quote["bid"], quote["ask"])
    signed = reservation.signed_change
    request = RiskRequest(f"{original_id}:quote:{content_hash(quote)[:12]}",
        p.expected_portfolio_id, "SMOKE_V001", now, now, signed,
        {"EXECUTION_SMOKE": signed}, p.sizing_method)
    decision = clone.evaluate_smoke(request, market)
    if (decision["decision"] != "APPROVE"
        or not math.isclose(decision["approved_change_lots"], signed, abs_tol=1e-9)
        or decision["measurements"].get("estimated_loss", math.inf) > reservation.estimated_loss + 1e-8
        or decision["measurements"].get("margin_required", math.inf) > reservation.margin + 1e-8):
        raise ValueError("fresh quote exceeds frozen risk limits or reserved envelope")
    return decision
