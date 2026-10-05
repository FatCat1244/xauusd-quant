"""Explicitly rearm a reconciled, never-submitted quote abort for one demo smoke test.

Uses the unchanged Stage18 coordinator/risk/native boundaries. This operator tool
does not migrate checkpoints, reset loss history, retry broker requests or admit
strategy operation. Its own source hash is recorded separately from src identity.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from xauusd_quant.alpha_portfolio.plan import freeze, versioned
from xauusd_quant.demo.config import load_demo_config
from xauusd_quant.demo.coordinator import Coordinator
from xauusd_quant.demo.journal import Journal
from xauusd_quant.demo.runs import account_directory, readiness
from xauusd_quant.demo.worker import DemoProcessBroker
from xauusd_quant.execution.readiness import sha256
from xauusd_quant.risk.contracts import MarketSnapshot, RiskRequest
from xauusd_quant.risk.policy import load_configuration
from xauusd_quant.shadow.config import load_shadow_config
from xauusd_quant.shadow.runs import code_identity

PLAN = {
    "plan_id": "DEMO_UNSUBMITTED_REARM_PLAN_V001",
    "scope": "explicit operator rearm after a never-submitted mechanical quote abort only",
    "eligibility": "SMOKE, quote-abort halts only, intact checkpoint, fresh identity, flat reconciled account",
    "budget": "one invocation, one entry send maximum, configured cleanup budget, <=300 seconds",
    "guards": "unchanged Stage18 exact quote/account binding, expiry, risk, native identity and ownership",
    "history": "preserve every intent, halt audit, daily loss, HWM, order-rate and turnover history",
    "retry": "no resubmission or automatic rearming after any attempted send; another quote abort stops",
    "acceptance": "verified entry and owned closure/deal accounting; no strategy or profit claim",
}
QUOTE_HALTS = {
    "MATERIAL_STATE_CHANGED_REVALIDATION_REQUIRED", "POST_CHECK_STATE_CHANGED",
    "IDENTITY_OR_READINESS_FAILURE",
}


def rearm_unsubmitted(
    coordinator: Coordinator, snapshot: dict[str, Any], now: datetime, *, operator: str,
) -> None:
    """Audited local operator transition; authorizes this process, never a restart."""
    c = coordinator
    p = c.risk.configuration.policy
    if c.config.run_type != "SMOKE" or p is None or not operator.strip():
        raise ValueError("explicit operator and dedicated SMOKE configuration required")
    if (not c.intents or not c.halts or not set(c.halts) <= QUOTE_HALTS
        or not set(c.risk.state.halts) <= {f"KILL:{reason}" for reason in QUOTE_HALTS}
        or not {"MATERIAL_STATE_CHANGED_REVALIDATION_REQUIRED", "POST_CHECK_STATE_CHANGED"}
        .intersection(c.halts)):
        raise ValueError("only an explicit never-submitted quote abort can be rearmed")
    # Scan the full durable history as well as current state: terminal rejection
    # of a prior attempted send must never turn this into a retry tool.
    prior_intents = [intent for row in c.journal.rows if row["event"] == "checkpoint"
                     for intent in row["data"].get("intents", {}).values()]
    if any(v.get("submission_utc") for v in [*prior_intents, *c.intents.values()]):
        raise ValueError("prior submission attempt prohibits this rearming path")
    if any(row["event"] == "broker_response" for row in c.journal.rows):
        raise ValueError("broker submission evidence prohibits this rearming path")
    c._validate(snapshot, now)
    quote = snapshot["quote"]
    market = MarketSnapshot(datetime.fromtimestamp(quote["time_msc"] / 1000, UTC), now,
                            quote["bid"], quote["ask"])
    if (snapshot["permissions"] is not True or snapshot["positions"] or snapshot["orders"]
        or c.risk._market_reasons(market, now, reducing=False)):
        raise ValueError("fresh executable quote and permitted flat account required")
    if not c.reconcile(snapshot, now).get("verified_flat"):
        raise ValueError("verified flat reconciliation required before rearming")
    if any(v["state"] != "REJECTED" for v in c.intents.values()):
        raise ValueError("all previous unsubmitted intents must be retired")
    prior_halts = c.halts.copy()
    c.risk.rearm(now, operator=operator, reason="explicit never-submitted quote-abort recovery")
    c._record("execution_rearm", {"operator": operator, "prior_halts": prior_halts,
        "received_utc": now.isoformat(), "plan_id": PLAN["plan_id"],
        "script_sha256": sha256(Path(__file__)), "same_process_only": True})
    c.halts.clear()
    # This runtime flag normally rejects restart arming. The explicit audited
    # operator transition permits only this same process to arm; it is not saved.
    c.recovered = False
    c.persist()


def run(root: Path, config_path: Path, run_id: str, operator: str) -> dict[str, Any]:
    versioned(run_id)
    if not run_id.replace("_", "").isalnum():
        raise ValueError("path-safe run identity required")
    config = load_demo_config(config_path)
    directory = root / "results/demo/runs" / run_id
    if directory.exists():
        raise ValueError("immutable run already exists")
    freeze(directory / "plan.json", PLAN)
    report = readiness(root, config)
    freeze(directory / "readiness.json", report)
    if report["blocking_reasons"] or config.run_type != "SMOKE":
        raise ValueError("configured mechanical SMOKE only; strategy stays blocked")
    terminal = load_shadow_config(root / str(config.terminal_config_path))
    risk = load_configuration(root / str(config.risk_config_path))
    source = code_identity(root)
    freeze(directory / "provenance.json", {"code_identity": source,
        "script_sha256": sha256(Path(__file__)), "execution_identity": config.identity,
        "risk_identity": risk.identity, "operator": operator, "plan_id": PLAN["plan_id"]})
    journal = Journal(account_directory(root, terminal))
    broker = DemoProcessBroker(config, terminal)
    c: Coordinator | None = None
    started = time.monotonic()
    failure: str | None = None
    try:
        c = Coordinator(config, terminal, risk, broker, journal, run_id, source)
        broker.connect()
        since = c.opened_utc or datetime.now(UTC) - timedelta(seconds=2)
        snapshot = broker.snapshot(since)
        now = datetime.fromisoformat(snapshot["received_utc"])
        rearm_unsubmitted(c, snapshot, now, operator=operator)
        c.initialize(snapshot, now, arm=True)
        p = risk.policy
        assert p is not None
        if time.monotonic() - started >= float(config.max_duration_seconds or 0) - float(config.cleanup_seconds or 0):
            raise ValueError("rearming consumed entry time; preserve cleanup capacity")
        target = float(config.quantity_lots or 0) * (1 if config.side == "BUY" else -1)
        request = RiskRequest(f"{run_id}:entry", p.expected_portfolio_id, "SMOKE_V001",
            now, now, target, {"EXECUTION_SMOKE": target}, p.sizing_method)
        intent = c.propose(request, snapshot, now,
            expires_utc=now + timedelta(seconds=float(config.intent_ttl_seconds or 0)))
        if intent["state"] == "RISK_APPROVED":
            fresh = broker.snapshot(since)
            c.submit(request.intent_id, fresh, datetime.fromisoformat(fresh["received_utc"]))
        hold_until = min(started + float(config.max_duration_seconds or 0) - float(config.cleanup_seconds or 0),
                         time.monotonic() + float(config.hold_seconds or 0))
        while time.monotonic() < hold_until:
            snapshot = broker.snapshot(since)
            now = datetime.fromisoformat(snapshot["received_utc"])
            if not c.reconcile(snapshot, now).get("reconciled") or not snapshot["positions"] or c.halts:
                break
            time.sleep(min(1.0, max(0, hold_until - time.monotonic())))
    except (Exception, KeyboardInterrupt) as exc:
        failure = "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else type(exc).__name__
        if c is not None:
            c.halt("EXPLICIT_REARM_SMOKE_ABORTED", datetime.now(UTC))
    finally:
        # Always reconcile and attempt only verified owned reduction, even if an
        # exception occurred after a send. Unknown ownership never permits exit.
        try:
            if c is not None and c.baseline is not None:
                p = risk.policy
                assert p is not None
                for sequence in range(int(config.cleanup_request_budget or 0) + 1):
                    if time.monotonic() - started >= float(config.max_duration_seconds or 0):
                        break
                    snapshot = broker.snapshot(c.opened_utc or datetime.now(UTC))
                    now = datetime.fromisoformat(snapshot["received_utc"])
                    reconciled = c.reconcile(snapshot, now)
                    if not reconciled.get("reconciled") or not snapshot["positions"]:
                        break
                    if sequence == int(config.cleanup_request_budget or 0):
                        break
                    close = RiskRequest(f"{run_id}:close:{sequence}", p.expected_portfolio_id,
                        "SMOKE_V001", now, now, 0, {"EXECUTION_SMOKE": 0}, p.sizing_method)
                    intent = c.propose(close, snapshot, now,
                        expires_utc=now + timedelta(seconds=float(config.intent_ttl_seconds or 0)))
                    if intent["state"] != "RISK_APPROVED":
                        break
                    fresh = broker.snapshot(c.opened_utc or now)
                    result = c.submit(close.intent_id, fresh, datetime.fromisoformat(fresh["received_utc"]))
                    response = result.get("response") or {}
                    if response.get("retcode") != 10010:
                        final = broker.snapshot(c.opened_utc or now)
                        c.reconcile(final, datetime.fromisoformat(final["received_utc"]))
                        break
                c.armed = False
                c.persist()
        except (Exception, KeyboardInterrupt):
            failure = "CLEANUP_UNRESOLVED"
        finally:
            broker.shutdown()
            journal.close()
    result = c.summary() if c is not None else {"verified_flat": False, "submissions": 0}
    result.update({"failure": failure, "code_identity": source,
        "script_sha256": sha256(Path(__file__)), "wall_seconds": time.monotonic() - started,
        "status": "VERIFIED_DEMO_LIFECYCLE" if not failure and result.get("verified_flat")
        and result.get("verified_entry_deals") and result.get("verified_close_deals")
        else "NO_VERIFIED_LIFECYCLE", "strategy_active": False})
    freeze(directory / "verdict.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--operator", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = run(root, args.demo_config, args.run_id, args.operator)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "VERIFIED_DEMO_LIFECYCLE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
