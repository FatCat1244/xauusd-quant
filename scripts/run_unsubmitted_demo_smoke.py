"""Explicitly rearm a reconciled, never-submitted quote abort for one demo smoke test.

The default retains exact quote binding. Explicit --fresh-quotes uses full Stage16
revalidation for one MARKET smoke, retaining loss history and reservations. Only
the documented never-submitted predecessor can migrate; no broker retries.
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
from xauusd_quant.demo.worker import VALIDATION_CODES, DemoProcessBroker
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
LEGACY_CODE = "7b33d23c30f78900d67f31c8cdbc534397e97718f29f6c3bfbf835fb52c8650c"
FRESH_PLAN = {
    **PLAN, "plan_id": "DEMO_FRESH_QUOTE_PLAN_V002",
    "guards": "same quantity, unchanged hard limits; two parent and two native Stage16 evaluations; MARKET SMOKE only",
    "reservation": "persist supplied position/account/sleeve allowance and free-margin headroom before check; never release unresolved risk",
    "migration": "known predecessor source only; identical configurations, intact full journal without submissions, fresh identity and verified flatness; audit both source identities",
    "retry": "one entry send maximum; no blind retry; retain all previous failures and halts in audit",
}


def resume_code(journal: Journal, current: str, *, fresh_quotes: bool) -> str:
    """Permit only the documented unsubmitted predecessor, never a general bypass."""
    if not fresh_quotes or not journal.rows:
        return current
    if journal.rows[-1]["event"] != "checkpoint":
        raise ValueError("intact final checkpoint required for migration")
    saved = journal.rows[-1]["data"]
    previous = saved["code_identity"]
    if previous == current:
        return current
    if previous != LEGACY_CODE:
        raise ValueError("unknown predecessor code; migration prohibited")
    if (any(v.get("submission_utc") for row in journal.rows if row["event"] == "checkpoint"
            for v in row["data"].get("intents", {}).values())
        or any(row["event"] == "broker_response" for row in journal.rows)):
        raise ValueError("any submission evidence prohibits code migration")
    return previous


def rearm_unsubmitted(
    coordinator: Coordinator, snapshot: dict[str, Any], now: datetime, *, operator: str,
    new_code: str | None = None,
) -> None:
    """Audited local operator transition; authorizes this process, never a restart."""
    c = coordinator
    p = c.risk.configuration.policy
    if c.config.run_type != "SMOKE" or p is None or not operator.strip():
        raise ValueError("explicit operator and dedicated SMOKE configuration required")
    allowed = QUOTE_HALTS | ({"EXPLICIT_REARM_SMOKE_ABORTED"} if c.fresh_quotes else set())
    if (not c.intents or not c.halts or not set(c.halts) <= allowed
        or not set(c.risk.state.halts) <= {f"KILL:{reason}" for reason in allowed}
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
    if new_code is not None and new_code != c.code:
        if not c.fresh_quotes or c.code != LEGACY_CODE:
            raise ValueError("unsupported checkpoint migration")
        c._record("execution_code_migration", {"old_code_identity": c.code,
            "new_code_identity": new_code, "operator": operator,
            "received_utc": now.isoformat(), "plan_id": FRESH_PLAN["plan_id"],
            "verified_flat": True, "history_preserved": True})
        c.code = new_code
    prior_halts = c.halts.copy()
    c.risk.rearm(now, operator=operator, reason="explicit never-submitted quote-abort recovery")
    c._record("execution_rearm", {"operator": operator, "prior_halts": prior_halts,
        "received_utc": now.isoformat(), "plan_id": (FRESH_PLAN if c.fresh_quotes else PLAN)["plan_id"],
        "script_sha256": sha256(Path(__file__)), "same_process_only": True})
    c.halts.clear()
    # This runtime flag normally rejects restart arming. The explicit audited
    # operator transition permits only this same process to arm; it is not saved.
    c.recovered = False
    c.persist()


def run(root: Path, config_path: Path, run_id: str, operator: str, *, fresh_quotes: bool = False) -> dict[str, Any]:
    versioned(run_id)
    if not run_id.replace("_", "").isalnum():
        raise ValueError("path-safe run identity required")
    config = load_demo_config(config_path)
    directory = root / "results/demo/runs" / run_id
    if directory.exists():
        raise ValueError("immutable run already exists")
    plan = FRESH_PLAN if fresh_quotes else PLAN
    freeze(directory / "plan.json", plan)
    report = readiness(root, config)
    freeze(directory / "readiness.json", report)
    if report["blocking_reasons"] or config.run_type != "SMOKE":
        raise ValueError("configured mechanical SMOKE only; strategy stays blocked")
    terminal = load_shadow_config(root / str(config.terminal_config_path))
    risk = load_configuration(root / str(config.risk_config_path))
    source = code_identity(root)
    freeze(directory / "provenance.json", {"code_identity": source,
        "script_sha256": sha256(Path(__file__)), "execution_identity": config.identity,
        "risk_identity": risk.identity, "operator": operator, "plan_id": plan["plan_id"]})
    journal = Journal(account_directory(root, terminal))
    broker = DemoProcessBroker(config, terminal)
    c: Coordinator | None = None
    started = time.monotonic()
    failure: str | None = None
    failure_reason: str | None = None
    phase = "RESTORE"
    try:
        prior_code = resume_code(journal, source, fresh_quotes=fresh_quotes)
        c = Coordinator(config, terminal, risk, broker, journal, run_id, prior_code,
                        **({"fresh_quotes": True} if fresh_quotes else {}))
        phase = "CONNECT"
        broker.connect()
        since = c.opened_utc or datetime.now(UTC) - timedelta(seconds=2)
        snapshot = broker.snapshot(since)
        now = datetime.fromisoformat(snapshot["received_utc"])
        phase = "OPERATOR_REARM"
        rearm_unsubmitted(c, snapshot, now, operator=operator,
                          new_code=source if fresh_quotes else None)
        c.initialize(snapshot, now, arm=True)
        p = risk.policy
        assert p is not None
        if time.monotonic() - started >= float(config.max_duration_seconds or 0) - float(config.cleanup_seconds or 0):
            raise ValueError("rearming consumed entry time; preserve cleanup capacity")
        target = float(config.quantity_lots or 0) * (1 if config.side == "BUY" else -1)
        request = RiskRequest(f"{run_id}:entry", p.expected_portfolio_id, "SMOKE_V001",
            now, now, target, {"EXECUTION_SMOKE": target}, p.sizing_method)
        phase = "PROPOSE"
        intent = c.propose(request, snapshot, now,
            expires_utc=now + timedelta(seconds=float(config.intent_ttl_seconds or 0)))
        if intent["state"] == "RISK_APPROVED":
            phase = "SUBMIT"
            fresh = broker.snapshot(since)
            c.submit(request.intent_id, fresh, datetime.fromisoformat(fresh["received_utc"]))
        phase = "HOLD"
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
        failure_reason = next((code for code in VALIDATION_CODES.values()
                               if str(exc).endswith(": " + code)), "UNCLASSIFIED")
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
        "failure_reason": failure_reason, "last_phase": phase,
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
    parser.add_argument("--fresh-quotes", action="store_true",
                        help="explicit bounded Stage16 quote revalidation and guarded legacy migration")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = run(root, args.demo_config, args.run_id, args.operator, fresh_quotes=args.fresh_quotes)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "VERIFIED_DEMO_LIFECYCLE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
