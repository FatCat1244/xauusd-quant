"""Versioned readiness, bounded smoke/precheck/recovery; no unqualified strategy run."""
from __future__ import annotations

import json
import re
import time
from dataclasses import fields
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ..alpha_portfolio.plan import freeze
from ..execution.config import content_hash
from ..execution.readiness import sha256
from ..execution.runs import process_memory
from ..risk.contracts import AccountSpec, InstrumentSpec, RiskRequest
from ..risk.policy import RiskPolicy, load_configuration
from ..risk.runs import evidence
from ..shadow.config import load_shadow_config
from ..shadow.runs import code_identity, read_frozen
from ..utils.config import Config
from .config import DemoConfig, load_demo_config
from .coordinator import Coordinator, settings
from .journal import Journal
from .worker import DemoProcessBroker

PLAN: dict[str, Any] = {
    "plan_id": "DEMO_VALIDATION_PLAN_V001", "stage": 18,
    "types": {"SMOKE": "one configured mechanical entry then verified closure; no alpha claim",
              "STRATEGY": "blocked without eligible frozen portfolio, live compatibility/equality and supplied risk"},
    "offline": "deterministic fake broker; known prices/deals/cash; failure paths and isolated guard mutations",
    "native": "explicit identity, DEMO enum, dedicated initially flat account, supplied limits and units",
    "budget": "<=300 seconds, one smoke entry send, <=3 cleanup requests, <=3 reconciliation polls per phase",
    "retry": "no entry resubmission; unknown/ambiguous outcomes halt; only verified terminal partial close permits remainder close",
    "request": "market FOK/IOC, REQUEST/INSTANT/MARKET; PROCESS_EXIT horizon policy only",
    "risk": "Stage16 authority, durable reservations, daily loss/HWM persist, separate no-model smoke policy",
    "revalidation": "any economic quote/account/book change abstains; no price chasing or automatic filling alternatives",
    "ownership": "flat baseline plus durable request and unique order/history/deal/position chain; magic/comment insufficient alone",
    "accounting": "actual deal profit/commission/swap/fee; balance reconciliation; known external balance flows excluded from trading gain",
    "acceptance": "demonstrate identities, request semantics, lifecycle, known fills/closure/accounting; no profitability selection",
    "history": "new runtime evidence; prior Stage17 native capture empty, strategy evidence blocked; reserved historical data untouched",
    "stop": "Stage18 only, no real account, deployment or automatic Stage19",
}


def prerequisites(root: Path) -> dict[str, Any]:
    upstream = evidence(root)
    names = ["results/shadow/NATIVE_VERIFICATION_V001.json",
        "results/shadow/runs/EXNESS_CAPTURE_V001/summary.json",
        "results/shadow/runs/EXNESS_EQUALITY_V001/comparison.json",
        "results/shadow/FINAL_VERIFICATION_V001.json"]
    sources = {name: sha256(root / name) if (root / name).is_file() else None for name in names}
    comparison = read_frozen(root / names[2]) if (root / names[2]).is_file() else {}
    return {"stage15_16": upstream, "stage17_sources": sources,
            "eligible_alphas": upstream["eligible_alphas"],
            "full_pipeline_equality_verified": comparison.get("full_pipeline_equality") is True,
            "actual_stage17_compared_observations": comparison.get("observations", {}),
            "feed_transfer_verified": False, "strategy_native_binding_implemented": False,
            "historical_studies_reproduced": False}


def readiness(root: Path, config: DemoConfig) -> dict[str, Any]:
    risk = load_configuration(root / (config.risk_config_path or "config/risk.yaml"))
    missing = {"policy": [f.name for f in fields(RiskPolicy)] if risk.policy is None else [],
        "account": [f.name for f in fields(AccountSpec)] if risk.account is None else [],
        "instrument": [f.name for f in fields(InstrumentSpec)] if risk.instrument is None else []}
    reasons = []
    if not config.configured:
        reasons.append("EXECUTION_UNCONFIGURED_OR_NOT_DEDICATED")
    if not risk.configured:
        reasons.append("RISK_UNCONFIGURED")
    if config.configured and risk.configured:
        try:
            settings(config, risk, offline_synthetic=False)
        except ValueError:
            reasons.append("INCOMPATIBLE_OR_UNVERIFIED_SUPPLIED_SETTINGS")
    if config.terminal_config_path is None:
        reasons.append("EXPLICIT_TERMINAL_REFERENCE_MISSING")
    else:
        terminal = load_shadow_config(root / config.terminal_config_path)
        if not terminal.configured:
            reasons.append("TERMINAL_IDENTITY_INCOMPLETE")
    history = prerequisites(root)
    if config.run_type == "STRATEGY":
        if not history["eligible_alphas"]:
            reasons.append("NO_ELIGIBLE_ALPHAS")
        if not history["full_pipeline_equality_verified"]:
            reasons.append("FULL_LIVE_PIPELINE_EQUALITY_UNVERIFIED")
        reasons += ["FEED_TRANSFER_UNASSESSED", "NATIVE_STRATEGY_BINDING_BLOCKED"]
    return {"stage": 18, "status": "BLOCKED" if reasons else "CONFIGURED_PREFLIGHT_REQUIRED",
        "execution_configuration": config.public(), "missing_execution_fields": config.missing,
        "missing_risk_fields": missing, "blocking_reasons": reasons, "evidence": history,
        "native_preflight_passed": False, "broker_precheck_completed": False,
        "demo_submissions": 0, "verified_fills": 0, "verified_closures": 0,
        "strategy_active": False, "real_money_readiness": False, "stage19_started": False}


def account_directory(root: Path, terminal: Any) -> Path:
    # Same account in another terminal or differently tuned polling config shares a lock.
    key = content_hash({"login": terminal.expected_login, "server": terminal.expected_server,
                        "company": terminal.expected_company})
    return root / "runtime/demo" / key


def native_run(root: Path, config: DemoConfig, directory: Path, command: str, run_id: str) -> dict[str, Any]:
    report = readiness(root, config)
    if command == "demo-strategy":
        if not report["evidence"]["eligible_alphas"]:
            report["blocking_reasons"].append("NO_ELIGIBLE_ALPHAS")
        report["blocking_reasons"] += ["FULL_LIVE_PIPELINE_EQUALITY_UNVERIFIED",
                                       "FEED_TRANSFER_UNASSESSED", "NATIVE_STRATEGY_BINDING_BLOCKED"]
        report["blocking_reasons"] = list(dict.fromkeys(report["blocking_reasons"]))
        report["status"] = "BLOCKED"
    freeze(directory / "readiness.json", report)
    if report["blocking_reasons"]:
        freeze(directory / "verdict.json", report)
        return report
    if command == "demo-strategy" or config.run_type != "SMOKE":
        report["blocking_reasons"] = ["STRATEGY_BINDING_AND_SCIENTIFIC_EVIDENCE_BLOCKED"]
        report["status"] = "BLOCKED"
        freeze(directory / "verdict.json", report)
        return report
    terminal = load_shadow_config(root / str(config.terminal_config_path))
    risk = load_configuration(root / str(config.risk_config_path))
    source = code_identity(root)
    specification = {"specification_id": config.specification_id,
        "configuration": config.public(), "configuration_sha256": config.identity,
        "risk_configuration_sha256": risk.identity, "code_identity": source,
        "classification": "frozen execution lifecycle specification, not an alpha"}
    freeze(root / "results/demo/specifications" / f"{config.specification_id}.json", specification)
    freeze(directory / "specification.json", specification)
    journal = Journal(account_directory(root, terminal))
    broker = DemoProcessBroker(config, terminal)
    coordinator: Coordinator | None = None
    started = time.monotonic()
    since = datetime.now(UTC) - timedelta(seconds=2)
    native_connected = False
    failure: str | None = None
    try:
        coordinator = Coordinator(config, terminal, risk, broker, journal, run_id, source)
        broker.connect()
        native_connected = True
        snapshot = broker.snapshot(coordinator.opened_utc or since)
        now = datetime.fromisoformat(snapshot["received_utc"])
        coordinator.initialize(snapshot, now, arm=False)
        if command not in ("demo-reconcile", "demo-recover-close"):
            if time.monotonic() - started >= float(config.max_duration_seconds or 0) - float(config.cleanup_seconds or 0):
                raise ValueError("initialization consumed entry budget; preserve cleanup capacity")
            p = risk.policy
            assert p is not None
            if p.warmup_snapshots > 16:
                raise ValueError("warmup exceeds bounded native snapshot budget")
            for _ in range(p.warmup_snapshots):
                if coordinator.risk.state.state == "READY":
                    break
                snapshot = broker.snapshot(coordinator.opened_utc or since)
                now = datetime.fromisoformat(snapshot["received_utc"])
                coordinator.initialize(snapshot, now, arm=False)
            coordinator.initialize(snapshot, now, arm=True)
        if command not in ("demo-reconcile", "demo-recover-close"):
            p = risk.policy
            assert p is not None
            target = float(config.quantity_lots or 0) * (1 if config.side == "BUY" else -1)
            request = RiskRequest(f"{run_id}:entry", p.expected_portfolio_id, "SMOKE_V001", now, now,
                                  target, {"EXECUTION_SMOKE": target}, p.sizing_method)
            intent = coordinator.propose(request, snapshot, now,
                expires_utc=now + timedelta(seconds=float(config.intent_ttl_seconds or 0)))
            if intent["state"] == "RISK_APPROVED":
                fresh = broker.snapshot(coordinator.opened_utc or since)
                coordinator.submit(request.intent_id, fresh, datetime.fromisoformat(fresh["received_utc"]),
                                   precheck_only=command == "demo-precheck")
        # Bounded history lag recovery, never resubmission of an entry.
        for attempt in range(3):
            if time.monotonic() - started >= float(config.max_duration_seconds or 0):
                break
            snapshot = broker.snapshot(coordinator.opened_utc or since)
            result = coordinator.reconcile(snapshot, datetime.fromisoformat(snapshot["received_utc"]))
            if result.get("reconciled"):
                break
            if attempt < 2:
                time.sleep(1)
        if command == "demo-smoke" and coordinator.last_snapshot and coordinator.last_snapshot["positions"]:
            remaining = float(config.max_duration_seconds or 0) - float(config.cleanup_seconds or 0) - (time.monotonic() - started)
            hold_until = time.monotonic() + max(0, min(float(config.hold_seconds or 0), remaining))
            while time.monotonic() < hold_until:
                time.sleep(min(1, hold_until - time.monotonic()))
                snapshot = broker.snapshot(coordinator.opened_utc or since)
                state = coordinator.reconcile(snapshot, datetime.fromisoformat(snapshot["received_utc"]))
                if not state.get("reconciled") or coordinator.risk.state.halts or not snapshot["positions"]:
                    break
        if command in ("demo-smoke", "demo-recover-close") and config.shutdown_policy == "CLOSE_OWNED":
            for sequence in range(int(config.cleanup_request_budget or 0)):
                if time.monotonic() - started >= float(config.max_duration_seconds or 0):
                    break
                snapshot = broker.snapshot(coordinator.opened_utc or since)
                now = datetime.fromisoformat(snapshot["received_utc"])
                reconciliation = coordinator.reconcile(snapshot, now)
                if not snapshot["positions"] or not reconciliation.get("reconciled"):
                    break
                p = risk.policy
                assert p is not None
                close = RiskRequest(f"{run_id}:close:{sequence}", p.expected_portfolio_id,
                    "SMOKE_V001", now, now, 0, {"EXECUTION_SMOKE": 0}, p.sizing_method)
                intent = coordinator.propose(close, snapshot, now,
                    expires_utc=now + timedelta(seconds=float(config.intent_ttl_seconds or 0)))
                if intent["state"] != "RISK_APPROVED":
                    break
                fresh = broker.snapshot(coordinator.opened_utc or since)
                outcome = coordinator.submit(close.intent_id, fresh, datetime.fromisoformat(fresh["received_utc"]))
                for attempt in range(3):
                    if time.monotonic() - started >= float(config.max_duration_seconds or 0):
                        break
                    snapshot = broker.snapshot(coordinator.opened_utc or since)
                    state = coordinator.reconcile(snapshot, datetime.fromisoformat(snapshot["received_utc"]))
                    if state.get("reconciled"):
                        break
                    if attempt < 2:
                        time.sleep(1)
                # Only a terminal partial close with verified remaining quantity
                # may create another reduction. Rejections/unknowns never loop.
                if outcome.get("response", {}) is None or outcome.get("response", {}).get("retcode") != 10010:
                    break
        coordinator.armed = False
        coordinator.persist()
    except (Exception, KeyboardInterrupt) as exc:
        failure = "INTERRUPTED_RECONCILIATION_REQUIRED" if isinstance(exc, KeyboardInterrupt) else "IDENTITY_OR_READINESS_FAILURE" if isinstance(exc, (ValueError, RuntimeError)) else "NATIVE_OR_STORAGE_FAILURE"
        if coordinator is not None:
            try:
                coordinator.halt(failure, datetime.now(UTC))
            except Exception:
                failure = "FAILURE_WITH_UNPERSISTED_HALT; RETAIN_JOURNAL_AND_RECONCILE"
    finally:
        try:
            broker.shutdown()
        finally:
            journal.close()
    verdict = coordinator.summary() if coordinator is not None else {"submissions": 0, "verified_deals": 0,
        "verified_flat": False, "unresolved_intents": ["STATE_RESTORATION_OR_INITIALIZATION_FAILED"]}
    verdict.update({"stage": 18, "status": "HALTED_OR_UNRESOLVED" if failure or not verdict.get("verified_flat") else "COMPLETED",
        "failure": failure, "native_connectivity_completed": native_connected, "code_identity": source,
        "classification": "demo execution lifecycle only; no strategy validity", "wall_seconds": time.monotonic() - started,
        "resources": process_memory(), "strategy_active": False, "broker_calls_bounded_but_outcomes_not_guaranteed": True})
    if command == "demo-smoke" and not verdict.get("verified_entry_deals") and not failure:
        verdict["status"] = "NO_DEMO_LIFECYCLE_DEMONSTRATED"
    freeze(directory / "verdict.json", verdict)
    return verdict


def cli_command(config: Config, args: Any) -> int:
    root = config.project_root
    execution = load_demo_config(args.demo_config or root / "config/demo.yaml")
    if not re.fullmatch(r"[A-Z][A-Z0-9_]+_V\d{3}", args.run_id):
        raise ValueError("versioned path-safe identity required")
    directory = root / "results/demo/runs" / args.run_id
    if directory.exists():
        raise ValueError("immutable run exists; bump version")
    freeze(directory / "plan.json", PLAN)
    if args.command in ("demo-plan", "demo-readiness"):
        report = readiness(root, execution)
        freeze(directory / "readiness.json", report)
        print(json.dumps({k: report[k] for k in ("status", "blocking_reasons", "missing_execution_fields", "missing_risk_fields")}, indent=2))
    elif args.command == "demo-preflight":
        path = args.terminal_config or (root / execution.terminal_config_path if execution.terminal_config_path else None)
        if path is None:
            raise ValueError("explicit local terminal configuration required for read-only preflight")
        terminal = load_shadow_config(path)
        broker = DemoProcessBroker(execution, terminal)
        report = {"native_connectivity_completed": False, "demo_submissions": 0, "broker_precheck_completed": False}
        try:
            metadata = broker.connect()
            snapshot = broker.snapshot(datetime.now(UTC) - timedelta(seconds=2))
            freeze(directory / "local_account_snapshot.json", snapshot)
            quote = snapshot["quote"]
            age = datetime.fromisoformat(snapshot["received_utc"]).timestamp() - quote["time_msc"] / 1000
            report |= {"native_connectivity_completed": True, "demo_mode_verified": metadata["demo_mode_verified"],
                "execution_permissions_verified": snapshot["permissions"],
                "disabled_permissions": [k for k, v in snapshot["permission_checks"].items() if not v],
                "account_mode": {0: "NETTING", 2: "HEDGING"}.get(snapshot["account"]["margin_mode"], "UNSUPPORTED"),
                "account_has_positions": bool(snapshot["positions"]), "account_has_orders": bool(snapshot["orders"]),
                "fresh_quote": 0 <= age <= terminal.max_quote_age_seconds,
                "code_identity": code_identity(root), "status": "READ_ONLY_PREFLIGHT_ONLY"}
        except Exception:
            report |= {"status": "BLOCKED", "reason": "IDENTITY_NATIVE_OR_METADATA_FAILURE"}
        finally:
            broker.shutdown()
        freeze(directory / "preflight.json", report)
        print(json.dumps(report, indent=2))
    else:
        report = native_run(root, execution, directory, args.command, args.run_id)
        print(json.dumps(report, indent=2))
    return 1 if report.get("status") in ("BLOCKED", "HALTED_OR_UNRESOLVED", "NO_DEMO_LIFECYCLE_DEMONSTRATED") else 0
