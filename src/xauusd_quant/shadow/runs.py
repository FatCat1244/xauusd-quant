"""Bounded capture/replay CLI and immutable local artifacts; sanitized summaries."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import re
import time
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, TextIO

from ..alpha_portfolio.plan import freeze
from ..execution.config import content_hash
from ..execution.io import _json_safe
from ..execution.readiness import sha256
from ..execution.runs import process_memory
from ..features.factory_config import load_features_config
from ..risk.policy import load_configuration
from ..risk.runs import evidence
from ..utils.config import Config
from .adapter import IdentityFailure, ReadFailure, ReadOnlyFeed
from .config import ShadowConfig, load_shadow_config
from .ingestion import ContinuityFailure, Cursor, Tick
from .pipeline import ShadowPipeline
from .worker import MT5ProcessFeed

PLAN: dict[str, Any] = {
    "plan_id": "SHADOW_VALIDATION_PLAN_V001", "stage": 17,
    "universe": "eligible frozen Stage15 alphas only; actual empty universe remains inactive",
    "native": "explicit authenticated Exness demo terminal, official read-only MT5 functions",
    "time": "UTC [open,close), source milliseconds, receipt sequence and publication clock separately",
    "capture_budget": "<=300 seconds, <=100000 accepted events, bounded batches/buffers, sequential",
    "acceptance": "identity/order isolation, occurrence continuity, prefix/chunk equality, persistent halts",
    "comparison": "same accepted observations and recorded publication clocks, float32 rtol=1e-5 atol=1e-7",
    "gaps": "no flat bars; ambiguity/gap/late corrections halt new intentions; no automatic rearm",
    "backfill": "history warms features only; no retrospective shadow trade decisions or fills",
    "economics": "Stage12 local fills only; missing broker costs/margin/feed compatibility block conclusions",
    "evidence": "offline software tests separate from native connectivity, actual capture and full model equality",
    "synthetic_study": {"capture": "36 fixed ticks, 180 synthetic seconds, replay same publication sequence",
        "governed": "26 step-trend ticks; fixed ret_1*0.2 mapping, hand-checked fills and cash",
        "controls": "same fixed pipeline with flat prices and missing uncertainty; no profitability selection",
        "max_wall_seconds": 120, "max_private_bytes": 1073741824},
    "stop": "no broker orders, deployment, strategy tuning or automatic Stage18",
}


def code_identity(root: Path) -> str:
    files = {str(p.relative_to(root)): sha256(p) for p in sorted((root / "src").rglob("*.py"))}
    return content_hash(files)


def freeze_runtime(path: Path, body: dict[str, Any]) -> Path:
    used = sum(p.stat().st_size for p in path.parent.iterdir() if p.is_file())
    estimate = len(json.dumps(_json_safe(body), indent=2, allow_nan=False).encode("utf-8")) + 1024
    if used + estimate > 256 * 1024 * 1024:
        raise OSError("bounded checkpoint storage budget exhausted")
    return freeze(path, body)


class Store:
    """Append-only bounded records. Failed storage is fatal, never silently discarded."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory
        self.stream: TextIO = (directory / "records.jsonl").open("x", encoding="utf-8", newline="\n")
        self.counts: Counter[str] = Counter()
        self.bytes_written = 0
        self.hasher = hashlib.sha256()

    def __call__(self, table: str, row: dict[str, Any]) -> None:
        line = json.dumps({"table": table, "row": _json_safe(row)}, allow_nan=False) + "\n"
        self.bytes_written += len(line.encode("utf-8"))
        if self.bytes_written > 64 * 1024 * 1024:
            raise OSError("bounded record storage budget exhausted")
        self.stream.write(line)
        self.hasher.update(line.encode("utf-8"))
        self.counts[table] += 1

    def flush(self) -> None:
        import os
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self) -> None:
        self.stream.close()


def readiness(root: Path, config: ShadowConfig) -> dict[str, Any]:
    upstream = evidence(root)
    risk = load_configuration(root / "config/risk.yaml")
    required = [n for n in ("terminal_path", "expected_login", "expected_server", "expected_company", "symbol")
                if getattr(config, n) is None]
    return {"stage": 17, "status": "BLOCKED", "strategy_active": False,
            "configuration": config.public(), "missing_terminal_fields": required,
            "native_package_installed": importlib.util.find_spec("MetaTrader5") is not None,
            "python_version": platform.python_version(), "platform": platform.system(),
            "upstream": upstream, "risk_configured": risk.configured,
            "alpha_status": "NO_ELIGIBLE_ALPHAS" if not upstream["eligible_alphas"] else "LIVE_BINDING_AUDIT_REQUIRED",
            "native_connectivity_completed": False, "live_observation_completed": False,
            "full_feature_model_equality_completed": False, "stage18_ready": False,
            "remaining": ["explicit local demo terminal identity", "optional native package installation",
                "eligible frozen policy/model/feature artifacts", "Exness feed-transfer validation",
                "supplied account/instrument/risk/cost terms", "compatible context and health diagnostics",
                "bounded recorded-feed equality and prospective shadow evaluation", "separate Stage18 authorization"],
            "past_results_reproduced": False, "frozen_artifacts": frozen_inventory(root)}


def frozen_inventory(root: Path) -> dict[str, Any]:
    from .models import forecast_units
    entries = []
    for namespace, pattern in (("ml_research", "MODEL_SPEC_*.json"), ("ensemble_research", "ENSEMBLE_SPEC_*.json")):
        for tf in ("5m", "15m"):
            for path in sorted((root / "results" / namespace / tf / "frozen").glob(pattern)):
                body = json.loads(path.read_text(encoding="utf-8"))
                features = body.get("features", [])
                if namespace == "ensemble_research":
                    features = sorted({n for c in body.get("constituents", []) for n in c["features"]})
                entries.append({"identity": body["spec_id"], "spec_sha256": sha256(path),
                    "kind": namespace, "timeframe": tf, "target": body["target"],
                    "output_units": forecast_units(body), "feature_count": len(features),
                    "feature_names": features, "binary_loaded": False,
                    "live_transfer_verified": False})
                if len(entries) > 256:
                    raise ValueError("frozen inventory exceeds bounded Stage17 universe")
    return {"entries": entries, "models": sum(e["kind"] == "ml_research" for e in entries),
            "ensembles": sum(e["kind"] == "ensemble_research" for e in entries),
            "scope": "saved frozen spec metadata and hashes only; no binaries or historic market rows read"}


def checkpoint(config: ShadowConfig, cursor: Cursor, pipeline: ShadowPipeline,
               code: str, sink: Store | None = None) -> dict[str, Any]:
    body = {"configuration_identity": config.identity, "code_identity": code,
            "cursor": asdict(cursor), "pipeline": pipeline.state()}
    if sink is not None:
        body["audit_prefix"] = {"bytes": sink.bytes_written, "sha256": sink.hasher.hexdigest()}
    return {"schema": "SHADOW_CHECKPOINT_V001", "body": body,
            "content_sha256": content_hash(body)}


def restore_checkpoint(config: ShadowConfig, pipeline: ShadowPipeline, code: str,
                       path: Path) -> Cursor:
    envelope = read_frozen(path)
    body = envelope["body"]
    if (envelope.get("schema") != "SHADOW_CHECKPOINT_V001"
        or envelope.get("content_sha256") != content_hash(body)
        or body["configuration_identity"] != config.identity
        or body["code_identity"] != code):
        raise ValueError("invalid or incompatible shadow checkpoint; do not resume")
    cursor = Cursor.restore(body["cursor"], config.batch_size)
    pipeline.restore(body["pipeline"])
    if pipeline.bars.last_sequence != cursor.sequence:
        raise ValueError("cursor and pipeline sequence do not reconcile")
    if "audit_prefix" in body:
        prefix = body["audit_prefix"]
        size = prefix["bytes"]
        if type(size) is not int or not 0 <= size <= 64 * 1024 * 1024:
            raise ValueError("invalid audit prefix size")
        digest = hashlib.sha256()
        with (path.parent / "records.jsonl").open("rb") as stream:
            left = size
            while left:
                chunk = stream.read(min(left, 1 << 20))
                if not chunk:
                    raise ValueError("checkpoint audit prefix truncated")
                digest.update(chunk)
                left -= len(chunk)
        if digest.hexdigest() != prefix["sha256"]:
            raise ValueError("checkpoint audit prefix changed")
    return cursor


def read_frozen(path: Path) -> dict[str, Any]:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(envelope, dict) or "body" not in envelope or envelope.get("content_sha256") != content_hash(envelope["body"]):
        raise ValueError("invalid immutable shadow artifact")
    return dict(envelope["body"])


def capture(root: Path, config: ShadowConfig, directory: Path, *,
            adapter: ReadOnlyFeed | None = None, resume: Path | None = None,
            wall: Callable[[], datetime] | None = None,
            monotonic: Callable[[], float] = time.monotonic,
            sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    if not config.configured:
        raise IdentityFailure("native capture requires an explicit local demo configuration")
    native_requested = adapter is None
    adapter = adapter or MT5ProcessFeed(config)
    wall = wall or (lambda: datetime.now(UTC))
    started = wall()
    deadline = monotonic() + config.duration_seconds
    source_code = code_identity(root)
    sink = Store(directory)
    pipeline = ShadowPipeline(config, load_features_config(root=root), sink)
    cursor = Cursor(int((started - timedelta(seconds=config.backfill_seconds)).timestamp()))
    try:
        if resume is not None:
            cursor = restore_checkpoint(config, pipeline, source_code, resume)
            pipeline.halt("RESTART_CONTINUITY_UNVERIFIED", started)
        freeze(directory / "plan.json", PLAN)
        sink.flush()
        freeze(directory / "initial_state.json", checkpoint(config, cursor, pipeline, source_code, sink))
    except Exception:
        sink.close()
        adapter.shutdown()
        raise
    count, polls, retries = 0, 0, 0
    latest_msc: int | None = None
    reconnects = 0
    status, reason = "COMPLETED", "DURATION_OR_EVENT_LIMIT"
    connected = False
    live_ticks = 0
    fresh_ticks = 0
    peak_memory = process_memory()
    initial_bars = pipeline.counts["bars"]
    observation_start: int | None = None
    observation_end: int | None = None
    before = monotonic()
    try:
        sink("preflight", adapter.connect())
        connected = True
        while monotonic() < deadline and count < config.max_events:
            try:
                quote = adapter.latest()
                latest_msc = quote.get("time_msc") if quote else None
                if latest_msc is None or not 0 <= wall().timestamp() - latest_msc / 1000 <= config.max_quote_age_seconds:
                    sink("health", {"reason": "NO_FRESH_LIVE_QUOTE", "receipt_utc": wall().isoformat()})
                rows = adapter.ticks(cursor.retrieval_start, config.batch_size)
                receipt = wall()
                polls += 1
                accepted = cursor.ingest(rows, receipt, batch_limit=config.batch_size,
                    live_start=started, max_future_seconds=config.max_future_seconds)
                if len(accepted) > config.max_events - count:
                    sink("health", {"reason": "EVENT_BUDGET_BATCH_NOT_PROCESSED",
                        "retrieved_ticks": len(accepted), "remaining_budget": config.max_events - count,
                        "batch_sha256": content_hash(rows), "receipt_utc": receipt.isoformat()})
                    raise ContinuityFailure("remaining event budget cannot accommodate retrieved batch")
                if accepted:
                    processing_started = monotonic()
                    sink("retrieval", {"received_utc": receipt.isoformat(), "retrieved": len(rows),
                        "new": len(accepted), "source_batch_sha256": content_hash(rows)})
                    for tick in accepted:
                        sink("ticks", tick.resolved())
                    clocks = pipeline.consume(accepted, wall)
                    memory = process_memory()
                    if int(memory.get("private_bytes") or 0) > int(peak_memory.get("private_bytes") or 0):
                        peak_memory = memory
                    if int(memory.get("private_bytes") or 0) > 1073741824:
                        raise ContinuityFailure("declared private-memory budget exceeded")
                    sink("processing", {"receipt_utc": receipt.isoformat(),
                        "wall_seconds": monotonic() - processing_started, "batch_ticks": len(accepted),
                        "source_to_receipt_seconds": receipt.timestamp() - accepted[-1].time_msc / 1000,
                        "uncompleted_tick_buffers": {tf: len(v) for tf, v in pipeline.bars.buffers.items()},
                        "synchronized_transport_latency": "unverified"})
                    sink("batch_commit", {"last_sequence": accepted[-1].sequence,
                        "count": len(accepted), "publication_clocks": clocks,
                        "receipt_utc": receipt.isoformat(), "content_sha256": content_hash(
                            {"ticks": [t.resolved() for t in accepted], "publication_clocks": clocks})})
                    sink.flush()
                    count += len(accepted)
                    live_ticks += sum(not t.backfill for t in accepted)
                    fresh_ticks += sum(not t.backfill and 0 <= (t.received_utc - t.event_utc).total_seconds() <= config.max_quote_age_seconds for t in accepted)
                    observation_start = observation_start or accepted[0].time_msc
                    observation_end = accepted[-1].time_msc
                    freeze_runtime(directory / f"checkpoint_{polls:06d}.json", checkpoint(config, cursor, pipeline, source_code, sink))
                else:
                    sink("health", {"reason": "EMPTY_OR_REPEATED_BATCH", "receipt_utc": receipt.isoformat()})
                retries = 0
            except IdentityFailure:
                raise
            except ReadFailure:
                pipeline.halt("DISCONNECTION_CONTINUITY_UNVERIFIED", wall())
                if retries >= config.reconnect_attempts:
                    raise
                retries += 1
                reconnects += 1
                adapter.shutdown()
                sleep(min(2 ** (retries - 1), 4))
                sink("preflight", adapter.connect())
                # Successful reconnect verifies identity; persistent pipeline/risk halts remain.
            remaining = deadline - monotonic()
            if remaining > 0 and count < config.max_events:
                sleep(min(config.poll_seconds, remaining))
    except IdentityFailure:
        status, reason = "HALTED", "IDENTITY_VERIFICATION_FAILURE"
    except ReadFailure:
        status, reason = "HALTED", "READ_API_FAILURE"
    except ContinuityFailure:
        status, reason = "HALTED", "INGESTION_OR_STREAM_CONTINUITY_FAILURE"
    except ValueError:
        status, reason = "HALTED", "INPUT_OR_CHECKPOINT_VALIDATION_FAILURE"
    except OSError:
        status, reason = "HALTED", "STORAGE_FAILURE"
    except (RuntimeError, TypeError):
        status, reason = "HALTED", "PIPELINE_RUNTIME_FAILURE"
        # Do not log native exception text or claim the last incomplete batch committed.
    except KeyboardInterrupt:
        status, reason = "INTERRUPTED", "USER_INTERRUPT"
    finally:
        adapter.shutdown()
        sink.close()
    summary = {"status": status, "reason": reason, "native_connectivity_completed": connected,
        "accepted_committed_ticks": count, "retrieval_polls": polls, "reconnections": reconnects,
        "coverage_start_msc": observation_start, "coverage_end_msc": observation_end,
        "bars": pipeline.counts["bars"] - initial_bars, "shadow_strategy_active": False,
        "live_ticks": live_ticks if native_requested else 0, "nonbackfill_ticks": live_ticks,
        "fresh_live_ticks": fresh_ticks if native_requested else 0, "backfill_ticks": count - live_ticks,
        "freshness_observation": "event/receipt lag diagnostic; synchronized transport latency unverified",
        "code_identity": source_code, "wall_seconds": monotonic() - before,
        "memory": peak_memory,
        "records": dict(sink.counts), "record_bytes": sink.bytes_written,
        "halts": sorted(pipeline.halts | pipeline.bars.halts), "blocked_reasons": list(pipeline.blocked),
        "broker_orders": 0, "local_shadow_fills": 0, "stage18_ready": False,
        "account_state": "broker observations separate; no strategy simulator without supplied terms"}
    summary["native_connectivity_completed"] = connected and native_requested
    summary["offline_adapter_connectivity_completed"] = connected and not native_requested
    summary["live_observation_completed"] = native_requested and fresh_ticks > 0
    summary["source"] = "exness_mt5" if native_requested else "synthetic_read_only_adapter"
    freeze(directory / "summary.json", summary)
    freeze(directory / "manifest.json", {"source": summary["source"], "code_identity": source_code,
        "plan_id": PLAN["plan_id"], "configuration": config.public(),
        "files": {p.name: sha256(p) for p in directory.iterdir() if p.is_file()},
        "broker_account_values": "never logged", "native_api": "official MT5 Python reads only",
        "ordering": "stable server occurrence order assumed; no unique broker tick IDs or proof of completeness"})
    return summary


def replay(root: Path, config: ShadowConfig, recorded: Path, directory: Path) -> dict[str, Any]:
    sink = Store(directory)
    pipeline = ShadowPipeline(config, load_features_config(root=root), sink)
    initial = read_frozen(recorded / "initial_state.json")
    manifest = read_frozen(recorded / "manifest.json")
    if sha256(recorded / "records.jsonl") != manifest["files"]["records.jsonl"]:
        sink.close()
        raise ValueError("recorded source manifest mismatch or interrupted modification")
    if initial["body"]["configuration_identity"] != config.identity or initial["content_sha256"] != content_hash(initial["body"]):
        sink.close()
        raise ValueError("recorded source/checkpoint identity incompatible")
    if initial["body"]["code_identity"] != code_identity(root):
        sink.close()
        raise ValueError("code changed since capture; freeze a new comparison version")
    pipeline.restore(initial["body"]["pipeline"])
    pending: list[Tick] = []
    batches = 0
    try:
        with (recorded / "records.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                entry = json.loads(line)
                if entry["table"] == "ticks":
                    pending.append(Tick.restore(entry["row"]))
                    if len(pending) > config.batch_size:
                        raise ValueError("recorded batch exceeds bounded replay size")
                elif entry["table"] == "batch_commit":
                    row = entry["row"]
                    if len(pending) != row["count"] or not pending or pending[-1].sequence != row["last_sequence"]:
                        raise ValueError("recorded batch commit does not reconcile")
                    if row.get("content_sha256") != content_hash({"ticks": [t.resolved() for t in pending],
                                                                   "publication_clocks": row["publication_clocks"]}):
                        raise ValueError("recorded batch content identity mismatch")
                    clocks = iter(row["publication_clocks"])
                    pipeline.consume(pending, recorded_clock(clocks))
                    if next(clocks, None) is not None:
                        raise ValueError("unused publication clock in replay")
                    pending = []
                    batches += 1
                elif entry["table"] == "health" and entry["row"].get("new_actions") == "halted":
                    health = entry["row"]
                    if health["reason"] not in pipeline.halts:
                        pipeline.halt(health["reason"], datetime.fromisoformat(health["receipt_utc"]))
        if pending:
            raise ValueError("interrupted uncommitted tick tail; cannot claim full equality")
        sink.flush()
    finally:
        sink.close()
    report = {"batches": batches, "ticks": pipeline.counts["ticks"], "bars": pipeline.counts["bars"],
        "source_records_sha256": sha256(recorded / "records.jsonl"),
        "code_identity": code_identity(root), "classification": "recorded-feed replay, not a new live observation",
        "predictions": "BLOCKED: no eligible live bindings", "portfolio_risk_equality": "UNAVAILABLE",
        "broker_execution": "unavailable"}
    freeze(directory / "summary.json", report)
    return report


def recorded_clock(clocks: Iterator[str]) -> Callable[[], datetime]:
    return lambda: datetime.fromisoformat(next(clocks))


def compare(recorded: Path, replayed: Path) -> dict[str, Any]:
    import numpy as np

    tables = {"bars", "features", "predictions", "uncertainty", "shadow_actions", "approved_shadow_actions", "risk_decisions", "fills"}

    def entries(path: Path) -> Any:
        with (path / "records.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                entry = json.loads(line)
                if entry["table"] in tables:
                    yield entry

    def equal(a: Any, b: Any, numeric: bool = False) -> bool:
        if isinstance(a, dict) and isinstance(b, dict):
            return set(a) == set(b) and all(equal(a[k], b[k], numeric) for k in a)
        if isinstance(a, list) and isinstance(b, list):
            return len(a) == len(b) and all(equal(x, y, numeric) for x, y in zip(a, b, strict=True))
        if numeric and isinstance(a, float) and isinstance(b, float):
            return bool(np.isclose(a, b, rtol=1e-5, atol=1e-7))
        return a == b

    import itertools
    counts: Counter[str] = Counter()
    mismatches = 0
    valid_predictions = 0
    for a, b in itertools.zip_longest(entries(recorded), entries(replayed)):
        if a is not None:
            counts[a["table"]] += 1
            valid_predictions += int(a["table"] == "predictions" and a["row"].get("value") is not None)
        numeric = a is not None and a["table"] in {"features", "predictions", "uncertainty"}
        mismatches += int(not equal(a, b, numeric))
    return {"equal": mismatches == 0, "mismatches": mismatches, "observations": dict(counts),
            "rtol": 1e-5, "atol": 1e-7, "integer_and_clock_comparison": "exact",
            "valid_predictions": valid_predictions,
            "full_model_equality": valid_predictions > 0 and mismatches == 0,
            "full_pipeline_equality": valid_predictions > 0 and counts["risk_decisions"] > 0 and mismatches == 0,
            "scope": "only observed tables; absent model/portfolio/risk stages remain untested"}


def cli_command(config: Config, args: Any) -> int:
    root = config.project_root
    shadow = load_shadow_config(args.shadow_config or root / "config/shadow.yaml")
    if args.command == "shadow-validate":
        report = readiness(root, shadow)
        if args.run_id:
            if not re.fullmatch(r"[A-Z][A-Z0-9_]+_V\d{3}", args.run_id):
                raise ValueError("versioned path-safe identity required")
            target = root / "results/shadow/runs" / args.run_id
            if target.exists():
                raise ValueError("immutable run exists; bump version")
            freeze(target / "readiness.json", report)
            freeze(target / "plan.json", PLAN)
        public = {k: report[k] for k in ("status", "strategy_active", "missing_terminal_fields",
            "native_package_installed", "python_version", "risk_configured", "alpha_status", "stage18_ready")}
        public["frozen_spec_counts"] = {k: report["frozen_artifacts"][k] for k in ("models", "ensembles")}
        print(json.dumps(public, indent=2))
        return 0
    if not re.fullmatch(r"[A-Z][A-Z0-9_]+_V\d{3}", args.run_id):
        raise ValueError("versioned path-safe run identity required")
    directory = root / "results/shadow/runs" / args.run_id
    if directory.exists():
        raise ValueError("immutable run exists; bump version")
    if args.command == "shadow-preflight":
        adapter = MT5ProcessFeed(shadow)
        try:
            report = adapter.connect()
        finally:
            adapter.shutdown()
        freeze(directory / "preflight.json", report)
    elif args.command in ("shadow-capture", "shadow-run"):
        # Both commands are currently capture+blocked diagnostics; no eligible universe.
        report = capture(root, shadow, directory, resume=args.resume)
    elif args.command == "shadow-replay":
        report = replay(root, shadow, args.recorded, directory)
    else:
        report = compare(args.recorded, args.replayed)
        freeze(directory / "comparison.json", report)
    print(json.dumps(report, indent=2))
    return 1 if report.get("status") == "HALTED" or report.get("equal") is False else 0
