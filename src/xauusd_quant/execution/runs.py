"""Offline Stage 12 CLI runs: freeze assumptions first, then consume bounded inputs.

Real economics require an input sidecar, artifact identities and chronology. Historical
diagnostics need the explicit flag and are never promoted. Reserved evaluation stays
with the existing final-test modules; this layer refuses new 2022+ execution access.
"""

from __future__ import annotations

import ctypes
import heapq
import json
import os
import re
import time
from collections.abc import Iterable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from typing import Any

import yaml

from ..utils.config import Config
from .config import ExecutionConfig, content_hash, load_execution_config
from .engine import BarClose, ExecutionEngine, Quote
from .io import (
    DiskRecorder,
    bar_closes,
    forecast_records,
    parse_utc,
    tick_quotes,
    validate_interval,
    write_new_json,
)
from .policy import Forecast
from .readiness import (
    READINESS_VERSION,
    code_identity,
    inventory,
    promotion_status,
    scientific_gates,
    sha256,
)


def process_memory() -> dict[str, int | str | None]:
    """Actual own-process counters, without another dependency or system-memory guarantee."""
    if os.name != "nt":
        return {
            "measurement": "Windows process counters unavailable",
            "peak_working_set_bytes": None,
        }

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            *[
                (name, ctypes.c_size_t)
                for name in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                    "PrivateUsage",
                )
            ],
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    if not psapi.GetProcessMemoryInfo(
        kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise OSError("GetProcessMemoryInfo failed")
    return {
        "measurement": "own Windows process; peak includes imports/preflight",
        "peak_working_set_bytes": counters.PeakWorkingSetSize,
        "working_set_bytes": counters.WorkingSetSize,
        "private_bytes": counters.PrivateUsage,
        "peak_commit_bytes": counters.PeakPagefileUsage,
    }


def new_run(config: Config, run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*_V[0-9]{3,}", run_id):
        raise ValueError("run id must be a safe versioned name ending _V001 (or later version)")
    path = config.project_root / "results/execution" / run_id
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ValueError(
            f"{run_id} already exists; outputs are immutable, bump the version"
        ) from exc
    return path


def _manifest(
    config: Config, execution: ExecutionConfig, kind: str, extras: dict[str, Any]
) -> dict[str, Any]:
    return {
        "stage": 12,
        "schema_version": "EXECUTION_RUN_V001",
        "kind": kind,
        "created_utc": datetime.now(UTC),
        "resolved_execution": execution.resolved(),
        "configuration_sha256": content_hash(execution.resolved()),
        "resolved_data_configuration": config.to_dict(),
        "data_configuration_sha256": content_hash(config.to_dict()),
        "dependencies": {name: version(name) for name in ("numpy", "polars", "pyarrow", "PyYAML")},
        "code": code_identity(config.project_root),
        "randomness": "none; deterministic scenario",
        "policy": {
            "id": execution.policy_id,
            "direction": "sign of expected log mid return",
            "quantity": "fixed lots; no optimization",
            "exit": "H subsequent observed nonempty bar closes",
            "position_mode": "single net position; overlaps rejected",
        },
        "cost_semantics": "quote-side cash flows already include spread; adverse slippage scenario; continuous funding",
        "benchmark": "midpoints at matched fill timestamps, counterfactual and non-executable",
        **extras,
    }


def synthetic_smoke(config: Config, run_id: str) -> dict[str, Any]:
    directory = new_run(config, run_id)
    c = ExecutionConfig()
    t = datetime(2021, 1, 4, 10, tzinfo=UTC)
    f = Forecast.from_bar(
        forecast_id="fixture-long", bar_open_utc=t, bar_index=0, value=0.001, config=c
    )
    events: list[Quote | BarClose] = [
        Quote(t, t.replace(tzinfo=None), 100, 102, 0),
        BarClose(t + timedelta(minutes=5), 0),
        Quote(
            t + timedelta(minutes=5, seconds=1),
            (t + timedelta(minutes=5, seconds=1)).replace(tzinfo=None),
            100,
            102,
            1,
        ),
        BarClose(t + timedelta(minutes=10), 1),
        Quote(
            t + timedelta(minutes=10, seconds=1),
            (t + timedelta(minutes=10, seconds=1)).replace(tzinfo=None),
            105,
            107,
            2,
        ),
    ]
    # The synthetic inter-quote interval deliberately spans five minutes; allow it.
    c = replace(c, max_gap_ms=600_000)
    write_new_json(
        directory / "manifest.json",
        _manifest(
            config,
            c,
            "synthetic_execution_smoke",
            {
                "data": "declared synthetic quotes; NOT historical evidence",
                "forecast_provenance": "fixed signed-return fixture",
                "readiness": "synthetic_only",
                "evaluation_history_classification": "synthetic",
                "all_evaluated_variants": [c.resolved()],
            },
        ),
    )
    start, cpu = time.perf_counter(), time.process_time()
    with DiskRecorder(directory) as record:
        engine = ExecutionEngine(c, [f], record)
        engine.consume(events)
        result = engine.finish(events[-1].timestamp_utc)
    result["resources"] = {
        "elapsed_seconds": time.perf_counter() - start,
        "cpu_seconds": time.process_time() - cpu,
        **process_memory(),
    }
    result["output"] = str(directory)
    write_new_json(directory / "summary.json", result)
    return result


def readiness_run(config: Config, execution: ExecutionConfig, run_id: str) -> dict[str, Any]:
    directory = new_run(config, run_id)
    facts = inventory(config)
    gates = {
        "data_provenance": facts["data_metadata_status"],
        "forecast_chronology": "failed",
        "null_evidence": "unknown",
        "untouched_evaluation": "failed",
        "broker_specification": "passed"
        if execution.specification_status == "verified_supplied"
        else "unknown",
        "forecast_units_and_horizon": "unknown",
        "evidence_identity": "unknown",
    }
    result = {
        "version": READINESS_VERSION,
        "gates": gates,
        "promotion_status": promotion_status(gates),
        "inventory": facts,
        "output": str(directory),
        "minimum_repair": "fold-local feature identities/counts, eligibility/null/correlation evidence; no globally filtered candidates; independently untouched outcomes",
    }
    write_new_json(directory / "manifest.json", _manifest(config, execution, "readiness", {}))
    write_new_json(directory / "readiness.json", result)
    return result


def _preflight(
    config: Config,
    c: ExecutionConfig,
    path: Path,
    metadata_path: Path,
    start: datetime,
    end: datetime,
    facts: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    validate_interval(config, start, end)
    if (end - start).total_seconds() > 366 * 86400:
        raise ValueError(
            "configured runs are bounded to 366 days; replay longer intervals as separate research runs"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required = {
        "forecast_id",
        "forecast_sha256",
        "timeframe",
        "target",
        "units",
        "horizon_bars",
        "dataset_version",
        "bar_dataset_version",
        "timestamp_convention",
        "feature_set_ids",
        "model_provenance",
        "feature_provenance",
        "evaluation_history_classification",
    }
    if required - metadata.keys():
        raise ValueError(f"missing forecast metadata: {sorted(required - metadata.keys())}")
    if metadata["timestamp_convention"] != "broker_local_bar_open":
        raise ValueError("unknown forecast availability convention")
    if sha256(path) != metadata["forecast_sha256"]:
        raise ValueError("forecast file hash differs from sidecar")
    bar_manifest = json.loads(
        (config.bars_dir(c.timeframe) / "_manifest.json").read_text(encoding="utf-8")
    )
    if (
        metadata["dataset_version"] != facts["dataset_version"]
        or metadata["bar_dataset_version"] != bar_manifest["bar_dataset_version"]
    ):
        raise ValueError("forecast data versions do not match verified local ticks/bars")
    gates = scientific_gates(metadata, start, c, evaluation_end=end)
    gates["data_provenance"] = facts["data_metadata_status"]
    evidence = metadata.get("evidence_files") or {}
    gates["evidence_identity"] = "unknown"
    verified = []
    if evidence:
        for name, expected in evidence.items():
            file = config.project_root / name
            if (
                not file.resolve().is_relative_to(config.project_root.resolve())
                or not file.is_file()
                or sha256(file) != expected
            ):
                raise ValueError(f"evidence file missing or hash differs: {name}")
            verified.append({"path": name, "sha256": expected})
        # An unrelated hashed file does not substantiate the declared audit.
        roles = metadata.get("evidence_roles") or {}
        required_roles = (
            "model",
            "features",
            "chronology",
            "nulls",
            "specification",
            "evaluation_history",
        )
        if all(
            isinstance(roles.get(role), list)
            and roles[role]
            and all(name in evidence for name in roles[role])
            for role in required_roles
        ):
            gates["evidence_identity"] = "passed"
    metadata["verified_evidence_files"] = verified
    if facts.get("evaluation_history"):
        # Local pre-2022 rows fed the historical development studies. A new sidecar
        # cannot erase the actual recorded inspection history.
        gates["untouched_evaluation"] = "failed"
    if gates["forecast_units_and_horizon"] != "passed":
        raise ValueError("forecast target, units or horizon incompatible with policy")
    return metadata, gates


def _event_key(event: Quote | BarClose) -> tuple[datetime, int, int]:
    return (
        (event.timestamp_utc, 0, event.sequence)
        if isinstance(event, Quote)
        else (event.timestamp_utc, 1, event.bar_index)
    )


def diagnostic_or_backtest(
    config: Config,
    execution: ExecutionConfig,
    run_id: str,
    start: datetime,
    end: datetime,
    *,
    forecast_path: Path | None,
    metadata_path: Path | None,
    historical: bool = False,
    quote_probe: bool = False,
    max_quotes: int = 50_000,
    scenario_set: Path | None = None,
) -> dict[str, Any]:
    validate_interval(config, start, end)
    if max_quotes < 1:
        raise ValueError("max_quotes must be positive")
    directory = new_run(config, run_id)
    started, cpu = time.perf_counter(), time.process_time()
    write_new_json(
        directory / "request.json",
        _manifest(
            config,
            execution,
            "preflight_request",
            {
                "start_utc": start,
                "end_utc_exclusive": end,
                "forecast_path": str(forecast_path),
                "historical_diagnostic": historical,
                "quote_probe": quote_probe,
                "scenario_set_sha256": sha256(scenario_set) if scenario_set else None,
            },
        ),
    )
    facts = inventory(config)
    if facts["data_metadata_status"] != "passed":
        write_new_json(directory / "readiness.json", facts)
        raise ValueError("tick identity/coverage preflight failed; see readiness.json")
    metadata: dict[str, Any] = {}
    gates = {
        "data_provenance": "passed",
        "forecast_chronology": "unknown",
        "null_evidence": "unknown",
        "broker_specification": "unknown",
        "untouched_evaluation": "unknown",
        "evidence_identity": "unknown",
        "forecast_units_and_horizon": "unknown",
    }
    variants = [execution]
    if scenario_set is not None:
        overrides = yaml.safe_load(scenario_set.read_text(encoding="utf-8"))
        if not isinstance(overrides, list) or not 1 <= len(overrides) <= 4:
            raise ValueError("scenario set must be a prespecified list of 1..4 overrides")
        allowed = {
            "scenario_id",
            "latency_ms",
            "commission_account_per_lot_per_leg",
            "slippage_usd_per_ounce_per_leg",
        }
        for body in overrides:
            if not isinstance(body, dict) or set(body) - allowed or "scenario_id" not in body:
                raise ValueError(
                    "scenario overrides support only identity, latency, commission, slippage"
                )
        variants = [replace(execution, **body) for body in overrides]
        if len({v.scenario_id for v in variants}) != len(variants):
            raise ValueError("duplicate scenario identity")
    if quote_probe:
        if scenario_set is not None or forecast_path is not None:
            raise ValueError("quote resource probe takes no forecasts or scenario set")
    else:
        if forecast_path is None or metadata_path is None:
            report = {
                "promotion_status": "historical_diagnostic_only",
                "gates": gates,
                "blocker": "forecast path and hash-verified forecast metadata required",
                "inventory": facts,
            }
            write_new_json(directory / "readiness.json", report)
            raise ValueError(
                "missing forecast input/sidecar; see readiness.json; no economic run executed"
            )
        metadata, gates = _preflight(
            config, execution, forecast_path, metadata_path, start, end, facts
        )
    status = promotion_status(gates)
    report = {"promotion_status": status, "gates": gates, "inventory": facts}
    write_new_json(directory / "readiness.json", report)
    if not quote_probe and status != "eligible_for_economic_review" and not historical:
        raise ValueError(
            "scientific/specification readiness unresolved; use --historical-diagnostic only for labelled diagnostics"
        )
    manifest = _manifest(
        config,
        execution,
        "quote_resource_probe" if quote_probe else "offline_backtest",
        {
            "start_utc": start,
            "end_utc_exclusive": end,
            "max_quotes": max_quotes if quote_probe else None,
            "forecast_metadata": metadata,
            "forecast_file": str(forecast_path) if forecast_path else None,
            "forecast_metadata_sha256": sha256(metadata_path) if metadata_path else None,
            "data_manifest_sha256": facts["manifest_sha256"],
            "dataset_version": facts["dataset_version"],
            "full_dataset_coverage": [facts["first_timestamp"], facts["last_timestamp"]],
            "readiness_gates": gates,
            "promotion_status": status,
            "evaluation_history_classification": "historical_diagnostic"
            if historical
            else "quote_only_no_forecasts"
            if quote_probe
            else "supplied_verified",
            "all_evaluated_variants": [
                {"config": v.resolved(), "hash": content_hash(v.resolved())} for v in variants
            ],
        },
    )
    # Every variant is registered before opening quote or forecast values.
    write_new_json(directory / "manifest.json", manifest)
    summaries = []
    for number, c in enumerate(variants):
        part = directory / f"scenario_{number + 1:02}"
        part.mkdir()
        forecasts = (
            forecast_records(forecast_path, metadata, config, c, start, end)
            if forecast_path is not None and not quote_probe
            else []
        )
        quotes = tick_quotes(config, c, start, end)
        count = 0
        first_quote = last_quote = None
        with DiskRecorder(part) as record:
            engine = ExecutionEngine(c, forecasts, record)
            quote_events: Iterable[Quote | BarClose] = quotes
            events = (
                quote_events
                if quote_probe
                else heapq.merge(quote_events, bar_closes(config, c, start, end), key=_event_key)
            )
            for event in events:
                engine.consume([event])
                if isinstance(event, Quote):
                    first_quote = first_quote or event
                    last_quote = event
                    count += 1
                if quote_probe and count >= max_quotes:
                    break
            if last_quote is None:
                raise ValueError("no quotes in declared interval")
            cutoff = last_quote.timestamp_utc if quote_probe else end
            result = engine.finish(cutoff)
            result["by_exit_month_utc"] = record.periods
        result.update(
            {
                "scenario_id": c.scenario_id,
                "configuration_sha256": content_hash(c.resolved()),
                "quote_coverage_utc": [
                    first_quote.timestamp_utc if first_quote else None,
                    last_quote.timestamp_utc,
                ],
                "quote_coverage_local": [
                    first_quote.timestamp_local if first_quote else None,
                    last_quote.timestamp_local,
                ],
                "probe_limit_reached": quote_probe and count >= max_quotes,
                "interpretation": "quote-only software resource probe; no forecast economics"
                if quote_probe
                else "historical diagnostic; cannot establish tradable edge"
                if historical
                else "evidence supplied; economic review only, not a profitability verdict",
            }
        )
        if quote_probe:
            result.pop("cutoff_equity_return", None)
        write_new_json(part / "summary.json", result)
        summaries.append(result)
    result = {
        "output": str(directory),
        "promotion_status": status,
        "gates": gates,
        "scenarios": summaries,
        "resources": {
            "elapsed_seconds": time.perf_counter() - started,
            "cpu_seconds": time.process_time() - cpu,
            **process_memory(),
        },
    }
    write_new_json(directory / "summary.json", result)
    return result


def cli_command(config: Config, args: Any) -> int:
    c = load_execution_config(
        args.execution_config or config.project_root / "config/execution.yaml"
    )
    if args.command == "execution-smoke":
        result = synthetic_smoke(config, args.run_id)
    elif args.command == "execution-readiness":
        result = readiness_run(config, c, args.run_id)
    else:
        result = diagnostic_or_backtest(
            config,
            c,
            args.run_id,
            parse_utc(args.start),
            parse_utc(args.end),
            forecast_path=args.forecasts,
            metadata_path=args.forecast_metadata,
            historical=args.historical_diagnostic,
            quote_probe=args.quote_probe,
            max_quotes=args.max_quotes,
            scenario_set=args.scenario_set,
        )
    print(
        json.dumps(
            {
                "output": result["output"],
                "promotion_status": result.get("promotion_status"),
                "gates": result.get("gates"),
                "resources": result.get("resources"),
            },
            indent=2,
        )
    )
    return (
        1
        if args.command == "execution-readiness"
        and result["promotion_status"] != "eligible_for_economic_review"
        else 0
    )
