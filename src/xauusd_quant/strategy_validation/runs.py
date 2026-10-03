"""Freeze bounded trials, then execute continuously with the existing tick accounting path."""

from __future__ import annotations

import heapq
import json
import math
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ..execution.config import ExecutionConfig, content_hash, load_execution_config
from ..execution.engine import BarClose, ExecutionEngine, Quote
from ..execution.io import DiskRecorder, tick_quotes, validate_interval, write_new_json
from ..execution.policy import Forecast
from ..execution.readiness import code_identity, sha256
from ..execution.runs import _event_key, process_memory
from ..utils.config import Config
from .metrics import (
    AuditRecorder,
    aggregate_folds,
    concentration,
    daily_uncertainty,
    evidence_verdict,
    fold_metrics,
    snapshot,
)
from .pipeline import DevelopmentSource, FrozenPipeline, fit_fold
from .plan import Candidate, ExperimentPlan, Fold, Scenario, freeze_plan, load_plan
from .policies import consume_forecasts, policy_spec, stress_quotes
from .readiness import REQUIRED_GATES, assess
from .source import BarDevelopmentSource, SyntheticSource


class TrialLedger:
    def __init__(self, recorder: DiskRecorder, budget: int) -> None:
        self.recorder, self.budget, self.attempts = recorder, budget, 0

    def call(self, kind: str, specification: dict[str, Any], operation: Callable[[], Any]) -> Any:
        if self.attempts >= self.budget:
            raise RuntimeError("frozen trial budget exhausted; no expansion")
        self.attempts += 1
        number = self.attempts
        identity = content_hash({"kind": kind, "specification": specification})
        self.recorder(
            "trial_ledger",
            {
                "trial": number,
                "kind": kind,
                "identity": identity,
                "specification": specification,
                "status": "attempted",
            },
        )
        try:
            result = operation()
        except Exception as error:
            self.recorder(
                "trial_ledger",
                {
                    "trial": number,
                    "kind": kind,
                    "identity": identity,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "reason": str(error),
                },
            )
            if isinstance(error, (RuntimeError, AssertionError)):
                raise
            return None
        self.recorder(
            "trial_ledger",
            {"trial": number, "kind": kind, "identity": identity, "status": "completed"},
        )
        return result


def new_run(root: Path, run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*_V\d{3,}", run_id):
        raise ValueError("versioned safe run_id required")
    path = root / "results/strategy_validation/runs" / run_id
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ValueError("immutable run exists; bump version") from error
    return path


def execution_scenario(execution: ExecutionConfig, scenario: Scenario) -> ExecutionConfig:
    if scenario.latency_ms is not None and scenario.latency_ms < execution.latency_ms:
        raise ValueError("latency stress must not reduce base latency")
    return replace(
        execution,
        scenario_id=scenario.scenario_id,
        commission_account_per_lot_per_leg=execution.commission_account_per_lot_per_leg
        * scenario.commission_multiplier,
        slippage_usd_per_ounce_per_leg=execution.slippage_usd_per_ounce_per_leg
        * scenario.slippage_multiplier,
        latency_ms=scenario.latency_ms if scenario.latency_ms is not None else execution.latency_ms,
    )


QuoteFactory = Callable[[ExecutionConfig], Iterable[Quote]]


def execute_trial(
    plan: ExperimentPlan,
    execution: ExecutionConfig,
    scenario: Scenario,
    forecasts: list[Forecast],
    closes: list[BarClose],
    quotes: QuoteFactory,
    threshold: float,
    kind: str,
    directory: Path,
) -> dict[str, Any]:
    """One engine from first to last fold, with left-boundary marks and zero capital resets."""
    directory.mkdir()
    c = execution_scenario(execution, scenario)
    with DiskRecorder(directory) as sink:
        recorder = AuditRecorder(sink)
        engine = ExecutionEngine(
            c, consume_forecasts(forecasts, c, plan, threshold, kind, recorder), recorder
        )
        quote_events: Iterable[Quote | BarClose] = stress_quotes(quotes(c), scenario)
        close_events: Iterable[Quote | BarClose] = closes
        events = iter(heapq.merge(quote_events, close_events, key=_event_key))
        next_event = next(events, None)
        while next_event is not None and next_event.timestamp_utc < plan.start:
            engine.consume([next_event])
            next_event = next(events, None)
        before = snapshot(engine, plan.start)
        # The report denominator starts at the declared evaluation, not schedule warm-up.
        engine.first_clock = plan.start
        midnight = plan.start.replace(hour=0, minute=0, second=0, microsecond=0)
        days = []
        if midnight == plan.start:
            days.append(before["mark"])
        midnight += timedelta(days=1)
        day_cuts = []
        while midnight <= plan.end:
            day_cuts.append(midnight)
            midnight += timedelta(days=1)
        cuts = sorted({f.end for f in plan.folds} | set(day_cuts))
        fold_results = []
        fold_number = 0
        fold_peak = before["mark"]["equity_account"]
        fold_drawdown = 0.0
        final_summary: dict[str, Any] = {}
        for cutoff in cuts:
            while next_event is not None and next_event.timestamp_utc < cutoff:
                engine.consume([next_event])
                if isinstance(next_event, Quote):
                    mark = engine.mark(next_event.timestamp_utc)
                    equity = mark["equity_account"]
                    if equity is not None:
                        fold_peak = max(fold_peak, equity) if fold_peak is not None else equity
                        fold_drawdown = max(fold_drawdown, fold_peak - equity)
                next_event = next(events, None)
            if cutoff == plan.end:
                final_summary = engine.finish(cutoff)
            after = snapshot(engine, cutoff)
            if cutoff in day_cuts:
                days.append(after["mark"])
            if cutoff == plan.folds[fold_number].end:
                metrics = fold_metrics(plan.folds[fold_number].fold_id, before, after)
                metrics["running_max_drawdown_account"] = engine.max_drawdown
                equity = after["mark"]["equity_account"]
                if equity is not None and fold_peak is not None:
                    fold_drawdown = max(fold_drawdown, fold_peak - equity)
                metrics["max_drawdown_within_fold_account"] = fold_drawdown
                metrics["drawdown_coverage"] = "known actual quote and boundary marks only"
                sink("fold_metrics", metrics)
                fold_results.append(metrics)
                fold_number += 1
                before = after
                fold_peak = equity
                fold_drawdown = 0.0
        aggregate = aggregate_folds(fold_results, c.initial_cash_account)
        if engine.counts["quotes_observed"] == 0:
            raise ValueError("no executable quote observations in declared interval")
        if final_summary["open_position"] is not None:
            recorder.maximum_holding_seconds = max(
                recorder.maximum_holding_seconds,
                (plan.end - final_summary["open_position"]["entry_utc"]).total_seconds(),
            )
        result = {
            "execution": final_summary,
            "aggregate": aggregate,
            "folds": fold_results,
            "daily_marks": days,
            "uncertainty": daily_uncertainty(days, plan, recorder.maximum_holding_seconds),
            "concentration": concentration(recorder.trades, fold_results, plan),
            "policy": policy_spec(threshold, c, kind),
            "scenario": scenario,
            "execution_configuration_sha256": content_hash(c.resolved()),
            "comparison": "same input opportunities; trade populations may differ; not matched-trade inference",
        }
        for trade in recorder.trades:
            sink(
                "trade_fold_attribution",
                {
                    "position_id": trade["position_id"],
                    "entry_fold": next(
                        (f.fold_id for f in plan.folds if f.start <= trade["entry_utc"] < f.end),
                        None,
                    ),
                    "exit_fold": next(
                        (f.fold_id for f in plan.folds if f.start <= trade["exit_utc"] < f.end),
                        None,
                    ),
                },
            )
    # Scenario dataclass serialized explicitly (no implicit Python repr in JSON).
    from dataclasses import asdict

    result["scenario"] = asdict(scenario)
    write_new_json(directory / "summary.json", result)
    return result


def run_framework(
    config: Config,
    execution: ExecutionConfig,
    plan: ExperimentPlan,
    run_id: str,
    *,
    historical_diagnostic: bool = False,
    evidence_path: Path | None = None,
    source: DevelopmentSource | None = None,
    quotes: QuoteFactory | None = None,
    readiness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_interval(config, plan.start, plan.end)
    if readiness is not None and (
        plan.evaluation_history_classification != "software_correctness"
        or not isinstance(source, SyntheticSource)
    ):
        raise ValueError("readiness overrides are allowed only for declared synthetic fixtures")
    c = replace(execution, timeframe=plan.timeframe, horizon_bars=plan.horizon_bars)
    for scenario in plan.scenarios:
        execution_scenario(c, scenario)
    plan_path = freeze_plan(config.project_root, plan, c)
    directory = new_run(config.project_root, run_id)
    started, cpu = time.perf_counter(), time.process_time()
    write_new_json(
        directory / "request.json",
        {
            "stage": 13,
            "plan_file_sha256": sha256(plan_path),
            "code": code_identity(config.project_root),
            "execution": c.resolved(),
            "evidence_path": str(evidence_path) if evidence_path else None,
            "historical_diagnostic": historical_diagnostic,
            "status": "preflight_attempted",
        },
    )
    try:
        assessment = readiness or assess(config, c, plan, plan_path, evidence_path)
    except Exception as error:
        write_new_json(
            directory / "preflight_failure.json",
            {"status": "BLOCKED", "error_type": type(error).__name__, "reason": str(error)},
        )
        raise
    write_new_json(directory / "readiness.json", assessment)
    variants: list[dict[str, Any]] = [
        {
            "candidate": candidate.candidate_id,
            "threshold": threshold,
            "kind": "forecast",
            "scenario": s.scenario_id,
        }
        for candidate in plan.candidates
        for threshold in plan.policy_thresholds
        for s in plan.scenarios
    ] + [
        {"candidate": name, "threshold": 0.0, "kind": name, "scenario": s.scenario_id}
        for name in ("no_trading", "random_direction")
        for s in plan.scenarios
    ]
    classification = plan.evaluation_history_classification
    blocked = any(assessment["gates"].get(key) != "passed" for key in REQUIRED_GATES)
    if blocked:
        classification = "historical_diagnostic"
    write_new_json(
        directory / "manifest.json",
        {
            "stage": 13,
            "schema_version": "STRATEGY_RUN_V001",
            "plan_path": str(plan_path),
            "plan_file_sha256": sha256(plan_path),
            "created_utc": datetime.now(UTC),
            "code": code_identity(config.project_root),
            "data_configuration": config.to_dict(),
            "data_configuration_sha256": content_hash(config.to_dict()),
            "execution_configuration": c.resolved(),
            "fold_definitions": [f.__dict__ for f in plan.folds],
            "evaluation_history_classification": classification,
            "all_economic_variants": variants,
            "readiness_gates": assessment["gates"],
            "seeds": {"controls_and_bootstrap": plan.seed},
            "prior_search_exposure": plan.prior_search_exposure,
            "expected_trial_count": plan.expected_trials,
            "stopping_rule": plan.stopping_rule,
            "benchmark_scope": "no trading and serial block-random direction on valid feature opportunities; not pipeline nulls",
        },
    )
    with DiskRecorder(directory) as recorder:
        if blocked and not historical_diagnostic:
            for candidate in plan.candidates:
                recorder(
                    "trial_ledger",
                    {
                        "candidate": candidate.candidate_id,
                        "status": "blocked",
                        "reason": "readiness failed before opening evaluation values",
                    },
                )
            verdicts = {
                c.candidate_id: {"status": "BLOCKED", "reasons": assessment["gates"]}
                for c in plan.candidates
            }
            result = {
                "output": str(directory),
                "candidate_verdicts": verdicts,
                "attempted_trials": 0,
            }
            write_new_json(directory / "verdict.json", result)
            return result
        if source is None:
            source = BarDevelopmentSource(config, c, plan)
        write_new_json(
            directory / "source_identity.json",
            getattr(source, "identity", {"status": "unknown supplied source identity"}),
        )
        if quotes is None:

            def quote_stream(scenario: ExecutionConfig) -> Iterable[Quote]:
                return tick_quotes(config, scenario, plan.start, plan.end)

            quotes = quote_stream
        ledger = TrialLedger(recorder, plan.trial_budget)
        pipelines: dict[tuple[str, str], FrozenPipeline] = {}
        for candidate in plan.candidates:
            for fold in plan.folds:

                def freeze(candidate: Candidate = candidate, fold: Fold = fold) -> FrozenPipeline:
                    return fit_fold(source, plan, fold, candidate, ledger.call)

                pipeline = ledger.call(
                    "outer_freeze",
                    {
                        "candidate": candidate.candidate_id,
                        "fold": fold.fold_id,
                        "cutoff": fold.start,
                    },
                    freeze,
                )
                if pipeline is not None:
                    pipelines[candidate.candidate_id, fold.fold_id] = pipeline
                    path = directory / "frozen" / fold.fold_id
                    path.mkdir(parents=True, exist_ok=True)
                    write_new_json(path / f"{candidate.candidate_id}.json", pipeline.resolved())
        rows, closes = source.evaluation(plan.start, plan.end)
        predictions: dict[str, list[Forecast]] = {}
        for candidate in plan.candidates:
            records = []
            if any((candidate.candidate_id, fold.fold_id) not in pipelines for fold in plan.folds):
                continue
            for row in rows:
                available = row.available_at_utc + timedelta(milliseconds=c.computation_delay_ms)
                fold = next(f for f in plan.folds if f.start <= available < f.end)
                pipeline = pipelines[candidate.candidate_id, fold.fold_id]
                path = directory / "frozen" / fold.fold_id / f"{candidate.candidate_id}.json"
                value = pipeline.predict(row, prediction_at=available)
                forecast = Forecast.from_bar(
                    forecast_id=f"{fold.fold_id}:{candidate.candidate_id}:{row.row_id}",
                    bar_open_utc=row.bar_open_utc,
                    bar_index=row.bar_index,
                    value=value,
                    config=c,
                    provenance_id=sha256(path),
                )
                recorder(
                    "predictions",
                    {
                        **forecast.__dict__,
                        "fold_id": fold.fold_id,
                        "frozen_spec_sha256": sha256(path),
                        "feature_identities": pipeline.features,
                        "feature_count": len(pipeline.features),
                        "fitting_cutoff_utc": pipeline.cutoff_utc,
                    },
                )
                records.append(forecast)
            predictions[candidate.candidate_id] = records
        controls = [
            Forecast.from_bar(
                forecast_id=f"benchmark:{row.row_id}",
                bar_open_utc=row.bar_open_utc,
                bar_index=row.bar_index,
                value=1e-4,
                config=c,
                provenance_id=plan.plan_id,
            )
            for row in rows
        ]
        results = []
        for number, variant in enumerate(variants):
            forecasts = (
                predictions.get(variant["candidate"]) if variant["kind"] == "forecast" else controls
            )
            scenario = next(s for s in plan.scenarios if s.scenario_id == variant["scenario"])

            def evaluate(
                forecasts: list[Forecast] | None = forecasts,
                scenario: Scenario = scenario,
                variant: dict[str, Any] = variant,
                number: int = number,
            ) -> dict[str, Any]:
                if forecasts is None:
                    raise ValueError("candidate has a failed fold; partial curves forbidden")
                return execute_trial(
                    plan,
                    c,
                    scenario,
                    forecasts,
                    closes,
                    quotes,
                    variant["threshold"],
                    variant["kind"],
                    directory / f"trial_{number + 1:03}",
                )

            result = ledger.call("economic_evaluation", variant, evaluate)
            results.append({"variant": variant, "result": result})
        verdicts = {}
        for candidate in plan.candidates:
            family = [r for r in results if r["variant"]["candidate"] == candidate.candidate_id]
            primary = family[0]["result"]
            if primary is None or any(r["result"] is None for r in family):
                verdicts[candidate.candidate_id] = {
                    "status": "BLOCKED",
                    "reasons": ["failed registered trial/fold"],
                }
            else:
                verdicts[candidate.candidate_id] = evidence_verdict(
                    assessment["gates"], primary, plan, [r["result"] for r in family[1:]]
                )
        result = {
            "output": str(directory),
            "candidate_verdicts": verdicts,
            "attempted_trials": ledger.attempts,
            "planned_trials": plan.expected_trials,
            "results": results,
            "selection": "full registered family reported; no winner selected from outer results",
            "evaluation_history_classification": classification,
            "resources": {
                "elapsed_seconds": time.perf_counter() - started,
                "cpu_seconds": time.process_time() - cpu,
                **process_memory(),
            },
        }
    write_new_json(
        directory / "robustness.json",
        {
            "family": results,
            "interpretation": "all fixed neighbors/scenarios; independent populations; no optimization",
        },
    )
    write_new_json(directory / "verdict.json", result)
    return result


def synthetic_inputs(
    plan: ExperimentPlan, execution: ExecutionConfig
) -> tuple[SyntheticSource, QuoteFactory]:
    seconds = execution.bar_seconds
    start = min(
        datetime.fromisoformat(f.training_start_utc.replace("Z", "+00:00")) for f in plan.folds
    )
    count = int((plan.end - start).total_seconds() // seconds) + 1
    opens = [start + timedelta(seconds=i * seconds) for i in range(count)]
    closes = [2000 * math.exp(0.00004 * i + 0.0003 * math.sin(i / 7)) for i in range(count)]
    source = SyntheticSource(opens, closes, execution, plan.horizon_bars)

    def quotes(config: ExecutionConfig) -> Iterable[Quote]:
        del config
        sequence = 0
        for index, stamp in enumerate(opens):
            for offset in (0, 1, seconds - 1):
                at = stamp + timedelta(seconds=offset)
                if plan.start <= at < plan.end:
                    midpoint = closes[index]
                    yield Quote(
                        at, at.replace(tzinfo=None), midpoint - 0.1, midpoint + 0.1, sequence
                    )
                    sequence += 1

    return source, quotes


def cli_command(config: Config, args: Any) -> int:
    plan = load_plan(args.plan or config.project_root / "config/strategy_validation.yaml")
    execution = replace(
        load_execution_config(
            args.execution_config or config.project_root / "config/execution.yaml"
        ),
        timeframe=plan.timeframe,
        horizon_bars=plan.horizon_bars,
    )
    validate_interval(config, plan.start, plan.end)
    path = freeze_plan(config.project_root, plan, execution)
    if args.command == "strategy-plan":
        print(
            json.dumps(
                {"plan": str(path), "sha256": sha256(path), "expected_trials": plan.expected_trials}
            )
        )
        return 0
    if args.command == "strategy-readiness":
        directory = new_run(config.project_root, args.run_id)
        result = assess(config, execution, plan, path, args.evidence)
        result["output"] = str(directory)
        write_new_json(directory / "readiness.json", result)
        write_new_json(
            directory / "manifest.json",
            {
                "plan_sha256": sha256(path),
                "code": code_identity(config.project_root),
                "execution": execution.resolved(),
            },
        )
        print(json.dumps({"output": str(directory), "gates": result["gates"]}, indent=2))
        return 0 if all(v == "passed" for v in result["gates"].values()) else 1
    if args.command == "strategy-smoke":
        # Small training fixture, independent of local historical price outcomes.
        from .plan import Fold

        t = datetime(2021, 1, 4, tzinfo=UTC)

        def iso(minutes: int) -> str:
            return (t + timedelta(minutes=minutes)).isoformat()

        folds = tuple(
            Fold(
                f"F{i + 1}",
                iso(0),
                iso(900 + i * 180),
                iso(1080 + i * 180),
                ((iso(240), iso(360)), (iso(480), iso(600))),
            )
            for i in range(2)
        )
        plan = replace(
            plan,
            plan_id=f"SYNTHETIC_{plan.timeframe.upper()}_V001",
            folds=folds,
            evaluation_history_classification="software_correctness",
        )
        execution = replace(
            execution, max_gap_ms=max(execution.max_gap_ms, execution.bar_seconds * 2000)
        )
        source, quotes = synthetic_inputs(plan, execution)
        synthetic_gates = dict.fromkeys(
            (
                "data_provenance",
                "fold_local_pipeline",
                "null_evidence",
                "execution_specification",
                "evidence_identity",
                "evaluation_history_declared",
            ),
            "passed",
        )
        result = run_framework(
            config,
            execution,
            plan,
            args.run_id,
            source=source,
            quotes=quotes,
            readiness={
                "gates": synthetic_gates,
                "classification": "software_correctness",
                "meaning": "fixture declarations only, no market/null/broker evidence",
            },
        )
    else:
        result = run_framework(
            config,
            execution,
            plan,
            args.run_id,
            historical_diagnostic=args.historical_diagnostic,
            evidence_path=args.evidence,
        )
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in ("output", "candidate_verdicts", "attempted_trials", "resources")
            },
            indent=2,
        )
    )
    return 1 if any(v["status"] == "BLOCKED" for v in result["candidate_verdicts"].values()) else 0
