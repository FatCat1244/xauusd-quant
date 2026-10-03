"""Freeze, fit, calibrate and evaluate bounded chronological forecasts; never trade or unlock tests."""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from ..execution.config import ExecutionConfig, content_hash, load_execution_config
from ..execution.io import DiskRecorder, parse_utc, validate_interval, write_new_json
from ..execution.readiness import code_identity, sha256
from ..execution.runs import process_memory
from ..research.residual_stationarity import random_walk_control
from ..strategy_validation.pipeline import FrozenPipeline, fit_fold
from ..strategy_validation.plan import Candidate, Fold
from ..strategy_validation.readiness import assess
from ..strategy_validation.runs import TrialLedger
from ..strategy_validation.source import BarDevelopmentSource
from ..utils.config import Config
from .comparison import addition_verdict, compare, loss
from .data import GridData, Point, RidgeSource, Sample
from .diagnostics import series_diagnostics
from .monitor import ErrorCUSUM
from .plan import MODELS, RETURN_MODELS, EconometricPlan, freeze_plan, load_plan
from .regression import RegressionFit, fit_regression, validate_fit
from .state_space import LocalLevelState, fit_local_level
from .uncertainty import DelayedIntervals
from .volatility import VarianceState, fit_garch


class SyntheticBars:
    """Declared fixture prices only; no real-data readiness override is accepted."""

    def __init__(self, opens: list[datetime], prices: list[float], seconds: int) -> None:
        self.opens, self.prices, self.seconds = opens, prices, seconds
        self.accesses: list[dict[str, Any]] = []
        self.identity = {
            "kind": "declared synthetic",
            "fixture_sha256": content_hash({"opens": opens, "prices": prices}),
        }

    def __call__(self, start: datetime, cutoff: datetime) -> GridData:
        self.accesses.append({"start": start, "cutoff": cutoff})
        indices = [
            i
            for i, t in enumerate(self.opens)
            if start - timedelta(days=7) <= t and t + timedelta(seconds=self.seconds) < cutoff
        ]
        return GridData(
            [self.opens[i] for i in indices], [self.prices[i] for i in indices], self.seconds
        )


@dataclass
class Model:
    name: str
    fit: FrozenPipeline | RegressionFit | VarianceState | LocalLevelState | None
    spec: dict[str, Any]
    spec_hash: str = ""
    error: str | None = None

    def observe(self, data: GridData, index: int) -> dict[str, Any] | None:
        if self.error:
            return None
        state = self.fit
        if not isinstance(state, (VarianceState, LocalLevelState)):
            return None
        at = data.closes[index]
        if state.last_utc is not None and at <= state.last_utc:
            return None
        if isinstance(state, VarianceState):
            r = data.returns[index]
            state.observe(
                float(r) if math.isfinite(r) else None, at, contiguous=bool(data.contiguous[index])
            )
            return None
        return state.observe(float(data.logs[index]), at, contiguous=bool(data.contiguous[index]))

    def predict(self, point: Point, at: datetime) -> float:
        if point.available_utc > at or at < self.spec["fitting_cutoff_utc"]:
            raise ValueError("forecast features/model not available at decision time")
        if self.error:
            raise ValueError(self.error)
        if isinstance(self.fit, RegressionFit):
            return self.fit.predict(point, at)
        if isinstance(self.fit, FrozenPipeline):
            return self.fit.predict(point.stage13, prediction_at=at)
        if isinstance(self.fit, VarianceState):
            return self.fit.forecast()
        if isinstance(self.fit, LocalLevelState):
            return self.fit.forecast_return()
        if self.name == "ZERO":
            return 0.0
        raise ValueError("missing fitted specification; no fallback")


def fit_model(
    name: str,
    data: GridData,
    samples: list[Sample],
    start: datetime,
    cutoff: datetime,
    plan: EconometricPlan,
    fold: Fold,
    source: RidgeSource,
    ledger: TrialLedger,
) -> Model:
    if name not in MODELS:
        raise ValueError("model outside frozen family")
    validate_fit(samples, cutoff, plan.minimum_training_rows)
    if any(t >= cutoff for t in data.closes):
        raise ValueError("model fitting materialized a future bar")
    pieces = data.return_segments(start, cutoff)
    longest = max(pieces, key=len)
    spec: dict[str, Any] = {
        "model": name,
        "fold_id": fold.fold_id,
        "fitting_cutoff_utc": cutoff,
        "outer_information_cutoff_utc": fold.start,
        "training_label_as_of_utc": max(s.outcome.matured_utc for s in samples),
        "parameter_information_as_of_utc": max(data.closes),
        "training_samples": len(samples),
        "training_identity": content_hash([s.point.row_id for s in samples]),
        "permitted_bar_values_sha256": content_hash({"opens": data.opens, "logs": list(data.logs)}),
        "feature_selection": "fixed added model identities; Stage13 reference alone uses nested count selection",
        "horizon_bars": plan.horizon_bars,
        "grid_seconds": plan.seconds,
        "target": "future_return" if name in RETURN_MODELS else "realized_variance",
        "units": "log_mid_return" if name in RETURN_MODELS else "log_return_squared",
        "inference": "retrospective as-of reconstruction, not prospective model creation",
    }
    fitted: FrozenPipeline | RegressionFit | VarianceState | LocalLevelState | None
    if name == "STAGE13_RIDGE":
        fitting_fold = replace(fold, evaluation_start_utc=cutoff.isoformat())
        fitted = fit_fold(
            source,
            plan.validation_plan,
            fitting_fold,
            Candidate(name, "ridge", "preserved aligned reference"),
            ledger.call,
        )
    elif name in ("ARX", "HAR"):
        fitted = fit_regression(
            samples, cutoff, name, minimum=plan.minimum_training_rows, hac_lags=plan.hac_lags
        )
    elif name == "KALMAN":
        fitted = LocalLevelState(
            fit_local_level(
                data.level_segments(start, cutoff),
                minimum=plan.minimum_training_rows,
                max_iterations=plan.optimizer_iterations,
            )
        )
    elif name in ("ROLLING", "EWMA", "GARCH"):
        initial = float(np.square(np.concatenate(pieces)).mean())
        garch = (
            fit_garch(
                pieces, minimum=plan.minimum_training_rows, max_iterations=plan.optimizer_iterations
            )
            if name == "GARCH"
            else None
        )
        fitted = VarianceState(
            name,
            garch.initialization_variance if garch else initial,
            plan.horizon_bars,
            garch=garch,
            decay=plan.ewma_lambda,
        )
    else:
        fitted = None
    model = Model(name, fitted, spec)
    if isinstance(fitted, (VarianceState, LocalLevelState)):
        for i in range(len(data.opens)):
            model.observe(data, i)
    spec["fitted"] = fitted.resolved() if fitted is not None else {"constant_return": 0.0}
    if name in ("ARX", "GARCH", "KALMAN"):
        spec["return_stationarity_and_dependence"] = series_diagnostics(
            longest, label="training_contiguous_returns"
        )
    if name == "ARX":
        # Residual diagnostic uses one actual contiguous segment, never closes compressed across gaps.
        segments: list[list[Sample]] = []
        for sample in samples:
            if not segments or sample.point.available_utc - segments[-1][
                -1
            ].point.available_utc != timedelta(seconds=plan.seconds):
                segments.append([])
            segments[-1].append(sample)
        assert isinstance(fitted, RegressionFit)
        segment = max(segments, key=len)
        # In-sample fitted residuals are explicitly descriptive, not historical forecasts.
        x = np.asarray([s.point.arx for s in segment])
        predictions = fitted.intercept + ((x - fitted.means) / fitted.scales) @ fitted.coefficients
        residuals = np.asarray([s.outcome.log_return for s in segment]) - predictions
        spec["in_sample_residual_diagnostics"] = series_diagnostics(
            residuals, label="ARX_fit_residuals_descriptive"
        )
    if name == "GARCH":
        assert isinstance(fitted, VarianceState) and fitted.garch is not None
        from .volatility import variance_path

        h = variance_path(
            longest,
            fitted.garch.omega,
            fitted.garch.alpha,
            fitted.garch.beta,
            fitted.garch.initialization_variance,
        )
        spec["standardized_innovation_diagnostics"] = series_diagnostics(
            longest / np.sqrt(h), label="GARCH_zero_mean_standardized_returns"
        )
    return model


def run_phase(
    data: GridData,
    start: datetime,
    end: datetime,
    phase: str,
    models: dict[str, Model],
    intervals: dict[tuple[str, str], DelayedIntervals],
    monitor: ErrorCUSUM | None,
    plan: EconometricPlan,
    execution: ExecutionConfig,
    fold: Fold,
    recorder: DiskRecorder,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    points = {p.index: p for p in data.points(start, end)}
    labels = data.outcomes(plan.horizon_bars, plan.outcome_delay_ms, end)
    pending_ids = set().union(*(set(s.pending) for s in intervals.values()))
    mature = sorted(
        (o for o in labels.values() if o.start_utc >= start or o.row_id in pending_ids),
        key=lambda o: (o.matured_utc, o.row_id),
    )
    cursor = 0
    forecasts: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []

    def deliver(now: datetime) -> None:
        nonlocal cursor
        while cursor < len(mature) and mature[cursor].matured_utc < now:
            outcome = mature[cursor]
            for (name, method), wrapper in intervals.items():
                update = wrapper.deliver(outcome, now)
                if update is not None:
                    row = {
                        **update,
                        "model": name,
                        "method": method,
                        "phase": phase,
                        "fold_id": fold.fold_id,
                    }
                    recorder("calibration_updates", row)
                    updates.append(row)
                    if monitor is not None and name == "ARX" and method == "rolling":
                        recorder(
                            "change_monitor",
                            {
                                **monitor.update(update["error"], outcome.matured_utc, now),
                                "fold_id": fold.fold_id,
                                "forecast_issued_utc": update["issued_utc"],
                            },
                        )
            cursor += 1

    # Explicit batches exercise state carry. Gap/closure rules, never batch edges, reset a state.
    for batch in range(0, len(data.opens), 64):
        for i in range(batch, min(batch + 64, len(data.opens))):
            if data.closes[i] < start:
                continue
            at = data.closes[i] + timedelta(milliseconds=execution.computation_delay_ms)
            if at >= end:
                continue
            deliver(at)
            for model in models.values():
                try:
                    state = model.observe(data, i)
                    if state is not None:
                        recorder(
                            "filtered_states",
                            {**state, "fold_id": fold.fold_id, "phase": phase, "model": model.name},
                        )
                except ValueError as error:
                    model.error = str(error)
                    recorder(
                        "state_failures",
                        {
                            "model": model.name,
                            "fold_id": fold.fold_id,
                            "phase": phase,
                            "available_utc": at,
                            "reason": str(error),
                        },
                    )
            point = points.get(i)
            if point is None:
                continue
            for name, model in models.items():
                row: dict[str, Any] = {
                    "model": name,
                    "fold_id": fold.fold_id,
                    "phase": phase,
                    "row_id": point.row_id,
                    "bar_open_utc": point.bar_open_utc,
                    "available_utc": at.isoformat(),
                    "target_end_utc": (
                        point.available_utc + timedelta(seconds=plan.seconds * plan.horizon_bars)
                    ).isoformat(),
                    "target": model.spec["target"],
                    "units": model.spec["units"],
                    "horizon_bars": plan.horizon_bars,
                    "grid_seconds": plan.seconds,
                    "frozen_spec_sha256": model.spec_hash,
                }
                try:
                    value = model.predict(point, at)
                    if not math.isfinite(value):
                        raise ValueError("nonfinite forecast")
                    row.update(value=value, status="ok")
                    for (candidate, method), wrapper in intervals.items():
                        if candidate == name:
                            target_end = point.available_utc + timedelta(
                                seconds=plan.seconds * plan.horizon_bars
                            )
                            uncertainty = wrapper.issue(
                                point.row_id,
                                value,
                                at,
                                target_end,
                                target_end + timedelta(milliseconds=plan.outcome_delay_ms),
                            )
                            recorder(
                                "uncertainty",
                                {
                                    **uncertainty,
                                    "model": name,
                                    "method": method,
                                    "fold_id": fold.fold_id,
                                    "phase": phase,
                                },
                            )
                except ValueError as error:
                    row.update(value=None, status="rejected", reason=str(error))
                recorder("forecasts", row)
                forecasts.append(row)
            # Missing future grid outcomes expire explicitly, never enter residual calibration.
            for (name, method), wrapper in intervals.items():
                for expired in wrapper.expire(at):
                    recorder(
                        "missing_outcomes",
                        {
                            "row_id": expired,
                            "model": name,
                            "method": method,
                            "fold_id": fold.fold_id,
                            "phase": phase,
                            "observed_utc": at,
                        },
                    )
    deliver(end)
    return forecasts, updates


def evaluate(
    config: Config,
    execution: ExecutionConfig,
    plan: EconometricPlan,
    run_id: str,
    *,
    synthetic: SyntheticBars | None = None,
    readiness_only: bool = False,
) -> dict[str, Any]:
    started, cpu = time.perf_counter(), time.process_time()
    validate_interval(config, plan.folds[0].start, plan.folds[-1].end)
    if synthetic is not None and plan.evaluation_history_classification != "software_correctness":
        raise ValueError("synthetic reader cannot override real scientific readiness")
    if synthetic is None and plan.evaluation_history_classification == "software_correctness":
        raise ValueError(
            "software correctness requires a declared synthetic reader; real paths refused"
        )
    path = freeze_plan(config.project_root, plan, execution)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*_V\d{3,}", run_id):
        raise ValueError("versioned safe run id required")
    output = config.project_root / "results/econometrics/runs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    write_new_json(
        output / "request.json",
        {
            "run_id": run_id,
            "plan": str(path),
            "plan_sha256": sha256(path),
            "command": "readiness" if readiness_only else "forecast_evaluation",
            "synthetic": synthetic is not None,
        },
    )
    try:
        if synthetic is None:
            readiness = assess(config, execution, plan.validation_plan, path)
            base = BarDevelopmentSource(config, execution, plan.validation_plan)

            def reader(start: datetime, cutoff: datetime) -> GridData:
                opens, prices = base.completed_bars(start, cutoff)
                return GridData(opens, prices, plan.seconds)

            identity = base.identity
        else:
            reader = synthetic
            identity = synthetic.identity
            readiness = {
                "gates": {
                    "data_provenance": "passed",
                    "fold_local_pipeline": "passed",
                    "null_evidence": "unknown",
                    "execution_specification": "unknown",
                    "evidence_identity": "unknown",
                    "evaluation_history_declared": "passed",
                },
                "classification": "software_correctness",
                "inventory": {"fixture": identity},
            }
        if readiness["gates"]["data_provenance"] != "passed":
            raise ValueError("data provenance failed; cannot open economic/statistical values")
    except (OSError, ValueError, KeyError) as error:
        write_new_json(
            output / "preflight_failure.json", {"reason": str(error), "type": type(error).__name__}
        )
        result = {
            "output": str(output),
            "status": "BLOCKED BY MISSING INPUTS OR EVIDENCE",
            "attempted_trials": 0,
            "reason": str(error),
        }
        write_new_json(output / "verdict.json", result)
        return result
    write_new_json(output / "readiness.json", readiness)
    write_new_json(
        output / "manifest.json",
        {
            "stage": "13.5",
            "schema_version": "ECONOMETRIC_RUN_V001",
            "plan_sha256": sha256(path),
            "code": code_identity(config.project_root),
            "source": identity,
            "models": MODELS,
            "planned_attempts": plan.expected_trials,
            "execution": execution.resolved(),
            "history": plan.evaluation_history_classification,
            "prior_search_exposure": plan.prior_search_exposure,
            "economic_comparisons": "not authorized by readiness; no new policy/economics in this plan",
        },
    )
    if readiness_only:
        result = {
            "output": str(output),
            "status": "BLOCKED BY MISSING INPUTS OR EVIDENCE",
            "gates": readiness["gates"],
        }
        write_new_json(output / "verdict.json", result)
        return result
    scored: list[dict[str, Any]] = []
    failures: set[str] = set()
    coverage: list[dict[str, Any]] = []
    stability: list[dict[str, Any]] = []
    with DiskRecorder(output) as recorder:
        ledger = TrialLedger(recorder, plan.trial_budget)
        ridge_source = RidgeSource(reader, plan.horizon_bars, plan.outcome_delay_ms)
        for fold in plan.folds:
            if time.perf_counter() - started > plan.wall_budget_seconds:
                raise RuntimeError("frozen econometric wall budget exhausted; no expansion")
            training_start, cutoff = (
                parse_utc(fold.training_start_utc),
                plan.calibration_start(fold),
            )
            recorder(
                "fold_definitions",
                {
                    "fold_id": fold.fold_id,
                    "training_start": training_start,
                    "fitting_cutoff": cutoff,
                    "calibration_end": fold.start,
                    "evaluation_end": fold.end,
                    "embargo": "zero; preceding-only training and information-interval purge",
                },
            )
            training = reader(training_start, cutoff)
            samples = training.samples(
                training_start, cutoff, plan.horizon_bars, plan.outcome_delay_ms
            )
            recorder(
                "measurement_diagnostics",
                {"fold_id": fold.fold_id, **training.noise_diagnostic(training_start, cutoff)},
            )
            models: dict[str, Model] = {}
            directory = output / "frozen" / fold.fold_id
            directory.mkdir(parents=True)
            for name in MODELS:
                fitted = ledger.call(
                    "model_fit",
                    {"model": name, "fold_id": fold.fold_id, "cutoff": cutoff},
                    partial(
                        fit_model,
                        name,
                        training,
                        samples,
                        training_start,
                        cutoff,
                        plan,
                        fold,
                        ridge_source,
                        ledger,
                    ),
                )
                file = directory / f"{name}.json"
                if fitted is None:
                    failures.add(name)
                    write_new_json(
                        file,
                        {
                            "model": name,
                            "fold_id": fold.fold_id,
                            "status": "failed; inspect immutable trial ledger",
                        },
                    )
                    continue
                write_new_json(file, fitted.spec)
                fitted.spec_hash = sha256(file)
                models[name] = fitted
                if isinstance(fitted.fit, RegressionFit):
                    stability.append(
                        {
                            "model": name,
                            "fold_id": fold.fold_id,
                            "cutoff": cutoff,
                            "coefficients_raw_units": fitted.fit.diagnostics[
                                "coefficients_raw_units"
                            ],
                            "intercept_raw_units": fitted.fit.diagnostics["intercept_raw_units"],
                            "interpretation": "chronological description, not a significance/search correction",
                        }
                    )
            wrappers: dict[tuple[str, str], DelayedIntervals] = {
                (name, method): DelayedIntervals(
                    window=plan.residual_window,
                    minimum=plan.residual_minimum,
                    alpha=plan.alpha,
                    gamma=plan.adaptive_gamma if method == "adaptive" else 0.0,
                    horizon=plan.horizon_bars,
                    seconds=plan.seconds,
                )
                for name in ("ARX", "STAGE13_RIDGE")
                if name in models
                for method in ("rolling", "adaptive")
            }
            # Model files exist before calibration values; calibrated state exists before outer values.
            calibration = reader(cutoff, fold.start)
            _, calibration_updates = run_phase(
                calibration,
                cutoff,
                fold.start,
                "calibration",
                models,
                wrappers,
                None,
                plan,
                execution,
                fold,
                recorder,
            )
            for name in ("ARX", "STAGE13_RIDGE"):
                for method in ("rolling", "adaptive"):
                    calibration_state = wrappers.get((name, method))

                    def validate_calibration(
                        state: DelayedIntervals | None = calibration_state,
                        outer_cutoff: datetime = fold.start,
                    ) -> dict[str, Any]:
                        if state is None or len(state.errors) < plan.residual_minimum:
                            raise ValueError(
                                "missing model/insufficient preceding calibration; no future seeding"
                            )
                        if state.last_maturity is None or state.last_maturity >= outer_cutoff:
                            raise ValueError("calibration used outer outcomes")
                        return state.resolved()

                    ledger.call(
                        "uncertainty_calibration",
                        {"model": name, "method": method, "fold_id": fold.fold_id},
                        validate_calibration,
                    )
            errors = [
                r["error"]
                for r in calibration_updates
                if r["model"] == "ARX" and r["method"] == "rolling"
            ]
            scale = (
                math.sqrt(float(np.square(errors).mean()))
                if len(errors) >= plan.residual_minimum
                else None
            )

            def initialize_monitor(scale: float | None = scale) -> ErrorCUSUM:
                if scale is None or scale <= 0:
                    raise ValueError("insufficient preceding monitor calibration; no future scale")
                return ErrorCUSUM(scale, drift=plan.monitor_drift, threshold=plan.monitor_threshold)

            monitor = ledger.call(
                "monitor_initialization",
                {"model": "ARX", "fold_id": fold.fold_id},
                initialize_monitor,
            )
            write_new_json(
                directory / "CALIBRATION_STATE.json",
                {
                    "as_of_exclusive": fold.start,
                    "wrappers": {
                        f"{name}:{method}": state.resolved()
                        for (name, method), state in wrappers.items()
                    },
                    "monitor": monitor.resolved()
                    if monitor
                    else {
                        "status": "untested",
                        "reason": "insufficient/degenerate preceding calibration errors",
                    },
                    "model_states": {
                        name: m.fit.resolved()
                        for name, m in models.items()
                        if isinstance(m.fit, (LocalLevelState, VarianceState))
                    },
                },
            )
            evaluation = reader(fold.start, fold.end)
            forecasts, updates = run_phase(
                evaluation,
                fold.start,
                fold.end,
                "evaluation",
                models,
                wrappers,
                monitor,
                plan,
                execution,
                fold,
                recorder,
            )
            outcomes = evaluation.outcomes(plan.horizon_bars, plan.outcome_delay_ms, fold.end)
            counts: Counter[str] = Counter()
            for prediction in forecasts:
                outcome = outcomes.get(prediction["row_id"])
                if prediction["status"] != "ok":
                    failures.add(prediction["model"])
                    counts["invalid_forecasts"] += 1
                    continue
                if outcome is None:
                    recorder(
                        "unscored_forecasts",
                        {
                            **prediction,
                            "reason": "gap/end/publication boundary; no outcome fabrication",
                        },
                    )
                    counts["censored_forecasts"] += 1
                    continue
                actual = (
                    outcome.log_return
                    if prediction["target"] == "future_return"
                    else outcome.realized_variance
                )
                row = {
                    **prediction,
                    "actual": actual,
                    "outcome_matured_utc": outcome.matured_utc.isoformat(),
                    "status": "scored",
                    "loss": loss(prediction["value"], actual, prediction["target"]),
                }
                recorder("forecast_scores", row)
                scored.append(row)
                counts["scored_forecasts"] += 1
            for name, method in wrappers:
                usable = [
                    r
                    for r in updates
                    if r["model"] == name
                    and r["method"] == method
                    and r["issued_utc"] >= fold.start
                    and r["covered"] is not None
                ]
                metric = {
                    "fold_id": fold.fold_id,
                    "model": name,
                    "method": method,
                    "evaluated_intervals": len(usable),
                    "coverage": sum(r["covered"] for r in usable) / len(usable) if usable else None,
                    "mean_width": float(np.mean([r["width"] for r in usable])) if usable else None,
                    "type": "future_outcome_prediction_interval",
                    "nominal_coverage": 1 - plan.alpha,
                }
                recorder("coverage", metric)
                coverage.append(metric)
                for row_id in wrappers[name, method].pending:
                    recorder(
                        "censored_calibration",
                        {
                            "fold_id": fold.fold_id,
                            "model": name,
                            "method": method,
                            "row_id": row_id,
                            "reason": "fold end/outcome unavailable; no calibration value inferred",
                        },
                    )
            recorder("fold_counts", {"fold_id": fold.fold_id, **counts})
            write_new_json(
                directory / "FINAL_STATE.json",
                {
                    "wrappers": {
                        f"{name}:{method}": s.resolved() for (name, method), s in wrappers.items()
                    },
                    "monitor": monitor.resolved() if monitor else None,
                    "model_states": {
                        name: m.fit.resolved()
                        for name, m in models.items()
                        if isinstance(m.fit, (LocalLevelState, VarianceState))
                    },
                },
            )
            memory = process_memory()
            private_bytes = memory.get("private_bytes")
            if (
                isinstance(private_bytes, int)
                and private_bytes > plan.private_memory_budget_bytes
                or time.perf_counter() - started > plan.wall_budget_seconds
            ):
                recorder(
                    "resource_stop",
                    {"memory": memory, "elapsed_seconds": time.perf_counter() - started},
                )
                raise RuntimeError("frozen econometric resource budget exceeded; no expansion")
        comparisons = [
            compare(scored, name, baseline, plan)
            for name, baseline in (
                ("ARX", "ZERO"),
                ("ARX", "STAGE13_RIDGE"),
                ("KALMAN", "ZERO"),
                ("KALMAN", "STAGE13_RIDGE"),
                ("GARCH", "ROLLING"),
                ("GARCH", "EWMA"),
                ("HAR", "ROLLING"),
                ("HAR", "EWMA"),
            )
        ]
        verdicts = {
            name: addition_verdict(
                [c for c in comparisons if c["candidate"] == name],
                readiness["gates"],
                plan,
                failed=name in failures,
            )
            for name in ("ARX", "KALMAN", "GARCH", "HAR")
        }
        # Conditional numerical/coverage fixtures never replace old pipeline null evidence.
        recorder(
            "control_notes",
            {
                "prior_pipeline_nulls": "preserved; matching new-plan null evidence unknown",
                "random_direction": "not used as replacement null",
                "detrending": "no residual-direction claim",
            },
        )
    write_new_json(output / "comparisons.json", comparisons)
    write_new_json(output / "coefficient_stability.json", stability)
    result = {
        "output": str(output),
        "status": "completed_retrospective_diagnostic",
        "candidate_verdicts": verdicts,
        "attempted_trials": ledger.attempts,
        "failed_models": sorted(failures),
        "coverage": coverage,
        "economic_comparison": {
            "status": "BLOCKED BY MISSING INPUTS OR EVIDENCE",
            "reasons": [k for k, v in readiness["gates"].items() if v != "passed"],
            "executed": False,
            "scope": "no economic candidate registered here; original Stage 13 baseline preserved",
        },
        "uncertainty_status": "prediction intervals empirical only; not mean confidence; no dependent-data guarantee",
        "monitor_status": "health diagnostic only, no automatic adaptations",
        "resources": {
            "elapsed_seconds": time.perf_counter() - started,
            "cpu_seconds": time.process_time() - cpu,
            **process_memory(),
        },
    }
    write_new_json(output / "verdict.json", result)
    return result


def synthetic_inputs(plan: EconometricPlan) -> SyntheticBars:
    start = parse_utc(plan.folds[0].training_start_utc) - timedelta(days=1)
    n = int((plan.folds[-1].end - start).total_seconds() / plan.seconds) + 2
    rng = np.random.default_rng(plan.seed)
    shocks = rng.normal(0, 0.0004, n)
    returns = np.zeros(n)
    for i in range(1, n):
        returns[i] = 0.25 * returns[i - 1] + shocks[i]
    prices = list(2000 * np.exp(np.cumsum(returns)))
    return SyntheticBars(
        [start + timedelta(seconds=i * plan.seconds) for i in range(n)], prices, plan.seconds
    )


def fixture_monitor(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    sequence = rng.normal(size=500)
    origin = datetime(2021, 1, 1, tzinfo=UTC)
    reports: dict[str, Any] = {}
    for name, values in (
        ("unchanged", sequence),
        ("changed", sequence + np.r_[np.zeros(250), np.full(250, 3.0)]),
    ):
        state = ErrorCUSUM(1.0)
        alarms = []
        for i, value in enumerate(values):
            matured = origin + timedelta(minutes=i)
            record = state.update(float(value), matured, matured + timedelta(seconds=1))
            if record["alarm"]:
                alarms.append(i)
        reports[name] = {
            "alarms": alarms,
            "false_alarms_before_change": sum(i < 250 for i in alarms),
            "first_postchange_delay": next((i - 250 for i in alarms if i >= 250), None)
            if name == "changed"
            else None,
        }
    reports["detrended_random_walk"] = series_diagnostics(
        random_walk_control(640, 32, seed=seed), label="same_existing_rolling_detrend_fixture"
    )
    reports["interpretation"] = "fixed synthetic controls, not passing real pipeline null evidence"
    return reports


def cli_command(config: Config, args: Any) -> int:
    plan = load_plan(args.plan or config.project_root / "config/econometrics.yaml")
    execution = replace(
        load_execution_config(
            args.execution_config or config.project_root / "config/execution.yaml"
        ),
        timeframe=plan.timeframe,
        horizon_bars=plan.horizon_bars,
    )
    if args.command == "econometric-plan":
        path = freeze_plan(config.project_root, plan, execution)
        print(
            json.dumps(
                {"plan": str(path), "sha256": sha256(path), "attempts": plan.expected_trials},
                indent=2,
            )
        )
        return 0
    if args.command == "econometric-smoke":
        plan = replace(
            plan,
            plan_id=plan.plan_id.replace("_V", "_SYNTH_V"),
            evaluation_history_classification="software_correctness",
            prior_search_exposure="synthetic only; separate from historical attempts",
        )
        result = evaluate(config, execution, plan, args.run_id, synthetic=synthetic_inputs(plan))
        write_new_json(
            Path(result["output"]) / "synthetic_controls.json", fixture_monitor(plan.seed)
        )
    else:
        result = evaluate(
            config,
            execution,
            plan,
            args.run_id,
            readiness_only=args.command == "econometric-readiness",
        )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                in (
                    "output",
                    "status",
                    "candidate_verdicts",
                    "resources",
                    "attempted_trials",
                    "failed_models",
                )
            },
            indent=2,
        )
    )
    return (
        1
        if args.command == "econometric-readiness"
        or result["status"] == "BLOCKED BY MISSING INPUTS OR EVIDENCE"
        else 0
    )
