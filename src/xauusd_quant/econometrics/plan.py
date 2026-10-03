"""Small immutable hypotheses and budgets; all designs remain retrospective research."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from ..execution.config import ExecutionConfig, content_hash
from ..execution.io import write_new_json
from ..strategy_validation.plan import Candidate, ExperimentPlan, Fold, Scenario

MODELS = ("ZERO", "STAGE13_RIDGE", "ARX", "KALMAN", "ROLLING", "EWMA", "GARCH", "HAR")
RETURN_MODELS = MODELS[:4]
VARIANCE_MODELS = MODELS[4:]


@dataclass(frozen=True)
class EconometricPlan:
    plan_id: str
    timeframe: str
    folds: tuple[Fold, ...]
    horizon_bars: int = 1
    calibration_hours: int = 24
    residual_window: int = 128
    residual_minimum: int = 32
    alpha: float = 0.2
    adaptive_gamma: float = 0.01
    ewma_lambda: float = 0.94
    hac_lags: int = 3
    monitor_drift: float = 0.5
    monitor_threshold: float = 8.0
    optimizer_iterations: int = 200
    minimum_training_rows: int = 100
    maximum_rows: int = 20000
    wall_budget_seconds: float = 240.0
    private_memory_budget_bytes: int = 1073741824
    trial_budget: int = 64
    comparison_block_minutes: int = 60
    bootstrap_replicates: int = 199
    minimum_evaluation_days: int = 5
    minimum_comparison_rows: int = 200
    minimum_loss_improvement: float = 0.01
    seed: int = 135013
    outcome_delay_ms: int = 0
    evaluation_history_classification: str = "historical_reconstructed"
    prior_search_exposure: str = "132 Stage 13 attempts plus 34 misclassified development fixture attempts; older searches unknown"
    revision_reason: str = "initial retrospective bounded econometric hypotheses"
    supersedes_plan: str | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*_V\d{3,}", self.plan_id):
            raise ValueError("safe versioned econometric plan id required")
        if self.timeframe not in ("5m", "15m") or not 1 <= self.horizon_bars <= 12:
            raise ValueError("supported grid is 5m/15m and horizon 1..12")
        if self.evaluation_history_classification not in (
            "software_correctness",
            "historical_reconstructed",
            "historical_diagnostic",
        ):
            raise ValueError("inspected development research cannot claim prospective evidence")
        integers = (
            self.calibration_hours,
            self.residual_window,
            self.residual_minimum,
            self.hac_lags,
            self.optimizer_iterations,
            self.minimum_training_rows,
            self.maximum_rows,
            self.private_memory_budget_bytes,
            self.trial_budget,
            self.comparison_block_minutes,
            self.bootstrap_replicates,
            self.minimum_evaluation_days,
            self.minimum_comparison_rows,
        )
        if any(type(v) is not int or v <= 0 for v in integers):
            raise ValueError("positive integer budgets/windows required")
        if type(self.outcome_delay_ms) is not int or self.outcome_delay_ms < 0:
            raise ValueError("nonnegative declared outcome publication delay required")
        if (
            self.residual_minimum > self.residual_window
            or self.maximum_rows < self.minimum_training_rows
        ):
            raise ValueError("incompatible history bounds")
        if not 0.01 < self.alpha < 0.5 or not 0 < self.ewma_lambda < 1:
            raise ValueError("invalid interval coverage or EWMA decay")
        for value in (
            self.alpha,
            self.adaptive_gamma,
            self.monitor_drift,
            self.monitor_threshold,
            self.wall_budget_seconds,
            self.minimum_loss_improvement,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("finite positive assumptions required")
        if self.expected_trials > self.trial_budget:
            raise ValueError("registered family exceeds trial budget")
        if not self.revision_reason or self.supersedes_plan == self.plan_id:
            raise ValueError("revision needs reason and distinct version")
        # Reuse Stage 13's chronological schedule and strict development-period validation.
        _ = self.validation_plan
        for fold in self.folds:
            if self.calibration_start(fold) <= datetime.fromisoformat(
                fold.training_start_utc.replace("Z", "+00:00")
            ):
                raise ValueError("calibration must follow training")
            if any(
                datetime.fromisoformat(b.replace("Z", "+00:00")) >= self.calibration_start(fold)
                for _, b in fold.inner_intervals
            ):
                raise ValueError("inner selection must precede calibration holdout")
        if (self.folds[-1].end - self.folds[0].start) > timedelta(days=2):
            raise ValueError("initial extension limited to two evaluation days per design")

    @property
    def seconds(self) -> int:
        return 300 if self.timeframe == "5m" else 900

    @property
    def expected_trials(self) -> int:
        # Eight model fits, four wrapper calibrations, one monitor initialization per fold.
        return (len(MODELS) + 5) * len(self.folds) + sum(
            2 * len(f.inner_intervals) for f in self.folds
        )

    @property
    def validation_plan(self) -> ExperimentPlan:
        return ExperimentPlan(
            plan_id=self.plan_id,
            timeframe=self.timeframe,
            horizon_bars=self.horizon_bars,
            research_question="Bounded incremental econometric information under aligned causal targets",
            folds=self.folds,
            candidates=(Candidate("STAGE13_RIDGE", "ridge", "preserved inner-count reference"),),
            scenarios=(Scenario("BASE"),),
            policy_thresholds=(0.0,),
            trial_budget=self.trial_budget,
            max_training_rows=self.maximum_rows,
            evaluation_history_classification=self.evaluation_history_classification,
            prior_search_exposure=self.prior_search_exposure,
        )

    def calibration_start(self, fold: Fold) -> datetime:
        return fold.start - timedelta(hours=self.calibration_hours)

    def resolved(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "models": MODELS,
            "questions": {
                "ARX": "Do two return lags and prior volatility beat zero and fold-local Stage 13 ridge?",
                "GARCH": "Does zero-mean Gaussian GARCH(1,1) beat rolling/EWMA variance?",
                "HAR": "Do nonnegative 1/3/12-grid variance components beat rolling/EWMA?",
                "KALMAN": "Does a filtered log local level improve future return MSE?",
                "intervals": "Do prior matured residual intervals approach 80% outcome coverage at useful widths?",
                "monitor": "Does a bounded CUSUM flag changed matured errors with acceptable fixture false alarms?",
            },
            "targets": {
                "return": "future grid-horizon sum log(mid_close[t+k]/mid_close[t+k-1]); log_mid_return",
                "variance": "sum of future contiguous grid squared log returns; log_return_squared",
                "availability": "bar open + grid duration + declared computation delay",
                "outcome": "horizon bar close + declared publication delay; strictly earlier before update",
            },
            "specifications": {
                "ARX": "OLS constant + return_1 + return_2 + sqrt(mean past 12 squared returns); training scaling; segmented Bartlett HAC",
                "GARCH": "zero conditional mean, Gaussian quasi-MLE; omega>0, alpha,beta>=0, alpha+beta<=0.995; no failure substitution",
                "HAR": "intraday HAR-style NNLS constant + mean RV over 1/3/12 contiguous grid returns; levels, no log retransformation",
                "KALMAN": "log-price local level; prior-only q/r Gaussian MLE; filter only; reset at gaps, never smooth",
                "benchmarks": "zero return; existing Stage 13 ridge/count selection refitted on aligned targets; rolling12 and EWMA0.94 variance",
            },
            "primary_metrics": "matched return MSE and variance QLIKE; relative loss improvement, coverage and widths separate",
            "inference": "prespecified within-contiguous-block loss bootstrap; minimum five calendar days; no DM p-value or search correction",
            "economic_diagnostics": "BLOCKED unless Stage 13 scientific and verified execution gates pass; no new economics in this design",
            "selection": "fixed added models; only existing ridge count uses preceding inner splits; all parameters estimated before holdout",
            "training": "expanding before preceding holdout; label-interval purge, no future-side training; no additional embargo",
            "measurement": "no interpolation, no returns across gaps; short/medium/long windows are grid intervals, never claimed trading days",
            "quote_noise": "training-only lag1 return correlation and nonoverlapping 3-grid RV ratio; descriptive signature diagnostic",
            "acceptance": ">=1% relative return MSE improvement or >=0.01 absolute QLIKE improvement; positive lower conditional block interval, >=60% positive folds, >=5 calendar days, provenance/null gates; no prospective claim",
            "stopping": "fixed eight-model family, no new variants; budget/resource/invariant failure stops; no Stage 14",
            "second_path": "15m only after 5m chronology/output/resource checks, regardless of outcome sign",
            "interval_type": "future-outcome prediction interval, never mean-confidence bound or profitable-trade lower bound",
            "monitor_action": "health output only; no strategy switches or evaluation-driven refitting",
        }


def load_plan(path: Path) -> EconometricPlan:
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ValueError("econometric plan must be a mapping")
    body["folds"] = tuple(
        Fold(**{**f, "inner_intervals": tuple(tuple(x) for x in f["inner_intervals"])})
        for f in body["folds"]
    )
    return EconometricPlan(**body)


def freeze_plan(root: Path, plan: EconometricPlan, execution: ExecutionConfig) -> Path:
    if (execution.timeframe, execution.horizon_bars) != (plan.timeframe, plan.horizon_bars):
        raise ValueError("execution/measurement target horizons differ")
    if execution.computation_delay_ms >= plan.seconds * 1000:
        raise ValueError("forecast computation delay reaches the next grid outcome")
    directory = root / "results/econometrics/plans"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{plan.plan_id}.json"
    body = {"plan": plan.resolved(), "execution": execution.resolved()}
    digest = content_hash(body)
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if (
            previous.get("content_sha256") != digest
            or content_hash({k: previous[k] for k in body}) != digest
        ):
            raise ValueError("immutable econometric plan changed; new version and reason required")
    else:
        write_new_json(
            path,
            {
                **body,
                "content_sha256": digest,
                "registered_utc": datetime.now(UTC),
                "classification": "retrospective; previous research inspection remains recorded",
            },
        )
    return path
