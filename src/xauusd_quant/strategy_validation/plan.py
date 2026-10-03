"""Strict, immutable experiment designs registered before evaluation values are opened."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from ..execution.config import ExecutionConfig, content_hash
from ..execution.io import parse_utc, write_new_json

FEATURE_POOL = ("return_1", "momentum_3", "volatility_3")


@dataclass(frozen=True)
class Fold:
    fold_id: str
    training_start_utc: str
    evaluation_start_utc: str
    evaluation_end_utc: str
    inner_intervals: tuple[tuple[str, str], ...]

    @property
    def start(self) -> datetime:
        return parse_utc(self.evaluation_start_utc)

    @property
    def end(self) -> datetime:
        return parse_utc(self.evaluation_end_utc)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    model: str
    rationale: str


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    commission_multiplier: float = 1.0
    slippage_multiplier: float = 1.0
    latency_ms: int | None = None
    spread_multiplier: float = 1.0


@dataclass(frozen=True)
class ExperimentPlan:
    plan_id: str
    research_question: str
    timeframe: str
    horizon_bars: int
    folds: tuple[Fold, ...]
    candidates: tuple[Candidate, ...]
    scenarios: tuple[Scenario, ...]
    policy_thresholds: tuple[float, ...] = (0.0, 0.00001)
    feature_counts: tuple[int, ...] = (1, 2)
    feature_pool: tuple[str, ...] = FEATURE_POOL
    ridge_alpha: float = 1.0
    training_window: str = "expanding"
    rolling_days: int | None = None
    inner_selection: str = "mean_chronological_validation_mse_smallest_count_tie"
    target: str = "future_return"
    units: str = "log_mid_return"
    primary_metric: str = "marked_equity_change_account"
    evaluation_history_classification: str = "historical_reconstructed"
    prior_search_exposure: str = "unknown; historical feature/model/ensemble research inspected"
    revision_reason: str = "initial Stage 13 retrospective design; not prospective registration"
    supersedes_plan: str | None = None
    trial_budget: int = 64
    seed: int = 130013
    random_direction_block_opportunities: int = 16
    bootstrap_days: int = 5
    bootstrap_replicates: int = 199
    minimum_daily_observations: int = 20
    minimum_folds: int = 3
    minimum_closed_trades: int = 30
    minimum_positive_fold_fraction: float = 0.6
    maximum_positive_trade_concentration: float = 0.5
    maximum_positive_fold_concentration: float = 0.75
    remove_largest_trades: int = 3
    max_training_rows: int = 200_000
    max_evaluation_days: int = 31
    fold_boundary_mode: str = "continuous_carry"
    stopping_rule: str = (
        "fixed_budget_no_expansion; candidate failure recorded; invariant failure aborts"
    )
    sizing: str = "constant_execution_config_lots_no_compounding"
    overnight: str = "allowed_with_declared_continuous_funding"
    embargo_basis: str = (
        "none: only past training; labels strictly before cutoff; no future-side training"
    )
    uncertainty_scope: str = "conditional_on_frozen_policy; not whole discovery-process correction"

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*_V\d{3,}", self.plan_id):
            raise ValueError("plan_id must be versioned")
        if self.timeframe not in ("5m", "15m") or self.horizon_bars < 1:
            raise ValueError("only declared 5m/15m positive observed-bar horizons supported")
        if (self.target, self.units) != ("future_return", "log_mid_return"):
            raise ValueError("direction requires expected log mid return, not volatility/residual")
        if (
            self.primary_metric != "marked_equity_change_account"
            or self.fold_boundary_mode != "continuous_carry"
        ):
            raise ValueError("only marked-equity metric and continuous carry supported")
        if self.training_window not in ("expanding", "rolling"):
            raise ValueError("declare expanding or rolling")
        if self.training_window == "rolling" and (
            self.rolling_days is None or self.rolling_days <= 0
        ):
            raise ValueError("rolling window requires positive rolling_days")
        if self.inner_selection != "mean_chronological_validation_mse_smallest_count_tie":
            raise ValueError("unsupported inner procedure")
        if self.feature_pool != FEATURE_POOL or not self.feature_counts:
            raise ValueError("only prespecified causal feature pool supported")
        if any(type(n) is not int or not 1 <= n <= len(FEATURE_POOL) for n in self.feature_counts):
            raise ValueError("invalid feature count")
        if (
            not self.policy_thresholds
            or self.policy_thresholds[0] != 0
            or len(self.policy_thresholds) > 3
        ):
            raise ValueError("primary sign threshold zero; at most two registered neighbors")
        if any(not math.isfinite(x) or x < 0 for x in self.policy_thresholds):
            raise ValueError("invalid policy threshold")
        if len(set(self.policy_thresholds)) != len(self.policy_thresholds):
            raise ValueError("duplicate threshold")
        if not 1 <= len(self.candidates) <= 2 or not 1 <= len(self.scenarios) <= 4:
            raise ValueError("small family required: 1..2 models, 1..4 scenarios")
        for candidate in self.candidates:
            if candidate.model not in ("ridge", "historical_mean") or not candidate.rationale:
                raise ValueError("unsupported model or missing rationale")
        for values in (self.candidates, self.scenarios, self.folds):
            ids = [
                getattr(v, "candidate_id", getattr(v, "scenario_id", getattr(v, "fold_id", "")))
                for v in values
            ]
            if len(ids) != len(set(ids)) or any(
                not re.fullmatch(r"[A-Za-z0-9_-]+", x) for x in ids
            ):
                raise ValueError("identities must be unique safe names")
        for scenario in self.scenarios:
            if any(
                not math.isfinite(x) or x < 1
                for x in (
                    scenario.commission_multiplier,
                    scenario.slippage_multiplier,
                    scenario.spread_multiplier,
                )
            ):
                raise ValueError("stress multipliers must be finite >= 1")
            if scenario.latency_ms is not None and (
                type(scenario.latency_ms) is not int or scenario.latency_ms < 0
            ):
                raise ValueError("invalid scenario latency")
        if not self.folds or len(self.folds) > 12:
            raise ValueError("1..12 outer folds required")
        previous = None
        for fold in self.folds:
            train = parse_utc(fold.training_start_utc)
            if not train < fold.start < fold.end or (
                previous is not None and previous != fold.start
            ):
                raise ValueError("outer folds must be chronological, contiguous, after training")
            if not 1 <= len(fold.inner_intervals) <= 3:
                raise ValueError("1..3 preceding inner intervals required")
            inner_end = train
            for a, b in fold.inner_intervals:
                start, end = parse_utc(a), parse_utc(b)
                if not train < start < end < fold.start or start < inner_end:
                    raise ValueError("inner folds must precede outer cutoff without overlap")
                inner_end = end
            previous = fold.end
        if (self.end - self.start).total_seconds() > self.max_evaluation_days * 86400:
            raise ValueError("representative evaluation exceeds frozen bounded coverage")
        positive_ints = (
            self.trial_budget,
            self.random_direction_block_opportunities,
            self.bootstrap_days,
            self.bootstrap_replicates,
            self.minimum_daily_observations,
            self.minimum_folds,
            self.minimum_closed_trades,
            self.remove_largest_trades,
            self.max_training_rows,
            self.max_evaluation_days,
        )
        if any(type(x) is not int or x < 1 for x in positive_ints):
            raise ValueError("budgets and evidence requirements must be positive integers")
        if not math.isfinite(self.ridge_alpha) or self.ridge_alpha <= 0:
            raise ValueError("positive ridge alpha required")
        if any(
            not 0 < x <= 1
            for x in (
                self.minimum_positive_fold_fraction,
                self.maximum_positive_trade_concentration,
                self.maximum_positive_fold_concentration,
            )
        ):
            raise ValueError("invalid promotion fractions")
        if self.evaluation_history_classification not in (
            "software_correctness",
            "historical_diagnostic",
            "historical_reconstructed",
        ):
            raise ValueError("this development framework cannot claim prospective evaluation")
        if not self.revision_reason or (
            self.supersedes_plan and self.supersedes_plan == self.plan_id
        ):
            raise ValueError("revisions need a distinct version and reason")
        if self.expected_trials > self.trial_budget:
            raise ValueError("planned trials exceed budget")

    @property
    def start(self) -> datetime:
        return self.folds[0].start

    @property
    def end(self) -> datetime:
        return self.folds[-1].end

    @property
    def expected_trials(self) -> int:
        fit = sum(len(f.inner_intervals) * len(self.feature_counts) + 1 for f in self.folds)
        return fit * len(self.candidates) + (
            len(self.candidates) * len(self.policy_thresholds) + 2
        ) * len(self.scenarios)

    def training_start(self, fold: Fold, cutoff: datetime) -> datetime:
        start = parse_utc(fold.training_start_utc)
        if self.training_window == "rolling":
            assert self.rolling_days is not None
            start = max(start, cutoff - timedelta(days=self.rolling_days))
        return start

    def resolved(self) -> dict[str, Any]:
        return asdict(self)


def load_plan(path: Path) -> ExperimentPlan:
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ValueError("plan must be a mapping")
    body = dict(body)
    body["folds"] = tuple(
        Fold(**{**f, "inner_intervals": tuple(tuple(x) for x in f["inner_intervals"])})
        for f in body["folds"]
    )
    body["candidates"] = tuple(Candidate(**c) for c in body["candidates"])
    body["scenarios"] = tuple(Scenario(**s) for s in body["scenarios"])
    for key in ("feature_counts", "feature_pool", "policy_thresholds"):
        if key in body:
            body[key] = tuple(body[key])
    return ExperimentPlan(**body)


def freeze_plan(root: Path, plan: ExperimentPlan, execution: ExecutionConfig) -> Path:
    """Idempotent identical design; changed design/costs under same identity raises."""
    directory = root / "results/strategy_validation/plans"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{plan.plan_id}.json"
    body = {"plan": plan.resolved(), "execution": execution.resolved()}
    digest = content_hash(body)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if (
            existing["content_sha256"] != digest
            or content_hash({k: existing[k] for k in body}) != digest
        ):
            raise ValueError("immutable plan changed: bump version and record revision reason")
    else:
        write_new_json(
            path,
            {
                **body,
                "content_sha256": digest,
                "registration_scope": "frozen before this run, after historical research; retrospective",
            },
        )
    return path
