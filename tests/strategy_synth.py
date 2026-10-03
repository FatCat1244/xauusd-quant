"""Small Stage 13 fixtures with declared chronology and costs."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

from xauusd_quant.execution.config import ExecutionConfig
from xauusd_quant.strategy_validation.plan import Candidate, ExperimentPlan, Fold, Scenario
from xauusd_quant.strategy_validation.runs import synthetic_inputs

T = datetime(2021, 1, 4, tzinfo=UTC)


def at(minutes: float) -> datetime:
    return T + timedelta(minutes=minutes)


def fixture() -> tuple[ExperimentPlan, ExecutionConfig, Any, Any]:
    folds = tuple(
        Fold(
            f"F{i + 1}",
            at(0).isoformat(),
            at(900 + 60 * i).isoformat(),
            at(960 + 60 * i).isoformat(),
            (
                (at(240).isoformat(), at(360).isoformat()),
                (at(480).isoformat(), at(600).isoformat()),
            ),
        )
        for i in range(2)
    )
    plan = ExperimentPlan(
        "SYNTHETIC_PLAN_V001",
        "Does fixed policy survive declared costs?",
        "5m",
        1,
        folds,
        (Candidate("RIDGE", "ridge", "small declared causal pool"),),
        (Scenario("BASE"),),
        evaluation_history_classification="software_correctness",
    )
    execution = replace(ExecutionConfig(), max_gap_ms=10_000_000, max_quote_age_ms=10_000_000)
    source, quotes = synthetic_inputs(plan, execution)
    return plan, execution, source, quotes


def gates() -> dict[str, str]:
    return dict.fromkeys(
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
