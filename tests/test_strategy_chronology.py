"""Leakage tests validate permitted inputs, not just scored model chronology."""

from dataclasses import replace
from typing import Any

import pytest

from strategy_synth import at, fixture
from xauusd_quant.execution.readiness import ADAPTIVE_CHOICES
from xauusd_quant.strategy_validation.pipeline import Selection, fit, fit_fold, purge
from xauusd_quant.strategy_validation.readiness import validate_adaptive_evidence


def direct(kind: str, spec: dict[str, Any], operation: Any) -> Any:
    return operation()


def test_deliberately_leaking_feature_selector_is_rejected() -> None:
    plan, _, source, _ = fixture()
    fold = plan.folds[0]
    rows = purge(source.training(at(0), fold.start), at(0), fold.start)

    def selector(rows: Any, count: int) -> Selection:
        return Selection(("return_1",), fold.start, tuple(r.row.row_id for r in rows))

    with pytest.raises(ValueError, match="leaking feature selector"):
        fit(rows, plan.candidates[0], fold, 1, fold.start, plan, selector)

    def identities(rows: Any, count: int) -> Selection:
        return Selection(("return_1",), rows[-1].label_available_at_utc, ("outer-outcome",))

    with pytest.raises(ValueError, match="outside permitted training"):
        fit(rows, plan.candidates[0], fold, 1, fold.start, plan, identities)


def test_label_intervals_and_delayed_publication_crossing_cutoff_are_purged() -> None:
    _, _, source, _ = fixture()
    sample = source.training(at(0), at(200))[0]
    cutoff = at(100)
    crossing = replace(sample, label_end_utc=cutoff, label_available_at_utc=cutoff)
    delayed = replace(sample, label_available_at_utc=at(101))
    valid = replace(sample, label_end_utc=at(99), label_available_at_utc=at(99))
    assert purge([crossing, delayed, valid], at(0), cutoff) == [valid]


def test_inner_selection_never_requests_outer_outcomes() -> None:
    plan, _, source, _ = fixture()
    fold = plan.folds[0]
    original = source.training

    def guarded(start: Any, cutoff: Any) -> Any:
        assert cutoff <= fold.start, "inner selection requested outer outcomes"
        return original(start, cutoff)

    source.training = guarded
    pipeline = fit_fold(source, plan, fold, plan.candidates[0], direct)
    assert pipeline.training_label_as_of_utc < fold.start
    assert all(r["cutoff"] < fold.start for r in pipeline.inner_trials)
    assert all(access["kind"] == "training" for access in source.accesses)


def test_later_data_cannot_change_an_earlier_frozen_fold() -> None:
    plan, c, source, _ = fixture()
    before = fit_fold(source, plan, plan.folds[0], plan.candidates[0], direct)
    from xauusd_quant.strategy_validation.source import SyntheticSource

    future = SyntheticSource(
        [*source.opens, at(1500), at(1505)], [*source.closes, 1e8, 1e-8], c, plan.horizon_bars
    )
    after = fit_fold(future, plan, plan.folds[0], plan.candidates[0], direct)
    assert before == after
    assert source.evaluation(plan.start, plan.end) == future.evaluation(plan.start, plan.end)


@pytest.mark.parametrize("choice", ADAPTIVE_CHOICES)
def test_outcome_dependent_ensemble_and_adaptive_choices_respect_cutoff(choice: str) -> None:
    records = dict.fromkeys(ADAPTIVE_CHOICES, at(500).isoformat())
    assert validate_adaptive_evidence(records, at(900)) == "passed"
    records[choice] = at(1000).isoformat()
    assert validate_adaptive_evidence(records, at(900)) == "failed"
    del records[choice]
    assert validate_adaptive_evidence(records, at(900)) == "unknown"
