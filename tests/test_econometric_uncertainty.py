"""Delayed and overlapping outcomes, interval meanings, reload and causal change diagnostics."""

import json
from dataclasses import replace
from datetime import timedelta

import pytest

from econometric_synth import at
from xauusd_quant.econometrics.data import Outcome
from xauusd_quant.econometrics.monitor import ErrorCUSUM
from xauusd_quant.econometrics.runs import fixture_monitor
from xauusd_quant.econometrics.uncertainty import DelayedIntervals


def outcome(key: str, index: int, value: float = 1, horizon: int = 1, delay: int = 0) -> Outcome:
    end = at(index + horizon)
    return Outcome(
        key, at(index), end, end + timedelta(seconds=delay), value, value * value, horizon, 300
    )


def seed_intervals(gamma: float = 0.0) -> DelayedIntervals:
    state = DelayedIntervals(window=8, minimum=4, gamma=gamma)
    for i in range(4):
        item = outcome(str(i), i * 2)
        state.issue(str(i), 0.0, at(i * 2), item.end_utc, item.matured_utc)
        state.deliver(item, item.matured_utc + timedelta(seconds=1))
    return state


def test_calibration_waits_for_publication_and_overlap_does_not_make_labels_available() -> None:
    state = DelayedIntervals(horizon=3)
    item = outcome("A", 0, horizon=3, delay=300)
    state.issue("A", 0, at(0), item.end_utc, item.matured_utc)
    state.issue("B", 0, at(1), at(4), at(5))
    with pytest.raises(ValueError, match="not strictly matured"):
        state.deliver(item, item.matured_utc)
    assert not state.errors and state.updates == 0
    state.deliver(item, item.matured_utc + timedelta(seconds=1))
    assert list(state.errors) == [1] and set(state.pending) == {"B"}


def test_intervals_are_future_outcome_ranges_not_mean_confidence_bounds() -> None:
    state = seed_intervals()
    record = state.issue("next", 0.1, at(8), at(9), at(9))
    assert record["lower"] == pytest.approx(-0.9) and record["upper"] == pytest.approx(1.1)
    assert record["width"] == 2
    assert record["type"] == "future_outcome_prediction_interval"
    assert "not conditional-mean" in record["guarantee"]
    assert record["calibration_as_of_utc"] < record["issued_utc"]


def test_adaptive_alpha_responds_only_after_a_matured_miss() -> None:
    state = seed_intervals(0.01)
    before = state.alpha
    record = state.issue("miss", 0, at(8), at(9), at(9))
    assert record["current_alpha"] == before
    update = state.deliver(outcome("miss", 8, 3), at(9.1))
    assert update is not None and not update["covered"]
    assert state.alpha == pytest.approx(0.192)


def test_calibration_reload_preserves_pending_delays_and_next_intervals() -> None:
    original = seed_intervals(0.01)
    original.issue("pending", 0, at(8), at(9), at(9))
    restored = DelayedIntervals.restore(json.loads(json.dumps(original.resolved())))
    item = outcome("pending", 8, 0.5)
    assert original.deliver(item, at(9.1)) == restored.deliver(item, at(9.1))
    assert original.issue("next", 0, at(10), at(11), at(11)) == restored.issue(
        "next", 0, at(10), at(11), at(11)
    )
    assert original.resolved() == restored.resolved()


def test_delayed_delivery_cannot_be_used_for_a_backdated_decision() -> None:
    state = DelayedIntervals()
    item = outcome("old", 0)
    state.issue("old", 0, at(0), at(1), at(1))
    state.deliver(item, at(4))
    with pytest.raises(ValueError, match="backdate"):
        state.issue("backdated", 0, at(2), at(3), at(3))


def test_missing_outcomes_expire_without_residual_imputation_and_wrong_horizon_rejects() -> None:
    state = seed_intervals()
    state.issue("missing", 0, at(8), at(9), at(9))
    before = list(state.errors)
    assert state.expire(at(10)) == ["missing"]
    assert list(state.errors) == before
    state.issue("bad", 0, at(10), at(11), at(11))
    with pytest.raises(ValueError, match="horizon/publication mismatch"):
        state.deliver(replace(outcome("bad", 10), horizon_bars=2), at(12))


def test_cusum_waits_for_matured_errors_and_matches_reload() -> None:
    state = ErrorCUSUM(1.0)
    with pytest.raises(ValueError, match="unmatured"):
        state.update(3, at(1), at(1))
    state.update(0.5, at(1), at(1.1))
    restored = ErrorCUSUM.restore(json.loads(json.dumps(state.resolved())))
    for i in range(2, 8):
        assert state.update(3.0, at(i), at(i + 0.1)) == restored.update(3.0, at(i), at(i + 0.1))
    assert state.alarms > 0


def test_fixed_changed_and_unchanged_controls_report_false_alarms_and_delay() -> None:
    report = fixture_monitor(135013)
    assert report == fixture_monitor(135013)
    assert report["changed"]["first_postchange_delay"] is not None
    assert 0 <= report["changed"]["first_postchange_delay"] <= 10
    assert (
        report["unchanged"]["false_alarms_before_change"]
        == report["changed"]["false_alarms_before_change"]
    )
    assert "not passing real pipeline null" in report["interpretation"]
