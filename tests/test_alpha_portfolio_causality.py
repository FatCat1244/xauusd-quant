"""Availability, fold-local eligibility and independently checked allocation."""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from xauusd_quant.alpha_portfolio.allocation import Allocation, fit, minimum_variance
from xauusd_quant.alpha_portfolio.intents import Intent, ObservedBarPolicy
from xauusd_quant.alpha_portfolio.portfolio import Portfolio, WeightUpdate
from xauusd_quant.alpha_portfolio.registry import ELIGIBLE, SURVIVES, Alpha, policy
from xauusd_quant.alpha_portfolio.runs import universe_at
from xauusd_quant.execution.config import ExecutionConfig
from xauusd_quant.execution.engine import MemoryRecorder
from xauusd_quant.execution.policy import Forecast

T = datetime(2021, 1, 4, tzinfo=UTC)


def candidate() -> Alpha:
    roles = ("model", "features", "data", "chronology", "nulls", "economics", "execution")
    return Alpha(
        "A",
        "A_V001",
        "5m",
        1,
        {**policy("5m", 1), "maximum_age_seconds": 600},
        {**dict.fromkeys(roles, "identity"), "gates": dict.fromkeys(roles, "passed")},
        SURVIVES,
        ELIGIBLE,
        (T - timedelta(seconds=1)).isoformat(),
        (),
    )


def test_fold_eligibility_cannot_backdate_stage14_universe() -> None:
    a = candidate()
    assert universe_at([a], T) == (a,)
    assert universe_at([a], T - timedelta(seconds=2)) == ()
    assert universe_at([replace(a, evidence_available_utc=T.isoformat())], T) == ()
    assert universe_at([replace(a, evidence_available_utc=None)], T) == ()


@pytest.mark.parametrize("field", ["synthetic", "diagnostic_only"])
def test_fixtures_and_diagnostics_never_scientifically_eligible(field: str) -> None:
    assert not replace(candidate(), **{field: True}).eligible_at(T)


def test_missing_negative_evidence_never_promotes() -> None:
    a = candidate()
    assert not replace(a, stage14_verdict="BLOCKED").eligible_at(T)
    assert not replace(a, provenance={}).eligible_at(T)
    assert not replace(a, eligibility="REJECTED").eligible_at(T)
    assert not replace(a, software_ready=False).eligible_at(T)
    assert not replace(a, policy={}).eligible_at(T)
    assert not replace(a, policy={**a.policy, "maximum_age_seconds": None}).eligible_at(T)


def risk_data() -> tuple[list[datetime], list[datetime], np.ndarray]:
    stamps = [T - timedelta(seconds=(40 - i) * 300) for i in range(40)]
    values = np.random.default_rng(15).normal(0, 0.001, (40, 2))
    return stamps, stamps.copy(), values


def test_independent_minimum_variance_arithmetic_and_constraint_boundary() -> None:
    # Independent two-sleeve scalar derivation: w1=(v2-c)/(v1+v2-2c).
    c = np.array([[4.0, 1.0], [1.0, 9.0]])
    assert minimum_variance(c) == pytest.approx([8 / 11, 3 / 11])
    assert minimum_variance(np.array([[1.0, 2.0], [2.0, 5.0]])) == pytest.approx([1, 0])
    assert minimum_variance(np.array([[2.0]])) == pytest.approx([1])


@pytest.mark.parametrize(
    "c",
    [
        np.zeros((2, 2)),
        np.array([[1.0, 2.0], [2.0, 1.0]]),
        np.array([[1.0, float("nan")], [0.0, 1.0]]),
        np.ones((2, 3)),
    ],
)
def test_invalid_covariance_explicit_rejection(c: np.ndarray) -> None:
    with pytest.raises(ValueError):
        minimum_variance(c)


def test_future_append_no_change_to_frozen_weight_identity() -> None:
    stamps, matured, values = risk_data()
    a = fit(("A", "B"), "minimum_variance_shrunk", T, stamps, matured, values)
    b = fit(
        ("A", "B"),
        "minimum_variance_shrunk",
        T,
        [*stamps, T + timedelta(seconds=300)],
        [*matured, T + timedelta(seconds=400)],
        np.vstack([values, [100, -100]]),
    )
    assert a.resolved() == b.resolved()
    sample = values[-32:]
    s = np.cov(sample, rowvar=False)
    assert np.asarray(a.diagnostics["shrunk_covariance"]) == pytest.approx(
        0.5 * s + 0.5 * np.diag(np.diag(s))
    )


def test_matured_returns_required_for_weights() -> None:
    stamps, matured, values = risk_data()
    normal = fit(("A", "B"), "minimum_variance_shrunk", T, stamps, matured, values)
    matured[-1] = T  # last otherwise-prior observation not published before cutoff
    delayed = fit(("A", "B"), "minimum_variance_shrunk", T, stamps, matured, values)
    assert delayed.diagnostics["training_sha256"] != normal.diagnostics["training_sha256"]
    assert delayed.information_as_of_utc < T
    values[-1] = [1000, -1000]
    assert (
        fit(("A", "B"), "minimum_variance_shrunk", T, stamps, matured, values).resolved()
        == delayed.resolved()
    )


def test_fallback_declared_for_missing_singular_and_solver_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stamps, matured, values = risk_data()
    values[:] = 0
    a = fit(("A", "B"), "minimum_variance_shrunk", T, stamps, matured, values)
    assert a.weights == {"A": 0.5, "B": 0.5}
    assert "equal:" in a.diagnostics["fallback"]
    assert fit(("A", "B"), "minimum_variance_shrunk", T, [], [], np.empty((0, 2))).diagnostics[
        "fallback"
    ]

    def fail(_: np.ndarray) -> np.ndarray:
        raise np.linalg.LinAlgError("fixture failed solve")

    monkeypatch.setattr("xauusd_quant.alpha_portfolio.allocation.minimum_variance", fail)
    _, _, valid = risk_data()
    assert (
        "failed solve"
        in fit(("A", "B"), "minimum_variance_shrunk", T, stamps, matured, valid).diagnostics[
            "fallback"
        ]
    )


def test_missing_rows_complete_case_no_silent_imputation() -> None:
    stamps, matured, values = risk_data()
    values[-1, 0] = np.nan
    a = fit(("A", "B"), "minimum_variance_shrunk", T, stamps, matured, values)
    assert a.diagnostics["rows"] == 31 and a.diagnostics["missing_rows"] == 1


def test_weight_information_cannot_equal_or_follow_effective_time() -> None:
    with pytest.raises(ValueError, match="prior"):
        Allocation("BAD_V001", T, T, {"A": 1}, "equal", {})
    with pytest.raises(ValueError, match="fully allocated"):
        Allocation("BAD_V001", T, T - timedelta(seconds=1), {"A": 1.000001}, "equal", {})


def test_different_frequency_observed_holding_exit_and_availability() -> None:
    for timeframe, seconds in (("5m", 300), ("15m", 900)):
        config = ExecutionConfig(timeframe=timeframe)
        p = ObservedBarPolicy("A", "A_V001", config, 2 * seconds)
        f = Forecast.from_bar(
            forecast_id="F", bar_open_utc=T, bar_index=0, value=0.001, config=config
        )
        p.on_bar(0, T + timedelta(seconds=seconds))
        with pytest.raises(ValueError, match="available"):
            p.on_forecast(f, T + timedelta(seconds=seconds - 1))
        i = p.on_forecast(f, T + timedelta(seconds=seconds))
        assert i and i.target == 1
        assert p.on_forecast(f, T + timedelta(seconds=seconds + 1)) is None
        # Next *observed* bar may be delayed; no future bar used to fix its exit.
        exit_intent = p.on_bar(1, T + timedelta(seconds=seconds + seconds // 2))
        assert exit_intent and exit_intent.target == 0


def test_policy_invalid_forecast_clears_and_does_not_use_volatility_direction() -> None:
    config = ExecutionConfig()
    p = ObservedBarPolicy("A", "A_V001", config, 1000)
    at = T + timedelta(seconds=300)
    p.on_bar(0, at)
    f = Forecast.from_bar(forecast_id="F", bar_open_utc=T, bar_index=0, value=0.1, config=config)
    assert p.on_forecast(f, at)
    i = p.on_forecast(replace(f, target="future_volatility", units="log_volatility"), at)
    assert i and i.target == 0 and i.state == "invalid"


def test_future_publication_refused_after_bar_already_completed() -> None:
    config = ExecutionConfig()
    p = ObservedBarPolicy("A", "A_V001", config, 1000)
    at = T + timedelta(seconds=300)
    p.on_bar(0, at)
    f = Forecast.from_bar(forecast_id="F", bar_open_utc=T, bar_index=0, value=0.1, config=config)
    with pytest.raises(ValueError, match="available"):
        p.on_forecast(
            replace(f, available_at_utc=at + timedelta(seconds=100)), at + timedelta(seconds=50)
        )


def test_scientific_constructor_refuses_future_universe_and_incompatible_intent() -> None:
    alpha = candidate()
    sink = MemoryRecorder()
    with pytest.raises(ValueError, match="ineligible"):
        Portfolio.from_alphas(
            ExecutionConfig(), (replace(alpha, stage14_verdict="BLOCKED"),), T, 0.02, sink
        )
    with pytest.raises(ValueError, match="ineligible"):
        Portfolio.from_alphas(ExecutionConfig(), (alpha,), T - timedelta(seconds=2), 0.02, sink)
    p = Portfolio.from_alphas(ExecutionConfig(), (alpha,), T, 0.02, sink)
    p.consume(
        [WeightUpdate(Allocation("A_V001", T, T - timedelta(seconds=1), {"A": 1}, "equal", {}))]
    )
    with pytest.raises(ValueError, match="horizon"):
        p.consume(
            [
                Intent(
                    "A",
                    "A_V001",
                    T,
                    T,
                    T + timedelta(seconds=1),
                    1.0,
                    "active",
                    "synthetic",
                    "15m",
                    1,
                )
            ]
        )


def test_three_sleeves_singular_empirical_shrinkage_remains_bounded() -> None:
    weights = minimum_variance(np.diag([1.0, 2.0, 4.0]))
    assert weights == pytest.approx([4 / 7, 2 / 7, 1 / 7])
    stamps, matured, values = risk_data()
    duplicated = np.column_stack([values[:, 0], values[:, 0], values[:, 1]])
    allocation = fit(("A", "B", "C"), "minimum_variance_shrunk", T, stamps, matured, duplicated)
    assert allocation.diagnostics["fallback"] is None
    assert sum(allocation.weights.values()) == pytest.approx(1)


def test_policy_reload_and_stale_publication() -> None:
    config = ExecutionConfig()
    p = ObservedBarPolicy("A", "A_V001", config, 10)
    at = T + timedelta(seconds=300)
    p.on_bar(0, at)
    f = Forecast.from_bar(forecast_id="F", bar_open_utc=T, bar_index=0, value=0.1, config=config)
    p.on_forecast(f, at)
    q = ObservedBarPolicy("A", "A_V001", config, 10)
    q.restore(json.loads(json.dumps(p.state(), default=lambda t: t.isoformat())))
    assert q.on_bar(1, at + timedelta(seconds=1)) == p.on_bar(1, at + timedelta(seconds=1))
    stale = q.on_forecast(f, at + timedelta(seconds=20))
    assert stale and stale.target == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"target": float("nan")},
        {"target": 2},
        {"units": "probability"},
        {"available_utc": T - timedelta(seconds=1)},
        {"state": "invalid", "target": 1},
    ],
)
def test_intent_contract_rejects_unsupported_inputs(changes: dict[str, object]) -> None:
    body = {
        "alpha_id": "A",
        "specification_id": "A_V001",
        "decision_utc": T,
        "available_utc": T,
        "valid_until_utc": T + timedelta(seconds=1),
        "target": 1.0,
        "state": "active",
        "rationale": "synthetic",
        "timeframe": "5m",
        "horizon_bars": 1,
    }
    with pytest.raises(ValueError):
        Intent(**{**body, **changes})
