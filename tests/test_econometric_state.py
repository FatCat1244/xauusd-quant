"""Causal recursive state prefixes, partition carry, reload and numerical failures."""

import json
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from econometric_synth import at, grid, returns_process
from xauusd_quant.econometrics.runs import Model
from xauusd_quant.econometrics.state_space import LocalLevelFit, LocalLevelState, fit_local_level
from xauusd_quant.econometrics.volatility import VarianceState


def filter_states(values: np.ndarray, *, reload_at: int | None = None) -> list[dict]:
    state = LocalLevelState(LocalLevelFit(1e-7, 2e-7, {}))
    output = []
    for i, value in enumerate(values):
        if i == reload_at:
            state = LocalLevelState.restore(json.loads(json.dumps(state.resolved())))
        output.append(state.observe(float(value), at(i), contiguous=i != 0))
    return output


def test_filtered_state_prefix_is_unchanged_by_future_observations() -> None:
    rng = np.random.default_rng(39)
    latent = np.cumsum(rng.normal(0.00005, 0.0002, 500))
    observed = latent + rng.normal(0, 0.0004, 500)
    first = filter_states(observed[:250])
    extended = filter_states(np.r_[observed, np.full(100, 1000)])
    assert first == extended[:250]
    assert all(row["mode"] == "causal_filter_no_smoothing" for row in first)


def test_replay_observes_current_bar_only_even_when_future_bars_are_in_memory() -> None:
    data = grid(300)
    model = Model(
        "KALMAN", LocalLevelState(LocalLevelFit(1e-7, 2e-7, {})), {"fitting_cutoff_utc": at(0)}
    )
    for i in range(40):
        emitted = model.observe(data, i)
        assert emitted is not None
        prefix = grid(300)
        # A future price alteration cannot change any emitted current state.
        prefix.logs[i + 1 :] = 10000
        other = Model(
            "KALMAN", LocalLevelState(LocalLevelFit(1e-7, 2e-7, {})), {"fitting_cutoff_utc": at(0)}
        )
        for j in range(i + 1):
            record = other.observe(prefix, j)
        assert record == emitted
    point = data.points(at(40), at(50))[0]
    with pytest.raises(ValueError, match="not available"):
        model.predict(point, point.available_utc.replace(microsecond=0) - timedelta(seconds=1))


def test_filter_and_variance_state_batch_reload_are_exact() -> None:
    values = grid(300).logs
    assert filter_states(values) == filter_states(values, reload_at=137)
    continuous = VarianceState("EWMA", 1e-7, 3)
    partitioned = VarianceState("EWMA", 1e-7, 3)
    returns = returns_process(300)
    for i, value in enumerate(returns):
        continuous.observe(float(value), at(i), contiguous=True)
    for a, b in ((0, 63), (63, 200), (200, 300)):
        for i in range(a, b):
            partitioned.observe(float(returns[i]), at(i), contiguous=True)
        partitioned = VarianceState.restore(json.loads(json.dumps(partitioned.resolved())))
    assert continuous.resolved() == partitioned.resolved()
    assert continuous.forecast() == partitioned.forecast()


def test_session_gap_resets_filter_without_fabricating_observations() -> None:
    state = LocalLevelState(LocalLevelFit(0.01, 0.02, {}))
    state.observe(1, at(0), contiguous=False)
    state.observe(2, at(1), contiguous=True)
    reset = state.observe(10, at(300), contiguous=False)
    assert reset["gap_reset"] and reset["filtered_log_level"] == 10
    assert state.forecast_return() == 0
    assert reset["innovation"] is None


def test_parameter_fit_on_drifting_latent_fixture_and_explicit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(24)
    level = np.cumsum(rng.normal(0.00002, 0.0002, 600))
    observed = level + rng.normal(0, 0.0004, 600)
    fit = fit_local_level([observed])
    assert fit.process_variance > 0 and fit.observation_variance > 0
    with pytest.raises(ValueError, match="insufficient"):
        fit_local_level([observed[:10]])
    monkeypatch.setattr(
        "xauusd_quant.econometrics.state_space.minimize",
        lambda *a, **k: SimpleNamespace(success=False, message="forced failure"),
    )
    with pytest.raises(ValueError, match="nonconvergence"):
        fit_local_level([observed])


def test_nonpositive_or_unordered_state_updates_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        LocalLevelFit(0, 0.1, {})
    state = LocalLevelState(LocalLevelFit(0.1, 0.1, {}))
    state.observe(1, at(1), contiguous=False)
    with pytest.raises(ValueError, match="nonchronological"):
        state.observe(2, at(1), contiguous=True)
