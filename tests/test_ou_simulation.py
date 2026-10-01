"""The OU simulator, parameter recovery, and the controls.

The simulator is the ruler every other result is measured against, so it is
checked against its own closed forms: innovation variance, stationary
variance, autocorrelation b^h, and a deterministic seed. The estimator must
then recover known parameters from long samples, and the random-walk negative
control must never be passed off as strong mean reversion.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from xauusd_quant.features.rolling_regression import rolling_ols
from xauusd_quant.models import load_ou_config
from xauusd_quant.models.ornstein_uhlenbeck import (
    OUValidityRules,
    ar1_coefficients,
    decay_ratio,
    expected_value,
    fit_ar1,
    innovation_variance,
    simulate_ar1,
    simulate_ou,
    simulate_random_walk,
)
from xauusd_quant.research.ou_estimation import control_series, ou_parameter_frame

RULES = OUValidityRules()


def test_the_simulator_uses_the_exact_innovation_variance():
    theta, mu, sigma, dt = 0.3, 1.5, 0.8, 2.0
    x = simulate_ou(theta, mu, sigma, 400_000, dt=dt, seed=1)
    a, b = ar1_coefficients(theta, mu, dt)
    innovations = x[1:] - (a + b * x[:-1])
    expected = sigma ** 2 * (1 - math.exp(-2 * theta * dt)) / (2 * theta)
    assert innovation_variance(theta, sigma, dt) == pytest.approx(expected)
    assert innovations.var() == pytest.approx(expected, rel=0.01)


def test_the_simulated_path_is_stationary_from_the_first_value():
    theta, mu, sigma = 0.05, -2.0, 0.4
    starts = np.array([simulate_ou(theta, mu, sigma, 1, seed=s)[0] for s in range(4_000)])
    assert starts.mean() == pytest.approx(mu, abs=0.08)
    assert starts.std() == pytest.approx(sigma / math.sqrt(2 * theta), rel=0.05)


def test_the_autocorrelation_decays_as_b_to_the_h():
    x = simulate_ou(0.2, 0.0, 1.0, 300_000, seed=4)
    for h in (1, 3, 10):
        rho = np.corrcoef(x[h:], x[:-h])[0, 1]
        assert rho == pytest.approx(math.exp(-0.2 * h), abs=0.01)


def test_a_seed_makes_the_simulation_deterministic():
    np.testing.assert_array_equal(simulate_ou(0.1, 0, 1, 1000, seed=5),
                                  simulate_ou(0.1, 0, 1, 1000, seed=5))
    assert not np.array_equal(simulate_ou(0.1, 0, 1, 1000, seed=5),
                              simulate_ou(0.1, 0, 1, 1000, seed=6))


@pytest.mark.parametrize(("theta", "mu", "sigma"), [(0.1, 0.0, 0.3), (0.02, 5.0, 1.0),
                                                     (0.7, -1.0, 0.05)])
def test_known_parameters_are_recovered_from_a_long_sample(theta, mu, sigma):
    x = simulate_ou(theta, mu, sigma, 400_000, seed=12)
    fit = fit_ar1(x, rules=RULES)
    assert fit.ou.state == "valid"
    assert fit.ou.theta == pytest.approx(theta, rel=0.06)
    assert fit.ou.mu == pytest.approx(mu, abs=4 * sigma / math.sqrt(2 * theta) / 50)
    assert fit.ou.sigma == pytest.approx(sigma, rel=0.02)
    assert fit.std_ratio == pytest.approx(1.0, abs=0.03)


def test_small_windows_are_biased_toward_faster_reversion():
    """The finite-sample (Kendall) bias the OU reference exists to measure."""
    theta = 0.05
    x = simulate_ou(theta, 0.0, 1.0, 100_000, seed=3)
    frame = ou_parameter_frame(x, None, ou_window=64, config=load_ou_config())
    median_b = float(np.median(frame["ou_b"].drop_nulls().to_numpy()))
    true_b = math.exp(-theta)
    kendall = true_b - (1 + 3 * true_b) / 63
    assert median_b < true_b - 0.03, "a 64-bar window must underestimate b"
    assert median_b == pytest.approx(kendall, abs=0.02)


def test_expected_path_and_decay_ratio():
    assert decay_ratio(0.1, 5) == pytest.approx(math.exp(-0.5))
    assert expected_value(3.0, 1.0, 0.1, 5) == pytest.approx(1.0 + 2.0 * math.exp(-0.5))
    theta = -math.log(0.9)
    hl = math.log(2) / theta
    for k, remaining in ((1, 0.5), (2, 0.25), (3, 0.125)):
        assert decay_ratio(theta, k * hl) == pytest.approx(remaining)


def _median_half_life(x: np.ndarray, window: int) -> float:
    frame = ou_parameter_frame(x, None, ou_window=window, config=load_ou_config())
    return float(np.median(frame["ou_half_life_bars"].drop_nulls().to_numpy()))


def test_random_walk_negative_control_is_not_strong_mean_reversion():
    """A raw random walk: b ~ 1 and no unit-root rejection on the whole sample."""
    walk = simulate_random_walk(50_000, 1.0, seed=8)
    fit = fit_ar1(walk, rules=RULES)
    assert fit.b > 0.999
    assert fit.df_statistic > -3.5
    frame = ou_parameter_frame(walk, None, ou_window=512, config=load_ou_config())
    assert np.median(frame["ou_b"].drop_nulls().to_numpy()) > 0.98


def test_a_windowed_half_life_of_a_random_walk_scales_with_the_window():
    """The diagnostic that separates an artefact from a property.

    In-window OLS finds b < 1 even for a random walk (the Dickey-Fuller
    bias), so every window reports a finite half-life - but it grows roughly
    in proportion to the window, because it is a property of the window. A
    genuine OU's estimates converge to the true half-life instead.
    """
    walk = simulate_random_walk(200_000, 1.0, seed=8)
    walk_ratio = _median_half_life(walk, 512) / _median_half_life(walk, 64)
    ou = simulate_ou(-math.log(0.933), 0.0, 1.0, 200_000, seed=8)   # half-life 10 bars
    ou_ratio = _median_half_life(ou, 512) / _median_half_life(ou, 64)
    assert walk_ratio > 4.0
    assert ou_ratio < 1.8
    assert _median_half_life(ou, 512) == pytest.approx(10.0, rel=0.15)


def test_a_detrended_random_walk_looks_mean_reverting_which_is_why_it_is_the_control():
    """Detrending alone manufactures a finite 'half-life' - the core caveat."""
    walk = simulate_random_walk(60_000, 1e-3, x0=7.0, seed=2)
    residual = rolling_ols(walk, 128).residual
    fit = fit_ar1(residual, rules=RULES)
    assert fit.ou.state == "valid"
    assert 5 < fit.ou.half_life_bars < 200, (
        "a detrended random walk shows a finite half-life purely by construction"
    )


def test_controls_have_the_requested_length_scale_and_determinism():
    returns = np.random.default_rng(1).normal(0, 2e-4, 5_000)
    a = control_series("control_random_walk", 5_000, regression_window=64,
                       log_returns=returns, reference_fit=None, seed=7)
    b = control_series("control_random_walk", 5_000, regression_window=64,
                       log_returns=returns, reference_fit=None, seed=7)
    assert a is not None and a.size == 5_000
    np.testing.assert_array_equal(a, b)
    assert np.isnan(a[:63]).all() and np.isfinite(a[63:]).all()
    shuffled = control_series("control_shuffled_returns", 5_000, regression_window=64,
                              log_returns=returns, reference_fit=None, seed=7)
    assert shuffled is not None and np.isfinite(shuffled[63:]).all()
    assert control_series("reference_ou", 5_000, regression_window=64, log_returns=returns,
                          reference_fit=None, seed=7) is None


def test_ar1_simulator_matches_its_recursion():
    x = simulate_ar1(0.4, -0.5, 0.0, 6, x0=2.0)
    expected = [2.0]
    for _ in range(5):
        expected.append(0.4 - 0.5 * expected[-1])
    np.testing.assert_allclose(x, expected)
