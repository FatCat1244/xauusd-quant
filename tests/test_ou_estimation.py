"""The OU estimator is right before it is pointed at data.

The static fit must agree with statsmodels to machine precision (coefficients,
OLS and Newey-West standard errors); the rolling fit must agree with an
independent per-window fit at every bar; and the AR(1) -> OU mapping must
reproduce its closed forms exactly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import statsmodels.api as sm

from xauusd_quant.models.ornstein_uhlenbeck import (
    OUValidityRules,
    fit_ar1,
    fit_ar1_pairs,
    innovation_variance,
    map_ar1_to_ou,
    rolling_ar1,
    rolling_ou,
    simulate_ar1,
    simulate_ou,
    stationary_std,
)

RULES = OUValidityRules()


def test_static_fit_matches_statsmodels_exactly():
    x = simulate_ar1(0.3, 0.8, 1.5, 5_000, seed=11)
    fit = fit_ar1(x, rules=RULES, hac_lags=6)
    ols = sm.OLS(x[1:], sm.add_constant(x[:-1])).fit()
    assert fit.a == pytest.approx(ols.params[0], rel=1e-10)
    assert fit.b == pytest.approx(ols.params[1], rel=1e-10)
    assert fit.se_a == pytest.approx(ols.bse[0], rel=1e-8)
    assert fit.se_b == pytest.approx(ols.bse[1], rel=1e-8)
    assert fit.r_squared == pytest.approx(ols.rsquared, rel=1e-10)
    assert fit.innovation_var == pytest.approx(ols.scale, rel=1e-10)
    hac = sm.OLS(x[1:], sm.add_constant(x[:-1])).fit(
        cov_type="HAC", cov_kwds={"maxlags": 6, "use_correction": False})
    assert fit.hac_se_a == pytest.approx(hac.bse[0], rel=1e-8)
    assert fit.hac_se_b == pytest.approx(hac.bse[1], rel=1e-8)


def test_t_statistics_and_the_dickey_fuller_form():
    x = simulate_ar1(0.0, 0.6, 1.0, 3_000, seed=2)
    fit = fit_ar1(x, rules=RULES)
    assert fit.t_b == pytest.approx(fit.b / fit.se_b)
    assert fit.df_statistic == pytest.approx((fit.b - 1.0) / fit.se_b)


@pytest.mark.parametrize("window", [4, 64, 257])
def test_rolling_fit_equals_an_independent_fit_of_every_window(window):
    x = simulate_ou(0.08, 1.0, 0.2, 1_500, seed=4)
    rolling = rolling_ar1(x, window, chunk_rows=1024)
    for t in range(window - 1, x.size, 37):
        segment = x[t - window + 1: t + 1]
        ols = sm.OLS(segment[1:], sm.add_constant(segment[:-1])).fit()
        assert rolling.b[t] == pytest.approx(ols.params[1], rel=1e-9, abs=1e-12)
        assert rolling.a[t] == pytest.approx(ols.params[0], rel=1e-9, abs=1e-12)
        assert rolling.innovation_var[t] == pytest.approx(ols.scale, rel=1e-8)
        assert rolling.r_squared[t] == pytest.approx(ols.rsquared, rel=1e-8, abs=1e-12)
        assert rolling.window_mean[t] == pytest.approx(segment.mean(), rel=1e-10, abs=1e-12)
        assert rolling.window_std[t] == pytest.approx(segment.std(ddof=1), rel=1e-9)


def test_rolling_fit_is_accurate_far_from_zero_and_across_blocks():
    """A large offset and small noise stress the shifted prefix sums."""
    x = 5_000.0 + simulate_ou(0.05, 0.0, 1e-3, 40_000, seed=9)
    rolling = rolling_ar1(x, 128, chunk_rows=1024)
    for t in (127, 1023, 1024 + 127, 20_000, 39_999):
        segment = x[t - 127: t + 1]
        ols = sm.OLS(segment[1:], sm.add_constant(segment[:-1])).fit()
        assert rolling.b[t] == pytest.approx(ols.params[1], rel=1e-7)


def test_block_size_changes_only_rounding():
    """Longer blocks accumulate longer prefix sums, so the last bits may move.

    Measured at ~1e-11 relative here. Exact, bit-for-bit equality is promised
    only between runs with the same block structure - which is what the
    no-look-ahead guarantee needs (a longer series never changes the blocks
    an earlier output belongs to) and what tests/test_ou_no_leakage.py checks.
    """
    x = simulate_ou(0.1, 0.5, 0.3, 30_000, seed=5)
    a = rolling_ar1(x, 64, chunk_rows=1024)
    b = rolling_ar1(x, 64, chunk_rows=100_000)
    np.testing.assert_allclose(a.b, b.b, rtol=1e-9, atol=1e-12, equal_nan=True)


# ---------------------------------------------------------------------------
# The AR(1) -> OU mapping
# ---------------------------------------------------------------------------
def test_mapping_reproduces_the_closed_forms():
    a, b, var = 0.05, 0.9, 0.04
    mapping = map_ar1_to_ou(a, b, var, rules=RULES)
    theta = -math.log(b)
    assert mapping.state[0] == "valid"
    assert mapping.theta[0] == pytest.approx(theta)
    assert mapping.mu[0] == pytest.approx(a / (1 - b))
    sigma = math.sqrt(var * 2 * theta / (1 - math.exp(-2 * theta)))
    assert mapping.sigma[0] == pytest.approx(sigma)
    assert mapping.stationary_std[0] == pytest.approx(sigma / math.sqrt(2 * theta))
    assert mapping.stationary_std[0] == pytest.approx(math.sqrt(var / (1 - b * b)))
    assert mapping.half_life_bars[0] == pytest.approx(math.log(2) / theta)
    assert mapping.mr_per_bar[0] == pytest.approx(1 - b)


def test_sigma_mapping_round_trips_through_the_innovation_variance():
    theta, sigma = 0.07, 0.35
    var = innovation_variance(theta, sigma)
    b = math.exp(-theta)
    mapping = map_ar1_to_ou(0.0, b, var, rules=RULES)
    assert mapping.sigma[0] == pytest.approx(sigma, rel=1e-12)
    assert mapping.theta[0] == pytest.approx(theta, rel=1e-12)


def test_dt_scales_theta_and_sigma_but_not_the_half_life_in_bars():
    rules = OUValidityRules(dt=5.0)
    mapping = map_ar1_to_ou(0.0, 0.9, 0.01, rules=rules)
    unit = map_ar1_to_ou(0.0, 0.9, 0.01, rules=RULES)
    assert mapping.theta[0] == pytest.approx(unit.theta[0] / 5.0)
    assert mapping.half_life_bars[0] == pytest.approx(unit.half_life_bars[0])
    assert mapping.stationary_std[0] == pytest.approx(unit.stationary_std[0])
    assert stationary_std(mapping.sigma[0], mapping.theta[0]) == pytest.approx(
        mapping.stationary_std[0])


def test_equilibrium_is_estimated_not_forced_to_zero():
    x = simulate_ou(0.1, 0.004, 0.001, 200_000, seed=21)
    fit = fit_ar1(x, rules=RULES)
    assert fit.ou.mu == pytest.approx(0.004, abs=3e-4)
    assert fit.mu_ci_lower < 0.004 < fit.mu_ci_upper
    assert fit.a / (1 - fit.b) == pytest.approx(fit.ou.mu)


def test_rolling_ou_zscore_and_std_ratio_use_the_window_estimates():
    x = simulate_ou(0.1, 0.0, 0.2, 3_000, seed=8)
    result = rolling_ou(x, 256, rules=RULES)
    t = 2_000
    m = result.mapping
    assert result.zscore[t] == pytest.approx((x[t] - m.mu[t]) / m.stationary_std[t])
    assert result.std_ratio[t] == pytest.approx(result.fits.window_std[t] / m.stationary_std[t])
    predicted = result.fits.a[t - 1] + result.fits.b[t - 1] * x[t - 1]
    assert result.innovation[t] == pytest.approx(x[t] - predicted)


def test_pooled_pairs_fit_ignores_order():
    x = simulate_ar1(0.0, 0.7, 1.0, 5_000, seed=3)
    u, v = x[:-1], x[1:]
    perm = np.random.default_rng(0).permutation(u.size)
    ordered = fit_ar1_pairs(u, v, rules=RULES)
    shuffled = fit_ar1_pairs(u[perm], v[perm], rules=RULES)
    assert ordered.b == pytest.approx(shuffled.b, rel=1e-12)
    assert ordered.se_b == pytest.approx(shuffled.se_b, rel=1e-12)
