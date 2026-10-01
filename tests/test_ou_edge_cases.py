"""Numerical safety: degenerate inputs give labelled states, never explosions.

Constant windows, perfect fits, b at or near 0 and 1, negative b, NaN and
infinity, and tiny samples must each produce a deterministic, labelled
outcome - a state and a warning flag - and never an infinite or absurd
number in a parameter column.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from xauusd_quant.models import load_ou_config
from xauusd_quant.models.ornstein_uhlenbeck import (
    OUValidityRules,
    fit_ar1,
    rolling_ar1,
    rolling_ou,
    simulate_ar1,
)
from xauusd_quant.research.ou_estimation import b_classification, ou_parameter_frame

RULES = OUValidityRules()
PARAMS = ("ou_theta", "ou_mu", "ou_sigma", "ou_stationary_std", "ou_half_life_bars",
          "ou_zscore", "ou_std_ratio")


def assert_no_infinities(frame) -> None:
    for column in PARAMS + ("ou_a", "ou_b", "ou_r_squared", "ou_innovation"):
        values = frame[column].drop_nulls().to_numpy()
        assert np.isfinite(values).all(), f"{column} holds a non-finite value"


def test_a_constant_series_is_degenerate_not_a_number():
    frame = ou_parameter_frame(np.full(500, 0.25), None, ou_window=64, config=load_ou_config())
    states = set(frame["ou_state"].drop_nulls().to_list())
    assert states == {"insufficient_data", "degenerate"}
    assert frame["ou_b"].null_count() == frame.height
    assert frame.filter(frame["ou_state"] == "degenerate")["ou_numerical_warning"].all()
    assert_no_infinities(frame)


def test_constant_static_fit_is_degenerate():
    fit = fit_ar1(np.full(200, 3.0), rules=RULES)
    assert fit.ou.state == "degenerate"
    assert math.isnan(fit.b)


def test_a_perfect_fit_has_zero_sigma_and_no_zscore():
    """Deterministic decay: theta, mu and the half-life exist; sigma is 0; Z is undefined.

    The path decays slowly enough (b = 0.98 from a deviation of ~100) that
    every window's variation stays far above float rounding, so every fit is
    exact. (With fast decay the late windows vary by ~1e-9 and their "perfect"
    fit is limited by rounding - a genuinely tiny sigma, correctly reported.)
    """
    x = simulate_ar1(0.01, 0.98, 0.0, 300, x0=100.0)
    frame = ou_parameter_frame(x, None, ou_window=64, config=load_ou_config())
    fitted = frame.filter(frame["ou_state"] == "valid")
    assert fitted.height == 300 - 63
    early = fitted.head(20)
    assert early["ou_sigma"].max() == 0.0
    assert early["ou_numerical_warning"].all()
    assert early["ou_zscore"].null_count() == early.height
    # Late windows sit ~100 below the block's shift constant, so rounding in the
    # prefix sums leaves an honest, tiny sigma rather than an exact zero.
    assert fitted["ou_sigma"].max() < 1e-4
    assert fitted["ou_half_life_bars"][0] == pytest.approx(-math.log(2) / math.log(0.98),
                                                           rel=1e-6)
    assert_no_infinities(frame)


def test_negative_b_is_reported_as_non_positive():
    x = simulate_ar1(0.0, -0.6, 1.0, 5_000, seed=1)
    frame = ou_parameter_frame(x, None, ou_window=128, config=load_ou_config())
    summary = b_classification(frame, source="t", config=load_ou_config())
    assert summary["fraction_non_positive"] > 0.95
    assert summary["valid_ou_fraction"] < 0.05
    assert_no_infinities(frame)


def test_b_near_zero_gives_a_tiny_half_life_not_an_error():
    x = simulate_ar1(0.0, 0.02, 1.0, 20_000, seed=2)
    fit = fit_ar1(x, rules=RULES)
    if fit.ou.state == "valid":
        assert 0 < fit.ou.half_life_bars < 0.5
    else:
        assert fit.ou.state == "non_positive"


def test_b_at_or_above_one_never_receives_ou_parameters():
    explosive = simulate_ar1(0.0, 1.002, 0.01, 3_000, x0=1.0, seed=3)
    frame = ou_parameter_frame(explosive, None, ou_window=256, config=load_ou_config())
    over = frame.filter(frame["ou_b"] > 1.0)
    assert over.height > 0
    assert set(over["ou_state"].to_list()) <= {"explosive", "unit_root"}
    for column in PARAMS:
        assert over[column].null_count() == over.height
    assert_no_infinities(frame)


def test_nan_and_infinity_are_missing_values_deterministically():
    x = simulate_ar1(0.0, 0.7, 1.0, 1_000, seed=4)
    x[300] = np.nan
    x[600] = np.inf
    x[601] = -np.inf
    a = ou_parameter_frame(x, None, ou_window=50, config=load_ou_config())
    b = ou_parameter_frame(x.copy(), None, ou_window=50, config=load_ou_config())
    assert a["ou_b"].fill_null(-9).to_list() == b["ou_b"].fill_null(-9).to_list()
    assert a["ou_state"][300:350].to_list() == ["insufficient_data"] * 50
    assert a["ou_state"][600:651].to_list() == ["insufficient_data"] * 51
    assert_no_infinities(a)


def test_tiny_samples_are_insufficient():
    assert fit_ar1(np.array([1.0, 2.0]), rules=RULES).ou.state == "insufficient_data"
    frame = ou_parameter_frame(np.arange(10.0), None, ou_window=64, config=load_ou_config())
    assert set(frame["ou_state"].to_list()) == {"insufficient_data"}
    with pytest.raises(ValueError, match="at least 4"):
        rolling_ar1(np.arange(10.0), 3)


def test_a_trending_window_is_handled_without_blowing_up():
    """x_t = t is an exact unit root with drift: b = 1 exactly in every window."""
    result = rolling_ou(np.arange(1_000, dtype=float), 64, rules=RULES)
    states = set(np.asarray(result.mapping.state)[63:].tolist())
    assert states <= {"unit_root"}
    assert np.isnan(result.mapping.half_life_bars).all()
