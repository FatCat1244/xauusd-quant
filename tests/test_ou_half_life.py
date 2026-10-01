"""Half-life: the formula, where it does not apply, and honest summaries.

HL = -ln 2 / ln b exists only for 0 < b < 1. For b <= 0, b = 1 and b > 1 the
formula must never be applied - not even silently through NaN arithmetic that
happens to look reasonable - and each case must be classified and reported
separately. A near-unit-root b gives a half-life that is computable and
meaningless, so summaries must treat it as censored rather than let it
dominate a mean.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from xauusd_quant.models import load_ou_config
from xauusd_quant.models.ornstein_uhlenbeck import (
    OUValidityRules,
    fit_ar1,
    half_life_bars_from_b,
    map_ar1_to_ou,
    simulate_ar1,
    theta_from_b,
)
from xauusd_quant.research.ou_estimation import half_life_summary, ou_parameter_frame

RULES = OUValidityRules()


def test_half_life_of_b_one_half_is_exactly_one_bar():
    assert half_life_bars_from_b(0.5) == pytest.approx(1.0, abs=1e-15)


def test_half_life_of_b_point_nine():
    expected = -math.log(2) / math.log(0.9)
    assert half_life_bars_from_b(0.9) == pytest.approx(expected, rel=1e-14)
    assert expected == pytest.approx(6.578813478960585)


@pytest.mark.parametrize("b", [0.0, -0.2, -1.0, 1.0, 1.0000001, 1.5, float("nan")])
def test_no_half_life_or_theta_outside_the_unit_interval(b):
    assert math.isnan(half_life_bars_from_b(b))
    assert math.isnan(theta_from_b(b))


@pytest.mark.parametrize(
    ("b", "state"),
    [
        (0.5, "valid"),
        (0.0, "non_positive"),
        (-0.4, "non_positive"),
        (1.0, "unit_root"),
        (1.0 + 5e-10, "unit_root"),
        (1.2, "explosive"),
        (0.99999, "near_unit_root"),       # HL ~ 69,000 bars > 10,000 cap
    ],
)
def test_each_case_is_classified_and_only_valid_ones_get_parameters(b, state):
    mapping = map_ar1_to_ou(0.01, b, 0.01, rules=RULES)
    assert mapping.state[0] == state
    has_parameters = not math.isnan(mapping.theta[0])
    assert has_parameters == (state == "valid")
    for column in (mapping.mu, mapping.sigma, mapping.stationary_std, mapping.half_life_bars):
        assert math.isnan(column[0]) == (state != "valid")


def test_the_reportable_cap_sets_the_near_unit_root_boundary():
    rules = OUValidityRules(max_half_life_bars=100.0)
    b_edge = 2 ** (-1 / 100)
    assert map_ar1_to_ou(0.0, b_edge * 0.9999, 0.01, rules=rules).state[0] == "valid"
    assert map_ar1_to_ou(0.0, b_edge * 1.00001, 0.01, rules=rules).state[0] == "near_unit_root"


def test_a_narrower_valid_range_is_respected_and_labelled():
    rules = OUValidityRules(valid_b_min=0.2, valid_b_max=0.99)
    assert map_ar1_to_ou(0.0, 0.1, 0.01, rules=rules).state[0] == "outside_valid_range"
    assert map_ar1_to_ou(0.0, 0.5, 0.01, rules=rules).state[0] == "valid"


@pytest.mark.parametrize(("b", "tolerance"), [(0.5, 0.03), (0.8, 0.04), (0.95, 0.05)])
def test_synthetic_ar1_half_lives_are_recovered(b, tolerance):
    x = simulate_ar1(0.0, b, 1.0, 200_000, seed=17)
    fit = fit_ar1(x, rules=RULES)
    expected = -math.log(2) / math.log(b)
    assert fit.ou.state == "valid"
    assert fit.ou.half_life_bars == pytest.approx(expected, rel=tolerance)


def test_summary_censors_near_unit_root_windows_instead_of_averaging_them():
    config = load_ou_config()
    frame = pl.DataFrame({
        "ou_n_pairs": [10] * 10,
        "ou_half_life_bars": [1.0, 2.0, 3.0, 4.0, 5.0, None, None, None, None, None],
        "ou_state": ["valid"] * 5 + ["near_unit_root"] * 3 + ["explosive"] * 2,
    })
    out = half_life_summary(frame, source="t", config=config)
    assert out["valid_windows"] == 5
    assert out["censored_above_cap"] == 3
    assert out["valid_fraction"] == pytest.approx(0.5)
    assert out["mean_reportable"] == pytest.approx(3.0)
    # Pooled with 3 censored (=+inf) values: p25 is observed, p75 is above the cap.
    assert out["p25"] is not None
    assert out["p75"] is None and out["p75_above_cap"] is True


def test_a_random_walk_is_never_given_a_confident_short_half_life():
    """No reversion at all: most windows land near, at or above a unit root."""
    walk = np.cumsum(np.random.default_rng(3).normal(size=20_000))
    config = load_ou_config()
    frame = ou_parameter_frame(walk, None, ou_window=256, config=config)
    b = frame["ou_b"].drop_nulls().to_numpy()
    assert np.median(b) > 0.97
    fit = fit_ar1(walk, rules=RULES)
    assert fit.b > 0.999
    assert fit.df_statistic > -3.5, "a random walk must not reject a unit root decisively"
