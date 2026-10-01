"""IC decay and the empirical information half-life (Prompt #8, Steps 23-24)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from xauusd_quant.alpha.decay import decay_summary, information_half_life

LAGS = np.array([1, 2, 3, 5, 10, 20, 50])


def test_an_exponential_decay_recovers_its_half_life() -> None:
    ic = 0.2 * np.exp(-math.log(2) * LAGS / 5.0)
    se = np.full(LAGS.size, 1e-4)
    fit = information_half_life(LAGS, ic, se, zero_z=2.0, min_points=4)
    assert fit["fit"] == "exponential"
    assert fit["alpha_information_half_life"] == pytest.approx(5.0, rel=1e-6)
    assert fit["ic0"] == pytest.approx(0.2, rel=1e-6)


def test_a_rebounding_curve_is_not_called_exponential() -> None:
    ic = np.array([0.27, 0.23, 0.20, 0.15, 0.09, 0.27, 0.15])   # the daily cycle at 1h
    fit = information_half_life(LAGS, ic, np.full(7, 0.01), zero_z=2.0, min_points=4)
    assert fit["alpha_information_half_life"] is None
    assert fit["fit"].startswith("not exponential")


def test_no_half_life_without_enough_significant_points() -> None:
    ic = np.array([0.05, 0.001, 0.0, -0.001, 0.0, 0.0, 0.0])
    fit = information_half_life(LAGS, ic, np.full(7, 0.005), zero_z=2.0, min_points=4)
    assert fit["alpha_information_half_life"] is None
    weak = information_half_life(LAGS, np.full(7, 0.001), np.full(7, 0.01), zero_z=2.0,
                                 min_points=4)
    assert weak["fit"] == "not fitted: the peak is indistinguishable from zero"


def test_decay_summary_reads_peak_half_reversal_and_zero() -> None:
    ic = np.array([0.10, 0.08, 0.05, 0.02, -0.03, -0.005, 0.0])
    se = np.full(7, 0.005)
    out = decay_summary(LAGS, ic, se, zero_z=2.0)
    assert out["peak_horizon"] == 1 and out["peak_ic"] == pytest.approx(0.10)
    assert out["half_decay_horizon"] == 3
    assert out["sign_reversal_horizon"] == 10
    assert out["zero_horizon"] == 20
    assert decay_summary(LAGS, np.full(7, np.nan), se, zero_z=2.0)["peak_horizon"] is None
