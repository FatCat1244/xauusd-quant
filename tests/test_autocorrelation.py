"""Autocorrelation and Ljung-Box, against sequences with known structure."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from xauusd_quant.research.autocorrelation import (
    acf_absolute_returns,
    acf_frame,
    acf_returns,
    acf_squared_returns,
    autocorrelation,
    effect_size_label,
    ljung_box_test,
)


def ar1(n: int, phi: float, *, seed: int = 11, sigma: float = 1.0) -> np.ndarray:
    """AR(1) with a known lag-1 autocorrelation of exactly ``phi``."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, n)
    out = np.empty(n)
    out[0] = noise[0]
    for i in range(1, n):
        out[i] = phi * out[i - 1] + noise[i]
    return out


def white_noise(n: int, *, seed: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 1.0, n)


# ---------------------------------------------------------------------------
# Known structure is recovered
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("phi", [0.7, 0.4, -0.5])
def test_ar1_lag1_autocorrelation_is_recovered(phi):
    result = autocorrelation(ar1(20_000, phi), max_lag=5)
    assert result.values[0] == pytest.approx(phi, abs=0.03)


def test_ar1_autocorrelation_decays_geometrically():
    """For an AR(1), rho_k = phi^k."""
    phi = 0.6
    result = autocorrelation(ar1(40_000, phi), max_lag=5)
    for k, rho in zip(result.lags, result.values, strict=True):
        assert rho == pytest.approx(phi**k, abs=0.04), f"lag {k}"


def test_white_noise_autocorrelation_sits_inside_the_band():
    result = autocorrelation(white_noise(20_000), max_lag=20)
    outside = sum(abs(v) > result.confidence_bound for v in result.values)
    # ~5% of lags may cross a 95% band by chance; allow generous slack.
    assert outside <= 4, f"{outside} of 20 lags outside the band"


def test_alternating_series_has_negative_lag1_autocorrelation():
    values = np.array([1.0, -1.0] * 5000)
    result = autocorrelation(values, max_lag=2)
    assert result.values[0] == pytest.approx(-1.0, abs=0.01)
    assert result.values[1] == pytest.approx(1.0, abs=0.01)


def test_lag_zero_is_excluded():
    result = autocorrelation(white_noise(1000), max_lag=10)
    assert result.lags[0] == 1
    assert len(result.lags) == len(result.values) == 10


# ---------------------------------------------------------------------------
# Confidence band and reporting
# ---------------------------------------------------------------------------
def test_confidence_bound_shrinks_with_sample_size():
    small = autocorrelation(white_noise(1_000, seed=1), max_lag=5)
    large = autocorrelation(white_noise(100_000, seed=1), max_lag=5)
    assert large.confidence_bound < small.confidence_bound
    assert large.confidence_bound == pytest.approx(1.96 / np.sqrt(100_000), rel=0.01)


def test_result_frame_carries_effect_size_and_sample_count():
    frame = autocorrelation(ar1(5_000, 0.5), max_lag=3).to_frame()
    assert frame.columns == [
        "lag", "autocorrelation", "conf_lower", "conf_upper",
        "significant", "effect_size", "observations", "series",
    ]
    assert frame["observations"][0] == 5_000
    assert frame["effect_size"][0] in {"moderate", "large"}


@pytest.mark.parametrize(
    ("rho", "expected"),
    [(0.0005, "negligible"), (0.02, "very small"), (0.07, "small"),
     (0.15, "moderate"), (0.6, "large"), (-0.6, "large")],
)
def test_effect_size_labels(rho, expected):
    assert effect_size_label(rho) == expected


def test_a_tiny_but_significant_autocorrelation_is_labelled_negligible():
    """The central caution of this module, made executable.

    A million observations makes rho = 0.003 'significant'. The label must
    still say it is negligible.
    """
    values = ar1(1_000_000, 0.003, seed=5)
    result = autocorrelation(values, max_lag=1)
    frame = result.to_frame()
    assert abs(result.values[0]) < 0.01
    assert frame["effect_size"][0] == "negligible"


def test_notes_warn_about_large_sample_significance():
    result = autocorrelation(white_noise(50_000), max_lag=5)
    assert any("effect_size" in note for note in result.notes)


# ---------------------------------------------------------------------------
# The three series
# ---------------------------------------------------------------------------
def test_volatility_clustering_shows_in_abs_and_squared_not_raw():
    """Sign-randomised, magnitude-clustered returns: the canonical case."""
    rng = np.random.default_rng(17)
    magnitude = np.abs(ar1(40_000, 0.85, seed=2)) + 0.05
    signs = rng.choice([-1.0, 1.0], size=magnitude.size)
    returns = magnitude * signs

    raw = acf_returns(returns, max_lag=10)
    absolute = acf_absolute_returns(returns, max_lag=10)
    squared = acf_squared_returns(returns, max_lag=10)

    assert abs(raw.values[0]) < 0.05, "signs are random: raw ACF must be near zero"
    assert absolute.values[0] > 0.3, "|r| must show the clustering"
    assert squared.values[0] > 0.2, "r^2 must show the clustering"
    assert absolute.series_name == "abs_returns"
    assert squared.series_name == "squared_returns"


def test_acf_frame_stacks_multiple_series():
    values = ar1(3_000, 0.3)
    frame = acf_frame([
        acf_returns(values, max_lag=5),
        acf_absolute_returns(values, max_lag=5),
        acf_squared_returns(values, max_lag=5),
    ])
    assert frame.height == 15
    assert set(frame["series"].unique()) == {
        "returns", "abs_returns", "squared_returns"
    }


# ---------------------------------------------------------------------------
# Ljung-Box
# ---------------------------------------------------------------------------
def test_ljung_box_rejects_for_an_autocorrelated_series():
    results = ljung_box_test(ar1(5_000, 0.5), lags=(5, 10))
    assert all(r.reject_at_5pct for r in results)
    assert all(r.p_value < 1e-6 for r in results)


def test_ljung_box_does_not_reject_for_white_noise():
    results = ljung_box_test(white_noise(5_000, seed=21), lags=(5, 10, 20))
    assert not any(r.reject_at_5pct for r in results)


def test_ljung_box_reports_effect_size_beside_the_p_value():
    results = ljung_box_test(ar1(5_000, 0.5), lags=(5,))
    result = results[0]
    assert result.mean_abs_autocorrelation > 0
    assert result.max_abs_autocorrelation >= result.mean_abs_autocorrelation
    assert result.effect_size in {"small", "moderate", "large", "very small"}


def test_ljung_box_warns_on_very_large_samples():
    results = ljung_box_test(
        white_noise(150_000), lags=(5,), large_sample_threshold=100_000
    )
    assert results[0].warning is not None
    assert "transaction costs" in results[0].warning


def test_ljung_box_skips_lags_beyond_the_sample():
    results = ljung_box_test(white_noise(50), lags=(5, 10, 500))
    assert [r.lag for r in results] == [5, 10]


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------
def test_polars_series_input_and_null_handling():
    values = pl.Series("r", [0.1, None, -0.2, 0.3, None, -0.1] * 200)
    result = autocorrelation(values, max_lag=3)
    assert result.observations == 800, "nulls must be dropped, not zero-filled"


def test_too_few_observations_is_an_explicit_error():
    with pytest.raises(ValueError, match="at least 3 observations"):
        autocorrelation(np.array([1.0, 2.0]), max_lag=1)


def test_max_lag_is_capped_by_sample_size():
    result = autocorrelation(white_noise(30), max_lag=100)
    assert result.max_lag <= 28
    assert any("reduced" in note for note in result.notes)


# ---------------------------------------------------------------------------
# The Ljung-Box statistic is computed in closed form from the FFT ACF rather
# than by statsmodels' O(n^2) route. It must agree with statsmodels exactly.
# ---------------------------------------------------------------------------
def _reference_ljung_box(values, lags):
    from statsmodels.stats.diagnostic import acorr_ljungbox

    table = acorr_ljungbox(values, lags=list(lags), return_df=True)
    return {
        lag: (float(table.loc[lag, "lb_stat"]), float(table.loc[lag, "lb_pvalue"]))
        for lag in lags
    }


def _ar1(n, phi, seed=11):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 1.0, n)
    out = np.empty(n)
    out[0] = noise[0]
    for i in range(1, n):
        out[i] = phi * out[i - 1] + noise[i]
    return out


@pytest.mark.parametrize("series_name", ["white", "ar1", "ma31"])
def test_ljung_box_matches_statsmodels_exactly(series_name):
    rng = np.random.default_rng(11)
    n = 20_000
    if series_name == "white":
        values = rng.normal(0.0, 1.0, n)
    elif series_name == "ar1":
        values = _ar1(n, 0.4)
    else:
        white = rng.normal(0.0, 1.0, n + 30)
        values = np.convolve(white, np.ones(31), "valid")[:n]

    lags = (5, 10, 20, 50)
    reference = _reference_ljung_box(values, lags)
    results = ljung_box_test(values, lags=lags, series_name=series_name)

    assert [r.lag for r in results] == list(lags)
    for result in results:
        expected_stat, expected_p = reference[result.lag]
        assert result.statistic == pytest.approx(expected_stat, rel=1e-10)
        assert result.p_value == pytest.approx(expected_p, rel=1e-10, abs=1e-300)


def test_ljung_box_statistic_is_monotone_in_the_lag():
    """Q(h) is a cumulative sum of non-negative terms."""
    results = ljung_box_test(_ar1(5_000, 0.3), lags=(5, 10, 20, 50))
    statistics = [r.statistic for r in results]
    assert statistics == sorted(statistics)
