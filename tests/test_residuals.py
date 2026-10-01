"""Residual statistics: the two Z-score methods, distributions and ACF."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from xauusd_quant.features import load_regression_config
from xauusd_quant.features.rolling_regression import rolling_regression_features
from xauusd_quant.research.residual_stationarity import (
    random_walk_control,
    residual_stationarity,
)
from xauusd_quant.research.residuals import (
    add_quantile_buckets,
    r_squared_statistics,
    residual_autocorrelation,
    residual_distribution,
    slope_statistics,
)


@pytest.fixture
def regression_config():
    return load_regression_config()


def make_bars(n: int = 2000, *, seed: int = 5, drift: float = 0.0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    prices = 400.0 * np.exp(np.cumsum(rng.normal(drift, 0.001, n)))
    start = datetime(2024, 1, 2, 1, 0)
    return pl.DataFrame({
        "timestamp": [start + timedelta(minutes=5 * i) for i in range(n)],
        "open": prices, "high": prices * 1.0002, "low": prices * 0.9998,
        "close": prices,
        "first_bid": prices - 0.2, "first_ask": prices + 0.2,
        "last_bid": prices - 0.2, "last_ask": prices + 0.2,
        "tick_count": np.full(n, 100, dtype=np.int64),
        "volume": np.full(n, 1000.0),
        "mean_spread": np.full(n, 0.4),
    })


@pytest.fixture
def features(regression_config):
    frame, _ = rolling_regression_features(
        make_bars(), window=64, config=regression_config, timeframe="5m"
    )
    return frame.filter(pl.col("residual").is_not_null())


# ---------------------------------------------------------------------------
# The two Z-score methods are genuinely different
# ---------------------------------------------------------------------------
def test_method_a_divides_by_the_fit_standard_error(regression_config):
    """Z_fit = eps_t / s_fit, with no mean subtracted (the OLS mean is zero)."""
    frame, _ = rolling_regression_features(
        make_bars(800), window=50, config=regression_config, timeframe="5m"
    )
    residual = frame["residual"].to_numpy()
    std = frame["residual_std_fit"].to_numpy()
    z = frame["residual_zscore_fit"].to_numpy()
    ok = np.isfinite(z)
    assert z[ok] == pytest.approx(residual[ok] / std[ok], rel=1e-12)


def test_method_b_subtracts_a_rolling_mean(regression_config):
    """Z_rolling = (eps_t - mu_t) / sigma_t over the residual series."""
    frame, _ = rolling_regression_features(
        make_bars(1200), window=64, config=regression_config, timeframe="5m"
    )
    residual = frame["residual"]
    window = 64
    expected = frame.select(
        (
            (pl.col("residual") - pl.col("residual").rolling_mean(window, min_samples=window))
            / pl.col("residual").rolling_std(window, min_samples=window, ddof=1)
        ).alias("z")
    )["z"].to_numpy()
    actual = frame["residual_zscore_rolling"].to_numpy()
    ok = np.isfinite(expected) & np.isfinite(actual)
    assert ok.sum() > 100
    assert actual[ok] == pytest.approx(expected[ok], rel=1e-10)
    assert residual is not None


def test_the_two_methods_are_kept_in_separate_columns(regression_config):
    frame, _ = rolling_regression_features(
        make_bars(1000), window=64, config=regression_config, timeframe="5m"
    )
    assert "residual_zscore_fit" in frame.columns
    assert "residual_zscore_rolling" in frame.columns
    a = frame["residual_zscore_fit"].to_numpy()
    b = frame["residual_zscore_rolling"].to_numpy()
    ok = np.isfinite(a) & np.isfinite(b)
    assert not np.allclose(a[ok], b[ok]), "the two methods must not be silently mixed"


def test_a_known_residual_sequence_gives_the_expected_rolling_zscore():
    """Hand-checked Method B on a deliberately simple series."""
    values = [0.0] * 10 + [1.0]
    frame = pl.DataFrame({"r": values})
    window = 10
    z = frame.select(
        (
            (pl.col("r") - pl.col("r").rolling_mean(window, min_samples=window))
            / pl.col("r").rolling_std(window, min_samples=window, ddof=1)
        ).alias("z")
    )["z"].to_list()
    # The last window is nine zeros and a one: mean 0.1, sd = sqrt(0.1).
    assert z[-1] == pytest.approx((1.0 - 0.1) / np.sqrt(0.1), rel=1e-12)


# ---------------------------------------------------------------------------
# Slope
# ---------------------------------------------------------------------------
def test_slope_statistics_on_a_known_uptrend(regression_config):
    """A steady exponential uptrend must give an all-positive slope."""
    n = 1000
    prices = 400.0 * np.exp(0.0002 * np.arange(n))
    bars = make_bars(n).with_columns(pl.Series("close", prices))
    frame, _ = rolling_regression_features(
        bars, window=64, config=regression_config, timeframe="5m"
    )
    valid = frame.filter(pl.col("regression_slope").is_not_null())
    stats_out = slope_statistics(
        valid, timeframe="5m", window=64, config=regression_config
    )
    assert stats_out.mean == pytest.approx(0.0002, rel=1e-6)
    assert stats_out.positive_frequency == pytest.approx(1.0)
    assert stats_out.negative_frequency == pytest.approx(0.0)
    assert stats_out.mean_pct_per_bar == pytest.approx(np.expm1(0.0002), rel=1e-4)


def test_slope_statistics_are_not_annualised(regression_config, features):
    stats_out = slope_statistics(
        features, timeframe="5m", window=64, config=regression_config
    )
    assert any("Not annualised" in note for note in stats_out.notes)


# ---------------------------------------------------------------------------
# R-squared
# ---------------------------------------------------------------------------
def test_r_squared_is_one_for_a_perfect_trend(regression_config):
    n = 500
    prices = 400.0 * np.exp(0.0001 * np.arange(n))
    bars = make_bars(n).with_columns(pl.Series("close", prices))
    frame, _ = rolling_regression_features(
        bars, window=64, config=regression_config, timeframe="5m"
    )
    r2 = frame["r_squared"].to_numpy()
    ok = np.isfinite(r2)
    assert r2[ok] == pytest.approx(1.0, abs=1e-9)


def test_r_squared_table_reports_per_quartile_residual_behaviour(
    regression_config, features
):
    table = r_squared_statistics(
        features, timeframe="5m", window=64, config=regression_config
    )
    assert not table.is_empty()
    buckets = set(table["bucket"].to_list())
    assert "all" in buckets
    assert {"Q1", "Q2", "Q3", "Q4"} <= buckets
    assert "residual_std" in table.columns
    assert "abs_z_gt_2_frequency" in table.columns


def test_quantile_buckets_split_roughly_evenly(features):
    bucketed = add_quantile_buckets(
        features, column="r_squared", label="r2_bucket", buckets=4
    )
    counts = bucketed.group_by("r2_bucket").len().sort("r2_bucket")
    sizes = counts["len"].to_list()
    assert len(sizes) == 4
    assert max(sizes) / min(sizes) < 1.2, f"buckets are badly unbalanced: {sizes}"


# ---------------------------------------------------------------------------
# Residual distribution
# ---------------------------------------------------------------------------
def test_residual_distribution_covers_both_zscore_series(regression_config, features):
    dist = residual_distribution(
        features, timeframe="5m", window=64, config=regression_config
    )
    assert "residual" in dist.series
    assert "residual_zscore_fit" in dist.series
    assert "residual_zscore_rolling" in dist.series
    table = dist.to_frame()
    assert "p0_1" in table.columns and "p99_9" in table.columns


def test_residual_distribution_states_the_zero_mean_construction(
    regression_config, features
):
    dist = residual_distribution(
        features, timeframe="5m", window=64, config=regression_config
    )
    joined = " ".join(dist.notes)
    assert "zero by OLS construction" in joined
    assert "Gaussianity is not assumed" in joined


def test_symmetry_is_reported_rather_than_assumed(regression_config, features):
    dist = residual_distribution(
        features, timeframe="5m", window=64, config=regression_config
    )
    assert any("skewness" in note for note in dist.notes)


# ---------------------------------------------------------------------------
# Residual autocorrelation
# ---------------------------------------------------------------------------
def test_residual_acf_covers_all_three_series(regression_config, features):
    acf, ljung, notes = residual_autocorrelation(
        features, timeframe="5m", window=64, config=regression_config
    )
    assert set(acf["series"].unique()) == {"residual", "residual_diff", "abs_residual"}
    assert not ljung.is_empty()
    assert any("mechanical" in note for note in notes)


def test_raw_residual_acf_is_high_by_construction(regression_config, features):
    """Overlapping windows guarantee it; the test pins the expectation down."""
    acf, _, _ = residual_autocorrelation(
        features, timeframe="5m", window=64, config=regression_config
    )
    lag1 = acf.filter((pl.col("series") == "residual") & (pl.col("lag") == 1))
    assert lag1["autocorrelation"][0] > 0.5, (
        "consecutive residuals share 63 of 64 bars, so lag-1 must be high"
    )


# ---------------------------------------------------------------------------
# Stationarity, and the control that makes it interpretable
# ---------------------------------------------------------------------------
def test_random_walk_control_is_detrended_the_same_way():
    control = random_walk_control(2000, 64)
    assert control.size > 1500
    assert np.all(np.isfinite(control))
    # Detrending forces it near zero regardless of the walk's level.
    assert abs(float(np.mean(control))) < 0.01


def test_stationarity_compares_against_the_control(regression_config, features):
    result = residual_stationarity(
        features, timeframe="5m", window=64, config=regression_config
    )
    assert result.adf is not None
    assert result.control_adf is not None
    assert result.control_verdict != "not evaluated"
    assert result.control_matches is not None
    assert "local detrending" in result.interpretation


def test_stationarity_of_a_detrended_random_walk_is_flagged(regression_config):
    """The point of the control: a pure random walk looks stationary once detrended."""
    bars = make_bars(3000, seed=99)
    frame, _ = rolling_regression_features(
        bars, window=64, config=regression_config, timeframe="5m"
    )
    valid = frame.filter(pl.col("residual").is_not_null())
    result = residual_stationarity(
        valid, timeframe="5m", window=64, config=regression_config
    )
    # The input IS a random walk, so residual and control should agree.
    assert result.control_matches is True
    assert any("artefact of local detrending" in note for note in result.notes)
