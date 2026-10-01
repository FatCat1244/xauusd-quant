"""Rolling OLS: correctness against reference implementations, and numerics."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from xauusd_quant.features import (
    fit_window,
    load_regression_config,
    rolling_ols,
    theil_sen_slope,
    window_design,
)
from xauusd_quant.features.rolling_regression import rolling_regression_features


@pytest.fixture
def regression_config():
    return load_regression_config()


def bars_from_prices(prices, start: datetime | None = None) -> pl.DataFrame:
    prices = np.asarray(prices, dtype=np.float64)
    n = prices.size
    start = start or datetime(2024, 1, 2, 1, 0)
    return pl.DataFrame({
        "timestamp": [start + timedelta(minutes=5 * i) for i in range(n)],
        "open": prices, "high": prices * 1.0001, "low": prices * 0.9999,
        "close": prices,
        "first_bid": prices - 0.2, "first_ask": prices + 0.2,
        "last_bid": prices - 0.2, "last_ask": prices + 0.2,
        "tick_count": np.full(n, 100, dtype=np.int64),
        "volume": np.full(n, 1000.0),
        "mean_spread": np.full(n, 0.4),
    })


# ---------------------------------------------------------------------------
# Window geometry
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n", [3, 32, 64, 128, 512])
def test_window_design_matches_direct_computation(n):
    """Sxx = N(N^2-1)/12 must equal the directly summed value."""
    design = window_design(n)
    direct = float(np.sum((np.arange(n) - (n - 1) / 2.0) ** 2))
    assert design.sxx_centred == pytest.approx(direct, rel=1e-12)
    assert design.x_mean == pytest.approx((n - 1) / 2.0)
    assert design.x_centred.sum() == pytest.approx(0.0, abs=1e-9)


def test_window_below_three_bars_is_rejected():
    with pytest.raises(ValueError, match="at least 3 bars"):
        window_design(2)


# ---------------------------------------------------------------------------
# The specified worked example: a perfect linear trend
# ---------------------------------------------------------------------------
def test_perfect_linear_trend_is_recovered_exactly():
    """y = 2 + 3x over one window: beta=3, alpha=2, residual=0, R^2=1."""
    n = 50
    y = 2.0 + 3.0 * np.arange(n, dtype=np.float64)
    fit = fit_window(y)
    assert fit.slope == pytest.approx(3.0, abs=1e-12)
    assert fit.intercept == pytest.approx(2.0, abs=1e-9)
    assert fit.residual_last == pytest.approx(0.0, abs=1e-9)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-12)
    assert fit.residual_std == pytest.approx(0.0, abs=1e-9)


def test_perfect_trend_rolling_is_exact_at_every_bar():
    n, window = 300, 64
    y = 2.0 + 3.0 * np.arange(n, dtype=np.float64)
    result = rolling_ols(y, window)
    valid = np.isfinite(result.slope)
    assert result.slope[valid] == pytest.approx(3.0, abs=1e-9)
    assert result.residual[valid] == pytest.approx(0.0, abs=1e-8)
    assert result.r_squared[valid] == pytest.approx(1.0, abs=1e-9)
    # The intercept is the fitted value at the window's own x=0, i.e. at t-N+1.
    t = 100
    assert result.intercept[t] == pytest.approx(2.0 + 3.0 * (t - window + 1), abs=1e-6)


def test_fitted_value_reproduces_the_price_on_a_perfect_trend():
    n, window = 200, 32
    y = 5.0 - 0.25 * np.arange(n, dtype=np.float64)
    result = rolling_ols(y, window)
    valid = np.isfinite(result.fitted)
    assert result.fitted[valid] == pytest.approx(y[valid], abs=1e-9)


# ---------------------------------------------------------------------------
# Against reference implementations
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("window", [16, 64, 128])
def test_rolling_ols_matches_numpy_polyfit(window):
    rng = np.random.default_rng(3)
    y = np.cumsum(rng.normal(0.0, 0.001, 600)) + np.log(400.0)
    result = rolling_ols(y, window)
    x = np.arange(window, dtype=np.float64)

    for t in range(window - 1, y.size, 37):
        chunk = y[t - window + 1 : t + 1]
        slope, intercept = np.polyfit(x, chunk, 1)
        fitted = intercept + slope * (window - 1)
        sse = float(np.sum((chunk - (intercept + slope * x)) ** 2))
        sst = float(np.sum((chunk - chunk.mean()) ** 2))
        assert result.slope[t] == pytest.approx(slope, rel=1e-9, abs=1e-15)
        assert result.intercept[t] == pytest.approx(intercept, rel=1e-9, abs=1e-12)
        assert result.fitted[t] == pytest.approx(fitted, rel=1e-9, abs=1e-12)
        assert result.residual[t] == pytest.approx(chunk[-1] - fitted, abs=1e-12)
        assert result.r_squared[t] == pytest.approx(1.0 - sse / sst, rel=1e-8)


def test_rolling_ols_matches_lstsq():
    """Independent check against the general least-squares solver."""
    rng = np.random.default_rng(19)
    y = np.cumsum(rng.normal(0.0, 0.002, 400)) + 6.0
    window = 48
    result = rolling_ols(y, window)
    design = np.column_stack([np.ones(window), np.arange(window, dtype=np.float64)])
    for t in (window - 1, 150, 399):
        chunk = y[t - window + 1 : t + 1]
        (intercept, slope), *_ = np.linalg.lstsq(design, chunk, rcond=None)
        assert result.slope[t] == pytest.approx(slope, rel=1e-9)
        assert result.intercept[t] == pytest.approx(intercept, rel=1e-9)


def test_vectorised_path_matches_the_single_window_reference():
    """rolling_ols must agree with fit_window, its own reference implementation."""
    rng = np.random.default_rng(23)
    y = np.cumsum(rng.normal(0.0, 0.0015, 500)) + np.log(1800.0)
    window = 96
    result = rolling_ols(y, window)
    for t in (window - 1, 200, 350, 499):
        reference = fit_window(y[t - window + 1 : t + 1])
        assert result.slope[t] == pytest.approx(reference.slope, rel=1e-10)
        assert result.residual[t] == pytest.approx(reference.residual_last, abs=1e-14)
        assert result.r_squared[t] == pytest.approx(reference.r_squared, rel=1e-10)
        assert result.residual_std[t] == pytest.approx(reference.residual_std, rel=1e-10)


def test_chunking_does_not_change_results():
    """Peak-memory chunking is an implementation detail, not a behaviour."""
    rng = np.random.default_rng(31)
    y = np.cumsum(rng.normal(0.0, 0.001, 2000)) + 6.0
    window = 64
    big = rolling_ols(y, window, chunk_elements=10_000_000)
    small = rolling_ols(y, window, chunk_elements=10_000)
    for name in ("slope", "intercept", "residual", "r_squared", "residual_std"):
        assert np.array_equal(
            getattr(big, name), getattr(small, name), equal_nan=True
        ), name


# ---------------------------------------------------------------------------
# Window boundaries
# ---------------------------------------------------------------------------
def test_exactly_n_observations_are_used():
    """A step change must enter the window exactly when it should."""
    window = 20
    y = np.zeros(100)
    y[50] = 1.0                       # a single spike
    result = rolling_ols(y, window)
    # The spike is inside the window for t = 50 .. 50+window-1 and nowhere else.
    affected = np.flatnonzero(np.abs(result.slope) > 1e-12)
    assert affected.min() == 50
    assert affected.max() == 50 + window - 1
    assert result.n_observations[60] == window


def test_first_valid_index_is_the_window_minus_one():
    for window in (5, 32, 100):
        result = rolling_ols(np.arange(300, dtype=np.float64), window)
        first = int(np.argmax(np.isfinite(result.slope)))
        assert first == window - 1
        assert np.all(np.isnan(result.slope[: window - 1]))


def test_series_shorter_than_the_window_produces_no_fits():
    result = rolling_ols(np.arange(10, dtype=np.float64), 50)
    assert np.all(np.isnan(result.residual))
    assert any("fewer than" in note for note in result.notes)


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------
def test_flat_prices_give_nan_not_infinity():
    """Zero residual variance must never produce an infinite Z-score."""
    window = 32
    y = np.full(200, np.log(400.0))
    result = rolling_ols(y, window)
    valid = np.isfinite(result.slope)
    assert result.slope[valid] == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.isnan(result.r_squared[valid])), "R^2 of a flat window is undefined"
    assert np.all(np.isnan(result.residual_std[valid]))
    assert result.degenerate_windows > 0
    assert any("degenerate" in note or "flat" in note for note in result.notes)


def test_flat_prices_produce_nan_zscore_in_the_feature_table(regression_config):
    features, diagnostics = rolling_regression_features(
        bars_from_prices(np.full(200, 400.0)), window=32,
        config=regression_config, timeframe="5m",
    )
    z = features["residual_zscore_fit"].to_numpy().astype(np.float64)
    assert not np.any(np.isinf(z)), "an infinite Z-score escaped"
    assert np.all(np.isnan(z[31:])), "a flat window must give NaN, not a number"
    assert diagnostics["degenerate_windows"] > 0


def test_a_perfect_trend_also_gives_nan_zscore(regression_config):
    """R^2 = 1 means zero residual variance, so Z is undefined, not infinite."""
    prices = 400.0 * np.exp(0.0001 * np.arange(300))
    features, _ = rolling_regression_features(
        bars_from_prices(prices), window=64, config=regression_config, timeframe="5m"
    )
    z = features["residual_zscore_fit"].to_numpy().astype(np.float64)
    assert not np.any(np.isinf(z))


def test_non_positive_prices_are_rejected_for_a_log_model(regression_config):
    prices = np.full(200, 400.0)
    prices[50] = -1.0
    with pytest.raises(ValueError, match="non-positive prices"):
        rolling_regression_features(
            bars_from_prices(prices), window=32,
            config=regression_config, timeframe="5m",
        )


# ---------------------------------------------------------------------------
# The feature table
# ---------------------------------------------------------------------------
def test_feature_table_has_both_zscore_methods(regression_config):
    rng = np.random.default_rng(7)
    prices = 400.0 * np.exp(np.cumsum(rng.normal(0, 0.001, 800)))
    features, diagnostics = rolling_regression_features(
        bars_from_prices(prices), window=64, config=regression_config, timeframe="5m"
    )
    assert "residual_zscore_fit" in features.columns
    assert "residual_zscore_rolling" in features.columns
    # They answer different questions and must not be identical.
    a = features["residual_zscore_fit"].to_numpy()
    b = features["residual_zscore_rolling"].to_numpy()
    both = np.isfinite(a) & np.isfinite(b)
    assert both.sum() > 100
    assert not np.allclose(a[both], b[both]), "the two Z methods collapsed into one"
    assert diagnostics["zscore_rolling_primary_window"] == 64


def test_zscore_fit_is_residual_over_fit_standard_error(regression_config):
    """Method A is exactly eps_t / s_fit, with no mean subtracted."""
    rng = np.random.default_rng(13)
    prices = 400.0 * np.exp(np.cumsum(rng.normal(0, 0.001, 400)))
    features, _ = rolling_regression_features(
        bars_from_prices(prices), window=50, config=regression_config, timeframe="5m"
    )
    residual = features["residual"].to_numpy()
    std = features["residual_std_fit"].to_numpy()
    z = features["residual_zscore_fit"].to_numpy()
    ok = np.isfinite(residual) & np.isfinite(std) & np.isfinite(z)
    assert ok.sum() > 100
    assert z[ok] == pytest.approx(residual[ok] / std[ok], rel=1e-12)


def test_fitted_price_is_the_exponential_of_the_fitted_log_price(regression_config):
    rng = np.random.default_rng(17)
    prices = 400.0 * np.exp(np.cumsum(rng.normal(0, 0.001, 300)))
    features, _ = rolling_regression_features(
        bars_from_prices(prices), window=40, config=regression_config, timeframe="5m"
    )
    log_fit = features["fitted_log_price"].to_numpy()
    price_fit = features["fitted_price"].to_numpy()
    ok = np.isfinite(log_fit)
    assert price_fit[ok] == pytest.approx(np.exp(log_fit[ok]), rel=1e-12)


def test_residual_equals_log_price_minus_fitted(regression_config):
    rng = np.random.default_rng(29)
    prices = 400.0 * np.exp(np.cumsum(rng.normal(0, 0.001, 300)))
    features, _ = rolling_regression_features(
        bars_from_prices(prices), window=40, config=regression_config, timeframe="5m"
    )
    y = features["log_price"].to_numpy()
    fitted = features["fitted_log_price"].to_numpy()
    residual = features["residual"].to_numpy()
    ok = np.isfinite(fitted)
    assert residual[ok] == pytest.approx(y[ok] - fitted[ok], abs=1e-14)


# ---------------------------------------------------------------------------
# Robust comparison
# ---------------------------------------------------------------------------
def test_theil_sen_recovers_a_clean_slope():
    y = 3.0 + 2.0 * np.arange(40, dtype=np.float64)
    assert theil_sen_slope(y) == pytest.approx(2.0, abs=1e-12)


def test_an_outlier_at_the_window_centre_barely_moves_the_slope():
    """Leverage is lowest at the middle: a central outlier hits R^2, not beta.

    Worth pinning down, because it is the reason a low R^2 does not imply a
    distorted trend estimate.
    """
    y = 3.0 + 2.0 * np.arange(40, dtype=np.float64)
    y[20] += 500.0
    fit = fit_window(y)
    assert fit.slope == pytest.approx(2.0, abs=0.1), "central outlier moved the slope"
    assert fit.r_squared < 0.2, "R^2 should collapse even though beta survives"
    assert theil_sen_slope(y) == pytest.approx(2.0, abs=1e-9)


def test_theil_sen_resists_a_high_leverage_outlier_that_moves_ols():
    """At the window edge leverage is maximal, and OLS does move."""
    y = 3.0 + 2.0 * np.arange(40, dtype=np.float64)
    y[-1] += 500.0
    ols_slope = fit_window(y).slope
    assert ols_slope > 3.0, f"a high-leverage outlier should drag OLS, got {ols_slope}"
    assert theil_sen_slope(y) == pytest.approx(2.0, abs=1e-9), (
        "Theil-Sen should be unmoved"
    )


# ---------------------------------------------------------------------------
# Missing values: deterministic, contained, never interpolated
# ---------------------------------------------------------------------------
def test_a_missing_price_poisons_only_the_windows_that_contain_it():
    """One NaN invalidates exactly the N windows spanning it, and no others."""
    window = 32
    n = 200
    rng = np.random.default_rng(3)
    y = np.log(400.0) + np.cumsum(rng.normal(0.0, 0.001, n))
    hole = 100
    y[hole] = np.nan

    result = rolling_ols(y, window)
    invalid = set(np.flatnonzero(~np.isfinite(result.residual)).tolist())
    expected = set(range(window - 1)) | set(range(hole, hole + window))
    assert invalid == expected, "a missing price leaked outside its own windows"


def test_missing_value_handling_is_deterministic():
    """Same input, same output - bit for bit, across repeated calls."""
    window = 48
    rng = np.random.default_rng(9)
    y = np.log(400.0) + np.cumsum(rng.normal(0.0, 0.001, 400))
    y[[50, 51, 200]] = np.nan

    first = rolling_ols(y, window)
    second = rolling_ols(y, window)
    for name in ("slope", "intercept", "residual", "r_squared", "residual_std"):
        assert np.array_equal(
            getattr(first, name), getattr(second, name), equal_nan=True
        ), f"{name} differed between two identical calls"


def test_a_null_close_is_not_interpolated(regression_config):
    """A gap stays a gap: no fill, no carry-forward, no silent repair."""
    window = 32
    n = 300
    rng = np.random.default_rng(4)
    bars = bars_from_prices(400.0 * np.exp(np.cumsum(rng.normal(0.0, 0.001, n))))
    holed = bars.with_columns(
        pl.when(pl.int_range(pl.len()) == 150)
        .then(None)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    features, diagnostics = rolling_regression_features(
        holed, window=window, config=regression_config, timeframe="5m"
    )
    # Warm-up (31) plus the 32 windows containing the hole.
    assert features["residual"].null_count() == (window - 1) + window
    assert features["residual"][150:150 + window].null_count() == window
    assert features["residual"][150 + window] is not None
    assert diagnostics["valid_fits"] == n - (window - 1) - window
