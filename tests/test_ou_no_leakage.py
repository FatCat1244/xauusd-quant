"""No look-ahead in the OU features, checked to exact equality.

For a timestamp t, the rolling OU parameters are computed from the residuals
ending at t. Appending future residuals - or future bars, end to end through
the regression - must not change a single historical value: a, b, theta, mu,
sigma, the half-life, the stationary std, the OU Z-score, the innovation or
the state. The computation is deterministic, so equality is exact, not
approximate; the block structure of the rolling sums is designed so that a
longer series never changes an earlier block.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from xauusd_quant.features import load_regression_config
from xauusd_quant.features.rolling_regression import rolling_regression_features
from xauusd_quant.models import load_ou_config
from xauusd_quant.models.ornstein_uhlenbeck import simulate_ou
from xauusd_quant.research.ou_estimation import ou_parameter_frame

COLUMNS = (
    "ou_a", "ou_b", "ou_theta", "ou_mu", "ou_sigma", "ou_half_life_bars",
    "ou_stationary_std", "ou_zscore", "ou_innovation", "ou_innovation_standardized",
    "ou_r_squared", "ou_window_std", "ou_std_ratio",
)


def assert_identical(a: pl.DataFrame, b: pl.DataFrame, columns=COLUMNS) -> None:
    for column in columns:
        left = a[column].fill_null(np.nan).to_numpy()
        right = b[column].fill_null(np.nan).to_numpy()
        same = (left == right) | (np.isnan(left) & np.isnan(right))
        assert same.all(), f"{column} changed at {np.flatnonzero(~same)[:5]}"
    assert a["ou_state"].to_list() == b["ou_state"].to_list()


@pytest.fixture(scope="module")
def ou_config():
    return load_ou_config()


@pytest.mark.parametrize("window", [64, 256])
def test_appending_future_residuals_changes_nothing_before_them(ou_config, window):
    series = simulate_ou(0.08, 0.0, 1e-3, 30_000, seed=1)
    cutoff = 20_000                      # straddles several rolling-sum blocks
    short = ou_parameter_frame(series[:cutoff], None, ou_window=window, config=ou_config)
    full = ou_parameter_frame(series, None, ou_window=window, config=ou_config)
    assert_identical(short, full.head(cutoff))


def test_a_huge_future_shock_does_not_move_the_past(ou_config):
    series = simulate_ou(0.08, 0.0, 1e-3, 5_000, seed=2)
    shocked = series.copy()
    shocked[4_000:] += 50.0              # absurd jump, entirely in the future
    a = ou_parameter_frame(series, None, ou_window=128, config=ou_config)
    b = ou_parameter_frame(shocked, None, ou_window=128, config=ou_config)
    assert_identical(a.head(4_000), b.head(4_000))
    assert not np.allclose(a["ou_b"].fill_null(0).to_numpy()[4_050:],
                           b["ou_b"].fill_null(0).to_numpy()[4_050:])


def test_the_first_values_wait_for_a_full_window_and_are_never_backfilled(ou_config):
    series = simulate_ou(0.1, 0.0, 1.0, 1_000, seed=3)
    frame = ou_parameter_frame(series, None, ou_window=128, config=ou_config)
    b = frame["ou_b"]
    assert b[:127].null_count() == 127
    assert b[127] is not None
    assert frame["ou_state"][:127].to_list() == ["insufficient_data"] * 127
    assert frame["ou_innovation"][:128].null_count() == 128, "innovation needs the fit at t-1"
    for column in COLUMNS:
        assert frame[column].is_nan().sum() == 0, f"{column} carries NaN instead of null"


def test_missing_residuals_make_their_windows_insufficient_not_bridged(ou_config):
    series = simulate_ou(0.1, 0.0, 1.0, 2_000, seed=4)
    series[1_000] = np.nan
    frame = ou_parameter_frame(series, None, ou_window=64, config=ou_config)
    states = frame["ou_state"].to_list()
    assert all(s == "insufficient_data" for s in states[1_000:1_064])
    assert states[1_064] != "insufficient_data"


def make_bars(n: int, *, seed: int = 11) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    prices = 400.0 * np.exp(np.cumsum(rng.normal(0.0, 0.0008, n)))
    start = datetime(2024, 1, 2, 1, 0)
    return pl.DataFrame({
        "timestamp": [start + timedelta(minutes=5 * i) for i in range(n)],
        "open": prices, "high": prices * 1.0002, "low": prices * 0.9998, "close": prices,
        "first_bid": prices - 0.2, "first_ask": prices + 0.2,
        "last_bid": prices - 0.2, "last_ask": prices + 0.2,
        "tick_count": np.full(n, 100, dtype=np.int64), "volume": np.full(n, 1000.0),
        "mean_spread": np.full(n, 0.4),
    })


def test_end_to_end_future_bars_change_no_historical_ou_value(ou_config):
    """Bars -> rolling regression -> residual -> rolling OU, truncated vs full."""
    regression = load_regression_config()
    bars = make_bars(4_000)
    cutoff = 3_000

    def ou_from(frame: pl.DataFrame) -> pl.DataFrame:
        features, _ = rolling_regression_features(frame, window=64, config=regression,
                                                  timeframe="5m")
        residual = features["residual"].fill_null(np.nan).to_numpy()
        return ou_parameter_frame(residual, features["timestamp"], ou_window=128,
                                  config=ou_config)

    short = ou_from(bars.head(cutoff))
    full = ou_from(bars).head(cutoff)
    assert short["timestamp"].to_list() == full["timestamp"].to_list()
    assert_identical(short, full)
    # Warm-up spans both windows: N-1 regression bars, then M-1 residuals.
    first = short["ou_b"].to_numpy()
    assert np.isnan(first[: 63 + 127]).all() and np.isfinite(first[63 + 127])
