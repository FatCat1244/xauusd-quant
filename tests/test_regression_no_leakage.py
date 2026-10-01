"""Look-ahead verification for the rolling-regression features.

The contract: a feature at time ``t`` is a function of bars ``t-N+1 .. t`` and
nothing else. Appending later data must not change a single historical value.

These are the most important tests in Prompt #3. A leak here would make every
downstream residual statistic optimistic in a way no amount of careful
reporting could undo, so each feature column is checked individually and to
exact equality - not `approx` - because the computation is genuinely
deterministic and bit-identical.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from xauusd_quant.features import load_regression_config, rolling_ols
from xauusd_quant.features.rolling_regression import rolling_regression_features

FEATURE_COLUMNS = (
    "regression_intercept",
    "regression_slope",
    "fitted_log_price",
    "fitted_price",
    "residual",
    "residual_pct",
    "residual_std_fit",
    "r_squared",
    "residual_zscore_fit",
    "residual_zscore_rolling",
)


@pytest.fixture
def regression_config():
    return load_regression_config()


def make_bars(n: int, *, seed: int = 11, start: datetime | None = None) -> pl.DataFrame:
    """Synthetic bars with a realistic log-price path."""
    rng = np.random.default_rng(seed)
    prices = 400.0 * np.exp(np.cumsum(rng.normal(0.0, 0.0008, n)))
    start = start or datetime(2024, 1, 2, 1, 0)
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


# ---------------------------------------------------------------------------
# The core guarantee
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("window", [32, 64, 128])
def test_appending_future_bars_changes_no_historical_feature(regression_config, window):
    """Truncated dataset vs full dataset: values at shared timestamps must match."""
    full = make_bars(1200)
    cutoff = 800
    truncated = full.head(cutoff)

    a, _ = rolling_regression_features(
        truncated, window=window, config=regression_config, timeframe="5m"
    )
    b, _ = rolling_regression_features(
        full, window=window, config=regression_config, timeframe="5m"
    )
    b = b.head(cutoff)

    assert a["timestamp"].to_list() == b["timestamp"].to_list()
    for column in FEATURE_COLUMNS:
        left = a[column].to_numpy().astype(np.float64)
        right = b[column].to_numpy().astype(np.float64)
        assert np.array_equal(left, right, equal_nan=True), (
            f"{column} changed when future bars were appended (window={window})"
        )


def test_a_huge_future_spike_cannot_reach_back(regression_config):
    """The sharpest form: one enormous price far in the future."""
    window = 64
    base = make_bars(600)
    spiked = base.with_columns(
        pl.when(pl.int_range(pl.len()) == 500)
        .then(pl.col("close") * 5.0)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    a, _ = rolling_regression_features(
        base, window=window, config=regression_config, timeframe="5m"
    )
    b, _ = rolling_regression_features(
        spiked, window=window, config=regression_config, timeframe="5m"
    )
    # Everything strictly before the spike's own window must be untouched.
    for column in ("regression_slope", "residual", "r_squared", "residual_zscore_fit"):
        left = a[column].to_numpy()[: 500]
        right = b[column].to_numpy()[: 500]
        assert np.array_equal(left, right, equal_nan=True), f"{column} leaked backwards"

    # And the spike must affect its own bar onward, or the test proves nothing.
    assert not np.array_equal(
        a["residual"].to_numpy()[500:520],
        b["residual"].to_numpy()[500:520],
        equal_nan=True,
    ), "the spike had no forward effect; the test is vacuous"


def test_bar_t_depends_only_on_its_own_window(regression_config):
    """Perturbing a bar outside the window must not change the value at t."""
    window = 32
    n = 300
    bars = make_bars(n)
    target = 200

    a, _ = rolling_regression_features(
        bars, window=window, config=regression_config, timeframe="5m"
    )
    # Change a bar well before the window [t-31, t].
    outside = target - window - 10
    perturbed = bars.with_columns(
        pl.when(pl.int_range(pl.len()) == outside)
        .then(pl.col("close") * 1.5)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    b, _ = rolling_regression_features(
        perturbed, window=window, config=regression_config, timeframe="5m"
    )
    for column in ("regression_slope", "regression_intercept", "residual", "r_squared"):
        assert a[column][target] == b[column][target], (
            f"{column} at t depended on a bar outside its window"
        )


def test_rolling_ols_is_prefix_stable():
    """Growing the series one bar at a time reproduces the same history."""
    rng = np.random.default_rng(5)
    y = np.cumsum(rng.normal(0, 0.001, 400)) + np.log(400)
    window = 50
    full = rolling_ols(y, window)
    for cutoff in (120, 250, 399):
        partial = rolling_ols(y[:cutoff], window)
        assert np.array_equal(
            partial.residual, full.residual[:cutoff], equal_nan=True
        ), f"residual history changed at cutoff {cutoff}"
        assert np.array_equal(
            partial.slope, full.slope[:cutoff], equal_nan=True
        ), f"slope history changed at cutoff {cutoff}"


# ---------------------------------------------------------------------------
# Warm-up must not be filled
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("window", [32, 128, 256])
def test_warmup_bars_are_null_not_backfilled(regression_config, window):
    """The first N-1 bars have no full window and must stay empty."""
    bars = make_bars(window * 4)
    features, diagnostics = rolling_regression_features(
        bars, window=window, config=regression_config, timeframe="5m"
    )
    for column in ("regression_slope", "residual", "r_squared", "fitted_log_price"):
        head = features[column].to_numpy()[: window - 1]
        assert np.all(np.isnan(head)), f"{column} was filled during warm-up"
    assert features["residual"].to_numpy()[window - 1] is not None
    assert np.isfinite(features["residual"].to_numpy()[window - 1])
    assert diagnostics["warmup_bars"] == window - 1


@pytest.mark.parametrize("window", [32, 128, 256])
def test_missing_values_are_null_and_never_nan(regression_config, window):
    """Polars keeps NaN through `is_not_null()`; missing must be null.

    Every consumer filters with `filter(pl.col("residual").is_not_null())`. If
    warm-up rows were NaN instead of null they would survive that filter, which
    inflates the reported fit count by N-1 and - because NaN sorts high in
    Polars - drags every conditioning quantile edge upward.
    """
    n = window * 4
    features, _ = rolling_regression_features(
        make_bars(n), window=window, config=regression_config, timeframe="5m"
    )
    for column in FEATURE_COLUMNS:
        series = features[column]
        assert series.is_nan().sum() == 0, f"{column} contains NaN rather than null"

    assert features["residual"].null_count() == window - 1
    valid = features.filter(pl.col("residual").is_not_null())
    assert valid.height == n - (window - 1)
    assert np.all(np.isfinite(valid["residual"].to_numpy()))


def test_quantile_edges_are_not_dragged_by_warmup_rows(regression_config):
    """The concrete consequence: conditioning buckets must use real fits only."""
    window = 64
    features, _ = rolling_regression_features(
        make_bars(500), window=window, config=regression_config, timeframe="5m"
    )
    valid = features.filter(pl.col("r_squared").is_not_null())
    finite = valid["r_squared"].to_numpy()
    assert np.all(np.isfinite(finite))
    for quantile in (0.25, 0.5, 0.75):
        polars_edge = valid.select(pl.col("r_squared").quantile(quantile)).item()
        assert polars_edge == pytest.approx(
            float(np.quantile(finite, quantile)), rel=1e-6
        ), f"quantile {quantile} disagrees with the finite-only value"


def test_warmup_is_not_affected_by_series_length(regression_config):
    """A longer series must not retroactively grant the early bars a fit."""
    window = 64
    short, _ = rolling_regression_features(
        make_bars(200), window=window, config=regression_config, timeframe="5m"
    )
    long, _ = rolling_regression_features(
        make_bars(2000), window=window, config=regression_config, timeframe="5m"
    )
    assert np.isnan(short["residual"].to_numpy()[: window - 1]).all()
    assert np.isnan(long["residual"].to_numpy()[: window - 1]).all()


# ---------------------------------------------------------------------------
# The rolling Z-score is causal too
# ---------------------------------------------------------------------------
def test_rolling_zscore_uses_only_past_residuals(regression_config):
    """Method B must not see future residuals either."""
    window = 32
    full = make_bars(800)
    cutoff = 500
    a, _ = rolling_regression_features(
        full.head(cutoff), window=window, config=regression_config, timeframe="5m"
    )
    b, _ = rolling_regression_features(
        full, window=window, config=regression_config, timeframe="5m"
    )
    left = a["residual_zscore_rolling"].to_numpy().astype(np.float64)
    right = b["residual_zscore_rolling"].to_numpy().astype(np.float64)[:cutoff]
    assert np.array_equal(left, right, equal_nan=True)


def test_every_rolling_zscore_variant_is_causal(regression_config):
    """All configured Method B windows, not just the primary one."""
    window = 64
    full = make_bars(900)
    cutoff = 600
    a, _ = rolling_regression_features(
        full.head(cutoff), window=window, config=regression_config, timeframe="5m"
    )
    b, _ = rolling_regression_features(
        full, window=window, config=regression_config, timeframe="5m"
    )
    variants = [c for c in a.columns if c.startswith("residual_zscore_rolling_")]
    assert variants, "no rolling Z-score variants were produced"
    for column in variants:
        assert np.array_equal(
            a[column].to_numpy().astype(np.float64),
            b[column].to_numpy().astype(np.float64)[:cutoff],
            equal_nan=True,
        ), f"{column} is not causal"


def test_trailing_volatility_is_causal(regression_config):
    """The conditioning variable must be causal as well as the features."""
    window = 32
    full = make_bars(700)
    cutoff = 450
    a, _ = rolling_regression_features(
        full.head(cutoff), window=window, config=regression_config, timeframe="5m"
    )
    b, _ = rolling_regression_features(
        full, window=window, config=regression_config, timeframe="5m"
    )
    assert np.array_equal(
        a["trailing_volatility"].to_numpy().astype(np.float64),
        b["trailing_volatility"].to_numpy().astype(np.float64)[:cutoff],
        equal_nan=True,
    )


# ---------------------------------------------------------------------------
# No forward columns anywhere in the feature table
# ---------------------------------------------------------------------------
def test_feature_table_contains_no_forward_columns(regression_config):
    """Forward returns belong to outcome analysis, never to features."""
    features, _ = rolling_regression_features(
        make_bars(500), window=64, config=regression_config, timeframe="5m"
    )
    leaked = [c for c in features.columns if c.startswith("fwd_") or "future" in c.lower()]
    assert not leaked, f"forward-looking columns in the feature table: {leaked}"
