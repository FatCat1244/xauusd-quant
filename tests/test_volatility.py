"""Volatility estimators, annualisation policy and regime buckets."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from xauusd_quant.research.volatility import (
    annualization_factor,
    ewma_volatility,
    realized_volatility,
    rolling_volatility,
    volatility_clustering,
    volatility_regimes,
)


# ---------------------------------------------------------------------------
# Rolling standard deviation
# ---------------------------------------------------------------------------
def test_rolling_volatility_matches_numpy():
    values = [0.01, -0.02, 0.015, 0.005, -0.03, 0.02, 0.0, -0.01]
    frame = pl.DataFrame({"r": values})
    out = frame.select(rolling_volatility(pl.col("r"), 3).alias("v"))["v"].to_list()
    assert out[:2] == [None, None], "an incomplete window must not emit a value"
    for i in range(2, len(values)):
        expected = float(np.std(values[i - 2 : i + 1], ddof=1))
        assert out[i] == pytest.approx(expected), f"index {i}"


def test_rolling_volatility_uses_only_the_trailing_window():
    """Appending future data must not change an earlier value."""
    base = [0.01, -0.02, 0.015, 0.005, -0.03]
    short = pl.DataFrame({"r": base}).select(
        rolling_volatility(pl.col("r"), 3).alias("v"))["v"].to_list()
    long = pl.DataFrame({"r": [*base, 9.9, -9.9]}).select(
        rolling_volatility(pl.col("r"), 3).alias("v"))["v"].to_list()
    assert short == pytest.approx(long[: len(base)], nan_ok=True)


def test_constant_series_has_zero_volatility():
    frame = pl.DataFrame({"r": [0.01] * 20})
    out = frame.select(rolling_volatility(pl.col("r"), 5).alias("v"))["v"].to_list()
    assert all(v == pytest.approx(0.0) for v in out[4:])


@pytest.mark.parametrize("window", [0, 1])
def test_rolling_volatility_rejects_tiny_windows(window):
    with pytest.raises(ValueError, match="window must be >= 2"):
        rolling_volatility(pl.col("r"), window)


# ---------------------------------------------------------------------------
# Realized volatility
# ---------------------------------------------------------------------------
def test_realized_volatility_is_the_root_of_summed_squares():
    values = [0.01, -0.02, 0.015, 0.005]
    frame = pl.DataFrame({"r": values})
    out = frame.select(realized_volatility(pl.col("r"), 4).alias("rv"))["rv"].to_list()
    assert out[3] == pytest.approx(float(np.sqrt(np.sum(np.square(values)))))


def test_realized_volatility_differs_from_rolling_std_by_the_mean_term():
    """RV does not subtract a mean; rolling std does."""
    values = [0.02] * 10          # non-zero mean, zero variance
    frame = pl.DataFrame({"r": values})
    rv = frame.select(realized_volatility(pl.col("r"), 5).alias("v"))["v"].to_list()[-1]
    sd = frame.select(rolling_volatility(pl.col("r"), 5).alias("v"))["v"].to_list()[-1]
    assert sd == pytest.approx(0.0)
    assert rv == pytest.approx(float(np.sqrt(5 * 0.02**2)))


# ---------------------------------------------------------------------------
# EWMA
# ---------------------------------------------------------------------------
def test_ewma_volatility_tracks_a_variance_regime_shift():
    rng = np.random.default_rng(5)
    quiet = rng.normal(0, 0.001, 500)
    loud = rng.normal(0, 0.010, 500)
    out = ewma_volatility(np.concatenate([quiet, loud]), lambda_=0.94, warmup=50)
    assert out[400] < out[900], "EWMA must rise into the high-variance regime"
    assert out[900] > 5 * out[400]


def test_ewma_uses_only_past_returns():
    """sigma_t is built from r_{t-1} and earlier, never r_t."""
    values = np.concatenate([np.full(100, 0.001), np.array([10.0]), np.full(20, 0.001)])
    out = ewma_volatility(values, lambda_=0.94, warmup=20)
    spike_index = 100
    # The huge return at t=100 must not affect sigma at t=100, only from t=101.
    assert out[spike_index] == pytest.approx(out[spike_index - 1], rel=0.05)
    assert out[spike_index + 1] > 10 * out[spike_index]


def test_ewma_warmup_prefix_is_nan():
    out = ewma_volatility(np.random.default_rng(1).normal(0, 0.01, 100), warmup=30)
    assert np.all(np.isnan(out[:30]))
    assert np.all(np.isfinite(out[30:]))


@pytest.mark.parametrize("lam", [0.0, 1.0, -0.5, 1.5])
def test_ewma_rejects_invalid_lambda(lam):
    with pytest.raises(ValueError, match="lambda must be in"):
        ewma_volatility(np.zeros(10), lambda_=lam)


# ---------------------------------------------------------------------------
# Annualisation is opt-in
# ---------------------------------------------------------------------------
def test_annualization_is_disabled_by_default(research_config):
    info = annualization_factor("5m", research_config)
    assert info.enabled is False
    assert info.factor is None
    assert "equity conventions" in info.note or "252" in info.note


def test_annualization_uses_xauusd_hours_when_enabled(research_config_factory):
    config = research_config_factory(
        volatility={"annualization": {
            "enabled": True, "trading_days_per_year": 260, "hours_per_trading_day": 23.0,
        }}
    )
    info = annualization_factor("1h", config)
    assert info.enabled is True
    # 23 bars/day x 260 days = 5,980 one-hour bars per year.
    assert info.bars_per_year == pytest.approx(5980.0)
    assert info.factor == pytest.approx(np.sqrt(5980.0))
    assert "260" in info.note


def test_explicit_bars_per_year_overrides_the_derivation(research_config_factory):
    config = research_config_factory(
        volatility={"annualization": {"enabled": True, "bars_per_year": 1000.0}}
    )
    info = annualization_factor("5m", config)
    assert info.bars_per_year == 1000.0
    assert info.factor == pytest.approx(np.sqrt(1000.0))


# ---------------------------------------------------------------------------
# Clustering and regimes
# ---------------------------------------------------------------------------
def test_volatility_clustering_table_covers_all_three_series():
    rng = np.random.default_rng(9)
    magnitude = np.abs(rng.normal(0, 1, 5000))
    returns = magnitude * rng.choice([-1.0, 1.0], 5000)
    table = volatility_clustering(returns, max_lag=5)
    assert set(table["series"].unique()) == {
        "returns", "abs_returns", "squared_returns"
    }
    assert table.height == 15


def test_volatility_regimes_bucket_by_trailing_volatility():
    rng = np.random.default_rng(4)
    n = 2000
    returns = np.concatenate([
        rng.normal(0, 0.0005, n // 2), rng.normal(0, 0.005, n // 2)
    ])
    frame = pl.DataFrame({"ret_log": returns}).with_columns(
        rolling_volatility(pl.col("ret_log"), 50).alias("vol_50")
    )
    regimes = volatility_regimes(
        frame, return_column="ret_log", volatility_column="vol_50", window=50
    )
    table = regimes.table.sort("volatility_regime")
    assert table.height == 4
    volatilities = table["mean_trailing_volatility"].to_list()
    assert volatilities == sorted(volatilities), "buckets must be ordered by volatility"
    assert "Q1_lowest" in table["volatility_regime"].to_list()
    assert any("not a regime model" in n for n in regimes.notes)


def test_volatility_regimes_handles_an_empty_frame():
    frame = pl.DataFrame({"ret_log": [None] * 5, "vol_50": [None] * 5},
                         schema={"ret_log": pl.Float64, "vol_50": pl.Float64})
    regimes = volatility_regimes(
        frame, return_column="ret_log", volatility_column="vol_50", window=50
    )
    assert regimes.table.is_empty()
    assert regimes.notes
