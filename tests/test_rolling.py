"""Rolling statistics: correctness, and the causality guarantee.

The central property under test is that a value at ``t`` cannot change when
data after ``t`` arrives. That is what makes these usable as features later.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy import stats

from conftest import make_bars
from xauusd_quant.research.rolling import (
    RollingSpec,
    rolling_autocorrelation,
    rolling_frame,
    rolling_kurtosis,
    rolling_mean,
    rolling_skewness,
    rolling_statistics,
    rolling_variance,
)


def series_of(values: list[float]) -> pl.Series:
    return pl.Series("r", values, dtype=pl.Float64)


# ---------------------------------------------------------------------------
# Correctness against NumPy / SciPy
# ---------------------------------------------------------------------------
def test_rolling_mean_matches_numpy():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    out = pl.DataFrame({"r": values}).select(
        rolling_mean(pl.col("r"), 3).alias("m"))["m"].to_list()
    assert out[:2] == [None, None]
    for i in range(2, len(values)):
        assert out[i] == pytest.approx(float(np.mean(values[i - 2 : i + 1])))


def test_rolling_variance_matches_numpy():
    values = [1.0, 3.0, 2.0, 8.0, 5.0, 4.0]
    out = pl.DataFrame({"r": values}).select(
        rolling_variance(pl.col("r"), 4).alias("v"))["v"].to_list()
    for i in range(3, len(values)):
        assert out[i] == pytest.approx(float(np.var(values[i - 3 : i + 1], ddof=1)))


def test_rolling_skewness_matches_scipy():
    rng = np.random.default_rng(4)
    values = rng.gamma(2.0, 1.0, 200)
    out = rolling_skewness(series_of(values.tolist()), 50)
    for i in (49, 100, 199):
        expected = float(stats.skew(values[i - 49 : i + 1], bias=False))
        assert out[i] == pytest.approx(expected)


def test_rolling_kurtosis_matches_scipy_and_is_excess():
    rng = np.random.default_rng(6)
    values = rng.normal(0, 1, 500)
    out = rolling_kurtosis(series_of(values.tolist()), 100)
    expected = float(stats.kurtosis(values[100 - 100 : 100], fisher=True, bias=False))
    assert out[99] == pytest.approx(expected)
    # Gaussian data should give excess kurtosis near zero.
    assert abs(np.nanmean(out)) < 1.0


def test_rolling_autocorrelation_recovers_known_structure():
    """Alternating +/-1 has lag-1 autocorrelation of -1."""
    values = [1.0, -1.0] * 100
    out = rolling_autocorrelation(series_of(values), window=50, lag=1)
    assert out[60] == pytest.approx(-1.0, abs=1e-9)
    out2 = rolling_autocorrelation(series_of(values), window=50, lag=2)
    assert out2[60] == pytest.approx(1.0, abs=1e-9)


def test_rolling_autocorrelation_rejects_a_bad_lag():
    with pytest.raises(ValueError, match="lag must be >= 1"):
        rolling_autocorrelation(series_of([1.0, 2.0]), window=5, lag=0)


# ---------------------------------------------------------------------------
# Causality: the property that matters
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "fn", [rolling_mean, rolling_variance], ids=["mean", "variance"]
)
def test_expression_statistics_ignore_future_data(fn):
    base = [1.0, 2.0, 3.0, 4.0, 5.0]
    short = pl.DataFrame({"r": base}).select(fn(pl.col("r"), 3).alias("x"))["x"].to_list()
    extended = pl.DataFrame({"r": [*base, 1000.0, -1000.0]}).select(
        fn(pl.col("r"), 3).alias("x"))["x"].to_list()
    assert short == pytest.approx(extended[: len(base)], nan_ok=True)


@pytest.mark.parametrize(
    "fn", [rolling_skewness, rolling_kurtosis], ids=["skew", "kurtosis"]
)
def test_moment_statistics_ignore_future_data(fn):
    rng = np.random.default_rng(12)
    base = rng.normal(0, 1, 300).tolist()
    short = fn(series_of(base), 50)
    extended = fn(series_of([*base, 500.0, -500.0]), 50)
    np.testing.assert_allclose(short, extended[: len(base)], equal_nan=True)


def test_rolling_autocorrelation_ignores_future_data():
    rng = np.random.default_rng(13)
    base = rng.normal(0, 1, 300).tolist()
    short = rolling_autocorrelation(series_of(base), 50, lag=1)
    extended = rolling_autocorrelation(series_of([*base, 99.0, -99.0]), 50, lag=1)
    np.testing.assert_allclose(short, extended[: len(base)], equal_nan=True)


def test_a_future_spike_cannot_change_an_earlier_value():
    """The sharpest form of the test: one huge value far in the future."""
    values = [0.001] * 200
    clean = rolling_mean(pl.col("r"), 20)
    a = pl.DataFrame({"r": values}).select(clean.alias("m"))["m"].to_list()
    spiked = [*values[:150], 9999.0, *values[151:]]
    b = pl.DataFrame({"r": spiked}).select(clean.alias("m"))["m"].to_list()
    assert a[:150] == pytest.approx(b[:150], nan_ok=True)
    assert b[150] != pytest.approx(a[150]), "the spike must affect its own bar onward"


# ---------------------------------------------------------------------------
# include_current_bar
# ---------------------------------------------------------------------------
def test_excluding_the_current_bar_lags_everything_by_one():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    frame = pl.DataFrame({"r": values})
    inclusive = frame.select(
        rolling_mean(pl.col("r"), 3, include_current_bar=True).alias("m"))["m"].to_list()
    exclusive = frame.select(
        rolling_mean(pl.col("r"), 3, include_current_bar=False).alias("m"))["m"].to_list()
    assert exclusive[1:] == pytest.approx(inclusive[:-1], nan_ok=True)
    assert exclusive[0] is None


def test_excluded_current_bar_uses_only_strictly_earlier_data():
    """With include_current_bar=False the value at t knows nothing about bar t."""
    values = [1.0] * 10
    spiked = [*values[:5], 1000.0, *values[6:]]
    frame_a = pl.DataFrame({"r": values})
    frame_b = pl.DataFrame({"r": spiked})
    a = frame_a.select(
        rolling_mean(pl.col("r"), 3, include_current_bar=False).alias("m"))["m"].to_list()
    b = frame_b.select(
        rolling_mean(pl.col("r"), 3, include_current_bar=False).alias("m"))["m"].to_list()
    assert a[5] == pytest.approx(b[5]), "bar 5's own value must not enter its feature"
    assert b[6] != pytest.approx(a[6]), "it must enter from bar 6 onward"


def test_moment_statistics_honour_include_current_bar():
    rng = np.random.default_rng(21)
    values = rng.normal(0, 1, 200).tolist()
    inclusive = rolling_skewness(series_of(values), 50, include_current_bar=True)
    exclusive = rolling_skewness(series_of(values), 50, include_current_bar=False)
    np.testing.assert_allclose(exclusive[1:], inclusive[:-1], equal_nan=True)


# ---------------------------------------------------------------------------
# The assembled feature set
# ---------------------------------------------------------------------------
def test_rolling_statistics_produces_the_configured_columns(research_config):
    frame = pl.DataFrame({"ret_log": np.random.default_rng(1).normal(0, 0.001, 500)})
    out, spec = rolling_statistics(frame, column="ret_log", window=50,
                                   config=research_config)
    for suffix in ("mean", "var", "volatility", "skew", "excess_kurtosis"):
        assert f"roll50_{suffix}" in out.columns
    for lag in research_config.rolling.autocorr_lags:
        assert f"roll50_acf{lag}" in out.columns
    assert isinstance(spec, RollingSpec)
    assert spec.window == 50


def test_rolling_spec_records_the_windowing_convention(research_config):
    frame = pl.DataFrame({"ret_log": np.zeros(200)})
    _, spec = rolling_statistics(frame, column="ret_log", window=20,
                                 config=research_config)
    payload = spec.to_dict()
    assert payload["window_description"] in ("[t-N+1, t]", "[t-N, t-1]")
    assert "Backward-looking" in payload["causality"]
    assert payload["include_current_bar"] == research_config.rolling.include_current_bar


def test_rolling_frame_covers_every_configured_window(research_config):
    bars = make_bars(600, seed=5)
    frame = bars.select("timestamp").with_columns(
        pl.Series("ret_log", np.random.default_rng(2).normal(0, 0.001, 600))
    )
    out, specs = rolling_frame(frame, column="ret_log", config=research_config)
    assert [s.window for s in specs] == list(research_config.rolling.windows)
    for window in research_config.rolling.windows:
        assert f"roll{window}_volatility" in out.columns


def test_min_periods_prevents_a_value_from_a_short_window(research_config_factory):
    config = research_config_factory(rolling={"min_periods_fraction": 1.0})
    frame = pl.DataFrame({"ret_log": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out, spec = rolling_statistics(frame, column="ret_log", window=4, config=config)
    assert spec.min_periods == 4
    assert out["roll4_mean"].to_list()[:3] == [None, None, None]
