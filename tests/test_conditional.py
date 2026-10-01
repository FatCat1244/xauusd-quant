"""Conditional and reversal analysis, and the feature/outcome boundary.

The most important tests here are the leakage ones: a forward return must
never be able to influence a condition, a threshold or a rolling statistic.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from conftest import make_bars
from xauusd_quant.research.conditional import (
    bootstrap_mean_ci,
    conditional_return_analysis,
    reversal_summary,
)
from xauusd_quant.research.returns import assert_no_forward_columns, prepare_returns


@pytest.fixture
def prepared(research_config):
    bars = make_bars(4000, seed=42)
    return prepare_returns(bars, "5m", research_config)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------
def test_analysis_covers_every_bucket_and_horizon(prepared, research_config):
    analysis = conditional_return_analysis(
        prepared.frame, timeframe="5m", return_column=prepared.return_column,
        config=research_config, price_column=prepared.price_column,
    )
    cond = research_config.conditional
    expected = (
        (len(cond.lower_quantiles) + len(cond.upper_quantiles))
        * len(cond.forward_horizons)
    )
    assert len(analysis.buckets) == expected
    assert set(analysis.thresholds) == {
        *[f"lower_q{q:g}" for q in cond.lower_quantiles],
        *[f"upper_q{q:g}" for q in cond.upper_quantiles],
    }


def test_lower_thresholds_are_negative_and_ordered(prepared, research_config):
    analysis = conditional_return_analysis(
        prepared.frame, timeframe="5m", return_column=prepared.return_column,
        config=research_config, price_column=prepared.price_column,
    )
    lower = [analysis.thresholds[f"lower_q{q:g}"]
             for q in sorted(research_config.conditional.lower_quantiles)]
    assert lower == sorted(lower), "a lower quantile must be a smaller return"
    upper = [analysis.thresholds[f"upper_q{q:g}"]
             for q in sorted(research_config.conditional.upper_quantiles)]
    assert upper == sorted(upper)


def test_reversal_and_continuation_probabilities_sum_to_one(prepared, research_config):
    analysis = conditional_return_analysis(
        prepared.frame, timeframe="5m", return_column=prepared.return_column,
        config=research_config, price_column=prepared.price_column,
    )
    for bucket in analysis.buckets:
        assert bucket.prob_reversal + bucket.prob_continuation == pytest.approx(1.0)


def test_thin_samples_are_flagged_not_hidden(prepared, research_config):
    analysis = conditional_return_analysis(
        prepared.frame, timeframe="5m", return_column=prepared.return_column,
        config=research_config, price_column=prepared.price_column,
    )
    # The 0.1% bucket on 4,000 bars holds only a handful of observations.
    tiny = [b for b in analysis.buckets if b.quantile == 0.001]
    assert tiny and all(b.thin_sample for b in tiny)
    assert any("thin_sample" in w for w in analysis.warnings)


def test_analysis_states_its_own_limitations(prepared, research_config):
    analysis = conditional_return_analysis(
        prepared.frame, timeframe="5m", return_column=prepared.return_column,
        config=research_config, price_column=prepared.price_column,
    )
    joined = " ".join(analysis.notes)
    assert "OUTCOMES only" in joined
    assert "overlap" in joined, "overlapping windows caveat must be present"
    assert "transaction costs" in joined or "no costs" in joined.lower()


# ---------------------------------------------------------------------------
# Correctness of the conditional statistics
# ---------------------------------------------------------------------------
def test_forward_return_statistics_match_a_manual_computation(research_config_factory):
    """Recompute one cell by hand and compare."""
    config = research_config_factory(
        conditional={"lower_quantiles": [0.10], "upper_quantiles": [0.90],
                     "forward_horizons": [1], "min_samples_warning": 1,
                     "bootstrap": {"enabled": False}}
    )
    bars = make_bars(1000, seed=3)
    series = prepare_returns(bars, "5m", config)
    frame = series.frame

    analysis = conditional_return_analysis(
        frame, timeframe="5m", return_column=series.return_column,
        config=config, price_column=series.price_column,
    )
    bucket = next(b for b in analysis.buckets
                  if b.direction == "lower" and b.horizon == 1)

    # Manual: select bars at or below the 10% quantile, take next-bar log return.
    returns = frame[series.return_column]
    threshold = float(returns.quantile(0.10))
    prices = frame[series.price_column].to_numpy()
    mask = frame[series.return_column].to_numpy()
    selected = np.where(np.nan_to_num(mask, nan=np.inf) <= threshold)[0]
    selected = selected[selected < len(prices) - 1]
    manual = np.log(prices[selected + 1] / prices[selected])

    assert bucket.observations == len(manual)
    assert bucket.mean_forward_return == pytest.approx(float(np.mean(manual)), rel=1e-9)
    assert bucket.prob_positive == pytest.approx(float(np.mean(manual > 0)))


def test_reversal_is_defined_relative_to_the_move_direction(research_config_factory):
    """After a fall a reversal is a rise; after a rise it is a fall."""
    config = research_config_factory(
        conditional={"lower_quantiles": [0.10], "upper_quantiles": [0.90],
                     "forward_horizons": [1], "min_samples_warning": 1,
                     "bootstrap": {"enabled": False}}
    )
    series = prepare_returns(make_bars(2000, seed=8), "5m", config)
    analysis = conditional_return_analysis(
        series.frame, timeframe="5m", return_column=series.return_column,
        config=config, price_column=series.price_column,
    )
    lower = next(b for b in analysis.buckets if b.direction == "lower")
    upper = next(b for b in analysis.buckets if b.direction == "upper")
    assert lower.prob_reversal == pytest.approx(lower.prob_positive)
    assert upper.prob_reversal == pytest.approx(1.0 - upper.prob_positive, abs=1e-12)


def test_a_constructed_reversal_is_detected(research_config_factory):
    """Build a series that always bounces, and check the statistics see it."""
    config = research_config_factory(
        conditional={"lower_quantiles": [0.20], "upper_quantiles": [0.80],
                     "forward_horizons": [1], "min_samples_warning": 1,
                     "bootstrap": {"enabled": False}},
        returns={"drop_session_gap_returns": False},
    )
    # Strict alternation: every down move is followed by an up move.
    prices, value = [2000.0], 2000.0
    for i in range(1, 600):
        value *= 1.002 if i % 2 else 0.998
        prices.append(value)
    start = datetime(2024, 1, 2, 1, 0)
    bars = pl.DataFrame({
        "timestamp": [start + timedelta(minutes=5 * i) for i in range(len(prices))],
        "close": prices, "open": prices,
        "high": [p * 1.0001 for p in prices], "low": [p * 0.9999 for p in prices],
        "tick_count": [100] * len(prices), "volume": [1000.0] * len(prices),
        "mean_spread": [0.3] * len(prices),
        "first_bid": prices, "first_ask": prices,
        "last_bid": prices, "last_ask": prices,
    })
    series = prepare_returns(bars, "5m", config)
    analysis = conditional_return_analysis(
        series.frame, timeframe="5m", return_column=series.return_column,
        config=config, price_column=series.price_column,
    )
    lower = next(b for b in analysis.buckets if b.direction == "lower")
    assert lower.prob_reversal == pytest.approx(1.0), "every down move bounces"
    assert lower.mean_forward_return > 0


# ---------------------------------------------------------------------------
# No leakage
# ---------------------------------------------------------------------------
def test_the_input_frame_carries_no_forward_columns(prepared):
    assert_no_forward_columns(prepared.frame, context="conditional input")


def test_conditions_are_unaffected_by_data_after_the_bar(research_config_factory):
    """A bar's bucket must not change when later bars are appended.

    Thresholds are whole-sample by design, so this test pins the *condition
    value* itself: the return at t is what it was, regardless of the future.
    """
    config = research_config_factory(
        conditional={"lower_quantiles": [0.10], "upper_quantiles": [0.90],
                     "forward_horizons": [1], "bootstrap": {"enabled": False}}
    )
    bars = make_bars(1000, seed=15)
    short = prepare_returns(bars.head(600), "5m", config)
    long = prepare_returns(bars, "5m", config)
    a = short.frame[short.return_column].to_list()
    b = long.frame[long.return_column].to_list()[:600]
    assert a == pytest.approx(b, nan_ok=True)


def test_forward_returns_never_reach_the_output_tables(prepared, research_config):
    analysis = conditional_return_analysis(
        prepared.frame, timeframe="5m", return_column=prepared.return_column,
        config=research_config, price_column=prepared.price_column,
    )
    table = reversal_summary(analysis)
    leaked = [c for c in table.columns if c.startswith("fwd_")]
    assert not leaked, f"raw forward columns leaked into the summary: {leaked}"


def test_last_bars_have_no_forward_outcome(research_config_factory):
    """A forward return cannot exist past the end of the data."""
    config = research_config_factory(
        conditional={"lower_quantiles": [0.5], "upper_quantiles": [0.5],
                     "forward_horizons": [20], "min_samples_warning": 1,
                     "bootstrap": {"enabled": False}}
    )
    series = prepare_returns(make_bars(500, seed=2), "5m", config)
    analysis = conditional_return_analysis(
        series.frame, timeframe="5m", return_column=series.return_column,
        config=config, price_column=series.price_column,
    )
    bucket = next(b for b in analysis.buckets if b.horizon == 20)
    # Roughly half the bars are selected, but the final 20 can have no outcome.
    assert bucket.observations < series.frame.height


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def test_bootstrap_interval_brackets_the_sample_mean():
    rng = np.random.default_rng(1)
    values = rng.normal(0.001, 0.01, 5000)
    low, high = bootstrap_mean_ci(values, iterations=500, seed=1)
    assert low is not None and high is not None
    assert low < float(np.mean(values)) < high


def test_bootstrap_is_deterministic_for_a_given_seed():
    values = np.random.default_rng(2).normal(0, 1, 1000)
    a = bootstrap_mean_ci(values, iterations=200, seed=7)
    b = bootstrap_mean_ci(values, iterations=200, seed=7)
    assert a == b


def test_bootstrap_declines_rather_than_inventing_an_interval():
    assert bootstrap_mean_ci(np.array([1.0, 2.0, 3.0])) == (None, None)
    big = np.zeros(100)
    assert bootstrap_mean_ci(big, max_samples=10) == (None, None)


def test_reversal_summary_marks_intervals_that_exclude_zero(prepared, research_config):
    analysis = conditional_return_analysis(
        prepared.frame, timeframe="5m", return_column=prepared.return_column,
        config=research_config, price_column=prepared.price_column,
    )
    table = reversal_summary(analysis)
    assert "ci_excludes_zero" in table.columns
    for row in table.iter_rows(named=True):
        if row["mean_ci_lower"] is None:
            assert row["ci_excludes_zero"] is None
        else:
            expected = row["mean_ci_lower"] > 0 or row["mean_ci_upper"] < 0
            assert row["ci_excludes_zero"] == expected


def test_too_little_data_is_reported_not_analysed(research_config):
    tiny = prepare_returns(make_bars(50, seed=1), "5m", research_config)
    analysis = conditional_return_analysis(
        tiny.frame, timeframe="5m", return_column=tiny.return_column,
        config=research_config, price_column=tiny.price_column,
    )
    assert analysis.buckets == []
    assert any("skipped" in w for w in analysis.warnings)
