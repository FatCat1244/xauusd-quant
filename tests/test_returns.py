"""Return mathematics, and the feature/outcome boundary."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import polars as pl
import pytest

from xauusd_quant.research.returns import (
    assert_no_forward_columns,
    cumulative_returns,
    forward_returns,
    log_returns,
    mark_session_gaps,
    multi_period_returns,
    prepare_returns,
    resolve_price_column,
    simple_returns,
)


def frame_of(prices: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"close": prices})


# ---------------------------------------------------------------------------
# The worked example from the specification: 100 -> 101 -> 99
# ---------------------------------------------------------------------------
def test_simple_returns_of_the_worked_example():
    out = frame_of([100.0, 101.0, 99.0]).select(simple_returns(pl.col("close")).alias("r"))
    values = out["r"].to_list()
    assert values[0] is None
    assert values[1] == pytest.approx(0.01)
    assert values[2] == pytest.approx(-2.0 / 101.0)


def test_log_returns_of_the_worked_example():
    out = frame_of([100.0, 101.0, 99.0]).select(log_returns(pl.col("close")).alias("r"))
    values = out["r"].to_list()
    assert values[0] is None
    assert values[1] == pytest.approx(math.log(101 / 100))
    assert values[2] == pytest.approx(math.log(99 / 101))


def test_cumulative_return_recovers_the_total_move():
    """100 -> 99 is -1% overall, whichever return type gets there."""
    frame = frame_of([100.0, 101.0, 99.0])
    for return_type in ("log", "simple"):
        fn = log_returns if return_type == "log" else simple_returns
        total = frame.select(
            cumulative_returns(fn(pl.col("close")), return_type=return_type).alias("c")
        )["c"].to_list()[-1]
        assert total == pytest.approx(-0.01), return_type


def test_log_and_simple_returns_agree_for_small_moves():
    frame = frame_of([100.0, 100.05, 100.10, 100.02])
    simple = frame.select(simple_returns(pl.col("close")).alias("r"))["r"].to_list()[1:]
    logs = frame.select(log_returns(pl.col("close")).alias("r"))["r"].to_list()[1:]
    for s, lg in zip(simple, logs, strict=True):
        assert lg == pytest.approx(s, abs=1e-5)


def test_log_returns_are_additive_across_periods():
    """The property that makes log returns the analysis default."""
    frame = frame_of([100.0, 103.0, 99.0, 105.0])
    one = frame.select(log_returns(pl.col("close")).alias("r"))["r"].to_list()
    three = frame.select(log_returns(pl.col("close"), periods=3).alias("r"))["r"].to_list()
    assert three[3] == pytest.approx(one[1] + one[2] + one[3])


@pytest.mark.parametrize("periods", [0, -1])
def test_non_positive_periods_are_rejected(periods):
    for fn in (simple_returns, log_returns):
        with pytest.raises(ValueError, match="periods must be >= 1"):
            fn(pl.col("close"), periods=periods)


# ---------------------------------------------------------------------------
# Backward multi-period returns
# ---------------------------------------------------------------------------
def test_multi_period_returns_look_backward_only():
    prices = [100.0 + i for i in range(10)]
    out = frame_of(prices).with_columns(multi_period_returns(pl.col("close"), [2, 5]))
    for horizon in (2, 5):
        column = out[f"ret_log_{horizon}"].to_list()
        assert column[:horizon] == [None] * horizon, "leading values must be null"
        assert column[horizon] == pytest.approx(
            math.log(prices[horizon] / prices[0])
        ), "value at t must compare against t-n, not t+n"


def test_backward_return_at_t_is_unchanged_by_later_data():
    """The defining property of a causal feature."""
    base = [100.0, 101.0, 102.0, 103.0]
    short = frame_of(base).with_columns(multi_period_returns(pl.col("close"), [2]))
    long = frame_of([*base, 500.0, 0.5]).with_columns(
        multi_period_returns(pl.col("close"), [2])
    )
    assert short["ret_log_2"].to_list() == pytest.approx(
        long["ret_log_2"].to_list()[: len(base)], nan_ok=True
    )


# ---------------------------------------------------------------------------
# Forward returns are outcomes, and must look forward
# ---------------------------------------------------------------------------
def test_forward_returns_look_forward_and_are_named_as_outcomes():
    prices = [100.0, 110.0, 121.0, 133.1]
    out = frame_of(prices).with_columns(forward_returns(pl.col("close"), [1, 2]))
    assert out.columns[-2:] == ["fwd_log_1", "fwd_log_2"]
    assert out["fwd_log_1"].to_list()[0] == pytest.approx(math.log(110 / 100))
    assert out["fwd_log_2"].to_list()[0] == pytest.approx(math.log(121 / 100))
    # The tail cannot look past the end of the data.
    assert out["fwd_log_1"].to_list()[-1] is None
    assert out["fwd_log_2"].to_list()[-2:] == [None, None]


def test_forward_and_backward_returns_are_mirror_images():
    prices = [100.0, 104.0, 99.0, 107.0]
    frame = frame_of(prices).with_columns(
        log_returns(pl.col("close")).alias("back"),
        *forward_returns(pl.col("close"), [1]),
    )
    back = frame["back"].to_list()
    fwd = frame["fwd_log_1"].to_list()
    # The forward return at t is the backward return at t+1.
    assert fwd[0] == pytest.approx(back[1])
    assert fwd[1] == pytest.approx(back[2])


def test_assert_no_forward_columns_catches_a_leak():
    clean = frame_of([1.0, 2.0]).with_columns(log_returns(pl.col("close")).alias("ret"))
    assert_no_forward_columns(clean)  # must not raise

    leaked = clean.with_columns(forward_returns(pl.col("close"), [1]))
    with pytest.raises(ValueError, match="forward-looking column"):
        assert_no_forward_columns(leaked, context="feature set")


@pytest.mark.parametrize("horizon", [0, -3])
def test_forward_horizons_must_be_positive(horizon):
    with pytest.raises(ValueError, match="forward horizon must be >= 1"):
        forward_returns(pl.col("close"), [horizon])


# ---------------------------------------------------------------------------
# Session gaps
# ---------------------------------------------------------------------------
def test_session_gaps_are_detected_from_the_bar_spacing():
    base = datetime(2024, 5, 1, 10, 0)
    stamps = [base, base + timedelta(minutes=5), base + timedelta(minutes=10),
              base + timedelta(hours=49)]          # weekend
    frame = pl.DataFrame({"timestamp": stamps})
    flags = frame.select(
        mark_session_gaps(pl.col("timestamp"), "5m").alias("gap")
    )["gap"].to_list()
    assert flags == [True, False, False, True], "first bar and the weekend are gaps"


def test_price_column_resolution_maps_sources_to_bar_columns():
    assert resolve_price_column("mid", "close") == "close"
    assert resolve_price_column("bid", "close") == "last_bid"
    assert resolve_price_column("ask", "close") == "last_ask"
    assert resolve_price_column("ask", "open") == "first_ask"
    assert resolve_price_column("bid", "open") == "first_bid"
    with pytest.raises(KeyError, match="Unknown price_source"):
        resolve_price_column("midpoint", "close")


# ---------------------------------------------------------------------------
# prepare_returns
# ---------------------------------------------------------------------------
def synthetic_bars(n: int = 300, start: datetime | None = None) -> pl.DataFrame:
    start = start or datetime(2024, 5, 1, 1, 0)
    prices = [2000.0]
    for i in range(1, n):
        prices.append(prices[-1] * (1 + 0.0001 * ((i * 37) % 11 - 5)))
    return pl.DataFrame({
        "timestamp": [start + timedelta(minutes=5 * i) for i in range(n)],
        "open": prices,
        "high": [p * 1.0002 for p in prices],
        "low": [p * 0.9998 for p in prices],
        "close": prices,
        "tick_count": [100 + (i % 50) for i in range(n)],
        "volume": [1000.0 + i for i in range(n)],
        "mean_spread": [0.3 + 0.01 * (i % 7) for i in range(n)],
        "first_bid": [p - 0.15 for p in prices],
        "first_ask": [p + 0.15 for p in prices],
        "last_bid": [p - 0.15 for p in prices],
        "last_ask": [p + 0.15 for p in prices],
    })


def test_prepare_returns_builds_the_expected_columns(research_config):
    series = prepare_returns(synthetic_bars(), "5m", research_config)
    for column in ("ret_log", "ret_simple", "abs_return", "squared_return",
                   "is_session_gap"):
        assert column in series.frame.columns
    assert series.return_column == "ret_log"
    assert series.price_column == "close"
    assert series.rows_with_return > 0


def test_prepare_returns_produces_no_forward_columns(research_config):
    """The feature frame must be free of outcomes by construction."""
    series = prepare_returns(synthetic_bars(), "5m", research_config)
    assert_no_forward_columns(series.frame, context="prepare_returns output")


def test_prepare_returns_nulls_the_gap_return(research_config):
    bars = synthetic_bars(100)
    # Push the second half a week forward to create a weekend gap.
    bars = bars.with_columns(
        pl.when(pl.int_range(pl.len()) >= 50)
        .then(pl.col("timestamp") + pl.duration(days=3))
        .otherwise(pl.col("timestamp"))
        .alias("timestamp")
    )
    series = prepare_returns(bars, "5m", research_config)
    assert series.session_gap_returns_dropped >= 1
    gap_rows = series.frame.filter(pl.col("is_session_gap"))
    assert gap_rows["ret_log"].null_count() == gap_rows.height


def test_prepare_returns_rejects_an_empty_frame(research_config):
    empty = synthetic_bars(2).clear()
    with pytest.raises(ValueError, match="No bars supplied"):
        prepare_returns(empty, "5m", research_config)


def test_prepare_returns_reports_a_missing_price_column(research_config):
    bars = synthetic_bars().drop("close")
    with pytest.raises(KeyError, match="close"):
        prepare_returns(bars, "5m", research_config)
