"""Independent arithmetic, supported conventions and dependence behavior."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from xauusd_quant.robustness.resampling import block_indices, bootstrap, segments
from xauusd_quant.robustness.statistics import ReturnDefinition, equity_returns, holm, performance


def definition() -> ReturnDefinition:
    return ReturnDefinition(
        86400,
        100.0,
        "fixed_initial_cash",
        "fixed lots no reset",
        "none",
        "reject",
        "funded elapsed UTC",
        "single net engine position",
        "dependent blocks",
        "none",
    )


def test_regular_equity_definition_not_irregular_trade_sharpe() -> None:
    times = [datetime(2021, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(3)]
    np.testing.assert_allclose(equity_returns(times, [100, 110, 105], definition()), [0.1, -0.05])
    with pytest.raises(ValueError, match="irregular"):
        equity_returns(
            [times[0], times[1], times[2] + timedelta(hours=1)], [100, 110, 105], definition()
        )
    with pytest.raises(ValueError, match="finite"):
        equity_returns(times, [100, np.nan, 105], definition())
    with pytest.raises(ValueError, match="exposure"):
        replace(definition(), overnight="")
    with pytest.raises(ValueError, match="annualization"):
        replace(definition(), annualization="sqrt252")
    result = performance(np.array([0.1, -0.05]), definition=definition(), duration_seconds=172800)
    assert result["sample_variance"] == pytest.approx(0.01125)
    assert result["deflated_sharpe"]["status"] == "unavailable"
    assert (
        performance(np.zeros(5), definition=definition(), duration_seconds=432000)[
            "unannualized_mean_over_sd"
        ]
        is None
    )
    with pytest.raises(ValueError):
        performance(np.array([np.nan, 1]), definition=definition(), duration_seconds=172800)
    with pytest.raises(ValueError, match="duration"):
        performance(np.zeros(5), definition=definition(), duration_seconds=10)


def test_holm_independent_hand_example_and_complete_family() -> None:
    # Sorted .01, .03, .04 -> cumulative max(.03, .06, .04) = .03, .06, .06.
    result = holm(
        ("a", "b", "c"),
        {"a": 0.03, "b": 0.01, "c": 0.04},
        validity="valid_marginal_pvalues_fixed_analysis",
    )
    assert result["adjusted"] == pytest.approx({"a": 0.06, "b": 0.03, "c": 0.06})
    assert result["rejected"] == {"a": False, "b": True, "c": False}
    with pytest.raises(ValueError, match="complete"):
        holm(("a", "b"), {"a": 0.01}, validity="valid_marginal_pvalues_fixed_analysis")
    with pytest.raises(ValueError, match="finite"):
        holm(("a",), {"a": np.nan}, validity="valid_marginal_pvalues_fixed_analysis")
    assert holm(("a",), {"a": None}, validity="unsupported")["status"] == "unavailable"


def test_block_draw_matches_independent_index_construction() -> None:
    rng = np.random.default_rng(17)
    starts = rng.integers(0, 4, 3)
    expected = np.concatenate([np.arange(i, i + 3) for i in starts])[:6]
    # ceil(6/3) = 2 starts: independently use first two of three precomputed draws.
    np.testing.assert_array_equal(block_indices(6, 3, np.random.default_rng(17)), expected)
    with pytest.raises(ValueError, match="segment"):
        block_indices(3, 4, rng)


def test_segment_boundaries_short_segments_and_overlap_are_explicit() -> None:
    origin = datetime(2021, 1, 1, tzinfo=UTC)
    times = [origin + timedelta(hours=i) for i in (0, 1, 2, 4, 5, 6)]
    assert segments(times, ["x"] * 6, 3600) == [slice(0, 3), slice(3, 6)]
    arguments = {
        "seconds": 3600,
        "horizon": 2,
        "block": 4,
        "replicates": 19,
        "seed": 1,
        "minimum_days": 1,
        "minimum_rows": 1,
        "minimum_blocks": 1,
    }
    result = bootstrap(np.arange(6.0), times, ["x"] * 6, **arguments)
    assert result["status"] == "insufficient" and result["interval"] is None
    with pytest.raises(ValueError, match="overlapping"):
        bootstrap(np.arange(6.0), times, ["x"] * 6, **{**arguments, "block": 1})


def test_dependent_uncertainty_reproducible_and_wider_than_singletons() -> None:
    rng = np.random.default_rng(2)
    x = np.zeros(2400)
    for i in range(1, len(x)):
        x[i] = 0.8 * x[i - 1] + rng.normal()
    times = [datetime(2021, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(len(x))]
    arguments = {
        "seconds": 3600,
        "horizon": 1,
        "replicates": 199,
        "seed": 5,
        "minimum_days": 5,
        "minimum_rows": 200,
        "minimum_blocks": 8,
    }
    blocked = bootstrap(x, times, ["a"] * len(x), block=12, **arguments)
    repeated = bootstrap(x, times, ["a"] * len(x), block=12, **arguments)
    assert blocked == repeated
    singleton = bootstrap(x, times, ["a"] * len(x), block=1, **arguments)
    assert blocked["bootstrap_mean_sd"] > 1.8 * singleton["bootstrap_mean_sd"]
    assert (
        bootstrap(np.zeros(len(x)), times, ["a"] * len(x), block=12, **arguments)["status"]
        == "unavailable"
    )
