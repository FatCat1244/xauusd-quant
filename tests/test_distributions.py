"""Distribution statistics, tail comparison and responsible normality testing."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from scipy import stats

from xauusd_quant.research.distributions import (
    describe_distribution,
    gaussian_tail_comparison,
    normality_tests,
    summary_to_frame,
)


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------
def test_moments_match_numpy_and_scipy():
    values = np.random.default_rng(5).normal(0.002, 0.01, 10_000)
    summary = describe_distribution(values)
    assert summary.count == 10_000
    assert summary.mean == pytest.approx(float(np.mean(values)))
    assert summary.median == pytest.approx(float(np.median(values)))
    assert summary.std == pytest.approx(float(np.std(values, ddof=1)))
    assert summary.variance == pytest.approx(float(np.var(values, ddof=1)))
    assert summary.skewness == pytest.approx(float(stats.skew(values, bias=False)))
    assert summary.excess_kurtosis == pytest.approx(
        float(stats.kurtosis(values, fisher=True, bias=False))
    )


def test_gaussian_data_has_near_zero_excess_kurtosis():
    values = np.random.default_rng(9).normal(0, 1, 200_000)
    summary = describe_distribution(values)
    assert abs(summary.excess_kurtosis) < 0.1
    assert abs(summary.skewness) < 0.05


def test_heavy_tailed_data_has_large_positive_excess_kurtosis():
    values = np.random.default_rng(9).standard_t(df=3, size=100_000)
    summary = describe_distribution(values)
    assert summary.excess_kurtosis > 2, "Student-t(3) is markedly heavy-tailed"


def test_quantiles_are_named_as_percentiles_and_ordered():
    values = np.random.default_rng(11).normal(0, 1, 50_000)
    summary = describe_distribution(values)
    assert "p50" in summary.quantiles and "p0_1" in summary.quantiles
    assert summary.quantiles["p0_1"] < summary.quantiles["p50"] < summary.quantiles["p99_9"]
    assert summary.quantiles["p50"] == pytest.approx(summary.median, abs=1e-9)


def test_nulls_and_non_finite_values_are_dropped():
    series = pl.Series("r", [0.1, None, 0.2, float("inf"), -0.1, float("nan")])
    summary = describe_distribution(series)
    assert summary.count == 3


def test_empty_input_is_an_explicit_error():
    with pytest.raises(ValueError, match="no finite observations"):
        describe_distribution(np.array([np.nan, np.inf]))


# ---------------------------------------------------------------------------
# Gaussian tail comparison
# ---------------------------------------------------------------------------
def test_gaussian_data_gives_tail_ratios_near_one():
    values = np.random.default_rng(3).normal(0, 1, 500_000)
    tails = gaussian_tail_comparison(values, sigma_levels=(2, 3))
    for tail in tails:
        assert tail["ratio_observed_to_gaussian"] == pytest.approx(1.0, abs=0.25)


def test_heavy_tails_produce_ratios_far_above_one():
    values = np.random.default_rng(3).standard_t(df=3, size=200_000)
    tails = gaussian_tail_comparison(values, sigma_levels=(3, 5))
    ratios = {t["sigma"]: t["ratio_observed_to_gaussian"] for t in tails}
    assert ratios[3.0] > 2
    assert ratios[5.0] > ratios[3.0], "the excess should grow further out"


def test_tail_comparison_reports_counts_and_expected_counts():
    values = np.random.default_rng(4).normal(0, 1, 10_000)
    tail = gaussian_tail_comparison(values, sigma_levels=(2,))[0]
    assert tail["observed_count"] >= 0
    assert tail["gaussian_expected_count"] == pytest.approx(
        10_000 * 2 * stats.norm.sf(2)
    )
    assert 0 <= tail["observed_rate"] <= 1


def test_tail_comparison_declines_on_degenerate_input():
    assert gaussian_tail_comparison(np.ones(100)) == []
    assert gaussian_tail_comparison(np.array([1.0])) == []


# ---------------------------------------------------------------------------
# Normality testing, and its caveats
# ---------------------------------------------------------------------------
def test_normality_tests_do_not_reject_for_modest_gaussian_samples():
    values = np.random.default_rng(7).normal(0, 1, 5_000)
    result = normality_tests(values, large_sample_threshold=1_000_000)
    assert all(not t["reject_at_5pct"] for t in result["tests"])


def test_normality_tests_reject_for_clearly_non_gaussian_data():
    values = np.random.default_rng(7).standard_t(df=2, size=5_000)
    result = normality_tests(values, large_sample_threshold=1_000_000)
    assert all(t["reject_at_5pct"] for t in result["tests"])


def test_large_samples_carry_an_explicit_warning():
    """The core caution: huge n makes these tests reject almost anything."""
    values = np.random.default_rng(8).normal(0, 1, 150_000)
    result = normality_tests(values, large_sample_threshold=100_000)
    assert result["large_sample_warning"] is not None
    assert "effect sizes" in result["large_sample_warning"]
    for test in result["tests"]:
        assert test["warning"] is not None


def test_small_samples_carry_no_such_warning():
    values = np.random.default_rng(8).normal(0, 1, 1_000)
    result = normality_tests(values, large_sample_threshold=100_000)
    assert result["large_sample_warning"] is None


def test_interpretation_is_always_present():
    result = normality_tests(np.random.default_rng(1).normal(0, 1, 100))
    assert "not exactly Gaussian" in result["interpretation"]
    assert "tradability" in result["interpretation"]


def test_individual_tests_can_be_disabled():
    values = np.random.default_rng(2).normal(0, 1, 1_000)
    only_jb = normality_tests(values, jarque_bera=True, dagostino_k2=False)
    assert [t["test"] for t in only_jb["tests"]] == ["jarque_bera"]
    only_k2 = normality_tests(values, jarque_bera=False, dagostino_k2=True)
    assert [t["test"] for t in only_k2["tests"]] == ["dagostino_k2"]


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------
def test_summary_flattens_into_a_single_row_frame():
    values = np.random.default_rng(6).normal(0, 0.01, 20_000)
    summary = describe_distribution(values)
    summary.tails = gaussian_tail_comparison(values, sigma_levels=(3, 5))
    frame = summary_to_frame(summary)
    assert frame.height == 1
    for column in ("count", "mean", "std", "skewness", "excess_kurtosis", "p50"):
        assert column in frame.columns
    assert "tail_3sigma_ratio_vs_gaussian" in frame.columns


def test_summary_notes_explain_the_kurtosis_convention():
    summary = describe_distribution(np.random.default_rng(1).normal(0, 1, 1000))
    assert any("relative to the Gaussian" in note for note in summary.notes)
