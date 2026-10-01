"""Stationarity tests against series with known behaviour.

Exact p-values are not asserted - they depend on lag selection and sample size.
What is asserted is the qualitative behaviour every downstream reading depends
on: a random walk must look non-stationary, stationary noise must look
stationary, and the two tests' opposite nulls must be handled correctly.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from xauusd_quant.research.stationarity import (
    adf_test,
    kpss_test,
    segmented_stationarity,
    stationarity_report,
    subsample,
)


def random_walk(n: int = 3000, *, seed: int = 13, drift: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(drift, 1.0, n)) + 100.0


def stationary_noise(n: int = 3000, *, seed: int = 13) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 1.0, n)


def ar1(n: int, phi: float, *, seed: int = 13) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    out[0] = rng.normal()
    for i in range(1, n):
        out[i] = phi * out[i - 1] + rng.normal()
    return out


# ---------------------------------------------------------------------------
# ADF: H0 is a unit root
# ---------------------------------------------------------------------------
def test_adf_rejects_the_unit_root_for_stationary_noise():
    result = adf_test(stationary_noise())
    assert result.reject_at_5pct is True
    assert result.p_value < 0.01
    assert result.null_hypothesis.startswith("unit root")


def test_adf_does_not_reject_for_a_random_walk():
    result = adf_test(random_walk())
    assert result.reject_at_5pct is False
    assert result.p_value > 0.05


def test_adf_reports_critical_values_and_lag_selection():
    result = adf_test(stationary_noise())
    assert set(result.critical_values) >= {"1%", "5%", "10%"}
    assert result.lags_used is not None and result.lags_used >= 0
    assert result.observations > 0


def test_adf_on_a_differenced_random_walk_looks_stationary():
    """Differencing a unit-root series should remove the unit root."""
    walk = random_walk(4000)
    assert adf_test(walk).reject_at_5pct is False
    assert adf_test(np.diff(walk)).reject_at_5pct is True


# ---------------------------------------------------------------------------
# KPSS: H0 is stationarity - the opposite direction
# ---------------------------------------------------------------------------
def test_kpss_does_not_reject_stationarity_for_noise():
    result = kpss_test(stationary_noise())
    assert result.reject_at_5pct is False
    assert result.null_hypothesis == "series is stationary"


def test_kpss_rejects_stationarity_for_a_random_walk():
    result = kpss_test(random_walk())
    assert result.reject_at_5pct is True


def test_kpss_flags_a_bounded_p_value():
    """statsmodels clamps p to its table; the result must say so."""
    result = kpss_test(random_walk(3000))
    assert result.p_value_is_bounded is True
    assert result.note and "bound" in result.note


# ---------------------------------------------------------------------------
# The two tests combined
# ---------------------------------------------------------------------------
def test_report_calls_stationary_noise_stationary(research_config):
    report = stationarity_report(
        stationary_noise(), series_name="noise", config=research_config
    )
    assert report.verdict == "stationary"
    assert "both tests point the same way" in report.interpretation
    assert report.adf is not None and report.kpss is not None


def test_report_calls_a_random_walk_non_stationary(research_config):
    report = stationarity_report(
        random_walk(), series_name="walk", config=research_config
    )
    assert report.verdict == "non-stationary (unit root)"


def test_report_always_explains_the_opposing_nulls(research_config):
    report = stationarity_report(
        stationary_noise(), series_name="noise", config=research_config
    )
    assert any("opposites" in note for note in report.notes)


def test_report_is_not_reduced_to_a_single_boolean(research_config):
    """Both tests must survive into the output, not be collapsed."""
    report = stationarity_report(
        random_walk(), series_name="walk", config=research_config
    )
    payload = report.to_dict()
    assert payload["adf"]["null_hypothesis"] != payload["kpss"]["null_hypothesis"]
    assert payload["verdict"] and payload["interpretation"]


def test_report_declines_on_too_few_observations(research_config):
    report = stationarity_report(
        np.array([1.0, 2.0, 3.0]), series_name="tiny", config=research_config
    )
    assert report.verdict == "not evaluated"
    assert "too few" in report.interpretation


def test_only_adf_enabled_is_labelled_as_such(research_config_factory):
    config = research_config_factory(
        stationarity={"adf_enabled": True, "kpss_enabled": False}
    )
    report = stationarity_report(
        stationary_noise(), series_name="noise", config=config
    )
    assert "ADF only" in report.verdict
    assert report.kpss is None


# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------
def test_subsample_preserves_order_and_is_deterministic():
    values = np.arange(1000, dtype=float)
    a, was_a = subsample(values, 100)
    b, was_b = subsample(values, 100)
    assert was_a and was_b
    assert np.array_equal(a, b), "subsampling must be deterministic"
    assert np.all(np.diff(a) > 0), "ordering must be preserved"
    assert a.size <= 100


def test_subsample_is_a_no_op_below_the_cap():
    values = np.arange(50, dtype=float)
    out, was = subsample(values, 100)
    assert was is False
    assert np.array_equal(out, values)


def test_subsampling_is_recorded_in_the_result():
    result = adf_test(stationary_noise(5000), max_observations=1000)
    assert result.note and "subsample" in result.note.lower()


# ---------------------------------------------------------------------------
# Segmented
# ---------------------------------------------------------------------------
def test_segmented_stationarity_tests_each_year(research_config):
    from datetime import datetime, timedelta

    n = 20_000          # ~833 days, so the series genuinely spans 2022-2024
    start = datetime(2022, 1, 1)
    frame = pl.DataFrame({
        "timestamp": [start + timedelta(hours=i) for i in range(n)],
        "r": stationary_noise(n),
    })
    result = segmented_stationarity(
        frame, series_column="r", time_column="timestamp",
        series_name="r", config=research_config,
    )
    segments = {s["segment"] for s in result.segments}
    assert {"2022", "2023"} <= segments
    tested = [s for s in result.segments if s.get("adf")]
    assert len(tested) >= 2, "at least two years should have been tested"
    table = result.to_frame()
    assert "adf_p_value" in table.columns and "kpss_p_value" in table.columns


def test_segments_below_the_minimum_are_skipped_not_guessed(research_config_factory):
    from datetime import datetime, timedelta

    config = research_config_factory(
        stationarity={"segmented": {"enabled": True, "by": "year",
                                    "min_observations": 100_000}}
    )
    frame = pl.DataFrame({
        "timestamp": [datetime(2022, 1, 1) + timedelta(hours=i) for i in range(500)],
        "r": stationary_noise(500),
    })
    result = segmented_stationarity(
        frame, series_column="r", time_column="timestamp", series_name="r", config=config
    )
    assert all("skipped" in s["verdict"] for s in result.segments)
    assert all(s["adf"] is None for s in result.segments)


def test_segmented_can_be_disabled(research_config_factory):
    from datetime import datetime, timedelta

    config = research_config_factory(
        stationarity={"segmented": {"enabled": False}}
    )
    frame = pl.DataFrame({
        "timestamp": [datetime(2022, 1, 1) + timedelta(hours=i) for i in range(100)],
        "r": stationary_noise(100),
    })
    result = segmented_stationarity(
        frame, series_column="r", time_column="timestamp", series_name="r", config=config
    )
    assert result.segments == []
    assert any("disabled" in n for n in result.notes)
