"""Fixed synthetic sensitivity checks retain failures and do not select a neighbor."""

import numpy as np

from xauusd_quant.robustness.econometric_controls import (
    interval_sensitivity,
    monitor_sensitivity,
    state_sensitivity,
)


def test_kalman_whole_neighborhood_has_prior_fit_and_positive_covariance() -> None:
    result = state_sensitivity(140014)
    assert result["prior_fit_rows"] == 300
    assert [r["q_multiplier"] for r in result["results"]] == [0.5, 1.0, 2.0]
    assert all(r["final_covariance"] > 0 and np.isfinite(r["mse"]) for r in result["results"])
    assert result == state_sensitivity(140014)


def test_interval_whole_neighborhood_retains_coverage_and_width() -> None:
    result = interval_sensitivity(140014)
    assert len(result) == 4
    assert all(
        r["evaluated"] == 768 and 0 <= r["coverage"] <= 1 and r["mean_width"] > 0 for r in result
    )
    assert {(r["window"], r["gamma"]) for r in result} == {
        (64, 0.0),
        (128, 0.0),
        (64, 0.01),
        (128, 0.01),
    }


def test_monitor_false_alarms_and_shift_delays_have_monte_carlo_uncertainty() -> None:
    results = monitor_sensitivity(140014, 12)
    assert [r["threshold"] for r in results] == [6.0, 8.0, 10.0]
    assert all(len(r["null_alarms_per_600"]) == 12 for r in results)
    assert all(r["maximum_binomial_se"] > 0.1 for r in results)
    # Controlled large shift must be detected promptly; null false positives are allowed.
    assert all(1 <= d <= 10 for r in results for d in r["shift_detection_delay_observations"])
