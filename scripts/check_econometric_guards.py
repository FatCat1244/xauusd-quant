"""Break econometric leakage guards in isolated copies; retain the Stage 12/13 guard audit."""

from check_strategy_guards import MUTATIONS, main

MUTATIONS.extend(
    [
        (
            "econometric_label_publication_cutoff",
            "econometrics/regression.py",
            "or s.outcome.matured_utc >= cutoff",
            "or (False and s.outcome.matured_utc >= cutoff)",
            "tests/test_econometric_models.py::test_fit_rejects_crossing_labels_and_mixed_horizons",
        ),
        (
            "econometric_materialized_future_fit",
            "econometrics/runs.py",
            "if any(t >= cutoff for t in data.closes):",
            "if False and any(t >= cutoff for t in data.closes):",
            "tests/test_econometric_framework.py::test_model_fit_refuses_a_materialized_future_bar_even_with_purged_labels",
        ),
        (
            "econometric_no_cross_gap_outcomes",
            "econometrics/data.py",
            "if matured >= cutoff or not self.contiguous[i + 1 : j + 1].all():",
            "if matured >= cutoff:",
            "tests/test_econometric_models.py::test_targets_use_contiguous_grid_and_actual_information_intervals",
        ),
        (
            "econometric_current_observation_only",
            "econometrics/runs.py",
            "float(data.logs[index])",
            "float(data.logs[min(index + 1, len(data.logs) - 1)])",
            "tests/test_econometric_state.py::test_replay_observes_current_bar_only_even_when_future_bars_are_in_memory",
        ),
        (
            "econometric_forecast_feature_availability",
            "econometrics/runs.py",
            'if point.available_utc > at or at < self.spec["fitting_cutoff_utc"]:',
            'if False and (point.available_utc > at or at < self.spec["fitting_cutoff_utc"]):',
            "tests/test_econometric_state.py::test_replay_observes_current_bar_only_even_when_future_bars_are_in_memory",
        ),
        (
            "econometric_residual_maturity",
            "econometrics/uncertainty.py",
            "if outcome.matured_utc >= now:",
            "if False and outcome.matured_utc >= now:",
            "tests/test_econometric_uncertainty.py::test_calibration_waits_for_publication_and_overlap_does_not_make_labels_available",
        ),
        (
            "econometric_calibration_decision_clock",
            "econometrics/uncertainty.py",
            "if self.last_update_utc is not None and issued < self.last_update_utc:",
            "if False and self.last_update_utc is not None and issued < self.last_update_utc:",
            "tests/test_econometric_uncertainty.py::test_delayed_delivery_cannot_be_used_for_a_backdated_decision",
        ),
        (
            "econometric_monitor_maturity",
            "econometrics/monitor.py",
            "if matured >= now:",
            "if False and matured >= now:",
            "tests/test_econometric_uncertainty.py::test_cusum_waits_for_matured_errors_and_matches_reload",
        ),
        (
            "econometric_unknown_provenance",
            "econometrics/comparison.py",
            'gates.get(k) == "passed"',
            'True or gates.get(k) == "passed"',
            "tests/test_econometric_framework.py::test_missing_provenance_cannot_promote_favorable_effects",
        ),
        (
            "econometric_real_fixture_classification",
            "econometrics/runs.py",
            'if synthetic is None and plan.evaluation_history_classification == "software_correctness":',
            'if False and synthetic is None and plan.evaluation_history_classification == "software_correctness":',
            "tests/test_econometric_framework.py::test_final_period_and_real_readiness_cannot_be_overridden_by_fixtures",
        ),
    ]
)

if __name__ == "__main__":
    raise SystemExit(main())
