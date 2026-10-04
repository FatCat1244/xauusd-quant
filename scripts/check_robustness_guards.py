"""Break Stage14 and retained Stage12/13.5 leakage guards in isolated source copies."""

from check_econometric_guards import MUTATIONS, main

MUTATIONS.extend(
    [
        (
            "robustness_saved_artifact_hash",
            "robustness/reports.py",
            "or sha256(file) != digest",
            "or False",
            "tests/test_robustness_source_identity.py::test_changed_saved_source_refused_before_outcome_materialization",
        ),
        (
            "robustness_saved_reserved_source",
            "robustness/reports.py",
            'validate_interval(\n            config, parse_utc(fold["training_start_utc"]), parse_utc(fold["evaluation_end_utc"])\n        )',
            "None",
            "tests/test_robustness_framework.py::test_reserved_source_refused_before_opening_outcomes",
        ),
        (
            "robustness_saved_label_cutoff",
            "robustness/reports.py",
            'or parse_utc(spec["training_label_as_of_utc"]) >= cutoff',
            "or False",
            "tests/test_robustness_framework.py::test_saved_forecast_maturity_and_fit_cutoff_are_audited",
        ),
        (
            "robustness_matured_saved_outcomes",
            "robustness/reports.py",
            "if not start <= available < target_end <= matured < end:",
            "if not start <= available < target_end < end:",
            "tests/test_robustness_framework.py::test_saved_forecast_maturity_and_fit_cutoff_are_audited",
        ),
        (
            "robustness_unknown_gates",
            "robustness/reports.py",
            'missing = [k for k in required if gates.get(k) != "passed"]',
            "missing = []",
            "tests/test_robustness_framework.py::test_missing_gate_and_nan_cannot_promote",
        ),
        (
            "robustness_null_selection_path",
            "robustness/studies.py",
            "fit_fold(source, plan, fold, plan.candidates[0], ledger.call, selector=selector)",
            "fit_fold(source, replace(plan, feature_counts=(1,)), fold, plan.candidates[0], ledger.call, selector=selector)",
            "tests/test_robustness_framework.py::test_perturbed_null_pipeline_repeats_inner_selection_before_scored_rows",
        ),
    ]
)

if __name__ == "__main__":
    raise SystemExit(main())
