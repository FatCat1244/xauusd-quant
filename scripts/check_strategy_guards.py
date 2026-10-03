"""Re-break Stage 13 guards and retained Stage 12 guards in isolated source copies."""

from check_execution_guards import MUTATIONS, main

CHRONOLOGY = "tests/test_strategy_chronology.py::"
FRAMEWORK = "tests/test_strategy_framework.py::"

MUTATIONS.extend(
    [
        (
            "selector_as_of",
            "strategy_validation/pipeline.py",
            "if selection.as_of_utc >= cutoff:",
            "if False and selection.as_of_utc >= cutoff:",
            CHRONOLOGY + "test_deliberately_leaking_feature_selector_is_rejected",
        ),
        (
            "selector_training_identities",
            "strategy_validation/pipeline.py",
            "if not set(selection.used_row_ids) <= {r.row.row_id for r in rows}:",
            "if False and not set(selection.used_row_ids) <= {r.row.row_id for r in rows}:",
            CHRONOLOGY + "test_deliberately_leaking_feature_selector_is_rejected",
        ),
        (
            "actual_label_intervals",
            "strategy_validation/pipeline.py",
            "            and sample.label_end_utc < cutoff\n            and sample.label_available_at_utc < cutoff",
            "            and True",
            CHRONOLOGY + "test_label_intervals_and_delayed_publication_crossing_cutoff_are_purged",
        ),
        (
            "inner_outer_isolation",
            "strategy_validation/pipeline.py",
            "source.training(start, end)",
            "source.training(start, fold.end)",
            CHRONOLOGY + "test_inner_selection_never_requests_outer_outcomes",
        ),
        (
            "bar_prices_before_materialization",
            "strategy_validation/source.py",
            'pl.col("timestamp_utc") + pl.duration(seconds=self.execution.bar_seconds)\n                        < cutoff',
            "pl.lit(True)",
            "tests/test_strategy_source.py::test_development_bar_values_are_cut_before_materialization",
        ),
        (
            "ensemble_selection_chronology",
            "strategy_validation/readiness.py",
            "elif parse_utc(value) >= cutoff:",
            "elif False and parse_utc(value) >= cutoff:",
            CHRONOLOGY + "test_outcome_dependent_ensemble_and_adaptive_choices_respect_cutoff",
        ),
        (
            "unknown_null_promotion",
            "strategy_validation/metrics.py",
            'failures = sorted(k for k in required if gates.get(k) != "passed")',
            "failures = []",
            FRAMEWORK + "test_unknown_null_blocks_even_a_favorable_result",
        ),
        (
            "continuous_boundary_accounting",
            "strategy_validation/metrics.py",
            'or previous["end_mark"] != current["start_mark"]',
            "or False",
            "tests/test_strategy_execution.py::test_fold_boundary_carries_position_costs_and_rejects_duplicate_exposure",
        ),
        (
            "duplicate_position_aggregation",
            "strategy_validation/metrics.py",
            "if position in self.seen_positions:",
            "if False and position in self.seen_positions:",
            FRAMEWORK + "test_duplicate_trade_aggregation_is_rejected",
        ),
        (
            "readiness_before_inputs",
            "strategy_validation/runs.py",
            'blocked = any(assessment["gates"].get(key) != "passed" for key in REQUIRED_GATES)',
            "blocked = False",
            FRAMEWORK + "test_readiness_blocks_before_any_evaluation_values",
        ),
    ]
)

if __name__ == "__main__":
    raise SystemExit(main())
