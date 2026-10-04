"""Disable Stage15 guards in isolated copies, retaining the 42 preceding-stage checks."""

from check_robustness_guards import MUTATIONS, main

MUTATIONS.extend(
    [
        (
            "portfolio_prior_eligibility",
            "alpha_portfolio/registry.py",
            "and parse_utc(self.evidence_available_utc) < cutoff",
            "and True",
            "tests/test_alpha_portfolio_causality.py::test_fold_eligibility_cannot_backdate_stage14_universe",
        ),
        (
            "portfolio_prior_weight_information",
            "alpha_portfolio/allocation.py",
            "if utc_time(self.information_as_of_utc) >= utc_time(self.effective_utc):",
            "if False:",
            "tests/test_alpha_portfolio_causality.py::test_weight_information_cannot_equal_or_follow_effective_time",
        ),
        (
            "portfolio_matured_risk_observations",
            "alpha_portfolio/allocation.py",
            "if t < effective and m < effective",
            "if t < effective",
            "tests/test_alpha_portfolio_causality.py::test_matured_returns_required_for_weights",
        ),
        (
            "portfolio_future_risk_observations",
            "alpha_portfolio/allocation.py",
            "if t < effective and m < effective",
            "if True",
            "tests/test_alpha_portfolio_causality.py::test_future_append_no_change_to_frozen_weight_identity",
        ),
        (
            "portfolio_future_forecast_publication",
            "alpha_portfolio/intents.py",
            "if utc_time(forecast.available_at_utc) > at:",
            "if False:",
            "tests/test_alpha_portfolio_causality.py::test_future_publication_refused_after_bar_already_completed",
        ),
        (
            "portfolio_own_frequency_contract",
            "alpha_portfolio/portfolio.py",
            "and (event.timeframe, event.horizon_bars) != self.contracts[event.alpha_id]",
            "and False",
            "tests/test_alpha_portfolio_causality.py::test_scientific_constructor_refuses_future_universe_and_incompatible_intent",
        ),
        (
            "portfolio_checkpoint_budget",
            "alpha_portfolio/portfolio.py",
            'or state.get("configuration_sha256") != portfolio.state()["configuration_sha256"]',
            "or False",
            "tests/test_alpha_portfolio_account.py::test_state_reload_chunks_same_records_and_funding",
        ),
        (
            "portfolio_scientific_entry",
            "alpha_portfolio/portfolio.py",
            "if any(not alpha.eligible_at(cutoff) for alpha in alphas):",
            "if False:",
            "tests/test_alpha_portfolio_causality.py::test_scientific_constructor_refuses_future_universe_and_incompatible_intent",
        ),
        (
            "portfolio_immutable_specification",
            "alpha_portfolio/plan.py",
            'if previous.get("content_sha256") != digest or content_hash(previous["body"]) != digest:',
            "if False:",
            "tests/test_alpha_portfolio_framework.py::test_plan_and_registry_immutable_and_missing_inputs_rejected",
        ),
    ]
)

MUTATIONS.append(
    (
        "portfolio_serialized_timestamp_identity",
        "alpha_portfolio/plan.py",
        "body = _json_safe(body)",
        "body = body",
        "tests/test_alpha_portfolio_framework.py::test_frozen_datetime_hash_matches_serialized_body_and_reuse",
    )
)

MUTATIONS.extend(
    [
        (
            "portfolio_complete_policy_contract",
            "alpha_portfolio/registry.py",
        "complete_policy\n            and not self.synthetic",
            "not self.synthetic",
            "tests/test_alpha_portfolio_causality.py::test_missing_negative_evidence_never_promotes",
        ),
        (
            "portfolio_corrected_fit_inventory",
            "alpha_portfolio/registry.py",
            'run = f"STRATEGY_{timeframe.upper()}_DIAGNOSTIC_V002"',
            'run = f"STRATEGY_{timeframe.upper()}_DIAGNOSTIC_V001"',
            "tests/test_alpha_portfolio_framework.py::test_corrected_stage13_fits_referenced_without_erasing_failed_attempt",
        ),
    ]
)

if __name__ == "__main__":
    raise SystemExit(main())
