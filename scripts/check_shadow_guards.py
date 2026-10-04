"""Break read-only/identity/availability/continuity/risk guards in isolated copies."""

from check_execution_guards import MUTATIONS, main

MUTATIONS[:] = [
    ("demo_identity", "shadow/adapter.py", "or account.trade_mode != self.__native.ACCOUNT_TRADE_MODE_DEMO",
     "or False", "tests/test_shadow_adapter.py::test_wrong_or_real_identity_blocks"),
    ("forbidden_order_trap", "shadow/adapter.py", "self.__native.symbol_info_tick(self.config.symbol)",
     "self.__native.order_send(self.config.symbol)", "tests/test_shadow_adapter.py::test_read_only_calls_and_privacy"),
    ("overlap_occurrences", "shadow/ingestion.py", "if overlap[:len(self.boundary_hashes)] != self.boundary_hashes:",
     "if False:", "tests/test_shadow_ingestion.py::test_changed_or_missing_overlap_blocks"),
    ("publication_clock", "shadow/pipeline.py",
     "if available.tzinfo is None or available < bar.available_utc or (ticks and available < ticks[-1].received_utc):",
     "if False:", "tests/test_shadow_streaming.py::test_backfill_never_creates_live_actions_and_early_publication_rejected"),
    ("scientific_eligibility", "shadow/pipeline.py",
     "if not (offline_synthetic and b.alpha.synthetic) and (cutoff is None or not b.alpha.eligible_at(cutoff) or b.feed_evidence_sha256 is None):",
     "if False:", "tests/test_shadow_pipeline.py::test_synthetic_binding_cannot_enter_actual_constructor"),
    ("risk_boundary", "execution/engine.py", "if self._risk_authority is not None and (",
     "if False and (", "tests/test_risk_portfolio.py::test_raw_call_cannot_reuse_current_authorized_context"),
    ("reserved_period", "execution/io.py", "if local_time(b, config) > datetime(2022, 1, 1):",
     "if False and local_time(b, config) > datetime(2022, 1, 1):",
     "tests/test_execution_readiness.py::test_utc_is_explicit_and_stage12_cannot_open_a_new_reserved_path"),
    ("current_quote_before_target", "alpha_portfolio/portfolio.py", "self._prepare_quote(event)",
     "pass", "tests/test_shadow_pipeline.py::test_governed_synthetic_path_has_hand_checkable_prediction_and_local_fills"),
]

if __name__ == "__main__":
    raise SystemExit(main())
