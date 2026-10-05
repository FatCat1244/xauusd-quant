"""Disable demo guards in isolated copies and require each canary to fail."""

from check_execution_guards import MUTATIONS, main

MUTATIONS[:] = [
    ("final_demo_enum", "demo/broker.py",
     "or account.trade_mode != self.__native.ACCOUNT_TRADE_MODE_DEMO",
     "or False", "tests/test_demo_native.py::test_final_identity_read_rejects_racing_real_account"),
    ("private_broker_capability", "demo/broker.py",
     "if self.__authority is None or permit.authority is not self.__authority or permit.request_sha256 != content_hash(request):",
     "if False:", "tests/test_demo_native.py::test_shadow_read_only_and_raw_strategy_cannot_invoke_demo"),
    ("check_success_semantics", "demo/coordinator.py",
     'check.get("retcode") != 0', 'False',
     "tests/test_demo_lifecycle.py::test_check_success_code_distinct_and_failed_check_never_sends"),
    ("durable_submission_attempt", "demo/coordinator.py",
     "self.persist()  # Crash here is uncertain even if native call never started.",
     "pass  # Deliberately broken isolated canary.",
     "tests/test_demo_lifecycle.py::test_reservation_and_attempt_are_durable_before_broker_call"),
    ("material_state_revalidation", "demo/coordinator.py",
     'if not intent["closing"] and ((not self.fresh_quotes and (current["quote"]["bid"]',
     'if False and ((not self.fresh_quotes and (current["quote"]["bid"]',
     "tests/test_demo_guards.py::test_missing_expired_or_changed_approval_blocks_submission[quote]"),
    ("recovered_session_never_rearms", "demo/coordinator.py",
     'if (self.recovered and not pure_precheck) or self.halts or self.risk.state.state != "READY" or not snapshot["permissions"]:',
     'if False:',
     "tests/test_demo_lifecycle.py::test_recovery_after_original_runtime_can_close_but_never_reenter"),
    ("symbol_filling_mapping", "demo/broker.py",
     'return 0 if policy == "FOK" else 1', 'return flag',
     "tests/test_demo_native.py::test_filling_flags_are_mapped_not_copied"),
    ("native_state_binding", "demo/broker.py",
     'if current_state != expected_state:', 'if False:',
     "tests/test_demo_native.py::test_native_handoff_revalidates_expiry_and_economic_state[cash]"),
    ("reserved_period", "execution/io.py",
     'if local_time(b, config) > datetime(2022, 1, 1):',
     'if False and local_time(b, config) > datetime(2022, 1, 1):',
     "tests/test_execution_readiness.py::test_utc_is_explicit_and_stage12_cannot_open_a_new_reserved_path"),
    ("fresh_quote_stage16", "demo/coordinator.py",
     'self._fresh_risk(intent, current, now)', 'pass  # Deliberately disabled Stage16 check.',
     "tests/test_demo_fresh_quotes.py::test_fresh_quote_does_not_bypass_limits[budget]"),
    ("quote_receipt_clock", "demo/broker.py",
     'now = datetime.now(UTC)  # Quote receipt follows API retrieval, not earlier identity reads.',
     'pass  # Deliberately reuse pre-retrieval time.',
     "tests/test_demo_fresh_quotes.py::test_quote_receipt_clock_is_sampled_after_api_retrieval"),
]

if __name__ == "__main__":
    raise SystemExit(main())
