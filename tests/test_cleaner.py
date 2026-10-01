"""Cleaning policy: what gets dropped, what gets kept, and the audit trail."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from conftest import canonical_ticks
from xauusd_quant.data.cleaner import Cleaner
from xauusd_quant.data.validator import FLAG_PREFIX, StreamValidator
from xauusd_quant.utils.config import CleaningConfig

BASE = datetime(2021, 1, 4, 1, 0, 0)


def run(config, frame, cleaning: CleaningConfig | None = None):
    """Validate then clean *frame*, returning the kept rows and the summary."""
    flagged = StreamValidator(config).validate(frame)
    cleaner = Cleaner(cleaning or config.cleaning)
    kept = cleaner.clean(flagged)
    return kept, cleaner.summary


# ---------------------------------------------------------------------------
# Default policy
# ---------------------------------------------------------------------------
def test_clean_rows_all_survive(clean_config):
    frame = canonical_ticks([BASE + timedelta(seconds=i) for i in range(10)],
                            [2000.0] * 10, [2000.3] * 10)
    kept, summary = run(clean_config, frame)
    assert kept.height == 10
    assert summary.rows_dropped == 0


def test_structurally_unusable_rows_are_dropped_by_default(clean_config):
    frame = canonical_ticks(
        [BASE, BASE + timedelta(seconds=1), BASE + timedelta(seconds=2)],
        [2000.0, 0.0, None], [2000.3, 2000.3, 2000.3],
    )
    kept, summary = run(clean_config, frame)
    assert kept.height == 1
    assert summary.dropped_by_reason["missing_bid_or_ask"] == 1
    assert summary.dropped_by_reason["non_positive_price"] == 1


def test_suspicious_but_real_rows_are_kept_and_flagged(clean_config):
    """Crossed quotes, extreme spreads and duplicates are preserved by default."""
    frame = canonical_ticks(
        [BASE, BASE, BASE + timedelta(seconds=2)],
        [2000.5, 2000.5, 2000.0], [2000.1, 2000.5, 2080.0],
    )
    kept, summary = run(clean_config, frame)
    assert kept.height == 3, "nothing should be removed under the default policy"
    assert summary.rows_dropped == 0
    assert summary.flagged_by_check["ask_below_bid"] == 1
    assert summary.flagged_by_check["extreme_spread"] == 1
    assert summary.flagged_kept["ask_below_bid"] == 1


def test_flag_columns_are_stripped_from_the_output(clean_config):
    frame = canonical_ticks([BASE], [2000.0], [2000.3])
    kept, _ = run(clean_config, frame)
    assert not [c for c in kept.columns if c.startswith(FLAG_PREFIX)]
    assert "__drop_reason" not in kept.columns


def test_prices_are_never_modified(clean_config):
    frame = canonical_ticks([BASE, BASE + timedelta(seconds=1)],
                            [2000.123, 1999.987], [2000.456, 2000.111])
    kept, _ = run(clean_config, frame)
    assert kept["bid"].to_list() == pytest.approx([2000.123, 1999.987])
    assert kept["ask"].to_list() == pytest.approx([2000.456, 2000.111])


def test_no_rows_are_ever_added(clean_config):
    """A gap must not be filled in by the cleaner."""
    frame = canonical_ticks([BASE, BASE + timedelta(hours=5)], [2000.0] * 2, [2000.3] * 2)
    kept, summary = run(clean_config, frame)
    assert kept.height == 2
    assert summary.rows_output <= summary.rows_input


# ---------------------------------------------------------------------------
# Configurable policy
# ---------------------------------------------------------------------------
def test_enabling_a_drop_rule_removes_those_rows(clean_config):
    policy = CleaningConfig(drop_ask_below_bid=True)
    frame = canonical_ticks([BASE, BASE + timedelta(seconds=1)],
                            [2000.5, 2000.0], [2000.1, 2000.3])
    kept, summary = run(clean_config, frame, policy)
    assert kept.height == 1
    assert summary.dropped_by_reason["ask_below_bid"] == 1


def test_disabling_every_rule_keeps_everything(clean_config):
    policy = CleaningConfig(
        drop_unparseable_timestamp=False, drop_missing_bid_or_ask=False,
        drop_non_positive_price=False, drop_non_finite=False,
    )
    frame = canonical_ticks([BASE, BASE + timedelta(seconds=1)],
                            [0.0, None], [2000.3, 2000.3])
    kept, summary = run(clean_config, frame, policy)
    assert kept.height == 2
    assert summary.rows_dropped == 0
    assert summary.enabled_drop_rules == []


def test_duplicate_dropping_is_opt_in(clean_config):
    frame = canonical_ticks([BASE, BASE], [2000.0, 2000.0], [2000.3, 2000.3])
    kept_default, _ = run(clean_config, frame)
    assert kept_default.height == 2

    kept_strict, summary = run(clean_config, frame, CleaningConfig(
        drop_exact_duplicate_rows=True))
    assert kept_strict.height == 1
    assert summary.dropped_by_reason["exact_duplicate_rows"] == 1


# ---------------------------------------------------------------------------
# Audit trail integrity
# ---------------------------------------------------------------------------
def test_reason_counts_sum_exactly_to_rows_dropped(clean_config):
    """A row tripping several rules is attributed to exactly one of them."""
    policy = CleaningConfig(
        drop_non_positive_price=True, drop_ask_below_bid=True, drop_extreme_spread=True
    )
    frame = canonical_ticks(
        [BASE + timedelta(seconds=i) for i in range(4)],
        [0.0, 2000.5, 2000.0, 2000.0],       # row 0 is non-positive AND crossed
        [-1.0, 2000.1, 2080.0, 2000.3],
    )
    _, summary = run(clean_config, frame, policy)
    assert sum(summary.dropped_by_reason.values()) == summary.rows_dropped
    payload = summary.to_dict()
    assert payload["consistency_check"]["reasons_sum_to_dropped"]
    assert payload["consistency_check"]["input_equals_output_plus_dropped"]


def test_overlapping_flags_are_all_counted_independently(clean_config):
    """`flagged_by_check` may overlap even though `dropped_by_reason` cannot."""
    frame = canonical_ticks([BASE], [2000.5], [2000.1])
    _, summary = run(clean_config, frame)
    assert summary.flagged_by_check["ask_below_bid"] == 1
    assert summary.flagged_by_check["negative_spread"] == 1


def test_counters_accumulate_across_batches(clean_config):
    cleaner = Cleaner(clean_config.cleaning)
    validator = StreamValidator(clean_config)
    for i in range(3):
        frame = canonical_ticks(
            [BASE + timedelta(seconds=i * 2), BASE + timedelta(seconds=i * 2 + 1)],
            [2000.0, 0.0], [2000.3, 2000.3],
        )
        cleaner.clean(validator.validate(frame))
    assert cleaner.summary.rows_input == 6
    assert cleaner.summary.rows_output == 3
    assert cleaner.summary.rows_dropped == 3


def test_summary_records_the_policy_that_produced_it(clean_config):
    _, summary = run(clean_config, canonical_ticks([BASE], [2000.0], [2000.3]))
    assert summary.policy["drop_non_positive_price"] is True
    assert summary.policy["drop_ask_below_bid"] is False
    assert "non_positive_price" in summary.enabled_drop_rules


def test_drop_rate_is_reported(clean_config):
    frame = canonical_ticks([BASE, BASE + timedelta(seconds=1)],
                            [2000.0, 0.0], [2000.3, 2000.3])
    _, summary = run(clean_config, frame)
    assert summary.to_dict()["drop_rate"] == pytest.approx(0.5)
