"""Validation checks, including behaviour across batch boundaries."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from conftest import canonical_ticks
from xauusd_quant.data.converter import TickConverter
from xauusd_quant.data.validator import CHECKS, StreamValidator, flag_column

BASE = datetime(2021, 1, 4, 1, 0, 0)


def counts_for(config, frame: pl.DataFrame) -> dict[str, int]:
    validator = StreamValidator(config)
    validator.validate(frame)
    return validator.report().counts


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def test_clean_data_raises_no_flags(clean_config):
    frame = canonical_ticks(
        [BASE + timedelta(milliseconds=100 * i) for i in range(50)],
        [2000.0 + i * 0.01 for i in range(50)],
        [2000.3 + i * 0.01 for i in range(50)],
    )
    assert all(v == 0 for v in counts_for(clean_config, frame).values())


def test_detects_crossed_quotes_and_negative_spread(clean_config):
    frame = canonical_ticks([BASE, BASE + timedelta(seconds=1)],
                            [2000.0, 2000.5], [2000.3, 2000.1])
    counts = counts_for(clean_config, frame)
    assert counts["ask_below_bid"] == 1
    assert counts["negative_spread"] == 1


def test_detects_non_positive_prices(clean_config):
    frame = canonical_ticks([BASE, BASE + timedelta(seconds=1)],
                            [0.0, -1.0], [2000.3, 2000.1])
    assert counts_for(clean_config, frame)["non_positive_price"] == 2


def test_detects_extreme_spread_against_the_configured_threshold(clean_config):
    frame = canonical_ticks([BASE, BASE + timedelta(seconds=1)],
                            [2000.0, 2000.0], [2000.3, 2080.0])
    assert counts_for(clean_config, frame)["extreme_spread"] == 1


def test_detects_out_of_range_prices(clean_config):
    frame = canonical_ticks([BASE, BASE + timedelta(seconds=1)],
                            [99.0, 2000.0], [99.3, 2000.3])
    assert counts_for(clean_config, frame)["price_out_of_range"] == 1


def test_detects_missing_quotes_and_non_finite_values(clean_config):
    frame = canonical_ticks(
        [BASE + timedelta(seconds=i) for i in range(3)],
        [2000.0, None, float("inf")],
        [2000.3, 2000.3, 2000.3],
    )
    counts = counts_for(clean_config, frame)
    assert counts["missing_bid_or_ask"] == 1
    assert counts["non_finite"] == 1


def test_detects_nan_prices(clean_config):
    frame = canonical_ticks([BASE, BASE + timedelta(seconds=1)],
                            [2000.0, float("nan")], [2000.3, 2000.3])
    assert counts_for(clean_config, frame)["non_finite"] == 1


def test_detects_duplicate_timestamps_but_not_on_distinct_ones(clean_config):
    frame = canonical_ticks([BASE, BASE, BASE + timedelta(seconds=1)],
                            [2000.0, 2001.0, 2002.0], [2000.3, 2001.3, 2002.3])
    counts = counts_for(clean_config, frame)
    assert counts["duplicate_timestamps"] == 1
    assert counts["exact_duplicate_rows"] == 0  # same time, different prices


def test_detects_exact_duplicate_rows(clean_config):
    frame = canonical_ticks([BASE, BASE, BASE + timedelta(seconds=1)],
                            [2000.0, 2000.0, 2002.0], [2000.3, 2000.3, 2002.3])
    counts = counts_for(clean_config, frame)
    assert counts["exact_duplicate_rows"] == 1
    assert counts["duplicate_timestamps"] == 1


def test_detects_non_monotonic_timestamps(clean_config):
    frame = canonical_ticks(
        [BASE, BASE + timedelta(seconds=2), BASE + timedelta(seconds=1)],
        [2000.0, 2001.0, 2002.0], [2000.3, 2001.3, 2002.3],
    )
    assert counts_for(clean_config, frame)["non_monotonic"] == 1


def test_detects_negative_volume(clean_config):
    frame = canonical_ticks([BASE, BASE + timedelta(seconds=1)],
                            [2000.0, 2000.0], [2000.3, 2000.3], [120.0, -5.0])
    assert counts_for(clean_config, frame)["negative_volume"] == 1


# ---------------------------------------------------------------------------
# Gaps
# ---------------------------------------------------------------------------
def test_records_large_gaps_with_their_boundaries(clean_config):
    frame = canonical_ticks(
        [BASE, BASE + timedelta(seconds=1), BASE + timedelta(hours=3)],
        [2000.0] * 3, [2000.3] * 3,
    )
    report = StreamValidator(clean_config)
    report.validate(frame)
    result = report.report()
    assert result.gap_count == 1
    assert result.largest_gap_seconds == pytest.approx(3 * 3600 - 1)
    assert result.gaps[0].start.startswith("2021-01-04 01:00:01")


def test_gaps_below_the_threshold_are_ignored(clean_config):
    frame = canonical_ticks([BASE, BASE + timedelta(minutes=30)],
                            [2000.0] * 2, [2000.3] * 2)
    assert StreamValidator(clean_config).validate(frame) is not None
    validator = StreamValidator(clean_config)
    validator.validate(frame)
    assert validator.report().gap_count == 0


# ---------------------------------------------------------------------------
# Cross-batch behaviour
# ---------------------------------------------------------------------------
def test_results_are_identical_whatever_the_batch_split(clean_config):
    """Duplicate/ordering/gap checks must not depend on where batches break."""
    stamps = [BASE, BASE, BASE + timedelta(seconds=1), BASE,
              BASE + timedelta(hours=4), BASE + timedelta(hours=4)]
    bids = [2000.0, 2000.0, 2001.0, 2002.0, 2003.0, 2003.0]
    asks = [b + 0.3 for b in bids]
    whole = canonical_ticks(stamps, bids, asks)

    single = StreamValidator(clean_config)
    single.validate(whole)

    split = StreamValidator(clean_config)
    for lo, hi in [(0, 1), (1, 3), (3, 4), (4, 6)]:
        split.validate(canonical_ticks(stamps[lo:hi], bids[lo:hi], asks[lo:hi]))

    assert single.report().counts == split.report().counts
    assert single.report().gap_count == split.report().gap_count


def test_first_row_of_a_later_batch_is_still_checked(clean_config):
    """Without carry-over state this duplicate would slip through unseen."""
    first = canonical_ticks([BASE], [2000.0], [2000.3])
    second = canonical_ticks([BASE], [2000.0], [2000.3])
    validator = StreamValidator(clean_config)
    validator.validate(first)
    validator.validate(second)
    counts = validator.report().counts
    assert counts["duplicate_timestamps"] == 1
    assert counts["exact_duplicate_rows"] == 1


# ---------------------------------------------------------------------------
# Report shape
# ---------------------------------------------------------------------------
def test_report_exposes_every_declared_check(clean_config):
    frame = canonical_ticks([BASE], [2000.0], [2000.3])
    validator = StreamValidator(clean_config)
    validator.validate(frame)
    assert set(validator.report().counts) == set(CHECKS)


def test_flag_columns_are_attached_without_dropping_rows(clean_config):
    frame = canonical_ticks([BASE, BASE + timedelta(seconds=1)],
                            [0.0, 2000.0], [2000.3, 2000.3])
    flagged = StreamValidator(clean_config).validate(frame)
    assert flagged.height == 2, "the validator must never remove rows"
    assert flagged[flag_column("non_positive_price")].to_list() == [True, False]


def test_report_records_the_thresholds_it_used(clean_config):
    validator = StreamValidator(clean_config)
    validator.validate(canonical_ticks([BASE], [2000.0], [2000.3]))
    assert validator.report().thresholds["max_spread"] == 50.0


def test_examples_are_collected_and_bounded(clean_config):
    stamps = [BASE + timedelta(seconds=i) for i in range(40)]
    frame = canonical_ticks(stamps, [0.0] * 40, [2000.3] * 40)
    validator = StreamValidator(clean_config)
    validator.validate(frame)
    examples = validator.report().examples["non_positive_price"]
    assert 0 < len(examples) <= clean_config.validation.max_reported_examples


# ---------------------------------------------------------------------------
# Raw-input integration: every defect in the dirty file is found
# ---------------------------------------------------------------------------
def test_dirty_file_trips_every_expected_check(dirty_csv, config_factory):
    config = config_factory(dirty_csv)
    converter = TickConverter(config)
    validator = StreamValidator(config)
    for batch in converter.iter_canonical_batches(dirty_csv):
        validator.validate(batch)
    # Malformed rows never reach a batch, so they come from the reader.
    validator.note_malformed(converter.malformed_rows)
    report = validator.report()

    assert report.malformed_rows == 1, "the 3-field row must be rejected by the parser"
    for check in ("missing_timestamp", "unparseable_timestamp", "missing_bid_or_ask",
                  "non_positive_price", "ask_below_bid", "negative_spread",
                  "extreme_spread", "negative_volume", "exact_duplicate_rows",
                  "duplicate_timestamps", "non_monotonic", "price_out_of_range"):
        assert report.counts[check] >= 1, f"{check} was not detected"
    assert report.gap_count == 1


# ---------------------------------------------------------------------------
# Duplicates that are NOT adjacent
#
# The 2021-2026 part of the real export repeats a whole hour of ticks at each
# weekly splice, so the two copies sit thousands of rows apart. Comparing each
# row only with its predecessor cannot see that, which is how 2.68 M duplicate
# rows went unreported in the first conversion of that dataset.
# ---------------------------------------------------------------------------
def _repeated_block(gap_rows: int):
    """A block of ticks, then `gap_rows` unrelated ticks, then the block again."""
    block = [(BASE + timedelta(milliseconds=100 * i), 2000.0 + i * 0.1) for i in range(5)]
    filler = [(BASE + timedelta(seconds=10 + i), 2100.0 + i * 0.1) for i in range(gap_rows)]
    rows = block + filler + block
    stamps = [r[0] for r in rows]
    bids = [r[1] for r in rows]
    return stamps, bids, [b + 0.3 for b in bids]


def test_detects_a_repeated_block_far_from_its_original(clean_config):
    stamps, bids, asks = _repeated_block(gap_rows=400)
    frame = canonical_ticks(stamps, bids, asks)
    counts = counts_for(clean_config, frame)
    assert counts["exact_duplicate_rows"] == 5
    assert counts["duplicate_timestamps"] == 5


def test_only_the_later_copy_of_a_repeat_is_flagged(clean_config):
    """The first occurrence must survive cleaning."""
    stamps, bids, asks = _repeated_block(gap_rows=50)
    flagged = StreamValidator(clean_config).validate(canonical_ticks(stamps, bids, asks))
    marks = flagged[flag_column("exact_duplicate_rows")].to_list()
    assert marks[:5] == [False] * 5, "the original block must not be flagged"
    assert marks[-5:] == [True] * 5, "the repeated block must be flagged"
    assert sum(marks) == 5


def test_repeat_is_still_found_when_it_straddles_a_batch_boundary(clean_config):
    stamps, bids, asks = _repeated_block(gap_rows=50)
    validator = StreamValidator(clean_config)
    total = 0
    for lo, hi in [(0, 30), (30, 45), (45, len(stamps))]:
        out = validator.validate(canonical_ticks(stamps[lo:hi], bids[lo:hi], asks[lo:hi]))
        total += int(out[flag_column("exact_duplicate_rows")].sum())
    assert total == 5
    assert validator.report().counts["exact_duplicate_rows"] == 5


@pytest.mark.parametrize("splits", [[60], [5, 55], [3, 7, 40, 58], [1, 2, 3, 4, 5]])
def test_duplicate_count_does_not_depend_on_batching(clean_config, splits):
    stamps, bids, asks = _repeated_block(gap_rows=50)
    bounds = [0, *splits, len(stamps)]
    validator = StreamValidator(clean_config)
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        if hi > lo:
            validator.validate(canonical_ticks(stamps[lo:hi], bids[lo:hi], asks[lo:hi]))
    assert validator.report().counts["exact_duplicate_rows"] == 5


def test_duplicates_beyond_the_window_are_not_claimed_to_be_found(clean_csv, config_factory):
    """The window is a documented limit, so make it observable rather than silent."""
    config = config_factory(clean_csv, validation={"duplicate_window_rows": 10})
    stamps, bids, asks = _repeated_block(gap_rows=400)
    validator = StreamValidator(config)
    for lo in range(0, len(stamps), 20):
        validator.validate(
            canonical_ticks(stamps[lo:lo + 20], bids[lo:lo + 20], asks[lo:lo + 20])
        )
    # With a 10-row window and a 400-row separation the repeat is out of reach.
    assert validator.report().counts["exact_duplicate_rows"] == 0


def test_same_timestamp_with_different_prices_is_not_an_exact_duplicate(clean_config):
    """Two genuine ticks in one millisecond must not be treated as corruption."""
    frame = canonical_ticks([BASE, BASE, BASE + timedelta(seconds=1)],
                            [2000.0, 2001.0, 2002.0], [2000.3, 2001.3, 2002.3])
    counts = counts_for(clean_config, frame)
    assert counts["duplicate_timestamps"] == 1
    assert counts["exact_duplicate_rows"] == 0
