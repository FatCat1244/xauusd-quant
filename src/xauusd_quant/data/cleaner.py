"""Conservative cleaning with a full audit trail.

Guarantees, in force under every configuration:

* Bid and ask values are never modified, interpolated or forward-filled.
* No tick is ever fabricated.
* A row is removed only when the configuration explicitly says so, and every
  removal is counted and attributed to a reason.
* When a row trips several checks it is attributed to the *first* one in
  :data:`xauusd_quant.data.validator.CHECKS`, so per-reason counts sum exactly
  to ``rows_dropped``. The independent per-check tallies live alongside, in
  ``flagged_by_check``, and those may overlap.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import polars as pl

from ..utils.config import CleaningConfig
from .validator import CHECKS, FLAG_PREFIX, flag_column

__all__ = ["CleaningSummary", "Cleaner"]


@dataclass
class CleaningSummary:
    """Audit trail for everything the cleaner removed, and why."""

    rows_input: int = 0
    rows_output: int = 0
    rows_dropped: int = 0
    dropped_by_reason: dict[str, int] = field(default_factory=dict)
    flagged_by_check: dict[str, int] = field(default_factory=dict)
    flagged_kept: dict[str, int] = field(default_factory=dict)
    enabled_drop_rules: list[str] = field(default_factory=list)
    policy: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dropped_by_reason"] = {k: v for k, v in self.dropped_by_reason.items() if v}
        payload["flagged_by_check"] = {k: v for k, v in self.flagged_by_check.items() if v}
        payload["flagged_kept"] = {k: v for k, v in self.flagged_kept.items() if v}
        payload["drop_rate"] = (
            self.rows_dropped / self.rows_input if self.rows_input else 0.0
        )
        payload["consistency_check"] = {
            "reasons_sum_to_dropped": sum(self.dropped_by_reason.values()) == self.rows_dropped,
            "input_equals_output_plus_dropped":
                self.rows_input == self.rows_output + self.rows_dropped,
        }
        return payload

    def summary_line(self) -> str:
        reasons = {k: v for k, v in self.dropped_by_reason.items() if v}
        return (
            f"in={self.rows_input:,} out={self.rows_output:,} "
            f"dropped={self.rows_dropped:,} reasons={reasons or 'none'}"
        )


class Cleaner:
    """Applies the configured drop rules to validated batches.

    The cleaner is stateful only in its counters; each batch is cleaned purely
    from its own flag columns, so behaviour does not depend on batch size.
    """

    def __init__(self, config: CleaningConfig) -> None:
        self._cfg = config
        self._drop_checks = self._resolve_drop_checks(config)
        self.summary = CleaningSummary(
            dropped_by_reason=dict.fromkeys(CHECKS, 0),
            flagged_by_check=dict.fromkeys(CHECKS, 0),
            flagged_kept=dict.fromkeys(CHECKS, 0),
            enabled_drop_rules=list(self._drop_checks),
            policy=asdict(config),
        )

    @staticmethod
    def _resolve_drop_checks(config: CleaningConfig) -> tuple[str, ...]:
        """Map ``drop_*`` settings onto check names, preserving CHECKS order."""
        # `drop_unparseable_timestamp` covers both "cell was empty" and
        # "cell could not be parsed": neither yields a usable time.
        aliases = {
            "missing_timestamp": config.drop_unparseable_timestamp,
            "unparseable_timestamp": config.drop_unparseable_timestamp,
            "missing_bid_or_ask": config.drop_missing_bid_or_ask,
            "non_finite": config.drop_non_finite,
            "non_positive_price": config.drop_non_positive_price,
            "price_out_of_range": config.drop_price_out_of_range,
            "ask_below_bid": config.drop_ask_below_bid,
            "negative_spread": config.drop_negative_spread,
            "extreme_spread": config.drop_extreme_spread,
            "negative_volume": config.drop_negative_volume,
            "invalid_volume": config.drop_negative_volume,
            "exact_duplicate_rows": config.drop_exact_duplicate_rows,
            "duplicate_timestamps": config.drop_duplicate_timestamps,
            "non_monotonic": config.drop_non_monotonic,
        }
        return tuple(c for c in CHECKS if aliases.get(c, False))

    def clean(self, flagged: pl.DataFrame) -> pl.DataFrame:
        """Drop the configured rows from *flagged* and strip the flag columns."""
        self.summary.rows_input += flagged.height
        if flagged.is_empty():
            return flagged.drop([c for c in flagged.columns if c.startswith(FLAG_PREFIX)])

        present = [c for c in CHECKS if flag_column(c) in flagged.columns]
        totals = flagged.select(
            [pl.col(flag_column(c)).sum().alias(c) for c in present]
        ).row(0)
        for check, total in zip(present, totals, strict=True):
            self.summary.flagged_by_check[check] += int(total or 0)

        drops = [c for c in self._drop_checks if c in present]
        if not drops:
            self._finish(flagged, present)
            return flagged.drop([c for c in flagged.columns if c.startswith(FLAG_PREFIX)])

        # First-match attribution: a chained when/then in CHECKS order, so a
        # row tripping several rules is counted under exactly one of them.
        # (Polars returns a different type once the chain extends, hence Any.)
        reason: Any = pl.when(pl.col(flag_column(drops[0]))).then(pl.lit(drops[0]))
        for check in drops[1:]:
            reason = reason.when(pl.col(flag_column(check))).then(pl.lit(check))
        labelled = flagged.with_columns(reason.otherwise(None).alias("__drop_reason"))

        counts = (
            labelled.filter(pl.col("__drop_reason").is_not_null())
            .group_by("__drop_reason")
            .len()
        )
        for row in counts.iter_rows(named=True):
            self.summary.dropped_by_reason[row["__drop_reason"]] += int(row["len"])

        kept = labelled.filter(pl.col("__drop_reason").is_null())
        self._finish(kept, present)
        return kept.drop(
            [c for c in kept.columns if c.startswith(FLAG_PREFIX) or c == "__drop_reason"]
        )

    def _finish(self, kept: pl.DataFrame, present: list[str]) -> None:
        """Update output counters and record which flags survived cleaning."""
        self.summary.rows_output += kept.height
        self.summary.rows_dropped = self.summary.rows_input - self.summary.rows_output
        if kept.is_empty():
            return
        survivors = [c for c in present if flag_column(c) in kept.columns]
        if not survivors:
            return
        totals = kept.select([pl.col(flag_column(c)).sum().alias(c) for c in survivors]).row(0)
        for check, total in zip(survivors, totals, strict=True):
            self.summary.flagged_kept[check] += int(total or 0)
