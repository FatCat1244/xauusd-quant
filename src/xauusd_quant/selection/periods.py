r"""Chronological periods, purging, embargo and the reserved-test guard (Prompt #9, Steps 4,
37-40, 61-62).

Three consecutive periods: **development** (selection happens here),
**validation** (validates selection choices) and **reserved test** (untouched
until Prompt #10+). Feature *values* may be computed anywhere - they are causal
- but an *outcome* at bar ``t`` with horizon ``h`` uses bars ``t+1 .. t+h``, so

* a target value is usable only if its whole outcome window lies inside the
  period of ``t`` (:func:`outcome_safe`): the last ``h`` bars of development
  would otherwise read validation prices, and the last ``h`` bars of
  validation reserved-test prices;
* every target value of the reserved period is removed at load
  (:class:`ReservedGuard`) and no selection code can ask for one back;
* a training span that ends right before an evaluation span loses the rows
  whose outcome reaches the evaluation span (the *purge*, per target horizon)
  and an extra configured gap (the *embargo*).

Nested folds inside development train strictly before each validation block
(expanding windows); nothing is shuffled.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import polars as pl

from .config import FeatureSelectionConfig, Period

__all__ = [
    "Fold",
    "ReservedGuard",
    "ReservedPeriodError",
    "SplitRows",
    "chronological_folds",
    "first_row_at",
    "outcome_safe",
    "period_mask",
    "period_rows",
]


class ReservedPeriodError(RuntimeError):
    """Selection code asked for an outcome of the reserved test period."""


def _ts(d: date) -> datetime:
    return datetime(d.year, d.month, d.day)


def first_row_at(timestamps: pl.Series, when: date | None) -> int:
    """Index of the first bar at or after *when* (len(timestamps) if none; 0 for None)."""
    if when is None:
        return int(timestamps.len())
    return int(timestamps.search_sorted(_ts(when), side="left"))


def period_rows(timestamps: pl.Series, period: Period) -> tuple[int, int]:
    """[first, last) row range of a period in a time-sorted series."""
    lo = first_row_at(timestamps, period.start)
    hi = int(timestamps.len()) if period.end is None else first_row_at(timestamps, period.end)
    return lo, hi


def period_mask(timestamps: pl.Series, period: Period) -> np.ndarray:
    lo, hi = period_rows(timestamps, period)
    mask = np.zeros(int(timestamps.len()), dtype=bool)
    mask[lo:hi] = True
    return mask


def outcome_safe(values: np.ndarray, horizon: int, boundaries: list[int]) -> np.ndarray:
    """A copy of a target column with every value whose outcome window crosses a boundary
    removed: row ``t`` is kept only if ``t + horizon`` lies before the next boundary row."""
    out = np.array(values, dtype=np.float32)
    for b in boundaries:
        lo = max(0, b - horizon)
        out[lo:b] = np.nan
    return out


class ReservedGuard:
    """Holds the first reserved-test row; masks and refuses reserved outcomes."""

    def __init__(self, reserved_start_row: int, n_rows: int) -> None:
        self.reserved_start_row = int(reserved_start_row)
        self.n_rows = int(n_rows)
        self.masked_values = 0

    def mask(self, values: np.ndarray) -> np.ndarray:
        out = np.array(values, dtype=np.float32)
        self.masked_values += int(np.isfinite(out[self.reserved_start_row:]).sum())
        out[self.reserved_start_row:] = np.nan
        return out

    def check_rows(self, rows: np.ndarray | slice, what: str) -> None:
        """Refuse any outcome request that reaches into the reserved period."""
        if isinstance(rows, slice):
            stop = self.n_rows if rows.stop is None else rows.stop
            reaches = stop > self.reserved_start_row
        else:
            idx = np.asarray(rows)
            if idx.dtype == bool:
                reaches = bool(idx[self.reserved_start_row:].any())
            else:
                reaches = bool(idx.size) and int(idx.max()) >= self.reserved_start_row
        if reaches:
            raise ReservedPeriodError(f"{what}: outcome rows reach the reserved test period "
                                      f"(row >= {self.reserved_start_row}) - refused")


@dataclass(frozen=True)
class SplitRows:
    """One chronological split: training and evaluation row ranges (evaluation after training)."""

    name: str
    train: tuple[int, int]            # [first, last) before purging
    evaluate: tuple[int, int]         # [first, last)

    def train_rows(self, horizon: int, embargo: int) -> tuple[int, int]:
        """Training rows whose outcome window ends before the evaluation span, minus the embargo."""
        lo, hi = self.train
        cut = min(hi, self.evaluate[0] - horizon - embargo)
        return lo, max(lo, cut)


Fold = SplitRows


def chronological_folds(timestamps: pl.Series, cfg: FeatureSelectionConfig) -> list[SplitRows]:
    """The nested development folds, then the final development -> validation split."""
    dev_lo, dev_hi = period_rows(timestamps, cfg.periods.development)
    val_lo, val_hi = period_rows(timestamps, cfg.periods.validation)
    out = []
    for f in cfg.periods.folds:
        v_lo = first_row_at(timestamps, f.validate_start)
        v_hi = first_row_at(timestamps, f.validate_end)
        out.append(SplitRows(name=f"dev_{f.validate_start.year}_{f.validate_end.year - 1}",
                             train=(dev_lo, v_lo), evaluate=(v_lo, v_hi)))
    out.append(SplitRows(name="development_to_validation", train=(dev_lo, dev_hi),
                         evaluate=(val_lo, val_hi)))
    return out
