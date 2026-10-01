r"""Chronological walk-forward splits with purging and embargo (Prompt #10, Steps 11-14).

A fold scores one validation block ``[v_lo, v_hi)``. Its training span ends
``h + embargo`` bars before the block (the *purge*: a training target at ``t``
uses bars ``t+1 .. t+h``, so any ``t >= v_lo - h`` would read validation
prices; the embargo adds a configured margin). The last ``inner_fraction`` of
the span is the *inner slice*, used only for early stopping and probability
calibration, and it is purged from the fitting rows the same way:

.. code-block:: text

    [ fit rows ........ ] gap(h+e) [ inner ] gap(h+e) [ validation block ]

``expanding`` training starts at the first bar; ``rolling`` starts
``rolling_years`` before the block. Nothing is shuffled, and every fold's rows
are strictly before its block. No fold reaches the reserved test period: the
datasets hold no row at or after it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime

import numpy as np
import polars as pl

from .config import WalkForwardConfig

__all__ = [
    "Fold",
    "SplitError",
    "first_row_at",
    "refit_schedule",
    "walk_forward_folds",
]


class SplitError(ValueError):
    """A split would put a validation outcome inside its training rows."""


def _ts(d: date) -> datetime:
    return datetime(d.year, d.month, d.day)


def first_row_at(timestamps: pl.Series, when: date) -> int:
    """Index of the first bar at or after *when* (len if none)."""
    return int(timestamps.search_sorted(_ts(when), side="left"))


@dataclass(frozen=True)
class Fold:
    """One chronological split; every range is ``[lo, hi)`` in row numbers."""

    index: int
    name: str
    fit: tuple[int, int]
    inner: tuple[int, int]
    validate: tuple[int, int]
    horizon: int
    embargo: int
    scheme: str

    @property
    def span(self) -> tuple[int, int]:
        """The whole training span (fit + purge gap + inner slice)."""
        return self.fit[0], self.inner[1]

    def check(self) -> None:
        """Refuse any overlap between an outcome window and a later range."""
        gap = self.horizon + self.embargo
        if not (self.fit[0] <= self.fit[1] <= self.inner[0] <= self.inner[1]
                <= self.validate[0] <= self.validate[1]):
            raise SplitError(f"{self.name}: ranges out of chronological order")
        if self.fit[1] > self.inner[0] - gap and self.inner[1] > self.inner[0]:
            raise SplitError(f"{self.name}: fit outcomes reach the inner slice")
        if self.inner[1] > self.validate[0] - gap:
            raise SplitError(f"{self.name}: training outcomes reach the validation block")


def walk_forward_folds(timestamps: pl.Series, wf: WalkForwardConfig, *, horizon: int,
                       scheme: str | None = None, rolling_years: int | None = None,
                       inner_fraction: float | None = None) -> list[Fold]:
    """The configured validation blocks as purged, embargoed folds for one horizon."""
    scheme = scheme or wf.scheme
    years = rolling_years if rolling_years is not None else wf.rolling_years
    inner_share = wf.inner_fraction if inner_fraction is None else inner_fraction
    gap = int(horizon) + int(wf.embargo_bars)
    out = []
    for i, (start, end) in enumerate(wf.validation_blocks):
        v_lo, v_hi = first_row_at(timestamps, start), first_row_at(timestamps, end)
        if v_hi <= v_lo:
            continue
        span_hi = v_lo - gap
        span_lo = 0 if scheme == "expanding" else first_row_at(
            timestamps, date(start.year - years, start.month, start.day))
        if span_hi - span_lo < 10 * gap:
            raise SplitError(f"block {start}..{end}: training span too short ({span_hi - span_lo})")
        n_inner = max(1, int(round(inner_share * (span_hi - span_lo))))
        inner = (span_hi - n_inner, span_hi)
        fit = (span_lo, max(span_lo, inner[0] - gap))
        fold = Fold(index=i, name=f"wf{i + 1}_{start.year}_{end.year - 1}", fit=fit, inner=inner,
                    validate=(v_lo, v_hi), horizon=int(horizon), embargo=int(wf.embargo_bars),
                    scheme=scheme)
        fold.check()
        out.append(fold)
    return out


def refit_schedule(timestamps: pl.Series, start: date, end: date, every_months: int, *,
                   horizon: int, embargo: int, inner_fraction: float) -> list[Fold]:
    """Expanding refits every *every_months* months inside ``[start, end)`` (Step 51)."""
    out = []
    months = []
    y, m = start.year, start.month
    while date(y, m, 1) < end:
        months.append(date(y, m, 1))
        m += every_months
        while m > 12:
            m -= 12
            y += 1
    months.append(end)
    gap = horizon + embargo
    for i, (a, b) in enumerate(zip(months[:-1], months[1:], strict=True)):
        v_lo, v_hi = first_row_at(timestamps, a), first_row_at(timestamps, min(b, end))
        if v_hi <= v_lo:
            continue
        span_hi = v_lo - gap
        n_inner = max(1, int(round(inner_fraction * span_hi)))
        inner = (span_hi - n_inner, span_hi)
        fold = Fold(index=i, name=f"refit{every_months}m_{a.isoformat()}", fit=(0, inner[0] - gap),
                    inner=inner, validate=(v_lo, v_hi), horizon=horizon, embargo=embargo,
                    scheme="expanding")
        fold.check()
        out.append(fold)
    return out


def with_fit_start(fold: Fold, lo: int, name: str) -> Fold:
    """The same fold with a later fitting start (learning curves, rolling windows)."""
    out = replace(fold, fit=(max(fold.fit[0], lo), fold.fit[1]), name=name)
    if out.fit[1] - out.fit[0] < 1000:
        raise SplitError(f"{name}: fewer than 1000 fitting rows")
    out.check()
    return out


def rows_of(rng: tuple[int, int]) -> np.ndarray:
    return np.arange(rng[0], rng[1], dtype=np.int64)
