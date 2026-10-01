"""Chronological walk-forward folds with purge and embargo (Prompt #10, Steps 11-14).

No validation date may appear in a fold's fitting or inner rows, and the gap
between them must cover the outcome horizon plus the embargo, so the last
training label's outcome window ends before the first scored bar.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from ml_synth import RESERVED, small_config, stamps, synthetic_ml_data
from xauusd_quant.ml.splits import (
    SplitError,
    first_row_at,
    refit_schedule,
    walk_forward_folds,
    with_fit_start,
)
from xauusd_quant.ml.training import final_fold

TS = stamps()


@pytest.mark.parametrize("scheme", ["expanding", "rolling"])
@pytest.mark.parametrize("h", [1, 5, 20])
def test_no_validation_date_is_in_training(scheme: str, h: int) -> None:
    wf = small_config().walk_forward
    folds = walk_forward_folds(TS, wf, horizon=h, scheme=scheme)
    assert len(folds) == len(wf.validation_blocks) == 5
    for fold, (start, end) in zip(folds, wf.validation_blocks, strict=True):
        fit = np.arange(*fold.fit)
        inner = np.arange(*fold.inner)
        val = np.arange(*fold.validate)
        assert fit.size > 1000 and inner.size > 0 and val.size > 0
        assert not set(fit) & set(val) and not set(inner) & set(val) and not set(fit) & set(inner)
        val_dates = set(TS.gather(val).dt.date().to_list())
        train_dates = set(TS.gather(np.concatenate([fit, inner])).dt.date().to_list())
        assert not val_dates & train_dates
        assert TS[int(fit[-1])] < TS[int(inner[0])] < TS[int(val[0])]
        # the validation rows are exactly the block's bars
        assert TS[int(val[0])].date() >= start and TS[int(val[-1])].date() < end


@pytest.mark.parametrize("h", [1, 5, 20])
def test_purge_covers_the_horizon_and_the_embargo(h: int) -> None:
    wf = small_config().walk_forward
    for fold in walk_forward_folds(TS, wf, horizon=h):
        gap = h + wf.embargo_bars
        assert fold.inner[0] - fold.fit[1] >= gap
        assert fold.validate[0] - fold.inner[1] >= gap
        # the last labelled row's outcome window (t, t + h] ends before the next segment
        assert (fold.fit[1] - 1) + h < fold.inner[0]
        assert (fold.inner[1] - 1) + h < fold.validate[0]
    wider = replace(wf, embargo_bars=wf.embargo_bars + 30)
    a = walk_forward_folds(TS, wf, horizon=h)[2]
    b = walk_forward_folds(TS, wider, horizon=h)[2]
    assert b.validate == a.validate and b.inner[1] == a.inner[1] - 30


def test_rolling_folds_start_within_the_lookback() -> None:
    wf = small_config().walk_forward
    for fold, (start, _) in zip(walk_forward_folds(TS, wf, horizon=5, scheme="rolling"),
                                wf.validation_blocks, strict=True):
        lo = first_row_at(TS, date(start.year - wf.rolling_years, start.month, start.day))
        assert fold.fit[0] == lo
        assert fold.scheme == "rolling"


def test_refits_never_train_on_their_own_block() -> None:
    folds = refit_schedule(TS, date(2018, 1, 1), date(2022, 1, 1), 3, horizon=5, embargo=12,
                           inner_fraction=0.1)
    assert len(folds) == 16
    for a, b in zip(folds[:-1], folds[1:], strict=True):
        assert a.validate[1] == b.validate[0]                  # contiguous, no overlap
    for f in folds:
        assert f.fit[0] == 0 and f.fit[1] + 17 <= f.inner[0] and f.inner[1] + 17 <= f.validate[0]
    assert folds[0].validate[0] == first_row_at(TS, date(2018, 1, 1))
    assert folds[-1].validate[1] == first_row_at(TS, date(2022, 1, 1)) == TS.len()


def test_a_too_short_training_span_is_refused() -> None:
    wf = replace(small_config().walk_forward,
                 validation_blocks=((date(2006, 1, 9), date(2007, 1, 1)),))
    with pytest.raises(SplitError, match="training span too short"):
        walk_forward_folds(TS, wf, horizon=20)


def test_the_final_fold_ends_at_the_last_labelled_row() -> None:
    data = synthetic_ml_data()
    cfg = small_config()
    for h in (1, 5, 20):
        fold = final_fold(data, cfg, horizon=h, scheme="expanding", rolling_years=5)
        assert fold.validate == (data.n, data.n)               # nothing scored here
        assert fold.inner[1] == data.n - h
        assert fold.inner[0] - fold.fit[1] >= h + cfg.walk_forward.embargo_bars
        y = data.targets_raw[f"target_return_{h}"]
        assert np.isfinite(y[fold.inner[1] - 1]) and not np.isfinite(y[fold.inner[1]:]).any()
    rolling = final_fold(data, cfg, horizon=5, scheme="rolling", rolling_years=5)
    assert rolling.fit[0] == first_row_at(data.timestamps, date(RESERVED.year - 5, 1, 1))


def test_a_later_fit_start_keeps_the_purge() -> None:
    fold = walk_forward_folds(TS, small_config().walk_forward, horizon=5)[-1]
    late = with_fit_start(fold, fold.fit[1] - 2000, "late")
    assert late.fit == (fold.fit[1] - 2000, fold.fit[1])
    assert late.inner == fold.inner and late.validate == fold.validate
    with pytest.raises(SplitError, match="fewer than 1000"):
        with_fit_start(fold, fold.fit[1] - 500, "too_late")
