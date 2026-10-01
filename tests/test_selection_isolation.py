"""The reserved test period and the evaluation rows never steer a selection (Prompt #9,
Steps 4, 37-40, 61-62, 69).

The synthetic target store poisons every reserved-period target (``1e30``) and
overwrites the reserved years' target files with garbage: loading must succeed
without opening them and without keeping one reserved value. Targets are purged
at the development / validation boundary by their own horizon, the guard
refuses any request that reaches the reserved rows, and a nested split's
selection is bit-for-bit the same whatever the evaluation rows' outcomes are.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import numpy as np
import polars as pl
import pytest

from selection_synth import POISON, TF, selection_config, write_stores
from xauusd_quant.alpha.config import load_alpha_config
from xauusd_quant.selection.data import load_selection_data
from xauusd_quant.selection.periods import (
    ReservedGuard,
    ReservedPeriodError,
    SplitRows,
    chronological_folds,
    outcome_safe,
)
from xauusd_quant.selection.selection_cv import method_orderings, prepare_split, split_evidence


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    root = tmp_path_factory.mktemp("selection_store")
    fcfg, tcfg = write_stores(root)
    cfg = selection_config()
    return cfg, fcfg, tcfg, load_selection_data(cfg, fcfg, tcfg, TF)


def test_reserved_outcomes_are_never_loaded(loaded) -> None:
    cfg, _, _, data = loaded
    reserved = datetime(2022, 1, 1)
    assert data.timestamps.max() < reserved                  # rows stop before the reserved start
    assert data.n == int((pl.Series(data.timestamps) < reserved).sum())
    finite = data.targets[np.isfinite(data.targets)]
    assert finite.size and np.abs(finite).max() < POISON / 1e6   # no poisoned value came through
    assert data.guard.reserved_start_row == data.n
    # the corrupt 2022-2023 target files were never opened (reading one raises), yet the
    # reserved *feature values* were read for the availability check
    assert data.reserved_feature_missing["late_start"] == 1.0
    assert data.reserved_feature_missing["sig_a"] == 0.0


def test_targets_are_purged_at_the_period_boundary_by_their_horizon(loaded) -> None:
    _, _, _, data = loaded
    dev_hi = data.rows["development"][1]
    for kind, h, col in data.target_meta:
        y = data.target(col)
        if kind == "reversion":                              # synthetic noise, defined everywhere
            assert np.isnan(y[dev_hi - h:dev_hi]).all() and np.isfinite(y[dev_hi - h - 1])
            continue
        assert np.isnan(y[dev_hi - h:dev_hi]).all(), col     # outcome window crosses into 2018
        assert np.isnan(y[data.n - h:]).all(), col           # ... or into the reserved period
    raw = np.arange(10, dtype=np.float32)
    np.testing.assert_array_equal(np.isnan(outcome_safe(raw, 3, [6, 10])),
                                  [False, False, False, True, True, True, False, True, True, True])


def test_the_guard_refuses_reserved_rows(loaded) -> None:
    cfg, _, _, data = loaded
    guard = ReservedGuard(100, 150)
    guard.check_rows(slice(0, 100), "ok")
    guard.check_rows(np.arange(100), "ok")
    for rows in (slice(90, 101), np.array([5, 120]), np.arange(150) >= 149):
        with pytest.raises(ReservedPeriodError):
            guard.check_rows(rows, "reaches")
    bad = SplitRows(name="leaky", train=(0, 1000), evaluate=(data.n - 50, data.n + 10))
    with pytest.raises(ReservedPeriodError):
        prepare_split(data, bad, [0, 1], cfg)
    assert all(s.evaluate[1] <= data.n for s in chronological_folds(data.timestamps, cfg))


def _selection(data, cfg, split, columns):
    work = prepare_split(data, split, columns, cfg)
    ev = split_evidence(work, data, cfg, load_alpha_config(), with_mi=False)
    orders = {t: method_orderings(work, ev, data, t, cfg)
              for t in ("target_return_1", "target_realized_vol_5")}
    return work, ev, orders


def test_evaluation_outcomes_never_influence_the_training_selection(loaded) -> None:
    cfg, _, _, data = loaded
    cfg = replace(cfg, mrmr=replace(cfg.mrmr, relevance=("abs_rank_ic",)))
    split = chronological_folds(data.timestamps, cfg)[-1]    # development -> validation
    columns = [data.names.index(n) for n in ("sig_a", "sig_a_twin", "vol_x", "vol_x_slow",
                                             "noise_1", "noise_2", "drifting")]
    columns += [data.names.index(p) for p in data.probes]
    work, ev, orders = _selection(data, cfg, split, columns)
    assert any(orders["target_realized_vol_5"].values())    # something was selected
    poisoned = replace(data, targets=data.targets.copy(), features=data.features.copy())
    e_lo, e_hi = split.evaluate
    rng = np.random.default_rng(5)
    poisoned.targets[e_lo:e_hi] = rng.normal(size=(e_hi - e_lo, data.targets.shape[1])) * 1e3
    poisoned.features[e_lo:e_hi] = rng.normal(size=(e_hi - e_lo, data.features.shape[1]))
    work2, ev2, orders2 = _selection(poisoned, cfg, split, columns)
    np.testing.assert_array_equal(work.x_train, work2.x_train)
    np.testing.assert_array_equal(work.moments.sxy, work2.moments.sxy)
    assert ev.equals(ev2)
    assert orders == orders2
    assert not np.array_equal(work.y_eval, work2.y_eval, equal_nan=True)   # the test can tell


def test_training_rows_are_purged_and_embargoed_before_evaluation(loaded) -> None:
    cfg, _, _, data = loaded
    h, emb = max(cfg.targets.horizons), cfg.periods.embargo_bars
    for split in chronological_folds(data.timestamps, cfg):
        lo, hi = split.train_rows(h, emb)
        assert hi <= split.evaluate[0] - h - emb and lo == split.train[0]
