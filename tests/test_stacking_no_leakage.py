"""Second-level fits see earlier out-of-sample rows only (Prompt #11, Steps 5, 21-24, 33;
Rule 2).

The meta walk-forward scores block ``k`` with whatever was fitted on the rows of
blocks ``< k`` whose outcomes had resolved ``h + embargo`` bars before block ``k``.
Corrupting the labels of block ``k``, of every later block and of the purged edge
of block ``k - 1`` must therefore change nothing that is applied to block ``k`` -
the meta-model, the weights, the calibrators. Each test also shows it can fail: a
fit that does see block ``k``'s labels comes out different.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ensemble_synth import synthetic_pair
from xauusd_quant.ensemble.calibration import calibration_variants
from xauusd_quant.ensemble.meta_model import fit_meta_model
from xauusd_quant.ensemble.stacking import meta_inputs, stack_block
from xauusd_quant.ensemble.weighting import diversity_weights, performance_weights

MODELS = ["a|half", "b|half", "c|weak"]
EMBARGO = 12


def _poisoned(pair, k: int):  # type: ignore[no-untyped-def]
    """Labels of block k and later, and of the purged edge of block k-1, flipped."""
    y = pair.label.copy()
    first = int(pair.rows[pair.block_positions(k)[0]])
    edge = (pair.block == k - 1) & (pair.rows >= first - (pair.horizon + EMBARGO))
    bad = (pair.block >= k) | edge
    assert edge.sum() == pair.horizon + EMBARGO          # the edge really is there
    y[bad] = (1.0 - y[bad]) if pair.is_classification else -3.0 * y[bad]
    return replace(pair, label=y, label_raw=y.copy())


def test_history_is_earlier_blocks_minus_the_purged_edge() -> None:
    pair = synthetic_pair()
    for k in (1, 2, 3, 4):
        hist = pair.history(k, EMBARGO)
        first = int(pair.rows[pair.block_positions(k)[0]])
        expected = np.flatnonzero((pair.block < k)
                                  & (pair.rows < first - (pair.horizon + EMBARGO)))
        assert np.array_equal(hist, expected)
        assert (pair.rows[hist] + pair.horizon + EMBARGO < first).all()
    assert pair.history(0, EMBARGO).size == 0              # nothing before the first block


def test_the_meta_model_never_sees_the_scored_block_or_later_labels() -> None:
    for task in ("classification", "regression"):
        pair = synthetic_pair(task)
        k = 2
        pred, meta, hist = stack_block(pair, MODELS, k, embargo=EMBARGO)
        bad = _poisoned(pair, k)
        pred2, meta2, hist2 = stack_block(bad, MODELS, k, embargo=EMBARGO)
        assert np.array_equal(hist, hist2)
        np.testing.assert_array_equal(meta.coef, meta2.coef)
        np.testing.assert_array_equal(pred, pred2)
        # the check has teeth: a meta-model fitted with block k's (poisoned) labels differs
        pos = pair.block_positions(k)
        rows = np.concatenate([hist, pos])
        x, names = meta_inputs(bad, MODELS, rows)
        leaky = fit_meta_model(x, bad.label[rows], names, constituents=len(MODELS), task=task)
        assert not np.allclose(leaky.coef, meta.coef, rtol=1e-3, atol=1e-6)


def test_meta_model_inputs_are_the_stored_out_of_sample_predictions() -> None:
    pair = synthetic_pair()
    k = 3
    hist = pair.history(k, EMBARGO)
    x, names = meta_inputs(pair, MODELS, hist)
    np.testing.assert_array_equal(x, pair.matrix(MODELS)[hist])     # nothing refitted
    assert names == MODELS
    _, meta, used = stack_block(pair, MODELS, k, embargo=EMBARGO)
    assert np.array_equal(used, hist) and meta.params["rows"] == hist.size


def test_weights_and_calibrators_are_fitted_on_history_only() -> None:
    pair = synthetic_pair()
    k = 3
    bad = _poisoned(pair, k)
    hist = pair.history(k, EMBARGO)
    p = pair.matrix(MODELS)
    for which in (pair, bad):
        assert np.array_equal(which.history(k, EMBARGO), hist)
    w1, _ = performance_weights(p[hist], pair.label[hist], pair.base[hist], pair.task,
                                shrink=0.5)
    w2, _ = performance_weights(p[hist], bad.label[hist], bad.base[hist], bad.task, shrink=0.5)
    np.testing.assert_array_equal(w1, w2)
    d1, _ = diversity_weights(p[hist], pair.label[hist], pair.base[hist], pair.task, lam=0.5,
                              shrink=0.5, duplicate_correlation=0.999)
    d2, _ = diversity_weights(p[hist], bad.label[hist], bad.base[hist], bad.task, lam=0.5,
                              shrink=0.5, duplicate_correlation=0.999)
    np.testing.assert_array_equal(d1, d2)
    c1, _ = calibration_variants(pair, MODELS, k, embargo=EMBARGO,
                                 methods=["sigmoid", "isotonic"], min_rows=1000)
    c2, _ = calibration_variants(bad, MODELS, k, embargo=EMBARGO,
                                 methods=["sigmoid", "isotonic"], min_rows=1000)
    for name in c1:
        np.testing.assert_array_equal(c1[name], c2[name])
    # and a weight fitted on block k's own rows would differ
    pos = pair.block_positions(k)
    w3, _ = performance_weights(p[pos], bad.label[pos], bad.base[pos], bad.task, shrink=0.0)
    assert not np.allclose(w3, performance_weights(p[hist], pair.label[hist], pair.base[hist],
                                                   pair.task, shrink=0.0)[0])
