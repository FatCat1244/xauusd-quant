"""Simple meta-models, complementary predictors and the noise model (Prompt #11, Steps
21-22, 25-26, 76, 78).

Synthetic streams only: two weak predictors that each see one half of a hidden
signal must be combined into something better than either (logistic and ridge
stacking, and the plain average); a pure-noise constituent must receive almost no
weight; a fitted meta-model serialises to JSON and reproduces its predictions
exactly; a missing constituent prediction is never filled.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ensemble_synth import synthetic_pair
from xauusd_quant.ensemble.averaging import weighted_mean
from xauusd_quant.ensemble.meta_model import MetaModel, fit_meta_model
from xauusd_quant.ensemble.stability import score
from xauusd_quant.ensemble.stacking import stack_block
from xauusd_quant.ensemble.weighting import performance_weights

EMBARGO = 12


def _metric(pair, pred: np.ndarray, pos: np.ndarray) -> float:  # type: ignore[no-untyped-def]
    """Skill of *pred* on the pair positions *pos* (pred: one value per pair row, or one
    per position of *pos*)."""
    m = score(pair.task, pred[pos] if pred.size == pair.n else pred, pair.label[pos],
              pair.base[pos], pair.label_raw[pos])
    key = "log_loss_skill" if pair.is_classification else "mse_skill"
    return float(m[key])


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_stacking_combines_complementary_weak_predictors(task: str) -> None:
    pair = synthetic_pair(task)
    k = 4
    pos = pair.block_positions(k)
    pred, meta, _ = stack_block(pair, ["a|half", "b|half"], k, embargo=EMBARGO)
    stacked = _metric(pair, pred, pos)
    a = _metric(pair, pair.prediction("a|half"), pos)
    b = _metric(pair, pair.prediction("b|half"), pos)
    assert stacked > max(a, b) + 0.05                    # complementary halves add up
    shares = meta.weight_shares()
    assert shares["a|half"] == pytest.approx(0.5, abs=0.1)
    if task == "regression":                             # the plain average helps too
        avg = weighted_mean(pair.matrix(["a|half", "b|half"]))
        assert _metric(pair, avg, pos) > max(a, b)


def test_a_noise_constituent_gets_almost_no_weight() -> None:
    pair = synthetic_pair()
    k = 4
    models = ["a|half", "b|half", "n|noise"]
    _, meta, hist = stack_block(pair, models, k, embargo=EMBARGO)
    assert meta.weight_shares()["n|noise"] < 0.05
    w, q = performance_weights(pair.matrix(models)[hist], pair.label[hist], pair.base[hist],
                               pair.task, shrink=0.0)
    assert q[2] < 0 and w[2] == 0.0                      # negative skill: no raw weight


def test_a_meta_model_round_trips_through_json_exactly() -> None:
    pair = synthetic_pair("regression")
    _, meta, _ = stack_block(pair, ["a|half", "b|half", "c|weak"], 3, embargo=EMBARGO)
    again = MetaModel.from_dict(json.loads(json.dumps(meta.to_dict())))
    x = pair.matrix(["a|half", "b|half", "c|weak"])[:500]
    np.testing.assert_array_equal(meta.predict(x), again.predict(x))


def test_missing_inputs_are_never_filled_and_context_defaults_to_its_centre() -> None:
    rng = np.random.default_rng(1)
    x = np.column_stack([rng.random(2000) * 0.8 + 0.1, rng.random(2000) * 0.8 + 0.1,
                         rng.normal(size=2000)])
    y = (rng.random(2000) < x[:, 0]).astype(float)
    meta = fit_meta_model(x, y, ["m1", "m2", "context:vol"], constituents=2,
                          task="classification")
    probe = x[:3].copy()
    probe[0, 1] = np.nan                                  # a constituent is missing
    probe[1, 2] = np.nan                                  # a context value is missing
    out = meta.predict(probe)
    assert np.isnan(out[0])
    centred = probe[1].copy()
    centred[2] = meta.center[2]
    assert out[1] == pytest.approx(float(meta.predict(centred[None, :])[0]))
    with pytest.raises(ValueError, match="too few"):
        fit_meta_model(x[:50], y[:50], ["m1", "m2", "context:vol"], constituents=2,
                       task="classification")
