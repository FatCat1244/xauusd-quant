"""Walk-forward units: every family, reproducibility, variants and nulls (Prompt #10,
Steps 5-10, 17-22, 41-43, 67-69).

Synthetic data only. A unit must score exactly its validation block, reproduce
itself bit for bit with the same seed, find the reversion structure built into
the data, and find nothing once the training labels are shifted away from their
features.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
import pytest

from ml_synth import small_config, synthetic_ml_data
from xauusd_quant.ml.datasets import MLData
from xauusd_quant.ml.splits import Fold, walk_forward_folds
from xauusd_quant.ml.training import UnitResult, UnitSpec, run_unit, time_decay_weights


@cache
def _data() -> MLData:
    return synthetic_ml_data()


def _fold(h: int, index: int = 4) -> Fold:
    return walk_forward_folds(_data().timestamps, small_config().walk_forward, horizon=h)[index]


def _run(target: str, family: str, *, h: int = 5, variant: dict[str, Any] | None = None,
         params: dict[str, Any] | None = None, index: int = 4, **kw: Any) -> UnitResult:
    cfg = small_config()
    data = _data()
    spec = UnitSpec(target=target, horizon=h, family=family, feature_set="standard",
                    features=data.feature_set("standard"), fold=_fold(h, index),
                    params=params if params is not None else cfg.model_params(family),
                    variant=dict(variant or {}), seed=cfg.random_seed, **kw)
    return run_unit(data, spec, cfg.targets[target], cfg)


CLASSIFIERS = ["constant", "logistic_l2", "logistic_l1", "logistic_en", "random_forest",
               "xgboost", "lightgbm", "catboost"]
REGRESSORS = ["constant", "ols", "ridge", "elastic_net", "random_forest", "xgboost",
              "lightgbm", "catboost"]


@pytest.mark.parametrize("family", CLASSIFIERS)
def test_every_classifier_scores_exactly_its_block(family: str) -> None:
    res = _run("mean_reversion", family)
    fold = _fold(5)
    rows = res.predictions["row"].to_numpy()
    assert rows.min() >= fold.validate[0] and rows.max() < fold.validate[1]
    assert np.all(np.diff(rows) > 0)
    for col in ("prediction", "cal_platt", "cal_isotonic"):
        p = res.predictions[col].to_numpy()
        assert np.isfinite(p).all() and p.min() >= 0.0 and p.max() <= 1.0
    assert set(res.metrics) == {"raw", "platt", "isotonic"}
    if family == "constant":                             # the baseline itself: zero skill
        assert abs(res.metrics["raw"]["log_loss_skill"]) < 1e-9
        assert np.ptp(res.predictions["prediction"].to_numpy()) == 0.0


@pytest.mark.parametrize("family", REGRESSORS)
def test_every_regressor_scores_exactly_its_block(family: str) -> None:
    res = _run("future_volatility", family)
    fold = _fold(5)
    rows = res.predictions["row"].to_numpy()
    assert rows.min() >= fold.validate[0] and rows.max() < fold.validate[1]
    assert set(res.metrics) == {"raw"}
    assert np.isfinite(res.predictions["prediction"].to_numpy()).all()
    if family not in ("constant", "elastic_net"):
        assert res.metrics["raw"]["rank_ic"] > 0.3       # vol drives the synthetic volatility


@pytest.mark.parametrize("family", ["random_forest", "xgboost", "lightgbm", "catboost",
                                    "logistic_en"])
def test_training_is_reproducible(family: str) -> None:
    a = _run("mean_reversion", family)
    b = _run("mean_reversion", family)
    for col in a.predictions.columns:
        np.testing.assert_array_equal(a.predictions[col].to_numpy(),
                                      b.predictions[col].to_numpy())
    assert a.metrics == b.metrics


def test_the_structure_is_found_and_the_shifted_target_null_finds_nothing() -> None:
    real = _run("mean_reversion", "lightgbm")
    null = _run("mean_reversion", "lightgbm", variant={"null": "shift", "repeat": 0})
    assert real.metrics["platt"]["auc"] > 0.62
    assert real.metrics["platt"]["log_loss_skill"] > 0.02
    assert abs(null.metrics["platt"]["auc"] - 0.5) < 0.05
    assert null.metrics["platt"]["log_loss_skill"] < 0.01
    # the validation labels stay real: both units score the same rows against the same y
    np.testing.assert_array_equal(real.predictions["row"].to_numpy(),
                                  null.predictions["row"].to_numpy())


def test_noise_columns_and_permuted_features() -> None:
    base = _run("mean_reversion", "lightgbm")
    noisy = _run("mean_reversion", "lightgbm", variant={"noise": 3})
    permuted = _run("mean_reversion", "lightgbm", variant={"permute_features": 1})
    assert noisy.info["design_columns"] == base.info["design_columns"] + 3
    assert abs(permuted.metrics["platt"]["auc"] - 0.5) < 0.05
    assert base.metrics["platt"]["auc"] > permuted.metrics["platt"]["auc"] + 0.1


def test_booster_missing_value_modes_both_run() -> None:
    native = _run("mean_reversion", "lightgbm")
    imputed = _run("mean_reversion", "lightgbm", variant={"tree_missing": "imputed"})
    assert native.info["preprocessing"] == "tree_native"
    assert imputed.info["preprocessing"] == "tree_imputed"
    assert imputed.info["design_columns"] == native.info["design_columns"] + 1   # gappy flag


def test_time_decay_weights_halve_every_half_life() -> None:
    ts = _data().timestamps
    rows = np.arange(ts.len())
    w = time_decay_weights(ts, rows, 2.0)
    assert w[-1] == pytest.approx(1.0)
    years = (ts[-1] - ts[0]).total_seconds() / (365.25 * 86400)
    assert w[0] == pytest.approx(2.0 ** (-years / 2.0), rel=1e-9)
    assert np.all(np.diff(w) >= 0)
    weighted = _run("mean_reversion", "logistic_l2", variant={"weighting": 2.0})
    assert weighted.metrics["platt"]["n"] > 0


def test_a_fold_without_enough_labels_is_refused() -> None:
    cfg = small_config()
    data = _data()
    fold = _fold(5)
    short = Fold(index=9, name="short", fit=(fold.fit[0], fold.fit[0] + 500), inner=fold.inner,
                 validate=fold.validate, horizon=5, embargo=12, scheme="expanding")
    spec = UnitSpec(target="mean_reversion", horizon=5, family="logistic_l2",
                    feature_set="standard", features=data.feature_set("standard"), fold=short,
                    params=cfg.model_params("logistic_l2"), seed=cfg.random_seed)
    with pytest.raises(ValueError, match="too few labelled rows"):
        run_unit(data, spec, cfg.targets["mean_reversion"], cfg)
