"""Model serialization, artifacts and frozen specifications (Prompt #10, Steps 71-75).

Every family reloads from its native format and reproduces its predictions; an
artifact whose files were changed after saving is refused; a frozen spec cannot
be replaced under the same id, and an edited or unfrozen spec is refused.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from xauusd_quant.ml.calibration import fit_calibrator
from xauusd_quant.ml.models import load_model, make_model, preprocessing_kind
from xauusd_quant.ml.preprocessing import FeatureOrderError, Preprocessor
from xauusd_quant.ml.registry import (
    SpecError,
    freeze_spec,
    load_artifact,
    load_frozen_spec,
    model_id,
    save_artifact,
)

NAMES = ["f0", "f1", "f2", "f3"]
PARAMS = {"xgboost": {"n_estimators": 30, "min_child_weight": 5},
          "lightgbm": {"n_estimators": 30, "min_data_in_leaf": 20},
          "catboost": {"iterations": 30},
          "random_forest": {"n_estimators": 6, "min_samples_leaf": 20}}


def _xy(task: str, n: int = 3000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 4)).astype(np.float32)
    x[rng.random(n) < 0.05, 2] = np.nan
    z = x[:, 0] - 0.5 * x[:, 1] ** 2 + 0.3 * rng.standard_normal(n)
    return x, ((z > 0).astype(np.float64) if task == "classification" else z)


def _fitted(family: str, task: str) -> tuple:
    x, y = _xy(task)
    pre = Preprocessor.fit(x[:2000], NAMES, kind=preprocessing_kind(family))
    model = make_model(family, task, PARAMS.get(family, {}), seed=7, threads=1)
    model.fit(pre.transform(x[:2000], NAMES), y[:2000],
              x_inner=pre.transform(x[2000:2500], NAMES), y_inner=y[2000:2500])
    return model, pre, x[2500:], x[2000:2500], y[2000:2500]


@pytest.mark.parametrize(("family", "task"), [
    ("constant", "classification"), ("logistic_l2", "classification"),
    ("logistic_en", "classification"), ("ridge", "regression"), ("ols", "regression"),
    ("elastic_net", "regression"), ("random_forest", "classification"),
    ("random_forest", "regression"), ("xgboost", "classification"), ("xgboost", "regression"),
    ("lightgbm", "classification"), ("lightgbm", "regression"),
    ("catboost", "classification"), ("catboost", "regression")])
def test_every_family_reloads_with_identical_predictions(family: str, task: str,
                                                          tmp_path: Path) -> None:
    model, pre, x_test, _, _ = _fitted(family, task)
    before = model.predict(pre.transform(x_test, NAMES))
    model.save(tmp_path)
    again = load_model(tmp_path)
    after = again.predict(pre.transform(x_test, NAMES))
    np.testing.assert_allclose(after, before, rtol=1e-7, atol=1e-12)
    assert again.family == family and again.task == task
    if task == "classification":
        assert before.min() >= 0.0 and before.max() <= 1.0


def _spec(**changes: object) -> dict:
    spec = {"spec_id": model_id("lightgbm", "mean_reversion", "5m", 5), "family": "lightgbm",
            "target": "mean_reversion", "target_definition": {"source": "residual_shrinks"},
            "horizon": 5, "timeframe": "5m", "feature_set": "standard",
            "feature_set_id": "FS_STANDARD", "feature_set_hash": "abc", "features": NAMES,
            "params": PARAMS["lightgbm"], "preprocessing": {"kind": "tree_native", "clip": 8.0},
            "calibration": "platt", "training_policy": {"scheme": "expanding"}, "seed": 7,
            "dataset_versions": {"tick_dataset_version": "ticks-x"}}
    spec.update(changes)
    return spec


def test_an_artifact_round_trips_and_detects_tampering(tmp_path: Path) -> None:
    model, pre, x_test, x_cal, y_cal = _fitted("lightgbm", "classification")
    cal = fit_calibrator(model.predict(pre.transform(x_cal, NAMES)), y_cal, "platt")
    spec = load_frozen_spec(freeze_spec(_spec(), tmp_path / "frozen"))
    art = save_artifact(tmp_path / "models" / spec["spec_id"], model=model, preprocessor=pre,
                        calibrator=cal, spec=spec, extra={"training": {"base": 0.5}})
    m2, p2, c2, manifest = load_artifact(art)
    first = cal.apply(model.predict(pre.transform(x_test, NAMES)))
    again = c2.apply(m2.predict(p2.transform(x_test, NAMES)))
    np.testing.assert_allclose(again, first, rtol=1e-7, atol=1e-12)
    assert manifest["spec_hash"] == spec["content_hash"] and manifest["features"] == NAMES
    assert manifest["model_size_bytes"] > 0
    with pytest.raises(FeatureOrderError):
        p2.transform(x_test[:, [1, 0, 2, 3]], ["f1", "f0", "f2", "f3"])
    pre_file = art / "preprocessor.json"
    body = json.loads(pre_file.read_text(encoding="utf-8"))
    body["features"] = ["f1", "f0", "f2", "f3"]                 # a silent column swap
    pre_file.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(SpecError, match="file hash differs"):
        load_artifact(art)


def test_frozen_specs_are_immutable(tmp_path: Path) -> None:
    path = freeze_spec(_spec(), tmp_path)
    assert path.name == "MODEL_SPEC_LGBM_REVERSION_5M_H5_V001.json"
    assert freeze_spec(_spec(), tmp_path) == path                # same content: no-op
    with pytest.raises(SpecError, match="already frozen with other content"):
        freeze_spec(_spec(params={"n_estimators": 31}), tmp_path)
    spec = load_frozen_spec(path)
    assert spec["frozen"] is True
    body = json.loads(path.read_text(encoding="utf-8"))
    body["features"] = NAMES[:3]
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(SpecError, match="content hash does not match"):
        load_frozen_spec(path)
    body["frozen"] = False
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(SpecError, match="not frozen"):
        load_frozen_spec(path)
    with pytest.raises(SpecError, match="no model spec"):
        load_frozen_spec(tmp_path / "MODEL_SPEC_NOPE.json")


def test_model_ids_name_family_target_timeframe_horizon_and_version() -> None:
    assert model_id("xgboost", "mean_reversion", "5m", 5) == "XGB_REVERSION_5M_H5_V001"
    assert model_id("logistic_l2", "future_volatility", "15m", 20, 3) == "LOGL2_VOL_15M_H20_V003"
    assert model_id("constant", "direction_up_cost", "5m", 1) == "CONST_UPCOST_5M_H1_V001"
