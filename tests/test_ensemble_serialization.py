"""A frozen ensemble behind one interface, and its refusals (Prompt #11, Steps 66-70).

Real constituents (tiny logistic / LightGBM / ridge models on features the factory
engines compute from synthetic bars) are saved as artifacts with their frozen
specs; the ensemble is loaded through :class:`EnsembleModel`, and:

* a reloaded ensemble reproduces its predictions exactly; one row through
  ``predict`` / ``predict_proba`` equals the batch row; the combination equals the
  frozen rule applied by hand;
* a constituent artifact trained from another spec version, a spec hash that is
  not the frozen one, or a live feature manifest that differs fails loudly (no
  silent substitution);
* a missing artifact, a missing feature or a constituent without a prediction is
  ``ENSEMBLE_INVALID`` - fail closed, never a reduced ensemble.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from ensemble_synth import NAMES, bars_and_features, frozen_ensemble, live_manifest
from xauusd_quant.ensemble.inference import (
    EnsembleInvalidError,
    EnsembleModel,
    FeatureVersionMismatchError,
    ModelVersionMismatchError,
)
from xauusd_quant.ensemble.registry import freeze_ensemble_spec, load_frozen_ensemble_spec

ROWS = np.arange(1990, 2390, 5)


def _load(path: Path, info: dict, **kw: object) -> EnsembleModel:
    return EnsembleModel.load(path, models_path=info["models"],
                              constituent_dir=info["constituents"],
                              live_manifests=kw.get("manifests", live_manifest(info["members"])))


def _x() -> dict[str, np.ndarray]:
    _, stored, _, _ = bars_and_features()
    return {f: stored[f][ROWS] for f in NAMES}


@pytest.mark.parametrize(("task", "method"), [("classification", "weights"),
                                              ("classification", "stacking"),
                                              ("regression", "simple_average")])
def test_a_reloaded_ensemble_reproduces_its_predictions(tmp_path: Path, task: str,
                                                        method: str) -> None:
    path, info = frozen_ensemble(tmp_path, task=task, method=method)
    m1 = _load(path, info)
    a = m1.predict_batch(_x())
    b = _load(path, info).predict_batch(_x())
    np.testing.assert_array_equal(a["prediction"], b["prediction"])
    np.testing.assert_array_equal(a["constituents"], b["constituents"])
    assert (a["status"] == "ok").all() and np.isfinite(a["prediction"]).all()
    if method == "weights":                               # the frozen rule, by hand
        np.testing.assert_allclose(a["prediction"], a["constituents"] @ np.array([0.25, 0.75]))
    if method == "simple_average":
        np.testing.assert_allclose(a["prediction"], a["constituents"].mean(axis=1))
    one = {f: float(v[7]) for f, v in _x().items()}
    assert m1.predict(one) == pytest.approx(float(a["prediction"][7]), rel=1e-6)
    if task == "classification":
        assert 0.0 < m1.predict_proba(one) < 1.0
    else:
        with pytest.raises(TypeError, match="expected value"):
            m1.predict_proba(one)
    assert m1.describe()["constituents"] == [c["model_id"] for c in info["members"]]


def test_a_constituent_of_another_version_fails_clearly(tmp_path: Path) -> None:
    path, info = frozen_ensemble(tmp_path)
    # "MODEL_XGB_V4 expected, MODEL_XGB_V5 loaded": the V001 directory now holds an artifact
    # trained from another spec (other parameters, version 2)
    from ensemble_synth import _constituent

    other = _constituent(tmp_path, "lightgbm", "classification", NAMES, version=2,
                         params={"n_estimators": 12, "min_data_in_leaf": 40})
    v1 = info["models"] / info["members"][1]["model_id"]
    shutil.rmtree(v1)
    shutil.copytree(info["models"] / other["model_id"], v1)
    with pytest.raises(ModelVersionMismatchError, match="trained from"):
        _load(path, info)


def test_a_spec_hash_that_is_not_the_frozen_one_fails_clearly(tmp_path: Path) -> None:
    path, info = frozen_ensemble(tmp_path)
    spec = load_frozen_ensemble_spec(path)
    spec["constituents"][0]["spec_hash"] = "0" * 32       # frozen against another version
    spec["spec_id"] = spec["spec_id"].replace("V001", "V002")
    for key in ("frozen", "content_hash", "frozen_utc", "ENSEMBLE_SPEC_FROZEN"):
        spec.pop(key, None)
    other = freeze_ensemble_spec(spec, path.parent)
    with pytest.raises(ModelVersionMismatchError, match="no silent substitution"):
        _load(other, info)


def test_an_artifact_that_leaves_a_file_unhashed_is_refused(tmp_path: Path) -> None:
    path, info = frozen_ensemble(tmp_path)
    man_path = info["models"] / info["members"][0]["model_id"] / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    man["files"].pop("preprocessor.json")                 # the file would go unchecked
    man_path.write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(ModelVersionMismatchError, match="does not hash"):
        _load(path, info)


def test_a_feature_manifest_mismatch_fails_clearly(tmp_path: Path) -> None:
    path, info = frozen_ensemble(tmp_path)
    good = live_manifest(info["members"])
    _load(path, info, manifests=good)                     # loads
    for bad in ({"FS_SYNTH": {"hash": "hash-other", "features": NAMES}},
                {"FS_SYNTH": {"hash": "hash-synth", "features": [*NAMES[1:], NAMES[0]]}},
                {}):
        with pytest.raises(FeatureVersionMismatchError):
            _load(path, info, manifests=bad)


def test_missing_constituents_or_inputs_are_ensemble_invalid(tmp_path: Path) -> None:
    path, info = frozen_ensemble(tmp_path)
    model = _load(path, info)
    x = _x()
    x.pop("log_rv_20")
    with pytest.raises(EnsembleInvalidError, match="unavailable"):
        model.predict_batch(x)
    # a constituent that cannot predict a row: the row is invalid, not re-weighted
    member = model.constituents[0]
    original = member.predict

    def broken(xx: np.ndarray) -> np.ndarray:
        out = original(xx)
        out[::3] = np.nan
        return out

    member.predict = broken                               # type: ignore[method-assign]
    res = model.predict_batch(_x())
    assert (res["status"][::3] == "ENSEMBLE_INVALID").all()
    assert np.isnan(res["prediction"][::3]).all() and np.isfinite(res["prediction"][1::3]).all()
    with pytest.raises(EnsembleInvalidError):
        model.predict({f: float(v[0]) for f, v in _x().items()})
    shutil.rmtree(info["models"] / info["members"][0]["model_id"])
    with pytest.raises(EnsembleInvalidError, match="no artifact"):
        _load(path, info)


def test_a_retained_single_model_is_a_one_member_combination(tmp_path: Path) -> None:
    """When no ensemble earns its complexity the rule's single model is frozen; the
    other constituents stay loaded for the companions (simple average) only."""
    path, info = frozen_ensemble(tmp_path, method="single")
    model = _load(path, info)
    res = model.predict_batch(_x())
    np.testing.assert_array_equal(res["prediction"], res["constituents"][:, 1])
    assert model.describe()["method"] == "single"


def test_state_weights_need_their_context(tmp_path: Path) -> None:
    path, info = frozen_ensemble(tmp_path, method="regime")
    model = _load(path, info)
    with pytest.raises(EnsembleInvalidError, match="state inputs"):
        model.predict_batch(_x())
    n = ROWS.size
    ctx = {"regime_p0": np.ones(n), "regime_p1": np.zeros(n)}
    res = model.predict_batch(_x(), ctx)
    np.testing.assert_allclose(res["prediction"], res["constituents"] @ np.array([0.9, 0.1]))
    ctx_nan = {"regime_p0": np.full(n, np.nan), "regime_p1": np.full(n, np.nan)}
    res2 = model.predict_batch(_x(), ctx_nan)             # no state -> the frozen fallback
    np.testing.assert_allclose(res2["prediction"], res2["constituents"].mean(axis=1))
    assert json.loads(path.read_text(encoding="utf-8"))["combination"]["kind"] == "state_weights"
