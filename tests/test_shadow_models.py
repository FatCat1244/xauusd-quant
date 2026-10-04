"""Existing frozen artifact reload and rejecting altered/missing live inputs."""

import json
from pathlib import Path

import numpy as np
import pytest

from xauusd_quant.execution.readiness import sha256
from xauusd_quant.ml.calibration import Calibrator
from xauusd_quant.ml.models import Model
from xauusd_quant.ml.preprocessing import Preprocessor
from xauusd_quant.ml.registry import freeze_spec, save_artifact
from xauusd_quant.shadow.models import FrozenEnsemble, FrozenModel


def artifact(root: Path) -> tuple[Path, Path]:
    # Fixed tiny earlier fitting set, unrelated to tick replay outcomes.
    x, y = np.array([[0.], [.1], [.2]]), np.array([.001, .002, .003])
    spec = {"spec_id": "CONST_RETURN_SYNTHETIC_V001", "family": "constant",
        "target": "future_return", "timeframe": "1m", "horizon": 1,
        "target_definition": {"source": "return", "task": "regression", "transform": "none"},
        "features": ["ret_1"], "feature_set_id": "SYNTHETIC_FEATURES_V001"}
    frozen = freeze_spec(spec, root / "specs")
    body = json.loads(frozen.read_text())
    directory = root / "artifacts"
    save_artifact(directory, model=Model("constant", "regression").fit(x, y),
        preprocessor=Preprocessor.fit(x, ("ret_1",), kind="linear"),
        calibrator=Calibrator("none"), spec=body, extra={})
    return frozen, directory


def test_frozen_predictor_reload_known_mean_and_missing_inputs(tmp_path: Path) -> None:
    spec, directory = artifact(tmp_path)
    model = FrozenModel.load(spec, directory, sha256(directory / "manifest.json"))
    assert model.predict({"ret_1": .05}) == pytest.approx(.002)
    assert model.model_id == "CONST_RETURN_SYNTHETIC_V001"
    with pytest.raises(ValueError, match="missing"):
        model.predict({"ret_1": float("nan")})
    with pytest.raises(KeyError):
        model.predict({})


def test_changed_manifest_and_spec_block_before_loading(tmp_path: Path) -> None:
    spec, directory = artifact(tmp_path)
    path = directory / "manifest.json"
    digest = sha256(path)
    body = json.loads(path.read_text())
    body["features"] = ["future_return"]
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="pinned"):
        FrozenModel.load(spec, directory, digest)
    with pytest.raises(ValueError, match="mismatch"):
        FrozenModel.load(spec, directory, sha256(path))


def test_ensemble_requires_pinned_spec_and_live_manifests(tmp_path: Path) -> None:
    file = tmp_path / "ensemble.json"
    file.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="pinned"):
        FrozenEnsemble.load(file, tmp_path, "invalid")
    with pytest.raises(ValueError, match="live feature"):
        FrozenEnsemble.load(file, tmp_path, sha256(file))
