r"""Model ids, frozen model specifications and model artifacts (Prompt #10, Steps 71-75, 100).

* **Model id**: ``<FAMILY>_<TARGET>_<TF>_H<h>_V<nnn>`` (``XGB_REVERSION_5M_H5_V001``).
* **Frozen spec** (``MODEL_SPEC_<id>.json``): everything that defines a model -
  family, hyperparameters, target definition, horizon, ordered features and
  their manifest id / hash, preprocessing, calibration method, training-window
  policy, seed, data versions - plus the development evidence it was chosen
  on, ``"frozen": true`` and a content hash. Specs are immutable: a different
  spec under an existing id is refused (bump the version). ``xq ml-final-test``
  accepts nothing else.
* **Artifact** (``data/models/<id>/``): the fitted model in its native format,
  ``preprocessor.json``, ``calibrator.json`` and ``manifest.json`` (the
  registry entry: spec hash, training rows and dates, file hashes, versions).
  :func:`load_artifact` verifies the file hashes before returning anything.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..features.manifest import digest
from ..utils.clock import utc_now_iso
from ..utils.paths import atomic_write_text, ensure_dir
from .calibration import Calibrator
from .models import FAMILY_PREFIX, Model, load_model
from .preprocessing import Preprocessor

__all__ = [
    "SpecError",
    "TARGET_TAGS",
    "freeze_spec",
    "load_artifact",
    "load_frozen_spec",
    "model_id",
    "save_artifact",
]

TARGET_TAGS = {"mean_reversion": "REVERSION", "mean_reversion_half": "REVHALF",
               "residual_reduction": "RESIDRED", "future_return": "RETURN",
               "direction_up_cost": "UPCOST", "direction_down_cost": "DOWNCOST",
               "future_volatility": "VOL", "future_abs_move": "ABSMOVE"}
_HASHED = ("spec_id", "family", "target", "target_definition", "horizon", "timeframe",
           "feature_set", "feature_set_id", "feature_set_hash", "features", "params",
           "preprocessing", "calibration", "training_policy", "seed", "dataset_versions")


class SpecError(RuntimeError):
    """A model specification is missing, not frozen, tampered with, or conflicting."""


def model_id(family: str, target: str, timeframe: str, horizon: int, version: int = 1) -> str:
    return (f"{FAMILY_PREFIX[family]}_{TARGET_TAGS.get(target, target.upper())}_"
            f"{timeframe.upper()}_H{int(horizon)}_V{int(version):03d}")


def _spec_hash(spec: dict[str, Any]) -> str:
    return digest({k: spec.get(k) for k in _HASHED}, 16)


def freeze_spec(spec: dict[str, Any], directory: Path) -> Path:
    """Write ``MODEL_SPEC_<id>.json`` once; same content is a no-op, different content refused."""
    body = dict(spec)
    body["frozen"] = True
    body["content_hash"] = _spec_hash(body)
    body.setdefault("frozen_utc", utc_now_iso())
    path = ensure_dir(directory) / f"MODEL_SPEC_{body['spec_id']}.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("content_hash") != body["content_hash"]:
            raise SpecError(f"{body['spec_id']} is already frozen with other content "
                            f"({old.get('content_hash')} vs {body['content_hash']}) - bump the "
                            "version")
        return path
    atomic_write_text(path, json.dumps(body, indent=1, default=str) + "\n")
    return path


def load_frozen_spec(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SpecError(f"no model spec at {path}")
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("frozen") is not True:
        raise SpecError(f"{path.name}: not frozen - the final test needs MODEL_SPEC_FROZEN")
    if _spec_hash(spec) != spec.get("content_hash"):
        raise SpecError(f"{path.name}: content hash does not match (edited after freezing?)")
    return dict(spec)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_artifact(directory: Path, *, model: Model, preprocessor: Preprocessor,
                  calibrator: Calibrator, spec: dict[str, Any], extra: dict[str, Any]) -> Path:
    """Write the model, its preprocessing and calibration, and the registry entry."""
    directory = ensure_dir(directory)
    model_file = model.save(directory)
    atomic_write_text(directory / "preprocessor.json",
                      json.dumps(preprocessor.to_dict(), indent=1) + "\n")
    atomic_write_text(directory / "calibrator.json",
                      json.dumps(calibrator.to_dict(), indent=1) + "\n")
    files = {p.name: _sha256(p) for p in (model_file, directory / "model_meta.json",
                                          directory / "preprocessor.json",
                                          directory / "calibrator.json")}
    manifest = {"model_id": spec["spec_id"], "spec_hash": spec.get("content_hash"),
                "family": model.family, "task": model.task, "target": spec["target"],
                "horizon": spec["horizon"], "timeframe": spec["timeframe"],
                "feature_set_id": spec["feature_set_id"], "features": spec["features"],
                "hyperparameters": model.params, "seed": model.seed,
                "preprocessing": preprocessor.kind, "calibration": calibrator.method,
                "dataset_versions": spec.get("dataset_versions"),
                "git_commit": spec.get("git_commit"), "files": files,
                "model_size_bytes": int(model_file.stat().st_size), "created_utc": utc_now_iso(),
                **extra}
    atomic_write_text(directory / "manifest.json", json.dumps(manifest, indent=1, default=str)
                      + "\n")
    return directory


def load_artifact(directory: Path) -> tuple[Model, Preprocessor, Calibrator, dict[str, Any]]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        if _sha256(directory / name) != expected:
            raise SpecError(f"{directory.name}/{name}: file hash differs from the registry entry")
    model = load_model(directory)
    pre = Preprocessor.from_dict(json.loads((directory / "preprocessor.json").read_text(
        encoding="utf-8")))
    cal = Calibrator.from_dict(json.loads((directory / "calibrator.json").read_text(
        encoding="utf-8")))
    return model, pre, cal, manifest
