r"""Ensemble ids, frozen ensemble specifications and constituent ids (Prompt #11,
Steps 59-61).

* **Ensemble id**: ``ENS_<TARGET>_<TF>_H<h>_V<nnn>`` (``ENS_REVERSION_5M_H5_V001``).
* **Constituent id**: ``<FAMILY>_<SET>_<TARGET>_<TF>_H<h>_V<nnn>``
  (``LGBM_EXT_REVERSION_5M_H5_V001``) - a Prompt #10 model spec (family, defaults,
  features, preprocessing, Platt calibration for probabilities, expanding
  training) frozen in the ensemble namespace, so it never collides with a Prompt
  #10 model id.
* **Frozen ensemble spec** (``ENSEMBLE_SPEC_<id>.json``): constituent model ids
  *with their spec hashes*, feature-set ids and hashes, target, horizon, the
  combination method and its fitted parameters (weights, meta-model,
  state weights, dynamic-weight rule), the final calibrator, the companions
  evaluated beside it (simple average, best individual, constant), training and
  refit policy, data versions, the development evidence and a content hash.
  Immutable: a different spec under an existing id raises - bump the version.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..features.manifest import digest
from ..ml.models import FAMILY_PREFIX
from ..ml.registry import TARGET_TAGS
from ..utils.clock import utc_now_iso
from ..utils.paths import atomic_write_text, ensure_dir

__all__ = ["EnsembleSpecError", "SET_TAGS", "constituent_id", "ensemble_id",
           "freeze_ensemble_spec", "load_frozen_ensemble_spec", "spec_path"]

SET_TAGS = {"standard": "STD", "extended": "EXT", "target": "TGT", "minimal": "MIN"}
_HASHED = ("spec_id", "timeframe", "target", "target_definition", "horizon", "task", "method",
           "role", "input_kind", "constituents", "combination", "calibration", "companions",
           "context_inputs", "training_policy", "refit_policy", "dataset_versions",
           "missing_constituent_policy")


class EnsembleSpecError(RuntimeError):
    """An ensemble spec is missing, not frozen, tampered with, or conflicting."""


def ensemble_id(target: str, timeframe: str, horizon: int, version: int = 1) -> str:
    return (f"ENS_{TARGET_TAGS.get(target, target.upper())}_{timeframe.upper()}_H{int(horizon)}"
            f"_V{int(version):03d}")


def constituent_id(family: str, feature_set: str, target: str, timeframe: str, horizon: int,
                   version: int = 1) -> str:
    return (f"{FAMILY_PREFIX[family]}_{SET_TAGS.get(feature_set, feature_set.upper())}_"
            f"{TARGET_TAGS.get(target, target.upper())}_{timeframe.upper()}_H{int(horizon)}_"
            f"V{int(version):03d}")


def spec_path(directory: Path, spec_id: str) -> Path:
    return directory / f"ENSEMBLE_SPEC_{spec_id}.json"


def _hash(spec: dict[str, Any]) -> str:
    return digest({k: spec.get(k) for k in _HASHED}, 16)


def freeze_ensemble_spec(spec: dict[str, Any], directory: Path) -> Path:
    """Write ``ENSEMBLE_SPEC_<id>.json`` once; same content is a no-op, other content refused."""
    body = dict(spec)
    body["frozen"] = True
    body["ENSEMBLE_SPEC_FROZEN"] = True
    body["content_hash"] = _hash(body)
    body.setdefault("frozen_utc", utc_now_iso())
    path = spec_path(ensure_dir(directory), body["spec_id"])
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("content_hash") != body["content_hash"]:
            raise EnsembleSpecError(f"{body['spec_id']} is already frozen with other content "
                                    f"({old.get('content_hash')} vs {body['content_hash']}) - "
                                    "never overwrite an ensemble version: bump it")
        return path
    atomic_write_text(path, json.dumps(body, indent=1, default=str) + "\n")
    return path


def load_frozen_ensemble_spec(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise EnsembleSpecError(f"no ensemble spec at {path}")
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("frozen") is not True:
        raise EnsembleSpecError(f"{path.name}: not frozen - the final test needs a frozen "
                                "ENSEMBLE_SPEC")
    if _hash(spec) != spec.get("content_hash"):
        raise EnsembleSpecError(f"{path.name}: content hash does not match (edited after "
                                "freezing?)")
    return dict(spec)
