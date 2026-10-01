"""Immutable, versioned regime-model artefacts (Step 56).

Every fitted model - each walk-forward refit, each offline fit - is stored once
under an ID such as ``HMM_5M_K4_00017``: family, timeframe, number of states
and a sequence number within that triple. A stored model is never overwritten.
Its content (parameters, scaler, alignment and training metadata) is hashed;
registering identical content again returns the existing ID, so reruns do not
inflate the sequence, and different content always gets a new number.

Layout::

    data/regime_models/
        HMM_5M_K4/
            _index.json          content hash -> model ID, next sequence number
            HMM_5M_K4_00017.json the artefact (parameters + scaler + metadata)

A model is restored with :func:`load_model`, which rebuilds the fitted object
and its scaler; the restored model gives the same probabilities (tested).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..research.study_io import clean_json
from ..utils.clock import utc_now_iso
from ..utils.paths import atomic_write_text, ensure_dir
from .clustering import KMeansModel
from .gmm import GMMModel
from .hmm import HMMModel
from .preprocessing import FittedScaler

__all__ = ["ModelRegistry", "build_model", "family_code", "load_model"]

_FAMILY = {"kmeans": "KMEANS", "gmm": "GMM", "gmm_diag": "GMMDIAG", "gmm_tied": "GMMTIED",
           "hmm": "HMM"}


def family_code(model: str) -> str:
    if model not in _FAMILY:
        raise KeyError(f"unknown model family {model!r}")
    return _FAMILY[model]


def build_model(payload: dict[str, Any]) -> KMeansModel | GMMModel | HMMModel:
    kind = payload.get("kind")
    if kind == "kmeans":
        return KMeansModel.from_dict(payload)
    if kind == "gmm":
        return GMMModel.from_dict(payload)
    if kind == "hmm":
        return HMMModel.from_dict(payload)
    raise ValueError(f"unknown model kind {kind!r}")


def _digest(payload: dict[str, Any]) -> str:
    blob = json.dumps(clean_json(payload), sort_keys=True, separators=(",", ":"),
                      default=str).encode()
    return hashlib.blake2b(blob, digest_size=12).hexdigest()


class ModelRegistry:
    """Append-only store of fitted regime models."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _group(self, model: str, timeframe: str, states: int) -> Path:
        return self.root / f"{family_code(model)}_{timeframe.upper()}_K{states}"

    def register(self, *, model: str, timeframe: str, states: int, fitted: Any,
                 scaler: FittedScaler | None, metadata: dict[str, Any]) -> str:
        """Store *fitted* (never overwriting) and return its model ID."""
        group = ensure_dir(self._group(model, timeframe, states))
        index_path = group / "_index.json"
        index: dict[str, Any] = (json.loads(index_path.read_text(encoding="utf-8"))
                                 if index_path.exists() else {"next": 1, "models": {}})
        content = {"parameters": fitted.to_dict(),
                   "scaler": scaler.to_dict() if scaler is not None else None,
                   "metadata": {k: v for k, v in metadata.items() if k != "registered_utc"}}
        digest = _digest(content)
        if digest in index["models"]:
            return str(index["models"][digest])
        number = int(index["next"])
        model_id = f"{group.name}_{number:05d}"
        path = group / f"{model_id}.json"
        if path.exists():                       # never overwrite: skip to a free number
            while path.exists():
                number += 1
                model_id = f"{group.name}_{number:05d}"
                path = group / f"{model_id}.json"
        artefact = {"model_id": model_id, "content_hash": digest, "family": model,
                    "timeframe": timeframe, "states": states, "registered_utc": utc_now_iso(),
                    **content}
        atomic_write_text(path, json.dumps(clean_json(artefact), indent=1, default=str) + "\n")
        index["models"][digest] = model_id
        index["next"] = number + 1
        atomic_write_text(index_path, json.dumps(index, indent=1) + "\n")
        return model_id

    def path(self, model_id: str) -> Path:
        group = model_id.rsplit("_", 1)[0]
        return self.root / group / f"{model_id}.json"

    def load(self, model_id: str) -> dict[str, Any]:
        return dict(json.loads(self.path(model_id).read_text(encoding="utf-8")))


def load_model(registry: ModelRegistry, model_id: str
               ) -> tuple[KMeansModel | GMMModel | HMMModel, FittedScaler | None, dict[str, Any]]:
    """The fitted model, its frozen scaler and its metadata."""
    artefact = registry.load(model_id)
    scaler = (FittedScaler.from_dict(artefact["scaler"]) if artefact.get("scaler") else None)
    return build_model(artefact["parameters"]), scaler, artefact.get("metadata", {})
