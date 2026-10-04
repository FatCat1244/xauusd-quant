"""Existing frozen predictive contracts; no live fitting or eligibility promotion."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..execution.readiness import sha256
from ..ml.registry import load_artifact, load_frozen_spec


class Predictor(Protocol):
    model_id: str
    feature_id: str
    features: tuple[str, ...]
    target: str
    units: str
    timeframe: str
    horizon: int
    identity: str
    def predict(self, values: Mapping[str, float]) -> float: ...


@dataclass
class FrozenModel:
    model_id: str
    feature_id: str
    features: tuple[str, ...]
    target: str
    units: str
    timeframe: str
    horizon: int
    identity: str
    model: Any
    preprocessor: Any
    calibrator: Any

    @classmethod
    def load(cls, spec_path: Path, directory: Path, manifest_sha256: str) -> FrozenModel:
        if sha256(directory / "manifest.json") != manifest_sha256:
            raise ValueError("pinned model manifest changed")
        spec = load_frozen_spec(spec_path)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("model_id"), manifest.get("spec_hash"), manifest.get("features")) != (
            spec["spec_id"], spec["content_hash"], spec["features"]
        ):
            raise ValueError("frozen model and fitted artifact mismatch")
        # Validate names before existing loader opens native model/pickle files.
        if any(Path(name).name != name for name in manifest["files"]):
            raise ValueError("artifact file must stay inside its directory")
        model, pre, cal, _ = load_artifact(directory)
        pre.check(tuple(spec["features"]))
        return cls(spec["spec_id"], spec["feature_set_id"], tuple(spec["features"]),
                   spec["target"], forecast_units(spec), spec["timeframe"], int(spec["horizon"]),
                   manifest_sha256, model, pre, cal)

    def predict(self, values: Mapping[str, float]) -> float:
        x = np.asarray([[values[n] for n in self.features]], dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError("missing live features are not silently imputed")
        return float(self.calibrator.apply(self.model.predict(
            self.preprocessor.transform(x, self.features)))[0])


class FrozenEnsemble:
    """Use the existing hash-verified ensemble; unsupported context is explicit."""

    def __init__(self, ensemble: Any, identity: str) -> None:
        self.ensemble = ensemble
        self.model_id = ensemble.spec_id
        self.feature_id = str(ensemble.spec.get("feature_manifest_id", ""))
        self.features = tuple(ensemble.required_features())
        self.target, self.timeframe = ensemble.spec["target"], ensemble.spec["timeframe"]
        self.units = forecast_units(ensemble.spec)
        self.horizon, self.identity = int(ensemble.spec["horizon"]), identity
        if ensemble.required_context():
            raise ValueError("ensemble requires a compatible live context/health service")

    @classmethod
    def load(cls, spec_path: Path, models_path: Path, manifest_sha256: str, *,
             constituent_dir: Path | None = None,
             live_manifests: dict[str, dict[str, Any]] | None = None) -> FrozenEnsemble:
        from ..ensemble.inference import EnsembleModel
        if sha256(spec_path) != manifest_sha256:
            raise ValueError("pinned ensemble specification changed")
        if live_manifests is None:
            raise ValueError("hash-verified compatible live feature manifests required")
        model = EnsembleModel.load(spec_path, models_path=models_path,
                                   constituent_dir=constituent_dir, live_manifests=live_manifests)
        return cls(model, manifest_sha256)

    def predict(self, values: Mapping[str, float]) -> float:
        if not all(np.isfinite(values[n]) for n in self.features):
            raise ValueError("all frozen ensemble features must be available")
        return float(self.ensemble.predict(values))


def forecast_units(spec: dict[str, Any]) -> str:
    definition = spec.get("target_definition") or {}
    if (spec.get("target") == "future_return" and definition.get("source") == "return"
        and definition.get("task") == "regression"):
        if definition.get("transform") == "none":
            return "log_mid_return"
        if definition.get("transform") == "vol_scaled":
            return "volatility_scaled_return"
    return "unsupported_or_unknown_live_units"
