r"""One frozen ensemble behind one interface (Prompt #11, Steps 66-70).

:meth:`EnsembleModel.load` reads a frozen ``ENSEMBLE_SPEC`` (hash-verified), and
for every constituent its frozen ``MODEL_SPEC`` and its artifact (file hashes
verified) - and refuses, loudly:

* a constituent spec whose content hash is not the one the ensemble was frozen
  with, or an artifact trained from another spec version
  (:class:`ModelVersionMismatchError`, Step 69 - ``MODEL_XGB_V4`` expected,
  ``MODEL_XGB_V5`` loaded fails);
* a constituent whose feature manifest (id, hash and ordered names) differs from
  the live manifest it is given (:class:`FeatureVersionMismatchError`, Step 70);
* a missing artifact (:class:`EnsembleInvalidError`).

Prediction (:meth:`EnsembleModel.predict_batch`, :meth:`predict`,
:meth:`predict_proba`): each constituent's frozen preprocessing, model and
calibrator; the frozen combination (average, median, weights, stacked
meta-model, state weights, trailing weights); the frozen final calibrator. A
missing feature or a constituent without a finite prediction makes the row
``ENSEMBLE_INVALID`` (Step 68: fail closed - no reduced ensemble, no
renormalised weights, unless such a fallback is validated and frozen; none is).
Outputs are a probability or an expected value and the constituents' spread -
never a decision.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..ml.calibration import Calibrator
from ..ml.models import Model
from ..ml.preprocessing import Preprocessor
from ..ml.registry import load_artifact, load_frozen_spec
from .averaging import median_combination, weighted_mean
from .meta_model import MetaModel
from .registry import EnsembleSpecError, load_frozen_ensemble_spec
from .weighting import apply_state_weights

__all__ = ["Constituent", "EnsembleInvalidError", "EnsembleModel", "FeatureVersionMismatchError",
           "ModelVersionMismatchError", "current_manifests"]

INVALID = "ENSEMBLE_INVALID"


class EnsembleInvalidError(RuntimeError):
    """A required constituent (or its input) is unavailable: ``ENSEMBLE_INVALID``."""

    status = INVALID


class ModelVersionMismatchError(EnsembleSpecError):
    """A constituent spec or artifact is not the version the ensemble was frozen with."""


class FeatureVersionMismatchError(EnsembleSpecError):
    """A constituent's feature manifest differs from the live feature manifest."""


def current_manifests(manifest_dir: Path) -> dict[str, dict[str, Any]]:
    """The Prompt #9 manifests of one timeframe, keyed by feature-set id."""
    from ..selection.manifest import load_manifest

    out = {}
    for p in sorted(manifest_dir.glob("*.json")):
        m = load_manifest(p)
        out[m["feature_set_id"]] = {"hash": m["content_hash"],
                                    "features": [f["name"] for f in m["features"]]}
    return out


@dataclass
class Constituent:
    model_id: str
    spec: dict[str, Any]
    model: Model
    pre: Preprocessor
    cal: Calibrator
    features: list[str]
    manifest: dict[str, Any]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Calibrated probability (classification) or expected value, one per row of *x*."""
        return np.asarray(self.cal.apply(self.model.predict(self.pre.transform(x, self.features))),
                          dtype=np.float64)


class EnsembleModel:
    def __init__(self, spec: dict[str, Any], constituents: list[Constituent]) -> None:
        self.spec = spec
        self.constituents = constituents
        comb = dict(spec["combination"])
        self.kind = str(comb["kind"])
        self.combination = comb
        self.meta = MetaModel.from_dict(comb["meta_model"]) if self.kind == "stacking" else None
        cal = spec.get("calibration") or {"method": "none"}
        self.calibrator = Calibrator.from_dict(cal)
        self.task = str(spec["task"])

    @property
    def spec_id(self) -> str:
        return str(self.spec["spec_id"])

    @property
    def names(self) -> list[str]:
        return [c["name"] for c in self.spec["constituents"]]

    # -- loading -------------------------------------------------------------------
    @classmethod
    def load(cls, spec_path: Path, *, models_path: Path, constituent_dir: Path | None = None,
             live_manifests: dict[str, dict[str, Any]] | None = None) -> EnsembleModel:
        spec = load_frozen_ensemble_spec(spec_path)
        cdir = constituent_dir or spec_path.parent.parent / "constituents"
        loaded = []
        for c in spec["constituents"]:
            mid = c["model_id"]
            cspec = load_frozen_spec(cdir / f"MODEL_SPEC_{mid}.json")
            if cspec.get("content_hash") != c["spec_hash"]:
                raise ModelVersionMismatchError(
                    f"{spec['spec_id']}: constituent {mid} is frozen as {c['spec_hash']}, the spec "
                    f"on disk is {cspec.get('content_hash')} - no silent substitution")
            art = models_path / mid
            if not (art / "manifest.json").exists():
                raise EnsembleInvalidError(f"{spec['spec_id']}: constituent {mid} has no artifact "
                                      f"under {art} ({INVALID}) - run `xq ensemble-finalize`")
            # every prediction-bearing file must be in the hashed inventory - a manifest that
            # left one out would leave it unchecked
            meta = json.loads((art / "model_meta.json").read_text(encoding="utf-8"))
            listed = set(json.loads((art / "manifest.json").read_text(encoding="utf-8"))
                         .get("files") or {})
            missing = {str(meta.get("file")), "model_meta.json", "preprocessor.json",
                       "calibrator.json"} - listed
            if missing:
                raise ModelVersionMismatchError(f"{spec['spec_id']}: artifact {mid} does not hash "
                                                f"{sorted(missing)}")
            model, pre, cal, man = load_artifact(art)          # file hashes verified
            if man.get("model_id") != mid or man.get("spec_hash") != c["spec_hash"]:
                raise ModelVersionMismatchError(
                    f"{spec['spec_id']}: artifact {art.name} was trained from "
                    f"{man.get('model_id')} / {man.get('spec_hash')}, expected {mid} / "
                    f"{c['spec_hash']}")
            features = list(cspec["features"])
            if features != list(c["features"]):
                raise FeatureVersionMismatchError(f"{mid}: constituent spec and ensemble spec list "
                                             "different features")
            if live_manifests is not None:
                live = live_manifests.get(c["feature_set_id"])
                if live is None or live["hash"] != c["feature_set_hash"] \
                        or list(live["features"]) != features:
                    raise FeatureVersionMismatchError(
                        f"{mid}: expects feature manifest {c['feature_set_id']} "
                        f"({c['feature_set_hash']}), the live manifest is "
                        f"{None if live is None else live['hash']}")
            loaded.append(Constituent(model_id=mid, spec=cspec, model=model, pre=pre, cal=cal,
                                      features=features, manifest=man))
        return cls(spec, loaded)

    # -- prediction ----------------------------------------------------------------
    def required_features(self) -> list[str]:
        out: list[str] = []
        for c in self.constituents:
            out += [f for f in c.features if f not in out]
        return out

    def required_context(self) -> list[str]:
        if self.kind == "state_weights":
            return list(self.combination["state_inputs"])
        if self.kind == "dynamic":
            return ["dynamic_weights"]
        if self.kind == "stacking" and self.meta is not None:
            return [n.removeprefix("context:") for n in self.meta.inputs[self.meta.constituents:]]
        return []

    def constituent_matrix(self, x: Mapping[str, np.ndarray]) -> np.ndarray:
        missing = [f for f in self.required_features() if f not in x]
        if missing:
            raise EnsembleInvalidError(f"{self.spec_id}: features {missing} unavailable ({INVALID})")
        n = len(np.asarray(x[self.required_features()[0]]).reshape(-1))
        cols = []
        for c in self.constituents:
            xc = np.column_stack([np.asarray(x[f], dtype=np.float32).reshape(-1)
                                  for f in c.features]).astype(np.float32)
            if xc.shape[0] != n:
                raise EnsembleInvalidError(f"{self.spec_id}: feature columns of different lengths")
            cols.append(c.predict(xc))
        return np.column_stack(cols)

    def combine(self, p: np.ndarray, context: Mapping[str, Any] | None = None) -> np.ndarray:
        comb = self.combination
        ctx = context or {}
        if self.kind == "single":                      # the rule retained one model
            out = np.asarray(p[:, self.names.index(str(comb["member"]))], dtype=np.float64)
        elif self.kind == "simple_average":
            out = weighted_mean(p)
        elif self.kind == "median":
            out = median_combination(p)
        elif self.kind == "weights":
            out = weighted_mean(p, np.asarray(comb["weights"], dtype=np.float64))
        elif self.kind == "stacking":
            assert self.meta is not None
            names = self.required_context()
            if any(n not in ctx for n in names):
                raise EnsembleInvalidError(f"{self.spec_id}: context {names} unavailable ({INVALID})")
            extra = [np.asarray(ctx[n], dtype=np.float64).reshape(-1) for n in names]
            out = self.meta.predict(np.column_stack([p, *extra]) if extra else p)
        elif self.kind == "state_weights":
            names = list(comb["state_inputs"])
            if any(n not in ctx for n in names):
                raise EnsembleInvalidError(f"{self.spec_id}: state inputs {names} unavailable "
                                      f"({INVALID})")
            if comb.get("states") == "volatility":
                q = np.asarray(ctx[names[0]], dtype=np.float64).reshape(-1)
                nb = len(comb["weights"])
                probs = np.full((q.size, nb), np.nan)
                ok = np.isfinite(q) & (q >= 0)
                probs[ok] = 0.0
                probs[np.flatnonzero(ok), q[ok].astype(np.int64)] = 1.0
            else:
                probs = np.column_stack([np.asarray(ctx[n], dtype=np.float64).reshape(-1)
                                         for n in names])
            rw = apply_state_weights(probs, np.asarray(comb["weights"], dtype=np.float64),
                                     np.asarray(comb["fallback"], dtype=np.float64))
            out = weighted_mean(p, rw)
        elif self.kind == "dynamic":
            if "dynamic_weights" not in ctx:
                raise EnsembleInvalidError(f"{self.spec_id}: trailing weights unavailable - the "
                                      f"model-health service is required ({INVALID})")
            out = weighted_mean(p, np.asarray(ctx["dynamic_weights"], dtype=np.float64))
        else:                                              # pragma: no cover - spec-checked
            raise EnsembleSpecError(f"unknown combination {self.kind!r}")
        return np.asarray(self.calibrator.apply(out), dtype=np.float64)

    def predict_batch(self, x: Mapping[str, np.ndarray],
                      context: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Every row: the ensemble output, the constituents' outputs and their spread; rows
        where any constituent has no finite output are ``ENSEMBLE_INVALID`` (NaN)."""
        p = self.constituent_matrix(x)
        valid = np.isfinite(p).all(axis=1)
        out = self.combine(p, context)
        out = np.where(valid & np.isfinite(out), out, np.nan)
        spread = np.where(valid, p.std(axis=1), np.nan) if p.shape[1] > 1 else np.zeros(len(out))
        return {"prediction": out, "constituents": p, "disagreement": spread,
                "status": np.where(np.isfinite(out), "ok", INVALID)}

    def _one(self, feature_vector: Mapping[str, float], context: Mapping[str, Any] | None
             ) -> float:
        x = {k: np.array([v], dtype=np.float64) for k, v in feature_vector.items()}
        ctx: dict[str, np.ndarray] | None = None
        if context is not None:
            ctx = {}
            for k, v in context.items():
                a = np.asarray(v, dtype=np.float64)
                ctx[k] = a.reshape(1, -1) if k == "dynamic_weights" else a.reshape(1)
        res = self.predict_batch(x, ctx)
        value = float(res["prediction"][0])
        if not np.isfinite(value):
            raise EnsembleInvalidError(f"{self.spec_id}: {INVALID} - a constituent prediction is "
                                  "missing")
        return value

    def predict(self, feature_vector: Mapping[str, float],
                context: Mapping[str, Any] | None = None) -> float:
        """The ensemble's output for one bar: an expected value (regression) or a probability
        (classification). Raises :class:`EnsembleInvalidError` instead of guessing."""
        return self._one(feature_vector, context)

    def predict_proba(self, feature_vector: Mapping[str, float],
                      context: Mapping[str, Any] | None = None) -> float:
        if self.task != "classification":
            raise TypeError(f"{self.spec_id} predicts an expected value, not a probability")
        return self._one(feature_vector, context)

    def describe(self) -> dict[str, Any]:
        return {"spec_id": self.spec_id, "method": self.spec.get("method"), "task": self.task,
                "constituents": [c.model_id for c in self.constituents],
                "features": len(self.required_features()), "context": self.required_context(),
                "calibration": self.calibrator.method}


def load_json(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return out
