r"""Fold-specific preprocessing, fitted on training rows only (Prompt #10, Steps 15-16, 73).

A :class:`Preprocessor` is fitted on one fold's fitting rows and then frozen:

* ``linear``      median imputation, a missing indicator per feature that was
                  missing in training, robust scaling (median / IQR, the
                  standard deviation where the IQR is zero) and a symmetric clip;
* ``tree_imputed`` median imputation and missing indicators, no scaling;
* ``tree_native``  nothing but the order check - the booster sees NaN itself.

Every call checks the feature order against the order it was fitted with and
refuses a mismatch (:class:`FeatureOrderError`), so a column swapped in a live
pipeline cannot be scored silently. The artifact serializes to JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["FeatureOrderError", "Preprocessor"]

KINDS = ("linear", "tree_imputed", "tree_native")


class FeatureOrderError(ValueError):
    """The columns given are not the columns (or not in the order) the artifact was fitted on."""


@dataclass
class Preprocessor:
    kind: str
    features: tuple[str, ...]
    medians: np.ndarray | None = None
    centers: np.ndarray | None = None
    scales: np.ndarray | None = None
    indicators: tuple[int, ...] = ()
    clip: float | None = None

    @classmethod
    def fit(cls, x: np.ndarray, features: list[str] | tuple[str, ...], *, kind: str,
            clip: float | None = 8.0) -> Preprocessor:
        """Fit on training rows *x* (n, p) only."""
        if kind not in KINDS:
            raise ValueError(f"unknown preprocessing kind {kind!r}")
        feats = tuple(features)
        if x.shape[1] != len(feats):
            raise FeatureOrderError(f"{x.shape[1]} columns for {len(feats)} feature names")
        if kind == "tree_native":
            return cls(kind=kind, features=feats)
        xs = np.asarray(x, dtype=np.float64)
        finite = np.isfinite(xs)
        medians = np.zeros(len(feats))
        for j in range(len(feats)):
            col = xs[finite[:, j], j]
            medians[j] = float(np.median(col)) if col.size else 0.0
        indicators = tuple(int(j) for j in range(len(feats)) if (~finite[:, j]).any())
        if kind == "tree_imputed":
            return cls(kind=kind, features=feats, medians=medians, indicators=indicators)
        filled = np.where(finite, xs, medians[None, :])
        centers = np.median(filled, axis=0)
        q75, q25 = np.percentile(filled, [75, 25], axis=0)
        scales = q75 - q25
        sd = filled.std(axis=0)
        scales = np.where(scales > 1e-12, scales, np.where(sd > 1e-12, sd, 1.0))
        return cls(kind=kind, features=feats, medians=medians, centers=centers, scales=scales,
                   indicators=indicators, clip=clip)

    def check(self, features: list[str] | tuple[str, ...]) -> None:
        if tuple(features) != self.features:
            missing = [f for f in self.features if f not in features]
            extra = [f for f in features if f not in self.features]
            raise FeatureOrderError(
                "feature order differs from the fitted artifact"
                + (f"; missing {missing}" if missing else "")
                + (f"; unexpected {extra}" if extra else "")
                + ("" if missing or extra else "; same names, different order"))

    def transform(self, x: np.ndarray, features: list[str] | tuple[str, ...]) -> np.ndarray:
        """The model design of rows *x*; deterministic, row by row (streaming-safe)."""
        self.check(features)
        if self.kind == "tree_native":
            return np.asarray(x, dtype=np.float32)
        xs = np.asarray(x, dtype=np.float64)
        missing = ~np.isfinite(xs)
        assert self.medians is not None
        out = np.where(missing, self.medians[None, :], xs)
        if self.kind == "linear":
            assert self.centers is not None and self.scales is not None
            out = (out - self.centers[None, :]) / self.scales[None, :]
            if self.clip is not None:
                out = np.clip(out, -self.clip, self.clip)
        if self.indicators:
            out = np.concatenate([out, missing[:, list(self.indicators)].astype(np.float64)],
                                 axis=1)
        return out.astype(np.float32 if self.kind == "tree_imputed" else np.float64)

    def output_names(self) -> list[str]:
        return [*self.features, *[f"missing__{self.features[j]}" for j in self.indicators]]

    def to_dict(self) -> dict[str, Any]:
        def arr(a: np.ndarray | None) -> list[float] | None:
            return None if a is None else [float(v) for v in a]

        return {"kind": self.kind, "features": list(self.features), "medians": arr(self.medians),
                "centers": arr(self.centers), "scales": arr(self.scales),
                "indicators": list(self.indicators), "clip": self.clip}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Preprocessor:
        def arr(v: Any) -> np.ndarray | None:
            return None if v is None else np.asarray(v, dtype=np.float64)

        return cls(kind=str(d["kind"]), features=tuple(d["features"]), medians=arr(d["medians"]),
                   centers=arr(d["centers"]), scales=arr(d["scales"]),
                   indicators=tuple(int(j) for j in d["indicators"]), clip=d.get("clip"))
