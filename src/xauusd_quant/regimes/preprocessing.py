r"""Scaling and the missing-feature policy - fitted on training rows only.

A regime model never sees raw features. Each feature is centred and scaled by
a :class:`FittedScaler` estimated from the **training period alone**:

``robust``    :math:`x' = (x - \mathrm{median}) / \mathrm{IQR}`
``standard``  :math:`x' = (x - \mathrm{mean}) / \mathrm{sd}`

and optionally clipped to :math:`|x'| \le` ``clip``. A causal model's scaler is
fitted on its own training window and frozen with the model until the next
refit; nothing from the inference period reaches it. Offline (full-sample)
fits use a full-sample scaler and say so - their scaled values are
``offline_scaled`` and never features.

Missing features are never filled with zero. The policy
(``preprocessing.missing_policy``):

``marginalize``  score a bar on its observed features: the Gaussian log-density
                 of the observed coordinates is the exact marginal of the
                 model's density (see :mod:`.emissions`). Default.
``exclude``      score only bars with every feature observed.
``median_fill``  replace a missing value by the *training* median.

Whatever the policy, a bar is scored only if at least ``min_feature_coverage``
of its features and every ``required_features`` entry are observed; any other
bar is *unscored* (a hidden-Markov filter then only propagates its prior).
Models are always **fitted** on complete cases. Coverage and missingness are
recorded per bar, so whether missingness itself carries information can be
measured rather than assumed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "FittedScaler",
    "MissingPolicy",
    "clip_fractions",
    "coverage",
    "fit_scaler",
    "scorable_rows",
]


@dataclass(frozen=True)
class FittedScaler:
    """Per-feature centre and scale estimated on one training set, then frozen."""

    method: str
    features: tuple[str, ...]
    center: np.ndarray
    scale: np.ndarray
    clip: float | None
    rows: int
    training_start: str | None = None
    training_end: str | None = None
    medians: np.ndarray | None = None    # raw training medians (median_fill)

    @property
    def dimension(self) -> int:
        return len(self.features)

    def transform(self, values: np.ndarray, *, clip: bool = True) -> np.ndarray:
        """Scaled copy of *values* (rows x features); NaN stays NaN."""
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.dimension:
            raise ValueError(f"expected (rows, {self.dimension}) values, got {x.shape}")
        z = (x - self.center) / self.scale
        if clip and self.clip is not None:
            np.clip(z, -self.clip, self.clip, out=z)       # NaN passes through
        return z

    def inverse(self, scaled: np.ndarray) -> np.ndarray:
        return np.asarray(scaled, dtype=np.float64) * self.scale + self.center

    def means_to_raw(self, means: np.ndarray) -> np.ndarray:
        return np.asarray(means, dtype=np.float64) * self.scale + self.center

    def covariances_to_raw(self, covariances: np.ndarray) -> np.ndarray:
        s = self.scale
        return np.asarray(covariances, dtype=np.float64) * s[:, None] * s[None, :]

    def map_to(self, other: FittedScaler, means: np.ndarray,
               covariances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Gaussian parameters in this scaler's space re-expressed in *other*'s.

        Exact for an affine map (clipping aside): ``x = z s + c`` so
        ``z' = (z s + c - c') / s'``. Used to warm-start a refit from the
        previous period's model.
        """
        if other.features != self.features:
            raise ValueError("scalers describe different features")
        ratio = self.scale / other.scale
        new_means = (np.asarray(means) * self.scale + self.center - other.center) / other.scale
        new_cov = np.asarray(covariances) * ratio[:, None] * ratio[None, :]
        return new_means, new_cov

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method, "features": list(self.features),
            "center": [float(v) for v in self.center], "scale": [float(v) for v in self.scale],
            "clip": self.clip, "rows": int(self.rows), "training_start": self.training_start,
            "training_end": self.training_end,
            "medians": None if self.medians is None else [float(v) for v in self.medians],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FittedScaler:
        return cls(method=str(data["method"]), features=tuple(data["features"]),
                   center=np.asarray(data["center"], dtype=np.float64),
                   scale=np.asarray(data["scale"], dtype=np.float64),
                   clip=None if data.get("clip") is None else float(data["clip"]),
                   rows=int(data["rows"]), training_start=data.get("training_start"),
                   training_end=data.get("training_end"),
                   medians=(None if data.get("medians") is None
                            else np.asarray(data["medians"], dtype=np.float64)))

    def fingerprint(self) -> str:
        """The scaler version: a digest of exactly what it does to a value."""
        payload = {k: v for k, v in self.to_dict().items()
                   if k not in ("training_start", "training_end", "rows")}
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return f"scaler-{hashlib.blake2b(blob, digest_size=8).hexdigest()}"


def fit_scaler(values: np.ndarray, features: tuple[str, ...] | list[str], *, method: str,
               clip: float | None, rows: np.ndarray | None = None,
               training_start: str | None = None,
               training_end: str | None = None) -> FittedScaler:
    """Centre and scale of each column over the training *rows* (finite values only).

    A zero IQR (a discrete feature concentrated on one value) falls back to the
    standard deviation, and a zero standard deviation to 1, so a degenerate
    column is left unscaled rather than divided by zero.
    """
    x = np.asarray(values, dtype=np.float64)
    if rows is not None:
        x = x[rows]
    if x.ndim != 2 or x.shape[1] != len(features):
        raise ValueError("values and features disagree")
    if x.shape[0] < 2:
        raise ValueError("a scaler needs at least two training rows")
    center = np.empty(x.shape[1])
    scale = np.empty(x.shape[1])
    medians = np.empty(x.shape[1])
    for j in range(x.shape[1]):
        column = x[:, j]
        column = column[np.isfinite(column)]
        if column.size < 2:
            raise ValueError(f"feature {features[j]!r} has fewer than two finite training values")
        q25, q50, q75 = np.quantile(column, [0.25, 0.5, 0.75])
        medians[j] = q50
        sd = float(column.std())
        if method == "robust":
            center[j] = q50
            spread = q75 - q25
            scale[j] = spread if spread > 0 else (sd if sd > 0 else 1.0)
        elif method == "standard":
            center[j] = float(column.mean())
            scale[j] = sd if sd > 0 else 1.0
        else:
            raise ValueError(f"unknown scaling {method!r}")
    return FittedScaler(method=method, features=tuple(features), center=center, scale=scale,
                        clip=None if clip is None else float(clip), rows=int(x.shape[0]),
                        training_start=training_start, training_end=training_end,
                        medians=medians)


def coverage(values: np.ndarray) -> np.ndarray:
    """Share of finite features in each row."""
    x = np.asarray(values)
    return np.isfinite(x).mean(axis=1) if x.size else np.zeros(x.shape[0])


def scorable_rows(values: np.ndarray, features: tuple[str, ...] | list[str], *,
                  min_coverage: float, required: tuple[str, ...] | list[str]) -> np.ndarray:
    """Rows a model may score: enough coverage and every required feature observed."""
    x = np.asarray(values)
    finite = np.isfinite(x)
    ok = finite.mean(axis=1) >= min_coverage - 1e-12
    for name in required:
        if name in features:
            ok &= finite[:, list(features).index(name)]
    return ok


def clip_fractions(scaled_unclipped: np.ndarray, clip: float | None) -> np.ndarray:
    """Per feature, the share of finite scaled values beyond the clip."""
    z = np.asarray(scaled_unclipped)
    if clip is None:
        return np.zeros(z.shape[1])
    finite = np.isfinite(z)
    with np.errstate(invalid="ignore"):
        beyond = (np.abs(z) > clip) & finite
    counts = finite.sum(axis=0)
    return np.where(counts > 0, beyond.sum(axis=0) / np.maximum(counts, 1), np.nan)


@dataclass(frozen=True)
class MissingPolicy:
    """How bars with missing features are scored (never how models are fitted)."""

    policy: str
    min_coverage: float
    required: tuple[str, ...]

    def prepare(self, scaled: np.ndarray, features: tuple[str, ...],
                scaler: FittedScaler) -> tuple[np.ndarray, np.ndarray]:
        """(values to score, scorable mask). Unscorable rows keep their NaNs.

        ``median_fill`` puts the training median (in scaled units) into missing
        cells of scorable rows; the other policies leave NaN for the emission
        code to marginalise over (``marginalize``) or refuse (``exclude``).
        """
        z = np.array(scaled, dtype=np.float64, copy=True)
        finite = np.isfinite(z)
        if self.policy == "exclude":
            ok = finite.all(axis=1)
            return z, ok
        ok = scorable_rows(z, features, min_coverage=self.min_coverage, required=self.required)
        if self.policy == "median_fill":
            if scaler.medians is None:
                raise ValueError("median_fill needs the scaler's training medians")
            fill = (scaler.medians - scaler.center) / scaler.scale
            rows = ok & ~finite.all(axis=1)
            if rows.any():
                sub = z[rows]
                holes = ~np.isfinite(sub)
                sub[holes] = np.broadcast_to(fill, sub.shape)[holes]
                z[rows] = sub
        return z, ok
