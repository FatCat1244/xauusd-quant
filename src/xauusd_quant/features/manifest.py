"""Schema guards and versions for the feature matrix and the target store (Steps 3, 7, 77).

Architectural safeguards, not conventions:

* :func:`assert_feature_schema` refuses any column that is not registered,
  any column the registry marks non-live-safe, and anything that looks like an
  outcome or an offline quantity (``target_``, ``fwd_``, ``future``,
  ``offline``, ``smooth``, ``viterbi``, ``label``) - whatever its registration.
* :func:`assert_target_schema` refuses a target table with any column that is
  not ``timestamp`` or ``target_*``.

Versions chain: a feature's version hashes its definition, its source versions
(bar dataset, regression store, stored engine fingerprints, regime model
version) and the code of its family; the matrix version hashes every feature
version it holds.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..research.study_io import code_fingerprint
from .registry import FeatureSpec

__all__ = [
    "FORBIDDEN_MARKERS",
    "FeatureSchemaError",
    "assert_feature_schema",
    "assert_target_schema",
    "digest",
    "family_code_version",
    "feature_version",
]

#: Substrings no feature column may contain, registered or not.
FORBIDDEN_MARKERS: tuple[str, ...] = ("target_", "fwd_", "future", "offline", "smooth",
                                      "viterbi", "label")
_FAMILY_CODE = {
    "returns": ("features/families.py",),
    "volatility": ("features/families.py", "regimes/dataset.py"),
    "autocorrelation": ("features/families.py",),
    "regression": ("features/families.py", "features/rolling_regression.py",
                   "features/store.py"),
    "ou": ("features/families.py", "models/ornstein_uhlenbeck.py"),
    "fft": ("features/families.py", "features/spectral.py", "features/spectral_bands.py",
            "features/spectral_entropy.py"),
    "wavelet": ("features/families.py", "features/wavelet.py", "features/wavelet_energy.py",
                "features/wavelet_entropy.py", "features/wavelet_causal.py"),
    "regime": ("features/joins.py",),
    "microstructure": ("features/families.py", "regimes/dataset.py"),
    "time": ("features/families.py", "research/intraday.py"),
    "interaction": ("features/interactions.py",),
}


class FeatureSchemaError(RuntimeError):
    """A column that must never be in a feature (or target) table."""


def assert_feature_schema(columns: Iterable[str], registry: dict[str, FeatureSpec], *,
                          live_safe_only: bool = True) -> None:
    """Refuse unregistered, non-live-safe or outcome-like columns in a feature matrix."""
    problems = []
    for name in columns:
        if name == "timestamp":
            continue
        lowered = name.lower()
        marker = next((m for m in FORBIDDEN_MARKERS if m in lowered), None)
        if marker is not None:
            problems.append(f"{name!r} contains {marker!r} (an outcome or offline quantity)")
            continue
        spec = registry.get(name)
        if spec is None:
            problems.append(f"{name!r} is not registered (no anonymous columns)")
        elif live_safe_only and not spec.live_safe:
            problems.append(f"{name!r} is registered live_safe = false")
    if problems:
        raise FeatureSchemaError("feature matrix refused: " + "; ".join(problems))


def assert_target_schema(columns: Iterable[str]) -> None:
    """A target table holds ``timestamp`` and ``target_*`` columns only."""
    bad = [c for c in columns if c != "timestamp" and not c.startswith("target_")]
    if bad:
        raise FeatureSchemaError(f"target table refused: non-target column(s) {bad}")


def digest(payload: Any, size: int = 8) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.blake2b(blob, digest_size=size).hexdigest()


def family_code_version(family: str) -> str:
    base = Path(__file__).resolve().parent.parent
    rels = _FAMILY_CODE.get(family, ("features/families.py",))
    return code_fingerprint(base / rel for rel in rels)


def feature_version(spec: FeatureSpec, *, sources: dict[str, Any], config_fingerprint: str,
                    code_version: str) -> str:
    """``feat-<timeframe>-<hash>`` of the definition, source versions and family code."""
    payload = {"id": spec.feature_id, "definition": spec.definition, "window": spec.window,
               "source_series": spec.source_series, "sources": sources,
               "config": config_fingerprint, "code": code_version}
    return f"feat-{spec.timeframe}-{digest(payload)}"
