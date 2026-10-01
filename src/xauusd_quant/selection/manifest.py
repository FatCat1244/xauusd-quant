r"""Immutable, ordered feature-set manifests (Prompt #9, Steps 54-59, 68).

A manifest is what Prompt #10 consumes: an ordered list of registered
features with everything needed to rebuild the matrix exactly and to run it
live - feature id, name, family, dtype, transformation, window, warm-up
history, live-safe flag, source module, feature version, refresh frequency,
declared live cost - plus the set's provenance (dataset / bar / factory
versions, the selection and development periods, the reserved test period,
config fingerprints, git commit).

**Order** never comes from column discovery or from a selection ranking: it
is the configured family order, then the registry order within a family.
**Immutability**: a manifest is identified by ``FEATURESET_<TF>_<SET>_V<nnn>``
and a content hash over its ordered features and versions; writing a
different content under an existing id is refused (bump the version).
:func:`matrix_from_manifest` loads exactly those columns, in that order, from
the stored factory matrix and refuses a feature whose stored version differs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from ..features.factory import current_version_dir
from ..features.factory_config import FeatureFactoryConfig
from ..features.manifest import digest
from ..utils.paths import atomic_write_text

__all__ = [
    "ManifestConflictError",
    "build_manifest",
    "load_manifest",
    "matrix_from_manifest",
    "ordered_features",
    "refresh_frequency",
    "write_manifest",
]


class ManifestConflictError(RuntimeError):
    """A different feature set already holds this manifest id."""


def ordered_features(names: list[str], registry: dict[str, dict[str, Any]],
                     families_order: tuple[str, ...]) -> list[str]:
    """Deterministic order: configured family order, then registry order."""
    position = {n: i for i, n in enumerate(registry)}
    rank = {f: i for i, f in enumerate(families_order)}
    return sorted(set(names), key=lambda n: (rank.get(str(registry[n].get("family")), len(rank)),
                                             position.get(n, 1 << 30)))


def refresh_frequency(spec: dict[str, Any]) -> str:
    if spec.get("family") == "regime":
        return ("per bar (forward filter) with the walk-forward model refitted each quarter "
                "off the live path")
    if spec.get("family") == "time":
        return "per bar (clock only)"
    return "per bar (trailing window)"


def build_manifest(set_name: str, timeframe: str, features: list[str],
                   registry: dict[str, dict[str, Any]], *, version: int,
                   families_order: tuple[str, ...], provenance: dict[str, Any],
                   evidence: dict[str, dict[str, Any]] | None = None,
                   effective_rank: float | None = None, note: str = "") -> dict[str, Any]:
    ordered = ordered_features(features, registry, families_order)
    rows = []
    for pos, name in enumerate(ordered):
        spec = registry[name]
        rows.append({
            "position": pos, "feature_id": spec.get("feature_id"), "name": name,
            "family": spec.get("family"), "dtype": spec.get("dtype", "float32"),
            "transformation": "none - the stored factory value (causal by construction)",
            "window": spec.get("window"), "required_history": int(spec.get("min_history") or 0),
            "live_safe": bool(spec.get("live_safe", True)),
            "source_module": spec.get("source_module"), "source_series": spec.get("source_series"),
            "feature_version": spec.get("feature_version"),
            "refresh": refresh_frequency(spec), "declared_cost": spec.get("cost"),
            "incremental_update": spec.get("incremental_update"),
            "selection_evidence": (evidence or {}).get(name, {})})
    content = [{"name": r["name"], "feature_id": r["feature_id"],
                "feature_version": r["feature_version"]} for r in rows]
    set_id = f"FEATURESET_{timeframe.upper()}_{set_name.upper()}_V{version:03d}"
    return {
        "feature_set_id": set_id, "set": set_name, "timeframe": timeframe, "version": version,
        "content_hash": digest(content, 16), "features": rows, "feature_count": len(rows),
        "max_history_required": max((r["required_history"] for r in rows), default=0),
        "effective_rank": effective_rank, "note": note, **provenance,
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> bool:
    """Write once. Same content: no-op (False). Different content, same id: refused."""
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("feature_set_id") == manifest["feature_set_id"] and \
                old.get("content_hash") != manifest["content_hash"]:
            raise ManifestConflictError(
                f"{manifest['feature_set_id']} already exists with other features "
                f"({old.get('content_hash')} vs {manifest['content_hash']}) - bump the version")
        if old.get("content_hash") == manifest["content_hash"]:
            return False
    atomic_write_text(path, json.dumps(manifest, indent=1, default=str) + "\n")
    return True


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    content = [{"name": r["name"], "feature_id": r["feature_id"],
                "feature_version": r["feature_version"]} for r in manifest["features"]]
    if digest(content, 16) != manifest["content_hash"]:
        raise ManifestConflictError(f"{path}: content hash does not match its features")
    if [r["position"] for r in manifest["features"]] != list(range(len(manifest["features"]))):
        raise ManifestConflictError(f"{path}: positions are not 0..n-1 in order")
    return manifest


def matrix_from_manifest(manifest: dict[str, Any], fcfg: FeatureFactoryConfig, *,
                         start: Any = None, end: Any = None) -> pl.DataFrame:
    """The stored factory matrix restricted to the manifest's columns, in manifest order."""
    timeframe = manifest["timeframe"]
    base = current_version_dir(fcfg.factory_path, timeframe)
    registry = {r["name"]: r for r in json.loads((base / "registry.json").read_text(
        encoding="utf-8"))}
    names = [r["name"] for r in manifest["features"]]
    stale = [r["name"] for r in manifest["features"]
             if (registry.get(r["name"]) or {}).get("feature_version") != r["feature_version"]]
    if stale:
        raise ManifestConflictError(f"{manifest['feature_set_id']}: stored feature versions "
                                    f"differ for {stale} - rebuild or re-select")
    files = sorted(base.glob("year=*/part-0.parquet"))
    frame = pl.read_parquet(files, columns=["timestamp", *names]).sort("timestamp")
    if start is not None:
        frame = frame.filter(pl.col("timestamp") >= pl.lit(start).cast(frame["timestamp"].dtype))
    if end is not None:
        frame = frame.filter(pl.col("timestamp") < pl.lit(end).cast(frame["timestamp"].dtype))
    return frame.select("timestamp", *names)
