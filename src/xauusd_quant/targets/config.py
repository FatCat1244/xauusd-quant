"""Typed configuration of the target factory (Prompt #8, ``config/targets.yaml``)."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..utils.config import ConfigError, _as_dict, _expand
from ..utils.paths import find_project_root, resolve_path

__all__ = [
    "TARGET_FAMILIES",
    "TargetConfig",
    "default_targets_config_path",
    "load_targets_config",
]

DEFAULT_TARGETS_RELPATH = "config/targets.yaml"
TARGETS_CONFIG_ENV_VAR = "XAUUSD_TARGETS_CONFIG"

#: Every outcome family the factory can build, and what kind of outcome it is.
TARGET_FAMILIES: dict[str, str] = {
    "return": "direction",
    "abs_return": "magnitude",
    "realized_vol": "volatility",
    "residual_change": "residual",
    "residual_reduction": "residual",
    "residual_shrinks": "residual",
    "ou_deviation_reduction": "residual",
    "max_up": "excursion",
    "max_down": "excursion",
}


@dataclass(frozen=True, slots=True)
class TargetConfig:
    schema_version: str
    horizons: tuple[int, ...]
    families: tuple[str, ...]
    regression_window: int
    residual_scale: str
    ou_window: int
    weekend_gap_hours: float
    targets_path: Path
    project_root: Path
    config_path: Path

    def columns(self, family: str | None = None) -> list[str]:
        """Target column names, ``target_<family>_<h>``, in family then horizon order."""
        fams = self.families if family is None else (family,)
        return [f"target_{f}_{h}" for f in fams for h in self.horizons]

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = _as_dict(self)
        return out

    def fingerprint(self) -> str:
        payload = self.to_dict()
        for key in ("project_root", "config_path", "targets_path"):
            payload.pop(key, None)
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        return hashlib.blake2b(blob, digest_size=8).hexdigest()


def default_targets_config_path(root: Path | None = None) -> Path:
    root = root or find_project_root()
    return resolve_path(os.environ.get(TARGETS_CONFIG_ENV_VAR) or DEFAULT_TARGETS_RELPATH, root)


_TOP = ("schema_version", "horizons", "families", "residual", "ou", "weekend_gap_hours",
        "storage")


def load_targets_config(path: str | os.PathLike[str] | None = None, *,
                        root: Path | None = None) -> TargetConfig:
    root = root or find_project_root()
    cfg_path = resolve_path(path, root) if path is not None else default_targets_config_path(root)
    if not cfg_path.exists():
        raise ConfigError(f"Target configuration file not found: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{cfg_path}: top level must be a mapping")
    data: dict[str, Any] = _expand(raw)
    unknown = sorted(set(data) - set(_TOP))
    if unknown:
        raise ConfigError(f"{cfg_path}: unknown top-level key(s) {unknown}")
    horizons = tuple(sorted({int(h) for h in data.get("horizons") or ()}))
    if not horizons or horizons[0] < 1:
        raise ConfigError("horizons: at least one, each >= 1")
    families = tuple(str(f) for f in data.get("families") or ())
    bad = [f for f in families if f not in TARGET_FAMILIES]
    if bad or not families:
        raise ConfigError(f"families: unknown {bad}; expected from {list(TARGET_FAMILIES)}")
    residual = data.get("residual") or {}
    scale = str(residual.get("scale", "trailing_volatility"))
    if scale != "trailing_volatility":
        raise ConfigError("residual.scale: only trailing_volatility (sigma_t, known at t)")
    storage = data.get("storage") or {}
    return TargetConfig(
        schema_version=str(data.get("schema_version", "1.0.0")),
        horizons=horizons,
        families=families,
        regression_window=int(residual.get("regression_window", 128)),
        residual_scale=scale,
        ou_window=int((data.get("ou") or {}).get("window", 256)),
        weekend_gap_hours=float(data.get("weekend_gap_hours", 24.0)),
        targets_path=resolve_path(storage.get("targets_path") or "data/targets", root),
        project_root=root,
        config_path=cfg_path,
    )
