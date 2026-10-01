r"""Hyperparameter trials, neighbourhoods and complexity grids (Prompt #10, Steps 44-47).

Every trial is a research experiment: it is fitted on the same walk-forward
blocks as everything else (never the reserved period), recorded as a unit and
counted in the ledger. The search is deliberately small - random draws from
the grids of ``config/ml.yaml`` with a fixed seed - because a large search
mostly buys selection bias (Critical Rule 6).

* :func:`search_trials` - the configured number of random draws per family;
* :func:`neighbour_trials` - one grid step down and up in each parameter
  around a chosen trial, so a winner can be checked for a stable neighbourhood;
* :func:`complexity_trials` - one parameter varied alone (LightGBM
  ``num_leaves``), the complexity curve.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import MLConfig

__all__ = ["complexity_trials", "neighbour_trials", "search_trials"]

_SEED_OFFSET = {"lightgbm": 17, "xgboost": 29}


def search_trials(cfg: MLConfig, family: str) -> list[dict[str, Any]]:
    """The randomized trials of *family* (deterministic from the configured seed)."""
    grid = cfg.search.get(family) or {}
    count = int((cfg.search.get("trials") or {}).get(family, 0))
    rng = np.random.default_rng(cfg.random_seed + _SEED_OFFSET.get(family, 41))
    return [{k: v[int(rng.integers(len(v)))] for k, v in grid.items()} for _ in range(count)]


def neighbour_trials(params: dict[str, Any], grid: dict[str, list[Any]],
                     names: list[str]) -> list[tuple[str, dict[str, Any]]]:
    """``(label, trial)`` one grid step down / up in each of *names* around *params*."""
    out = []
    for name in names:
        values = list(grid.get(name, []))
        if params.get(name) not in values:
            continue
        i = values.index(params[name])
        for step, j in (("down", i - 1), ("up", i + 1)):
            if 0 <= j < len(values):
                out.append((f"{name}-{step}", {**params, name: values[j]}))
    return out


def complexity_trials(cfg: MLConfig) -> list[tuple[int, dict[str, Any]]]:
    """LightGBM with ``num_leaves`` varied alone (other parameters at their defaults)."""
    base = {k: v for k, v in (cfg.models.get("lightgbm") or {}).items() if k != "enabled"}
    return [(int(leaves), {**base, "num_leaves": int(leaves), "max_depth": -1})
            for leaves in (cfg.search.get("complexity") or {}).get("num_leaves", [])]
