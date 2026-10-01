"""Hyperparameter trials, neighbourhoods and the complexity grid (Prompt #10, Steps 44-47).

Trials are drawn from the configured grids with a fixed seed, so a rerun plans
exactly the same units (and finds them cached); neighbourhoods step one grid
value down and up; the complexity curve varies ``num_leaves`` alone.
"""

from __future__ import annotations

import numpy as np

from ml_synth import small_config
from xauusd_quant.ml.hyperparameters import complexity_trials, neighbour_trials, search_trials


def _original_draws(cfg, family: str) -> list[dict]:  # noqa: ANN001 - the pre-module algorithm
    grid = cfg.search.get(family) or {}
    count = int((cfg.search.get("trials") or {}).get(family, 0))
    rng = np.random.default_rng(cfg.random_seed + (17 if family == "lightgbm" else 29))
    trials = []
    for _ in range(count):
        trials.append({k: v[int(rng.integers(len(v)))] for k, v in grid.items()})
    return trials


def test_search_trials_are_fixed_by_the_seed() -> None:
    cfg = small_config()
    for family, count in (("lightgbm", 12), ("xgboost", 8)):
        trials = search_trials(cfg, family)
        assert len(trials) == count == cfg.search["trials"][family]
        assert trials == search_trials(cfg, family) == _original_draws(cfg, family)
        grid = cfg.search[family]
        for trial in trials:
            assert set(trial) == set(grid) and all(trial[k] in grid[k] for k in grid)
    assert search_trials(cfg, "catboost") == []


def test_neighbours_step_one_grid_value_each_way() -> None:
    grid = {"num_leaves": [7, 15, 31, 63], "learning_rate": [0.02, 0.05, 0.1]}
    params = {"num_leaves": 7, "learning_rate": 0.05, "other": 1}
    out = dict(neighbour_trials(params, grid, ["num_leaves", "learning_rate", "absent"]))
    assert set(out) == {"num_leaves-up", "learning_rate-down", "learning_rate-up"}
    assert out["num_leaves-up"] == {"num_leaves": 15, "learning_rate": 0.05, "other": 1}
    assert out["learning_rate-down"]["learning_rate"] == 0.02
    assert neighbour_trials({"num_leaves": 8}, grid, ["num_leaves"]) == []   # off the grid


def test_the_complexity_curve_varies_num_leaves_alone() -> None:
    cfg = small_config()
    trials = complexity_trials(cfg)
    base = cfg.model_params("lightgbm")
    assert [leaves for leaves, _ in trials] == cfg.search["complexity"]["num_leaves"]
    for leaves, params in trials:
        assert params == {**base, "num_leaves": leaves, "max_depth": -1}
