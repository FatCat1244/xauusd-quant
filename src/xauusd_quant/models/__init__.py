"""Stochastic models of the regression residual.

Currently the Ornstein-Uhlenbeck process, estimated through its exact AR(1)
discretisation. The mathematics lives in :mod:`.ornstein_uhlenbeck` and the
typed configuration (``config/ou.yaml``) in :mod:`.config`.

These are *descriptive* models. Nothing here defines an entry, an exit or a
position, and nothing chooses a window or a threshold.
"""

from __future__ import annotations

from .config import OUConfig, load_ou_config
from .ornstein_uhlenbeck import (
    OU_STATES,
    AR1Fit,
    OUValidityRules,
    RollingOU,
    decay_ratio,
    expected_value,
    fit_ar1,
    half_life_bars_from_b,
    map_ar1_to_ou,
    rolling_ar1,
    rolling_ou,
    simulate_ar1,
    simulate_ou,
    simulate_random_walk,
    theta_from_b,
)

__all__ = [
    "OU_STATES",
    "AR1Fit",
    "OUConfig",
    "OUValidityRules",
    "RollingOU",
    "decay_ratio",
    "expected_value",
    "fit_ar1",
    "half_life_bars_from_b",
    "load_ou_config",
    "map_ar1_to_ou",
    "rolling_ar1",
    "rolling_ou",
    "simulate_ar1",
    "simulate_ou",
    "simulate_random_walk",
    "theta_from_b",
]
