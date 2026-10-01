"""Feature engineering.

Currently the rolling-regression detrending model. Every feature here is
causal: a value at time ``t`` is a function of bars at or before ``t`` only,
so appending later data cannot change it.

Forward-looking quantities live in the research layer's outcome analysis, never
here.
"""

from __future__ import annotations

from .config import RegressionConfig, load_regression_config
from .rolling_regression import (
    FEATURE_COLUMNS,
    RollingRegressionResult,
    fit_window,
    rolling_ols,
    rolling_regression_features,
    theil_sen_slope,
    window_design,
)

__all__ = [
    "FEATURE_COLUMNS",
    "RegressionConfig",
    "RollingRegressionResult",
    "fit_window",
    "load_regression_config",
    "rolling_ols",
    "rolling_regression_features",
    "theil_sen_slope",
    "window_design",
]
