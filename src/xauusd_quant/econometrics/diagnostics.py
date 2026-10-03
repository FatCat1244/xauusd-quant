"""Reuse stationarity/serial tests; gap-aware HAC describes predictive coefficients only."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from ..research.autocorrelation import ljung_box_test
from ..research.ou_diagnostics import _arch_lm
from ..research.stationarity import adf_test, kpss_test


def segmented_hac(
    x: np.ndarray, residuals: np.ndarray, times: Sequence[datetime], seconds: int, lags: int
) -> np.ndarray:
    """Bartlett Newey-West score covariance; pairs must have their actual grid separation."""
    n, p = x.shape
    if n <= p or len(times) != n or residuals.shape != (n,) or lags < 0:
        raise ValueError("invalid HAC design/history")
    score = x * residuals[:, None]
    meat = score.T @ score
    lookup = {t: i for i, t in enumerate(times)}
    if len(lookup) != n:
        raise ValueError("duplicate HAC timestamps")
    for lag in range(1, lags + 1):
        pairs = [
            (i, lookup[t - timedelta(seconds=seconds * lag)])
            for i, t in enumerate(times)
            if t - timedelta(seconds=seconds * lag) in lookup
        ]
        if pairs:
            a, b = np.asarray(pairs).T
            cross = score[a].T @ score[b]
            meat += (1 - lag / (lags + 1)) * (cross + cross.T)
    bread = np.linalg.inv(x.T @ x)
    covariance = bread @ meat @ bread * (n / (n - p))
    if not np.isfinite(covariance).all() or np.any(np.diag(covariance) < -1e-14):
        raise ValueError("invalid HAC covariance; inference unknown")
    return covariance


def series_diagnostics(values: np.ndarray, *, label: str) -> dict[str, Any]:
    """Input must be one contiguous chronological segment, capped without sparse resampling."""
    x = np.asarray(values, dtype=float)[-512:]
    result: dict[str, Any] = {
        "series": label,
        "observations": len(x),
        "scope": "last <=512 observations of longest contiguous training segment",
        "interpretation": "failure to reject is not proof; stationarity is not economic reversion; HAC cannot repair leakage/search",
    }
    if len(x) < 30 or not np.isfinite(x).all() or np.std(x) <= 1e-15:
        return {**result, "status": "untested", "reason": "insufficient/invalid/constant series"}
    for name, function in (("adf", adf_test), ("kpss", kpss_test)):
        try:
            test = function(x, max_observations=512).to_dict()
            result[name] = (
                test
                if math.isfinite(test["statistic"]) and math.isfinite(test["p_value"])
                else {"status": "untested"}
            )
        except (ValueError, np.linalg.LinAlgError) as error:
            result[name] = {"status": "untested", "reason": str(error)}
    result["ljung_box"] = [r.to_dict() for r in ljung_box_test(x, lags=(3, 6), series_name=label)]
    statistic, p_value = _arch_lm(x, 3)
    result["arch_lm"] = {
        "null": "no ARCH dependence through lag 3",
        "statistic": statistic,
        "p_value": p_value,
        "status": "measured" if p_value is not None and math.isfinite(p_value) else "untested",
    }
    result["status"] = "diagnostics_only"
    return result
