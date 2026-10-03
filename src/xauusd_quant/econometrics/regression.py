"""Fixed interpretable ARX/NNLS HAR-style predictors; all fitting uses preceding labels."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from scipy.optimize import nnls

from ..execution.config import content_hash
from .data import Point, Sample
from .diagnostics import segmented_hac


def validate_fit(samples: Sequence[Sample], cutoff: datetime, minimum: int) -> None:
    if len(samples) < minimum:
        raise ValueError("insufficient permitted training history")
    if any(
        s.point.available_utc >= cutoff
        or s.outcome.end_utc >= cutoff
        or s.outcome.matured_utc >= cutoff
        or s.outcome.start_utc < s.point.available_utc
        or s.outcome.matured_utc < s.outcome.end_utc
        or s.outcome.end_utc - s.outcome.start_utc
        != timedelta(seconds=s.outcome.horizon_bars * s.outcome.seconds)
        for s in samples
    ):
        raise ValueError("fit received future or crossing outcome information")
    horizons = {(s.outcome.horizon_bars, s.outcome.seconds) for s in samples}
    if len(horizons) != 1:
        raise ValueError("mixed target horizons/grids in one fit")


@dataclass(frozen=True)
class RegressionFit:
    model: str
    cutoff_utc: datetime
    horizon_bars: int
    grid_seconds: int
    names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    diagnostics: dict[str, Any]
    training_identity: str
    label_as_of_utc: datetime

    def predict(self, point: Point, at: datetime) -> float:
        if point.available_utc > at or at < self.cutoff_utc:
            raise ValueError("prediction used unavailable features/model")
        x = np.asarray(point.arx if self.model == "ARX" else point.rv)
        value = float(self.intercept + ((x - self.means) / self.scales) @ self.coefficients)
        if not np.isfinite(value) or (self.model == "HAR" and value <= 0):
            raise ValueError("invalid forecast; no fallback/clipping")
        return value

    def resolved(self) -> dict[str, Any]:
        return asdict(self)


def fit_regression(
    samples: Sequence[Sample],
    cutoff: datetime,
    model: str,
    *,
    minimum: int = 100,
    hac_lags: int = 3,
) -> RegressionFit:
    validate_fit(samples, cutoff, minimum)
    if model not in ("ARX", "HAR"):
        raise ValueError("unsupported interpretable model")
    x = np.asarray([s.point.arx if model == "ARX" else s.point.rv for s in samples])
    y = np.asarray(
        [s.outcome.log_return if model == "ARX" else s.outcome.realized_variance for s in samples]
    )
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("invalid fitting values")
    means = x.mean(axis=0) if model == "ARX" else np.zeros(3)
    scales = x.std(axis=0)
    if np.any(scales <= 1e-15):
        raise ValueError("degenerate predictor; no silent replacement")
    z = np.column_stack((np.ones(len(x)), (x - means) / scales))
    if np.linalg.matrix_rank(z) != z.shape[1]:
        raise ValueError("rank-deficient predictive regression")
    if model == "ARX":
        coef = np.linalg.lstsq(z, y, rcond=None)[0]
    else:
        # Scale the response so NNLS's absolute stopping tolerance is appropriate.
        response_scale = float(y.mean())
        if response_scale <= 0:
            raise ValueError("nonpositive realized-variance training target")
        coef = nnls(z, y / response_scale)[0] * response_scale
    residuals = y - z @ coef
    lag = max(hac_lags, samples[0].outcome.horizon_bars - 1)
    diagnostics: dict[str, Any] = {
        "observations": len(x),
        "training_mse": float(np.square(residuals).mean()),
        "coefficients_raw_units": list(coef[1:] / scales),
        "intercept_raw_units": float(coef[0] - (means / scales) @ coef[1:]),
        "interpretation": "predictive coefficients, never causal effects; HAR is constrained variance levels, not log volatility",
    }
    if model == "ARX":
        cov = segmented_hac(
            z, residuals, [s.point.available_utc for s in samples], samples[0].outcome.seconds, lag
        )
        diagnostics.update(
            hac_lags=lag,
            hac_standard_errors=list(np.sqrt(np.maximum(np.diag(cov), 0))),
            hac_scope="conditional fixed regression; actual-grid score pairs; no selection/search correction",
        )
    else:
        diagnostics["coefficient_uncertainty"] = (
            "unknown; ordinary unconstrained OLS/HAC inference is not NNLS inference"
        )
    return RegressionFit(
        model,
        cutoff,
        samples[0].outcome.horizon_bars,
        samples[0].outcome.seconds,
        ("return_1", "return_2", "volatility_12") if model == "ARX" else ("rv_1", "rv_3", "rv_12"),
        tuple(means),
        tuple(scales),
        tuple(coef[1:]),
        float(coef[0]),
        diagnostics,
        content_hash([s.point.row_id for s in samples]),
        max(s.outcome.matured_utc for s in samples),
    )
