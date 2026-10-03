"""Local-level log observation model; prior-only parameters, Joseph filtering, no smoothing."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class LocalLevelFit:
    process_variance: float
    observation_variance: float
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(v) and v > 0 for v in (self.process_variance, self.observation_variance)
        ):
            raise ValueError("local-level variances must be finite and positive")

    def resolved(self) -> dict[str, Any]:
        return asdict(self)


def fit_local_level(
    segments: Sequence[np.ndarray], *, minimum: int = 100, max_iterations: int = 200
) -> LocalLevelFit:
    pieces = [np.asarray(x, dtype=float) for x in segments if len(x) >= 2]
    if sum(len(x) - 1 for x in pieces) < minimum or not all(np.isfinite(x).all() for x in pieces):
        raise ValueError("insufficient/invalid local-level history")
    difference_variance = float(
        np.square(np.concatenate([np.diff(x) * 10000 for x in pieces])).mean()
    )
    if difference_variance <= 1e-10:
        raise ValueError("degenerate local-level innovations")
    scaled = [(x - x[0]) * 10000 for x in pieces]

    def likelihood(log_variances: np.ndarray) -> float:
        q, r = np.exp(log_variances)
        value = 0.0
        for series in scaled:
            level, covariance = float(series[0]), float(r)
            for observed in series[1:]:
                prior = covariance + q
                innovation_variance = prior + r
                innovation = observed - level
                value += (
                    math.log(innovation_variance) + innovation * innovation / innovation_variance
                ) / 2
                gain = prior / innovation_variance
                level += gain * innovation
                covariance = (1 - gain) ** 2 * prior + gain * gain * r
        return value / sum(len(x) - 1 for x in scaled)

    bounds = [(math.log(difference_variance * 1e-6), math.log(difference_variance * 100))] * 2
    optimum = minimize(
        likelihood,
        np.log([difference_variance * 0.5] * 2),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iterations, "ftol": 1e-10},
    )
    if not optimum.success or not math.isfinite(optimum.fun) or not np.isfinite(optimum.x).all():
        raise ValueError(f"local-level nonconvergence: {optimum.message}; no fallback")
    q, r = (float(v) / 1e8 for v in np.exp(optimum.x))
    return LocalLevelFit(
        q,
        r,
        {
            "converged": True,
            "optimizer": "scipy L-BFGS-B",
            "iterations": int(optimum.nit),
            "equations": "x_t=x_(t-1)+w_t; y_t=x_t+v_t, independent zero-mean Gaussian noises",
            "initialization": "first segment observation as level, covariance=r; condition on first observation",
            "gap_rule": "reset to first post-gap observation; no interpolated latent path",
            "bounds": "q/r in [1e-6,100] times training difference second moment; weak identification at boundaries reported",
            "at_parameter_boundary": any(
                abs(v - a) < 1e-5 or abs(v - b) < 1e-5
                for v, (a, b) in zip(optimum.x, bounds, strict=True)
            ),
            "interpretation": "estimated latent log level, not true fair value; filter only",
        },
    )


class LocalLevelState:
    def __init__(self, fit: LocalLevelFit) -> None:
        self.fit = fit
        self.level: float | None = None
        self.covariance = fit.observation_variance
        self.last_observation: float | None = None
        self.last_utc: datetime | None = None

    def observe(
        self, observed_log_price: float, at: datetime, *, contiguous: bool
    ) -> dict[str, Any]:
        if not math.isfinite(observed_log_price) or (
            self.last_utc is not None and at <= self.last_utc
        ):
            raise ValueError("invalid/nonchronological local-level observation")
        if self.level is None or not contiguous:
            self.level, self.covariance = observed_log_price, self.fit.observation_variance
            innovation = None
            reset = True
        else:
            prior = self.covariance + self.fit.process_variance
            variance = prior + self.fit.observation_variance
            innovation = observed_log_price - self.level
            gain = prior / variance
            self.level += gain * innovation
            self.covariance = (1 - gain) ** 2 * prior + gain * gain * self.fit.observation_variance
            reset = False
        if (
            not math.isfinite(self.level)
            or not math.isfinite(self.covariance)
            or self.covariance <= 0
        ):
            raise ValueError("numerically invalid local-level state")
        self.last_observation, self.last_utc = observed_log_price, at
        return {
            "available_utc": at,
            "filtered_log_level": self.level,
            "state_variance": self.covariance,
            "innovation": innovation,
            "gap_reset": reset,
            "mode": "causal_filter_no_smoothing",
        }

    def forecast_return(self) -> float:
        if self.level is None or self.last_observation is None:
            raise ValueError("local-level state not initialized")
        # E[y_(t+h)|past]-y_t in log units under the fixed no-drift local-level model.
        return self.level - self.last_observation

    def resolved(self) -> dict[str, Any]:
        return {
            "fit": self.fit.resolved(),
            "level": self.level,
            "covariance": self.covariance,
            "last_observation": self.last_observation,
            "last_utc": self.last_utc.isoformat() if self.last_utc else None,
        }

    @classmethod
    def restore(cls, body: dict[str, Any]) -> LocalLevelState:
        state = cls(LocalLevelFit(**body["fit"]))
        if not math.isfinite(body["covariance"]) or body["covariance"] <= 0:
            raise ValueError("invalid persisted filter covariance")
        for key in ("level", "last_observation"):
            if body[key] is not None and not math.isfinite(body[key]):
                raise ValueError("invalid persisted filter level")
        state.level, state.covariance, state.last_observation = (
            body["level"],
            body["covariance"],
            body["last_observation"],
        )
        state.last_utc = datetime.fromisoformat(body["last_utc"]) if body["last_utc"] else None
        return state
