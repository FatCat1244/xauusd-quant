"""Zero-mean variance benchmarks with positive constrained GARCH and causal recursion."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter


@dataclass(frozen=True)
class GarchFit:
    omega: float
    alpha: float
    beta: float
    initialization_variance: float
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(v)
            for v in (self.omega, self.alpha, self.beta, self.initialization_variance)
        ):
            raise ValueError("nonfinite GARCH specification")
        if (
            self.omega <= 0
            or self.initialization_variance <= 0
            or min(self.alpha, self.beta) < 0
            or self.alpha + self.beta > 0.99500001
        ):
            raise ValueError("invalid/nonstationary GARCH parameters")

    def resolved(self) -> dict[str, Any]:
        return asdict(self)


def variance_path(
    x: np.ndarray, omega: float, alpha: float, beta: float, initial: float
) -> np.ndarray:
    forcing = omega + alpha * np.r_[0.0, np.square(x[:-1])]
    forcing[0] = initial
    return np.asarray(lfilter([1.0], [1.0, -beta], forcing), dtype=float)


def fit_garch(
    segments: Sequence[np.ndarray], *, minimum: int = 100, max_iterations: int = 200
) -> GarchFit:
    pieces = [np.asarray(x, dtype=float) * 10000 for x in segments if len(x)]
    if sum(len(x) for x in pieces) < minimum or not all(np.isfinite(x).all() for x in pieces):
        raise ValueError("insufficient/invalid GARCH training history")
    sample_variance = float(np.square(np.concatenate(pieces)).mean())
    if sample_variance <= 1e-10:
        raise ValueError("degenerate GARCH variance")

    def objective(parameters: np.ndarray) -> float:
        omega, alpha, beta = parameters
        if omega <= 0 or min(alpha, beta) < 0 or alpha + beta >= 1:
            return 1e30
        initial = omega / (1 - alpha - beta)
        value = 0.0
        for x in pieces:
            h = variance_path(x, omega, alpha, beta, initial)
            if not np.isfinite(h).all() or np.any(h <= 0):
                return 1e30
            value += float(np.sum(np.log(h) + np.square(x) / h)) / 2
        return value / sum(len(x) for x in pieces)

    optimum = minimize(
        objective,
        [sample_variance * 0.05, 0.05, 0.90],
        method="SLSQP",
        bounds=[(1e-10, sample_variance * 10), (0, 0.995), (0, 0.995)],
        constraints=[{"type": "ineq", "fun": lambda p: 0.995 - p[1] - p[2]}],
        options={"maxiter": max_iterations, "ftol": 1e-9},
    )
    if not optimum.success or not np.isfinite(optimum.x).all() or not math.isfinite(optimum.fun):
        raise ValueError(f"GARCH nonconvergence: {optimum.message}; no fallback")
    omega, alpha, beta = (float(v) for v in optimum.x)
    unconditional = omega / (1 - alpha - beta)
    if alpha + beta >= 0.995 - 1e-6 or unconditional > 1000 * sample_variance:
        raise ValueError("unstable/boundary GARCH estimate; no fallback")
    return GarchFit(
        omega / 1e8,
        alpha,
        beta,
        unconditional / 1e8,
        {
            "optimizer": "scipy SLSQP",
            "converged": True,
            "iterations": int(optimum.nit),
            "mean": "zero",
            "innovations": "Gaussian quasi-likelihood, not asserted Gaussian data",
            "persistence": alpha + beta,
            "parameters_scale": "returns multiplied by 10000 for fitting; variance restored",
            "gaps": "independent segment reinitialization at stationary variance; no missing returns synthesized",
            "fallback": "none",
        },
    )


class VarianceState:
    """Recursive observed-return state; partition boundaries have no semantics."""

    def __init__(
        self,
        model: str,
        initial: float,
        horizon: int = 1,
        *,
        garch: GarchFit | None = None,
        decay: float = 0.94,
    ) -> None:
        if model not in ("ROLLING", "EWMA", "GARCH") or not math.isfinite(initial) or initial <= 0:
            raise ValueError("invalid variance state")
        if not 1 <= horizon <= 12 or not 0 < decay < 1 or (model == "GARCH" and garch is None):
            raise ValueError("invalid variance horizon/parameters")
        self.model, self.initial, self.horizon, self.garch, self.decay = (
            model,
            initial,
            horizon,
            garch,
            decay,
        )
        self.next_variance = initial
        self.history: deque[float] = deque(maxlen=12)
        self.last_utc: datetime | None = None

    def observe(self, value: float | None, at: datetime, *, contiguous: bool) -> None:
        if self.last_utc is not None and at <= self.last_utc:
            raise ValueError("variance update not strictly chronological")
        if value is not None and not math.isfinite(value):
            raise ValueError("invalid observed return")
        self.last_utc = at
        if not contiguous or value is None:
            self.history.clear()
            self.next_variance = self.initial
            return
        self.history.append(value)
        if self.model == "GARCH":
            assert self.garch is not None
            self.next_variance = (
                self.garch.omega
                + self.garch.alpha * value * value
                + self.garch.beta * self.next_variance
            )
        elif self.model == "EWMA":
            self.next_variance = self.decay * self.next_variance + (1 - self.decay) * value * value
        elif len(self.history) == 12:
            self.next_variance = float(np.mean(np.square(self.history)))
        if not math.isfinite(self.next_variance) or self.next_variance <= 0:
            raise ValueError("invalid recursive variance; no silent replacement")

    def forecast(self) -> float:
        if self.model == "ROLLING" and len(self.history) < 12:
            raise ValueError("rolling variance warm-up incomplete")
        total, step = 0.0, self.next_variance
        for _ in range(self.horizon):
            total += step
            if self.model == "GARCH":
                assert self.garch is not None
                step = self.garch.omega + (self.garch.alpha + self.garch.beta) * step
        if not math.isfinite(total) or total <= 0:
            raise ValueError("nonpositive/nonfinite future variance")
        return total

    def resolved(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "initial": self.initial,
            "horizon": self.horizon,
            "decay": self.decay,
            "garch": self.garch.resolved() if self.garch else None,
            "next_variance": self.next_variance,
            "history": list(self.history),
            "last_utc": self.last_utc.isoformat() if self.last_utc else None,
        }

    @classmethod
    def restore(cls, body: dict[str, Any]) -> VarianceState:
        state = cls(
            body["model"],
            body["initial"],
            body["horizon"],
            garch=GarchFit(**body["garch"]) if body["garch"] else None,
            decay=body["decay"],
        )
        if (
            not math.isfinite(body["next_variance"])
            or body["next_variance"] <= 0
            or len(body["history"]) > 12
        ):
            raise ValueError("invalid persisted variance state")
        state.next_variance = body["next_variance"]
        state.history.extend(body["history"])
        state.last_utc = datetime.fromisoformat(body["last_utc"]) if body["last_utc"] else None
        return state
