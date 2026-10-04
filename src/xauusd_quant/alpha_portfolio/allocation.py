"""Prior-only aligned sleeve risk estimates; one fixed constrained objective."""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from ..execution.config import content_hash
from ..execution.policy import utc_time


@dataclass(frozen=True)
class Allocation:
    allocation_id: str
    effective_utc: datetime
    information_as_of_utc: datetime
    weights: dict[str, float]
    method: str
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        if utc_time(self.information_as_of_utc) >= utc_time(self.effective_utc):
            raise ValueError("weights require strictly prior information")
        values = list(self.weights.values())
        if any(not np.isfinite(w) or w < 0 for w in values):
            raise ValueError("nonnegative finite allocations required")
        if values and not np.isclose(sum(values), 1.0, rtol=0, atol=1e-12):
            raise ValueError("fully allocated unit budget required")
        if not self.allocation_id or self.method not in ("equal", "minimum_variance_shrunk"):
            raise ValueError("allocation identity and declared method required")

    def resolved(self) -> dict[str, Any]:
        return asdict(self)


def minimum_variance(covariance: np.ndarray) -> np.ndarray:
    """Minimize w'Cw, w>=0, sum(w)=1 over at most three comparable sleeves.

    Exhaustively inspect active sets. This is not an expected-return or leverage
    optimizer. Positive-definiteness is required; equal fallback lives in fit().
    """
    c = np.asarray(covariance, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or not 1 <= len(c) <= 3:
        raise ValueError("square covariance for 1..3 sleeves required")
    if not np.isfinite(c).all() or not np.allclose(c, c.T, rtol=1e-10, atol=1e-18):
        raise ValueError("finite symmetric covariance required")
    if np.linalg.eigvalsh(c).min() <= 1e-18:
        raise ValueError("positive definite covariance required")
    best: np.ndarray | None = None
    score = float("inf")
    for size in range(1, len(c) + 1):
        for indices in itertools.combinations(range(len(c)), size):
            index = list(indices)
            solution = np.linalg.solve(c[np.ix_(index, index)], np.ones(size))
            solution /= solution.sum()
            if not np.isfinite(solution).all() or (solution < -1e-12).any():
                continue
            weights = np.zeros(len(c))
            weights[index] = np.maximum(solution, 0)
            weights /= weights.sum()
            objective = float(weights @ c @ weights)
            if objective < score:
                score, best = objective, weights
    if best is None:
        raise ValueError("infeasible allocation")
    return best


def fit(
    identities: tuple[str, ...],
    method: str,
    effective: datetime,
    stamps: list[datetime],
    matured: list[datetime],
    returns: np.ndarray,
    *,
    window: int = 32,
    shrinkage: float = 0.5,
) -> Allocation:
    """Filter by observation AND outcome publication before fitting; future ignored.

    Inputs: same 5-minute grid, fixed-initial-capital standalone *net* returns with
    the same exposure budget. Complete cases only, at least16; gaps never bridged
    by resampling. Covariance is descriptive risk, not independence or mean skill.
    """
    effective = utc_time(effective)
    if len(set(identities)) != len(identities) or len(identities) > 3:
        raise ValueError("unique bounded alpha universe required")
    if method not in ("equal", "minimum_variance_shrunk") or window != 32 or shrinkage != 0.5:
        raise ValueError("fixed prespecified methods and estimation settings")
    data = np.asarray(returns, dtype=float)
    if data.shape != (len(stamps), len(identities)) or len(matured) != len(stamps):
        raise ValueError("aligned timestamp/maturity/return dimensions required")
    times = [utc_time(t) for t in stamps]
    labels = [utc_time(t) for t in matured]
    if any(b <= a for a, b in itertools.pairwise(times)) or any(
        m < t for t, m in zip(times, labels, strict=True)
    ):
        raise ValueError("ordered observations and valid maturity required")
    indices = [
        i
        for i, (t, m) in enumerate(zip(times, labels, strict=True))
        if t < effective and m < effective
    ][-window:]
    selected = data[indices].copy()
    finite = np.isfinite(selected).all(axis=1)
    sample = selected[finite]
    weights = np.full(len(identities), 1 / len(identities)) if identities else np.array([])
    info = max((labels[i] for i in indices), default=effective - timedelta(days=1))
    diagnostics: dict[str, Any] = {
        "normalization": "fixed signed unit intent; no fitted scaling",
        "return_units": "5-minute standalone net equity increments / fixed initial cash",
        "rows": len(sample),
        "missing_rows": int((~finite).sum()),
        "training_sha256": content_hash(
            {
                "times": [times[i] for i in indices],
                "matured": [labels[i] for i in indices],
                "values": np.nan_to_num(selected).tolist(),
                "finite": finite.tolist(),
            }
        ),
        "fallback": None,
        "uncertainty": "short dependent sample; no standard errors or diversification proof",
    }
    if method == "minimum_variance_shrunk" and identities:
        try:
            if len(sample) < 16:
                raise ValueError("insufficient complete aligned rows")
            if any(
                (times[b] - times[a]).total_seconds() != 300 for a, b in itertools.pairwise(indices)
            ):
                raise ValueError("nonuniform risk observation grid")
            empirical = np.atleast_2d(np.cov(sample, rowvar=False, ddof=1))
            covariance = (1 - shrinkage) * empirical + shrinkage * np.diag(np.diag(empirical))
            weights = minimum_variance(covariance)
            diagnostics.update(
                {
                    "sample_covariance": empirical.tolist(),
                    "shrunk_covariance": covariance.tolist(),
                    "eigenvalues": np.linalg.eigvalsh(covariance).tolist(),
                    "first_half_covariance": np.atleast_2d(
                        np.cov(sample[: len(sample) // 2], rowvar=False)
                    ).tolist(),
                    "last_half_covariance": np.atleast_2d(
                        np.cov(sample[len(sample) // 2 :], rowvar=False)
                    ).tolist(),
                    "negative_coloss_fraction": ((sample < 0).all(axis=1)).mean().item(),
                    "tail_coloss_status": "unavailable: short dependent training sample; no tail inference",
                }
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            diagnostics["fallback"] = f"equal: {error}"
    body = {
        "effective": effective,
        "information": info,
        "identities": identities,
        "method": method,
        "weights": weights.tolist(),
        "diagnostics": diagnostics,
    }
    return Allocation(
        f"ALLOCATION_{content_hash(body)[:16]}_V001",
        effective,
        info,
        dict(zip(identities, weights.tolist(), strict=True)),
        method,
        diagnostics,
    )
