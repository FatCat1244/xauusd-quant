"""Strict return conventions and complete-family Holm correction, no IID time-series errors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReturnDefinition:
    sampling_seconds: int
    capital_account: float
    capital_convention: str
    exposure: str
    compounding: str
    missing: str
    overnight: str
    overlapping_trades: str
    dependence: str
    annualization: str

    def __post_init__(self) -> None:
        if type(self.sampling_seconds) is not int or self.sampling_seconds <= 0:
            raise ValueError("regular positive sampling required; trade PnL is not returns")
        if not math.isfinite(self.capital_account) or self.capital_account <= 0:
            raise ValueError("finite positive capital required")
        if (
            self.capital_convention != "fixed_initial_cash"
            or self.compounding != "none"
            or self.missing != "reject"
            or self.annualization != "none"
        ):
            raise ValueError(
                "only fixed capital additive returns, missing rejection, no annualization supported"
            )
        if not all((self.exposure, self.overnight, self.overlapping_trades, self.dependence)):
            raise ValueError("exposure/overnight/overlap/dependence must be explicit")


def equity_returns(
    times: list[datetime], equity: list[float], definition: ReturnDefinition
) -> np.ndarray:
    values = np.asarray(equity, dtype=float)
    if len(times) != len(values) or len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("complete finite chronological equity required; no silent fill")
    if any(t.tzinfo is None or t.utcoffset() is None for t in times):
        raise ValueError("explicit UTC clocks required")
    if any(
        b - a != timedelta(seconds=definition.sampling_seconds)
        for a, b in zip(times, times[1:], strict=False)
    ):
        raise ValueError("irregular or missing equity marks cannot become regular returns")
    return np.diff(values) / definition.capital_account


def performance(
    returns: np.ndarray, *, definition: ReturnDefinition, duration_seconds: float
) -> dict[str, Any]:
    x = np.asarray(returns, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("finite return vector required")
    if (
        not math.isfinite(duration_seconds)
        or duration_seconds != len(x) * definition.sampling_seconds
    ):
        raise ValueError("evaluation duration must match declared regular return sampling")
    variance = float(np.var(x, ddof=1))
    return {
        "observations": len(x),
        "evaluation_duration_seconds": duration_seconds,
        "sampling_seconds": definition.sampling_seconds,
        "capital_account": definition.capital_account,
        "additive_return": float(x.sum()),
        "mean": float(x.mean()),
        "sample_variance": variance,
        "unannualized_mean_over_sd": float(x.mean() / math.sqrt(variance))
        if variance > 0
        else None,
        "ratio_status": "defined_descriptive_only" if variance > 0 else "unavailable_zero_variance",
        "inference": "no IID standard error; use declared block resampling",
        "deflated_sharpe": {
            "status": "unavailable",
            "reason": "unknown effective selection exposure and unsupported dependence-adjusted moments",
        },
    }


def holm(
    family: tuple[str, ...], pvalues: dict[str, float | None], *, validity: str, alpha: float = 0.05
) -> dict[str, Any]:
    """Step-down Bonferroni: ordered p[i] <= alpha/(m-i); arbitrary dependence.

    Verified against Goeman/Solari (2010), section 3, Holm critical-value formula. Requires valid
    marginal p-values and the complete frozen family; repeated analyses not repaired.
    """
    if not family or len(set(family)) != len(family) or set(pvalues) != set(family):
        raise ValueError("complete unique frozen hypothesis family required")
    if not 0 < alpha < 1 or not math.isfinite(alpha):
        raise ValueError("invalid alpha")
    for p in pvalues.values():
        if p is not None and (not math.isfinite(p) or not 0 <= p <= 1):
            raise ValueError("valid finite p-values required; NaN is not nonsignificance")
    if validity != "valid_marginal_pvalues_fixed_analysis" or any(
        p is None for p in pvalues.values()
    ):
        return {
            "status": "unavailable",
            "family": family,
            "reason": "missing or unsupported marginal p-values; never correct favorable subset",
        }
    ordered = sorted(family, key=lambda k: (pvalues[k], k))
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, name in enumerate(ordered):
        value = pvalues[name]
        assert value is not None
        running = max(running, (len(family) - i) * value)
        adjusted[name] = min(1.0, running)
    return {
        "status": "calculated",
        "adjusted": adjusted,
        "rejected": {k: p <= alpha for k, p in adjusted.items()},
        "scope": "FWER for supplied fixed family and valid marginal p-values; no repair of adaptive historical searches",
    }
