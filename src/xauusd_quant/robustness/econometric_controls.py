"""Prespecified synthetic state, interval and monitor sensitivity using Stage13.5 APIs."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from ..econometrics.data import Outcome
from ..econometrics.monitor import ErrorCUSUM
from ..econometrics.state_space import LocalLevelState, fit_local_level
from ..econometrics.uncertainty import DelayedIntervals
from ..strategy_validation.runs import TrialLedger
from .plan import RobustnessPlan


def state_sensitivity(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    latent = 7.5 + np.cumsum(rng.normal(0, 0.0003, 900))
    logs = latent + rng.normal(0, 0.0002, len(latent))
    fit = fit_local_level([logs[:300]])  # Parameters never see evaluation observations.
    origin = datetime(2021, 1, 1, tzinfo=UTC)
    results = []
    for multiplier in (0.5, 1.0, 2.0):
        state = LocalLevelState(replace(fit, process_variance=fit.process_variance * multiplier))
        errors = []
        for i, value in enumerate(logs):
            at = origin + timedelta(seconds=300 * i)
            state.observe(float(value), at, contiguous=i > 0)
            if 300 <= i < len(logs) - 1:
                errors.append((logs[i + 1] - value - state.forecast_return()) ** 2)
        results.append(
            {
                "q_multiplier": multiplier,
                "mse": float(np.mean(errors)),
                "final_covariance": state.covariance,
                "state": state.resolved(),
            }
        )
    return {
        "prior_fit_rows": 300,
        "fit": fit.resolved(),
        "results": results,
        "scope": "synthetic known local level; q neighborhood reported whole; no smoother or economic claim",
    }


def interval_sensitivity(seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    errors = rng.normal(0, 0.0005, 800)
    # Declared shift in outcome scale, independent of coverage results.
    errors[400:] *= 2
    origin = datetime(2021, 1, 1, tzinfo=UTC)
    results = []
    for window, gamma in ((64, 0.0), (128, 0.0), (64, 0.01), (128, 0.01)):
        state = DelayedIntervals(window=window, gamma=gamma)
        covered: list[bool] = []
        widths = []
        for i, error in enumerate(errors):
            at = origin + timedelta(seconds=600 * i)
            end = at + timedelta(seconds=300)
            mature = end + timedelta(seconds=30)
            state.issue(str(i), 0.0, at, end, mature)
            outcome = Outcome(str(i), at, end, mature, float(error), float(error**2), 1, 300)
            row = state.deliver(outcome, mature + timedelta(microseconds=1))
            assert row is not None
            if row["covered"] is not None:
                covered.append(row["covered"])
                widths.append(row["width"])
        results.append(
            {
                "window": window,
                "gamma": gamma,
                "nominal_coverage": 0.8,
                "evaluated": len(covered),
                "coverage": float(np.mean(covered)),
                "mean_width": float(np.mean(widths)),
                "final_alpha": state.alpha,
                "scope": "matured synthetic outcome intervals; scale shifts; no exchangeability guarantee",
            }
        )
    return results


def monitor_sensitivity(seed: int, replicates: int) -> list[dict[str, Any]]:
    origin = datetime(2021, 1, 1, tzinfo=UTC)
    results = []
    for threshold in (6.0, 8.0, 10.0):
        unchanged_alarms, delays = [], []
        for replicate in range(replicates):
            noise = np.random.default_rng(seed + replicate).normal(size=600)
            unchanged = ErrorCUSUM(1.0, threshold=threshold)
            shifted = ErrorCUSUM(1.0, threshold=threshold)
            first = None
            for i, value in enumerate(noise):
                mature = origin + timedelta(seconds=i)
                unchanged.update(float(value), mature, mature + timedelta(microseconds=1))
                row = shifted.update(
                    float(value + (3 if i >= 300 else 0)),
                    mature,
                    mature + timedelta(microseconds=1),
                )
                if i >= 300 and row["alarm"] and first is None:
                    first = i - 300 + 1
            unchanged_alarms.append(unchanged.alarms)
            delays.append(first)
        events = sum(n > 0 for n in unchanged_alarms)
        p = events / replicates
        results.append(
            {
                "threshold": threshold,
                "replicates": replicates,
                "null_alarms_per_600": unchanged_alarms,
                "null_any_alarm_fraction": p,
                "monte_carlo_binomial_se": math.sqrt(p * (1 - p) / replicates),
                "maximum_binomial_se": 0.5 / math.sqrt(replicates),
                "shift_detection_delay_observations": delays,
                "scope": "unchanged Gaussian control and +3 sigma at300; health only; finite simulation cannot prove zero false alarms",
            }
        )
    return results


def controls(plan: RobustnessPlan, ledger: TrialLedger) -> dict[str, Any]:
    state = ledger.call(
        "synthetic_state_neighborhood",
        {"seed": plan.seed, "q_multipliers": [0.5, 1.0, 2.0]},
        lambda: state_sensitivity(plan.seed),
    )
    intervals = ledger.call(
        "synthetic_interval_neighborhood",
        {"seed": plan.seed, "window": [64, 128], "gamma": [0.0, 0.01]},
        lambda: interval_sensitivity(plan.seed),
    )
    monitor = ledger.call(
        "synthetic_monitor_neighborhood",
        {"seed": plan.seed, "threshold": [6.0, 8.0, 10.0], "replicates": plan.discovery_replicates},
        lambda: monitor_sensitivity(plan.seed, plan.discovery_replicates),
    )
    return {
        "state": state,
        "intervals": intervals,
        "monitor": monitor,
        "status": "synthetic sensitivity only; no market promotion or future performance inference",
    }
