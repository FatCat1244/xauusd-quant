"""Known synthetic return/variance/state processes; no performance requirements on markets."""

from datetime import UTC, datetime, timedelta

import numpy as np

from xauusd_quant.econometrics.data import GridData
from xauusd_quant.econometrics.plan import EconometricPlan
from xauusd_quant.econometrics.runs import SyntheticBars
from xauusd_quant.execution.config import ExecutionConfig
from xauusd_quant.strategy_validation.plan import Fold

ORIGIN = datetime(2021, 5, 1, tzinfo=UTC)


def at(index: float) -> datetime:
    return ORIGIN + timedelta(seconds=300 * index)


def returns_process(n: int = 1100, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = np.zeros(n)
    variance = 2e-7
    for i in range(1, n):
        variance = 2e-8 + 0.1 * result[i - 1] ** 2 + 0.8 * variance
        result[i] = 0.35 * result[i - 1] + np.sqrt(variance) * rng.normal()
    return result


def grid(n: int = 1100) -> GridData:
    prices = 2000 * np.exp(np.cumsum(returns_process(n)))
    return GridData([at(i) for i in range(n)], list(prices), 300)


def fixture() -> tuple[EconometricPlan, ExecutionConfig, SyntheticBars]:
    folds = tuple(
        Fold(
            f"F{i + 1}",
            at(0).isoformat(),
            at(1000 + i * 20).isoformat(),
            at(1020 + i * 20).isoformat(),
            (
                (at(200).isoformat(), at(240).isoformat()),
                (at(350).isoformat(), at(390).isoformat()),
            ),
        )
        for i in range(2)
    )
    plan = EconometricPlan(
        "ECON_SYNTH_V001", "5m", folds, evaluation_history_classification="software_correctness"
    )
    values = grid()
    source = SyntheticBars(values.opens, list(np.exp(values.logs)), 300)
    return plan, ExecutionConfig(), source
