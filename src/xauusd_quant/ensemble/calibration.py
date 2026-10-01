r"""Calibration of ensemble probabilities (Prompt #11, Steps 33-34).

An average of calibrated probabilities is not itself calibrated in general
(averaging pulls probabilities towards the middle). Three orders are compared,
each with its calibrator fitted on **history rows only** (earlier out-of-sample
blocks, :meth:`~.data.PairPredictions.history`) and scored on the next block:

* ``A_calibrate_then_average``: the mean of the constituents' Platt probabilities
  (each calibrated on its own fold's inner slice in Prompt #10) - no further step;
* ``B_average_then_calibrate_<m>``: the mean of the *raw* probabilities, then one
  calibrator ``m`` (sigmoid = Platt, or isotonic) fitted on the history;
* ``C_both_<m>``: the mean of the Platt probabilities, calibrated again on the history.

The raw mean is reported beside them. Calibrators come from
:mod:`xauusd_quant.ml.calibration` (Platt on the logit, isotonic only with enough
rows); nothing here sees the reserved period.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..ml.calibration import Calibrator, fit_calibrator
from .data import PairPredictions

__all__ = ["calibration_variants", "fit_ensemble_calibrator"]

_METHOD = {"sigmoid": "platt", "isotonic": "isotonic"}


def fit_ensemble_calibrator(p_hist: np.ndarray, y_hist: np.ndarray, method: str, *,
                            min_rows: int) -> Calibrator:
    ok = np.isfinite(p_hist) & np.isfinite(y_hist)
    return fit_calibrator(p_hist[ok], y_hist[ok], _METHOD[method], min_rows=min_rows)


def calibration_variants(pair: PairPredictions, models: list[str], k: int, *, embargo: int,
                         methods: list[str], min_rows: int
                         ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Every calibration order's predictions on block *k* (and the calibrators' parameters)."""
    if not pair.is_classification:
        raise ValueError("calibration variants apply to probabilities only")
    hist = pair.history(k, embargo)
    pos = pair.block_positions(k)
    raw_all = pair.matrix(models, raw=True).mean(axis=1)
    cal_all = pair.matrix(models).mean(axis=1)
    out = {"raw_average": raw_all[pos], "A_calibrate_then_average": cal_all[pos]}
    params: dict[str, dict[str, Any]] = {}
    for m in methods:
        for tag, series in (("B_average_then_calibrate", raw_all), ("C_both", cal_all)):
            cal = fit_ensemble_calibrator(series[hist], pair.label[hist], m, min_rows=min_rows)
            name = f"{tag}_{m}"
            out[name] = cal.apply(series[pos])
            params[name] = {"method": cal.method, "history_rows": int(hist.size),
                            **({k2: v for k2, v in cal.params.items() if k2 in ("a", "b", "rows",
                                                                                "reason")})}
    return out, params
