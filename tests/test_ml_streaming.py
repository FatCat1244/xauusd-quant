"""Streaming predictions equal batch predictions (Prompt #10, Steps 92-94).

A frozen model is replayed bar by bar: the features are recomputed from a
rolling buffer of bars up to ``t`` only, go through the frozen preprocessing,
model and calibrator, and must reproduce the batch prediction at ``t``. A
stored feature value that differs from its causal recomputation must make the
check fail (live-readiness FAIL), so the check cannot pass vacuously.
"""

from __future__ import annotations

from functools import cache

import numpy as np

from feature_synth import synthetic_bars
from xauusd_quant.features import load_regression_config
from xauusd_quant.features.factory import compute_families
from xauusd_quant.features.factory_config import load_features_config
from xauusd_quant.features.joins import EngineProvider
from xauusd_quant.features.registry import build_registry
from xauusd_quant.features.spectral_config import load_spectral_config
from xauusd_quant.features.wavelet_config import load_wavelet_config
from xauusd_quant.ml.calibration import fit_calibrator
from xauusd_quant.ml.models import make_model, preprocessing_kind
from xauusd_quant.ml.preprocessing import Preprocessor
from xauusd_quant.ml.streaming import stream_predictions
from xauusd_quant.models import load_ou_config
from xauusd_quant.research.config import load_research_config

N = 2600
NAMES = ["ret_1", "ret_5", "ret_z_64", "log_rv_20", "log_rv_64", "log_ewma_vol_94", "tod_sin",
         "session_london"]
ROWS = [2000 + 37 * k for k in range(8)]


@cache
def _configs() -> tuple:
    return (load_features_config(), load_regression_config(), load_ou_config(),
            load_spectral_config(), load_wavelet_config(), load_research_config())


@cache
def _batch() -> tuple:
    fcfg, regression, ou, spectral, wavelet, research = _configs()
    bars = synthetic_bars(N)
    provider = EngineProvider(bars=bars, regression_config=regression, spectral_config=spectral,
                              wavelet_config=wavelet)
    values, _ = compute_families(bars, provider, fcfg, ou, research,
                                 families=("returns", "volatility", "time"), log=False,
                                 collect=False)
    registry = {s.name: s.to_dict() for s in build_registry(fcfg, "1h", 3600.0)}
    stored = {n: np.asarray(values[n], dtype=np.float32) for n in NAMES}
    close = np.asarray(bars.close, dtype=np.float64)
    nxt = np.full(N, np.nan)
    nxt[:-1] = np.log(close[1:] / close[:-1])
    return bars, stored, registry, nxt


def _frozen(family: str) -> tuple:
    _, stored, _, nxt = _batch()
    task = "regression" if family == "ridge" else "classification"
    x = np.column_stack([stored[n] for n in NAMES]).astype(np.float32)
    y = (nxt > 0).astype(np.float64) if task == "classification" else np.abs(nxt)
    fit, cal_rows = np.arange(200, 1700), np.arange(1720, 1980)
    pre = Preprocessor.fit(x[fit], NAMES, kind=preprocessing_kind(family))
    params = {"n_estimators": 30, "min_data_in_leaf": 20} if family == "lightgbm" else {}
    model = make_model(family, task, params, seed=3, threads=1)
    model.fit(pre.transform(x[fit], NAMES), y[fit])
    method = "platt" if task == "classification" else "none"
    cal = fit_calibrator(model.predict(pre.transform(x[cal_rows], NAMES)), y[cal_rows], method)
    return model, pre, cal


def _stream(family: str, stored: dict[str, np.ndarray]) -> dict:
    fcfg, regression, ou, spectral, wavelet, research = _configs()
    bars, _, registry, _ = _batch()
    model, pre, cal = _frozen(family)
    return stream_predictions(model, pre, cal, NAMES, registry, bars, stored, ROWS, buffer=700,
                              fcfg=fcfg, ou=ou, research=research, regression=regression,
                              spectral=spectral, wavelet=wavelet)


def test_streaming_equals_batch_for_every_model_kind() -> None:
    _, stored, _, _ = _batch()
    for family in ("lightgbm", "logistic_l2", "ridge"):
        out = _stream(family, stored)
        assert out["passed"], (family, out["max_abs_prediction_difference"])
        assert out["steps"] == len(ROWS) and out["mismatches"] == 0
        assert set(out["families_recomputed"]) == {"returns", "volatility", "time"}
        assert out["max_abs_feature_difference"] < 1e-5
        np.testing.assert_allclose(out["live"], out["batch"], rtol=1e-4, atol=1e-6)


def test_a_stored_value_that_differs_from_its_causal_recomputation_fails() -> None:
    _, stored, _, _ = _batch()
    bad = {k: v.copy() for k, v in stored.items()}
    bad["log_rv_20"][ROWS[3]] += 0.5                       # computed some other way
    out = _stream("logistic_l2", bad)
    assert not out["passed"] and out["mismatches"] == 1
    assert out["max_abs_feature_difference"] >= 0.49
