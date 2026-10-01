"""Probability calibration (Prompt #10, Steps 26-29).

Calibrators are fitted on the fold's inner slice - after the fitting rows,
purged, before the validation block - so no row is used both to fit or score
the model and to calibrate it.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml_synth import small_config, synthetic_ml_data
from xauusd_quant.ml.calibration import (
    Calibrator,
    expected_calibration_error,
    fit_calibrator,
    reliability,
)
from xauusd_quant.ml.splits import walk_forward_folds
from xauusd_quant.ml.training import UnitSpec, run_unit


def _labels(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95, n)
    return p, (rng.random(n) < p).astype(np.float64)


def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def test_calibrators_are_fitted_on_the_inner_slice_only() -> None:
    data = synthetic_ml_data()
    cfg = small_config()
    h = 5
    fold = walk_forward_folds(data.timestamps, cfg.walk_forward, horizon=h)[3]
    names = data.feature_set("standard")
    spec = UnitSpec(target="mean_reversion", horizon=h, family="lightgbm",
                    feature_set="standard", features=names, fold=fold,
                    params=cfg.model_params("lightgbm"), seed=cfg.random_seed)
    tspec = cfg.targets["mean_reversion"]
    res = run_unit(data, spec, tspec, cfg, keep_model=True)
    y = data.target(tspec, h, cfg.log_floor).y
    inner = np.arange(*fold.inner)
    inner = inner[np.isfinite(y[inner])]
    # the calibration rows touch neither the fitting rows nor the scored block
    gap = h + cfg.walk_forward.embargo_bars
    assert fold.fit[1] + gap <= inner.min() and inner.max() + gap < fold.validate[0]
    assert res.model is not None and res.preprocessor is not None
    p_inner = res.model.predict(res.preprocessor.transform(data.design(names)[inner], names))
    for method in ("platt", "isotonic"):
        again = fit_calibrator(p_inner, y[inner], method,
                               min_rows=int(cfg.calibration["isotonic_min_rows"]))
        assert res.calibrators[method].to_dict() == again.to_dict()
    assert res.calibrators["platt"].params["rows"] == inner.size


def test_platt_repairs_a_miscalibrated_score_out_of_sample() -> None:
    p, y = _labels(40_000)
    raw = p ** 2.5                                         # ranks right, scale wrong
    cal = fit_calibrator(raw[:20_000], y[:20_000], "platt")
    fixed = cal.apply(raw[20_000:])
    assert _log_loss(fixed, y[20_000:]) < _log_loss(raw[20_000:], y[20_000:]) - 0.01
    assert expected_calibration_error(fixed, y[20_000:]) < \
        expected_calibration_error(raw[20_000:], y[20_000:]) / 3
    # a monotone map: the ranking (AUC) is unchanged
    assert np.all(np.diff(fixed[np.argsort(raw[20_000:])]) >= -1e-12)


def test_isotonic_needs_enough_rows_and_stays_monotone_and_bounded() -> None:
    p, y = _labels(30_000, seed=1)
    small = fit_calibrator(p[:500], y[:500], "isotonic", min_rows=20_000)
    assert small.method == "none" and "fewer than 20000" in small.params["reason"]
    iso = fit_calibrator(p, y, "isotonic", min_rows=20_000)
    grid = np.linspace(-0.5, 1.5, 401)
    out = iso.apply(grid)
    assert iso.method == "isotonic"
    assert np.all(np.diff(out) >= 0) and out.min() >= 0.0 and out.max() <= 1.0


def test_platt_with_one_class_falls_back_to_the_raw_probability() -> None:
    cal = fit_calibrator(np.full(100, 0.3), np.ones(100), "platt")
    assert cal.method == "none"
    np.testing.assert_array_equal(cal.apply(np.array([0.2, 0.7])), [0.2, 0.7])


def test_reliability_bins_cover_every_prediction() -> None:
    p, y = _labels(200_000, seed=2)
    table = reliability(p, y, bins=20)
    assert sum(r["n"] for r in table) == p.size
    assert all(r["lo"] <= r["mean_predicted"] <= r["hi"] for r in table)
    assert expected_calibration_error(p, y) < 0.01             # calibrated by construction
    assert expected_calibration_error(np.clip(p + 0.1, 0, 1), y) > 0.08
    uniform = reliability(p, y, bins=10, strategy="uniform")
    assert len(uniform) == 10


def test_calibrator_round_trip_is_exact() -> None:
    p, y = _labels(25_000, seed=3)
    for method in ("none", "platt", "isotonic"):
        cal = fit_calibrator(p, y, method, min_rows=1000)
        back = Calibrator.from_dict(cal.to_dict())
        np.testing.assert_array_equal(cal.apply(p[:1000]), back.apply(p[:1000]))
    with pytest.raises(ValueError, match="unknown calibration"):
        fit_calibrator(p, y, "beta")
