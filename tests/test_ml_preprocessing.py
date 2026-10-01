"""Fold-specific preprocessing (Prompt #10, Steps 15-16, 73).

The preprocessor is fitted on a fold's fitting rows only, refuses a feature
order it was not fitted with, and treats missing values the same way every
time - in a batch, row by row, and after a JSON round trip.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml_synth import small_config, synthetic_ml_data
from xauusd_quant.ml.preprocessing import FeatureOrderError, Preprocessor
from xauusd_quant.ml.splits import walk_forward_folds
from xauusd_quant.ml.training import UnitSpec, run_unit

NAMES = ["a", "b", "c"]


def _x(seed: int = 0, n: int = 4000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 3)) * [1.0, 10.0, 0.1] + [0.0, 5.0, -1.0]
    x[rng.random(n) < 0.1, 1] = np.nan
    return x.astype(np.float32)


def test_parameters_come_from_the_fitting_rows_only() -> None:
    x = _x()
    fit, val = x[:3000], x[3000:].copy()
    pre = Preprocessor.fit(fit, NAMES, kind="linear")
    before = pre.to_dict()
    val[:, 0] += 1000.0                                  # a shifted validation block
    pre.transform(val, NAMES)
    assert pre.to_dict() == before
    f64 = fit.astype(np.float64)
    assert pre.medians is not None and pre.centers is not None
    np.testing.assert_allclose(pre.medians[1], np.nanmedian(f64[:, 1]))
    filled = np.where(np.isfinite(f64), f64, pre.medians)
    np.testing.assert_allclose(pre.centers, np.median(filled, axis=0))
    assert pre.indicators == (1,)                        # only b was missing in training


def test_a_unit_fits_its_preprocessor_on_its_own_fitting_rows() -> None:
    data = synthetic_ml_data()
    cfg = small_config()
    fold = walk_forward_folds(data.timestamps, cfg.walk_forward, horizon=5)[1]
    spec = UnitSpec(target="future_volatility", horizon=5, family="ridge",
                    feature_set="standard", features=data.feature_set("standard"), fold=fold,
                    params=cfg.model_params("ridge"), seed=cfg.random_seed)
    res = run_unit(data, spec, cfg.targets["future_volatility"], cfg, keep_model=True)
    y = data.target(cfg.targets["future_volatility"], 5, cfg.log_floor).y
    rows = np.arange(*fold.fit)
    rows = rows[np.isfinite(y[rows])]
    x = data.design(spec.features)[rows].astype(np.float64)
    assert res.preprocessor is not None and res.preprocessor.medians is not None
    np.testing.assert_allclose(res.preprocessor.medians, np.nanmedian(x, axis=0))
    assert res.info["fit_rows"] == rows.size


def test_feature_order_mismatch_fails_clearly() -> None:
    x = _x()
    for kind in ("linear", "tree_imputed", "tree_native"):
        pre = Preprocessor.fit(x, NAMES, kind=kind)
        with pytest.raises(FeatureOrderError, match="same names, different order"):
            pre.transform(x[:, [1, 0, 2]], ["b", "a", "c"])
        with pytest.raises(FeatureOrderError, match=r"missing \['c'\]"):
            pre.transform(x[:, :2], ["a", "b"])
        with pytest.raises(FeatureOrderError, match=r"unexpected \['d'\]"):
            pre.transform(np.column_stack([x, x[:, 0]]), [*NAMES, "d"])
    with pytest.raises(FeatureOrderError, match="2 columns for 3"):
        Preprocessor.fit(x[:, :2], NAMES, kind="linear")


@pytest.mark.parametrize("kind", ["linear", "tree_imputed", "tree_native"])
def test_missing_values_are_handled_deterministically(kind: str) -> None:
    x = _x()
    pre = Preprocessor.fit(x[:3000], NAMES, kind=kind)
    later = x[3000:].copy()
    later[:5, 0] = np.nan                                # missing only after training
    a = pre.transform(later, NAMES)
    b = pre.transform(later, NAMES)
    rows = np.vstack([pre.transform(later[i:i + 1], NAMES) for i in range(50)])
    again = Preprocessor.from_dict(pre.to_dict()).transform(later, NAMES)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a[:50], rows)          # streaming = batch, row by row
    np.testing.assert_array_equal(a, again)
    if kind == "tree_native":
        assert np.isnan(a[:5, 0]).all() and a.shape[1] == 3
        return
    assert np.isfinite(a).all()
    assert a.shape[1] == 4 and pre.output_names()[-1] == "missing__b"
    miss_b = ~np.isfinite(later[:, 1])
    np.testing.assert_array_equal(a[:, 3], miss_b.astype(a.dtype))
    assert pre.medians is not None
    if kind == "tree_imputed":
        np.testing.assert_allclose(a[:5, 0], pre.medians[0], rtol=1e-6)


def test_extreme_values_are_clipped_after_scaling() -> None:
    x = _x()
    pre = Preprocessor.fit(x, NAMES, kind="linear", clip=8.0)
    wild = x[:3].copy()
    wild[0, 0] = 1e9
    wild[1, 2] = -1e9
    out = pre.transform(wild, NAMES)
    assert out[0, 0] == 8.0 and out[1, 2] == -8.0
    assert np.abs(out[:, :3]).max() <= 8.0
