"""Streaming ensemble predictions equal batch predictions (Prompt #11, Step 67).

A frozen ensemble is replayed bar by bar: the features are recomputed once per bar
from a rolling buffer of bars up to ``t`` only, every constituent runs its frozen
preprocessing, model and calibrator, the frozen combination merges them and the
result must reproduce the batch prediction at ``t``. A stored feature value that
differs from its causal recomputation must make the check fail.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ensemble_synth import STREAM_ROWS, bars_and_features, configs, frozen_ensemble, live_manifest
from xauusd_quant.ensemble.inference import EnsembleModel
from xauusd_quant.ensemble.streaming import stream_ensemble


def _run(tmp_path: Path, stored: dict[str, np.ndarray], *, task: str = "classification",
         method: str = "weights", context: dict[str, np.ndarray] | None = None) -> dict:
    path, info = frozen_ensemble(tmp_path, task=task, method=method)
    model = EnsembleModel.load(path, models_path=info["models"],
                               constituent_dir=info["constituents"],
                               live_manifests=live_manifest(info["members"]))
    fcfg, regression, ou, spectral, wavelet, research = configs()
    bars, _, registry, _ = bars_and_features()
    return stream_ensemble(model, registry, bars, stored, STREAM_ROWS, buffer=700, fcfg=fcfg,
                           ou=ou, research=research, regression=regression, spectral=spectral,
                           wavelet=wavelet, context=context)


@pytest.mark.parametrize(("task", "method"), [("classification", "weights"),
                                              ("classification", "stacking"),
                                              ("regression", "simple_average")])
def test_streaming_equals_batch(tmp_path: Path, task: str, method: str) -> None:
    _, stored, _, _ = bars_and_features()
    out = _run(tmp_path, stored, task=task, method=method)
    assert out["passed"], out
    assert out["steps"] == len(STREAM_ROWS) and out["mismatches"] == 0
    assert out["max_abs_feature_difference"] < 1e-5
    assert out["max_abs_constituent_difference"] < 1e-4


def test_state_weights_stream_with_the_same_context(tmp_path: Path) -> None:
    _, stored, _, _ = bars_and_features()
    n = len(STREAM_ROWS)
    rng = np.random.default_rng(0)
    p0 = rng.random(n)
    out = _run(tmp_path, stored, method="regime",
               context={"regime_p0": p0, "regime_p1": 1.0 - p0})
    assert out["passed"], out


def test_a_stored_value_that_differs_from_its_causal_recomputation_fails(tmp_path: Path) -> None:
    _, stored, _, _ = bars_and_features()
    bad = {k: v.copy() for k, v in stored.items()}
    bad["log_rv_20"][STREAM_ROWS[3]] += 0.5               # computed some other way
    out = _run(tmp_path, bad)
    assert not out["passed"] and out["mismatches"] == 1
    assert out["max_abs_feature_difference"] >= 0.49
