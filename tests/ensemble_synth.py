"""Synthetic inputs for the Prompt #11 ensemble tests.

Nothing here reads the real dataset.

* :func:`synthetic_pair`: a :class:`~xauusd_quant.ensemble.data.PairPredictions` with
  five contiguous blocks of consecutive bars, a constant baseline and constituents of
  known quality - two that each see half of a hidden signal (complementary), one
  weaker one, and pure noise - for classification or regression.
* :func:`frozen_ensemble`: real frozen constituents (tiny logistic / LightGBM /
  ridge models on features computed by the factory engines from synthetic bars),
  their ``MODEL_SPEC`` files and artifacts, and a frozen ``ENSEMBLE_SPEC`` over them -
  the inputs of the inference, serialization and streaming tests.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from xauusd_quant.ensemble.data import PairPredictions
from xauusd_quant.ensemble.meta_model import fit_meta_model
from xauusd_quant.ensemble.registry import constituent_id, ensemble_id, freeze_ensemble_spec
from xauusd_quant.ml.calibration import fit_calibrator
from xauusd_quant.ml.models import make_model, preprocessing_kind
from xauusd_quant.ml.preprocessing import Preprocessor
from xauusd_quant.ml.registry import freeze_spec, load_frozen_spec, save_artifact

BLOCKS = ("wf1_2011_2012", "wf2_2013_2014", "wf3_2015_2017", "wf4_2018_2019", "wf5_2020_2021")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def synthetic_pair(task: str = "classification", *, per_block: int = 3000, horizon: int = 5,
                   seed: int = 0, start_row: int = 10_000) -> PairPredictions:
    """Out-of-sample predictions of four constituents on five blocks of consecutive bars.

    ``a|half`` and ``b|half`` each see one half of the hidden signal ``s = u + v`` (their
    errors are complementary), ``c|weak`` sees a quarter of it, ``n|noise`` nothing.
    """
    rng = np.random.default_rng(seed)
    n = per_block * len(BLOCKS)
    u, v = rng.normal(size=n), rng.normal(size=n)
    s = u + v
    if task == "classification":
        y = (rng.random(n) < _sigmoid(1.2 * s)).astype(np.float64)
        preds = {"a|half": _sigmoid(0.9 * u), "b|half": _sigmoid(0.9 * v),
                 "c|weak": _sigmoid(0.3 * (u + v) + 0.6 * rng.normal(size=n)),
                 "n|noise": _sigmoid(0.4 * rng.normal(size=n))}
    else:
        y = s + rng.normal(size=n)
        preds = {"a|half": 0.9 * u, "b|half": 0.9 * v,
                 "c|weak": 0.3 * s + 0.6 * rng.normal(size=n),
                 "n|noise": 0.4 * rng.normal(size=n)}
    block = np.repeat(np.arange(len(BLOCKS), dtype=np.int16), per_block)
    base = np.empty(n)
    for k in range(len(BLOCKS)):                       # the block's "training mean"
        m = block == k
        base[m] = float(np.mean(y[:max(1, int(m.argmax()))])) if k else float(np.mean(y))
    start = datetime(2011, 1, 3)
    ts = pl.Series("timestamp", [start + timedelta(hours=i) for i in range(n)],
                   dtype=pl.Datetime("us"))
    raw = {"constant|standard": base, **{k: v.astype(np.float64) for k, v in preds.items()}}
    cal = dict(raw) if task == "classification" else {}
    cal.pop("constant|standard", None)
    models = {m: {"family": m.split("|")[0], "feature_set": m.split("|")[1],
                  "feature_set_id": f"FS_{m.split('|')[1].upper()}", "blocks": list(BLOCKS)}
              for m in raw}
    return PairPredictions(
        timeframe="1h", target="mean_reversion" if task == "classification" else "future_return",
        horizon=horizon, task=task, rows=np.arange(start_row, start_row + n, dtype=np.int64),
        timestamps=ts, block=block, block_names=list(BLOCKS), label=y.astype(np.float64),
        label_raw=y.astype(np.float64), base=base, raw=raw, calibrated=cal, models=models,
        versions={"tick_dataset_version": "ticks-synthetic"}, calibrated_inputs=True)


# ---------------------------------------------------------------------------
# Real frozen constituents on synthetic bars (inference / streaming tests)
# ---------------------------------------------------------------------------
N_BARS = 2600
NAMES = ["ret_1", "ret_5", "ret_z_64", "log_rv_20", "log_rv_64", "log_ewma_vol_94", "tod_sin",
         "session_london"]
STREAM_ROWS = [2000 + 37 * k for k in range(8)]


@cache
def configs() -> tuple[Any, ...]:
    from xauusd_quant.features import load_regression_config
    from xauusd_quant.features.factory_config import load_features_config
    from xauusd_quant.features.spectral_config import load_spectral_config
    from xauusd_quant.features.wavelet_config import load_wavelet_config
    from xauusd_quant.models import load_ou_config
    from xauusd_quant.research.config import load_research_config

    return (load_features_config(), load_regression_config(), load_ou_config(),
            load_spectral_config(), load_wavelet_config(), load_research_config())


@cache
def bars_and_features() -> tuple[Any, dict[str, np.ndarray], dict[str, dict[str, Any]],
                                  np.ndarray]:
    from feature_synth import synthetic_bars
    from xauusd_quant.features.factory import compute_families
    from xauusd_quant.features.joins import EngineProvider
    from xauusd_quant.features.registry import build_registry

    fcfg, regression, ou, spectral, wavelet, research = configs()
    bars = synthetic_bars(N_BARS)
    provider = EngineProvider(bars=bars, regression_config=regression, spectral_config=spectral,
                              wavelet_config=wavelet)
    values, _ = compute_families(bars, provider, fcfg, ou, research,
                                 families=("returns", "volatility", "time"), log=False,
                                 collect=False)
    registry = {s.name: s.to_dict() for s in build_registry(fcfg, "1h", 3600.0)}
    stored = {n: np.asarray(values[n], dtype=np.float32) for n in NAMES}
    close = np.asarray(bars.close, dtype=np.float64)
    nxt = np.full(N_BARS, np.nan)
    nxt[:-1] = np.log(close[1:] / close[:-1])
    return bars, stored, registry, nxt


def _constituent(root: Path, family: str, task: str, features: list[str], *,
                 version: int = 1, params: dict[str, Any] | None = None
                 ) -> dict[str, Any]:
    _, stored, _, nxt = bars_and_features()
    x = np.column_stack([stored[n] for n in features]).astype(np.float32)
    y = (nxt > 0).astype(np.float64) if task == "classification" else np.abs(nxt)
    fit, cal_rows = np.arange(200, 1700), np.arange(1720, 1980)
    kind = preprocessing_kind(family)
    pre = Preprocessor.fit(x[fit], features, kind=kind)
    p = params if params is not None else (
        {"n_estimators": 30, "min_data_in_leaf": 20} if family == "lightgbm" else {})
    model = make_model(family, task, p, seed=3, threads=1)
    model.fit(pre.transform(x[fit], features), y[fit])
    method = "platt" if task == "classification" else "none"
    cal = fit_calibrator(model.predict(pre.transform(x[cal_rows], features)), y[cal_rows], method)
    target = "mean_reversion" if task == "classification" else "future_return"
    body = {"spec_id": constituent_id(family, "standard", target, "1h", 5, version),
            "family": family, "target": target, "target_definition": {"task": task},
            "horizon": 5, "timeframe": "1h", "feature_set": "standard",
            "feature_set_id": "FS_SYNTH", "feature_set_hash": "hash-synth",
            "features": list(features), "params": p,
            "preprocessing": {"kind": kind, "clip": 8.0}, "calibration": method,
            "training_policy": {"scheme": "expanding"}, "seed": 3,
            "dataset_versions": {"tick_dataset_version": "ticks-synthetic"}}
    path = freeze_spec(body, root / "constituents")
    spec = load_frozen_spec(path)
    save_artifact(root / "models" / spec["spec_id"], model=model, preprocessor=pre,
                  calibrator=cal, spec=spec, extra={})
    return {"name": f"{family}|standard", "model_id": spec["spec_id"],
            "spec_hash": spec["content_hash"], "family": family, "feature_set": "standard",
            "feature_set_id": "FS_SYNTH", "feature_set_hash": "hash-synth",
            "features": list(features), "calibration": method}


def frozen_ensemble(root: Path, *, task: str = "classification", method: str = "weights",
                    families: tuple[str, ...] | None = None) -> tuple[Path, dict[str, Any]]:
    """A frozen ensemble over real constituents; (spec path, the members' descriptions)."""
    fams = families or (("logistic_l2", "lightgbm") if task == "classification"
                        else ("ridge", "lightgbm"))
    members = [_constituent(root, f, task, NAMES) for f in fams]
    comb: dict[str, Any]
    if method == "weights":
        comb = {"kind": "weights", "weights": [0.25, 0.75][:len(members)]}
    elif method == "simple_average":
        comb = {"kind": "simple_average"}
    elif method == "single":
        comb = {"kind": "single", "member": members[1]["name"]}
    elif method == "stacking":
        rng = np.random.default_rng(5)
        xs = rng.random((500, len(members))) * 0.6 + 0.2 if task == "classification" \
            else rng.normal(size=(500, len(members)))
        ys = (rng.random(500) < xs.mean(axis=1)).astype(float) if task == "classification" \
            else xs.sum(axis=1) + rng.normal(size=500)
        meta = fit_meta_model(xs, ys, [m["name"] for m in members],
                              constituents=len(members), task=task)
        comb = {"kind": "stacking", "meta_model": meta.to_dict()}
    elif method == "regime":
        comb = {"kind": "state_weights", "states": "regime",
                "state_inputs": ["regime_p0", "regime_p1"],
                "weights": [[0.9, 0.1], [0.2, 0.8]], "fallback": [0.5, 0.5]}
    else:
        raise ValueError(method)
    target = "mean_reversion" if task == "classification" else "future_return"
    spec = {"spec_id": ensemble_id(target, "1h", 5), "timeframe": "1h", "target": target,
            "target_definition": {"task": task}, "horizon": 5, "task": task,
            "method": method, "role": "candidate", "input_kind": "test",
            "constituents": members, "combination": comb, "calibration": {"method": "none"},
            "companions": {"simple_average": [m["name"] for m in members],
                           "best_individual": members[0]["name"]},
            "context_inputs": [], "training_policy": {}, "refit_policy": "frozen",
            "dataset_versions": {"tick_dataset_version": "ticks-synthetic"},
            "missing_constituent_policy": "fail_closed"}
    path = freeze_ensemble_spec(spec, root / "frozen")
    return path, {"members": members, "models": root / "models",
                  "constituents": root / "constituents"}


def live_manifest(members: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {"FS_SYNTH": {"hash": "hash-synth", "features": list(members[0]["features"])}}
