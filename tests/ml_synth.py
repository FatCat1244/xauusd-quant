"""Synthetic development data for the Prompt #10 tests.

Nothing here reads the real dataset. :func:`synthetic_ml_data` builds an
in-memory :class:`~xauusd_quant.ml.datasets.MLData` with three bars every
weekday from 2006 to the reserved start (2022-01-01), so the five configured
walk-forward blocks (2011-2021) all exist:

* ``sig`` drives the next returns (a direction signal), ``vol`` drives their
  scale (a volatility signal), ``noise_a`` / ``noise_b`` are pure noise and
  ``gappy`` is noise with 20 % missing values;
* the residual ``epsilon`` is an AR(1) (``phi = 0.9``), so the reversion
  labels have a known, learnable dependence on ``|epsilon|`` through ``resid``;
* every target column is purged at the end by its horizon, as
  :func:`~xauusd_quant.ml.datasets.load_ml_data` does at the reserved start.

:func:`small_config` is the project's ``config/ml.yaml`` with tiny boosters so
a unit fits in well under a second.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from xauusd_quant.ml.config import MLConfig, load_ml_config
from xauusd_quant.ml.datasets import MLData, build_context

RESERVED = date(2022, 1, 1)
HORIZONS = (1, 5, 20)
FAMILY = {"sig": "returns", "vol": "volatility", "resid": "regression", "noise_a": "returns",
          "noise_b": "autocorrelation", "gappy": "ou", "log_rv_20": "volatility",
          "spread_rel": "microstructure"}
STANDARD = ["sig", "vol", "resid", "noise_a", "gappy"]


def stamps(start: date = date(2006, 1, 2), end: date = RESERVED) -> pl.Series:
    out = []
    d = start
    while d < end:
        if d.weekday() < 5:
            out += [datetime(d.year, d.month, d.day, h) for h in (1, 9, 17)]
        d += timedelta(days=1)
    return pl.Series("timestamp", out, dtype=pl.Datetime("us"))


def _ar1(rng: np.random.Generator, n: int, phi: float) -> np.ndarray:
    from scipy.signal import lfilter

    return lfilter([np.sqrt(1 - phi * phi)], [1.0, -phi], rng.standard_normal(n))


def _ahead(v: np.ndarray, h: int) -> np.ndarray:
    out = np.full(v.size, np.nan)
    out[:-h] = v[h:]
    return out


def synthetic_ml_data(seed: int = 0) -> MLData:
    ts = stamps()
    n = ts.len()
    rng = np.random.default_rng(seed)
    sig = _ar1(rng, n, 0.3)
    vol = _ar1(rng, n, 0.98)
    eps = _ar1(rng, n, 0.9)
    sigma = 0.004 * np.exp(0.5 * vol)
    r_next = 0.25 * sig * sigma + sigma * rng.standard_normal(n)     # return of bar t+1
    gappy = rng.standard_normal(n)
    gappy[rng.random(n) < 0.2] = np.nan
    features = {"sig": sig, "vol": vol, "resid": eps, "noise_a": rng.standard_normal(n),
                "noise_b": _ar1(rng, n, 0.9), "gappy": gappy,
                "log_rv_20": np.log(sigma) + 0.1 * rng.standard_normal(n),
                "spread_rel": 0.5 + 0.1 * np.abs(rng.standard_normal(n))}
    features = {k: np.asarray(v, dtype=np.float32) for k, v in features.items()}
    csum = np.concatenate(([0.0], np.cumsum(r_next)))
    csq = np.concatenate(([0.0], np.cumsum(r_next ** 2)))
    raw: dict[str, np.ndarray] = {}
    for h in HORIZONS:
        ret = np.full(n, np.nan)
        ret[: n - h] = csum[h:n] - csum[: n - h]
        rv = np.full(n, np.nan)
        rv[: n - h] = np.sqrt((csq[h:n] - csq[: n - h]) / h)
        later = _ahead(eps, h)
        raw[f"target_return_{h}"] = ret
        raw[f"target_abs_return_{h}"] = np.abs(ret)
        raw[f"target_realized_vol_{h}"] = rv
        raw[f"target_residual_reduction_{h}"] = (np.abs(eps) - np.abs(later)) / sigma
        raw[f"target_residual_shrinks_{h}"] = np.where(np.isfinite(later),
                                                       (np.abs(later) < np.abs(eps)) * 1.0,
                                                       np.nan)
    # every outcome window ends inside the data: the last h rows of each column are NaN,
    # as load_ml_data leaves them at the reserved start
    registry = {name: {"name": name, "family": fam, "live_safe": True, "feature_version": "fv",
                       "min_history": 0} for name, fam in FAMILY.items()}
    manifests = {"standard": list(STANDARD), "minimal": ["sig", "vol", "resid"],
                 "extended": [*STANDARD, "noise_b"],
                 "target_reversion": ["resid", "vol", "noise_a"],
                 "target_direction": ["sig", "noise_a"],
                 "target_volatility": ["vol", "noise_b"],
                 "target_magnitude": ["vol", "sig"]}
    data = MLData(timeframe="1h", timestamps=ts, reserved_start=RESERVED, features=features,
                  manifests=manifests, manifest_ids={k: f"FS_{k.upper()}" for k in manifests},
                  manifest_hashes={k: f"hash-{k}" for k in manifests}, registry=registry,
                  targets_raw=raw, epsilon=eps.astype(np.float64),
                  sigma=sigma.astype(np.float64),
                  versions={"tick_dataset_version": "ticks-synthetic",
                            "factory_version": "factory-synthetic",
                            "target_version": "targets-synthetic"})
    data.context = build_context(data)
    return data


def small_config(root: Path | None = None) -> MLConfig:
    """``config/ml.yaml`` with tiny models (and paths under *root*)."""
    cfg = load_ml_config()
    models = {k: dict(v) for k, v in cfg.models.items()}
    models["xgboost"].update(n_estimators=40, min_child_weight=5, early_stopping_rounds=10)
    models["lightgbm"].update(n_estimators=40, min_data_in_leaf=20, early_stopping_rounds=10)
    models["catboost"].update(iterations=40, early_stopping_rounds=10)
    models["random_forest"].update(n_estimators=8, min_samples_leaf=20, max_samples=0.5)
    out = replace(cfg, models=models, threads=1,
                  calibration={**cfg.calibration, "isotonic_min_rows": 300},
                  explain={**cfg.explain, "shap_rows_per_fold": 400, "top_features": 2,
                           "ale_bins": 5})
    if root is not None:
        out = replace(out, results_path=root / "ml_research", models_path=root / "models",
                      ledger_path=root / "ledger.parquet",
                      registry_path=root / "model_registry.yaml")
    return out
