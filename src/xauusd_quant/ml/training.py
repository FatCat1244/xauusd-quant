r"""One research unit: fit a model on a fold, calibrate, predict the block (Prompt #10).

A *unit* is (target, horizon, model family, feature list, fold, variant). It

1. takes the fold's fitting rows, inner slice and validation block (rows
   without a label dropped);
2. fits the fold-specific :class:`~.preprocessing.Preprocessor` on the fitting
   rows only and freezes it;
3. fits the model on the fitting rows (boosters stop early on the inner slice);
4. fits the calibrators on the inner slice (classification);
5. predicts the validation block - raw and calibrated - and returns those
   out-of-sample predictions with their row numbers.

Variants (all recorded, all counted as trials):

* ``weighting``: time-decay weights :math:`w_t = 2^{-age_t / half\text{-}life}`
  (age from the end of the fitting rows);
* ``null="shift"``: the training labels circularly shifted inside the training
  span by at least 10 % of it - features and labels no longer belong together,
  the validation labels stay real (Step 67);
* ``noise``: extra AR(1) noise columns appended to the design (Step 68);
* ``permute_features``: every feature column permuted independently in the
  training rows (Step 69);
* ``tree_missing="imputed"``: boosters on imputed features instead of NaN.

Explanations (SHAP from the boosters' own TreeSHAP, permutation importance, ALE
curves) are computed on a sample of the validation block when asked.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from .calibration import Calibrator, fit_calibrator
from .config import MLConfig, TargetSpec
from .datasets import MLData, TargetArrays
from .evaluation import classification_metrics, regression_metrics
from .models import Model, make_model, preprocessing_kind
from .preprocessing import Preprocessor
from .splits import Fold

__all__ = ["UnitResult", "UnitSpec", "final_fold", "run_unit", "time_decay_weights",
           "train_final"]

_SECONDS_PER_YEAR = 365.25 * 86400.0


@dataclass
class UnitSpec:
    target: str
    horizon: int
    family: str
    feature_set: str                   # a manifest name or a variant label
    features: list[str]
    fold: Fold
    params: dict[str, Any]
    variant: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    explain: bool = False

    def key(self) -> str:
        v = ",".join(f"{k}={self.variant[k]}" for k in sorted(self.variant)) or "base"
        return (f"{self.target}|h{self.horizon}|{self.family}|{self.feature_set}|{v}|"
                f"{self.fold.name}")


@dataclass
class UnitResult:
    spec: UnitSpec
    predictions: pl.DataFrame          # row, prediction (+ cal_platt, cal_isotonic)
    metrics: dict[str, Any]
    info: dict[str, Any]
    explain: dict[str, Any] = field(default_factory=dict)
    model: Model | None = None
    preprocessor: Preprocessor | None = None
    calibrators: dict[str, Calibrator] = field(default_factory=dict)


def time_decay_weights(timestamps: pl.Series, rows: np.ndarray, half_life_years: float
                       ) -> np.ndarray:
    ts = timestamps.gather(rows).dt.epoch("s").to_numpy().astype(np.float64)
    age = (ts.max() - ts) / _SECONDS_PER_YEAR
    return np.power(2.0, -age / half_life_years)


def _label_rows(y: np.ndarray, rng: tuple[int, int]) -> np.ndarray:
    idx = np.arange(rng[0], rng[1], dtype=np.int64)
    return idx[np.isfinite(y[idx])]


def _noise_columns(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((n, k), dtype=np.float32)
    phis = [0.0, 0.5, 0.9, 0.99, 0.999]
    for j in range(k):
        phi = phis[j % len(phis)]
        e = rng.standard_normal(n)
        if phi == 0.0:
            out[:, j] = e
            continue
        from scipy.signal import lfilter

        out[:, j] = lfilter([np.sqrt(1 - phi * phi)], [1.0, -phi], e)
    return out


def run_unit(data: MLData, spec: UnitSpec, tspec: TargetSpec, cfg: MLConfig, *,
             target: TargetArrays | None = None, keep_model: bool = False) -> UnitResult:
    started = time.perf_counter()
    ta = target or data.target(tspec, spec.horizon, cfg.log_floor)
    y_all = ta.y
    fold = spec.fold
    fit_rows = _label_rows(y_all, fold.fit)
    inner_rows = _label_rows(y_all, fold.inner)
    val_rows = _label_rows(y_all, fold.validate)
    if fit_rows.size < 1000 or val_rows.size < 100:
        raise ValueError(f"{spec.key()}: too few labelled rows ({fit_rows.size} fit, "
                         f"{val_rows.size} validation)")
    names = list(spec.features)
    x_all = data.design(names, key=spec.feature_set if spec.feature_set in data.manifests
                        else None)
    x_fit, x_inner, x_val = x_all[fit_rows], x_all[inner_rows], x_all[val_rows]
    y_fit, y_inner, y_val = y_all[fit_rows], y_all[inner_rows], y_all[val_rows]
    v = spec.variant
    if v.get("null") == "shift":
        span = np.concatenate([fit_rows, inner_rows])
        ys = y_all[span]
        rng = np.random.default_rng(spec.seed + 101 * int(v.get("repeat", 0)))
        k = int(rng.integers(int(0.1 * span.size), int(0.9 * span.size)))
        shifted = np.roll(ys, k)
        y_fit, y_inner = shifted[:fit_rows.size], shifted[fit_rows.size:]
    if v.get("permute_features"):
        rng = np.random.default_rng(spec.seed + 7)
        x_fit = x_fit.copy()
        x_inner = x_inner.copy()
        for j in range(x_fit.shape[1]):
            x_fit[:, j] = x_fit[rng.permutation(x_fit.shape[0]), j]
            x_inner[:, j] = x_inner[rng.permutation(x_inner.shape[0]), j]
    if int(v.get("noise", 0)) > 0:
        k = int(v["noise"])
        noise = _noise_columns(data.n, k, spec.seed + 13)
        x_fit = np.concatenate([x_fit, noise[fit_rows]], axis=1)
        x_inner = np.concatenate([x_inner, noise[inner_rows]], axis=1)
        x_val = np.concatenate([x_val, noise[val_rows]], axis=1)
        names = [*names, *[f"noise_{j}" for j in range(k)]]
    weight = None
    if v.get("weighting"):
        weight = time_decay_weights(data.timestamps, fit_rows, float(v["weighting"]))
    kind = preprocessing_kind(spec.family, str(v.get("tree_missing",
                                                     cfg.preprocessing.get("tree_missing",
                                                                           "native"))))
    pre = Preprocessor.fit(x_fit, names, kind=kind,
                           clip=float(cfg.preprocessing.get("clip", 8.0)))
    xf = pre.transform(x_fit, names)
    xi = pre.transform(x_inner, names)
    xv = pre.transform(x_val, names)
    del x_fit
    model = make_model(spec.family, tspec.task, spec.params, seed=spec.seed,
                       threads=cfg.threads)
    model.fit(xf, y_fit, weight=weight, x_inner=xi if xi.shape[0] else None,
              y_inner=y_inner if xi.shape[0] else None)
    p_val = model.predict(xv)
    cols: dict[str, Any] = {"row": val_rows.astype(np.int32), "prediction": p_val.astype(np.float32)}
    calibrators: dict[str, Calibrator] = {}
    base = float(np.mean(y_fit))
    if tspec.is_classification:
        p_inner = model.predict(xi) if xi.shape[0] else np.zeros(0)
        for method in cfg.calibration.get("methods", ("none", "platt", "isotonic")):
            if method == "none":
                continue
            cal = fit_calibrator(p_inner, y_inner, method,
                                 min_rows=int(cfg.calibration.get("isotonic_min_rows", 20000)))
            calibrators[method] = cal
            cols[f"cal_{method}"] = cal.apply(p_val).astype(np.float32)
        metrics = {"raw": classification_metrics(p_val, y_val, base_rate=base)}
        for method in calibrators:
            metrics[method] = classification_metrics(cols[f"cal_{method}"].astype(np.float64),
                                                     y_val, base_rate=base)
    else:
        metrics = {"raw": regression_metrics(p_val, y_val, y_raw=ta.y_raw[val_rows],
                                             base_value=base)}
    info = {**model.info, "fit_rows": int(fit_rows.size), "inner_rows": int(inner_rows.size),
            "validation_rows": int(val_rows.size), "base": base,
            "preprocessing": kind, "design_columns": len(pre.output_names()),
            "seconds": time.perf_counter() - started}
    explain: dict[str, Any] = {}
    if spec.explain:
        from .explainability import explain_unit

        explain = explain_unit(model, pre, xv, y_val, val_rows, names, data, tspec,
                               cfg, seed=spec.seed)
    return UnitResult(spec=spec, predictions=pl.DataFrame(cols), metrics=metrics, info=info,
                      explain=explain, model=model if keep_model else None,
                      preprocessor=pre if keep_model else None,
                      calibrators=calibrators if keep_model else {})


def final_fold(data: MLData, cfg: MLConfig, *, horizon: int, scheme: str,
               rolling_years: int) -> Fold:
    """The whole development span as one fold: fit rows, purge, inner slice, end of data.

    The 'validation' range is empty: the final model is scored only by
    ``xq ml-final-test`` on the reserved period.
    """
    from datetime import date as _date

    from .splits import first_row_at

    n = data.n
    gap = horizon + cfg.walk_forward.embargo_bars
    span_hi = n - horizon                               # the last labelled row + 1
    span_lo = 0
    if scheme == "rolling":
        end = data.reserved_start
        span_lo = first_row_at(data.timestamps,
                               _date(end.year - rolling_years, end.month, end.day))
    n_inner = max(1, int(round(cfg.walk_forward.inner_fraction * (span_hi - span_lo))))
    inner = (span_hi - n_inner, span_hi)
    return Fold(index=-1, name=f"final_{scheme}", fit=(span_lo, inner[0] - gap), inner=inner,
                validate=(n, n), horizon=horizon, embargo=cfg.walk_forward.embargo_bars,
                scheme=scheme)


def train_final(data: MLData, spec: dict[str, Any], tspec: TargetSpec, cfg: MLConfig
                ) -> tuple[Model, Preprocessor, Calibrator, dict[str, Any]]:
    """Fit a frozen spec on its development training window (no reserved row is present)."""
    policy = spec["training_policy"]
    h = int(spec["horizon"])
    fold = final_fold(data, cfg, horizon=h, scheme=policy.get("scheme", "expanding"),
                      rolling_years=int(policy.get("rolling_years", 5)))
    ta = data.target(tspec, h, cfg.log_floor)
    fit_rows = _label_rows(ta.y, fold.fit)
    inner_rows = _label_rows(ta.y, fold.inner)
    names = list(spec["features"])
    x_all = data.design(names)
    x_fit, x_inner = x_all[fit_rows], x_all[inner_rows]
    y_fit, y_inner = ta.y[fit_rows], ta.y[inner_rows]
    weight = None
    if policy.get("weighting"):
        weight = time_decay_weights(data.timestamps, fit_rows, float(policy["weighting"]))
    pre = Preprocessor.fit(x_fit, names, kind=spec["preprocessing"]["kind"],
                           clip=spec["preprocessing"].get("clip"))
    xf, xi = pre.transform(x_fit, names), pre.transform(x_inner, names)
    model = make_model(spec["family"], tspec.task, spec["params"], seed=int(spec["seed"]),
                       threads=cfg.threads)
    model.fit(xf, y_fit, weight=weight, x_inner=xi, y_inner=y_inner)
    method = spec.get("calibration", "none") if tspec.is_classification else "none"
    cal = (fit_calibrator(model.predict(xi), y_inner, method,
                          min_rows=int(cfg.calibration.get("isotonic_min_rows", 20000)))
           if method != "none" else Calibrator("none"))
    info = {"fit_rows": int(fit_rows.size), "inner_rows": int(inner_rows.size),
            "fit_first": str(data.timestamps[int(fit_rows[0])]),
            "fit_last": str(data.timestamps[int(fit_rows[-1])]),
            "inner_first": str(data.timestamps[int(inner_rows[0])]),
            "inner_last": str(data.timestamps[int(inner_rows[-1])]),
            "base": float(np.mean(y_fit)), **model.info}
    return model, pre, cal, info
