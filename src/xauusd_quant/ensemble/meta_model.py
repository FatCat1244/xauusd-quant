r"""Second-level models of stacking (Prompt #11, Steps 21-26).

Deliberately simple (Step 22): **logistic regression** for a classification
target, on the logits of the constituents' probabilities, and **ridge
regression** for a continuous target, on their expected values - never a
booster. Inputs are standardised with the centre and scale of the rows the
meta-model is fitted on, and those rows are always out-of-sample predictions of
the constituents (Rule 2; :mod:`.stacking` builds them).

Context inputs (Step 26: volatility percentile, regime entropy / confidence,
model disagreement, OOD score - a handful, never the feature matrix) are
standardised the same way; a missing context value is set to the fitted centre
(it then contributes nothing). A missing *constituent* prediction is never
filled: the row gets no prediction.

A fitted :class:`MetaModel` serialises to plain JSON (:meth:`to_dict`) and a
reloaded one reproduces its predictions exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["MetaModel", "fit_meta_model"]

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)
    return np.log(q / (1.0 - q))


@dataclass
class MetaModel:
    kind: str                          # logistic_regression | ridge
    inputs: list[str]                  # constituent names, then context names
    constituents: int                  # the first `constituents` inputs are model predictions
    logit_inputs: bool                 # constituent probabilities enter as logits
    center: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    params: dict[str, Any] = field(default_factory=dict)

    def design(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(standardised design, rows with every constituent prediction present)."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.inputs):
            raise ValueError(f"meta-model expects {len(self.inputs)} inputs, got {x.shape}")
        z = x.copy()
        k = self.constituents
        valid = np.isfinite(z[:, :k]).all(axis=1)
        if self.logit_inputs:
            z[:, :k] = _logit(z[:, :k])
        ctx = z[:, k:]
        ctx[~np.isfinite(ctx)] = np.broadcast_to(self.center[k:], ctx.shape)[~np.isfinite(ctx)]
        z[:, k:] = ctx
        z = (z - self.center) / self.scale
        z[~valid] = 0.0
        return z, valid

    def predict(self, x: np.ndarray) -> np.ndarray:
        z, valid = self.design(x)
        eta = z @ self.coef + self.intercept
        out = 1.0 / (1.0 + np.exp(-eta)) if self.kind == "logistic_regression" else eta
        return np.where(valid, out, np.nan)

    def weight_shares(self) -> dict[str, float]:
        """|coefficient| shares of the constituent inputs (on the standardised scale)."""
        c = np.abs(self.coef[:self.constituents])
        total = float(c.sum())
        return {n: (float(v) / total if total > 0 else 0.0)
                for n, v in zip(self.inputs[:self.constituents], c, strict=True)}

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "inputs": list(self.inputs), "constituents": self.constituents,
                "logit_inputs": self.logit_inputs, "center": self.center.tolist(),
                "scale": self.scale.tolist(), "coef": self.coef.tolist(),
                "intercept": float(self.intercept), "params": self.params}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MetaModel:
        return cls(kind=str(d["kind"]), inputs=[str(s) for s in d["inputs"]],
                   constituents=int(d["constituents"]), logit_inputs=bool(d["logit_inputs"]),
                   center=np.asarray(d["center"], dtype=np.float64),
                   scale=np.asarray(d["scale"], dtype=np.float64),
                   coef=np.asarray(d["coef"], dtype=np.float64),
                   intercept=float(d["intercept"]), params=dict(d.get("params") or {}))


def _evenly(n: int, size: int) -> np.ndarray:
    if n <= size:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, size).astype(np.int64))


def fit_meta_model(x: np.ndarray, y: np.ndarray, names: list[str], *, constituents: int,
                   task: str, logistic_c: float = 1.0, ridge_alpha: float = 1.0,
                   max_rows: int | None = None, seed: int = 0) -> MetaModel:
    """Fit on out-of-sample rows *x* (constituent predictions, then context) and labels *y*."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x[:, :constituents]).all(axis=1) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if max_rows and x.shape[0] > max_rows:
        sel = _evenly(x.shape[0], int(max_rows))
        x, y = x[sel], y[sel]
    if x.shape[0] < 100:
        raise ValueError(f"too few out-of-sample rows to fit a meta-model ({x.shape[0]})")
    logit_inputs = task == "classification"
    z = x.copy()
    if logit_inputs:
        z[:, :constituents] = _logit(z[:, :constituents])
    center = np.array([np.nanmedian(z[:, j]) if np.isfinite(z[:, j]).any() else 0.0
                       for j in range(z.shape[1])])
    filled = np.where(np.isfinite(z), z, center[None, :])
    scale = filled.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    zs = (filled - center) / scale
    if task == "classification":
        from sklearn.linear_model import LogisticRegression

        est = LogisticRegression(C=float(logistic_c), l1_ratio=0.0, solver="lbfgs",
                                 max_iter=1000, random_state=seed)
        est.fit(zs, y.astype(np.int64))
        coef, intercept, kind = np.ravel(est.coef_), float(est.intercept_[0]), \
            "logistic_regression"
    else:
        from sklearn.linear_model import Ridge

        est = Ridge(alpha=float(ridge_alpha)).fit(zs, y)
        coef, intercept, kind = np.ravel(est.coef_), float(est.intercept_), "ridge"
    return MetaModel(kind=kind, inputs=list(names), constituents=int(constituents),
                     logit_inputs=logit_inputs, center=center, scale=scale,
                     coef=coef.astype(np.float64), intercept=intercept,
                     params={"logistic_C": logistic_c, "ridge_alpha": ridge_alpha,
                             "rows": int(x.shape[0])})
