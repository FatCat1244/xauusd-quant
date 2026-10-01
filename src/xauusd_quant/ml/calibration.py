r"""Probability calibration (Prompt #10, Steps 26-29).

Calibrators map a model's raw probability to a calibrated one and are fitted on
the fold's **inner slice** - rows after the fitting rows (purged), before the
validation block - never on rows the model was fitted on, never on the block
it is scored on, never on the final test.

* ``platt``: logistic regression of the label on ``logit(p)``
  (:math:`\tilde p = \sigma(a\,\mathrm{logit}(p) + b)`);
* ``isotonic``: a monotone step function (pool-adjacent-violators), linear
  between knots, clipped outside them; fitted only with enough rows;
* ``none``: the raw probability.

:func:`reliability` bins predictions and compares the mean prediction with the
observed frequency (the reliability diagram); :func:`expected_calibration_error`
summarises it, weighted by bin counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["Calibrator", "expected_calibration_error", "fit_calibrator", "reliability"]

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)
    return np.log(q / (1.0 - q))


@dataclass
class Calibrator:
    method: str
    params: dict[str, Any] = field(default_factory=dict)

    def apply(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        if self.method == "none":
            return p
        if self.method == "platt":
            z = self.params["a"] * _logit(p) + self.params["b"]
            return 1.0 / (1.0 + np.exp(-z))
        if self.method == "isotonic":
            xs = np.asarray(self.params["x"], dtype=np.float64)
            ys = np.asarray(self.params["y"], dtype=np.float64)
            return np.clip(np.interp(p, xs, ys), 0.0, 1.0)
        raise ValueError(f"unknown calibration method {self.method!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "params": self.params}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Calibrator:
        return cls(method=str(d["method"]), params=dict(d.get("params") or {}))


def fit_calibrator(p: np.ndarray, y: np.ndarray, method: str, *,
                   min_rows: int = 20_000) -> Calibrator:
    """Fit *method* on calibration rows (raw probability *p*, 0/1 label *y*)."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if method == "none":
        return Calibrator("none")
    if method == "platt":
        from sklearn.linear_model import LogisticRegression

        if len(np.unique(y)) < 2:
            return Calibrator("none", {"reason": "one class in the calibration rows"})
        z = _logit(p)[:, None]
        est = LogisticRegression(C=1e6, max_iter=1000).fit(z, y.astype(np.int64))
        return Calibrator("platt", {"a": float(est.coef_[0, 0]), "b": float(est.intercept_[0]),
                                    "rows": int(len(y))})
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        if len(y) < min_rows:
            return Calibrator("none", {"reason": f"fewer than {min_rows} calibration rows"})
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y)
        return Calibrator("isotonic", {"x": [float(v) for v in iso.X_thresholds_],
                                       "y": [float(v) for v in iso.y_thresholds_],
                                       "rows": int(len(y))})
    raise ValueError(f"unknown calibration method {method!r}")


def reliability(p: np.ndarray, y: np.ndarray, *, bins: int = 20,
                strategy: str = "quantile") -> list[dict[str, Any]]:
    """Mean prediction vs observed frequency per bin (the reliability diagram)."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if p.size == 0:
        return []
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    else:
        edges = np.linspace(0.0, 1.0, bins + 1)
    if edges.size < 2:
        edges = np.array([p.min(), p.max() + 1e-12])
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, edges.size - 2)
    rows = []
    for b in range(edges.size - 1):
        m = idx == b
        if not m.any():
            continue
        rows.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]),
                     "mean_predicted": float(p[m].mean()), "observed": float(y[m].mean()),
                     "n": int(m.sum())})
    return rows


def expected_calibration_error(p: np.ndarray, y: np.ndarray, *, bins: int = 20) -> float | None:
    table = reliability(p, y, bins=bins)
    n = sum(r["n"] for r in table)
    if n == 0:
        return None
    return float(sum(r["n"] * abs(r["mean_predicted"] - r["observed"]) for r in table) / n)
