r"""How quickly a feature's information decays (Prompt #8, Steps 23-24).

Two curves, kept apart because they answer different questions:

* the **horizon curve** IC(h) = corr(x_t, R_{t,h}) of the cumulative targets.
  A cumulative return over ``h`` bars dilutes a one-bar signal like
  :math:`1/\sqrt h` even if the information itself never fades, so its shape
  is read descriptively (peak, half, sign reversal, indistinguishable from 0);
* the **marginal curve** IC(k) = corr(x_t, r_{t+k}) of one-bar outcomes ``k``
  bars ahead - information about the ``k``-th bar alone. Its decay is what the
  empirical ``alpha_information_half_life`` describes, when (and only when) it
  decays roughly exponentially:

.. math:: |IC(k)| \approx IC_0 e^{-\lambda k}, \qquad HL_{IC} = \ln 2 / \lambda .

The fit uses the points from the peak on, while the IC keeps its sign and
stays distinguishable from zero; it is reported only with at least
``min_fit_points`` points and an :math:`R^2 \ge 0.8` of the log-linear fit -
otherwise "not exponential" with the reason. This half-life has nothing to do
with the OU half-life of a residual.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = ["decay_summary", "information_half_life"]


def decay_summary(horizons: np.ndarray, ic: np.ndarray, se: np.ndarray, *,
                  zero_z: float) -> dict[str, Any]:
    """Peak, half-decay, sign-reversal and zero horizons of one IC curve."""
    h = np.asarray(horizons, dtype=np.float64)
    ic = np.asarray(ic, dtype=np.float64)
    se = np.asarray(se, dtype=np.float64)
    ok = np.isfinite(ic)
    out: dict[str, Any] = {"peak_horizon": None, "peak_ic": None, "half_decay_horizon": None,
                           "sign_reversal_horizon": None, "zero_horizon": None}
    if not ok.any():
        return out
    i_peak = int(np.nanargmax(np.where(ok, np.abs(ic), -1.0)))
    peak = ic[i_peak]
    out["peak_horizon"] = int(h[i_peak])
    out["peak_ic"] = float(peak)
    for j in range(i_peak + 1, h.size):
        if ok[j] and abs(ic[j]) <= abs(peak) / 2:
            out["half_decay_horizon"] = int(h[j])
            break
    for j in range(i_peak + 1, h.size):
        if ok[j] and np.sign(ic[j]) == -np.sign(peak) and abs(ic[j]) > zero_z * se[j]:
            out["sign_reversal_horizon"] = int(h[j])
            break
    with np.errstate(invalid="ignore", divide="ignore"):
        indist = np.abs(ic) < zero_z * se
    for j in range(h.size):
        if ok[j] and indist[j]:
            out["zero_horizon"] = int(h[j])
            break
    return out


def information_half_life(lags: np.ndarray, ic: np.ndarray, se: np.ndarray, *, zero_z: float,
                          min_points: int) -> dict[str, Any]:
    """Exponential fit of |IC(k)| on the marginal curve, or the reason it is not fitted."""
    k = np.asarray(lags, dtype=np.float64)
    ic = np.asarray(ic, dtype=np.float64)
    se = np.asarray(se, dtype=np.float64)
    ok = np.isfinite(ic) & np.isfinite(se)
    if ok.sum() < min_points:
        return {"alpha_information_half_life": None, "fit": "not fitted: too few points"}
    i_peak = int(np.nanargmax(np.where(ok, np.abs(ic), -1.0)))
    sign = np.sign(ic[i_peak])
    pts = []
    for j in range(i_peak, k.size):
        if not ok[j] or np.sign(ic[j]) != sign or abs(ic[j]) < zero_z * se[j]:
            break
        pts.append(j)
    if abs(ic[i_peak]) < zero_z * se[i_peak]:
        return {"alpha_information_half_life": None,
                "fit": "not fitted: the peak is indistinguishable from zero"}
    if len(pts) < min_points:
        return {"alpha_information_half_life": None,
                "fit": f"not fitted: {len(pts)} decaying point(s) beyond the peak before the IC "
                       "loses its sign or significance",
                "significant_through_lag": int(k[pts[-1]]) if pts else None}
    kk, yy = k[pts], np.log(np.abs(ic[pts]))
    slope, intercept = np.polyfit(kk, yy, 1)
    fitted = intercept + slope * kk
    ss_res = float(((yy - fitted) ** 2).sum())
    ss_tot = float(((yy - yy.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    if slope >= 0:
        return {"alpha_information_half_life": None, "fit": "not fitted: |IC| does not decay",
                "log_fit_r2": r2}
    if r2 < 0.8:
        return {"alpha_information_half_life": None,
                "fit": f"not exponential (log-linear R^2 {r2:.2f} < 0.8)", "log_fit_r2": r2}
    lam = -float(slope)
    return {"alpha_information_half_life": math.log(2.0) / lam, "decay_rate": lam,
            "ic0": float(math.exp(intercept)) * float(sign), "log_fit_r2": r2,
            "fit": "exponential", "points": len(pts)}
