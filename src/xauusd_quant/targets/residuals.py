r"""Residual- and OU-reversion targets (Prompt #8, Steps 15, 16).

With :math:`\epsilon_t` the Prompt #3 rolling-regression residual (each from
the window ending at its own bar) and :math:`\sigma_t` the trailing return
volatility known at ``t``:

.. math::

    \Delta\epsilon_{t,h} = (\epsilon_{t+h} - \epsilon_t) / \sigma_t, \qquad
    \mathrm{Reduction}_{t,h} = (|\epsilon_t| - |\epsilon_{t+h}|) / \sigma_t,

    \mathrm{Shrinks}_{t,h} = \mathbb 1\{|\epsilon_{t+h}| < |\epsilon_t|\},

    \mathrm{OUReduction}_{t,h} = (|\epsilon_t - \mu_t| - |\epsilon_{t+h} - \mu_t|) / \sigma_t .

Scaling by :math:`\sigma_t` (known at ``t``) keeps the high-volatility eras
from dominating a pooled correlation. :math:`\mu_t` is the OU equilibrium
estimated from bars ``<= t`` and is held fixed over the horizon - a
re-estimated, future equilibrium would be a different, offline concept.
A positive reduction means the residual shrank.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ou_deviation_reduction", "residual_change", "residual_reduction",
           "residual_shrinks"]


def _ahead(values: np.ndarray, h: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    out = np.full(x.size, np.nan)
    if 0 < h < x.size:
        out[:-h] = x[h:]
    return out


def _scaled(values: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    s = np.asarray(sigma, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 0, values / np.where(s > 0, s, 1.0), np.nan)


def residual_change(eps: np.ndarray, sigma: np.ndarray, h: int) -> np.ndarray:
    return _scaled(_ahead(eps, h) - np.asarray(eps, dtype=np.float64), sigma)


def residual_reduction(eps: np.ndarray, sigma: np.ndarray, h: int) -> np.ndarray:
    e = np.asarray(eps, dtype=np.float64)
    return _scaled(np.abs(e) - np.abs(_ahead(e, h)), sigma)


def residual_shrinks(eps: np.ndarray, h: int) -> np.ndarray:
    e = np.asarray(eps, dtype=np.float64)
    later = _ahead(e, h)
    out = np.full(e.size, np.nan)
    ok = np.isfinite(e) & np.isfinite(later)
    out[ok] = (np.abs(later[ok]) < np.abs(e[ok])).astype(np.float64)
    return out


def ou_deviation_reduction(eps: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
                           h: int) -> np.ndarray:
    e = np.asarray(eps, dtype=np.float64)
    m = np.asarray(mu, dtype=np.float64)
    return _scaled(np.abs(e - m) - np.abs(_ahead(e, h) - m), sigma)
