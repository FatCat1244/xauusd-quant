r"""Future return and absolute-move targets (Prompt #8, Steps 14, 18).

.. math:: R_{t,h} = \ln(C_{t+h} / C_t), \qquad A_{t,h} = |R_{t,h}|

``h`` counts bars of the bar sequence (trading time). The last ``h`` rows have
no outcome and are missing. These are outcomes: they use bars after ``t`` and
never enter a feature table.
"""

from __future__ import annotations

import numpy as np

__all__ = ["forward_log_return", "future_abs_return", "future_return"]


def forward_log_return(log_price: np.ndarray, h: int) -> np.ndarray:
    """ln P_{t+h} - ln P_t for every t (NaN for the last h rows)."""
    lp = np.asarray(log_price, dtype=np.float64)
    out = np.full(lp.size, np.nan)
    if 0 < h < lp.size:
        out[:-h] = lp[h:] - lp[:-h]
    return out


def future_return(close: np.ndarray, h: int) -> np.ndarray:
    return forward_log_return(np.log(np.asarray(close, dtype=np.float64)), h)


def future_abs_return(close: np.ndarray, h: int) -> np.ndarray:
    return np.abs(future_return(close, h))
