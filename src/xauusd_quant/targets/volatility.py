r"""Future realised-volatility target (Prompt #8, Step 17).

.. math:: RV_{t,h} = \sqrt{\sum_{i=1}^{h} r_{t+i}^2}, \qquad r_s = \ln C_s - \ln C_{s-1}

The sum starts at the first return *after* ``t``: the bar-``t`` return is
known at ``t`` and belongs to the features' side.
"""

from __future__ import annotations

import numpy as np

__all__ = ["future_realized_vol"]


def future_realized_vol(close: np.ndarray, h: int) -> np.ndarray:
    """sqrt(r_{t+1}^2 + ... + r_{t+h}^2) (NaN where any of them is missing)."""
    lp = np.log(np.asarray(close, dtype=np.float64))
    n = lp.size
    r2 = np.full(n, np.nan)
    r2[1:] = np.diff(lp) ** 2
    finite = np.isfinite(r2)
    csum = np.concatenate(([0.0], np.cumsum(np.where(finite, r2, 0.0))))
    bad = np.concatenate(([0], np.cumsum(~finite, dtype=np.int64)))
    out = np.full(n, np.nan)
    if 0 < h < n:
        t = np.arange(n - h)
        total = csum[t + h + 1] - csum[t + 1]          # r_{t+1} .. r_{t+h}
        clean = (bad[t + h + 1] - bad[t + 1]) == 0
        out[:n - h] = np.where(clean, np.sqrt(np.maximum(total, 0.0)), np.nan)
    return out
