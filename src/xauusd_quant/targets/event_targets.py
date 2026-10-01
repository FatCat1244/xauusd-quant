r"""Maximum future excursions in research units (Prompt #8, Step 19).

.. math::

    \mathrm{MaxUp}_{t,H} = \max_{1 \le i \le H} \ln(H_{t+i} / C_t), \qquad
    \mathrm{MaxDown}_{t,H} = \min_{1 \le i \le H} \ln(L_{t+i} / C_t) \;(\le 0 \text{ typically})

from the bars' mid highs and lows after ``t``, relative to the close at ``t``.
They describe how far price travelled over the horizon in either direction;
they are not trade MFE / MAE (there is no entry, no side, no cost). A null
control without intrabar highs and lows has no excursion targets.
"""

from __future__ import annotations

import numpy as np
import polars as pl

__all__ = ["future_max_down", "future_max_up"]


def _forward_extreme(values: np.ndarray, h: int, kind: str) -> np.ndarray:
    """max / min of values[t+1 .. t+h] for every t (NaN if any is missing or beyond the end)."""
    x = np.asarray(values, dtype=np.float64)
    n = x.size
    s = pl.Series("x", x).fill_nan(None)
    rolled = (s.rolling_max(window_size=h, min_samples=h) if kind == "max"
              else s.rolling_min(window_size=h, min_samples=h))
    trailing = rolled.cast(pl.Float64).fill_null(np.nan).to_numpy()   # over [s-h+1, s]
    out = np.full(n, np.nan)
    if 0 < h < n:
        out[:n - h] = trailing[h:]                                     # [t+1, t+h] at s = t+h
    return out


def future_max_up(close: np.ndarray, high: np.ndarray, h: int) -> np.ndarray:
    return _forward_extreme(np.log(high), h, "max") - np.log(np.asarray(close, dtype=np.float64))


def future_max_down(close: np.ndarray, low: np.ndarray, h: int) -> np.ndarray:
    return _forward_extreme(np.log(low), h, "min") - np.log(np.asarray(close, dtype=np.float64))
