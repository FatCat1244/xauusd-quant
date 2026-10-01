r"""Model disagreement and ensemble entropy - uncertainty, not alpha (Prompt #11,
Steps 8-9, 46-48; Rule 7).

* :func:`disagreement_measures`: per row, across the constituents' predictions,
  the standard deviation :math:`D_t = \mathrm{Std}(p_{1,t}, .., p_{M,t})`, the range
  :math:`\max p - \min p` and the mean pairwise absolute difference.
* :func:`binary_entropy`: :math:`H(p) = -p \ln p - (1-p)\ln(1-p)` of the *ensemble*
  probability - how undecided the combined forecast is. Stored separately from
  the disagreement: a 0.5 forecast every model agrees on has maximal entropy and
  zero disagreement.
* :func:`grouped_means`: means of any per-row quantity by a causal grouping
  (volatility quartile, regime state, spread quartile, session, year).

Nothing here becomes a rule: no threshold on disagreement, no filter.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["binary_entropy", "disagreement_measures", "grouped_means"]

_EPS = 1e-12


def disagreement_measures(p: np.ndarray) -> dict[str, np.ndarray]:
    p = np.asarray(p, dtype=np.float64)
    ok = np.isfinite(p).all(axis=1)
    m = p.shape[1]
    std = np.full(p.shape[0], np.nan)
    rng = np.full(p.shape[0], np.nan)
    pair = np.full(p.shape[0], np.nan)
    if m >= 2 and ok.any():
        q = p[ok]
        std[ok] = q.std(axis=1)
        rng[ok] = q.max(axis=1) - q.min(axis=1)
        # mean |p_i - p_j| over pairs from the sorted values: sum_k (2k - m + 1) x_(k)
        s = np.sort(q, axis=1)
        coef = 2.0 * np.arange(m) - m + 1
        pair[ok] = (s @ coef) / (m * (m - 1) / 2.0)
    return {"prediction_std": std, "prediction_range": rng, "pairwise_disagreement": pair}


def binary_entropy(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)
    out = -(q * np.log(q) + (1.0 - q) * np.log(1.0 - q))
    return np.where(np.isfinite(np.asarray(p, dtype=np.float64)), out, np.nan)


def grouped_means(values: dict[str, np.ndarray], groups: np.ndarray, *,
                  labels: dict[int, str] | None = None, min_rows: int = 1
                  ) -> list[dict[str, Any]]:
    """Mean of every array in *values* inside each group code (negative codes skipped)."""
    out = []
    groups = np.asarray(groups)
    for g in np.unique(groups):
        if g < 0:
            continue
        m = groups == g
        if m.sum() < min_rows:
            continue
        row: dict[str, Any] = {"group": int(g), "label": (labels or {}).get(int(g), str(int(g))),
                               "rows": int(m.sum())}
        for name, v in values.items():
            x = np.asarray(v, dtype=np.float64)[m]
            row[name] = float(np.nanmean(x)) if np.isfinite(x).any() else None
        out.append(row)
    return out
