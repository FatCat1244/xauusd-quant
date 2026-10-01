r"""Shape of a spectrum: entropy, flatness and concentration.

All three describe *how* power is spread over frequencies, not whether any of
it is useful. White noise has a flat expected spectrum, yet a finite window of
white noise still shows peaks; that is why every figure here is read against
null controls, never as a level.

Spectral entropy, with :math:`p_k = P_k / \sum_j P_j` over the ``K`` usable bins:

.. math:: H = -\sum_k p_k \ln p_k, \qquad H_{norm} = H / \ln K \in [0, 1]

Spectral flatness, computed in log space so no product can underflow:

.. math:: SF = \exp\Big(\tfrac1K \sum_k \ln P_k\Big) \Big/ \tfrac1K \sum_k P_k

Both are 1 for a perfectly flat spectrum and fall as power concentrates.
"""

from __future__ import annotations

import numpy as np

__all__ = ["concentration", "spectral_entropy", "spectral_flatness"]

_TINY = np.finfo(np.float64).tiny


def spectral_entropy(power: np.ndarray, total: np.ndarray, *, normalise: bool = True) -> np.ndarray:
    """Entropy of each row's power distribution (0 log 0 = 0)."""
    k = power.shape[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        shares = power / total[:, None]
        log_shares = np.log(np.maximum(shares, _TINY))
        entropy = -(shares * log_shares).sum(axis=1)
    if normalise:
        entropy = entropy / np.log(k) if k > 1 else np.full_like(entropy, np.nan)
    return entropy


def spectral_flatness(power: np.ndarray) -> np.ndarray:
    """Geometric over arithmetic mean of each row's power, in log space."""
    with np.errstate(invalid="ignore", divide="ignore"):
        log_geometric = np.log(np.maximum(power, _TINY)).mean(axis=1)
        log_arithmetic = np.log(power.mean(axis=1))
        return np.exp(log_geometric - log_arithmetic)


def concentration(sorted_shares: np.ndarray, counts: tuple[int, ...] = (1, 3, 5)) -> dict[int, np.ndarray]:
    """Share of power in the ``n`` strongest bins, for each ``n`` in *counts*.

    *sorted_shares* holds each row's largest shares in descending order (as
    many columns as the largest count needs, or fewer if the window has fewer
    bins - then the share is over all of them).
    """
    return {n: sorted_shares[:, :n].sum(axis=1) for n in counts}
