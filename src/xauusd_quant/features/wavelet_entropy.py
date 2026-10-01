r"""Wavelet (scale) entropy.

With the band energy shares :math:`p_j` (:mod:`wavelet_energy`),

.. math:: H_W = -\sum_j p_j \ln p_j, \qquad H_{W,norm} = H_W / \ln B

where ``B`` is the number of bands. Low entropy: energy in few scales; high:
spread across scales. It is a description of how the window's variance is
distributed over timescales - not a measure of predictability, and it is
always read beside the same statistic on the null controls.
"""

from __future__ import annotations

import numpy as np

__all__ = ["effective_scales", "wavelet_entropy"]


def wavelet_entropy(shares: np.ndarray, *, normalise: bool = True) -> np.ndarray:
    """Entropy of each column of *shares* ``(bands, bars)``; NaN where undefined."""
    p = np.asarray(shares, dtype=np.float64)
    undefined = np.isnan(p).any(axis=0)
    safe = np.where(np.isnan(p) | (p <= 0), 1.0, p)
    h = -(np.where(np.isnan(p), 0.0, p) * np.log(safe)).sum(axis=0)
    if normalise and p.shape[0] > 1:
        h = h / np.log(p.shape[0])
    return np.where(undefined, np.nan, h)


def effective_scales(shares: np.ndarray) -> np.ndarray:
    """:math:`\\exp(H_W)`: the number of equally loaded scales with the same entropy."""
    return np.exp(wavelet_entropy(shares, normalise=False))
