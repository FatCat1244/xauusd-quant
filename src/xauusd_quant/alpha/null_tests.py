r"""Null controls for predictive research (Prompt #8, Steps 50-53).

How large can an IC look when there is no information? Four answers:

* **Circular shifts** of the target (Step 50): the target column rotated by a
  random 10-90 % of the sample. Each series keeps its own autocorrelation and
  distribution - only the alignment is destroyed - so a persistent feature
  meets a persistent target by chance exactly as often as in the real data.
* **Permutations** of the target rows: the textbook shuffle. It also destroys
  the autocorrelation, so its null is much narrower than the circular one; it
  is reported beside it to show how optimistic it is.
* **Wrong alignments** (Step 51): the target moved by days in either
  direction. Information that survives a wrong alignment is a slow-moving
  common component (trend, era, volatility level), not a forecast. A feature
  deliberately taken one bar *ahead* shows the size of real look-ahead.
* **Synthetic noise features** (Step 52): Gaussian white noise, persistent
  AR(1) noise (phi 0.9, 0.99, 0.999) and real feature values reordered by AR(1)
  noise (matched marginals) go through the same IC pipeline, test counts and
  FDR - the best of many noise features shows what selection alone produces.

The random-walk pipeline controls (Step 53) are built by the factory on a null
price path and scored by the same engine (``research/feature_research.py``).

For speed, the shift and permutation nulls use one matrix product per draw:
each column is rank-transformed, standardised over its finite values and its
missing values set to zero, so :math:`IC_0 = Z_X^\top Z_Y / n` - the
correlation shrunk by the missing share. Real and null values use the same
estimator; the null quantile is rescaled to IC units by ``n / n_pairs``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .information_coefficient import rank_scores

__all__ = [
    "NullMatrices",
    "circular_shift_null",
    "pair_counts",
    "permutation_null",
    "shifted_ic",
    "standardized_ranks",
    "standardized_scores",
    "synthetic_noise_features",
]


@dataclass
class NullMatrices:
    """Standardised rank matrices (missing = 0), float32, for the fast null estimator."""

    z: np.ndarray               # (n, p)
    present: np.ndarray         # (n, p) bool


def standardized_ranks(columns: list[np.ndarray]) -> NullMatrices:
    """Rank-transform, then :func:`standardized_scores`."""
    return standardized_scores([rank_scores(values) for values in columns])


def standardized_scores(columns: list[np.ndarray]) -> NullMatrices:
    """Columns already rank-transformed (NaN = missing), standardised over their finite values."""
    n = columns[0].size
    z = np.zeros((n, len(columns)), dtype=np.float32)
    present = np.zeros((n, len(columns)), dtype=bool)
    for j, values in enumerate(columns):
        u = np.asarray(values, dtype=np.float64)
        ok = np.isfinite(u)
        if ok.sum() > 2:
            v = u[ok]
            sd = float(v.std())
            if sd > 0:
                z[ok, j] = ((v - v.mean()) / sd).astype(np.float32)
        present[:, j] = ok
    return NullMatrices(z=z, present=present)


def shifted_ic(zx: np.ndarray, zy: np.ndarray, shift: int) -> np.ndarray:
    """IC_0 of x_t with y_(t+shift) (circularly): (p, q) = zx^T roll(zy, -shift) / n."""
    n = zx.shape[0]
    s = shift % n
    if s == 0:
        return (zx.T @ zy) / n
    # x_t pairs with y_(t+s): rows 0..n-s-1 with s..n-1, then the wrap-around
    return (zx[: n - s].T @ zy[s:] + zx[n - s:].T @ zy[: s]) / n


def circular_shift_null(zx: np.ndarray, zy: np.ndarray, *, draws: int, min_fraction: float,
                        seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(draws, p, q) IC_0 under random circular shifts, and the shifts used."""
    n = zx.shape[0]
    rng = np.random.default_rng(seed)
    lo, hi = int(min_fraction * n), int((1.0 - min_fraction) * n)
    shifts = rng.integers(lo, hi, size=draws)
    out = np.empty((draws, zx.shape[1], zy.shape[1]), dtype=np.float32)
    for i, s in enumerate(shifts):
        out[i] = shifted_ic(zx, zy, int(s))
    return out, shifts


def permutation_null(zx: np.ndarray, zy: np.ndarray, *, draws: int, seed: int) -> np.ndarray:
    """(draws, p, q) IC_0 with the rows of one side permuted.

    The feature rows are permuted (the smaller matrix; a random re-pairing is
    the same null whichever side moves).
    """
    rng = np.random.default_rng(seed)
    out = np.empty((draws, zx.shape[1], zy.shape[1]), dtype=np.float32)
    n = zx.shape[0]
    for i in range(draws):
        perm = rng.permutation(n)
        out[i] = (zx[perm].T @ zy) / n
    return out


def pair_counts(present_x: np.ndarray, present_y: np.ndarray, *, block: int = 262_144
                ) -> np.ndarray:
    """(p, q) number of rows where both columns are present, in row blocks (bounded memory)."""
    out = np.zeros((present_x.shape[1], present_y.shape[1]), dtype=np.float64)
    for lo in range(0, present_x.shape[0], block):
        a = present_x[lo:lo + block].astype(np.float32)
        b = present_y[lo:lo + block].astype(np.float32)
        out += (a.T @ b).astype(np.float64)
    return out


def _ar1(n: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    from scipy.signal import lfilter

    e = rng.standard_normal(n)
    start = rng.standard_normal() / np.sqrt(1.0 - phi * phi)   # stationary y_(-1)
    x, _ = lfilter([1.0], [1.0, -phi], e, zi=[phi * start])
    return x


def synthetic_noise_features(n: int, *, white: int, ar1: tuple[float, ...], per_phi: int,
                             matched: dict[str, np.ndarray], seed: int
                             ) -> dict[str, tuple[np.ndarray, str]]:
    """Noise features with no information: name -> (float32 values, description).

    A matched-marginal feature keeps the real feature's missing bars and puts
    its finite values in the order of an AR(1) path.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, tuple[np.ndarray, str]] = {}
    for i in range(white):
        out[f"noise_white_{i:02d}"] = (rng.standard_normal(n).astype(np.float32),
                                       "N(0, 1) white noise")
    for phi in ar1:
        tag = f"{phi:.3f}".rstrip("0").replace("0.", "")
        for i in range(per_phi):
            out[f"noise_ar1_{tag}_{i:02d}"] = (_ar1(n, phi, rng).astype(np.float32),
                                               f"AR(1) noise, phi = {phi}")
    for i, (name, values) in enumerate(matched.items()):
        x = np.asarray(values, dtype=np.float64)
        ok = np.isfinite(x)
        noise = _ar1(int(ok.sum()), 0.99, rng)
        out_values = np.full(n, np.nan, dtype=np.float32)
        order = np.argsort(np.argsort(noise))
        out_values[ok] = np.sort(x[ok])[order]
        out[f"noise_matched_{i:02d}"] = (out_values, f"the values of {name} reordered by "
                                                     "AR(1) noise (phi 0.99): same marginal, "
                                                     "no information")
    return out
