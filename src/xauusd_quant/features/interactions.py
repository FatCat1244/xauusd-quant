r"""Registered pairwise interactions (Prompt #8, Steps 30, 43).

Only the pairs in ``config/features.yaml`` are built - never all N^2. Each
component is standardised causally before multiplying,

.. math:: z_t(x) = \mathrm{clip}\!\left(\frac{x_t - \bar x_{t,W}}{s_{t,W}}, \pm 10\right),

with the trailing mean and standard deviation over the finite values among the
last ``W`` bars (``interaction_standardisation_days`` trading days; at least
90 % of the window present, and never a partial first window), so the product
is centred on "both unusual" rather than on the raw scale of either feature,
and depends only on bars ``<= t``. A component missing at ``t`` gives a missing
product. (A window that demanded *every* value would lose ~10 % of the 1h bars
to the occasional invalid OU fit and nearly all of the 5m ones.)
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from .factory_config import FeatureFactoryConfig

__all__ = ["causal_zscore", "interaction_columns"]


def causal_zscore(values: np.ndarray, window: int, *, clip: float = 10.0,
                  min_share: float = 0.9) -> np.ndarray:
    """(x_t - trailing mean) / trailing sd over the last *window* bars, clipped.

    The moments use the finite values in the window and need at least
    *min_share* of it; the first ``window - 1`` bars are always missing.
    """
    x = np.asarray(values, dtype=np.float64)
    need = max(2, int(math.ceil(min_share * window)))
    s = pl.Series("x", x).fill_nan(None)
    frame = pl.DataFrame({"x": s}).select(
        pl.col("x").rolling_mean(window_size=window, min_samples=need).alias("m"),
        pl.col("x").rolling_std(window_size=window, min_samples=need).alias("s"))
    mean = frame["m"].fill_null(np.nan).to_numpy()
    sd = frame["s"].fill_null(np.nan).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(sd > 0, (x - mean) / np.where(sd > 0, sd, 1.0), np.nan)
    z[: window - 1] = np.nan
    return np.clip(z, -clip, clip)


def interaction_columns(features: dict[str, np.ndarray], cfg: FeatureFactoryConfig,
                        bar_seconds: float) -> dict[str, np.ndarray]:
    """Every registered interaction whose two components exist in *features*."""
    window = cfg.interaction_standardisation_days * cfg.bars_per_day(bar_seconds)
    cache: dict[str, np.ndarray] = {}
    out: dict[str, np.ndarray] = {}
    for ix in cfg.interactions:
        if ix.a not in features or ix.b not in features:
            continue
        for name in (ix.a, ix.b):
            if name not in cache:
                cache[name] = causal_zscore(features[name], window)
        out[ix.name] = cache[ix.a] * cache[ix.b]
    return out
