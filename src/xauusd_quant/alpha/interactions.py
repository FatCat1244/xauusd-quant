r"""Does a registered interaction add information to its two components? (Step 44)

For an interaction :math:`x_3 = z(x_1) z(x_2)` the IC of :math:`x_3` is read
beside the ICs of :math:`x_1` and :math:`x_2` (from the main IC table), and a
small out-of-sample diagnostic compares, on the chronological folds of
Prompts #6/#7 (train strictly before each block, embargoed by the longest
horizon), a ridge model with the two components against the same model plus
the product. The out-of-sample :math:`R^2` gain is the incremental
information; these linear fits are diagnostics, not a predictive system.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from ..research.wavelet_predictiveness import (
    _design,
    _select,
    chronological_folds,
    fit_linear_models,
)

__all__ = ["TEST_BLOCKS", "interaction_increment"]

#: The chronological test blocks of Prompts #6 and #7 (expanding training before each).
TEST_BLOCKS: tuple[tuple[int, int], ...] = ((2011, 2014), (2015, 2018), (2019, 2022),
                                            (2023, 2026))


def interaction_increment(timestamps: pl.Series, a: np.ndarray, b: np.ndarray, ix: np.ndarray,
                          targets: dict[str, np.ndarray], *, ridge: float, max_rows: int,
                          embargo: int) -> list[dict[str, Any]]:
    """Fold-level OOS R^2 of components vs components + interaction, per target."""
    n = a.size
    rows = np.arange(n)
    usable = np.isfinite(a) & np.isfinite(b) & np.isfinite(ix)
    for y in targets.values():
        usable &= np.isfinite(y)
    idx = rows[usable]
    if idx.size > max_rows:
        idx = idx[np.linspace(0, idx.size - 1, max_rows).astype(np.int64)]
    if idx.size < 1000:
        return []
    ts = timestamps.gather(pl.Series(idx))
    folds = chronological_folds(ts, idx, TEST_BLOCKS, embargo)
    out: list[dict[str, Any]] = []
    blocks = {"components": {"a": a[idx], "b": b[idx]}, "interaction": {"ix": ix[idx]}}
    tgt = {k: v[idx] for k, v in targets.items()}
    for label, train, test in folds:
        design = _design(blocks, train, order=("components", "interaction"))
        subsets = {"components": _select(design, ("components",)),
                   "components+interaction": _select(design, ("components", "interaction"))}
        for r in fit_linear_models(design, tgt, train, test, ridge=ridge, subsets=subsets):
            out.append({"fold": label, **r})
    return out
