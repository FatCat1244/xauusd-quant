r"""Stacking on out-of-sample predictions only (Prompt #11, Steps 5, 21-26).

Base models :math:`M_1 .. M_K` produce walk-forward out-of-sample predictions
:math:`p_{1,t} .. p_{K,t}` (Prompt #10: each block predicted by models fitted
before it). A meta-model :math:`g(p_{1,t}, .., p_{K,t})` is then fitted **only on
those predictions** - never on a base model's predictions of its own training
rows - and chronologically: for evaluation block *k* it sees the rows of blocks
``< k`` whose outcomes had resolved ``h + embargo`` bars before block *k*
(:meth:`~.data.PairPredictions.history`). Rows are never shuffled across time.

:func:`meta_inputs` assembles the meta-model's design for a set of positions;
:func:`stack_block` fits on the history and predicts one block;
:func:`fit_stack` fits on every out-of-sample row (the frozen meta-model, used
only after its spec is frozen).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .data import PairPredictions
from .meta_model import MetaModel, fit_meta_model

__all__ = ["fit_stack", "meta_inputs", "stack_block"]


def meta_inputs(pair: PairPredictions, models: list[str], positions: np.ndarray,
                context: dict[str, np.ndarray] | None = None
                ) -> tuple[np.ndarray, list[str]]:
    """(rows x (K + C) design, input names): constituent predictions, then context.

    *context* maps a name to one value per position of the pair (already causal).
    """
    cols = [pair.prediction(m)[positions] for m in models]
    names = list(models)
    for name, values in (context or {}).items():
        cols.append(np.asarray(values, dtype=np.float64)[positions])
        names.append(f"context:{name}")
    return np.column_stack(cols), names


def stack_block(pair: PairPredictions, models: list[str], k: int, *, embargo: int,
                context: dict[str, np.ndarray] | None = None,
                params: dict[str, Any] | None = None, seed: int = 0
                ) -> tuple[np.ndarray, MetaModel, np.ndarray]:
    """(predictions on block *k*, the meta-model, the history positions it was fitted on)."""
    params = params or {}
    hist = pair.history(k, embargo)
    if hist.size == 0:
        raise ValueError(f"block {k}: no earlier out-of-sample rows to fit a meta-model on")
    xh, names = meta_inputs(pair, models, hist, context)
    meta = fit_meta_model(xh, pair.label[hist], names, constituents=len(models), task=pair.task,
                          logistic_c=float(params.get("logistic_C", 1.0)),
                          ridge_alpha=float(params.get("ridge_alpha", 1.0)),
                          max_rows=params.get("max_rows"), seed=seed)
    pos = pair.block_positions(k)
    xb, _ = meta_inputs(pair, models, pos, context)
    return meta.predict(xb), meta, hist


def fit_stack(pair: PairPredictions, models: list[str], *,
              context: dict[str, np.ndarray] | None = None,
              params: dict[str, Any] | None = None, seed: int = 0) -> MetaModel:
    """The meta-model on every out-of-sample row of the pair (for a frozen ensemble)."""
    params = params or {}
    pos = np.flatnonzero(np.isfinite(pair.label))
    x, names = meta_inputs(pair, models, pos, context)
    return fit_meta_model(x, pair.label[pos], names, constituents=len(models), task=pair.task,
                          logistic_c=float(params.get("logistic_C", 1.0)),
                          ridge_alpha=float(params.get("ridge_alpha", 1.0)),
                          max_rows=params.get("max_rows"), seed=seed)
