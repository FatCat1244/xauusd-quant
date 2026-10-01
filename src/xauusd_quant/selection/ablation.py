r"""Family ablation, leave-one-family-out and permutation importance (Prompt #9, Steps 29-34).

All fits are the rank-linear ridge of :mod:`.linear` on the training rows of a
split, scored on its evaluation rows; the feature pool is the *quality*
universe (not outcome-selected), so a family is judged by what it carries,
not by what a selector already kept.

* **Cumulative ablation**: statistics (returns, volatility, autocorrelation)
  -> + regression -> + OU -> + FFT -> + wavelet -> + regime -> +
  microstructure -> + time -> + interactions.
* **Leave one family out**: everything, minus one family at a time - a drop
  means the family carried something the others do not.
* **Permutation importance** on the evaluation rows with *block* permutation:
  the rows are cut into blocks of ``permutation_block_bars`` bars and one
  feature's blocks are shuffled, which breaks its alignment with the outcome
  but keeps its own short-range structure (a row-level shuffle would destroy
  that too and exaggerate the loss). Correlated features absorb each other's
  information, so the same is repeated for whole **redundancy clusters**; a
  feature whose own importance is ~0 while its cluster's is large is not
  useless - its twin carries it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .linear import evaluate, ridge
from .selection_cv import SplitWork

__all__ = ["STATISTICS", "cumulative_groups", "fit_and_score", "leave_one_family_out",
           "permutation_importance"]

STATISTICS = ("returns", "volatility", "autocorrelation")
_STEPS = ("regression", "ou", "fft", "wavelet", "regime", "microstructure", "time",
          "interaction")


def cumulative_groups() -> list[tuple[str, tuple[str, ...]]]:
    out: list[tuple[str, tuple[str, ...]]] = [("statistics", STATISTICS)]
    fams = list(STATISTICS)
    for step in _STEPS:
        fams.append(step)
        out.append((f"+{step}", tuple(fams)))
    return out


def fit_and_score(work: SplitWork, cols: list[int], j: int, alpha: float
                  ) -> tuple[dict[str, Any], np.ndarray | None]:
    """Ridge on training moments of *cols*; metrics on the evaluation rows, and the coefs."""
    if not cols:
        return {"rank_ic": None, "r2": None, "n": 0}, None
    q, c, mx, sd, my = work.moments.standardized(cols)
    beta = ridge(q, c[:, [j]], alpha)[:, 0]
    pred = ((work.x_eval[:, cols] - mx) / sd) @ beta + my[j]
    return evaluate(pred, work.y_eval_raw[:, j], work.y_eval[:, j], float(my[j])), beta


def leave_one_family_out(work: SplitWork, universe: list[int], family_of: list[str], j: int,
                         alpha: float) -> list[dict[str, Any]]:
    full, _ = fit_and_score(work, universe, j, alpha)
    rows = [{"variant": "all", "features": len(universe), **full}]
    groups = {"statistics": STATISTICS, **{f: (f,) for f in _STEPS}}
    for name, fams in groups.items():
        cols = [c for c in universe if family_of[c] not in fams]
        if len(cols) == len(universe):
            continue
        m, _ = fit_and_score(work, cols, j, alpha)
        rows.append({"variant": f"-{name}", "features": len(cols), **m,
                     "delta_rank_ic": _delta(m.get("rank_ic"), full.get("rank_ic")),
                     "delta_r2": _delta(m.get("r2"), full.get("r2"))})
    return rows


def _delta(a: Any, b: Any) -> float | None:
    return None if a is None or b is None else float(a) - float(b)


def permutation_importance(work: SplitWork, cols: list[int], j: int, alpha: float, *,
                           block: int, repeats: int, seed: int,
                           groups: dict[str, list[int]] | None = None) -> list[dict[str, Any]]:
    """Mean drop in evaluation rank IC when one feature's (or one group's) blocks are shuffled."""
    q, c, mx, sd, my = work.moments.standardized(cols)
    beta = ridge(q, c[:, [j]], alpha)[:, 0]
    z = (work.x_eval[:, cols] - mx) / sd
    base = z @ beta + my[j]
    ref = evaluate(base, work.y_eval_raw[:, j], work.y_eval[:, j], float(my[j]))["rank_ic"]
    n = z.shape[0]
    starts = np.arange(0, n, max(1, block))
    rng = np.random.default_rng(seed)
    rows = []
    units: list[tuple[str, list[int]]] = [(f"feature:{k}", [i]) for i, k in enumerate(cols)]
    if groups:
        units += [(f"cluster:{g}", [cols.index(c) for c in members if c in cols])
                  for g, members in groups.items()]
    for label, members in units:
        if not members or ref is None:
            continue
        drops = []
        for _ in range(repeats):
            order = rng.permutation(starts.size)
            idx = np.concatenate([np.arange(starts[b], min(n, starts[b] + block)) for b in order])
            pred = base.copy()
            for m in members:
                pred += (z[idx[:n], m] - z[:, m]) * beta[m]
            r = evaluate(pred, work.y_eval_raw[:, j], work.y_eval[:, j], float(my[j]))["rank_ic"]
            if r is not None:
                drops.append(ref - r)
        rows.append({"unit": label, "members": len(members),
                     "importance_rank_ic": float(np.mean(drops)) if drops else None,
                     "importance_sd": float(np.std(drops)) if len(drops) > 1 else None,
                     "coefficient": float(beta[members[0]]) if len(members) == 1 else None,
                     "reference_rank_ic": ref})
    return rows
