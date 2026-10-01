r"""Transitions, durations, regime age and transition calibration (Steps 16-17, 29-31).

For a transition matrix :math:`A` with geometric state durations the expected
duration of state ``i`` is :math:`D_i = 1/(1-A_{ii})` bars. Observed durations
are the run lengths of a hard state sequence; comparing the two says whether
the geometric (memoryless) assumption holds.

A **run** ends when the state changes or a bar is unscored (label ``-1``),
and runs touching either end of the sequence are flagged *censored*: their
true length is unknown, so they are counted but kept apart in the summaries.

``regime_age`` counts bars since the current hard state began, from past
labels only (causal). The model-based transition features are

.. math::

    P(S_{t+1} = j \mid X_{\le t}) = \sum_i p_i A_{ij},\qquad
    P(\text{leave}) = P(S_{t+1} \ne S_t \mid X_{\le t}) = 1 - \sum_i p_i A_{ii},

expectations under the frozen model, not future information. Their
calibration is read against the realised next-bar change of the filtered
hard state.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

__all__ = [
    "duration_summary",
    "leave_probability",
    "regime_age",
    "run_lengths",
    "transition_calibration",
    "transition_counts",
]


def run_lengths(labels: np.ndarray) -> pl.DataFrame:
    """One row per run of a constant label (``-1`` breaks runs and is not a run)."""
    lab = np.asarray(labels, dtype=np.int64)
    n = lab.size
    if n == 0:
        return pl.DataFrame(schema={"state": pl.Int64, "start": pl.Int64, "length": pl.Int64,
                                    "censored": pl.Boolean})
    change = np.ones(n, dtype=bool)
    change[1:] = lab[1:] != lab[:-1]
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], n)
    states = lab[starts]
    keep = states >= 0
    starts, ends, states = starts[keep], ends[keep], states[keep]
    censored = (starts == 0) | (ends == n)
    return pl.DataFrame({"state": states, "start": starts, "length": ends - starts,
                         "censored": censored})


def duration_summary(runs: pl.DataFrame, n_states: int, *,
                     expected: np.ndarray | None = None) -> pl.DataFrame:
    """Per state: run count, mean/median/quantiles/max of uncensored run lengths.

    *expected* adds the model's geometric expectation ``1/(1 - A_ii)``.
    """
    rows: list[dict[str, Any]] = []
    total = int(runs["length"].sum()) if runs.height else 0
    for k in range(n_states):
        part = runs.filter(pl.col("state") == k)
        full = part.filter(~pl.col("censored"))["length"].to_numpy().astype(np.float64)
        row: dict[str, Any] = {
            "state": k, "runs": part.height, "censored_runs": part.height - full.size,
            "bars": int(part["length"].sum()) if part.height else 0,
            "share_of_bars": (int(part["length"].sum()) / total) if total and part.height else 0.0,
        }
        if full.size:
            row.update({"mean_duration": float(full.mean()),
                        "median_duration": float(np.median(full)),
                        **{f"p{int(q * 100)}_duration": float(np.quantile(full, q))
                           for q in (0.1, 0.25, 0.75, 0.9)},
                        "max_duration": float(full.max()),
                        "single_bar_runs": float(np.mean(full == 1))})
        if expected is not None:
            row["expected_duration"] = float(expected[k])
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def transition_counts(labels: np.ndarray, n_states: int) -> np.ndarray:
    """Counts of consecutive (from, to) pairs; pairs touching ``-1`` are skipped."""
    lab = np.asarray(labels, dtype=np.int64)
    a, b = lab[:-1], lab[1:]
    ok = (a >= 0) & (b >= 0)
    counts = np.zeros((n_states, n_states), dtype=np.int64)
    np.add.at(counts, (a[ok], b[ok]), 1)
    return counts


def regime_age(labels: np.ndarray) -> np.ndarray:
    """Bars since the current label began (1 on its first bar); -1 where unscored. Causal."""
    lab = np.asarray(labels, dtype=np.int64)
    n = lab.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    idx = np.arange(n)
    change = np.ones(n, dtype=bool)
    change[1:] = lab[1:] != lab[:-1]
    start = np.maximum.accumulate(np.where(change, idx, 0))
    return np.where(lab >= 0, idx - start + 1, -1)


def leave_probability(probs: np.ndarray, transition: np.ndarray) -> np.ndarray:
    r""":math:`1 - \sum_i p_i A_{ii}`: the chance the state changes at the next bar."""
    p = np.asarray(probs, dtype=np.float64)
    return 1.0 - p @ np.diag(transition)


def transition_calibration(predicted: np.ndarray, realised: np.ndarray, *,
                           bins: int) -> pl.DataFrame:
    """Reliability table: predicted probability deciles vs the realised frequency."""
    p = np.asarray(predicted, dtype=np.float64)
    y = np.asarray(realised, dtype=np.float64)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if p.size == 0:
        return pl.DataFrame()
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    which = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, max(edges.size - 2, 0))
    rows = []
    for b in range(max(edges.size - 1, 1)):
        m = which == b
        if not m.any():
            continue
        rows.append({"bin": b, "lower": float(edges[b]), "upper": float(edges[min(b + 1,
                                                                                  edges.size - 1)]),
                     "observations": int(m.sum()), "mean_predicted": float(p[m].mean()),
                     "observed_frequency": float(y[m].mean())})
    return pl.DataFrame(rows)
