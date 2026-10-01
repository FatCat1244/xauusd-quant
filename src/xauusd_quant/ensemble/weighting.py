r"""Ensemble weights estimated from earlier out-of-sample rows only (Prompt #11,
Steps 12-14, 27-32, 55-57).

Quality is always the same quantity: the **skill against the constant** on the
fitted scale, :math:`Q_i = 1 - \sum_t L(\hat y_{i,t}, y_t) / \sum_t L(b_t, y_t)`,
with :math:`L` the log loss (classification) or the squared error (regression)
and :math:`b_t` the constant model's prediction (the training mean of the row's
block). Rank IC stays an evaluation metric; it is not what a weighted average of
values optimises.

* :func:`performance_weights`: :math:`w_i \propto \max(Q_i, 0)`, then
  :func:`shrink_to_equal` (:math:`w' = (1-\lambda) w + \lambda / M`); equal weights
  when no model has positive skill. Never a trading objective.
* :func:`diversity_weights`: :math:`w_i \propto \max(Q_i,0)\,(1 - \lambda R_i)`, with
  :math:`R_i` the mean correlation of model *i*'s errors with the others'. Models
  whose predictions correlate at or above ``duplicate_correlation`` form one
  *cluster* that gets one weight, split equally - a copy adds no information.
* :func:`conditional_weights` / :func:`apply_state_weights`: one weight vector per
  state, estimated with soft (regime probabilities) or hard (volatility buckets)
  row weights, mixed at time *t* by the state probabilities known at *t*:
  :math:`w_i(t) = \sum_k P(S_t = k)\, w_{i,k}`.
* :func:`dynamic_weights`: trailing model health, updated once per trading day
  from rows whose label had **resolved before** that day's first bar (row number
  ``+ h <`` the update bar) within the last ``window_days`` trading days. The
  weights of a day therefore never change when later rows are appended
  (Step 32; :func:`dynamic_weights` is tested for exactly that).
* :func:`weight_entropy`, :func:`turnover`: diagnostics (Steps 56-57).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .averaging import equal_weights, normalise

__all__ = ["DynamicWeights", "apply_state_weights", "conditional_weights", "diversity_weights",
           "duplicate_clusters", "dynamic_weights", "error_correlation", "loss_rows",
           "performance_weights", "shrink_to_equal", "skill", "turnover", "weight_entropy"]

_EPS = 1e-6


def loss_rows(task: str, pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-row loss: log loss (classification, probabilities clipped) or squared error."""
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if task == "classification":
        p = np.clip(pred, _EPS, 1.0 - _EPS)
        if p.ndim == 2:
            return -(y[:, None] * np.log(p) + (1.0 - y[:, None]) * np.log(1.0 - p))
        return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    if pred.ndim == 2:
        return (pred - y[:, None]) ** 2
    return (pred - y) ** 2


def skill(task: str, pred: np.ndarray, y: np.ndarray, base: np.ndarray,
          row_weight: np.ndarray | None = None) -> np.ndarray | float:
    """Skill against the per-row constant *base*; (M,) for an (n, M) *pred*."""
    lp = loss_rows(task, pred, y)
    lb = loss_rows(task, base, y)
    if row_weight is None:
        num = lp.sum(axis=0)
        den = lb.sum()
    else:
        rw = np.asarray(row_weight, dtype=np.float64)
        num = (lp * (rw[:, None] if lp.ndim == 2 else rw)).sum(axis=0)
        den = float((lb * rw).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        out = 1.0 - num / den if den > 0 else np.full(np.shape(num), np.nan)
    if np.ndim(out) == 0:
        return float(out)
    return np.asarray(out, dtype=np.float64)


def shrink_to_equal(w: np.ndarray, lam: float) -> np.ndarray:
    """:math:`(1-\\lambda) w + \\lambda / M` (Step 13)."""
    w = normalise(w)
    if not 0.0 <= lam <= 1.0:
        raise ValueError("shrink must lie in [0, 1]")
    return (1.0 - lam) * w + lam / w.size


def performance_weights(p: np.ndarray, y: np.ndarray, base: np.ndarray, task: str, *,
                        shrink: float = 0.0, row_weight: np.ndarray | None = None
                        ) -> tuple[np.ndarray, np.ndarray]:
    """(weights, quality) from history rows; equal weights when no model has skill > 0."""
    q = np.atleast_1d(np.asarray(skill(task, p, y, base, row_weight), dtype=np.float64))
    raw = np.where(np.isfinite(q), np.maximum(q, 0.0), 0.0)
    w = normalise(raw) if raw.sum() > 0 else equal_weights(raw.size)
    return shrink_to_equal(w, shrink), q


def error_correlation(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Correlation matrix of the errors :math:`y - \\hat y_i` (Step 7)."""
    e = np.asarray(y, dtype=np.float64)[:, None] - np.asarray(p, dtype=np.float64)
    if e.shape[1] == 1:
        return np.ones((1, 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(e, rowvar=False)
    return np.where(np.isfinite(c), c, 0.0)


def duplicate_clusters(p: np.ndarray, threshold: float) -> list[list[int]]:
    """Groups of constituents that are copies of each other - identical columns, or
    predictions correlating at or above *threshold* (single linkage) - in the order of
    their first member. Identical columns are caught directly: a constant column has no
    defined correlation, and a copy of it is still a copy."""
    p = np.asarray(p, dtype=np.float64)
    m = p.shape[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(p, rowvar=False) if m > 1 else np.ones((1, 1))
    c = np.where(np.isfinite(c), c, 0.0)
    for i in range(m):
        for j in range(i + 1, m):
            if np.array_equal(p[:, i], p[:, j], equal_nan=True):
                c[i, j] = c[j, i] = 1.0
    parent = list(range(m))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(m):
        for j in range(i + 1, m):
            if c[i, j] >= threshold:
                parent[find(j)] = find(i)
    groups: dict[int, list[int]] = {}
    for i in range(m):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def diversity_weights(p: np.ndarray, y: np.ndarray, base: np.ndarray, task: str, *,
                      lam: float, shrink: float, duplicate_correlation: float
                      ) -> tuple[np.ndarray, dict[str, Any]]:
    """Quality x (1 - lambda x redundancy), one weight per duplicate cluster (Step 14)."""
    q = np.atleast_1d(np.asarray(skill(task, p, y, base), dtype=np.float64))
    clusters = duplicate_clusters(p, duplicate_correlation)
    reps = [max(g, key=lambda i: q[i] if np.isfinite(q[i]) else -np.inf) for g in clusters]
    ec = error_correlation(p[:, reps], y)
    c = len(reps)
    redundancy = np.array([float(np.mean([max(ec[a, b], 0.0) for b in range(c) if b != a]))
                           if c > 1 else 0.0 for a in range(c)])
    qr = np.array([q[i] for i in reps])
    score = np.where(np.isfinite(qr), np.maximum(qr, 0.0), 0.0) * (1.0 - lam * redundancy)
    score = np.maximum(score, 0.0)
    wc = normalise(score) if score.sum() > 0 else equal_weights(c)
    wc = shrink_to_equal(wc, shrink)
    w = np.zeros(p.shape[1])
    for g, wg in zip(clusters, wc, strict=True):
        w[g] = wg / len(g)
    return w, {"quality": q, "clusters": clusters, "redundancy": redundancy,
               "cluster_weights": wc}


def conditional_weights(p: np.ndarray, y: np.ndarray, base: np.ndarray, task: str,
                        state_prob: np.ndarray, *, shrink: float, min_rows: float
                        ) -> tuple[np.ndarray, np.ndarray, list[bool]]:
    """(K, M) weights per state from soft row weights, the unconditional (M,) weights, and
    which states had enough (effective) rows to be estimated on their own."""
    w0, _ = performance_weights(p, y, base, task, shrink=shrink)
    probs = np.asarray(state_prob, dtype=np.float64)
    ok = np.isfinite(probs).all(axis=1)
    out = np.empty((probs.shape[1], p.shape[1]))
    own = []
    for k in range(probs.shape[1]):
        rw = np.where(ok, probs[:, k], 0.0)
        if rw.sum() < min_rows:
            out[k] = w0
            own.append(False)
            continue
        out[k], _ = performance_weights(p, y, base, task, shrink=shrink, row_weight=rw)
        own.append(True)
    return out, w0, own


def apply_state_weights(state_prob: np.ndarray, state_weights: np.ndarray,
                        fallback: np.ndarray) -> np.ndarray:
    """Per-row weights :math:`\\sum_k P(S_t=k) w_{\\cdot,k}`; rows without a state get *fallback*."""
    probs = np.asarray(state_prob, dtype=np.float64)
    ok = np.isfinite(probs).all(axis=1) & (probs.sum(axis=1) > 0)
    out = np.tile(np.asarray(fallback, dtype=np.float64), (probs.shape[0], 1))
    if ok.any():
        pk = probs[ok] / probs[ok].sum(axis=1, keepdims=True)
        out[ok] = pk @ state_weights
    return out


@dataclass
class DynamicWeights:
    """Trailing weights of the evaluated rows, and the per-update table."""

    row_weights: np.ndarray            # (n_eval, M)
    update_positions: np.ndarray       # position (in the pair) of each update's first row
    update_weights: np.ndarray         # (n_updates, M)
    update_rows_used: np.ndarray       # resolved rows behind each update


def dynamic_weights(p: np.ndarray, y: np.ndarray, base: np.ndarray, rows: np.ndarray,
                    day: np.ndarray, evaluate: np.ndarray, *, task: str, horizon: int,
                    window_days: int, min_rows: int, shrink: float,
                    first_bar: np.ndarray | None = None,
                    valid: np.ndarray | None = None) -> DynamicWeights:
    """Daily trailing-skill weights for the positions *evaluate* (Steps 30-32).

    *p*, *y*, *base*, *rows* (bar numbers) and *day* (trading-day index, non-decreasing)
    describe every out-of-sample row in time order; only rows whose label resolved
    before the update bar (``rows + h < bar``) and that fall in the last *window_days*
    trading days enter a day's weights. The update bar is the first bar of the day -
    *first_bar* gives it per row from the full timeline when *p* holds only a subset
    of the day's bars (otherwise the first row of the day here). Rows with ``valid``
    False (a missing prediction or label) keep their place in time but contribute no
    loss and do not count toward *min_rows* - nothing is made up for them.
    """
    p = np.asarray(p, dtype=np.float64)
    n, m = p.shape
    if not (np.diff(rows) > 0).all() or not (np.diff(day) >= 0).all():
        raise ValueError("rows must be strictly increasing and days non-decreasing")
    ok = (np.isfinite(p).all(axis=1) & np.isfinite(np.asarray(y, dtype=np.float64))
          & np.isfinite(np.asarray(base, dtype=np.float64)))
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)
    loss = np.where(ok[:, None], loss_rows(task, np.where(ok[:, None], p, 0.5),
                                           np.where(ok, y, 0.0)), 0.0)
    lbase = np.where(ok, loss_rows(task, np.where(ok, base, 0.5), np.where(ok, y, 0.0)), 0.0)
    cum = np.vstack([np.zeros((1, m)), np.cumsum(loss, axis=0)])
    cumb = np.concatenate([[0.0], np.cumsum(lbase)])
    cumn = np.concatenate([[0], np.cumsum(ok.astype(np.int64))])
    evaluate = np.asarray(evaluate, dtype=np.int64)
    eval_days = day[evaluate]
    starts = evaluate[np.r_[True, np.diff(eval_days) > 0]]       # first evaluated row of a day
    day_first = np.searchsorted(day, day, side="left")           # first row of each row's day
    w_rows = np.empty((evaluate.size, m))
    upd_w = np.empty((starts.size, m))
    used = np.empty(starts.size, dtype=np.int64)
    for u, s in enumerate(starts):
        first = int(day_first[s])                                # the day's first row here
        bar = int(rows[first]) if first_bar is None else min(int(rows[first]),
                                                             int(first_bar[s]))
        hi = int(np.searchsorted(rows, bar - horizon, side="left"))   # rows + h < bar
        lo = int(np.searchsorted(day, int(day[first]) - window_days, side="left"))
        hi = max(hi, lo)
        used[u] = int(cumn[hi] - cumn[lo])                      # valid resolved rows only
        den = cumb[hi] - cumb[lo]
        if used[u] < min_rows or den <= 0:
            upd_w[u] = equal_weights(m)
            continue
        q = 1.0 - (cum[hi] - cum[lo]) / den
        raw = np.maximum(np.where(np.isfinite(q), q, 0.0), 0.0)
        w = normalise(raw) if raw.sum() > 0 else equal_weights(m)
        upd_w[u] = shrink_to_equal(w, shrink)
    which = np.searchsorted(starts, evaluate, side="right") - 1
    w_rows[:] = upd_w[which]
    return DynamicWeights(row_weights=w_rows, update_positions=starts, update_weights=upd_w,
                          update_rows_used=used)


def weight_entropy(w: np.ndarray) -> tuple[float, float]:
    """(H_w = -sum w ln w, effective number of models exp(H_w)) (Step 56)."""
    w = np.asarray(w, dtype=np.float64)
    nz = w[w > 0]
    h = float(-(nz * np.log(nz)).sum())
    return h, float(np.exp(h))


def turnover(weights: np.ndarray) -> np.ndarray:
    """:math:`\\sum_i |w_{i,t} - w_{i,t-1}|` between consecutive weight vectors (Step 57)."""
    w = np.asarray(weights, dtype=np.float64)
    if w.shape[0] < 2:
        return np.zeros(0)
    return np.abs(np.diff(w, axis=0)).sum(axis=1)
