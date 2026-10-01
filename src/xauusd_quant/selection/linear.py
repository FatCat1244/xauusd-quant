r"""Diagnostic linear models on rank-transformed features (Prompt #9, Steps 20-23, 32, 35-36).

These models measure whether a reduced feature set still carries forward
information; they are not the ML system of Prompt #10.

**Rank-linear design.** Every feature is mapped through the empirical CDF of
its *training* values, :math:`u = F_{train}(x) \in (0,1)`, centred
(:math:`u - 1/2`; a missing value becomes the centre, i.e. mean imputation) and
standardised with training moments. Continuous targets are mapped through
their own training CDF the same way. This keeps heavy tails from steering the
fits, matches the rank IC used everywhere else, and needs nothing from the
evaluation rows but the frozen training CDFs.

**Sufficient statistics.** A fit needs only ``n``, :math:`\sum x`,
:math:`\sum x x^\top`, :math:`\sum y`, :math:`\sum y^2` and :math:`\sum x y`
of the training rows (:class:`Moments`); moments of disjoint blocks add, so a
resample of quarters is fitted from quarter sums with no second pass.

* :func:`ridge`: :math:`\hat\beta = (Q + \alpha I)^{-1} c` on the standardised
  Gram :math:`Q` and cross-moments :math:`c` (``alpha`` fixed in the config,
  never tuned on evaluation rows);
* :func:`lasso_path`: the Lasso path from :math:`(Q, c)` alone (active-set
  coordinate descent, warm-started down a log grid of penalties); the order in
  which features enter is a selection ranking. The **elastic net** is the naive
  elastic net of Zou and Hastie with a fixed ridge weight :math:`\lambda_2`:
  the same path on :math:`(Q + \lambda_2 I, c)` (``elastic_net_l2``);
* :func:`logistic_l1_path`: L1 logistic regression by FISTA on a subsample of
  training rows, for the binary direction view (no scikit-learn dependency);
* :func:`evaluate`: rank IC of predictions, and out-of-sample :math:`R^2` of the
  CDF-mapped target against the training mean.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..alpha.information_coefficient import rank_scores

__all__ = [
    "Moments",
    "SortedTraining",
    "auc",
    "evaluate",
    "fit_cdfs",
    "lasso_path",
    "logistic_l1_path",
    "moments_of",
    "ridge",
]


@dataclass
class SortedTraining:
    """Frozen training CDFs: the sorted finite training values of every column."""

    values: list[np.ndarray]

    def transform(self, x: np.ndarray, j: int) -> np.ndarray:
        """Centred CDF scores u - 1/2 of *x* under column *j*'s training values (0 if missing)."""
        s = self.values[j]
        out = np.zeros(x.size, dtype=np.float64)
        ok = np.isfinite(x)
        if s.size == 0:
            return out
        lo = np.searchsorted(s, x[ok], side="left")
        hi = np.searchsorted(s, x[ok], side="right")
        out[ok] = (lo + hi) / (2.0 * s.size) - 0.5
        return out

    def matrix(self, x: np.ndarray, columns: list[int]) -> np.ndarray:
        return np.column_stack([self.transform(x[:, c], i) for i, c in enumerate(columns)]) \
            if columns else np.zeros((x.shape[0], 0))


def fit_cdfs(x: np.ndarray, columns: list[int], lo: int, hi: int) -> SortedTraining:
    return SortedTraining([np.sort(v[np.isfinite(v)]) for v in
                           (np.asarray(x[lo:hi, c], dtype=np.float64) for c in columns)])


@dataclass
class Moments:
    """Sufficient statistics of a design block (additive over disjoint row blocks)."""

    n: float
    sx: np.ndarray        # (p,)
    sxx: np.ndarray       # (p, p)
    sy: np.ndarray        # (T,)
    syy: np.ndarray       # (T,)
    sxy: np.ndarray       # (p, T)

    def __add__(self, other: Moments) -> Moments:
        return Moments(self.n + other.n, self.sx + other.sx, self.sxx + other.sxx,
                       self.sy + other.sy, self.syy + other.syy, self.sxy + other.sxy)

    def standardized(self, columns: list[int] | None = None
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(Q correlation-scale Gram, c cross-moments, x means, x sds, y means)."""
        idx = np.arange(self.sx.size) if columns is None else np.asarray(columns, dtype=np.int64)
        n = self.n
        mx = self.sx[idx] / n
        cov = self.sxx[np.ix_(idx, idx)] / n - np.outer(mx, mx)
        sd = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
        q = cov / np.outer(sd, sd)
        my = self.sy / n
        cxy = self.sxy[idx] / n - np.outer(mx, my)
        c = cxy / sd[:, None]
        return q, c, mx, sd, my


def moments_of(x: np.ndarray, y: np.ndarray) -> Moments:
    """Moments of rows where every y column is finite (x already transformed, no NaN)."""
    ok = np.isfinite(y).all(axis=1)
    xs, ys = x[ok], y[ok]
    return Moments(float(ok.sum()), xs.sum(axis=0), xs.T @ xs, ys.sum(axis=0),
                   (ys * ys).sum(axis=0), xs.T @ ys)


def ridge(q: np.ndarray, c: np.ndarray, alpha: float) -> np.ndarray:
    """(p, T) coefficients on standardised features."""
    if q.shape[0] == 0:
        return np.zeros((0, c.shape[1] if c.ndim == 2 else 1))
    return np.linalg.solve(q + alpha * np.eye(q.shape[0]), c)


def _soft(z: float, t: float) -> float:
    return z - t if z > t else z + t if z < -t else 0.0


def lasso_path(q: np.ndarray, c: np.ndarray, *, l2: float = 0.0, max_features: int = 75,
               n_alphas: int = 60, min_ratio: float = 1e-3, tol: float = 1e-8,
               max_sweeps: int = 1000) -> dict[str, Any]:
    r"""Lasso path of one target from the standardised Gram (``l2`` > 0: naive elastic net).

    Minimises :math:`\tfrac12\beta^\top(Q + \lambda_2 I)\beta - c^\top\beta + \alpha|\beta|_1`
    by active-set coordinate descent, warm-started down ``n_alphas`` log-spaced
    penalties from :math:`\alpha_{max} = \max|c|`. Features are recorded in the
    order they first become non-zero (``order``); the path stops once
    *max_features* are active. Returns ``order``, ``alphas`` and ``coefs``.
    """
    p = q.shape[0]
    if p == 0 or not np.isfinite(c).all() or float(np.max(np.abs(c))) == 0.0:
        return {"order": [], "alphas": np.zeros(0), "coefs": np.zeros((0, p))}
    g = np.asarray(q, dtype=np.float64) + l2 * np.eye(p)
    diag = np.diag(g).copy()
    a_max = float(np.max(np.abs(c)))
    alphas = a_max * np.logspace(0, np.log10(min_ratio), n_alphas)
    beta = np.zeros(p)
    grad = np.asarray(c, dtype=np.float64).copy()          # c - G beta
    active: list[int] = []
    order: list[int] = []
    coefs = []
    for alpha in alphas:
        while True:
            inactive = np.setdiff1d(np.arange(p), active)
            viol = inactive[np.abs(grad[inactive]) > alpha * (1 + 1e-12)]
            for j in viol[np.argsort(-np.abs(grad[viol]))]:
                active.append(int(j))
            for _ in range(max_sweeps):
                delta = 0.0
                for j in active:
                    old = beta[j]
                    new = _soft(grad[j] + diag[j] * old, alpha) / diag[j]
                    if new != old:
                        d = new - old
                        grad -= g[:, j] * d
                        beta[j] = new
                        delta = max(delta, abs(d))
                        if old == 0.0 and j not in order:
                            order.append(j)
                if delta < tol:
                    break
            inactive = np.setdiff1d(np.arange(p), active)
            if not (np.abs(grad[inactive]) > alpha * (1 + 1e-12)).any():
                break
        coefs.append(beta.copy())
        if int((beta != 0).sum()) >= max_features:
            break
    return {"order": order[:max_features], "alphas": alphas[:len(coefs)],
            "coefs": np.asarray(coefs)}


def logistic_l1_path(x: np.ndarray, y: np.ndarray, lambdas: tuple[float, ...], *,
                     max_iter: int = 2000, tol: float = 1e-7) -> dict[str, Any]:
    r"""L1 logistic regression along decreasing penalties (x standardised, y in {0, 1}).

    Minimises :math:`\tfrac1n\sum_i \ell(b_0 + x_i^\top\beta, y_i) + \lambda|\beta|_1`
    (intercept unpenalised) by FISTA proximal gradient with the Lipschitz step
    :math:`1/L`, :math:`L = (1 + \|x\|_2^2/n)/4`, warm-started along *lambdas*.
    """
    n, p = x.shape
    lam_sorted = sorted(lambdas, reverse=True)
    lip = (1.0 + float(np.linalg.norm(x, 2)) ** 2 / n) / 4.0
    step = 1.0 / lip
    mean_y = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    b0 = np.log(mean_y / (1 - mean_y))
    beta = np.zeros(p)
    coefs = np.zeros((len(lam_sorted), p))
    for i, lam in enumerate(lam_sorted):
        zb0, zb = b0, beta.copy()
        tk = 1.0
        for _ in range(max_iter):
            eta = np.clip(zb0 + x @ zb, -30, 30)
            r = 1.0 / (1.0 + np.exp(-eta)) - y
            g0 = float(r.mean())
            g = (x.T @ r) / n
            nb0 = zb0 - step * g0
            nb = zb - step * g
            nb = np.sign(nb) * np.maximum(np.abs(nb) - step * lam, 0.0)
            tn = (1.0 + np.sqrt(1.0 + 4.0 * tk * tk)) / 2.0
            zb0 = nb0 + (tk - 1.0) / tn * (nb0 - b0)
            zb = nb + (tk - 1.0) / tn * (nb - beta)
            change = max(abs(nb0 - b0), float(np.max(np.abs(nb - beta))) if p else 0.0)
            b0, beta, tk = nb0, nb, tn
            if change < tol:
                break
        coefs[i] = beta
    entry = np.full(p, np.inf)
    for i in range(len(lam_sorted)):
        newly = (coefs[i] != 0) & ~np.isfinite(entry)
        entry[newly] = i
    return {"lambdas": np.asarray(lam_sorted), "coefs": coefs, "entry": entry}


def evaluate(pred: np.ndarray, y_raw: np.ndarray, y_cdf: np.ndarray, train_mean: float
             ) -> dict[str, float | None]:
    """Rank IC of predictions with the raw target, and R^2 of the CDF-mapped target."""
    ok = np.isfinite(pred) & np.isfinite(y_raw) & np.isfinite(y_cdf)
    if ok.sum() < 100:
        return {"rank_ic": None, "r2": None, "n": int(ok.sum())}
    with np.errstate(invalid="ignore", divide="ignore"):
        ric = float(np.corrcoef(rank_scores(pred[ok]), rank_scores(y_raw[ok]))[0, 1])
    sse = float(((y_cdf[ok] - pred[ok]) ** 2).sum())
    sst = float(((y_cdf[ok] - train_mean) ** 2).sum())
    return {"rank_ic": ric if np.isfinite(ric) else None,
            "r2": 1.0 - sse / sst if sst > 0 else None, "n": int(ok.sum())}


def auc(y: np.ndarray, score: np.ndarray) -> float | None:
    ok = np.isfinite(y) & np.isfinite(score)
    pos = y[ok] > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    r = rank_scores(score[ok]) * ok.sum() + 0.5
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
