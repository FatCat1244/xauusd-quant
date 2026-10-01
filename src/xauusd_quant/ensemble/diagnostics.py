r"""Where ensembles fail, and whether the information lasts (Prompt #11, Steps 37-38,
46-50).

* :func:`block_ood_percentiles`: the Prompt #10 out-of-distribution score - a
  robust Mahalanobis distance of a bar's features from the training rows of the
  fold that predicted it, as a percentile of the training rows' own distances.
  One scorer per walk-forward block, fitted on that block's fitting rows only.
* :func:`condition_codes`: causal groupings of every row (regime state, volatility
  and spread quartiles, session, spectral and wavelet entropy quintiles, OOD
  decile, disagreement quintile).
* :func:`failure_table`: the largest ``1 - q`` share of ensemble losses - how much
  more often they happen in each condition than overall (lift), and how often the
  individual models were wrong on the same rows (Step 49).
* :func:`common_mode`: rows where *every* constituent is beyond its own
  ``q`` loss quantile at once, against the rate independent errors would give
  (:math:`(1-q)^M`) - shared information fails together (Step 50).
* :func:`loss_by_quantile`: mean loss of the ensemble / best model by quantile of
  an uncertainty proxy (disagreement, OOD; Steps 46, 48).
* :func:`alpha_decay`: rank IC of one score with the same outcome family at other
  horizons (Step 37).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..alpha.information_coefficient import rank_scores
from ..ml.diagnostics import OODScorer

__all__ = ["alpha_decay", "block_ood_percentiles", "common_mode", "condition_codes",
           "failure_table", "loss_by_quantile", "quantile_codes"]


def block_ood_percentiles(x: np.ndarray, folds: list[Any]) -> np.ndarray:
    """OOD percentile of every validation row, from a scorer fitted on its fold's fit rows."""
    out = np.full(x.shape[0], np.nan)
    for fold in folds:
        tr = x[fold.fit[0]:fold.fit[1]]
        if tr.shape[0] < 1000:
            continue
        scorer = OODScorer.fit(tr)
        lo, hi = fold.validate
        out[lo:hi] = scorer.percentile(x[lo:hi])
    return out


def quantile_codes(v: np.ndarray, q: int) -> np.ndarray:
    """0..q-1 by rank among the finite values (-1 where missing)."""
    v = np.asarray(v, dtype=np.float64)
    code = np.full(v.size, -1, dtype=np.int64)
    ok = np.isfinite(v)
    if ok.sum() >= q:
        code[ok] = np.clip((rank_scores(v[ok]) * q).astype(np.int64), 0, q - 1)
    return code


def condition_codes(context: dict[str, np.ndarray], extra: dict[str, np.ndarray]
                    ) -> dict[str, np.ndarray]:
    """Grouping codes of every row (all known at t), from context arrays and extras."""
    out: dict[str, np.ndarray] = {}
    for name in ("regime_state", "vol_quartile", "spread_quartile", "session"):
        if name in context:
            out[name] = np.asarray(context[name], dtype=np.int64)
    if "fft_entropy_256" in context:
        out["spectral_entropy_quintile"] = quantile_codes(context["fft_entropy_256"], 5)
    if "wav_entropy_512" in context:
        out["wavelet_entropy_quintile"] = quantile_codes(context["wav_entropy_512"], 5)
    if "ood_score" in extra:
        out["ood_decile"] = quantile_codes(extra["ood_score"], 10)
    if "disagreement" in extra:
        out["disagreement_quintile"] = quantile_codes(extra["disagreement"], 5)
    return out


def failure_table(ens_loss: np.ndarray, member_loss: np.ndarray, members: list[str],
                  codes: dict[str, np.ndarray], *, quantile: float, min_rows: int = 200
                  ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """(lift by condition, summary) of the largest ensemble losses."""
    ok = np.isfinite(ens_loss) & np.isfinite(member_loss).all(axis=1)
    cut = float(np.quantile(ens_loss[ok], quantile))
    big = ok & (ens_loss >= cut)
    overall = float(big.sum()) / max(1.0, float(ok.sum()))
    rows = []
    for name, code in codes.items():
        for c in np.unique(code[ok]):
            if c < 0:
                continue
            m = ok & (code == c)
            if m.sum() < min_rows:
                continue
            rate = float(big[m].sum() / m.sum())
            rows.append({"condition": name, "code": int(c), "rows": int(m.sum()),
                         "large_error_rate": rate, "lift": rate / overall if overall else None})
    member_cuts = np.array([np.quantile(member_loss[ok, j], quantile)
                            for j in range(member_loss.shape[1])])
    also = (member_loss[big] >= member_cuts[None, :])
    summary = {"rows": int(ok.sum()), "large_error_rows": int(big.sum()), "loss_cut": cut,
               "mean_ensemble_loss_large": float(ens_loss[big].mean()) if big.any() else None,
               "mean_member_loss_large": {m: float(member_loss[big, j].mean())
                                          for j, m in enumerate(members)} if big.any() else {},
               "member_also_large_share": {m: float(also[:, j].mean())
                                           for j, m in enumerate(members)} if big.any() else {},
               "all_members_also_large_share": float(also.all(axis=1).mean()) if big.any()
               else None}
    return rows, summary


def common_mode(member_loss: np.ndarray, codes: dict[str, np.ndarray], months: np.ndarray, *,
                quantile: float, min_rows: int = 200
                ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rows where every constituent is beyond its own *quantile* loss at the same time."""
    ok = np.isfinite(member_loss).all(axis=1)
    m_count = member_loss.shape[1]
    cuts = np.array([np.quantile(member_loss[ok, j], quantile) for j in range(m_count)])
    allw = ok & (member_loss >= cuts[None, :]).all(axis=1)
    rate = float(allw.sum()) / max(1.0, float(ok.sum()))
    p_one = 1.0 - quantile
    rows = []
    for name, code in codes.items():
        for c in np.unique(code[ok]):
            if c < 0:
                continue
            msk = ok & (code == c)
            if msk.sum() < min_rows:
                continue
            r = float(allw[msk].sum() / msk.sum())
            rows.append({"condition": name, "code": int(c), "rows": int(msk.sum()),
                         "common_mode_rate": r, "lift": r / rate if rate else None})
    by_month: dict[int, int] = {}
    for mo in months[allw]:
        by_month[int(mo)] = by_month.get(int(mo), 0) + 1
    top = sorted(by_month.items(), key=lambda kv: -kv[1])[:10]
    summary = {"rows": int(ok.sum()), "members": m_count, "quantile": quantile,
               "common_mode_rows": int(allw.sum()), "common_mode_rate": rate,
               "rate_if_independent": p_one ** m_count, "rate_if_identical": p_one,
               "ratio_to_independent": rate / (p_one ** m_count) if m_count else None,
               "top_months": [{"month": k, "rows": v} for k, v in top]}
    return rows, summary


def loss_by_quantile(proxy: np.ndarray, losses: dict[str, np.ndarray], *, q: int = 5
                     ) -> list[dict[str, Any]]:
    """Mean of every loss series by quantile of an uncertainty proxy."""
    code = quantile_codes(proxy, q)
    out = []
    for c in range(q):
        m = code == c
        if not m.any():
            continue
        row: dict[str, Any] = {"quantile": c + 1, "rows": int(m.sum()),
                               "mean_proxy": float(np.nanmean(np.asarray(proxy)[m]))}
        for name, v in losses.items():
            x = np.asarray(v, dtype=np.float64)[m]
            row[name] = float(np.nanmean(x)) if np.isfinite(x).any() else None
        out.append(row)
    return out


def alpha_decay(score: np.ndarray, outcomes: dict[int, np.ndarray]) -> list[dict[str, Any]]:
    """Rank IC of *score* with the outcome at every horizon (rows with both finite)."""
    out: list[dict[str, Any]] = []
    s = np.asarray(score, dtype=np.float64)
    for h, y in sorted(outcomes.items()):
        y = np.asarray(y, dtype=np.float64)
        ok = np.isfinite(s) & np.isfinite(y)
        if ok.sum() < 100:
            out.append({"horizon": int(h), "rows": int(ok.sum()), "rank_ic": None})
            continue
        r = float(np.corrcoef(rank_scores(s[ok]), rank_scores(y[ok]))[0, 1])
        out.append({"horizon": int(h), "rows": int(ok.sum()),
                    "rank_ic": r if np.isfinite(r) else None})
    return out
