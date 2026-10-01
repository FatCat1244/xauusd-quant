r"""Are the regimes stable - across refits, schemes, years and eras? (Steps 27, 39-41)

A regime that changes definition every refit is not a regime. Every
walk-forward refit is aligned to the previous one, and the components of its
change are reported **separately** (no combined score):

``mean_shift``        distance between an aligned state's means, in units of the
                      previous scaler (IQRs)
``bhattacharyya``     distance between the aligned Gaussians (means and covariances)
``log_det_change``    change in log |Sigma| (covariance volume)
``transition``        largest change in a transition probability (HMM)
``duration``          change in ln expected duration (HMM)
``overlap``           adjusted Rand index between the previous and the new model's
                      labels on the *same* bars (the new period): does the new model
                      carve the market the same way?
``anchor``            distance of each state from its definition in the first refit:
                      drift of the definition over 2008-2026

Expanding vs rolling training (Step 27) is compared on out-of-sample
likelihood, state consistency (agreement on common bars), transition
consistency and duration stability - never on trading performance.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import polars as pl

from .regime_analysis import agreement

__all__ = [
    "defensible_states",
    "era_definitions",
    "finite_or_none",
    "k_rule_verdict",
    "refit_stability_summary",
    "scheme_comparison",
]


def _json_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    try:
        out = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError):
        return None
    return out if isinstance(out, list) else None


def refit_stability_summary(refits: pl.DataFrame) -> dict[str, Any]:
    """Median and 90th percentile of each stability component across refits."""
    fitted = refits.filter(pl.col("status") == "fitted") if "status" in refits.columns else refits
    out: dict[str, Any] = {"refits": fitted.height}
    if fitted.is_empty():
        return out
    for column in ("drift_max_mean_shift_scaled", "drift_max_bhattacharyya",
                   "drift_max_transition_change", "alignment_cost", "overlap_ari",
                   "overlap_agreement", "oos_ll_per_obs"):
        if column in fitted.columns:
            v = fitted[column].cast(pl.Float64).drop_nulls().to_numpy()
            v = v[np.isfinite(v)]
            if v.size:
                out[f"{column}__median"] = float(np.median(v))
                out[f"{column}__p10"] = float(np.quantile(v, 0.1))
                out[f"{column}__p90"] = float(np.quantile(v, 0.9))
    anchors = [a for a in (_json_list(x) for x in fitted["anchor_bhattacharyya"].to_list())
               if a] if "anchor_bhattacharyya" in fitted.columns else []
    if anchors:
        last = np.asarray(anchors[-1], dtype=np.float64)
        out["final_anchor_bhattacharyya"] = last.tolist()
        out["max_anchor_bhattacharyya"] = float(np.nanmax(np.asarray(anchors, dtype=np.float64)))
    durations = [d for d in (_json_list(x) for x in fitted["expected_durations"].to_list())
                 if d] if "expected_durations" in fitted.columns else []
    if durations:
        arr = np.log(np.clip(np.asarray(durations, dtype=np.float64), 1.0, 1e9))
        out["expected_duration_median_by_state"] = np.exp(np.median(arr, axis=0)).tolist()
        out["expected_duration_log_sd_by_state"] = arr.std(axis=0).tolist()
    flags = fitted["flag_degenerate"].to_list() if "flag_degenerate" in fitted.columns else []
    out["degenerate_refits"] = int(sum(bool(f) for f in flags))
    if "flag_flags" in fitted.columns:
        names = [f for value in fitted["flag_flags"].to_list() if value for f in value.split(",")]
        out["degenerate_flag_counts"] = {n: names.count(n) for n in sorted(set(names))}
    return out


def era_definitions(refits: pl.DataFrame, features: tuple[str, ...], *,
                    eras: tuple[tuple[int, int], ...]) -> pl.DataFrame:
    """Mean (over refits in each era) of every aligned state's raw feature means (answer F)."""
    fitted = refits.filter(pl.col("status") == "fitted") if "status" in refits.columns else refits
    rows = []
    for record in fitted.iter_rows(named=True):
        means = _json_list(record.get("means_raw"))
        if not means:
            continue
        year = int(str(record["infer_start"])[:4])
        era = next((f"{a}-{b}" for a, b in eras if a <= year <= b), None)
        if era is None:
            continue
        for state, values in enumerate(means):
            for name, value in zip(features, values, strict=False):
                rows.append({"era": era, "state": state, "feature": name, "value": value})
    if not rows:
        return pl.DataFrame()
    return (pl.DataFrame(rows).group_by("era", "state", "feature")
            .agg(pl.col("value").mean().alias("mean_of_state_means"), pl.len().alias("refits"))
            .sort("state", "feature", "era"))


def scheme_comparison(results: dict[str, Any], family: str) -> list[dict[str, Any]]:
    """Expanding vs rolling (and quarterly vs monthly): likelihood, consistency, stability."""
    rows = []
    frames = {name: r.frame for name, r in results.items() if r is not None and
              not r.frame.is_empty()}
    for name, result in results.items():
        if result is None:
            continue
        summary = refit_stability_summary(result.refits)
        fitted = result.refits.filter(pl.col("status") == "fitted") if "status" in \
            result.refits.columns else result.refits
        weighted = None
        if {"oos_complete_rows", "oos_ll_per_obs"} <= set(fitted.columns):
            weights = fitted["oos_complete_rows"].cast(pl.Float64).to_numpy()
            ll = fitted["oos_ll_per_obs"].cast(pl.Float64).to_numpy()
            ok = np.isfinite(ll) & (weights > 0)
            if ok.any():
                weighted = float(np.average(ll[ok], weights=weights[ok]))
        rows.append({
            "scheme": name, "refits": summary.get("refits"),
            "oos_ll_per_obs_weighted": weighted,
            "overlap_ari_median": summary.get("overlap_ari__median"),
            "max_bhattacharyya_drift_median": summary.get("drift_max_bhattacharyya__median"),
            "max_transition_change_median": summary.get("drift_max_transition_change__median"),
            "expected_duration_log_sd": summary.get("expected_duration_log_sd_by_state"),
            "degenerate_refits": summary.get("degenerate_refits"),
            "fit_seconds_total": float(fitted["fit_seconds"].sum()) if "fit_seconds" in
            fitted.columns else None,
        })
    names = list(frames)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            joined = frames[a].select("timestamp", pl.col(f"{family}_state").alias("a")).join(
                frames[b].select("timestamp", pl.col(f"{family}_state").alias("b")),
                on="timestamp")
            if joined.height:
                rows.append({"scheme": f"{a} vs {b}", **{
                    f"state_{k}": v for k, v in agreement(joined["a"].to_numpy(),
                                                          joined["b"].to_numpy()).items()}})
    return rows


def defensible_states(candidates: list[dict[str, Any]], *, margin: float,
                      min_overlap_ari: float) -> dict[str, Any]:
    """The registered K rule (REG-H-004), applied to one model family on one timeframe.

    ``candidates`` holds one row per K with ``test_ll_per_obs`` (chronological
    hold-out), ``degenerate`` (offline full fit) and ``overlap_ari`` (median
    walk-forward refit overlap); K = 1 carries only its hold-out likelihood.
    K* is the largest K such that every step 1 -> 2 -> ... -> K improves the
    hold-out likelihood by more than *margin* nats per bar, and every K on the
    way is non-degenerate with a refit overlap of at least *min_overlap_ari*.
    K* = 1 means no number of discrete states is defensible.

    A criterion that cannot be evaluated - no hold-out likelihood, or no
    walk-forward (a model run offline only), or a non-finite value - stops the
    rule *incomplete* (``complete`` False): K* is then only a lower bound, and
    :func:`k_rule_verdict` reads it as untested unless K >= 2 was already reached.
    """
    by_k = {int(c["states"]): c for c in candidates}
    best = 1
    complete = True
    reasons = []
    for k in sorted(by_k):
        if k == 1:
            continue
        cur, prev = by_k[k], by_k.get(k - 1)
        ll, ll_prev = finite_or_none(cur.get("test_ll_per_obs")), finite_or_none(
            (prev or {}).get("test_ll_per_obs"))
        if ll is None or ll_prev is None:
            reasons.append(f"K={k}: hold-out likelihood unavailable (untested)")
            complete = False
            break
        if ll - ll_prev <= margin:
            reasons.append(f"K={k}: hold-out gain {ll - ll_prev:+.4f} <= {margin}")
            break
        if cur.get("degenerate"):
            reasons.append(f"K={k}: degenerate ({cur.get('flags')})")
            break
        ari = finite_or_none(cur.get("overlap_ari"))
        if ari is None:
            reasons.append(f"K={k}: no walk-forward refit overlap (untested)")
            complete = False
            break
        if ari < min_overlap_ari:
            reasons.append(f"K={k}: refit overlap ARI {ari:.3f} < {min_overlap_ari}")
            break
        best = k
    return {"defensible_states": best, "complete": complete,
            "stopped_because": reasons[-1] if reasons else "every K passed",
            "margin": margin, "min_overlap_ari": min_overlap_ari}


def k_rule_verdict(rule: dict[str, Any]) -> str:
    """REG-H-004's verdict from :func:`defensible_states` (an unavailable step is untested)."""
    if rule["defensible_states"] >= 2:
        return "discrete_states_defensible"
    return "no_discrete_states_defensible" if rule.get("complete", True) else "untested"


def finite_or_none(value: Any) -> float | None:
    """A number a verdict may compare, or None (NaN fails every comparison silently)."""
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None
