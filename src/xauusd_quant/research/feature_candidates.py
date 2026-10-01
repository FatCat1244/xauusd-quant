r"""Representatives of redundant features, the candidate manifest, ablation sets (Steps 42, 67-75).

**Representatives.** Features in one hierarchical cluster (|Spearman| >= the
cluster threshold, average linkage) carry the same information. Among the
members that beat the nulls, one is kept - by, in order: the better
preliminary status, the larger evidence score (|rank IC| x yearly sign
consistency), the cheaper live cost, the shorter warm-up, the name. Every other
passing member whose own |rho| with the representative is at least
``max_redundancy_rho`` becomes ``redundant``. This is a choice among
near-duplicates for the next stage's convenience, not a claim that the
representative is the "true" signal; the reason string says who was kept.

**Candidate manifest.** ``candidate_features.json`` lists the features whose
final status is strong_candidate, candidate or weak_candidate, each with the
evidence behind the status, its cost and warm-up, and the features it
represents. Nothing here is a trading signal.

**Ablation sets.** The cumulative family groups A-G of ``config/features.yaml``
(plus H = G + registered interactions) as feature-id lists, twice: every
default-eligible registered feature (prior status allows it, live-safe,
valid), and only the research candidates. Prompt #9 would compare models on
these sets; nothing is fitted here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..alpha.ranking import status_rank

__all__ = ["ablation_sets", "candidate_manifest", "choose_representatives"]

_CANDIDATES = ("strong_candidate", "candidate", "weak_candidate")
_COST_ORDER = {"cheap": 0, "moderate": 1, "expensive": 2}


def choose_representatives(prelim: dict[str, dict[str, Any]], clusters: dict[str, int],
                           abs_rho: dict[tuple[str, str], float], *, threshold: float
                           ) -> dict[str, str]:
    """feature -> the representative that makes it redundant (only for redundant features).

    *prelim* maps each feature to its preliminary ``status`` (classified with
    no redundancy), ``score`` (|rank IC| x sign consistency), ``cost`` and
    ``min_history``; *abs_rho* holds |Spearman| of feature pairs.
    """
    members: dict[int, list[str]] = {}
    for f, c in clusters.items():
        members.setdefault(c, []).append(f)
    out: dict[str, str] = {}
    for group in members.values():
        passing = [f for f in group if prelim.get(f, {}).get("status") in _CANDIDATES]
        if len(passing) < 2:
            continue

        def key(f: str) -> tuple[Any, ...]:
            p = prelim[f]
            score = p.get("score")
            return (status_rank(p["status"]),
                    -(score if score is not None and np.isfinite(score) else 0.0),
                    _COST_ORDER.get(str(p.get("cost")), 3), int(p.get("min_history") or 0), f)

        ordered = sorted(passing, key=key)
        rep = ordered[0]
        for f in ordered[1:]:
            rho = abs_rho.get((f, rep), abs_rho.get((rep, f)))
            if rho is not None and np.isfinite(rho) and rho >= threshold:
                out[f] = rep
    return out


def candidate_manifest(timeframe: str, status_rows: list[dict[str, Any]],
                       provenance: dict[str, Any]) -> dict[str, Any]:
    """The machine-readable list of research candidates of one timeframe."""
    kept = [r for r in status_rows if r.get("status") in _CANDIDATES]
    kept.sort(key=lambda r: (status_rank(r["status"]), -(r.get("best_abs_rank_ic") or 0.0)))
    represents: dict[str, list[str]] = {}
    for r in status_rows:
        if r.get("status") == "redundant" and r.get("representative"):
            represents.setdefault(r["representative"], []).append(r["feature"])
    counts: dict[str, int] = {}
    for r in status_rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    fields = ("feature_id", "family", "status", "reason", "best_kind", "best_target",
              "best_rank_ic", "best_q_value", "sign_consistency", "recent_ic", "era_sign_flips",
              "beyond_pipeline_null", "cluster", "window", "min_history", "cost",
              "measured_cost", "incremental_update", "prior_status", "kinds_passing")
    return {
        "timeframe": timeframe,
        "what_this_is": ("features whose rank IC with some outcome kind beats the circular-shift "
                         "null, survives FDR, is not matched on the random-walk / shuffled "
                         "pipeline nulls, and keeps its sign through time; descriptive "
                         "research evidence, not a trading signal and not a model"),
        "status_counts": dict(sorted(counts.items(), key=lambda kv: status_rank(kv[0]))),
        "features": [{"feature": r["feature"], **{k: r.get(k) for k in fields},
                      "represents": sorted(represents.get(r["feature"], []))} for r in kept],
        "provenance": provenance,
    }


def ablation_sets(registry: list[dict[str, Any]], statuses: dict[str, str],
                  groups: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    """Cumulative family groups A-G (+ H with interactions): registry defaults and candidates."""
    order = list(groups)
    full = dict(groups)
    if order:
        full["H"] = (*groups[order[-1]], "interaction")

    def ids(families: tuple[str, ...], *, research: bool) -> list[str]:
        out = []
        for r in registry:
            if r.get("family") not in families or r.get("excluded_from_default_candidates"):
                continue
            if research and statuses.get(r["name"]) not in _CANDIDATES:
                continue
            out.append(str(r["feature_id"]))
        return out

    return {
        "definition": {k: list(v) for k, v in full.items()},
        "registry_default": {k: ids(v, research=False) for k, v in full.items()},
        "research_candidates": {k: ids(v, research=True) for k, v in full.items()},
        "note": ("A-G add one family group at a time; H adds the registered interactions. "
                 "registry_default = live-safe, valid features whose prior status does not "
                 "exclude them; research_candidates = those with a candidate status here"),
    }
