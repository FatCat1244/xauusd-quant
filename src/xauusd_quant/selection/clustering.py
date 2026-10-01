r"""Hierarchical clusters, representatives and parameter-family reduction
(Prompt #9, Steps 10-12).

Clusters: average linkage on ``d = 1 - |Spearman|`` over development rows, cut
at ``1 - cluster_abs_corr``. Within a cluster the **representative** is chosen
by broad evidence, never by the largest full-history IC. The documented,
lexicographic rule (config ``representative``):

1. passes the development null screen for some target kind;
2. yearly sign consistency of its best kind (rounded to ``consistency_round``);
3. |median yearly rank IC| of that kind;
4. recent-development relevance (|IC| in the recent development years with
   the pooled sign; 0 if reversed);
5. lower missingness in development;
6. cheaper declared live cost;
7. shorter warm-up;
8. name (a deterministic tie-break).

**Parameter families** (one definition, several windows): the members are
kept greedily in representative order, and a member is dropped when its
|Spearman| with an already-kept member of the family is at least the
cluster threshold - "one representative, or a small diverse subset".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..features.redundancy import cluster_features

__all__ = [
    "EvidenceKey",
    "cluster_members",
    "parameter_family_reduction",
    "representative_order",
]

_COST = {"cheap": 0, "moderate": 1, "expensive": 2}


@dataclass(frozen=True)
class EvidenceKey:
    """One feature's evidence for the lexicographic representative rule."""

    name: str
    passes_null: bool
    sign_consistency: float | None
    median_yearly_ic: float | None
    recent_relevance: float | None
    missing_development: float
    cost: str | None
    min_history: int

    def sort_key(self, round_to: float) -> tuple[Any, ...]:
        cons = self.sign_consistency if self.sign_consistency is not None else 0.0
        return (0 if self.passes_null else 1,
                -round(cons / round_to) * round_to,
                -abs(self.median_yearly_ic or 0.0),
                -(self.recent_relevance or 0.0),
                self.missing_development,
                _COST.get(str(self.cost), 3),
                self.min_history,
                self.name)


def representative_order(keys: list[EvidenceKey], round_to: float) -> list[EvidenceKey]:
    return sorted(keys, key=lambda k: k.sort_key(round_to))


def cluster_members(abs_spearman: np.ndarray, names: list[str], *, threshold: float,
                    method: str = "average") -> tuple[dict[int, list[str]], np.ndarray]:
    labels, link = cluster_features(abs_spearman, threshold=threshold, method=method)
    out: dict[int, list[str]] = {}
    for name, label in zip(names, labels, strict=True):
        out.setdefault(int(label), []).append(name)
    return out, link


def parameter_family_reduction(names: list[str], registry: dict[str, dict[str, Any]],
                               abs_spearman: np.ndarray, keys: dict[str, EvidenceKey], *,
                               threshold: float, round_to: float) -> list[dict[str, Any]]:
    """Per parameter family: kept members (greedy, representative order) and the dropped ones."""
    index = {n: i for i, n in enumerate(names)}
    families: dict[str, list[str]] = {}
    for n in names:
        fam = (registry.get(n) or {}).get("parameter_family")
        if fam:
            families.setdefault(str(fam), []).append(n)
    rows = []
    for fam, members in sorted(families.items()):
        if len(members) < 2:
            continue
        ordered = [k.name for k in representative_order([keys[m] for m in members if m in keys],
                                                         round_to)]
        kept: list[str] = []
        dropped: list[dict[str, Any]] = []
        for m in ordered:
            rho = [abs_spearman[index[m], index[k]] for k in kept]
            near = max(rho) if rho else 0.0
            if rho and near >= threshold:
                twin = kept[int(np.argmax(rho))]
                dropped.append({"feature": m, "near_duplicate_of": twin, "abs_spearman": near})
            else:
                kept.append(m)
        windows = sorted(((registry[m].get("window") or 0), m) for m in members)
        adjacent = [float(abs_spearman[index[a], index[b]])
                    for (_, a), (_, b) in zip(windows, windows[1:], strict=False)]
        rows.append({"parameter_family": fam, "members": [m for _, m in windows],
                     "adjacent_abs_spearman": adjacent, "kept": kept, "dropped": dropped,
                     "decision": ("one representative" if len(kept) == 1
                                  else f"a diverse subset of {len(kept)}")})
    return rows
