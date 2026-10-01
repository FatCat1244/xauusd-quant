"""Clusters, representatives, parameter families and set collinearity (Prompt #9, Steps 9-12,
47-50, 69).

Highly correlated synthetic features must share a cluster; the representative
comes from the documented lexicographic rule (null screen first, then yearly
consistency, never the largest IC alone); a parameter family keeps one member
per near-duplicate group; effective rank and VIF behave as defined.
"""

from __future__ import annotations

import numpy as np
import pytest

from xauusd_quant.features.redundancy import pairwise_correlation
from xauusd_quant.selection.clustering import (
    EvidenceKey,
    cluster_members,
    parameter_family_reduction,
    representative_order,
)
from xauusd_quant.selection.redundancy import effective_rank, set_collinearity


def _data(n: int = 20_000) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(1)
    a, b, c = rng.normal(size=(3, n))
    cols = {"a_20": a, "a_40": a + 0.1 * rng.normal(size=n), "a_80": a + 0.12 * rng.normal(size=n),
            "b_20": b, "b_40": b + 0.1 * rng.normal(size=n), "c": c,
            "a_tail": np.exp(a)}                            # monotone in a: |Spearman| = 1
    return np.column_stack(list(cols.values())), list(cols)


def test_highly_correlated_features_cluster_together() -> None:
    x, names = _data()
    rho, _ = pairwise_correlation(x, rank=True)
    clusters, link = cluster_members(np.abs(rho), names, threshold=0.9)
    groups = sorted(sorted(m) for m in clusters.values())
    assert sorted(["a_20", "a_40", "a_80", "a_tail"]) in groups
    assert ["b_20", "b_40"] in groups and ["c"] in groups
    assert link.shape == (len(names) - 1, 4)


def _key(name: str, **kw: object) -> EvidenceKey:
    base = {"passes_null": True, "sign_consistency": 0.8, "median_yearly_ic": 0.05,
            "recent_relevance": 0.05, "missing_development": 0.0, "cost": "cheap",
            "min_history": 20}
    base.update(kw)
    return EvidenceKey(name=name, **base)  # type: ignore[arg-type]


def test_the_representative_rule_is_lexicographic_and_not_the_largest_ic() -> None:
    keys = [_key("big_ic_coin_flip", sign_consistency=0.55, median_yearly_ic=0.20),
            _key("steady", sign_consistency=0.90, median_yearly_ic=0.04),
            _key("fails_null", passes_null=False, sign_consistency=1.0, median_yearly_ic=0.5)]
    order = [k.name for k in representative_order(keys, 0.1)]
    assert order == ["steady", "big_ic_coin_flip", "fails_null"]
    # equal evidence: lower missingness, then cheaper, then shorter warm-up, then the name
    tied = [_key("z", missing_development=0.0, cost="expensive"),
            _key("y", missing_development=0.0, cost="cheap", min_history=500),
            _key("x", missing_development=0.0, cost="cheap", min_history=20),
            _key("w", missing_development=0.3)]
    assert [k.name for k in representative_order(tied, 0.1)] == ["x", "y", "z", "w"]


def test_parameter_family_keeps_one_member_per_near_duplicate_group() -> None:
    x, names = _data()
    rho, _ = pairwise_correlation(x, rank=True)
    registry = {n: {"parameter_family": n.rsplit("_", 1)[0] if n != "c" else None,
                    "window": int(n.rsplit("_", 1)[1]) if n[-1].isdigit() else 0}
                for n in names}
    registry["a_tail"]["parameter_family"] = None
    keys = {n: _key(n) for n in names}
    rows = {r["parameter_family"]: r for r in parameter_family_reduction(
        names, registry, np.abs(rho), keys, threshold=0.9, round_to=0.1)}
    assert rows["a"]["kept"] == ["a_20"] and len(rows["a"]["dropped"]) == 2
    assert rows["b"]["kept"] == ["b_20"]
    assert rows["a"]["members"] == ["a_20", "a_40", "a_80"]           # window order


def test_effective_rank_and_vif() -> None:
    assert effective_rank(np.eye(6)) == pytest.approx(6.0)
    assert effective_rank(np.ones((6, 6))) == pytest.approx(1.0, abs=1e-6)
    x, names = _data()
    rho, _ = pairwise_correlation(x, rank=True)
    independent = set_collinearity(rho, [0, 3, 5], names)
    assert independent["max_vif"] < 1.05 and independent["effective_rank"] > 2.9
    dup = set_collinearity(rho, [0, 1, 2, 3], names)
    assert dup["max_vif"] > 20 and dup["condition_number"] > 50
    assert sorted(dup["max_pair"]) in (["a_20", "a_40"], ["a_40", "a_80"], ["a_20", "a_80"])
