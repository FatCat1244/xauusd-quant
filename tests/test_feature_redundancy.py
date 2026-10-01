"""Redundancy: correlations, clusters, graph, non-monotone duplicates, representatives
(Prompt #8, Steps 38-42)."""

from __future__ import annotations

import numpy as np
import pytest

from xauusd_quant.features.redundancy import (
    cluster_features,
    evenly_spaced_rows,
    feature_nmi_matrix,
    nonmonotone_pairs,
    pairwise_correlation,
    redundancy_groups,
)
from xauusd_quant.research.feature_candidates import (
    ablation_sets,
    candidate_manifest,
    choose_representatives,
)


def _features(n: int = 6000, seed: int = 0) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = a + 0.05 * rng.normal(size=n)                        # near-duplicate of a
    c = rng.normal(size=n)                                   # independent
    d = np.abs(c) + 0.05 * rng.normal(size=n)                # V-shaped in c: |rho| ~ 0
    x = np.column_stack([a, b, c, d])
    x[::11, 1] = np.nan
    return x, ["a", "b", "c", "d"]


def test_pairwise_correlation_is_pairwise_complete() -> None:
    x, _ = _features()
    r, n = pairwise_correlation(x)
    ok = np.isfinite(x[:, 0]) & np.isfinite(x[:, 1])
    assert r[0, 1] == pytest.approx(np.corrcoef(x[ok, 0], x[ok, 1])[0, 1], abs=1e-10)
    assert n[0, 1] == ok.sum() and n[0, 2] == x.shape[0]
    rs, _ = pairwise_correlation(x, rank=True)
    assert abs(rs[2, 3]) < 0.05 and rs[0, 1] > 0.99


def test_clusters_and_graph_join_only_near_duplicates() -> None:
    x, names = _features()
    rs, _ = pairwise_correlation(x, rank=True)
    labels, _ = cluster_features(np.abs(rs), threshold=0.9)
    assert labels[0] == labels[1] and len({labels[0], labels[2], labels[3]}) == 3
    groups, edges = redundancy_groups(np.abs(rs), names, threshold=0.7)
    assert groups == [["a", "b"]] and len(edges) == 1


def test_mutual_information_finds_a_v_shaped_duplicate() -> None:
    x, names = _features()
    rs, _ = pairwise_correlation(x, rank=True)
    nmi = feature_nmi_matrix(x, bins=20)
    hidden = nonmonotone_pairs(nmi, names, np.abs(rs), nmi_threshold=0.3, corr_ceiling=0.7)
    assert [(h["a"], h["b"]) for h in hidden] == [("c", "d")]


def test_evenly_spaced_rows() -> None:
    rows = evenly_spaced_rows(1000, 10)
    assert rows[0] == 0 and rows[-1] == 999 and rows.size == 10
    assert evenly_spaced_rows(5, 10).tolist() == [0, 1, 2, 3, 4]


def test_the_representative_is_the_best_evidence_and_only_close_copies_are_redundant() -> None:
    prelim = {"a": {"status": "candidate", "score": 0.05, "cost": "cheap", "min_history": 20},
              "b": {"status": "strong_candidate", "score": 0.03, "cost": "expensive",
                    "min_history": 500},
              "c": {"status": "candidate", "score": 0.20, "cost": "cheap", "min_history": 20},
              "d": {"status": "failed_null", "score": None, "cost": "cheap", "min_history": 1}}
    clusters = {"a": 1, "b": 1, "c": 1, "d": 1}
    rho = {("a", "b"): 0.95, ("b", "c"): 0.85, ("a", "c"): 0.9, ("a", "d"): 0.99}
    reps = choose_representatives(prelim, clusters, rho, threshold=0.9)
    assert reps == {"a": "b"}          # b has the better status; c is too far from b (0.85)


def test_manifests_list_candidates_and_cumulative_family_groups() -> None:
    registry = [{"name": "x", "feature_id": "X_1H", "family": "returns",
                 "excluded_from_default_candidates": False},
                {"name": "y", "feature_id": "Y_1H", "family": "fft",
                 "excluded_from_default_candidates": True},
                {"name": "z", "feature_id": "Z_1H", "family": "interaction",
                 "excluded_from_default_candidates": False}]
    sets = ablation_sets(registry, {"x": "candidate", "z": "failed_null"},
                         {"A": ("returns",), "B": ("returns", "fft")})
    assert sets["registry_default"] == {"A": ["X_1H"], "B": ["X_1H"], "H": ["X_1H", "Z_1H"]}
    assert sets["research_candidates"]["H"] == ["X_1H"]
    rows = [{"feature": "x", "status": "candidate", "best_abs_rank_ic": 0.03},
            {"feature": "w", "status": "redundant", "representative": "x"},
            {"feature": "z", "status": "failed_null"}]
    manifest = candidate_manifest("1h", rows, {"factory_version": "v"})
    assert [f["feature"] for f in manifest["features"]] == ["x"]
    assert manifest["features"][0]["represents"] == ["w"]
    assert manifest["status_counts"] == {"candidate": 1, "redundant": 1, "failed_null": 1}
