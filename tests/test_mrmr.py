"""mRMR prefers relevant, non-redundant features (Prompt #9, Steps 13, 41, 69).

On controlled data where the target is ``a + b``: IC ranking takes ``a`` and
its near-copies first; mRMR takes ``a`` then ``b``; the hard |rho| limit never
admits a near-duplicate; the effect-stability score zeroes a coin-flip sign.
"""

from __future__ import annotations

import numpy as np
import pytest

from xauusd_quant.features.redundancy import pairwise_correlation
from xauusd_quant.selection.mrmr import effect_stability_score, ic_ranking, mrmr


def _problem(n: int = 30_000) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(7)
    a, b = rng.normal(size=(2, n))
    y = a + 0.7 * b + 2.0 * rng.normal(size=n)
    feats = {"a": a, "a_copy": a + 0.05 * rng.normal(size=n),
             "a_copy2": a + 0.2 * rng.normal(size=n), "b": b,
             "noise": rng.normal(size=n)}
    x = np.column_stack(list(feats.values()))
    rel = np.abs([np.corrcoef(x[:, j], y)[0, 1] for j in range(x.shape[1])])
    rho, _ = pairwise_correlation(x, rank=True)
    return rel, np.abs(rho), list(feats)


def test_ic_ranking_picks_the_near_copies_and_mrmr_does_not() -> None:
    rel, corr, names = _problem()
    cand = list(range(len(names)))
    naive = [names[i] for i in ic_ranking(rel, cand, k=3)]
    assert set(naive) == {"a", "a_copy", "a_copy2"}
    steps = mrmr(rel, corr, cand, k=2, lam=1.0, hard_limit=0.95)
    assert [names[s.index] for s in steps] == [naive[0], "b"]
    assert steps[1].redundancy < 0.1 and steps[0].redundancy == 0.0


@pytest.mark.parametrize("scheme", ["difference", "quotient"])
def test_both_schemes_prefer_the_non_redundant_feature(scheme: str) -> None:
    rel, corr, names = _problem()
    steps = mrmr(rel, corr, list(range(len(names))), k=2, scheme=scheme, hard_limit=0.999)
    assert names[steps[1].index] == "b"


def test_the_hard_limit_never_admits_a_near_duplicate() -> None:
    rel, corr, names = _problem()
    steps = mrmr(rel, corr, list(range(len(names))), k=5, lam=0.0, hard_limit=0.95)
    chosen = [s.index for s in steps]
    assert all(corr[i, j] < 0.95 for i in chosen for j in chosen if i != j)
    assert "a_copy" not in [names[i] for i in chosen] or "a" not in [names[i] for i in chosen]


def test_candidates_limit_and_order_are_respected() -> None:
    rel, corr, names = _problem()
    only = [names.index("b"), names.index("noise")]
    steps = mrmr(rel, corr, only, k=5)
    assert [names[s.index] for s in steps] == ["b", "noise"]
    assert mrmr(rel, corr, [], k=5) == [] and mrmr(rel, corr, only, k=0) == []


def test_effect_stability_score() -> None:
    score = effect_stability_score(np.array([0.1, 0.1, -0.2, np.nan]),
                                   np.array([1.0, 0.5, 0.75, 0.9]))
    np.testing.assert_allclose(score, [0.1, 0.0, 0.1, 0.0])
