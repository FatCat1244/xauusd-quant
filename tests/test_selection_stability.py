"""Stability selection over chronological blocks (Prompt #9, Steps 16-19, 69).

Resamples are whole quarters, reproducible from the seed; a strongly relevant
feature is selected in (nearly) every resample, a near-duplicate and pure noise
almost never; and the probe-stopping rule is what makes the frequencies
informative - a plain top-k count saturates at 1 for every candidate once the
pool is not much larger than k.
"""

from __future__ import annotations

import numpy as np
import pytest

from xauusd_quant.features.redundancy import pairwise_correlation
from xauusd_quant.selection.mrmr import mrmr
from xauusd_quant.selection.stability import (
    jaccard,
    mean_pairwise_jaccard,
    quarter_resamples,
    rank_stability,
    selection_frequency,
    until_first_probe,
    year_subsets,
)

QUARTERS = [f"{y}Q{q}" for y in range(2003, 2018) for q in range(1, 5)]


def test_quarter_resamples_are_whole_quarters_and_reproducible() -> None:
    a = quarter_resamples(QUARTERS, count=20, fraction=0.5, seed=3)
    b = quarter_resamples(QUARTERS, count=20, fraction=0.5, seed=3)
    assert a == b and len(a) == 20
    assert all(len(r) == 30 and len(set(r)) == 30 and set(r) <= set(QUARTERS) for r in a)
    assert all(r == sorted(r) for r in a)
    assert a != quarter_resamples(QUARTERS, count=20, fraction=0.5, seed=4)


def test_year_subsets_partition_the_years() -> None:
    subs = year_subsets(QUARTERS, ("odd_years", "even_years", "early", "middle", "late"))
    assert set(subs["odd_years"]) | set(subs["even_years"]) == set(QUARTERS)
    assert not set(subs["odd_years"]) & set(subs["even_years"])
    thirds = subs["early"] + subs["middle"] + subs["late"]
    assert sorted(thirds) == sorted(QUARTERS) and subs["early"][0] == "2003Q1"


def test_until_first_probe() -> None:
    probes = frozenset({"p1", "p2"})
    assert until_first_probe(["a", "b", "p1", "c"], probes, 5) == (["a", "b"], 3)
    assert until_first_probe(["p2", "a"], probes, 5) == ([], 1)
    assert until_first_probe(["a", "b", "c"], probes, 2) == (["a", "b"], None)
    assert until_first_probe([], probes, 3) == ([], None)


def test_frequency_jaccard_and_rank_stability() -> None:
    runs = [["a", "b"], ["a", "c"], ["a", "b"], ["a"]]
    freq = selection_frequency(runs, ["a", "b", "c", "d"])
    assert freq == {"a": 1.0, "b": 0.5, "c": 0.25, "d": 0.0}
    assert jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)
    assert jaccard([], []) == 1.0
    assert mean_pairwise_jaccard([["a"], ["a"]]) == 1.0 and mean_pairwise_jaccard([["a"]]) is None
    same = [np.array([3.0, 2.0, 1.0])] * 3
    assert rank_stability(same) == pytest.approx(1.0)


def _block_selections(k: int, probe_stop: bool) -> tuple[dict[str, float], list[str]]:
    """mRMR on 40 resamples of 'quarters' of synthetic rows; y = a + 0.5 b + noise."""
    rng = np.random.default_rng(11)
    q, per = 60, 400
    n = q * per
    a, b, c = rng.normal(size=(3, n))
    y = a + 0.5 * b + 0.15 * c + 3.0 * rng.normal(size=n)
    feats = {"a": a, "a_twin": a + 0.03 * rng.normal(size=n), "b": b, "c": c,
             **{f"probe_{i}": rng.normal(size=n) for i in range(6)}}
    names = list(feats)
    x = np.column_stack(list(feats.values()))
    corr = np.abs(pairwise_correlation(x[::7], rank=True)[0])
    probes = frozenset(n for n in names if n.startswith("probe_"))
    blocks = np.arange(n) // per
    runs = []
    for r in range(40):
        pick = np.isin(blocks, np.random.default_rng(r).choice(q, q // 2, replace=False))
        rel = np.abs([np.corrcoef(x[pick, j], y[pick])[0, 1] for j in range(len(names))])
        order = [names[s.index] for s in mrmr(rel, corr, list(range(len(names))), k=k + 1,
                                               hard_limit=0.95)]
        runs.append(until_first_probe(order, probes, k)[0] if probe_stop else order[:k])
    return selection_frequency(runs, names), names


def test_block_resamples_give_the_expected_selection_frequencies() -> None:
    freq, names = _block_selections(k=3, probe_stop=True)
    assert freq["a"] + freq["a_twin"] == pytest.approx(1.0)     # one of the pair, every time
    assert freq["b"] >= 0.9
    assert all(freq[n] == 0.0 for n in names if n.startswith("probe_"))
    # a plain top-k over a pool no larger than k cannot tell noise from signal
    top, _ = _block_selections(k=9, probe_stop=False)
    assert all(top[n] == 1.0 for n in names if n.startswith("probe_"))
