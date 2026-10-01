r"""Selection stability under chronological resampling (Prompt #9, Steps 16-19).

Rows are never shuffled. A resample is a set of whole calendar **quarters**
of the development period: half of them drawn without replacement
(``resamples`` times), plus the named year subsets - odd years, even years and
the early / middle / late thirds of the development years. The selector runs on
each subset; then

.. math:: \mathrm{SelectionFrequency}_i = \frac{\#\{\text{runs selecting } i\}}{\#\text{runs}},

the **Jaccard** similarity of two selected sets
:math:`J(S_a,S_b) = |S_a\cap S_b| / |S_a\cup S_b|` (reported as the mean over
all pairs of runs), and the **rank stability**: the mean pairwise Spearman
correlation of the relevance rankings of all features across runs.

Noise probes run through the same selector, and a run's selection **stops at
the first probe** (the random-probe rule of Stoppiglia, Dreyfus, Dubois &
Oussar 2003; Bi et al. 2003): the features that entered before any pure noise
series did, at most ``k``. A plain top-``k`` count cannot be calibrated this
way - when a target's candidate pool is not much larger than ``k``, every
candidate, probe or not, is selected in every run and the frequencies carry no
information. A feature is called stable when its probe-stopped frequency
reaches the configured threshold; the top-``k`` frequencies are kept beside it
as a diagnostic.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from ..alpha.information_coefficient import rank_scores

__all__ = [
    "jaccard",
    "mean_pairwise_jaccard",
    "quarter_resamples",
    "rank_stability",
    "selection_frequency",
    "until_first_probe",
    "year_subsets",
]


def until_first_probe(order: list[str], probes: set[str] | frozenset[str], k: int
                      ) -> tuple[list[str], int | None]:
    """The features ranked before the first probe (at most *k*), and that probe's position.

    The position is 1-based within *order*; ``None`` when no probe appears in it
    (the selector ran out of candidates or was stopped before reaching one).
    """
    out: list[str] = []
    for pos, name in enumerate(order, start=1):
        if name in probes:
            return out[:k], pos
        out.append(name)
    return out[:k], None


def quarter_resamples(quarters: list[str], *, count: int, fraction: float, seed: int
                      ) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    size = max(1, int(round(fraction * len(quarters))))
    return [sorted(rng.choice(quarters, size=size, replace=False).tolist()) for _ in range(count)]


def year_subsets(quarters: list[str], names: tuple[str, ...]) -> dict[str, list[str]]:
    years = sorted({int(q[:4]) for q in quarters})
    thirds = np.array_split(np.array(years), 3)
    groups = {"odd_years": [y for y in years if y % 2 == 1],
              "even_years": [y for y in years if y % 2 == 0],
              "early": thirds[0].tolist(), "middle": thirds[1].tolist(),
              "late": thirds[2].tolist()}
    return {name: [q for q in quarters if int(q[:4]) in set(groups[name])]
            for name in names if name in groups}


def selection_frequency(runs: list[list[str]], names: list[str]) -> dict[str, float]:
    if not runs:
        return dict.fromkeys(names, 0.0)
    counts = dict.fromkeys(names, 0)
    for run in runs:
        for n in set(run):
            if n in counts:
                counts[n] += 1
    return {n: counts[n] / len(runs) for n in names}


def jaccard(a: set[str] | list[str], b: set[str] | list[str]) -> float:
    sa, sb = set(a), set(b)
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0


def mean_pairwise_jaccard(runs: list[list[str]]) -> float | None:
    pairs = list(combinations(range(len(runs)), 2))
    if not pairs:
        return None
    return float(np.mean([jaccard(runs[i], runs[j]) for i, j in pairs]))


def rank_stability(scores: list[np.ndarray]) -> float | None:
    """Mean pairwise Spearman correlation of feature scores across runs."""
    ranks = [rank_scores(np.nan_to_num(np.asarray(s, dtype=np.float64), nan=-1.0))
             for s in scores]
    pairs = list(combinations(range(len(ranks)), 2))
    if not pairs:
        return None
    vals = []
    for i, j in pairs:
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.corrcoef(ranks[i], ranks[j])[0, 1]
        if np.isfinite(r):
            vals.append(r)
    return float(np.mean(vals)) if vals else None
