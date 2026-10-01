"""Label switching (Step 23): renumbered models align back; label-invariant evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from xauusd_quant.regimes.diagnostics import (
    adjusted_rand,
    contingency,
    cramers_v,
    degenerate_flags,
    eta_squared,
    normalized_mutual_info,
)
from xauusd_quant.regimes.emissions import GaussianStates, bhattacharyya
from xauusd_quant.regimes.hmm import HMMModel
from xauusd_quant.regimes.state_alignment import align_states, canonical_order, cost_matrix


def _model(seed: int = 0, k: int = 4, d: int = 3) -> HMMModel:
    rng = np.random.default_rng(seed)
    means = rng.normal(0, 3, (k, d))
    cov = np.stack([np.eye(d) * (0.5 + j) for j in range(k)])
    a = rng.dirichlet(np.ones(k), size=k) * 0.1 + np.eye(k) * 0.9
    return HMMModel(start=np.full(k, 1 / k), transition=a / a.sum(axis=1, keepdims=True),
                    states=GaussianStates(means, cov), covariance_type="full", reg_covar=1e-4,
                    pseudocount=1.0)


@pytest.mark.parametrize("seed", range(5))
def test_a_permuted_model_aligns_back_to_the_reference(seed: int) -> None:
    ref = _model(seed)
    perm = np.random.default_rng(seed + 100).permutation(4)
    shuffled = ref.permuted(perm)
    cost = cost_matrix(ref.states.means, ref.states.covariances, shuffled.states.means,
                       shuffled.states.covariances)
    alignment = align_states(cost)
    aligned = shuffled.permuted(alignment.order)
    np.testing.assert_array_equal(aligned.states.means, ref.states.means)
    np.testing.assert_array_equal(aligned.transition, ref.transition)
    assert alignment.total_cost == pytest.approx(0.0, abs=1e-12)


def test_alignment_survives_small_perturbations_and_reports_ambiguity() -> None:
    ref = _model(1)
    rng = np.random.default_rng(9)
    noisy = HMMModel(start=ref.start, transition=ref.transition,
                     states=GaussianStates(ref.states.means + rng.normal(0, 0.05, (4, 3)),
                                           ref.states.covariances * 1.05),
                     covariance_type="full", reg_covar=1e-4, pseudocount=1.0)
    perm = np.array([2, 0, 3, 1])
    alignment = align_states(cost_matrix(ref.states.means, ref.states.covariances,
                                         noisy.permuted(perm).states.means,
                                         noisy.permuted(perm).states.covariances))
    np.testing.assert_array_equal(perm[alignment.order], np.arange(4))
    assert (alignment.ambiguity < 0.1).all(), "a clear match has a small cost ratio"


def test_bhattacharyya_is_invariant_to_a_common_affine_map() -> None:
    rng = np.random.default_rng(2)
    m1, m2 = rng.normal(size=3), rng.normal(size=3)
    c1 = np.cov(rng.normal(size=(20, 3)).T) + 0.1 * np.eye(3)
    c2 = np.cov(rng.normal(size=(20, 3)).T) + 0.1 * np.eye(3)
    scale, shift = np.array([2.0, 0.5, 10.0]), np.array([1.0, -3.0, 7.0])
    s = np.diag(scale)
    before = bhattacharyya(m1, c1, m2, c2)
    after = bhattacharyya(m1 * scale + shift, s @ c1 @ s, m2 * scale + shift, s @ c2 @ s)
    assert after == pytest.approx(before, rel=1e-10)
    assert bhattacharyya(m1, c1, m1, c1) == pytest.approx(0.0, abs=1e-12)


def test_canonical_order_sorts_by_the_reference_feature() -> None:
    means = np.array([[3.0, 0.0], [-1.0, 5.0], [1.0, -2.0]])
    np.testing.assert_array_equal(canonical_order(means, 0), [1, 2, 0])


def test_agreement_measures_ignore_renumbering() -> None:
    rng = np.random.default_rng(3)
    a = rng.integers(0, 3, 5000)
    b = np.where(rng.random(5000) < 0.8, a, rng.integers(0, 3, 5000))
    relabelled = np.array([2, 0, 1])[b]
    for fn in (adjusted_rand, normalized_mutual_info):
        assert fn(a, b) == pytest.approx(fn(a, relabelled), abs=1e-12)
    assert cramers_v(contingency(a, b)) == pytest.approx(cramers_v(contingency(a, relabelled)))
    assert adjusted_rand(a, a) == pytest.approx(1.0)
    assert abs(adjusted_rand(a, rng.integers(0, 3, 5000))) < 0.01


def test_eta_squared_is_one_for_perfect_separation_and_near_zero_by_chance() -> None:
    labels = np.repeat([0, 1], 500)
    assert eta_squared(labels.astype(float) * 10, labels) == pytest.approx(1.0)
    noise = np.random.default_rng(4).normal(size=1000)
    assert eta_squared(noise, np.random.default_rng(5).integers(0, 2, 1000)) < 0.01


def test_degenerate_models_are_flagged() -> None:
    cov = np.stack([np.eye(2), np.eye(2)])
    flags = degenerate_flags(np.array([0.995, 0.005]), covariances=cov,
                             separation=np.array([[0.0, 0.01], [0.01, 0.0]]),
                             transition=np.array([[0.2, 0.8], [0.3, 0.7]]),
                             min_state_fraction=0.01, max_state_fraction=0.9,
                             max_condition_number=1e8, duplicate_bhattacharyya=0.05,
                             min_expected_duration=2.0, max_expected_duration=1e5)
    assert flags["degenerate"]
    for name in ("tiny_state", "dominant_state", "duplicate_states", "flickering_state"):
        assert flags[name], name
    collapsed = degenerate_flags(np.array([0.5, 0.5]),
                                 covariances=np.stack([np.eye(2), np.diag([1.0, 1e-12])]),
                                 separation=None, transition=None, min_state_fraction=0.01,
                                 max_state_fraction=0.9, max_condition_number=1e8,
                                 duplicate_bhattacharyya=0.05, min_expected_duration=2.0,
                                 max_expected_duration=1e5)
    assert collapsed["covariance_collapse"]
