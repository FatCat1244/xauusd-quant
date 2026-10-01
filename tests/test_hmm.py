"""The Gaussian HMM: exact recursions, missing evidence, fitting, serialisation (Steps 13-18).

The blocked forward/backward recursions are pinned against the textbook scaled
recursions and, for short sequences, against brute-force enumeration of every
state path.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from xauusd_quant.regimes.emissions import GaussianStates, log_densities
from xauusd_quant.regimes.hmm import (
    HMMModel,
    fit_hmm,
    forward_backward,
    forward_filter,
    stationary_distribution,
    viterbi_path,
)
from xauusd_quant.regimes.registry import build_model
from xauusd_quant.regimes.synthetic import known_hmm


def _naive(log_b: np.ndarray, a: np.ndarray, pi: np.ndarray):
    n, k = log_b.shape
    m = log_b.max(axis=1)
    b = np.exp(log_b - m[:, None])
    alpha = np.zeros((n, k))
    c = np.zeros(n)
    u = pi * b[0]
    c[0] = u.sum()
    alpha[0] = u / c[0]
    for t in range(1, n):
        u = (alpha[t - 1] @ a) * b[t]
        c[t] = u.sum()
        alpha[t] = u / c[t]
    beta = np.ones((n, k))
    for t in range(n - 2, -1, -1):
        beta[t] = a @ (b[t + 1] * beta[t + 1]) / c[t + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    xi = np.zeros((k, k))
    for t in range(n - 1):
        x = alpha[t][:, None] * a * (b[t + 1] * beta[t + 1])[None, :]
        xi += x / x.sum()
    return alpha, np.log(c) + m, gamma, xi


def _random_chain(rng: np.random.Generator, k: int) -> tuple[np.ndarray, np.ndarray]:
    a = rng.dirichlet(np.ones(k) * 0.5, size=k) * 0.3 + np.eye(k) * 0.7
    return a / a.sum(axis=1, keepdims=True), rng.dirichlet(np.ones(k))


@pytest.mark.parametrize("k", [2, 3, 5])
@pytest.mark.parametrize("n", [1, 9, 257, 1500])
@pytest.mark.parametrize("block", [16, 64, 512])
def test_blocked_recursions_equal_the_textbook_ones(k: int, n: int, block: int) -> None:
    rng = np.random.default_rng(k * 1000 + n + block)
    a, pi = _random_chain(rng, k)
    log_b = rng.normal(0, 3, (n, k)) - 40
    log_b[rng.random(n) < 0.1] = 0.0                      # bars without evidence
    alpha, log_pred, gamma, xi = _naive(log_b, a, pi)
    for exact in (True, False):
        f_alpha, f_pred = forward_filter(log_b, a, pi, block_length=block, exact=exact)
        np.testing.assert_allclose(f_alpha, alpha, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(f_pred, log_pred, atol=1e-9)
    post = forward_backward(log_b, a, pi, block_length=block)
    np.testing.assert_allclose(post.gamma, gamma, atol=1e-10)
    np.testing.assert_allclose(post.xi_sum, xi, atol=1e-8 * max(n, 1))
    assert post.log_likelihood == pytest.approx(log_pred.sum(), abs=1e-7 * max(n, 1))


def test_likelihood_and_viterbi_match_brute_force_enumeration() -> None:
    rng = np.random.default_rng(3)
    k, n = 3, 6
    a, pi = _random_chain(rng, k)
    log_b = rng.normal(0, 1, (n, k))
    total = 0.0
    best, best_score = None, -np.inf
    for path in itertools.product(range(k), repeat=n):
        score = np.log(pi[path[0]]) + log_b[0, path[0]]
        for t in range(1, n):
            score += np.log(a[path[t - 1], path[t]]) + log_b[t, path[t]]
        total += np.exp(score)
        if score > best_score:
            best, best_score = path, score
    _, log_pred = forward_filter(log_b, a, pi, block_length=4)
    assert log_pred.sum() == pytest.approx(np.log(total), abs=1e-10)
    assert tuple(viterbi_path(log_b, a, pi)) == best


def test_a_bar_without_evidence_equals_marginalising_every_feature() -> None:
    rng = np.random.default_rng(4)
    states = GaussianStates(rng.normal(size=(2, 3)), np.stack([np.eye(3), 2 * np.eye(3)]))
    x = rng.normal(size=(50, 3))
    x[10] = np.nan                                         # all features missing
    x[20, 1] = np.nan                                      # one feature missing
    ld = log_densities(x, states)
    assert np.all(ld[10] == 0.0), "no evidence: the same (zero) log-density in every state"
    from scipy.stats import multivariate_normal

    for j in range(2):
        obs = [0, 2]
        expected = multivariate_normal(states.means[j, obs],
                                       states.covariances[j][np.ix_(obs, obs)]).logpdf(x[20, obs])
        assert ld[20, j] == pytest.approx(expected, abs=1e-10)


def test_stationary_distribution_and_expected_durations() -> None:
    a = np.array([[0.9, 0.1], [0.3, 0.7]])
    pi = stationary_distribution(a)
    np.testing.assert_allclose(pi @ a, pi, atol=1e-12)
    np.testing.assert_allclose(pi, [0.75, 0.25], atol=1e-12)
    model = HMMModel(start=pi, transition=a,
                     states=GaussianStates(np.zeros((2, 1)), np.ones((2, 1, 1))),
                     covariance_type="full", reg_covar=1e-4, pseudocount=1.0)
    np.testing.assert_allclose(model.expected_durations(), [10.0, 1 / 0.3])


def test_baum_welch_recovers_a_known_three_state_hmm() -> None:
    series = known_hmm(20_000, seed=11, dimension=3)
    model = fit_hmm(series.values, 3, n_init=2, max_iter=200, seed=5)
    # match fitted states to the truth by their means
    cost = ((model.states.means[:, None, :] - series.means[None, :, :]) ** 2).sum(axis=2)
    order = np.argmin(cost, axis=0)
    assert len(set(order.tolist())) == 3
    fitted = model.permuted(order)
    np.testing.assert_allclose(np.diag(fitted.transition), np.diag(series.transition), atol=0.01)
    np.testing.assert_allclose(fitted.states.means, series.means, atol=0.1)
    accuracy = np.mean(np.argmax(fitted.filter(series.values).probs, axis=1) == series.states)
    assert accuracy > 0.95


def test_serialised_model_gives_identical_probabilities() -> None:
    series = known_hmm(4_000, seed=2, dimension=3)
    model = fit_hmm(series.values, 3, n_init=1, max_iter=30, seed=1)
    restored = build_model(model.to_dict())
    assert isinstance(restored, HMMModel)
    a = model.filter(series.values).probs
    b = restored.filter(series.values).probs
    assert np.array_equal(a, b)


def test_filtered_probabilities_are_proper_distributions() -> None:
    series = known_hmm(3_000, seed=3, dimension=3)
    model = fit_hmm(series.values, 3, n_init=1, max_iter=20, seed=1)
    result = model.filter(series.values)
    np.testing.assert_allclose(result.probs.sum(axis=1), 1.0, atol=1e-12)
    assert (result.probs >= 0).all()
    nxt = model.predict_next(result.probs)
    np.testing.assert_allclose(nxt.sum(axis=1), 1.0, atol=1e-12)
