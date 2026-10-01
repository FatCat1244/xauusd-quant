"""Gaussian mixtures and K-Means (Steps 10-12): recovery, covariance types, criteria."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from xauusd_quant.regimes.clustering import fit_kmeans, silhouette_estimate
from xauusd_quant.regimes.emissions import GaussianStates, log_densities, parameter_count
from xauusd_quant.regimes.gmm import GMMModel, fit_gmm, state_entropy
from xauusd_quant.regimes.registry import build_model
from xauusd_quant.research.regime_analysis import single_gaussian


def _two_blobs(n: int = 6000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < 0.3).astype(int)
    means = np.array([[0.0, 0.0, 0.0], [4.0, -3.0, 2.0]])
    x = means[labels] + rng.normal(size=(n, 3)) * np.where(labels[:, None] == 1, 0.5, 1.0)
    return x, labels


def test_log_densities_equal_scipy_including_the_padded_chunks() -> None:
    rng = np.random.default_rng(1)
    k, d = 3, 4
    means = rng.normal(size=(k, d))
    cov = np.stack([np.cov(rng.normal(size=(50, d)).T) + 0.1 * np.eye(d) for _ in range(k)])
    x = rng.normal(size=(40_000, d))
    ld = log_densities(x, GaussianStates(means, cov))
    for j in range(k):
        expected = multivariate_normal(means[j], cov[j]).logpdf(x)
        np.testing.assert_allclose(ld[:, j], expected, rtol=1e-10, atol=1e-10)


def test_em_recovers_two_well_separated_components() -> None:
    x, labels = _two_blobs()
    model = fit_gmm(x, 2, n_init=2, seed=3)
    probs, _ = model.posterior(x)
    predicted = np.argmax(probs, axis=1)
    accuracy = max(np.mean(predicted == labels), np.mean(predicted != labels))
    assert accuracy > 0.99
    assert sorted(np.round(model.weights, 1).tolist()) == [0.3, 0.7]


@pytest.mark.parametrize("covariance_type", ["full", "diag", "tied"])
def test_covariance_types_are_respected(covariance_type: str) -> None:
    x, _ = _two_blobs(seed=2)
    model = fit_gmm(x, 2, covariance_type=covariance_type, n_init=1, seed=1)
    cov = model.states.covariances
    if covariance_type == "diag":
        off = cov - np.einsum("kii->ki", cov)[:, :, None] * np.eye(3)[None]
        assert np.all(off == 0.0)
    if covariance_type == "tied":
        assert np.array_equal(cov[0], cov[1])
    assert model.n_parameters == 1 + parameter_count(2, 3, covariance_type)


def test_information_criteria_prefer_the_true_number_of_components() -> None:
    x, _ = _two_blobs(seed=4)
    one = single_gaussian(x, 1e-4)
    one_bic = -2 * one.log_likelihood + (3 + 6) * np.log(x.shape[0])
    two = fit_gmm(x, 2, n_init=2, seed=1)
    three = fit_gmm(x, 3, n_init=2, seed=1)
    assert two.bic() < one_bic
    assert two.bic() < three.bic()


def test_posterior_rows_sum_to_one_and_entropy_is_bounded() -> None:
    x, _ = _two_blobs(seed=5)
    model = fit_gmm(x, 2, n_init=1, seed=1)
    x[5, :] = np.nan                                     # an unscored bar
    probs, marginal = model.posterior(x)
    assert np.isnan(probs[5]).all() and np.isnan(marginal[5])
    ok = ~np.isnan(probs).any(axis=1)
    np.testing.assert_allclose(probs[ok].sum(axis=1), 1.0, atol=1e-12)
    h = state_entropy(probs)
    assert np.isnan(h[5])
    assert (h[ok] >= -1e-12).all() and (h[ok] <= np.log(2) + 1e-12).all()


def test_gmm_round_trip_through_the_registry_format() -> None:
    x, _ = _two_blobs(seed=6)
    model = fit_gmm(x, 2, n_init=1, seed=1)
    restored = build_model(model.to_dict())
    assert isinstance(restored, GMMModel)
    assert np.array_equal(model.posterior(x)[0], restored.posterior(x)[0])


def test_kmeans_finds_the_blobs_and_reports_a_high_silhouette() -> None:
    x, labels = _two_blobs(seed=7)
    model = fit_kmeans(x, 2, n_init=3, seed=1)
    predicted, _ = model.predict(x)
    accuracy = max(np.mean(predicted == labels), np.mean(predicted != labels))
    assert accuracy > 0.99
    assert silhouette_estimate(x, predicted, sample=2000) > 0.5
    one_blob = np.random.default_rng(1).normal(size=(4000, 3))
    random_split = (one_blob[:, 0] > 0).astype(int)
    assert silhouette_estimate(one_blob, random_split, sample=2000) < 0.4


def test_kmeans_assigns_partially_observed_rows_on_their_observed_features() -> None:
    x, _ = _two_blobs(seed=8)
    model = fit_kmeans(x, 2, n_init=1, seed=1)
    row = x[:1].copy()
    full_label, _ = model.predict(row)
    row[0, 2] = np.nan
    partial_label, _ = model.predict(row)
    assert partial_label[0] == full_label[0]
