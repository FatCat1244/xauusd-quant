"""PCA and the diagnostic linear models are fitted on training rows only (Prompt #9,
Steps 20-28, 69).

Appending future observations - absurd ones - must not move a historically
fitted PCA or any projection of a historical row; fitting on the longer sample
does move it (so the test can tell). The Lasso / elastic-net paths computed from
sufficient statistics recover a known support, moments add over blocks, and the
FISTA L1 logistic path enters the true features first.
"""

from __future__ import annotations

import numpy as np
import pytest

from xauusd_quant.selection.linear import (
    evaluate,
    fit_cdfs,
    lasso_path,
    logistic_l1_path,
    moments_of,
    ridge,
)
from xauusd_quant.selection.pca import fit_pca, pca_ridge


def _design(n: int = 6000, p: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, 3))
    x = base @ rng.normal(size=(3, p)) + 0.5 * rng.normal(size=(n, p))
    y = (x[:, 0] - 0.5 * x[:, 3] + rng.normal(size=n))[:, None]
    return x, y


def test_appending_future_rows_does_not_alter_a_fitted_pca() -> None:
    x, y = _design()
    t = 4000
    cols = list(range(x.shape[1]))
    model = fit_pca(moments_of(x[:t], y[:t]), cols)
    wild = np.random.default_rng(9).standard_cauchy(size=(2000, x.shape[1])) * 1e3
    x2 = np.vstack([x[:t], wild])
    y2 = np.vstack([y[:t], np.random.default_rng(1).normal(size=(2000, 1)) * 1e3])
    again = fit_pca(moments_of(x2[:t], y2[:t]), cols)
    for field in ("mean", "sd", "eigenvalues", "loadings"):
        np.testing.assert_array_equal(getattr(model, field), getattr(again, field))
    np.testing.assert_array_equal(model.transform(x[:t], 3), again.transform(x2[:t], 3))
    # a row's projection uses that row only
    np.testing.assert_array_equal(model.transform(x2, 3)[:t], model.transform(x[:t], 3))
    leaky = fit_pca(moments_of(x2, y2), cols)
    assert not np.allclose(leaky.loadings[:, 0], model.loadings[:, 0])


def test_training_cdfs_are_frozen() -> None:
    x, _ = _design()
    cdfs = fit_cdfs(x, [0, 1], 0, 3000)
    before = cdfs.matrix(x[:3000], [0, 1])
    x[3000:] = 1e9                                          # the future changes
    np.testing.assert_array_equal(fit_cdfs(x, [0, 1], 0, 3000).matrix(x[:3000], [0, 1]), before)
    assert np.all(np.abs(before) <= 0.5)


def test_moments_add_over_blocks_and_ridge_matches_least_squares() -> None:
    x, y = _design()
    whole = moments_of(x, y)
    parts = moments_of(x[:2500], y[:2500]) + moments_of(x[2500:], y[2500:])
    np.testing.assert_allclose(parts.sxx, whole.sxx, rtol=1e-12)
    np.testing.assert_allclose(parts.sxy, whole.sxy, rtol=1e-12)
    q, c, mx, sd, my = whole.standardized()
    beta = ridge(q, c, 0.0)
    z = (x - mx) / sd
    ls, *_ = np.linalg.lstsq(z, y - my, rcond=None)
    np.testing.assert_allclose(beta, ls, rtol=1e-6, atol=1e-8)
    m = pca_ridge(fit_pca(whole, list(range(x.shape[1]))), whole, 3, 0.01)
    assert m.shape == (3, 1)


def test_lasso_and_elastic_net_paths_recover_the_support() -> None:
    rng = np.random.default_rng(2)
    n, p = 20_000, 12
    x = rng.normal(size=(n, p))
    x[:, 5] = 0.6 * x[:, 0] + 0.8 * x[:, 5]                 # a correlated distractor
    y = (1.0 * x[:, 0] - 0.6 * x[:, 3] + 0.3 * x[:, 7] + rng.normal(size=n))[:, None]
    q, c, *_ = moments_of(x, y).standardized()
    for l2 in (0.0, 0.1):
        path = lasso_path(q, c[:, 0], l2=l2, max_features=3)
        assert path["order"] == [0, 3, 7], l2
        assert path["coefs"].shape[1] == p
    empty = lasso_path(q[:0, :0], c[:0, 0])
    assert empty["order"] == []


def test_logistic_l1_path_enters_the_true_features_first() -> None:
    rng = np.random.default_rng(4)
    n = 8000
    x = rng.normal(size=(n, 6))
    prob = 1.0 / (1.0 + np.exp(-(1.2 * x[:, 1] - 0.8 * x[:, 4])))
    yb = (rng.random(n) < prob).astype(float)
    path = logistic_l1_path(x, yb, (0.2, 0.1, 0.05, 0.01))
    entry = path["entry"]
    assert np.isfinite(entry[1]) and np.isfinite(entry[4])
    assert entry[1] <= entry[4] < min(entry[j] for j in (0, 2, 3, 5))
    assert np.sign(path["coefs"][-1, 1]) == 1 and np.sign(path["coefs"][-1, 4]) == -1


def test_evaluate_scores_rank_ic_and_r2() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=5000)
    good = evaluate(y + 0.1 * rng.normal(size=5000), y, y, 0.0)
    assert good["rank_ic"] > 0.95 and good["r2"] > 0.95
    short = evaluate(y[:50], y[:50], y[:50], 0.0)
    assert short["rank_ic"] is None and short["n"] == 50


@pytest.mark.parametrize("k", [1, 3])
def test_pca_components_are_orthonormal_with_a_fixed_sign(k: int) -> None:
    x, y = _design()
    model = fit_pca(moments_of(x, y), list(range(x.shape[1])))
    v = model.loadings[:, :k]
    np.testing.assert_allclose(v.T @ v, np.eye(k), atol=1e-10)
    top = np.argmax(np.abs(v), axis=0)
    assert np.all(v[top, np.arange(k)] > 0)
    assert np.all(np.diff(model.eigenvalues) <= 1e-12)
