"""Null controls, multiple testing, test ids and feature statuses (Prompt #8, Steps 50-59, 70).

The statuses are the conjunction of documented conditions; an undefined
statistic is never evidence; a residual claim needs the random-walk control
(invariant 9); a pipeline null vetoes only what it reproduces.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from statsmodels.stats.multitest import multipletests

from xauusd_quant.alpha.config import load_alpha_config
from xauusd_quant.alpha.information_coefficient import rank_scores
from xauusd_quant.alpha.mutual_information import (
    binned_codes,
    conditional_mi,
    conditional_mi_from_codes,
    copula_mi,
    joint_counts,
    mi_from_joint,
    sentinel_codes,
)
from xauusd_quant.alpha.null_tests import (
    circular_shift_null,
    pair_counts,
    permutation_null,
    shifted_ic,
    standardized_ranks,
    synthetic_noise_features,
)
from xauusd_quant.alpha.ranking import (
    KindEvidence,
    TestRegistry,
    benjamini_hochberg,
    classify,
    passes_nulls,
)
from xauusd_quant.research.feature_reports import pipeline_verdicts


@pytest.fixture(scope="module")
def alpha():
    return load_alpha_config()


def _ev(kind: str = "volatility", ic: float = 0.1, q: float = 1e-6, shift: bool = True,
        pipe: bool | None = True, sign: float = 0.9, recent: bool | None = True, flips: int = 0,
        slow: bool | None = False) -> KindEvidence:
    return KindEvidence(kind=kind, target=f"target_{kind}_5", rank_ic=ic, q_value=q,
                        beyond_shift_null=shift, beyond_pipeline_null=pipe, sign_consistency=sign,
                        recent_same_sign=recent, era_sign_flips=flips, slow_component=slow)


def _classify(evidence, alpha, **kw):
    return classify("f", evidence, live_safe=kw.get("live_safe", True),
                    invalid_reason=kw.get("invalid"), cluster_better=kw.get("better"),
                    cfg=alpha.classification, fdr_q=alpha.multiple_testing.fdr_q)[0]


def test_statuses_are_conjunctions_of_documented_conditions(alpha) -> None:
    assert _classify([_ev(ic=0.10)], alpha) == "strong_candidate"
    assert _classify([_ev(ic=0.03)], alpha) == "candidate"
    assert _classify([_ev(ic=0.01)], alpha) == "weak_candidate"
    assert _classify([_ev(ic=0.10, flips=1)], alpha) == "candidate"
    assert _classify([_ev(sign=0.5)], alpha) == "unstable"
    assert _classify([_ev(recent=False)], alpha) == "unstable"
    assert _classify([_ev(kind="direction", slow=True)], alpha) == "unstable"
    assert _classify([_ev(kind="volatility", slow=True)], alpha) == "strong_candidate"
    assert _classify([_ev(shift=False)], alpha) == "failed_null"
    assert _classify([_ev(q=0.2)], alpha) == "failed_null"
    assert _classify([_ev(pipe=False)], alpha) == "failed_null"
    assert _classify([_ev()], alpha, better="g") == "redundant"
    assert _classify([_ev()], alpha, live_safe=False) == "non_causal"
    assert _classify([_ev()], alpha, invalid="no slow band") == "invalid"


def test_an_undefined_statistic_is_never_evidence(alpha) -> None:
    assert _classify([_ev(ic=float("nan"))], alpha) == "failed_null"
    assert _classify([_ev(q=float("nan"))], alpha) == "failed_null"
    assert not passes_nulls(_ev(q=float("nan")), 0.05)
    assert _classify([_ev(sign=float("nan"))], alpha) == "unstable"


def test_a_residual_claim_needs_the_random_walk_control(alpha) -> None:
    assert not passes_nulls(_ev(kind="residual", pipe=None), 0.05)
    assert passes_nulls(_ev(kind="residual", pipe=True), 0.05)
    assert passes_nulls(_ev(kind="volatility", pipe=None), 0.05)
    status, reason, _ = classify("f", [_ev(kind="residual", pipe=None)], live_safe=True,
                                 invalid_reason=None, cluster_better=None,
                                 cfg=alpha.classification, fdr_q=0.05)
    assert status == "failed_null" and "invariant 9" in reason


def test_a_pipeline_null_vetoes_only_what_it_reproduces(alpha) -> None:
    ev = pl.DataFrame({
        "kind": ["residual", "residual", "volatility", "direction", "volatility"],
        "value": [-0.70, 0.04, 0.30, 0.05, 0.30],
        "pipeline_random_walk_ic": [-0.72, 0.001, 0.002, 0.0, None],
        "pipeline_random_walk_se": [0.01, 0.01, 0.01, 0.01, None],
        "pipeline_shuffled_returns_ic": [-0.69, -0.002, 0.001, 0.0, None],
        "pipeline_shuffled_returns_se": [0.01, 0.01, 0.01, 0.01, None],
        "pipeline_sign_flip_ic": [-0.70, 0.0, 0.28, 0.045, None],
        "pipeline_sign_flip_se": [0.01, 0.01, 0.01, 0.01, None],
    })
    out = pipeline_verdicts(ev, alpha)
    assert out["beyond_pipeline_null"].to_list() == [False, True, True, False, None]
    # the sign-flipped null keeps volatility: it may veto direction, never volatility
    assert out["matched_by_null"][3] == "sign_flip"
    assert out["mechanical_share"][0] == pytest.approx(0.72 / 0.70)


def test_benjamini_hochberg_matches_statsmodels() -> None:
    p = np.random.default_rng(0).uniform(size=500) ** 3
    p[::50] = np.nan
    q = benjamini_hochberg(p)
    ok = np.isfinite(p)
    np.testing.assert_allclose(q[ok], multipletests(p[ok], method="fdr_bh")[1], rtol=1e-12)
    assert np.isnan(q[~ok]).all()


def test_test_ids_are_permanent_and_remember_their_first_definition(tmp_path) -> None:
    path = tmp_path / "registry.parquet"
    reg = TestRegistry(path)
    ids = reg.assign(["k1", "k2"], ["f", "f"], preregistered=True)
    assert ids == ["ALPHA-H-000001", "ALPHA-H-000002"]
    again = TestRegistry(path)
    assert again.assign(["k2", "k3"], ["f", "f"], preregistered=False) == [
        "ALPHA-H-000002", "ALPHA-H-000003"]
    assert again.preregistered(["ALPHA-H-000002", "ALPHA-H-000003"]) == [True, False]
    assert TestRegistry(path).size == 3


def test_shift_and_permutation_nulls_use_the_same_estimator() -> None:
    rng = np.random.default_rng(1)
    n = 5000
    x = rng.normal(size=(n, 2))
    y = rng.normal(size=(n, 3))
    zx = standardized_ranks([x[:, j] for j in range(2)])
    zy = standardized_ranks([y[:, j] for j in range(3)])
    s = 1234
    direct = np.array([[np.corrcoef(rank_scores(x[:, i]), np.roll(rank_scores(y[:, j]), -s))
                        [0, 1] for j in range(3)] for i in range(2)])
    np.testing.assert_allclose(shifted_ic(zx.z, zy.z, s), direct, atol=1e-5)
    shifts, used = circular_shift_null(zx.z, zy.z, draws=30, min_fraction=0.1, seed=3)
    assert shifts.shape == (30, 2, 3) and (used >= 500).all() and (used < 4500).all()
    perm = permutation_null(zx.z, zy.z, draws=10, seed=4)
    assert perm.shape == (10, 2, 3) and np.abs(perm).max() < 0.1
    np.testing.assert_array_equal(pair_counts(zx.present, zy.present), np.full((2, 3), n))


def test_noise_features_carry_no_information_by_construction() -> None:
    x = np.random.default_rng(2).normal(size=3000)
    x[::7] = np.nan
    noise = synthetic_noise_features(3000, white=2, ar1=(0.9,), per_phi=2, matched={"x": x},
                                     seed=5)
    assert set(noise) == {"noise_white_00", "noise_white_01", "noise_ar1_9_00",
                          "noise_ar1_9_01", "noise_matched_00"}
    matched = noise["noise_matched_00"][0]
    assert np.array_equal(np.isnan(matched), np.isnan(x))
    np.testing.assert_allclose(np.sort(matched[np.isfinite(matched)]),
                               np.sort(x[np.isfinite(x)]).astype(np.float32))
    ar = noise["noise_ar1_9_00"][0].astype(np.float64)
    assert np.corrcoef(ar[1:], ar[:-1])[0, 1] == pytest.approx(0.9, abs=0.03)


def test_copula_mutual_information() -> None:
    rng = np.random.default_rng(3)
    n = 200_000
    rho = 0.6
    a = rng.normal(size=n)
    b = rho * a + np.sqrt(1 - rho ** 2) * rng.normal(size=n)
    bins = 20
    ba, bb = binned_codes(rank_scores(a), bins), binned_codes(rank_scores(b), bins)
    mi, pairs = copula_mi(ba, bb, bins)
    assert pairs == n
    assert mi == pytest.approx(-0.5 * np.log(1 - rho ** 2), rel=0.1)    # 20 bins: a bit low
    indep = copula_mi(ba, binned_codes(rank_scores(rng.normal(size=n)), bins), bins)[0]
    assert abs(indep) < 2e-3
    cx = sentinel_codes(ba, bins, scale=bins + 1)
    cy = sentinel_codes(bb, bins)
    assert mi_from_joint(joint_counts(cx, cy, bins))[0] == pytest.approx(mi, abs=1e-12)
    shifted = copula_mi(ba, np.roll(bb, -777), bins)[0]
    assert mi_from_joint(joint_counts(cx, cy, bins, shift=777))[0] == pytest.approx(
        shifted, abs=1e-12)
    groups = (np.arange(n) % 4).astype(np.int64)
    groups[:10] = -1
    ref = conditional_mi(ba, bb, groups, bins)[0]
    assert conditional_mi_from_codes(cx, cy, groups, 4, bins)[0] == pytest.approx(ref, abs=1e-12)


def test_preregistration_matches_by_pattern_family_and_horizon(alpha) -> None:
    assert alpha.preregistration("log_rv_20", "realized_vol", 20) == "ALPHA-PR-01"
    assert alpha.preregistration("reg_resid_z_128", "residual_change", 5) == "ALPHA-PR-03"
    assert alpha.preregistration("log_rv_20", "residual_change", 5) is None
    assert alpha.nulls.veto_nulls("volatility") == ("random_walk", "shuffled_returns")
    assert "sign_flip" in alpha.nulls.veto_nulls("residual")
