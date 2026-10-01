"""Synthetic controls (Steps 44-48): known regimes are found, absent regimes are not invented.

A known 3-state HMM must come back as three states with the right transition
behaviour; one stationary distribution must not yield persistent, well
separated "regimes"; a smoothly drifting volatility must be recognisable as a
continuum (states that only step to their neighbours); and destroying the
time order must destroy HMM persistence.
"""

from __future__ import annotations

import numpy as np
import pytest

from xauusd_quant.regimes.changepoints import pelt_mean
from xauusd_quant.regimes.config import load_regime_config
from xauusd_quant.regimes.gmm import fit_gmm
from xauusd_quant.regimes.hmm import fit_hmm
from xauusd_quant.regimes.synthetic import known_hmm, single_regime, smooth_continuum
from xauusd_quant.research.regime_nulls import (
    adjacency_share,
    block_bootstrap_rows,
    persistence_row,
    shuffle_rows,
)
from xauusd_quant.research.regime_reports import increment_verdict
from xauusd_quant.research.regime_stability import defensible_states, k_rule_verdict

REGIME = load_regime_config()


def _holdout_ll(model, x: np.ndarray, cut: int) -> float:
    return float(model.filter(x).log_predictive[cut:].mean())


def test_the_known_three_state_hmm_is_recovered_and_three_is_chosen() -> None:
    series = known_hmm(24_000, seed=21, dimension=3)
    cut = 16_000
    scores = {}
    for k in (2, 3, 4):
        model = fit_hmm(series.values[:cut], k, n_init=2, max_iter=150, seed=k)
        scores[k] = _holdout_ll(model, series.values, cut)
    assert scores[3] - scores[2] > 0.05, "a real third state is worth a lot of likelihood"
    assert scores[4] - scores[3] < 0.01, "a fourth state adds almost nothing out of sample"


def test_one_gaussian_regime_gives_no_persistent_distinct_states() -> None:
    series = single_regime(20_000, seed=22, dimension=3)
    hmm = fit_hmm(series.values, 2, n_init=2, max_iter=100, seed=1)
    gmm = fit_gmm(series.values, 2, n_init=2, seed=1)
    row = persistence_row(hmm, gmm, series.values, vol_index=0, regime=REGIME)
    assert abs(row["persistence_excess"]) < 0.05, "no temporal structure to find"
    assert row["temporal_gain_per_obs"] < 0.002
    assert row["min_bhattacharyya"] < 0.5, "the two 'states' overlap heavily"


def test_heavy_tails_make_mixtures_prefer_more_components_without_any_regime() -> None:
    series = single_regime(20_000, seed=23, dimension=3, heavy_tails=True)
    one = fit_gmm(series.values, 1, n_init=1, seed=1)
    two = fit_gmm(series.values, 2, n_init=2, seed=1)
    assert two.bic() < one.bic(), "BIC rewards modelling the tails - it is not regime evidence"
    hmm = fit_hmm(series.values, 2, n_init=1, max_iter=100, seed=1)
    row = persistence_row(hmm, two, series.values, vol_index=0, regime=REGIME)
    assert abs(row["persistence_excess"]) < 0.05, "...and the components do not persist"


def test_a_smooth_continuum_is_cut_into_neighbouring_bands() -> None:
    series = smooth_continuum(30_000, seed=24, dimension=3, persistence=0.999)
    hmm = fit_hmm(series.values, 4, n_init=1, max_iter=100, seed=1)
    order = np.argsort(hmm.states.means[:, 0])
    share = adjacency_share(hmm.transition, order)
    assert share is not None and share > 0.9, "a continuum only steps to adjacent bands"
    assert np.median(hmm.expected_durations()) > 20, "and its bands look persistent"


def test_shuffling_rows_destroys_persistence_and_block_bootstrap_keeps_it_locally() -> None:
    series = known_hmm(16_000, seed=25, dimension=3)
    real = fit_hmm(series.values, 3, n_init=1, max_iter=100, seed=1)
    shuffled = fit_hmm(shuffle_rows(series.values, 1), 3, n_init=1, max_iter=100, seed=1)
    assert np.median(real.expected_durations()) > 20
    assert np.median(shuffled.expected_durations()) < 3
    blocks = fit_hmm(block_bootstrap_rows(series.values, 400, 2), 3, n_init=1, max_iter=100,
                     seed=1)
    assert np.median(blocks.expected_durations()) > 10, "blocks longer than the regimes keep them"


def test_the_k_rule_stops_at_the_first_failing_step() -> None:
    candidates = [{"states": 1, "test_ll_per_obs": -10.0},
                  {"states": 2, "test_ll_per_obs": -9.5, "degenerate": False, "overlap_ari": 0.9},
                  {"states": 3, "test_ll_per_obs": -9.2, "degenerate": False, "overlap_ari": 0.8},
                  {"states": 4, "test_ll_per_obs": -9.199, "degenerate": False, "overlap_ari": 0.8},
                  {"states": 5, "test_ll_per_obs": -8.0, "degenerate": False, "overlap_ari": 0.9}]
    verdict = defensible_states(candidates, margin=0.005, min_overlap_ari=0.5)
    assert verdict["defensible_states"] == 3
    candidates[1]["degenerate"] = True
    assert defensible_states(candidates, margin=0.005, min_overlap_ari=0.5)[
        "defensible_states"] == 1


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_the_k_rule_reads_an_unavailable_statistic_as_untested(missing: float | None) -> None:
    # A model run offline only (the 5m GMM, every diagonal GMM) has no walk-forward refit
    # overlap: that step cannot fail, it is untested - and NaN must not pass either.
    candidates = [{"states": 1, "test_ll_per_obs": -10.0},
                  {"states": 2, "test_ll_per_obs": -9.5, "degenerate": False,
                   "overlap_ari": missing},
                  {"states": 3, "test_ll_per_obs": -9.2, "degenerate": False, "overlap_ari": 0.8}]
    rule = defensible_states(candidates, margin=0.005, min_overlap_ari=0.5)
    assert rule["defensible_states"] == 1 and not rule["complete"]
    assert k_rule_verdict(rule) == "untested"
    candidates[1]["overlap_ari"] = 0.9
    candidates[2]["test_ll_per_obs"] = missing
    rule = defensible_states(candidates, margin=0.005, min_overlap_ari=0.5)
    assert rule["defensible_states"] == 2 and not rule["complete"]
    assert k_rule_verdict(rule) == "discrete_states_defensible", "K >= 2 already established"
    candidates[1]["test_ll_per_obs"] = -10.0
    rule = defensible_states(candidates, margin=0.005, min_overlap_ari=0.5)
    assert rule["complete"] and k_rule_verdict(rule) == "no_discrete_states_defensible"


@pytest.mark.parametrize("missing", [None, float("nan")])
def test_an_undefined_increment_is_untested_never_a_verdict(missing: float | None) -> None:
    nulls = {"random_walk": 0.001, "block_bootstrap_1024": 0.002}

    def verdict(soft: float | None, rnd: float | None, folds: int,
                null_values: dict[str, float | None]) -> str:
        return increment_verdict(soft, rnd, folds, null_values, no_increment="none",
                                 beyond="beyond")[0]

    assert verdict(missing, 0.0, 4, nulls) == "untested"
    assert verdict(0.01, 0.0, 4, nulls) == "beyond"
    assert verdict(0.0015, 0.0, 4, nulls) == "within_pipeline_nulls"
    assert verdict(0.01, 0.0, 2, nulls) == "none", "fewer than 3 of 4 folds"
    assert verdict(0.01, missing, 4, nulls) == "beyond", "no random draws: compared with 0"
    assert verdict(0.01, 0.0, 4, {"random_walk": missing,
                                  "block_bootstrap_1024": missing}) == \
        "untested_against_pipeline_nulls"
    assert verdict(0.0015, 0.0, 4, {"random_walk": missing,
                                    "block_bootstrap_1024": 0.002}) == "within_pipeline_nulls"


@pytest.mark.parametrize("shift", [1.0, 3.0])
def test_pelt_finds_mean_shifts_at_the_right_places(shift: float) -> None:
    rng = np.random.default_rng(int(shift))
    y = np.concatenate([rng.normal(0, 1, 300), rng.normal(shift, 1, 400),
                        rng.normal(0, 1, 300)])
    points, _ = pelt_mean(y, min_size=20)
    assert len(points) == 2
    assert abs(points[0] - 300) <= 25 and abs(points[1] - 700) <= 25
    flat, _ = pelt_mean(rng.normal(0, 1, 1000), min_size=20)
    assert flat == []
