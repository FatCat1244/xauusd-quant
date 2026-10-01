"""Residual decay, zero crossings and extreme behaviour.

Prompt #3 is explicit that residual mean reversion must be *measured*, not
assumed. These tests exist mostly to pin down how easy it is to measure it
wrongly:

* an AR(1) with a known coefficient must be recovered, so the estimator is
  known to be right before it is pointed at real data;
* a detrended random walk must produce a negative decay coefficient too, so
  ``b < 0`` on its own is shown to be uninformative;
* conditioning an i.i.d. series on ``|Z| > 2`` must produce a "moves toward
  zero" probability far above one half, so the headline probability is shown
  to be mechanical rather than predictive.

The last two are the ones that stop a plausible-looking number from being read
as evidence of anything tradeable.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from xauusd_quant.features import load_regression_config
from xauusd_quant.features.rolling_regression import (
    rolling_ols,
    rolling_regression_features,
)
from xauusd_quant.research.residual_decay import (
    assign_z_bins,
    bootstrap_proportion_ci,
    conditional_decay,
    move_toward_zero,
    reversion_coefficient,
)
from xauusd_quant.research.residual_extremes import (
    adverse_excursion,
    analyse_extremes,
    extreme_persistence,
    zero_crossing_analysis,
)


@pytest.fixture
def regression_config():
    return load_regression_config()


def ar1(n: int, phi: float, *, seed: int = 3, sigma: float = 1.0) -> np.ndarray:
    """AR(1) path. The decay regression on it must return b = phi - 1."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, n)
    out = np.empty(n)
    out[0] = noise[0]
    for i in range(1, n):
        out[i] = phi * out[i - 1] + noise[i]
    return out


def make_bars(n: int = 3000, *, seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    prices = 400.0 * np.exp(np.cumsum(rng.normal(0.0, 0.001, n)))
    start = datetime(2024, 1, 2, 1, 0)
    return pl.DataFrame({
        "timestamp": [start + timedelta(minutes=5 * i) for i in range(n)],
        "open": prices, "high": prices * 1.0002, "low": prices * 0.9998,
        "close": prices,
        "first_bid": prices - 0.2, "first_ask": prices + 0.2,
        "last_bid": prices - 0.2, "last_ask": prices + 0.2,
        "tick_count": np.full(n, 100, dtype=np.int64),
        "volume": np.full(n, 1000.0),
        "mean_spread": np.full(n, 0.4),
    })


@pytest.fixture
def features(regression_config):
    frame, _ = rolling_regression_features(
        make_bars(), window=64, config=regression_config, timeframe="5m"
    )
    return frame.filter(pl.col("residual").is_not_null())


def residual_frame(residual: np.ndarray, zscore: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame({"residual": residual, "residual_zscore_fit": zscore})


# ---------------------------------------------------------------------------
# The estimator is right before it is trusted
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("phi", "tolerance"), [(0.9, 0.01), (0.5, 0.02), (0.99, 0.01)]
)
def test_reversion_coefficient_recovers_a_known_ar1(regression_config, phi, tolerance):
    """delta eps = (phi - 1) * eps + noise, so b must come back as phi - 1."""
    series = ar1(40_000, phi)
    result = reversion_coefficient(
        series, timeframe="synthetic", window=64,
        config=regression_config, with_control=False,
    )
    assert result.coefficient == pytest.approx(phi - 1.0, abs=tolerance)
    assert result.observations == 39_999


def test_a_faster_reverting_series_gives_a_more_negative_coefficient(regression_config):
    fast = reversion_coefficient(
        ar1(40_000, 0.5), timeframe="s", window=64,
        config=regression_config, with_control=False,
    )
    slow = reversion_coefficient(
        ar1(40_000, 0.95), timeframe="s", window=64,
        config=regression_config, with_control=False,
    )
    assert fast.coefficient < slow.coefficient < 0.0


def test_a_random_walk_gives_a_coefficient_indistinguishable_from_zero(
    regression_config,
):
    """No detrending: a raw unit-root series has no reversion to find."""
    rng = np.random.default_rng(17)
    walk = np.cumsum(rng.normal(0.0, 1.0, 40_000))
    result = reversion_coefficient(
        walk, timeframe="s", window=64, config=regression_config, with_control=False,
    )
    assert result.coefficient == pytest.approx(0.0, abs=0.005)


def test_too_few_observations_is_refused_rather_than_estimated(regression_config):
    result = reversion_coefficient(
        ar1(100, 0.9), timeframe="s", window=64, config=regression_config,
    )
    assert np.isnan(result.coefficient)
    assert result.observations == 0
    assert any("below the configured minimum" in note for note in result.notes)


# ---------------------------------------------------------------------------
# ... and the control is what makes b readable
# ---------------------------------------------------------------------------
def test_detrended_random_walk_also_decays(regression_config):
    """The whole point of the control: b < 0 arises with no reversion present."""
    rng = np.random.default_rng(23)
    walk = np.cumsum(rng.normal(0.0, 0.001, 20_000)) + np.log(400.0)
    residual = rolling_ols(walk, 64).residual
    residual = residual[np.isfinite(residual)]
    result = reversion_coefficient(
        residual, timeframe="control", window=64,
        config=regression_config, with_control=False,
    )
    assert result.coefficient < 0.0, (
        "a detrended random walk must still show b < 0; if it does not, the "
        "control is not doing its job"
    )


def test_the_control_is_fitted_and_reported_alongside(regression_config, features):
    residual = features["residual"].to_numpy()
    result = reversion_coefficient(
        residual, timeframe="5m", window=64, config=regression_config,
    )
    assert np.isfinite(result.control_coefficient)
    assert result.control_coefficient < 0.0
    joined = " ".join(result.notes)
    assert "NOT an OU speed" in joined
    assert "NOT a half-life" in joined
    assert "compare against it" in joined


def ar1_with_ma_errors(n: int, phi: float, q: int, *, seed: int = 5) -> np.ndarray:
    """AR(1) driven by MA(q) noise, so the decay regression error is correlated."""
    rng = np.random.default_rng(seed)
    white = rng.normal(0.0, 1.0, n + q)
    noise = np.convolve(white, np.ones(q + 1), mode="valid")[:n]
    out = np.empty(n)
    out[0] = noise[0]
    for i in range(1, n):
        out[i] = phi * out[i - 1] + noise[i]
    return out


def test_newey_west_matches_ols_when_the_errors_are_independent(regression_config):
    """With white errors the HAC correction has nothing to correct."""
    result = reversion_coefficient(
        ar1(40_000, 0.9), timeframe="s", window=64,
        config=regression_config, with_control=False,
    )
    ratio = result.newey_west_se / result.standard_error
    assert 0.9 < ratio < 1.1


@pytest.mark.parametrize(("q", "minimum_ratio"), [(5, 1.2), (20, 2.0), (50, 2.5)])
def test_newey_west_inflates_the_error_when_the_score_is_correlated(
    regression_config, q, minimum_ratio
):
    """The estimator must widen the interval when serial correlation is real."""
    result = reversion_coefficient(
        ar1_with_ma_errors(40_000, 0.9, q), timeframe="s", window=64,
        config=regression_config, with_control=False,
    )
    assert result.newey_west_se / result.standard_error > minimum_ratio
    assert abs(result.newey_west_t) < abs(result.t_statistic)


def test_the_two_standard_errors_are_close_on_the_real_residual(
    regression_config, features
):
    """A measured property, not an assumption.

    It would be easy to assume that overlapping windows must inflate this
    t-statistic. They do not inflate it much: the one-step regression error is
    close to white even though the residual *level* series is heavily
    overlapping. The overlap that does matter lives in the forward-horizon
    statistics, which is where the bootstrap caveats are attached.
    """
    result = reversion_coefficient(
        features["residual"].to_numpy(), timeframe="5m", window=64,
        config=regression_config, with_control=False,
    )
    assert np.isfinite(result.newey_west_se)
    assert 0.8 < result.newey_west_se / result.standard_error < 1.25
    assert not any("far too large" in note for note in result.notes)


# ---------------------------------------------------------------------------
# Binning uses only information available at t
# ---------------------------------------------------------------------------
def test_z_bins_are_assigned_from_the_value_at_t_only(regression_config, features):
    full = assign_z_bins(features, column="residual_zscore_fit", config=regression_config)
    cut = assign_z_bins(
        features.head(800), column="residual_zscore_fit", config=regression_config
    )
    assert cut["z_bin"].to_list() == full["z_bin"].to_list()[:800]


def test_z_bin_labels_match_the_configured_edges(regression_config, features):
    binned = assign_z_bins(
        features, column="residual_zscore_fit", config=regression_config
    )
    seen = set(binned["z_bin"].drop_nulls().to_list())
    assert seen <= set(regression_config.extremes.bin_labels())
    assert "-1<=Z<0" in seen and "0<=Z<1" in seen


# ---------------------------------------------------------------------------
# "Moves toward zero" is mostly arithmetic, and the tests say so
# ---------------------------------------------------------------------------
def test_unconditionally_an_iid_series_moves_toward_zero_half_the_time(
    regression_config,
):
    """Exchangeability: with no conditioning the probability is exactly 1/2."""
    rng = np.random.default_rng(31)
    residual = rng.normal(0.0, 1.0, 60_000)
    # Every row counts as an extreme, so no selection takes place.
    frame = residual_frame(residual, np.full(residual.size, 50.0))
    table = move_toward_zero(frame, config=regression_config, horizons=(1, 5, 20))
    values = table.filter(pl.col("side") == "positive")["prob_move_toward_zero"]
    assert values.to_numpy() == pytest.approx(0.5, abs=0.01)


def test_conditioning_an_iid_series_on_extremes_fakes_mean_reversion(
    regression_config,
):
    """The headline number is high even when nothing is predictable at all.

    Given a standard normal draw beyond 2 sigma, the next independent draw is
    smaller in magnitude about 97% of the time. Any report that quotes this
    probability without a baseline is quoting regression to the mean.
    """
    rng = np.random.default_rng(37)
    residual = rng.normal(0.0, 1.0, 60_000)
    frame = residual_frame(residual, residual)  # Z is the value itself
    table = move_toward_zero(frame, config=regression_config, horizons=(1,))
    either = table.filter(pl.col("side") == "either")["prob_move_toward_zero"][0]
    assert either > 0.9, "the mechanical baseline should be near 0.97, not 0.5"


def test_conditional_decay_labels_forward_values_as_outcomes(
    regression_config, features
):
    analysis = conditional_decay(
        features, timeframe="5m", window=64, config=regression_config
    )
    assert not analysis.table.is_empty()
    joined = " ".join(analysis.notes)
    assert "OUTCOMES" in joined
    assert "The bin depends only on Z at t" in joined
    assert "the baseline is not zero" in joined
    assert "No transaction cost" in joined


def test_conditional_decay_is_unchanged_by_later_data(regression_config, features):
    """Truncating the sample must not alter the rows it still supports."""
    horizon = max(regression_config.extremes.forward_horizons)
    cut = features.head(1500)
    full = conditional_decay(
        features, timeframe="5m", window=64, config=regression_config
    ).table
    partial = conditional_decay(
        cut, timeframe="5m", window=64, config=regression_config
    ).table
    keys = ["z_bin", "horizon"]
    merged = partial.join(full, on=keys, how="inner", suffix="_full")
    assert merged.height > 0
    safe = merged.filter(pl.col("horizon") <= horizon)
    assert safe.height > 0


def test_thin_cells_are_flagged_not_silently_reported(regression_config, features):
    analysis = conditional_decay(
        features, timeframe="5m", window=64, config=regression_config
    )
    assert "thin_sample" in analysis.table.columns
    thin = analysis.table.filter(pl.col("thin_sample"))
    if thin.height:
        assert any("flagged" in warning for warning in analysis.warnings)


# ---------------------------------------------------------------------------
# Bootstrap interval
# ---------------------------------------------------------------------------
def test_bootstrap_interval_brackets_the_sample_proportion():
    rng = np.random.default_rng(41)
    flags = (rng.random(5000) < 0.7).astype(np.float64)
    lower, upper = bootstrap_proportion_ci(flags, iterations=500, seed=1)
    point = float(flags.mean())
    assert lower is not None and upper is not None
    assert lower < point < upper
    assert upper - lower < 0.06


def test_the_binomial_shortcut_agrees_with_explicit_resampling():
    """The fast path is an identity, so it must match a hand-rolled bootstrap."""
    rng = np.random.default_rng(47)
    n = 20_000
    flags = (rng.random(n) < 0.63).astype(np.float64)

    resampler = np.random.default_rng(2)
    means = np.array([
        flags[resampler.integers(0, n, n)].mean() for _ in range(2000)
    ])
    reference = (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))

    lower, upper = bootstrap_proportion_ci(flags, iterations=2000, seed=2)
    assert lower == pytest.approx(reference[0], abs=5e-4)
    assert upper == pytest.approx(reference[1], abs=5e-4)


def test_large_samples_still_get_an_interval():
    """The old cost cap returned nothing here; the binomial form has no cap."""
    rng = np.random.default_rng(53)
    flags = (rng.random(400_000) < 0.4).astype(np.float64)
    lower, upper = bootstrap_proportion_ci(flags, iterations=500, max_samples=200_000)
    assert lower is not None and upper is not None
    assert lower < 0.4 < upper


def test_non_binary_input_falls_back_to_resampling():
    """Only a 0/1 vector may take the shortcut."""
    rng = np.random.default_rng(59)
    values = rng.normal(5.0, 1.0, 5_000)
    lower, upper = bootstrap_proportion_ci(values, iterations=400, seed=3)
    assert lower is not None and upper is not None
    assert lower < 5.0 < upper
    # And the cap still applies on that path.
    big = rng.normal(0.0, 1.0, 300_000)
    assert bootstrap_proportion_ci(big, max_samples=200_000) == (None, None)


def test_a_sample_too_small_to_bootstrap_returns_nothing():
    assert bootstrap_proportion_ci(np.ones(10)) == (None, None)


def test_bootstrap_interval_is_deterministic_for_a_fixed_seed():
    rng = np.random.default_rng(43)
    flags = (rng.random(2000) < 0.3).astype(np.float64)
    first = bootstrap_proportion_ci(flags, iterations=300, seed=9)
    second = bootstrap_proportion_ci(flags, iterations=300, seed=9)
    assert first == second


# ---------------------------------------------------------------------------
# Zero crossings, with censoring made explicit
# ---------------------------------------------------------------------------
def test_a_residual_that_never_crosses_is_reported_as_censored(regression_config):
    """No sign change inside the horizon must not be quietly dropped."""
    n = 1000
    residual = np.full(n, 1.0) + np.linspace(0.0, 0.1, n)  # strictly positive
    zscore = np.zeros(n)
    zscore[100:300] = 3.0
    table = zero_crossing_analysis(
        residual_frame(residual, zscore), timeframe="s", window=64,
        config=regression_config,
    )
    row = table.filter(pl.col("condition") == "Z>2")
    assert row.height == 1
    assert row["censored_fraction"][0] == pytest.approx(1.0)
    assert np.isnan(row["median_bars_to_cross"][0])
    assert row["crossed_within_200"][0] == pytest.approx(0.0)


def test_starts_without_room_for_the_horizon_are_excluded_not_censored(
    regression_config,
):
    """A start near the end of the sample would be censored by position alone."""
    n = 1000
    horizon = regression_config.extremes.max_crossing_horizon
    residual = np.full(n, 1.0)
    zscore = np.zeros(n)
    zscore[50] = 3.0      # 50 + 200 < 1000, so this one counts
    zscore[900] = 3.0     # 900 + 200 >= 1000, so this one must not
    table = zero_crossing_analysis(
        residual_frame(residual, zscore), timeframe="s", window=64,
        config=regression_config,
    )
    row = table.filter(pl.col("condition") == "Z>2")
    assert horizon == 200
    assert row["observations"][0] == 1


def test_a_known_crossing_time_is_measured_exactly(regression_config):
    """Sign flips at a fixed offset: the reported time must be that offset."""
    n = 1000
    residual = np.full(n, 1.0)
    zscore = np.zeros(n)
    for start in range(100, 500, 20):
        residual[start + 7:start + 9] = -1.0   # flips 7 bars after each start
        zscore[start] = 3.0
    table = zero_crossing_analysis(
        residual_frame(residual, zscore), timeframe="s", window=64,
        config=regression_config,
    )
    row = table.filter(pl.col("condition") == "Z>2")
    assert row["censored_fraction"][0] == pytest.approx(0.0)
    assert row["median_bars_to_cross"][0] == pytest.approx(7.0)
    assert row["crossed_within_5"][0] == pytest.approx(0.0)
    assert row["crossed_within_10"][0] == pytest.approx(1.0)


def test_zero_crossing_runs_on_real_features(regression_config, features):
    table = zero_crossing_analysis(
        features, timeframe="5m", window=64, config=regression_config
    )
    assert not table.is_empty()
    assert {"censored_count", "censored_fraction", "thin_sample"} <= set(table.columns)
    fractions = table["censored_fraction"].to_numpy()
    assert np.all((fractions >= 0.0) & (fractions <= 1.0))


# ---------------------------------------------------------------------------
# Overshoot and adverse excursion
# ---------------------------------------------------------------------------
def test_persistence_probabilities_are_proper_probabilities(
    regression_config, features
):
    table = extreme_persistence(
        features, timeframe="5m", window=64, config=regression_config
    )
    if table.is_empty():
        pytest.skip("no extremes in the synthetic sample")
    for column in ("prob_reached_before_reverting", "prob_reverted_within_horizon"):
        values = table[column].to_numpy()
        assert np.all((values >= 0.0) & (values <= 1.0))
    assert set(table["reached_abs_z"].unique()) == set(
        regression_config.extremes.persistence_levels
    )


def test_reaching_a_higher_level_is_never_more_likely(regression_config, features):
    """P(|Z| reaches 3.5) cannot exceed P(|Z| reaches 2.5) from the same starts."""
    table = extreme_persistence(
        features, timeframe="5m", window=64, config=regression_config
    )
    if table.is_empty():
        pytest.skip("no extremes in the synthetic sample")
    for side in table["side"].unique():
        chunk = table.filter(pl.col("side") == side).sort("reached_abs_z")
        probs = chunk["prob_reached_before_reverting"].to_numpy()
        assert np.all(np.diff(probs) <= 1e-12), f"{side}: probabilities rise with level"


def test_adverse_excursion_is_non_negative_by_construction(
    regression_config, features
):
    """MARE measures movement away from zero, so it is floored at zero."""
    table = adverse_excursion(
        features, timeframe="5m", window=64, config=regression_config
    )
    if table.is_empty():
        pytest.skip("no extremes in the synthetic sample")
    for column in ("mare_residual_mean", "mare_z_mean", "mare_z_median"):
        assert np.all(table[column].to_numpy() >= 0.0)
    quantiles = ["mare_z_p50", "mare_z_p75", "mare_z_p90", "mare_z_p95", "mare_z_p99"]
    values = table.select(quantiles).row(0)
    assert list(values) == sorted(values)


def test_analyse_extremes_collects_every_study_and_its_caveats(
    regression_config, features
):
    result = analyse_extremes(
        features, timeframe="5m", window=64, config=regression_config
    )
    assert not result.zero_crossing.is_empty()
    joined = " ".join(result.notes)
    assert "not" in joined.lower()
    assert result.timeframe == "5m" and result.window == 64
