"""Strict causality of the live-safe regime features (Steps 15, 70-73).

At a historical cut-off ``t`` the filtered probabilities, most likely state,
confidence, entropy, next-state probabilities and regime age must be identical
whether or not later bars exist - bit for bit, at many cut-offs, through the
whole walk-forward procedure. Smoothed probabilities are shown to change when
future data is appended, and the stored-feature schema refuses them.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from xauusd_quant.regimes.causal_inference import (
    enforce_live_safe,
    refit_schedule,
    regime_feature_frame,
    run_walk_forward,
)
from xauusd_quant.regimes.config import load_regime_config
from xauusd_quant.regimes.hmm import fit_hmm
from xauusd_quant.regimes.preprocessing import MissingPolicy, fit_scaler
from xauusd_quant.regimes.registry import ModelRegistry, load_model
from xauusd_quant.regimes.synthetic import known_hmm
from xauusd_quant.regimes.transitions import regime_age

FEATURES = ("log_rv_20", "return_z_20", "r_squared")


def _config(**causal: object):
    return load_regime_config(overrides={
        "feature_sets": {"core": list(FEATURES)},
        "feature_groups": {"volatility": ["log_rv_20"], "returns": ["return_z_20"],
                           "regression": ["r_squared"]},
        "causal": {"first_inference": "2020-01-01", "min_training_observations": 500,
                   "burn_in_bars": 300, "refit_max_iter": 20, "fresh_init_every": 3,
                   **causal},
        "models": {"hmm": {"states": [3], "n_init": 1, "max_iter": 40, "block_length": 64},
                   "gmm": {"components": [3], "n_init": 1, "max_iter": 60},
                   "kmeans": {"clusters": [3], "n_init": 1}},
        "preprocessing": {"required_features": ["log_rv_20"]},
    })


def _frame(n: int, seed: int = 0) -> pl.DataFrame:
    series = known_hmm(n, seed=seed, dimension=3)
    start = datetime(2019, 7, 1)                  # hourly: inference from 2020-01 to ~mid-2020
    stamps = [start + timedelta(hours=i) for i in range(n)]
    x = series.values.copy()
    x[np.random.default_rng(seed).random(n) < 0.02, 1] = np.nan     # a few missing values
    return pl.DataFrame({"timestamp": pl.Series(stamps, dtype=pl.Datetime("us")),
                         **{f: x[:, j] for j, f in enumerate(FEATURES)}})


def test_filtered_probabilities_ignore_future_bars_bit_for_bit() -> None:
    series = known_hmm(6000, seed=1, dimension=3)
    model = fit_hmm(series.values[:3000], 3, n_init=1, max_iter=30, seed=1, block_length=64)
    full = model.filter(series.values)
    rng = np.random.default_rng(2)
    for cut in [1, 63, 64, 65, 2999, 3001, 5999]:
        prefix = model.filter(series.values[:cut])
        assert np.array_equal(prefix.probs, full.probs[:cut]), cut
        assert np.array_equal(prefix.log_predictive, full.log_predictive[:cut]), cut
    wild = np.vstack([series.values[:4000], rng.standard_cauchy((2000, 3)) * 1e3])
    assert np.array_equal(model.filter(wild).probs[:4000], full.probs[:4000])


def test_smoothed_probabilities_do_change_with_future_bars() -> None:
    series = known_hmm(4000, seed=3, dimension=3)
    model = fit_hmm(series.values, 3, n_init=1, max_iter=30, seed=1)
    smoothed_short = model.smooth(series.values[:2000])
    smoothed_long = model.smooth(series.values)
    assert not np.allclose(smoothed_short[-50:], smoothed_long[1950:2000], atol=1e-6), \
        "smoothing uses later bars - that is why it is never live-safe"
    with pytest.raises(ValueError, match="smoothed"):
        enforce_live_safe(["timestamp", "hmm_smoothed_p0"])
    with pytest.raises(ValueError, match="smoothed|offline|forward"):
        enforce_live_safe(["timestamp", "viterbi_state"])
    with pytest.raises(ValueError, match="not a registered"):
        enforce_live_safe(["timestamp", "hmm_something_else"])
    enforce_live_safe(["timestamp", "hmm_p0", "hmm_state", "next_state_p2", "regime_age_bars",
                       "leave_probability", "regime_model_id"])


@pytest.mark.parametrize("family", ["hmm", "gmm", "kmeans"])
def test_walk_forward_outputs_are_prefix_invariant(family: str) -> None:
    regime = _config()
    frame = _frame(9000)
    full = run_walk_forward(frame, regime, timeframe="1h", family=family, states=3,
                            scheme="expanding", refit="quarterly")
    assert full.frame.height > 3000
    cutoffs = [int(v) for v in (full.frame["timestamp"].len() * np.array([0.3, 0.7]))]
    for cut_row in cutoffs:
        cut_time = full.frame["timestamp"][cut_row]
        prefix_frame = frame.filter(pl.col("timestamp") <= cut_time)
        extended = pl.concat([prefix_frame, frame.filter(pl.col("timestamp") > cut_time).with_columns(
            (pl.col("log_rv_20") * 50.0).alias("log_rv_20"))])        # a wild, different future
        short = run_walk_forward(prefix_frame, regime, timeframe="1h", family=family, states=3,
                                 scheme="expanding", refit="quarterly")
        long = run_walk_forward(extended, regime, timeframe="1h", family=family, states=3,
                                scheme="expanding", refit="quarterly")
        n = short.frame.height
        columns = [c for c in short.frame.columns if c != "regime_model_id"]
        assert short.frame.select(columns).equals(long.frame.head(n).select(columns)), cut_row
        assert short.frame["regime_model_id"].equals(long.frame["regime_model_id"].head(n))


def test_the_scaler_and_every_refit_see_only_the_past() -> None:
    periods = refit_schedule(datetime(2003, 5, 5), datetime(2026, 9, 18),
                             first_inference=datetime(2008, 1, 1), frequency="quarterly",
                             scheme="rolling", rolling_years=5)
    for p in periods:
        assert p.train_end == p.infer_start
        assert p.train_start < p.train_end
        assert p.train_start >= datetime(2003, 5, 5)
        assert (p.train_end - p.train_start).days <= 5 * 366 + 1
    starts = [p.infer_start for p in periods]
    assert starts == sorted(starts) and starts[0] == datetime(2008, 1, 1)
    assert all(a.infer_end == b.infer_start for a, b in zip(periods, periods[1:], strict=False))
    x = np.random.default_rng(1).normal(size=(1000, 2))
    scaler = fit_scaler(x, ("a", "b"), method="robust", clip=None, rows=np.arange(500))
    x_future = x.copy()
    x_future[500:] *= 1e6
    again = fit_scaler(x_future, ("a", "b"), method="robust", clip=None, rows=np.arange(500))
    np.testing.assert_array_equal(scaler.center, again.center)
    np.testing.assert_array_equal(scaler.scale, again.scale)


def test_regime_age_depends_only_on_past_labels() -> None:
    labels = np.array([0, 0, 1, 1, 1, -1, 2, 2, 0])
    np.testing.assert_array_equal(regime_age(labels), [1, 2, 1, 2, 3, -1, 1, 2, 1])
    for cut in range(1, labels.size + 1):
        np.testing.assert_array_equal(regime_age(labels[:cut]), regime_age(labels)[:cut])


def test_missing_feature_policy_is_deterministic() -> None:
    z = np.array([[0.0, 1.0, 2.0], [np.nan, 1.0, 2.0], [np.nan, np.nan, 2.0],
                  [1.0, np.nan, np.nan]])
    scaler = fit_scaler(np.random.default_rng(0).normal(size=(100, 3)), ("a", "b", "c"),
                        method="robust", clip=None)
    marginal = MissingPolicy("marginalize", 0.6, ("c",))
    values, ok = marginal.prepare(z, ("a", "b", "c"), scaler)
    np.testing.assert_array_equal(ok, [True, True, False, False])
    assert np.isnan(values[1, 0]), "marginalised, never filled"
    exclude = MissingPolicy("exclude", 0.6, ("c",))
    np.testing.assert_array_equal(exclude.prepare(z, ("a", "b", "c"), scaler)[1],
                                  [True, False, False, False])
    fill = MissingPolicy("median_fill", 0.6, ("c",))
    values, ok = fill.prepare(z, ("a", "b", "c"), scaler)
    assert np.isfinite(values[1]).all() and ok[1]
    again, _ = fill.prepare(z, ("a", "b", "c"), scaler)
    assert np.array_equal(values, again, equal_nan=True)


def test_registered_models_are_immutable_and_restore_exactly(tmp_path) -> None:
    regime = _config()
    frame = _frame(6000, seed=4)
    registry = ModelRegistry(tmp_path / "models")
    result = run_walk_forward(frame, regime, timeframe="1h", family="hmm", states=3,
                              scheme="expanding", refit="quarterly", registry=registry)
    assert result.model_ids and all(m.startswith("HMM_1H_K3_") for m in result.model_ids)
    assert len(set(result.model_ids)) == len(result.model_ids)
    again = run_walk_forward(frame, regime, timeframe="1h", family="hmm", states=3,
                             scheme="expanding", refit="quarterly", registry=registry)
    assert again.model_ids == result.model_ids, "identical content keeps its ID"
    model, scaler, metadata = load_model(registry, result.model_ids[-1])
    assert metadata["causal"] and metadata["live_safe"]
    assert scaler is not None and metadata["train_end"] < metadata["infer_start"]
    table = regime_feature_frame(result)
    assert "log_score" not in table.columns and "complete" not in table.columns
