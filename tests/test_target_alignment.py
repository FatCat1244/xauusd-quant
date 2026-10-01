"""Targets pair feature_t with outcome_(t+h) and live apart from features (Steps 14-20, 77).

Each target family is checked against its definition, the first outcome bar is
t+1 (never t), later bars move a target only within its horizon, the OU
equilibrium is held at its value at t, weekend spans are flagged, and the
schema guards keep targets and features in separate tables.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from feature_synth import hourly_stamps, synthetic_bars
from xauusd_quant.features.manifest import (
    FeatureSchemaError,
    assert_feature_schema,
    assert_target_schema,
)
from xauusd_quant.features.registry import FeatureSpec
from xauusd_quant.targets.alignment import (
    TargetInputs,
    build_targets,
    check_alignment,
    target_kind,
)
from xauusd_quant.targets.config import load_targets_config
from xauusd_quant.targets.event_targets import future_max_down, future_max_up
from xauusd_quant.targets.residuals import (
    ou_deviation_reduction,
    residual_change,
    residual_reduction,
    residual_shrinks,
)
from xauusd_quant.targets.returns import future_abs_return, future_return
from xauusd_quant.targets.volatility import future_realized_vol


def test_future_return_pairs_t_with_t_plus_h() -> None:
    close = np.array([100.0, 101.0, 99.0, 102.0, 104.0])
    r = future_return(close, 2)
    np.testing.assert_allclose(r[:3], np.log(close[2:] / close[:3]))
    assert np.isnan(r[3:]).all()
    np.testing.assert_allclose(future_abs_return(close, 2)[:3], np.abs(r[:3]))


def test_realized_vol_starts_with_the_first_return_after_t() -> None:
    close = np.full(10, 100.0)
    close[5:] = 110.0                                       # the only move is the bar-5 return
    rv = future_realized_vol(close, 2)
    jump = abs(np.log(110.0 / 100.0))
    assert rv[5] == 0.0                                     # r_5 is known at t = 5: not an outcome
    np.testing.assert_allclose(rv[3:5], [jump, jump])       # t = 3, 4 see r_5 ahead
    assert rv[2] == 0.0 and np.isnan(rv[8:]).all()


def test_excursions_use_the_bars_after_t_only() -> None:
    close = np.full(8, 100.0)
    high = np.full(8, 100.5)
    low = np.full(8, 99.5)
    high[4], low[4] = 120.0, 80.0                           # an extreme bar at index 4
    up, down = future_max_up(close, high, 3), future_max_down(close, low, 3)
    assert up[4] == pytest.approx(np.log(100.5 / 100.0))    # bar 4 itself is not ahead of 4
    assert up[1] == pytest.approx(np.log(1.2)) and down[3] == pytest.approx(np.log(0.8))
    assert np.isnan(up[5:]).all()


def test_residual_targets_match_their_definitions() -> None:
    eps = np.array([2.0, -1.0, 0.5, -3.0, 1.0])
    sigma = np.array([1.0, 2.0, 0.5, 1.0, 1.0])
    np.testing.assert_allclose(residual_change(eps, sigma, 1)[:4],
                               (eps[1:] - eps[:-1]) / sigma[:-1])
    np.testing.assert_allclose(residual_reduction(eps, sigma, 2)[:3],
                               (np.abs(eps[:3]) - np.abs(eps[2:])) / sigma[:3])
    np.testing.assert_array_equal(residual_shrinks(eps, 1)[:4], [1.0, 1.0, 0.0, 1.0])
    assert np.isnan(residual_shrinks(eps, 1)[4])
    mu = np.array([0.5, 0.0, 0.0, 0.0, 9.0])
    np.testing.assert_allclose(ou_deviation_reduction(eps, mu, sigma, 1)[:4],
                               (np.abs(eps[:4] - mu[:4]) - np.abs(eps[1:] - mu[:4])) / sigma[:4])


def test_the_ou_equilibrium_is_held_at_its_value_at_t() -> None:
    eps = np.random.default_rng(1).normal(size=50)
    sigma = np.ones(50)
    mu = np.zeros(50)
    later = mu.copy()
    later[21:] = 5.0                                        # a later re-estimate of mu
    a = ou_deviation_reduction(eps, mu, sigma, 5)
    b = ou_deviation_reduction(eps, later, sigma, 5)
    np.testing.assert_array_equal(a[:21], b[:21])


def test_a_target_moves_only_when_a_bar_within_its_horizon_moves() -> None:
    tcfg = load_targets_config()
    bs = synthetic_bars(600)
    n = bs.size
    eps = np.sin(np.arange(n) / 7.0)
    sigma = np.full(n, 0.01)
    mu = np.zeros(n)
    base = build_targets(TargetInputs(bars=bs, residual=eps, sigma=sigma, ou_mu=mu), tcfg)
    k = 400
    changed = synthetic_bars(600)
    changed.close[k:] *= 1.5
    changed.high[k:] *= 1.5
    changed.low[k:] *= 1.5
    eps2 = eps.copy()
    eps2[k:] += 3.0
    other = build_targets(TargetInputs(bars=changed, residual=eps2, sigma=sigma, ou_mu=mu), tcfg)
    for h in tcfg.horizons:
        for fam in tcfg.families:
            col = f"target_{fam}_{h}"
            a = base[col].to_numpy()[: k - h]
            b = other[col].to_numpy()[: k - h]
            assert np.array_equal(a, b, equal_nan=True), col
            if fam in ("return", "residual_change"):
                assert not np.array_equal(base[col].to_numpy()[k - h:k],
                                          other[col].to_numpy()[k - h:k], equal_nan=True), col


def test_weekend_flags_mark_spans_that_cross_a_weekend() -> None:
    tcfg = load_targets_config()
    stamps = hourly_stamps(200)
    bs = synthetic_bars(200)
    table = build_targets(TargetInputs(bars=bs, residual=None, sigma=None, ou_mu=None), tcfg)
    friday_last = int(np.flatnonzero(stamps.dt.weekday().to_numpy() == 5)[-1])
    flag = table["target_meta_weekend_1"].to_numpy()
    assert flag[friday_last] == 1 and flag[friday_last - 1] == 0
    assert table["target_meta_weekend_5"].to_numpy()[friday_last - 4] == 1


def test_target_tables_hold_only_targets_and_features_never_hold_them() -> None:
    tcfg = load_targets_config()
    bs = synthetic_bars(300)
    table = build_targets(TargetInputs(bars=bs, residual=None, sigma=None, ou_mu=None), tcfg)
    assert_target_schema(table.columns)
    assert all(c == "timestamp" or c.startswith("target_") for c in table.columns)
    with pytest.raises(FeatureSchemaError):
        assert_target_schema(["timestamp", "log_rv_20"])
    spec = FeatureSpec(feature_id="LOG_RV_20_1H_20", name="log_rv_20", family="volatility",
                       definition="d", source_module="m", source_series="s", timeframe="1h",
                       window=20, min_history=20, live_safe=True, dtype="float32",
                       missing_policy="p", cost="cheap", incremental_update="u",
                       parameter_family="log_rv", prior_status="candidate", prior_evidence="")
    assert_feature_schema(["timestamp", "log_rv_20"], {"log_rv_20": spec})
    for leak in ("target_return_5", "fwd_return_1", "future_rv", "smoothed_p0",
                 "offline_state", "viterbi_state", "state_label", "unregistered"):
        with pytest.raises(FeatureSchemaError):
            assert_feature_schema(["timestamp", "log_rv_20", leak], {"log_rv_20": spec})


def test_a_shifted_pairing_is_refused() -> None:
    stamps = hourly_stamps(50)
    features = pl.DataFrame({"timestamp": stamps, "x": np.arange(50.0)})
    targets = pl.DataFrame({"timestamp": stamps, "target_return_1": np.arange(50.0)})
    check_alignment(features, targets)
    shifted = targets.with_columns(pl.col("timestamp").shift(-1).fill_null(
        stamps[-1] + (stamps[-1] - stamps[-2])))
    with pytest.raises(ValueError):
        check_alignment(features, shifted)
    with pytest.raises(ValueError):
        check_alignment(features, targets.head(49))


def test_target_kinds() -> None:
    assert target_kind("target_return_5") == "direction"
    assert target_kind("target_abs_return_1") == "magnitude"
    assert target_kind("target_realized_vol_20") == "volatility"
    assert target_kind("target_ou_deviation_reduction_50") == "residual"
    assert target_kind("target_max_down_3") == "excursion"
