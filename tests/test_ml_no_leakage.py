"""Time-series leakage is refused (Prompt #10, Steps 7-10, 38-40, Critical Rule 5).

A design column must be a registered, live-safe factory feature; outcome-like
names, dates, timestamps and row numbers are refused; a column that tracks a
target almost perfectly (a leakage canary) is refused however it is named;
derived labels never reach past the purged end of the data.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml_synth import small_config, synthetic_ml_data
from xauusd_quant.ml.datasets import LeakageError, assert_no_leakage, derive_target

REGISTRY = {"good": {"live_safe": True}, "leaky": {"live_safe": False},
            "canary": {"live_safe": True}, "weak": {"live_safe": True}}


@pytest.mark.parametrize("name", ["target_return_5", "fwd_ret_5", "future_vol", "offline_ridge",
                                  "smoothed_state", "viterbi_state", "ret_lead_1",
                                  "x_shifted_back"])
def test_outcome_like_names_are_refused(name: str) -> None:
    with pytest.raises(LeakageError, match="outcome or an offline"):
        assert_no_leakage([name], {name: {"live_safe": True}})


@pytest.mark.parametrize("name", ["timestamp", "timestamp_utc", "date", "year", "row",
                                  "row_number", "index"])
def test_dates_timestamps_and_row_numbers_are_never_predictors(name: str) -> None:
    with pytest.raises(LeakageError, match="time / row metadata"):
        assert_no_leakage([name], {name: {"live_safe": True}})


def test_unregistered_and_non_causal_features_are_refused() -> None:
    with pytest.raises(LeakageError, match="not a registered factory feature"):
        assert_no_leakage(["good", "mystery"], REGISTRY)
    with pytest.raises(LeakageError, match="not live-safe"):
        assert_no_leakage(["good", "leaky"], REGISTRY)
    assert_no_leakage(["good"], REGISTRY)


def test_a_leakage_canary_is_rejected() -> None:
    rng = np.random.default_rng(0)
    y = rng.standard_normal(20_000)
    good = rng.standard_normal(20_000)
    weak = 0.3 * y + rng.standard_normal(20_000)          # a real, modest signal passes
    canary = y + 0.01 * rng.standard_normal(20_000)       # knows its own outcome
    x = np.column_stack([good, weak, canary])
    assert_no_leakage(["good", "weak"], REGISTRY, x=x[:, :2], targets=[y])
    with pytest.raises(LeakageError, match="canary: rank correlation"):
        assert_no_leakage(["good", "weak", "canary"], REGISTRY, x=x, targets=[y])
    flipped = np.column_stack([good, -canary])
    with pytest.raises(LeakageError, match="leakage canary"):
        assert_no_leakage(["good", "canary"], REGISTRY, x=flipped, targets=[y])


def test_the_design_matrix_runs_the_gate() -> None:
    data = synthetic_ml_data()
    data.registry["target_return_5"] = {"live_safe": True}
    data.features["target_return_5"] = data.targets_raw["target_return_5"].astype(np.float32)
    with pytest.raises(LeakageError, match="target_return_5"):
        data.design(["sig", "target_return_5"])
    x = data.design(["sig", "vol"])
    assert x.shape == (data.n, 2) and x.dtype == np.float32


def test_every_manifest_is_free_of_time_and_outcome_columns() -> None:
    data = synthetic_ml_data()
    for names in data.manifests.values():
        assert_no_leakage(names, data.registry)


def test_derived_labels_never_reach_past_the_purged_end() -> None:
    data = synthetic_ml_data()
    cfg = small_config()
    for name, spec in cfg.targets.items():
        for h in spec.horizons:
            if f"target_return_{h}" not in data.targets_raw:
                continue
            ta = derive_target(spec, h, data.targets_raw, epsilon=data.epsilon,
                               sigma=data.sigma, spread_rel=data.features["spread_rel"],
                               log_floor=cfg.log_floor)
            assert not np.isfinite(ta.y[-h:]).any(), (name, h)
            assert np.isfinite(ta.y[: -h - 200]).mean() > 0.95, (name, h)
            if spec.is_classification:
                vals = np.unique(ta.y[np.isfinite(ta.y)])
                assert set(vals.tolist()) <= {0.0, 1.0}, name
    half = cfg.targets["mean_reversion_half"]
    ta = derive_target(half, 5, data.targets_raw, epsilon=data.epsilon, sigma=data.sigma,
                       spread_rel=None, log_floor=cfg.log_floor)
    full = derive_target(cfg.targets["mean_reversion"], 5, data.targets_raw,
                         epsilon=data.epsilon, sigma=data.sigma, spread_rel=None,
                         log_floor=cfg.log_floor)
    ok = np.isfinite(ta.y) & np.isfinite(full.y)
    assert ta.y[ok].mean() < full.y[ok].mean()            # halving is harder than shrinking
    assert np.all(ta.y[ok] <= full.y[ok])                 # and implies it
