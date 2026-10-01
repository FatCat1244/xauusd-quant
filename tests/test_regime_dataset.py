"""The regime feature space: definitions, manifest, redundancy rule, causal transforms (Steps 2-8)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from xauusd_quant.regimes.config import load_regime_config
from xauusd_quant.regimes.dataset import (
    EXCLUDED_FEATURES,
    FEATURE_SPECS,
    _prefix_span_sum,
    _rolling_abs_acf1,
    _rolling_percentile,
    feature_manifest,
    feature_matrix,
    redundancy_table,
)
from xauusd_quant.regimes.preprocessing import clip_fractions, fit_scaler
from xauusd_quant.utils.config import ConfigError

REGIME = load_regime_config()


def test_every_configured_feature_is_registered_documented_and_causal() -> None:
    for name in REGIME.features():
        spec = FEATURE_SPECS[name]
        assert spec.causal and spec.rationale and spec.description and spec.lookback
        assert REGIME.group_of(name) == spec.group
    assert "residual_zscore" in EXCLUDED_FEATURES and "hour_of_day" in EXCLUDED_FEATURES
    assert not set(EXCLUDED_FEATURES) & set(REGIME.features())


def test_unknown_or_duplicate_features_are_config_errors() -> None:
    with pytest.raises(ConfigError, match="not a registered"):
        load_regime_config(overrides={"feature_sets": {"core": ["log_rv_20", "made_up"]}})
    with pytest.raises(ConfigError, match="duplicate"):
        load_regime_config(overrides={"feature_sets": {"core": ["log_rv_20", "log_rv_20"]}})
    with pytest.raises(ConfigError):
        load_regime_config(overrides={"causal": {"primary_scheme": "sideways"}})


def test_rolling_percentile_equals_brute_force_with_ties_and_gaps() -> None:
    rng = np.random.default_rng(0)
    x = np.round(rng.normal(size=600), 1)                  # many ties, like spreads
    x[[50, 51, 300]] = np.nan
    window = 40
    got = _rolling_percentile(x, window)
    for t in range(window - 1, x.size):
        w = x[t - window + 1:t + 1]
        valid = w[np.isfinite(w)]
        if not np.isfinite(x[t]) or valid.size < int(0.9 * window):
            assert np.isnan(got[t]), t
            continue
        less = (valid < x[t]).sum()
        equal = (valid == x[t]).sum()
        expected = (less + (equal + 1) / 2 - 0.5) / valid.size
        assert got[t] == pytest.approx(expected, abs=1e-12), t
    assert np.isnan(got[:window - 1]).all()


def test_causal_transforms_ignore_the_future() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=2000)
    future = np.concatenate([x[:1500], rng.normal(0, 100, 500)])
    for fn in (lambda v: _rolling_percentile(v, 100), lambda v: _rolling_abs_acf1(v, 50),
               lambda v: _prefix_span_sum(v, 20)):
        a, b = fn(x), fn(future)
        np.testing.assert_array_equal(a[:1500], b[:1500])


def test_the_redundancy_rule_flags_near_duplicates() -> None:
    rng = np.random.default_rng(2)
    base = rng.normal(size=5000)
    frame = pl.DataFrame({"log_rv_20": base, "rv_percentile": base + rng.normal(0, 0.01, 5000),
                          "r_squared": rng.normal(size=5000)})
    table = redundancy_table(frame, ("log_rv_20", "rv_percentile", "r_squared"),
                             sample_rows=10_000)
    regime = load_regime_config(overrides={
        "feature_sets": {"core": ["log_rv_20", "rv_percentile", "r_squared"]},
        "feature_groups": {"volatility": ["log_rv_20", "rv_percentile"],
                           "regression": ["r_squared"]}})
    manifest = feature_manifest(regime, "core", table)
    assert not manifest["passes_redundancy_rule"]
    assert manifest["duplicate_pairs"][0]["a"] == "log_rv_20"
    assert "residual_zscore" in manifest["excluded"]


def test_scalers_use_training_rows_only_and_report_clipping() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_t(3, size=(4000, 2))
    x[3000:] += 100.0
    robust = fit_scaler(x, ("a", "b"), method="robust", clip=5.0, rows=np.arange(3000))
    assert abs(robust.center[0]) < 0.1, "the shifted future does not move the centre"
    z = robust.transform(x)
    assert np.nanmax(np.abs(z)) <= 5.0
    fractions = clip_fractions(robust.transform(x, clip=False), 5.0)
    assert (fractions > 0.2).all(), "the shifted quarter is clipped and that is reported"
    standard = fit_scaler(x, ("a", "b"), method="standard", clip=None, rows=np.arange(3000))
    np.testing.assert_allclose(standard.transform(x[:3000]).mean(axis=0), 0.0, atol=1e-10)
    restored = type(robust).from_dict(robust.to_dict())
    np.testing.assert_array_equal(restored.transform(x), z)
    assert restored.fingerprint() == robust.fingerprint()


def test_feature_matrix_turns_nulls_and_infinities_into_nan() -> None:
    frame = pl.DataFrame({"a": [1.0, None, np.inf], "b": [np.nan, 2.0, 3.0]})
    x = feature_matrix(frame, ("a", "b"))
    assert np.isnan(x[1, 0]) and np.isnan(x[2, 0]) and np.isnan(x[0, 1])
    assert x[0, 0] == 1.0 and x[2, 1] == 3.0


@pytest.mark.realdata
def test_the_integrity_gate_passes_on_the_current_1h_dataset() -> None:
    from xauusd_quant import load_config
    from xauusd_quant.features import load_regression_config
    from xauusd_quant.features.spectral_config import load_spectral_config
    from xauusd_quant.features.wavelet_config import load_wavelet_config
    from xauusd_quant.models import load_ou_config
    from xauusd_quant.regimes.dataset import integrity_gate

    config = load_config()
    if not config.bars_dir("1h").exists():
        pytest.skip("no real bars on this machine")
    gate = integrity_gate(config, load_regression_config(), load_ou_config(),
                          load_spectral_config(), load_wavelet_config(), REGIME, "1h",
                          recompute_check_bars=500)
    assert gate["passed"], gate["problems"]
    assert gate["versions"]["recompute_check"]["matches"]
