"""No look-ahead, prefix invariance, live computability, missing data and the causal schema.

Steps 58, 59 and 61 of Prompt #6: a causal feature at bar t must be identical
whether or not later bars exist, at many cut-offs, and must be computable from
the bounded buffer a live process would hold.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from xauusd_quant.features.wavelet_causal import (
    CAUSAL_FEATURES,
    _enforce_schema,
    feature_lookback,
    rolling_wavelet,
    wavelet_feature_frame,
)
from xauusd_quant.features.wavelet_config import load_wavelet_config
from xauusd_quant.research.wavelet_analysis import boundary_study

CONFIG = load_wavelet_config()
WINDOW = 256


def _series(n: int, seed: int = 0) -> np.ndarray:
    """A residual-like series with volatility regimes and a slow wander."""
    rng = np.random.default_rng(seed)
    vol = np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    x = np.cumsum(rng.normal(size=n) * vol)
    return x - np.convolve(x, np.ones(64) / 64, mode="same")


def _stamps(n: int) -> pl.Series:
    start = np.datetime64("2020-01-01T00:00", "us")
    return pl.Series("timestamp", start + np.arange(n) * np.timedelta64(5, "m"))


def _frame(values: np.ndarray, *, window: int = WINDOW, slots: np.ndarray | None = None
           ) -> pl.DataFrame:
    result = rolling_wavelet(values, window=window, config=CONFIG, bar_seconds=300.0,
                             missing_slots=slots)
    return wavelet_feature_frame(result, _stamps(values.size), float32=False)


def test_appending_future_bars_changes_no_feature_at_or_before_t():
    x = _series(5000)
    base = _frame(x)
    future = np.concatenate([x, 1e4 * np.random.default_rng(9).standard_cauchy(2000)])
    extended = _frame(future).head(5000)
    assert base["wavelet_window_valid"].sum() > 4000, "the test must not be vacuous"
    assert base["wavelet_entropy"].n_unique() > 100
    assert base.equals(extended), "a future bar changed a historical feature"


def test_a_future_spike_does_not_leak_into_the_past():
    x = _series(3000, seed=1)
    spiked = x.copy()
    spiked[2000] += 1e6
    a, b = _frame(x), _frame(spiked)
    assert a.head(2000).equals(b.head(2000))
    assert not a.slice(2000, 10).equals(b.slice(2000, 10)), "the spike is seen from t onwards"


def test_prefix_invariance_at_random_cutoffs():
    x = _series(6000, seed=2)
    full = _frame(x)
    for cut in np.random.default_rng(3).integers(2 * WINDOW, 6000, size=8):
        prefix = _frame(x[:cut + 1])
        assert prefix.tail(1).equals(full.slice(int(cut), 1)), f"cut-off {cut}"


def test_every_feature_is_computable_from_a_live_buffer():
    x = _series(6000, seed=4)
    full = _frame(x)
    for t in (1500, 3333, 5999):
        buffer = x[t - 2 * WINDOW + 1:t + 1]                     # the longest lookback
        live = _frame(buffer).tail(1)
        row = full.slice(t, 1)
        for column in live.columns[1:]:
            a, b = live[column][0], row[column][0]
            if a is None or b is None:
                assert a is None and b is None, column
            elif isinstance(a, float):
                assert a == pytest.approx(b, rel=1e-9, abs=1e-12), column
            else:
                assert a == b, column


def test_the_schema_lists_every_stored_feature_as_causal_and_live_safe():
    frame = _frame(_series(1200, seed=5))
    for column in frame.columns:
        if column == "timestamp":
            continue
        assert column in CAUSAL_FEATURES or column.startswith("wavelet_energy_share_"), column
        if column in CAUSAL_FEATURES:
            assert CAUSAL_FEATURES[column].causal and CAUSAL_FEATURES[column].live_safe
        assert feature_lookback(column, WINDOW, WINDOW // 4) <= 2 * WINDOW


def test_offline_or_unregistered_columns_are_refused():
    with pytest.raises(ValueError, match="offline"):
        _enforce_schema({"timestamp": None, "offline_cwt_power": None})
    with pytest.raises(ValueError, match="not a registered"):
        _enforce_schema({"timestamp": None, "centred_scalogram_energy": None})
    _enforce_schema({"timestamp": None, "wavelet_entropy": None,
                     "wavelet_energy_share_d3": None})


def test_lookbacks_are_declared():
    assert feature_lookback("wavelet_entropy", 512, 128) == 512
    assert feature_lookback("wavelet_scale_drift", 512, 128) == 640
    assert feature_lookback("wavelet_energy_log_change", 512, 128) == 1024
    with pytest.raises(KeyError):
        feature_lookback("offline_cwt_power", 512, 128)


def test_missing_values_and_gaps_invalidate_windows_deterministically():
    x = _series(3000, seed=6)
    x[1500] = np.nan
    slots = np.zeros(x.size)
    slots[2500] = 0.2 * WINDOW                                   # an unscheduled gap
    a = rolling_wavelet(x, window=WINDOW, config=CONFIG, bar_seconds=300.0, missing_slots=slots)
    b = rolling_wavelet(x, window=WINDOW, config=CONFIG, bar_seconds=300.0, missing_slots=slots)
    assert np.array_equal(a.valid, b.valid)
    assert not a.valid[1500:1500 + WINDOW].any(), "every window holding the NaN is invalid"
    assert a.valid[1499] and a.valid[1500 + WINDOW]
    assert not a.valid[2500:2500 + WINDOW - 1].any(), "windows spanning the gap are invalid"
    assert np.isnan(a.entropy[~a.valid]).all()
    assert a.counts["incomplete"] == WINDOW and a.counts["too_many_missing_bars"] > 0


def test_constant_data_yields_no_scale_and_no_entropy():
    result = rolling_wavelet(np.full(2000, 1234.5), window=WINDOW, config=CONFIG,
                             bar_seconds=60.0)
    assert not result.valid.any()
    assert np.isnan(result.entropy).all()
    assert (result.dominant == -1).all()
    assert result.counts["zero_energy"] == result.counts["windows"]


def test_the_causal_default_beats_windowed_padding_at_the_right_edge():
    config = load_wavelet_config(overrides={"boundary": {"trials": 40}})
    _, summary = boundary_study(config)
    stationary = {r["method"]: r for r in summary.filter(~pl.col("step")).iter_rows(named=True)}
    causal = stationary["causal_modwt"]
    assert causal["median_padding_spread"] == 0.0, "no padding, so nothing depends on it"
    windowed = [r["median_abs_log_error"] for m, r in stationary.items()
                if m.startswith("windowed_dwt_")]
    assert causal["median_abs_log_error"] < min(windowed)
