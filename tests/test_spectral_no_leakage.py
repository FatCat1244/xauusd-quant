"""No spectral feature at bar t may change when bars after t are appended.

Each window's features depend on that window's values alone, so equality is
exact, not approximate - across history lengths and chunk sizes.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from xauusd_quant.features.spectral import rolling_spectrum, spectral_feature_frame
from xauusd_quant.features.spectral_config import load_spectral_config

ARRAYS = ("valid", "missing_fraction", "total_power", "entropy", "flatness", "centroid",
          "top_bins", "top_amplitude", "top_shares", "dominant_phase", "dominant_phase_valid")


@pytest.fixture(scope="module")
def config():
    return load_spectral_config()


def same(a, b):
    return np.array_equal(a, b, equal_nan=np.issubdtype(np.asarray(a).dtype, np.floating))


@pytest.mark.parametrize("fft_window", [64, 256])
def test_appending_future_bars_changes_nothing_before_them(config, fft_window):
    rng = np.random.default_rng(1)
    full = np.cumsum(rng.normal(size=4000)) * 0.01 + rng.normal(size=4000)
    slots = np.zeros(4000)
    slots[rng.choice(4000, 30, replace=False)] = rng.integers(1, 40, 30)
    cut = 2500
    short = rolling_spectrum(full[:cut], fft_window=fft_window, config=config, bar_seconds=60.0,
                             missing_slots=slots[:cut])
    long = rolling_spectrum(full, fft_window=fft_window, config=config, bar_seconds=60.0,
                            missing_slots=slots)
    for name in ARRAYS:
        assert same(getattr(short, name), getattr(long, name)[:cut]), name
    for band in short.band_shares:
        assert same(short.band_shares[band], long.band_shares[band][:cut]), band
    for m in short.concentration:
        assert same(short.concentration[m], long.concentration[m][:cut]), m


def test_a_future_spike_moves_nothing_before_it(config):
    x = np.random.default_rng(2).normal(size=3000)
    spiked = x.copy()
    spiked[2000] = 1e6
    a = rolling_spectrum(x, fft_window=128, config=config, bar_seconds=60.0)
    b = rolling_spectrum(spiked, fft_window=128, config=config, bar_seconds=60.0)
    for name in ARRAYS:
        assert same(getattr(a, name)[:2000], getattr(b, name)[:2000]), name
    assert not same(a.entropy[2000:2128], b.entropy[2000:2128])


def test_chunk_size_never_changes_a_value():
    x = np.random.default_rng(3).normal(size=5000)
    small = load_spectral_config(overrides={"rolling_fft": {"chunk_elements": 2048}})
    large = load_spectral_config(overrides={"rolling_fft": {"chunk_elements": 8_000_000}})
    a = rolling_spectrum(x, fft_window=128, config=small, bar_seconds=60.0)
    b = rolling_spectrum(x, fft_window=128, config=large, bar_seconds=60.0)
    for name in ARRAYS:
        assert same(getattr(a, name), getattr(b, name)), name


def test_the_stored_feature_rows_are_unchanged_by_later_data(config):
    x = np.random.default_rng(4).normal(size=3000)
    stamps = pl.Series("timestamp", np.arange(3000))
    a = spectral_feature_frame(rolling_spectrum(x[:2000], fft_window=256, config=config,
                                                bar_seconds=60.0), stamps[:2000], config)
    b = spectral_feature_frame(rolling_spectrum(x, fft_window=256, config=config,
                                                bar_seconds=60.0), stamps, config)
    assert a.equals(b.head(2000))
