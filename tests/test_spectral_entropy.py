"""Spectral shape: entropy, flatness, concentration, bands and centroid."""

from __future__ import annotations

import numpy as np
import pytest

from xauusd_quant.features.spectral import rolling_spectrum
from xauusd_quant.features.spectral_config import load_spectral_config
from xauusd_quant.features.spectral_entropy import (
    concentration,
    spectral_entropy,
    spectral_flatness,
)
from xauusd_quant.research.fft_analysis import sinusoid


@pytest.fixture(scope="module")
def config():
    return load_spectral_config()


def test_a_flat_spectrum_has_entropy_and_flatness_one():
    power = np.ones((1, 50))
    total = power.sum(axis=1)
    assert spectral_entropy(power, total)[0] == pytest.approx(1.0)
    assert spectral_flatness(power)[0] == pytest.approx(1.0)


def test_a_single_spike_has_entropy_and_flatness_near_zero():
    power = np.zeros((1, 50))
    power[0, 7] = 5.0
    total = power.sum(axis=1)
    assert spectral_entropy(power, total)[0] == pytest.approx(0.0, abs=1e-12)
    assert spectral_flatness(power)[0] < 1e-10


def test_concentration_sums_the_largest_shares():
    shares = np.array([[0.5, 0.2, 0.1, 0.1, 0.05]])
    got = concentration(shares)
    assert got[1][0] == pytest.approx(0.5)
    assert got[3][0] == pytest.approx(0.8)
    assert got[5][0] == pytest.approx(0.95)


def test_a_sinusoid_is_more_concentrated_than_white_noise(config):
    n = 256
    rng = np.random.default_rng(8)
    noise = rng.normal(size=5000)
    tone = sinusoid(5000, [(16 / n, 1.0, 0.0)], noise_std=0.3, seed=9)
    a = rolling_spectrum(noise, fft_window=n, config=config, bar_seconds=60.0)
    b = rolling_spectrum(tone, fft_window=n, config=config, bar_seconds=60.0)
    assert np.nanmedian(b.entropy) < np.nanmedian(a.entropy) - 0.2
    assert np.nanmedian(b.flatness) < np.nanmedian(a.flatness) / 2
    assert np.nanmedian(b.concentration[1]) > np.nanmedian(a.concentration[1]) * 5


def test_a_pure_on_bin_tone_keeps_two_thirds_of_its_power_in_one_hann_bin(config):
    """Hann spreads an on-bin tone as 1/2 : 1 : 1/2 in amplitude, 1/4 : 1 : 1/4 in power."""
    n = 128
    result = rolling_spectrum(sinusoid(n, [(12 / n, 1.0, 0.4)]), fft_window=n, config=config,
                              bar_seconds=60.0)
    assert result.concentration[1][-1] == pytest.approx(1 / (1 + 2 / 4), abs=0.01)
    assert result.concentration[3][-1] == pytest.approx(1.0, abs=0.005)
    assert result.centroid[-1] == pytest.approx(12 / n, rel=1e-3)


def test_white_noise_has_no_stable_dominant_cycle(config):
    """Disjoint windows of white noise rarely share their dominant bin."""
    n = 256
    x = np.random.default_rng(10).normal(size=60_000)
    result = rolling_spectrum(x, fft_window=n, config=config, bar_seconds=60.0)
    bins = result.dominant_bin
    ends = np.arange(n - 1, x.size - n, n)
    same = np.mean(bins[ends] == bins[ends + n])
    assert same < 0.05, f"{same:.3f} of disjoint windows share a dominant bin"
    assert np.nanmedian(result.flatness) > 0.4


def test_band_shares_partition_the_power(config):
    x = np.random.default_rng(11).normal(size=3000)
    result = rolling_spectrum(x, fft_window=128, config=config, bar_seconds=60.0)
    total = sum(result.band_shares[name] for name in ("low", "mid", "high"))
    assert np.allclose(total[result.valid], 1.0, atol=1e-5)
    # White noise spreads power in proportion to band width: 10 % / 30 % / 60 %.
    assert np.nanmedian(result.band_shares["high"]) == pytest.approx(0.6, abs=0.08)
