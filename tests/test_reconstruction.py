"""Fourier reconstruction inside a window and extrapolation beyond it."""

from __future__ import annotations

import numpy as np
import pytest

from xauusd_quant.features.spectral_bands import usable_bins
from xauusd_quant.features.spectral_config import load_spectral_config
from xauusd_quant.research.fft_analysis import sinusoid
from xauusd_quant.research.spectral_reconstruction import (
    extrapolation_table,
    reconstruct,
    reconstruction_table,
    sample_window_ends,
)


@pytest.fixture(scope="module")
def config():
    return load_spectral_config()


def test_all_components_reproduce_the_window_exactly():
    x = np.random.default_rng(1).normal(size=64)
    assert np.allclose(reconstruct(x, np.arange(1, 33)), x)


def test_selected_components_reproduce_injected_ones():
    n = 128
    parts = [(5 / n, 1.0, 0.4), (17 / n, 0.5, -1.0)]
    x = 3.0 + sinusoid(n, parts)
    assert np.allclose(reconstruct(x, np.array([5, 17])), x, atol=1e-10)
    only_first = reconstruct(x, np.array([5]))
    assert np.allclose(only_first, 3.0 + sinusoid(n, parts[:1]), atol=1e-10)


def test_explained_variance_follows_parseval(config):
    n = 128
    x = sinusoid(4 * n, [(5 / n, 1.0, 0.4), (17 / n, 0.5, -1.0)])
    ends = np.array([n - 1, 2 * n - 1, 4 * n - 1])
    table = reconstruction_table(x, ends, n, usable_bins=usable_bins(n, config.spectrum),
                                 components=(1, 2), source="t")
    ev = dict(zip(table["components"].to_list(), table["explained_variance_median"].to_list(),
                  strict=True))
    assert ev[1] == pytest.approx(1.0 / 1.25, abs=1e-9)   # 1^2 / (1^2 + 0.5^2)
    assert ev[2] == pytest.approx(1.0, abs=1e-9)
    assert table.filter(table["components"] == 2)["rmse_median"][0] < 1e-9


def test_a_whole_cycle_extrapolates_exactly_and_noise_does_not(config):
    n = 128
    tone = sinusoid(6000, [(8 / n, 1.0, 0.2)])
    ends = sample_window_ends(np.arange(6000) >= n - 1, 200)
    table = extrapolation_table(tone, ends, n, usable_bins=usable_bins(n, config.spectrum),
                                components=(1,), horizons=(1, 10), source="t")
    assert (table["rmse"] < 1e-9).all()
    assert (table["skill_vs_last"] > 0.99).all()
    noise = np.random.default_rng(2).normal(size=6000)
    null = extrapolation_table(noise, ends, n, usable_bins=usable_bins(n, config.spectrum),
                               components=(3,), horizons=(5,), source="t")
    assert null["skill_vs_mean"][0] < 0.05, "periodic continuation of noise predicts nothing"


def test_window_sampling_is_even_and_bounded():
    valid = np.zeros(1000, dtype=bool)
    valid[100:] = True
    ends = sample_window_ends(valid, 50)
    assert ends.size == 50 and ends[0] == 100 and ends[-1] == 999
    assert sample_window_ends(valid, 5000).size == 900
