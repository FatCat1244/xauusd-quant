"""The rolling FFT engine and its conventions, checked on known signals.

Positive controls are mandatory: if a sinusoid of known frequency, amplitude
and phase does not come back where it was put, nothing measured on market
data means anything.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from xauusd_quant.features.spectral import (
    rolling_spectrum,
    spectral_feature_frame,
    window_function,
    window_validity,
)
from xauusd_quant.features.spectral_bands import usable_bins
from xauusd_quant.features.spectral_config import load_spectral_config
from xauusd_quant.research.fft_analysis import (
    aliased_frequency,
    benchmark_rolling_fft,
    power_spectral_density,
    recover_components,
    resolution_table,
    sinusoid,
    window_spectrum,
)


@pytest.fixture(scope="module")
def config():
    return load_spectral_config()


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def test_hann_matches_its_definition_and_is_symmetric():
    n = 64
    w = window_function("hann", n)
    k = np.arange(n)
    assert np.allclose(w, 0.5 * (1 - np.cos(2 * np.pi * k / (n - 1))))
    assert w[0] == pytest.approx(0.0) and w[-1] == pytest.approx(0.0)
    assert np.allclose(w, w[::-1])
    assert np.all(window_function("rectangular", n) == 1.0)


def test_dc_is_never_a_component_and_nyquist_only_when_configured(config):
    bins = usable_bins(64, config.spectrum)
    assert bins[0] == 1 and bins[-1] == 31, "DC and the Nyquist bin 32 are excluded"
    keep = load_spectral_config(overrides={"spectrum": {"exclude_nyquist_if_needed": False}})
    assert usable_bins(64, keep.spectrum)[-1] == 32
    assert usable_bins(63, config.spectrum)[-1] == 31, "an odd window has no Nyquist bin"


def test_a_single_sinusoid_is_recovered_by_the_rolling_engine(config):
    """Frequency, amplitude and phase at the window's last bar."""
    n, freq, amp, phase0 = 256, 16 / 256, 2.0, 0.9
    x = sinusoid(3000, [(freq, amp, phase0)], noise_std=0.2, seed=1)
    result = rolling_spectrum(x, fft_window=n, config=config, bar_seconds=300.0)
    t = 2999
    assert result.valid[t]
    assert result.dominant_bin[t] == 16
    assert result.dominant_period_bars[t] == pytest.approx(16.0)
    assert result.dominant_period_seconds[t] == pytest.approx(16 * 300.0)
    assert result.top_amplitude[t, 0] == pytest.approx(amp, rel=0.05)
    true_phase = wrap(2 * math.pi * freq * t + phase0)
    assert abs(wrap(result.dominant_phase[t] - true_phase)) < 0.05


def test_two_components_are_found_and_ranked_by_power(config):
    n = 256
    x = sinusoid(n, [(10 / n, 1.0, 0.2), (40 / n, 0.5, 1.4)], noise_std=0.05, seed=2)
    got = recover_components(x, config, 2)
    assert got["bin"].to_list() == [10, 40]
    assert got["amplitude"].to_list() == pytest.approx([1.0, 0.5], rel=0.08)


def test_leakage_beside_a_strong_peak_is_not_a_second_component(config):
    """Hann leaks half the amplitude into each neighbour; that is still one cycle."""
    n = 64
    x = sinusoid(n, [(8 / n, 1.0, 0.0)])
    result = rolling_spectrum(x, fft_window=n, config=config, bar_seconds=60.0)
    bins = result.top_bins[-1]
    assert bins[0] == 8
    assert 7 not in bins[1:] and 9 not in bins[1:]


def test_close_frequencies_need_a_long_enough_window(config):
    table = resolution_table(config)
    for row in table.iter_rows(named=True):
        if row["separation_in_bins"] >= 2.5:
            assert row["resolved"], row
        if row["separation_in_bins"] <= 1.0:
            assert not row["resolved"], row


def test_aliasing_folds_fast_oscillations_below_nyquist(config):
    assert aliased_frequency(0.8) == pytest.approx(0.2)
    assert aliased_frequency(1.25) == pytest.approx(0.25)
    # A 0.8 cycle-per-bar cosine sampled once per bar looks like 0.2.
    n = 200
    spec = window_spectrum(sinusoid(n, [(0.8, 1.0, 0.0)]), config)
    peak = spec.sort("power", descending=True).row(0, named=True)
    assert peak["frequency_cycles_per_bar"] == pytest.approx(0.2)


def test_psd_of_white_noise_integrates_to_its_variance():
    rng = np.random.default_rng(3)
    sigma = 1.7
    x = rng.normal(0.0, sigma, 2 ** 16)
    psd = power_spectral_density(x)
    freq = psd["frequency_cycles_per_bar"].to_numpy()
    area = np.trapezoid(psd["psd_per_bar"].to_numpy(), freq)
    assert area == pytest.approx(sigma ** 2, rel=0.05)
    per_second = power_spectral_density(x, bar_seconds=300.0)
    assert "psd_per_second" in per_second.columns


def test_unscheduled_gaps_invalidate_windows_but_the_policy_is_a_share(config):
    n = 64
    x = np.random.default_rng(4).normal(size=300)
    slots = np.zeros(300)
    slots[150] = 2          # 2 missing slots between bars 149 and 150: 2/66 = 3 %
    slots[200] = 10         # 10 missing slots: 10/74 = 13.5 %
    _, fraction, valid = window_validity(x, n, missing_slots=slots, tolerance=0.05)
    only_150 = np.arange(150 - n + 1, 200 - n + 1)     # hold bar 150 but not bar 200
    assert np.allclose(fraction[only_150], 2 / (n + 2))
    assert valid[only_150].all(), "3 % is within a 5 % tolerance"
    both = np.arange(200 - n + 1, 150)
    assert np.allclose(fraction[both], 12 / (n + 12))
    only_200 = np.arange(150, 200)
    assert np.allclose(fraction[only_200], 10 / (n + 10))
    assert not valid[np.arange(200 - n + 1, 200)].any(), "13.5 % is not"
    assert fraction[150] == pytest.approx(10 / (n + 10)), (
        "the gap before a window's first bar lies outside it")


def test_constant_input_yields_no_components_and_no_phase(config):
    result = rolling_spectrum(np.full(400, 3.25), fft_window=64, config=config, bar_seconds=60.0)
    assert not result.valid.any()
    assert np.isnan(result.dominant_phase).all()
    assert (result.top_bins == -1).all()
    assert result.counts["zero_power"] == 400 - 64 + 1


def test_warm_up_bars_are_missing_not_backfilled(config):
    x = np.random.default_rng(5).normal(size=500)
    result = rolling_spectrum(x, fft_window=128, config=config, bar_seconds=60.0)
    assert not result.valid[:127].any() and result.valid[127:].all()
    assert np.isnan(result.entropy[:127]).all()
    frame = spectral_feature_frame(result, pl.Series("t", np.arange(500)), config)
    assert frame["spectral_entropy"][:127].null_count() == 127
    assert frame["spectral_entropy"][127:].null_count() == 0


def test_the_feature_frame_keeps_every_unit_explicit(config):
    x = sinusoid(600, [(8 / 128, 1.0, 0.0)], noise_std=0.1, seed=6)
    result = rolling_spectrum(x, fft_window=128, config=config, bar_seconds=900.0)
    frame = spectral_feature_frame(result, pl.Series("t", np.arange(600)), config)
    last = frame.row(-1, named=True)
    assert last["fft_period_bars_1"] == pytest.approx(16.0)
    assert last["fft_period_seconds_1"] == pytest.approx(16.0 * 900.0)
    assert last["fft_freq_1"] == pytest.approx(1 / 16)
    assert last["fft_phase1_sin"] ** 2 + last["fft_phase1_cos"] ** 2 == pytest.approx(1.0,
                                                                                         abs=1e-5)
    assert last["low_power_share"] + last["mid_power_share"] + last["high_power_share"] == (
        pytest.approx(1.0, abs=1e-5))
    assert frame.schema["spectral_entropy"] == pl.Float32


def test_the_benchmark_reports_every_stage(config):
    table = benchmark_rolling_fft(config, windows=500)
    assert set(table["fft_window"].to_list()) == set(config.fft_windows)
    assert (table["windows_per_second"] > 0).all()
