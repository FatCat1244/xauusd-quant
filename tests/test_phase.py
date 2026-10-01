"""Phase: circular statistics, validity, continuation and bucketing."""

from __future__ import annotations

import math

import numpy as np
import pytest

from xauusd_quant.features.spectral import rolling_spectrum
from xauusd_quant.features.spectral_config import load_spectral_config
from xauusd_quant.research.fft_analysis import sinusoid
from xauusd_quant.research.spectral_phase import (
    circular_mean,
    circular_summary,
    phase_advance_consistency,
    phase_bucket_index,
    phase_bucket_table,
    phase_projection,
    rayleigh_test,
    resultant_length,
)


@pytest.fixture(scope="module")
def config():
    return load_spectral_config()


def test_the_circular_mean_of_angles_near_pi_is_pi_not_zero():
    angles = np.array([math.pi - 0.05, -math.pi + 0.05, math.pi - 0.02, -math.pi + 0.02])
    assert abs(abs(circular_mean(angles)) - math.pi) < 1e-9
    assert np.mean(angles) == pytest.approx(0.0), "the arithmetic mean gets it exactly wrong"


def test_resultant_length_and_rayleigh():
    assert resultant_length(np.full(100, 1.3)) == pytest.approx(1.0)
    uniform = np.random.default_rng(1).uniform(-math.pi, math.pi, 20_000)
    assert resultant_length(uniform) < 0.03
    assert rayleigh_test(uniform)[1] > 0.01
    concentrated = np.random.default_rng(2).vonmises(0.5, 4.0, 500)
    assert rayleigh_test(concentrated)[1] < 1e-10
    assert circular_summary(concentrated)["circular_mean"] == pytest.approx(0.5, abs=0.1)


def test_phase_buckets_are_circular_sectors():
    b = phase_bucket_index(np.array([-math.pi, -math.pi + 1e-9, 0.0, math.pi - 1e-9, np.nan]), 8)
    assert b.tolist() == [0, 0, 4, 7, -1]


def test_a_negligible_component_has_no_phase():
    strict = load_spectral_config(overrides={"spectrum": {"phase_min_power_share": 0.5}})
    noise = np.random.default_rng(3).normal(size=2000)
    result = rolling_spectrum(noise, fft_window=128, config=strict, bar_seconds=60.0)
    assert result.valid.sum() > 0
    assert not result.dominant_phase_valid.any(), "no noise bin holds half the power"
    assert np.isnan(result.dominant_phase).all()


def test_a_steady_cycle_advances_its_phase_between_disjoint_windows(config):
    n = 128
    tone = sinusoid(20_000, [(10 / n, 1.0, 0.3)], noise_std=0.2, seed=4)
    got = phase_advance_consistency(rolling_spectrum(tone, fft_window=n, config=config,
                                                     bar_seconds=60.0), lag=n)
    assert got["same_bin_fraction"] > 0.99 and got["error_resultant_length"] > 0.95
    noise = np.random.default_rng(5).normal(size=20_000)
    null = phase_advance_consistency(rolling_spectrum(noise, fft_window=n, config=config,
                                                      bar_seconds=60.0), lag=n)
    assert null["same_bin_fraction"] < 0.1


def test_the_projected_component_tracks_a_real_cycle_but_not_noise(config):
    n = 128
    tone = sinusoid(20_000, [(10 / n, 1.0, 0.3)], noise_std=0.3, seed=6)
    result = rolling_spectrum(tone, fft_window=n, config=config, bar_seconds=60.0)
    table = phase_projection(result, tone, (1, 5))
    assert table.filter(table["horizon"] == 5)["pearson"][0] > 0.8
    noise = np.random.default_rng(7).normal(size=20_000)
    null = phase_projection(rolling_spectrum(noise, fft_window=n, config=config,
                                             bar_seconds=60.0), noise, (5,))
    assert abs(null["pearson"][0]) < 0.1 or null["pearson"][0] < 0.5


def test_phase_bucket_table_reports_means_and_an_effect_size():
    rng = np.random.default_rng(8)
    phase = rng.uniform(-math.pi, math.pi, 50_000)
    outcome = np.cos(phase) + rng.normal(0, 1.0, phase.size)
    table = phase_bucket_table(phase, {"y": outcome}, buckets=8, source="t")
    summary = table.filter(table["bucket"] == -1).row(0, named=True)
    assert summary["spread_over_std"] > 1.0 and summary["f_statistic"] > 100
    flat = phase_bucket_table(phase, {"y": rng.normal(size=phase.size)}, buckets=8, source="t")
    assert flat.filter(flat["bucket"] == -1)["spread_over_std"][0] < 0.1
