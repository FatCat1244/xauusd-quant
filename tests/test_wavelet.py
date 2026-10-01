"""Wavelet transforms: the causal MODWT, DWT reconstruction, the offline CWT and ridges."""

from __future__ import annotations

import numpy as np
import pytest
import pywt

from xauusd_quant.features.wavelet import (
    NON_CAUSAL_LABEL,
    causal_modwt,
    circular_modwt,
    cone_of_influence,
    cwt_scales,
    dwt_components,
    equivalent_filter_length,
    extract_ridges,
    max_level,
    modwt_filters,
    offline_cwt,
    scale_to_period,
    windowed_dwt_edge,
)


def test_filter_lengths_and_levels_per_window():
    f = modwt_filters("db4", 6)
    assert f.lengths == tuple(equivalent_filter_length(8, j) for j in range(1, 7))
    assert f.lengths == (8, 22, 50, 106, 218, 442)
    assert f.scaling.size == f.lengths[-1]
    assert [max_level(n, "db4", 16) for n in (128, 256, 512, 1024)] == [4, 5, 6, 7]


def test_modwt_energy_decomposes_the_variance_exactly():
    x = np.random.default_rng(1).normal(size=2048)
    w, v = circular_modwt(x, modwt_filters("db4", 6))
    assert (w ** 2).sum() + (v ** 2).sum() == pytest.approx((x ** 2).sum(), rel=1e-12)


def test_per_level_energy_matches_pywavelets_swt():
    x = np.random.default_rng(2).normal(size=1024)
    w, _ = circular_modwt(x, modwt_filters("db4", 5))
    swt = pywt.swt(x, "db4", level=5, norm=True, trim_approx=True)   # [cA5, cD5 .. cD1]
    expected = [float((d ** 2).sum()) for d in swt[1:][::-1]]
    assert [float((row ** 2).sum()) for row in w] == pytest.approx(expected, rel=1e-10)


def test_causal_modwt_equals_the_circular_one_away_from_the_start():
    x = np.random.default_rng(3).normal(size=1500)
    f = modwt_filters("db4", 5)
    causal = causal_modwt(x, f)
    circ, _ = circular_modwt(x, f)
    for j, length in enumerate(f.lengths):
        assert np.allclose(causal.detail[j, length - 1:], circ[j, length - 1:])
        assert not causal.finite_history[j, length - 2]
        assert causal.finite_history[j, length - 1]


def test_causal_coefficients_never_change_when_bars_are_appended():
    rng = np.random.default_rng(4)
    x = rng.normal(size=3000)
    f = modwt_filters("db4", 6)
    short = causal_modwt(x, f)
    long = causal_modwt(np.concatenate([x, 1e3 * rng.normal(size=500)]), f)
    assert np.array_equal(short.detail, long.detail[:, :3000])        # bit for bit
    assert np.array_equal(short.scaling, long.scaling[:3000])


def test_a_missing_value_invalidates_exactly_the_coefficients_that_read_it():
    x = np.random.default_rng(5).normal(size=600)
    x[300] = np.nan
    f = modwt_filters("db4", 3)
    c = causal_modwt(x, f)
    for j, length in enumerate(f.lengths):
        bad = np.flatnonzero(~c.finite_history[j, length - 1:]) + length - 1
        assert bad.min() == 300 and bad.max() == 300 + length - 1


def test_dwt_levels_reconstruct_the_series_and_isolate_their_scale():
    rng = np.random.default_rng(6)
    t = np.arange(1024)
    slow = np.sin(2 * np.pi * t / 128)
    fast = 0.5 * np.sin(2 * np.pi * t / 4)
    x = slow + fast + 0.05 * rng.normal(size=t.size)
    parts = dwt_components(x, "db4", level=6)
    assert set(parts) == {"a6", *[f"d{j}" for j in range(1, 7)]}
    assert np.allclose(sum(parts.values()), x, atol=1e-10)              # x = A_J + sum D_j
    energy = {k: float((v ** 2).mean()) for k, v in parts.items()}
    assert max(energy, key=energy.__getitem__) in ("d6", "a6")        # the slow cycle
    assert energy["d1"] + energy["d2"] > 0.8 * (fast ** 2).mean()      # the fast one


def test_scale_to_period_uses_the_wavelets_centre_frequency():
    scales = np.array([2.0, 8.0, 32.0])
    assert np.allclose(scale_to_period("cmor1.5-1.0", scales), scales)          # C = 1
    assert np.allclose(scale_to_period("mexh", scales), scales / 0.25)
    assert np.allclose(scale_to_period("morl", scales), scales / 0.8125)


def test_cone_of_influence_is_zero_at_the_edges_and_symmetric():
    coi = cone_of_influence(501, "cmor1.5-1.0")
    assert coi[0] == 0 and coi[-1] == 0
    assert np.allclose(coi, coi[::-1])
    assert coi[250] == coi.max()


def test_offline_cwt_of_a_known_sinusoid_peaks_at_its_period_and_is_labelled():
    t = np.arange(2048)
    x = np.sin(2 * np.pi * t / 24.0)
    scal = offline_cwt(x, cwt_scales(2, 256, 48), wavelet="cmor1.5-1.0", bar_seconds=60.0)
    assert scal.label == NON_CAUSAL_LABEL
    reliable = scal.reliable
    mean_power = np.where(reliable, scal.power, np.nan)
    peak = scal.periods_bars[np.nanargmax(np.nanmean(mean_power, axis=1))]
    assert peak == pytest.approx(24.0, rel=0.1)
    assert scal.periods_seconds[0] == pytest.approx(scal.periods_bars[0] * 60.0)


def test_ridges_follow_a_chirp_and_noise_ridges_are_short():
    rng = np.random.default_rng(7)
    n = 4096
    period = 8.0 * (64.0 / 8.0) ** (np.arange(n) / (n - 1))
    x = np.sin(np.cumsum(2 * np.pi / period)) + 0.1 * rng.normal(size=n)
    scal = offline_cwt(x, cwt_scales(2, 256, 48), wavelet="cmor1.5-1.0", bar_seconds=1.0)
    ridges = extract_ridges(scal, power_ratio=3.0)
    longest = max(ridges, key=lambda r: r.length)
    truth = period[longest.start:longest.end + 1]
    assert longest.length > n / 2
    assert np.corrcoef(np.log(longest.periods_bars), np.log(truth))[0, 1] > 0.99
    assert longest.log_period_drift > 0
    noise = offline_cwt(rng.normal(size=n), cwt_scales(2, 256, 48), wavelet="cmor1.5-1.0",
                        bar_seconds=1.0)
    noise_ridges = extract_ridges(noise, power_ratio=3.0)
    assert max(r.cycles for r in noise_ridges) < longest.cycles / 5


def test_windowed_dwt_edge_depends_on_padding_mode():
    x = np.sin(2 * np.pi * np.arange(512) / 22.4 + 0.3)
    edges = {m: windowed_dwt_edge(x, "db4", 6, m) for m in ("zero", "symmetric", "reflect",
                                                           "constant", "periodization")}
    assert all(e.shape == (6,) and np.isfinite(e).all() for e in edges.values())
    level4 = [e[3] for e in edges.values()]
    assert max(level4) > 2 * min(level4), "the newest coefficient depends on padding"
