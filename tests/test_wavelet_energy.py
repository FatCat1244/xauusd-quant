"""Band layout, physical periods, energy shares, groups and concentration."""

from __future__ import annotations

import numpy as np
import pytest

from xauusd_quant.features.wavelet_causal import rolling_wavelet
from xauusd_quant.features.wavelet_config import load_wavelet_config
from xauusd_quant.features.wavelet_energy import (
    active_scales,
    band_layout,
    concentration,
    energy_shares,
    group_share,
    white_noise_shares,
)


def _layout(window=512, bar_seconds=300.0, boundary=14_400.0, edges=(3_600.0, 28_800.0)):
    return band_layout(window, wavelet="db4", bar_seconds=bar_seconds, min_coefficients=16,
                       fast_slow_boundary_seconds=boundary, band_edges_seconds=edges)


def test_band_periods_follow_the_wavelet_and_the_bar_length():
    five = _layout(bar_seconds=300.0)
    hour = _layout(bar_seconds=3600.0)
    assert five.levels == 6 and five.bands == 7
    assert np.allclose(five.period_bars[:6], 2.0 ** np.arange(1, 7) / 0.7142857, rtol=1e-3)
    assert five.period_bars[-1] == pytest.approx(np.sqrt(128 * 512))    # approximation band
    assert np.allclose(hour.period_seconds, five.period_seconds * 12)   # same bars, other time


def test_groups_are_physical_so_a_timeframe_can_lack_one():
    five, hour = _layout(bar_seconds=300.0), _layout(bar_seconds=3600.0)
    assert five.groups["high"].any() and five.fast.any() and (~five.fast).any()
    assert not hour.groups["high"].any(), "no band shorter than an hour on hourly bars"
    shares = energy_shares(np.ones((hour.bands, 3)))
    assert np.isnan(group_share(shares, hour.groups["high"])).all()
    one_hour_band = [row["band"] for row in five.describe() if 2400 < row["period_seconds"] < 4800]
    assert one_hour_band, "the 5m layout has a band near one hour"


def test_shares_sum_to_one_and_undefined_where_energy_vanishes():
    energies = np.array([[1.0, 0.0], [3.0, 0.0]])
    shares = energy_shares(energies)
    assert shares[:, 0] == pytest.approx([0.25, 0.75])
    assert np.isnan(shares[:, 1]).all()


def test_concentration_and_active_scales_on_known_shares():
    shares = np.array([[0.7, 0.25], [0.2, 0.25], [0.1, 0.25], [0.0, 0.25]])
    assert concentration(shares, 1) == pytest.approx([0.7, 0.25])
    assert concentration(shares, 3) == pytest.approx([1.0, 0.75])
    assert active_scales(shares) == pytest.approx([1.0, 0.0])        # share > 1/4 only


def test_white_noise_baseline_matches_simulation():
    config = load_wavelet_config()
    rng = np.random.default_rng(0)
    result = rolling_wavelet(rng.normal(size=30_000), window=256, config=config,
                             bar_seconds=300.0)
    expected = white_noise_shares(result.layout)
    assert expected.sum() == pytest.approx(1.0)
    assert np.nanmean(result.shares, axis=1) == pytest.approx(expected, abs=0.01)


def test_a_sinusoid_puts_its_energy_in_the_band_that_holds_its_period():
    config = load_wavelet_config()
    t = np.arange(8000)
    x = np.sin(2 * np.pi * t / 22.4) + 0.05 * np.random.default_rng(1).normal(size=t.size)
    result = rolling_wavelet(x, window=512, config=config, bar_seconds=300.0)
    d4 = result.layout.band_names().index("d4")
    assert np.nanmedian(result.shares[d4]) > 0.6
    assert (result.dominant[result.valid] == d4).all()
