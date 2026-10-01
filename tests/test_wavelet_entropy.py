"""Wavelet (scale) entropy on known shares and on synthetic signals."""

from __future__ import annotations

import numpy as np
import pytest

from xauusd_quant.features.wavelet_causal import rolling_wavelet
from xauusd_quant.features.wavelet_config import load_wavelet_config
from xauusd_quant.features.wavelet_entropy import effective_scales, wavelet_entropy


def test_entropy_of_known_shares():
    shares = np.array([[1.0, 0.25, np.nan], [0.0, 0.25, 0.5], [0.0, 0.25, 0.5],
                       [0.0, 0.25, 0.0]])
    h = wavelet_entropy(shares)
    assert h[0] == pytest.approx(0.0)               # one scale
    assert h[1] == pytest.approx(1.0)               # uniform
    assert np.isnan(h[2]), "undefined shares give no entropy"
    assert effective_scales(shares[:, 1:2])[0] == pytest.approx(4.0)
    assert wavelet_entropy(shares[:, 1:2], normalise=False)[0] == pytest.approx(np.log(4))


def test_a_single_scale_signal_has_lower_entropy_than_broadband_noise():
    config = load_wavelet_config()
    rng = np.random.default_rng(2)
    t = np.arange(10_000)
    tone = np.sin(2 * np.pi * t / 11.2) + 0.05 * rng.normal(size=t.size)
    noise = rng.normal(size=t.size)
    h_tone = np.nanmedian(rolling_wavelet(tone, window=512, config=config,
                                          bar_seconds=60.0).entropy)
    h_noise = np.nanmedian(rolling_wavelet(noise, window=512, config=config,
                                           bar_seconds=60.0).entropy)
    assert h_tone < h_noise - 0.3
    assert 0.0 <= h_tone <= 1.0 and 0.0 <= h_noise <= 1.0
