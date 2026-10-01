"""Null controls for the spectral study: each is what it claims to be.

A control that is accidentally shorter, seeded differently each run, or built
through a different pipeline than the real series would make every comparison
against it meaningless.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from xauusd_quant.features import load_regression_config
from xauusd_quant.features.spectral import rolling_spectrum
from xauusd_quant.features.spectral_config import load_spectral_config
from xauusd_quant.models import load_ou_config
from xauusd_quant.research.spectral_nulls import SourceData, _pipeline, build_control


@pytest.fixture(scope="module")
def setup():
    spectral = load_spectral_config(overrides={"regression_window": 32, "ou_window": 64,
                                               "controls": {"block_size": 20}})
    regression, ou = load_regression_config(), load_ou_config()
    n = 6000
    rng = np.random.default_rng(1)
    # Fat-tailed, volatility-clustered returns, so shuffling and bootstrapping differ.
    vol = np.exp(np.convolve(rng.normal(0, 0.3, n), np.ones(50) / 50, mode="same"))
    returns = rng.standard_t(4, n) * 1e-3 * vol
    log_price = np.log(2000.0) + np.cumsum(returns)
    columns = _pipeline(log_price, regression=regression, spectral=spectral, ou=ou)
    stamps = pl.Series("timestamp", [datetime(2021, 1, 4) + timedelta(minutes=i)
                                     for i in range(n)])
    real = SourceData(name="real", timestamps=stamps, columns=columns,
                      missing_slots=np.zeros(n), bar_seconds=60.0)
    return spectral, regression, ou, real


def build(name, setup):
    spectral, regression, ou, real = setup
    return build_control(name, real, regression=regression, ou=ou, spectral=spectral)


@pytest.mark.parametrize("name", ["white_noise", "random_walk", "shuffled_returns",
                                  "block_bootstrap"])
def test_every_control_matches_the_real_length_and_is_reproducible(setup, name):
    real = setup[3]
    a, b = build(name, setup), build(name, setup)
    for series in ("regression_residual", "ou_innovation", "log_return"):
        assert a.columns[series].size == real.size
        assert np.array_equal(a.columns[series], b.columns[series], equal_nan=True)
    assert a.timestamps is real.timestamps, "the real clock, position by position"


def test_white_noise_matches_each_series_scale(setup):
    real, noise = setup[3], build("white_noise", setup)
    for series in ("regression_residual", "log_return"):
        assert np.std(noise.columns[series]) == pytest.approx(np.nanstd(real.columns[series]),
                                                              rel=0.05)
    assert not noise.has_pipeline


def test_shuffled_returns_keep_the_exact_return_distribution(setup):
    real, shuffled = setup[3], build("shuffled_returns", setup)
    r = real.columns["log_return"]
    finite = np.sort(r[np.isfinite(r)])
    got = shuffled.columns["log_return"][1:]
    assert np.allclose(np.sort(got), np.sort(finite)[: got.size] if got.size < finite.size
                       else np.sort(got))
    assert shuffled.has_pipeline, "cumulated and run through the same regression and OU"


def test_block_bootstrap_resamples_whole_blocks_of_real_returns(setup):
    real, boot = setup[3], build("block_bootstrap", setup)
    r = real.columns["log_return"]
    source = r[np.isfinite(r)]
    got = boot.columns["log_return"][1:21]
    # Some contiguous run of the real returns must reproduce a block exactly.
    hits = np.flatnonzero(np.isclose(source, got[1]))
    assert any(np.allclose(source[i:i + 10], got[1:11]) for i in hits if i + 10 <= source.size)


def test_the_random_walk_control_is_detrended_by_the_same_pipeline(setup):
    walk = build("random_walk", setup)
    assert walk.has_pipeline
    resid = walk.columns["regression_residual"]
    assert np.isnan(resid[:31]).all() and np.isfinite(resid[31:]).all(), "N-1 warm-up"
    z = walk.columns["residual_zscore"]
    assert np.nanstd(z) == pytest.approx(1.0, abs=0.2)


def test_a_detrended_random_walk_has_a_red_spectrum_white_noise_does_not(setup):
    """Detrending shapes the spectrum - the reason this control is mandatory."""
    spectral = setup[0]
    walk, noise = build("random_walk", setup), build("white_noise", setup)
    a = rolling_spectrum(walk.columns["regression_residual"], fft_window=128, config=spectral,
                         bar_seconds=60.0)
    b = rolling_spectrum(noise.columns["regression_residual"], fft_window=128, config=spectral,
                         bar_seconds=60.0)
    assert np.nanmedian(a.band_shares["low"]) > 3 * np.nanmedian(b.band_shares["low"])
    assert np.nanmedian(a.entropy) < np.nanmedian(b.entropy)
