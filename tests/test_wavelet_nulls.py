"""The wavelet layer's null controls: the Prompt #5 builders, block-size variants, roles."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from xauusd_quant.features.config import load_regression_config
from xauusd_quant.features.wavelet_causal import rolling_wavelet
from xauusd_quant.features.wavelet_config import load_wavelet_config
from xauusd_quant.models import load_ou_config
from xauusd_quant.research.spectral_nulls import (
    SourceData,
    bootstrap_block_size,
    build_control,
)
from xauusd_quant.research.wavelet_reports import source_roles
from xauusd_quant.research.wavelet_stability import band_persistence


def _real(n: int = 3000, seed: int = 0) -> SourceData:
    rng = np.random.default_rng(seed)
    vol = np.exp(np.cumsum(rng.normal(0, 0.03, n)))
    returns = rng.normal(0, 1e-3, n) * vol
    returns[0] = np.nan
    stamps = pl.Series("timestamp", np.datetime64("2020-01-01T00:00", "us")
                       + np.arange(n) * np.timedelta64(5, "m"))
    columns = {name: rng.normal(size=n) for name in ("regression_residual", "ou_innovation",
                                                      "abs_ou_innovation")}
    columns["log_return"] = returns
    return SourceData(name="real", timestamps=stamps, columns=columns,
                      missing_slots=np.zeros(n), bar_seconds=300.0)


@pytest.fixture(scope="module")
def settings():
    wavelet = load_wavelet_config(overrides={"regression_window": 32, "ou_window": 64})
    return wavelet, load_regression_config(), load_ou_config()


def test_the_null_list_includes_every_block_size_once(settings):
    wavelet, _, _ = settings
    names = wavelet.controls.enabled()
    assert names == ["white_noise", "random_walk", "shuffled_returns", "block_bootstrap",
                     "block_bootstrap_64", "block_bootstrap_1024"]
    assert bootstrap_block_size("block_bootstrap", 256) == 256
    assert bootstrap_block_size("block_bootstrap_64", 256) == 64
    assert bootstrap_block_size("random_walk", 256) is None


def test_roles_decide_which_nulls_get_which_analyses(settings):
    roles = source_roles(settings[0])
    assert roles["real"] == "real"
    assert roles["random_walk"] == roles["block_bootstrap"] == "incremental"
    assert roles["shuffled_returns"] == "core"
    assert roles["white_noise"] == roles["block_bootstrap_64"] == "shape"


def test_block_size_variants_are_reproducible_and_distinct(settings):
    wavelet, regression, ou = settings
    real = _real()
    a = build_control("block_bootstrap_64", real, regression=regression, ou=ou, spectral=wavelet)
    b = build_control("block_bootstrap_64", real, regression=regression, ou=ou, spectral=wavelet)
    c = build_control("block_bootstrap_1024", real, regression=regression, ou=ou,
                      spectral=wavelet)
    base = build_control("block_bootstrap", real, regression=regression, ou=ou,
                         spectral=wavelet)
    assert np.array_equal(a.columns["log_return"], b.columns["log_return"], equal_nan=True)
    assert not np.array_equal(a.columns["log_return"], c.columns["log_return"], equal_nan=True)
    assert not np.array_equal(a.columns["log_return"], base.columns["log_return"],
                              equal_nan=True)
    assert "ou_state_code" in a.columns and a.has_pipeline
    with pytest.raises(ValueError):
        build_control("block_bootstrap_1", real, regression=regression, ou=ou, spectral=wavelet)


def test_white_noise_has_no_stable_long_lived_dominant_scale(settings):
    wavelet = load_wavelet_config()
    x = np.random.default_rng(11).normal(size=40_000)
    result = rolling_wavelet(x, window=512, config=wavelet, bar_seconds=300.0)
    beyond = band_persistence(result, [512, 1024], tolerance=1, source="white_noise")
    assert (beyond["same_band"] < 0.3).all()
    counts = np.bincount(result.dominant[result.valid], minlength=result.layout.bands)
    assert (counts > 0).all(), "every band dominates sometimes: no preferred scale"
