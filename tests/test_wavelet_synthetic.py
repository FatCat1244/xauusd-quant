"""Positive controls: a localised cycle, a chirp and two scales at once (Steps 37-39)."""

from __future__ import annotations

import polars as pl
import pytest

from xauusd_quant.features.spectral_config import load_spectral_config
from xauusd_quant.features.wavelet_config import load_wavelet_config
from xauusd_quant.research.wavelet_analysis import (
    chirp_control,
    localized_oscillation_control,
    multiscale_control,
)


@pytest.fixture(scope="module")
def configs():
    return load_wavelet_config(), load_spectral_config()


def test_the_wavelet_localises_a_temporary_cycle_the_fft_smears(configs):
    wavelet, spectral = configs
    table = {r["method"]: r for r in localized_oscillation_control(wavelet, spectral)
             .iter_rows(named=True)}
    causal = table["causal_modwt_local_energy"]
    fft = next(r for m, r in table.items() if m.startswith("rolling_fft"))
    offline = table["offline_cwt_power"]
    # onset seen later by every causal method, but the wavelet by its filter delay only
    assert 0 <= causal["start_error"] < 150
    assert fft["start_error"] > 2 * causal["start_error"]
    assert abs(offline["start_error"]) < 60 and offline["label"].startswith("NON-CAUSAL")
    assert abs(causal["width_error"]) < 150


def test_the_dominant_scale_follows_a_chirp(configs):
    wavelet, spectral = configs
    table = {r["method"]: r for r in chirp_control(wavelet, spectral).iter_rows(named=True)}
    assert table["offline_cwt_ridge"]["log_period_correlation"] > 0.99
    causal = table["causal_modwt_local_centroid"]
    assert causal["log_period_correlation"] > 0.95
    assert causal["slope_vs_true"] > 0.7, "the causal estimate rises with the true period"


def test_a_slow_cycle_and_a_fast_burst_land_in_different_bands(configs):
    wavelet, _ = configs
    checks = {r["check"]: r["value"] for r in multiscale_control(wavelet).iter_rows(named=True)}
    assert checks["slow bands' median window share"] > 0.6
    assert checks["fast local energy, burst / outside"] > 5
    assert checks["offline CWT fast power, burst / outside"] > 5
    assert checks["offline CWT slow power, burst / outside"] == pytest.approx(1.0, rel=0.25)
    assert isinstance(multiscale_control(wavelet), pl.DataFrame)
