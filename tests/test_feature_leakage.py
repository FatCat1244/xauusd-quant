"""No feature of the factory may use a bar after its own (Prompt #8, Steps 5, 77, 78).

Every registered family is computed on a synthetic path and on prefixes of
it cut at random bars, and on the prefix followed by absurd future bars
(Cauchy jumps, huge spreads and tick counts). Every value before the cut must
be identical, bit for bit (NaN where NaN). Registered interactions and the
causal scaling / winsorising transformations are held to the same rule, and a
full-sample z-score - the classic normalisation leak - is shown to fail it,
so the test can tell.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest

from feature_synth import synthetic_bars, with_wild_future
from xauusd_quant.features import load_regression_config
from xauusd_quant.features.factory import compute_families
from xauusd_quant.features.factory_config import load_features_config
from xauusd_quant.features.interactions import causal_zscore, interaction_columns
from xauusd_quant.features.joins import EngineProvider
from xauusd_quant.features.registry import build_registry
from xauusd_quant.features.spectral_config import load_spectral_config
from xauusd_quant.features.transformations import scaling_variants, winsorize_causal
from xauusd_quant.features.wavelet_config import load_wavelet_config
from xauusd_quant.models import load_ou_config
from xauusd_quant.research.config import load_research_config

N = 3600
FAMILIES = ("returns", "volatility", "autocorrelation", "regression", "ou", "fft", "wavelet",
            "regime", "microstructure", "time")


@cache
def _configs() -> tuple:
    return (load_features_config(), load_regression_config(), load_ou_config(),
            load_spectral_config(), load_wavelet_config(), load_research_config())


def _compute(bs) -> dict[str, np.ndarray]:
    fcfg, regression, ou, spectral, wavelet, research = _configs()
    provider = EngineProvider(bars=bs, regression_config=regression, spectral_config=spectral,
                              wavelet_config=wavelet)
    values, _ = compute_families(bs, provider, fcfg, ou, research, families=FAMILIES,
                                 log=False, collect=False)
    values.update(interaction_columns(values, fcfg, bs.bar_seconds))
    return values


@cache
def _full() -> dict[str, np.ndarray]:
    return _compute(synthetic_bars(N))


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True))


def test_every_registered_feature_is_computed() -> None:
    fcfg = _configs()[0]
    names = {s.name for s in build_registry(fcfg, "1h", 3600.0)}
    assert names == set(_full())


@pytest.mark.slow
@pytest.mark.parametrize("cut", [1100, 1733, 2600])
def test_prefix_invariance_of_every_feature(cut: int) -> None:
    full = _full()
    prefix = _compute(synthetic_bars(N, cut=cut))
    bad = [name for name, values in prefix.items() if not _same(values, full[name][:cut])]
    assert not bad, f"features that change when later bars exist (cut {cut}): {bad}"
    defined = [name for name, values in prefix.items() if np.isfinite(values[-200:]).any()]
    assert len(defined) > 0.8 * len(prefix)                 # the check is not vacuous


@pytest.mark.slow
def test_wild_future_bars_change_nothing_before_them() -> None:
    cut = 2400
    full = _full()
    wild = _compute(with_wild_future(synthetic_bars(N), cut))
    bad = [name for name, values in wild.items() if not _same(values[:cut], full[name][:cut])]
    assert not bad, f"features that see the future: {bad}"
    moved = [name for name, values in wild.items()
             if not _same(values[cut:], full[name][cut:])]
    assert len(moved) > 0.8 * len(wild)                     # the future really was different


def test_causal_scaling_and_winsorising_are_prefix_invariant() -> None:
    bs = synthetic_bars(N)
    x = np.log(bs.tick_count + 1.0)
    x[::17] = np.nan
    full = scaling_variants(x, bs.timestamps, 200, min_history=300)
    wfull = winsorize_causal(x, bs.timestamps, 0.01, 0.99, min_history=300)
    for cut in (700, 2222):
        part = scaling_variants(x[:cut], bs.timestamps.head(cut), 200, min_history=300)
        for name, values in part.items():
            assert _same(values, full[name][:cut]), (name, cut)
        assert _same(winsorize_causal(x[:cut], bs.timestamps.head(cut), 0.01, 0.99,
                                      min_history=300), wfull[:cut])
    wild = x.copy()
    wild[2500:] = 1e9
    assert _same(causal_zscore(wild, 200)[:2500], causal_zscore(x, 200)[:2500])


def test_a_full_sample_zscore_would_be_caught() -> None:
    x = np.log(synthetic_bars(N).tick_count + 1.0)

    def leaky(v: np.ndarray) -> np.ndarray:
        return (v - np.nanmean(v)) / np.nanstd(v)

    assert not _same(leaky(x[:1500]), leaky(x)[:1500])


def test_causal_zscore_tolerates_sparse_missing_values() -> None:
    x = np.random.default_rng(3).normal(size=5000)
    x[::50] = np.nan                                        # 2 % missing: every window has some
    z = causal_zscore(x, 400)
    assert np.isfinite(z[400:][np.isfinite(x[400:])]).all()
    assert np.isnan(z[:399]).all()                          # never a partial first window


def test_regime_features_are_missing_without_a_regime_store() -> None:
    full = _full()
    regime = [k for k in full if k.startswith("regime_")]
    assert regime and all(np.isnan(full[k]).all() for k in regime)
