"""Streaming reconstruction matches the batch causal computation (Prompt #9, Steps 53-54,
60, 69).

Bars are replayed one at a time into a rolling buffer shorter than the history;
every selected feature recomputed from the buffer must equal the stored batch
value (float32 rounding). A stored value that was computed differently - here,
one altered cell - is reported as a mismatch, so the check can fail.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import polars as pl

from feature_synth import synthetic_bars
from xauusd_quant.features import load_regression_config
from xauusd_quant.features.factory import compute_families
from xauusd_quant.features.factory_config import load_features_config
from xauusd_quant.features.joins import EngineProvider
from xauusd_quant.features.registry import build_registry
from xauusd_quant.features.spectral_config import load_spectral_config
from xauusd_quant.features.wavelet_config import load_wavelet_config
from xauusd_quant.models import load_ou_config
from xauusd_quant.research.config import load_research_config
from xauusd_quant.selection.live import live_reconstruction, required_buffer

N = 2600
NAMES = ["ret_1", "ret_5", "ret_z_64", "ret_kurt_64", "log_rv_20", "log_rv_64",
         "log_ewma_vol_94", "tod_sin", "session_london"]


@cache
def _configs() -> tuple:
    return (load_features_config(), load_regression_config(), load_ou_config(),
            load_spectral_config(), load_wavelet_config(), load_research_config())


@cache
def _batch() -> tuple:
    fcfg, regression, ou, spectral, wavelet, research = _configs()
    bars = synthetic_bars(N)
    provider = EngineProvider(bars=bars, regression_config=regression, spectral_config=spectral,
                              wavelet_config=wavelet)
    values, _ = compute_families(bars, provider, fcfg, ou, research,
                                 families=("returns", "volatility", "time"), log=False,
                                 collect=False)
    registry = {s.name: s.to_dict() for s in build_registry(fcfg, "1h", 3600.0)}
    stored = pl.DataFrame({"timestamp": bars.timestamps, **{n: values[n] for n in NAMES}})
    return bars, stored, registry


def _replay(stored: pl.DataFrame, **kw) -> dict:
    fcfg, regression, ou, spectral, wavelet, research = _configs()
    bars, _, registry = _batch()
    return live_reconstruction(bars, stored, NAMES, registry, fcfg=fcfg, ou=ou,
                               research=research, regression=regression, spectral=spectral,
                               wavelet=wavelet, **kw)


def test_streaming_reconstruction_matches_the_batch_values() -> None:
    _, stored, registry = _batch()
    assert all(n in registry for n in NAMES)
    out = _replay(stored, start=2000, steps=12, buffer=700, stride=37)
    assert out["passed"], out["failed"]
    assert out["steps"] == 12 and out["features_checked"] == NAMES
    assert set(out["families"]) == {"returns", "volatility", "time"}
    checked = [n for n in NAMES if out["max_abs_difference"][n] >= 0.0]
    assert len(checked) == len(NAMES)


def test_an_altered_stored_value_is_caught() -> None:
    _, stored, _ = _batch()
    row = 2000 + 37 * 3
    bad = stored.with_columns(pl.when(pl.int_range(pl.len()) == row)
                              .then(pl.col("log_rv_20") * 1.01).otherwise(pl.col("log_rv_20"))
                              .alias("log_rv_20"))
    out = _replay(bad, start=2000, steps=6, buffer=700, stride=37)
    assert not out["passed"] and out["failed"] == ["log_rv_20"]
    assert out["mismatches"]["log_rv_20"] == 1


def test_required_buffer_covers_twice_the_longest_warm_up() -> None:
    _, _, registry = _batch()
    buf = required_buffer(registry, NAMES, percentile_window=100)
    longest = max(int(registry[n]["min_history"]) for n in NAMES)
    assert buf >= 2 * longest + 3000
    assert np.isfinite(buf)
