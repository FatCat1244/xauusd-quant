"""The stored feature matrix and targets against the real dataset (Prompt #8).

Skipped unless a 1h feature matrix has been built (`xq build-feature-factory -t 1h`).
The stored matrix must hold only registered, live-safe columns on exactly the
current bars; its values must equal a fresh computation on a *prefix* of the
real bars (stored sets, store-backed regression and engine recomputation are
the same thing, and nothing saw later bars); the targets must sit on the same
bars in their own table.
"""

from __future__ import annotations

import numpy as np
import pytest

from xauusd_quant import load_config
from xauusd_quant.features import load_regression_config
from xauusd_quant.features.factory import (
    compute_families,
    current_version_dir,
    load_bar_series,
    load_feature_matrix,
)
from xauusd_quant.features.factory_config import load_features_config
from xauusd_quant.features.interactions import interaction_columns
from xauusd_quant.features.joins import EngineProvider
from xauusd_quant.features.manifest import FORBIDDEN_MARKERS
from xauusd_quant.features.spectral_config import load_spectral_config
from xauusd_quant.features.wavelet_config import load_wavelet_config
from xauusd_quant.models import load_ou_config
from xauusd_quant.research.config import load_research_config
from xauusd_quant.targets.alignment import check_alignment, load_targets
from xauusd_quant.targets.config import load_targets_config

pytestmark = pytest.mark.realdata
TF = "1h"
CUT = 30_000


@pytest.fixture(scope="module")
def stored():
    fcfg = load_features_config()
    try:
        current_version_dir(fcfg.factory_path, TF)
    except FileNotFoundError:
        pytest.skip("no stored 1h feature matrix; run `xq build-feature-factory -t 1h`")
    return load_feature_matrix(fcfg.factory_path, TF)


def test_the_matrix_holds_only_registered_live_safe_columns_on_the_bars(stored) -> None:
    frame, manifest, registry = stored
    names = [r["name"] for r in registry]
    assert frame.columns == ["timestamp", *names]
    assert all(r["live_safe"] for r in registry)
    assert not [c for c in names if any(m in c.lower() for m in FORBIDDEN_MARKERS)]
    bars = load_bar_series(load_config(), TF)
    assert frame["timestamp"].equals(bars.timestamps)
    assert manifest["bar_dataset_version"] == manifest["dataset_lineage"]["bar_dataset_version"]


@pytest.mark.slow
def test_stored_values_equal_a_fresh_computation_on_a_prefix(stored) -> None:
    frame, _, registry = stored
    fcfg = load_features_config()
    bars = load_bar_series(load_config(), TF)
    from dataclasses import replace

    prefix = replace(bars, timestamps=bars.timestamps.head(CUT), close=bars.close[:CUT],
                     open=bars.open[:CUT], high=bars.high[:CUT], low=bars.low[:CUT],
                     median_spread=bars.median_spread[:CUT], tick_count=bars.tick_count[:CUT],
                     missing_slots=bars.missing_slots[:CUT])
    provider = EngineProvider(bars=prefix, regression_config=load_regression_config(),
                              spectral_config=load_spectral_config(),
                              wavelet_config=load_wavelet_config())
    values, _ = compute_families(prefix, provider, fcfg, load_ou_config(),
                                 load_research_config(), log=False, collect=False)
    values.update(interaction_columns(values, fcfg, prefix.bar_seconds))
    bad = []
    for r in registry:
        name = r["name"]
        if r["family"] == "regime" or "regime" in name:
            continue                               # the regime store cannot be recomputed here
        fresh = np.asarray(values[name], dtype=np.float32)
        old = frame[name].to_numpy()[:CUT].astype(np.float32)
        same_missing = np.array_equal(np.isnan(fresh), np.isnan(old))
        both = np.isfinite(fresh) & np.isfinite(old)
        scale = np.maximum(np.abs(old[both]), 1.0)
        close = np.all(np.abs(fresh[both] - old[both]) <= 2e-5 * scale)
        if not (same_missing and close):
            bad.append(name)
    assert not bad, f"stored values differ from a prefix recomputation: {bad}"


def test_targets_sit_on_the_same_bars_in_their_own_table(stored) -> None:
    frame, _, _ = stored
    tcfg = load_targets_config()
    targets, manifest = load_targets(tcfg.targets_path, TF)
    check_alignment(frame.select("timestamp"), targets)
    assert all(c == "timestamp" or c.startswith("target_") for c in targets.columns)
    assert not set(targets.columns) & set(frame.columns) - {"timestamp"}
    r1 = targets["target_return_1"].to_numpy()
    close = load_bar_series(load_config(), TF).close
    np.testing.assert_allclose(r1[:-1], np.log(close[1:] / close[:-1]).astype(np.float32),
                               rtol=1e-5, atol=1e-7)
