"""The feature factory: null paths, storage, versions, quality tables (Prompt #8, Steps 2-13, 53).

Pipeline nulls are whole OHLC paths on the real timestamps (so range features
and excursion targets exist on them); a stored matrix round-trips with its
registry, catalog and CURRENT pointer; building twice gives the same values
and version; the quality tables describe missingness and drift without
imputing anything.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest
from scipy.stats import ks_2samp, wasserstein_distance

from feature_synth import synthetic_bars
from xauusd_quant.features import load_regression_config
from xauusd_quant.features.factory import (
    FactoryResult,
    compute_families,
    current_version_dir,
    load_feature_matrix,
    null_bar_series,
    write_feature_matrix,
)
from xauusd_quant.features.factory_config import load_features_config
from xauusd_quant.features.interactions import causal_zscore, interaction_columns
from xauusd_quant.features.joins import EngineProvider, SourceMismatchError, align_to_bars
from xauusd_quant.features.quality import (
    distribution_row,
    drift_rows,
    ks_and_wasserstein,
    missingness_tables,
)
from xauusd_quant.features.registry import build_registry
from xauusd_quant.features.spectral_config import load_spectral_config
from xauusd_quant.features.wavelet_config import load_wavelet_config
from xauusd_quant.models import load_ou_config
from xauusd_quant.research.config import load_research_config

NULLS = ("random_walk", "shuffled_returns", "sign_flip", "block_bootstrap_64")


@pytest.mark.parametrize("name", NULLS)
def test_null_paths_are_valid_ohlc_on_the_real_timestamps(name: str) -> None:
    real = synthetic_bars(3000)
    null = null_bar_series(name, real, seed=11)
    assert null.timestamps.equals(real.timestamps) and null.size == real.size
    assert null.high is not None and null.low is not None and null.open is not None
    body_hi = np.maximum(null.open, null.close)
    body_lo = np.minimum(null.open, null.close)
    assert (null.high >= body_hi * (1 - 1e-12)).all() and (null.low <= body_lo * (1 + 1e-12)).all()
    np.testing.assert_allclose(null.open[1:], null.close[:-1])       # gapless: open = last close
    again = null_bar_series(name, real, seed=11)
    np.testing.assert_array_equal(again.close, null.close)          # reproducible
    assert not np.allclose(null.close, real.close)


def test_each_null_keeps_what_it_should() -> None:
    real = synthetic_bars(3000)
    r = real.log_returns()[1:]
    shuffled = null_bar_series("shuffled_returns", real, seed=1).log_returns()[1:]
    np.testing.assert_allclose(np.sort(shuffled), np.sort(r), atol=1e-9, rtol=0)
    flipped = null_bar_series("sign_flip", real, seed=1).log_returns()[1:]
    np.testing.assert_allclose(np.abs(flipped), np.abs(r), atol=1e-9)     # same |r|, same order
    assert 0.3 < np.mean(np.sign(flipped) == np.sign(r)) < 0.7
    rw = null_bar_series("random_walk", real, seed=1).log_returns()[1:]
    assert np.std(rw) == pytest.approx(np.std(r), rel=0.1)
    ac = lambda v: np.corrcoef(np.abs(v[1:]), np.abs(v[:-1]))[0, 1]    # noqa: E731
    assert ac(r) > 0.1 and abs(ac(shuffled)) < 0.06 and ac(flipped) == pytest.approx(ac(r))


def test_building_twice_gives_identical_features() -> None:
    fcfg = load_features_config()
    bs = synthetic_bars(2500)

    def build() -> dict[str, np.ndarray]:
        provider = EngineProvider(bars=bs, regression_config=load_regression_config(),
                                  spectral_config=load_spectral_config(),
                                  wavelet_config=load_wavelet_config())
        values, _ = compute_families(bs, provider, fcfg, load_ou_config(),
                                     load_research_config(), log=False, collect=False)
        return values

    a, b = build(), build()
    assert set(a) == set(b)
    assert all(np.array_equal(a[k], b[k], equal_nan=True) for k in a)
    assert all(v.dtype == np.float32 for v in a.values())


def test_a_matrix_round_trips_with_its_registry_and_pointer(tmp_path) -> None:
    fcfg = load_features_config()
    bs = synthetic_bars(900)
    specs = [s for s in build_registry(fcfg, "1h", 3600.0) if s.family == "returns"]
    frame = pl.DataFrame({"timestamp": bs.timestamps,
                          **{s.name: np.arange(bs.size, dtype=np.float32) for s in specs}})
    result = FactoryResult(timeframe="1h", frame=frame, registry=specs,
                           manifest={"factory_version": "factory-1h-test", "rows": bs.size})
    base = write_feature_matrix(result, tmp_path)
    assert current_version_dir(tmp_path, "1h") == base
    loaded, manifest, registry = load_feature_matrix(tmp_path, "1h", columns=[specs[0].name])
    assert loaded.columns == ["timestamp", specs[0].name] and loaded.height == bs.size
    assert [r["name"] for r in registry] == [s.name for s in specs]
    assert all("excluded_from_default_candidates" in r for r in registry)
    catalog = json.loads((base / "feature_catalog.json").read_text(encoding="utf-8"))
    assert any(r["family"] == "non_causal" for r in catalog)
    expected = {f"year={y}" for y in bs.timestamps.dt.year().unique().to_list()}
    assert {p.parent.name for p in base.glob("year=*/part-0.parquet")} == expected


def test_sources_are_joined_by_exact_timestamp_only() -> None:
    stamps = synthetic_bars(50).timestamps
    frame = pl.DataFrame({"timestamp": stamps.gather(list(range(0, 50, 2))),
                          "x": np.arange(25.0)})
    joined = align_to_bars(frame, stamps, "test")
    assert joined.height == 50 and joined["x"].null_count() == 25
    stray = pl.DataFrame({"timestamp": [stamps[0].replace(minute=30)], "x": [1.0]})
    with pytest.raises(SourceMismatchError):
        align_to_bars(stray, stamps, "test")
    dup = pl.concat([frame, frame.head(1)])
    with pytest.raises(SourceMismatchError):
        align_to_bars(dup, stamps, "test")


def test_interactions_multiply_causal_zscores_of_their_components() -> None:
    fcfg = load_features_config()
    rng = np.random.default_rng(0)
    n = 5000
    feats = {ix.a: rng.normal(size=n) for ix in fcfg.interactions}
    feats.update({ix.b: rng.normal(size=n) for ix in fcfg.interactions})
    out = interaction_columns(feats, fcfg, 3600.0)
    window = fcfg.interaction_standardisation_days * fcfg.bars_per_day(3600.0)
    ix = fcfg.interactions[0]
    expected = causal_zscore(feats[ix.a], window) * causal_zscore(feats[ix.b], window)
    np.testing.assert_array_equal(out[ix.name], expected)
    missing = dict(feats)
    missing.pop(ix.b)
    assert ix.name not in interaction_columns(missing, fcfg, 3600.0)


def test_quality_tables_describe_without_imputing() -> None:
    bs = synthetic_bars(2000)
    x = np.log(bs.tick_count + 1.0)
    x[:100] = np.nan
    frame = pl.DataFrame({"timestamp": bs.timestamps, "x": x})
    state = np.where(np.arange(bs.size) < 1000, 0, 1)
    overall, by_year, by_month = missingness_tables(frame, ["x"], state)
    row = overall.row(0, named=True)
    assert row["missing_share"] == pytest.approx(0.05)
    assert row["missing_share_state_0"] == pytest.approx(0.1)
    assert row["missing_share_state_1"] == 0.0
    years = bs.timestamps.dt.year().to_numpy()
    dist = distribution_row("x", x, years)
    assert dist["count"] == 1900 and dist["missing"] == 100
    assert "near_constant" not in dist["flags"]
    const = distribution_row("c", np.ones(500), np.full(500, 2020))
    assert "near_constant" in const["flags"]


def test_drift_statistics_agree_with_scipy() -> None:
    rng = np.random.default_rng(4)
    ref = np.sort(rng.standard_t(4, 50_000))
    sample = np.sort(rng.standard_t(4, 8_000) * 1.3 + 0.2)
    ks, w1 = ks_and_wasserstein(sample, ref)
    assert ks == pytest.approx(ks_2samp(sample, ref).statistic, abs=2e-3)
    assert w1 == pytest.approx(wasserstein_distance(sample, ref), rel=0.02)
    years = np.repeat([2019, 2020], 5000)
    x = np.concatenate((rng.normal(size=5000), rng.normal(size=5000) + 2.0))
    rows = {r["year"]: r for r in drift_rows("x", x, years)}
    assert rows[2020]["psi"] > 0.25 and rows[2020]["median_shift_iqr"] > 0.2
    assert rows[2019]["ks"] > 0.2 and rows[2020]["wasserstein_sd"] > 0.5
