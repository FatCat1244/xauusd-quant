"""The versioned regression-feature store.

Three properties matter. A loaded frame must be *identical* to a freshly
computed one, so caching can never change a research result. A cache made
from other bars or settings must be refused, so features from one dataset
version can never be mixed with another. And the store must not keep
redundant copies of columns that do not depend on the window.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from conftest import make_clean_rows, write_csv
from xauusd_quant.data.resampler import bar_dataset_version
from xauusd_quant.features import load_regression_config
from xauusd_quant.features.rolling_regression import rolling_regression_features
from xauusd_quant.features.store import (
    BASE_COLUMNS,
    RegressionFeatureStore,
    StaleFeaturesError,
)
from xauusd_quant.research.regression_reports import load_bars


def test_a_loaded_frame_is_identical_to_a_fresh_computation(world):
    config, regression = world
    store = RegressionFeatureStore(config, regression)
    store.build("1m", [32, 64])
    bars = load_bars(config, "1m", None, None)
    for window in (32, 64):
        fresh, _ = rolling_regression_features(
            bars, window=window, config=regression, timeframe="1m"
        )
        loaded = store.load("1m", window)
        assert loaded.columns == fresh.columns
        assert loaded.equals(fresh), f"window {window}: the cache changed the data"


def test_a_column_subset_is_the_full_frames_values_read_from_disk(world):
    """The OU study reads seven columns; they must equal the full frame's."""
    config, regression = world
    store = RegressionFeatureStore(config, regression)
    store.build("1m", [32])
    full = store.load("1m", 32)
    # Stored, base, and all three re-derived columns, in a caller's own order.
    wanted = ["residual", "timestamp", "residual_zscore_rolling", "residual_pct",
              "log_return", "fitted_price"]
    subset = store.load("1m", 32, wanted)
    assert subset.columns == wanted
    assert subset.equals(full.select(wanted))
    for alone in (["log_return"], ["residual"]):
        assert store.load("1m", 32, alone).equals(full.select(alone)), alone
    with pytest.raises(KeyError, match="not_a_column"):
        store.load("1m", 32, ["residual", "not_a_column"])


def test_window_independent_columns_are_stored_once(world):
    config, regression = world
    store = RegressionFeatureStore(config, regression)
    store.build("1m", [32, 64])
    directory = store.timeframe_dir("1m")
    base = pl.read_parquet(directory / "base.parquet")
    assert tuple(base.columns) == BASE_COLUMNS
    for window in (32, 64):
        specific = pl.read_parquet(directory / f"window={window}" / "features.parquet")
        assert not set(specific.columns) & set(BASE_COLUMNS)
        assert "residual_zscore_rolling" not in specific.columns, "alias stored twice"


def test_rebuilt_bars_make_the_cache_stale_and_it_is_refused(world):
    config, regression = world
    store = RegressionFeatureStore(config, regression)
    store.build("1m", [32])
    manifest = store.manifest("1m")
    assert manifest is not None
    assert manifest["bar_dataset_version"] == bar_dataset_version(config, "1m")

    # Pretend the bars were rebuilt from a different tick dataset.
    bar_manifest_path = config.bars_dir("1m") / "_manifest.json"
    text = bar_manifest_path.read_text(encoding="utf-8")
    bar_manifest_path.write_text(
        text.replace(manifest["bar_dataset_version"], "bars-1m-somethingelse"),
        encoding="utf-8",
    )
    with pytest.raises(StaleFeaturesError, match="current bars"):
        store.load("1m", 32)

    # Rebuilding removes the stale cache instead of mixing it with the new one.
    store.build("1m", [32])
    fresh = store.manifest("1m")
    assert fresh is not None
    assert fresh["bar_dataset_version"] == "bars-1m-somethingelse"
    assert list(fresh["windows"]) == ["32"]


def test_changed_regression_settings_make_the_cache_stale(world, tmp_path):
    config, regression = world
    RegressionFeatureStore(config, regression).build("1m", [32])
    import yaml

    from xauusd_quant.features.config import load_regression_config as load

    raw = yaml.safe_load(regression.config_path.read_text(encoding="utf-8"))
    raw["conditioning"]["volatility_window"] = 20
    raw["features_path"] = str(regression.features_path).replace("\\", "/")
    path = tmp_path / "regression_changed.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    changed = load(path)
    with pytest.raises(StaleFeaturesError, match="different regression settings"):
        RegressionFeatureStore(config, changed).load("1m", 32)


def test_research_only_settings_do_not_invalidate_the_cache(world, tmp_path):
    config, regression = world
    RegressionFeatureStore(config, regression).build("1m", [32])
    import yaml

    from xauusd_quant.features.config import load_regression_config as load

    raw = yaml.safe_load(regression.config_path.read_text(encoding="utf-8"))
    raw["extremes"]["max_crossing_horizon"] = 300          # outcome analysis only
    raw["extremes"]["crossing_checkpoints"] = [5, 10, 300]
    raw["features_path"] = str(regression.features_path).replace("\\", "/")
    path = tmp_path / "regression_research_only.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert RegressionFeatureStore(config, load(path)).load("1m", 32).height > 0


def test_a_partial_frame_is_never_cached(world):
    config, regression = world
    bars = load_bars(config, "1m", None, None).head(500)
    frame, diagnostics = rolling_regression_features(
        bars, window=32, config=regression, timeframe="1m"
    )
    with pytest.raises(RuntimeError, match="only full-history features are cached"):
        RegressionFeatureStore(config, regression).save("1m", 32, frame, diagnostics)


def test_features_need_verified_bars(tmp_path, config_factory, monkeypatch):
    rows = make_clean_rows(datetime(2021, 1, 4, 1, 0, 0), 600)
    config = config_factory(write_csv(tmp_path / "t.csv", rows))
    monkeypatch.setenv("XAUUSD_FEATURES_PATH", str(tmp_path / "features").replace("\\", "/"))
    with pytest.raises(RuntimeError, match="build-bars"):
        RegressionFeatureStore(config, load_regression_config()).build("1m", [32])


def test_coverage_marks_current_and_stale_entries(world):
    config, regression = world
    store = RegressionFeatureStore(config, regression)
    store.build("1m", [32, 64])
    rows = store.coverage(["1m"])
    assert {r["window"] for r in rows} == {32, 64}
    assert all(r["current"] for r in rows)
    assert all(r["valid_fits"] < r["rows"] for r in rows), "warm-up rows are not valid fits"
