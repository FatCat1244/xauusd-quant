"""Synthetic stored feature matrix and target table for the Prompt #9 tests.

Nothing here reads the real dataset. :func:`write_stores` lays out a feature
factory store and a target store on a temporary directory exactly as the
factory writes them (``timeframe=/version=/year=/part-0.parquet``, manifests,
registry, ``CURRENT`` pointers), with three bars every weekday from 2003 to
2023, so the configured development (2003-2017), validation (2018-2021) and
reserved test (2022-) periods all exist.

The features are built to hit known outcomes: a direction signal and a
near-copy of it, a volatility driver and a smoothed relative, pure noise, and
one feature for each exclusion reason of the quality filter. The reserved
period's target values are poisoned (``1e30``) and, by default, its target year
files are overwritten with garbage bytes: a loader that opened them, or kept a
reserved value, fails loudly.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from xauusd_quant.features.factory import TIMESTAMP_CONVENTION
from xauusd_quant.features.factory_config import FeatureFactoryConfig, load_features_config
from xauusd_quant.selection.config import FeatureSelectionConfig, load_selection_config
from xauusd_quant.targets.config import TargetConfig, load_targets_config

TF = "1h"
POISON = 1.0e30
RESERVED = date(2022, 1, 1)

#: name -> (family, window, parameter family, min_history, extra registry fields)
FEATURES: dict[str, tuple[str, int | None, str | None, int, dict]] = {
    "sig_a": ("returns", 20, "sig", 21, {}),
    "sig_a_twin": ("returns", 40, "sig", 41, {}),
    "vol_x": ("volatility", 20, "vol", 21, {}),
    "vol_x_slow": ("volatility", 200, "vol", 201, {}),
    "noise_1": ("autocorrelation", 64, None, 65, {}),
    "noise_2": ("microstructure", 64, None, 65, {}),
    "rare_flag": ("time", None, None, 0, {}),
    "drifting": ("regression", 128, None, 129, {}),
    "const_flag": ("time", None, None, 0, {}),
    "tiny_flag": ("time", None, None, 0, {}),
    "gappy": ("ou", 256, None, 257, {}),
    "late_start": ("fft", 256, None, 257, {}),
    "huge": ("wavelet", 256, None, 257, {}),
    "invalid_one": ("fft", 1024, None, 1025, {"invalid_reason": "window above the day length"}),
    "leaky_smooth": ("regression", 128, None, 129, {"live_safe": False}),
}
#: the exclusion each feature is built to trigger (none for the first eight)
EXPECTED_REASON = {
    "const_flag": "EXCLUDE_NEAR_CONSTANT", "tiny_flag": "EXCLUDE_NEAR_CONSTANT",
    "gappy": "EXCLUDE_MISSINGNESS", "late_start": "EXCLUDE_UNAVAILABLE",
    "huge": "EXCLUDE_NUMERICAL", "invalid_one": "EXCLUDE_INVALID",
    "leaky_smooth": "EXCLUDE_NON_CAUSAL"}


def stamps(start: date = date(2003, 1, 1), end: date = date(2024, 1, 1)) -> pl.Series:
    """Three bars every weekday (01:00, 09:00, 17:00), weekends skipped."""
    out = []
    d = start
    while d < end:
        if d.weekday() < 5:
            out += [datetime(d.year, d.month, d.day, h) for h in (1, 9, 17)]
        d += timedelta(days=1)
    return pl.Series("timestamp", out, dtype=pl.Datetime("us"))


def _ar1(rng: np.random.Generator, n: int, phi: float) -> np.ndarray:
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + np.sqrt(1 - phi * phi) * e[i]
    return x


def synthetic_frames(seed: int = 0) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(feature frame, target frame) on the same timestamps."""
    ts = stamps()
    n = ts.len()
    rng = np.random.default_rng(seed)
    years = ts.dt.year().to_numpy()
    reserved = (ts >= datetime(2022, 1, 1)).to_numpy()
    sig = _ar1(rng, n, 0.3)
    vol = _ar1(rng, n, 0.98)
    f: dict[str, np.ndarray] = {                     # copies: warm-up NaNs go in below
        "sig_a": sig.copy(),
        "sig_a_twin": sig + 0.05 * rng.normal(size=n),
        "vol_x": vol.copy(),
        "vol_x_slow": 0.8 * vol + 0.6 * _ar1(rng, n, 0.99),
        "noise_1": rng.normal(size=n),
        "noise_2": _ar1(rng, n, 0.9),
        "rare_flag": (rng.random(n) < 0.004).astype(float),
        "drifting": rng.normal(size=n) + np.where(years >= 2012, 3.0, 0.0),
        "const_flag": np.ones(n),
        "tiny_flag": np.zeros(n),
        "gappy": np.where(rng.random(n) < 0.5, np.nan, rng.normal(size=n)),
        "late_start": np.where(reserved, np.nan, rng.normal(size=n)),
        "huge": rng.normal(size=n) * 1e8,
        "invalid_one": rng.normal(size=n),
        "leaky_smooth": np.convolve(sig, np.ones(9) / 9, mode="same"),
    }
    f["tiny_flag"][::7000] = 1.0                       # minority share ~0.0002 < 0.001
    for name, (_, _, _, warm, _) in FEATURES.items():
        f[name][:warm] = np.nan
    features = pl.DataFrame({"timestamp": ts, **{k: v.astype(np.float32) for k, v in f.items()}})
    # targets: horizon-h sums of a return process that sig_a and vol_x drive
    sigma = 0.004 * np.exp(0.5 * vol)
    r_next = 0.12 * sig * sigma + sigma * rng.standard_t(6, n) / np.sqrt(1.5)
    t: dict[str, np.ndarray] = {}
    for h in (1, 5, 20):
        c = np.concatenate(([0.0], np.cumsum(r_next)))
        ret = np.full(n, np.nan)
        ret[: n - h + 1] = c[h:] - c[: n - h + 1]
        c2 = np.concatenate(([0.0], np.cumsum(r_next ** 2)))
        rv = np.full(n, np.nan)
        rv[: n - h + 1] = np.log(np.sqrt((c2[h:] - c2[: n - h + 1]) / h))
        t[f"target_return_{h}"] = ret
        t[f"target_residual_reduction_{h}"] = rng.normal(size=n)
        t[f"target_realized_vol_{h}"] = rv
        t[f"target_abs_return_{h}"] = np.abs(ret)
    for v in t.values():
        v[reserved] = POISON
    targets = pl.DataFrame({"timestamp": ts, **{k: v.astype(np.float32) for k, v in t.items()}})
    return features, targets


def registry_rows(fcfg: FeatureFactoryConfig) -> list[dict]:
    rows = []
    for i, (name, (family, window, pfam, warm, extra)) in enumerate(FEATURES.items()):
        rows.append({"name": name, "feature_id": f"{TF}.{family}.{name}", "family": family,
                     "window": window, "parameter_family": pfam, "min_history": warm,
                     "cost": "cheap", "live_safe": True, "feature_version": f"fv-{i:03d}",
                     "dataset_version": "ticks-synthetic", "timeframe": TF,
                     "source_feed": fcfg.source_feed, "invalid_reason": None,
                     "source_module": "tests.selection_synth", **extra})
    return rows


def _write_years(frame: pl.DataFrame, base: Path) -> None:
    for (year,), part in frame.with_columns(pl.col("timestamp").dt.year().alias("_y")) \
            .group_by("_y", maintain_order=True):
        d = base / f"year={year}"
        d.mkdir(parents=True, exist_ok=True)
        part.drop("_y").write_parquet(d / "part-0.parquet")


def write_stores(root: Path, *, seed: int = 0, corrupt_reserved_targets: bool = True
                 ) -> tuple[FeatureFactoryConfig, TargetConfig]:
    """The synthetic factory + target stores under *root*; the configs pointing at them."""
    fcfg = replace(load_features_config(), factory_path=root / "factory",
                   results_path=root / "feature_research")
    tcfg = replace(load_targets_config(), targets_path=root / "targets")
    features, targets = synthetic_frames(seed)
    fbase = fcfg.factory_path / f"timeframe={TF}" / "version=factory-synthetic"
    _write_years(features, fbase)
    (fbase / "registry.json").write_text(json.dumps(registry_rows(fcfg)), encoding="utf-8")
    (fbase / "_manifest.json").write_text(json.dumps({
        "timeframe": TF, "dataset_lineage": {"partial": False},
        "timestamp_convention": TIMESTAMP_CONVENTION, "source_feed": fcfg.source_feed,
        "bar_dataset_version": "bars-synthetic", "tick_dataset_version": "ticks-synthetic",
        "factory_version": "factory-synthetic", "rows": features.height}), encoding="utf-8")
    (fbase.parent / "CURRENT").write_text("factory-synthetic\n", encoding="utf-8")
    tbase = tcfg.targets_path / f"timeframe={TF}" / "version=targets-synthetic"
    _write_years(targets, tbase)
    (tbase / "_manifest.json").write_text(json.dumps({
        "sources": {"bar_dataset_version": "bars-synthetic"},
        "target_version": "targets-synthetic"}), encoding="utf-8")
    (tbase.parent / "CURRENT").write_text("targets-synthetic\n", encoding="utf-8")
    if corrupt_reserved_targets:
        for year in (2022, 2023):
            (tbase / f"year={year}" / "part-0.parquet").write_bytes(b"not a parquet file")
    return fcfg, tcfg


def selection_config(**null_screen: int) -> FeatureSelectionConfig:
    """The project's selection config with a lighter null screen for tests."""
    cfg = load_selection_config()
    ns = replace(cfg.null_screen, circular_shifts=null_screen.get("circular_shifts", 60),
                 min_observations=null_screen.get("min_observations", 1000))
    return replace(cfg, null_screen=ns)
