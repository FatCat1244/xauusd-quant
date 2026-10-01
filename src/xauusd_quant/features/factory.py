r"""The feature factory: one versioned, causal feature matrix per timeframe (Steps 2, 7).

:func:`factory_gate` stops the build unless every source matches the current
canonical dataset: complete ticks, bars built from them, a current regression
store for every window, stored FFT / wavelet sets that are either current (and
reproduce a fresh computation of their last bars) or absent (then computed),
and a current live-safe regime store. The timestamp convention is the bars':
the broker-local wall clock (New York + 7 h) of each bar's open, ``[t, t+D)``.

:func:`build_feature_matrix` computes every registered family on that one bar
sequence, joins by exact timestamp, stores float32, and checks the schema
(registered, live-safe, no outcome-like column). :func:`write_feature_matrix`
writes ``data/features/factory/timeframe=<tf>/version=<v>/year=YYYY/`` with
``_manifest.json`` (lineage, every feature version and source, coverage) and
``registry.json``; ``CURRENT`` names the version :func:`load_feature_matrix`
reads. A null control's matrix is built with the same code
(:func:`null_bar_series` + :class:`~.joins.EngineProvider`) and never stored.
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..data.resampler import parse_timeframe
from ..data.versioning import dataset_lineage
from ..features.config import RegressionConfig
from ..features.spectral import rolling_spectrum, spectral_feature_frame
from ..features.spectral_config import SpectralConfig
from ..features.store import RegressionFeatureStore, StaleFeaturesError
from ..features.wavelet_causal import rolling_wavelet, wavelet_feature_frame
from ..features.wavelet_config import WaveletConfig
from ..models.config import OUConfig
from ..research.spectral_nulls import _block_bootstrap, bootstrap_block_size, missing_bar_slots
from ..research.study_io import clean_json
from ..utils.clock import utc_now_iso
from ..utils.config import Config
from ..utils.logging import get_logger
from ..utils.paths import atomic_write_text, ensure_dir
from . import families as fam
from .factory_config import FeatureFactoryConfig
from .interactions import interaction_columns
from .joins import EngineProvider, RealProvider, SourceMismatchError, stored_set_problems
from .manifest import (
    assert_feature_schema,
    digest,
    family_code_version,
    feature_version,
)
from .registry import FeatureSpec, build_registry, registry_frame_rows

__all__ = [
    "FactoryResult",
    "TIMESTAMP_CONVENTION",
    "build_feature_matrix",
    "factory_gate",
    "load_bar_series",
    "load_feature_matrix",
    "null_bar_series",
    "write_feature_matrix",
]

LOGGER = get_logger("features.factory")

FACTORY_VERSION = 1
TIMESTAMP_CONVENTION = ("broker-local wall clock (America/New_York + 7 h, US DST), bar open "
                        "time, bars aggregate [t, t+D) closed left")
_CURRENT = "CURRENT"
_MANIFEST = "_manifest.json"
_REGISTRY = "registry.json"
_CATALOG = "feature_catalog.json"


@dataclass
class FactoryResult:
    """A built matrix (float32 columns), its registry and what it was built from."""

    timeframe: str
    frame: pl.DataFrame
    registry: list[FeatureSpec]
    manifest: dict[str, Any]
    extras: dict[str, np.ndarray] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Bars
# ---------------------------------------------------------------------------
def load_bar_series(config: Config, timeframe: str) -> fam.BarSeries:
    files = sorted(config.bars_dir(timeframe).rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {timeframe} bars under {config.bars_dir(timeframe)}")
    bars = (pl.scan_parquet(files)
            .select("timestamp", "open", "high", "low", "close", "tick_count", "median_spread")
            .sort("timestamp").collect())
    if bars["timestamp"].is_duplicated().any():
        raise SourceMismatchError(f"{timeframe} bars: duplicate timestamps")
    bar_seconds = parse_timeframe(timeframe).total_seconds()

    def f64(name: str) -> np.ndarray:
        return bars[name].cast(pl.Float64).fill_null(np.nan).to_numpy()

    return fam.BarSeries(
        name="real", timeframe=timeframe, timestamps=bars["timestamp"], close=f64("close"),
        bar_seconds=bar_seconds, open=f64("open"), high=f64("high"), low=f64("low"),
        median_spread=f64("median_spread"), tick_count=f64("tick_count"),
        missing_slots=missing_bar_slots(bars["timestamp"], bar_seconds, config))


def null_bar_series(name: str, real: fam.BarSeries, *, seed: int,
                    substeps: int = 8) -> fam.BarSeries:
    """A pipeline null: a synthetic OHLC path on the real timestamps.

    ``random_walk``: Gaussian bars with the real close-to-close sd, each built
    from *substeps* Gaussian sub-steps - the open is the previous close, the
    high / low the extremes of the sub-path. ``shuffled_returns``: the real bars
    in random order, each keeping its close-to-close return *and* its own upper
    and lower wick (log distance of the high above max(open, close), of the low
    below min(open, close)). ``sign_flip``: the real bars in their real order
    with independent random signs (a flipped bar's wicks are mirrored) - a
    martingale with the real volatility path, so any direction or residual
    predictability left on it is mechanical or comes from volatility alone.
    ``block_bootstrap_<B>``: the same whole bars in circular blocks of ``B``
    (keeps volatility clustering within a block).
    Range features and excursion targets therefore exist on every null, so a
    mechanical link between a range feature and a residual target is exposed
    the same way as a regression feature's. Spread, activity, missing-bar slots
    and timestamps are the real ones: they are exogenous to a price path.
    """
    order = {"random_walk": 2, "shuffled_returns": 3, "sign_flip": 5}
    block = bootstrap_block_size(name, 256)
    r = real.log_returns()
    n = real.size
    o, h, lo = real.open, real.high, real.low
    has_range = o is not None and h is not None and lo is not None
    if o is not None and h is not None and lo is not None:
        lo_body = np.minimum(np.log(o), np.log(real.close))
        hi_body = np.maximum(np.log(o), np.log(real.close))
        up = np.log(h) - hi_body
        dn = lo_body - np.log(lo)
        bars = np.flatnonzero(np.isfinite(r) & np.isfinite(up) & np.isfinite(dn)
                              & (up >= 0) & (dn >= 0))
    else:
        up = dn = np.zeros(n)
        bars = np.flatnonzero(np.isfinite(r))
    note = ""
    if name == "random_walk":
        rng = np.random.default_rng(seed + order[name])
        sd = float(np.std(r[bars], ddof=1))
        path = np.cumsum(rng.normal(0.0, sd / np.sqrt(substeps), size=(n, substeps)), axis=1)
        steps = path[:, -1]
        wick_up = np.maximum(path.max(axis=1), 0.0) - np.maximum(steps, 0.0)
        wick_dn = np.minimum(steps, 0.0) - np.minimum(path.min(axis=1), 0.0)
        del path
        note = f"{substeps} Gaussian sub-steps per bar give the high / low"
    elif name == "sign_flip":
        rng = np.random.default_rng(seed + order[name])
        sign = np.where(rng.random(n) < 0.5, -1.0, 1.0)
        ok = np.isfinite(r) & np.isfinite(up) & np.isfinite(dn)
        steps = np.where(ok, r, 0.0) * sign
        wick_up = np.where(ok, np.where(sign > 0, up, dn), 0.0)
        wick_dn = np.where(ok, np.where(sign > 0, dn, up), 0.0)
        note = "real bars in real order, independent random signs (wicks mirrored)"
    else:
        if name in order:
            rng = np.random.default_rng(seed + order[name])
            idx = np.resize(rng.permutation(bars), n)
        elif block is not None:
            rng = np.random.default_rng([seed + 4, block])
            idx = _block_bootstrap(bars, n, block, rng)
        else:
            raise ValueError(f"unknown null control {name!r}")
        steps, wick_up, wick_dn = r[idx], up[idx], dn[idx]
        note = "whole real bars (return and both wicks) reordered"
    log_close = np.log(float(real.close[0])) + np.concatenate(([0.0], np.cumsum(steps[1:])))
    log_open = np.concatenate(([log_close[0]], log_close[:-1]))
    out = fam.BarSeries(name=name, timeframe=real.timeframe, timestamps=real.timestamps,
                        close=np.exp(log_close), bar_seconds=real.bar_seconds,
                        median_spread=real.median_spread, tick_count=real.tick_count,
                        missing_slots=real.missing_slots,
                        notes=[f"{name}: synthetic OHLC path, seed {seed}; {note}"])
    if has_range or name == "random_walk":
        out.open = np.exp(log_open)
        out.high = np.exp(np.maximum(log_open, log_close) + wick_up)
        out.low = np.exp(np.minimum(log_open, log_close) - wick_dn)
    return out


# ---------------------------------------------------------------------------
# The integrity gate
# ---------------------------------------------------------------------------
def _regime_dir(fcfg: FeatureFactoryConfig, regime_features_path: Path, timeframe: str) -> Path:
    r = fcfg.families.regime
    return (regime_features_path / f"timeframe={timeframe}" / f"model={r.model}"
            / f"k={r.states}" / f"scheme={r.scheme}")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def factory_gate(config: Config, regression: RegressionConfig, spectral: SpectralConfig,
                 wavelet: WaveletConfig, fcfg: FeatureFactoryConfig, regime_features_path: Path,
                 timeframe: str, *, bars: fam.BarSeries | None = None,
                 recompute_check_bars: int = 2000) -> dict[str, Any]:
    """Refuse to build unless every source matches the current canonical dataset."""
    lineage = dataset_lineage(config, timeframe)
    problems: list[str] = []
    notes: list[str] = []
    if lineage.get("tick_dataset_status") != "complete":
        problems.append(f"tick dataset is {lineage.get('tick_dataset_status')!r} - run "
                        "`xq convert` and `xq verify-dataset`")
    if not lineage.get("bars_match_ticks"):
        problems.append(f"{timeframe} bars were not built from the current ticks - run "
                        f"`xq build-bars --timeframe {timeframe}`")
    if lineage.get("partial"):
        problems.append(str(lineage.get("label")))
    store = RegressionFeatureStore(config, regression)
    for window in sorted(set(fcfg.families.regression.windows)
                         | {fcfg.families.ou.regression_window}):
        try:
            store.load(timeframe, window, ["timestamp"])
            notes.append(f"regression store N={window}: current")
        except (FileNotFoundError, StaleFeaturesError) as exc:
            problems.append(f"regression store N={window}: {exc}")
    stored: dict[str, Any] = {}
    for kind, series, window, root, engine in _stored_sets(fcfg, spectral, wavelet, timeframe):
        manifest = _read_json(root / _MANIFEST)
        if manifest is None:
            notes.append(f"{kind} {series} N={window}: not stored - computed by the engine")
            continue
        mismatch = stored_set_problems(manifest, lineage, engine=engine)
        if mismatch:
            notes.append(f"{kind} {series} N={window}: stored set stale ({', '.join(mismatch)}) "
                         "- recomputed by the engine instead")
            continue
        stored[f"{kind}_{series}_{window}"] = {"root": root, "engine": engine,
                                                "generated_utc": manifest.get("generated_utc")}
        notes.append(f"{kind} {series} N={window}: stored set current")
    regime_dir = _regime_dir(fcfg, regime_features_path, timeframe)
    manifest = _read_json(regime_dir / _MANIFEST)
    mismatch = stored_set_problems(manifest, lineage, rows=False)
    if manifest is None or mismatch or not manifest.get("live_safe"):
        problems.append(f"regime features {regime_dir}: "
                        f"{', '.join(mismatch) or 'missing or not live-safe'} - run "
                        f"`xq regime-walk-forward --timeframe {timeframe} --model "
                        f"{fcfg.families.regime.model} --states {fcfg.families.regime.states}`")
    else:
        notes.append(f"regime features: current ({manifest.get('regime_model_version')})")
    checks: dict[str, Any] = {}
    if not problems and recompute_check_bars > 0 and stored:
        bars = bars or load_bar_series(config, timeframe)
        checks = _tail_checks(stored, bars, config, regression, spectral, wavelet, fcfg,
                              recompute_check_bars)
        for name, check in checks.items():
            if not check["matches"]:
                problems.append(f"stored {name} differs from a fresh computation of its last "
                                f"{recompute_check_bars} bars ({check['detail']}) - rebuild it")
        if all(c["matches"] for c in checks.values()):
            notes.append(f"last {recompute_check_bars} bars of every stored FFT / wavelet set "
                         "recomputed: identical")
    return {"passed": not problems, "problems": problems, "notes": notes, "lineage": lineage,
            "stored_sets": {k: {**v, "root": str(v["root"])} for k, v in stored.items()},
            "recompute_checks": checks, "timestamp_convention": TIMESTAMP_CONVENTION}


def _stored_sets(fcfg: FeatureFactoryConfig, spectral: SpectralConfig, wavelet: WaveletConfig,
                 timeframe: str) -> list[tuple[str, str, int, Path, str]]:
    f, w = fcfg.families.fft, fcfg.families.wavelet
    out = []
    fft_sets = [(f.series, n) for n in f.windows] + [("regression_residual", f.residual_window)]
    if parse_timeframe(timeframe).total_seconds() >= 900:
        fft_sets.append(("abs_ou_innovation", f.abs_innovation_window))
    for series, n in fft_sets:
        out.append(("fft", series, n, spectral.features_path / f"timeframe={timeframe}"
                    / f"source={series}" / f"fft_window={n}", spectral.engine_fingerprint()))
    for n in w.windows:
        out.append(("wavelet", w.series, n, wavelet.features_path / f"timeframe={timeframe}"
                    / f"source={w.series}" / f"window={n}", wavelet.engine_fingerprint()))
    return out


def _tail_checks(stored: dict[str, Any], bars: fam.BarSeries, config: Config,
                 regression: RegressionConfig, spectral: SpectralConfig, wavelet: WaveletConfig,
                 fcfg: FeatureFactoryConfig, bars_n: int) -> dict[str, Any]:
    """Recompute the last *bars_n* bars of each stored set and compare (FFT exact, wavelet 1e-6)."""
    from ..models.ornstein_uhlenbeck import rolling_ou

    series: dict[str, np.ndarray] = {"log_return": bars.log_returns()}
    need = {k.split("_", 1)[1].rsplit("_", 1)[0] for k in stored}
    residual = None
    if "regression_residual" in need or "abs_ou_innovation" in need:
        store = RegressionFeatureStore(config, regression)
        frame = store.load(bars.timeframe, fcfg.families.ou.regression_window,
                           ["timestamp", "residual"])
        residual = frame["residual"].cast(pl.Float64).fill_null(np.nan).to_numpy()
        series["regression_residual"] = residual
    out: dict[str, Any] = {}
    for name, info in stored.items():
        kind, rest = name.split("_", 1)
        src, window_s = rest.rsplit("_", 1)
        window = int(window_s)
        n = bars.size
        lo = max(0, n - bars_n - 2 * window - 1)
        if src == "abs_ou_innovation":
            if "abs_ou_innovation" not in series and residual is not None:
                from ..models.config import load_ou_config

                ou = load_ou_config()
                lo_r = max(0, lo - 4 * fcfg.families.ou.full_window)
                fit = rolling_ou(residual[lo_r:], fcfg.families.ou.full_window,
                                 rules=ou.validity_rules(), chunk_rows=ou.numerical.chunk_rows)
                inn = np.full(n, np.nan)
                inn[lo_r:] = np.abs(fit.innovation)
                series["abs_ou_innovation"] = inn
            values = series.get("abs_ou_innovation")
            # The rolling OU is block-shifted: a later start changes its rounding.
            tol = 1e-6
        else:
            values = series.get(src)
            tol = 0.0 if kind == "fft" else 1e-6
        if values is None:
            continue
        ts = bars.timestamps.slice(lo)
        slots = bars.missing_slots[lo:] if bars.missing_slots is not None else None
        if kind == "fft":
            res = rolling_spectrum(values[lo:], fft_window=window, config=spectral,
                                   bar_seconds=bars.bar_seconds, missing_slots=slots)
            fresh = spectral_feature_frame(res, ts, spectral).tail(bars_n)
            columns = ["spectral_entropy", "low_power_share", "high_power_share"]
            pattern = "year=*/features.parquet"
            if src == "abs_ou_innovation":
                tol = 1e-4
        else:
            res2 = rolling_wavelet(values[lo:], window=window, config=wavelet,
                                   bar_seconds=bars.bar_seconds, missing_slots=slots)
            fresh = wavelet_feature_frame(res2, ts, float32=wavelet.features_output.float32
                                          ).tail(bars_n)
            columns = ["wavelet_entropy", "wavelet_fast_slow_log_ratio"]
            pattern = "year=*/part-0.parquet"
        files = sorted(Path(info["root"]).glob(pattern))
        first = fresh["timestamp"][0]
        old = (pl.read_parquet(files, columns=["timestamp", *columns]).sort("timestamp")
               .filter(pl.col("timestamp") >= first))
        worst = 0.0
        ok = old.height == fresh.height and old["timestamp"].equals(fresh["timestamp"])
        if ok:
            for c in columns:
                a = fresh[c].cast(pl.Float64).fill_null(np.nan).to_numpy()
                b = old[c].cast(pl.Float64).fill_null(np.nan).to_numpy()
                both = np.isfinite(a) & np.isfinite(b)
                same_missing = bool(np.array_equal(np.isnan(a), np.isnan(b)))
                diff = float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0
                worst = max(worst, diff)
                ok = ok and same_missing and diff <= tol
        out[name] = {"bars": bars_n, "matches": ok, "max_abs_difference": worst,
                     "tolerance": tol, "detail": "identical within tolerance" if ok
                     else f"max |diff| {worst:.2e}"}
    return out


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------
def _f32(values: np.ndarray) -> np.ndarray:
    out = np.array(values, dtype=np.float32)          # always a writable copy
    out[~np.isfinite(out)] = np.nan
    return out


def compute_families(bars: fam.BarSeries, provider: EngineProvider, fcfg: FeatureFactoryConfig,
                     ou: OUConfig, research: Any | None, *,
                     families: tuple[str, ...] | None = None, log: bool = True,
                     collect: bool = True
                     ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Every chosen family as float32 arrays (and the OU equilibria for the targets).

    *collect* runs the garbage collector after each family (large series);
    timing runs on short buffers turn it off - it costs ~60 ms by itself.
    """
    chosen = families or ("returns", "volatility", "autocorrelation", "regression", "ou", "fft",
                          "wavelet", "regime", "microstructure", "time")
    out: dict[str, np.ndarray] = {}
    extras: dict[str, np.ndarray] = {}

    def add(values: dict[str, np.ndarray], family: str, started: float) -> None:
        for name, arr in values.items():
            out[name] = _f32(arr)
        if log:
            LOGGER.info("[%s %s] %s: %d features in %.1fs", bars.timeframe, bars.name, family,
                        len(values), time.perf_counter() - started)

    provider.series["log_return"] = bars.log_returns()
    for family in chosen:
        started = time.perf_counter()
        if family == "returns":
            add(fam.returns_features(bars, fcfg), family, started)
        elif family == "volatility":
            add(fam.volatility_features(bars, fcfg), family, started)
        elif family == "autocorrelation":
            add(fam.autocorrelation_features(bars, fcfg), family, started)
        elif family == "regression":
            add(fam.regression_features(provider, fcfg), family, started)
            for n in fcfg.families.regression.windows:
                if n != fcfg.families.ou.regression_window:
                    provider.drop_regression(n)
        elif family == "ou":
            residual = provider.regression(fcfg.families.ou.regression_window)["residual"]
            provider.series["regression_residual"] = residual
            feats, ex = fam.ou_features(residual, fcfg, rules=ou.validity_rules(),
                                        chunk_rows=ou.numerical.chunk_rows)
            extras.update(ex)
            full = fcfg.families.ou.full_window
            if f"ou_innovation_{full}" in ex:
                provider.series["abs_ou_innovation"] = np.abs(ex[f"ou_innovation_{full}"])
            add(feats, family, started)
        elif family == "fft":
            if "regression_residual" not in provider.series:
                provider.series["regression_residual"] = provider.regression(
                    fcfg.families.ou.regression_window)["residual"]
            add(fam.fft_features(provider, fcfg, bars.bar_seconds), family, started)
        elif family == "wavelet":
            add(fam.wavelet_features(provider, fcfg), family, started)
        elif family == "regime":
            add(fam.regime_features(provider, fcfg, bars.size), family, started)
        elif family == "microstructure":
            add(fam.microstructure_features(bars, fcfg), family, started)
        elif family == "time":
            add(fam.time_features(bars, fcfg, research), family, started)
        if collect:
            gc.collect()
    return out, extras


def build_feature_matrix(config: Config, regression: RegressionConfig, ou: OUConfig,
                         spectral: SpectralConfig, wavelet: WaveletConfig, research: Any,
                         fcfg: FeatureFactoryConfig, regime_features_path: Path, timeframe: str,
                         *, gate: dict[str, Any], live_safe_only: bool = True,
                         families: tuple[str, ...] | None = None,
                         bars: fam.BarSeries | None = None) -> FactoryResult:
    """The real feature matrix of *timeframe* (refused unless *gate* passed)."""
    if not gate.get("passed"):
        raise RuntimeError(f"{timeframe}: factory gate failed - " + "; ".join(gate["problems"]))
    started = time.perf_counter()
    bars = bars or load_bar_series(config, timeframe)
    lineage = gate["lineage"]
    provider = RealProvider(bars=bars, regression_config=regression, spectral_config=spectral,
                            wavelet_config=wavelet, config=config, lineage=lineage,
                            regime_dir=_regime_dir(fcfg, regime_features_path, timeframe),
                            regime_states=fcfg.families.regime.states)
    values, extras = compute_families(bars, provider, fcfg, ou, research, families=families)
    specs = build_registry(fcfg, timeframe, bars.bar_seconds, families=families)
    by_name = {s.name: s for s in specs}
    values.update({k: _f32(v) for k, v in interaction_columns(values, fcfg,
                                                               bars.bar_seconds).items()})
    missing = [s.name for s in specs if s.name not in values]
    if missing:
        LOGGER.warning("[%s] registered but not computed (source absent): %s", timeframe, missing)
    specs = [s for s in specs if s.name in values]
    unregistered = sorted(set(values) - {s.name for s in specs})
    if unregistered:
        raise RuntimeError(f"{timeframe}: computed but unregistered: {unregistered}")
    frame = pl.DataFrame({"timestamp": bars.timestamps,
                          **{s.name: values[s.name] for s in specs}})
    assert_feature_schema(frame.columns, by_name, live_safe_only=live_safe_only)
    if live_safe_only:
        specs = [s for s in specs if s.live_safe]
    code = {family: family_code_version(family) for family in {s.family for s in specs}}
    sources = _source_versions(provider, lineage)
    config_fp = fcfg.fingerprint()
    versioned = []
    coverage = {}
    for s in specs:
        v = feature_version(s, sources=_sources_for(s, sources), config_fingerprint=config_fp,
                            code_version=code[s.family])
        versioned.append(_with_versions(s, lineage.get("tick_dataset_version"), v))
        arr = values[s.name]
        coverage[s.name] = float(np.isfinite(arr).mean())
    version = f"factory-{timeframe}-{digest([s.feature_version for s in versioned], 8)}"
    manifest = {
        "factory_version": version, "factory_schema": FACTORY_VERSION, "timeframe": timeframe,
        "rows": frame.height, "first_timestamp": str(frame["timestamp"][0]),
        "last_timestamp": str(frame["timestamp"][-1]), "features": len(versioned),
        "live_safe_only": live_safe_only, "timestamp_convention": TIMESTAMP_CONVENTION,
        "source_feed": fcfg.source_feed, "dataset_lineage": lineage,
        "tick_dataset_version": lineage.get("tick_dataset_version"),
        "bar_dataset_version": lineage.get("bar_dataset_version"),
        "features_config_fingerprint": config_fp, "sources": sources, "family_code": code,
        "coverage": coverage, "gate": {k: gate[k] for k in ("passed", "notes",
                                                             "recompute_checks")},
        "seconds": time.perf_counter() - started, "generated_utc": utc_now_iso(),
    }
    return FactoryResult(timeframe=timeframe, frame=frame, registry=versioned, manifest=manifest,
                         extras=extras)


def _source_versions(provider: RealProvider, lineage: dict[str, Any]) -> dict[str, Any]:
    out = {"bar_dataset_version": lineage.get("bar_dataset_version"),
           "tick_dataset_version": lineage.get("tick_dataset_version")}
    out.update(clean_json(provider.sources))
    return out


def _sources_for(spec: FeatureSpec, sources: dict[str, Any]) -> dict[str, Any]:
    base = {"bars": sources.get("bar_dataset_version")}
    if spec.family in ("regression", "ou") or spec.name.startswith(("fft_resid", "fft_abs_eta")):
        base.update({k: v for k, v in sources.items() if k.startswith("regression_")})
    if spec.family == "fft":
        base.update({k: v for k, v in sources.items() if k.startswith("fft_")})
    if spec.family == "wavelet":
        base.update({k: v for k, v in sources.items() if k.startswith("wavelet_")})
    if spec.family == "regime":
        base["regime"] = sources.get("regime")
    return base


def _with_versions(spec: FeatureSpec, dataset_version: str | None,
                   version: str) -> FeatureSpec:
    from dataclasses import replace

    return replace(spec, dataset_version=dataset_version, feature_version=version)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
def write_feature_matrix(result: FactoryResult, root: Path) -> Path:
    """``root/timeframe=<tf>/version=<v>/year=YYYY/part-0.parquet`` + manifest + registry."""
    version = result.manifest["factory_version"]
    base = ensure_dir(root / f"timeframe={result.timeframe}" / f"version={version}")
    frame = result.frame.with_columns(pl.col("timestamp").dt.year().alias("_year"))
    for (year,), part in frame.group_by("_year", maintain_order=True):
        directory = ensure_dir(base / f"year={year}")
        tmp = directory / "part-0.parquet.partial"
        part.drop("_year").write_parquet(tmp, compression="zstd", compression_level=3)
        tmp.replace(directory / "part-0.parquet")
    atomic_write_text(base / _REGISTRY, json.dumps(clean_json(
        [{**s.to_dict(), "excluded_from_default_candidates": s.status_excludes_default}
         for s in result.registry]), indent=1, default=str) + "\n")
    # the catalog adds the non-causal constructions, registered only to record their exclusion
    atomic_write_text(base / _CATALOG, json.dumps(clean_json(registry_frame_rows(
        result.registry)), indent=1, default=str) + "\n")
    atomic_write_text(base / _MANIFEST, json.dumps(clean_json(result.manifest), indent=1,
                                                   default=str) + "\n")
    atomic_write_text(base.parent / _CURRENT, version + "\n")
    return base


def current_version_dir(root: Path, timeframe: str) -> Path:
    pointer = root / f"timeframe={timeframe}" / _CURRENT
    if not pointer.exists():
        raise FileNotFoundError(f"no feature matrix for {timeframe} - run "
                                f"`xq build-feature-factory --timeframe {timeframe}`")
    return root / f"timeframe={timeframe}" / f"version={pointer.read_text().strip()}"


def load_feature_matrix(root: Path, timeframe: str, columns: list[str] | None = None
                        ) -> tuple[pl.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    """(frame, manifest, registry rows) of the current version; *columns* limits the read."""
    base = current_version_dir(root, timeframe)
    manifest = _read_json(base / _MANIFEST) or {}
    registry = json.loads((base / _REGISTRY).read_text(encoding="utf-8"))
    files = sorted(base.glob("year=*/part-0.parquet"))
    wanted = None if columns is None else ["timestamp", *[c for c in columns if c != "timestamp"]]
    frame = pl.read_parquet(files, columns=wanted).sort("timestamp")
    if frame.height != manifest.get("rows"):
        raise SourceMismatchError(f"{timeframe}: feature matrix has {frame.height} rows, manifest "
                             f"{manifest.get('rows')}")
    return frame, manifest, registry
