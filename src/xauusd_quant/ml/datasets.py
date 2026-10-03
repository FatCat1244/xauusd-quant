r"""Development datasets for the supervised research (Prompt #10, Steps 2, 7-10, 38-40, 92).

:func:`load_ml_data` reads, for one timeframe and **only rows before the
reserved test period** (the year files that start at or after it are never
opened; the rest are filtered at the scan):

* the Prompt #8 feature matrix, restricted to the features of the Prompt #9
  manifests (each manifest verified: content hash, live-safe flags, feature
  versions equal to the stored registry) plus a few *context* features used
  only to split the evaluation (regime state, volatility, spread, session);
* the Prompt #8 target table, every column purged at the reserved start by its
  own horizon (an outcome window may not reach reserved prices);
* the N = 128 regression residual and the trailing volatility (for the
  ``c``-reduction label), cut at the reserved start on load.

Derived labels (``residual_shrinks_c``, ``up_beyond_cost``,
``down_beyond_cost``) and the regression transforms are computed here and never
written back to a store. No date, timestamp or row number is ever a predictor
(Step 40): a design matrix holds manifest features only, in manifest order.

:func:`assert_no_leakage` is the gate every design passes: a column that is not
a registered live-safe feature, that is named like an outcome, or that tracks a
target almost perfectly (a leakage canary) is refused.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..alpha.conditioning import causal_quantile_buckets
from ..alpha.information_coefficient import rank_scores
from ..features.config import RegressionConfig
from ..features.factory import TIMESTAMP_CONVENTION, current_version_dir
from ..features.factory_config import FeatureFactoryConfig
from ..features.store import RegressionFeatureStore
from ..selection.config import FeatureSelectionConfig
from ..selection.manifest import load_manifest
from ..targets.config import TargetConfig
from ..utils.config import Config
from .config import MLConfig, TargetSpec

__all__ = [
    "CONTEXT_FEATURES",
    "LeakageError",
    "MLData",
    "MLIntegrityError",
    "TargetArrays",
    "assert_no_leakage",
    "derive_target",
    "load_ml_data",
    "read_years",
]

#: features read only to split evaluation results (never predictors unless a manifest has them)
CONTEXT_FEATURES = ("regime_p0", "regime_p1", "regime_p2", "regime_entropy", "log_rv_20",
                    "spread_percentile", "spread_rel", "session_asia", "session_london",
                    "session_london_ny_overlap", "session_new_york", "session_off_hours",
                    "reg_resid_z_128", "reg_slope_vol_128", "ou_valid_256", "fft_entropy_256",
                    "wav_entropy_512")
SESSION_ORDER = ("session_london_ny_overlap", "session_london", "session_new_york",
                 "session_asia", "session_off_hours")
_FORBIDDEN_PREFIXES = ("target_", "fwd_", "future_", "offline_", "smooth", "viterbi")
_META_COLUMNS = ("timestamp", "timestamp_utc", "date", "year", "row", "row_number", "index")


class MLIntegrityError(RuntimeError):
    """Stored features, targets and manifests do not describe the same dataset."""


class LeakageError(RuntimeError):
    """A design column could carry information from after its bar."""


def _midnight(d: date) -> datetime:
    return datetime(d.year, d.month, d.day)


def read_years(base: Path, columns: list[str], before: date | None) -> pl.DataFrame:
    """Year files of *base*, rows strictly before *before* only (later year files unopened)."""
    frames = []
    for path in sorted(base.glob("year=*/part-0.parquet")):
        year = int(path.parent.name.split("=")[1])
        if before is not None and date(year, 1, 1) >= before:
            continue
        lazy = pl.scan_parquet(path).select("timestamp", *columns)
        if before is not None:
            lazy = lazy.filter(pl.col("timestamp") < pl.lit(_midnight(before)))
        frames.append(lazy.collect())
    if not frames:
        raise MLIntegrityError(f"{base}: no rows before {before}")
    return pl.concat(frames).sort("timestamp")


@dataclass
class TargetArrays:
    """One target at one horizon: the fitted values and the untransformed outcome."""

    name: str
    horizon: int
    task: str
    y: np.ndarray                      # float64, the scale models are fitted on (NaN = no label)
    y_raw: np.ndarray                  # float64, the untransformed outcome (for rank IC)
    positive_rate: float | None = None

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.y)


@dataclass
class MLData:
    timeframe: str
    timestamps: pl.Series
    reserved_start: date
    features: dict[str, np.ndarray]
    manifests: dict[str, list[str]]
    manifest_ids: dict[str, str]
    manifest_hashes: dict[str, str]
    registry: dict[str, dict[str, Any]]
    targets_raw: dict[str, np.ndarray]
    epsilon: np.ndarray | None
    sigma: np.ndarray | None
    versions: dict[str, Any]
    context: dict[str, np.ndarray] = field(default_factory=dict)
    _designs: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def n(self) -> int:
        return int(self.timestamps.len())

    def feature_set(self, name: str) -> list[str]:
        if name not in self.manifests:
            raise KeyError(f"{self.timeframe}: no feature set {name!r} "
                           f"(have {sorted(self.manifests)})")
        return list(self.manifests[name])

    def design(self, names: list[str], key: str | None = None) -> np.ndarray:
        """(n, p) float32 matrix of *names*, in that order (cached per key)."""
        assert_no_leakage(names, self.registry)
        cache_key = key or "|".join(names)
        if cache_key not in self._designs:
            x = np.empty((self.n, len(names)), dtype=np.float32)
            for j, name in enumerate(names):
                x[:, j] = self.features[name]
            self._designs[cache_key] = x
        return self._designs[cache_key]

    def drop_designs(self) -> None:
        self._designs.clear()

    def target(self, spec: TargetSpec, horizon: int, log_floor: float) -> TargetArrays:
        return derive_target(spec, horizon, self.targets_raw, epsilon=self.epsilon,
                             sigma=self.sigma, spread_rel=self.features.get("spread_rel"),
                             log_floor=log_floor)

    def year(self) -> np.ndarray:
        return self.timestamps.dt.year().to_numpy()


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------
def _ahead(values: np.ndarray, h: int) -> np.ndarray:
    out = np.full(values.size, np.nan)
    if 0 < h < values.size:
        out[:-h] = values[h:]
    return out


def derive_target(spec: TargetSpec, h: int, raw: dict[str, np.ndarray], *,
                  epsilon: np.ndarray | None, sigma: np.ndarray | None,
                  spread_rel: np.ndarray | None, log_floor: float) -> TargetArrays:
    """The fitted and the untransformed values of *spec* at horizon *h* (NaN = no label)."""
    def col(family: str) -> np.ndarray:
        name = f"target_{family}_{h}"
        if name not in raw:
            raise KeyError(f"target column {name} was not loaded")
        return np.asarray(raw[name], dtype=np.float64)

    src = spec.source
    if src == "residual_shrinks":
        y = col("residual_shrinks")
        y_raw = y
    elif src == "residual_shrinks_c":
        if epsilon is None or spec.c is None:
            raise KeyError("residual_shrinks_c needs the regression residual")
        e = np.asarray(epsilon, dtype=np.float64)
        later = _ahead(e, h)
        y = np.full(e.size, np.nan)
        ok = np.isfinite(e) & np.isfinite(later) & (np.abs(e) > 0)
        y[ok] = (np.abs(later[ok]) < spec.c * np.abs(e[ok])).astype(np.float64)
        y_raw = y
    elif src in ("up_beyond_cost", "down_beyond_cost"):
        if spread_rel is None or spec.cost_multiple is None:
            raise KeyError(f"{src} needs spread_rel")
        r = col("return")
        buffer = spec.cost_multiple * np.asarray(spread_rel, dtype=np.float64) * 1e-4
        y = np.full(r.size, np.nan)
        ok = np.isfinite(r) & np.isfinite(buffer)
        y[ok] = (r[ok] > buffer[ok]) if src == "up_beyond_cost" else (r[ok] < -buffer[ok])
        y_raw = r
    else:
        family = {"residual_reduction": "residual_reduction", "return": "return",
                  "realized_vol": "realized_vol", "abs_return": "abs_return"}[src]
        y_raw = col(family)
        if spec.transform == "none":
            y = y_raw.copy()
        elif spec.transform == "log":
            with np.errstate(invalid="ignore", divide="ignore"):
                y = np.log(y_raw + log_floor)
        elif spec.transform == "vol_scaled":
            if sigma is None:
                raise KeyError("vol_scaled needs the trailing volatility")
            s = np.asarray(sigma, dtype=np.float64) * np.sqrt(h)
            with np.errstate(invalid="ignore", divide="ignore"):
                y = np.where(s > 0, y_raw / np.where(s > 0, s, 1.0), np.nan)
        else:                                            # pragma: no cover - config-checked
            raise ValueError(f"unknown transform {spec.transform!r}")
    y = np.where(np.isfinite(y), y, np.nan)
    rate = float(np.nanmean(y)) if spec.is_classification and np.isfinite(y).any() else None
    return TargetArrays(name=spec.name, horizon=h, task=spec.task, y=y,
                        y_raw=np.where(np.isfinite(y), y_raw, np.nan), positive_rate=rate)


# ---------------------------------------------------------------------------
# Leakage gate
# ---------------------------------------------------------------------------
def assert_no_leakage(names: list[str], registry: dict[str, dict[str, Any]], *,
                      x: np.ndarray | None = None, targets: list[np.ndarray] | None = None,
                      max_abs_rank_corr: float = 0.95) -> None:
    """Refuse outcome-like, unregistered, non-causal or suspiciously perfect columns."""
    problems = []
    for name in names:
        low = name.lower()
        if low in _META_COLUMNS:
            problems.append(f"{name}: time / row metadata is never a predictor")
        elif low.startswith(_FORBIDDEN_PREFIXES) or "shifted_back" in low or "_lead" in low:
            problems.append(f"{name}: named like an outcome or an offline quantity")
        elif name not in registry:
            problems.append(f"{name}: not a registered factory feature")
        elif not registry[name].get("live_safe", False):
            problems.append(f"{name}: registered as not live-safe")
    if problems:
        raise LeakageError("; ".join(problems))
    if x is not None and targets:
        for j, name in enumerate(names):
            xs = x[:, j]
            for y in targets:
                ok = np.isfinite(xs) & np.isfinite(y)
                if ok.sum() < 1000:
                    continue
                r = np.corrcoef(rank_scores(xs[ok]), rank_scores(y[ok]))[0, 1]
                if np.isfinite(r) and abs(r) > max_abs_rank_corr:
                    raise LeakageError(f"{name}: rank correlation {r:+.3f} with a target - "
                                       "a feature cannot know its own outcome (leakage canary)")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _manifests(scfg: FeatureSelectionConfig, timeframe: str
               ) -> tuple[dict[str, list[str]], dict[str, str], dict[str, str], dict[str, Any]]:
    base = scfg.results_path / timeframe / "manifests"
    names, ids, hashes, raw = {}, {}, {}, {}
    for path in sorted(base.glob("*.json")):
        m = load_manifest(path)                        # content hash + positions verified
        names[path.stem] = [f["name"] for f in m["features"]]
        ids[path.stem] = m["feature_set_id"]
        hashes[path.stem] = m["content_hash"]
        raw[path.stem] = m
    if not names:
        raise MLIntegrityError(f"{timeframe}: no Prompt #9 manifests under {base}")
    return names, ids, hashes, raw


def _check(timeframe: str, fmanifest: dict[str, Any], tmanifest: dict[str, Any],
           registry: dict[str, dict[str, Any]], manifests: dict[str, Any],
           fcfg: FeatureFactoryConfig) -> None:
    problems = []
    if fmanifest.get("timeframe") != timeframe:
        problems.append("matrix timeframe differs")
    if (fmanifest.get("dataset_lineage") or {}).get("partial"):
        problems.append("the feature matrix was built from a partial dataset")
    if fmanifest.get("timestamp_convention") != TIMESTAMP_CONVENTION:
        problems.append("timestamp convention differs")
    if fmanifest.get("source_feed") != fcfg.source_feed:
        problems.append("source feed differs")
    if (tmanifest.get("sources") or {}).get("bar_dataset_version") != \
            fmanifest.get("bar_dataset_version"):
        problems.append("targets and features were built from different bars")
    for set_name, m in manifests.items():
        if m.get("factory_version") != fmanifest.get("factory_version"):
            problems.append(f"manifest {set_name}: built on factory {m.get('factory_version')}, "
                            f"current {fmanifest.get('factory_version')}")
        for f in m["features"]:
            spec = registry.get(f["name"])
            if spec is None or spec.get("feature_version") != f["feature_version"] \
                    or not spec.get("live_safe", False):
                problems.append(f"manifest {set_name}: {f['name']} missing, stale or not "
                                "live-safe in the stored registry")
                break
    if problems:
        raise MLIntegrityError(f"{timeframe}: " + "; ".join(problems))


def load_ml_data(cfg: MLConfig, scfg: FeatureSelectionConfig, fcfg: FeatureFactoryConfig,
                 tcfg: TargetConfig, config: Config, regression: RegressionConfig,
                 timeframe: str, *, extra_horizons: tuple[int, ...] = ()) -> MLData:
    """Development features, outcome-safe targets and context of *timeframe* (rows < reserved)."""
    reserved = scfg.periods.reserved_test.start
    base = current_version_dir(fcfg.factory_path, timeframe)
    fmanifest = json.loads((base / "_manifest.json").read_text(encoding="utf-8"))
    reg_rows = json.loads((base / "registry.json").read_text(encoding="utf-8"))
    registry = {r["name"]: r for r in reg_rows}
    names, ids, hashes, raw_manifests = _manifests(scfg, timeframe)
    tptr = tcfg.targets_path / f"timeframe={timeframe}" / "CURRENT"
    tbase = tcfg.targets_path / f"timeframe={timeframe}" / f"version={tptr.read_text().strip()}"
    tmanifest = json.loads((tbase / "_manifest.json").read_text(encoding="utf-8"))
    _check(timeframe, fmanifest, tmanifest, registry, raw_manifests, fcfg)
    wanted = sorted({f for fs in names.values() for f in fs}
                    | {c for c in CONTEXT_FEATURES if c in registry})
    frame = read_years(base, wanted, reserved)
    stamps = frame["timestamp"]
    features = {c: frame[c].cast(pl.Float32).fill_null(np.nan).to_numpy() for c in wanted}
    del frame
    families = {"return", "abs_return", "realized_vol", "residual_reduction", "residual_shrinks"}
    horizons = sorted({h for spec in cfg.targets.values() for h in spec.horizons}
                      | set(extra_horizons))
    tcols = [f"target_{fam}_{h}" for fam in sorted(families) for h in horizons
             if f"target_{fam}_{h}" in tmanifest["targets"]]
    tframe = read_years(tbase, tcols, reserved)
    if not tframe["timestamp"].equals(stamps):
        raise MLIntegrityError(f"{timeframe}: targets and features are not on the same bars")
    n = int(stamps.len())
    targets_raw = {}
    for c in tcols:
        h = int(c.rsplit("_", 1)[1])
        v = np.array(tframe[c].cast(pl.Float64).fill_null(np.nan).to_numpy(), dtype=np.float64)
        v[max(0, n - h):] = np.nan                     # outcome windows crossing the reserved start
        targets_raw[c] = v
    del tframe
    store = RegressionFeatureStore(config, regression)
    rframe = store.load(timeframe, tcfg.regression_window,
                        ["timestamp", "residual", "trailing_volatility"],
                        before=reserved)
    rframe = rframe.filter(pl.col("timestamp") < pl.lit(_midnight(reserved))
                           .cast(rframe["timestamp"].dtype))           # cut at once
    if not rframe["timestamp"].equals(stamps):
        raise MLIntegrityError(f"{timeframe}: regression store and matrix differ in bars")
    eps = rframe["residual"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    sigma = rframe["trailing_volatility"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    del rframe
    versions = {"tick_dataset_version": fmanifest.get("tick_dataset_version"),
                "bar_dataset_version": fmanifest.get("bar_dataset_version"),
                "factory_version": fmanifest.get("factory_version"),
                "target_version": tmanifest.get("target_version"),
                "feature_sets": {k: {"id": ids[k], "hash": hashes[k]} for k in ids},
                "source_feed": fmanifest.get("source_feed")}
    data = MLData(timeframe=timeframe, timestamps=stamps, reserved_start=reserved,
                  features=features, manifests=names, manifest_ids=ids, manifest_hashes=hashes,
                  registry=registry, targets_raw=targets_raw, epsilon=eps, sigma=sigma,
                  versions=versions)
    data.context = build_context(data)
    return data


def build_context(data: MLData) -> dict[str, np.ndarray]:
    """Evaluation splits known at t: year, regime state, volatility quartile, spread, session."""
    n = data.n
    f = data.features
    ctx: dict[str, np.ndarray] = {"year": data.year().astype(np.int32)}
    probs = [f[p] for p in ("regime_p0", "regime_p1", "regime_p2") if p in f]
    if probs:
        stack = np.column_stack(probs).astype(np.float64)
        ok = np.isfinite(stack).all(axis=1)
        state = np.full(n, -1, dtype=np.int32)
        state[ok] = np.argmax(stack[ok], axis=1)
        ctx["regime_state"] = state
    if "log_rv_20" in f:
        per_day = max(1, int(round(n / max(1, data.timestamps.dt.date().n_unique()))))
        ctx["vol_quartile"] = causal_quantile_buckets(
            f["log_rv_20"].astype(np.float64), data.timestamps, 4,
            min_history=250 * per_day).astype(np.int32)
    if "spread_percentile" in f:
        sp = f["spread_percentile"]
        q = np.full(n, -1, dtype=np.int32)
        ok = np.isfinite(sp)
        q[ok] = np.clip((sp[ok] * 4).astype(np.int32), 0, 3)
        ctx["spread_quartile"] = q
    flags = [s for s in SESSION_ORDER if s in f]
    if flags:
        session = np.full(n, len(SESSION_ORDER), dtype=np.int32)       # "other"
        for code in range(len(SESSION_ORDER) - 1, -1, -1):
            name = SESSION_ORDER[code]
            if name in f:
                session[f[name] > 0.5] = code
        ctx["session"] = session
    return ctx


SESSION_LABELS = (*[s.removeprefix("session_") for s in SESSION_ORDER], "other")
