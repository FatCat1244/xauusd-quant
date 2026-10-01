r"""Predictive feature research, one timeframe at a time (Prompt #8).

Stages, each writing its tables when it finishes and a ``.done.json`` stamp
(data versions, research config, code) that lets a rerun skip it:

``factory``       integrity gate, the feature matrix and the target table (stored)
``quality``       missingness, distributions, drift (Steps 8-11)
``ic``            monthly moments of every feature x target -> pooled / yearly /
                  quarterly / era / recent / subsample / rolling IC, decay,
                  information half-life (Steps 21-27, 32-34, 62)
``conditioning``  IC within volatility quartiles, regime states and probabilities,
                  hour, session, weekday (Steps 28-31)
``mi``            copula mutual information against circular-shift and permutation
                  nulls, and conditional on volatility (Steps 35-37)
``redundancy``    correlation matrices, clusters, graph, feature-pair MI (Steps 38-42)
``deciles``       decile outcome curves, top-bottom spreads, their stability (46-49)
``interactions``  out-of-sample increments of registered interactions and
                  conditional effects (Steps 43-45)
``nulls``         circular-shift and permutation nulls, wrong alignments, the
                  look-ahead check, synthetic noise features (Steps 50-52)
``robustness``    parameter neighbourhoods, perturbations, scaling and
                  winsorisation research (Steps 12-13, 60-61)
``cost``          CPU / memory / update latency per family (Steps 64-66)
``pipeline``      the pipeline null controls: the factory run on a random walk,
                  shuffled returns and a block bootstrap (Step 53)

:mod:`.feature_reports` turns these tables into the multiple-testing counts,
the ledger, feature statuses, the candidate manifest and the plots. Nothing
here trains a predictive model, defines a trade or chooses anything by an
outcome; the small ridge fits of the interaction stage are diagnostics.

Memory: the target table (float32, column-major) and its rank scores stay in
memory; features are read from the stored matrix one column at a time. At 5m
(1.66M bars) that is ~1 GB held, ~2 GB at the peak of the null stage.
"""

from __future__ import annotations

import gc
import json
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import norm

from ..alpha.conditioning import (
    causal_quantile_buckets,
    condition_ics,
    condition_moments,
    month_codes,
)
from ..alpha.config import AlphaConfig
from ..alpha.decay import decay_summary, information_half_life
from ..alpha.information_coefficient import (
    MonthIndex,
    ic_with_errors,
    month_index,
    month_moments,
    rank_scores,
)
from ..alpha.interactions import interaction_increment
from ..alpha.mutual_information import (
    binned_codes,
    conditional_mi_from_codes,
    joint_counts,
    mi_from_joint,
    sentinel_codes,
)
from ..alpha.null_tests import (
    NullMatrices,
    circular_shift_null,
    pair_counts,
    permutation_null,
    shifted_ic,
    standardized_ranks,
    standardized_scores,
    synthetic_noise_features,
)
from ..alpha.stability import (
    alpha_health,
    consistency,
    era_ics,
    grouped_ics,
    rolling_ics,
    subsample_ics,
)
from ..data.resampler import parse_timeframe
from ..features.config import RegressionConfig
from ..features.factory import (
    build_feature_matrix,
    compute_families,
    current_version_dir,
    factory_gate,
    load_bar_series,
    null_bar_series,
    write_feature_matrix,
)
from ..features.factory_config import FeatureFactoryConfig
from ..features.families import BarSeries
from ..features.interactions import interaction_columns
from ..features.joins import EngineProvider
from ..features.manifest import digest, family_code_version
from ..features.quality import distribution_row, drift_rows, missingness_tables
from ..features.redundancy import (
    cluster_features,
    evenly_spaced_rows,
    feature_nmi_matrix,
    nonmonotone_pairs,
    pairwise_correlation,
    redundancy_groups,
)
from ..features.spectral_config import SpectralConfig
from ..features.store import RegressionFeatureStore
from ..features.transformations import scaling_variants, winsorize_causal
from ..features.wavelet_config import WaveletConfig
from ..models.config import OUConfig
from ..models.ornstein_uhlenbeck import rolling_ou
from ..targets.alignment import (
    TargetInputs,
    build_targets,
    check_alignment,
    load_targets,
    target_kind,
    target_manifest,
    target_version,
    write_targets,
)
from ..targets.config import TargetConfig
from ..utils.clock import utc_now_iso
from ..utils.config import Config
from ..utils.logging import get_logger
from ..utils.paths import atomic_write_text, ensure_dir
from .study_io import clean_json, code_fingerprint

__all__ = [
    "STAGES",
    "Configs",
    "ResearchContext",
    "ResearchData",
    "TargetBlocks",
    "compute_moments",
    "load_research_data",
    "run_timeframe",
    "write_table",
]

LOGGER = get_logger("research.feature_research")

STAGES: tuple[str, ...] = ("factory", "quality", "ic", "conditioning", "mi", "redundancy",
                           "deciles", "interactions", "nulls", "robustness", "cost",
                           "pipeline")
REPORT_VERSION = 1
_FACTORY_FAMILIES = ("returns", "volatility", "autocorrelation", "regression", "ou", "fft",
                     "wavelet", "regime", "microstructure", "time", "interaction")


@dataclass
class Configs:
    config: Config
    regression: RegressionConfig
    ou: OUConfig
    spectral: SpectralConfig
    wavelet: WaveletConfig
    research: Any
    features: FeatureFactoryConfig
    targets: TargetConfig
    alpha: AlphaConfig
    regime_features_path: Path


@dataclass
class ResearchContext:
    timeframe: str
    cfg: Configs
    out_dir: Path
    progress: Callable[[str], None] | None = None

    def log(self, message: str) -> None:
        LOGGER.info("[%s] %s", self.timeframe, message)
        if self.progress is not None:
            self.progress(f"[{self.timeframe}] {message}")

    def path(self, name: str) -> Path:
        return self.out_dir / name

    @property
    def bars_per_day(self) -> int:
        return self.cfg.features.bars_per_day(parse_timeframe(self.timeframe).total_seconds())


@dataclass
class TargetBlocks:
    """Every target column the IC uses: the stored targets plus the marginal decay columns.

    ``blocks`` groups column indices by family (one block per family, one per
    marginal curve); ``ranks`` holds the rank scores of every column
    (float32, column-major, so a column is a contiguous view).
    """

    names: list[str]
    blocks: list[tuple[str, list[int]]]
    horizons: dict[str, int]
    base: np.ndarray                  # the stored targets, (n, T0) column-major
    extra: np.ndarray                 # marginal columns, (n, T - T0) column-major
    ranks: np.ndarray                 # (n, T) column-major

    def column(self, j: int) -> np.ndarray:
        t0 = self.base.shape[1]
        return self.base[:, j] if j < t0 else self.extra[:, j - t0]

    def columns(self, idx: list[int]) -> list[np.ndarray]:
        return [self.column(j) for j in idx]

    def rank_columns(self, idx: list[int]) -> list[np.ndarray]:
        return [self.ranks[:, j] for j in idx]

    def index(self, name: str) -> int:
        return self.names.index(name)


@dataclass
class ResearchData:
    """What every stage reads: matrix columns (lazily, one at a time), targets, months."""

    timestamps: pl.Series
    feature_names: list[str]
    registry: list[dict[str, Any]]
    factory_manifest: dict[str, Any]
    target_names: list[str]
    target_manifest: dict[str, Any]
    months: MonthIndex
    matrix_dir: Path
    targets_loader: Callable[[], np.ndarray]
    _targets: np.ndarray | None = None
    _blocks: TargetBlocks | None = None
    _cache: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.timestamps.len())

    @property
    def targets(self) -> np.ndarray:
        if self._targets is None:
            self._targets = self.targets_loader()
        return self._targets

    def target(self, name: str) -> np.ndarray:
        return self.targets[:, self.target_names.index(name)]

    def drop_targets(self) -> None:
        self._targets = None
        self._blocks = None

    def blocks(self, ctx: ResearchContext) -> TargetBlocks:
        if self._blocks is None:
            self._blocks = target_blocks(ctx, self)
        return self._blocks

    def drop_blocks(self) -> None:
        self._blocks = None

    def feature(self, name: str) -> np.ndarray:
        if name not in self._cache:
            files = sorted(self.matrix_dir.glob("year=*/part-0.parquet"))
            col = pl.read_parquet(files, columns=["timestamp", name]).sort("timestamp")
            self._cache[name] = np.array(col[name].cast(pl.Float32).fill_null(np.nan)
                                         .to_numpy(), dtype=np.float32)
        return self._cache[name]

    def release(self) -> None:
        self._cache.clear()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def write_table(frame: pl.DataFrame | None, path: Path, *, csv: bool = True) -> None:
    """Parquet (and CSV unless *csv* is False); NaN -> null, Null columns as Float64,
    lists as JSON. A statistic that is undefined is missing, never a number that
    compares false (an undefined IC must not sort as the largest)."""
    if frame is None or frame.is_empty():
        return
    nulls = [c for c, d in frame.schema.items() if d == pl.Null]
    if nulls:
        frame = frame.with_columns(pl.col(nulls).cast(pl.Float64))
    frame = frame.with_columns(pl.col(pl.Float32, pl.Float64).fill_nan(None))
    ensure_dir(path.parent)
    frame.write_parquet(path.with_suffix(".parquet"))
    if csv:
        listy = [c for c, d in frame.schema.items() if isinstance(d, (pl.List, pl.Struct,
                                                                      pl.Array))]
        if listy:
            frame = frame.with_columns([pl.col(c).map_elements(
                lambda v: json.dumps(clean_json(v.to_list() if hasattr(v, "to_list") else v),
                                     default=str), return_dtype=pl.Utf8) for c in listy])
        frame.write_csv(path.with_suffix(".csv"))


def write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(clean_json(payload), indent=1, default=str) + "\n")


def _code_version(*rels: str) -> str:
    base = Path(__file__).resolve().parent.parent
    return code_fingerprint(base / rel for rel in rels)


_STAGE_CODE: dict[str, tuple[str, ...]] = {
    "quality": ("features/quality.py",),
    "ic": ("alpha/information_coefficient.py", "alpha/stability.py", "alpha/decay.py"),
    "conditioning": ("alpha/conditioning.py", "alpha/information_coefficient.py"),
    "mi": ("alpha/mutual_information.py", "alpha/conditioning.py"),
    "redundancy": ("features/redundancy.py", "alpha/mutual_information.py"),
    "deciles": (),
    "interactions": ("alpha/interactions.py", "alpha/conditioning.py",
                     "research/wavelet_predictiveness.py"),
    "nulls": ("alpha/null_tests.py", "alpha/information_coefficient.py", "alpha/stability.py"),
    "robustness": ("features/transformations.py", "features/interactions.py"),
    "cost": ("features/families.py", "features/factory.py"),
    "pipeline": ("features/families.py", "features/factory.py", "features/joins.py",
                 "targets/alignment.py", "alpha/information_coefficient.py"),
}


def _stamp(ctx: ResearchContext, data: ResearchData, stage: str) -> dict[str, Any]:
    code = (*_STAGE_CODE.get(stage, ()), "research/feature_research.py")
    return {"report_version": REPORT_VERSION, "stage": stage,
            "factory_version": data.factory_manifest.get("factory_version"),
            "target_version": data.target_manifest.get("target_version"),
            "alpha_config": ctx.cfg.alpha.fingerprint(), "code": _code_version(*code)}


def _reusable(path: Path, stamp: dict[str, Any]) -> bool:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("stamp") == stamp
    except (OSError, ValueError):
        return False


def _mark(path: Path, stamp: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
    write_json(path, {"stamp": stamp, "generated_utc": utc_now_iso(), **(extra or {})})


def _begin(ctx: ResearchContext, data: ResearchData, stage: str
           ) -> tuple[dict[str, Any], Path] | None:
    """(stamp, marker) of a stage that must run, or None when its outputs are current."""
    stamp = _stamp(ctx, data, stage)
    marker = ctx.path(f"{stage}/.done.json")
    if _reusable(marker, stamp):
        ctx.log(f"{stage}: current")
        return None
    ensure_dir(marker.parent)
    ctx.log(f"{stage}: running")
    return stamp, marker


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _horizon(target: str) -> int:
    return int(target.rsplit("_", 1)[1].removeprefix("lag"))


def _kind(target: str) -> str:
    if target.startswith("marginal_"):
        return "direction" if target.startswith("marginal_return") else "magnitude"
    return target_kind(target)


def _labels(values: list[str], codes: np.ndarray) -> pl.Series:
    """values[codes] as a string column, without a NumPy array of Python strings."""
    return pl.Series(values, dtype=pl.Utf8).gather(pl.Series(np.asarray(codes, dtype=np.int64)))


def _feature_release(data: ResearchData, name: str) -> np.ndarray:
    values = data.feature(name)
    data.release()
    return values


# ---------------------------------------------------------------------------
# Stage: factory (matrix + targets)
# ---------------------------------------------------------------------------
def _regime_version(cfg: Configs, timeframe: str) -> str | None:
    r = cfg.features.families.regime
    path = (cfg.regime_features_path / f"timeframe={timeframe}" / f"model={r.model}"
            / f"k={r.states}" / f"scheme={r.scheme}" / "_manifest.json")
    try:
        version = json.loads(path.read_text(encoding="utf-8")).get("regime_model_version")
    except (OSError, ValueError):
        return None
    return str(version) if version is not None else None


def _build_key(gate: dict[str, Any], cfg: Configs, timeframe: str) -> str:
    """Everything the stored matrix depends on; a current matrix is reused, never rebuilt."""
    lineage = gate["lineage"]
    stored = {k: v.get("generated_utc") for k, v in (gate.get("stored_sets") or {}).items()}
    return digest({"tick": lineage.get("tick_dataset_version"),
                   "bars": lineage.get("bar_dataset_version"), "stored": stored,
                   "regime": _regime_version(cfg, timeframe),
                   "features": cfg.features.fingerprint(),
                   "regression": cfg.regression.fingerprint(), "ou": cfg.ou.fingerprint(),
                   "spectral": cfg.spectral.engine_fingerprint(),
                   "wavelet": cfg.wavelet.engine_fingerprint(),
                   "code": {f: family_code_version(f) for f in _FACTORY_FAMILIES},
                   "factory": _code_version("features/factory.py", "features/registry.py",
                                            "features/joins.py", "features/interactions.py",
                                            "features/manifest.py")}, 12)


def stage_factory(ctx: ResearchContext, *, resume: bool = True) -> dict[str, Any]:
    """Gate, then the feature matrix and the targets (reused when every version matches)."""
    cfg = ctx.cfg
    tf = ctx.timeframe
    bars = load_bar_series(cfg.config, tf)
    gate = factory_gate(cfg.config, cfg.regression, cfg.spectral, cfg.wavelet, cfg.features,
                        cfg.regime_features_path, tf, bars=bars)
    write_json(ctx.path("integrity_gate.json"), gate)
    if not gate["passed"]:
        raise RuntimeError(f"{tf}: factory gate failed - " + "; ".join(gate["problems"]))
    key = _build_key(gate, cfg, tf)
    root = cfg.features.factory_path
    manifest = None
    if resume:
        try:
            base = current_version_dir(root, tf)
            old = json.loads((base / "_manifest.json").read_text(encoding="utf-8"))
            if old.get("build_key") == key:
                manifest = old
                ctx.log(f"feature matrix: current ({old.get('factory_version')})")
        except (OSError, ValueError):
            manifest = None
    if manifest is None:
        ctx.log("building the feature matrix")
        result = build_feature_matrix(cfg.config, cfg.regression, cfg.ou, cfg.spectral,
                                      cfg.wavelet, cfg.research, cfg.features,
                                      cfg.regime_features_path, tf, gate=gate, bars=bars)
        result.manifest["build_key"] = key
        write_feature_matrix(result, root)
        manifest = result.manifest
        ctx.log(f"feature matrix: {len(result.registry)} features x {result.frame.height:,} "
                f"bars in {manifest['seconds']:.0f}s ({manifest['factory_version']})")
        del result
        gc.collect()
    tmanifest = _targets(ctx, bars, gate, resume=resume)
    return {"factory": manifest, "targets": tmanifest, "gate": gate}


def _targets(ctx: ResearchContext, bars: BarSeries, gate: dict[str, Any], *,
             resume: bool) -> dict[str, Any]:
    cfg = ctx.cfg
    tcfg = cfg.targets
    tf = ctx.timeframe
    store = RegressionFeatureStore(cfg.config, cfg.regression)
    reg_manifest = store.manifest(tf) or {}
    reg_version = ((reg_manifest.get("windows") or {}).get(str(tcfg.regression_window))
                   or {}).get("feature_version")
    sources = {"timeframe": tf,
               "bar_dataset_version": gate["lineage"].get("bar_dataset_version"),
               "tick_dataset_version": gate["lineage"].get("tick_dataset_version"),
               "regression_feature_version": reg_version, "ou_config": cfg.ou.fingerprint(),
               "ou_window": tcfg.ou_window, "rows": bars.size}
    version = target_version(tcfg, sources)
    if resume:
        try:
            _, old = load_targets(tcfg.targets_path, tf, columns=["timestamp"])
            if old.get("target_version") == version:
                ctx.log(f"targets: current ({version})")
                return old
        except (OSError, ValueError):
            pass
    started = time.perf_counter()
    frame = store.load(tf, tcfg.regression_window,
                       ["timestamp", "residual", "trailing_volatility"])
    if not frame["timestamp"].equals(bars.timestamps):
        raise RuntimeError(f"{tf}: regression store and bars differ in timestamps")
    residual = frame["residual"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    sigma = frame["trailing_volatility"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    del frame
    fit = rolling_ou(residual, tcfg.ou_window, rules=cfg.ou.validity_rules(),
                     chunk_rows=cfg.ou.numerical.chunk_rows)
    mu = np.where(fit.mapping.valid, fit.mapping.mu, np.nan)
    del fit
    gc.collect()
    table = build_targets(TargetInputs(bars=bars, residual=residual, sigma=sigma, ou_mu=mu),
                          tcfg)
    manifest = target_manifest(table, tcfg, sources, started)
    write_targets(table, manifest, tcfg.targets_path, tf)
    ctx.log(f"targets: {len(manifest['targets'])} columns in {manifest['seconds']:.0f}s "
            f"({manifest['target_version']})")
    return manifest


def load_research_data(ctx: ResearchContext) -> ResearchData:
    """The current matrix of the timeframe and a lazy loader of its targets (checked aligned)."""
    cfg = ctx.cfg
    tf = ctx.timeframe
    base = current_version_dir(cfg.features.factory_path, tf)
    manifest = json.loads((base / "_manifest.json").read_text(encoding="utf-8"))
    registry = json.loads((base / "registry.json").read_text(encoding="utf-8"))
    files = sorted(base.glob("year=*/part-0.parquet"))
    stamps = pl.read_parquet(files, columns=["timestamp"]).sort("timestamp")
    if stamps.height != manifest.get("rows"):
        raise RuntimeError(f"{tf}: feature matrix rows {stamps.height} != manifest "
                           f"{manifest.get('rows')}")
    names = cfg.targets.columns()
    _, tmanifest = load_targets(cfg.targets.targets_path, tf, columns=["timestamp"])
    if (tmanifest.get("sources") or {}).get("bar_dataset_version") != manifest.get(
            "bar_dataset_version"):
        raise RuntimeError(f"{tf}: targets and features were built from different bars - "
                           "rerun the factory stage")

    def loader() -> np.ndarray:
        tframe, _ = load_targets(cfg.targets.targets_path, tf, columns=names)
        check_alignment(stamps, tframe)
        arr = np.empty((tframe.height, len(names)), dtype=np.float32, order="F")
        for j, c in enumerate(names):
            arr[:, j] = tframe[c].cast(pl.Float32).fill_null(np.nan).to_numpy()
        return arr

    return ResearchData(timestamps=stamps["timestamp"],
                        feature_names=[r["name"] for r in registry], registry=registry,
                        factory_manifest=manifest, target_names=names,
                        target_manifest=tmanifest, months=month_index(stamps["timestamp"]),
                        matrix_dir=base, targets_loader=loader)


def _regime_state(data: ResearchData, k: int) -> np.ndarray:
    """The most probable filtered state at each bar (-1 where the filter has no output)."""
    probs = [data.feature(f"regime_p{j}") for j in range(k)
             if f"regime_p{j}" in data.feature_names]
    out = np.full(data.n, -1, dtype=np.int64)
    if len(probs) == k:
        stack = np.column_stack(probs).astype(np.float64)
        ok = np.isfinite(stack).all(axis=1)
        out[ok] = np.argmax(stack[ok], axis=1)
    return out


# ---------------------------------------------------------------------------
# Stage: quality
# ---------------------------------------------------------------------------
def _longest_run(missing: np.ndarray) -> int:
    if not missing.any():
        return 0
    padded = np.concatenate(([False], missing, [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return int((edges[1::2] - edges[::2]).max())


def stage_quality(ctx: ResearchContext, data: ResearchData) -> None:
    begun = _begin(ctx, data, "quality")
    if begun is None:
        return
    stamp, marker = begun
    years = data.timestamps.dt.year().to_numpy()
    state = _regime_state(data, ctx.cfg.features.families.regime.states)
    data.release()
    registry = {r["name"]: r for r in data.registry}
    dist_rows: list[dict[str, Any]] = []
    drift: list[dict[str, Any]] = []
    policy: list[dict[str, Any]] = []
    overall: list[pl.DataFrame] = []
    by_year: list[pl.DataFrame] = []
    by_month: list[pl.DataFrame] = []
    chunk: dict[str, np.ndarray] = {}

    def flush() -> None:
        frame = pl.DataFrame({"timestamp": data.timestamps, **chunk})
        o, y, m = missingness_tables(frame, list(chunk), state)
        overall.append(o)
        by_year.append(y)
        by_month.append(m)
        chunk.clear()

    for i, name in enumerate(data.feature_names):
        x = _feature_release(data, name)
        dist_rows.append(distribution_row(name, x, years))
        drift.extend(drift_rows(name, x, years, seed=ctx.cfg.alpha.seed + i))
        missing = ~np.isfinite(x)
        spec = registry.get(name, {})
        first = int(np.argmax(~missing)) if (~missing).any() else x.size
        after = missing[first:]
        policy.append({"feature": name, "family": spec.get("family"),
                       "min_history_bars": int(spec.get("min_history") or 0),
                       "first_defined_bar": first,
                       "missing_share": float(missing.mean()),
                       "missing_share_after_first_value": float(after.mean())
                       if after.size else None,
                       "longest_missing_run_after_first_value": _longest_run(after),
                       "registry_policy": spec.get("missing_policy")})
        chunk[name] = x
        if len(chunk) >= 32:
            flush()
    if chunk:
        flush()
    q = "quality"
    write_table(pl.concat(overall, how="diagonal_relaxed"), ctx.path(f"{q}/missingness"))
    write_table(pl.concat(by_year, how="diagonal_relaxed"),
                ctx.path(f"{q}/missingness_by_year"))
    write_table(pl.concat(by_month, how="diagonal_relaxed"),
                ctx.path(f"{q}/missingness_by_month"), csv=False)
    write_table(pl.DataFrame(policy, infer_schema_length=None), ctx.path(f"{q}/missing_policy"))
    write_table(pl.DataFrame(dist_rows, infer_schema_length=None),
                ctx.path(f"{q}/feature_distributions"))
    drift_frame = pl.DataFrame(drift, infer_schema_length=None)
    write_table(drift_frame, ctx.path(f"{q}/feature_drift_yearly"))
    if not drift_frame.is_empty():
        summary = drift_frame.group_by("feature").agg(
            pl.col("psi").max().alias("max_yearly_psi"),
            pl.col("psi").median().alias("median_yearly_psi"),
            pl.col("ks").max().alias("max_yearly_ks"),
            pl.col("wasserstein_sd").max().alias("max_yearly_wasserstein_sd"),
            pl.col("median_shift_iqr").abs().max().alias("max_abs_median_shift_iqr"),
        ).with_columns((pl.col("max_yearly_psi") > 0.25).alias("psi_above_0_25")
                       ).sort("max_yearly_psi", descending=True)
        write_table(summary, ctx.path(f"{q}/feature_drift_summary"))
    _period_medians(ctx, data)
    _mark(marker, stamp)


def _period_medians(ctx: ResearchContext, data: ResearchData) -> None:
    """Quarterly and era medians of each feature in pooled-IQR units (drift through time)."""
    ts = data.timestamps
    quarter = (ts.dt.year().cast(pl.Int32) * 10 + ts.dt.quarter().cast(pl.Int32)).to_numpy()
    n = data.n
    eras = ctx.cfg.alpha.ic.eras
    era = np.minimum((np.arange(n) * eras) // n, eras - 1)
    frames = []
    for name in data.feature_names:
        x = _feature_release(data, name).astype(np.float64)
        ok = np.isfinite(x)
        if ok.sum() < 100:
            continue
        med = float(np.median(x[ok]))
        q25, q75 = np.quantile(x[ok], [0.25, 0.75])
        iqr = float(q75 - q25) or 1.0
        frame = pl.DataFrame({"q": quarter[ok], "e": era[ok], "x": x[ok]})
        for key, kind in (("q", "quarter"), ("e", "era")):
            grouped = frame.group_by(key).agg(pl.col("x").median().alias("m"),
                                              pl.len().alias("n")).filter(pl.col("n") > 30)
            period = ((pl.col(key) // 10).cast(pl.Utf8) + "Q" + (pl.col(key) % 10).cast(pl.Utf8)
                      if kind == "quarter" else "era_" + (pl.col(key) + 1).cast(pl.Utf8))
            frames.append(grouped.select(
                pl.lit(name).alias("feature"), period.alias("period"), pl.lit(kind).alias("kind"),
                ((pl.col("m") - med) / iqr).alias("median_shift_iqr"), pl.col("n")))
    if frames:
        write_table(pl.concat(frames, how="diagonal_relaxed").sort(["feature", "kind", "period"]),
                    ctx.path("quality/feature_period_medians"), csv=False)


# ---------------------------------------------------------------------------
# Stage: IC (monthly moments)
# ---------------------------------------------------------------------------
def target_blocks(ctx: ResearchContext, data: ResearchData, *,
                  base: np.ndarray | None = None, marginal: bool = True) -> TargetBlocks:
    """The stored targets (grouped by family) plus marginal one-bar columns for the decay."""
    tcfg = ctx.cfg.targets
    base = data.targets if base is None else base
    names = list(data.target_names)
    blocks: list[tuple[str, list[int]]] = []
    horizons: dict[str, int] = {}
    for family in tcfg.families:
        idx = [names.index(f"target_{family}_{h}") for h in tcfg.horizons]
        blocks.append((family, idx))
        for h in tcfg.horizons:
            horizons[f"target_{family}_{h}"] = h
    extra_cols: list[np.ndarray] = []
    if marginal:
        for family, one_name in (("return", "target_return_1"),
                                 ("abs_return", "target_abs_return_1")):
            if one_name not in names:
                continue
            one = base[:, data.target_names.index(one_name)]
            idx = []
            for k in ctx.cfg.alpha.decay.marginal_lags:
                shifted = np.full(one.size, np.nan, dtype=np.float32)
                if k - 1 < one.size:
                    shifted[: one.size - (k - 1)] = one[k - 1:]
                name = f"marginal_{family}_lag{k}"
                names.append(name)
                extra_cols.append(shifted)
                horizons[name] = k
                idx.append(len(names) - 1)
            blocks.append((f"marginal_{family}", idx))
    n = base.shape[0]
    extra = np.empty((n, len(extra_cols)), dtype=np.float32, order="F")
    for j, col in enumerate(extra_cols):
        extra[:, j] = col
    del extra_cols
    ranks = np.empty((n, len(names)), dtype=np.float32, order="F")
    t0 = base.shape[1]
    for j in range(len(names)):
        ranks[:, j] = rank_scores(base[:, j] if j < t0 else extra[:, j - t0])
    return TargetBlocks(names=names, blocks=blocks, horizons=horizons, base=base, extra=extra,
                        ranks=ranks)


def compute_moments(features: list[str], get: Callable[[str], np.ndarray], tb: TargetBlocks,
                    months: MonthIndex, *, log: Callable[[str], None] | None = None,
                    batch: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """(features, targets, months, 6) raw and rank moments of every feature x target column.

    Features are read *batch* at a time (float32) and scored against every
    target column with one matrix product per month (:func:`month_moments`).
    """
    raw = np.zeros((len(features), len(tb.names), months.size, 6))
    rnk = np.zeros_like(raw)
    columns = list(range(len(tb.names)))
    y_raw = tb.columns(columns)
    y_rank = tb.rank_columns(columns)
    started = time.perf_counter()
    for b0 in range(0, len(features), batch):
        names = features[b0:b0 + batch]
        xs = [np.asarray(get(name), dtype=np.float32) for name in names]
        xr = [rank_scores(x).astype(np.float32) for x in xs]
        raw[b0:b0 + len(names)] = month_moments(xs, y_raw, months.starts)
        rnk[b0:b0 + len(names)] = month_moments(xr, y_rank, months.starts)
        del xs, xr
        if log is not None:
            log(f"moments: {b0 + len(names)}/{len(features)} features "
                f"({time.perf_counter() - started:.0f}s)")
    return raw, rnk


def _grid(features: list[str], targets: list[str], keep: np.ndarray
          ) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    """(feature index, target index, base columns) of the kept cells of an (F, T) grid."""
    fi, ti = np.nonzero(keep)
    base = pl.DataFrame({"feature": _labels(features, fi), "target": _labels(targets, ti)})
    return fi, ti, base


def _long(features: list[str], targets: list[str], labels: list[str], ic: np.ndarray,
          n: np.ndarray, method: str, label: str) -> pl.DataFrame:
    gi, fi, ti = np.nonzero(np.isfinite(ic))
    return pl.DataFrame({"feature": _labels(features, fi), "target": _labels(targets, ti),
                         label: _labels(labels, gi)}).with_columns(
        pl.lit(method).alias("method"), pl.Series("ic", ic[gi, fi, ti]),
        pl.Series("n", n[gi, fi, ti]))


def ic_tables(ctx: ResearchContext, features: list[str], tb: TargetBlocks, months: MonthIndex,
              raw: np.ndarray, rnk: np.ndarray) -> dict[str, pl.DataFrame]:
    """Pooled, yearly, quarterly, consistency (era / recent / subsample / health), rolling."""
    a = ctx.cfg.alpha
    names = tb.names
    horizon = np.array([tb.horizons.get(t, -1) for t in names])
    kinds = [_kind(t) for t in names]
    recent_mask = np.array([lab >= a.ic.recent_start[:7] for lab in months.labels])
    q_codes = months.years * 10 + months.quarters
    hac = a.ic.hac_lag_months
    out: dict[str, list[pl.DataFrame]] = {"ic": [], "yearly": [], "quarterly": [],
                                          "consistency": [], "rolling": []}
    for method, mom in (("pearson", raw), ("spearman", rnk)):
        m = np.moveaxis(mom, 2, 0)                                   # (M, F, T, 6)
        stats = ic_with_errors(m, hac_lags=hac)
        keep = stats["n"] >= a.ic.min_observations
        fi, ti, base = _grid(features, names, keep)
        cols = {k: np.asarray(stats[k], dtype=np.float64)[fi, ti]
                for k in ("ic", "n", "se", "se_iid", "z", "p", "ci_low", "ci_high", "months")}
        out["ic"].append(base.with_columns(
            pl.Series("horizon", horizon[ti]), _labels(kinds, ti).alias("kind"),
            pl.lit(method).alias("method"), *[pl.Series(k, v) for k, v in cols.items()]))
        yl, yic, yn = grouped_ics(m, months.years, min_obs=a.ic.min_year_observations)
        out["yearly"].append(_long(features, names, [str(int(v)) for v in yl], yic, yn, method,
                                   "year"))
        if method == a.ic.headline:
            ql, qic, qn = grouped_ics(m, q_codes, min_obs=a.ic.min_quarter_observations)
            out["quarterly"].append(_long(features, names,
                                          [f"{int(q) // 10}Q{int(q) % 10}" for q in ql],
                                          qic, qn, method, "quarter"))
        recent = ic_with_errors(m, select=recent_mask, hac_lags=hac)
        recent_ic = np.where(recent["n"] >= a.ic.min_year_observations, recent["ic"], np.nan)
        cons = consistency(yic, stats["ic"], recent_ic)
        eras = era_ics(m, a.ic.eras, min_obs=a.ic.min_year_observations)
        subs = subsample_ics(m, months, min_obs=a.ic.min_year_observations)
        roll = rolling_ics(m, window=a.rolling.window_months, min_months=a.rolling.min_months,
                           min_obs=a.ic.min_year_observations, hac_lags=hac)
        health = alpha_health(roll["ic"], roll["z"], stats["ic"])
        pooled_sign = np.sign(stats["ic"])
        flips = np.where(np.isfinite(eras), np.sign(eras) != pooled_sign[None], False).sum(axis=0)
        ccols: dict[str, np.ndarray] = {"pooled_ic": stats["ic"][fi, ti],
                                        "recent_n": recent["n"][fi, ti]}
        ccols.update({k: np.asarray(v, dtype=np.float64)[fi, ti] for k, v in cons.items()})
        ccols.update({f"era_{e + 1}_ic": eras[e][fi, ti] for e in range(eras.shape[0])})
        ccols["era_sign_flips"] = flips[fi, ti]
        ccols.update({f"{k}_ic": v[fi, ti] for k, v in subs.items()})
        ccols.update({k: np.asarray(v, dtype=np.float64)[fi, ti] for k, v in health.items()})
        out["consistency"].append(base.with_columns(
            pl.Series("horizon", horizon[ti]), _labels(kinds, ti).alias("kind"),
            pl.lit(method).alias("method"),
            *[pl.Series(k, np.asarray(v, dtype=np.float64)) for k, v in ccols.items()]))
        # both methods: Pearson is strictly causal (past pairs, past values); the rank
        # version uses full-sample marginal ranks - a monotone transform of each column,
        # no later pair enters - and is the headline for its robustness to tails
        main = np.array([t.startswith("target_") for t in names])
        ric = np.where(main[None, None, :], roll["ic"], np.nan)
        mi_, fi2, ti2 = np.nonzero(np.isfinite(ric))
        out["rolling"].append(pl.DataFrame({
            "feature": _labels(features, fi2), "target": _labels(names, ti2),
            "month": _labels(months.labels, mi_)}).with_columns(
            pl.lit(method).alias("method"),
            pl.Series("rolling_ic", ric[mi_, fi2, ti2]),
            pl.Series("rolling_z", roll["z"][mi_, fi2, ti2])))
    return {k: pl.concat(v, how="diagonal_relaxed") if v else pl.DataFrame()
            for k, v in out.items()}


def decay_table(ctx: ResearchContext, ic: pl.DataFrame) -> pl.DataFrame:
    """Horizon and marginal decay curves per feature x target family x method (Steps 23-24)."""
    a = ctx.cfg.alpha
    if ic.is_empty():
        return pl.DataFrame()
    frame = ic.with_columns(pl.col("target").str.replace(r"_(lag)?\d+$", "").alias("family"))
    rows = []
    for (feature, family, method), part in frame.group_by(["feature", "family", "method"]):
        part = part.sort("horizon")
        h = part["horizon"].to_numpy()
        icv = part["ic"].to_numpy()
        se = part["se"].to_numpy()
        curve = "marginal" if str(family).startswith("marginal_") else "horizon"
        row: dict[str, Any] = {"feature": feature, "target_family": family, "method": method,
                               "curve": curve, "horizons": [int(v) for v in h],
                               "ics": [float(v) for v in icv]}
        row.update(decay_summary(h, icv, se, zero_z=a.decay.zero_z))
        if curve == "marginal":
            row.update(information_half_life(h, icv, se, zero_z=a.decay.zero_z,
                                             min_points=a.decay.min_fit_points))
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None).sort(["feature", "target_family",
                                                               "method"])


def stage_ic(ctx: ResearchContext, data: ResearchData) -> None:
    begun = _begin(ctx, data, "ic")
    if begun is None:
        return
    stamp, marker = begun
    tb = data.blocks(ctx)
    feats = data.feature_names
    raw, rnk = compute_moments(feats, lambda n: _feature_release(data, n), tb, data.months,
                               log=ctx.log)
    cache = ensure_dir(ctx.path("ic/cache"))
    np.save(cache / "moments_raw.npy", raw)
    np.save(cache / "moments_rank.npy", rnk)
    write_json(cache / "axes.json", {"features": feats, "targets": tb.names,
                                     "months": data.months.labels, "horizons": tb.horizons})
    ctx.log("ic: tables")
    tables = ic_tables(ctx, feats, tb, data.months, raw, rnk)
    del raw, rnk
    gc.collect()
    write_table(tables["ic"], ctx.path("ic/feature_ic"))
    write_table(tables["yearly"], ctx.path("ic/yearly_ic"), csv=False)
    write_table(tables["quarterly"], ctx.path("ic/quarterly_ic"), csv=False)
    write_table(tables["consistency"], ctx.path("ic/ic_consistency"))
    write_table(tables["rolling"], ctx.path("ic/rolling_ic"), csv=False)
    write_table(decay_table(ctx, tables["ic"]), ctx.path("ic/alpha_decay"))
    _mark(marker, stamp)


# ---------------------------------------------------------------------------
# Stage: conditioning
# ---------------------------------------------------------------------------
def conditions(ctx: ResearchContext, data: ResearchData
               ) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Every conditioning partition: name -> (code per bar, -1 = left out; labels)."""
    a = ctx.cfg.alpha.conditioning
    ts = data.timestamps
    out: dict[str, tuple[np.ndarray, list[str]]] = {}
    if a.volatility_feature in data.feature_names:
        vol = data.feature(a.volatility_feature).astype(np.float64)
        codes = causal_quantile_buckets(vol, ts, a.volatility_buckets,
                                        min_history=250 * ctx.bars_per_day)
        out["volatility_quartile"] = (codes, [f"Q{q + 1}" for q in range(a.volatility_buckets)])
    k = ctx.cfg.features.families.regime.states
    if all(f"regime_p{j}" in data.feature_names for j in range(k)):
        out["regime_state"] = (_regime_state(data, k), [f"state_{j}" for j in range(k)])
        for j in range(k):
            p = data.feature(f"regime_p{j}").astype(np.float64)
            ok = np.isfinite(p)
            for th in a.regime_thresholds:
                code = np.full(data.n, -1, dtype=np.int64)
                code[ok] = (p[ok] > th).astype(np.int64)
                out[f"regime_p{j}_gt_{th:g}"] = (code, [f"p{j}<={th:g}", f"p{j}>{th:g}"])
    if "hour" in a.intraday:
        out["hour"] = (ts.dt.hour().cast(pl.Int64).to_numpy(), [f"{h:02d}" for h in range(24)])
    if "weekday" in a.intraday:
        out["weekday"] = (ts.dt.weekday().cast(pl.Int64).to_numpy() - 1,
                          ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    if "session" in a.intraday and ctx.cfg.research is not None:
        from .intraday import assign_sessions

        labels = pl.DataFrame({"ts": ts}).select(
            assign_sessions(pl.col("ts"), ctx.cfg.research).alias("s"))["s"]
        names = ["off_hours", "asia", "london", "london_ny_overlap", "new_york"]
        code = np.full(data.n, -1, dtype=np.int64)
        for i, s in enumerate(names):
            code[(labels == s).to_numpy()] = i
        out["session"] = (code, names)
    data.release()
    return out


_CONDITION_GROUPS = {"volatility_conditioning": ("volatility_quartile",),
                     "intraday_ic": ("hour", "session", "weekday")}


@dataclass
class _Partition:
    """Rows of one conditioning partition sorted by (condition, month), excluded rows dropped."""

    order: np.ndarray                 # row indices, int64
    starts: np.ndarray                # first sorted row of every (condition, month) segment
    labels: list[str]


def _partition(codes: np.ndarray, labels: list[str], mcode: np.ndarray,
               n_months: int) -> _Partition:
    c = len(labels)
    key = np.where(codes >= 0, codes * n_months + mcode, c * n_months)
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    kept = int(np.searchsorted(sorted_key, c * n_months))
    starts = np.searchsorted(sorted_key[:kept], np.arange(c * n_months)).astype(np.int64)
    return _Partition(order=order[:kept], starts=starts, labels=labels)


def stage_conditioning(ctx: ResearchContext, data: ResearchData) -> None:
    begun = _begin(ctx, data, "conditioning")
    if begun is None:
        return
    stamp, marker = begun
    a = ctx.cfg.alpha
    tb = data.blocks(ctx)
    targets = [f"target_{f}_{h}" for f in a.conditioning.target_families
               for h in a.conditioning.horizons if f"target_{f}_{h}" in tb.names]
    y_cols = tb.rank_columns([tb.index(t) for t in targets])
    months = data.months.size
    mcode = month_codes(data.months.starts, data.n)
    plans = {name: _partition(codes, labels, mcode, months)
             for name, (codes, labels) in conditions(ctx, data).items()}
    horizons = np.array([_horizon(t) for t in targets])
    kinds = [target_kind(t) for t in targets]
    frames = []
    started = time.perf_counter()
    batch = 32
    feats = data.feature_names
    for b0 in range(0, len(feats), batch):
        names = feats[b0:b0 + batch]
        xr = [rank_scores(_feature_release(data, f)).astype(np.float32) for f in names]
        for cname, plan in plans.items():
            if plan.order.size == 0:
                continue
            mom = month_moments([x[plan.order] for x in xr], [y[plan.order] for y in y_cols],
                                plan.starts)
            mom = mom.reshape(len(names), len(targets), len(plan.labels), months, 6)
            for c, label in enumerate(plan.labels):
                stats = ic_with_errors(np.moveaxis(mom[:, :, c], 2, 0),
                                       hac_lags=a.ic.hac_lag_months)
                fi, ti = np.nonzero(stats["n"] > 0)
                if fi.size == 0:
                    continue
                enough = stats["n"][fi, ti] >= a.ic.min_year_observations

                def col(key: str, fi: np.ndarray = fi, ti: np.ndarray = ti,
                        enough: np.ndarray = enough, stats: dict[str, np.ndarray] = stats
                        ) -> np.ndarray:
                    return np.where(enough, stats[key][fi, ti], np.nan)

                frames.append(pl.DataFrame({
                    "feature": _labels(names, fi), "target": _labels(targets, ti),
                    "kind": _labels(kinds, ti),
                    "conditioning": _labels([cname], np.zeros(fi.size, dtype=np.int64)),
                    "condition": _labels([label], np.zeros(fi.size, dtype=np.int64))}
                ).with_columns(
                    pl.Series("horizon", horizons[ti]), pl.Series("ic", col("ic")),
                    pl.Series("n", stats["n"][fi, ti]), pl.Series("se", col("se")),
                    pl.Series("z", col("z")), pl.Series("p", col("p"))))
        del xr
        ctx.log(f"conditioning: {b0 + len(names)}/{len(feats)} features "
                f"({time.perf_counter() - started:.0f}s)")
    frame = pl.concat(frames, how="diagonal_relaxed").with_columns(
        pl.col(["ic", "se", "z", "p"]).fill_nan(None))
    conds = plans
    regime = tuple(c for c in conds if c.startswith("regime"))
    for name, part in (*_CONDITION_GROUPS.items(), ("regime_conditioning", regime)):
        write_table(frame.filter(pl.col("conditioning").is_in(list(part))),
                    ctx.path(f"conditioning/{name}"), csv=name != "intraday_ic")
    _mark(marker, stamp)


# ---------------------------------------------------------------------------
# Stage: mutual information
# ---------------------------------------------------------------------------
def stage_mi(ctx: ResearchContext, data: ResearchData) -> None:
    begun = _begin(ctx, data, "mi")
    if begun is None:
        return
    stamp, marker = begun
    a = ctx.cfg.alpha.mutual_information
    bins = a.bins
    tb = data.blocks(ctx)
    targets = [f"target_{f}_{h}" for f in a.target_families for h in a.horizons
               if f"target_{f}_{h}" in tb.names]
    cys = {t: sentinel_codes(binned_codes(tb.ranks[:, tb.index(t)], bins), bins)
           for t in targets}
    rng = np.random.default_rng(ctx.cfg.alpha.seed + 17)
    n = data.n
    shifts = rng.integers(int(0.1 * n), int(0.9 * n), size=a.null_shifts)
    perms = [rng.permutation(n).astype(np.int32) for _ in range(a.null_permutations)]
    vol = None
    vol_feature = ctx.cfg.alpha.conditioning.volatility_feature
    if a.conditional_on_volatility and vol_feature in data.feature_names:
        vol = causal_quantile_buckets(data.feature(vol_feature).astype(np.float64),
                                      data.timestamps, 4, min_history=250 * ctx.bars_per_day)
    rows = []
    started = time.perf_counter()
    for i, f in enumerate(data.feature_names):
        bx = binned_codes(rank_scores(_feature_release(data, f)), bins)
        cx = sentinel_codes(bx, bins, scale=bins + 1)
        permuted = [cx[p] for p in perms]
        for t in targets:
            cy = cys[t]
            mi, pairs = mi_from_joint(joint_counts(cx, cy, bins))
            ns = np.array([mi_from_joint(joint_counts(cx, cy, bins, shift=int(s)))[0]
                           for s in shifts])
            npm = np.array([mi_from_joint(joint_counts(px, cy, bins))[0] for px in permuted])
            cmi = (conditional_mi_from_codes(cx, cy, vol, 4, bins)[0] if vol is not None
                   else None)
            sd = float(np.std(ns, ddof=1)) if ns.size > 1 else float("nan")
            z = (mi - float(ns.mean())) / sd if sd > 0 else float("nan")
            rows.append({"feature": f, "target": t, "horizon": _horizon(t),
                         "kind": target_kind(t), "mi_nats": _num(mi), "pairs": int(pairs),
                         "null_shift_mean": _num(ns.mean()), "null_shift_sd": _num(sd),
                         "null_shift_max": _num(ns.max()),
                         "null_perm_mean": _num(npm.mean()), "null_perm_max": _num(npm.max()),
                         "mi_excess_over_shift_mean": _num(mi - ns.mean()),
                         "z_vs_shift": _num(z),
                         "p_shift_z": _num(norm.sf(z)) if np.isfinite(z) else None,
                         "p_shift_empirical": (1 + int((ns >= mi).sum())) / (1 + ns.size),
                         "beyond_shift_max": bool(np.isfinite(mi) and mi > ns.max()),
                         "conditional_mi_given_volatility": _num(cmi)})
        if (i + 1) % 16 == 0 or i + 1 == len(data.feature_names):
            ctx.log(f"mi: {i + 1}/{len(data.feature_names)} features "
                    f"({time.perf_counter() - started:.0f}s)")
    write_table(pl.DataFrame(rows, infer_schema_length=None), ctx.path("mi/mutual_information"))
    _mark(marker, stamp, {"estimator": "copula histogram, Miller-Madow", "bins": bins,
                          "null_shifts": a.null_shifts, "null_permutations": a.null_permutations})


# ---------------------------------------------------------------------------
# Stage: redundancy
# ---------------------------------------------------------------------------
def stage_redundancy(ctx: ResearchContext, data: ResearchData) -> None:
    begun = _begin(ctx, data, "redundancy")
    if begun is None:
        return
    stamp, marker = begun
    r = ctx.cfg.alpha.redundancy
    rows = evenly_spaced_rows(data.n, r.sample_rows)
    names = data.feature_names
    sample = np.empty((rows.size, len(names)), dtype=np.float64)
    for j, f in enumerate(names):
        sample[:, j] = _feature_release(data, f)[rows]
    ctx.log(f"redundancy: {len(names)} features x {rows.size:,} sampled bars")
    pearson, counts = pairwise_correlation(sample)
    spearman, _ = pairwise_correlation(sample, rank=True)
    abs_s = np.abs(spearman)
    labels, link = cluster_features(abs_s, threshold=r.cluster_abs_corr, method=r.linkage)
    groups, edges = redundancy_groups(abs_s, names, threshold=r.graph_abs_corr)
    mi_rows = evenly_spaced_rows(rows.size,
                                 ctx.cfg.alpha.mutual_information.feature_pairs_sample_rows)
    nmi = feature_nmi_matrix(sample[mi_rows], bins=ctx.cfg.alpha.mutual_information.bins)
    hidden = nonmonotone_pairs(nmi, names, abs_s, nmi_threshold=r.nonmonotone_nmi,
                               corr_ceiling=r.graph_abs_corr)
    del sample
    iu, ju = np.triu_indices(len(names), 1)
    write_table(pl.DataFrame({"a": _labels(names, iu), "b": _labels(names, ju)}).with_columns(
        pl.Series("pearson", pearson[iu, ju]), pl.Series("spearman", spearman[iu, ju]),
        pl.Series("normalized_mi", nmi[iu, ju]), pl.Series("pairs", counts[iu, ju])),
        ctx.path("redundancy/feature_correlations"), csv=False)
    write_table(pl.DataFrame({"feature": names, "cluster": labels.tolist()}),
                ctx.path("redundancy/feature_clusters_table"))
    clusters: dict[str, list[str]] = {}
    for f, c in zip(names, labels, strict=True):
        clusters.setdefault(str(int(c)), []).append(f)
    write_json(ctx.path("redundancy/feature_clusters.json"), {
        "threshold_abs_spearman": r.cluster_abs_corr, "linkage": r.linkage,
        "sample_rows": int(rows.size), "estimate": True,
        "clusters": dict(sorted(clusters.items(), key=lambda kv: -len(kv[1]))),
        "redundancy_groups": {"threshold_abs_spearman": r.graph_abs_corr, "groups": groups},
        "graph_edges": edges, "nonmonotone_pairs": hidden})
    cache = ensure_dir(ctx.path("redundancy/cache"))
    np.save(cache / "linkage.npy", link)
    np.save(cache / "spearman.npy", spearman)
    np.save(cache / "normalized_mi.npy", nmi)
    write_json(cache / "names.json", names)
    _mark(marker, stamp)


# ---------------------------------------------------------------------------
# Stage: deciles
# ---------------------------------------------------------------------------
def stage_deciles(ctx: ResearchContext, data: ResearchData) -> None:
    begun = _begin(ctx, data, "deciles")
    if begun is None:
        return
    stamp, marker = begun
    d = ctx.cfg.alpha.deciles
    q = d.quantiles
    targets = [f"target_{f}_{h}" for f in d.target_families for h in d.horizons
               if f"target_{f}_{h}" in data.target_names]
    years = data.timestamps.dt.year().to_numpy()
    uniq_years = np.unique(years)
    year_idx = np.searchsorted(uniq_years, years)
    n = data.n
    eras = ctx.cfg.alpha.ic.eras
    era = np.minimum((np.arange(n) * eras) // n, eras - 1)
    k = ctx.cfg.features.families.regime.states
    state = _regime_state(data, k)
    data.release()
    curves: list[dict[str, Any]] = []
    spreads: list[dict[str, Any]] = []
    monot: list[dict[str, Any]] = []
    started = time.perf_counter()
    for i, f in enumerate(data.feature_names):
        x = _feature_release(data, f).astype(np.float64)
        ok = np.isfinite(x)
        if ok.sum() < q * 50:
            continue
        edges = np.unique(np.quantile(x[ok], np.linspace(0, 1, q + 1)[1:-1]))
        dec = np.full(n, -1, dtype=np.int64)
        dec[ok] = np.searchsorted(edges, x[ok], side="right")
        nq = int(edges.size + 1)
        for t in targets:
            y = data.target(t).astype(np.float64)
            m = (dec >= 0) & np.isfinite(y)
            dm, ym = dec[m], y[m]
            cnt = np.bincount(dm, minlength=nq).astype(np.float64)
            s1 = np.bincount(dm, weights=ym, minlength=nq)
            s2 = np.bincount(dm, weights=ym * ym, minlength=nq)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = s1 / cnt
                sd = np.sqrt(np.maximum(s2 / cnt - mean ** 2, 0.0))
                se = sd / np.sqrt(cnt)
            ys = ym[np.argsort(dm, kind="stable")]
            bounds = np.concatenate(([0], np.cumsum(cnt).astype(np.int64)))
            medians = [float(np.median(ys[bounds[j]:bounds[j + 1]])) if cnt[j] > 0 else None
                       for j in range(nq)]
            for j in range(nq):
                curves.append({"feature": f, "target": t, "horizon": _horizon(t),
                               "kind": target_kind(t), "decile": j + 1, "n": int(cnt[j]),
                               "mean": _num(mean[j]), "median": medians[j], "sd": _num(sd[j]),
                               "ci_low": _num(mean[j] - 1.96 * se[j]),
                               "ci_high": _num(mean[j] + 1.96 * se[j])})
            valid = np.isfinite(mean)
            rho = None
            if valid.sum() > 2:
                pos = np.arange(nq)[valid].astype(np.float64)
                rho = float(np.corrcoef(pos, pl.Series(mean[valid]).rank().to_numpy())[0, 1])
            spread = float(mean[-1] - mean[0]) if valid[0] and valid[-1] else None
            code_y = year_idx[m] * nq + dm
            cy = np.bincount(code_y, minlength=uniq_years.size * nq).reshape(-1, nq)
            sy = np.bincount(code_y, weights=ym, minlength=uniq_years.size * nq).reshape(-1, nq)
            with np.errstate(invalid="ignore", divide="ignore"):
                yearly = sy[:, -1] / cy[:, -1] - sy[:, 0] / cy[:, 0]
            yearly = np.where((cy[:, 0] >= 30) & (cy[:, -1] >= 30), yearly, np.nan)
            same = np.isfinite(yearly) & (np.sign(yearly) == np.sign(spread or 0.0))
            for yi in np.flatnonzero(np.isfinite(yearly)):
                spreads.append({"feature": f, "target": t, "period": str(uniq_years[yi]),
                                "period_kind": "year", "top_minus_bottom": float(yearly[yi])})
            for label, codes, count in (("era", era, eras), ("regime_state", state, k)):
                sel = m & (codes >= 0)
                code = codes[sel] * nq + dec[sel]
                cc = np.bincount(code, minlength=count * nq).reshape(-1, nq)
                ss = np.bincount(code, weights=y[sel], minlength=count * nq).reshape(-1, nq)
                for g in range(count):
                    if cc[g, 0] >= 30 and cc[g, -1] >= 30:
                        spreads.append({"feature": f, "target": t, "period": f"{label}_{g}",
                                        "period_kind": label,
                                        "top_minus_bottom": float(ss[g, -1] / cc[g, -1]
                                                                  - ss[g, 0] / cc[g, 0])})
            monot.append({"feature": f, "target": t, "horizon": _horizon(t),
                          "kind": target_kind(t), "deciles": nq,
                          "monotonicity_spearman": rho, "top_minus_bottom": spread,
                          "top_minus_bottom_se": _num(np.sqrt(se[0] ** 2 + se[-1] ** 2)),
                          "years_with_spread": int(np.isfinite(yearly).sum()),
                          "years_same_sign": int(same.sum())})
        if (i + 1) % 16 == 0 or i + 1 == len(data.feature_names):
            ctx.log(f"deciles: {i + 1}/{len(data.feature_names)} features "
                    f"({time.perf_counter() - started:.0f}s)")
    write_table(pl.DataFrame(curves, infer_schema_length=None), ctx.path("deciles/decile_curves"),
                csv=False)
    write_table(pl.DataFrame(spreads, infer_schema_length=None),
                ctx.path("deciles/top_bottom_spread_through_time"), csv=False)
    write_table(pl.DataFrame(monot, infer_schema_length=None), ctx.path("deciles/monotonicity"))
    _mark(marker, stamp, {"note": "decile edges are full-sample quantiles of the feature: a "
                                  "descriptive grouping for an offline statistic, never a "
                                  "feature"})


# ---------------------------------------------------------------------------
# Stage: interactions
# ---------------------------------------------------------------------------
def stage_interactions(ctx: ResearchContext, data: ResearchData) -> None:
    begun = _begin(ctx, data, "interactions")
    if begun is None:
        return
    stamp, marker = begun
    a = ctx.cfg.alpha
    ix_cfg = a.interactions
    tb = data.blocks(ctx)
    targets = [f"target_{f}_{h}" for f in ix_cfg.target_families for h in ix_cfg.horizons
               if f"target_{f}_{h}" in tb.names]
    embargo = max(ctx.cfg.targets.horizons)
    mcode = month_codes(data.months.starts, data.n)
    y_cols = tb.rank_columns([tb.index(t) for t in targets])
    fold_rows: list[dict[str, Any]] = []
    cond_rows: list[dict[str, Any]] = []
    for ix in ctx.cfg.features.interactions:
        if not {ix.name, ix.a, ix.b} <= set(data.feature_names):
            continue
        xa = data.feature(ix.a).astype(np.float64)
        xb = data.feature(ix.b).astype(np.float64)
        xi = data.feature(ix.name).astype(np.float64)
        tdict = {t: data.target(t).astype(np.float64) for t in targets}
        for r in interaction_increment(data.timestamps, xa, xb, xi, tdict, ridge=ix_cfg.ridge,
                                       max_rows=ix_cfg.max_rows, embargo=embargo):
            fold_rows.append({"interaction": ix.name, "a": ix.a, "b": ix.b, **r})
        del tdict
        # conditional effect: the IC of a within causal quartiles of b (Step 45)
        codes = causal_quantile_buckets(xb, data.timestamps, 4,
                                        min_history=250 * ctx.bars_per_day)
        res = condition_ics(condition_moments(rank_scores(xa), y_cols, mcode, codes,
                                              data.months.size, 4),
                            min_obs=a.ic.min_year_observations,
                            hac_lags=a.ic.hac_lag_months)
        for c in range(4):
            for j, t in enumerate(targets):
                cond_rows.append({"interaction": ix.name, "feature": ix.a,
                                  "conditioned_on": ix.b, "quartile_of_b": f"Q{c + 1}",
                                  "target": t, "horizon": _horizon(t), "kind": target_kind(t),
                                  "ic": _num(res["ic"][c, j]), "n": int(res["n"][c, j]),
                                  "se": _num(res["se"][c, j]), "p": _num(res["p"][c, j])})
        data.release()
        ctx.log(f"interactions: {ix.name}")
    folds = pl.DataFrame(fold_rows, infer_schema_length=None)
    write_table(folds, ctx.path("interactions/interaction_oos_folds"))
    if not folds.is_empty():
        wide = folds.pivot(on="model", index=["interaction", "a", "b", "target", "fold"],
                           values="oos_r2")
        summary = wide.with_columns(
            (pl.col("components+interaction") - pl.col("components")).alias("delta_r2")
        ).group_by(["interaction", "a", "b", "target"]).agg(
            pl.col("delta_r2").mean().alias("mean_delta_oos_r2"),
            pl.col("delta_r2").min().alias("min_delta_oos_r2"),
            (pl.col("delta_r2") > 0).sum().alias("folds_better"),
            pl.len().alias("folds"),
            pl.col("components").mean().alias("mean_oos_r2_components"),
        ).sort(["interaction", "target"])
        write_table(summary, ctx.path("interactions/interaction_increment"))
    write_table(pl.DataFrame(cond_rows, infer_schema_length=None),
                ctx.path("interactions/conditional_effects"))
    _mark(marker, stamp, {"models": "ridge OLS diagnostics on chronological folds, trained "
                                    "strictly before each block, embargo = longest horizon"})


# ---------------------------------------------------------------------------
# Stage: nulls (shift, permutation, wrong alignment, look-ahead, synthetic noise)
# ---------------------------------------------------------------------------
def null_rows(ctx: ResearchContext, names: list[str], get: Callable[[str], np.ndarray],
              zy: NullMatrices, targets: list[str], *, seed: int, chunk: int = 24,
              label: str = "nulls") -> tuple[list[pl.DataFrame], list[pl.DataFrame], np.ndarray]:
    """Null rows for *names*: per test, per feature x target kind (max-T), per-draw maxima.

    Per test: |IC_0| against the circular-shift quantile, the permutation
    quantile and the wrong alignments. Per feature: Westfall-Young single-step
    studentized max-T over *all* its targets - each target's |IC_0| divided by
    the root mean square of its own shift null, the maximum over targets
    taken in every shift draw; a target kind passes when its best studentized
    |IC_0| exceeds the null quantile of that maximum. This controls the
    feature's family-wise error over every target, horizon and kind at once
    (a per-test quantile let 16 of 50 noise features through at 1h, a per-kind
    maximum 2 of 50).
    """
    nl = ctx.cfg.alpha.nulls
    min_obs = ctx.cfg.alpha.ic.min_observations
    n = zy.z.shape[0]
    days = ctx.bars_per_day
    wrong = [d * days for d in nl.time_shift_days] + [-d * days for d in nl.time_shift_days]
    wrong_names = [f"wrong_{'plus' if lag > 0 else 'minus'}_{abs(lag) // days}d"
                   for lag in wrong]
    kinds = [target_kind(t) for t in targets]
    by_kind = {k: np.array([j for j, kk in enumerate(kinds) if kk == k]) for k in dict.fromkeys(kinds)}
    horizons = np.array([_horizon(t) for t in targets])
    frames, kind_frames, maxima = [], [], []
    started = time.perf_counter()
    for c0 in range(0, len(names), chunk):
        part = names[c0:c0 + chunk]
        zx = standardized_ranks([get(f) for f in part])
        real = shifted_ic(zx.z, zy.z, 0)
        pairs = pair_counts(zx.present, zy.present)
        scale = n / np.maximum(pairs, 1.0)
        shift, _ = circular_shift_null(zx.z, zy.z, draws=nl.circular_shifts,
                                       min_fraction=nl.min_shift_fraction, seed=seed)
        perm = permutation_null(zx.z, zy.z, draws=nl.permutations, seed=seed + 1)
        abs_shift = np.abs(shift)
        q_shift = np.quantile(abs_shift, nl.null_quantile, axis=0)
        q_perm = np.quantile(np.abs(perm), nl.null_quantile, axis=0)
        pct = (abs_shift < np.abs(real)[None]).mean(axis=0)
        wrong_ic = [shifted_ic(zx.z, zy.z, int(lag)) * scale for lag in wrong]
        # the null's RMS (not its sd): a shift null off-centre (slow eras) stays conservative
        null_sd = np.sqrt(np.mean(np.square(shift, dtype=np.float64), axis=0))   # (p, q)
        usable = (pairs >= min_obs) & (null_sd > 0)
        sd_safe = np.where(usable, null_sd, 1.0)
        t_real = np.where(usable, np.abs(real) / sd_safe, -1.0)          # (p, q)
        t_null = np.where(usable[None], abs_shift / sd_safe[None], -1.0)  # (D, p, q)
        t_max = t_null.max(axis=2)                                       # (D, p)
        q_max = np.quantile(t_max, nl.null_quantile, axis=0)             # (p,)
        rows_idx = np.arange(len(part))
        for kind, idx in by_kind.items():
            best_j = idx[np.argmax(t_real[:, idx], axis=1)]
            best_t = t_real[rows_idx, best_j]
            p_max = (1 + (t_max >= best_t[None]).sum(axis=0)) / (1 + t_max.shape[0])
            defined = best_t >= 0
            kind_frames.append(pl.DataFrame({
                "feature": _labels(part, np.flatnonzero(defined)),
                "kind": _labels([kind], np.zeros(int(defined.sum()), dtype=np.int64)),
                "best_target_ic0": _labels(targets, best_j[defined])}).with_columns(
                pl.Series("targets_in_kind", np.full(int(defined.sum()), idx.size)),
                pl.Series("targets_in_family", usable.sum(axis=1)[defined]),
                pl.Series("best_abs_rank_ic0",
                          (np.abs(real) * scale)[rows_idx, best_j][defined].astype(np.float64)),
                pl.Series("best_studentized", best_t[defined].astype(np.float64)),
                pl.Series("max_null_q", q_max[defined].astype(np.float64)),
                pl.Series("max_null_p", p_max[defined].astype(np.float64)),
                pl.Series("beyond_kind_max_null", (best_t > q_max)[defined])))
        del t_null
        fi, ti = np.nonzero(pairs >= min_obs)
        frames.append(pl.DataFrame({
            "feature": _labels(part, fi), "target": _labels(targets, ti),
            "kind": _labels(kinds, ti)}).with_columns(
            pl.Series("horizon", horizons[ti]),
            pl.Series("rank_ic0", (real * scale)[fi, ti].astype(np.float64)),
            pl.Series("pairs", pairs[fi, ti]),
            pl.Series("shift_null_q", (q_shift * scale)[fi, ti].astype(np.float64)),
            pl.Series("shift_null_mean_abs",
                      (abs_shift.mean(axis=0) * scale)[fi, ti].astype(np.float64)),
            pl.Series("perm_null_q", (q_perm * scale)[fi, ti].astype(np.float64)),
            pl.Series("null_percentile", pct[fi, ti].astype(np.float64)),
            pl.Series("beyond_shift_null", (np.abs(real) > q_shift)[fi, ti]),
            *[pl.Series(w, v[fi, ti].astype(np.float64))
              for w, v in zip(wrong_names, wrong_ic, strict=True)]))
        maxima.append(abs_shift.max(axis=(1, 2)))
        del zx, shift, perm, abs_shift, wrong_ic
        gc.collect()
        ctx.log(f"{label}: {c0 + len(part)}/{len(names)} features "
                f"({time.perf_counter() - started:.0f}s)")
    return frames, kind_frames, np.max(np.column_stack(maxima), axis=1)


def stage_nulls(ctx: ResearchContext, data: ResearchData) -> None:
    begun = _begin(ctx, data, "nulls")
    if begun is None:
        return
    stamp, marker = begun
    a = ctx.cfg.alpha
    tb = data.blocks(ctx)
    targets = list(data.target_names)
    zy = standardized_scores([tb.ranks[:, tb.index(t)] for t in targets])
    frames, kframes, maxima = null_rows(ctx, data.feature_names,
                                        lambda f: _feature_release(data, f), zy, targets,
                                        seed=a.seed + 101)
    write_table(pl.concat(frames, how="diagonal_relaxed"), ctx.path("nulls/null_tests"))
    write_table(pl.concat(kframes, how="diagonal_relaxed"), ctx.path("nulls/kind_max_null"))
    write_table(pl.DataFrame({"draw": np.arange(maxima.size),
                              "max_abs_rank_ic0_over_all_pairs": maxima}),
                ctx.path("nulls/shift_null_maxima"))
    _lookahead(ctx, data)
    noise = _synthetic_features(ctx, data)
    nframes, nkframes, _ = null_rows(ctx, list(noise), lambda f: noise[f][0], zy, targets,
                                     seed=a.seed + 101, label="nulls (noise features)")
    write_table(pl.concat(nframes, how="diagonal_relaxed"),
                ctx.path("nulls/synthetic_noise_null_tests"))
    write_table(pl.concat(nkframes, how="diagonal_relaxed"),
                ctx.path("nulls/synthetic_noise_kind_max_null"))
    del zy
    gc.collect()
    _synthetic_ic(ctx, data, noise)
    _mark(marker, stamp, {"estimator": "rank IC with missing = 0 (IC_0), rescaled by n / pairs",
                          "circular_shifts": a.nulls.circular_shifts,
                          "permutations": a.nulls.permutations})


def _lookahead(ctx: ResearchContext, data: ResearchData) -> None:
    """A feature deliberately taken one bar ahead must look strongly predictive."""
    rows = []
    for f in ("ret_1", "log_rv_5", "reg_resid_z_128"):
        if f not in data.feature_names:
            continue
        x = data.feature(f).astype(np.float64)
        ahead = np.full(x.size, np.nan)
        ahead[:-1] = x[1:]
        for t in ("target_return_1", "target_abs_return_1", "target_residual_change_1"):
            if t not in data.target_names:
                continue
            y = data.target(t).astype(np.float64)
            for label, v in (("aligned: x_t", x), ("leaked: x_(t+1)", ahead)):
                ok = np.isfinite(v) & np.isfinite(y)
                rho = _corr(rank_scores(v[ok]), rank_scores(y[ok]))
                rows.append({"feature": f, "target": t, "alignment": label, "rank_ic": rho,
                             "n": int(ok.sum())})
    data.release()
    write_table(pl.DataFrame(rows, infer_schema_length=None), ctx.path("nulls/lookahead_check"))


_MATCHED = ("log_rv_20", "reg_resid_z_128", "ou_log_half_life_256", "fft_entropy_256",
            "wav_entropy_512", "spread_rel", "acf_lag1_256", "ret_z_20", "regime_p0",
            "log_tick_count")


def _synthetic_features(ctx: ResearchContext, data: ResearchData
                        ) -> dict[str, tuple[np.ndarray, str]]:
    s = ctx.cfg.alpha.nulls.synthetic
    chosen = [f for f in _MATCHED if f in data.feature_names][: s.matched_marginal]
    matched = {f: data.feature(f).astype(np.float64) for f in chosen}
    noise = synthetic_noise_features(data.n, white=s.white, ar1=s.ar1, per_phi=s.ar1_per_phi,
                                     matched=matched, seed=ctx.cfg.alpha.seed + 303)
    data.release()
    return noise


def _synthetic_ic(ctx: ResearchContext, data: ResearchData,
                  noise: dict[str, tuple[np.ndarray, str]]) -> None:
    """Noise features through the same moments and stability tables as the real ones."""
    tb = data.blocks(ctx)
    names = list(noise)
    raw, rnk = compute_moments(names, lambda nm: noise[nm][0], tb, data.months, log=ctx.log)
    tables = ic_tables(ctx, names, tb, data.months, raw, rnk)
    describe = pl.DataFrame({"feature": names, "description": [noise[nm][1] for nm in names]})
    write_table(tables["ic"].join(describe, on="feature", how="left"),
                ctx.path("nulls/synthetic_noise_ic"))
    write_table(tables["consistency"], ctx.path("nulls/synthetic_noise_consistency"))


# ---------------------------------------------------------------------------
# Stage: robustness (parameter neighbourhoods, perturbations, scaling, winsorising)
# ---------------------------------------------------------------------------
_PERTURBATIONS: dict[str, tuple[str, ...]] = {
    "log_rv_20": ("log_parkinson_20", "log_garman_klass_20", "log_ewma_vol_94", "log_rv_5",
                  "log_rv_64"),
    "reg_resid_z_128": ("reg_resid_z_64", "reg_resid_z_256", "reg_resid_vol_128"),
    "fft_entropy_256": ("fft_entropy_128", "fft_entropy_512"),
    "wav_entropy_512": ("wav_entropy_256", "wav_entropy_1024"),
    "wav_fast_slow_512": ("wav_fast_slow_256", "wav_fast_slow_1024"),
    "ou_log_half_life_256": ("ou_log_half_life_128", "ou_log_half_life_512"),
    "acf_lag1_256": ("acf_lag1_128", "acf_lag1_512"),
    "ret_z_20": ("ret_z_64", "ret_20"),
}
_SCALING_FEATURES = ("log_rv_20", "reg_resid_z_128", "ou_log_half_life_256", "spread_rel",
                     "ret_z_20", "fft_entropy_256", "wav_fast_slow_512", "log_tick_count")
_SCALING_TARGETS = ("target_return_5", "target_abs_return_5", "target_realized_vol_20",
                    "target_residual_reduction_5")


def stage_robustness(ctx: ResearchContext, data: ResearchData) -> None:
    begun = _begin(ctx, data, "robustness")
    if begun is None:
        return
    stamp, marker = begun
    ic_path = ctx.path("ic/feature_ic.parquet")
    if not ic_path.exists():
        raise RuntimeError("the robustness stage reads the ic stage's tables: run it first")
    head = pl.read_parquet(ic_path).filter(
        pl.col("target").str.starts_with("target_")
        & (pl.col("method") == ctx.cfg.alpha.ic.headline))
    fam = pl.DataFrame([{"feature": r["name"], "parameter_family": r.get("parameter_family"),
                         "window": r.get("window")} for r in data.registry
                        if r.get("parameter_family")], infer_schema_length=None)
    if not fam.is_empty():
        joined = head.join(fam, on="feature")
        stability = joined.group_by(["parameter_family", "target"]).agg(
            pl.len().alias("members"),
            pl.col("ic").abs().max().alias("max_abs_ic"),
            pl.col("ic").abs().min().alias("min_abs_ic"),
            (pl.col("ic") > 0).mean().alias("positive_share"),
            pl.col("ic").std().alias("ic_sd_across_windows"),
        ).filter(pl.col("members") > 1).with_columns(
            (pl.col("min_abs_ic") / pl.col("max_abs_ic")).alias("min_over_max_abs_ic"),
            pl.max_horizontal(pl.col("positive_share"), 1 - pl.col("positive_share"))
            .alias("sign_agreement"))
        write_table(stability.sort(["parameter_family", "target"]),
                    ctx.path("robustness/parameter_stability"))
        write_table(joined.select("parameter_family", "feature", "window", "target", "horizon",
                                  "kind", "ic", "se"),
                    ctx.path("robustness/parameter_surface"), csv=False)
    rows = []
    for base, others in _PERTURBATIONS.items():
        b = head.filter(pl.col("feature") == base).select("target", pl.col("ic").alias("base_ic"))
        for other in others:
            o = head.filter(pl.col("feature") == other).select(
                "target", pl.col("ic").alias("perturbed_ic"))
            for r in b.join(o, on="target").iter_rows(named=True):
                rows.append({"base": base, "perturbation": other, **r,
                             "same_sign": bool(np.sign(r["base_ic"])
                                               == np.sign(r["perturbed_ic"])),
                             "abs_ratio": _num(abs(r["perturbed_ic"]) / abs(r["base_ic"]))
                             if r["base_ic"] else None})
    write_table(pl.DataFrame(rows, infer_schema_length=None),
                ctx.path("robustness/perturbation_sensitivity"))
    _scaling_and_winsor(ctx, data)
    _mark(marker, stamp)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; NaN (no warning) when either side is constant."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.corrcoef(a, b)[0, 1])


def _pearson_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 1000:
        return float("nan"), float("nan"), int(ok.sum())
    xs, ys = x[ok], y[ok]
    return _corr(xs, ys), _corr(rank_scores(xs), rank_scores(ys)), int(ok.sum())


def _scaling_and_winsor(ctx: ResearchContext, data: ResearchData) -> None:
    targets = [t for t in _SCALING_TARGETS if t in data.target_names]
    window = ctx.cfg.features.interaction_standardisation_days * ctx.bars_per_day
    ys = {t: data.target(t).astype(np.float64) for t in targets}
    rows = []
    for f in _SCALING_FEATURES:
        if f not in data.feature_names:
            continue
        x = _feature_release(data, f).astype(np.float64)
        for variant, values in scaling_variants(x, data.timestamps, window).items():
            for t in targets:
                pear, spear, n = _pearson_spearman(values, ys[t])
                rows.append({"feature": f, "transformation": variant, "target": t,
                             "pearson_ic": _num(pear), "spearman_ic": _num(spear), "n": n})
    write_table(pl.DataFrame(rows, infer_schema_length=None),
                ctx.path("robustness/scaling_research"))
    wrows = []
    for f in data.feature_names:
        x = _feature_release(data, f).astype(np.float64)
        w = winsorize_causal(x, data.timestamps, 0.01, 0.99)
        defined = np.isfinite(w)
        clipped = float(np.mean(w[defined] != x[defined])) if defined.any() else None
        for t in targets:
            y = ys[t]
            ok = defined & np.isfinite(y)
            if ok.sum() < 1000:
                continue
            raw_ic = _corr(x[ok], y[ok])
            win_ic = _corr(w[ok], y[ok])
            wrows.append({"feature": f, "target": t, "pearson_ic_raw": _num(raw_ic),
                          "pearson_ic_winsorized": _num(win_ic),
                          "change": _num(win_ic - raw_ic), "share_clipped": clipped,
                          "n": int(ok.sum())})
    write_table(pl.DataFrame(wrows, infer_schema_length=None),
                ctx.path("robustness/winsorization_research"))


# ---------------------------------------------------------------------------
# Stage: cost
# ---------------------------------------------------------------------------
def _tail(bs: BarSeries, span: int) -> BarSeries:
    lo = max(0, bs.size - span)

    def cut(v: np.ndarray | None) -> np.ndarray | None:
        return None if v is None else v[lo:]

    return BarSeries(name=f"{bs.name}_tail", timeframe=bs.timeframe,
                     timestamps=bs.timestamps.slice(lo), close=bs.close[lo:],
                     bar_seconds=bs.bar_seconds, open=cut(bs.open), high=cut(bs.high),
                     low=cut(bs.low), median_spread=cut(bs.median_spread),
                     tick_count=cut(bs.tick_count), missing_slots=cut(bs.missing_slots))


def stage_cost(ctx: ResearchContext, data: ResearchData) -> None:
    begun = _begin(ctx, data, "cost")
    if begun is None:
        return
    stamp, marker = begun
    cfg = ctx.cfg
    c = cfg.alpha.cost
    sample = _tail(load_bar_series(cfg.config, ctx.timeframe), c.sample_bars)
    registry = data.registry
    rows: list[dict[str, Any]] = []
    for family in ("returns", "volatility", "autocorrelation", "regression", "ou", "fft",
                   "wavelet", "microstructure", "time"):
        specs = [r for r in registry if r.get("family") == family]
        if not specs:
            continue
        buffer = max(int(r.get("min_history") or 0) for r in specs)
        prereq: tuple[str, ...] = ("regression",) if family in ("ou", "fft") else ()
        provider = EngineProvider(bars=sample, regression_config=cfg.regression,
                                  spectral_config=cfg.spectral, wavelet_config=cfg.wavelet)
        tracemalloc.start()
        t0 = time.perf_counter()
        compute_families(sample, provider, cfg.features, cfg.ou, cfg.research,
                         families=(*prereq, family), log=False, collect=False)
        batch = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        # a live update: recompute the family on the shortest buffer ending at the latest bar
        tail = _tail(sample, min(sample.size, max(2 * buffer + 2, 64)))
        updates = []
        for _ in range(max(3, c.live_updates // 20)):
            prov = EngineProvider(bars=tail, regression_config=cfg.regression,
                                  spectral_config=cfg.spectral, wavelet_config=cfg.wavelet)
            t1 = time.perf_counter()
            compute_families(tail, prov, cfg.features, cfg.ou, cfg.research,
                             families=(*prereq, family), log=False, collect=False)
            updates.append(time.perf_counter() - t1)
        ms = 1000.0 * float(np.median(updates))
        category = ("cheap" if ms <= c.cheap_ms else "expensive" if ms >= c.expensive_ms
                    else "moderate")
        rows.append({"family": family, "features": len(specs), "sample_bars": sample.size,
                     "batch_seconds": batch, "batch_us_per_bar": 1e6 * batch / sample.size,
                     "peak_mb": peak / 2 ** 20, "live_buffer_bars": buffer,
                     "recompute_bars": tail.size, "live_recompute_ms": ms,
                     "measured_category": category,
                     "incremental_update": sorted({str(r.get("incremental_update"))
                                                   for r in specs}),
                     "includes": list(prereq)})
        ctx.log(f"cost: {family} {1e6 * batch / sample.size:.1f} us/bar batch, "
                f"{ms:.2f} ms per live recompute")
    regime_specs = [r for r in registry if r.get("family") == "regime"]
    if regime_specs:
        rows.append({"family": "regime", "features": len(regime_specs), "sample_bars": None,
                     "batch_seconds": None, "batch_us_per_bar": None, "peak_mb": None,
                     "live_buffer_bars": None, "recompute_bars": None, "live_recompute_ms": None,
                     "measured_category": "not measured here",
                     "incremental_update": ["forward filter O(K^2) per bar; a scheduled "
                                            "refit off the live path (Prompt #7)"],
                     "includes": []})
    write_table(pl.DataFrame(rows, infer_schema_length=None), ctx.path("cost/cost_profile"))
    buffers = pl.DataFrame([{"feature": r["name"], "family": r.get("family"),
                             "min_history_bars": r.get("min_history"),
                             "declared_cost": r.get("cost"),
                             "incremental_update": r.get("incremental_update")}
                            for r in registry], infer_schema_length=None)
    write_table(buffers.sort("min_history_bars", descending=True, nulls_last=True),
                ctx.path("cost/live_buffer_requirements"))
    _mark(marker, stamp, {"note": "timings on this machine; the relative sizes matter, not "
                                  "the absolute values"})


# ---------------------------------------------------------------------------
# Stage: pipeline nulls (random walk, shuffled returns, block bootstrap)
# ---------------------------------------------------------------------------
def stage_pipeline(ctx: ResearchContext, data: ResearchData) -> None:
    begun = _begin(ctx, data, "pipeline")
    if begun is None:
        return
    stamp, marker = begun
    cfg = ctx.cfg
    data.drop_targets()                      # the real targets are not needed here
    gc.collect()
    real = load_bar_series(cfg.config, ctx.timeframe)
    families = tuple(f for f in cfg.alpha.nulls.pipeline_families
                     if f in ("returns", "volatility", "autocorrelation", "regression", "ou",
                              "fft", "wavelet"))
    parts = []
    for name in cfg.alpha.nulls.pipeline:
        part = ctx.path(f"pipeline/{name}.parquet")
        pmark = ctx.path(f"pipeline/.{name}.json")
        if _reusable(pmark, {**stamp, "null": name}) and part.exists():
            parts.append(pl.read_parquet(part))
            ctx.log(f"pipeline null {name}: current")
            continue
        parts.append(_pipeline_null(ctx, data, real, name, families, part, pmark, stamp))
        gc.collect()
    write_table(pl.concat(parts, how="diagonal_relaxed"), ctx.path("pipeline/pipeline_null_ic"))
    _mark(marker, stamp, {"families": list(families),
                          "veto": list(cfg.alpha.nulls.pipeline_veto)})


def _save_quarterly_moments(ctx: ResearchContext, null: str, features: list[str],
                            targets: list[str], months: MonthIndex, rnk: np.ndarray) -> None:
    """A null's rank moments summed by calendar quarter (float64, ~30 MB at 5m).

    The feature selection of Prompt #9 re-reads the mechanical-effect veto on
    its own development period only; quarter sums allow any quarter-aligned
    period and batch-means errors over its quarters.
    """
    codes = months.years * 4 + (months.quarters - 1)
    starts = np.concatenate(([0], np.flatnonzero(np.diff(codes) != 0) + 1)).astype(np.int64)
    quarterly = np.add.reduceat(rnk, starts, axis=2)
    labels = [f"{int(c) // 4}Q{int(c) % 4 + 1}" for c in codes[starts]]
    cache = ensure_dir(ctx.path("pipeline/cache"))
    np.save(cache / f"{null}_moments_rank_quarterly.npy", quarterly)
    write_json(cache / f"{null}_axes.json", {"features": features, "targets": targets,
                                             "quarters": labels, "method": "spearman",
                                             "axes": "feature, target, quarter, moment"})


def _pipeline_null(ctx: ResearchContext, data: ResearchData, real: BarSeries, name: str,
                   families: tuple[str, ...], part: Path, pmark: Path,
                   stamp: dict[str, Any]) -> pl.DataFrame:
    cfg = ctx.cfg
    started = time.perf_counter()
    ctx.log(f"pipeline null {name}: features")
    bars = null_bar_series(name, real, seed=cfg.alpha.seed)
    provider = EngineProvider(bars=bars, regression_config=cfg.regression,
                              spectral_config=cfg.spectral, wavelet_config=cfg.wavelet)
    values, extras = compute_families(bars, provider, cfg.features, cfg.ou, cfg.research,
                                      families=families, log=False)
    # registered interactions whose two components exist on the null path
    for key, arr in interaction_columns(values, cfg.features, bars.bar_seconds).items():
        values[key] = np.asarray(arr, dtype=np.float32)
    reg = provider.regression(cfg.targets.regression_window)
    tframe = build_targets(TargetInputs(bars=bars, residual=reg["residual"], sigma=reg["sigma"],
                                        ou_mu=extras.get(f"ou_mu_{cfg.targets.ou_window}")),
                           cfg.targets)
    del provider, extras, reg
    gc.collect()
    null_targets = np.empty((tframe.height, len(data.target_names)), dtype=np.float32,
                            order="F")
    for j, c in enumerate(data.target_names):
        null_targets[:, j] = tframe[c].cast(pl.Float32).fill_null(np.nan).to_numpy()
    del tframe
    tb = target_blocks(ctx, data, base=null_targets, marginal=False)
    names = [f for f, v in values.items() if np.isfinite(v).sum() >= 1000]
    ctx.log(f"pipeline null {name}: {len(names)} features in "
            f"{time.perf_counter() - started:.0f}s; scoring")
    raw, rnk = compute_moments(names, lambda nm: values[nm], tb, data.months)
    _save_quarterly_moments(ctx, name, names, tb.names, data.months, rnk)
    horizons = np.array([tb.horizons[t] for t in tb.names])
    kinds = [target_kind(t) for t in tb.names]
    frames = []
    for method, mom in (("pearson", raw), ("spearman", rnk)):
        stats = ic_with_errors(np.moveaxis(mom, 2, 0), hac_lags=cfg.alpha.ic.hac_lag_months)
        fi, ti, base = _grid(names, tb.names, stats["n"] >= cfg.alpha.ic.min_observations)
        frames.append(base.with_columns(
            pl.lit(name).alias("null"), pl.lit(method).alias("method"),
            pl.Series("horizon", horizons[ti]), _labels(kinds, ti).alias("kind"),
            pl.Series("ic", stats["ic"][fi, ti]), pl.Series("se", stats["se"][fi, ti]),
            pl.Series("n", stats["n"][fi, ti])))
    frame = pl.concat(frames, how="diagonal_relaxed")
    ensure_dir(part.parent)
    frame.write_parquet(part)
    _mark(pmark, {**stamp, "null": name},
          {"seconds": time.perf_counter() - started, "notes": bars.notes,
           "features": len(names)})
    ctx.log(f"pipeline null {name}: done in {time.perf_counter() - started:.0f}s")
    return frame


# ---------------------------------------------------------------------------
# One timeframe
# ---------------------------------------------------------------------------
_RUNNERS: dict[str, Callable[[ResearchContext, ResearchData], None]] = {
    "quality": stage_quality, "ic": stage_ic, "conditioning": stage_conditioning,
    "mi": stage_mi, "redundancy": stage_redundancy, "deciles": stage_deciles,
    "interactions": stage_interactions, "nulls": stage_nulls, "robustness": stage_robustness,
    "cost": stage_cost, "pipeline": stage_pipeline,
}


def run_timeframe(cfg: Configs, timeframe: str, *, stages: tuple[str, ...] = STAGES,
                  resume: bool = True, progress: Callable[[str], None] | None = None
                  ) -> dict[str, Any]:
    """Run the chosen stages of one timeframe in the fixed order; returns timings."""
    unknown = sorted(set(stages) - set(STAGES))
    if unknown:
        raise ValueError(f"unknown stage(s) {unknown}; expected from {list(STAGES)}")
    out_dir = ensure_dir(cfg.features.results_path / timeframe)
    ctx = ResearchContext(timeframe=timeframe, cfg=cfg, out_dir=out_dir, progress=progress)
    started = time.perf_counter()
    info: dict[str, Any] = {"timeframe": timeframe, "stages": list(stages)}
    if "factory" in stages:
        built = stage_factory(ctx, resume=resume)
        info["factory_version"] = built["factory"].get("factory_version")
        info["target_version"] = built["targets"].get("target_version")
    data = load_research_data(ctx)
    timings: dict[str, float] = {}
    for stage in STAGES[1:]:
        if stage not in stages:
            continue
        if not resume:
            ctx.path(f"{stage}/.done.json").unlink(missing_ok=True)
        t0 = time.perf_counter()
        _RUNNERS[stage](ctx, data)
        data.release()
        gc.collect()
        timings[stage] = time.perf_counter() - t0
    info["timings"] = timings
    info["seconds"] = time.perf_counter() - started
    write_json(ctx.path("run_info.json"), {**info, "finished_utc": utc_now_iso()})
    return info
