r"""What the feature selection reads (Prompt #9, Steps 1-2, 61-62).

:func:`load_selection_data` reads, for one timeframe:

* the stored Prompt #8 feature matrix for the **development + validation**
  rows (feature values of the reserved period are read only where a step
  needs them - missingness, the live reconstruction - never with outcomes);
* the stored target table for the same rows only: the reserved period's
  target values are **not read at all** (the year files are filtered by
  timestamp at the reader), and every column is purged at the
  development / validation boundary by its own horizon
  (:func:`~.periods.outcome_safe`);
* the registry, the factory / target manifests and the dataset lineage, and
  refuses to run unless they match (canonical dataset version, timeframe,
  feature versions, timestamp convention, source-feed tag - Step 2).

Noise *probe* features (AR(1) paths and real features in AR(1)-noise order)
are appended as ``probe_*`` columns: they go through every selection method
and show the selection rate of pure noise. They never reach a manifest.

Rank transforms are always taken inside a period (development, or a training
fold): a rank over the whole history would carry the reserved period's
outcome distribution into a selection statistic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..alpha.information_coefficient import MonthIndex, month_index, rank_scores
from ..alpha.null_tests import synthetic_noise_features
from ..features.factory import TIMESTAMP_CONVENTION, current_version_dir
from ..features.factory_config import FeatureFactoryConfig
from ..targets.config import TargetConfig
from .config import FeatureSelectionConfig
from .periods import ReservedGuard, first_row_at, outcome_safe, period_rows

__all__ = ["SelectionData", "load_selection_data", "period_rank"]


class SelectionIntegrityError(RuntimeError):
    """The stored matrix, targets and registry do not describe the same dataset."""


@dataclass
class SelectionData:
    timeframe: str
    timestamps: pl.Series                  # rows [0, reserved_start) - development + validation
    names: list[str]                       # registered feature columns (and probes at the end)
    registry: dict[str, dict[str, Any]]
    features: np.ndarray                   # (n, p) float32, column-major
    probes: list[str]
    target_meta: list[tuple[str, int, str]]    # (kind, horizon, column)
    targets: np.ndarray                    # (n, T) float32, column-major, outcome-safe
    guard: ReservedGuard
    rows: dict[str, tuple[int, int]]       # development / validation / recent_development
    months: MonthIndex
    factory_manifest: dict[str, Any]
    target_manifest: dict[str, Any]
    reserved_feature_missing: dict[str, float] = field(default_factory=dict)
    matrix_dir: Path | None = None
    research_dir: Path | None = None           # Prompt #8 results of this timeframe

    @property
    def n(self) -> int:
        return int(self.timestamps.len())

    def column(self, name: str) -> np.ndarray:
        return self.features[:, self.names.index(name)]

    def target(self, column: str) -> np.ndarray:
        return self.targets[:, [c for _, _, c in self.target_meta].index(column)]

    def target_columns(self) -> list[str]:
        return [c for _, _, c in self.target_meta]

    def real_names(self) -> list[str]:
        return [n for n in self.names if n not in self.probes]


def period_rank(values: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Rank scores inside rows [lo, hi) only; NaN outside (float32)."""
    out = np.full(values.size, np.nan, dtype=np.float32)
    out[lo:hi] = rank_scores(values[lo:hi]).astype(np.float32)
    return out


def _starts_at_or_after(year: int, before: date | None) -> bool:
    """True when the whole calendar year lies at or after *before* (its file is never opened)."""
    return before is not None and date(year, 1, 1) >= before


def _read_years(base: Path, columns: list[str], before: date | None) -> pl.DataFrame:
    """Year files of *base*, rows strictly before *before* only.

    A year file that starts at or after *before* is never opened, and the row
    filter is applied by the scan, so a reserved-period value never reaches a
    frame (not even one that is filtered afterwards).
    """
    frames = []
    for path in sorted(base.glob("year=*/part-0.parquet")):
        year = int(path.parent.name.split("=")[1])
        if _starts_at_or_after(year, before):
            continue
        lazy = pl.scan_parquet(path).select("timestamp", *columns)
        if before is not None:
            lazy = lazy.filter(pl.col("timestamp") < pl.lit(_midnight(before)))
        frames.append(lazy.collect())
    return pl.concat(frames).sort("timestamp")


def _midnight(d: date) -> datetime:
    return datetime(d.year, d.month, d.day)


def _check_integrity(timeframe: str, manifest: dict[str, Any], tmanifest: dict[str, Any],
                     registry: list[dict[str, Any]], fcfg: FeatureFactoryConfig) -> None:
    problems = []
    if manifest.get("timeframe") != timeframe:
        problems.append(f"matrix timeframe {manifest.get('timeframe')!r} != {timeframe!r}")
    lineage = manifest.get("dataset_lineage") or {}
    if lineage.get("partial"):
        problems.append(str(lineage.get("label")))
    if manifest.get("timestamp_convention") != TIMESTAMP_CONVENTION:
        problems.append("timestamp convention differs from the factory's")
    if manifest.get("source_feed") != fcfg.source_feed:
        problems.append(f"source feed {manifest.get('source_feed')!r} != {fcfg.source_feed!r}")
    if (tmanifest.get("sources") or {}).get("bar_dataset_version") != \
            manifest.get("bar_dataset_version"):
        problems.append("targets and features were built from different bars")
    for r in registry:
        if not r.get("feature_version") or r.get("dataset_version") != \
                manifest.get("tick_dataset_version"):
            problems.append(f"{r.get('name')}: feature / dataset version missing or stale")
            break
        if r.get("timeframe") != timeframe or r.get("source_feed") != fcfg.source_feed:
            problems.append(f"{r.get('name')}: timeframe or source feed differs")
            break
    if problems:
        raise SelectionIntegrityError(f"{timeframe}: " + "; ".join(problems))


def load_selection_data(cfg: FeatureSelectionConfig, fcfg: FeatureFactoryConfig,
                        tcfg: TargetConfig, timeframe: str, *,
                        with_probes: bool = True) -> SelectionData:
    """Development + validation features and outcome-safe targets of *timeframe*."""
    base = current_version_dir(fcfg.factory_path, timeframe)
    manifest = json.loads((base / "_manifest.json").read_text(encoding="utf-8"))
    registry = json.loads((base / "registry.json").read_text(encoding="utf-8"))
    tpointer = tcfg.targets_path / f"timeframe={timeframe}" / "CURRENT"
    tbase = tcfg.targets_path / f"timeframe={timeframe}" / f"version={tpointer.read_text().strip()}"
    tmanifest = json.loads((tbase / "_manifest.json").read_text(encoding="utf-8"))
    _check_integrity(timeframe, manifest, tmanifest, registry, fcfg)
    reserved = cfg.periods.reserved_test.start
    names = [r["name"] for r in registry]
    # timestamps first (cheap), then one year of columns at a time into a column-major array
    stamps = _read_years(base, [], reserved)["timestamp"]
    n = int(stamps.len())
    features = np.empty((n, len(names)), dtype=np.float32, order="F")
    row = 0
    for path in sorted(base.glob("year=*/part-0.parquet")):
        year = int(path.parent.name.split("=")[1])
        if _starts_at_or_after(year, reserved):
            continue
        frame = (pl.scan_parquet(path).filter(pl.col("timestamp") < pl.lit(_midnight(reserved)))
                 .collect().sort("timestamp"))
        m = frame.height
        for j, name in enumerate(names):
            features[row:row + m, j] = frame[name].cast(pl.Float32).fill_null(np.nan).to_numpy()
        row += m
        del frame
    if row != n:
        raise SelectionIntegrityError(f"{timeframe}: {row} feature rows read, {n} timestamps")
    target_meta = cfg.targets.columns()
    tframe = _read_years(tbase, [c for _, _, c in target_meta], reserved)
    if not tframe["timestamp"].equals(stamps):
        raise SelectionIntegrityError(f"{timeframe}: targets and features are not on the same "
                                      "bars - refusing to pair them")
    dev_lo, dev_hi = period_rows(stamps, cfg.periods.development)
    val_lo, val_hi = period_rows(stamps, cfg.periods.validation)
    boundaries = [dev_hi, val_hi]                   # val_hi = n = the reserved start
    targets = np.empty((n, len(target_meta)), dtype=np.float32, order="F")
    for j, (_, h, col) in enumerate(target_meta):
        targets[:, j] = outcome_safe(tframe[col].cast(pl.Float32).fill_null(np.nan).to_numpy(),
                                     h, boundaries)
    del tframe
    guard = ReservedGuard(n, n)                     # nothing at or after row n was read
    recent = period_rows(stamps, cfg.periods.recent_development)
    probes: list[str] = []
    if with_probes and cfg.universe.probes > 0:
        features, names, probes = _add_probes(cfg, features, names, n)
    missing = _reserved_missing(base, [r["name"] for r in registry], reserved)
    return SelectionData(timeframe=timeframe, timestamps=stamps, names=names,
                         registry={r["name"]: r for r in registry}, features=features,
                         probes=probes, target_meta=target_meta, targets=targets, guard=guard,
                         rows={"development": (dev_lo, dev_hi), "validation": (val_lo, val_hi),
                               "recent_development": recent},
                         months=month_index(stamps), factory_manifest=manifest,
                         target_manifest=tmanifest, reserved_feature_missing=missing,
                         matrix_dir=base, research_dir=fcfg.results_path / timeframe)


def _add_probes(cfg: FeatureSelectionConfig, features: np.ndarray, names: list[str],
                n: int) -> tuple[np.ndarray, list[str], list[str]]:
    k = cfg.universe.probes
    matched_from = [nm for nm in ("log_rv_20", "reg_resid_z_128", "spread_rel", "ret_z_20")
                    if nm in names][: max(0, k - 6)]
    noise = synthetic_noise_features(
        n, white=0, ar1=(0.9, 0.99, 0.999), per_phi=2,
        matched={nm: features[:, names.index(nm)].astype(np.float64) for nm in matched_from},
        seed=cfg.seed + 7)
    probe_names = [f"probe_{p.removeprefix('noise_')}" for p in noise][:k]
    out = np.empty((n, features.shape[1] + len(probe_names)), dtype=np.float32, order="F")
    out[:, :features.shape[1]] = features
    for j, (_, (values, _)) in enumerate(list(noise.items())[: len(probe_names)]):
        out[:, features.shape[1] + j] = values
    return out, [*names, *probe_names], probe_names


def _reserved_missing(base: Path, names: list[str], reserved: Any) -> dict[str, float]:
    """Missing share of each feature's *values* in the reserved period (no outcome is read)."""
    counts = np.zeros(len(names))
    total = 0
    for path in sorted(base.glob("year=*/part-0.parquet")):
        year = int(path.parent.name.split("=")[1])
        if year < reserved.year:
            continue
        frame = pl.scan_parquet(path).filter(
            pl.col("timestamp") >= pl.lit(_midnight(reserved))).collect()
        total += frame.height
        for j, name in enumerate(names):
            counts[j] += frame[name].cast(pl.Float64).fill_nan(None).null_count()
    return {name: float(counts[j] / total) if total else float("nan")
            for j, name in enumerate(names)}


def first_reserved_row(timestamps: pl.Series, cfg: FeatureSelectionConfig) -> int:
    return first_row_at(timestamps, cfg.periods.reserved_test.start)
