r"""Ensemble inputs: aligned out-of-sample predictions and their coverage (Prompt #11,
Steps 4-5).

The input of every ensemble is one Prompt #10 prediction table
(``results/ml_research/<tf>/predictions/<target>_h<h>.parquet``): one row per
development bar of a walk-forward validation block, its label, and one column
per base model ``<family>|<set>`` (+ ``|platt`` for classification). Every value
in it is out of sample - predicted by a model fitted on rows before its block,
purged by the horizon plus the embargo.

:func:`load_pair` refuses a table that does not describe the current data
(dataset, bar, factory and target versions; target, horizon and timeframe; rows
and timestamps strictly increasing; blocks contiguous and in order) and records,
per model and block, how many rows carry a prediction
(:func:`ensemble_prediction_coverage`). Nothing is ever filled: a row missing any
constituent of an ensemble is left out of that ensemble, and counted.

The *meta walk-forward* (:meth:`PairPredictions.history`) gives, for an
evaluation block ``k``, the rows of the earlier blocks whose outcome had resolved
``h + embargo`` bars before block ``k`` begins - the only rows any weight,
meta-model or calibrator scored on block ``k`` may be fitted on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

__all__ = ["AlignmentError", "PairPredictions", "ensemble_prediction_coverage", "load_pair",
           "load_unit_predictions"]

_LEAD = ("timestamp", "row", "fold", "target", "horizon", "dataset_version", "label",
         "label_raw")


class AlignmentError(RuntimeError):
    """Constituent predictions do not describe the same rows, target, horizon or data."""


@dataclass
class PairPredictions:
    """Every base model's out-of-sample predictions of one target x horizon x timeframe."""

    timeframe: str
    target: str
    horizon: int
    task: str
    rows: np.ndarray                   # development row numbers (int64, strictly increasing)
    timestamps: pl.Series
    block: np.ndarray                  # block index per row (int16)
    block_names: list[str]
    label: np.ndarray                  # fitted scale (float64)
    label_raw: np.ndarray
    base: np.ndarray                   # the constant model's prediction (the block's training mean)
    raw: dict[str, np.ndarray]
    calibrated: dict[str, np.ndarray]
    models: dict[str, dict[str, Any]]
    versions: dict[str, Any]
    calibrated_inputs: bool = True
    _cache: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def n(self) -> int:
        return int(self.rows.size)

    @property
    def is_classification(self) -> bool:
        return self.task == "classification"

    @property
    def model_names(self) -> list[str]:
        return [m for m in self.raw if not m.startswith("constant|")]

    def prediction(self, model: str) -> np.ndarray:
        """The ensemble input of *model*: Platt probability (classification) or raw value."""
        if self.is_classification and self.calibrated_inputs:
            if model not in self.calibrated:
                raise AlignmentError(f"{model}: no calibrated (Platt) column")
            return self.calibrated[model]
        return self.raw[model]

    def matrix(self, models: list[str], *, raw: bool = False) -> np.ndarray:
        """(n, M) float64 predictions of *models*, NaN where a model has no prediction."""
        cols = [self.raw[m] if raw else self.prediction(m) for m in models]
        return np.column_stack(cols) if cols else np.empty((self.n, 0))

    def covered(self, models: list[str], *, raw: bool = False) -> np.ndarray:
        """Rows where every one of *models* has a finite prediction and the label is known."""
        ok = np.isfinite(self.label).copy()
        for m in models:
            ok &= np.isfinite(self.raw[m] if raw else self.prediction(m))
        return ok

    def block_positions(self, k: int) -> np.ndarray:
        return np.flatnonzero(self.block == k)

    def history(self, k: int, embargo: int) -> np.ndarray:
        """Positions of the rows any method scored on block *k* may be fitted on.

        Rows of earlier blocks only, and only those whose outcome window
        (``t+1 .. t+h``) closed ``embargo`` bars before the block's first bar.
        """
        first = self.block_positions(k)
        if first.size == 0:
            return np.zeros(0, dtype=np.int64)
        start_row = int(self.rows[first[0]])
        limit = start_row - (self.horizon + int(embargo))
        return np.flatnonzero((self.block < k) & (self.rows < limit))

    def family(self, model: str) -> str:
        return str(self.models.get(model, {}).get("family") or model.split("|")[0])

    def feature_set(self, model: str) -> str:
        return str(self.models.get(model, {}).get("feature_set") or model.split("|")[1])


def _check(frame: pl.DataFrame, meta: dict[str, Any], *, timeframe: str, target: str,
           horizon: int, expected: dict[str, Any] | None) -> list[str]:
    problems = []
    if meta.get("timeframe") != timeframe:
        problems.append(f"sidecar timeframe {meta.get('timeframe')!r} != {timeframe!r}")
    if meta.get("target") != target or int(meta.get("horizon", -1)) != int(horizon):
        problems.append("sidecar target / horizon differ")
    for col, want in (("target", target), ("horizon", horizon)):
        vals = frame[col].unique().to_list()
        if vals != [want]:
            problems.append(f"column {col} holds {vals}, expected [{want}]")
    if expected:
        versions = meta.get("versions") or {}
        for key, want in expected.items():
            if want is not None and versions.get(key) != want:
                problems.append(f"{key}: predictions built on {versions.get(key)}, current {want}")
        tick = expected.get("tick_dataset_version")
        if tick is not None and frame["dataset_version"].unique().to_list() != [tick]:
            problems.append("dataset_version column differs from the current tick dataset")
    rows = frame["row"].to_numpy()
    if rows.size and not (np.diff(rows) > 0).all():
        problems.append("rows are not strictly increasing (duplicates or disorder)")
    ts = frame["timestamp"]
    if ts.len() > 1 and not (ts.diff().drop_nulls().dt.total_microseconds().to_numpy() > 0).all():
        problems.append("timestamps are not strictly increasing")
    return problems


def _midnight(d: date) -> datetime:
    return datetime(d.year, d.month, d.day)


def load_pair(path: Path, *, timeframe: str, target: str, horizon: int, task: str,
              expected_versions: dict[str, Any] | None = None,
              calibrated_inputs: bool = True, reserved_start: date | None = None
              ) -> PairPredictions:
    """One prediction table, verified (see the module docstring); nothing is filled.

    With *reserved_start*, a table holding any row at or after it is refused before a
    single outcome of such a row is read (only its timestamp is counted).
    """
    if not path.exists():
        raise AlignmentError(f"no prediction table at {path} - run `xq ml-report` (Prompt #10)")
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    lazy = pl.scan_parquet(path)
    if reserved_start is not None:
        cut = pl.lit(_midnight(reserved_start))
        late = int(lazy.filter(pl.col("timestamp") >= cut).select(pl.len()).collect().item())
        if late:
            raise AlignmentError(f"{path.name}: {late} rows at or after the reserved start "
                                 f"{reserved_start} - a development table must hold none")
        lazy = lazy.filter(pl.col("timestamp") < cut)
    frame = lazy.collect()
    missing = [c for c in _LEAD if c not in frame.columns]
    if missing:
        raise AlignmentError(f"{path.name}: missing columns {missing}")
    problems = _check(frame, meta, timeframe=timeframe, target=target, horizon=horizon,
                      expected=expected_versions)
    if problems:
        raise AlignmentError(f"{path.name}: " + "; ".join(problems))
    folds = frame["fold"].to_list()
    names: list[str] = []
    for f in folds:                                   # blocks in order of appearance
        if not names or names[-1] != f:
            if f in names:
                raise AlignmentError(f"{path.name}: block {f} is not contiguous")
            names.append(f)
    if names != sorted(names):
        raise AlignmentError(f"{path.name}: blocks out of chronological order {names}")
    index = {n: i for i, n in enumerate(names)}
    block = np.array([index[f] for f in folds], dtype=np.int16)

    def arr(col: str) -> np.ndarray:
        return np.array(frame[col].cast(pl.Float64).fill_null(np.nan).to_numpy(),
                        dtype=np.float64)

    model_cols = [c for c in frame.columns if c not in _LEAD]
    raw = {c: arr(c) for c in model_cols if not c.endswith("|platt")}
    cal = {c.removesuffix("|platt"): arr(c) for c in model_cols if c.endswith("|platt")}
    const = [m for m in raw if m.startswith("constant|")]
    if not const:
        raise AlignmentError(f"{path.name}: no constant baseline column")
    unknown = sorted(set(raw) - set(meta.get("models") or {}))
    if unknown:
        raise AlignmentError(f"{path.name}: columns without a sidecar entry {unknown}")
    return PairPredictions(
        timeframe=timeframe, target=target, horizon=int(horizon), task=task,
        rows=frame["row"].to_numpy().astype(np.int64), timestamps=frame["timestamp"],
        block=block, block_names=names, label=arr("label"), label_raw=arr("label_raw"),
        base=raw[const[0]], raw=raw, calibrated=cal, models=dict(meta.get("models") or {}),
        versions=dict(meta.get("versions") or {}), calibrated_inputs=calibrated_inputs)


def ensemble_prediction_coverage(pair: PairPredictions) -> pl.DataFrame:
    """Per model and block: rows, rows with a prediction, share, first / last timestamp."""
    out = []
    labelled = np.isfinite(pair.label)
    for model in pair.raw:
        values = pair.prediction(model) if not model.startswith("constant|") else pair.raw[model]
        for k, name in enumerate(pair.block_names):
            pos = pair.block_positions(k)
            have = np.isfinite(values[pos])
            out.append({"timeframe": pair.timeframe, "target": pair.target,
                        "horizon": pair.horizon, "model": model, "block": name,
                        "rows": int(pos.size), "labelled": int(labelled[pos].sum()),
                        "with_prediction": int(have.sum()),
                        "share": float(have.mean()) if pos.size else None,
                        "first": str(pair.timestamps[int(pos[0])]) if pos.size else None,
                        "last": str(pair.timestamps[int(pos[-1])]) if pos.size else None,
                        "feature_set_id": (pair.models.get(model) or {}).get("feature_set_id")})
    return pl.DataFrame(out, infer_schema_length=None)


def load_unit_predictions(units_dir: Path, target: str, horizon: int, family: str,
                          feature_set: str, variant: str = "base", *,
                          development_rows: int | None = None) -> pl.DataFrame:
    """Out-of-sample predictions of one Prompt #10 unit configuration, every stored block.

    Columns ``row``, ``prediction`` (+ ``cal_platt``), ``block`` (the fold name). Empty
    when the configuration was not run. Row numbers index the development rows; with
    *development_rows* a unit holding a row beyond them is refused.
    """
    d = units_dir / target / f"h{horizon}" / family / feature_set / variant
    frames = []
    for p in sorted(d.glob("*.parquet")):
        if p.name.endswith(".partial"):
            continue
        f = pl.read_parquet(p)
        top = int(f["row"].to_numpy().max()) if f.height else -1
        if development_rows is not None and top >= development_rows:
            raise AlignmentError(f"{p}: row {top} is not a development row "
                                 f"(< {development_rows})")
        frames.append(f.with_columns(pl.lit(p.stem).alias("block")))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed").sort("row")
