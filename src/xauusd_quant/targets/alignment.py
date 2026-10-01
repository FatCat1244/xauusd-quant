r"""Building, storing and aligning the target table (Prompt #8, Steps 14-20, 77, 78).

A target row at bar ``t`` holds what happens over bars ``t+1 .. t+h``; a
feature row at bar ``t`` holds what is known at ``t``. Joining the two on the
same timestamp therefore pairs feature_t with outcome_{t+h} - the only pairing
the research uses. :func:`check_alignment` enforces identical timestamps before
any statistic is computed, so a shifted join cannot manufacture information.

The table is stored apart from every feature
(``data/targets/timeframe=<tf>/version=<v>/year=YYYY/``), holds only
``timestamp`` and ``target_*`` columns (:func:`~..features.manifest.assert_target_schema`),
and the weekend flags ``target_meta_weekend_<h>`` record when the span crosses
a weekend gap. Nothing from here is ever joined into the feature matrix.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..features.families import BarSeries
from ..features.manifest import assert_target_schema, digest
from ..research.study_io import clean_json, code_fingerprint
from ..utils.clock import utc_now_iso
from ..utils.paths import atomic_write_text, ensure_dir
from .config import TARGET_FAMILIES, TargetConfig
from .event_targets import future_max_down, future_max_up
from .residuals import (
    ou_deviation_reduction,
    residual_change,
    residual_reduction,
    residual_shrinks,
)
from .returns import future_abs_return, future_return
from .volatility import future_realized_vol

__all__ = [
    "TargetInputs",
    "build_targets",
    "check_alignment",
    "load_targets",
    "target_kind",
    "target_version",
    "write_targets",
]

_CURRENT = "CURRENT"
_MANIFEST = "_manifest.json"


@dataclass
class TargetInputs:
    """What the targets are computed from (all known at each t, except what lies ahead)."""

    bars: BarSeries
    residual: np.ndarray | None           # eps_t, N = regression_window
    sigma: np.ndarray | None              # sigma_t, trailing return sd
    ou_mu: np.ndarray | None              # mu_t of the rolling OU fit on bars <= t


def target_kind(column: str) -> str:
    """direction / magnitude / volatility / residual / excursion for a target column."""
    body = column.removeprefix("target_")
    family = body.rsplit("_", 1)[0]
    return TARGET_FAMILIES.get(family, "unknown")


def build_targets(inputs: TargetInputs, tcfg: TargetConfig) -> pl.DataFrame:
    """Every configured target family x horizon, float32, plus weekend flags."""
    bars = inputs.bars
    close = bars.close
    columns: dict[str, np.ndarray] = {}
    for family in tcfg.families:
        for h in tcfg.horizons:
            name = f"target_{family}_{h}"
            if family == "return":
                values = future_return(close, h)
            elif family == "abs_return":
                values = future_abs_return(close, h)
            elif family == "realized_vol":
                values = future_realized_vol(close, h)
            elif family in ("residual_change", "residual_reduction", "residual_shrinks",
                            "ou_deviation_reduction"):
                if inputs.residual is None or inputs.sigma is None:
                    values = np.full(bars.size, np.nan)
                elif family == "residual_change":
                    values = residual_change(inputs.residual, inputs.sigma, h)
                elif family == "residual_reduction":
                    values = residual_reduction(inputs.residual, inputs.sigma, h)
                elif family == "residual_shrinks":
                    values = residual_shrinks(inputs.residual, h)
                else:
                    values = (np.full(bars.size, np.nan) if inputs.ou_mu is None
                              else ou_deviation_reduction(inputs.residual, inputs.ou_mu,
                                                          inputs.sigma, h))
            elif family == "max_up":
                values = (np.full(bars.size, np.nan) if bars.high is None
                          else future_max_up(close, bars.high, h))
            elif family == "max_down":
                values = (np.full(bars.size, np.nan) if bars.low is None
                          else future_max_down(close, bars.low, h))
            else:                                          # pragma: no cover - config-checked
                raise ValueError(f"unknown target family {family!r}")
            arr = np.asarray(values, dtype=np.float32)
            arr[~np.isfinite(arr)] = np.nan
            columns[name] = arr
    gaps = _weekend_gaps(bars.timestamps, tcfg.weekend_gap_hours)
    for h in tcfg.horizons:
        columns[f"target_meta_weekend_{h}"] = _span_contains(gaps, h).astype(np.int8)
    frame = pl.DataFrame({"timestamp": bars.timestamps, **columns})
    assert_target_schema(frame.columns)
    return frame


def _weekend_gaps(timestamps: pl.Series, hours: float) -> np.ndarray:
    """gap[i] = 1 if the step into bar i spans more than *hours*."""
    step = (timestamps.diff().dt.total_seconds().fill_null(0).to_numpy()).astype(np.float64)
    return (step > hours * 3600.0).astype(np.int64)


def _span_contains(gaps: np.ndarray, h: int) -> np.ndarray:
    """True if any of the steps into bars t+1 .. t+h is a gap."""
    csum = np.concatenate(([0], np.cumsum(gaps)))
    n = gaps.size
    out = np.zeros(n, dtype=bool)
    if 0 < h < n:
        t = np.arange(n - h)
        out[:n - h] = (csum[t + h + 1] - csum[t + 1]) > 0
    return out


def check_alignment(features: pl.DataFrame, targets: pl.DataFrame) -> None:
    """Refuse to pair a feature table and a target table unless their bars are identical."""
    if features.height != targets.height or not features["timestamp"].equals(
            targets["timestamp"]):
        raise ValueError("feature and target tables are not on the same bars: refusing to "
                         "pair them (a shifted pairing can manufacture information)")


def target_version(tcfg: TargetConfig, sources: dict[str, Any]) -> str:
    base = Path(__file__).resolve().parent
    code = code_fingerprint(base / name for name in ("alignment.py", "returns.py",
                                                     "residuals.py", "volatility.py",
                                                     "event_targets.py"))
    return f"targets-{sources.get('timeframe')}-" + digest(
        {"config": tcfg.fingerprint(), "sources": sources, "code": code})


def write_targets(frame: pl.DataFrame, manifest: dict[str, Any], root: Path,
                  timeframe: str) -> Path:
    assert_target_schema(frame.columns)
    version = manifest["target_version"]
    base = ensure_dir(root / f"timeframe={timeframe}" / f"version={version}")
    with_year = frame.with_columns(pl.col("timestamp").dt.year().alias("_year"))
    for (year,), part in with_year.group_by("_year", maintain_order=True):
        directory = ensure_dir(base / f"year={year}")
        tmp = directory / "part-0.parquet.partial"
        part.drop("_year").write_parquet(tmp, compression="zstd", compression_level=3)
        tmp.replace(directory / "part-0.parquet")
    atomic_write_text(base / _MANIFEST, json.dumps(clean_json(manifest), indent=1,
                                                   default=str) + "\n")
    atomic_write_text(base.parent / _CURRENT, version + "\n")
    return base


def load_targets(root: Path, timeframe: str, columns: list[str] | None = None
                 ) -> tuple[pl.DataFrame, dict[str, Any]]:
    pointer = root / f"timeframe={timeframe}" / _CURRENT
    if not pointer.exists():
        raise FileNotFoundError(f"no targets for {timeframe} - run `xq build-feature-factory "
                                f"--timeframe {timeframe}`")
    base = root / f"timeframe={timeframe}" / f"version={pointer.read_text().strip()}"
    manifest = json.loads((base / _MANIFEST).read_text(encoding="utf-8"))
    files = sorted(base.glob("year=*/part-0.parquet"))
    wanted = None if columns is None else ["timestamp", *[c for c in columns if c != "timestamp"]]
    frame = pl.read_parquet(files, columns=wanted).sort("timestamp")
    return frame, manifest


def target_manifest(frame: pl.DataFrame, tcfg: TargetConfig, sources: dict[str, Any],
                    started: float) -> dict[str, Any]:
    names = [c for c in frame.columns if c.startswith("target_") and "_meta_" not in c]
    coverage = {c: float(np.isfinite(frame[c].cast(pl.Float64).fill_null(np.nan).to_numpy()
                                     ).mean()) for c in names}
    return {"target_version": target_version(tcfg, sources), "timeframe": sources["timeframe"],
            "rows": frame.height, "first_timestamp": str(frame["timestamp"][0]),
            "last_timestamp": str(frame["timestamp"][-1]), "targets": names,
            "kinds": {c: target_kind(c) for c in names}, "horizons": list(tcfg.horizons),
            "families": list(tcfg.families), "sources": sources,
            "targets_config_fingerprint": tcfg.fingerprint(), "coverage": coverage,
            "outcomes": "every column uses bars after t; never joined into a feature table",
            "seconds": time.perf_counter() - started, "generated_utc": utc_now_iso()}
