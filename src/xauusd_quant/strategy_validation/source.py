"""Causal three-feature bar source; cut development values before materialization.

Only bounded local bars are used. Target ends are subsequent observed bar closes,
not wall-clock interpolation across sessions. No existing selected feature store,
fitted model or reserved-period target loader is reused.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from ..data.timezones import to_utc_expr
from ..execution.config import ExecutionConfig, content_hash
from ..execution.engine import BarClose
from ..execution.io import local_time, validate_interval
from ..execution.readiness import sha256
from ..utils.config import Config
from .pipeline import FeatureRow, TrainingRow
from .plan import ExperimentPlan


def features(opens: Sequence[datetime], closes: Sequence[float], seconds: int) -> list[FeatureRow]:
    result = []
    for i in range(3, len(opens)):
        prices = closes[i - 3 : i + 1]
        if not all(math.isfinite(x) and x > 0 for x in prices):
            continue
        ret = np.diff(np.log(prices))
        result.append(
            FeatureRow(
                opens[i].isoformat(),
                opens[i],
                opens[i] + timedelta(seconds=seconds),
                (float(ret[-1]), float(ret.sum()), float(np.std(ret))),
                i,
            )
        )
    return result


class BarDevelopmentSource:
    def __init__(self, config: Config, execution: ExecutionConfig, plan: ExperimentPlan) -> None:
        self.config, self.execution, self.plan = config, execution, plan
        manifest = json.loads((config.bars_dir(plan.timeframe) / "_manifest.json").read_text())
        tick = json.loads((config.processed_data_path / "_manifest.json").read_text())
        if (
            manifest.get("status") != "complete"
            or manifest.get("tick_dataset_version") != tick.get("dataset_version")
            or manifest.get("convention")
            != {"label": "open", "closed": "left", "price_source": "mid", "time_basis": "timestamp"}
        ):
            raise ValueError("invalid bar identity or opening-time convention")
        if manifest.get("bar_fingerprint") != config.bar_fingerprint():
            raise ValueError("bar settings differ from current source configuration")
        directory = config.bars_dir(plan.timeframe)
        declared = set()
        for entry in manifest.get("years", []):
            file = directory / entry["path"]
            if (
                not file.resolve().is_relative_to(directory.resolve())
                or not file.is_file()
                or file.stat().st_size != entry["bytes"]
                or pq.read_metadata(file).num_rows != entry["rows"]
            ):
                raise ValueError("bar partition identity/coverage mismatch")
            declared.add(file.resolve())
        if not declared or declared != {p.resolve() for p in directory.glob("year=*/*.parquet")}:
            raise ValueError("bar partition files differ from complete manifest")
        self.identity = {
            "bar_dataset_version": manifest["bar_dataset_version"],
            "tick_dataset_version": tick["dataset_version"],
            "timeframe": plan.timeframe,
            "feature_pool": plan.feature_pool,
            "bar_manifest_sha256": sha256(directory / "_manifest.json"),
            "bar_partition_metadata_verified": True,
            "full_content_reverified": False,
        }

    def _bars(self, start: datetime, cutoff: datetime) -> tuple[list[datetime], list[float]]:
        validate_interval(self.config, start, cutoff)
        # UTC predicate at the scan, with source-local year pruning. Close < cutoff.
        lower = start - timedelta(
            days=7
        )  # explicit warm-up allowance, not executable interpolation
        local_lower, local_upper = local_time(lower, self.config), local_time(cutoff, self.config)
        opens: list[datetime] = []
        prices: list[float] = []
        for path in sorted(self.config.bars_dir(self.plan.timeframe).glob("year=*/*.parquet")):
            year = int(path.parent.name.split("=")[1])
            if year >= 2022 or year < local_lower.year or year > local_upper.year:
                continue
            frame = (
                pl.scan_parquet(path)
                .select("timestamp", "close")
                .filter(
                    (pl.col("timestamp") >= local_lower)
                    & (pl.col("timestamp") < datetime(2022, 1, 1))
                )
                .with_columns(to_utc_expr(self.config.timezone))
                .filter(
                    (pl.col("timestamp_utc") >= lower)
                    & (
                        pl.col("timestamp_utc") + pl.duration(seconds=self.execution.bar_seconds)
                        < cutoff
                    )
                )
                .limit(self.plan.max_training_rows - len(opens) + 1)
                .collect()
            )
            for _, close, stamp in frame.iter_rows():
                if stamp is None:
                    raise ValueError("unknown bar-close UTC availability")
                opens.append(stamp)
                prices.append(float(close) if close is not None else math.nan)
        if len(opens) > self.plan.max_training_rows:
            raise ValueError("bounded bar scan exceeds registered memory row cap")
        if any(a >= b for a, b in zip(opens, opens[1:], strict=False)):
            raise ValueError("bar clock must be strictly chronological")
        return opens, prices

    def training(self, start: datetime, cutoff: datetime) -> list[TrainingRow]:
        opens, closes = self._bars(start, cutoff)
        rows = features(opens, closes, self.execution.bar_seconds)
        result = []
        for row in rows:
            j = row.bar_index + self.plan.horizon_bars
            if row.available_at_utc < start or j >= len(opens):
                continue
            if not math.isfinite(closes[j]) or closes[j] <= 0:
                continue
            end = opens[j] + timedelta(seconds=self.execution.bar_seconds)
            result.append(
                TrainingRow(
                    row, math.log(closes[j] / closes[row.bar_index]), row.available_at_utc, end, end
                )
            )
        return result

    def evaluation(self, start: datetime, end: datetime) -> tuple[list[FeatureRow], list[BarClose]]:
        # Strict _bars close < cutoff; an extra microsecond admits a close exactly at end.
        # Do not read price values at/after end: the final boundary bar is scheduling only.
        opens, closes = self._bars(start, end)
        delay = timedelta(milliseconds=self.execution.computation_delay_ms)
        lower = start - timedelta(seconds=self.execution.bar_seconds) - delay
        included = [i for i, stamp in enumerate(opens) if stamp >= lower]
        indices = {old: index for index, old in enumerate(included)}
        timeline = [
            BarClose(opens[i] + timedelta(seconds=self.execution.bar_seconds), indices[i])
            for i in included
        ]
        rows = [
            replace(r, bar_index=indices[r.bar_index])
            for r in features(opens, closes, self.execution.bar_seconds)
            if r.bar_index in indices and start <= r.available_at_utc + delay < end
        ]
        return rows, timeline


class SyntheticSource:
    """Small declared fixture; no claims about real-data loader isolation from this class."""

    def __init__(
        self,
        opens: Sequence[datetime],
        closes: Sequence[float],
        execution: ExecutionConfig,
        horizon: int,
    ) -> None:
        self.opens, self.closes, self.execution, self.horizon = (
            list(opens),
            list(closes),
            execution,
            horizon,
        )
        self.accesses: list[dict[str, Any]] = []
        self.identity = {
            "kind": "declared synthetic",
            "rows": len(opens),
            "fixture_sha256": content_hash(
                {"opens": [t.isoformat() for t in opens], "closes": list(closes)}
            ),
        }

    def training(self, start: datetime, cutoff: datetime) -> list[TrainingRow]:
        self.accesses.append({"kind": "training", "start": start, "cutoff": cutoff})
        permitted = [
            i
            for i, t in enumerate(self.opens)
            if t + timedelta(seconds=self.execution.bar_seconds) < cutoff
        ]
        rows = features(
            [self.opens[i] for i in permitted],
            [self.closes[i] for i in permitted],
            self.execution.bar_seconds,
        )
        result = []
        for row in rows:
            j = row.bar_index + self.horizon
            if row.available_at_utc < start or j >= len(permitted):
                continue
            end = self.opens[permitted[j]] + timedelta(seconds=self.execution.bar_seconds)
            result.append(
                TrainingRow(
                    row,
                    math.log(self.closes[permitted[j]] / self.closes[permitted[row.bar_index]]),
                    row.available_at_utc,
                    end,
                    end,
                )
            )
        return result

    def evaluation(self, start: datetime, end: datetime) -> tuple[list[FeatureRow], list[BarClose]]:
        self.accesses.append({"kind": "features_only", "start": start, "end": end})
        delay = timedelta(milliseconds=self.execution.computation_delay_ms)
        lower = start - timedelta(seconds=self.execution.bar_seconds) - delay
        included = [
            i
            for i, t in enumerate(self.opens)
            if lower <= t and t + timedelta(seconds=self.execution.bar_seconds) < end
        ]
        indices = {old: index for index, old in enumerate(included)}
        # Derive only permitted features, never outcomes at/after cutoff.
        prefix = [
            i
            for i, t in enumerate(self.opens)
            if t + timedelta(seconds=self.execution.bar_seconds) < end
        ]
        all_rows = features(
            [self.opens[i] for i in prefix],
            [self.closes[i] for i in prefix],
            self.execution.bar_seconds,
        )
        rows = [
            replace(r, bar_index=indices[r.bar_index])
            for r in all_rows
            if r.bar_index in indices and start <= r.available_at_utc + delay < end
        ]
        timeline = [
            BarClose(self.opens[i] + timedelta(seconds=self.execution.bar_seconds), indices[i])
            for i in included
        ]
        return rows, timeline
