"""Finite-window canonical features on the new UTC feed; unsupported state blocks.

This intentionally does not import historical feature stores or final-test
loaders. Regime/time/recursive/wavelet state requires additional compatible
live services; registry flags alone do not provide those services.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from ..execution.config import content_hash
from ..features.factory_config import FeatureFactoryConfig
from ..features.families import (
    BarSeries,
    autocorrelation_features,
    microstructure_features,
    returns_features,
)
from ..features.registry import build_registry
from .bars import CompletedBar, restore_values, serialize_bar

SUPPORTED = {"returns": returns_features, "autocorrelation": autocorrelation_features,
             "microstructure": microstructure_features}


class Features:
    def __init__(self, config: FeatureFactoryConfig, timeframe: str,
                 names: tuple[str, ...] = ("ret_1",)) -> None:
        self.config, self.timeframe, self.names = config, timeframe, names
        registry = {s.name: s for s in build_registry(config, timeframe, int(timeframe[:-1]) * 60)}
        if not names or len(set(names)) != len(names) or any(n not in registry for n in names):
            raise ValueError("explicit registered feature dependencies required")
        self.specs = [registry[n] for n in names]
        if any(s.family not in SUPPORTED or not s.live_safe for s in self.specs):
            raise ValueError("required feature/context has no validated Stage17 streaming service")
        self.history_required = max(s.min_history for s in self.specs)
        if self.history_required > 2048:
            raise ValueError("feature history exceeds bounded Stage17 budget")
        self.history: list[dict[str, Any]] = []
        self.identity = content_hash({"features": [s.to_dict() for s in self.specs],
                                      "time_basis": "MT5 UTC", "buffer": self.history_required,
                                      "method": "canonical finite trailing windows V001"})

    def update(self, bar: CompletedBar) -> dict[str, Any]:
        if bar.timeframe != self.timeframe:
            raise ValueError("feature timeframe mismatch")
        if self.history and bar.open_utc <= self.history[-1]["timestamp"]:
            raise ValueError("completed feature bar duplicated or out of order")
        if not bar.valid or (self.history and (bar.open_utc - self.history[-1]["timestamp"]).total_seconds() != int(self.timeframe[:-1]) * 60):
            self.history.clear()  # Never bridge gaps with artificial flat bars.
        if bar.valid:
            self.history.append(bar.values)
        self.history = self.history[-self.history_required:]
        values: dict[str, float | None] = dict.fromkeys(self.names)
        if self.history:
            frame = pl.DataFrame(self.history)
            arrays = {name: frame[name].to_numpy().astype(float) for name in (
                "open", "high", "low", "close", "median_spread", "tick_count")}
            series = BarSeries("exness_read_only", self.timeframe, frame["timestamp"],
                               arrays["close"], int(self.timeframe[:-1]) * 60,
                               open=arrays["open"], high=arrays["high"], low=arrays["low"],
                               median_spread=arrays["median_spread"], tick_count=arrays["tick_count"])
            computed = {}
            for family in sorted({s.family for s in self.specs}):
                computed.update(SUPPORTED[family](series, self.config))
            for name in self.names:
                value = float(np.float32(computed[name][-1]))
                values[name] = value if np.isfinite(value) else None
        return {"feature_identity": self.identity, "warmup_bars": len(self.history),
                "required_bars": self.history_required,
                "ready": len(self.history) >= self.history_required and all(v is not None for v in values.values()),
                "values": values, "available_utc": bar.available_utc.isoformat()}

    def state(self) -> dict[str, Any]:
        return {"identity": self.identity, "history": [serialize_bar(CompletedBar(
            self.timeframe, row, row["timestamp"], True, True, ())) ["values"] for row in self.history]}

    def restore(self, body: dict[str, Any]) -> None:
        if body["identity"] != self.identity or len(body["history"]) > self.history_required:
            raise ValueError("feature checkpoint incompatible")
        rows = [restore_values(r) for r in body["history"]]
        if any(b["timestamp"] <= a["timestamp"] for a, b in zip(rows, rows[1:], strict=False)):
            raise ValueError("feature checkpoint not chronological")
        self.history = rows
