"""Regular-grid measurements and separate labels; no interpolation or cross-gap returns."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from ..execution.engine import BarClose
from ..execution.policy import utc_time
from ..strategy_validation.pipeline import FeatureRow, TrainingRow, purge


@dataclass(frozen=True)
class Point:
    row_id: str
    bar_open_utc: datetime
    available_utc: datetime
    index: int
    log_price: float
    returns: tuple[float, ...]  # most recent 12 grid returns, oldest first

    @property
    def arx(self) -> tuple[float, ...]:
        return self.returns[-1], self.returns[-2], math.sqrt(self.rv[2])

    @property
    def rv(self) -> tuple[float, ...]:
        return tuple(float(np.mean(np.square(self.returns[-n:]))) for n in (1, 3, 12))

    @property
    def stage13(self) -> FeatureRow:
        last = self.returns[-3:]
        return FeatureRow(
            self.row_id,
            self.bar_open_utc,
            self.available_utc,
            (last[-1], sum(last), float(np.std(last))),
            self.index,
        )


@dataclass(frozen=True)
class Outcome:
    row_id: str
    start_utc: datetime
    end_utc: datetime
    matured_utc: datetime
    log_return: float
    realized_variance: float
    horizon_bars: int
    seconds: int


@dataclass(frozen=True)
class Sample:
    point: Point
    outcome: Outcome


class GridData:
    """Small bounded permitted bars; inference features are computed from each prefix only."""

    def __init__(self, opens: Sequence[datetime], prices: Sequence[float], seconds: int) -> None:
        if seconds not in (300, 900) or len(opens) != len(prices):
            raise ValueError("incompatible price/grid arrays")
        if not all(math.isfinite(p) and p > 0 for p in prices):
            raise ValueError("invalid/nonpositive midpoint bar; no silent imputation")
        self.opens = [utc_time(t) for t in opens]
        self.logs = np.log(np.asarray(prices, dtype=float)) if len(prices) else np.empty(0)
        self.seconds = seconds
        if any(a >= b for a, b in zip(self.opens, self.opens[1:], strict=False)):
            raise ValueError("bar timestamps must be strictly chronological")
        if any(t.microsecond or int(t.timestamp()) % seconds for t in self.opens):
            raise ValueError("bar opening times do not lie on the declared grid")
        self.closes = [t + timedelta(seconds=seconds) for t in self.opens]
        self.contiguous = (
            np.asarray(
                [False]
                + [
                    b - a == timedelta(seconds=seconds)
                    for a, b in zip(self.opens, self.opens[1:], strict=False)
                ]
            )
            if self.opens
            else np.empty(0, dtype=bool)
        )
        self.returns = np.full(len(self.opens), np.nan)
        if len(self.opens) > 1:
            self.returns[1:] = np.where(self.contiguous[1:], np.diff(self.logs), np.nan)

    def points(self, start: datetime, end: datetime) -> list[Point]:
        result = []
        for i in range(12, len(self.opens)):
            if start <= self.closes[i] < end and self.contiguous[i - 11 : i + 1].all():
                result.append(
                    Point(
                        self.opens[i].isoformat(),
                        self.opens[i],
                        self.closes[i],
                        i,
                        float(self.logs[i]),
                        tuple(self.returns[i - 11 : i + 1]),
                    )
                )
        return result

    def outcomes(self, horizon: int, delay_ms: int, cutoff: datetime) -> dict[str, Outcome]:
        if not 1 <= horizon <= 12 or delay_ms < 0:
            raise ValueError("invalid outcome horizon/publication delay")
        result = {}
        for i in range(len(self.opens) - horizon):
            j = i + horizon
            end = self.closes[j]
            matured = end + timedelta(milliseconds=delay_ms)
            if matured >= cutoff or not self.contiguous[i + 1 : j + 1].all():
                continue
            result[self.opens[i].isoformat()] = Outcome(
                self.opens[i].isoformat(),
                self.closes[i],
                end,
                matured,
                float(self.logs[j] - self.logs[i]),
                float(np.square(self.returns[i + 1 : j + 1]).sum()),
                horizon,
                self.seconds,
            )
        return result

    def samples(
        self, start: datetime, cutoff: datetime, horizon: int, delay_ms: int
    ) -> list[Sample]:
        labels = self.outcomes(horizon, delay_ms, cutoff)
        raw = [
            Sample(p, labels[p.row_id]) for p in self.points(start, cutoff) if p.row_id in labels
        ]
        # Reuse Stage 13's actual-information-interval guard, including delayed publication.
        safe = purge(
            [
                TrainingRow(
                    s.point.stage13,
                    s.outcome.log_return,
                    s.outcome.start_utc,
                    s.outcome.end_utc,
                    s.outcome.matured_utc,
                )
                for s in raw
            ],
            start,
            cutoff,
        )
        permitted = {r.row.row_id for r in safe}
        return [s for s in raw if s.point.row_id in permitted]

    def return_segments(self, start: datetime, cutoff: datetime) -> list[np.ndarray]:
        indices = [
            i
            for i, t in enumerate(self.closes)
            if start <= t < cutoff and math.isfinite(self.returns[i])
        ]
        groups: list[list[int]] = []
        for i in indices:
            if not groups or i != groups[-1][-1] + 1:
                groups.append([])
            groups[-1].append(i)
        return [self.returns[g].copy() for g in groups]

    def level_segments(self, start: datetime, cutoff: datetime) -> list[np.ndarray]:
        groups: list[list[int]] = []
        for i, t in enumerate(self.closes):
            if not start <= t < cutoff:
                continue
            if not groups or not self.contiguous[i]:
                groups.append([])
            groups[-1].append(i)
        return [self.logs[g].copy() for g in groups]

    def noise_diagnostic(self, start: datetime, cutoff: datetime) -> dict[str, Any]:
        pieces = [p for p in self.return_segments(start, cutoff) if len(p) >= 6]
        if not pieces:
            return {"status": "unknown", "reason": "no contiguous training return segments"}
        a, b = np.concatenate([x[:-1] for x in pieces]), np.concatenate([x[1:] for x in pieces])
        rho = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else None
        fine, coarse = 0.0, 0.0
        for x in pieces:
            usable = x[: len(x) // 3 * 3].reshape(-1, 3)
            fine += float(np.square(usable).sum())
            coarse += float(np.square(usable.sum(axis=1)).sum())
        return {
            "status": "measured",
            "as_of_utc": cutoff,
            "grid_seconds": self.seconds,
            "lag1_return_correlation": rho,
            "coarse_3grid_rv_to_fine_ratio": coarse / fine if fine > 0 else None,
            "interpretation": "training-only signature; aggregation anchored at each contiguous segment, not optimized; neither ratio nor correlation identifies microstructure noise alone",
        }


Reader = Callable[[datetime, datetime], GridData]


class RidgeSource:
    """Same Stage 13 selector/model path, with this extension's regular-grid target alignment."""

    def __init__(self, reader: Reader, horizon: int, delay_ms: int) -> None:
        self.reader, self.horizon, self.delay_ms = reader, horizon, delay_ms

    def training(self, start: datetime, cutoff: datetime) -> list[TrainingRow]:
        data = self.reader(start, cutoff)
        return [
            TrainingRow(
                s.point.stage13,
                s.outcome.log_return,
                s.outcome.start_utc,
                s.outcome.end_utc,
                s.outcome.matured_utc,
            )
            for s in data.samples(start, cutoff, self.horizon, self.delay_ms)
        ]

    def evaluation(self, start: datetime, end: datetime) -> tuple[list[FeatureRow], list[BarClose]]:
        data = self.reader(start, end)
        return [p.stage13 for p in data.points(start, end)], [
            BarClose(t, i) for i, t in enumerate(data.closes)
        ]
