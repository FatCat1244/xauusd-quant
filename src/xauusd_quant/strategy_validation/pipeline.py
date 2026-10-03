"""Fold-local feature identities/counts and models; selectors see only purged preceding data.

The small new ridge/mean path does not import global selection manifests or ensemble
universes. Prediction features and training labels travel through separate interfaces.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Protocol

import numpy as np

from ..execution.config import content_hash
from ..execution.engine import BarClose
from .plan import FEATURE_POOL, Candidate, ExperimentPlan, Fold


@dataclass(frozen=True)
class FeatureRow:
    row_id: str
    bar_open_utc: datetime
    available_at_utc: datetime
    features: tuple[float, ...]
    bar_index: int = -1


@dataclass(frozen=True)
class TrainingRow:
    row: FeatureRow
    label: float
    label_start_utc: datetime
    label_end_utc: datetime
    label_available_at_utc: datetime


class DevelopmentSource(Protocol):
    def training(self, start: datetime, cutoff: datetime) -> Sequence[TrainingRow]: ...
    def evaluation(
        self, start: datetime, end: datetime
    ) -> tuple[list[FeatureRow], list[BarClose]]: ...


@dataclass(frozen=True)
class Selection:
    names: tuple[str, ...]
    as_of_utc: datetime
    used_row_ids: tuple[str, ...]


Selector = Callable[[Sequence[TrainingRow], int], Selection]


def purge(rows: Sequence[TrainingRow], start: datetime, cutoff: datetime) -> list[TrainingRow]:
    """Actual [label_start,label_end] intervals and label publication, not a row-count gap."""
    result = []
    for sample in rows:
        if (
            sample.label_end_utc < sample.label_start_utc
            or sample.label_available_at_utc < sample.label_end_utc
        ):
            raise ValueError("invalid label information interval")
        if (
            start <= sample.row.available_at_utc < cutoff
            and sample.label_start_utc >= sample.row.available_at_utc
            and sample.label_end_utc < cutoff
            and sample.label_available_at_utc < cutoff
        ):
            if not math.isfinite(sample.label) or not all(
                math.isfinite(x) for x in sample.row.features
            ):
                continue
            result.append(sample)
    return result


def select_features(rows: Sequence[TrainingRow], count: int) -> Selection:
    y = np.asarray([r.label for r in rows], dtype=float)
    x = np.asarray([r.row.features for r in rows], dtype=float)
    yc = y - y.mean()
    xc = x - x.mean(axis=0)
    den = np.sqrt((xc * xc).sum(axis=0) * (yc * yc).sum())
    scores = np.divide(
        np.abs((xc * yc[:, None]).sum(axis=0)), den, out=np.zeros(len(FEATURE_POOL)), where=den > 0
    )
    indices = sorted(range(len(FEATURE_POOL)), key=lambda i: (-scores[i], i))[:count]
    return Selection(
        tuple(FEATURE_POOL[i] for i in indices),
        max(r.label_available_at_utc for r in rows),
        tuple(r.row.row_id for r in rows),
    )


def validate_selection(
    selection: Selection, rows: Sequence[TrainingRow], count: int, cutoff: datetime
) -> None:
    if selection.as_of_utc >= cutoff:
        raise ValueError("leaking feature selector: as_of is not preceding cutoff")
    if not set(selection.used_row_ids) <= {r.row.row_id for r in rows}:
        raise ValueError("leaking feature selector: used rows outside permitted training")
    if (
        len(selection.names) != count
        or len(set(selection.names)) != count
        or not set(selection.names) <= set(FEATURE_POOL)
    ):
        raise ValueError("selector identities/count outside registered family")


@dataclass(frozen=True)
class FrozenPipeline:
    candidate_id: str
    fold_id: str
    cutoff_utc: datetime
    features: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    training_ids_sha256: str
    training_label_as_of_utc: datetime
    training_rows: int
    inner_trials: tuple[dict[str, Any], ...] = ()

    def predict(self, row: FeatureRow, *, prediction_at: datetime | None = None) -> float:
        if prediction_at is not None and row.available_at_utc > prediction_at:
            raise ValueError("prediction used unavailable features")
        if (prediction_at or row.available_at_utc) < self.cutoff_utc:
            raise ValueError("pipeline cannot score before its fitting cutoff")
        if not self.features:
            return self.intercept
        x = np.asarray([row.features[FEATURE_POOL.index(name)] for name in self.features])
        return float(self.intercept + ((x - self.means) / self.scales) @ self.coefficients)

    def resolved(self) -> dict[str, Any]:
        body = asdict(self)
        for key in ("cutoff_utc", "training_label_as_of_utc"):
            body[key] = body[key].isoformat()
        body["inner_trials"] = [{**r, "cutoff": r["cutoff"].isoformat()} for r in self.inner_trials]
        return body


def fit(
    rows: Sequence[TrainingRow],
    candidate: Candidate,
    fold: Fold,
    count: int,
    cutoff: datetime,
    plan: ExperimentPlan,
    selector: Selector = select_features,
) -> FrozenPipeline:
    if len(rows) < 8:
        raise ValueError("at least eight finite purged training examples required")
    if any(r.row.available_at_utc >= cutoff or r.label_available_at_utc >= cutoff for r in rows):
        raise ValueError("fit received future information")
    selection = selector(rows, count)
    validate_selection(selection, rows, count, cutoff)
    features = selection.names if candidate.model == "ridge" else ()
    x = np.asarray([[r.row.features[FEATURE_POOL.index(n)] for n in features] for r in rows])
    y = np.asarray([r.label for r in rows])
    intercept = float(y.mean())
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    z = (x - mean) / scale
    coef = np.linalg.solve(
        z.T @ z + plan.ridge_alpha * np.eye(len(features)), z.T @ (y - intercept)
    )
    return FrozenPipeline(
        candidate.candidate_id,
        fold.fold_id,
        cutoff,
        features,
        tuple(mean),
        tuple(scale),
        tuple(coef),
        intercept,
        content_hash([r.row.row_id for r in rows]),
        max(r.label_available_at_utc for r in rows),
        len(rows),
    )


Trial = Callable[[str, dict[str, Any], Callable[[], Any]], Any]


def fit_fold(
    source: DevelopmentSource,
    plan: ExperimentPlan,
    fold: Fold,
    candidate: Candidate,
    trial: Trial,
    selector: Selector = select_features,
) -> FrozenPipeline:
    """Inner MSE chooses count only. Every adaptive choice is repeated within each inner split."""
    scores: dict[int, list[float]] = {n: [] for n in plan.feature_counts}
    inner_records: list[dict[str, Any]] = []
    for inner_number, (a, b) in enumerate(fold.inner_intervals):
        from ..execution.io import parse_utc

        start, end = parse_utc(a), parse_utc(b)
        train = purge(
            source.training(plan.training_start(fold, start), start),
            plan.training_start(fold, start),
            start,
        )
        # Inner labels are fully observed before inner end, itself before outer start.
        validation = purge(source.training(start, end), start, end)
        for count in plan.feature_counts:
            spec = {
                "fold": fold.fold_id,
                "candidate": candidate.candidate_id,
                "inner": inner_number,
                "feature_count": count,
                "cutoff": start,
            }

            def evaluate(
                count: int = count,
                train: Sequence[TrainingRow] = train,
                start: datetime = start,
                validation: Sequence[TrainingRow] = validation,
            ) -> float:
                pipeline = fit(train, candidate, fold, count, start, plan, selector)
                if len(validation) < 4:
                    raise ValueError("insufficient purged inner validation")
                return float(
                    np.mean([(pipeline.predict(r.row) - r.label) ** 2 for r in validation])
                )

            value = trial("inner_fit", spec, evaluate)
            inner_records.append({**spec, "mse": value})
            if value is not None and math.isfinite(value):
                scores[count].append(value)
    eligible = [n for n, values in scores.items() if len(values) == len(fold.inner_intervals)]
    if not eligible:
        raise ValueError("all registered inner choices failed; candidate remains recorded")
    chosen = min(eligible, key=lambda n: (float(np.mean(scores[n])), n))
    start = plan.training_start(fold, fold.start)
    rows = purge(source.training(start, fold.start), start, fold.start)
    pipeline = fit(rows, candidate, fold, chosen, fold.start, plan, selector)
    from dataclasses import replace

    return replace(pipeline, inner_trials=tuple(inner_records))
