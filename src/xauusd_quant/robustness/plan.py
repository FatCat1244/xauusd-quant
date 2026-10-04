"""Immutable bounded robustness hypotheses registered before new diagnostic results."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from ..execution.config import content_hash
from ..execution.io import write_new_json
from ..execution.readiness import sha256

PAIRS = (
    ("ARX", "ZERO"),
    ("ARX", "STAGE13_RIDGE"),
    ("KALMAN", "ZERO"),
    ("KALMAN", "STAGE13_RIDGE"),
    ("GARCH", "ROLLING"),
    ("GARCH", "EWMA"),
    ("HAR", "ROLLING"),
    ("HAR", "EWMA"),
)


@dataclass(frozen=True)
class RobustnessPlan:
    plan_id: str
    source_runs: tuple[str, ...]
    pairs: tuple[tuple[str, str], ...] = PAIRS
    block_lengths: tuple[int, ...] = (4, 12, 24)
    bootstrap_replicates: int = 199
    monte_carlo_paths: int = 16
    discovery_replicates: int = 12
    seed: int = 140014
    minimum_days: int = 5
    minimum_rows: int = 200
    minimum_blocks: int = 8
    trial_budget: int = 256
    maximum_records: int = 25000
    wall_budget_seconds: float = 240.0
    private_memory_budget_bytes: int = 1073741824
    classification: str = "retrospective_previously_inspected"
    revision_reason: str = "initial bounded Stage 14 saved-development diagnostics"
    econometric_controls: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.econometric_controls) is not bool
            or type(self.private_memory_budget_bytes) is not int
        ):
            raise ValueError("explicit boolean controls and integer memory budget required")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*_V\d{3,}", self.plan_id):
            raise ValueError("safe versioned plan required")
        if not self.source_runs or len(self.source_runs) > 4:
            raise ValueError("1..4 explicit saved development runs required")
        for name in self.source_runs:
            if not re.fullmatch(r"ECON_(5M|15M)_DIAGNOSTIC_V\d{3,}", name):
                raise ValueError("only declared development econometric runs supported")
        if self.pairs != PAIRS:
            raise ValueError(
                "complete fixed predictive family required, including unfavorable pairs"
            )
        if self.block_lengths != (4, 12, 24):
            raise ValueError("fixed block sensitivity 4/12/24; revisions need code and new plan")
        for value in (
            self.bootstrap_replicates,
            self.monte_carlo_paths,
            self.discovery_replicates,
            self.minimum_days,
            self.minimum_rows,
            self.minimum_blocks,
            self.trial_budget,
            self.maximum_records,
        ):
            if type(value) is not int or value < 1:
                raise ValueError("positive integer evidence/budget requirements")
        if (
            self.bootstrap_replicates > 999
            or self.monte_carlo_paths > 32
            or self.discovery_replicates > 32
            or self.maximum_records > 50000
        ):
            raise ValueError("bounded computation only")
        if not 0 < self.wall_budget_seconds <= 600 or self.private_memory_budget_bytes <= 0:
            raise ValueError("finite bounded resources required")
        if self.classification != "retrospective_previously_inspected" or not self.revision_reason:
            raise ValueError("historical inspection cannot be erased")

    def resolved(self) -> dict[str, Any]:
        fields = asdict(self)
        if not self.econometric_controls:
            fields.pop("econometric_controls")
        body = {
            **fields,
            "candidate_universe": [
                "CAUSAL_RIDGE",
                "HISTORICAL_MEAN",
                "ARX",
                "KALMAN",
                "GARCH",
                "HAR",
                "LEGACY_ML_ENSEMBLES",
            ],
            "benchmarks": [
                "ZERO",
                "STAGE13_RIDGE",
                "ROLLING",
                "EWMA",
                "no_trading",
                "random_direction",
            ],
            "selection_exposure": "record all attempts; historical global feature/count/universe selection; effective trials unknown",
            "periods": "saved 2021 development records only; synthetic 2021 clocks; no new market outcome reads",
            "families": "predictive: all eight pairs at both timeframes; economic: ridge/mean fixed policies versus no trading; never combine p-values",
            "losses": "aligned MSE for return; QLIKE for variance; economic marked-equity change through Stage12",
            "uncertainty": "segmented moving-block percentile intervals 4/12/24; conditional frozen forecasts, not discovery correction",
            "correction": "Holm only for complete family with externally justified valid p-values; bootstrap intervals are not p-values",
            "stress": "base, double commission, extra .02 USD/oz slippage, 2000ms latency, double spread, financing +2 account/lot/day, 10% opportunity misses, declared 30s feed interruption",
            "scenario_distributions": "hypothetical independent path-level uniform commission 1..2x, extra slippage 0...04, latency 100..2000ms, spread 1..2x, financing +0..4, missed forecasts Bernoulli(.1); seed fixed; not measured broker behavior",
            "perturbations": "synthetic fold-local ridge alpha .5/1/2 and drop momentum group with fixed return/volatility pool; thresholds 0/1e-5; whole neighborhood retained",
            "nulls": "synthetic zero-relationship versus AR(.5) prices; repeat feature construction, nested count choice, scaling and fitting; economic policy through engine; no claim these replace matched real RW/sign-flip controls",
            "concentration": "top-three positive trade share, positive month share, monthly PnL, first/last chronological half; removal is retrospective attribution",
            "acceptance": "BLOCKED on missing gates/artifacts; INCONCLUSIVE below minima or unavailable inference; REJECTED on complete declared negative effects; survival requires all block lower bounds>0, >=60% positive folds, relative MSE>=.01 or QLIKE>=.01 and all evidence gates; economic survival requires verified terms/nulls and positive adverse scenarios",
            "stopping": "fixed family, no expansion for a failure; invariant/resource failure stops; Stage15 requires separate instruction",
            "unsupported_methods": "DSR/PBO/RealityCheck/SPA unavailable: historical adaptive trial universe unknown or incompatible; no invented effective independent trial count",
        }
        if self.econometric_controls:
            body["econometric_sensitivity"] = (
                "fixed synthetic Kalman q multiplier .5/1/2 fitted on preceding 300 log prices; residual window64/128 x gamma0/.01 with delayed labels; CUSUM threshold6/8/10 x 12 fixed seeds under unchanged/shifted Gaussian errors; no model/neighbor selection; measured market calibration remains insufficient"
            )
        return body


def load_plan(path: Path) -> RobustnessPlan:
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ValueError("robustness plan mapping required")
    body["source_runs"] = tuple(body["source_runs"])
    if "block_lengths" in body:
        body["block_lengths"] = tuple(body["block_lengths"])
    if "pairs" in body:
        body["pairs"] = tuple(tuple(p) for p in body["pairs"])
    return RobustnessPlan(**body)


def freeze_plan(root: Path, plan: RobustnessPlan) -> Path:
    path = root / "results/robustness/plans" / f"{plan.plan_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = plan.resolved()
    digest = content_hash(body)
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("content_sha256") != digest or content_hash(previous["plan"]) != digest:
            raise ValueError("immutable robustness plan changed; bump version")
    else:
        write_new_json(
            path, {"plan": body, "content_sha256": digest, "registered_utc": datetime.now(UTC)}
        )
    # Hash bytes without decoding outcome values. Bind saved evidence before the
    # analysis opens values; replays fail before reading a changed source artifact.
    sources = {}
    for run_id in plan.source_runs:
        directory = root / "results/econometrics/runs" / run_id
        files = [
            directory / name
            for name in ("request.json", "manifest.json", "readiness.json", "forecast_scores.jsonl")
        ]
        files.extend(sorted(directory.glob("frozen/*/*.json")))
        sources[run_id] = {
            str(file.relative_to(directory)): sha256(file) for file in files if file.is_file()
        }
    source_file = path.with_name(f"{plan.plan_id}_SOURCES.json")
    if source_file.exists():
        previous_sources = json.loads(source_file.read_text(encoding="utf-8"))
        if previous_sources["files"] != sources:
            raise ValueError("immutable source artifacts changed; new plan and audit required")
    else:
        write_new_json(
            source_file,
            {
                "files": sources,
                "registered_utc": datetime.now(UTC),
                "scope": "byte hashes only; outcomes previously inspected, no freshness claim",
            },
        )
    return path
