"""Hash-bound evidence inventory; no scientific status follows from code readiness."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..execution.io import parse_utc
from ..execution.policy import utc_time
from ..execution.readiness import sha256
from .plan import PortfolioPlan, freeze, freeze_plan

ELIGIBLE = "ELIGIBLE FOR DECLARED HISTORICAL PORTFOLIO RESEARCH"
SURVIVES = "SURVIVES DECLARED HISTORICAL TESTS"


@dataclass(frozen=True)
class Alpha:
    alpha_id: str
    specification_id: str
    timeframe: str
    horizon_bars: int
    policy: dict[str, Any]
    provenance: dict[str, Any]
    stage14_verdict: str
    eligibility: str
    evidence_available_utc: str | None
    limitations: tuple[str, ...]
    software_ready: bool = True
    diagnostic_only: bool = False
    synthetic: bool = False

    def eligible_at(self, cutoff: datetime) -> bool:
        cutoff = utc_time(cutoff)
        required = ("model", "features", "data", "chronology", "nulls", "economics", "execution")
        settings = self.policy
        maximum_age = settings.get("maximum_age_seconds")
        threshold = settings.get("threshold")
        complete_policy = (
            settings.get("forecast_target") == "future_return"
            and settings.get("forecast_units") == "log_mid_return"
            and settings.get("timeframe") == self.timeframe
            and settings.get("horizon_bars") == self.horizon_bars
            and type(maximum_age) is int
            and maximum_age > 0
            and isinstance(threshold, (int, float))
            and not isinstance(threshold, bool)
            and math.isfinite(threshold)
            and threshold >= 0
            and all(
                settings.get(key)
                for key in (
                    "id",
                    "availability",
                    "entry",
                    "holding",
                    "exit",
                    "overlap",
                    "exposure",
                    "execution",
                    "conditions",
                )
            )
        )
        return (
            complete_policy
            and not self.synthetic
            and not self.diagnostic_only
            and self.software_ready
            and self.eligibility == ELIGIBLE
            and self.stage14_verdict == SURVIVES
            and self.evidence_available_utc is not None
            and parse_utc(self.evidence_available_utc) < cutoff
            and all(self.provenance.get(key) for key in required)
            and all(self.provenance.get("gates", {}).get(key) == "passed" for key in required)
        )

    def resolved(self) -> dict[str, Any]:
        return asdict(self)


def policy(timeframe: str, horizon: int, threshold: float = 0) -> dict[str, Any]:
    return {
        "id": f"SIGNED_RETURN_{timeframe.upper()}_H{horizon}_T{threshold:g}_V001",
        "forecast_target": "future_return",
        "forecast_units": "log_mid_return",
        "availability": "bar close plus declared publication/computation delay",
        "entry": "sign if abs(expected_return)>fixed threshold else abstain",
        "threshold": threshold,
        "holding": "h subsequent observed bar closes",
        "exit": "observed holding end OR explicit maximum staleness/invalid evidence",
        "overlap": "ignore fresh directional forecasts during active intent; invalid clears",
        "horizon_bars": horizon,
        "timeframe": timeframe,
        "exposure": "signed unit research budget fraction, weighted then netted",
        "execution": "Stage12 Bid/Ask, strict latency/TTL/gap, full close before resize",
        "conditions": "verified return provenance, prior-only eligibility, valid fresh forecasts",
        "maximum_age_seconds": None,
    }


def build_registry(root: Path, plan: PortfolioPlan) -> dict[str, Any]:
    frozen = freeze_plan(root, plan)
    index = json.loads(frozen.with_name(f"{plan.plan_id}_SOURCES.json").read_text())["body"]
    directory = root / "results/robustness/runs" / plan.evidence_run
    for name, digest in index["files"].items():
        file = directory / name
        if digest is None or not file.is_file() or sha256(file) != digest:
            raise ValueError("missing or changed Stage14 evidence; no candidate promotion")
    verdict = json.loads((directory / "verdict.json").read_text(encoding="utf-8"))
    inventory = json.loads((directory / "inventory.json").read_text(encoding="utf-8"))
    candidates: list[Alpha] = []
    metadata = inventory["metadata"]
    data = metadata["dataset_version"]
    refs: dict[str, str] = {
        str((directory / name).relative_to(root)): digest for name, digest in index["files"].items()
    }
    for entry in metadata.get("evaluation_history", []):
        file = root / entry["path"]
        if not file.is_file() or sha256(file) != entry["sha256"]:
            raise ValueError("reserved access history changed since evidence audit")
        refs[entry["path"]] = entry["sha256"]
    upstream: dict[str, dict[str, str | None]] = {}
    for namespace, runs in (
        (
            "strategy_validation",
            (
                "STRATEGY_5M_DIAGNOSTIC_V001",
                "STRATEGY_5M_DIAGNOSTIC_V002",
                "STRATEGY_15M_DIAGNOSTIC_V002",
            ),
        ),
        ("econometrics", ("ECON_5M_DIAGNOSTIC_V001", "ECON_15M_DIAGNOSTIC_V001")),
        ("execution", ("EXEC_SMOKE_V002", "EXEC_QUOTE_PROBE_V002")),
    ):
        for run_id in runs:
            parent = root / "results" / namespace
            source = parent / "runs" / run_id if namespace != "execution" else parent / run_id
            entries = {}
            for name in (
                "manifest.json",
                "request.json",
                "readiness.json",
                "verdict.json",
                "summary.json",
                "trial_ledger.jsonl",
                "source_identity.json",
            ):
                file = source / name
                relative = str(file.relative_to(root))
                digest = sha256(file) if file.is_file() else None
                entries[relative] = digest
                if digest is not None:
                    refs[relative] = digest
            upstream[run_id] = entries

    def add(
        identity: str,
        timeframe: str,
        horizon: int,
        status: dict[str, Any],
        provenance: dict[str, Any],
        diagnostic: bool = False,
        threshold: float = 0,
    ) -> None:
        state = status.get("status", "BLOCKED")
        # Stage14 forecast survival alone never supplies portfolio economic evidence.
        eligibility = state if state in ("BLOCKED", "INCONCLUSIVE", "REJECTED") else "BLOCKED"
        reasons = status.get("reasons", ["matching policy economic evidence unavailable"])
        candidates.append(
            Alpha(
                identity,
                f"{identity}_SPEC_{plan.registry_id.rsplit('_', 1)[-1]}",
                timeframe,
                horizon,
                {} if diagnostic else policy(timeframe, horizon, threshold),
                {
                    "data": data,
                    **provenance,
                    "selection_exposure": inventory["selection_exposure_quantification"],
                    "historical_selection": inventory["retrospective_selection"],
                },
                state,
                eligibility,
                None,
                (
                    *reasons,
                    "no reconstructed fold-local eligibility; outcomes previously inspected",
                    "matching economic evidence and verified execution assumptions required",
                ),
                diagnostic_only=diagnostic,
            )
        )

    for timeframe in ("5m", "15m"):
        for model in ("CAUSAL_RIDGE", "HISTORICAL_MEAN"):
            run = f"STRATEGY_{timeframe.upper()}_DIAGNOSTIC_V002"
            artifacts = sorted(
                (root / "results/strategy_validation/runs" / run / "frozen").glob(f"*/{model}.json")
            )
            model_refs = {str(f.relative_to(root)): sha256(f) for f in artifacts}
            refs.update(model_refs)
            features = {}
            for file in artifacts:
                spec = json.loads(file.read_text(encoding="utf-8"))
                features[spec["fold_id"]] = {
                    "names": spec["features"],
                    "training_ids_sha256": spec["training_ids_sha256"],
                    "training_label_as_of_utc": spec["training_label_as_of_utc"],
                    "cutoff_utc": spec["cutoff_utc"],
                }
            for threshold in (0.0, 0.00001):
                add(
                    f"{model}_{timeframe.upper()}_T{threshold:g}",
                    timeframe,
                    1,
                    verdict["candidate_verdicts"][model],
                    {
                        "model": model_refs,
                        "missing_frozen_fit_artifacts": not bool(model_refs),
                        "declared_model": model,
                        "chronology": upstream[run],
                        "features": features,
                        "policy_evaluated": "see full Stage13 ledger; failures preserved",
                    },
                    threshold=threshold,
                )
        run = f"ECON_{timeframe.upper()}_DIAGNOSTIC_V001"
        for model in ("ARX", "KALMAN", "GARCH", "HAR"):
            artifacts = sorted(
                (root / "results/econometrics/runs" / run / "frozen").glob(f"*/{model}.json")
            )
            model_refs = {str(f.relative_to(root)): sha256(f) for f in artifacts}
            refs.update(model_refs)
            add(
                f"{model}_{timeframe.upper()}",
                timeframe,
                1,
                verdict["candidate_verdicts"][f"{run}:{model}"],
                {
                    "model": model_refs,
                    "features": "fixed causal econometric identities",
                    "chronology": run,
                    "policy_evaluated": False,
                },
                diagnostic=model in ("GARCH", "HAR"),
            )
    for spec in metadata["frozen_specs"]:
        file = root / spec["path"]
        if not file.resolve().is_relative_to(root.resolve()) or sha256(file) != spec["sha256"]:
            raise ValueError("upstream frozen identity changed; new evidence audit required")
        refs[spec["path"]] = spec["sha256"]
        body = json.loads(file.read_text(encoding="utf-8"))
        timeframe = body.get("timeframe", "15m" if "15M" in spec["id"] else "5m")
        add(
            spec["id"],
            timeframe,
            int(spec["horizon"]),
            verdict["candidate_verdicts"]["LEGACY_ML_ENSEMBLES"],
            {
                "model": {spec["path"]: spec["sha256"]},
                "features": body.get(
                    "feature_set_id", body.get("feature_set_ids", "see frozen spec")
                ),
                "chronology": "legacy global adaptive chronology unresolved",
                "policy_evaluated": False,
                "target": spec["target"],
            },
            diagnostic=spec["target"] != "future_return",
        )
    if len({c.alpha_id for c in candidates}) != len(candidates):
        raise ValueError("duplicate alpha identity")
    return {
        "registry_id": plan.registry_id,
        "candidates": [c.resolved() for c in candidates],
        "eligible_alphas": [],
        "result": "NO_ELIGIBLE_ALPHAS",
        "active": False,
        "diagnostics": "variance/interval/health/residual probabilities supply no trading direction",
        "evidence_files": refs,
        "dataset": metadata,
        "upstream_sources": upstream,
        "historical_recorded_statuses": inventory["historical_recorded_statuses"],
        "missing_prerequisites": inventory["missing_prerequisites"],
        "prior_results_reproduced": False,
    }


def freeze_registry(root: Path, plan: PortfolioPlan) -> Path:
    return freeze(
        root / "results/alpha_portfolio/registries" / f"{plan.registry_id}.json",
        build_registry(root, plan),
    )
