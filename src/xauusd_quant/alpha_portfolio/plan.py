"""Small immutable portfolio design, frozen before synthetic portfolio outcomes."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from ..execution.config import content_hash
from ..execution.io import _json_safe, write_new_json
from ..execution.readiness import sha256


def versioned(identity: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*_V\d{3,}", identity):
        raise ValueError("safe versioned identity required")


def freeze(path: Path, body: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Hash exactly the representation the shared JSON writer persists, including
    # ISO timestamps. datetime.__str__ cannot hash the serialized read-back body.
    body = _json_safe(body)
    digest = content_hash(body)
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("content_sha256") != digest or content_hash(previous["body"]) != digest:
            raise ValueError("immutable artifact changed; bump version")
    else:
        write_new_json(
            path, {"body": body, "content_sha256": digest, "registered_utc": datetime.now(UTC)}
        )
    return path


@dataclass(frozen=True)
class PortfolioPlan:
    plan_id: str
    registry_id: str
    evidence_run: str
    methods: tuple[str, ...] = ("equal", "minimum_variance_shrunk")
    seed: int = 150015
    budget_lots: float = 0.02
    covariance_rows: int = 32
    shrinkage: float = 0.5
    trial_budget: int = 24
    maximum_events: int = 10000
    wall_budget_seconds: float = 120
    private_memory_budget_bytes: int = 1073741824
    revision_reason: str = "initial bounded offline portfolio design"
    supersedes: str | None = None

    def __post_init__(self) -> None:
        for value in (self.plan_id, self.registry_id, self.evidence_run):
            versioned(value)
        if self.supersedes is not None:
            versioned(self.supersedes)
        if not self.revision_reason:
            raise ValueError("explicit plan revision reason required")
        if self.methods != ("equal", "minimum_variance_shrunk"):
            raise ValueError("only the two declared methods; revisions require a new design")
        if not math.isfinite(self.budget_lots) or not 0 < self.budget_lots <= 0.02:
            raise ValueError("frozen research lot budget <= .02 required")
        if self.shrinkage != 0.5 or self.covariance_rows != 32:
            raise ValueError("fixed 32-row window and half-diagonal shrinkage")
        for bound, maximum in (
            (self.trial_budget, 24),
            (self.maximum_events, 10000),
            (self.private_memory_budget_bytes, 1073741824),
        ):
            if type(bound) is not int or not 1 <= bound <= maximum:
                raise ValueError("bounded integer budget required")
        if not math.isfinite(self.wall_budget_seconds) or not 0 < self.wall_budget_seconds <= 120:
            raise ValueError("bounded wall budget required")

    def resolved(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "universe": "all recorded Stage14 candidates and primary legacy specs; no promotion or search expansion",
            "eligibility": "survives Stage14 + matching policy/economic/null/execution identities; eligibility available strictly before fold; missing blocks",
            "schedule": "synthetic two contiguous 30-minute folds in 2021; fit on strictly earlier matured 5-minute returns; update only at fold starts; real universe empty",
            "normalization": "fixed sign-to-unit-budget fraction; no fitted moments; policy observed-bar exits plus explicit maximum staleness; horizons remain attached to each intent",
            "risk": "aligned fixed-capital standalone 5-minute sleeve returns, complete cases, preceding32 rows; S*.5+diag(S)*.5; no IID errors; first/last16 sensitivity descriptive",
            "objective": "long-only fully allocated minimum w'Cw; bounded 1..3 sleeves; exhaustive active sets, no expected returns or leverage; equal fallback recorded on insufficient/singular/invalid inputs",
            "execution": "Stage12 shared account; round toward zero to .01 lot; full close then reopen on resize/reversal; no same-quote reentry; pending entry cancelled on changed target, exit retained; retry at next event after expiry/gap",
            "comparisons": "no trading, each synthetic standalone, equal, minimum_variance_shrunk, equal removal-of-one; base plus double commission/extra .02 slippage; no winner selection",
            "primary_metrics": "cutoff marked equity change, cash reconciliation, costs, turnover, net/gross-intended exposure, known-mark drawdown; no Sharpe or summed sleeve equity",
            "acceptance": "software invariants must pass; real evidence requires eligible universe and prior-only fold identities; none yields NO_ELIGIBLE_ALPHAS; synthetic results cannot promote",
            "history": "retrospective previously inspected development; synthetic correctness separate; no reserved outcomes; historical eligibility cannot be backdated",
            "stopping": "at most24 operations, sequential; no expansion after failure; no Stage16, broker, deployment, commit or push",
            "attribution": "absolute weighted intent shares among net-direction supporters frozen at actual entry; all trade cash/costs/exposure and unrealized allocated to those owners until close; excluded opposing intents recorded; attribution is convention, not causal contribution",
        }


def load_plan(path: Path) -> PortfolioPlan:
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ValueError("portfolio plan mapping required")
    body["methods"] = tuple(body["methods"])
    return PortfolioPlan(**body)


def freeze_plan(root: Path, plan: PortfolioPlan) -> Path:
    directory = root / "results/alpha_portfolio/plans"
    path = freeze(directory / f"{plan.plan_id}.json", plan.resolved())
    evidence = root / "results/robustness/runs" / plan.evidence_run
    files = {
        name: sha256(evidence / name) if (evidence / name).is_file() else None
        for name in ("verdict.json", "inventory.json", "manifest.json")
    }
    freeze(directory / f"{plan.plan_id}_SOURCES.json", {"run": plan.evidence_run, "files": files})
    return path
