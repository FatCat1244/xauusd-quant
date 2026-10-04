"""Reconstruct recorded exposure without inventing an effective independent trial count."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from ..execution.config import content_hash
from ..execution.readiness import inventory as upstream_inventory
from ..execution.readiness import sha256
from ..utils.config import Config


def ledger_view(path: Path) -> dict[str, Any]:
    attempts: dict[int, dict[str, Any]] = {}
    terminal: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        number = row["trial"]
        table = attempts if row["status"] == "attempted" else terminal
        if number in table or row["status"] not in ("attempted", "completed", "failed"):
            raise ValueError("duplicate or invalid trial record")
        table[number] = row
    if not terminal.keys() <= attempts.keys():
        raise ValueError("terminal trial without attempt")
    records = []
    for number, row in attempts.items():
        end = terminal.get(number, {})
        if end and end["identity"] != row["identity"]:
            raise ValueError("trial identity changed")
        spec = row["specification"]
        # Fold and cutoff identify refits, not a new candidate hypothesis.
        stable = {
            k: v
            for k, v in spec.items()
            if k not in ("fold", "fold_id", "cutoff", "inner", "as_of", "seed", "resample")
        }
        records.append(
            {
                **row,
                "status": end.get("status", "interrupted_or_abandoned"),
                "reason": end.get("reason"),
                "candidate_specification_key": content_hash(stable),
                "classification": "execution_scenario"
                if row["kind"] == "execution"
                else "repeated_chronological_fit"
                if "fit" in row["kind"] or "freeze" in row["kind"]
                else "diagnostic_or_wrapper",
                "seed_or_resample": {k: spec[k] for k in ("seed", "resample") if k in spec},
            }
        )
    return {
        "sha256": sha256(path),
        "attempts": len(records),
        "statuses": dict(Counter(r["status"] for r in records)),
        "records": records,
        "distinct_recorded_specification_keys": len(
            {r["candidate_specification_key"] for r in records}
        ),
        "effective_independent_trials": None,
    }


def build_inventory(config: Config) -> dict[str, Any]:
    root = config.project_root
    metadata = upstream_inventory(config)
    ledgers = []
    for directory in ("strategy_validation/runs", "econometrics/runs", "econometrics/audits"):
        for path in sorted((root / "results" / directory).glob("*/trial_ledger.jsonl")):
            view = ledger_view(path)
            synthetic = "SMOKE" in path.parent.name
            view.update(
                path=str(path.relative_to(root)),
                evaluation_history="software_correctness" if synthetic else "historical_inspected",
                classification_override="preserved misclassified real fixture"
                if "audits" in path.parts
                else None,
            )
            ledgers.append(view)
    registries = []
    for path in (root / "results").glob("*_research/test_registry.parquet"):
        registries.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": sha256(path),
                "rows": pq.read_metadata(path).num_rows,
                "columns": pq.read_schema(path).names,
                "meaning": "registered tests, not necessarily unique independent model trials",
            }
        )
    shared = root / "results/research_ledger.parquet"
    if shared.exists():
        registries.append(
            {
                "path": str(shared.relative_to(root)),
                "sha256": sha256(shared),
                "rows": pq.read_metadata(shared).num_rows,
                "columns": pq.read_schema(shared).names,
            }
        )
    history: Counter[str] = Counter()
    synthetic_counts: Counter[str] = Counter()
    for view in ledgers:
        counter = (
            synthetic_counts if view["evaluation_history"] == "software_correctness" else history
        )
        counter.update(view["statuses"])
    execution_runs = []
    for path in sorted((root / "results/execution").glob("*/manifest.json")):
        body = json.loads(path.read_text(encoding="utf-8"))
        execution_runs.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": sha256(path),
                "kind": body.get("kind"),
                "summary_present": (path.parent / "summary.json").exists(),
            }
        )
    return {
        "metadata": metadata,
        "ledgers": ledgers,
        "historical_recorded_statuses": dict(history),
        "synthetic_recorded_statuses": dict(synthetic_counts),
        "registries": registries,
        "execution_runs": execution_runs,
        "historical_search_completeness": "unknown; old grids, revisions and unrecorded abandoned searches not reconstructible from current ledgers",
        "selection_exposure_quantification": "partial recorded attempts; unknown adaptive history and dependence; no effective trial estimate",
        "retrospective_selection": "legacy model/features/ensemble universes inspected; Stage13 fixed neighborhoods retained, not winners; Stage13.5 additions fixed hypotheses",
        "historical_documentation_only": "earlier accuracy and large-history tables not rerun; metadata hashes prove identity, not scientific validity",
        "missing_prerequisites": [
            "matched prior-only random-walk/sign-flip pipeline evidence",
            "verified supplied execution/account terms",
            "audited full legacy fold-local feature/count/universe chronology",
            "adequate chronological coverage",
            "independently established uninspected period",
        ],
    }
