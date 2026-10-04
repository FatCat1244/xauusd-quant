"""Audit saved development forecasts, losses, calibration and historical attribution."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from ..econometrics.comparison import loss
from ..execution.config import content_hash
from ..execution.io import parse_utc, validate_interval, write_new_json
from ..execution.readiness import sha256
from ..strategy_validation.runs import TrialLedger
from ..utils.config import Config
from .plan import RobustnessPlan
from .resampling import bootstrap
from .statistics import holm


def read_records(path: Path, maximum: int) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) > maximum:
                    raise ValueError("saved artifact exceeds frozen record cap")
    return rows


def source_records(
    config: Config, path: Path, maximum: int, expected_files: dict[str, str] | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Refuse reserved plans before opening saved outcomes, then audit identity/maturity."""
    root = config.project_root.resolve()
    if not path.resolve().is_relative_to(root / "results/econometrics/runs"):
        raise ValueError("source outside declared econometric development runs")
    request = json.loads((path / "request.json").read_text(encoding="utf-8"))
    frozen = Path(request["plan"])
    if not frozen.resolve().is_relative_to(root / "results/econometrics/plans"):
        raise ValueError("source plan path escapes repository")
    if sha256(frozen) != request["plan_sha256"]:
        raise ValueError("source plan hash changed")
    body = json.loads(frozen.read_text(encoding="utf-8"))
    if content_hash({k: body[k] for k in ("plan", "execution")}) != body["content_sha256"]:
        raise ValueError("source frozen content changed")
    plan = body["plan"]
    if request.get("synthetic") or plan["evaluation_history_classification"] not in (
        "historical_diagnostic",
        "historical_reconstructed",
    ):
        raise ValueError("expected historical development provenance")
    for fold in plan["folds"]:
        validate_interval(
            config, parse_utc(fold["training_start_utc"]), parse_utc(fold["evaluation_end_utc"])
        )
    if expected_files is None or "forecast_scores.jsonl" not in expected_files:
        raise ValueError("frozen source artifact hashes required before opening outcomes")
    for relative, digest in expected_files.items():
        file = path / relative
        if (
            not file.resolve().is_relative_to(path.resolve())
            or not file.is_file()
            or sha256(file) != digest
        ):
            raise ValueError("frozen saved source artifact changed or missing")
    readiness = json.loads((path / "readiness.json").read_text(encoding="utf-8"))
    specs: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    for file in path.glob("frozen/*/*.json"):
        if file.stem in ("FINAL_STATE", "CALIBRATION_STATE"):
            continue
        spec = json.loads(file.read_text(encoding="utf-8"))
        if "model" in spec:
            specs[(spec["fold_id"], spec["model"])] = (sha256(file), spec)
    records = read_records(path / "forecast_scores.jsonl", maximum)
    seen = set()
    folds = {f["fold_id"]: f for f in plan["folds"]}
    for row in records:
        key = (row["fold_id"], row["model"], row["row_id"])
        if key in seen:
            raise ValueError("duplicate scored forecast identity")
        seen.add(key)
        fold = folds[row["fold_id"]]
        start, end = parse_utc(fold["evaluation_start_utc"]), parse_utc(fold["evaluation_end_utc"])
        available = parse_utc(row["available_utc"])
        target_end = parse_utc(row["target_end_utc"])
        matured = parse_utc(row["outcome_matured_utc"])
        if not start <= available < target_end <= matured < end:
            raise ValueError("invalid forecast availability/matured outcome/fold cutoff")
        digest, spec = specs[(row["fold_id"], row["model"])]
        if digest != row["frozen_spec_sha256"]:
            raise ValueError("saved forecast specification hash mismatch")
        if any(
            row[field] != spec[field]
            for field in ("target", "units", "grid_seconds", "horizon_bars")
        ):
            raise ValueError("saved forecast target units/horizon differ from fitted specification")
        cutoff = parse_utc(spec["fitting_cutoff_utc"])
        if (
            cutoff > available
            or parse_utc(spec["training_label_as_of_utc"]) >= cutoff
            or parse_utc(spec["parameter_information_as_of_utc"]) >= cutoff
        ):
            raise ValueError("saved fit used unavailable observations/labels")
        expected = loss(row["value"], row["actual"], row["target"])
        if not math.isfinite(row["loss"]) or not math.isclose(expected, row["loss"], abs_tol=1e-15):
            raise ValueError("saved loss arithmetic mismatch")
    return records, {
        "path": str(path.relative_to(root)),
        "plan_sha256": sha256(frozen),
        "scores_sha256": sha256(path / "forecast_scores.jsonl"),
        "records_audited": len(records),
        "gates": readiness["gates"],
        "dataset_version": readiness["inventory"]["dataset_version"],
        "classification": "retrospective_saved_development; no new market outcomes",
    }


def matched_effect(
    records: list[dict[str, Any]], candidate: str, benchmark: str, plan: RobustnessPlan
) -> dict[str, Any]:
    sides = [
        {(r["fold_id"], r["row_id"]): r for r in records if r["model"] == name}
        for name in (candidate, benchmark)
    ]
    c, b = sides
    if any(
        len(side) != sum(r["model"] == name for r in records)
        for side, name in zip(sides, (candidate, benchmark), strict=True)
    ):
        raise ValueError("duplicate comparison records")
    pairs = sorted(c.keys() & b.keys(), key=lambda k: c[k]["available_utc"])
    if not pairs:
        return {
            "candidate": candidate,
            "benchmark": benchmark,
            "status": "missing",
            "uncertainty": [],
        }
    for key in pairs:
        if any(
            c[key][field] != b[key][field]
            for field in (
                "actual",
                "available_utc",
                "target_end_utc",
                "outcome_matured_utc",
                "horizon_bars",
                "grid_seconds",
                "target",
                "units",
            )
        ):
            raise ValueError("unmatched target/availability/units")
    x = np.asarray([b[k]["loss"] - c[k]["loss"] for k in pairs])
    times = [parse_utc(c[k]["available_utc"]) for k in pairs]
    group = [k[0] for k in pairs]
    seconds, horizon, target = (c[pairs[0]][f] for f in ("grid_seconds", "horizon_bars", "target"))
    by_fold = {
        g: float(np.mean([v for v, label in zip(x, group, strict=True) if label == g]))
        for g in sorted(set(group))
    }
    baseline = float(np.mean([b[k]["loss"] for k in pairs]))
    uncertainty = [
        bootstrap(
            x,
            times,
            group,
            seconds=seconds,
            horizon=horizon,
            block=block,
            replicates=plan.bootstrap_replicates,
            seed=plan.seed,
            minimum_days=plan.minimum_days,
            minimum_rows=plan.minimum_rows,
            minimum_blocks=plan.minimum_blocks,
        )
        for block in plan.block_lengths
    ]
    return {
        "candidate": candidate,
        "benchmark": benchmark,
        "status": "measured",
        "rows": len(x),
        "candidate_rows": len(c),
        "benchmark_rows": len(b),
        "unmatched_rows": len(c.keys() ^ b.keys()),
        "target": target,
        "mean_improvement": float(x.mean()),
        "relative_mse_improvement": float(x.mean() / baseline)
        if target == "future_return" and baseline > 0
        else None,
        "fold_improvements": by_fold,
        "positive_fold_fraction": sum(v > 0 for v in by_fold.values()) / len(by_fold),
        "first_half_improvement": float(x[: len(x) // 2].mean()),
        "last_half_improvement": float(x[len(x) // 2 :].mean()),
        "uncertainty": uncertainty,
        "p_value": None,
        "dependence": "grid horizons and gaps explicit; block choices all reported; no IID or adaptive selection p-value",
    }


def candidate_verdict(effects: list[dict[str, Any]], gates: dict[str, str]) -> dict[str, Any]:
    required = ("data_provenance", "fold_local_pipeline", "evidence_identity", "null_evidence")
    missing = [k for k in required if gates.get(k) != "passed"]
    if not effects or any(e["status"] == "missing" for e in effects) or missing:
        return {
            "status": "BLOCKED",
            "reasons": missing or ["missing matched forecast observations"],
            "conditional_evidence": "INCONCLUSIVE"
            if effects
            and any(u["status"] != "estimated" for e in effects for u in e["uncertainty"])
            else "descriptive only",
        }
    for effect in effects:
        metric = (
            effect.get("relative_mse_improvement")
            if effect.get("target") == "future_return"
            else effect.get("mean_improvement")
        )
        values = [metric, effect.get("positive_fold_fraction")]
        values.extend(v for u in effect["uncertainty"] for v in (u.get("interval") or []))
        if any(v is None or not math.isfinite(v) for v in values) or not effect["uncertainty"]:
            return {"status": "INCONCLUSIVE", "reasons": ["missing/nonfinite required statistic"]}
    if any(u["status"] != "estimated" for e in effects for u in e["uncertainty"]):
        return {
            "status": "INCONCLUSIVE",
            "reasons": ["insufficient coverage/dependence-aware uncertainty"],
        }
    passes = all(
        e["positive_fold_fraction"] >= 0.6
        and (
            e["relative_mse_improvement"]
            if e["target"] == "future_return"
            else e["mean_improvement"]
        )
        >= 0.01
        and all(u["interval"][0] > 0 for u in e["uncertainty"])
        for e in effects
    )
    return {
        "status": "SURVIVES DECLARED HISTORICAL TESTS" if passes else "REJECTED",
        "reasons": [
            "conditional frozen checks only; full historical discovery uncertainty remains unknown"
        ],
    }


def concentration(trades: list[dict[str, Any]]) -> dict[str, Any]:
    monthly: dict[str, float] = defaultdict(float)
    gains = []
    for row in trades:
        value = row["net_pnl_account"]
        if not math.isfinite(value):
            raise ValueError("nonfinite trade attribution")
        monthly[str(row["exit_utc"])[:7]] += value
        gains.append(max(0.0, value))
    positive = sum(gains)
    month_positive = sum(max(0.0, v) for v in monthly.values())
    return {
        "closed_trades": len(trades),
        "monthly_closed_pnl": dict(monthly),
        "top_three_positive_trade_share": sum(sorted(gains, reverse=True)[:3]) / positive
        if positive
        else None,
        "largest_positive_month_share": max(monthly.values(), default=0) / month_positive
        if month_positive
        else None,
        "pnl_after_removing_three_largest_gains": sum(monthly.values())
        - sum(sorted(gains, reverse=True)[:3]),
        "scope": "retrospective closed-trade attribution; not executable removal policy, portfolio returns or full drawdown",
    }


def historical_reports(
    config: Config, plan: RobustnessPlan, directory: Path, ledger: TrialLedger
) -> dict[str, Any]:
    effects, assumptions, verdicts, diagnostic_refs = [], [], {}, []
    source_index = json.loads(
        (
            config.project_root / "results/robustness/plans" / f"{plan.plan_id}_SOURCES.json"
        ).read_text(encoding="utf-8")
    )["files"]
    for run_id in plan.source_runs:
        path = config.project_root / "results/econometrics/runs" / run_id
        result = ledger.call(
            "saved_source_audit",
            {"run": run_id},
            partial(source_records, config, path, plan.maximum_records, source_index[run_id]),
        )
        if result is None:
            verdicts[run_id] = {
                "status": "BLOCKED",
                "reasons": ["missing or invalid saved development artifact; see ledger"],
            }
            continue
        records, audit = result
        assumptions.append(audit)
        for candidate, baseline in plan.pairs:
            effect = ledger.call(
                "conditional_loss",
                {"run": run_id, "candidate": candidate, "benchmark": baseline},
                partial(matched_effect, records, candidate, baseline, plan),
            )
            effects.append({"source_run": run_id, "effect": effect})
        for candidate in ("ARX", "KALMAN", "GARCH", "HAR"):
            candidate_effects = [
                r["effect"]
                for r in effects
                if r["source_run"] == run_id
                and r["effect"]
                and r["effect"]["candidate"] == candidate
            ]
            expected_pair_count = sum(c == candidate for c, _ in plan.pairs)
            if len(candidate_effects) != expected_pair_count:
                candidate_effects.append({"status": "missing", "uncertainty": []})
            verdicts[f"{run_id}:{candidate}"] = candidate_verdict(candidate_effects, audit["gates"])
        for filename in (
            "coefficient_stability.json",
            "coverage.jsonl",
            "change_monitor.jsonl",
            "filtered_states.jsonl",
        ):
            file = path / filename
            diagnostic_refs.append(
                {
                    "source_run": run_id,
                    "path": str(file.relative_to(config.project_root)),
                    "sha256": sha256(file) if file.exists() else None,
                    "status": "existing diagnostic, not a new perturbation/refit",
                }
            )
    family = tuple(f"{run_id}:{c}:{b}" for run_id in plan.source_runs for c, b in plan.pairs)
    corrections = holm(
        family, dict.fromkeys(family), validity="unsupported_adaptive_short_dependent_history"
    )
    write_new_json(directory / "assumptions.json", assumptions)
    write_new_json(directory / "statistics.json", effects)
    write_new_json(
        directory / "corrections.json",
        {
            "predictive": corrections,
            "economic": {
                "status": "unavailable",
                "reason": "no eligible economic inferential family",
            },
        },
    )
    write_new_json(
        directory / "bootstrap.json",
        [
            {
                "source_run": r["source_run"],
                "candidate": r["effect"]["candidate"],
                "benchmark": r["effect"]["benchmark"],
                "uncertainty": r["effect"]["uncertainty"],
            }
            for r in effects
            if r["effect"]
        ],
    )
    write_new_json(directory / "econometric_diagnostics.json", diagnostic_refs)
    for name in ("CAUSAL_RIDGE", "HISTORICAL_MEAN", "LEGACY_ML_ENSEMBLES"):
        verdicts[name] = {
            "status": "BLOCKED",
            "reasons": [
                "matching null/verified execution evidence missing; legacy feature/universe chronology additionally unresolved"
            ],
        }
    # Preserve all existing economic populations, including failed attempts with no trades.
    attribution = []
    for path in (config.project_root / "results/strategy_validation/runs").glob(
        "*DIAGNOSTIC*/trial_*/trades.jsonl"
    ):
        rows = read_records(path, plan.maximum_records)
        attribution.append(
            {
                "path": str(path.relative_to(config.project_root)),
                "sha256": sha256(path),
                **concentration(rows),
            }
        )
    write_new_json(directory / "concentration_decay.json", attribution)
    write_new_json(
        directory / "perturbations.json",
        {
            "status": "BLOCKED for market selection conclusions",
            "reason": "no eligible evidence gates; synthetic fold-local perturbations in separate smoke report; saved coefficients/calibration referenced above",
        },
    )
    write_new_json(
        directory / "execution_stress.json",
        {
            "status": "BLOCKED for market economic conclusions",
            "reason": "verified terms and matching prior-only null/evidence missing; Stage13 stress records preserved; engine stress exercised in synthetic smoke",
        },
    )
    write_new_json(
        directory / "monte_carlo.json",
        {
            "status": "conditional synthetic scenario study only; see smoke run",
            "measured_broker_distribution": False,
        },
    )
    return {
        "candidate_verdicts": verdicts,
        "stage15_candidates": [],
        "uncertainty": "short saved conditional evidence insufficient; discovery uncertainty unknown",
        "reserved_outcomes_read": False,
    }
