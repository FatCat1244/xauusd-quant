"""Frozen Stage13 forecast diagnostics on recorded Exness bars; no trading path.

Reuse Stage13 feature construction and frozen inference. This script never fits,
loads historical market rows, promotes an alpha, opens MT5 or submits an intent.
Recorded-feed computations are retrospective software diagnostics, not live forecasts.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from xauusd_quant.alpha_portfolio.plan import freeze, versioned
from xauusd_quant.execution.config import content_hash
from xauusd_quant.execution.io import parse_utc, write_new_json
from xauusd_quant.execution.readiness import sha256
from xauusd_quant.features.factory_config import load_features_config
from xauusd_quant.features.registry import build_registry
from xauusd_quant.risk.runs import evidence
from xauusd_quant.shadow.config import load_shadow_config
from xauusd_quant.shadow.features import SUPPORTED
from xauusd_quant.shadow.runs import code_identity, compare, frozen_inventory, read_frozen
from xauusd_quant.strategy_validation.pipeline import FrozenPipeline
from xauusd_quant.strategy_validation.plan import FEATURE_POOL
from xauusd_quant.strategy_validation.source import features

PLAN_ID = "EXNESS_FROZEN_FORECAST_DIAGNOSTIC_V001"
MODEL_RUNS = {tf: f"STRATEGY_{tf.upper()}_DIAGNOSTIC_V002" for tf in ("5m", "15m")}


def load_pipeline(path: Path, expected_sha256: str) -> FrozenPipeline:
    if sha256(path) != expected_sha256:
        raise ValueError("pinned fitted pipeline changed")
    body = json.loads(path.read_text(encoding="utf-8"))
    body["cutoff_utc"] = parse_utc(body["cutoff_utc"])
    body["training_label_as_of_utc"] = parse_utc(body["training_label_as_of_utc"])
    body["features"] = tuple(body["features"])
    for key in ("means", "scales", "coefficients"):
        body[key] = tuple(body[key])
    body["inner_trials"] = tuple({**row, "cutoff": parse_utc(row["cutoff"])}
                                  for row in body["inner_trials"])
    model = FrozenPipeline(**body)
    n = len(model.features)
    if (model.candidate_id != "CAUSAL_RIDGE" or model.fold_id != "F02"
        or not 1 <= n <= 3 or len(set(model.features)) != n or not set(model.features) <= set(FEATURE_POOL)
        or any(len(v) != n for v in (model.means, model.scales, model.coefficients))
        or not all(math.isfinite(v) for v in (*model.means, *model.scales, *model.coefficients, model.intercept))
        or any(v <= 0 for v in model.scales) or model.training_rows < 8
        or model.training_label_as_of_utc >= model.cutoff_utc
        or any(row["cutoff"] >= model.cutoff_utc for row in model.inner_trials)):
        raise ValueError("valid prior-only frozen Stage13 ridge parameters required")
    return model


class Diagnostic:
    """Four consecutive completed closes; actual Stage13 features, fixed fit only."""

    def __init__(self, models: dict[str, FrozenPipeline], identity: str) -> None:
        self.models, self.identity = models, identity
        self.history: dict[str, list[tuple[str, float]]] = {tf: [] for tf in models}
        self.last_open: dict[str, str | None] = dict.fromkeys(models)

    def update(self, bar: dict[str, Any]) -> dict[str, Any]:
        tf = bar["timeframe"]
        model = self.models[tf]
        seconds = int(tf[:-1]) * 60
        opened = parse_utc(bar["values"]["timestamp"])
        available = parse_utc(bar["available_utc"])
        if available.timestamp() < opened.timestamp() + seconds:
            raise ValueError("incomplete or unavailable bar cannot enter a forecast")
        previous = self.last_open[tf]
        if previous is not None and opened <= parse_utc(previous):
            raise ValueError("duplicate or out-of-order completed bar")
        if previous is not None and (opened - parse_utc(previous)).total_seconds() != seconds:
            self.history[tf].clear()
        self.last_open[tf] = opened.isoformat()
        price = bar["values"]["close"]
        valid = bar["valid"] is True and isinstance(price, (int, float)) and not isinstance(price, bool)
        valid = valid and math.isfinite(price) and price > 0
        if not valid:
            self.history[tf].clear()
        else:
            self.history[tf].append((opened.isoformat(), float(price)))
            self.history[tf] = self.history[tf][-4:]
        values: dict[str, float] | None = None
        prediction: float | None = None
        reason = "INVALID_BAR" if not valid else "WARMUP"
        if len(self.history[tf]) == 4:
            times, prices = zip(*self.history[tf], strict=True)
            row = features([parse_utc(t) for t in times], prices, seconds)[-1]
            values = dict(zip(FEATURE_POOL, row.features, strict=True))
            prediction = model.predict(row, prediction_at=available)
            if not math.isfinite(prediction):
                raise ValueError("nonfinite frozen forecast")
            reason = "FORECAST_DIAGNOSTIC_ONLY"
        return {"timeframe": tf, "bar_open_utc": opened.isoformat(),
            "recorded_input_available_utc": available.isoformat(),
            "model_id": model.candidate_id, "fold_id": model.fold_id,
            "fit_cutoff_utc": model.cutoff_utc.isoformat(), "features": values,
            "value": prediction, "units": "log_mid_return", "horizon_bars": 1,
            "reason": reason, "backfill": bar["backfill"], "trade_intent": False,
            "classification": "recorded-feed inference reconstruction; not a live forecast"}

    def state(self) -> dict[str, Any]:
        return {"identity": self.identity, "history": self.history, "last_open": self.last_open}

    def restore(self, state: dict[str, Any]) -> None:
        if (state.get("identity") != self.identity or set(state["history"]) != set(self.models)
            or set(state["last_open"]) != set(self.models)):
            raise ValueError("diagnostic checkpoint incompatible")
        for tf, rows in state["history"].items():
            if len(rows) > 4 or any(not math.isfinite(p) or p <= 0 for _, p in rows):
                raise ValueError("invalid diagnostic history")
            seconds = int(tf[:-1]) * 60
            if any((parse_utc(b[0]) - parse_utc(a[0])).total_seconds() != seconds
                   for a, b in zip(rows, rows[1:], strict=False)):
                raise ValueError("diagnostic history must be consecutive")
            if rows and rows[-1][0] != state["last_open"][tf]:
                raise ValueError("diagnostic checkpoint watermark mismatch")
        self.history = {tf: [(t, p) for t, p in rows] for tf, rows in state["history"].items()}
        self.last_open = state["last_open"].copy()


def prepare(root: Path, shadow_path: Path, local_config: Path) -> Path:
    """Freeze a two-model software plan before capturing a new feed segment."""
    cfg = replace(load_shadow_config(shadow_path), configuration_id="SHADOW_FORECAST_OBSERVATION_V001",
                  backfill_seconds=3600, duration_seconds=300, max_events=30000)
    entries = {}
    for tf, run_id in MODEL_RUNS.items():
        path = root / "results/strategy_validation/runs" / run_id / "frozen/F02/CAUSAL_RIDGE.json"
        digest = sha256(path)
        load_pipeline(path, digest)
        historical = json.loads((path.parents[2] / "manifest.json").read_text(encoding="utf-8"))
        execution = historical["execution_configuration"]
        if (execution["timeframe"] != tf or execution["horizon_bars"] != 1
            or execution["policy_units"] != "log_mid_return"):
            raise ValueError("frozen source must declare matching timeframe, target units and horizon")
        entries[tf] = {"path": path.relative_to(root).as_posix(), "sha256": digest,
            "historical_run": run_id, "historical_manifest_sha256": sha256(path.parents[2] / "manifest.json")}
    plan = {"plan_id": PLAN_ID, "scope": "Stage18 independent forecast software diagnostics only",
        "models": entries, "selection": "both existing Stage13 ridge fits, last chronological fold F02; no performance selection, refit or policy search",
        "features": "existing Stage13 return_1, momentum_3, volatility_3 from four consecutive valid UTC closes",
        "target": "one-subsequent-observed-bar log midpoint return; training provenance retains Stage13 limitations",
        "capture": cfg.public(), "local_capture_config_identity": cfg.identity,
        "budget": "one <=300s read-only capture, <=30000 ticks, <=256 forecast bars, two fixed fits; no model grid",
        "acceptance": "finite forecasts; canonical batch/incremental arithmetic equality at rtol1e-12/atol1e-14; exact chunk/restart/prefix invariance; absent coverage is untested",
        "boundaries": "no fitting, targets, eligibility changes, alpha intents, risk rearming, broker requests or historical dataset reads",
        "history": "new-feed software observation; existing fits/outcomes inspected, no predictive or prospective performance claim",
        "code_identity": code_identity(root), "script_sha256": sha256(Path(__file__)),
        "evidence_before": evidence(root)}
    if local_config.exists():
        raise ValueError("local diagnostic configuration already exists; preserve it")
    plan_path = root / "results/shadow/forecast_diagnostics/plans" / f"{PLAN_ID}.json"
    freeze(plan_path, plan)
    local_config.parent.mkdir(parents=True, exist_ok=True)
    with local_config.open("x", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(asdict(cfg), stream, sort_keys=False)
    return plan_path


def legacy_readiness(root: Path) -> dict[str, Any]:
    """Saved spec/manifest metadata only; no binary loading or scientific promotion."""
    cfg = load_features_config(root=root)
    registries = {tf: {s.name: s for s in build_registry(cfg, tf, int(tf[:-1]) * 60)}
                  for tf in ("5m", "15m")}
    inventory = frozen_inventory(root)
    reports = []
    for entry in inventory["entries"]:
        specs = registries[entry["timeframe"]]
        unsupported = [name for name in entry["feature_names"]
                       if name not in specs or specs[name].family not in SUPPORTED or not specs[name].live_safe]
        manifest_path = root / "data/models" / entry["identity"] / "manifest.json"
        reports.append({"identity": entry["identity"], "kind": entry["kind"],
            "spec_sha256": entry["spec_sha256"], "required_features": entry["feature_names"],
            "unsupported_live_features": unsupported, "output_units": entry["output_units"],
            "maximum_feature_history_bars": max((specs[n].min_history for n in entry["feature_names"] if n in specs), default=0),
            "model_manifest_present_at_standard_path": manifest_path.is_file(),
            "model_manifest_sha256": sha256(manifest_path) if manifest_path.is_file() else None,
            "historical_selection_audit": "unresolved", "feed_transfer": "unassessed",
            "activation_status": "BLOCKED"})
    return {"models": inventory["models"], "ensembles": inventory["ensembles"], "entries": reports,
        "scope": "metadata feasibility audit; path presence is not compatible fitted-artifact validation"}


def amend_reporting(root: Path, original: Path, plan_id: str) -> Path:
    """Preserve V001; clarify partial coverage after inspecting its software results."""
    previous = read_frozen(original)
    versioned(plan_id)
    if not plan_id.startswith("EXNESS_FROZEN_FORECAST_DIAGNOSTIC_V") or plan_id == PLAN_ID:
        raise ValueError("new version of the same diagnostic plan required")
    if previous["plan_id"] != PLAN_ID or previous["code_identity"] != code_identity(root):
        raise ValueError("original plan and unchanged native source required")
    revised = {**previous, "plan_id": plan_id,
        "supersedes_plan_sha256": sha256(original), "script_sha256": sha256(Path(__file__)),
        "revision_reason": "V001 covered 5m only; explicitly classify each timeframe and overall partial coverage. CLI path display also corrected. No fitting, universe, numeric tolerance or acceptance change; same recorded segment, no new capture.",
        "history": "reporting revision after V001 outcome inspection; retrospective software verification only"}
    output = original.parent / f"{revised['plan_id']}.json"
    freeze(output, revised)
    return output


def forecast_coverage(predictions: list[dict[str, Any]], timeframes: tuple[str, ...]) -> dict[str, Any]:
    counts = Counter(p["timeframe"] for p in predictions if p["value"] is not None)
    status = ("FORECAST_DIAGNOSTICS_PASSED" if all(counts[tf] > 0 for tf in timeframes)
              else "PARTIAL_FORECAST_DIAGNOSTICS" if counts else "INCONCLUSIVE_NO_FORECASTS")
    return {"status": status, "finite_forecasts_by_timeframe": {tf: counts[tf] for tf in timeframes},
        "timeframe_status": {tf: "OBSERVED_INFERENCE_MATCHED" if counts[tf] else "UNTESTED_NO_FORECASTS"
                             for tf in timeframes}}


def evaluate(root: Path, plan_path: Path, recorded: Path, replayed: Path, run_id: str) -> dict[str, Any]:
    versioned(run_id)
    if not run_id.replace("_", "").isalnum():
        raise ValueError("safe versioned run identity required")
    plan = read_frozen(plan_path)
    if plan["code_identity"] != code_identity(root) or plan["script_sha256"] != sha256(Path(__file__)):
        raise ValueError("diagnostic code changed after plan freeze")
    initial = read_frozen(recorded / "initial_state.json")
    manifest = read_frozen(recorded / "manifest.json")
    if initial["body"]["configuration_identity"] != plan["local_capture_config_identity"]:
        raise ValueError("capture differs from the frozen diagnostic configuration")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or sha256(recorded / name) != expected:
            raise ValueError("captured evidence manifest mismatch")
    comparison = compare(recorded, replayed)
    if not comparison["equal"]:
        raise ValueError("recorded baseline replay differs")
    if set(plan["models"]) != set(MODEL_RUNS):
        raise ValueError("exactly the two prespecified forecast fits required")
    models = {}
    for tf, row in plan["models"].items():
        expected_path = root / "results/strategy_validation/runs" / MODEL_RUNS[tf] / "frozen/F02/CAUSAL_RIDGE.json"
        path = root / row["path"]
        if (path.resolve() != expected_path.resolve()
            or sha256(path.parents[2] / "manifest.json") != row["historical_manifest_sha256"]):
            raise ValueError("historical fitted source path or provenance changed")
        models[tf] = load_pipeline(path, row["sha256"])
    identity = content_hash(plan["models"])
    bars = []
    with (recorded / "records.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            entry = json.loads(line)
            if entry["table"] == "bars":
                bars.append(entry["row"])
                if len(bars) > 256:
                    raise ValueError("bounded forecast bar budget exceeded")
    diagnostic = Diagnostic(models, identity)
    predictions = [diagnostic.update(bar) for bar in bars]
    restarted = Diagnostic(models, identity)
    restarted_rows: list[dict[str, Any]] = []
    for start in range(0, len(bars), 2):
        restarted_rows.extend(restarted.update(bar) for bar in bars[start:start + 2])
        checkpoint = json.loads(json.dumps(restarted.state()))
        restarted = Diagnostic(models, identity)
        restarted.restore(checkpoint)
    prefix = Diagnostic(models, identity)
    prefix_rows = [prefix.update(bar) for bar in bars[:len(bars) // 2]]
    if restarted_rows != predictions or prefix_rows != predictions[:len(prefix_rows)]:
        raise ValueError("chunk/restart/prefix forecast invariance failed")
    errors = []
    histories: dict[str, list[dict[str, Any]]] = {tf: [] for tf in models}
    for bar, prediction in zip(bars, predictions, strict=True):
        tf = bar["timeframe"]
        seconds = int(tf[:-1]) * 60
        history = histories[tf]
        if not bar["valid"] or history and (parse_utc(bar["values"]["timestamp"]) - parse_utc(history[-1]["values"]["timestamp"])).total_seconds() != seconds:
            history.clear()
        if bar["valid"]:
            history.append(bar)
        if prediction["value"] is not None:
            rows = features([parse_utc(b["values"]["timestamp"]) for b in history],
                            [b["values"]["close"] for b in history], seconds)
            model = models[tf]
            x = np.asarray([[row.features[FEATURE_POOL.index(n)] for n in model.features] for row in rows])
            batch = model.intercept + ((x - model.means) / model.scales) @ np.asarray(model.coefficients)
            errors.append(abs(float(batch[-1]) - prediction["value"]))
            if not np.isclose(batch[-1], prediction["value"], rtol=1e-12, atol=1e-14):
                raise ValueError("canonical batch and incremental frozen inference differ")
    if evidence(root) != plan["evidence_before"]:
        raise ValueError("scientific records or reserved access logs changed during this study")
    directory = root / "results/shadow/forecast_diagnostics/runs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "forecasts.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for row in predictions:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    write_new_json(directory / "checkpoint.json", diagnostic.state())
    capture_summary = read_frozen(recorded / "summary.json")
    report = {"run_id": run_id, "plan_sha256": sha256(plan_path), "code_identity": code_identity(root),
        "script_sha256": sha256(Path(__file__)), "computed_at_utc": datetime.now(UTC).isoformat(),
        "classification": "retrospective recorded-feed forecast software diagnostics",
        "forecast_rows": len(predictions), "finite_forecasts": len(errors),
        **forecast_coverage(predictions, tuple(models)),
        "maximum_batch_prediction_difference": max(errors) if errors else None,
        "batch_equality": True if errors else None, "chunk_restart_equality": True,
        "prefix_invariance": True, "baseline_observed_equality": comparison,
        "capture_summary": capture_summary, "models": plan["models"],
        "legacy_readiness": legacy_readiness(root), "eligible_alphas": [],
        "strategy_active": False, "full_portfolio_risk_equality": "UNTESTED",
        "live_model_inference": False, "predictive_value": "UNASSESSED", "economics": "UNASSESSED",
        "reserved_access_log_hashes_unchanged": True, "broker_orders": 0,
        "observed_forecast_equality": True if errors else None}
    freeze(directory / "report.json", report)
    freeze(directory / "manifest.json", {"plan_sha256": sha256(plan_path),
        "source_records_sha256": sha256(recorded / "records.jsonl"),
        "files": {p.name: sha256(p) for p in directory.iterdir() if p.is_file()}})
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--shadow-config", type=Path, required=True)
    prep.add_argument("--local-capture-config", type=Path, required=True)
    amendment = sub.add_parser("amend-reporting")
    amendment.add_argument("--original-plan", type=Path, required=True)
    amendment.add_argument("--plan-id", required=True)
    run = sub.add_parser("evaluate")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--recorded", type=Path, required=True)
    run.add_argument("--replayed", type=Path, required=True)
    run.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.command == "prepare":
        print(prepare(root, args.shadow_config, args.local_capture_config).relative_to(root))
    elif args.command == "amend-reporting":
        print(amend_reporting(root, args.original_plan, args.plan_id).resolve().relative_to(root))
    else:
        report = evaluate(root, args.plan, args.recorded, args.replayed, args.run_id)
        print(json.dumps({k: report[k] for k in ("status", "finite_forecasts_by_timeframe",
            "maximum_batch_prediction_difference", "strategy_active", "broker_orders")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
