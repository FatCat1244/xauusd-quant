"""Immutable plans, retained failures, blocked promotion and saved-source cutoff guards."""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from xauusd_quant.execution.config import ExecutionConfig, content_hash
from xauusd_quant.execution.io import DiskRecorder, write_new_json
from xauusd_quant.execution.readiness import sha256
from xauusd_quant.robustness.inventory import ledger_view
from xauusd_quant.robustness.plan import RobustnessPlan, freeze_plan
from xauusd_quant.robustness.reports import candidate_verdict, source_records
from xauusd_quant.robustness.studies import fit_predictions, synthetic_fixture
from xauusd_quant.strategy_validation.runs import TrialLedger
from xauusd_quant.utils.config import load_config


def test_plan_immutable_no_prospective_claim_or_favorable_family_subset(tmp_path: Path) -> None:
    plan = RobustnessPlan("ROBUST_TEST_V001", ("ECON_5M_DIAGNOSTIC_V001",))
    file = freeze_plan(tmp_path, plan)
    assert freeze_plan(tmp_path, plan) == file
    with pytest.raises(ValueError, match="immutable"):
        freeze_plan(tmp_path, replace(plan, seed=4))
    with pytest.raises(ValueError, match="inspection"):
        replace(plan, classification="fresh")
    with pytest.raises(ValueError, match="complete"):
        replace(plan, pairs=(("ARX", "ZERO"),))


def test_trial_inventory_keeps_failed_and_interrupted_fits(tmp_path: Path) -> None:
    with DiskRecorder(tmp_path) as recorder:
        ledger = TrialLedger(recorder, 5)
        ledger.call("inner_fit", {"model": "RIDGE", "fold": "F1"}, lambda: 1)

        def failure() -> None:
            raise ValueError("preserved failure")

        assert ledger.call("inner_fit", {"model": "RIDGE", "fold": "F2"}, failure) is None
        recorder(
            "trial_ledger",
            {
                "trial": 3,
                "status": "attempted",
                "identity": "x",
                "kind": "outer_fit",
                "specification": {"model": "RIDGE", "fold": "F3"},
            },
        )
    view = ledger_view(tmp_path / "trial_ledger.jsonl")
    assert view["attempts"] == 3
    assert view["statuses"] == {"completed": 1, "failed": 1, "interrupted_or_abandoned": 1}
    assert view["distinct_recorded_specification_keys"] == 1
    assert view["effective_independent_trials"] is None


def test_missing_gate_and_nan_cannot_promote() -> None:
    effect = {
        "status": "measured",
        "target": "future_return",
        "relative_mse_improvement": 0.5,
        "positive_fold_fraction": 1.0,
        "uncertainty": [{"status": "estimated", "interval": [0.1, 0.2]}],
    }
    assert candidate_verdict([effect], {})["status"] == "BLOCKED"
    gates = dict.fromkeys(
        ("data_provenance", "fold_local_pipeline", "evidence_identity", "null_evidence"), "passed"
    )
    assert (
        candidate_verdict([{**effect, "relative_mse_improvement": float("nan")}], gates)["status"]
        != "SURVIVES DECLARED HISTORICAL TESTS"
    )
    assert candidate_verdict([effect], gates)["status"] == "SURVIVES DECLARED HISTORICAL TESTS"


def make_saved_source(tmp_path: Path, *, reserved: bool = False) -> tuple[Any, Path]:
    config = replace(load_config(), project_root=tmp_path)
    path = tmp_path / "results/econometrics/runs/ECON_5M_DIAGNOSTIC_V001"
    path.mkdir(parents=True)
    start = datetime(2022 if reserved else 2021, 6, 1, tzinfo=UTC)
    fold = {
        "fold_id": "F1",
        "training_start_utc": (start - timedelta(days=30)).isoformat(),
        "evaluation_start_utc": start.isoformat(),
        "evaluation_end_utc": (start + timedelta(days=1)).isoformat(),
    }
    plan = {"folds": [fold], "evaluation_history_classification": "historical_reconstructed"}
    body = {"plan": plan, "execution": ExecutionConfig().resolved()}
    frozen = tmp_path / "results/econometrics/plans/TEST_V001.json"
    frozen.parent.mkdir(parents=True)
    write_new_json(frozen, {**body, "content_sha256": content_hash(body)})
    write_new_json(
        path / "request.json",
        {"plan": str(frozen), "plan_sha256": sha256(frozen), "synthetic": False},
    )
    return config, path


def test_reserved_source_refused_before_opening_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, path = make_saved_source(tmp_path, reserved=True)
    from xauusd_quant.robustness import reports

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("reserved saved outcomes were opened")

    monkeypatch.setattr(reports, "read_records", forbidden)
    with pytest.raises(ValueError, match="2022"):
        source_records(config, path, 20)


def test_saved_forecast_maturity_and_fit_cutoff_are_audited(tmp_path: Path) -> None:
    config, path = make_saved_source(tmp_path)
    write_new_json(
        path / "readiness.json", {"gates": {}, "inventory": {"dataset_version": "synthetic"}}
    )
    cutoff = datetime(2021, 5, 31, tzinfo=UTC)
    spec = {
        "model": "ARX",
        "fold_id": "F1",
        "fitting_cutoff_utc": cutoff.isoformat(),
        "target": "future_return",
        "units": "log_mid_return",
        "grid_seconds": 300,
        "horizon_bars": 1,
        "training_label_as_of_utc": (cutoff - timedelta(minutes=5)).isoformat(),
        "parameter_information_as_of_utc": (cutoff - timedelta(minutes=5)).isoformat(),
    }
    file = path / "frozen/F1/ARX.json"
    file.parent.mkdir(parents=True)
    write_new_json(file, spec)
    row = {
        "model": "ARX",
        "fold_id": "F1",
        "row_id": "r",
        "available_utc": "2021-06-01T00:00:00+00:00",
        "units": "log_mid_return",
        "grid_seconds": 300,
        "horizon_bars": 1,
        "target_end_utc": "2021-06-01T00:05:00+00:00",
        "outcome_matured_utc": "2021-06-01T00:05:00+00:00",
        "frozen_spec_sha256": sha256(file),
        "value": 0.0,
        "actual": 0.1,
        "target": "future_return",
        "loss": 0.01,
    }
    scores = path / "forecast_scores.jsonl"
    scores.write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert len(source_records(config, path, 20, {"forecast_scores.jsonl": sha256(scores)})[0]) == 1
    scores.write_text(
        json.dumps({**row, "outcome_matured_utc": "2021-06-01T00:01:00+00:00"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="matured"):
        source_records(config, path, 20, {"forecast_scores.jsonl": sha256(scores)})
    spec["training_label_as_of_utc"] = cutoff.isoformat()
    file.write_text(json.dumps(spec), encoding="utf-8")
    scores.write_text(
        json.dumps({**row, "frozen_spec_sha256": sha256(file)}) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unavailable"):
        source_records(config, path, 20, {"forecast_scores.jsonl": sha256(scores)})


def test_perturbed_null_pipeline_repeats_inner_selection_before_scored_rows(tmp_path: Path) -> None:
    design, _, source, _ = synthetic_fixture(9, 0.0)
    with DiskRecorder(tmp_path) as sink:
        ledger = TrialLedger(sink, 20)
        forecasts, _, audit = fit_predictions(
            replace(design, ridge_alpha=0.5), source, ledger, drop_momentum=True
        )
    assert "momentum_3" not in audit["fit"]["features"]
    assert len(audit["fit"]["inner_trials"]) == 4
    assert len(forecasts) > 100
    assert all(r["cutoff"] < design.start.isoformat() for r in audit["fit"]["inner_trials"])
    assert all(r["cutoff"] <= design.start for r in source.accesses if r["kind"] == "training")
    assert ledger_view(tmp_path / "trial_ledger.jsonl")["attempts"] == 5
