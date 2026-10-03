"""Frozen chronology, matched observations, honest evidence and retained failed hypotheses."""

import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from econometric_synth import at, fixture, grid
from strategy_synth import gates
from xauusd_quant.econometrics.comparison import addition_verdict, compare, loss
from xauusd_quant.econometrics.data import RidgeSource
from xauusd_quant.econometrics.plan import freeze_plan
from xauusd_quant.econometrics.runs import SyntheticBars, evaluate, fit_model
from xauusd_quant.execution.io import DiskRecorder, validate_interval
from xauusd_quant.strategy_validation.runs import TrialLedger
from xauusd_quant.utils.config import load_config


def read_rows(file: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in file.read_text().splitlines()]


def test_plan_changes_need_new_version_and_scope_and_budgets_are_fixed(tmp_path: Path) -> None:
    plan, c, _ = fixture()
    file = freeze_plan(tmp_path, plan, c)
    original = file.read_bytes()
    assert freeze_plan(tmp_path, plan, c) == file
    with pytest.raises(ValueError, match="immutable"):
        freeze_plan(tmp_path, replace(plan, ewma_lambda=0.95), c)
    assert file.read_bytes() == original
    with pytest.raises(ValueError, match="budget"):
        replace(plan, trial_budget=1)
    with pytest.raises(ValueError, match="prospective"):
        replace(plan, evaluation_history_classification="prospective")
    assert plan.expected_trials == 34


def test_model_fit_refuses_a_materialized_future_bar_even_with_purged_labels(
    tmp_path: Path,
) -> None:
    plan, _, source = fixture()
    data = grid(800)
    samples = data.samples(at(0), at(600), 1, 0)
    with DiskRecorder(tmp_path) as recorder:
        ledger = TrialLedger(recorder, plan.trial_budget)
        with pytest.raises(ValueError, match="materialized a future bar"):
            fit_model(
                "ARX",
                data,
                samples,
                at(0),
                at(600),
                plan,
                plan.folds[0],
                RidgeSource(source, 1, 0),
                ledger,
            )


def test_model_and_calibration_files_exist_before_outer_values_and_reruns_match(
    tmp_path: Path,
) -> None:
    plan, c, source = fixture()
    config = replace(load_config(), project_root=tmp_path)

    class Guarded(SyntheticBars):
        def __call__(self, start: Any, cutoff: Any) -> Any:
            for fold in plan.folds:
                if start == fold.start and cutoff == fold.end:
                    directory = (
                        tmp_path / "results/econometrics/runs/FIRST_V001/frozen" / fold.fold_id
                    )
                    assert (directory / "CALIBRATION_STATE.json").is_file(), (
                        "outer values opened before calibration freeze"
                    )
                    assert len(list(directory.glob("*.json"))) == 9
            return super().__call__(start, cutoff)

    guarded = Guarded(source.opens, source.prices, source.seconds)
    first = evaluate(config, c, plan, "FIRST_V001", synthetic=guarded)
    second = evaluate(config, c, plan, "SECOND_V001", synthetic=source)
    a, b = Path(first["output"]), Path(second["output"])
    assert json.loads((a / "comparisons.json").read_text()) == json.loads(
        (b / "comparisons.json").read_text()
    )
    assert first["attempted_trials"] == second["attempted_trials"] == 34
    assert not first["economic_comparison"]["executed"]
    assert all(
        v["status"] != "PROMISING WITHIN DECLARED HISTORICAL SCOPE"
        for v in first["candidate_verdicts"].values()
    )
    scores = read_rows(a / "forecast_scores.jsonl")
    forecasts = read_rows(a / "forecasts.jsonl")
    assert all(
        datetime.fromisoformat(r["available_utc"])
        >= datetime.fromisoformat(r["bar_open_utc"]) + timedelta(seconds=plan.seconds)
        for r in forecasts
    )
    assert scores and all(
        r["outcome_matured_utc"] < plan.folds[int(r["fold_id"][-1]) - 1].end.isoformat()
        for r in scores
    )
    for file in (a / "frozen").glob("*/*.json"):
        spec = json.loads(file.read_text())
        if "training_label_as_of_utc" in spec:
            assert spec["training_label_as_of_utc"] < spec["fitting_cutoff_utc"]
            assert spec["parameter_information_as_of_utc"] < spec["fitting_cutoff_utc"]


def test_appending_future_bars_does_not_change_emitted_earlier_forecasts(tmp_path: Path) -> None:
    plan, c, source = fixture()
    config = replace(load_config(), project_root=tmp_path)

    first = evaluate(config, c, plan, "PREFIX_V001", synthetic=source)
    extended = SyntheticBars(
        source.opens + [at(i) for i in range(len(source.opens), len(source.opens) + 30)],
        source.prices + [99999.0] * 30,
        source.seconds,
    )
    second = evaluate(config, c, plan, "EXTENDED_V001", synthetic=extended)
    a, b = Path(first["output"]), Path(second["output"])
    assert (a / "forecasts.jsonl").read_bytes() == (b / "forecasts.jsonl").read_bytes()
    relative = f"frozen/{plan.folds[0].fold_id}/ARX.json"
    assert (a / relative).read_bytes() == (b / relative).read_bytes()


def test_missing_provenance_cannot_promote_favorable_effects() -> None:
    plan, _, _ = fixture()
    plan = replace(plan, evaluation_history_classification="historical_reconstructed")
    favorable = {
        "matched_rows": 1000,
        "mean_loss_improvement": 1.0,
        "return_relative_mse_improvement": 0.1,
        "positive_fold_fraction": 1.0,
        "uncertainty": {"status": "estimated", "mean_loss_improvement_95_interval": [0.1, 2.0]},
    }
    current = gates()
    current["evidence_identity"] = "unknown"
    assert addition_verdict([favorable], current, plan)["status"] == "INCONCLUSIVE"
    assert addition_verdict([favorable], {}, plan)["status"] == "INCONCLUSIVE"
    assert (
        addition_verdict([favorable], gates(), plan, failed=True)["status"]
        == "BLOCKED BY MISSING INPUTS OR EVIDENCE"
    )


def test_target_mismatch_and_duplicate_comparison_records_raise() -> None:
    plan, _, _ = fixture()
    row = {
        "fold_id": "F01",
        "row_id": "R1",
        "status": "scored",
        "target": "future_return",
        "units": "log_mid_return",
        "horizon_bars": 1,
        "grid_seconds": 300,
        "available_utc": at(1000).isoformat(),
        "actual": 0.02,
        "target_end_utc": at(1001).isoformat(),
        "value": 0.01,
    }
    a, b = {**row, "model": "ARX"}, {**row, "model": "ZERO", "value": 0.0}
    report = compare([a, b], "ARX", "ZERO", plan)
    assert report["mean_loss_improvement"] == pytest.approx(0.0003)
    assert report["uncertainty"]["status"] == "insufficient"
    with pytest.raises(ValueError, match="horizon/observations"):
        compare([a, {**b, "horizon_bars": 3}], "ARX", "ZERO", plan)
    with pytest.raises(ValueError, match="duplicate"):
        compare([a, a, b], "ARX", "ZERO", plan)
    assert loss(2, 1, "realized_variance") == pytest.approx(0.6931471805599453 + 0.5)


def test_final_period_and_real_readiness_cannot_be_overridden_by_fixtures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, c, source = fixture()
    config = replace(load_config(), project_root=tmp_path)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("software-classified fixture opened real metadata")

    monkeypatch.setattr("xauusd_quant.econometrics.runs.assess", forbidden)
    with pytest.raises(ValueError, match="real scientific"):
        evaluate(
            config,
            c,
            replace(plan, evaluation_history_classification="historical_reconstructed"),
            "BAD_V001",
            synthetic=source,
        )
    with pytest.raises(ValueError, match="2022"):
        validate_interval(config, at(0).replace(year=2022), at(1).replace(year=2022))
    with pytest.raises(ValueError, match="requires a declared synthetic"):
        evaluate(config, c, plan, "NO_FIXTURE_V001")

    def absent(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError("explicit missing input fixture; no real values opened")

    monkeypatch.setattr("xauusd_quant.econometrics.runs.assess", absent)
    blocked = evaluate(
        config,
        c,
        replace(plan, evaluation_history_classification="historical_reconstructed"),
        "MISSING_V001",
    )
    assert (
        blocked["attempted_trials"] == 0
        and blocked["status"] == "BLOCKED BY MISSING INPUTS OR EVIDENCE"
    )


def test_failed_model_trial_is_preserved_alongside_completed_benchmarks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, c, source = fixture()
    plan = replace(plan, folds=plan.folds[:1])
    config = replace(load_config(), project_root=tmp_path)

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("deliberate GARCH fitting failure")

    monkeypatch.setattr("xauusd_quant.econometrics.runs.fit_garch", fail)
    result = evaluate(config, c, plan, "FAILED_V001", synthetic=source)
    entries = read_rows(Path(result["output"]) / "trial_ledger.jsonl")
    assert any(r["status"] == "failed" and "GARCH" in r["reason"] for r in entries)
    assert len([r for r in entries if r["status"] == "attempted"]) == plan.expected_trials
    assert (
        result["candidate_verdicts"]["GARCH"]["status"] == "BLOCKED BY MISSING INPUTS OR EVIDENCE"
    )
