"""Immutable plans/registries, actual missing evidence and guarded offline entry."""

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from xauusd_quant.alpha_portfolio.plan import PortfolioPlan, freeze, freeze_plan
from xauusd_quant.alpha_portfolio.registry import build_registry, freeze_registry
from xauusd_quant.alpha_portfolio.runs import run
from xauusd_quant.execution.config import ExecutionConfig, content_hash
from xauusd_quant.execution.io import validate_interval
from xauusd_quant.utils.config import load_config


def evidence(root: Path) -> PortfolioPlan:
    plan = PortfolioPlan("PORTFOLIO_V001", "REGISTRY_V001", "EVIDENCE_V001")
    path = root / "results/robustness/runs" / plan.evidence_run
    path.mkdir(parents=True)
    verdict = {
        "candidate_verdicts": {
            **{
                m: {"status": "BLOCKED", "reasons": ["null/specification missing"]}
                for m in ("CAUSAL_RIDGE", "HISTORICAL_MEAN", "LEGACY_ML_ENSEMBLES")
            },
            **{
                f"ECON_{tf}_DIAGNOSTIC_V001:{m}": {
                    "status": "BLOCKED",
                    "reasons": ["matching null missing"],
                }
                for tf in ("5M", "15M")
                for m in ("ARX", "KALMAN", "GARCH", "HAR")
            },
        }
    }
    inventory = {
        "metadata": {"dataset_version": "synthetic", "frozen_specs": [], "evaluation_history": []},
        "selection_exposure_quantification": "unknown",
        "retrospective_selection": "unknown",
        "historical_recorded_statuses": {"failed": 1},
        "missing_prerequisites": ["matching nulls"],
    }
    for name, body in (
        ("verdict.json", verdict),
        ("inventory.json", inventory),
        ("manifest.json", {}),
    ):
        (path / name).write_text(json.dumps(body), encoding="utf-8")
    return plan


def test_registry_missing_science_inactive_and_failed_candidates_retained(tmp_path: Path) -> None:
    plan = evidence(tmp_path)
    registry = build_registry(tmp_path, plan)
    assert registry["result"] == "NO_ELIGIBLE_ALPHAS"
    assert not registry["eligible_alphas"] and not registry["active"]
    assert len(registry["candidates"]) == 16
    assert sum(c["diagnostic_only"] for c in registry["candidates"]) == 4
    assert all(c["eligibility"] == "BLOCKED" for c in registry["candidates"])
    assert registry["historical_recorded_statuses"]["failed"] == 1
    assert freeze_registry(tmp_path, plan) == freeze_registry(tmp_path, plan)


def test_plan_and_registry_immutable_and_missing_inputs_rejected(tmp_path: Path) -> None:
    plan = evidence(tmp_path)
    path = freeze_plan(tmp_path, plan)
    with pytest.raises(ValueError, match="immutable"):
        freeze(path, {"changed": True})
    other = tmp_path / "missing"
    with pytest.raises(ValueError, match="missing"):
        build_registry(other, plan)


def test_changed_evidence_refused_before_verdict_open(tmp_path: Path) -> None:
    plan = evidence(tmp_path)
    freeze_plan(tmp_path, plan)
    file = tmp_path / "results/robustness/runs" / plan.evidence_run / "verdict.json"
    file.write_text("OUTCOME MUST NOT BE PARSED", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable"):
        build_registry(tmp_path, plan)


def test_empty_evaluation_never_opens_market_forecasts_or_ticks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = evidence(tmp_path)
    config = replace(load_config(), project_root=tmp_path)
    (tmp_path / "config").mkdir()
    import yaml

    (tmp_path / "config/execution.yaml").write_text(
        yaml.safe_dump(ExecutionConfig().resolved()), encoding="utf-8"
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("market loader reached despite empty eligible universe")

    monkeypatch.setattr("xauusd_quant.execution.io.tick_quotes", forbidden)
    monkeypatch.setattr("xauusd_quant.execution.io.forecast_records", forbidden)
    report = run(config, plan, "EMPTY_V001", "evaluate")
    assert report["result"] == "NO_ELIGIBLE_ALPHAS"
    assert report["market_study_executed"] is False
    assert report["attempted_trials"] == 1
    assert (Path(report["output"]) / "shared_account_V001.json").is_file()
    with pytest.raises(FileExistsError):
        run(config, plan, "EMPTY_V001", "evaluate")


def test_retained_reserved_guard() -> None:
    with pytest.raises(ValueError, match="2022"):
        validate_interval(
            load_config(), datetime(2022, 1, 1, tzinfo=UTC), datetime(2022, 1, 2, tzinfo=UTC)
        )


def test_bounded_plan_rejects_search_expansion() -> None:
    plan = PortfolioPlan("P_V001", "R_V001", "E_V001")
    with pytest.raises(ValueError):
        replace(plan, methods=("equal", "kelly"))
    with pytest.raises(ValueError):
        replace(plan, trial_budget=100)
    with pytest.raises(ValueError):
        replace(plan, budget_lots=0.1)


def test_frozen_datetime_hash_matches_serialized_body_and_reuse(tmp_path: Path) -> None:
    body = {
        "effective_utc": datetime(2021, 1, 4, tzinfo=UTC),
        "nested": [{"cutoff": datetime(2021, 1, 3, tzinfo=UTC)}],
    }
    path = freeze(tmp_path / "allocation_V001.json", body)
    saved = json.loads(path.read_text())
    assert content_hash(saved["body"]) == saved["content_sha256"]
    assert freeze(path, body) == path


def test_corrected_stage13_fits_referenced_without_erasing_failed_attempt(tmp_path: Path) -> None:
    plan = evidence(tmp_path)
    path = tmp_path / "results/strategy_validation/runs/STRATEGY_5M_DIAGNOSTIC_V002/frozen/F01"
    path.mkdir(parents=True)
    file = path / "CAUSAL_RIDGE.json"
    file.write_text(
        json.dumps(
            {
                "fold_id": "F01",
                "features": ["return_1"],
                "training_ids_sha256": "actual-training-identity",
                "training_label_as_of_utc": "2021-05-31T23:55:00+00:00",
                "cutoff_utc": "2021-06-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    registry = build_registry(tmp_path, plan)
    ridge = next(c for c in registry["candidates"] if c["alpha_id"] == "CAUSAL_RIDGE_5M_T0")
    assert ridge["provenance"]["model"]
    assert not ridge["provenance"]["missing_frozen_fit_artifacts"]
    assert ridge["provenance"]["features"]["F01"]["names"] == ["return_1"]
    assert "STRATEGY_5M_DIAGNOSTIC_V001" in registry["upstream_sources"]
    assert ridge["eligibility"] == "BLOCKED"
