"""Unknown scientific/specification evidence cannot promote an economic result."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from xauusd_quant.execution.config import ExecutionConfig, load_execution_config
from xauusd_quant.execution.io import parse_utc, validate_interval, write_new_json
from xauusd_quant.execution.readiness import ADAPTIVE_CHOICES, promotion_status, scientific_gates
from xauusd_quant.execution.runs import new_run, synthetic_smoke
from xauusd_quant.utils.config import load_config

START = datetime(2021, 1, 4, tzinfo=UTC)


def evidence() -> dict[str, Any]:
    return {
        "timeframe": "5m",
        "target": "future_return",
        "units": "log_mid_return",
        "horizon_bars": 1,
        "feature_set_ids": ["NEW_FOLD_LOCAL_V002"],
        "evaluation_history_classification": "untouched_verified",
        "folds": [
            {
                "evaluation_start_utc": "2021-01-01T00:00:00Z",
                "evaluation_end_utc": "2022-01-01T00:00:00Z",
                "adaptive_as_of_utc": dict.fromkeys(ADAPTIVE_CHOICES, "2020-12-31T00:00:00Z"),
                "nulls": {"random_walk": "passed", "sign_flip": "passed"},
            }
        ],
    }


def test_missing_scientific_or_specification_evidence_cannot_promote() -> None:
    assert promotion_status({}) == "historical_diagnostic_only"
    c = ExecutionConfig()
    gates = scientific_gates({}, START, c)
    assert gates["null_evidence"] == "unknown"
    assert gates["broker_specification"] == "unknown"
    assert promotion_status(gates) == "historical_diagnostic_only"


@pytest.mark.parametrize("choice", ADAPTIVE_CHOICES)
def test_every_adaptive_choice_must_precede_the_evaluation_fold(choice: str) -> None:
    body = evidence()
    body["folds"][0]["adaptive_as_of_utc"][choice] = "2021-06-01T00:00:00Z"
    gates = scientific_gates(body, START, ExecutionConfig())
    assert gates["forecast_chronology"] == "failed"
    assert promotion_status(gates) == "historical_diagnostic_only"


def test_global_v001_selection_and_inspected_history_cannot_be_papered_over() -> None:
    body = evidence()
    body["feature_set_ids"] = ["FEATURESET_5M_STANDARD_V001"]
    body["evaluation_history_classification"] = "historical_inspected"
    gates = scientific_gates(body, START, ExecutionConfig())
    assert gates["forecast_chronology"] == gates["untouched_evaluation"] == "failed"
    del body["folds"][0]["nulls"]["sign_flip"]
    assert scientific_gates(body, START, ExecutionConfig())["null_evidence"] == "unknown"


def test_complete_supplied_audit_still_needs_data_and_evidence_identity() -> None:
    c = replace(
        ExecutionConfig(),
        specification_status="verified_supplied",
        specification_source="fixture specification",
    )
    gates = scientific_gates(evidence(), START, c)
    assert gates["forecast_chronology"] == "passed"
    assert promotion_status(gates) == "historical_diagnostic_only"
    gates.update(data_provenance="passed", evidence_identity="passed")
    assert promotion_status(gates) == "eligible_for_economic_review"


def test_audited_fold_at_start_cannot_cover_unaudited_later_forecasts() -> None:
    body = evidence()
    body["folds"][0]["evaluation_end_utc"] = "2021-02-01T00:00:00Z"
    end = datetime(2021, 4, 1, tzinfo=UTC)
    assert (
        scientific_gates(body, START, ExecutionConfig(), evaluation_end=end)["forecast_chronology"]
        == "failed"
    )
    later = dict(body["folds"][0])
    later.update(
        evaluation_start_utc="2021-03-01T00:00:00Z", evaluation_end_utc="2021-05-01T00:00:00Z"
    )
    body["folds"].append(later)
    assert (
        scientific_gates(body, START, ExecutionConfig(), evaluation_end=end)["forecast_chronology"]
        == "failed"
    )
    later["evaluation_start_utc"] = "2021-02-01T00:00:00Z"
    assert (
        scientific_gates(body, START, ExecutionConfig(), evaluation_end=end)["forecast_chronology"]
        == "passed"
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"quantity_lots": 0.015},
        {"contract_ounces_per_lot": float("nan")},
        {"commission_account_per_lot_per_leg": -1},
        {"slippage_usd_per_ounce_per_leg": -1},
        {"account_currency_per_usd": 0},
        {"latency_ms": -1},
        {"order_ttl_ms": 0},
        {"financing_long_account_per_lot_per_day": float("nan")},
        {"policy_units": "log_volatility"},
        {"end_policy": "retroactive_flatten"},
        {"batch_rows": 0},
    ],
)
def test_invalid_units_and_costs_fail_before_any_run(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        replace(ExecutionConfig(), **changes)


def test_financing_never_defaults_silently_in_file_configuration(tmp_path: Path) -> None:
    body = ExecutionConfig().resolved()
    del body["financing_long_account_per_lot_per_day"]
    path = tmp_path / "execution.yaml"
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    with pytest.raises(ValueError, match="explicit execution assumptions"):
        load_execution_config(path)


def test_utc_is_explicit_and_stage12_cannot_open_a_new_reserved_path() -> None:
    c = load_config()
    with pytest.raises(ValueError, match="explicit UTC"):
        parse_utc("2021-01-04T00:00:00")
    with pytest.raises(ValueError, match="2022"):
        validate_interval(c, parse_utc("2021-12-31T00:00:00Z"), parse_utc("2022-01-02T00:00:00Z"))


def test_synthetic_run_outputs_are_reconcilable_and_immutable(tmp_path: Path) -> None:
    config = replace(load_config(), project_root=tmp_path)
    result = synthetic_smoke(config, "TEST_SMOKE_V001")
    directory = Path(result["output"])
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["stage"] == 12 and manifest["evaluation_history_classification"] == "synthetic"
    assert len(manifest["configuration_sha256"]) == 64
    assert result["counts"]["fills"] == 2
    assert result["final_mark"]["reconciliation_error_account"] == pytest.approx(0, abs=1e-8)
    original = (directory / "manifest.json").read_bytes()
    with pytest.raises((ValueError, FileExistsError)):
        synthetic_smoke(config, "TEST_SMOKE_V001")
    assert (directory / "manifest.json").read_bytes() == original
    with pytest.raises(FileExistsError):
        write_new_json(directory / "summary.json", {})
    with pytest.raises(ValueError, match="versioned"):
        new_run(config, "../../data")
