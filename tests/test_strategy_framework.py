"""Immutable plans, recorded failures, deterministic family evaluation and honest verdicts."""

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from strategy_synth import at, fixture, gates
from xauusd_quant.execution.io import DiskRecorder, validate_interval
from xauusd_quant.strategy_validation.metrics import (
    AuditRecorder,
    daily_uncertainty,
    evidence_verdict,
)
from xauusd_quant.strategy_validation.plan import freeze_plan
from xauusd_quant.strategy_validation.runs import TrialLedger, run_framework
from xauusd_quant.utils.config import load_config


def test_plan_is_frozen_before_values_and_changes_need_new_version(tmp_path: Path) -> None:
    plan, c, _, _ = fixture()
    path = freeze_plan(tmp_path, plan, c)
    original = path.read_bytes()
    assert freeze_plan(tmp_path, plan, c) == path
    with pytest.raises(ValueError, match="immutable plan changed"):
        freeze_plan(tmp_path, replace(plan, policy_thresholds=(0.0, 0.00002)), c)
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="budget"):
        replace(plan, trial_budget=1)
    with pytest.raises(ValueError, match="cannot claim prospective"):
        replace(plan, evaluation_history_classification="prospective")


def test_nested_run_is_deterministic_and_full_family_remains_reported(tmp_path: Path) -> None:
    plan, c, source, quotes = fixture()
    config = replace(load_config(), project_root=tmp_path)
    ready = {"gates": gates(), "classification": "software_correctness"}
    first = run_framework(
        config, c, plan, "FIRST_V001", source=source, quotes=quotes, readiness=ready
    )
    _, _, source, quotes = fixture()
    second = run_framework(
        config, c, plan, "SECOND_V001", source=source, quotes=quotes, readiness=ready
    )
    assert first["results"] == second["results"]
    assert first["attempted_trials"] == plan.expected_trials == 14
    assert first["candidate_verdicts"]["RIDGE"]["status"] == "INCONCLUSIVE"
    directory = Path(first["output"])
    specs = list((directory / "frozen").glob("*/*.json"))
    assert len(specs) == 2
    assert all(
        json.loads(p.read_text())["training_label_as_of_utc"]
        < json.loads(p.read_text())["cutoff_utc"]
        for p in specs
    )
    manifests = json.loads((directory / "manifest.json").read_text())
    assert len(manifests["all_economic_variants"]) == 4
    assert first["results"][2]["variant"]["kind"] == "no_trading"
    assert first["results"][2]["result"]["execution"]["counts"].get("fills", 0) == 0


def test_failed_trials_are_retained_and_trial_budget_does_not_expand(tmp_path: Path) -> None:
    def fail() -> None:
        raise ValueError("negative or failed trial must remain recorded")

    with DiskRecorder(tmp_path) as recorder:
        ledger = TrialLedger(recorder, 1)
        assert ledger.call("fit", {"id": 1}, fail) is None
        with pytest.raises(RuntimeError, match="budget exhausted"):
            ledger.call("fit", {"id": 2}, fail)
    rows = [json.loads(line) for line in (tmp_path / "trial_ledger.jsonl").read_text().splitlines()]
    assert [r["status"] for r in rows] == ["attempted", "failed"]


def favorable_result() -> dict[str, Any]:
    """Hand-declared adequate evidence isolates the economic verdict's rules."""
    return {
        "aggregate": {
            "marked_equity_change_account": 100.0,
            "folds": 4,
            "known_fold_marks": 4,
            "positive_fold_fraction": 0.75,
        },
        "execution": {"counts": {"closed_positions": 40}, "open_position": None},
        "uncertainty": {"status": "estimated", "mean_daily_pnl_95_interval": [1.0, 3.0]},
        "concentration": {
            "largest_trade_positive_share": 0.2,
            "largest_fold_positive_share": 0.4,
            "closed_net_without_largest_positive_trades_account": 80.0,
        },
    }


def test_unknown_null_blocks_even_a_favorable_result() -> None:
    plan, _, _, _ = fixture()
    plan = replace(plan, evaluation_history_classification="historical_reconstructed")
    current = gates()
    current["null_evidence"] = "unknown"
    result = favorable_result()
    assert evidence_verdict(current, result, plan, [result])["status"] == "BLOCKED"
    assert evidence_verdict({}, {}, plan, [])["status"] == "BLOCKED"


def test_historical_verdict_requires_the_whole_registered_family() -> None:
    plan, _, _, _ = fixture()
    plan = replace(plan, evaluation_history_classification="historical_reconstructed")
    result = favorable_result()
    assert evidence_verdict(gates(), result, plan, [result])["status"] == (
        "PASSES DECLARED HISTORICAL CRITERIA"
    )
    adverse = favorable_result()
    adverse["aggregate"]["marked_equity_change_account"] = -1.0
    verdict = evidence_verdict(gates(), result, plan, [result, adverse])
    assert verdict["status"] == "REJECTED"
    assert not verdict["checks"]["all_registered_robustness"]


@pytest.mark.parametrize("pnl", [None, float("nan"), float("inf")])
def test_unknown_economics_remain_inconclusive(pnl: float | None) -> None:
    plan, _, _, _ = fixture()
    plan = replace(plan, evaluation_history_classification="historical_reconstructed")
    result = favorable_result()
    result["aggregate"]["marked_equity_change_account"] = pnl
    assert evidence_verdict(gates(), result, plan, [result])["status"] == "INCONCLUSIVE"


def test_readiness_blocks_before_any_evaluation_values(tmp_path: Path) -> None:
    plan, c, source, quotes = fixture()
    config = replace(load_config(), project_root=tmp_path)

    def forbidden(*args: Any) -> Any:
        pytest.fail("blocked run opened values")

    source.training = forbidden
    source.evaluation = forbidden
    current = gates()
    current["null_evidence"] = "unknown"
    result = run_framework(
        config,
        c,
        plan,
        "BLOCKED_V001",
        source=source,
        quotes=forbidden,
        readiness={"gates": current},
    )
    assert result["attempted_trials"] == 0
    assert result["candidate_verdicts"]["RIDGE"]["status"] == "BLOCKED"
    assert list((Path(result["output"]) / "trial_ledger.jsonl").read_text().splitlines())


def test_daily_bootstrap_requires_calendar_coverage_and_is_deterministic() -> None:
    plan, _, _, _ = fixture()
    marks = [
        {"timestamp_utc": at(0) + timedelta(days=i), "equity_account": 10000 + i} for i in range(26)
    ]
    result = daily_uncertainty(marks, plan, 300)
    assert result == daily_uncertainty(marks, plan, 300)
    assert result["mean_daily_pnl_95_interval"] == [1, 1]
    assert daily_uncertainty(marks[:5], plan, 300)["status"] == "insufficient"
    assert daily_uncertainty(marks, plan, 5 * 86400)["status"] == "unknown"
    marks[3]["equity_account"] = None
    assert daily_uncertainty(marks, plan, 300)["status"] == "unknown"


def test_duplicate_trade_aggregation_is_rejected() -> None:
    recorder = AuditRecorder(lambda *_: None)
    trade = {"position_id": 1, "entry_utc": at(0), "exit_utc": at(5)}
    recorder("trades", trade)
    with pytest.raises(ValueError, match="duplicate closed position"):
        recorder("trades", trade)


def test_reserved_execution_guard_cannot_be_bypassed_by_new_plan() -> None:
    config = load_config()
    from xauusd_quant.execution.io import parse_utc

    with pytest.raises(ValueError, match="2022"):
        validate_interval(
            config, parse_utc("2022-01-02T00:00:00Z"), parse_utc("2022-01-03T00:00:00Z")
        )
