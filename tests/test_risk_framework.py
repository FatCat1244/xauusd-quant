"""Immutable risk artifacts, complete ledgers and absence of market-data entry."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from risk_synth import ROOT, configuration
from xauusd_quant.alpha_portfolio.plan import freeze
from xauusd_quant.execution.config import content_hash
from xauusd_quant.risk.contracts import resolved
from xauusd_quant.risk.policy import load_configuration
from xauusd_quant.risk.runs import PLAN, freeze_plan, readiness, run
from xauusd_quant.utils.config import load_config


def test_configuration_schema_and_frozen_settings(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    cfg = configuration()
    freeze(path, resolved(cfg))
    envelope = json.loads(path.read_text())
    assert envelope["content_sha256"] == content_hash(envelope["body"])
    freeze(path, resolved(cfg))
    with pytest.raises(ValueError):
        freeze(path, {"different": "limits under same id"})
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("configuration_id: INVALID_V001\npolicy: {}\naccount: null\ninstrument: null\n", encoding="utf-8")
    with pytest.raises((KeyError, TypeError, ValueError)):
        load_configuration(invalid)


def test_no_registry_does_not_become_eligible_or_ready(tmp_path: Path) -> None:
    audit = readiness(tmp_path, configuration())
    assert audit["status"] == "BLOCKED" and audit["alpha_status"] == "NO_ELIGIBLE_ALPHAS"
    assert not audit["strategy_active"] and not audit["evidence"]["registry_available"]


def test_real_readiness_opens_no_ticks_and_preserves_run_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from xauusd_quant.execution import io

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("market-data access forbidden")
    monkeypatch.setattr(io, "tick_quotes", forbidden)
    config = replace(load_config(), project_root=tmp_path)
    report = run(config, "RISK_READINESS_V001", "readiness", ROOT / "config/risk.yaml")
    assert report["readiness"]["configuration_state"] == "UNCONFIGURED"
    assert not report["real_data_replay_completed"] and not report["strategy_active"]
    with pytest.raises(FileExistsError):
        run(config, "RISK_READINESS_V001", "readiness", ROOT / "config/risk.yaml")
    assert freeze_plan(tmp_path).is_file()


def test_bounded_replay_full_ledger_and_failed_exit_remains(tmp_path: Path) -> None:
    config = replace(load_config(), project_root=tmp_path)
    result = run(config, "RISK_SYNTH_V001", "replay", ROOT / "config/risk_synthetic.yaml")
    assert result["attempted_trials"] == 14 == PLAN["budget"]["trials"]
    assert len(result["synthetic_comparisons"]) == 14
    ledger = tmp_path / "results/risk/runs/RISK_SYNTH_V001/trial_ledger.jsonl"
    statuses = [json.loads(row)["status"] for row in ledger.read_text().splitlines()]
    assert statuses.count("completed") == 14 and statuses.count("attempted") == 14
    feed = next(x for x in result["synthetic_comparisons"] if x["scenario"] == "invalid_feed" and x["governed"])
    assert feed["risk"]["actual_account"]["signed_lots"] > 0 and feed["risk"]["unresolved_actions"]
    loss = next(x for x in result["synthetic_comparisons"] if x["scenario"] == "loss_halt" and x["governed"])
    assert loss["risk"]["actual_account"]["signed_lots"] == 0 and not loss["risk"]["unresolved_actions"]
    assert loss["net_closed_pnl"] < -40  # Gap exceeds configured drawdown; no ceiling guarantee.
