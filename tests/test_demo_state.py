"""Durability, concurrent writer exclusion and unchanged readiness gates."""
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from demo_synth import create, propose
from xauusd_quant.demo.config import DemoConfig
from xauusd_quant.demo.coordinator import Coordinator
from xauusd_quant.demo.journal import Journal
from xauusd_quant.demo.runs import account_directory, prerequisites, readiness

ROOT = Path(__file__).resolve().parents[1]


def test_missing_actual_settings_and_strategy_evidence_block() -> None:
    report = readiness(ROOT, DemoConfig())
    assert report["status"] == "BLOCKED"
    assert "quantity_lots" in report["missing_execution_fields"]
    assert report["missing_risk_fields"]["policy"]
    strategy = readiness(ROOT, DemoConfig(run_type="STRATEGY"))
    assert "NO_ELIGIBLE_ALPHAS" in strategy["blocking_reasons"]
    assert "FULL_LIVE_PIPELINE_EQUALITY_UNVERIFIED" in strategy["blocking_reasons"]
    assert not prerequisites(ROOT)["eligible_alphas"]


@pytest.mark.parametrize("kwargs", [{"mode": "REAL"}, {"mode": "CONTEST"}, {"entry_budget": 99}, {"max_duration_seconds": 301}, {"quantity_lots": float("nan")}, {"deviation_points": "1"}, {"dedicated_account_confirmed": "true"}, {"filling_policy": "RETURN"}])
def test_invalid_configuration_explicitly_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        DemoConfig(**kwargs)


def test_single_writer_excludes_another_process_and_releases_on_close(tmp_path: Path) -> None:
    j = Journal(tmp_path)
    program = "from pathlib import Path; from xauusd_quant.demo.journal import Journal; Journal(Path(__import__('sys').argv[1]))"
    try:
        other = subprocess.run([sys.executable, "-c", program, str(tmp_path)], capture_output=True, text=True, timeout=15)
        assert other.returncode != 0
        assert "already has a writer" in other.stderr
    finally:
        j.close()
    reopened = Journal(tmp_path)
    reopened.close()


def test_same_account_different_terminal_and_polling_share_lock(tmp_path: Path) -> None:
    c, _, j = create(tmp_path)
    try:
        different = replace(c.terminal, terminal_path="OTHER_TERMINAL.exe", duration_seconds=2, poll_seconds=.5)
        assert account_directory(ROOT, c.terminal) == account_directory(ROOT, different)
    finally:
        j.close()


def test_corrupt_or_truncated_journal_never_defaults_ready(tmp_path: Path) -> None:
    j = Journal(tmp_path)
    j.append("fixture", {"value": 1})
    j.close()
    path = tmp_path / "journal.jsonl"
    row = json.loads(path.read_text(encoding="utf-8"))
    row["body"]["data"]["value"] = 2
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="chain"):
        Journal(tmp_path)


def test_config_change_and_uncheckpointed_tail_do_not_erase_halt(tmp_path: Path) -> None:
    c, b, j = create(tmp_path)
    propose(c, b)
    c.halt("SYNTHETIC_MANUAL_HALT", b.now)
    cfg, terminal, risk = c.config, c.terminal, c.risk.configuration
    j.close()
    b.authority = None
    reopened = Journal(tmp_path / "runtime")
    try:
        with pytest.raises(ValueError, match="incompatible"):
            Coordinator(replace(cfg, magic=999), terminal, risk, b, reopened,
                        "RESTART_V001", "SYNTHETIC_CODE", offline_synthetic=True)
        reopened.append("uncheckpointed_failure", {"failure": True})
    finally:
        reopened.close()
    b.authority = None
    reopened = Journal(tmp_path / "runtime")
    try:
        with pytest.raises(ValueError, match="tail"):
            Coordinator(cfg, terminal, risk, b, reopened, "RESTART_V001", "SYNTHETIC_CODE", offline_synthetic=True)
    finally:
        reopened.close()
