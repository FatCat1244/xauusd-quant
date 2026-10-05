"""CLI/gate isolation: unconfigured runs cannot even construct native adapters."""
from pathlib import Path

import pytest

from xauusd_quant.cli import build_parser
from xauusd_quant.demo.config import DemoConfig, load_demo_config
from xauusd_quant.demo.runs import cli_command
from xauusd_quant.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_public_templates_remain_unconfigured() -> None:
    assert not load_demo_config(ROOT / "config/demo.yaml").configured
    assert not DemoConfig().configured


def test_demo_commands_are_distinct_from_shadow() -> None:
    parser = build_parser()
    for command in ("demo-plan", "demo-readiness", "demo-preflight", "demo-precheck",
                    "demo-smoke", "demo-strategy", "demo-reconcile", "demo-recover-close"):
        args = parser.parse_args([command, "--run-id", "SYNTHETIC_V001"])
        assert args.command == command
    with pytest.raises(SystemExit):
        parser.parse_args(["shadow-run", "--mode", "DEMO"])


@pytest.mark.parametrize("command", ["demo-smoke", "demo-strategy", "demo-precheck"])
def test_missing_config_blocks_before_native_construction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str) -> None:
    from dataclasses import replace

    from xauusd_quant.demo import runs

    def forbidden(*args: object, **kwargs: object) -> object:
        pytest.fail("blocked configuration reached native adapter")

    monkeypatch.setattr(runs, "DemoProcessBroker", forbidden)
    args = build_parser().parse_args([command, "--run-id", "BLOCKED_V001",
        "--demo-config", str(ROOT / "config/demo.yaml")])
    configuration = replace(load_config(), project_root=tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config/risk.yaml").write_bytes((ROOT / "config/risk.yaml").read_bytes())
    assert cli_command(configuration, args) == 1
    assert (tmp_path / "results/demo/runs/BLOCKED_V001/verdict.json").is_file()


def test_interruption_around_submission_preserves_uncertainty_and_stops_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace
    from typing import Any

    from demo_synth import FakeBroker, smoke_config, smoke_risk
    from shadow_synth import local_config
    from xauusd_quant.demo import runs
    from xauusd_quant.demo.broker import Permit

    cfg, risk = smoke_config(), smoke_risk()
    assert risk.policy is not None and risk.account is not None and risk.instrument is not None
    # Fake boundary test only: local terms cannot reach an installed native package.
    risk = replace(risk, policy=replace(risk.policy, status="verified_supplied"),
        account=replace(risk.account, status="verified_supplied"), instrument=replace(risk.instrument, status="verified_supplied"))
    terminal = local_config(tmp_path / "terminal64.exe")
    fake = FakeBroker(cfg)

    def interrupted(request: dict[str, Any], permit: Permit) -> None:
        fake.sent.append(request)
        raise KeyboardInterrupt

    fake.send = interrupted  # type: ignore[method-assign]
    monkeypatch.setattr(runs, "readiness", lambda *args: {"blocking_reasons": []})
    monkeypatch.setattr(runs, "load_configuration", lambda *args: risk)
    monkeypatch.setattr(runs, "load_shadow_config", lambda *args: terminal)
    monkeypatch.setattr(runs, "DemoProcessBroker", lambda *args: fake)
    directory = tmp_path / "study"
    result = runs.native_run(tmp_path, cfg, directory, "demo-smoke", "INTERRUPTED_V001")
    assert result["status"] == "HALTED_OR_UNRESOLVED"
    assert result["failure"] == "INTERRUPTED_RECONCILIATION_REQUIRED"
    assert not result["verified_flat"] and result["reservations"] == 1
    assert len(fake.sent) == 1 and fake.shutdown_count == 1
