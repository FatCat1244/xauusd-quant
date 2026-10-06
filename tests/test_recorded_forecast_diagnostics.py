"""Recorded-feed inference preserves training, clocks, gaps and bounded state."""
import inspect
import json
import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scripts import diagnose_recorded_forecasts as diagnostic

from xauusd_quant.alpha_portfolio.plan import freeze
from xauusd_quant.execution.readiness import sha256
from xauusd_quant.risk.runs import evidence
from xauusd_quant.shadow.runs import code_identity
from xauusd_quant.strategy_validation.pipeline import FrozenPipeline

T = datetime(2026, 10, 6, 12, tzinfo=UTC)


def model() -> FrozenPipeline:
    return FrozenPipeline("CAUSAL_RIDGE", "F02", T - timedelta(days=10),
        ("return_1", "momentum_3", "volatility_3"), (0., 0., 0.), (1., 1., 1.),
        (.2, -.1, .3), .00001, "SYNTHETIC_IDS", T - timedelta(days=11), 50)


def bar(index: int, *, valid: bool = True, timeframe: str = "5m") -> dict[str, Any]:
    seconds = int(timeframe[:-1]) * 60
    opened = T + timedelta(seconds=seconds * index)
    return {"timeframe": timeframe, "values": {"timestamp": opened.isoformat(),
        "close": 1800 * math.exp(.001 * index * index)}, "valid": valid,
        "available_utc": (opened + timedelta(seconds=seconds + 1)).isoformat(), "backfill": False}


def test_hand_checked_feature_and_frozen_prediction() -> None:
    service = diagnostic.Diagnostic({"5m": model()}, "PINNED")
    output = [service.update(bar(i)) for i in range(4)]
    assert all(r["value"] is None for r in output[:3])
    returns = np.asarray([.001, .003, .005])
    expected = (.005, .009, float(np.std(returns)))
    assert list(output[-1]["features"].values()) == pytest.approx(expected, abs=1e-14)
    assert output[-1]["value"] == pytest.approx(.00001 + .2 * expected[0] - .1 * expected[1] + .3 * expected[2])
    assert not output[-1]["trade_intent"]


def test_appending_future_bars_and_restarting_preserves_predictions() -> None:
    service = diagnostic.Diagnostic({"5m": model(), "15m": model()}, "PINNED")
    prefix = [service.update(bar(i)) for i in range(5)]
    saved = json.loads(json.dumps(service.state()))
    resumed = diagnostic.Diagnostic(service.models, "PINNED")
    resumed.restore(saved)
    appended = [service.update(bar(i)) for i in range(5, 8)]
    assert appended == [resumed.update(bar(i)) for i in range(5, 8)]
    independent = diagnostic.Diagnostic(service.models, "PINNED")
    assert prefix + appended == [independent.update(bar(i)) for i in range(8)]
    assert independent.history["15m"] == [] and len(independent.history["5m"]) == 4


@pytest.mark.parametrize("fault", ["gap", "invalid", "nan"])
def test_gap_and_invalid_bar_reset_warmup(fault: str) -> None:
    service = diagnostic.Diagnostic({"5m": model()}, "PINNED")
    for i in range(4):
        service.update(bar(i))
    row = bar(5 if fault == "gap" else 4, valid=fault != "invalid")
    if fault == "nan":
        row["values"]["close"] = float("nan")
    assert service.update(row)["value"] is None


@pytest.mark.parametrize("fault", ["duplicate", "unavailable", "future_fit"])
def test_causality_guards_reject(fault: str) -> None:
    service = diagnostic.Diagnostic({"5m": model()}, "PINNED")
    for i in range(0 if fault == "unavailable" else 3):
        service.update(bar(i))
    row = bar(0 if fault == "unavailable" else 3)
    if fault == "duplicate":
        row = bar(2)
    elif fault == "unavailable":
        row["available_utc"] = row["values"]["timestamp"]
    else:
        service.models["5m"] = replace(model(), cutoff_utc=T + timedelta(days=1))
    with pytest.raises(ValueError):
        service.update(row)


@pytest.mark.parametrize("fault", ["pin", "scale", "maturity", "feature"])
def test_fitted_artifact_requires_valid_frozen_inputs(tmp_path: Path, fault: str) -> None:
    path = tmp_path / "fit.json"
    body = model().resolved()
    if fault == "scale":
        body["scales"] = [0., 1., 1.]
    elif fault == "maturity":
        body["training_label_as_of_utc"] = body["cutoff_utc"]
    elif fault == "feature":
        body["features"] = ["target_return", "momentum_3", "volatility_3"]
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError):
        diagnostic.load_pipeline(path, "WRONG_HASH" if fault == "pin" else sha256(path))


def test_incompatible_or_nonconsecutive_checkpoint_rejected() -> None:
    service = diagnostic.Diagnostic({"5m": model()}, "PINNED")
    service.update(bar(0))
    service.update(bar(1))
    state = json.loads(json.dumps(service.state()))
    state["history"]["5m"][1][0] = bar(3)["values"]["timestamp"]
    with pytest.raises(ValueError):
        service.restore(state)
    state["identity"] = "OTHER"
    with pytest.raises(ValueError):
        service.restore(state)


def test_absent_timeframe_never_becomes_passing_coverage() -> None:
    report = diagnostic.forecast_coverage([{"timeframe": "5m", "value": .001},
                                          {"timeframe": "15m", "value": None}], ("5m", "15m"))
    assert report["status"] == "PARTIAL_FORECAST_DIAGNOSTICS"
    assert report["finite_forecasts_by_timeframe"] == {"5m": 1, "15m": 0}
    assert report["timeframe_status"]["15m"] == "UNTESTED_NO_FORECASTS"
    assert diagnostic.forecast_coverage([], ("5m", "15m"))["status"] == "INCONCLUSIVE_NO_FORECASTS"


def test_complete_recorded_workflow_preserves_partial_verdict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pins = {}
    for tf, run_id in diagnostic.MODEL_RUNS.items():
        fit = tmp_path / "results/strategy_validation/runs" / run_id / "frozen/F02/CAUSAL_RIDGE.json"
        fit.parent.mkdir(parents=True)
        fit.write_text(json.dumps(model().resolved()), encoding="utf-8")
        manifest = fit.parents[2] / "manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        pins[tf] = {"path": fit.relative_to(tmp_path).as_posix(), "sha256": sha256(fit),
                    "historical_manifest_sha256": sha256(manifest)}
    plan = tmp_path / "plan.json"
    freeze(plan, {"code_identity": code_identity(tmp_path), "script_sha256": sha256(Path(diagnostic.__file__)),
        "models": pins, "local_capture_config_identity": "SYNTHETIC", "evidence_before": evidence(tmp_path)})
    captured, replayed = tmp_path / "captured", tmp_path / "replayed"
    captured.mkdir()
    replayed.mkdir()
    rows = [{"table": "bars", "row": bar(i)} for i in range(7)]
    rows += [{"table": "bars", "row": bar(i, timeframe="15m")} for i in range(2)]
    data = "".join(json.dumps(row) + "\n" for row in rows)
    for directory in (captured, replayed):
        (directory / "records.jsonl").write_text(data, encoding="utf-8")
    freeze(captured / "initial_state.json", {"body": {"configuration_identity": "SYNTHETIC"}})
    freeze(captured / "summary.json", {"source": "SYNTHETIC_TEST_FIXTURE"})
    freeze(captured / "manifest.json", {"files": {"records.jsonl": sha256(captured / "records.jsonl")}})
    monkeypatch.setattr(diagnostic, "legacy_readiness", lambda root: {"source": "SYNTHETIC"})
    report = diagnostic.evaluate(tmp_path, plan, captured, replayed, "SYNTHETIC_RECORDED_V001")
    assert report["status"] == "PARTIAL_FORECAST_DIAGNOSTICS"
    assert report["finite_forecasts_by_timeframe"] == {"5m": 4, "15m": 0}
    assert report["timeframe_status"]["15m"] == "UNTESTED_NO_FORECASTS"
    assert not report["strategy_active"] and not report["broker_orders"]
    assert report["full_portfolio_risk_equality"] == "UNTESTED"
    with pytest.raises(FileExistsError):
        diagnostic.evaluate(tmp_path, plan, captured, replayed, "SYNTHETIC_RECORDED_V001")


def test_amendment_cli_accepts_relative_plan_path(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import sys

    monkeypatch.setattr(sys, "argv", ["diagnose_recorded_forecasts", "amend-reporting",
        "--original-plan", "relative.json", "--plan-id", "EXNESS_FROZEN_FORECAST_DIAGNOSTIC_V003"])
    monkeypatch.setattr(diagnostic, "amend_reporting", lambda *args: Path("README.md"))
    assert diagnostic.main() == 0
    assert capsys.readouterr().out.strip() == "README.md"


def test_disabled_availability_guard_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    original = inspect.getsource(diagnostic.Diagnostic.update)
    guard = "if available.timestamp() < opened.timestamp() + seconds:"
    assert guard in original
    import textwrap

    namespace = diagnostic.__dict__.copy()
    exec(compile(textwrap.dedent(original).replace(guard, "if False:"), "<broken-availability>", "exec"), namespace)
    monkeypatch.setattr(diagnostic.Diagnostic, "update", namespace["update"])
    with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
        test_causality_guards_reject("unavailable")
