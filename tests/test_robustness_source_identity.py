"""Frozen saved evidence binds values before materialization, without new market reads."""

from pathlib import Path
from typing import Any

import pytest

from test_robustness_framework import make_saved_source
from xauusd_quant.robustness import reports
from xauusd_quant.robustness.plan import RobustnessPlan, freeze_plan


def test_changed_saved_source_refused_before_outcome_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, path = make_saved_source(tmp_path)
    (path / "readiness.json").write_text(
        '{"gates": {}, "inventory": {"dataset_version": "synthetic"}}', encoding="utf-8"
    )
    source = path / "forecast_scores.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    plan = RobustnessPlan("ROBUST_IDENTITY_V001", (path.name,))
    freeze_plan(tmp_path, plan)
    original_digest = reports.sha256(source)
    source.write_text('{"actual": 999}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="source artifacts"):
        freeze_plan(tmp_path, plan)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("changed outcome artifact opened")

    monkeypatch.setattr(reports, "read_records", forbidden)
    with pytest.raises(ValueError, match="artifact changed"):
        reports.source_records(config, path, 20, {"forecast_scores.jsonl": original_digest})
