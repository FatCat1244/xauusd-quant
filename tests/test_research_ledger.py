"""The shared research ledger's upsert keys (Prompts #5-#10)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from xauusd_quant.research.research_ledger import ResearchLedger


def _row(hid: str, window: int, value: float, feature: str = "lightgbm/standard/base"
         ) -> dict[str, Any]:
    return {"hypothesis_id": hid, "timeframe": "15m", "input_series": "ml_research",
            "window": window, "feature": feature, "target": "mean_reversion", "horizon": 5,
            "metric": "log_loss_skill", "value": value, "verdict": "x"}


def test_the_full_key_keeps_one_row_per_window_and_feature(tmp_path: Path) -> None:
    ledger = ResearchLedger(tmp_path / "ledger.parquet")
    ledger.upsert([_row("SPEC-H-006", 64, 0.1), _row("SPEC-H-006", 128, 0.2)])
    assert ledger.upsert([_row("SPEC-H-006", 64, 0.3)]) == 2      # same key: replaced
    rows = ledger.load().sort("window")
    assert rows["value"].to_list() == [0.3, 0.2]


def test_replacing_by_id_keeps_one_row_when_the_window_changes(tmp_path: Path) -> None:
    """An ML-H configuration reported first with 1 block and later with 8 is one row."""
    ledger = ResearchLedger(tmp_path / "ledger.parquet")
    ledger.upsert([_row("ML-H-000246", 1, 0.1375), _row("ML-H-000247", 5, 0.02)],
                  replace_by=("hypothesis_id",))
    after = ledger.upsert([_row("ML-H-000246", 8, 0.1311)], replace_by=("hypothesis_id",))
    assert after == 2
    row = ledger.load().filter(ledger.load()["hypothesis_id"] == "ML-H-000246")
    assert row.height == 1 and row["window"][0] == 8 and row["value"][0] == 0.1311
    # the default (full key) would have kept the stale one-block row beside it
    ledger.upsert([_row("ML-H-000246", 1, 0.1375)])
    assert ledger.load().height == 3
