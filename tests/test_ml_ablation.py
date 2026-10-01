"""Family ablation deltas (Prompt #10, Steps 32-33).

Each ablation subset is read against LightGBM on the full Extended set, block by
block. Where Extended equals the default set (15m) it is never refitted under its
own name, so the default set's units are the reference.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from ml_synth import small_config
from xauusd_quant.research.model_ablation import ablation_table

BLOCKS = ("wf4_x", "wf5_x")


def _rows(feature_set: str, skill: tuple[float, float]) -> list[dict[str, Any]]:
    return [{"target": "mean_reversion", "horizon": 5, "family": "lightgbm",
             "feature_set": feature_set, "variant": "base", "fold": fold, "fold_index": 3 + i,
             "calibration": "platt", "log_loss_skill": skill[i], "path": "none"}
            for i, fold in enumerate(BLOCKS)]


@pytest.mark.parametrize("reference", ["extended", "standard"])
def test_ablation_reads_extended_or_the_identical_default_set(reference: str) -> None:
    cfg = small_config()
    rows = [*_rows(reference, (0.10, 0.12)),
            *_rows("ablation_minus_regression", (0.04, 0.05)),
            *_rows("ablation_forward_statistics", (0.03, 0.03))]
    if reference == "extended":                 # a different default set is never the reference
        rows += _rows("standard", (0.50, 0.50))
    out = ablation_table(pl.DataFrame(rows, infer_schema_length=None), cfg)
    by = {r["set"]: r for r in out.filter(pl.col("target") == "mean_reversion")
          .iter_rows(named=True)}
    assert by["minus_regression"]["delta_vs_extended"] == pytest.approx(-0.065)
    assert by["minus_regression"]["kind"] == "leave_one_out"
    assert by["forward_statistics"]["delta_vs_extended"] == pytest.approx(-0.08)
    assert {r["reference_set"] for r in by.values()} == {reference}
