r"""Family ablation of the supervised models (Prompt #10, Steps 32-33).

LightGBM is refitted on the Extended feature set with one feature family left
out (``minus_<family>``) and with families added one at a time in the configured
order (``forward_statistics``, ``forward_plus_regression``, ...), on the
ablation blocks of ``config/ml.yaml`` (2015-17 and 2018-19 by default). Each
variant is read as a block-by-block difference from the full Extended model on
the same blocks - never as a level of its own - so a family that "matters" is
one whose removal costs more than its SE. Where Extended equals the default
set (15m) it is never refitted under its own name, and the default set's units
are the reference (``reference_set``).

Families are the factory's (``returns``, ``volatility`` and ``autocorrelation``
form the statistics block). The ablation pool has no pipeline veto, so a
residual-target row can include mechanical effects (invariant 9).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from ..ml.config import MLConfig
from ..ml.evaluation import paired_blocks

__all__ = ["ABLATION_FAMILIES", "FORWARD_CHAIN", "STATISTICS", "ablation_sets", "ablation_table"]

STATISTICS = ("returns", "volatility", "autocorrelation")
ABLATION_FAMILIES = ("regression", "ou", "fft", "wavelet", "regime", "microstructure", "time")
FORWARD_CHAIN = ("statistics", "regression", "ou", "fft", "wavelet", "regime",
                 "microstructure", "time")


def ablation_sets(extended: list[str], registry: dict[str, dict[str, Any]]
                  ) -> dict[str, list[str]]:
    """Leave-one-family-out and forward-addition feature lists (Extended order kept)."""
    fam = {f: str(registry[f].get("family")) for f in extended}
    out: dict[str, list[str]] = {}
    for drop in ABLATION_FAMILIES:
        if any(v == drop for v in fam.values()):
            out[f"minus_{drop}"] = [f for f in extended if fam[f] != drop]
    included: set[str] = set(STATISTICS)
    for step in FORWARD_CHAIN:
        if step != "statistics":
            included.add(step)
        feats = [f for f in extended if fam[f] in included]
        label = "forward_statistics" if step == "statistics" else f"forward_plus_{step}"
        if feats and feats != extended:
            out[label] = feats
    return out


def ablation_table(folds: pl.DataFrame, cfg: MLConfig) -> pl.DataFrame:
    """Each ablation set against LightGBM on the full Extended set, same blocks."""
    from .ml_reports import _metric_rows, primary_metric

    if folds.is_empty():
        return pl.DataFrame()
    view = _metric_rows(folds, cfg).filter((pl.col("family") == "lightgbm")
                                           & (pl.col("variant") == "base"))
    blocks = [int(i) for i in cfg.ablation.get("folds", [3, 4])]
    rows = []
    for target, h in cfg.focus:
        metric = primary_metric(cfg.targets[target].task)
        part = view.filter((pl.col("target") == target) & (pl.col("horizon") == h)
                           & pl.col("fold_index").is_in(blocks))
        # Extended is not refitted where it equals the default set (15m): read that one
        ref_set = "extended" if part.filter(pl.col("feature_set") == "extended").height \
            else cfg.default_feature_set
        ref = {r["fold"]: r.get(metric) for r in part.filter(
            pl.col("feature_set") == ref_set).iter_rows(named=True)}
        if not ref:
            continue
        for (set_name,), p in part.filter(pl.col("feature_set").str.starts_with(
                "ablation_")).group_by(["feature_set"], maintain_order=True):
            mine = {r["fold"]: r.get(metric) for r in p.iter_rows(named=True)}
            common = sorted(set(mine) & set(ref))
            cmp = paired_blocks([mine[f] for f in common], [ref[f] for f in common])
            label = str(set_name).removeprefix("ablation_")
            kind = "leave_one_out" if label.startswith("minus_") else "forward"
            vals = [v for v in mine.values() if v is not None]
            rows.append({"target": target, "horizon": h, "metric": metric, "set": label,
                         "kind": kind, "family": label.removeprefix("minus_")
                         .removeprefix("forward_plus_").removeprefix("forward_"),
                         "mean": float(np.mean(vals)) if vals else None,
                         "extended_mean": float(np.mean([v for v in ref.values()
                                                         if v is not None])),
                         "reference_set": ref_set,
                         "delta_vs_extended": cmp.get("mean_diff"), "se": cmp.get("se"),
                         "blocks": cmp.get("blocks", 0), "wins": cmp.get("wins")})
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()
