"""The supervised research end to end on synthetic data (Prompt #10, Steps 23-66, 71-75,
95-97).

A reduced plan (two targets, four families) runs through the cached units, the
report (every table, the ML-H ledger rows, the figures), the freeze rules and
the final development models (artifacts, reload equality, latency). Nothing
here reads the real dataset or the reserved period.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
import yaml

from ml_synth import small_config, synthetic_ml_data
from xauusd_quant.ml.config import MLConfig
from xauusd_quant.ml.registry import load_frozen_spec
from xauusd_quant.research.ml_freeze import finalize, freeze_candidates
from xauusd_quant.research.ml_reports import build_ml_report, collect_units
from xauusd_quant.research.ml_research import MLContext, plan_stage, run_units


def _cfg(root: Path) -> MLConfig:
    cfg = small_config(root)
    keep = ("constant", "logistic_l2", "ridge", "lightgbm")
    models = {k: {**v, "enabled": k in keep} for k, v in cfg.models.items()}
    targets = {k: replace(cfg.targets[k], horizons=(5,))
               for k in ("mean_reversion", "future_volatility", "direction_up_cost")}
    return replace(cfg, models=models, targets=targets,
                   primary={"mean_reversion": (5,), "future_volatility": (5,)},
                   tree_comparison={"mean_reversion": 5, "future_volatility": 5},
                   focus=(("mean_reversion", 5), ("future_volatility", 5)),
                   search={**cfg.search, "targets": []},
                   nulls={**cfg.nulls, "shuffled_target_repeats": 1, "shuffled_folds": [4]})


def test_units_report_freeze_and_final_models(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    ctx = MLContext(timeframe="1h", cfgs=SimpleNamespace(ml=cfg),  # type: ignore[arg-type]
                    data=synthetic_ml_data(), out_dir=tmp_path / "ml_research" / "1h")
    for stage in ("grid", "trees", "feature_sets", "windows", "learning_curve", "decay",
                  "ablation", "nulls"):
        res = run_units(ctx, plan_stage(ctx, stage), stage=stage)
        assert res["failed"] == 0 and res["ran"] == res["units"] > 0, stage
    again = run_units(ctx, plan_stage(ctx, "grid"), stage="grid")
    assert again["ran"] == 0                                # every unit current: nothing refitted

    summary = build_ml_report(ctx, ledger=True, plots=True)
    tables = ctx.out_dir / "tables"
    for name in ("fold_metrics", "configuration_summary", "paired_vs_linear",
                 "paired_vs_constant", "pooled", "yearly", "groups", "deciles", "reliability",
                 "alpha_decay", "roc_pr", "probability_hist", "shap", "permutation",
                 "shap_stability", "prediction_correlation", "disagreement",
                 "distribution_shift", "ood_performance", "windows", "learning_curve",
                 "decay_by_year", "nulls", "failure_analysis", "ablation_deltas",
                 "null_comparison", "skill_beyond_base_rate"):
        assert (tables / f"{name}.parquet").exists(), name
    beyond = pl.read_parquet(tables / "skill_beyond_base_rate.parquet")
    rev = beyond.filter((pl.col("target") == "mean_reversion") & (pl.col("family") == "lightgbm")
                        & (pl.col("feature_set") == "standard"))
    # the synthetic reversion label has a stable base rate: the features carry the skill
    assert rev["skill_beyond_base_rate"][0] == pytest.approx(
        rev["model_skill"][0] - rev["recalibrated_constant_skill"][0])
    assert rev["skill_beyond_base_rate"][0] > 0.05
    nulls = pl.read_parquet(tables / "null_comparison.parquet")
    shifted = nulls.filter((pl.col("target") == "mean_reversion")
                           & (pl.col("family") == "lightgbm") & (pl.col("control") == "null"))
    assert shifted.height and (shifted["real_minus_control"] > 0.05).all()
    deltas = pl.read_parquet(tables / "ablation_deltas.parquet")
    minus_reg = deltas.filter((pl.col("target") == "mean_reversion")
                              & (pl.col("set") == "minus_regression"))
    assert minus_reg["delta_vs_extended"][0] < -0.05        # the residual feature carries it
    view = ctx.out_dir / "mean_reversion" / "h5"
    for part in ("comparisons/configuration_summary.csv", "baselines/fold_metrics.csv",
                 "lightgbm/fold_metrics.csv", "calibration/reliability.csv"):
        assert (view / part).exists(), part
    assert summary["ledger_rows_upserted"] == summary["configurations"] > 0
    ledger = pl.read_parquet(cfg.ledger_path)
    assert ledger["hypothesis_id"].str.starts_with("ML-H").all()
    for fig in ("roc_pr.png", "reliability.png", "probability_distribution.png",
                "prediction_deciles.png", "model_ic_decay.png", "yearly_performance.png",
                "fold_performance.png", "feature_importance.png", "shap_summary.png",
                "training_window.png", "prediction_correlation.png", "model_disagreement.png",
                "null_controls.png"):
        assert fig in summary["figures"], fig
        assert (ctx.out_dir / "plots" / fig).stat().st_size > 10_000
    # Steps 61 / 74: the out-of-sample predictions, one table per comparison pair
    assert sorted(summary["predictions"]) == ["future_volatility_h5.parquet",
                                              "mean_reversion_h5.parquet"]
    exported = pl.read_parquet(ctx.out_dir / "predictions" / "mean_reversion_h5.parquet")
    assert exported.columns[:8] == ["timestamp", "row", "fold", "target", "horizon",
                                    "dataset_version", "label", "label_raw"]
    assert {"lightgbm|standard", "lightgbm|standard|platt", "constant|standard"} <= \
        set(exported.columns)
    assert exported["timestamp"].dt.year().max() < ctx.data.reserved_start.year   # 2022
    assert exported["timestamp"].is_sorted() and exported["row"].n_unique() == exported.height
    units, _ = collect_units(ctx.out_dir)
    unit = units.filter((pl.col("target") == "mean_reversion") & (pl.col("family") == "lightgbm")
                        & (pl.col("feature_set") == "standard") & (pl.col("variant") == "base")
                        & (pl.col("calibration") == "raw")).sort("fold_index").row(-1, named=True)
    raw = pl.read_parquet(unit["path"]).select(pl.col("row").cast(pl.Int64), "prediction",
                                                "cal_platt")
    joined = raw.join(exported.select("row", "fold", "lightgbm|standard",
                                      "lightgbm|standard|platt"), on="row")
    assert joined.height == raw.height and (joined["fold"] == unit["fold"]).all()
    assert (joined["prediction"] == joined["lightgbm|standard"]).all()
    assert (joined["cal_platt"] == joined["lightgbm|standard|platt"]).all()
    vol = pl.read_parquet(ctx.out_dir / "predictions" / "future_volatility_h5.parquet")
    assert not [c for c in vol.columns if c.endswith("|platt")]       # expected values only
    pooled = pl.read_parquet(tables / "pooled.parquet")
    lgbm = pooled.filter((pl.col("target") == "mean_reversion") & (pl.col("family") == "lightgbm")
                         & (pl.col("calibration") == "platt"))
    assert lgbm["auc"][0] > 0.6                             # the synthetic reversion structure

    folds, _ = collect_units(ctx.out_dir)
    report = freeze_candidates(ctx, folds)
    decisions = {d["target"]: d for d in report["decisions"]}
    assert decisions["mean_reversion"]["frozen"] == "LGBM_REVERSION_1H_H5_V001"
    frozen = sorted((ctx.out_dir / "frozen").glob("MODEL_SPEC_*.json"))
    assert len(frozen) == len(report["specs"]) >= 3
    out = finalize(ctx, streaming=False)
    assert {m["model_id"] for m in out["models"]} == {load_frozen_spec(p)["spec_id"]
                                                      for p in frozen}
    for m in out["models"]:
        assert m["reload_equal"] is True and m["latency_ms_per_row"] > 0
        manifest = json.loads((cfg.models_path / m["model_id"] / "manifest.json").read_text(
            encoding="utf-8"))
        assert manifest["spec_hash"] == load_frozen_spec(
            ctx.out_dir / "frozen" / f"MODEL_SPEC_{m['model_id']}.json")["content_hash"]
    again_out = finalize(ctx, streaming=False)            # artifacts reused, not refitted
    assert {m["artifact"] for m in again_out["models"]} == {"existing"}
    registry = yaml.safe_load(cfg.registry_path.read_text(encoding="utf-8"))
    assert set(registry["models"]) == {m["model_id"] for m in out["models"]}
    entry = registry["models"]["LGBM_REVERSION_1H_H5_V001"]
    assert entry["reload_equal"] is True and entry["no_trading_output"] is True
    assert entry["outputs"] == "probability"
