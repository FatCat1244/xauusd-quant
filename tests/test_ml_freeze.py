"""The pre-registered freeze rules (Prompt #10, Step 75; Critical Rules 2-4, 7).

Development results only: a model must beat the constant baseline in at least
four of five blocks with a positive mean; among the eligible, the simplest
within one standard error of the best is frozen; a window, weighting or
calibration change needs a block-by-block advantage beyond one SE - and a
perfectly consistent difference (SE exactly zero) counts as one.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from ml_synth import small_config, synthetic_ml_data
from xauusd_quant.ml.registry import SpecError, load_frozen_spec
from xauusd_quant.research.ml_freeze import freeze_candidates
from xauusd_quant.research.ml_reports import summarise

FOLDS = [f"wf{i + 1}_x" for i in range(5)]
# dyadic values: block differences are exact, so a constant difference has an SE of exactly 0
RAW_LOSS = [88 / 128, 87 / 128, 86 / 128, 85 / 128, 84 / 128]
LOGISTIC = [8 / 1024, 12 / 1024, 10 / 1024, 9 / 1024, 11 / 1024]       # mean 0.009766
STEP = 1 / 1024


def _cls(family: str, skill: list[float], *, variant: str = "base",
         raw_loss: list[float] | None = None, platt_shift: float = STEP,
         iso_shift: float = STEP / 2, feature_set: str = "standard", path: str = "none",
         folds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> list[dict[str, Any]]:
    rows = []
    raw_loss = raw_loss or RAW_LOSS
    for i, fold in enumerate(FOLDS):
        if i not in folds:
            continue
        common = {"target": "mean_reversion", "horizon": 5, "family": family,
                  "feature_set": feature_set, "variant": variant, "fold": fold,
                  "fold_index": i, "fit_seconds": 1.0, "path": path, "mse_skill": None,
                  "rank_ic": 0.05}
        for calib, loss in (("raw", raw_loss[i]), ("platt", raw_loss[i] - platt_shift),
                            ("isotonic", raw_loss[i] - iso_shift)):
            rows.append({**common, "calibration": calib, "log_loss": loss, "auc": 0.55,
                         "log_loss_skill": skill[i], "brier_skill": skill[i] / 2})
    return rows


def _reg(target: str, h: int, family: str, rank_ic: list[float],
         mse_skill: list[float]) -> list[dict[str, Any]]:
    return [{"target": target, "horizon": h, "family": family, "feature_set": "standard",
             "variant": "base", "fold": fold, "fold_index": i, "fit_seconds": 1.0,
             "path": "none", "calibration": "raw", "log_loss": None, "auc": None,
             "log_loss_skill": None, "brier_skill": None, "rank_ic": rank_ic[i],
             "mse_skill": mse_skill[i]} for i, fold in enumerate(FOLDS)]


def _folds(**overrides: Any) -> pl.DataFrame:
    rows = [*_cls("constant", [0.0] * 5),
            *_cls("logistic_l2", LOGISTIC, **overrides),
            *_cls("lightgbm", [0.020, 0.005, 0.015, 0.008, 0.012]),         # best, SE 0.0026
            *_cls("catboost", [0.030, -0.010, 0.030, -0.005, 0.020]),       # 3 of 5 blocks
            *_cls("logistic_l2", [v + STEP for v in LOGISTIC],              # + STEP each block
                  variant="window-rolling5"),
            *_reg("future_volatility", 5, "constant", [0.0] * 5, [0.0] * 5),
            *_reg("future_volatility", 5, "ridge", [0.5] * 5, [0.1, -0.1, 0.1, -0.1, 0.1]),
            *_reg("future_volatility", 5, "lightgbm", [0.55] * 5, [0.1, 0.1, -0.05, -0.05, 0.1]),
            *_reg("future_return", 1, "constant", [0.0] * 5, [0.0] * 5),
            *_reg("future_return", 1, "ridge", [0.012, 0.018, 0.016, 0.011, 0.020], [0.001] * 5),
            *_reg("future_return", 1, "lightgbm", [0.010, 0.020, 0.015, 0.012, 0.018],
                  [0.001] * 5)]
    return pl.DataFrame(rows, infer_schema_length=None)


def _ctx(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(cfg=small_config(tmp_path), data=synthetic_ml_data(),
                           out_dir=tmp_path / "1h", timeframe="1h")


def test_summaries_count_blocks_beating_the_baseline() -> None:
    summ = summarise(_folds(), small_config()).filter(pl.col("variant") == "base")
    by = {(r["target"], r["family"]): r for r in summ.iter_rows(named=True)}
    assert by[("mean_reversion", "logistic_l2")]["blocks_beating_baseline"] == 5
    assert by[("mean_reversion", "catboost")]["blocks_beating_baseline"] == 3
    assert by[("future_volatility", "ridge")]["blocks_beating_baseline"] == 3
    assert by[("mean_reversion", "lightgbm")]["mean"] == pytest.approx(0.012)
    assert by[("mean_reversion", "lightgbm")]["metric"] == "log_loss_skill"
    assert by[("future_return", "ridge")]["metric"] == "rank_ic"


def test_the_simplest_model_within_one_se_is_frozen(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    report = freeze_candidates(ctx, _folds())
    assert report["MODEL_SPEC_FROZEN"] is True
    dec = {(d["target"], d["horizon"]): d for d in report["decisions"]}
    rev = dec[("mean_reversion", 5)]
    assert rev["eligible"] == 2                                 # catboost: 3 of 5 blocks
    assert rev["frozen"] == "LOGL2_REVERSION_1H_H5_V001"        # lightgbm is best, not simplest
    assert rev["companions"] == ["CONST_REVERSION_1H_H5_V001"]
    frozen = ctx.out_dir / "frozen"
    spec = load_frozen_spec(frozen / "MODEL_SPEC_LOGL2_REVERSION_1H_H5_V001.json")
    assert spec["development"]["best_family"] == "lightgbm"
    assert ["lightgbm", "standard"] in [list(p) for p in spec["development"]["within_one_se"]]
    assert spec["features"] == ctx.data.feature_set("standard")
    assert spec["no_trading_output"] is True and spec["outputs"] == "probability"
    # platt beats raw by exactly STEP in every block: SE 0, a consistent difference
    assert spec["calibration"] == "platt"
    # the rolling window is better by exactly STEP in every block: adopted
    assert spec["training_policy"]["scheme"] == "rolling"
    const = load_frozen_spec(frozen / "MODEL_SPEC_CONST_REVERSION_1H_H5_V001.json")
    assert const["role"] == "baseline" and const["calibration"] == "none"
    assert dec[("future_volatility", 5)]["frozen"] is None
    assert "4 of 5 blocks" in dec[("future_volatility", 5)]["reason"]
    ret = dec[("future_return", 1)]
    assert ret["frozen"] == "RIDGE_RETURN_1H_H1_V001"           # the reference: no companion
    assert ret["companions"] == ["CONST_RETURN_1H_H1_V001"]
    assert sorted(report["specs"]) == sorted(p.name for p in frozen.glob("MODEL_SPEC_*.json"))


def test_refreezing_is_idempotent_but_a_changed_decision_is_refused(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    first = freeze_candidates(ctx, _folds())
    again = freeze_candidates(ctx, _folds())
    assert sorted(first["specs"]) == sorted(again["specs"])
    # raw within one SE of platt now -> calibration "none": another spec under the same id
    with pytest.raises(SpecError, match="already frozen with other content"):
        freeze_candidates(ctx, _folds(platt_shift=0.0, iso_shift=0.0))


@pytest.mark.parametrize("extended_wins", [False, True])
def test_search_trials_are_read_against_defaults_on_the_searched_set_only(
        tmp_path: Path, extended_wins: bool) -> None:
    """LightGBM has default-parameter units on Standard *and* Extended on the search blocks;
    the trial (run on Standard) must be compared with Standard's, never with whichever
    set's row comes last - and a model frozen on Extended keeps the defaults."""
    ctx = _ctx(tmp_path)
    trial_json = tmp_path / "search_t00.json"
    trial_json.write_text('{"params": {"num_leaves": 7, "learning_rate": 0.05}}',
                          encoding="utf-8")
    standard = [20 / 1024, 22 / 1024, 18 / 1024, 21 / 1024, 19 / 1024]
    # Extended beats the trial on the search blocks (wf4, wf5) in both cases; it is the
    # best configuration overall only when extended_wins
    extended = ([100 / 1024] * 5 if extended_wins
                else [1 / 1024, 1 / 1024, 1 / 1024, 30 / 1024, 30 / 1024])
    rows = [*_cls("constant", [0.0] * 5),
            *_cls("lightgbm", standard),
            *_cls("lightgbm", extended, feature_set="extended"),        # after Standard's rows
            # the trial beats Standard's defaults by 4/1024 in both search blocks (SE 0)
            *_cls("lightgbm", [0.0, 0.0, 0.0, standard[3] + 4 / 1024, standard[4] + 4 / 1024],
                  variant="search-t00", path=str(trial_json.with_suffix(".parquet")),
                  folds=(3, 4))]
    report = freeze_candidates(ctx, pl.DataFrame(rows, infer_schema_length=None))
    rev = {(d["target"], d["horizon"]): d for d in report["decisions"]}[("mean_reversion", 5)]
    spec = load_frozen_spec(ctx.out_dir / "frozen" / "MODEL_SPEC_LGBM_REVERSION_1H_H5_V001.json")
    if extended_wins:
        assert rev["feature_set"] == "extended"
        assert spec["params"]["num_leaves"] == ctx.cfg.model_params("lightgbm")["num_leaves"]
        assert "the search ran on standard" in rev["hyperparameter_rule"]
    else:
        assert rev["feature_set"] == "standard"
        assert spec["params"] == {"num_leaves": 7, "learning_rate": 0.05}
        assert rev["hyperparameter_rule"].startswith("search trial search-t00: +0.0039")


def test_no_eligible_model_freezes_nothing_for_that_target(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    weak = _folds().with_columns(
        pl.when(pl.col("target") == "mean_reversion").then(pl.lit(-0.001))
        .otherwise(pl.col("log_loss_skill")).alias("log_loss_skill"))
    report = freeze_candidates(ctx, weak)
    dec = {(d["target"], d["horizon"]): d for d in report["decisions"]}
    assert dec[("mean_reversion", 5)]["frozen"] is None
    assert not (ctx.out_dir / "frozen" / "MODEL_SPEC_LOGL2_REVERSION_1H_H5_V001.json").exists()
