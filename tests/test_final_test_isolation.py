"""The reserved test period stays untouched (Prompt #10, Step 2, Steps 75-79, Critical
Rule 2).

Development loading never opens a reserved year file (the synthetic store's
reserved target files are garbage bytes, so opening one fails). The only code
that reads reserved rows refuses to run without a frozen, hash-verified spec,
refuses a second look without a logged reason, and is imported by nothing that
selects models, features, hyperparameters or calibration.
"""

from __future__ import annotations

import ast
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl
import pytest

from ml_synth import small_config, synthetic_ml_data
from selection_synth import (
    RESERVED,
    TF,
    registry_rows,
    selection_config,
    stamps,
    write_stores,
)
from xauusd_quant.ml import final_test
from xauusd_quant.ml.datasets import read_years
from xauusd_quant.ml.final_test import (
    FinalTestRefusedError,
    load_reserved_rows,
    prior_evaluations,
    run_final_test,
)
from xauusd_quant.ml.registry import SpecError, freeze_spec
from xauusd_quant.research.ml_research import STAGES, MLContext, plan_stage

SRC = Path(__file__).resolve().parents[1] / "src" / "xauusd_quant"


def test_development_loading_never_opens_reserved_year_files(tmp_path: Path) -> None:
    _, tcfg = write_stores(tmp_path)                      # reserved target files are garbage
    base = tcfg.targets_path / f"timeframe={TF}" / "version=targets-synthetic"
    frame = read_years(base, ["target_return_5"], RESERVED)
    assert frame["timestamp"].max() < datetime(2022, 1, 1)
    assert frame["target_return_5"].abs().max() < 1.0      # no poisoned reserved value
    with pytest.raises((pl.exceptions.PolarsError, OSError)):
        read_years(base, ["target_return_5"], None)       # opening them does fail


def test_the_recent_base_rate_is_the_final_inner_slice_of_development() -> None:
    """The final test also scores each model against the recent base rate: the mean label
    of the final model's inner slice - development rows, known before the reserved period."""
    from xauusd_quant.ml.training import final_fold

    cfg, data = small_config(), synthetic_ml_data()
    tspec = cfg.targets["mean_reversion"]
    for scheme in ("expanding", "rolling"):
        spec = {"horizon": 5, "training_policy": {"scheme": scheme, "rolling_years": 1}}
        fold = final_fold(data, cfg, horizon=5, scheme=scheme, rolling_years=1)
        y = data.target(tspec, 5, cfg.log_floor).y
        inner = y[fold.inner[0]:fold.inner[1]]
        assert fold.inner[1] <= data.n - 5                  # labelled development rows only
        assert final_test.recent_base(data, spec, tspec, cfg) == pytest.approx(
            float(np.nanmean(inner)))
        assert final_test.recent_base(data, spec, tspec, cfg) != pytest.approx(
            float(np.nanmean(y[fold.fit[0]:fold.fit[1]])), abs=1e-12)


def test_reserved_rows_are_refused_without_a_frozen_spec() -> None:
    for spec in ({}, {"frozen": False}, {"frozen": "yes"}):
        with pytest.raises(FinalTestRefusedError, match="only for a frozen model spec"):
            load_reserved_rows(None, "5m", spec)           # refused before any file access


class _FakeRegressionStore:
    """The regression store's residual and trailing volatility on the synthetic bars."""

    def __init__(self, *_: object) -> None:
        pass

    def load(self, _timeframe: str, _window: int, _columns: list[str], *,
             before: Any = None) -> pl.DataFrame:
        ts = stamps()
        n = ts.len()
        frame = pl.DataFrame({"timestamp": ts, "residual": np.sin(np.arange(n) / 50.0),
                              "trailing_volatility": np.full(n, 0.004)})
        return (frame.filter(pl.col("timestamp") < pl.lit(before).cast(ts.dtype))
                if before is not None else frame)


def test_a_frozen_spec_reads_exactly_the_reserved_rows(tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    fcfg, tcfg = write_stores(tmp_path, corrupt_reserved_targets=False)
    monkeypatch.setattr(final_test, "RegressionFeatureStore", _FakeRegressionStore)
    cfgs = SimpleNamespace(selection=selection_config(), features=fcfg, targets=tcfg,
                           config=None, regression=None)
    spec = {"frozen": True, "features": ["vol_x", "sig_a"], "horizon": 5,
            "feature_set_id": "FS", "feature_set_hash": "h"}
    res = load_reserved_rows(cfgs, TF, spec)
    ts = stamps()
    reserved = ts.filter(ts >= datetime(2022, 1, 1))
    assert res.timestamps.equals(reserved)                 # every reserved bar, nothing earlier
    assert res.reserved_start == RESERVED
    assert set(res.features) == {"vol_x", "sig_a"}
    assert set(res.targets_raw) == {"target_return_5", "target_abs_return_5",
                                    "target_realized_vol_5", "target_residual_reduction_5"}
    assert res.epsilon is not None and res.epsilon.size == res.n == reserved.len()
    assert "year" in res.context and res.context["year"].min() == 2022


def _dev_data(fcfg: Any, tcfg: Any, features: list[str]) -> Any:
    """Development MLData read from the synthetic stores (rows before the reserved start)."""
    from xauusd_quant.ml.datasets import MLData, build_context

    fbase = fcfg.factory_path / f"timeframe={TF}" / "version=factory-synthetic"
    tbase = tcfg.targets_path / f"timeframe={TF}" / "version=targets-synthetic"
    frame = read_years(fbase, features, RESERVED)
    tcols = ["target_return_5", "target_abs_return_5", "target_realized_vol_5",
             "target_residual_reduction_5"]
    tframe = read_years(tbase, tcols, RESERVED)
    n = frame.height
    raw = {}
    for c in tcols:
        v = np.array(tframe[c].cast(pl.Float64).to_numpy(), dtype=np.float64)
        v[n - 5:] = np.nan                             # outcome windows crossing the start
        raw[c] = v
    reg = _FakeRegressionStore().load(TF, 128, []).filter(
        pl.col("timestamp") < datetime(2022, 1, 1))
    data = MLData(timeframe=TF, timestamps=frame["timestamp"], reserved_start=RESERVED,
                  features={f: frame[f].cast(pl.Float32).fill_null(np.nan).to_numpy()
                            for f in features},
                  manifests={"frozen": features}, manifest_ids={"frozen": "FS"},
                  manifest_hashes={"frozen": "h"},
                  registry={r["name"]: r for r in registry_rows(fcfg)}, targets_raw=raw,
                  epsilon=reg["residual"].to_numpy(), sigma=reg["trailing_volatility"].to_numpy(),
                  versions={"tick_dataset_version": "ticks-synthetic"})
    data.context = build_context(data)
    return data


def test_the_final_test_end_to_end_on_synthetic_stores(tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    """A frozen ridge and a frozen logistic model evaluated once on the reserved period:
    artifacts saved and reloaded, reserved rows only, labels aligned with their bars (the
    reserved return is built from the *same* row's ``sig_a``, whose lag-1 autocorrelation
    is 0.3, so a feature / label misalignment would collapse the rank IC), the recent
    base rate from development rows, the audit trail and the access log."""
    fcfg, tcfg = write_stores(tmp_path, corrupt_reserved_targets=False)
    tbase = tcfg.targets_path / f"timeframe={TF}" / "version=targets-synthetic"
    fbase = fcfg.factory_path / f"timeframe={TF}" / "version=factory-synthetic"
    rng = np.random.default_rng(3)
    for year in (2022, 2023):                          # replace the poison with real labels
        feats = pl.read_parquet(fbase / f"year={year}" / "part-0.parquet")
        part = pl.read_parquet(tbase / f"year={year}" / "part-0.parquet")
        sig = feats["sig_a"].cast(pl.Float64).fill_null(0.0).to_numpy()
        ret = 0.004 * np.sqrt(5) * (0.5 * sig + rng.normal(size=sig.size))
        part = part.with_columns(pl.Series("target_return_5", ret.astype(np.float32)))
        part.write_parquet(tbase / f"year={year}" / "part-0.parquet")
    monkeypatch.setattr(final_test, "RegressionFeatureStore", _FakeRegressionStore)
    feats = ["sig_a", "vol_x"]
    dev = _dev_data(fcfg, tcfg, feats)
    cfg = small_config(tmp_path)
    cfgs = SimpleNamespace(selection=selection_config(), features=fcfg, targets=tcfg,
                           config=None, regression=None)
    common = {"timeframe": TF, "features": feats, "feature_set": "frozen",
              "feature_set_id": "FS", "feature_set_hash": "h", "seed": 7,
              "training_policy": {"scheme": "expanding", "rolling_years": 5},
              "preprocessing": {"kind": "linear", "clip": 8.0},
              "dataset_versions": {"tick_dataset_version": "ticks-synthetic"}}
    specs = [
        freeze_spec({**common, "spec_id": "RIDGE_RETURN_1H_H5_V001", "family": "ridge",
                     "target": "future_return", "horizon": 5, "params": {"alpha": 1.0},
                     "calibration": "none",
                     "target_definition": {"task": "regression"}}, tmp_path / "frozen"),
        freeze_spec({**common, "spec_id": "LOGL2_REVHALF_1H_H5_V001", "family": "logistic_l2",
                     "target": "mean_reversion_half", "horizon": 5,
                     "params": {"C": 1.0, "max_iter": 200}, "calibration": "platt",
                     "target_definition": {"task": "classification"}}, tmp_path / "frozen")]
    out = tmp_path / "final_test"
    results = run_final_test(cfgs, cfg, dev, specs, out_dir=out)
    by = {r["model_id"]: r for r in results}
    ret = by["RIDGE_RETURN_1H_H5_V001"]
    assert ret["reload_equal"] is True
    assert ret["first"].startswith("2022-01-03") and ret["last"].startswith("2023-12-29")
    assert ret["overall"]["rank_ic"] > 0.3                # aligned labels: the signal survives
    assert set(ret["by_year"]) == {2022, 2023}
    assert ret["recent_base"] == pytest.approx(final_test.recent_base(
        dev, json.loads(specs[0].read_text(encoding="utf-8")), cfg.targets["future_return"],
        cfg))
    assert "mse_skill" in ret["overall_vs_recent_base"]
    rev = by["LOGL2_REVHALF_1H_H5_V001"]
    assert 0.0 < rev["recent_base"] < 1.0
    assert rev["overall"]["n"] == rev["rows"] > 0 and "log_loss_skill" in rev["overall"]
    assert rev["summary"]["calibrated"] is True
    assert rev["summary"]["final_test_skill_vs_recent_base"] is not None
    audit = pl.read_parquet(out / "RIDGE_RETURN_1H_H5_V001_predictions.parquet")
    assert audit["timestamp"].min() >= datetime(2022, 1, 1)
    assert {"timestamp", "model_id", "prediction", "calibrated", "label", "fold",
            "feature_set_id", "dataset_version"} <= set(audit.columns)
    events = [json.loads(line)["event"] for line in
              (out / "access_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events == ["final_test_started", "evaluated", "evaluated"]
    with pytest.raises(FinalTestRefusedError, match="one-time"):
        run_final_test(cfgs, cfg, dev, specs, out_dir=out)


def _frozen(directory: Path) -> Path:
    return freeze_spec({"spec_id": "LGBM_REVERSION_1H_H5_V001", "family": "lightgbm",
                        "target": "mean_reversion", "horizon": 5, "timeframe": "1h",
                        "features": ["sig"], "params": {}, "feature_set_id": "FS",
                        "training_policy": {"scheme": "expanding"}, "seed": 1}, directory)


def test_the_final_test_refuses_unfrozen_or_edited_specs(tmp_path: Path) -> None:
    cfg = small_config(tmp_path)
    path = _frozen(tmp_path / "frozen")
    body = json.loads(path.read_text(encoding="utf-8"))
    body["params"] = {"num_leaves": 255}                   # edited after freezing
    path.write_text(json.dumps(body), encoding="utf-8")
    out = tmp_path / "final_test"
    with pytest.raises(SpecError, match="content hash"):
        run_final_test(None, cfg, None, [path], out_dir=out)   # type: ignore[arg-type]
    unfrozen = tmp_path / "MODEL_SPEC_X.json"
    unfrozen.write_text(json.dumps({"spec_id": "X"}), encoding="utf-8")
    with pytest.raises(SpecError, match="not frozen"):
        run_final_test(None, cfg, None, [unfrozen], out_dir=out)   # type: ignore[arg-type]
    assert not out.exists()                                # nothing was logged or read


def test_a_second_look_needs_a_logged_reason(tmp_path: Path) -> None:
    cfg = small_config(tmp_path)
    path = _frozen(tmp_path / "frozen")
    spec_hash = json.loads(path.read_text(encoding="utf-8"))["content_hash"]
    out = tmp_path / "final_test"
    out.mkdir()
    log = out / "access_log.jsonl"
    log.write_text(json.dumps({"utc": "2026-01-01T00:00:00Z", "event": "evaluated",
                               "spec_id": "LGBM_REVERSION_1H_H5_V001",
                               "spec_hash": spec_hash}) + "\n", encoding="utf-8")
    assert len(prior_evaluations(log, {spec_hash})) == 1
    with pytest.raises(FinalTestRefusedError, match="one-time"):
        run_final_test(None, cfg, None, [path], out_dir=out)       # type: ignore[arg-type]
    assert len(log.read_text(encoding="utf-8").splitlines()) == 1  # the refusal reads nothing
    with pytest.raises(AttributeError):                    # no development data given here
        run_final_test(None, cfg, None, [path], out_dir=out,       # type: ignore[arg-type]
                       repeat_reason="audit of a corrected loader")
    events = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert events[1]["event"] == "repeat_authorised"
    assert events[1]["reason"] == "audit of a corrected loader"


_SELECTION_MODULES = ("research/ml_research.py", "research/ml_reports.py",
                      "research/ml_freeze.py", "research/ml_plots.py", "ml/training.py",
                      "ml/datasets.py", "ml/models.py", "ml/calibration.py",
                      "ml/preprocessing.py", "ml/splits.py", "ml/explainability.py",
                      "ml/diagnostics.py", "ml/evaluation.py", "ml/streaming.py")


def test_no_selection_code_can_reach_the_reserved_period() -> None:
    for rel in _SELECTION_MODULES:
        tree = ast.parse((SRC / rel).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
        assert not any("final_test" in m or m == "load_reserved_rows" for m in imported), rel


def test_every_planned_unit_scores_development_rows_only(tmp_path: Path) -> None:
    data = synthetic_ml_data()
    cfg = small_config(tmp_path)
    cfgs = SimpleNamespace(ml=cfg)
    ctx = MLContext(timeframe="1h", cfgs=cfgs, data=data,  # type: ignore[arg-type]
                    out_dir=tmp_path / "1h")
    assert data.timestamps.max() < datetime(2022, 1, 1)
    for stage in STAGES:
        units = plan_stage(ctx, stage)
        assert units, stage
        for unit in units:
            lo, hi = unit.fold.validate
            assert 0 <= unit.fold.fit[0] < unit.fold.fit[1] <= unit.fold.inner[0] < lo
            assert hi <= data.n, (stage, unit.key())
