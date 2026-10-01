"""The ensemble final test: frozen specs only, logged, one-time - and nothing that
selects an ensemble can reach the reserved period (Prompt #11, Steps 59, 62-63;
Rule 4).

* no research, selection, weighting, stacking or calibration module imports the
  final-test code or the reserved loader;
* an unfrozen or edited spec is refused before anything is read; a second look
  needs a reason, which is logged;
* end to end on synthetic stores: two frozen constituents (ridge, elastic net) on
  development rows, their frozen simple average evaluated once on 2022-2023 -
  reserved rows only, labels aligned with their bars (a misalignment would collapse
  the rank IC), the companions (simple average, best individual, constant) in the
  same pass, the second-look note in the access log;
* the final test's light development loader reads development rows only, with
  every outcome window that crosses the reserved start purged.
"""

from __future__ import annotations

import ast
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from ml_synth import small_config
from selection_synth import RESERVED, TF, selection_config, stamps, write_stores
from test_final_test_isolation import _dev_data, _FakeRegressionStore
from xauusd_quant.ensemble import final_test as eft
from xauusd_quant.ensemble.final_test import (
    EnsembleFinalTestRefusedError,
    run_ensemble_final_test,
)
from xauusd_quant.ensemble.registry import EnsembleSpecError, freeze_ensemble_spec
from xauusd_quant.ml import final_test as mft
from xauusd_quant.ml.datasets import read_years
from xauusd_quant.ml.registry import freeze_spec, load_frozen_spec, save_artifact
from xauusd_quant.ml.training import train_final

SRC = Path(__file__).resolve().parents[1] / "src" / "xauusd_quant"
_SELECTION = ("research/ensemble_research.py", "research/ensemble_ablation.py",
              "research/ensemble_reports.py", "research/ensemble_plots.py",
              "research/ensemble_freeze.py", "ensemble/data.py", "ensemble/averaging.py",
              "ensemble/weighting.py", "ensemble/stacking.py", "ensemble/meta_model.py",
              "ensemble/calibration.py", "ensemble/diversity.py", "ensemble/disagreement.py",
              "ensemble/stability.py", "ensemble/diagnostics.py", "ensemble/inference.py",
              "ensemble/streaming.py", "ensemble/contract.py", "ensemble/registry.py",
              "ensemble/config.py")


def test_no_ensemble_selection_code_can_reach_the_reserved_period() -> None:
    for rel in _SELECTION:
        tree = ast.parse((SRC / rel).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
        assert not any("final_test" in m or m == "load_reserved_rows" for m in imported), rel


def _ensemble(tmp_path: Path, dev, cfg) -> Path:  # type: ignore[no-untyped-def]
    feats = ["sig_a", "vol_x"]
    common = {"timeframe": TF, "features": feats, "feature_set": "standard",
              "feature_set_id": "FS", "feature_set_hash": "h", "seed": 7, "horizon": 5,
              "target": "future_return", "target_definition": {"task": "regression"},
              "training_policy": {"scheme": "expanding", "rolling_years": 5},
              "preprocessing": {"kind": "linear", "clip": 8.0}, "calibration": "none",
              "dataset_versions": {"tick_dataset_version": "ticks-synthetic"}}
    members = []
    for fam, params in (("ridge", {"alpha": 1.0}), ("elastic_net", {"alpha": 1e-4,
                                                                    "l1_ratio": 0.5})):
        spec = load_frozen_spec(freeze_spec({**common, "spec_id": f"{fam.upper()}_STD_RET",
                                             "family": fam, "params": params},
                                            tmp_path / "constituents"))
        model, pre, cal, info = train_final(dev, spec, cfg.targets["future_return"], cfg)
        save_artifact(tmp_path / "models" / spec["spec_id"], model=model, preprocessor=pre,
                      calibrator=cal, spec=spec, extra={"training": info})
        members.append({"name": f"{fam}|standard", "model_id": spec["spec_id"],
                        "spec_hash": spec["content_hash"], "family": fam,
                        "feature_set": "standard", "feature_set_id": "FS",
                        "feature_set_hash": "h", "features": feats, "calibration": "none"})
    return freeze_ensemble_spec({
        "spec_id": "ENS_RETURN_1H_H5_V001", "timeframe": TF, "target": "future_return",
        "target_definition": {"task": "regression"}, "horizon": 5, "task": "regression",
        "method": "simple_average", "role": "candidate", "constituents": members,
        "combination": {"kind": "simple_average"}, "calibration": {"method": "none"},
        "companions": {"best_individual": "ridge|standard"}}, tmp_path / "frozen")


@pytest.fixture
def stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
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
    monkeypatch.setattr(mft, "RegressionFeatureStore", _FakeRegressionStore)
    monkeypatch.setattr(eft, "current_manifests",
                        lambda _p: {"FS": {"hash": "h", "features": ["sig_a", "vol_x"]}})
    dev = _dev_data(fcfg, tcfg, ["sig_a", "vol_x"])
    cfg = small_config(tmp_path)
    cfgs = SimpleNamespace(selection=selection_config(), features=fcfg, targets=tcfg,
                           config=None, regression=None)
    ecfg = SimpleNamespace(models_path=tmp_path / "models",
                           stability={"min_rows_year": 10, "min_rows_quarter": 10})
    return SimpleNamespace(dev=dev, cfg=cfg, cfgs=cfgs, ecfg=ecfg, root=tmp_path)


def test_the_ensemble_final_test_end_to_end(stores) -> None:  # type: ignore[no-untyped-def]
    s = stores
    spec_path = _ensemble(s.root, s.dev, s.cfg)
    out = s.root / "final_test"
    results = run_ensemble_final_test(s.cfgs, s.cfg, s.ecfg, s.dev, [spec_path], out_dir=out)
    r = results[0]
    assert r["first"].startswith("2022-01-03") and r["last"].startswith("2023-12-29")
    ov = r["overall"]
    assert ov["ensemble"]["rank_ic"] > 0.3                # aligned labels: the signal survives
    assert {"ensemble", "simple_average", "best_individual", "constant"} <= set(ov)
    np.testing.assert_allclose(ov["ensemble"]["rank_ic"], ov["simple_average"]["rank_ic"])
    assert {row["period"] for row in r["by_year"]} == {2022, 2023}
    assert r["summary"]["invalid_rows"] == 0 and "second look" in r["summary"]["note"]
    audit = pl.read_parquet(out / "ENS_RETURN_1H_H5_V001_predictions.parquet")
    assert audit["timestamp"].min() >= datetime(2022, 1, 1)   # reserved rows only
    assert audit.height == r["rows"] + int(audit["label"].is_nan().sum())
    events = [json.loads(line) for line in
              (out / "access_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["event"] for e in events] == ["final_test_started", "evaluated"]
    assert "second look" in events[0]["note"]
    with pytest.raises(EnsembleFinalTestRefusedError, match="one-time"):
        run_ensemble_final_test(s.cfgs, s.cfg, s.ecfg, s.dev, [spec_path], out_dir=out)
    run_ensemble_final_test(s.cfgs, s.cfg, s.ecfg, s.dev, [spec_path], out_dir=out,
                            repeat_reason="audit of a corrected loader")
    events = [json.loads(line) for line in
              (out / "access_log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert events[2]["event"] == "repeat_authorised"
    assert events[2]["reason"] == "audit of a corrected loader"


def test_an_interrupted_look_counts_and_live_config_must_match(stores) -> None:  # type: ignore[no-untyped-def]
    """A run that logged ``final_test_started`` and never finished was still a look; a
    live configuration or an artifact manifest that differs from the frozen ones is
    refused - all before any reserved row is read."""
    s = stores
    spec_path = _ensemble(s.root, s.dev, s.cfg)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    out = s.root / "final_test"
    out.mkdir()
    (out / "access_log.jsonl").write_text(json.dumps({
        "utc": "2026-01-01T00:00:00Z", "event": "final_test_started",
        "specs": {spec["spec_id"]: spec["content_hash"]}}) + "\n", encoding="utf-8")
    with pytest.raises(EnsembleFinalTestRefusedError, match="one-time"):
        run_ensemble_final_test(s.cfgs, s.cfg, s.ecfg, s.dev, [spec_path], out_dir=out)
    other = s.root / "other_test"
    tampered = {spec["spec_id"]: {"constituent_manifests": {
        spec["constituents"][0]["model_id"]: "0" * 64}}}
    with pytest.raises(EnsembleFinalTestRefusedError, match="changed after"):
        run_ensemble_final_test(s.cfgs, s.cfg, s.ecfg, s.dev, [spec_path], out_dir=other,
                                finalized=tampered)
    from dataclasses import replace

    changed = replace(s.cfg, targets={**s.cfg.targets, "future_return": replace(
        s.cfg.targets["future_return"], task="classification")})
    with pytest.raises(EnsembleFinalTestRefusedError, match="target task"):
        run_ensemble_final_test(s.cfgs, changed, s.ecfg, s.dev, [spec_path], out_dir=other)
    assert not (other / "access_log.jsonl").exists() or "evaluated" not in \
        (other / "access_log.jsonl").read_text(encoding="utf-8")


def test_unfrozen_or_edited_specs_are_refused_before_any_read(stores) -> None:  # type: ignore[no-untyped-def]
    s = stores
    spec_path = _ensemble(s.root, s.dev, s.cfg)
    body = json.loads(spec_path.read_text(encoding="utf-8"))
    body["combination"] = {"kind": "median"}              # edited after freezing
    spec_path.write_text(json.dumps(body), encoding="utf-8")
    out = s.root / "final_test"
    with pytest.raises(EnsembleSpecError, match="content hash"):
        run_ensemble_final_test(s.cfgs, s.cfg, s.ecfg, s.dev, [spec_path], out_dir=out)
    assert not out.exists()                               # nothing logged, nothing read


def test_the_slim_development_loader_reads_development_rows_only(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``slim_development_data`` - the final test's light loader, added after the first
    ensemble final test ran out of memory - reads like ``load_ml_data``: rows before the
    reserved start only (the reserved target files are garbage, so opening one fails),
    every outcome window that crosses the start purged, the residual and trailing
    volatility cut at the start, and every label of every target derivable from it."""
    from xauusd_quant.features import store as fstore

    fcfg, tcfg = write_stores(tmp_path)                   # reserved target files: garbage
    fbase = fcfg.factory_path / f"timeframe={TF}" / "version=factory-synthetic"
    tbase = tcfg.targets_path / f"timeframe={TF}" / "version=targets-synthetic"
    rng = np.random.default_rng(5)
    for part in sorted(fbase.glob("year=*/part-0.parquet")):   # the two context columns
        frame = pl.read_parquet(part)
        frame.with_columns(
            pl.Series("spread_rel", (1.0 + rng.random(frame.height)).astype(np.float32)),
            pl.Series("log_rv_20", rng.normal(-6.0, 0.5, frame.height).astype(np.float32)),
        ).write_parquet(part)
    registry = json.loads((fbase / "registry.json").read_text(encoding="utf-8"))
    registry += [{**registry[0], "name": "spread_rel"}, {**registry[0], "name": "log_rv_20"}]
    (fbase / "registry.json").write_text(json.dumps(registry), encoding="utf-8")
    families = ("return", "abs_return", "realized_vol", "residual_reduction", "residual_shrinks")
    for part in sorted(tbase.glob("year=*/part-0.parquet")):
        if int(part.parent.name.split("=")[1]) >= RESERVED.year:
            continue                                      # the reserved files stay garbage
        frame = pl.read_parquet(part)
        frame.with_columns(                               # RV itself (the store holds RV)
            *[pl.col(f"target_realized_vol_{h}").exp() for h in (1, 5, 20)],
            *[pl.Series(f"target_residual_shrinks_{h}",
                        (rng.random(frame.height) < 0.5).astype(np.float32)) for h in (1, 5, 20)],
        ).write_parquet(part)
    tman = json.loads((tbase / "_manifest.json").read_text(encoding="utf-8"))
    tman["targets"] = [f"target_{fam}_{h}" for fam in families for h in (1, 5, 20)]
    (tbase / "_manifest.json").write_text(json.dumps(tman), encoding="utf-8")
    monkeypatch.setattr(fstore, "RegressionFeatureStore", _FakeRegressionStore)
    cfgs = SimpleNamespace(selection=selection_config(), features=fcfg, targets=tcfg,
                           config=None, regression=None)

    dev = eft.slim_development_data(cfgs, TF, [1, 5])

    ts = stamps()
    assert dev.timestamps.equals(ts.filter(ts < datetime(2022, 1, 1)))
    assert dev.reserved_start == RESERVED
    assert set(dev.targets_raw) == {f"target_{fam}_{h}" for fam in families for h in (1, 5)}
    n = dev.n
    for name, v in dev.targets_raw.items():
        h = int(name.rsplit("_", 1)[1])
        assert np.isnan(v[n - h:]).all(), name            # windows crossing the start
        assert np.nanmax(np.abs(v)) < 1e3, name           # no poisoned reserved value
    direct = read_years(fbase, ["spread_rel", "log_rv_20"], RESERVED)
    for c in ("spread_rel", "log_rv_20"):
        np.testing.assert_array_equal(dev.features[c], direct[c].to_numpy())
    reg = _FakeRegressionStore().load(TF, 128, []).filter(
        pl.col("timestamp") < datetime(2022, 1, 1))
    np.testing.assert_array_equal(dev.epsilon, reg["residual"].to_numpy())
    np.testing.assert_array_equal(dev.sigma, reg["trailing_volatility"].to_numpy())
    cfg = small_config(tmp_path)
    for name, spec in cfg.targets.items():
        for h in sorted({1, 5} & set(spec.horizons)):
            y = dev.target(spec, h, cfg.log_floor).y
            assert np.isnan(y[n - h:]).all(), (name, h)     # no label crosses the start
            assert np.isfinite(y[: n - h]).mean() > 0.9, (name, h)
