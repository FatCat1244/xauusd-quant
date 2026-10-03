"""Regression guards for scan-time development isolation and interrupted-access logs."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl
import pytest

from ml_synth import small_config
from selection_synth import RESERVED, TF, selection_config, write_stores
from test_final_test_isolation import _frozen
from xauusd_quant.features.store import RegressionFeatureStore
from xauusd_quant.ml import datasets, final_test
from xauusd_quant.research import ensemble_research as ens


def test_regression_boundary_precedes_every_feature_materialization(
    world: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, regression = world
    store = RegressionFeatureStore(config, regression)
    store.build("1m", [32])
    full = store.load("1m", 32, ["timestamp", "residual", "trailing_volatility"])
    boundary = full["timestamp"][400]
    original = pl.LazyFrame.collect
    collected = []

    def guarded_collect(lazy: pl.LazyFrame, *args: Any, **kwargs: Any) -> pl.DataFrame:
        materialized = original(lazy, *args, **kwargs)
        collected.append(materialized)
        if set(materialized.columns) & {"residual", "trailing_volatility"}:
            assert materialized.height <= 400, "reserved regression values were materialized"
        return materialized

    monkeypatch.setattr(pl.LazyFrame, "collect", guarded_collect)
    part = store.load("1m", 32, ["timestamp", "residual", "trailing_volatility"], before=boundary)
    assert part.equals(full.head(400))
    assert len(collected) == 2
    assert store.load("1m", 32, ["timestamp"], before=date(2021, 1, 1)).is_empty()


def test_both_development_entry_points_supply_the_reserved_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from xauusd_quant.ensemble import final_test as eft
    from xauusd_quant.features import store as fstore

    fcfg, tcfg = write_stores(tmp_path)

    # Stop at the store call: the test verifies the actual end-to-end call argument,
    # while the preceding features/target scans must survive corrupt reserved files.
    class StopAtBoundaryError(Exception):
        pass

    class GuardStore:
        def __init__(self, *args: Any) -> None:
            pass

        def load(self, *args: Any, before: date | None = None) -> pl.DataFrame:
            assert before == RESERVED, "development caller failed to bound regression scan"
            raise StopAtBoundaryError

    base = fcfg.factory_path / f"timeframe={TF}" / "version=factory-synthetic"
    registry = json.loads((base / "registry.json").read_text())
    registry = [r for r in registry if r.get("live_safe")]
    manifest = {
        "features": [
            {"name": r["name"], "feature_version": r["feature_version"]} for r in registry
        ],
        "feature_set_id": "FS",
        "content_hash": "hash",
        "factory_version": "factory-synthetic",
    }
    monkeypatch.setattr(
        datasets,
        "_manifests",
        lambda *a: (
            {"standard": [r["name"] for r in registry]},
            {"standard": "FS"},
            {"standard": "hash"},
            {"standard": manifest},
        ),
    )
    tptr = tcfg.targets_path / f"timeframe={TF}" / "version=targets-synthetic" / "_manifest.json"
    tman = json.loads(tptr.read_text())
    tman["targets"] = [
        f"target_{family}_{h}"
        for family in ("return", "abs_return", "realized_vol", "residual_reduction")
        for h in (1, 5, 20)
    ]
    tptr.write_text(json.dumps(tman), encoding="utf-8")
    monkeypatch.setattr(datasets, "RegressionFeatureStore", GuardStore)
    monkeypatch.setattr(fstore, "RegressionFeatureStore", GuardStore)
    cfgs = SimpleNamespace(
        selection=selection_config(), features=fcfg, targets=tcfg, config=None, regression=None
    )
    with pytest.raises(StopAtBoundaryError):
        datasets.load_ml_data(small_config(tmp_path), cfgs.selection, fcfg, tcfg, None, None, TF)
    with pytest.raises(StopAtBoundaryError):
        eft.slim_development_data(cfgs, TF, [1, 5])


def test_interrupted_ml_attempt_is_a_look_and_requires_a_logged_repeat_reason(
    tmp_path: Path,
) -> None:
    spec = _frozen(tmp_path / "frozen")
    body = json.loads(spec.read_text())
    out = tmp_path / "final_test"
    out.mkdir()
    log = out / "access_log.jsonl"
    log.write_text(
        json.dumps(
            {
                "event": "final_test_started",
                "utc": "2021-01-01",
                "specs": {body["spec_id"]: body["content_hash"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert len(final_test.prior_evaluations(log, {body["content_hash"]})) == 1
    assert final_test.prior_evaluations(log, {"another-hash"}) == []
    with pytest.raises(final_test.FinalTestRefusedError, match="one-time"):
        final_test.run_final_test(None, small_config(tmp_path), None, [spec], out_dir=out)
    assert len(log.read_text().splitlines()) == 1
    with pytest.raises(AttributeError):
        final_test.run_final_test(
            None,
            small_config(tmp_path),
            None,
            [spec],
            out_dir=out,
            repeat_reason="explicit recovery of an interrupted attempt",
        )
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert events[1]["event"] == "repeat_authorised"
    assert events[1]["reason"] == "explicit recovery of an interrupted attempt"


def test_undefined_null_cannot_hide_beside_a_passing_null() -> None:
    status, _ = ens._null_verdict(
        [
            {"variant": "shift", "real_minus_control": 0.1, "blocks": 5, "wins": 5},
            {
                "variant": "pipeline-random_walk",
                "real_minus_control": float("nan"),
                "blocks": 5,
                "wins": 5,
            },
        ]
    )
    assert status == "untested"
    status, _ = ens._null_verdict(
        [{"variant": "shift", "real_minus_control": 0.1, "blocks": 5, "wins": 5}],
        expected=("shift", "pipeline-random_walk"),
    )
    assert status == "untested"


def test_ensemble_missing_null_never_enters_an_eligible_universe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(12)
    tspec = SimpleNamespace(kind="direction", task="regression", is_classification=False)
    data = SimpleNamespace(
        n=2000,
        registry={"sig": {"live_safe": True}},
        features={"sig": rng.normal(size=2000)},
        manifests={"standard": ["sig"]},
        manifest_ids={"standard": "FS"},
        target=lambda *args: SimpleNamespace(y=rng.normal(size=2000)),
    )
    ml = SimpleNamespace(
        targets={"future_return": tspec},
        log_floor=1e-5,
        walk_forward=SimpleNamespace(embargo_bars=1),
        tree_families=(),
        reference_linear=lambda task: "ridge",
    )
    context = SimpleNamespace(
        data=data,
        ml=ml,
        cfg=SimpleNamespace(
            eligibility={"min_folds_beating_baseline": 1, "registered_nulls": ["shift"]}
        ),
    )
    pair = SimpleNamespace(
        target="future_return",
        horizon=1,
        timeframe="1m",
        model_names=["ridge|standard"],
        block_names=["fold"],
        family=lambda m: "ridge",
        feature_set=lambda m: "standard",
        models={"ridge|standard": {"feature_set_id": "FS"}},
        block_positions=lambda k: np.array([0]),
        rows=np.array([800]),
    )
    table = pl.DataFrame(
        [
            {
                "target": "future_return",
                "horizon": 1,
                "family": "ridge",
                "feature_set": "standard",
                "variant": "base",
                "mean": 1.0,
                "min": 1.0,
                "blocks_beating_baseline": 1,
            }
        ]
    )
    monkeypatch.setattr(
        ens,
        "_tables10",
        lambda ctx: {
            "configuration_summary": table,
            "null_comparison": pl.DataFrame(),
            "skill_beyond_base_rate": pl.DataFrame(),
        },
    )
    monkeypatch.setattr(
        ens,
        "_unit_jsons",
        lambda *args: [
            {
                "features": ["sig"],
                "fold": {
                    "fit": [0, 500],
                    "inner": [600, 700],
                    "validate": [800, 1000],
                    "name": "fold",
                },
            }
        ],
    )
    result = ens.eligibility_table(context, pair)
    assert result["registered_null"][0] == "untested"
    assert result["status"][0] == "untested"
    assert result["flag_untested"][0]
