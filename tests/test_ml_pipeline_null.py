"""The pipeline-null stages of the supervised research (Prompt #10, invariant 9).

The registered null (a random walk) and the post-hoc ones (``nulls.pipeline_posthoc``,
the sign-flip martingale with the real volatility path) are planned as separate
variants on the same blocks, their unit stamps carry the null's name, and the
expensive null build is skipped when every unit of that null is already current.
The build itself needs the real bar store and is replaced here by the synthetic
data under the null's versions.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ml_synth import small_config, synthetic_ml_data
from xauusd_quant.research import ml_research
from xauusd_quant.research.ml_research import (
    MLContext,
    null_version,
    pending_units,
    pipeline_null_names,
    plan_stage,
    run_ml_research,
)


def _ctx(tmp_path: Path) -> MLContext:
    cfg = small_config(tmp_path)
    cfg = replace(cfg, nulls={**cfg.nulls, "pipeline": "random_walk",
                              "pipeline_posthoc": ["sign_flip"]})
    return MLContext(timeframe="1h", cfgs=SimpleNamespace(ml=cfg),  # type: ignore[arg-type]
                     data=synthetic_ml_data(), out_dir=tmp_path / "ml_research" / "1h")


def test_registered_and_posthoc_nulls_are_separate_variants(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    assert pipeline_null_names(ctx, "pipeline_null") == ["random_walk"]
    assert pipeline_null_names(ctx, "pipeline_null_posthoc") == ["sign_flip"]
    registered = plan_stage(ctx, "pipeline_null")
    posthoc = plan_stage(ctx, "pipeline_null_posthoc")
    assert len(registered) == len(posthoc) > 0
    assert {u.variant["pipeline"] for u in registered} == {"random_walk"}
    assert {u.variant["pipeline"] for u in posthoc} == {"sign_flip"}
    assert not {u.key() for u in registered} & {u.key() for u in posthoc}
    assert null_version(ctx, "sign_flip") == {"null": "sign_flip",
                                              "seed": ctx.cfg.random_seed}


def test_the_null_is_built_only_when_one_of_its_units_is_pending(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _ctx(tmp_path)
    built: list[str] = []

    def fake_build(c: MLContext, _features: list[str], name: str | None = None) -> Any:
        built.append(str(name))
        return replace(c.data, versions={**c.data.versions,
                                         "pipeline_null": null_version(c, str(name))},
                       _designs={})

    monkeypatch.setattr(ml_research, "load_context", lambda *_a, **_k: ctx)
    monkeypatch.setattr(ml_research, "build_pipeline_null", fake_build)
    stages = ("pipeline_null", "pipeline_null_posthoc")
    first = run_ml_research(ctx.cfgs, "1h", stages=stages)      # type: ignore[arg-type]
    assert built == ["random_walk", "sign_flip"]
    for stage in stages:
        s = first["stages"][stage]
        assert s["failed"] == 0 and s["ran"] == s["units"] > 0
    again = run_ml_research(ctx.cfgs, "1h", stages=stages)      # type: ignore[arg-type]
    assert built == ["random_walk", "sign_flip"]                # nothing rebuilt
    assert all(again["stages"][s]["ran"] == 0 for s in stages)
    # the units are current only under their own null: another null name is pending
    probe = SimpleNamespace(versions={**ctx.data.versions,
                                      "pipeline_null": null_version(ctx, "shuffled_returns")})
    assert pending_units(ctx, plan_stage(ctx, "pipeline_null"), probe)  # type: ignore[arg-type]
