"""Ensemble ids and frozen ensemble specs (Prompt #11, Steps 60-61).

Ids name the target, timeframe, horizon and version (``ENS_REVERSION_5M_H5_V001``);
constituents get their own namespace with the feature set in the id, so they never
collide with a Prompt #10 model id. A frozen spec is immutable: refreezing the same
content is a no-op, different content under the same id is refused, and a spec
edited after freezing (or never frozen) is refused on load.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from xauusd_quant.ensemble.registry import (
    EnsembleSpecError,
    constituent_id,
    ensemble_id,
    freeze_ensemble_spec,
    load_frozen_ensemble_spec,
)
from xauusd_quant.ml.registry import model_id


def _spec(**extra: object) -> dict:
    return {"spec_id": ensemble_id("mean_reversion", "5m", 5), "timeframe": "5m",
            "target": "mean_reversion", "horizon": 5, "task": "classification",
            "method": "simple_average", "role": "candidate",
            "constituents": [{"model_id": "LGBM_EXT_REVERSION_5M_H5_V001", "spec_hash": "x"}],
            "combination": {"kind": "simple_average"}, "calibration": {"method": "none"},
            **extra}


def test_ids_name_target_timeframe_horizon_set_and_version() -> None:
    assert ensemble_id("mean_reversion", "5m", 5) == "ENS_REVERSION_5M_H5_V001"
    assert ensemble_id("future_volatility", "15m", 5, 2) == "ENS_VOL_15M_H5_V002"
    cid = constituent_id("lightgbm", "extended", "mean_reversion", "5m", 5)
    assert cid == "LGBM_EXT_REVERSION_5M_H5_V001"
    assert cid != model_id("lightgbm", "mean_reversion", "5m", 5)   # no Prompt #10 collision
    assert constituent_id("random_forest", "target", "future_abs_move", "15m", 5) == \
        "RF_TGT_ABSMOVE_15M_H5_V001"


def test_a_frozen_spec_is_immutable(tmp_path: Path) -> None:
    path = freeze_ensemble_spec(_spec(), tmp_path)
    first = path.read_text(encoding="utf-8")
    assert freeze_ensemble_spec(_spec(), tmp_path) == path         # same content: no-op
    assert path.read_text(encoding="utf-8") == first
    body = json.loads(first)
    assert body["frozen"] is True and body["ENSEMBLE_SPEC_FROZEN"] is True
    with pytest.raises(EnsembleSpecError, match="bump it"):
        freeze_ensemble_spec(_spec(combination={"kind": "median"}), tmp_path)
    assert path.read_text(encoding="utf-8") == first                # never overwritten


def test_unfrozen_or_edited_specs_are_refused(tmp_path: Path) -> None:
    path = freeze_ensemble_spec(_spec(), tmp_path)
    assert load_frozen_ensemble_spec(path)["spec_id"] == "ENS_REVERSION_5M_H5_V001"
    body = json.loads(path.read_text(encoding="utf-8"))
    body["constituents"][0]["spec_hash"] = "another-version"        # edited after freezing
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(EnsembleSpecError, match="content hash"):
        load_frozen_ensemble_spec(path)
    loose = tmp_path / "ENSEMBLE_SPEC_X.json"
    loose.write_text(json.dumps(_spec()), encoding="utf-8")
    with pytest.raises(EnsembleSpecError, match="not frozen"):
        load_frozen_ensemble_spec(loose)
    with pytest.raises(EnsembleSpecError, match="no ensemble spec"):
        load_frozen_ensemble_spec(tmp_path / "missing.json")
