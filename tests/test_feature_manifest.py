"""Feature-set manifests: deterministic order, immutability, exact matrix reproduction
(Prompt #9, Steps 54-59, 68-69).
"""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from selection_synth import TF, registry_rows, synthetic_frames, write_stores
from xauusd_quant.selection.config import load_selection_config
from xauusd_quant.selection.manifest import (
    ManifestConflictError,
    build_manifest,
    load_manifest,
    matrix_from_manifest,
    ordered_features,
    write_manifest,
)

SET = ["vol_x", "sig_a", "noise_2", "rare_flag", "drifting", "noise_1"]


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    root = tmp_path_factory.mktemp("manifest_store")
    fcfg, _ = write_stores(root, corrupt_reserved_targets=False)
    registry = {r["name"]: r for r in registry_rows(fcfg)}
    return fcfg, registry


def _manifest(registry, names, version: int = 1):
    order = load_selection_config().manifest.families_order
    return build_manifest("standard", TF, names, registry, version=version, families_order=order,
                          provenance={"dataset_version": "ticks-synthetic"})


def test_order_is_family_then_registry_order_whatever_the_input_order(store) -> None:
    _, registry = store
    order = load_selection_config().manifest.families_order
    expected = ordered_features(SET, registry, order)
    assert expected == ["sig_a", "vol_x", "noise_1", "drifting", "noise_2", "rare_flag"]
    rng = random.Random(0)
    for _ in range(10):
        shuffled = SET[:]
        rng.shuffle(shuffled)
        assert ordered_features(shuffled, registry, order) == expected
        m = _manifest(registry, shuffled)
        assert [f["name"] for f in m["features"]] == expected
        assert [f["position"] for f in m["features"]] == list(range(len(SET)))
        assert m["content_hash"] == _manifest(registry, SET)["content_hash"]


def test_manifest_carries_what_prompt_10_needs(store) -> None:
    _, registry = store
    m = _manifest(registry, SET)
    assert m["feature_set_id"] == "FEATURESET_1H_STANDARD_V001"
    assert m["max_history_required"] == max(registry[n]["min_history"] for n in SET)
    for f in m["features"]:
        for key in ("feature_id", "family", "dtype", "transformation", "window",
                    "required_history", "live_safe", "feature_version", "refresh"):
            assert key in f, key
        assert f["live_safe"] is True


def test_manifests_are_immutable(store, tmp_path) -> None:
    _, registry = store
    path = tmp_path / "standard.json"
    m = _manifest(registry, SET)
    assert write_manifest(path, m) is True
    assert write_manifest(path, _manifest(registry, list(reversed(SET)))) is False   # same content
    with pytest.raises(ManifestConflictError):
        write_manifest(path, _manifest(registry, SET[:-1]))                           # other set
    assert write_manifest(tmp_path / "v2.json", _manifest(registry, SET[:-1], version=2))
    loaded = load_manifest(path)
    assert loaded["content_hash"] == m["content_hash"]
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["features"][0]["feature_version"] = "fv-edited"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ManifestConflictError):
        load_manifest(path)


def test_a_saved_manifest_reproduces_the_matrix_in_its_order(store, tmp_path) -> None:
    fcfg, registry = store
    m = _manifest(registry, SET)
    path = tmp_path / "m.json"
    write_manifest(path, m)
    frame = matrix_from_manifest(load_manifest(path), fcfg)
    names = [f["name"] for f in m["features"]]
    assert frame.columns == ["timestamp", *names]
    features, _ = synthetic_frames()
    for name in names:
        np.testing.assert_array_equal(frame[name].to_numpy(), features[name].to_numpy())
    part = matrix_from_manifest(m, fcfg, start=features["timestamp"][100],
                                end=features["timestamp"][200])
    assert part.height == 100 and part.columns == frame.columns


def test_a_stale_feature_version_is_refused(store) -> None:
    fcfg, registry = store
    m = _manifest(registry, SET)
    m["features"][2]["feature_version"] = "fv-older"
    with pytest.raises(ManifestConflictError):
        matrix_from_manifest(m, fcfg)
