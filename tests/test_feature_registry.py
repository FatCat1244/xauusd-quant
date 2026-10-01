"""The central feature registry (Prompt #8, Steps 1-3, 6).

Every feature has an id, a family, a definition, a warm-up and a live-safe
flag; names are unique; parameter families group the windows of one
definition; prior evidence from Prompts #2-#7 is attached as a status and the
failed / invalid / non-causal statuses never reach the default candidates.
"""

from __future__ import annotations

import pytest

from xauusd_quant.features.factory_config import FAMILIES, PRIOR_STATUSES, load_features_config
from xauusd_quant.features.manifest import feature_version
from xauusd_quant.features.registry import (
    NON_CAUSAL,
    build_registry,
    feature_id,
    registry_frame_rows,
)

TIMEFRAMES = {"1h": 3600.0, "30m": 1800.0, "15m": 900.0, "5m": 300.0}


@pytest.fixture(scope="module")
def fcfg():
    return load_features_config()


@pytest.mark.parametrize("tf", list(TIMEFRAMES))
def test_every_spec_is_complete_and_unique(fcfg, tf: str) -> None:
    specs = build_registry(fcfg, tf, TIMEFRAMES[tf])
    names = [s.name for s in specs]
    ids = [s.feature_id for s in specs]
    assert len(set(names)) == len(names) and len(set(ids)) == len(ids)
    for s in specs:
        assert s.family in FAMILIES or s.family == "interaction", s.name
        assert s.definition and s.source_module and s.timeframe == tf
        assert s.live_safe and s.min_history >= 0 and s.dtype == "float32"
        assert s.prior_status in PRIOR_STATUSES
        assert f"_{tf.upper()}" in s.feature_id and s.feature_id == s.feature_id.upper()
        assert s.window is None or s.window > 0
        assert s.source_feed == fcfg.source_feed


def test_abs_innovation_spectrum_only_from_15m(fcfg) -> None:
    for tf, seconds in TIMEFRAMES.items():
        names = {s.name for s in build_registry(fcfg, tf, seconds)}
        has = any(n.startswith("fft_abs_eta_") for n in names)
        assert has == (seconds >= 900), tf


def test_prior_statuses_follow_the_earlier_layers(fcfg) -> None:
    specs = {s.name: s for s in build_registry(fcfg, "1h", 3600.0)}
    assert specs["fft_abs_eta_entropy_256"].prior_status == "candidate"
    assert specs["fft_dominant_period_256"].prior_status == "failed_null_control"
    assert specs["fft_dominant_period_256"].status_excludes_default
    assert specs["wav_top3_share_512"].prior_status == "redundant"
    assert not specs["log_rv_20"].status_excludes_default
    five = {s.name: s for s in build_registry(fcfg, "5m", 300.0)}
    assert five["wav_fast_slow_256"].invalid_reason is not None
    assert five["wav_fast_slow_256"].status_excludes_default


def test_parameter_families_group_windows_of_one_definition(fcfg) -> None:
    specs = build_registry(fcfg, "1h", 3600.0)
    fams: dict[str, set[int | None]] = {}
    for s in specs:
        if s.parameter_family:
            fams.setdefault(s.parameter_family, set()).add(s.window)
    multi = {k: v for k, v in fams.items() if len(v) > 1}
    assert "log_rv" in multi and {5, 20, 64, 256, 1024} <= multi["log_rv"]
    assert any(k.startswith("reg_resid_z") for k in multi)


def test_non_causal_constructions_are_registered_only_to_be_excluded(fcfg) -> None:
    rows = registry_frame_rows(build_registry(fcfg, "1h", 3600.0))
    blocked = [r for r in rows if r["family"] == "non_causal"]
    assert {r["name"] for r in blocked} == set(NON_CAUSAL)
    assert all(not r["live_safe"] and r["excluded_from_default_candidates"] for r in blocked)
    assert all(r.get("excluded_from_default_candidates") is not None for r in rows)


def test_feature_ids_and_versions(fcfg) -> None:
    assert feature_id("ou_log_half_life", "5m", 256) == "OU_LOG_HALF_LIFE_5M_256"
    assert feature_id("tod_sin", "1h") == "TOD_SIN_1H"
    spec = next(s for s in build_registry(fcfg, "1h", 3600.0) if s.name == "log_rv_20")
    v1 = feature_version(spec, sources={"bars": "a"}, config_fingerprint="c", code_version="x")
    assert v1 == feature_version(spec, sources={"bars": "a"}, config_fingerprint="c",
                                 code_version="x")
    for kwargs in ({"sources": {"bars": "b"}, "config_fingerprint": "c", "code_version": "x"},
                   {"sources": {"bars": "a"}, "config_fingerprint": "d", "code_version": "x"},
                   {"sources": {"bars": "a"}, "config_fingerprint": "c", "code_version": "y"}):
        assert feature_version(spec, **kwargs) != v1


def test_interactions_are_only_the_registered_pairs(fcfg) -> None:
    specs = [s for s in build_registry(fcfg, "1h", 3600.0) if s.family == "interaction"]
    assert {s.name for s in specs} == {ix.name for ix in fcfg.interactions}
    assert len(specs) == len(fcfg.interactions) < 20          # never all N^2 pairs
