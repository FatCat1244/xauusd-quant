"""The quality filter excludes with reasons and never on outcomes (Prompt #9, Steps 2, 5-8, 69).

Each synthetic feature is built to trigger one exclusion; the filter must name
that reason and keep every well-formed feature - a rare binary flag included,
and a strongly drifting feature, whose drift is classified, not excluded.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from selection_synth import EXPECTED_REASON, FEATURES, TF, selection_config, write_stores
from xauusd_quant.selection.data import load_selection_data
from xauusd_quant.selection.filters import REASONS, drift_class, quality_filter


@pytest.fixture(scope="module")
def filtered(tmp_path_factory):
    root = tmp_path_factory.mktemp("filter_store")
    fcfg, tcfg = write_stores(root)
    cfg = selection_config()
    data = load_selection_data(cfg, fcfg, tcfg, TF, with_probes=False)
    return cfg, data, quality_filter(data, cfg)


def test_every_feature_gets_a_row_and_known_bad_features_are_excluded(filtered) -> None:
    _, _, table = filtered
    rows = {r["feature"]: r for r in table.iter_rows(named=True)}
    assert set(rows) == set(FEATURES)
    for name, reason in EXPECTED_REASON.items():
        assert reason in rows[name]["reasons"].split(","), (name, rows[name]["reasons"])
        assert not rows[name]["kept_quality"]
    kept = {n for n, r in rows.items() if r["kept_quality"]}
    assert kept == set(FEATURES) - set(EXPECTED_REASON)
    for r in rows.values():
        assert all(code in REASONS for code in filter(None, r["reasons"].split(",")))


def test_a_rare_binary_flag_is_kept_and_a_vanishing_one_is_not(filtered) -> None:
    _, _, table = filtered
    rows = {r["feature"]: r for r in table.iter_rows(named=True)}
    assert rows["rare_flag"]["unique_values"] == 2 and rows["rare_flag"]["kept_quality"]
    assert 0.001 < rows["rare_flag"]["minority_fraction"] < 0.01
    assert rows["tiny_flag"]["minority_fraction"] < 0.001


def test_drift_is_classified_not_excluded(filtered) -> None:
    cfg, data, table = filtered
    rows = {r["feature"]: r for r in table.iter_rows(named=True)}
    assert rows["drifting"]["drift_class"] == "strongly_drifting"
    assert rows["drifting"]["kept_quality"]
    assert rows["noise_1"]["drift_class"] == "stable"
    years = data.timestamps.dt.year().to_numpy()
    x = np.random.default_rng(0).normal(size=years.size)
    cls, psi, _ = drift_class(x, years, np.arange(years.size), cfg)
    assert cls == "stable" and psi < cfg.filters.drift["moderate_psi"]


def test_the_filter_reads_no_outcome(filtered) -> None:
    cfg, data, table = filtered
    blind = replace(data, targets=np.full_like(data.targets, np.nan))    # no target at all
    assert quality_filter(blind, cfg).equals(table)
