"""Pin Prompt #11 Steps 89-91: descriptive records, source statuses and versioning."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import polars as pl
import pytest

from xauusd_quant.ensemble.contract import (
    FIELDS,
    ContractError,
    build_record,
    contract_frame,
    prediction_version,
    records_from_frame,
    validate_record,
)

HORIZONS = dict.fromkeys(FIELDS, 4)
FLOOR = 0.125


def _sources() -> dict[str, dict[str, Any]]:
    return {
        field: {"ensemble_id": f"ENS_{field}_V001", "horizon_bars": 4, "status": "ok"}
        for field in FIELDS
    }


def _record() -> dict[str, Any]:
    return build_record(
        timestamp="2021-12-30T00:00:00",
        timeframe="5m",
        version="PRED_5M_V001-example",
        values={
            "reversion_probability": 0.75,
            "expected_return_vol_scaled": 0.25,
            "expected_log_volatility": math.log(0.5),
            "expected_log_abs_move": math.log(0.25),
        },
        sources=_sources(),
        sigma=0.5,
        horizons=HORIZONS,
        log_floor=FLOOR,
        uncertainty={
            "model_disagreement": 0.125,
            "disagreement_by_output": {"future_return": 0.25},
            "ood_score": 0.5,
            "regime_entropy": 0.25,
            "reversion_entropy": 0.5,
        },
    )


def test_built_record_validates_and_derived_values_use_declared_scales() -> None:
    record = _record()
    assert validate_record(record) is record
    assert record["expected_return"] == pytest.approx(0.25)
    assert record["expected_return"] == pytest.approx(
        record["expected_return_vol_scaled"] * 0.5 * math.sqrt(4)
    )
    assert record["expected_volatility"] == pytest.approx(0.375)
    assert record["expected_volatility"] == pytest.approx(
        math.exp(record["expected_log_volatility"]) - FLOOR
    )
    assert record["expected_abs_move"] == pytest.approx(0.125)


@pytest.mark.parametrize("key", ["signal", "position_size", "buy_threshold", "pnl"])
@pytest.mark.parametrize("location", ["root", "uncertainty", "source"])
def test_decision_like_keys_are_refused_at_every_tested_depth(
    key: str, location: str
) -> None:
    record = _record()
    if location == "root":
        record[key] = 0.5
    elif location == "uncertainty":
        record["uncertainty"][key] = 0.5
    else:
        record["sources"]["reversion_probability"]["details"] = {key: 0.5}
    with pytest.raises(ContractError, match="decision-like field"):
        validate_record(record)


@pytest.mark.parametrize(
    "field",
    [
        "timestamp",
        "timeframe",
        "prediction_version",
        *FIELDS,
        "expected_return",
        "expected_volatility",
        "expected_abs_move",
        "uncertainty",
        "sources",
    ],
)
def test_missing_required_field_is_refused(field: str) -> None:
    record = _record()
    del record[field]
    with pytest.raises(ContractError, match="missing fields"):
        validate_record(record)


@pytest.mark.parametrize("nested", [False, True])
def test_unknown_field_is_refused(nested: bool) -> None:
    record = _record()
    if nested:
        record["uncertainty"]["unexpected"] = 0.5
    else:
        record["unexpected"] = 0.5
    with pytest.raises(ContractError, match="unknown"):
        validate_record(record)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize(
    "field",
    [*FIELDS, "expected_return", "expected_volatility", "expected_abs_move"],
)
def test_non_finite_output_is_refused(field: str, value: float) -> None:
    record = _record()
    record[field] = value
    with pytest.raises(ContractError, match="finite number or null"):
        validate_record(record)


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_probability_outside_unit_interval_is_refused(probability: float) -> None:
    record = _record()
    record["reversion_probability"] = probability
    with pytest.raises(ContractError, match="not a probability"):
        validate_record(record)


@pytest.mark.parametrize("field", list(FIELDS))
def test_ok_source_requires_a_value(field: str) -> None:
    record = _record()
    record[field] = None
    with pytest.raises(ContractError, match="status 'ok' with value None"):
        validate_record(record)


@pytest.mark.parametrize(
    "status", ["not_frozen", "no_eligible_ensemble", "ENSEMBLE_INVALID", "missing_input"]
)
def test_non_ok_source_cannot_carry_a_value(status: str) -> None:
    record = _record()
    record["sources"]["reversion_probability"]["status"] = status
    with pytest.raises(ContractError, match="with value"):
        validate_record(record)


@pytest.mark.parametrize(
    "status", ["not_frozen", "no_eligible_ensemble", "ENSEMBLE_INVALID", "missing_input"]
)
def test_builder_nulls_values_from_non_ok_sources(status: str) -> None:
    sources = _sources()
    for source in sources.values():
        source["status"] = status
    record = build_record(
        timestamp="2021-12-30T00:00:00",
        timeframe="5m",
        version="example",
        values=dict.fromkeys(FIELDS, 0.25),
        sources=sources,
        sigma=0.5,
        horizons=HORIZONS,
        log_floor=FLOOR,
        uncertainty={},
    )
    for field in FIELDS:
        assert record[field] is None
        assert record["sources"][field]["status"] == status
    for field in ("expected_return", "expected_volatility", "expected_abs_move"):
        assert record[field] is None
    assert validate_record(record) is record


def test_frame_statuses_and_records_preserve_missing_and_unfrozen_outputs() -> None:
    start = datetime(2021, 12, 30)
    timestamps = pl.Series(
        "timestamp",
        [start + timedelta(minutes=5 * i) for i in range(3)],
        dtype=pl.Datetime("us"),
    )
    columns = {
        "reversion_probability": np.array([0.75, np.nan, 0.25]),
        "expected_return_vol_scaled": np.array([0.25, 0.5, np.nan]),
        "expected_log_volatility": np.array([math.log(0.5)] * 3),
        "expected_log_abs_move": np.array([math.log(0.25)] * 3),
    }
    statuses = dict.fromkeys(FIELDS, "ok")
    statuses["expected_log_abs_move"] = "no_eligible_ensemble"
    frame = contract_frame(
        timestamps,
        "5m",
        "example",
        columns,
        statuses,
        np.array([0.5] * 3),
        HORIZONS,
        FLOOR,
    )
    assert frame["status_reversion_probability"].to_list() == [
        "ok", "missing_input", "ok"
    ]
    assert frame["status_expected_return_vol_scaled"].to_list() == [
        "ok", "ok", "missing_input"
    ]
    assert frame["status_expected_log_volatility"].to_list() == ["ok"] * 3
    assert frame["status_expected_log_abs_move"].to_list() == [
        "no_eligible_ensemble"
    ] * 3
    assert frame["reversion_probability"].to_list() == [0.75, None, 0.25]
    assert frame["expected_log_abs_move"].to_list() == [None] * 3
    assert frame["expected_abs_move"].to_list() == [None] * 3
    assert frame["expected_return"].to_list() == [0.25, 0.5, None]
    assert frame["expected_volatility"].to_list() == pytest.approx([0.375] * 3)

    records = records_from_frame(
        frame, sources=_sources(), horizons=HORIZONS, log_floor=FLOOR
    )
    assert len(records) == 3
    for index, record in enumerate(records):
        assert validate_record(record) is record
        for field in FIELDS:
            assert record["sources"][field]["status"] == frame[f"status_{field}"][index]
    assert records[1]["reversion_probability"] is None
    assert records[2]["expected_return_vol_scaled"] is None
    assert records[2]["expected_return"] is None


def test_prediction_version_is_deterministic_and_tracks_ensemble_ids() -> None:
    ensembles = {"reversion_probability": "ENS_A", "expected_log_volatility": "ENS_B"}
    version = prediction_version("5m", ensembles)
    assert version == prediction_version("5m", ensembles)
    assert version == prediction_version("5m", dict(reversed(list(ensembles.items()))))
    assert version.startswith("PRED_5M_V001-")
    changed = {**ensembles, "reversion_probability": "ENS_A_V002"}
    assert version != prediction_version("5m", changed)
