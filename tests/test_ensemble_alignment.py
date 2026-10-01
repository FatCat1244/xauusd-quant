"""Pin Prompt #11 Steps 4-5: alignment, missing coverage and purged meta-history."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from xauusd_quant.ensemble.data import (
    AlignmentError,
    PairPredictions,
    ensemble_prediction_coverage,
    load_pair,
    load_unit_predictions,
)

MODEL = "logistic_l2|standard"
OTHER = "lightgbm|standard"
CONSTANT = "constant|standard"
VERSIONS = {
    "tick_dataset_version": "ticks-v1",
    "bar_dataset_version": "bars-v1",
    "factory_version": "factory-v1",
    "target_version": "targets-v1",
}


def _frame() -> pl.DataFrame:
    n = 8
    start = datetime(2021, 12, 30)
    return pl.DataFrame(
        {
            "timestamp": pl.Series(
                [start + timedelta(minutes=5 * i) for i in range(n)],
                dtype=pl.Datetime("us"),
            ),
            "row": pl.Series(range(n), dtype=pl.Int64),
            "fold": ["wf1"] * 4 + ["wf2"] * 4,
            "target": ["mean_reversion"] * n,
            "horizon": pl.Series([2] * n, dtype=pl.Int16),
            "dataset_version": ["ticks-v1"] * n,
            "label": pl.Series([0, 1] * 4, dtype=pl.Float32),
            "label_raw": pl.Series([0, 1] * 4, dtype=pl.Float32),
            MODEL: pl.Series([0.25] * n, dtype=pl.Float32),
            f"{MODEL}|platt": pl.Series([0.75] * n, dtype=pl.Float32),
            OTHER: pl.Series([0.375] * n, dtype=pl.Float32),
            f"{OTHER}|platt": pl.Series([0.625] * n, dtype=pl.Float32),
            CONSTANT: pl.Series([0.5] * n, dtype=pl.Float32),
        }
    )


def _meta() -> dict[str, Any]:
    return {
        "timeframe": "5m",
        "target": "mean_reversion",
        "horizon": 2,
        "models": {
            name: {
                "family": name.split("|")[0],
                "feature_set": "standard",
                "feature_set_id": "standard-v1",
                "blocks": ["wf1", "wf2"],
            }
            for name in (MODEL, OTHER, CONSTANT)
        },
        "versions": dict(VERSIONS),
    }


def _write(
    tmp_path: Path,
    frame: pl.DataFrame,
    meta: dict[str, Any] | None = None,
) -> Path:
    path = tmp_path / "predictions.parquet"
    frame.write_parquet(path)
    path.with_suffix(".json").write_text(
        json.dumps(_meta() if meta is None else meta), encoding="utf-8"
    )
    return path


def _load(path: Path, *, calibrated_inputs: bool = True) -> PairPredictions:
    return load_pair(
        path,
        timeframe="5m",
        target="mean_reversion",
        horizon=2,
        task="classification",
        expected_versions=VERSIONS,
        calibrated_inputs=calibrated_inputs,
    )


def test_valid_table_and_classification_input_selection(tmp_path: Path) -> None:
    path = _write(tmp_path, _frame())
    pair = _load(path)
    assert pair.n == 8
    assert pair.block_names == ["wf1", "wf2"]
    assert pair.model_names == [MODEL, OTHER]
    np.testing.assert_array_equal(pair.rows, np.arange(8))
    np.testing.assert_array_equal(pair.prediction(MODEL), np.full(8, 0.75))
    assert pair.covered([MODEL, OTHER]).all()

    raw_pair = _load(path, calibrated_inputs=False)
    np.testing.assert_array_equal(raw_pair.prediction(MODEL), np.full(8, 0.25))


@pytest.mark.parametrize(
    ("column", "value"),
    [("target", "future_return"), ("horizon", 3)],
)
def test_wrong_table_target_or_horizon_is_refused(
    tmp_path: Path, column: str, value: str | int
) -> None:
    frame = _frame()
    frame = frame.with_columns(pl.lit(value).cast(frame.schema[column]).alias(column))
    with pytest.raises(AlignmentError, match=f"column {column}"):
        _load(_write(tmp_path, frame))


@pytest.mark.parametrize(
    ("key", "value"),
    [("target", "future_return"), ("horizon", 3)],
)
def test_wrong_sidecar_target_or_horizon_is_refused(
    tmp_path: Path, key: str, value: str | int
) -> None:
    meta = _meta()
    meta[key] = value
    with pytest.raises(AlignmentError, match="sidecar target / horizon differ"):
        _load(_write(tmp_path, _frame(), meta))


@pytest.mark.parametrize("version_key", list(VERSIONS))
def test_stale_expected_version_is_refused(tmp_path: Path, version_key: str) -> None:
    path = _write(tmp_path, _frame())
    expected = {**VERSIONS, version_key: "new-version"}
    with pytest.raises(AlignmentError, match=version_key):
        load_pair(
            path,
            timeframe="5m",
            target="mean_reversion",
            horizon=2,
            task="classification",
            expected_versions=expected,
        )


@pytest.mark.parametrize("column", ["row", "timestamp"])
@pytest.mark.parametrize("disordered", [False, True])
def test_rows_and_timestamps_must_strictly_increase(
    tmp_path: Path, column: str, disordered: bool
) -> None:
    frame = _frame()
    values = frame[column].to_list()
    if disordered:
        values[1], values[2] = values[2], values[1]
    else:
        values[2] = values[1]
    frame = frame.with_columns(pl.Series(column, values, dtype=frame.schema[column]))
    message = "rows" if column == "row" else "timestamps"
    with pytest.raises(AlignmentError, match=f"{message} are not strictly increasing"):
        _load(_write(tmp_path, frame))


def test_non_contiguous_block_is_refused(tmp_path: Path) -> None:
    frame = _frame().with_columns(
        pl.Series("fold", ["wf1", "wf2", "wf1", "wf1", "wf2", "wf2", "wf2", "wf2"])
    )
    with pytest.raises(AlignmentError, match="block wf1 is not contiguous"):
        _load(_write(tmp_path, frame))


def test_column_without_sidecar_entry_is_refused(tmp_path: Path) -> None:
    frame = _frame().with_columns(pl.lit(0.5, dtype=pl.Float32).alias("ridge|extra"))
    with pytest.raises(AlignmentError, match="columns without a sidecar entry"):
        _load(_write(tmp_path, frame))


def test_missing_constant_is_refused(tmp_path: Path) -> None:
    with pytest.raises(AlignmentError, match="no constant baseline column"):
        _load(_write(tmp_path, _frame().drop(CONSTANT)))


@pytest.mark.parametrize("offset_days", [0, 1])
def test_reserved_boundary_and_later_rows_are_refused(
    tmp_path: Path, offset_days: int
) -> None:
    frame = _frame()
    timestamps = frame["timestamp"].to_list()
    timestamps[-1] = datetime(2022, 1, 1) + timedelta(days=offset_days)
    frame = frame.with_columns(
        pl.Series("timestamp", timestamps, dtype=pl.Datetime("us"))
    )
    path = _write(tmp_path, frame)
    with pytest.raises(AlignmentError, match="reserved start 2022-01-01"):
        load_pair(
            path,
            timeframe="5m",
            target="mean_reversion",
            horizon=2,
            task="classification",
            reserved_start=date(2022, 1, 1),
        )


def test_missing_inputs_remain_missing_and_coverage_counts_predictions(tmp_path: Path) -> None:
    frame = _frame().with_columns(
        pl.Series(
            f"{MODEL}|platt",
            [0.75, None, np.nan, 0.75, 0.75, 0.75, 0.75, 0.75],
            dtype=pl.Float32,
        ),
        pl.Series(
            f"{OTHER}|platt",
            [0.625, 0.625, 0.625, 0.625, None, 0.625, 0.625, 0.625],
            dtype=pl.Float32,
        ),
        pl.Series("label", [0, 1, 0, None, 0, 1, 0, 1], dtype=pl.Float32),
    )
    pair = _load(_write(tmp_path, frame))
    np.testing.assert_array_equal(
        pair.covered([MODEL, OTHER]),
        [True, False, False, False, False, True, True, True],
    )
    assert np.isnan(pair.prediction(MODEL)[1:3]).all()
    assert np.isnan(pair.prediction(OTHER)[4])
    assert np.isnan(pair.label[3])
    np.testing.assert_array_equal(
        pair.covered([MODEL], raw=True),
        [True, True, True, False, True, True, True, True],
    )

    coverage = ensemble_prediction_coverage(pair)
    by_key = {
        (record["model"], record["block"]): record
        for record in coverage.iter_rows(named=True)
    }
    expected = {
        (MODEL, "wf1"): 2,
        (MODEL, "wf2"): 4,
        (OTHER, "wf1"): 4,
        (OTHER, "wf2"): 3,
        (CONSTANT, "wf1"): 4,
        (CONSTANT, "wf2"): 4,
    }
    assert set(by_key) == set(expected)
    for key, count in expected.items():
        record = by_key[key]
        assert record["rows"] == 4
        assert record["with_prediction"] == count
        assert record["share"] == pytest.approx(count / 4)
        assert record["labelled"] == (3 if key[1] == "wf1" else 4)


@pytest.mark.parametrize(("embargo", "expected_rows"), [(0, [0, 1]), (1, [0]), (2, [])])
def test_history_excludes_unresolved_boundary_and_current_block(
    tmp_path: Path, embargo: int, expected_rows: list[int]
) -> None:
    pair = _load(_write(tmp_path, _frame()))
    positions = pair.history(1, embargo)
    np.testing.assert_array_equal(pair.rows[positions], expected_rows)
    assert (pair.block[positions] < 1).all()
    assert (pair.rows[positions] < 4 - (2 + embargo)).all()
    assert pair.history(0, embargo).size == 0
    assert pair.history(99, embargo).size == 0


def test_unit_predictions_respect_development_row_boundary(tmp_path: Path) -> None:
    units = tmp_path / "units"
    directory = units / "mean_reversion" / "h2" / "logistic_l2" / "standard" / "base"
    directory.mkdir(parents=True)
    pl.DataFrame(
        {
            "row": pl.Series([0, 4], dtype=pl.Int64),
            "prediction": pl.Series([0.25, 0.75], dtype=pl.Float32),
        }
    ).write_parquet(directory / "wf1.parquet")

    valid = load_unit_predictions(
        units, "mean_reversion", 2, "logistic_l2", "standard", development_rows=5
    )
    assert valid["row"].to_list() == [0, 4]
    assert valid["block"].to_list() == ["wf1", "wf1"]
    with pytest.raises(AlignmentError, match="row 4 is not a development row"):
        load_unit_predictions(
            units, "mean_reversion", 2, "logistic_l2", "standard", development_rows=4
        )
    missing = load_unit_predictions(
        units, "mean_reversion", 2, "ridge", "standard", development_rows=4
    )
    assert missing.shape == (0, 0)
