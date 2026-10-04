"""Retrieval occurrence prefixes, chronology and failure atomicity."""

from dataclasses import asdict
from datetime import timedelta

import pytest

from shadow_synth import T, row
from xauusd_quant.shadow.ingestion import ContinuityFailure, Cursor


def ingest(cursor: Cursor, rows: list[dict], limit: int = 10) -> list:
    return cursor.ingest(rows, T + timedelta(seconds=10), batch_limit=limit,
        live_start=T, max_future_seconds=0)


def test_same_timestamp_distinct_and_identical_occurrences_survive_restart() -> None:
    c = Cursor(int(T.timestamp()))
    first = [row(0), row(0), row(.1, bid=1801)]
    out = ingest(c, first)
    assert len(out) == 3 and [t.sequence for t in out] == [1, 2, 3]
    assert ingest(c, first) == []
    restored = Cursor.restore(asdict(c), 10)
    more = ingest(restored, [*first, row(.1, bid=1801), row(1)])
    assert [t.sequence for t in more] == [4, 5]
    assert more[0].time_msc == out[-1].time_msc
    assert ingest(restored, [row(1)]) == []


def test_changed_or_missing_overlap_blocks() -> None:
    c = Cursor(int(T.timestamp()))
    ingest(c, [row(0), row(.1)])
    saved = asdict(c)
    for rows in ([row(.1)], [row(0, bid=1790), row(.1)], [row(1)]):
        with pytest.raises(ContinuityFailure, match="prefix"):
            ingest(c, rows)
        assert asdict(c) == saved


@pytest.mark.parametrize("rows", [[row(.2), row(.1)], [row(0, bid=float("nan"))],
    [row(0, ask=1.)], [row(0, time_msc=-1)], [row(11)]])
def test_invalid_or_future_batch_never_advances(rows: list[dict]) -> None:
    c = Cursor(int(T.timestamp()))
    saved = asdict(c)
    with pytest.raises(ContinuityFailure):
        ingest(c, rows)
    assert asdict(c) == saved


def test_saturated_timestamp_and_empty_polls() -> None:
    c = Cursor(int(T.timestamp()))
    assert ingest(c, []) == []
    with pytest.raises(ContinuityFailure, match="saturated"):
        ingest(c, [row(0), row(0)], limit=2)
    assert c.sequence == 0


def test_corrupt_cursor_and_naive_receipt_rejected() -> None:
    c = Cursor(int(T.timestamp()))
    with pytest.raises(ValueError):
        Cursor.restore(asdict(c) | {"sequence": -1}, 10)
    with pytest.raises(ValueError):
        Cursor.restore(asdict(c) | {"boundary_hashes": ["a" * 64]}, 10)
    with pytest.raises(ContinuityFailure):
        c.ingest([row(0)], T.replace(tzinfo=None), batch_limit=10,
            live_start=T, max_future_seconds=0)
