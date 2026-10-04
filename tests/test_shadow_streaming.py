"""Canonical bars/features, prefix and restart equality, and availability guards."""

import math
from dataclasses import replace
from datetime import timedelta

import numpy as np
import polars as pl
import pytest

from shadow_synth import ROOT, T, feed, tick, trend
from xauusd_quant.data.resampler import resample_ticks
from xauusd_quant.execution.engine import MemoryRecorder
from xauusd_quant.features.factory_config import load_features_config
from xauusd_quant.shadow.bars import Bars, serialize_bar
from xauusd_quant.shadow.config import ShadowConfig
from xauusd_quant.shadow.features import Features
from xauusd_quant.shadow.ingestion import ContinuityFailure
from xauusd_quant.shadow.pipeline import ShadowPipeline


def test_boundaries_match_canonical_without_flat_bars() -> None:
    cfg = ShadowConfig(timeframes=("1m", "5m"), max_gap_seconds=300)
    inputs = [tick(0, 1, bid=1800), tick(59.999, 2, bid=1801), tick(60, 3, bid=1900)]
    bars = Bars(cfg)
    assert bars.consume(inputs[0]) == []
    assert bars.consume(inputs[1]) == []
    out = bars.consume(inputs[2])
    assert len(out) == 1 and out[0].timeframe == "1m"
    assert out[0].values["tick_count"] == 2
    assert out[0].values["close"] == pytest.approx(1801.1)
    assert out[0].available_utc == inputs[2].received_utc
    frame = pl.DataFrame({"timestamp": [t.event_utc for t in inputs],
        "bid": [t.bid for t in inputs], "ask": [t.ask for t in inputs],
        "mid": [(t.bid + t.ask) / 2 for t in inputs], "spread": [t.ask - t.bid for t in inputs]})
    canonical = resample_ticks(frame, "1m")
    assert out[0].values == canonical.row(0, named=True)
    assert bars.buffers["5m"] == inputs


def test_lateness_and_late_corrections_never_revise_decisions() -> None:
    b = Bars(ShadowConfig(timeframes=("1m",), lateness_seconds=2, max_gap_seconds=300))
    b.consume(tick(0, 1))
    assert b.consume(tick(60, 2)) == []
    out = b.consume(tick(62, 3))
    saved = serialize_bar(out[0])
    late = replace(tick(30, 4), received_utc=T + timedelta(seconds=63))
    with pytest.raises(ContinuityFailure, match="late"):
        b.consume(late)
    assert serialize_bar(out[0]) == saved and "LATE_TICK_AFTER_EMISSION" in b.halts


def test_gap_partial_start_and_overload_block() -> None:
    b = Bars(ShadowConfig(timeframes=("1m",)))
    b.consume(tick(5, 1))
    out = b.consume(tick(60, 2))
    assert not out[0].valid
    assert set(out[0].reasons) == {"PARTIAL_STARTUP_BAR", "GAP_AFFECTED_BAR"}
    assert "GAP_CONTINUITY_UNVERIFIED" in b.halts
    b = Bars(ShadowConfig(timeframes=("1m",), max_buffer_ticks=2))
    b.consume(tick(0, 1))
    b.consume(tick(1, 2))
    with pytest.raises(ContinuityFailure, match="overloaded"):
        b.consume(tick(2, 3))
    assert len(b.buffers["1m"]) == 2


def test_bar_restart_and_feature_warmup_known_formula() -> None:
    cfg = ShadowConfig(timeframes=("1m",), max_gap_seconds=300)
    b = Bars(cfg)
    b.consume(tick(0, 1, bid=1800))
    b.consume(tick(59, 2, bid=1801))
    restored = Bars.restore(cfg, b.state())
    first = b.consume(tick(60, 3, bid=1802))[0]
    assert serialize_bar(first) == serialize_bar(restored.consume(tick(60, 3, bid=1802))[0])
    second = b.consume(tick(120, 4, bid=1803))[0]
    features = Features(load_features_config(root=ROOT), "1m")
    assert features.history_required == 2
    assert not features.update(first)["ready"]
    state = features.state()
    actual = features.update(second)
    f2 = Features(load_features_config(root=ROOT), "1m")
    f2.restore(state)
    assert actual == f2.update(second)
    assert actual["ready"]
    assert actual["values"]["ret_1"] == float(np.float32(math.log(1802.1) - math.log(1801.1)))


def test_unsupported_context_and_incompatible_feature_state() -> None:
    cfg = load_features_config(root=ROOT)
    with pytest.raises(ValueError):
        Features(cfg, "5m", ("regime_confidence",))
    f = Features(cfg, "5m")
    with pytest.raises(ValueError, match="incompatible"):
        f.restore({"identity": "other", "history": []})


def test_prefix_chunk_and_reload_equality() -> None:
    cfg = ShadowConfig(timeframes=("1m",))
    fcfg = load_features_config(root=ROOT)
    full, prefix, chunks = MemoryRecorder(), MemoryRecorder(), MemoryRecorder()
    p = ShadowPipeline(cfg, fcfg, full)
    a = ShadowPipeline(cfg, fcfg, prefix)
    b = ShadowPipeline(cfg, fcfg, chunks)
    inputs = trend()
    feed(p, inputs)
    feed(a, inputs[:14])
    feed(b, inputs[:14])
    state = b.state()
    restored = ShadowPipeline(cfg, fcfg, chunks)
    restored.restore(state)
    feed(restored, inputs[14:])
    assert restored.state() == p.state()
    assert chunks.tables == full.tables
    for table, records in prefix.tables.items():
        assert full.tables[table][:len(records)] == records
    assert not full.tables.get("predictions") and not full.tables.get("fills")
    assert all(r["action"] == "NO_ACTION" for r in full.tables["shadow_actions"])


def test_backfill_never_creates_live_actions_and_early_publication_rejected() -> None:
    cfg = ShadowConfig(timeframes=("1m",))
    sink = MemoryRecorder()
    p = ShadowPipeline(cfg, load_features_config(root=ROOT), sink)
    inputs = [replace(t, backfill=True) for t in trend(15)]
    feed(p, inputs)
    assert all("BACKFILL_NO_RETROSPECTIVE_ACTION" in r["reasons"] for r in sink.tables["shadow_actions"])
    assert not sink.tables.get("orders")
    fresh = ShadowPipeline(cfg, load_features_config(root=ROOT), MemoryRecorder())
    feed(fresh, trend(6))  # No prior publication: isolates the input-availability guard.
    with pytest.raises(ContinuityFailure, match="publication"):
        fresh.consume([tick(60, 7)], lambda: T)


def test_future_receipt_and_incompatible_pipeline_checkpoint() -> None:
    cfg = ShadowConfig(timeframes=("1m",))
    p = ShadowPipeline(cfg, load_features_config(root=ROOT), MemoryRecorder())
    feed(p, trend(8))
    with pytest.raises(ValueError, match="incompatible"):
        p.restore(p.state() | {"identity": "other"})
    with pytest.raises(ContinuityFailure, match="receipt"):
        p.consume([replace(tick(70, 9), received_utc=T)])
