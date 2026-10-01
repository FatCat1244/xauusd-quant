"""Dataset lineage: which raw file, tick dataset and bars a result came from.

Every research artefact embeds :func:`dataset_lineage`, so any number can be
traced to the exact data behind it, and a result computed from an incomplete
tick dataset is labelled as such in its own provenance rather than depending
on someone remembering. The chain is::

    raw file (blake2b fingerprint)
      -> tick dataset  (ticks-<hash>: content digests of every partition)
        -> bars        (bars-<tf>-<hash>: built from that tick version)
          -> features  (regfeat-<tf>-w<N>-<hash>: built from those bars)

Each link records the version of the one before it, so a mismatch anywhere is
detectable.
"""

from __future__ import annotations

from typing import Any

from ..utils.config import Config
from .converter import load_dataset_manifest
from .resampler import load_bar_manifest

__all__ = ["PARTIAL_LABEL", "dataset_lineage"]

#: Stamped into the provenance of anything computed from an incomplete dataset.
PARTIAL_LABEL = "PARTIAL DEVELOPMENT SAMPLE — NOT FULL RESEARCH RESULT"


def dataset_lineage(config: Config, timeframe: str | None = None) -> dict[str, Any]:
    """Versions and coverage of the data a result is computed from."""
    ticks = load_dataset_manifest(config.processed_data_path) or {}
    lineage: dict[str, Any] = {
        "raw_fingerprint": ticks.get("raw_fingerprint"),
        "raw_files": ticks.get("source_files"),
        "tick_dataset_version": ticks.get("dataset_version"),
        "tick_dataset_status": ticks.get("status", "unknown (no manifest)"),
        "tick_first_timestamp": ticks.get("first_timestamp"),
        "tick_last_timestamp": ticks.get("last_timestamp"),
        "tick_rows": ticks.get("total_rows"),
        "tick_partitions": ticks.get("partition_count"),
    }
    if timeframe is not None:
        bars = load_bar_manifest(config.bars_dir(timeframe)) or {}
        lineage.update({
            "timeframe": timeframe,
            "bar_dataset_version": bars.get("bar_dataset_version"),
            "bar_built_from_tick_version": bars.get("tick_dataset_version"),
            "bar_rows": bars.get("rows"),
            "bar_first_timestamp": bars.get("first_timestamp"),
            "bar_last_timestamp": bars.get("last_timestamp"),
        })
        lineage["bars_match_ticks"] = (
            bool(bars) and bars.get("tick_dataset_version") == ticks.get("dataset_version")
        )
    partial = ticks.get("status") != "complete" or (
        timeframe is not None and not lineage.get("bars_match_ticks")
    )
    lineage["partial"] = partial
    lineage["label"] = PARTIAL_LABEL if partial else None
    return lineage
