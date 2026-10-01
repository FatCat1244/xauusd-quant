"""The research ledger: every hypothesis tested, and what came of it.

One row per (hypothesis, timeframe, input series, window, feature, target,
horizon), holding the definition, the measured result, a verdict against the
controls, when it was tested and on which dataset version. Failed hypotheses
stay in the ledger - a record of only the successes would be the
multiple-testing problem in its purest form.

The ledger is shared by every research layer; the hypothesis-id prefix says
which one (``SPEC-`` spectral, ``WAVE-`` wavelet, ``REG-`` regimes, ``ALPHA-``
predictive feature research). ``window`` is that layer's analysis window (the
FFT window, the rolling wavelet window). Ledgers written before it was shared
call the column ``fft_window``; they are read as ``window`` and rewritten so on
the next upsert.

Prompt #8 added optional columns, null for the earlier layers:
``preregistered`` (defined before any result was seen, or exploratory),
``test_family`` (the multiple-testing family), ``p_value`` / ``q_value``
(Benjamini-Hochberg within the family), ``conditioning``, ``feature_version``,
``source_feed`` and ``n_obs`` - so later stages can count trials and estimate
selection bias (deflated statistics, the probability of overfitting) without
re-deriving them.

Rows are upserted: re-testing a hypothesis on the same key replaces its row
(the date and dataset version say which run it came from). The ledger is a
Parquet file written atomically, with a CSV beside it for reading.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from ..utils.clock import utc_now_iso
from ..utils.paths import ensure_dir

__all__ = ["KEY", "ResearchLedger"]

#: Columns that identify one tested hypothesis instance.
KEY: tuple[str, ...] = ("hypothesis_id", "timeframe", "input_series", "window", "feature",
                        "target", "horizon")
_SCHEMA: dict[str, Any] = {
    "hypothesis_id": pl.Utf8, "definition": pl.Utf8, "timeframe": pl.Utf8,
    "input_series": pl.Utf8, "window": pl.Int64, "feature": pl.Utf8, "target": pl.Utf8,
    "horizon": pl.Int64, "metric": pl.Utf8, "value": pl.Float64, "control_value": pl.Float64,
    "verdict": pl.Utf8, "details": pl.Utf8, "date_tested": pl.Utf8,
    "dataset_version": pl.Utf8, "study": pl.Utf8,
    # Prompt #8 (null for earlier layers)
    "preregistered": pl.Boolean, "test_family": pl.Utf8, "p_value": pl.Float64,
    "q_value": pl.Float64, "conditioning": pl.Utf8, "feature_version": pl.Utf8,
    "source_feed": pl.Utf8, "n_obs": pl.Int64,
}
_LEGACY = {"fft_window": "window"}


class ResearchLedger:
    """A Parquet-backed, upserting table of tested hypotheses."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> pl.DataFrame:
        if not self.path.exists():
            return pl.DataFrame(schema=_SCHEMA)
        frame = pl.read_parquet(self.path)
        legacy = {old: new for old, new in _LEGACY.items()
                  if old in frame.columns and new not in frame.columns}
        return frame.rename(legacy) if legacy else frame

    def upsert(self, entries: list[dict[str, Any]], *,
               replace_by: tuple[str, ...] = KEY) -> int:
        """Add or replace *entries*; returns the ledger's row count afterwards.

        A new row replaces the old rows that share its *replace_by* columns - the
        full key by default (one id can hold many rows: features, windows,
        horizons). A layer whose ids are one configuration each replaces by
        ``("hypothesis_id",)``, so a configuration whose ``window`` changed
        between runs (the ML layer stores its block count there) keeps one row.
        """
        if not entries:
            return self.load().height
        stamp = utc_now_iso()
        rows = []
        for entry in entries:
            entry = {_LEGACY.get(k, k): v for k, v in entry.items()}
            row = {name: entry.get(name) for name in _SCHEMA}
            row["date_tested"] = row["date_tested"] or stamp
            if isinstance(entry.get("details"), (dict, list)):
                row["details"] = json.dumps(entry["details"], default=str, sort_keys=True)
            for key in ("feature", "target"):
                row[key] = row[key] or ""
            for key in ("window", "horizon"):
                value = row[key]
                row[key] = int(value) if value is not None else -1
            rows.append(row)
        new = pl.DataFrame(rows, schema=_SCHEMA)
        current = self.load()
        on = list(replace_by)
        merged = (pl.concat([current.join(new.select(on).unique(), on=on, how="anti"), new],
                            how="diagonal_relaxed")
                  .sort(list(KEY)))
        ensure_dir(self.path.parent)
        tmp = self.path.with_name(self.path.name + ".partial")
        merged.write_parquet(tmp)
        tmp.replace(self.path)
        merged.write_csv(self.path.with_suffix(".csv"))
        return merged.height
