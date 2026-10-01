r"""Live reconstruction: bars arriving one at a time (Prompt #9, Steps 53-54, 60).

For a selected set, historical bars are replayed as if they arrived live: at
each step the bar is appended to a rolling buffer, every needed family is
recomputed from the buffer alone, and the newest row is compared with the
stored batch matrix. A buffer never holds a bar after the one being produced,
so agreement proves the stored value used no later bar.

The buffer is ``2 x (longest warm-up) + convergence`` bars: rolling windows are
then complete, and the EWMA recursions (seeded at the buffer start) have
forgotten their seed (``0.99^3000 ~ 1e-13``). Values agree to float32
rounding, not bit for bit: running sums, the OU fits' block-shifted prefix
sums and centred moments start from a different first bar.

Regime features are not recomputed here: they come from the Prompt #7
walk-forward filter (a model refitted each quarter off the live path, then a
forward filter per bar), whose prefix invariance Prompt #7's tests pin bit for
bit; the manifest records that.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import numpy as np
import polars as pl

from ..features.factory import compute_families
from ..features.families import BarSeries
from ..features.interactions import interaction_columns
from ..features.joins import EngineProvider

__all__ = ["live_reconstruction", "required_buffer"]

_CONVERGENCE = 3000


def required_buffer(registry: dict[str, dict[str, Any]], names: list[str],
                    percentile_window: int) -> int:
    longest = max([int(registry[n].get("min_history") or 0) for n in names] + [percentile_window])
    return 2 * longest + _CONVERGENCE


def _slice(bs: BarSeries, lo: int, hi: int) -> BarSeries:
    def cut(v: np.ndarray | None) -> np.ndarray | None:
        return None if v is None else v[lo:hi]

    return replace(bs, name=f"{bs.name}_live", timestamps=bs.timestamps.slice(lo, hi - lo),
                   close=bs.close[lo:hi], open=cut(bs.open), high=cut(bs.high), low=cut(bs.low),
                   median_spread=cut(bs.median_spread), tick_count=cut(bs.tick_count),
                   missing_slots=cut(bs.missing_slots), notes=[])


def live_reconstruction(bars: BarSeries, stored: pl.DataFrame, names: list[str],
                        registry: dict[str, dict[str, Any]], *, fcfg: Any, ou: Any,
                        research: Any, regression: Any, spectral: Any, wavelet: Any,
                        start: int, steps: int, buffer: int, stride: int = 1,
                        rtol: float = 2e-5, atol: float = 1e-6) -> dict[str, Any]:
    """Replay *steps* bars from row *start*; compare each newest row with the stored matrix."""
    regime_ix = {ix.name for ix in fcfg.interactions
                 if "regime" in ((registry.get(ix.a) or {}).get("family"),
                                 (registry.get(ix.b) or {}).get("family"))}
    live_names = [n for n in names if registry[n].get("family") != "regime" and n not in regime_ix]
    families = tuple(sorted({str(registry[n].get("family")) for n in live_names} - {"interaction"}))
    needs_ix = any(registry[n].get("family") == "interaction" for n in live_names)
    if needs_ix:
        families = tuple(sorted(set(families) | {"returns", "volatility", "regression", "ou",
                                                 "fft", "wavelet", "microstructure", "time"}))
    if "ou" in families or "fft" in families:
        families = tuple(sorted(set(families) | {"regression"}))
    if any(n.startswith("fft_abs_eta") for n in live_names):
        families = tuple(sorted(set(families) | {"ou"}))     # |OU innovation| feeds the FFT
    order = ("returns", "volatility", "autocorrelation", "regression", "ou", "fft", "wavelet",
             "microstructure", "time")
    families = tuple(f for f in order if f in families)
    if not stored["timestamp"].equals(bars.timestamps):
        raise ValueError("the stored matrix and the bars are not on the same timestamps")
    rows = list(range(start, min(bars.size, start + steps * stride), stride))
    picked = stored.select(live_names)[rows]            # only the replayed rows
    batch_values = {n: picked[n].cast(pl.Float64).fill_null(np.nan).to_numpy()
                    for n in live_names}
    worst = dict.fromkeys(live_names, 0.0)
    mismatches: dict[str, int] = dict.fromkeys(live_names, 0)
    missing_disagree: dict[str, int] = dict.fromkeys(live_names, 0)
    started = time.perf_counter()
    for k, t in enumerate(rows):
        lo = max(0, t + 1 - buffer)
        window = _slice(bars, lo, t + 1)
        provider = EngineProvider(bars=window, regression_config=regression,
                                  spectral_config=spectral, wavelet_config=wavelet)
        values, _ = compute_families(window, provider, fcfg, ou, research, families=families,
                                     log=False, collect=False)
        if needs_ix:
            values.update(interaction_columns(values, fcfg, window.bar_seconds))
        for n in live_names:
            live = float(np.float32(values[n][-1]))
            batch = float(batch_values[n][k])
            if np.isnan(live) != np.isnan(batch):
                missing_disagree[n] += 1
                continue
            if np.isnan(live):
                continue
            diff = abs(live - batch)
            worst[n] = max(worst[n], diff)
            if diff > atol + rtol * abs(batch):
                mismatches[n] += 1
    seconds = time.perf_counter() - started
    failed = sorted(n for n in live_names if mismatches[n] or missing_disagree[n])
    return {"steps": len(rows), "start_row": start, "stride": stride, "buffer_bars": buffer,
            "families": list(families), "features_checked": live_names,
            "not_recomputed": [n for n in names if n not in live_names],
            "max_abs_difference": worst, "mismatches": mismatches,
            "missing_disagreements": missing_disagree, "failed": failed,
            "passed": not failed, "seconds": seconds,
            "seconds_per_bar": seconds / max(1, len(rows)),
            "tolerance": {"rtol": rtol, "atol": atol}}
