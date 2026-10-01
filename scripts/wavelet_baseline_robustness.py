#!/usr/bin/env python
"""POST-HOC, NOT PRE-REGISTERED: does the wavelet increment survive a wider baseline?

    python scripts/wavelet_baseline_robustness.py                    # 5m-1h, every window
    python scripts/wavelet_baseline_robustness.py -t 15m -w 512 --no-ledger
    python scripts/wavelet_baseline_robustness.py --resume          # after a stop

WAVE-H-007 (registered) found small increments of model D (C + wavelet) over
model C, beyond the random-walk and bootstrap nulls, almost all on
volatility-type targets. Baseline A carries volatility over at most 50 bars,
while a window's wavelet energy is itself an N-bar realised volatility, and no
baseline block carries the daily volatility cycle. This refits the same
models, rows, folds and ridge with A and B widened three ways
(``wavelet_predictiveness.BASELINE_EXTENSIONS``):

    hour_only  A + hour-of-day indicators
    rv_only    A + ln RV over 64/256/1024 bars; B + ln mean |OU innovation| over the same
    rich       both

on the real series and on every incremental null (each with the same widened
baseline), and records WAVE-P-001 in the research ledger: one row per real
(timeframe, series, window, baseline, target, horizon), verdicts by the
WAVE-H-007 rule. The ``original`` baseline is refitted as a check - it must
reproduce each study's stored WAVE-H-007 numbers - and gets no ledger row.

The widened baselines were chosen after WAVE-H-007 was evaluated. They can
qualify a registered finding, never promote one.

Outputs (``<wavelet results>/post_hoc/``): ``baseline_robustness.{parquet,csv}``
(fold-mean scores and increments), ``baseline_robustness_folds.parquet`` and
``summary.json`` (lineage per timeframe, versions, verdict counts, the
reproduction check, runtime). Each timeframe is written to ``parts/`` as soon
as it finishes, so a stopped run resumes with ``--resume`` (a part is reused
only if its data versions, windows, configuration and code fingerprints still
match). One timeframe's real source plus one null is in memory at a time (peak
~2.3 GB at 5m); measured: 5m ~45 min, 15m ~40 min, 30m ~25 min for all four
windows.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from xauusd_quant import load_config  # noqa: E402
from xauusd_quant.features import load_regression_config  # noqa: E402
from xauusd_quant.features.spectral import rolling_spectrum  # noqa: E402
from xauusd_quant.features.spectral_config import load_spectral_config  # noqa: E402
from xauusd_quant.features.wavelet_causal import rolling_wavelet  # noqa: E402
from xauusd_quant.features.wavelet_config import load_wavelet_config  # noqa: E402
from xauusd_quant.models import load_ou_config  # noqa: E402
from xauusd_quant.research.research_ledger import ResearchLedger  # noqa: E402
from xauusd_quant.research.spectral_nulls import build_control, build_real_source  # noqa: E402
from xauusd_quant.research.study_io import (  # noqa: E402
    clean_json,
    code_fingerprint,
    git_info,
    package_versions,
)
from xauusd_quant.research.wavelet_predictiveness import (  # noqa: E402
    BASELINE_EXTENSIONS,
    incremental_study,
    incremental_summary,
)
from xauusd_quant.research.wavelet_reports import (  # noqa: E402
    POST_HOC_HYPOTHESES,
    integrity_gate,
    post_hoc_baseline_entries,
)
from xauusd_quant.utils.clock import utc_now_iso  # noqa: E402
from xauusd_quant.utils.paths import atomic_write_text, ensure_dir  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CAVEAT = ("POST-HOC - NOT PRE-REGISTERED. The widened baselines were chosen after WAVE-H-007 "
          "was evaluated; this can only qualify that registered result. Descriptive research "
          "on sampled rows (estimates): not a signal, not a strategy.")
KEY = ["timeframe", "input_series", "window", "source", "baseline", "target", "horizon",
       "metric"]


def _write(frame: pl.DataFrame, path: Path, csv: bool = True) -> None:
    tmp = path.with_name(path.name + ".partial")
    frame.write_parquet(tmp)
    tmp.replace(path)
    if csv:
        frame.write_csv(path.with_suffix(".csv"))


def _reproduction(results: Path, summary: pl.DataFrame) -> dict[str, Any]:
    """Largest difference between the refitted 'original' baseline and the stored studies."""
    worst, compared, missing = 0.0, 0, []
    original = summary.filter(pl.col("baseline") == "original")
    for (tf, series, window), part in original.group_by("timeframe", "input_series", "window"):
        path = results / str(tf) / str(series) / f"window_{window}" / "incremental_summary.parquet"
        if not path.exists():
            missing.append(f"{tf}/{series}/window_{window}")
            continue
        stored = pl.read_parquet(path).select("source", "target", "horizon", "metric",
                                             "C", "D", "delta_D_C")
        joined = part.join(stored, on=["source", "target", "horizon", "metric"],
                           suffix="_stored")
        compared += joined.height
        for col in ("C", "D", "delta_D_C"):
            diff = (joined[col] - joined[f"{col}_stored"]).abs().max()
            if isinstance(diff, (int, float)) and np.isfinite(diff):
                worst = max(worst, float(diff))
    return {"rows_compared": compared, "max_abs_difference": worst, "studies_missing": missing,
            "reproduces_wave_h_007": compared > 0 and worst < 1e-9 and not missing}


def _fingerprints(regression: Any, ou: Any, spectral: Any, wavelet: Any) -> dict[str, str]:
    """Everything a stored timeframe part depends on besides the data versions."""
    code = [Path(__file__).resolve(),
            *(ROOT / "src" / "xauusd_quant" / "research" / name for name in (
                "wavelet_predictiveness.py", "wavelet_reports.py", "spectral_nulls.py",
                "spectral_ic.py")),
            *(ROOT / "src" / "xauusd_quant" / "features" / name for name in (
                "wavelet.py", "wavelet_causal.py", "wavelet_energy.py", "spectral.py"))]
    return {"code": code_fingerprint(code), "wavelet": wavelet.fingerprint(),
            "wavelet_engine": wavelet.engine_fingerprint(), "spectral": spectral.fingerprint(),
            "regression": regression.fingerprint(), "ou": ou.fingerprint()}


def _run_timeframe(tf: str, windows: list[int], config: Any, regression: Any, ou: Any,
                   spectral: Any, wavelet: Any) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Every source, series, window and baseline of one timeframe."""
    t0 = time.perf_counter()
    summaries: list[pl.DataFrame] = []
    folds: list[pl.DataFrame] = []
    real = build_real_source(config, regression, ou, wavelet, tf)
    for name in ("real", *wavelet.incremental.controls):
        source = real if name == "real" else build_control(
            name, real, regression=regression, ou=ou, spectral=wavelet)
        for series in wavelet.input_series:
            values = source.series(series)
            for window in windows:
                result = rolling_wavelet(values, window=window, config=wavelet,
                                         bar_seconds=source.bar_seconds,
                                         missing_slots=source.missing_slots)
                spectrum = rolling_spectrum(values, fft_window=window, config=spectral,
                                            bar_seconds=source.bar_seconds,
                                            missing_slots=source.missing_slots)
                tag = [pl.lit(tf).alias("timeframe"), pl.lit(series).alias("input_series"),
                       pl.lit(window).alias("window")]
                for baseline in BASELINE_EXTENSIONS:
                    tables = incremental_study(result, spectrum, source, wavelet,
                                               source_name=name, baseline=baseline,
                                               per_feature=False)
                    if not tables:
                        continue
                    labels = [*tag, pl.lit(baseline).alias("baseline")]
                    summaries.append(incremental_summary(tables["linear"], tables["binary"])
                                     .with_columns(labels))
                    for kind in ("linear", "binary"):
                        if not tables[kind].is_empty():
                            folds.append(tables[kind].with_columns(
                                *labels, pl.lit(kind).alias("kind")))
                del result, spectrum
        if source is not real:
            del source
            gc.collect()
        print(f"{tf} {name}: {time.perf_counter() - t0:.0f}s", flush=True)
    del real
    gc.collect()
    return (pl.concat(summaries, how="diagonal_relaxed"),
            pl.concat(folds, how="diagonal_relaxed"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-t", "--timeframes", nargs="+", default=["5m", "15m", "30m", "1h"])
    parser.add_argument("-w", "--windows", nargs="+", type=int, default=None,
                        help="rolling wavelet windows (default: every configured window)")
    parser.add_argument("--resume", action="store_true",
                        help="reuse timeframes already finished on the same data versions, "
                             "windows, configuration and code (parts/<tf>.*)")
    parser.add_argument("--no-ledger", action="store_true", help="do not write WAVE-P-001 rows")
    args = parser.parse_args()

    config, regression, ou = load_config(), load_regression_config(), load_ou_config()
    spectral, wavelet = load_spectral_config(), load_wavelet_config()
    windows = args.windows or list(wavelet.causal_features.rolling_windows)
    out_dir = ensure_dir(wavelet.results_path / "post_hoc")
    parts = ensure_dir(out_dir / "parts")
    prints = _fingerprints(regression, ou, spectral, wavelet)
    started = time.perf_counter()
    summaries: list[pl.DataFrame] = []
    folds: list[pl.DataFrame] = []
    lineage: dict[str, Any] = {}
    seconds: dict[str, float | str] = {}
    for tf in args.timeframes:
        gate = integrity_gate(config, regression, ou, spectral, wavelet, tf)
        if gate["problems"]:
            print(f"{tf}: integrity gate refused: {gate['problems']}", file=sys.stderr)
            return 2
        lineage[tf] = gate["lineage"]
        stamp = json.loads(json.dumps(clean_json({
            "timeframe": tf, "windows": windows, "fingerprints": prints,
            "tick_dataset_version": lineage[tf].get("tick_dataset_version"),
            "bar_dataset_version": lineage[tf].get("bar_dataset_version")})))
        meta = parts / f"{tf}.json"
        if args.resume and meta.exists():
            recorded = json.loads(meta.read_text(encoding="utf-8"))
            if {k: recorded.get(k) for k in stamp} == stamp:
                summaries.append(pl.read_parquet(parts / f"{tf}.parquet"))
                folds.append(pl.read_parquet(parts / f"{tf}_folds.parquet"))
                seconds[tf] = f"reused (computed in {recorded.get('seconds', 0):.0f}s)"
                print(f"{tf}: reused the stored part", flush=True)
                continue
            print(f"{tf}: stored part is stale; recomputing", flush=True)
        t0 = time.perf_counter()
        summary_tf, folds_tf = _run_timeframe(tf, windows, config, regression, ou, spectral,
                                              wavelet)
        seconds[tf] = time.perf_counter() - t0
        _write(summary_tf, parts / f"{tf}.parquet", csv=False)      # data first,
        _write(folds_tf, parts / f"{tf}_folds.parquet", csv=False)
        atomic_write_text(meta, json.dumps({**stamp, "seconds": seconds[tf],     # then the stamp
                                            "generated_utc": utc_now_iso()}, indent=2) + "\n")
        summaries.append(summary_tf)
        folds.append(folds_tf)

    summary = pl.concat(summaries, how="diagonal_relaxed").select(
        *KEY, pl.exclude(KEY)).sort(KEY)
    _write(summary, out_dir / "baseline_robustness.parquet")
    _write(pl.concat(folds, how="diagonal_relaxed"),
           out_dir / "baseline_robustness_folds.parquet", csv=False)
    check = _reproduction(wavelet.results_path, summary)

    entries: list[dict[str, Any]] = []
    for (tf, series, window), part in summary.group_by("timeframe", "input_series", "window",
                                                      maintain_order=True):
        entries += post_hoc_baseline_entries(
            part, timeframe=str(tf), series=str(series), window=int(str(window)),
            dataset_version=lineage[str(tf)].get("tick_dataset_version"),
            study=f"post_hoc/baseline_robustness/{tf}/{series}/window_{window}")
    verdicts = (pl.DataFrame([{"baseline": e["feature"].split("=", 1)[1],
                               "verdict": e["verdict"]} for e in entries],
                             schema={"baseline": pl.Utf8, "verdict": pl.Utf8})
                .group_by("baseline", "verdict").len().sort("baseline", "verdict"))
    if not args.no_ledger:
        ResearchLedger(wavelet.ledger_path).upsert(entries)
    payload = {
        "caveat": CAVEAT, "hypotheses": POST_HOC_HYPOTHESES,
        "baselines": BASELINE_EXTENSIONS, "timeframes": args.timeframes, "windows": windows,
        "sources": ["real", *wavelet.incremental.controls],
        "series": list(wavelet.input_series),
        "verdict_counts": verdicts.to_dicts(), "ledger_rows": len(entries),
        "ledger_written": not args.no_ledger, "original_baseline_check": check,
        "provenance": {
            "generated_utc": utc_now_iso(), "dataset_lineage": lineage,
            "fingerprints": prints, "git": git_info(ROOT),
            "packages": package_versions(("PyWavelets", "numpy", "scipy", "polars")),
            "sampling": {"incremental_rows": wavelet.incremental.max_rows,
                         "note": "fold scores are on evenly spaced sampled rows: estimates"},
            "seconds": {**seconds, "total": time.perf_counter() - started},
        },
    }
    atomic_write_text(out_dir / "summary.json",
                      json.dumps(clean_json(payload), indent=2, default=str) + "\n")
    print(verdicts)
    print("original baseline reproduces WAVE-H-007:", check["reproduces_wave_h_007"],
          f"(max |diff| {check['max_abs_difference']:.2e} over {check['rows_compared']} rows)")
    return 0 if check["reproduces_wave_h_007"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
