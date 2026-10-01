#!/usr/bin/env python
"""Benchmark the rolling-regression layer: time and peak memory.

    python scripts/benchmark_regression.py                    # 5m, all windows
    python scripts/benchmark_regression.py -t 1m -w 128 512
    python scripts/benchmark_regression.py -t 1m --full-report

By default only the fit itself is timed, which is the part that has to scale.
``--full-report`` additionally times the whole analysis, which is dominated by
the ADF autolag search rather than by the regression.

Peak memory is the process working set, read from the OS, so NumPy's
allocations are included - ``tracemalloc`` would miss them.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl  # noqa: E402

from xauusd_quant import load_config  # noqa: E402
from xauusd_quant.features import load_regression_config  # noqa: E402
from xauusd_quant.features.rolling_regression import (  # noqa: E402
    rolling_regression_features,
)
from xauusd_quant.research.regression_reports import (  # noqa: E402
    generate_regression_report,
    load_bars,
)


class _Counters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.wintypes.DWORD),
        ("PageFaultCount", ctypes.wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


if sys.platform == "win32":
    # HANDLE is pointer-sized. Without these declarations ctypes passes the
    # GetCurrentProcess pseudo-handle (-1) as a 32-bit int, which arrives as
    # 0x00000000FFFFFFFF on x64 and every call fails with no error set.
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _psapi = ctypes.WinDLL("psapi", use_last_error=True)
    _kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    _kernel32.GetCurrentProcess.argtypes = []
    _psapi.GetProcessMemoryInfo.restype = ctypes.wintypes.BOOL
    _psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(_Counters), ctypes.wintypes.DWORD
    ]
    _psapi.EmptyWorkingSet.restype = ctypes.wintypes.BOOL
    _psapi.EmptyWorkingSet.argtypes = [ctypes.c_void_p]


def memory_mib() -> tuple[float, float]:
    """Current and peak working set in MiB, or (nan, nan) off Windows."""
    if sys.platform != "win32":
        return float("nan"), float("nan")
    counters = _Counters()
    counters.cb = ctypes.sizeof(_Counters)
    handle = _kernel32.GetCurrentProcess()
    if not _psapi.GetProcessMemoryInfo(
        handle, ctypes.byref(counters), counters.cb
    ):
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
    return (
        counters.WorkingSetSize / 2**20,
        counters.PeakWorkingSetSize / 2**20,
    )


def trim_working_set() -> None:
    """Trim the working set so the next measurement starts from a known floor.

    Windows exposes no way to reset ``PeakWorkingSetSize`` - it is a
    process-lifetime high-water mark - so the per-window figure reported here
    is the *growth* in resident memory across the fit, not a reset peak. The
    lifetime peak is reported separately, for context only.
    """
    if sys.platform == "win32":
        _psapi.EmptyWorkingSet(_kernel32.GetCurrentProcess())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeframe", "-t", default="5m")
    parser.add_argument("--window", "-w", type=int, nargs="*", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--full-report", action="store_true")
    args = parser.parse_args(argv)

    config = load_config()
    regression = load_regression_config()
    windows = args.window or list(regression.regression.windows)

    bars = load_bars(config, args.timeframe, args.start, args.end)
    baseline_now, _ = memory_mib()
    print(f"timeframe      : {args.timeframe}")
    print(f"bars           : {bars.height:,}")
    print(f"span           : {bars['timestamp'].min()} -> {bars['timestamp'].max()}")
    print(f"resident after loading bars: {baseline_now:,.0f} MiB")
    print()

    rows = []
    for window in windows:
        gc.collect()
        trim_working_set()
        before, _ = memory_mib()

        started = time.perf_counter()
        features, diagnostics = rolling_regression_features(
            bars, window=window, config=regression, timeframe=args.timeframe
        )
        fit_seconds = time.perf_counter() - started
        after, lifetime_peak = memory_mib()

        row = {
            "window": window,
            "bars": bars.height,
            "fits": diagnostics["valid_fits"],
            "fit_seconds": round(fit_seconds, 3),
            "bars_per_second": int(bars.height / fit_seconds) if fit_seconds else 0,
            "resident_before_mib": round(before, 1),
            "resident_after_mib": round(after, 1),
            "fit_growth_mib": round(after - before, 1),
            "process_peak_mib": round(lifetime_peak, 1),
            "feature_columns": features.width,
        }

        if args.full_report:
            gc.collect()
            started = time.perf_counter()
            report = generate_regression_report(
                config, regression, timeframe=args.timeframe, window=window,
                start=args.start, end=args.end, make_plots=False, bars=bars,
            )
            row["report_seconds"] = round(time.perf_counter() - started, 1)
            row["warnings"] = len(report.warnings)

        del features
        rows.append(row)
        print(f"  window {window:>4}: fit {fit_seconds:6.2f}s  "
              f"({row['bars_per_second']:>9,} bars/s)  "
              f"resident {before:,.0f} -> {after:,.0f} MiB "
              f"(+{after - before:,.0f})"
              + (f"  report {row['report_seconds']:.1f}s" if args.full_report else ""))

    print()
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(pl.DataFrame(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
