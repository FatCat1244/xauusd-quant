#!/usr/bin/env python
"""Build OHLC bars from the converted Parquet dataset.

Thin wrapper over ``xq build-bars`` for people who prefer running a script.

    python scripts/build_bars.py --all
    python scripts/build_bars.py --timeframe 5m
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xauusd_quant.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["build-bars", *sys.argv[1:]]))
