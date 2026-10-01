#!/usr/bin/env python
"""Stream the raw tick file(s) into partitioned Parquet.

Thin wrapper over ``xq convert`` for people who prefer running a script.

    python scripts/convert_to_parquet.py
    python scripts/convert_to_parquet.py --limit-rows 2000000 --no-resume
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xauusd_quant.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["convert", *sys.argv[1:]]))
