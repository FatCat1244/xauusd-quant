#!/usr/bin/env python
"""Profile the raw XAUUSD file(s) and write a JSON report.

Thin wrapper over ``xq inspect`` for people who prefer running a script.

    python scripts/inspect_data.py
    python scripts/inspect_data.py --path "D:/data/ticks.csv" --blocks 80
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xauusd_quant.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["inspect", *sys.argv[1:]]))
