"""xauusd-quant: quantitative research platform for XAUUSD.

Current scope is the data foundation only: inspection, validation, cleaning,
Parquet storage, time bars and query helpers. Strategy, signal, feature,
machine-learning and backtesting code is deliberately absent - see the README.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .data.schema import SCHEMA_VERSION
from .utils.config import Config, load_config
from .utils.logging import get_logger, setup_logging

__all__ = [
    "SCHEMA_VERSION",
    "Config",
    "__version__",
    "get_logger",
    "load_config",
    "setup_logging",
]
