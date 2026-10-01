"""Logging setup.

Policy for this project: log progress and aggregates, never per-tick detail.
A full conversion of the ~34 GB dataset emits on the order of a few hundred
lines. Anything that would scale with row count belongs in a report file, not
in the log.
"""

from __future__ import annotations

import logging
import logging.config
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

from .paths import ensure_dir, find_project_root, resolve_path

__all__ = ["JOB_LOGGER", "get_logger", "log_duration", "setup_logging"]

JOB_LOGGER = "xauusd_quant"
DEFAULT_LOGGING_RELPATH = "config/logging.yaml"

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the project namespace."""
    if name.startswith(JOB_LOGGER):
        return logging.getLogger(name)
    return logging.getLogger(f"{JOB_LOGGER}.{name}")


def setup_logging(
    config_path: str | os.PathLike[str] | None = None,
    *,
    level: str | None = None,
    log_file: str | os.PathLike[str] | None = None,
    root: Path | None = None,
    force: bool = False,
) -> None:
    """Configure logging from ``config/logging.yaml``.

    Falls back to a plain console configuration when the YAML file is missing
    or unusable, so logging never becomes the reason a pipeline run fails.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    root = root or find_project_root()
    path = resolve_path(config_path or DEFAULT_LOGGING_RELPATH, root)
    level = (level or os.environ.get("XAUUSD_LOG_LEVEL") or "INFO").upper()

    applied = False
    if path.exists():
        try:
            spec: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            _prepare_file_handlers(spec, root, log_file)
            logging.config.dictConfig(spec)
            applied = True
        except Exception as exc:  # noqa: BLE001 - logging must never be fatal
            logging.basicConfig(level=level, format="%(asctime)s | %(levelname)-8s | %(message)s")
            logging.getLogger(JOB_LOGGER).warning(
                "Falling back to basic logging; could not apply %s: %s", path, exc
            )

    if not applied:
        logging.basicConfig(level=level, format="%(asctime)s | %(levelname)-8s | %(message)s")

    # The package logger stays at DEBUG so the rotating file handler keeps the
    # full trace; `level` only controls how chatty the console is.
    console_level = getattr(logging, level, logging.INFO)
    pkg_logger = logging.getLogger(JOB_LOGGER)
    pkg_logger.setLevel(logging.DEBUG)
    for handler in pkg_logger.handlers:
        if handler.__class__.__name__ in {"RichHandler", "StreamHandler"}:
            handler.setLevel(console_level)
    _CONFIGURED = True


def _prepare_file_handlers(
    spec: dict[str, Any], root: Path, log_file: str | os.PathLike[str] | None
) -> None:
    """Anchor file-handler paths at the repo root and create their directories."""
    override = log_file or os.environ.get("XAUUSD_LOG_FILE")
    for handler in (spec.get("handlers") or {}).values():
        if not isinstance(handler, dict) or "filename" not in handler:
            continue
        target = resolve_path(override or handler["filename"], root)
        ensure_dir(target.parent)
        handler["filename"] = str(target)


@contextmanager
def log_duration(logger: logging.Logger, what: str, level: int = logging.INFO) -> Iterator[None]:
    """Log the start and wall-clock duration of a long-running operation."""
    logger.log(level, "%s: start", what)
    started = time.perf_counter()
    try:
        yield
    except BaseException as exc:
        logger.error("%s: FAILED after %.1fs (%s: %s)",
                     what, time.perf_counter() - started, type(exc).__name__, exc)
        raise
    else:
        logger.log(level, "%s: done in %s", what, format_duration(time.perf_counter() - started))


def format_duration(seconds: float) -> str:
    """Compact human-readable duration, e.g. ``1h 03m 12s``."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"
