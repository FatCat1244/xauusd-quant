"""Filesystem helpers.

Every path in this project flows through here so that relative paths in YAML
always mean "relative to the repository root", regardless of the working
directory the CLI happens to be invoked from.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

__all__ = [
    "PROJECT_ROOT_MARKERS",
    "atomic_write_bytes",
    "atomic_write_text",
    "dir_size_bytes",
    "ensure_dir",
    "find_project_root",
    "human_bytes",
    "resolve_path",
]

PROJECT_ROOT_MARKERS: tuple[str, ...] = ("pyproject.toml", ".git")


def find_project_root(start: Path | None = None) -> Path:
    """Walk upwards from *start* until a repository-root marker is found.

    Falls back to the current working directory when no marker exists, which
    keeps the package importable from a wheel installed outside a checkout.
    """
    origin = (start or Path(__file__)).resolve()
    for candidate in (origin, *origin.parents):
        if candidate.is_dir() and any((candidate / m).exists() for m in PROJECT_ROOT_MARKERS):
            return candidate
    return Path.cwd().resolve()


def resolve_path(value: str | os.PathLike[str], root: Path | None = None) -> Path:
    """Expand ``~``/environment variables and anchor relative paths at *root*."""
    path = Path(os.path.expandvars(str(value))).expanduser()
    if path.is_absolute():
        return path
    return ((root or find_project_root()) / path).resolve()


def ensure_dir(path: Path) -> Path:
    """Create *path* (and parents) if missing and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write *payload* to *path* via a temporary file plus ``os.replace``.

    An interrupted run can therefore never leave a half-written manifest or
    report behind, which matters because those files drive resume decisions.
    On Windows a file that was just written can be held open for a moment by
    a scanner or indexer, and the rename then fails with "Access is denied";
    the rename is retried for about 3 s before the error is raised.
    """
    ensure_dir(path.parent)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        tmp.write_bytes(payload)
        for attempt in range(12):
            try:
                tmp.replace(path)
                break
            except PermissionError:
                if attempt == 11:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Text counterpart of :func:`atomic_write_bytes`."""
    atomic_write_bytes(path, text.encode(encoding))


def dir_size_bytes(path: Path, pattern: str = "**/*") -> int:
    """Total size of every file under *path* matching *pattern*."""
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.glob(pattern) if f.is_file())


def human_bytes(n: float) -> str:
    """Format a byte count using binary units (1 KiB = 1024 B)."""
    step = 1024.0
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < step or unit == "TiB":
            return f"{n:,.2f} {unit}" if unit != "B" else f"{int(n):,} B"
        n /= step
    return f"{n:,.2f} TiB"  # pragma: no cover - unreachable
