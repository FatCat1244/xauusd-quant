"""Plumbing shared by the research studies: tables, JSON, provenance.

The spectral and wavelet layers write the same kinds of artefacts - a table as
Parquet and CSV, a JSON digest, a provenance block - so they share this code
rather than each keeping a private copy.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import polars as pl

from ..utils.paths import atomic_write_text, ensure_dir

__all__ = [
    "OutputSettings",
    "StudyWriter",
    "clean_json",
    "code_fingerprint",
    "git_info",
    "package_versions",
    "stack_tables",
]

_SHA = re.compile(r"^[0-9a-f]{40}$")


class OutputSettings(Protocol):
    """Which formats a study writes (any config section with these flags)."""

    @property
    def write_parquet(self) -> bool: ...

    @property
    def write_csv(self) -> bool: ...


class StudyWriter:
    """Writes one study's tables into its directory and records every file."""

    def __init__(self, directory: Path, output: OutputSettings, files: list[str]) -> None:
        self.directory = ensure_dir(directory)
        self.output = output
        self.files = files

    def table(self, frame: pl.DataFrame | None, name: str, *, csv: bool = True) -> None:
        if frame is None or frame.is_empty():
            return
        nulls = [c for c, d in frame.schema.items() if d == pl.Null]
        if nulls:
            frame = frame.with_columns(pl.col(nulls).cast(pl.Float64))
        if self.output.write_parquet:
            frame.write_parquet(self.directory / f"{name}.parquet")
            self.files.append(f"{name}.parquet")
        if self.output.write_csv and csv:
            frame.write_csv(self.directory / f"{name}.csv")
            self.files.append(f"{name}.csv")

    def json(self, payload: dict[str, Any], name: str) -> None:
        atomic_write_text(self.directory / f"{name}.json",
                          json.dumps(clean_json(payload), indent=2, default=str) + "\n")
        self.files.append(f"{name}.json")


def clean_json(value: Any) -> Any:
    """NumPy scalars to Python ones, and non-finite floats to None, recursively."""
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def stack_tables(parts: Iterable[pl.DataFrame | None]) -> pl.DataFrame:
    """Concatenate per-source tables, tolerating all-null (``Null``-typed) columns."""
    frames = [p for p in parts if isinstance(p, pl.DataFrame) and not p.is_empty()]
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def git_info(root: Path) -> dict[str, Any]:
    """The checked-out commit, or None - also when the repository has no commits yet.

    ``git rev-parse HEAD`` prints the literal ``HEAD`` in a repository without
    commits; only a real 40-hex id is recorded.
    """
    try:
        done = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                              text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return {"commit": None}
    commit = done.stdout.strip()
    return {"commit": commit if _SHA.match(commit) else None}


def code_fingerprint(paths: Iterable[Path]) -> str:
    """Digest of source files: a code version that works without git history."""
    digest = hashlib.blake2b(digest_size=8)
    for path in sorted(Path(p) for p in paths):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def package_versions(distributions: Iterable[str]) -> dict[str, str | None]:
    """Installed versions of the named distributions (None when absent)."""
    out: dict[str, str | None] = {}
    for name in distributions:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out
