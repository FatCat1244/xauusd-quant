"""UTC timestamps for reports.

Every report the pipeline writes is stamped through here, so the format is
identical everywhere and the deprecated naive ``datetime.utcnow()`` never
appears in the codebase.

The ISO strings are deliberately *naive* (no ``+00:00`` suffix) but always
denote UTC. They label when a report was produced; they are never mixed with
market timestamps, which live in broker-local time.
"""

from __future__ import annotations

from datetime import UTC, datetime

__all__ = ["utc_now", "utc_now_iso", "utc_from_timestamp_iso"]


def utc_now() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


def utc_now_iso() -> str:
    """Current UTC time as ``YYYY-MM-DDTHH:MM:SS``."""
    return datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="seconds")


def utc_from_timestamp_iso(epoch_seconds: float) -> str:
    """Render a POSIX timestamp (e.g. ``st_mtime``) as a UTC ISO string."""
    return (
        datetime.fromtimestamp(epoch_seconds, UTC)
        .replace(tzinfo=None)
        .isoformat(timespec="seconds")
    )
