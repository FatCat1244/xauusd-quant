"""Timezone interpretation for naive source timestamps.

The raw XAUUSD export carries naive wall-clock timestamps with no zone
information. Guessing silently would be the single most damaging thing this
pipeline could do, so the rules here are:

* The canonical ``timestamp`` column is **never** rewritten. It is always the
  wall clock exactly as it appeared in the source file.
* ``timestamp_utc`` is a separate, optional, clearly-labelled derived column.
* The rule used to derive it comes from configuration and is recorded in every
  report, so a reader can always tell what was assumed.

The ``anchored_dst`` mode exists because broker "server time" commonly tracks a
*foreign* DST calendar. This dataset is UTC+2 in US winter and UTC+3 in US
summer, switching on US dates - a combination no IANA zone reproduces. Modelling
it as ``America/New_York + 7h`` is exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal, cast
from zoneinfo import ZoneInfo

import polars as pl

from ..utils.config import TimezoneConfig

__all__ = ["TimezoneResolution", "describe_policy", "to_utc_expr", "utc_offset_at"]


@dataclass(frozen=True, slots=True)
class TimezoneResolution:
    """What the pipeline assumed about the source timezone, for reporting."""

    mode: str
    source_label: str
    description: str
    emits_utc_column: bool
    determined: bool
    evidence: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "source_label": self.source_label,
            "description": self.description,
            "emits_utc_column": self.emits_utc_column,
            "determined": self.determined,
            "evidence": self.evidence,
        }


def describe_policy(cfg: TimezoneConfig, evidence: str | None = None) -> TimezoneResolution:
    """Summarise the configured policy for inclusion in JSON reports."""
    return TimezoneResolution(
        mode=cfg.mode,
        source_label=cfg.source_label,
        description=cfg.describe(),
        emits_utc_column=cfg.emit_utc_column,
        determined=cfg.mode != "naive",
        evidence=evidence,
    )


def to_utc_expr(cfg: TimezoneConfig, column: str = "timestamp") -> pl.Expr:
    """Polars expression converting a naive *column* to a UTC-aware column.

    Raises
    ------
    ValueError
        If the policy is ``naive``; there is nothing to convert to and the
        caller must not pretend otherwise.
    """
    if cfg.mode == "naive":
        raise ValueError(
            "timezone.mode is 'naive': the source zone is unknown, so a UTC "
            "column cannot be derived. Configure a timezone mode first."
        )

    expr = pl.col(column)
    # The config layer has already validated these against the same sets,
    # so narrowing here is a typing formality rather than a new check.
    ambiguous = cast(Literal["earliest", "latest", "raise", "null"], cfg.on_ambiguous)
    non_existent = cast(Literal["raise", "null"], cfg.on_non_existent)

    if cfg.mode == "fixed_offset":
        # local = UTC + offset  =>  UTC = local - offset
        return (
            expr.dt.offset_by(_hours_to_duration(-cfg.fixed_offset_hours))
            .dt.replace_time_zone("UTC")
            .alias("timestamp_utc")
        )

    if cfg.mode == "iana":
        return (
            expr.dt.replace_time_zone(
                cfg.iana_tz, ambiguous=ambiguous, non_existent=non_existent
            )
            .dt.convert_time_zone("UTC")
            .alias("timestamp_utc")
        )

    # anchored_dst: local = anchor wall clock + offset_hours.
    # Undo the offset to recover the anchor's wall clock, localise it in the
    # anchor zone (which supplies the correct DST offset for that instant), and
    # convert to UTC.
    return (
        expr.dt.offset_by(_hours_to_duration(-cfg.offset_hours))
        .dt.replace_time_zone(
            cfg.anchor_tz, ambiguous=ambiguous, non_existent=non_existent
        )
        .dt.convert_time_zone("UTC")
        .alias("timestamp_utc")
    )


def utc_offset_at(cfg: TimezoneConfig, when: datetime) -> float:
    """Offset in hours such that ``utc = local - offset`` at *when*.

    Provided for diagnostics and documentation. Prefer :func:`to_utc_expr` for
    bulk conversion - it stays inside Polars and never round-trips to Python.
    """
    if cfg.mode == "naive":
        raise ValueError("timezone.mode is 'naive': no offset is defined.")
    if cfg.mode == "fixed_offset":
        return cfg.fixed_offset_hours

    zone_name = cfg.iana_tz if cfg.mode == "iana" else cfg.anchor_tz
    if zone_name is None:  # pragma: no cover - the config layer forbids this
        raise ValueError("timezone.iana_tz must be set when mode == 'iana'")
    zone = ZoneInfo(zone_name)
    shift = 0.0 if cfg.mode == "iana" else cfg.offset_hours
    anchor_wall = when - timedelta(hours=shift)
    anchor_offset = anchor_wall.replace(tzinfo=zone).utcoffset()
    assert anchor_offset is not None
    # local = anchor_wall + shift and anchor_wall = utc + anchor_offset,
    # so utc = local - (shift + anchor_offset).
    return shift + anchor_offset.total_seconds() / 3600.0


def _hours_to_duration(hours: float) -> str:
    """Render a (possibly fractional, possibly negative) hour count for Polars."""
    total_seconds = round(hours * 3600)
    sign = "-" if total_seconds < 0 else ""
    return f"{sign}{abs(total_seconds)}s"
