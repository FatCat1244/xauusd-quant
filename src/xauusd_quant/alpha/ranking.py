r"""Multiple testing, test identity, feature statuses and manifests (Steps 54-59, 67-70, 75).

**Every test counts.** A test is one (timeframe, feature, target, horizon,
metric, condition). :class:`TestRegistry` gives each a permanent id
``ALPHA-H-nnnnnn`` the first time it is defined (append-only, never reused),
and records whether its definition existed before any result was seen
(``preregistered``) or was added afterwards (``exploratory``).

**False discoveries.** Benjamini-Hochberg q-values within each test family
(timeframe x metric x target kind x conditioning kind) and across all tests;
Bonferroni thresholds are reported as a conservative reference, never as the
only filter - effect size, stability and replication are read beside them.

**Statuses** are the conjunction of documented conditions (config
``classification``); the reason string of each feature says which held:

==================  ======================================================
``non_causal``      not live-safe (never computed)
``invalid``         undefined by construction at this timeframe
``failed_null``     no target kind where |rank IC| beats the circular-shift
                    null quantile *and* the FDR q-value is below the level
                    *and* the random-walk / shuffled pipeline nulls do not
                    match it (where they were computed)
``unstable``        beats the nulls pooled, but its yearly sign agrees with
                    the pooled sign in too few years, recent history reverses
                    it, or (direction only) the IC survives a 60-day
                    misalignment - a slow level, not a forecast
``redundant``       beats the nulls, but a cluster-mate (|rho| >= threshold)
                    has stronger evidence
``strong_candidate`` beats every null, |rank IC| >= strong floor, yearly sign
                    consistency >= strong share, recent sign kept
``candidate``       beats the nulls, |rank IC| >= floor, consistency >= share
``weak_candidate``  beats the nulls with a small or less consistent effect
==================  ======================================================

Nothing is called profitable: the statuses describe statistical information.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..utils.clock import utc_now_iso
from ..utils.paths import ensure_dir
from .config import ClassificationConfig

__all__ = [
    "STATUS_ORDER",
    "KindEvidence",
    "PIPELINE_REQUIRED",
    "TestRegistry",
    "ablation_manifests",
    "benjamini_hochberg",
    "classify",
    "passes_nulls",
    "status_rank",
]

STATUS_ORDER = ("strong_candidate", "candidate", "weak_candidate", "redundant", "unstable",
                "failed_null", "invalid", "non_causal")


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH q-values (monotone step-up); NaN p-values stay NaN and are not counted."""
    p = np.asarray(p, dtype=np.float64)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return q
    order = np.argsort(p[ok])
    ranked = p[ok][order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    vals = np.empty(m)
    vals[order] = np.minimum(ranked, 1.0)
    q[ok] = vals
    return q


class TestRegistry:
    """Append-only mapping test key -> ``<prefix>-nnnnnn`` (Steps 55-56).

    The prefix names the layer: ``ALPHA-H`` for the feature research (Prompt #8),
    ``SEL-H`` for the feature-selection experiments (Prompt #9).
    """

    __test__ = False                  # a registry of statistical tests, not a pytest class

    _SCHEMA = {"key": pl.Utf8, "test_id": pl.Utf8, "family": pl.Utf8,
               "preregistered": pl.Boolean, "first_defined_utc": pl.Utf8}

    def __init__(self, path: Path, *, prefix: str = "ALPHA-H") -> None:
        self.path = path
        self.prefix = prefix
        self._frame = (pl.read_parquet(path) if path.exists()
                       else pl.DataFrame(schema=self._SCHEMA))

    def assign(self, keys: list[str], families: list[str], *, preregistered: bool) -> list[str]:
        known = dict(zip(self._frame["key"].to_list(), self._frame["test_id"].to_list(),
                         strict=True))
        start = self._frame.height
        new_rows: list[dict[str, Any]] = []
        ids = []
        stamp = utc_now_iso()
        for key, family in zip(keys, families, strict=True):
            tid = known.get(key)
            if tid is None:
                tid = f"{self.prefix}-{start + len(new_rows) + 1:06d}"
                known[key] = tid
                new_rows.append({"key": key, "test_id": tid, "family": family,
                                 "preregistered": preregistered, "first_defined_utc": stamp})
            ids.append(tid)
        if new_rows:
            self._frame = pl.concat([self._frame, pl.DataFrame(new_rows, schema=self._SCHEMA)])
            ensure_dir(self.path.parent)
            tmp = self.path.with_name(self.path.name + ".partial")
            self._frame.write_parquet(tmp)
            tmp.replace(self.path)
        return ids

    def preregistered(self, ids: list[str]) -> list[bool]:
        lookup = dict(zip(self._frame["test_id"].to_list(),
                          self._frame["preregistered"].to_list(), strict=True))
        return [bool(lookup.get(i, False)) for i in ids]

    @property
    def size(self) -> int:
        return self._frame.height


@dataclass
class KindEvidence:
    """The best test of one feature for one target kind."""

    kind: str
    target: str | None
    rank_ic: float
    q_value: float
    beyond_shift_null: bool
    beyond_pipeline_null: bool | None
    sign_consistency: float
    recent_same_sign: bool | None
    era_sign_flips: int
    slow_component: bool | None = None       # |IC| survives a 60-day misalignment


def _finite(v: float | None) -> bool:
    return v is not None and bool(np.isfinite(v))


#: Target kinds whose evidence is uninterpretable without the pipeline nulls: a residual is
#: predictable by construction on a detrended random walk (invariant 9).
PIPELINE_REQUIRED = ("residual",)


def passes_nulls(e: KindEvidence, fdr_q: float) -> bool:
    """Beyond the circular-shift null, FDR-significant, not matched by a veto pipeline null.

    For residual targets the pipeline nulls must have been scored and beaten:
    "not scored" is not evidence there.
    """
    pipeline_ok = (e.beyond_pipeline_null is True if e.kind in PIPELINE_REQUIRED
                   else e.beyond_pipeline_null is not False)
    return (bool(e.beyond_shift_null) and _finite(e.q_value) and e.q_value < fdr_q
            and pipeline_ok)


def classify(feature: str, evidence: list[KindEvidence], *, live_safe: bool,
             invalid_reason: str | None, cluster_better: str | None,
             cfg: ClassificationConfig, fdr_q: float) -> tuple[str, str, str | None]:
    """(status, reason, best kind) of one feature from its per-kind evidence.

    Undefined statistics (None / NaN) never pass a condition: a kind without a
    finite rank IC and q-value is untested, not evidence.
    """
    if not live_safe:
        return "non_causal", f"{feature} is not live-safe - never computed", None
    if invalid_reason:
        return "invalid", invalid_reason, None
    evidence = [e for e in evidence if _finite(e.rank_ic)]
    passing = [e for e in evidence if passes_nulls(e, fdr_q)]
    if not passing:
        best = max(evidence, key=lambda e: abs(e.rank_ic), default=None)
        if best is None:
            return "failed_null", "no defined rank IC for any target kind", None
        detail = (f"no target kind beats the circular-shift null with FDR q < {fdr_q}; best "
                  f"|rank IC| {abs(best.rank_ic):.4f} ({best.target})")
        vetoed = [e for e in evidence if e.beyond_shift_null and _finite(e.q_value)
                  and e.q_value < fdr_q and e.beyond_pipeline_null is False]
        unscored = [e for e in evidence if e.beyond_shift_null and _finite(e.q_value)
                    and e.q_value < fdr_q and e.kind in PIPELINE_REQUIRED
                    and e.beyond_pipeline_null is None]
        if vetoed:
            v = max(vetoed, key=lambda e: abs(e.rank_ic))
            detail = (f"beats the shift null but is reproduced on the information-free "
                      f"pipeline nulls (random walk / shuffled / sign-flipped bars) - a "
                      f"mechanical effect of its construction ({v.target}, |rank IC| "
                      f"{abs(v.rank_ic):.4f})")
        elif unscored:
            v = max(unscored, key=lambda e: abs(e.rank_ic))
            detail = (f"only residual evidence ({v.target}, |rank IC| {abs(v.rank_ic):.4f}) and "
                      "the feature cannot be computed on the pipeline nulls - a residual claim "
                      "without the random-walk control is not interpretable (invariant 9)")
        return "failed_null", detail, None
    fast = [e for e in passing if not (e.kind == "direction" and e.slow_component)]
    if not fast:
        best = max(passing, key=lambda e: abs(e.rank_ic))
        return ("unstable", f"beats the nulls only through a slow level ({best.target}, rank IC "
                f"{best.rank_ic:+.4f}): the IC survives a 60-day misalignment - not a "
                "short-horizon forecast", best.kind)
    ranked = sorted(fast, key=lambda e: -abs(e.rank_ic))
    best = ranked[0]
    stable = [e for e in ranked if _finite(e.sign_consistency)
              and e.sign_consistency >= cfg.candidate_sign_years
              and e.recent_same_sign is not False]
    if not stable:
        share = (f"{best.sign_consistency:.0%}" if _finite(best.sign_consistency)
                 else "an undefined share")
        why = (f"beats the nulls ({best.kind}: {best.target}, rank IC {best.rank_ic:+.4f}, "
               f"q {best.q_value:.1e}) but the yearly sign agrees in only {share} of years"
               + ("" if best.recent_same_sign is not False else
                  " and recent history (2021+) has the opposite sign"))
        return "unstable", why, best.kind
    top = stable[0]
    if cluster_better is not None:
        return ("redundant", f"beats the nulls ({top.kind}: {top.target}) but {cluster_better} "
                f"carries the same information with stronger evidence", top.kind)
    effect = abs(top.rank_ic)
    pipeline = ("; beyond the random-walk pipeline nulls" if top.beyond_pipeline_null
                else "; not scored on the pipeline nulls" if top.beyond_pipeline_null is None
                else "")
    base = (f"{top.kind}: {top.target}, rank IC {top.rank_ic:+.4f}, q {top.q_value:.1e}, "
            f"yearly sign {top.sign_consistency:.0%}, eras flipping {top.era_sign_flips}"
            f"{pipeline}")
    if (effect >= cfg.strong_abs_rank_ic and top.sign_consistency >= cfg.strong_sign_years
            and top.recent_same_sign and top.era_sign_flips == 0
            and top.beyond_pipeline_null is not False):
        return "strong_candidate", base, top.kind
    if effect >= cfg.min_abs_rank_ic:
        return "candidate", base, top.kind
    return "weak_candidate", base + f" (|rank IC| below {cfg.min_abs_rank_ic})", top.kind


def status_rank(status: str) -> int:
    return STATUS_ORDER.index(status) if status in STATUS_ORDER else len(STATUS_ORDER)


def ablation_manifests(registry: list[dict[str, Any]], groups: dict[str, tuple[str, ...]],
                       *, default_only: bool) -> dict[str, list[str]]:
    """Cumulative family groups A-G as feature-id lists (default candidates only if asked)."""
    out: dict[str, list[str]] = {}
    for label, families in groups.items():
        out[label] = [r["feature_id"] for r in registry
                      if r.get("family") in families
                      and (not default_only or not r.get("excluded_from_default_candidates"))]
    return out
