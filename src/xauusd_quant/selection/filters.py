r"""The feature-quality filter with reason codes (Prompt #9, Steps 2, 5-8).

Every registered feature gets one row: kept or excluded, and *why*. Nothing
is discarded silently. Exclusion reasons, in order:

========================  ===============================================================
``EXCLUDE_NON_CAUSAL``    registered ``live_safe = false`` (never computed)
``EXCLUDE_INVALID``       undefined by construction at this timeframe (config ``invalid``)
``EXCLUDE_MISSINGNESS``   missing share above the configured limit in development,
                          validation or recent development (after the feature's warm-up)
``EXCLUDE_UNAVAILABLE``   missing in the reserved period's *feature values* (a feature
                          that cannot be computed recently cannot run live)
``EXCLUDE_NEAR_CONSTANT`` one value in more than ``max_dominant_fraction`` of bars, fewer
                          than ``min_unique_values`` values, or a binary flag whose
                          minority value is rarer than ``binary_min_minority``
``EXCLUDE_NUMERICAL``     non-finite values that are not missing, or |x| above the limit
``EXCLUDE_FAILED_NULL``   no target kind beats the development-period max-T null with
                          FDR q < level (decided later, by :mod:`.relevance`)
========================  ===============================================================

Drift is *classified*, never a reason on its own (``stable`` /
``moderately_drifting`` / ``strongly_drifting`` from the yearly PSI and median
shift within development and validation); Prompt #10 needs to know it.

All of this reads feature *values* only - no outcome enters the quality filter.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from ..features.quality import _proportions
from .config import FeatureSelectionConfig
from .data import SelectionData

__all__ = ["REASONS", "drift_class", "quality_filter"]

REASONS = ("EXCLUDE_NON_CAUSAL", "EXCLUDE_INVALID", "EXCLUDE_MISSINGNESS", "EXCLUDE_UNAVAILABLE",
           "EXCLUDE_NEAR_CONSTANT", "EXCLUDE_NUMERICAL", "EXCLUDE_FAILED_NULL")


def _missing_share(x: np.ndarray, lo: int, hi: int, warmup_end: int) -> float:
    lo = max(lo, warmup_end)
    if hi <= lo:
        return 1.0
    seg = x[lo:hi]
    return float(np.mean(~np.isfinite(seg)))


def drift_class(values: np.ndarray, years: np.ndarray, rows: np.ndarray, cfg: FeatureSelectionConfig
                ) -> tuple[str, float | None, float | None]:
    """(class, max yearly PSI, max |median shift| in IQR units) over the given rows."""
    x = values[rows].astype(np.float64)
    y = years[rows]
    ok = np.isfinite(x)
    if ok.sum() < 1000:
        return "undetermined", None, None
    ref = np.sort(x[ok])
    if ref.size > 50_000:
        ref = ref[np.linspace(0, ref.size - 1, 50_000).astype(np.int64)]
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, 11)))
    if edges.size < 3:
        return "stable", 0.0, 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    expected = _proportions(ref, edges)
    med = float(np.median(ref))
    iqr = float(np.quantile(ref, 0.75) - np.quantile(ref, 0.25)) or 1.0
    psi_max, shift_max = 0.0, 0.0
    for yr in np.unique(y[ok]):
        s = x[ok & (y == yr)]
        if s.size < 200:
            continue
        actual = _proportions(s, edges)
        psi_max = max(psi_max, float(((actual - expected) * np.log(actual / expected)).sum()))
        shift_max = max(shift_max, abs(float(np.median(s)) - med) / iqr)
    d = cfg.filters.drift
    if psi_max >= d["strong_psi"] or shift_max >= d["strong_median_shift_iqr"]:
        return "strongly_drifting", psi_max, shift_max
    if psi_max >= d["moderate_psi"]:
        return "moderately_drifting", psi_max, shift_max
    return "stable", psi_max, shift_max


def _near_constant(x: np.ndarray, cfg: FeatureSelectionConfig) -> tuple[bool, dict[str, Any]]:
    f = x[np.isfinite(x)]
    if f.size == 0:
        return True, {"unique_values": 0, "dominant_fraction": None, "minority_fraction": None}
    vc = pl.Series(f).value_counts(sort=True)
    unique = vc.height
    dominant = float(vc["count"][0] / f.size)
    minority = float(vc["count"][-1] / f.size) if unique == 2 else None
    nc = cfg.filters.near_constant
    if unique < nc["min_unique_values"]:
        return True, {"unique_values": unique, "dominant_fraction": dominant,
                      "minority_fraction": minority}
    if unique == 2:                        # a binary flag: rare events are allowed
        flagged = minority is not None and minority < nc["binary_min_minority"]
    else:
        flagged = dominant > nc["max_dominant_fraction"]
    return flagged, {"unique_values": unique, "dominant_fraction": dominant,
                     "minority_fraction": minority}


def quality_filter(data: SelectionData, cfg: FeatureSelectionConfig, *,
                   prompt8_status: dict[str, str] | None = None) -> pl.DataFrame:
    """One row per registered feature: metrics, drift class, reason codes, kept flag."""
    years = data.timestamps.dt.year().to_numpy()
    lim = cfg.filters.max_missing_fraction
    dev = data.rows["development"]
    val = data.rows["validation"]
    recent = data.rows["recent_development"]
    all_rows = np.arange(dev[0], val[1])
    rows = []
    for name in data.real_names():
        spec = data.registry[name]
        x = data.column(name)
        reasons: list[str] = []
        if not spec.get("live_safe", True):
            reasons.append("EXCLUDE_NON_CAUSAL")
        if spec.get("invalid_reason"):
            reasons.append("EXCLUDE_INVALID")
        warm = int(spec.get("min_history") or 0)
        first = int(np.argmax(np.isfinite(x))) if np.isfinite(x).any() else data.n
        warmup_end = max(warm, first)
        miss = {"development": _missing_share(x, *dev, warmup_end),
                "validation": _missing_share(x, *val, warmup_end),
                "recent_development": _missing_share(x, *recent, warmup_end)}
        reserved_missing = data.reserved_feature_missing.get(name)
        if any(miss[k] > lim.get(k, 1.0) for k in miss):
            reasons.append("EXCLUDE_MISSINGNESS")
        if reserved_missing is not None and np.isfinite(reserved_missing) and \
                reserved_missing > lim.get("validation", 1.0):
            reasons.append("EXCLUDE_UNAVAILABLE")
        flat, nc = _near_constant(x[dev[0]:val[1]], cfg)
        if flat:
            reasons.append("EXCLUDE_NEAR_CONSTANT")
        raw = x[dev[0]:val[1]]
        finite = raw[np.isfinite(raw)]
        n_inf = int(np.isinf(raw).sum())
        if n_inf or (finite.size and float(np.abs(finite).max()) > cfg.filters.numerical[
                "max_abs_value"]):
            reasons.append("EXCLUDE_NUMERICAL")
        cls, psi, shift = drift_class(x, years, all_rows, cfg)
        rows.append({
            "feature": name, "feature_id": spec.get("feature_id"), "family": spec.get("family"),
            "window": spec.get("window"), "parameter_family": spec.get("parameter_family"),
            "min_history": warm, "cost": spec.get("cost"), "live_safe": spec.get("live_safe"),
            "prior_status": spec.get("prior_status"),
            "prompt8_status": (prompt8_status or {}).get(name),
            "missing_development": miss["development"], "missing_validation": miss["validation"],
            "missing_recent_development": miss["recent_development"],
            "missing_reserved_values": reserved_missing, **nc,
            "infinite_values": n_inf, "drift_class": cls, "max_yearly_psi": psi,
            "max_median_shift_iqr": shift,
            "reasons": ",".join(reasons), "kept_quality": not reasons})
    return pl.DataFrame(rows, infer_schema_length=None)
