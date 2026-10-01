r"""Outcome evidence inside one chronological span (Prompt #9, Steps 5, 11, 41-43).

Everything here is computed from the rows of a *training span* only - the
development period, a nested training fold or a block resample - so that
feature selection stays inside chronological training methodology:

* rank transforms inside the span (:func:`~.data.period_rank`);
* monthly moments -> pooled rank IC with Newey-West batch-means errors,
  Benjamini-Hochberg q per target, yearly ICs, yearly sign consistency, the
  median yearly IC, the IC of the span's last years (``recent``);
* the studentized Westfall-Young max-T circular-shift screen of Prompt #8, on
  the span's rows (:func:`null_screen`);
* the pipeline-null veto of Prompt #8 re-read on the span's quarters only
  (the stored null moments hold no real outcome).

Probe features (pure noise) go through the same computation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..alpha.information_coefficient import (
    MonthIndex,
    correlation_from_sums,
    ic_with_errors,
    month_index,
    month_moments,
    rank_scores,
)
from ..alpha.mutual_information import binned_codes, joint_counts, mi_from_joint, sentinel_codes
from ..alpha.null_tests import (
    circular_shift_null,
    pair_counts,
    shifted_ic,
    standardized_scores,
)
from ..alpha.ranking import benjamini_hochberg
from ..alpha.stability import consistency, grouped_ics
from ..targets.alignment import target_kind
from .config import FeatureSelectionConfig

__all__ = [
    "SpanEvidence",
    "mutual_information",
    "null_screen",
    "pipeline_veto",
    "span_evidence",
]


@dataclass
class SpanEvidence:
    """Per feature x target evidence of one span (long table) and per feature x kind flags."""

    span: tuple[int, int]
    table: pl.DataFrame                # feature, target, kind, horizon, ic, se, z, p, q, ...
    kind_null: pl.DataFrame            # feature, kind, beyond_max_t, max_t_p
    moments: np.ndarray                # (F, T, months, 6) of the span (rank scores)
    months: MonthIndex


def _span_months(timestamps: pl.Series, lo: int, hi: int) -> MonthIndex:
    return month_index(timestamps.slice(lo, hi - lo))


def span_moments(features: np.ndarray, targets: np.ndarray, timestamps: pl.Series, lo: int,
                 hi: int, *, batch: int = 32) -> tuple[np.ndarray, MonthIndex]:
    """(F, T, M, 6) monthly moments of in-span rank scores (features and targets)."""
    months = _span_months(timestamps, lo, hi)
    ys = [rank_scores(targets[lo:hi, j]).astype(np.float32) for j in range(targets.shape[1])]
    out = np.zeros((features.shape[1], targets.shape[1], months.size, 6))
    for b0 in range(0, features.shape[1], batch):
        xs = [rank_scores(features[lo:hi, j]).astype(np.float32)
              for j in range(b0, min(features.shape[1], b0 + batch))]
        out[b0:b0 + len(xs)] = month_moments(xs, ys, months.starts)
    return out, months


def span_evidence(names: list[str], target_meta: list[tuple[str, int, str]],
                  features: np.ndarray, targets: np.ndarray, timestamps: pl.Series, lo: int,
                  hi: int, cfg: FeatureSelectionConfig, *, recent_start_row: int | None = None,
                  with_nulls: bool = True, seed: int = 0) -> SpanEvidence:
    """IC, stability and null evidence of every feature x target inside rows [lo, hi)."""
    ns = cfg.null_screen
    mom, months = span_moments(features, targets, timestamps, lo, hi)
    m = np.moveaxis(mom, 2, 0)                              # (M, F, T, 6)
    stats = ic_with_errors(m, hac_lags=ns.hac_lag_months)
    yl, yic, yn = grouped_ics(m, months.years, min_obs=500)
    recent_mask = np.zeros(months.size, dtype=bool)
    if recent_start_row is not None and recent_start_row < hi:
        recent_month0 = int(np.searchsorted(months.starts, recent_start_row - lo, side="left"))
        recent_mask[recent_month0:] = True
    recent = ic_with_errors(m, select=recent_mask, hac_lags=ns.hac_lag_months) \
        if recent_mask.any() else None
    recent_ic = (np.where(recent["n"] >= 2000, recent["ic"], np.nan) if recent is not None
                 else np.full(stats["ic"].shape, np.nan))
    cons = consistency(yic, stats["ic"], recent_ic)
    q = np.full(stats["p"].shape, np.nan)
    for j in range(stats["p"].shape[1]):
        q[:, j] = benjamini_hochberg(stats["p"][:, j])
    rows = []
    for i, f in enumerate(names):
        for j, (kind, h, col) in enumerate(target_meta):
            if stats["n"][i, j] < ns.min_observations:
                continue
            rows.append({"feature": f, "target": col, "kind": kind, "horizon": h,
                         "ic": _num(stats["ic"][i, j]), "se": _num(stats["se"][i, j]),
                         "z": _num(stats["z"][i, j]), "p": _num(stats["p"][i, j]),
                         "q": _num(q[i, j]), "n": int(stats["n"][i, j]),
                         "years": int(cons["years"][i, j]),
                         "median_yearly_ic": _num(cons["median_ic"][i, j]),
                         "sign_consistency": _num(cons["sign_consistency"][i, j]),
                         "worst_year_ic": _num(cons["worst_year_ic"][i, j]),
                         "recent_ic": _num(recent_ic[i, j])})
    table = pl.DataFrame(rows, infer_schema_length=None)
    kind_null = (null_screen(names, target_meta, features, targets, lo, hi, cfg, seed=seed)
                 if with_nulls else pl.DataFrame())
    del yl, yn
    return SpanEvidence(span=(lo, hi), table=table, kind_null=kind_null, moments=mom,
                        months=months)


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def null_screen(names: list[str], target_meta: list[tuple[str, int, str]],
                features: np.ndarray, targets: np.ndarray, lo: int, hi: int,
                cfg: FeatureSelectionConfig, *, seed: int, chunk: int = 24) -> pl.DataFrame:
    """Studentized max-T circular-shift screen inside rows [lo, hi) (Prompt #8's definition).

    Per feature: every target's |IC_0| is divided by the RMS of its own shift
    null; a target kind passes when its best studentized |IC_0| exceeds the
    null quantile of the per-draw maximum over *all* targets of the feature.
    """
    ns = cfg.null_screen
    zy = standardized_scores([rank_scores(targets[lo:hi, j]) for j in range(targets.shape[1])])
    kinds = [k for k, _, _ in target_meta]
    by_kind = {k: np.array([j for j, kk in enumerate(kinds) if kk == k])
               for k in dict.fromkeys(kinds)}
    rows = []
    for c0 in range(0, len(names), chunk):
        part = names[c0:c0 + chunk]
        zx = standardized_scores([rank_scores(features[lo:hi, c0 + i]) for i in range(len(part))])
        real = shifted_ic(zx.z, zy.z, 0)
        pairs = pair_counts(zx.present, zy.present)
        shift, _ = circular_shift_null(zx.z, zy.z, draws=ns.circular_shifts,
                                       min_fraction=ns.min_shift_fraction, seed=seed)
        rms = np.sqrt(np.mean(np.square(shift, dtype=np.float64), axis=0))
        usable = (pairs >= ns.min_observations) & (rms > 0)
        scale = np.where(usable, rms, 1.0)
        t_real = np.where(usable, np.abs(real) / scale, -1.0)
        t_max = np.where(usable[None], np.abs(shift) / scale[None], -1.0).max(axis=2)
        q_max = np.quantile(t_max, ns.null_quantile, axis=0)
        for i, f in enumerate(part):
            for kind, idx in by_kind.items():
                best = float(t_real[i, idx].max())
                if best < 0:
                    continue
                p_max = (1 + int((t_max[:, i] >= best).sum())) / (1 + t_max.shape[0])
                rows.append({"feature": f, "kind": kind, "best_studentized": best,
                             "max_null_q": float(q_max[i]), "max_t_p": p_max,
                             "beyond_max_t": bool(best > q_max[i])})
        del zx, shift
    return pl.DataFrame(rows, infer_schema_length=None)


def mutual_information(names: list[str], target_meta: list[tuple[str, int, str]],
                       features: np.ndarray, targets: np.ndarray, lo: int, hi: int, *,
                       bins: int) -> np.ndarray:
    """(F, T) copula MI (nats, Miller-Madow) of in-span rank scores."""
    cys = [sentinel_codes(binned_codes(rank_scores(targets[lo:hi, j]), bins), bins)
           for j in range(targets.shape[1])]
    out = np.full((len(names), len(target_meta)), np.nan)
    for i in range(len(names)):
        cx = sentinel_codes(binned_codes(rank_scores(features[lo:hi, i]), bins), bins,
                            scale=bins + 1)
        for j, cy in enumerate(cys):
            out[i, j] = mi_from_joint(joint_counts(cx, cy, bins))[0]
    return out


def pipeline_veto(results_dir: Path, quarters: list[str], cfg: FeatureSelectionConfig,
                  a: Any) -> pl.DataFrame:
    """Veto nulls' rank IC and SE over the chosen quarters (feature, target, null, ic, se).

    Reads Prompt #8's ``pipeline/cache/<null>_moments_rank_quarterly.npy``; the
    nulls are synthetic paths, so no real outcome enters. Empty if absent.
    """
    del cfg
    cache = results_dir / "pipeline" / "cache"
    frames = []
    for null in (*a.nulls.pipeline_veto, *a.nulls.pipeline_sign_veto):
        path = cache / f"{null}_moments_rank_quarterly.npy"
        axes_path = cache / f"{null}_axes.json"
        if not path.exists() or not axes_path.exists():
            continue
        axes = json.loads(axes_path.read_text(encoding="utf-8"))
        mom = np.load(path)                                   # (F, T, Q, 6)
        keep = np.array([q in set(quarters) for q in axes["quarters"]])
        if not keep.any():
            continue
        stats = ic_with_errors(np.moveaxis(mom[:, :, keep], 2, 0), hac_lags=1)
        fi, ti = np.nonzero(np.isfinite(stats["ic"]))
        frames.append(pl.DataFrame({
            "feature": [axes["features"][i] for i in fi],
            "target": [axes["targets"][j] for j in ti]}).with_columns(
            pl.lit(null).alias("null"), pl.Series("ic", stats["ic"][fi, ti]),
            pl.Series("se", stats["se"][fi, ti])))
    return pl.concat(frames) if frames else pl.DataFrame()


def quarter_labels(timestamps: pl.Series, lo: int, hi: int) -> list[str]:
    ts = timestamps.slice(lo, hi - lo)
    labels = (ts.dt.year().cast(pl.Utf8) + "Q" + ts.dt.quarter().cast(pl.Utf8)).unique()
    return sorted(labels.to_list())


def kind_of(column: str) -> str:
    return target_kind(column)


def correlation_of_sums(sums: np.ndarray) -> np.ndarray:
    return correlation_from_sums(sums)
