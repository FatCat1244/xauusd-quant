r"""Information coefficients of spectral features - exploratory, not a signal.

:math:`IC(h) = \mathrm{Corr}(S_t, Y_{t+h})` with a spectral feature ``S_t``
computed from bars up to ``t`` and an outcome ``Y`` that begins after ``t``:

``future_return``                  :math:`\ln P_{t+h} - \ln P_t`
``future_residual_change``         :math:`\epsilon_{t+h} - \epsilon_t`
``future_abs_residual_reduction``  :math:`|\epsilon_t| - |\epsilon_{t+h}|`
``future_volatility``              :math:`\sqrt{h^{-1}\sum_{k=1}^{h} r_{t+k}^2}`
``future_innovation_magnitude``    :math:`h^{-1}\sum_{k=1}^{h} |\eta_{t+k}|`

``ic_table`` is the engine; any layer passes it its features at the sampled
bars (the wavelet layer does), ``ic_study`` feeds it the spectral ones.

Pearson IC and Spearman rank IC are computed over the whole history and
within every year, quarter and volatility quartile. An overall IC can hide a
sign that flips from year to year, so each test is summarised by the mean,
median and dispersion of its yearly values, a t-statistic across years and
the share of years with a positive IC. The t-statistic treats years as
independent, which they roughly are; bars are not - forward outcomes for
``h > 1`` overlap, so no bar-level significance is reported at all.

Every (feature, target, horizon) evaluated is counted: the number of tests is
part of the result. The largest |IC| among many is expected to look
interesting by chance, which is why each study also computes the same ICs on
null controls, and why nothing here is ever used to choose a feature.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from ..features.spectral import RollingSpectrum
from ..features.spectral_config import SpectralConfig
from .spectral_nulls import SourceData

__all__ = ["TARGETS", "feature_columns", "ic_study", "ic_table", "target_columns"]

#: Every forward outcome ``target_columns`` can build.
TARGETS: tuple[str, ...] = (
    "future_return", "future_residual_change", "future_abs_residual_reduction",
    "future_volatility", "future_innovation_magnitude",
)


FEATURES: tuple[str, ...] = (
    "spectral_entropy", "spectral_flatness", "spectral_centroid_norm", "top1_power_share",
    "top3_power_share", "top5_power_share", "low_power_share", "mid_power_share",
    "high_power_share", "log_dominant_period", "dominant_phase_sin", "dominant_phase_cos",
)


def feature_columns(result: RollingSpectrum, names: tuple[str, ...],
                    rows: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """The spectral features an IC study reads, NaN where not defined.

    With *rows*, only those bars are materialised (a full-length float64
    copy of ten features is ~630 MB at 23 years of 1-minute bars).
    """
    missing = [n for n in names if n not in FEATURES]
    if missing:
        raise KeyError(f"unknown IC feature(s) {missing}; known: {list(FEATURES)}")
    idx: slice | np.ndarray = slice(None) if rows is None else rows

    def at(values: np.ndarray) -> np.ndarray:
        return np.asarray(values[idx], dtype=np.float64)

    phase = np.where(result.dominant_phase_valid[idx], at(result.dominant_phase), np.nan)
    out: dict[str, np.ndarray] = {}
    for name in names:
        if name == "spectral_entropy":
            out[name] = at(result.entropy)
        elif name == "spectral_flatness":
            out[name] = at(result.flatness)
        elif name == "spectral_centroid_norm":
            out[name] = at(result.centroid) / 0.5
        elif name.startswith("top"):
            out[name] = at(result.concentration[int(name[3])])
        elif name.endswith("_power_share"):
            out[name] = at(result.band_shares[name.removesuffix("_power_share")])
        elif name == "log_dominant_period":
            with np.errstate(divide="ignore", invalid="ignore"):
                out[name] = np.log(at(result.dominant_period_bars))
        elif name == "dominant_phase_sin":
            out[name] = np.sin(phase)
        elif name == "dominant_phase_cos":
            out[name] = np.cos(phase)
    return out


def target_columns(source: SourceData, targets: tuple[str, ...], horizons: tuple[int, ...],
                   rows: np.ndarray) -> dict[str, np.ndarray]:
    """Forward outcomes ``<target>_h<h>`` at bars *rows* (NaN past the end of history).

    Window sums use prefix sums, so an outcome over ``t+1 .. t+h`` costs the
    same at every horizon; one missing value inside the span makes it NaN.
    """
    unknown = [t for t in targets if t not in TARGETS]
    if unknown:
        raise KeyError(f"unknown IC target(s) {unknown}; known: {list(TARGETS)}")
    n = source.size
    out: dict[str, np.ndarray] = {}
    returns = source.columns["log_return"].astype(np.float64)
    log_price = np.concatenate(([0.0], np.cumsum(np.nan_to_num(returns[1:]))))
    residual = source.columns.get("regression_residual")
    innovation = source.columns.get("ou_innovation")

    def span_sums(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        finite = np.isfinite(values)
        return (np.concatenate(([0.0], np.cumsum(np.where(finite, values, 0.0)))),
                np.concatenate(([0], np.cumsum(~finite, dtype=np.int64))))

    squares = span_sums(returns ** 2) if "future_volatility" in targets else None
    magnitude = (span_sums(np.abs(innovation.astype(np.float64)))
                 if innovation is not None and "future_innovation_magnitude" in targets else None)
    for h in horizons:
        ahead = rows + h
        inside = ahead < n
        safe = np.where(inside, ahead, 0)
        if "future_return" in targets:
            out[f"future_return_h{h}"] = np.where(inside, log_price[safe] - log_price[rows],
                                                  np.nan)
        if residual is not None:
            now = residual[rows]
            later = np.where(inside, residual[safe], np.nan)
            if "future_residual_change" in targets:
                out[f"future_residual_change_h{h}"] = later - now
            if "future_abs_residual_reduction" in targets:
                out[f"future_abs_residual_reduction_h{h}"] = np.abs(now) - np.abs(later)
        # the bars t+1 .. t+h are prefix indices rows+1 .. rows+h+1
        lo = rows + 1
        hi = np.where(inside, safe + 1, lo)
        if squares is not None:
            total, gaps = squares
            ok = inside & (gaps[hi] - gaps[lo] == 0)
            out[f"future_volatility_h{h}"] = np.where(
                ok, np.sqrt(np.maximum(total[hi] - total[lo], 0.0) / h), np.nan)
        if magnitude is not None:
            total, gaps = magnitude
            ok = inside & (gaps[hi] - gaps[lo] == 0)
            out[f"future_innovation_magnitude_h{h}"] = np.where(
                ok, (total[hi] - total[lo]) / h, np.nan)
    return out


def _summarise(per_period: pl.DataFrame, prefix: str) -> dict[str, Any]:
    values = per_period.drop_nulls().to_series().to_numpy()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {f"{prefix}_periods": 0}
    std = float(values.std(ddof=1)) if values.size > 1 else float("nan")
    return {
        f"{prefix}_periods": int(values.size),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_std": std,
        f"{prefix}_t": (float(values.mean() / (std / np.sqrt(values.size)))
                        if values.size > 1 and std > 0 else None),
        f"{prefix}_positive_share": float(np.mean(values > 0)),
    }

def ic_study(
    result: RollingSpectrum, source: SourceData, config: SpectralConfig, *, max_rows: int,
    source_name: str,
) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    """IC and rank IC of every configured spectral (feature, target, horizon).

    Estimated on an evenly spaced sample of at most *max_rows* valid bars.
    """
    cfg = config.ic
    rows = sample_rows(result.valid, max_rows)
    features = feature_columns(result, cfg.features, rows)
    targets = target_columns(source, cfg.targets, cfg.horizons, rows)
    volatility = (source.columns["trailing_volatility"][rows]
                  if "trailing_volatility" in source.columns else None)
    return ic_table(features, targets, timestamps=source.timestamps.gather(pl.Series(rows)),
                    volatility=volatility, source_name=source_name)


def sample_rows(valid: np.ndarray, max_rows: int) -> np.ndarray:
    """Indices of the valid bars, thinned evenly to at most *max_rows*."""
    rows = np.flatnonzero(valid)
    if rows.size > max_rows:
        rows = rows[np.linspace(0, rows.size - 1, max_rows).round().astype(np.int64)]
    return rows


def ic_table(
    features: dict[str, np.ndarray], targets: dict[str, np.ndarray], *, timestamps: pl.Series,
    volatility: np.ndarray | None, source_name: str,
) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    """IC and rank IC of every (feature, target) pair, overall and by period.

    *features* and *targets* (``<target>_h<h>``) are aligned with
    *timestamps*, one entry per sampled bar. Returns the summary table (one
    row per test), the per-period table (one row per test and year / quarter
    / volatility bucket) and the test count: the pairs whose rank IC is
    defined. A feature that is NaN wherever the target exists (a fast/slow
    ratio with no slow band) keeps its row, but it is not a test.
    """
    if timestamps.len() < 100 or not targets or not features:
        return pl.DataFrame(), pl.DataFrame(), 0
    columns: dict[str, Any] = {"timestamp": timestamps}
    columns.update({f"f__{k}": v for k, v in features.items()})
    columns.update({f"y__{k}": v for k, v in targets.items()})
    if volatility is not None:
        columns["volatility"] = volatility
    frame = pl.DataFrame(columns).with_columns(pl.col(pl.Float64).fill_nan(None))
    frame = frame.with_columns(
        pl.col("timestamp").dt.year().alias("year"),
        (pl.col("timestamp").dt.year().cast(pl.Utf8) + "Q"
         + pl.col("timestamp").dt.quarter().cast(pl.Utf8)).alias("quarter"))
    if "volatility" in frame.columns:
        edges = frame["volatility"].drop_nulls().quantile(0.25), frame["volatility"].drop_nulls(
        ).quantile(0.5), frame["volatility"].drop_nulls().quantile(0.75)
        frame = frame.with_columns(
            pl.when(pl.col("volatility").is_null()).then(None)
            .when(pl.col("volatility") <= edges[0]).then(pl.lit("Q1"))
            .when(pl.col("volatility") <= edges[1]).then(pl.lit("Q2"))
            .when(pl.col("volatility") <= edges[2]).then(pl.lit("Q3"))
            .otherwise(pl.lit("Q4")).alias("volatility_bucket"))
    pairs = [(f, y) for f in features for y in targets]
    exprs = []
    for f, y in pairs:
        exprs.append(pl.corr(f"f__{f}", f"y__{y}").alias(f"ic|{f}|{y}"))
        exprs.append(pl.corr(f"f__{f}", f"y__{y}", method="spearman").alias(f"rank|{f}|{y}"))
    overall = frame.select(exprs).row(0, named=True)
    groupings = [("year", "year"), ("quarter", "quarter")]
    if "volatility_bucket" in frame.columns:
        groupings.append(("volatility_bucket", "volatility"))
    per_group = {name: frame.filter(pl.col(col).is_not_null()).group_by(col).agg(exprs)
                 .sort(col) for col, name in groupings}
    summary_rows: list[dict[str, Any]] = []
    period_rows: list[dict[str, Any]] = []
    for f, y in pairs:
        target, horizon = y.rsplit("_h", 1)
        row: dict[str, Any] = {
            "source": source_name, "feature": f, "target": target, "horizon": int(horizon),
            "observations": int(frame.select(pl.col(f"f__{f}").is_not_null()
                                             & pl.col(f"y__{y}").is_not_null()).sum().item()),
            "ic": overall[f"ic|{f}|{y}"], "rank_ic": overall[f"rank|{f}|{y}"],
        }
        for name, table in per_group.items():
            if name in ("year", "quarter"):
                row.update(_summarise(table.select(f"rank|{f}|{y}"), f"{name}ly_rank_ic"
                                      if name == "year" else "quarterly_rank_ic"))
                row.update(_summarise(table.select(f"ic|{f}|{y}"), f"{name}ly_ic"
                                      if name == "year" else "quarterly_ic"))
            key = table.columns[0]
            for record in table.select(key, f"ic|{f}|{y}", f"rank|{f}|{y}").iter_rows():
                period_rows.append({"source": source_name, "feature": f, "target": target,
                                    "horizon": int(horizon), "grouping": name,
                                    "group": str(record[0]), "ic": record[1],
                                    "rank_ic": record[2]})
        summary_rows.append(row)
    defined = sum(1 for r in summary_rows
                  if r["rank_ic"] is not None and np.isfinite(r["rank_ic"]))
    return (pl.DataFrame(summary_rows, infer_schema_length=None),
            pl.DataFrame(period_rows, infer_schema_length=None), defined)
