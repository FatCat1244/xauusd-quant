r"""Is a spectral feature a property of the market, or of one window?

A dominant period that jumps every bar, or a frequency that leaves the top-K
set as soon as the window moves on, is not a cycle. This module measures
persistence and stability, always in forms that can be computed identically
on the null controls:

* **dominant-period stability**: how often the dominant bin changes from one
  bar to the next, how long it stays, and how often it is the same ``h`` bars
  later. Windows fewer than ``N`` bars apart share data, so only lags of at
  least ``N`` measure persistence beyond overlap;
* **frequency persistence**:
  :math:`Overlap_t(h) = |TopK_t \cap TopK_{t+h}| / K`, where two frequencies
  match when :math:`|f_1-f_2|/f_1 < \delta`;
* **stability through time**: medians by month, quarter, year and
  equal-length era - 2003 is not assumed to resemble 2026;
* **conditioning**: by volatility, trend strength and OU reversion speed, and
  by hour and session on the configured (broker) clock;
* **change**: the distribution of bar-to-bar and window-to-window changes of
  entropy, centroid and dominant period, for later regime work.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import polars as pl

from ..features.spectral import RollingSpectrum
from ..features.spectral_config import ConditioningConfig
from ..models.ornstein_uhlenbeck import OU_STATES
from .config import ResearchConfig
from .intraday import assign_sessions
from .ou_conditional import TREND_LABELS_5, _bucket

__all__ = [
    "METRICS",
    "change_distribution",
    "conditioning_table",
    "dominant_period_stability",
    "era_table",
    "frequency_persistence",
    "intraday_table",
    "metrics_frame",
    "segment_table",
    "summary_exprs",
]

#: Per-bar spectral metrics summarised throughout the study.
METRICS: tuple[str, ...] = (
    "spectral_entropy", "spectral_flatness", "spectral_centroid_norm", "top1_power_share",
    "top3_power_share", "top5_power_share", "low_power_share", "mid_power_share",
    "high_power_share", "dominant_period_bars", "dominant_period_seconds",
)
_OU_SPEED = ("fast", "medium", "slow")
_NEAR_UNIT_ROOT = float(OU_STATES.index("near_unit_root"))


def metrics_frame(result: RollingSpectrum, timestamps: pl.Series,
                  extra: dict[str, np.ndarray] | None = None) -> pl.DataFrame:
    """Valid windows only: timestamp, the spectral metrics, and any *extra* columns."""
    valid = result.valid
    columns: dict[str, Any] = {"timestamp": timestamps.filter(pl.Series(valid))}

    def f32(values: np.ndarray) -> np.ndarray:
        return np.asarray(values[valid], dtype=np.float32)

    columns.update({
        "spectral_entropy": f32(result.entropy),
        "spectral_flatness": f32(result.flatness),
        "spectral_centroid_norm": f32(result.centroid_norm),
        **{f"top{m}_power_share": f32(arr) for m, arr in result.concentration.items()},
        **{f"{name}_power_share": f32(arr) for name, arr in result.band_shares.items()},
        "dominant_period_bars": f32(result.dominant_period_bars),
        "dominant_period_seconds": f32(result.dominant_period_seconds),
        "dominant_bin": result.dominant_bin[valid].astype(np.int16),
    })
    for name, values in (extra or {}).items():
        columns[name] = f32(values)
    frame = pl.DataFrame(columns)
    return frame.with_columns(pl.col(pl.Float32).fill_nan(None))


def dominant_period_stability(result: RollingSpectrum, lags: list[int]) -> dict[str, Any]:
    """Switching, run length and same-bin persistence of the dominant component."""
    bins = result.dominant_bin.astype(np.int64)
    ok = bins > 0
    both = ok[1:] & ok[:-1]
    out: dict[str, Any] = {"valid_windows": int(ok.sum())}
    if both.sum() < 10:
        return out
    change = bins[1:] != bins[:-1]
    out["switching_fraction_lag1"] = float(change[both].mean())
    # Runs of an unchanged dominant bin over consecutive valid windows.
    breaks = np.flatnonzero(~both | change) + 1
    segments = np.diff(np.concatenate(([0], breaks, [bins.size])))
    starts = np.concatenate(([0], breaks))
    runs = segments[ok[starts[: segments.size]]]
    if runs.size:
        out.update({"run_length_median": float(np.median(runs)),
                    "run_length_mean": float(runs.mean()),
                    "run_length_p90": float(np.quantile(runs, 0.9))})
    period = result.dominant_period_bars.astype(np.float64)
    rel = np.abs(np.diff(period))[both] / period[:-1][both]
    out["relative_period_change_median"] = float(np.median(rel))
    out["relative_period_change_p90"] = float(np.quantile(rel, 0.9))
    for h in lags:
        if h <= 0 or h >= bins.size:
            continue
        pair = ok[h:] & ok[:-h]
        if pair.sum():
            out[f"same_dominant_bin_lag_{h}"] = float((bins[h:] == bins[:-h])[pair].mean())
    return out


def frequency_persistence(result: RollingSpectrum, lags: list[int], *, top_k: int,
                          tolerance: float, max_pairs: int = 500_000) -> list[dict[str, Any]]:
    """Mean top-K overlap between windows ``h`` bars apart (tolerance matching).

    Estimated on at most *max_pairs* evenly spaced pairs per lag.
    """
    freqs = result.top_bins[:, :top_k].astype(np.float64) / result.fft_window
    freqs[result.top_bins[:, :top_k] < 0] = np.nan
    counts = np.isfinite(freqs).sum(axis=1)
    rows = []
    for h in lags:
        if h <= 0 or h >= freqs.shape[0]:
            continue
        idx = np.flatnonzero((counts[:-h] > 0) & (counts[h:] > 0))
        if idx.size == 0:
            continue
        if idx.size > max_pairs:
            idx = idx[np.linspace(0, idx.size - 1, max_pairs).round().astype(np.int64)]
        a, b = freqs[idx][:, :, None], freqs[idx + h][:, None, :]
        with np.errstate(invalid="ignore"):
            match = (np.abs(a - b) / a < tolerance).any(axis=2)
        overlap = match.sum(axis=1) / counts[idx]
        rows.append({"lag": h, "lag_in_windows": h / result.fft_window, "pairs": int(idx.size),
                     "mean_overlap": float(overlap.mean()),
                     "median_overlap": float(np.median(overlap)),
                     "full_overlap_fraction": float(np.mean(overlap >= 1.0))})
    return rows


def change_distribution(frame: pl.DataFrame, lags: list[int], quantiles: tuple[float, ...],
                        *, source: str) -> pl.DataFrame:
    """Quantiles of metric changes between windows ``h`` valid bars apart."""
    metrics = ("spectral_entropy", "spectral_centroid_norm", "top1_power_share",
               "high_power_share")
    rows = []
    log_period = np.log(frame["dominant_period_bars"].to_numpy().astype(np.float64))
    for h in lags:
        if h <= 0 or h >= frame.height:
            continue
        series = {m: frame[m].to_numpy().astype(np.float64) for m in metrics}
        series["log_dominant_period"] = log_period
        for name, values in series.items():
            delta = values[h:] - values[:-h]
            delta = delta[np.isfinite(delta)]
            if delta.size < 10:
                continue
            q = np.quantile(delta, quantiles)
            iqr = float(np.quantile(delta, 0.75) - np.quantile(delta, 0.25))
            rows.append({"source": source, "metric": name, "lag": h,
                         "observations": int(delta.size), "std": float(delta.std()),
                         **{f"q{int(round(p * 100)):02d}": float(v)
                            for p, v in zip(quantiles, q, strict=True)},
                         "share_beyond_3_iqr": (float(np.mean(np.abs(delta) > 3 * iqr))
                                                if iqr > 0 else None)})
    return pl.DataFrame(rows, infer_schema_length=None)


def summary_exprs(metrics: tuple[str, ...] = METRICS, *, period: str = "dominant_period_bars",
                  seconds: str = "dominant_period_seconds",
                  mode: str = "dominant_bin") -> list[pl.Expr]:
    """Per-group summaries: medians of *metrics*, the period's IQR, the modal *mode*.

    The defaults are the spectral metrics; another layer passes its own names.
    """
    return [
        pl.len().alias("windows"),
        *[pl.col(m).median().alias(f"median_{m}") for m in metrics if m != seconds],
        pl.col(period).quantile(0.25).alias(f"{period}_p25"),
        pl.col(period).quantile(0.75).alias(f"{period}_p75"),
        pl.col(seconds).median().alias(f"median_{seconds}"),
        pl.col(mode).mode().first().alias(f"modal_{mode}"),
        (pl.col(mode) == pl.col(mode).mode().first()).mean().alias(f"modal_{mode}_share"),
    ]


def _summaries(summaries: list[pl.Expr] | None = None) -> list[pl.Expr]:
    return summary_exprs() if summaries is None else summaries


def segment_table(frame: pl.DataFrame, *, by: str, source: str,
                  total_bars: pl.DataFrame | None = None,
                  summaries: list[pl.Expr] | None = None) -> pl.DataFrame:
    """Medians of every metric by month, quarter or year.

    *total_bars* (segment -> bars) turns window counts into valid fractions.
    """
    t = pl.col("timestamp")
    label = {
        "year": t.dt.year().cast(pl.Utf8),
        "quarter": t.dt.year().cast(pl.Utf8) + "Q" + t.dt.quarter().cast(pl.Utf8),
        "month": t.dt.strftime("%Y-%m"),
    }[by]
    table = (frame.with_columns(label.alias("segment")).group_by("segment")
             .agg(_summaries(summaries)).sort("segment"))
    if total_bars is not None:
        table = table.join(total_bars, on="segment", how="left").with_columns(
            (pl.col("windows") / pl.col("bars")).alias("valid_fraction"))
    return table.with_columns(pl.lit(source).alias("source")).select(
        "source", "segment", pl.exclude("source", "segment"))


def era_table(frame: pl.DataFrame, *, eras: int, source: str,
              summaries: list[pl.Expr] | None = None) -> pl.DataFrame:
    """Equal-length chronological blocks: boundaries by arithmetic, not by choice."""
    first, last = frame["timestamp"].min(), frame["timestamp"].max()
    if not isinstance(first, datetime) or not isinstance(last, datetime):
        return pl.DataFrame()
    span = (last - first) / eras
    edges = [first + span * i for i in range(eras + 1)]
    names = (["1_early", "2_middle", "3_recent"] if eras == 3
             else [f"{i + 1}_block" for i in range(eras)])
    label = pl.lit(names[-1])
    for i in reversed(range(eras - 1)):
        label = pl.when(pl.col("timestamp") < edges[i + 1]).then(pl.lit(names[i])).otherwise(label)
    table = (frame.with_columns(label.alias("segment")).group_by("segment")
             .agg(*_summaries(summaries), pl.col("timestamp").min().alias("first_timestamp"),
                  pl.col("timestamp").max().alias("last_timestamp"))
             .sort("segment"))
    return table.with_columns(pl.lit(source).alias("source")).select(
        "source", "segment", pl.exclude("source", "segment"))


def conditioning_table(frame: pl.DataFrame, cfg: ConditioningConfig, *, source: str,
                       summaries: list[pl.Expr] | None = None) -> pl.DataFrame:
    """Metric medians by volatility, trend and OU-speed bucket.

    Volatility: quartiles of trailing volatility (Prompt #2/#3 definition).
    Trend: quintiles of the regression slope, labelled strongly negative to
    strongly positive. OU speed: terciles of the half-life among valid OU
    fits (fast = shortest), plus ``invalid`` for windows without one - or,
    when the frame carries ``ou_state_code``, ``near_unit_root`` (a
    stationary fit whose half-life is past the reportable cap) apart from the
    other ``invalid`` states.
    """
    tables = []
    for column, buckets, labels, variable in (
        ("trailing_volatility", cfg.volatility_buckets, None, "volatility"),
        ("regression_slope", cfg.trend_buckets, TREND_LABELS_5, "trend"),
    ):
        if column not in frame.columns:
            continue
        bucketed, edges = _bucket(frame, column, buckets, labels)
        order = list(labels) if labels else [f"Q{i + 1}" for i in range(buckets)]
        table = (bucketed.filter(pl.col("bucket").is_not_null()).group_by("bucket")
                 .agg(_summaries(summaries)))
        rank = {name: i for i, name in enumerate(order)}
        tables.append(table.with_columns(
            pl.lit(variable).alias("variable"),
            pl.col("bucket").replace_strict(rank, return_dtype=pl.Int32).alias("bucket_rank"),
        ))
    if "ou_half_life_bars" in frame.columns and "ou_valid" in frame.columns:
        valid = frame.filter((pl.col("ou_valid") > 0) & pl.col("ou_half_life_bars").is_not_null())
        if valid.height:
            bucketed, _ = _bucket(valid, "ou_half_life_bars", cfg.ou_speed_buckets,
                                  _OU_SPEED if cfg.ou_speed_buckets == 3 else None)
            invalid = frame.filter(~((pl.col("ou_valid") > 0)
                                     & pl.col("ou_half_life_bars").is_not_null()))
            names = list(_OU_SPEED if cfg.ou_speed_buckets == 3 else
                         [f"Q{i + 1}" for i in range(cfg.ou_speed_buckets)])
            if "ou_state_code" in frame.columns:
                label = (pl.when(pl.col("ou_state_code") == _NEAR_UNIT_ROOT)
                         .then(pl.lit("near_unit_root")).otherwise(pl.lit("invalid")))
                names += ["near_unit_root", "invalid"]
            else:
                label = pl.lit("invalid")
                names.append("invalid")
            combined = pl.concat([
                bucketed.filter(pl.col("bucket").is_not_null()),
                invalid.with_columns(label.alias("bucket")),
            ], how="diagonal_relaxed")
            table = combined.group_by("bucket").agg(_summaries(summaries))
            tables.append(table.with_columns(
                pl.lit("ou_speed").alias("variable"),
                pl.col("bucket").replace_strict({n: i for i, n in enumerate(names)},
                                                return_dtype=pl.Int32).alias("bucket_rank"),
            ))
    if not tables:
        return pl.DataFrame()
    return (pl.concat(tables, how="diagonal_relaxed")
            .with_columns(pl.lit(source).alias("source"),
                          (pl.col("windows") < cfg.min_samples_warning).alias("thin_sample"))
            .select("source", "variable", "bucket", "bucket_rank", pl.exclude(
                "source", "variable", "bucket", "bucket_rank"))
            .sort("variable", "bucket_rank"))


def intraday_table(frame: pl.DataFrame, cfg: ConditioningConfig, research: ResearchConfig, *,
                   source: str, summaries: list[pl.Expr] | None = None) -> pl.DataFrame:
    """Metric medians by hour and by session, on the broker clock of the bars."""
    tables = []
    if cfg.by_hour:
        tables.append(frame.with_columns(
            pl.col("timestamp").dt.hour().cast(pl.Int32).cast(pl.Utf8).str.zfill(2).alias("group"))
            .group_by("group").agg(_summaries(summaries))
            .with_columns(pl.lit("hour").alias("grouping")))
    if cfg.by_session and research.intraday.sessions.definitions:
        tables.append(frame.with_columns(assign_sessions(pl.col("timestamp"), research).alias("group"))
                      .group_by("group").agg(_summaries(summaries))
                      .with_columns(pl.lit("session").alias("grouping")))
    if not tables:
        return pl.DataFrame()
    return (pl.concat(tables, how="diagonal_relaxed").with_columns(pl.lit(source).alias("source"))
            .select("source", "grouping", "group", pl.exclude("source", "grouping", "group"))
            .sort("grouping", "group"))
