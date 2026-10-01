"""Cross-timeframe and through-time comparison.

Two questions, both descriptive:

* **Multi-timeframe** — how do the same statistics differ between 1m and 1h?
  The output is a table, not a ranking. There is deliberately no "best
  timeframe" column: choosing one requires a strategy and a cost model, and
  neither exists yet.
* **Structural stability** — do those statistics hold from year to year? A
  number that swings between years is a warning about any model fitted to the
  pooled history, and that warning is more useful than the pooled number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

from .config import ResearchConfig

__all__ = [
    "StabilityAnalysis",
    "TimeframeMetrics",
    "multi_timeframe_table",
    "stability_analysis",
    "timeframe_metrics",
]


@dataclass
class TimeframeMetrics:
    """Headline statistics for a single timeframe."""

    timeframe: str
    observations: int
    first_timestamp: str | None
    last_timestamp: str | None
    return_mean: float
    return_std: float
    skewness: float
    excess_kurtosis: float
    acf_lag1: float
    acf_lag5: float
    abs_acf_lag1: float
    abs_acf_lag5: float
    squared_acf_lag1: float
    mean_spread: float | None
    median_spread: float | None
    mean_tick_count: float | None
    reversal_prob_bottom1pct_h1: float | None = None
    mean_fwd_return_bottom1pct_h5: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def timeframe_metrics(
    *,
    timeframe: str,
    frame: pl.DataFrame,
    return_column: str,
    acf_result: Any = None,
    abs_acf_result: Any = None,
    squared_acf_result: Any = None,
    conditional: Any = None,
    time_basis: str = "timestamp",
) -> TimeframeMetrics:
    """Collect the headline statistics for one timeframe into a single row."""
    returns = frame[return_column].drop_nulls()
    array = returns.to_numpy().astype(np.float64)
    array = array[np.isfinite(array)]

    span_lo = span_hi = None
    if time_basis in frame.columns and frame.height:
        span_lo = str(frame[time_basis].min())
        span_hi = str(frame[time_basis].max())

    metrics = TimeframeMetrics(
        timeframe=timeframe,
        observations=int(array.size),
        first_timestamp=span_lo,
        last_timestamp=span_hi,
        return_mean=float(np.mean(array)) if array.size else float("nan"),
        return_std=float(np.std(array, ddof=1)) if array.size > 1 else float("nan"),
        skewness=float(stats.skew(array, bias=False)) if array.size > 2 else float("nan"),
        excess_kurtosis=(
            float(stats.kurtosis(array, fisher=True, bias=False))
            if array.size > 3 else float("nan")
        ),
        acf_lag1=_lag(acf_result, 1),
        acf_lag5=_lag(acf_result, 5),
        abs_acf_lag1=_lag(abs_acf_result, 1),
        abs_acf_lag5=_lag(abs_acf_result, 5),
        squared_acf_lag1=_lag(squared_acf_result, 1),
        mean_spread=_column_stat(frame, "mean_spread", "mean"),
        median_spread=_column_stat(frame, "mean_spread", "median"),
        mean_tick_count=_column_stat(frame, "tick_count", "mean"),
    )

    if conditional is not None:
        table = conditional.to_frame()
        if not table.is_empty():
            hit = table.filter(
                (pl.col("direction") == "lower")
                & (pl.col("quantile") == 0.01)
                & (pl.col("horizon") == 1)
            )
            if not hit.is_empty():
                metrics.reversal_prob_bottom1pct_h1 = float(hit["prob_reversal"][0])
            hit5 = table.filter(
                (pl.col("direction") == "lower")
                & (pl.col("quantile") == 0.01)
                & (pl.col("horizon") == 5)
            )
            if not hit5.is_empty():
                metrics.mean_fwd_return_bottom1pct_h5 = float(
                    hit5["mean_forward_return"][0]
                )
    return metrics


def multi_timeframe_table(metrics: list[TimeframeMetrics]) -> pl.DataFrame:
    """Stack per-timeframe metrics into the comparison table.

    Rows are metrics and columns are timeframes, which is the orientation that
    makes a like-for-like comparison readable.
    """
    if not metrics:
        return pl.DataFrame()
    wide = pl.DataFrame([m.to_dict() for m in metrics])
    value_columns = [c for c in wide.columns if c != "timeframe"]
    return (
        wide.unpivot(index="timeframe", on=value_columns,
                     variable_name="metric", value_name="value")
        .pivot(values="value", index="metric", on="timeframe")
    )


@dataclass
class StabilityAnalysis:
    """The same statistics recomputed per chronological segment."""

    segment_by: str
    return_column: str
    table: pl.DataFrame = field(default_factory=pl.DataFrame)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_by": self.segment_by,
            "return_column": self.return_column,
            "segments": self.table.height,
            "notes": self.notes,
            "warnings": self.warnings,
        }


def stability_analysis(
    frame: pl.DataFrame,
    *,
    return_column: str,
    config: ResearchConfig,
    time_basis: str = "timestamp",
) -> StabilityAnalysis:
    """Recompute headline statistics within each year (or quarter).

    Segments below ``stability.min_observations`` are reported with their count
    and otherwise left blank, rather than filled with numbers nobody should
    trust.
    """
    cfg = config.stability
    result = StabilityAnalysis(segment_by=cfg.by, return_column=return_column)
    if not cfg.enabled:
        result.notes.append("Stability analysis is disabled in configuration.")
        return result

    usable = frame.filter(pl.col(return_column).is_not_null())
    if usable.is_empty() or time_basis not in usable.columns:
        result.warnings.append("No usable observations for stability analysis.")
        return result

    label = (
        pl.col(time_basis).dt.year().cast(pl.Utf8)
        if cfg.by == "year"
        else pl.col(time_basis).dt.year().cast(pl.Utf8)
        + "Q" + pl.col(time_basis).dt.quarter().cast(pl.Utf8)
    )
    tagged = usable.with_columns(label.alias("segment"))
    threshold = _f(
        tagged.select(
            pl.col(return_column).abs().quantile(config.intraday.extreme_quantile)
        ).item()
    )

    rows: list[dict[str, Any]] = []
    for segment in sorted(tagged["segment"].unique().to_list()):
        chunk = tagged.filter(pl.col("segment") == segment)
        array = chunk[return_column].drop_nulls().to_numpy().astype(np.float64)
        array = array[np.isfinite(array)]
        row: dict[str, Any] = {"segment": segment, "observations": int(array.size)}
        if array.size < cfg.min_observations:
            row["sufficient_data"] = False
            rows.append(row)
            continue

        row.update({
            "sufficient_data": True,
            "mean": float(np.mean(array)),
            "std": float(np.std(array, ddof=1)),
            "skew": float(stats.skew(array, bias=False)),
            "excess_kurtosis": float(stats.kurtosis(array, fisher=True, bias=False)),
            "acf1": _acf1(array),
            "abs_acf1": _acf1(np.abs(array)),
            "extreme_frequency": float(np.mean(np.abs(array) > threshold)),
            "min_return": float(np.min(array)),
            "max_return": float(np.max(array)),
        })
        row["mean_spread"] = _column_stat(chunk, "mean_spread", "mean")
        row["mean_tick_count"] = _column_stat(chunk, "tick_count", "mean")
        rows.append(row)

    result.table = pl.DataFrame(rows, infer_schema_length=None).sort("segment")
    thin = [r for r in rows if not r.get("sufficient_data", False)]
    if thin:
        result.warnings.append(
            f"{len(thin)} segment(s) fell below {cfg.min_observations:,} observations "
            "and were left unpopulated: "
            f"{[r['segment'] for r in thin]}."
        )
    result.notes += [
        f"Segments are {cfg.by}s on the `{time_basis}` clock.",
        f"extreme_frequency uses a single whole-period |r| threshold "
        f"({threshold:.6g}), so segments are comparable to each other.",
        "Variation between segments is evidence about structural stability; it "
        "does not by itself invalidate the pooled statistics, but any model "
        "fitted to the pooled history inherits that variation.",
    ]
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _lag(acf_result: Any, lag: int) -> float:
    """Pull one lag out of an AcfResult, or NaN when unavailable."""
    if acf_result is None:
        return float("nan")
    try:
        return float(acf_result.values[acf_result.lags.index(lag)])
    except (ValueError, IndexError):
        return float("nan")


def _acf1(array: np.ndarray) -> float:
    """Lag-1 autocorrelation of a dense array."""
    if array.size < 3:
        return float("nan")
    a, b = array[1:], array[:-1]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _column_stat(frame: pl.DataFrame, column: str, stat: str) -> float | None:
    """Mean/median of a column when present, else None."""
    if column not in frame.columns:
        return None
    series = frame[column].drop_nulls()
    if series.is_empty():
        return None
    return _f(series.mean() if stat == "mean" else series.median())


def _f(value: Any) -> float:
    """Coerce a Polars aggregate to float.

    Polars aggregates are Optional in the type stubs because an empty frame
    yields null. Callers here guard against that, but an empty group would give
    NaN rather than a crash, which is the right failure mode for a statistic.
    """
    return float("nan") if value is None else float(value)


def _i(value: Any) -> int:
    """Integer counterpart of :func:`_f`; zero when the aggregate is null."""
    return 0 if value is None else int(value)
