"""Orchestration for the rolling-regression research.

Runs the whole suite for one ``(timeframe, window)`` pair and writes the
results in machine-readable form. ``summary.json`` carries a compact
``model_summary`` block designed to be consumed directly by the next stage.

Every report records provenance - git commit, all three config fingerprints,
package versions, the exact date range and observation count - so a number can
be traced back to the code and settings that produced it.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..data.versioning import dataset_lineage
from ..features.config import RegressionConfig
from ..features.rolling_regression import rolling_regression_features
from ..utils.clock import utc_now_iso
from ..utils.config import Config
from ..utils.logging import format_duration, get_logger
from ..utils.paths import atomic_write_text, ensure_dir
from . import regression_plots as rplots
from .residual_decay import (
    ReversionCoefficient,
    conditional_decay,
    move_toward_zero,
    reversion_coefficient,
)
from .residual_extremes import analyse_extremes
from .residual_stationarity import residual_stationarity, segmented_residual_stationarity
from .residuals import (
    ResidualDistribution,
    add_quantile_buckets,
    r_squared_statistics,
    residual_autocorrelation,
    residual_distribution,
    slope_by_group,
    slope_statistics,
)

__all__ = [
    "RegressionReport",
    "generate_regression_report",
    "load_bars",
    "regression_provenance",
    "write_comparison_tables",
]

LOGGER = get_logger("research.regression_reports")


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def _git_commit(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
            text=True, timeout=10, check=False,
        )
        if commit.returncode != 0:
            return {"available": False, "reason": "not a git repository or no commits"}
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True,
            text=True, timeout=10, check=False,
        )
        return {
            "available": True,
            "commit": commit.stdout.strip(),
            "dirty": bool(status.stdout.strip()),
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def _package_versions() -> dict[str, str]:
    import importlib.metadata as md

    out: dict[str, str] = {}
    for name in ("polars", "numpy", "scipy", "statsmodels", "pyarrow", "matplotlib"):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:  # pragma: no cover
            out[name] = "not installed"
    return out


def regression_provenance(
    config: Config,
    regression: RegressionConfig,
    *,
    timeframe: str,
    window: int,
    start: str | None,
    end: str | None,
    observations: int,
    research_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Everything needed to reproduce or date a regression report."""
    return {
        "generated_utc": utc_now_iso(),
        "timeframe": timeframe,
        "regression_window": window,
        "requested_start": start,
        "requested_end": end,
        "observations": observations,
        "git": _git_commit(config.project_root),
        "data_config_fingerprint": config.fingerprint(),
        "regression_config_fingerprint": regression.fingerprint(),
        "research_config_fingerprint": research_fingerprint,
        "data_schema_version": config.schema_version,
        "regression_schema_version": regression.schema_version,
        "instrument": config.instrument,
        "price_transform": regression.regression.price_transform,
        "price_source": regression.regression.price_source,
        "include_current_bar": regression.regression.include_current_bar,
        "timezone": {
            "mode": config.timezone.mode,
            "description": config.timezone.describe(),
            "time_basis": regression.conditioning.time_basis,
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
        "dataset": dataset_lineage(config, None if timeframe == "all" else timeframe),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
@dataclass
class RegressionReport:
    """Outcome of analysing one (timeframe, window) pair."""

    timeframe: str
    window: int
    output_dir: Path
    observations: int
    first_timestamp: str | None
    last_timestamp: str | None
    model_summary: dict[str, Any] = field(default_factory=dict)
    files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    feature_rows: int = 0


class _Writer:
    """Writes a table in every configured format and records what it wrote."""

    def __init__(self, directory: Path, config: RegressionConfig, report: RegressionReport):
        self.directory = ensure_dir(directory)
        self.cfg = config.output
        self.report = report

    def table(self, frame: pl.DataFrame | None, name: str) -> None:
        if frame is None or frame.is_empty():
            return
        if self.cfg.write_parquet:
            frame.write_parquet(self.directory / f"{name}.parquet")
            self.report.files.append(f"{name}.parquet")
        if self.cfg.write_csv:
            frame.write_csv(self.directory / f"{name}.csv")
            self.report.files.append(f"{name}.csv")

    def json(self, payload: dict[str, Any], name: str) -> None:
        if not self.cfg.write_json:
            return
        atomic_write_text(
            self.directory / f"{name}.json",
            json.dumps(payload, indent=2, default=_json_default) + "\n",
        )
        self.report.files.append(f"{name}.json")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def load_bars(
    config: Config, timeframe: str, start: str | None, end: str | None
) -> pl.DataFrame:
    """Load bars for a timeframe with an optional date filter, lazily."""
    directory = config.bars_dir(timeframe)
    files = sorted(directory.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No bars for timeframe {timeframe!r} under {directory}. "
            "Run `xq build-bars --all` first."
        )
    lazy = pl.scan_parquet(files)
    if start:
        lazy = lazy.filter(pl.col("timestamp") >= _parse_date(start, "start"))
    if end:
        lazy = lazy.filter(pl.col("timestamp") < _parse_date(end, "end"))
    return lazy.sort("timestamp").collect()


def _parse_date(value: str, what: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"--{what}={value!r} is not a recognised date (use YYYY-MM-DD).")


def generate_regression_report(
    config: Config,
    regression: RegressionConfig,
    *,
    timeframe: str,
    window: int,
    start: str | None = None,
    end: str | None = None,
    make_plots: bool = True,
    bars: pl.DataFrame | None = None,
) -> RegressionReport:
    """Run the full residual research suite for one timeframe and window."""
    started = time.perf_counter()
    out_dir = regression.results_dir(timeframe, window)
    bars = bars if bars is not None else load_bars(config, timeframe, start, end)

    report = RegressionReport(
        timeframe=timeframe, window=window, output_dir=out_dir,
        observations=bars.height,
        first_timestamp=str(bars["timestamp"].min()) if bars.height else None,
        last_timestamp=str(bars["timestamp"].max()) if bars.height else None,
    )
    if bars.height < window * 3:
        report.warnings.append(
            f"Only {bars.height} bars for a {window}-bar window; need at least "
            f"{window * 3} for meaningful residual statistics. Skipped."
        )
        return report

    writer = _Writer(out_dir, regression, report)
    features, diagnostics = rolling_regression_features(
        bars, window=window, config=regression, timeframe=timeframe
    )
    report.feature_rows = features.height

    if regression.output.cache_features:
        # Only a full-history frame computed from verified bars is cached; the
        # store records the bar dataset version so it can never be read
        # against different bars.
        if start is None and end is None:
            from ..features.store import RegressionFeatureStore

            try:
                RegressionFeatureStore(config, regression).save(
                    timeframe, window, features, diagnostics
                )
            except RuntimeError as exc:
                report.warnings.append(f"Features not cached: {exc}")
        else:
            report.warnings.append(
                "Features not cached: --start/--end restrict the bars, and only "
                "full-history features are cached."
            )

    valid = features.filter(pl.col("residual").is_not_null())
    if valid.height < 100:
        report.warnings.append(f"Only {valid.height} valid fits; analysis skipped.")
        return report

    LOGGER.info("[%s w=%d] %s bars, %s valid fits", timeframe, window,
                f"{bars.height:,}", f"{valid.height:,}")

    summary: dict[str, Any] = {
        "provenance": regression_provenance(
            config, regression, timeframe=timeframe, window=window,
            start=start, end=end, observations=valid.height,
        ),
        "feature_diagnostics": diagnostics,
    }

    # -- slope and R^2 ------------------------------------------------------
    slope = slope_statistics(valid, timeframe=timeframe, window=window, config=regression)
    writer.table(slope.to_frame(), "slope_statistics")
    summary["slope"] = slope.to_dict()

    r2_table = r_squared_statistics(
        valid, timeframe=timeframe, window=window, config=regression
    )
    writer.table(r2_table, "r_squared_statistics")

    # -- residual distribution ---------------------------------------------
    distribution = residual_distribution(
        valid, timeframe=timeframe, window=window, config=regression
    )
    writer.table(distribution.to_frame(), "residual_distribution")
    summary["residual_distribution"] = distribution.to_dict()
    report.warnings += distribution.warnings

    # -- autocorrelation ----------------------------------------------------
    acf_table, lb_table, acf_notes = residual_autocorrelation(
        valid, timeframe=timeframe, window=window, config=regression
    )
    writer.table(acf_table, "residual_acf")
    writer.table(lb_table, "residual_ljung_box")
    summary["autocorrelation_notes"] = acf_notes

    # -- stationarity -------------------------------------------------------
    stationarity = residual_stationarity(
        valid, timeframe=timeframe, window=window, config=regression
    )
    segments = segmented_residual_stationarity(
        valid, timeframe=timeframe, window=window, config=regression
    )
    writer.json(stationarity.to_dict(), "stationarity")
    writer.table(segments, "stationarity_segments")
    summary["stationarity"] = stationarity.to_dict()

    # -- reversion coefficient ---------------------------------------------
    residual_array = valid["residual"].to_numpy().astype(np.float64)
    reversion = reversion_coefficient(
        residual_array, timeframe=timeframe, window=window, config=regression
    )
    summary["reversion"] = reversion.to_dict()

    # -- decay and extremes -------------------------------------------------
    decay = conditional_decay(
        valid, timeframe=timeframe, window=window, config=regression
    )
    writer.table(decay.table, "decay_analysis")
    report.warnings += decay.warnings

    extremes = analyse_extremes(
        valid, timeframe=timeframe, window=window, config=regression
    )
    writer.table(extremes.zero_crossing, "zero_crossing")
    writer.table(extremes.persistence, "extreme_persistence")
    writer.table(extremes.excursion, "adverse_excursion")
    report.warnings += extremes.warnings
    summary["extremes_notes"] = extremes.notes

    # -- conditioning -------------------------------------------------------
    conditioned = _add_conditioning_columns(valid, regression)
    writer.table(
        move_toward_zero(conditioned, config=regression, group_column="vol_bucket"),
        "volatility_conditioning",
    )
    writer.table(
        move_toward_zero(conditioned, config=regression, group_column="slope_bucket"),
        "slope_conditioning",
    )
    writer.table(
        move_toward_zero(conditioned, config=regression, group_column="r2_bucket"),
        "r_squared_conditioning",
    )
    if regression.conditioning.by_hour:
        writer.table(
            _intraday_residuals(conditioned, regression), "intraday"
        )
    if regression.conditioning.by_year:
        writer.table(
            _yearly_stability(conditioned, regression, timeframe, window),
            "yearly_stability",
        )
    writer.table(slope_by_group(conditioned, group_column="year"), "slope_by_year")

    # -- compact machine-readable block for the next stage ------------------
    summary["model_summary"] = _model_summary(
        timeframe=timeframe, window=window, valid=valid, distribution=distribution,
        acf_table=acf_table, stationarity=stationarity, reversion=reversion,
        decay=decay.table, crossing=extremes.zero_crossing, config=regression,
    )
    summary["warnings"] = report.warnings
    report.model_summary = summary["model_summary"]

    if make_plots and regression.plots.enabled:
        report.warnings += _make_plots(
            regression, timeframe, window, valid, acf_table, decay.table,
            extremes.zero_crossing, report,
        )

    writer.json(summary, "summary")
    report.duration_seconds = time.perf_counter() - started
    LOGGER.info("[%s w=%d] written to %s in %s", timeframe, window, out_dir,
                format_duration(report.duration_seconds))
    return report


# ---------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------
def _add_conditioning_columns(
    frame: pl.DataFrame, config: RegressionConfig
) -> pl.DataFrame:
    """Attach the bucket labels used by the conditioning studies.

    Every conditioning variable is known at ``t``: trailing volatility, the
    slope and R-squared of the current fit, the hour and the year.
    """
    out = frame
    for column, label in (
        ("trailing_volatility", "vol_bucket"),
        ("regression_slope", "slope_bucket"),
        ("r_squared", "r2_bucket"),
    ):
        if column in out.columns:
            out = add_quantile_buckets(
                out, column=column, label=label,
                buckets=config.conditioning.quantile_buckets,
            )
    basis = config.conditioning.time_basis
    if basis not in out.columns:
        basis = "timestamp"
    return out.with_columns(
        pl.col(basis).dt.hour().alias("hour"),
        pl.col(basis).dt.year().cast(pl.Utf8).alias("year"),
    )


def _intraday_residuals(
    frame: pl.DataFrame, config: RegressionConfig
) -> pl.DataFrame:
    """Residual behaviour by hour of the broker clock."""
    threshold = config.conditioning.extreme_abs_z
    hourly = (
        frame.group_by("hour")
        .agg(
            pl.len().alias("observations"),
            pl.col("residual").std().alias("residual_std"),
            pl.col("residual").mean().alias("residual_mean"),
            (pl.col("residual_zscore_fit").abs() > threshold)
            .mean().alias("abs_z_gt_threshold_frequency"),
            pl.col("r_squared").mean().alias("mean_r_squared"),
            pl.col("regression_slope").mean().alias("mean_slope"),
        )
        .sort("hour")
    )
    toward = move_toward_zero(
        frame, config=config, group_column="hour", horizons=(5,)
    )
    if toward.is_empty():
        return hourly
    either = (
        toward.filter(pl.col("side") == "either")
        .select(
            pl.col("hour").cast(pl.Int64),
            pl.col("prob_move_toward_zero").alias("prob_toward_zero_h5"),
            pl.col("observations").alias("extreme_observations"),
        )
    )
    return hourly.with_columns(pl.col("hour").cast(pl.Int64)).join(
        either, on="hour", how="left"
    ).sort("hour")


def _yearly_stability(
    frame: pl.DataFrame, config: RegressionConfig, timeframe: str, window: int
) -> pl.DataFrame:
    """Recompute the headline statistics within each year."""
    rows: list[dict[str, Any]] = []
    for year in sorted(frame["year"].unique().drop_nulls().to_list()):
        chunk = frame.filter(pl.col("year") == year)
        residual = chunk["residual"].drop_nulls().to_numpy().astype(np.float64)
        residual = residual[np.isfinite(residual)]
        row: dict[str, Any] = {
            "timeframe": timeframe, "window": window, "year": year,
            "observations": int(residual.size),
        }
        if residual.size < config.decay.min_observations:
            row["sufficient_data"] = False
            rows.append(row)
            continue

        reversion = reversion_coefficient(
            residual, timeframe=timeframe, window=window, config=config,
            segment=str(year), with_control=False,
        )
        zscores = chunk["residual_zscore_fit"].drop_nulls().to_numpy()
        zscores = zscores[np.isfinite(zscores)]
        from scipy import stats as scipy_stats

        row.update({
            "sufficient_data": True,
            "residual_std": float(np.std(residual, ddof=1)),
            "residual_skew": float(scipy_stats.skew(residual, bias=False)),
            "residual_excess_kurtosis": float(
                scipy_stats.kurtosis(residual, fisher=True, bias=False)
            ),
            "residual_acf1": _acf1(residual),
            "reversion_coefficient": reversion.coefficient,
            "reversion_newey_west_t": reversion.newey_west_t,
            "mean_slope": _mean(chunk, "regression_slope"),
            "mean_r_squared": _mean(chunk, "r_squared"),
            "abs_z_gt_2_frequency": (
                float(np.mean(np.abs(zscores) > config.conditioning.extreme_abs_z))
                if zscores.size else float("nan")
            ),
        })
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()


def _mean(frame: pl.DataFrame, column: str) -> float:
    """Mean of a column as a plain float, or NaN when it is empty."""
    value = frame[column].mean()
    return float(value) if isinstance(value, (int, float)) else float("nan")


# ---------------------------------------------------------------------------
# Compact summary
# ---------------------------------------------------------------------------
def _model_summary(
    *, timeframe: str, window: int, valid: pl.DataFrame,
    distribution: ResidualDistribution,
    acf_table: pl.DataFrame, stationarity: Any, reversion: ReversionCoefficient,
    decay: pl.DataFrame,
    crossing: pl.DataFrame, config: RegressionConfig,
) -> dict[str, Any]:
    """The compact block the next stage consumes."""
    residual = distribution.series.get("residual", {})
    zfit = distribution.series.get("residual_zscore_fit", {})
    zscores = valid["residual_zscore_fit"].drop_nulls().to_numpy()
    zscores = zscores[np.isfinite(zscores)]
    threshold = config.conditioning.extreme_abs_z

    def acf_at(series: str, lag: int) -> float | None:
        if acf_table.is_empty():
            return None
        hit = acf_table.filter((pl.col("series") == series) & (pl.col("lag") == lag))
        return float(hit["autocorrelation"][0]) if hit.height else None

    def toward(horizon: int) -> float | None:
        if decay.is_empty():
            return None
        extreme_bins = [b for b in decay["z_bin"].unique().to_list()
                        if b.startswith("Z<-") or b.startswith("Z>=")]
        hit = decay.filter(
            pl.col("z_bin").is_in(extreme_bins) & (pl.col("horizon") == horizon)
        )
        if hit.is_empty():
            return None
        weights = hit["observations"].to_numpy().astype(float)
        values = hit["prob_move_toward_zero"].to_numpy()
        return float(np.average(values, weights=weights)) if weights.sum() else None

    median_cross = None
    if not crossing.is_empty():
        hit = crossing.filter(pl.col("condition") == f"|Z|>{threshold:g}")
        if hit.height:
            value = hit["median_bars_to_cross"][0]
            median_cross = float(value) if value is not None and np.isfinite(value) else None

    adf = (stationarity.adf or {})
    kpss = (stationarity.kpss or {})
    return {
        "timeframe": timeframe,
        "regression_window": window,
        "observations": int(valid.height),
        "residual_std": residual.get("std"),
        "residual_skew": residual.get("skewness"),
        "residual_kurtosis": residual.get("excess_kurtosis"),
        "zscore_fit_skew": zfit.get("skewness"),
        "zscore_fit_kurtosis": zfit.get("excess_kurtosis"),
        "lag1_residual_acf": acf_at("residual", 1),
        "lag1_residual_diff_acf": acf_at("residual_diff", 1),
        "decay_coefficient": reversion.coefficient,
        "decay_coefficient_control": reversion.control_coefficient,
        "decay_newey_west_t": reversion.newey_west_t,
        "adf_statistic": adf.get("statistic"),
        "adf_pvalue": adf.get("p_value"),
        "kpss_statistic": kpss.get("statistic"),
        "kpss_pvalue": kpss.get("p_value"),
        "stationarity_verdict": stationarity.verdict,
        "stationarity_control_verdict": stationarity.control_verdict,
        "stationarity_control_matches": stationarity.control_matches,
        "extreme_count_abs_z_gt_2": int(np.sum(np.abs(zscores) > 2.0)),
        "extreme_frequency_abs_z_gt_2": (
            float(np.mean(np.abs(zscores) > 2.0)) if zscores.size else None
        ),
        "move_toward_zero_probability_5": toward(5),
        "move_toward_zero_probability_10": toward(10),
        "move_toward_zero_probability_20": toward(20),
        "median_zero_crossing_bars": median_cross,
        "caveat": (
            "Descriptive only. decay_coefficient is not an OU speed and not a "
            "half-life; compare it against decay_coefficient_control, which is the "
            "same statistic on a detrended random walk. No transaction cost is "
            "applied anywhere and nothing here is a trading signal."
        ),
    }


def _acf1(array: np.ndarray) -> float:
    if array.size < 3:
        return float("nan")
    a, b = array[1:], array[:-1]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _make_plots(
    config: RegressionConfig, timeframe: str, window: int, valid: pl.DataFrame,
    acf_table: pl.DataFrame, decay: pl.DataFrame, crossing: pl.DataFrame,
    report: RegressionReport,
) -> list[str]:
    """Render every figure, tolerating individual failures."""
    warnings: list[str] = []
    directory = config.plots_dir(timeframe, window)
    suffix = config.plots.format
    jobs = [
        ("price_and_fit", lambda p: rplots.plot_price_and_fit(
            valid, path=p, config=config, timeframe=timeframe, window=window)),
        ("residual_series", lambda p: rplots.plot_residual_series(
            valid, path=p, config=config, timeframe=timeframe, window=window)),
        ("residual_zscore", lambda p: rplots.plot_residual_zscore(
            valid, path=p, config=config, timeframe=timeframe, window=window)),
        ("residual_histogram", lambda p: rplots.plot_residual_histogram(
            valid, path=p, config=config, timeframe=timeframe, window=window)),
        ("residual_qq", lambda p: rplots.plot_residual_qq(
            valid, path=p, config=config, timeframe=timeframe, window=window)),
        ("residual_acf", lambda p: rplots.plot_residual_acf(
            acf_table, path=p, config=config, timeframe=timeframe, window=window)),
        ("decay_curve", lambda p: rplots.plot_decay_curve(
            decay, path=p, config=config, timeframe=timeframe, window=window)),
        ("zero_crossing_curve", lambda p: rplots.plot_zero_crossing_curve(
            crossing, path=p, config=config, timeframe=timeframe, window=window)),
    ]
    for name, fn in jobs:
        path = directory / f"{name}.{suffix}"
        try:
            fn(path)
            report.files.append(f"plots/{path.name}")
        except Exception as exc:  # noqa: BLE001 - a failed plot must not lose the tables
            warnings.append(f"plot {name} failed: {type(exc).__name__}: {exc}")
    return warnings


# ---------------------------------------------------------------------------
# Comparison across windows and timeframes
# ---------------------------------------------------------------------------
def write_comparison_tables(
    config: Config, regression: RegressionConfig, reports: list[RegressionReport]
) -> list[Path]:
    """Cross-window and cross-timeframe comparison tables.

    Deliberately contains no "best window" or "best timeframe" column. The
    question is whether an effect persists across neighbouring settings, not
    where it happens to be largest - a maximum over a grid is a selection
    effect, not a finding.
    """
    summaries = [r.model_summary for r in reports if r.model_summary]
    if not summaries:
        return []

    directory = ensure_dir(regression.results_path)
    written: list[Path] = []
    long = pl.DataFrame(summaries, infer_schema_length=None).drop("caveat", strict=False)

    path = directory / "model_comparison.csv"
    long.write_csv(path)
    written.append(path)
    if regression.output.write_parquet:
        long.write_parquet(directory / "model_comparison.parquet")
        written.append(directory / "model_comparison.parquet")

    metrics = [
        "observations", "residual_std", "residual_kurtosis", "lag1_residual_acf",
        "lag1_residual_diff_acf", "decay_coefficient", "decay_coefficient_control",
        "adf_statistic", "adf_pvalue", "extreme_frequency_abs_z_gt_2",
        "move_toward_zero_probability_5", "move_toward_zero_probability_10",
        "median_zero_crossing_bars",
    ]
    available = [m for m in metrics if m in long.columns]
    pivot = (
        long.with_columns(
            (pl.col("timeframe") + pl.lit("/w") + pl.col("regression_window").cast(pl.Utf8))
            .alias("model")
        )
        .unpivot(index="model", on=available, variable_name="metric", value_name="value")
        .pivot(values="value", index="metric", on="model")
    )
    path = directory / "model_comparison_matrix.csv"
    pivot.write_csv(path)
    written.append(path)

    payload = {
        "generated_utc": utc_now_iso(),
        "models": summaries,
        "provenance": regression_provenance(
            config, regression, timeframe="all", window=0, start=None, end=None,
            observations=sum(int(s.get("observations") or 0) for s in summaries),
        ),
        "notes": [
            "Comparison only. No window or timeframe is recommended: picking one "
            "requires a strategy and a cost model, neither of which exists yet.",
            "Compare decay_coefficient against decay_coefficient_control. Any "
            "bounded zero-mean series gives a negative coefficient, so the control "
            "is the baseline, not zero.",
            "An effect that appears at one window and vanishes at its neighbours is "
            "a selection effect, not a finding.",
        ],
    }
    path = directory / "model_comparison.json"
    atomic_write_text(path, json.dumps(payload, indent=2, default=_json_default) + "\n")
    written.append(path)
    return written
