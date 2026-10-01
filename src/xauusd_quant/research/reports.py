"""Research report generation.

Runs every analysis for one timeframe and writes the results in
machine-readable form: Parquet for the larger tables, CSV for eyeballing, JSON
for summaries and metadata. Plots are written too, but never as the only copy
of a result - a future prompt has to be able to *consume* these outputs, not
squint at them.

Every report carries a provenance block (git commit, both config fingerprints,
package versions, the exact date range and observation count) so a number can
always be traced back to the code and settings that produced it.
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

import polars as pl

from ..data.versioning import dataset_lineage
from ..utils.clock import utc_now_iso
from ..utils.config import Config
from ..utils.logging import format_duration, get_logger
from ..utils.paths import atomic_write_text, ensure_dir
from . import plots as plot_module
from .autocorrelation import (
    acf_absolute_returns,
    acf_frame,
    acf_returns,
    acf_squared_returns,
    ljung_box_frame,
    ljung_box_test,
)
from .comparison import (
    TimeframeMetrics,
    multi_timeframe_table,
    stability_analysis,
    timeframe_metrics,
)
from .conditional import conditional_return_analysis, reversal_summary
from .config import ResearchConfig
from .distributions import (
    describe_distribution,
    gaussian_tail_comparison,
    normality_tests,
    summary_to_frame,
)
from .intraday import hourly_analysis, session_analysis, weekday_analysis
from .returns import prepare_returns
from .rolling import rolling_frame
from .spread import activity_analysis, spread_analysis
from .stationarity import segmented_stationarity, stationarity_report
from .volatility import (
    annualization_factor,
    ewma_volatility,
    realized_volatility,
    rolling_volatility,
    volatility_regimes,
)

__all__ = [
    "ResearchReport",
    "generate_multi_timeframe_summary",
    "generate_report",
    "provenance",
]

LOGGER = get_logger("research.reports")


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def _git_commit(root: Path) -> dict[str, Any]:
    """Current commit and dirty state, when this is a git checkout."""
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
    for name in ("polars", "pyarrow", "duckdb", "numpy", "scipy",
                 "statsmodels", "matplotlib"):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:  # pragma: no cover - optional dependency
            out[name] = "not installed"
    return out


def provenance(
    config: Config,
    research: ResearchConfig,
    *,
    timeframe: str,
    start: str | None,
    end: str | None,
    observations: int,
) -> dict[str, Any]:
    """Everything needed to reproduce or date a report."""
    return {
        "generated_utc": utc_now_iso(),
        "timeframe": timeframe,
        "requested_start": start,
        "requested_end": end,
        "observations": observations,
        "git": _git_commit(config.project_root),
        "data_config_fingerprint": config.fingerprint(),
        "research_config_fingerprint": research.fingerprint(),
        "data_schema_version": config.schema_version,
        "research_schema_version": research.schema_version,
        "instrument": config.instrument,
        "timezone": {
            "mode": config.timezone.mode,
            "description": config.timezone.describe(),
            "time_basis": research.intraday.time_basis,
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
        "data_config_path": str(config.config_path),
        "research_config_path": str(research.config_path),
        "dataset": dataset_lineage(config, timeframe),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
@dataclass
class ResearchReport:
    """Result of analysing one timeframe."""

    timeframe: str
    output_dir: Path
    observations: int
    first_timestamp: str | None
    last_timestamp: str | None
    metrics: TimeframeMetrics | None = None
    files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    summary: dict[str, Any] = field(default_factory=dict)


class _Writer:
    """Writes a table in every configured format and records what it wrote."""

    def __init__(self, directory: Path, research: ResearchConfig, report: ResearchReport):
        self.directory = ensure_dir(directory)
        self.cfg = research.output
        self.report = report

    def table(self, frame: pl.DataFrame, name: str) -> None:
        if frame is None or frame.is_empty():
            return
        if self.cfg.write_parquet:
            path = self.directory / f"{name}.parquet"
            frame.write_parquet(path)
            self.report.files.append(path.name)
        if self.cfg.write_csv:
            path = self.directory / f"{name}.csv"
            frame.write_csv(path)
            self.report.files.append(path.name)

    def json(self, payload: dict[str, Any], name: str) -> None:
        if not self.cfg.write_json:
            return
        path = self.directory / f"{name}.json"
        atomic_write_text(path, json.dumps(payload, indent=2, default=str) + "\n")
        self.report.files.append(path.name)


def generate_report(
    config: Config,
    research: ResearchConfig,
    *,
    timeframe: str,
    start: str | None = None,
    end: str | None = None,
    make_plots: bool = True,
) -> ResearchReport:
    """Run the full analysis suite for one timeframe and write the results."""
    started = time.perf_counter()
    out_dir = research.results_dir(timeframe)
    bars = _load_bars(config, timeframe, start, end)

    report = ResearchReport(
        timeframe=timeframe,
        output_dir=out_dir,
        observations=bars.height,
        first_timestamp=str(bars["timestamp"].min()) if bars.height else None,
        last_timestamp=str(bars["timestamp"].max()) if bars.height else None,
    )
    if bars.height < 100:
        report.warnings.append(
            f"Only {bars.height} bars available for {timeframe}; analysis skipped."
        )
        return report

    writer = _Writer(out_dir, research, report)
    LOGGER.info("[%s] %s bars from %s to %s", timeframe, f"{bars.height:,}",
                report.first_timestamp, report.last_timestamp)

    series = prepare_returns(bars, timeframe, research)
    frame = series.frame
    return_column = series.return_column
    returns = series.returns
    tz_description = config.timezone.describe()

    summary: dict[str, Any] = {
        "provenance": provenance(
            config, research, timeframe=timeframe, start=start, end=end,
            observations=int(returns.len()),
        ),
        "returns": series.to_dict(),
        "bars": {
            "count": bars.height,
            "first_timestamp": report.first_timestamp,
            "last_timestamp": report.last_timestamp,
        },
    }

    # -- distribution ------------------------------------------------------
    dist = describe_distribution(
        returns, name=f"{timeframe}_{series.return_type}_returns",
        quantiles=research.distribution.quantiles,
    )
    if research.distribution.gaussian_reference:
        dist.tails = gaussian_tail_comparison(
            returns, sigma_levels=research.distribution.tail_sigma_levels
        )
    dist.normality = normality_tests(
        returns,
        jarque_bera=research.normality.jarque_bera,
        dagostino_k2=research.normality.dagostino_k2,
        large_sample_threshold=research.normality.large_sample_warning_threshold,
    )
    writer.table(summary_to_frame(dist), "distribution")
    if dist.tails:
        writer.table(pl.DataFrame(dist.tails), "gaussian_tail_comparison")
    summary["distribution"] = dist.to_dict()

    # -- autocorrelation ---------------------------------------------------
    max_lag = research.autocorrelation.max_lag
    level = research.autocorrelation.confidence_level
    acf_r = acf_returns(returns, max_lag=max_lag, confidence_level=level)
    acf_abs = acf_absolute_returns(returns, max_lag=max_lag, confidence_level=level)
    acf_sq = acf_squared_returns(returns, max_lag=max_lag, confidence_level=level)
    writer.table(acf_frame([acf_r, acf_abs, acf_sq]), "autocorrelation")

    import numpy as np

    raw = returns.to_numpy().astype(float)
    lb_rows = []
    for name, values in (
        ("returns", raw), ("abs_returns", np.abs(raw)), ("squared_returns", raw**2)
    ):
        lb_rows += ljung_box_test(
            values, lags=research.autocorrelation.ljung_box_lags, series_name=name,
            large_sample_threshold=research.normality.large_sample_warning_threshold,
        )
    writer.table(ljung_box_frame(lb_rows), "ljung_box")
    summary["autocorrelation"] = {
        "returns": acf_r.to_dict(),
        "abs_returns": acf_abs.to_dict(),
        "squared_returns": acf_sq.to_dict(),
        "ljung_box": [r.to_dict() for r in lb_rows],
    }

    # -- volatility --------------------------------------------------------
    annual = annualization_factor(timeframe, research)
    vol_windows = research.volatility.rolling_windows
    frame = frame.with_columns(
        [
            rolling_volatility(pl.col(return_column), int(w)).alias(f"vol_{w}")
            for w in vol_windows
        ]
        + [
            realized_volatility(
                pl.col(return_column), int(research.volatility.realized_vol_window)
            ).alias("realized_vol")
        ]
    )
    frame = frame.with_columns(
        pl.Series("ewma_vol", ewma_volatility(
            frame[return_column], lambda_=research.volatility.ewma_lambda
        ))
    )
    primary_window = int(vol_windows[min(1, len(vol_windows) - 1)])
    regimes = volatility_regimes(
        frame, return_column=return_column, volatility_column=f"vol_{primary_window}",
        quantiles=research.volatility.regime_quantiles, window=primary_window,
    )
    writer.table(regimes.table, "volatility_regimes")
    vol_cols = [f"vol_{w}" for w in vol_windows] + ["realized_vol", "ewma_vol"]
    writer.table(
        frame.select([c for c in vol_cols if c in frame.columns]).describe(),
        "volatility",
    )
    summary["volatility"] = {
        "annualization": annual.to_dict(),
        "rolling_windows": list(vol_windows),
        "ewma_lambda": research.volatility.ewma_lambda,
        "regimes": regimes.to_dict(),
    }

    # -- rolling statistics -------------------------------------------------
    rolling_table, specs = rolling_frame(
        frame.select(["timestamp", return_column]), column=return_column,
        config=research, time_column="timestamp",
    )
    writer.table(rolling_table.tail(50_000), "rolling_statistics")
    summary["rolling"] = {"specs": [s.to_dict() for s in specs]}

    # -- stationarity -------------------------------------------------------
    summary["stationarity"] = _stationarity_block(frame, series, research, writer)

    # -- conditional / reversal --------------------------------------------
    conditional = conditional_return_analysis(
        frame, timeframe=timeframe, return_column=return_column,
        config=research, price_column=series.price_column,
    )
    reversal = reversal_summary(conditional)
    writer.table(reversal, "extreme_return_analysis")
    summary["conditional"] = conditional.to_dict()
    report.warnings += conditional.warnings

    # -- intraday / weekday / session --------------------------------------
    hourly = weekday = sessions = None
    if research.intraday.enabled:
        hourly = hourly_analysis(
            frame, return_column=return_column, config=research,
            timezone_description=tz_description,
        )
        writer.table(hourly.table, "intraday")
        if research.intraday.weekday_enabled:
            weekday = weekday_analysis(
                frame, return_column=return_column, config=research,
                timezone_description=tz_description,
            )
            writer.table(weekday.table, "weekday")
            report.warnings += weekday.warnings
        if research.intraday.sessions.enabled:
            sessions = session_analysis(
                frame, return_column=return_column, config=research,
                timezone_description=tz_description,
            )
            writer.table(sessions.table, "session")
        summary["intraday"] = {
            "hourly": hourly.to_dict() if hourly else None,
            "weekday": weekday.to_dict() if weekday else None,
            "session": sessions.to_dict() if sessions else None,
        }

    # -- spread and activity ------------------------------------------------
    basis = series.time_basis
    spread = spread_analysis(
        frame, timeframe=timeframe, config=research, return_column=return_column,
        volatility_column=f"vol_{primary_window}", time_basis=basis,
    )
    if spread is not None:
        writer.table(spread.to_frame(), "spread_analysis")
        writer.table(spread.by_hour, "spread_by_hour")
        summary["spread"] = spread.to_dict()

    activity = activity_analysis(
        frame, timeframe=timeframe, config=research, time_basis=basis
    )
    if activity is not None:
        writer.table(activity.to_frame(), "activity")
        writer.table(activity.by_hour, "activity_by_hour")
        summary["activity"] = activity.to_dict()
        report.warnings += activity.warnings

    # -- structural stability ----------------------------------------------
    stability = stability_analysis(
        frame, return_column=return_column, config=research, time_basis=basis
    )
    writer.table(stability.table, "stability")
    summary["stability"] = stability.to_dict()
    report.warnings += stability.warnings

    # -- headline metrics ---------------------------------------------------
    report.metrics = timeframe_metrics(
        timeframe=timeframe, frame=frame, return_column=return_column,
        acf_result=acf_r, abs_acf_result=acf_abs, squared_acf_result=acf_sq,
        conditional=conditional, time_basis=basis,
    )
    summary["headline_metrics"] = report.metrics.to_dict()

    # -- plots --------------------------------------------------------------
    if make_plots and research.plots.enabled:
        report.warnings += _make_plots(
            research, timeframe, frame, returns, [acf_r, acf_abs, acf_sq],
            hourly, reversal, stability.table, tz_description, report,
        )

    summary["warnings"] = report.warnings
    writer.json(summary, "summary")
    report.summary = summary
    report.duration_seconds = time.perf_counter() - started
    LOGGER.info("[%s] report written to %s in %s", timeframe, out_dir,
                format_duration(report.duration_seconds))
    return report


def _stationarity_block(
    frame: pl.DataFrame, series: Any, research: ResearchConfig, writer: _Writer
) -> dict[str, Any]:
    """ADF/KPSS on each configured series, plus the segmented view."""
    price_column = series.price_column
    return_column = series.return_column
    available = {
        "price": frame[price_column],
        "log_price": frame.select(pl.col(price_column).log())[price_column],
        "log_returns": frame[return_column],
    }
    reports = []
    for name in research.stationarity.series:
        values = available.get(name)
        if values is None:
            continue
        reports.append(stationarity_report(values, series_name=name, config=research))

    segmented = segmented_stationarity(
        frame.with_columns(pl.col(price_column).log().alias("__log_price")),
        series_column=return_column, time_column="timestamp",
        series_name=return_column, config=research,
    )
    writer.table(segmented.to_frame(), "stationarity_segments")
    payload = {
        "series": [r.to_dict() for r in reports],
        "segmented": segmented.to_dict(),
        "note": (
            "ADF H0 = unit root (non-stationary); KPSS H0 = stationary. The nulls "
            "are opposites, so the two tests together are more informative than "
            "either alone."
        ),
    }
    writer.json(payload, "stationarity")
    return payload


def _make_plots(
    research: ResearchConfig, timeframe: str, frame: pl.DataFrame, returns: Any,
    acf_results: list[Any], hourly: Any, reversal: pl.DataFrame,
    stability_table: pl.DataFrame, tz_description: str, report: ResearchReport,
) -> list[str]:
    """Render every configured figure, tolerating individual failures."""
    warnings: list[str] = []
    plot_module.use_style(research)
    directory = research.plots_dir(timeframe)
    suffix = research.plots.format

    jobs = [
        ("return_histogram", lambda p: plot_module.plot_return_histogram(
            returns, path=p, config=research, timeframe=timeframe)),
        ("qq_plot", lambda p: plot_module.plot_qq(
            returns, path=p, config=research, timeframe=timeframe)),
        ("autocorrelation", lambda p: plot_module.plot_acf(
            acf_results, path=p, config=research, timeframe=timeframe)),
        ("rolling_volatility", lambda p: plot_module.plot_rolling_volatility(
            frame, path=p, config=research, timeframe=timeframe,
            columns=[c for c in frame.columns if c.startswith("vol_")])),
        ("conditional_returns", lambda p: plot_module.plot_conditional_returns(
            reversal, path=p, config=research, timeframe=timeframe)),
        ("stability", lambda p: plot_module.plot_yearly_comparison(
            stability_table, path=p, config=research, timeframe=timeframe)),
    ]
    if hourly is not None and not hourly.table.is_empty():
        jobs.append(("intraday_profile", lambda p: plot_module.plot_by_hour(
            hourly.table, path=p, config=research, timeframe=timeframe,
            timezone_description=tz_description)))

    for name, fn in jobs:
        path = directory / f"{name}.{suffix}"
        try:
            fn(path)
            report.files.append(f"plots/{path.name}")
        except Exception as exc:  # noqa: BLE001 - a failed plot must not lose the tables
            warnings.append(f"plot {name} failed: {type(exc).__name__}: {exc}")
    return warnings


# ---------------------------------------------------------------------------
# Multi-timeframe
# ---------------------------------------------------------------------------
def generate_multi_timeframe_summary(
    config: Config,
    research: ResearchConfig,
    reports: list[ResearchReport],
) -> Path | None:
    """Write the cross-timeframe comparison table.

    Deliberately has no "best timeframe" column: picking one needs a strategy
    and a cost model, neither of which exists at this stage.
    """
    metrics = [r.metrics for r in reports if r.metrics is not None]
    if not metrics:
        return None

    directory = ensure_dir(research.results_path)
    table = multi_timeframe_table(metrics)
    table.write_csv(directory / "multi_timeframe_comparison.csv")
    if research.output.write_parquet:
        table.write_parquet(directory / "multi_timeframe_comparison.parquet")

    payload = {
        "generated_utc": utc_now_iso(),
        "timeframes": [m.timeframe for m in metrics],
        "metrics": [m.to_dict() for m in metrics],
        "provenance": provenance(
            config, research, timeframe="all", start=None, end=None,
            observations=sum(m.observations for m in metrics),
        ),
        "notes": [
            "Comparison only. No timeframe is recommended: that requires a "
            "strategy and a transaction-cost model, which do not exist yet.",
            "Autocorrelations at short timeframes are estimated from far more "
            "observations, so their confidence bands are far tighter. Compare "
            "magnitudes, not significance, across timeframes.",
        ],
    }
    path = directory / "multi_timeframe_comparison.json"
    atomic_write_text(path, json.dumps(payload, indent=2, default=str) + "\n")
    return path


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load_bars(
    config: Config, timeframe: str, start: str | None, end: str | None
) -> pl.DataFrame:
    """Load bars for a timeframe, honouring an optional date filter.

    Uses a lazy scan with a pushed-down predicate so a date-filtered request
    never materialises the whole series.
    """
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
    raise ValueError(
        f"--{what}={value!r} is not a recognised date. Use YYYY-MM-DD or "
        "'YYYY-MM-DD HH:MM:SS'."
    )
