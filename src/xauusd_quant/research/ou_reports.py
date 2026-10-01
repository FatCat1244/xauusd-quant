"""Orchestration for the Ornstein-Uhlenbeck research (Prompt #4).

Runs the whole OU study for one ``(timeframe, regression window N, OU window M)``
and writes it to ``results/ou_research/<tf>/regression_<N>/ou_<M>/``.

Three series go through the identical pipeline:

``residual``             the XAUUSD rolling-regression residual
``control_random_walk``  the same regression's residual on a random walk with
                         the same bar count and step size - no reversion at all
``reference_ou``         an exact OU with the residual's whole-sample parameters -
                         what a constant OU would look like when estimated
                         exactly this way

They are processed one after another and only their summary tables are kept,
so peak memory is one series' per-bar table, not three.

``summary.json`` carries the machine-readable ``model_summary`` block the next
stage consumes, and full provenance - including the dataset lineage (raw
fingerprint, tick / bar / feature versions and coverage), so a number computed
from an incomplete dataset says so in its own file.
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..data.versioning import PARTIAL_LABEL, dataset_lineage
from ..features.config import RegressionConfig
from ..features.rolling_regression import rolling_regression_features
from ..features.store import RegressionFeatureStore, feature_fingerprint
from ..models.config import OUConfig
from ..utils.clock import utc_now_iso
from ..utils.config import Config
from ..utils.logging import format_duration, get_logger
from ..utils.paths import atomic_write_text, ensure_dir
from . import ou_plots
from .config import ResearchConfig
from .ou_conditional import (
    attach_conditioning,
    intraday_conditioning,
    r2_conditioning,
    trend_conditioning,
    volatility_conditioning,
)
from .ou_decay import (
    extreme_events,
    extreme_summary,
    first_passage_summary,
    forecast_evaluation,
    realized_decay_comparison,
)
from .ou_diagnostics import innovation_diagnostics
from .ou_estimation import (
    b_classification,
    control_series,
    control_step_std,
    equilibrium_summary,
    expected_decay_table,
    fit_row,
    half_life_summary,
    ou_parameter_frame,
    parameter_distribution,
    random_walk_features,
    static_fit,
    zscore_comparison,
)
from .ou_stability import (
    era_table,
    half_life_stability,
    recent_versus_history,
    rolling_half_life_profile,
    segment_table,
)
from .regression_reports import _git_commit, _package_versions, load_bars

__all__ = [
    "OUReport",
    "generate_ou_report",
    "load_ou_reports",
    "reusable_ou_report",
    "write_ou_comparison",
]

LOGGER = get_logger("research.ou_reports")

CAVEAT = (
    "Descriptive research only. An OU fit is not evidence of profitability: no "
    "transaction cost, entry, exit or position exists anywhere in this layer. Every "
    "OU statistic of a rolling-regression residual must be read against "
    "control_random_walk (the same pipeline on a detrended random walk, which shows "
    "OU-like reversion purely from detrending) and reference_ou (an exact OU with the "
    "fitted parameters, which shows what estimation noise alone produces). Half-lives "
    "are estimates with wide, reported dispersion - not laws of the market."
)

#: Version of what a study writes. --resume reuses only studies of this version.
#: 2: the volatility / trend / R^2 conditioning tables include the random-walk
#:    control, conditioned on its own slope, R^2 and volatility.
OU_REPORT_VERSION = 2

#: Sources drawn in the estimated-vs-realised half-decay scatter.
SCATTER_SOURCES: tuple[str, ...] = ("residual", "reference_ou")

#: The regression-feature columns the OU study reads; nothing else is loaded.
FEATURE_COLUMNS: tuple[str, ...] = (
    "timestamp", "residual", "residual_zscore_rolling", "regression_slope",
    "r_squared", "trailing_volatility", "log_return",
)


@dataclass
class OUReport:
    """Outcome of one (timeframe, N, M) OU study."""

    timeframe: str
    regression_window: int
    ou_window: int
    output_dir: Path
    observations: int = 0
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    model_summary: dict[str, Any] = field(default_factory=dict)
    files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    #: Read back from a current summary.json (``--resume``), not recomputed.
    reused: bool = False


class _Writer:
    def __init__(self, directory: Path, ou: OUConfig, report: OUReport) -> None:
        self.directory = ensure_dir(directory)
        self.cfg = ou.output
        self.report = report

    def table(self, frame: pl.DataFrame | None, name: str, *, csv: bool = True) -> None:
        if frame is None or frame.is_empty():
            return
        frame = _typed(frame)
        if self.cfg.write_parquet:
            frame.write_parquet(self.directory / f"{name}.parquet")
            self.report.files.append(f"{name}.parquet")
        if self.cfg.write_csv and csv:
            frame.write_csv(self.directory / f"{name}.csv")
            self.report.files.append(f"{name}.csv")

    def json(self, payload: dict[str, Any], name: str) -> None:
        if not self.cfg.write_json:
            return
        atomic_write_text(self.directory / f"{name}.json",
                          json.dumps(payload, indent=2, default=_json_default) + "\n")
        self.report.files.append(f"{name}.json")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return str(value)


def _scatter_events(events: pl.DataFrame, level: float) -> pl.DataFrame:
    """The part of an event table the half-decay scatter reads, and nothing more."""
    if events.is_empty():
        return events
    return events.select("abs_z", "fp_50", "half_life_estimated").filter(
        pl.col("abs_z") > level)


def _typed(frame: pl.DataFrame) -> pl.DataFrame:
    """Give every all-null column a concrete type before it reaches disk.

    A statistic undefined for every row of a table (a censored ratio quantile,
    say) is built as dtype ``Null``. Every such column in these tables is an
    optional number, so it is written as Float64 and a reader stacking tables
    from several studies never meets a ``Null`` column.
    """
    nulls = [name for name, dtype in frame.schema.items() if dtype == pl.Null]
    return frame.with_columns(pl.col(nulls).cast(pl.Float64)) if nulls else frame


def _stack(parts: list[Any]) -> pl.DataFrame:
    """Concatenate one table's per-source parts, whose columns may differ.

    ``diagonal_relaxed`` because a statistic can be undefined for every row
    of one source and defined for another's: the fully censored source then
    carries a ``Null`` column that a strict diagonal concat refuses to stack on
    the other's Float64 (the failure that stopped the first full grid).
    """
    frames = [p for p in parts if isinstance(p, pl.DataFrame) and not p.is_empty()]
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def _clean(value: Any) -> Any:
    """Recursively replace non-finite floats by None so JSON stays strict."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _bars_per_day(timeframe: str) -> int:
    from ..data.resampler import parse_timeframe

    return max(1, int(round(86_400 / parse_timeframe(timeframe).total_seconds())))


# ---------------------------------------------------------------------------
# One series through the pipeline
# ---------------------------------------------------------------------------
def _series_tables(
    frame: pl.DataFrame, *, source: str, ou: OUConfig, ou_window: int,
    structural_lags: dict[str, int], static_residuals: np.ndarray | None,
) -> dict[str, Any]:
    """Everything computed identically for the real residual and every control."""
    tables: dict[str, Any] = {
        "parameter_distribution": parameter_distribution(frame, source=source),
        "half_life": half_life_summary(frame, source=source, config=ou),
        "b_classification": b_classification(frame, source=source, config=ou),
        "expected_decay": expected_decay_table(frame, source=source, config=ou),
        "forecast": forecast_evaluation(frame, source=source, config=ou),
        "stability": half_life_stability(frame, ou_window=ou_window, source=source, config=ou),
        "zscores": zscore_comparison(frame, source=source),
    }
    events = extreme_events(frame, config=ou)
    tables["events"] = events
    tables["extremes"] = extreme_summary(events, source=source, config=ou)
    tables["realized"] = realized_decay_comparison(events, source=source, config=ou)
    tables["first_passage"] = first_passage_summary(events, source=source, config=ou)
    valid_prev = np.concatenate(([False], frame["ou_valid"].to_numpy()[:-1]))
    series = {
        "rolling_one_step": np.where(
            valid_prev, frame["ou_innovation"].fill_null(np.nan).to_numpy(), np.nan),
        "rolling_one_step_standardized": np.where(
            valid_prev, frame["ou_innovation_standardized"].fill_null(np.nan).to_numpy(),
            np.nan),
    }
    if static_residuals is not None:
        series["static_in_sample"] = static_residuals
    summary, acf = innovation_diagnostics(series, source=source, config=ou,
                                          structural_lags=structural_lags)
    tables["innovations"] = summary
    tables["innovation_acf"] = acf
    tables["half_life_values"] = frame["ou_half_life_bars"].drop_nulls()
    return tables


def _headline(tables: dict[str, Any], fit_row_: dict[str, Any], *, primary: float) -> dict[str, Any]:
    """The Step 41 metrics for one series."""
    hl, cls = tables["half_life"], tables["b_classification"]
    params = {r["parameter"]: r for r in tables["parameter_distribution"].iter_rows(named=True)}

    def median(name: str) -> float | None:
        return params.get(name, {}).get("p50")

    innovations = tables["innovations"]
    lag1 = None
    if not innovations.is_empty() and "acf1" in innovations.columns:
        hit = innovations.filter(pl.col("series") == "rolling_one_step")
        lag1 = hit["acf1"][0] if hit.height else None
    realized = tables["realized"]
    corr = ratio = None
    if not realized.is_empty():
        hit = realized.filter((pl.col("abs_z_level") == primary)
                              & (pl.col("subset") == "all starts"))
        if hit.height:
            corr = hit["spearman_realized_vs_estimated"][0]
            ratio = hit["median_ratio"][0]
    forecast = tables["forecast"]
    skill = None
    if not forecast.is_empty():
        hit = forecast.filter((pl.col("horizon") == 5) & (pl.col("subset") == "all"))
        skill = hit["skill_vs_persistence"][0] if hit.height else None
    return {
        "windows_estimated": cls.get("windows_estimated"),
        "valid_ou_fraction": cls.get("valid_ou_fraction"),
        "fraction_b_ge_1": cls.get("fraction_b_ge_1"),
        "fraction_b_le_0": cls.get("fraction_b_le_0"),
        "near_unit_root_fraction": cls.get("near_unit_root_fraction"),
        "median_b": median("b"),
        "median_theta": median("theta"),
        "median_mu": median("mu"),
        "median_sigma": median("sigma"),
        "median_half_life_bars": hl.get("median"),
        "half_life_p25": hl.get("p25"),
        "half_life_p75": hl.get("p75"),
        "half_life_p5": hl.get("p5"),
        "half_life_p95": hl.get("p95"),
        "median_ou_r_squared": median("r_squared"),
        "median_std_ratio": median("std_ratio"),
        "innovation_lag1_acf": lag1,
        "estimated_vs_realized_half_life_correlation": corr,
        "realized_to_estimated_half_life_median_ratio": ratio,
        "forecast_skill_vs_persistence_h5": skill,
        "half_life_disjoint_median_abs_log_change": tables["stability"].get(
            "disjoint_median_abs_log_change"),
        "static_b": fit_row_.get("b"),
        "static_half_life_bars": fit_row_.get("half_life_bars"),
        "static_state": fit_row_.get("state"),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def generate_ou_report(
    config: Config,
    regression: RegressionConfig,
    research: ResearchConfig,
    ou: OUConfig,
    *,
    timeframe: str,
    regression_window: int,
    ou_window: int,
    features: pl.DataFrame | None = None,
    start: str | None = None,
    end: str | None = None,
    make_plots: bool = True,
    write_parameters: bool | None = None,
) -> OUReport:
    """Run the full OU study for one (timeframe, N, M) and write every table."""
    started = time.perf_counter()
    out_dir = ou.results_dir(timeframe, regression_window, ou_window)
    report = OUReport(timeframe=timeframe, regression_window=regression_window,
                      ou_window=ou_window, output_dir=out_dir)
    lineage = dataset_lineage(config, timeframe)
    filtered = start is not None or end is not None
    feature_version = None
    if features is None:
        if filtered:
            bars = load_bars(config, timeframe, start, end)
            features, _ = rolling_regression_features(
                bars, window=regression_window, config=regression, timeframe=timeframe)
            del bars
        else:
            store = RegressionFeatureStore(config, regression)
            features = store.load_or_build(timeframe, regression_window,
                                           columns=FEATURE_COLUMNS)
            entry = ((store.manifest(timeframe) or {}).get("windows") or {}).get(
                str(regression_window)) or {}
            feature_version = entry.get("feature_version")
    if lineage.get("partial") or filtered:
        label = PARTIAL_LABEL if lineage.get("partial") else "DATE-FILTERED SUBSET"
        report.warnings.append(
            f"{label}: computed from {lineage.get('tick_dataset_status')} ticks"
            + (f" restricted to {start}..{end}" if filtered else "")
            + ". Not a statement about the full 2003-2026 history."
        )

    # Only what the OU study reads (the cache already returns just these).
    features = features.select([c for c in FEATURE_COLUMNS if c in features.columns])
    residual = features["residual"].fill_null(np.nan).to_numpy().astype(np.float64)
    report.observations = int(np.isfinite(residual).sum())
    report.first_timestamp = str(features["timestamp"].min())
    report.last_timestamp = str(features["timestamp"].max())
    if report.observations < ou_window * 3:
        report.warnings.append(
            f"Only {report.observations} residuals for an OU window of {ou_window}; skipped.")
        return report
    LOGGER.info("[%s N=%d M=%d] %s residuals, %s -> %s", timeframe, regression_window,
                ou_window, f"{report.observations:,}", report.first_timestamp,
                report.last_timestamp)

    writer = _Writer(out_dir, ou, report)
    structural = {
        "regression_window": regression_window,
        "ou_window": ou_window,
        "one_day": _bars_per_day(timeframe),
    }
    primary = ou.extremes.primary_abs_z
    log_returns = features["log_return"].fill_null(np.nan).to_numpy() if (
        "log_return" in features.columns) else np.diff(residual, prepend=np.nan)

    # -- the real residual --------------------------------------------------
    frame = ou_parameter_frame(residual, features["timestamp"], ou_window=ou_window, config=ou)
    frame = frame.with_columns(features["residual_zscore_rolling"]).pipe(
        attach_conditioning, features, ou_window=ou_window)
    del features        # the frame now carries every column the study reads
    fit = static_fit(residual, ou)
    fits = [fit_row(fit, source="residual")]
    real = _series_tables(frame, source="residual", ou=ou, ou_window=ou_window,
                          structural_lags=structural, static_residuals=fit.residuals)
    headlines = {"residual": _headline(real, fits[0], primary=primary)}
    collected: dict[str, list[Any]] = {k: [v] for k, v in real.items()
                                       if isinstance(v, pl.DataFrame) and k != "events"}
    collected["half_life"] = [real["half_life"]]
    collected["b_classification"] = [real["b_classification"]]
    collected["stability"] = [real["stability"]]
    hl_values = {"residual": real["half_life_values"]}

    equilibrium = equilibrium_summary(frame, fit, config=ou)
    years = segment_table(frame, by="year", config=ou) if ou.segmented_estimation.yearly else None
    quarters = (segment_table(frame, by="quarter", config=ou)
                if ou.segmented_estimation.quarterly else None)
    eras = era_table(frame, config=ou)
    recent = recent_versus_history(frame)
    conditioning = {
        "volatility_conditioning": volatility_conditioning(frame, source="residual", config=ou),
        "trend_conditioning": trend_conditioning(frame, source="residual", config=ou),
        "r2_conditioning": r2_conditioning(frame, source="residual", config=ou),
        "intraday_conditioning": intraday_conditioning(frame, source="residual", config=ou,
                                                       research=research),
    }
    profile = rolling_half_life_profile(frame, config=ou)

    write = write_parameters
    if write is None:
        policy = ou.rolling_estimation.write_parameters
        rep = ou.representative
        write = policy == "all" or (
            policy == "representative" and regression_window == rep.regression_window
            and ou_window == rep.ou_window)
    if write:
        writer.table(frame.select(
            [c for c in frame.columns if c not in ("log_return",)]), "parameters", csv=False)
    if ou.extremes.write_events and not real["events"].is_empty():
        writer.table(real["events"].filter(pl.col("abs_z") > primary), "extreme_events",
                     csv=False)
    # From here on only what is still to be read is kept: at 23 years of
    # 1-minute bars the frame, the profile and each source's events are each
    # hundreds of MB, and the controls below need room for their own frames.
    events_by_source = {"residual": _scatter_events(real["events"], primary)}
    del real

    if make_plots and ou.plots.enabled:
        report.warnings += _plots_real(frame, profile, years, ou, report, timeframe,
                                       regression_window, ou_window)
    writer.table(profile.drop_nulls().gather_every(max(1, profile.height // 20_000)),
                 "rolling_half_life_profile", csv=False)
    del frame, profile
    # Drawing leaves a reference cycle behind: matplotlib's tight_layout/savefig
    # keep the plotting calls' execution frames alive, and with them the frame
    # they drew (~0.5 GB at 5m, ~2.4 GB at 1m). Without a full collection it
    # stayed resident through every control below.
    gc.collect()

    # -- controls and reference, one at a time ------------------------------
    for offset, name in enumerate(ou.controls.enabled()):
        seed = ou.controls.seed + offset
        extras: pl.DataFrame | None = None
        series: np.ndarray | None
        if name == "control_random_walk":
            # The very walk control_series builds, with its own slope, R^2 and
            # volatility, so conditioning is read against the control too.
            extras = random_walk_features(
                residual.size, regression_window=regression_window,
                step_std=control_step_std(log_returns), seed=seed,
                volatility_window=regression.conditioning.volatility_window,
            )
            series = extras["residual"].fill_null(np.nan).to_numpy()
        else:
            series = control_series(
                name, residual.size, regression_window=regression_window,
                log_returns=log_returns, reference_fit=fit, seed=seed,
            )
        if series is None:
            report.warnings.append(f"{name}: not built (the whole-sample fit is not a valid OU).")
            continue
        control_frame = ou_parameter_frame(series, None, ou_window=ou_window, config=ou)
        if extras is not None:
            conditioned = control_frame.pipe(attach_conditioning, extras, ou_window=ou_window)
            for key, build in (("volatility_conditioning", volatility_conditioning),
                               ("trend_conditioning", trend_conditioning),
                               ("r2_conditioning", r2_conditioning)):
                conditioning[key] = _stack([conditioning[key],
                                            build(conditioned, source=name, config=ou)])
            del conditioned, extras
        control_fit = static_fit(series, ou)
        fits.append(fit_row(control_fit, source=name))
        tables = _series_tables(control_frame, source=name, ou=ou, ou_window=ou_window,
                                structural_lags=structural,
                                static_residuals=control_fit.residuals)
        headlines[name] = _headline(tables, fits[-1], primary=primary)
        for key, value in tables.items():
            if isinstance(value, pl.DataFrame) and key != "events":
                collected.setdefault(key, []).append(value)
        collected["half_life"].append(tables["half_life"])
        collected["b_classification"].append(tables["b_classification"])
        collected["stability"].append(tables["stability"])
        if name in SCATTER_SOURCES:
            events_by_source[name] = _scatter_events(tables["events"], primary)
        hl_values[name] = tables["half_life_values"]
        del control_frame, series, tables
        gc.collect()

    # -- tables -------------------------------------------------------------
    def stack(key: str) -> pl.DataFrame:
        return _stack(collected.get(key, []))

    def rows(key: str) -> pl.DataFrame:
        return pl.DataFrame([_clean(r) for r in collected.get(key, [])], infer_schema_length=None)

    writer.table(pl.DataFrame([_clean(r) for r in fits], infer_schema_length=None), "static_fit")
    writer.table(stack("parameter_distribution"), "parameter_distribution")
    writer.table(rows("half_life"), "half_life")
    writer.table(rows("b_classification"), "b_classification")
    writer.table(rows("stability"), "half_life_stability")
    writer.table(stack("expected_decay"), "expected_decay")
    writer.table(stack("zscores"), "zscore_comparison")
    writer.table(stack("forecast"), "forecast_evaluation")
    writer.table(stack("extremes"), "extreme_deviations")
    writer.table(stack("realized"), "realized_decay_comparison")
    writer.table(stack("first_passage"), "first_passage")
    writer.table(stack("innovations"), "innovation_diagnostics")
    writer.table(stack("innovation_acf"), "innovation_acf")
    writer.table(years, "yearly_stability")
    writer.table(quarters, "quarterly_stability")
    writer.table(eras, "era_stability")
    writer.table(pl.DataFrame([_clean(equilibrium)]), "equilibrium")
    for name, table in conditioning.items():
        writer.table(table, name)

    if make_plots and ou.plots.enabled:
        report.warnings += _plots_comparison(events_by_source, hl_values, stack("innovation_acf"),
                                             ou, report, timeframe, regression_window,
                                             ou_window)

    # -- summary ------------------------------------------------------------
    model_summary = {
        "timeframe": timeframe,
        "regression_window": regression_window,
        "ou_window": ou_window,
        "observations": report.observations,
        "first_timestamp": report.first_timestamp,
        "last_timestamp": report.last_timestamp,
        **headlines["residual"],
        "static_mu": equilibrium.get("static_mu"),
        "static_mu_ci": [equilibrium.get("static_mu_ci_lower"),
                         equilibrium.get("static_mu_ci_upper")],
        "equilibrium_flag_for_investigation": equilibrium.get("static_flag_for_investigation"),
        "control_random_walk": headlines.get("control_random_walk"),
        "reference_ou": headlines.get("reference_ou"),
        "partial_dataset": bool(lineage.get("partial") or filtered),
        "dataset_label": lineage.get("label"),
        "caveat": CAVEAT,
    }
    summary = {
        "model_summary": model_summary,
        "provenance": {
            "generated_utc": utc_now_iso(),
            "git": _git_commit(config.project_root),
            "dataset": lineage,
            "report_version": OU_REPORT_VERSION,
            "regression_feature_version": feature_version,
            "regression_feature_fingerprint": feature_fingerprint(regression),
            "requested_start": start,
            "requested_end": end,
            "ou_config_fingerprint": ou.fingerprint(),
            "regression_config_fingerprint": regression.fingerprint(),
            "research_config_fingerprint": research.fingerprint(),
            "data_config_fingerprint": config.fingerprint(),
            "estimation_method": list(ou.estimation_methods),
            "dt": ou.dt,
            "packages": _package_versions(),
        },
        "static_fits": fits,
        "equilibrium": equilibrium,
        "recent_versus_history": recent,
        "half_life": collected["half_life"],
        "b_classification": collected["b_classification"],
        "stability": collected["stability"],
        "controls": {
            "built": list(headlines),
            "description": {
                "control_random_walk": "same rolling regression on a Gaussian random walk "
                                       "with the same bar count and the data's log-return std",
                "reference_ou": "exact OU simulation with the residual's whole-sample "
                                "theta, mu and sigma, estimated with the same windows",
            },
        },
        "notes": [
            "Rolling estimates at t use residuals t-M+1..t only; forward values appear "
            "only in outcome tables (forecast, extremes, realized decay, first passage).",
            "Consecutive rolling windows share M-1 of M observations, and consecutive "
            "extreme starts share most of their future: effective sample sizes are far "
            "below the counts shown. 'episode starts' rows keep one start per excursion.",
            "Half-lives above the reportable cap are censored, not dropped: see "
            "censored_above_cap and the p.._above_cap flags.",
            "A bar is a bar: weekends and the daily break are one step like any other, "
            "because the regression is defined on bar counts.",
        ],
        "warnings": report.warnings,
    }
    writer.json(_clean(summary), "summary")
    report.model_summary = _clean(model_summary)
    report.duration_seconds = time.perf_counter() - started
    LOGGER.info("[%s N=%d M=%d] written to %s in %s", timeframe, regression_window, ou_window,
                out_dir, format_duration(report.duration_seconds))
    return report


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _plots_real(frame: pl.DataFrame, profile: pl.DataFrame, years: pl.DataFrame | None,
                ou: OUConfig, report: OUReport, timeframe: str, n: int, m: int) -> list[str]:
    warnings: list[str] = []
    directory = ou.plots_dir(timeframe, n, m)
    tag = f"{timeframe}  N={n}  M={m}"
    std = frame["ou_innovation_standardized"].fill_null(np.nan).to_numpy()
    jobs = [
        ("rolling_half_life", lambda p: ou_plots.plot_rolling_half_life(
            frame, profile, path=p, config=ou, title=f"{tag}  rolling OU half-life")),
        ("rolling_theta", lambda p: ou_plots.plot_rolling_theta(
            frame, path=p, config=ou, title=f"{tag}  rolling theta")),
        ("rolling_b", lambda p: ou_plots.plot_rolling_b(
            frame, path=p, config=ou, title=f"{tag}  rolling AR(1) coefficient")),
        ("ou_equilibrium", lambda p: ou_plots.plot_ou_equilibrium(
            frame, path=p, config=ou, title=f"{tag}  residual and rolling equilibrium")),
        ("ou_zscore", lambda p: ou_plots.plot_ou_zscore(
            frame, path=p, config=ou, title=f"{tag}  OU Z-score")),
        ("innovation_distribution", lambda p: ou_plots.plot_innovation_distribution(
            std, path=p, config=ou, title=f"{tag}  standardised one-step innovations")),
    ]
    if years is not None and not years.is_empty():
        jobs.append(("parameter_stability_by_year", lambda p: ou_plots.plot_parameter_stability(
            years, path=p, config=ou, title=f"{tag}  OU half-life by year")))
    for name, job in jobs:
        path = directory / f"{name}.{ou.plots.format}"
        try:
            job(path)
            report.files.append(f"plots/{path.name}")
        except Exception as exc:  # noqa: BLE001 - a failed plot must not lose the tables
            warnings.append(f"plot {name} failed: {type(exc).__name__}: {exc}")
    return warnings


def _plots_comparison(events: dict[str, pl.DataFrame], hl_values: dict[str, pl.Series],
                      acf: pl.DataFrame, ou: OUConfig, report: OUReport, timeframe: str,
                      n: int, m: int) -> list[str]:
    warnings: list[str] = []
    directory = ou.plots_dir(timeframe, n, m)
    tag = f"{timeframe}  N={n}  M={m}"
    frames = {k: pl.DataFrame({"ou_half_life_bars": v}) for k, v in hl_values.items()}
    jobs = [
        ("estimated_vs_realized_half_decay", lambda p: ou_plots.plot_estimated_vs_realized(
            {k: v for k, v in events.items() if k in SCATTER_SOURCES},
            path=p, config=ou, title=f"{tag}  estimated half-life vs realised half-decay")),
        ("half_life_distribution", lambda p: ou_plots.plot_half_life_distribution(
            frames, path=p, config=ou, title=f"{tag}  rolling half-life: residual vs controls")),
        ("innovation_acf", lambda p: ou_plots.plot_innovation_acf(
            acf, series="rolling_one_step", path=p, config=ou,
            title=f"{tag}  one-step innovation autocorrelation")),
    ]
    for name, job in jobs:
        path = directory / f"{name}.{ou.plots.format}"
        try:
            job(path)
            report.files.append(f"plots/{path.name}")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"plot {name} failed: {type(exc).__name__}: {exc}")
    return warnings


# ---------------------------------------------------------------------------
# Cross-model comparison
# ---------------------------------------------------------------------------
def _read_summary(path: Path) -> dict[str, Any] | None:
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return summary if isinstance(summary, dict) and summary.get("model_summary") else None


def _report_from_summary(path: Path, summary: dict[str, Any], *,
                         reused: bool = False) -> OUReport:
    model = summary["model_summary"]
    return OUReport(
        timeframe=model["timeframe"], regression_window=int(model["regression_window"]),
        ou_window=int(model["ou_window"]), output_dir=path.parent,
        observations=int(model.get("observations") or 0),
        first_timestamp=model.get("first_timestamp"), last_timestamp=model.get("last_timestamp"),
        model_summary=model, warnings=list(summary.get("warnings") or []), reused=reused,
    )


def load_ou_reports(ou: OUConfig) -> list[OUReport]:
    """Every OU study already on disk, rebuilt from its summary.json."""
    reports: list[OUReport] = []
    for path in sorted(ou.results_path.glob("*/regression_*/ou_*/summary.json")):
        summary = _read_summary(path)
        if summary is not None:
            reports.append(_report_from_summary(path, summary))
    return reports


def reusable_ou_report(
    config: Config, regression: RegressionConfig, research: ResearchConfig, ou: OUConfig,
    *, timeframe: str, regression_window: int, ou_window: int,
    need_parameters: bool = False, need_plots: bool = False,
) -> OUReport | None:
    """The study already on disk for (timeframe, N, M) if it is current, else None.

    Current means: computed over the full history (no date filter, not a
    partial dataset), from the same raw file, tick, bar and feature versions,
    under the same four configurations a fresh run would use, and holding the
    files this run was asked for. Anything else is recomputed, so ``--resume``
    cannot pass a stale study off as a new one. Code changes that alter what a
    study writes bump ``OU_REPORT_VERSION``; any other change that alters values
    needs a run without ``--resume``.
    """
    directory = ou.results_dir(timeframe, regression_window, ou_window)
    path = directory / "summary.json"
    summary = _read_summary(path)
    if summary is None or summary["model_summary"].get("partial_dataset"):
        return None
    lineage = dataset_lineage(config, timeframe)
    if lineage.get("partial"):
        return None
    recorded = summary.get("provenance") or {}
    if recorded.get("report_version") != OU_REPORT_VERSION:
        return None
    dataset = recorded.get("dataset") or {}
    for key in ("raw_fingerprint", "tick_dataset_version", "bar_dataset_version"):
        if lineage.get(key) is None or dataset.get(key) != lineage.get(key):
            return None
    store = RegressionFeatureStore(config, regression)
    entry = ((store.manifest(timeframe) or {}).get("windows") or {}).get(
        str(regression_window)) or {}
    expected = {
        "regression_feature_version": entry.get("feature_version"),
        "regression_feature_fingerprint": feature_fingerprint(regression),
        "requested_start": None,
        "requested_end": None,
        "ou_config_fingerprint": ou.fingerprint(),
        "regression_config_fingerprint": regression.fingerprint(),
        "research_config_fingerprint": research.fingerprint(),
        "data_config_fingerprint": config.fingerprint(),
    }
    if entry.get("feature_version") is None or any(
            recorded.get(key) != value for key, value in expected.items()):
        return None
    if need_parameters and not (directory / "parameters.parquet").exists():
        return None
    if need_plots and not any(ou.plots_dir(timeframe, regression_window, ou_window).glob("*.png")):
        return None
    return _report_from_summary(path, summary, reused=True)


def write_ou_comparison(config: Config, ou: OUConfig, reports: list[OUReport]) -> list[Path]:
    """Multi-timeframe table (Step 31) and the N x M window matrix (Step 32).

    No column ranks anything. The question is whether a finding survives
    neighbouring settings, not where it is largest.
    """
    summaries = [r.model_summary for r in reports if r.model_summary]
    if not summaries:
        return []
    directory = ensure_dir(ou.results_path)
    written: list[Path] = []
    flat: list[dict[str, Any]] = []
    for s in summaries:
        row = {k: v for k, v in s.items()
               if not isinstance(v, (dict, list)) and k not in ("caveat",)}
        for source in ("control_random_walk", "reference_ou"):
            for key, value in (s.get(source) or {}).items():
                if not isinstance(value, (dict, list)):
                    row[f"{source}__{key}"] = value
        flat.append(row)
    long = pl.DataFrame(flat, infer_schema_length=None)
    long.write_csv(directory / "ou_comparison.csv")
    long.write_parquet(directory / "ou_comparison.parquet")
    written += [directory / "ou_comparison.csv", directory / "ou_comparison.parquet"]

    metrics = ["valid_ou_fraction", "median_b", "median_theta", "median_half_life_bars",
               "half_life_p25", "half_life_p75", "median_sigma", "median_ou_r_squared",
               "near_unit_root_fraction", "innovation_lag1_acf",
               "estimated_vs_realized_half_life_correlation",
               "realized_to_estimated_half_life_median_ratio",
               "forecast_skill_vs_persistence_h5"]
    rep = ou.representative
    representative = long.filter((pl.col("regression_window") == rep.regression_window)
                                 & (pl.col("ou_window") == rep.ou_window))
    if not representative.is_empty():
        table_rows = []
        for metric in metrics:
            for prefix, label in (("", "residual"), ("control_random_walk__", "control"),
                                  ("reference_ou__", "reference")):
                column = f"{prefix}{metric}"
                if column not in representative.columns:
                    continue
                entry: dict[str, Any] = {"metric": metric, "series": label}
                for rec in representative.iter_rows(named=True):
                    entry[rec["timeframe"]] = rec.get(column)
                table_rows.append(entry)
        path = directory / "timeframe_comparison.csv"
        pl.DataFrame(table_rows, infer_schema_length=None).write_csv(path)
        written.append(path)

    matrix_columns = ["timeframe", "regression_window", "ou_window", "observations",
                      *[m for m in metrics if m in long.columns],
                      *[c for c in long.columns if c.startswith("control_random_walk__")
                        and c.split("__")[1] in ("valid_ou_fraction", "median_half_life_bars",
                                                 "median_ou_r_squared")]]
    matrix = long.select([c for c in matrix_columns if c in long.columns]).sort(
        "timeframe", "regression_window", "ou_window")
    path = directory / "window_matrix.csv"
    matrix.write_csv(path)
    written.append(path)
    if ou.plots.enabled:
        for timeframe in matrix["timeframe"].unique().to_list():
            part = matrix.filter(pl.col("timeframe") == timeframe)
            for metric in ("median_half_life_bars", "valid_ou_fraction", "median_ou_r_squared"):
                if metric not in part.columns or part[metric].null_count() == part.height:
                    continue
                target = directory / "plots" / f"window_matrix_{timeframe}_{metric}.png"
                try:
                    ou_plots.plot_window_matrix(part, metric=metric, path=target, config=ou,
                                                title=f"{timeframe}  {metric}: regression "
                                                      "window x OU window")
                    written.append(target)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("window matrix plot failed: %s", exc)

    payload = {
        "generated_utc": utc_now_iso(),
        "dataset": dataset_lineage(config),
        "models": summaries,
        "notes": [
            "No window or timeframe is ranked or recommended; choosing one needs a "
            "strategy and a cost model, neither of which exists.",
            "Read every residual metric beside its control_random_walk__ and "
            "reference_ou__ counterpart.",
        ],
    }
    path = directory / "ou_comparison.json"
    atomic_write_text(path, json.dumps(_clean(payload), indent=2, default=_json_default) + "\n")
    written.append(path)
    return written
