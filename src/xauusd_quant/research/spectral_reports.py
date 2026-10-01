"""Orchestration of the spectral study (Prompt #5).

One timeframe at a time: the integrity gate, then every source (the real data
and each null control, one in memory at a time), every input series and
every FFT window. Per (timeframe, series, window) study the outputs go to
``results/spectral_research/<tf>/<series>/fft_<N>/``:

``summary.json``                   machine-readable digest + full provenance
``spectral_distribution``          quantiles of every spectral metric, by source
``entropy_summary``                entropy, flatness and concentration, by source
``power_bands``                    band-share distributions, by source
``mean_spectrum``                  average power share per frequency bin, by source
``dominant_period_distribution``   how often each bin is the dominant one, by source
``dominant_period_stability``      switching, run length, persistence at lags
``frequency_persistence``          top-K overlap between windows h bars apart
``spectral_changes``               distributions of metric changes
``yearly/quarterly/monthly/era_stability``
``conditioning``, ``intraday``     by volatility, trend, OU speed; hour, session
``residual_outcomes``              entropy / concentration buckets vs residual decay
``phase_analysis``, ``phase_consistency``, ``phase_projection``
``reconstruction_metrics``, ``extrapolation_metrics``
``ic_analysis``, ``ic_by_period``  information coefficients, with per-period detail
``feature_correlation``, ``feature_redundancy``
``null_control_comparison``        every headline metric, real beside each null
``plots/``

Every hypothesis evaluated is also written to the research ledger with its
verdict. The verdict rules below were fixed before any real result existed;
they compare the real series with the null controls, never with zero.
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
from ..features.spectral import RollingSpectrum, rolling_spectrum, spectral_feature_frame
from ..features.spectral_bands import usable_bins
from ..features.spectral_config import SpectralConfig
from ..features.store import RegressionFeatureStore, StaleFeaturesError
from ..models.config import OUConfig
from ..utils.clock import utc_now_iso
from ..utils.config import Config
from ..utils.logging import get_logger
from ..utils.paths import atomic_write_text, ensure_dir
from .config import ResearchConfig
from .fft_analysis import window_spectra
from .ou_estimation import num
from .research_ledger import ResearchLedger
from .spectral_ic import feature_columns, ic_study
from .spectral_nulls import SourceData, build_control, build_real_source
from .spectral_phase import phase_advance_consistency, phase_bucket_table, phase_projection
from .spectral_reconstruction import extrapolation_table, reconstruction_table, sample_window_ends
from .spectral_stability import (
    METRICS,
    change_distribution,
    conditioning_table,
    dominant_period_stability,
    era_table,
    frequency_persistence,
    intraday_table,
    metrics_frame,
    segment_table,
)
from .study_io import StudyWriter, clean_json, git_info, stack_tables

__all__ = [
    "SPECTRAL_REPORT_VERSION",
    "SpectralStudy",
    "distribution_comparison",
    "generate_spectral_timeframe",
    "hypothesis_test_counts",
    "ic_chance_rates",
    "integrity_gate",
    "load_spectral_studies",
    "reusable_spectral_study",
    "write_spectral_comparison",
]

LOGGER = get_logger("research.spectral_reports")

#: Version of what a study writes; --resume reuses only studies of this version.
SPECTRAL_REPORT_VERSION = 1

CAVEAT = (
    "Descriptive research only. A spectral peak can come from finite samples, windowing, "
    "rolling detrending, volatility clustering or chance; every statistic here is read "
    "against white noise, a detrended random walk, shuffled and block-bootstrapped returns. "
    "No trading signal, entry, exit, sizing or cost exists in this layer, and no parameter "
    "was chosen for performance."
)

#: Values per reconstruction sample (windows x N), a memory bound, not a setting.
RECONSTRUCTION_VALUES = 20_000_000

#: The effect size (in null inter-quartile ranges) below which a distributional
#: difference from a null is treated as no difference. Fixed a priori.
NULL_EFFECT_THRESHOLD = 0.25

HYPOTHESES: dict[str, str] = {
    "SPEC-H-001": "Low spectral entropy is associated with stronger subsequent residual decay "
                  "after |Z| > 2 (Q1 vs Q4 entropy, 10 bars), beyond the random-walk control.",
    "SPEC-H-002": "Spectral concentration (median top-3 power share) exceeds every null control.",
    "SPEC-H-003": "The dominant frequency bin persists across disjoint windows (lag N) more "
                  "often than under every null control.",
    "SPEC-H-004": "The dominant component's phase is related to the subsequent residual change "
                  "(5 bars) beyond the random-walk control.",
    "SPEC-H-005": "The OU innovation retains spectral structure (entropy, flatness, top-3 "
                  "share) beyond shuffled and block-bootstrapped returns.",
    "SPEC-H-006": "A spectral feature carries forward information (rank IC) larger than any "
                  "IC seen on the null controls, with a sign stable across years.",
    "SPEC-H-007": "Carrying the top 3 Fourier components forward beats the last-value and "
                  "window-mean baselines at 5 bars, and beats the random-walk control.",
    "SPEC-H-008": "Spectral shape (median entropy) depends on the volatility regime more than "
                  "the random-walk control's does.",
    "SPEC-H-009": "Spectral shape (median entropy) depends on trend strength more than the "
                  "random-walk control's does.",
    "SPEC-H-010": "A spectral feature is distinct from existing features (max |Spearman| with "
                  "regression, OU and volatility features below 0.3).",
}


@dataclass
class SpectralStudy:
    """Outcome of one (timeframe, input series, FFT window) study."""

    timeframe: str
    series: str
    fft_window: int
    output_dir: Path
    summary: dict[str, Any] = field(default_factory=dict)
    files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    reused: bool = False


_clean = clean_json
_stack = stack_tables


# ---------------------------------------------------------------------------
# Integrity gate
# ---------------------------------------------------------------------------
def integrity_gate(config: Config, regression: RegressionConfig, ou: OUConfig,
                   spectral: SpectralConfig, timeframe: str) -> dict[str, Any]:
    """Refuse to mix artefacts from different dataset versions.

    Passes only if the tick dataset is complete, the bars were built from it,
    and the regression-feature cache for the configured window was computed
    from those bars under the current settings. A missing cache is built
    from current bars (no mismatch possible); a *stale* one fails the gate.
    The Prompt #4 OU study of the same (N, M) is checked for information
    only: the spectral study recomputes OU innovations from current features.
    """
    lineage = dataset_lineage(config, timeframe)
    problems: list[str] = []
    notes: list[str] = []
    if lineage.get("tick_dataset_status") != "complete":
        problems.append(f"tick dataset is {lineage.get('tick_dataset_status')!r}, not complete")
    if not lineage.get("bars_match_ticks"):
        problems.append(f"{timeframe} bars were not built from the current tick dataset "
                        f"({lineage.get('bar_built_from_tick_version')} vs "
                        f"{lineage.get('tick_dataset_version')})")
    store = RegressionFeatureStore(config, regression)
    entry = ((store.manifest(timeframe) or {}).get("windows") or {}).get(
        str(spectral.regression_window))
    feature_version = None
    if entry is None:
        notes.append(f"no cached {timeframe} features for N={spectral.regression_window}; "
                     "they will be computed from the current bars")
    else:
        try:
            store.load(timeframe, spectral.regression_window, ["timestamp"])
            feature_version = entry.get("feature_version")
        except StaleFeaturesError as exc:
            problems.append(f"regression features are stale: {exc}")
    ou_summary = (ou.results_dir(timeframe, spectral.regression_window, spectral.ou_window)
                  / "summary.json")
    if ou_summary.exists():
        recorded = json.loads(ou_summary.read_text(encoding="utf-8")).get("provenance", {})
        dataset = recorded.get("dataset") or {}
        same = (dataset.get("tick_dataset_version") == lineage.get("tick_dataset_version")
                and dataset.get("bar_dataset_version") == lineage.get("bar_dataset_version")
                and (feature_version is None
                     or recorded.get("regression_feature_version") == feature_version))
        notes.append(f"Prompt #4 OU study N={spectral.regression_window} "
                     f"M={spectral.ou_window}: {'current' if same else 'STALE (not used)'}")
    else:
        notes.append("no Prompt #4 OU study for this (N, M) on disk (not required)")
    if lineage.get("partial"):
        problems.append(PARTIAL_LABEL)
    return {"passed": not problems, "problems": problems, "notes": notes, "lineage": lineage,
            "regression_feature_version": feature_version}


# ---------------------------------------------------------------------------
# One pass: one source, one series, one window
# ---------------------------------------------------------------------------
def _lags(fft_window: int, spectral: SpectralConfig) -> list[int]:
    lags = {int(h) for h in spectral.persistence.lags_in_bars}
    lags |= {max(1, int(round(f * fft_window))) for f in spectral.persistence.lags_in_windows}
    return sorted(lags)


def _distribution(frame: pl.DataFrame, source: str) -> pl.DataFrame:
    rows = []
    for metric in METRICS:
        if metric not in frame.columns:
            continue
        values = frame[metric].drop_nulls()
        if values.len() == 0:
            continue
        rows.append({"source": source, "metric": metric, "count": values.len(),
                     "mean": values.mean(), "std": values.std(),
                     **{f"p{q}": values.quantile(q / 100) for q in (5, 25, 50, 75, 95)}})
    return pl.DataFrame(rows, infer_schema_length=None)


def _histograms(frame: pl.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for metric in ("spectral_entropy", "spectral_flatness", "top3_power_share"):
        values = frame[metric].drop_nulls().to_numpy()
        if values.size:
            density, edges = np.histogram(values, bins=60, range=(0.0, 1.0), density=True)
            out[metric] = (edges, density)
    return out


def _outcomes(metric: np.ndarray, source: SourceData, spectral: SpectralConfig, *,
              metric_name: str, source_name: str) -> pl.DataFrame:
    """P(the residual shrinks) after an extreme Z, by quartile of a spectral metric."""
    cfg = spectral.outcomes
    eps = source.columns["regression_residual"]
    rows = []
    valid = np.isfinite(metric)
    if valid.sum() < 1000:
        return pl.DataFrame()
    edges = np.quantile(metric[valid], np.linspace(0, 1, cfg.buckets + 1)[1:-1])
    bucket = np.where(valid, np.searchsorted(edges, metric, side="right"), -1)
    for z_name, centre in (("residual_zscore", np.zeros_like(eps)),
                           ("ou_zscore", source.columns["ou_mu"])):
        z = source.columns[z_name]
        extreme = valid & np.isfinite(z) & (np.abs(z) > cfg.extreme_abs_z) & np.isfinite(centre)
        now = np.abs(eps - centre)
        for h in cfg.horizons:
            later = np.full(eps.size, np.nan)
            later[:-h] = np.abs(eps[h:] - centre[:-h])
            ok = extreme & np.isfinite(later) & np.isfinite(now)
            for b in range(cfg.buckets):
                sel = ok & (bucket == b)
                n = int(sel.sum())
                if n < 30:
                    continue
                shrank = later[sel] < now[sel]
                rows.append({"source": source_name, "metric": metric_name, "extreme_z": z_name,
                             "horizon": h, "bucket": f"Q{b + 1}", "observations": n,
                             "prob_residual_shrinks": float(shrank.mean()),
                             "std_error": float(np.sqrt(shrank.mean() * (1 - shrank.mean()) / n)),
                             "median_ratio_later_to_now": float(np.median(later[sel] / now[sel]))})
    return pl.DataFrame(rows, infer_schema_length=None)


def _redundancy(result: RollingSpectrum, source: SourceData, series: str,
                spectral: SpectralConfig, max_rows: int = 300_000) -> tuple[pl.DataFrame,
                                                                            pl.DataFrame]:
    """Spearman correlations among spectral features and with existing ones."""
    rows = np.flatnonzero(result.valid)
    if rows.size > max_rows:
        rows = rows[np.linspace(0, rows.size - 1, max_rows).round().astype(np.int64)]
    features = feature_columns(result, spectral.ic.features, rows)
    x = pl.Series(source.columns[series]).fill_nan(None)
    rolling_acf1 = pl.DataFrame({"x": x}).select(
        pl.rolling_corr(pl.col("x"), pl.col("x").shift(1), window_size=result.fft_window,
                        min_samples=result.fft_window)).to_series().to_numpy()
    existing = {
        "abs_residual_zscore": np.abs(source.columns["residual_zscore"]),
        "regression_r_squared": source.columns["r_squared"],
        "abs_regression_slope": np.abs(source.columns["regression_slope"]),
        "log_ou_half_life": np.log(np.where(source.columns["ou_valid"] > 0,
                                            source.columns["ou_half_life_bars"], np.nan)),
        "trailing_volatility": source.columns["trailing_volatility"],
        f"rolling_acf1_{series}": rolling_acf1,
    }
    frame = pl.DataFrame({**features,
                          **{k: np.asarray(v, dtype=np.float64)[rows] for k, v in existing.items()}}
                         ).with_columns(pl.all().fill_nan(None))
    names = list(frame.columns)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    values = frame.select(
        *[pl.corr(a, b).alias(f"p|{a}|{b}") for a, b in pairs],
        *[pl.corr(a, b, method="spearman").alias(f"s|{a}|{b}") for a, b in pairs],
    ).row(0, named=True)
    correlation = pl.DataFrame([{"a": a, "b": b, "pearson": values[f"p|{a}|{b}"],
                                 "spearman": values[f"s|{a}|{b}"]} for a, b in pairs])
    red_rows = []
    for f in features:
        with_existing = correlation.filter(
            ((pl.col("a") == f) & pl.col("b").is_in(list(existing)))
            | ((pl.col("b") == f) & pl.col("a").is_in(list(existing))))
        with_spectral = correlation.filter(
            ((pl.col("a") == f) & pl.col("b").is_in(list(features)))
            | ((pl.col("b") == f) & pl.col("a").is_in(list(features))))
        top = with_existing.with_columns(pl.col("spearman").abs().alias("abs")).sort(
            "abs", descending=True, nulls_last=True).head(1)
        other = with_spectral.with_columns(pl.col("spearman").abs().alias("abs")).sort(
            "abs", descending=True, nulls_last=True).head(1)
        red_rows.append({
            "feature": f,
            "max_abs_spearman_existing": top["abs"][0] if top.height else None,
            "most_similar_existing": (top["b"][0] if top.height and top["a"][0] == f
                                      else top["a"][0] if top.height else None),
            "max_abs_spearman_spectral": other["abs"][0] if other.height else None,
            "most_similar_spectral": (other["b"][0] if other.height and other["a"][0] == f
                                      else other["a"][0] if other.height else None),
        })
    return correlation, pl.DataFrame(red_rows, infer_schema_length=None)


def analyse_pass(result: RollingSpectrum, source: SourceData, series: str,
                 spectral: SpectralConfig, research: ResearchConfig, *,
                 real: bool) -> dict[str, Any]:
    """Every per-source table of one (series, window) study."""
    name = source.name
    n = result.fft_window
    values = source.series(series)
    extra = {}
    if source.has_pipeline:
        extra = {k: source.columns[k] for k in ("trailing_volatility", "regression_slope",
                                                "ou_half_life_bars", "ou_valid")}
    frame = metrics_frame(result, source.timestamps, extra)
    lags = _lags(n, spectral)
    out: dict[str, Any] = {"source": name, "counts": dict(result.counts)}
    out["distribution"] = _distribution(frame, name)
    bins = result.layout.bins
    out["mean_spectrum"] = pl.DataFrame({
        "source": name, "bin": bins, "frequency_cycles_per_bar": bins / n,
        "normalised_frequency": bins / n / 0.5, "period_bars": n / bins.astype(np.float64),
        "period_seconds": n / bins.astype(np.float64) * source.bar_seconds,
        "mean_power_share": result.mean_share})
    total = max(int(result.dominant_histogram.sum()), 1)
    out["dominant_histogram"] = pl.DataFrame({
        "source": name, "bin": bins, "period_bars": n / bins.astype(np.float64),
        "period_seconds": n / bins.astype(np.float64) * source.bar_seconds,
        "share_of_windows": result.dominant_histogram / total})
    out["stability"] = {"source": name, **dominant_period_stability(result, lags)}
    out["persistence"] = [{"source": name, **r} for r in frequency_persistence(
        result, lags, top_k=spectral.persistence.top_k,
        tolerance=spectral.persistence.relative_tolerance)]
    out["phase_consistency"] = [{"source": name, **phase_advance_consistency(result, lag)}
                                for lag in (n, 2 * n)]
    out["changes"] = change_distribution(frame, [1, n], spectral.stability.change_quantiles,
                                         source=name)
    years = pl.DataFrame({"timestamp": source.timestamps}).group_by(
        pl.col("timestamp").dt.year().cast(pl.Utf8).alias("segment")).len("bars")
    out["yearly"] = segment_table(frame, by="year", source=name, total_bars=years)
    out["eras"] = era_table(frame, eras=spectral.stability.eras, source=name)
    if real or name == "random_walk":
        out["quarterly"] = segment_table(frame, by="quarter", source=name)
        if real:
            out["monthly"] = segment_table(frame, by="month", source=name)
        if source.has_pipeline:
            out["conditioning"] = conditioning_table(frame, spectral.conditioning, source=name)
        out["intraday"] = intraday_table(frame, spectral.conditioning, research, source=name)
    out["histograms"] = _histograms(frame)
    usable = usable_bins(n, spectral.spectrum)
    # The sample is a (windows x N) matrix; cap it at 20M values for long windows.
    ends = sample_window_ends(result.valid, min(spectral.reconstruction.max_windows,
                                                max(1000, RECONSTRUCTION_VALUES // n)))
    out["reconstruction"] = reconstruction_table(values, ends, n, usable_bins=usable,
                                                 components=spectral.reconstruction.components,
                                                 source=name)
    out["extrapolation"] = extrapolation_table(values, ends, n, usable_bins=usable,
                                               components=tuple(k for k in (1, 3, 5)
                                                                if k <= usable.size),
                                               horizons=spectral.reconstruction.horizons,
                                               source=name)
    # Phase outcomes only where a phase exists (a full-length array per
    # horizon would be ~63 MB each at 1-minute scale).
    with_phase = np.flatnonzero(result.dominant_phase_valid)
    phase = result.dominant_phase[with_phase].astype(np.float64)

    def change(series_values: np.ndarray, h: int) -> np.ndarray:
        ahead = with_phase + h
        inside = ahead < series_values.size
        later = np.where(inside, series_values[np.where(inside, ahead, 0)], np.nan)
        return later - series_values[with_phase]

    outcomes = {f"self_change_h{h}": change(values, h) for h in spectral.phase.horizons}
    projections = [phase_projection(result, values, spectral.phase.horizons, target_name="self")]
    if source.has_pipeline:
        eps = source.columns["regression_residual"]
        projections.append(phase_projection(result, values, spectral.phase.horizons,
                                            target=eps, target_name="regression_residual"))
        outcomes.update({f"residual_change_h{h}": change(eps, h)
                         for h in spectral.phase.horizons})
        out["outcomes"] = _stack([
            _outcomes(np.asarray(result.entropy, dtype=np.float64), source, spectral,
                      metric_name="spectral_entropy", source_name=name),
            _outcomes(np.asarray(result.concentration[3], dtype=np.float64), source, spectral,
                      metric_name="top3_power_share", source_name=name),
            _outcomes(np.asarray(result.concentration[1], dtype=np.float64), source, spectral,
                      metric_name="top1_power_share", source_name=name),
        ])
        max_rows = spectral.ic.max_rows if real else spectral.ic.max_rows_control
        ic, ic_periods, tests = ic_study(result, source, spectral, max_rows=max_rows,
                                         source_name=name)
        out["ic"], out["ic_periods"], out["ic_tests"] = ic, ic_periods, tests
    out["phase_buckets"] = phase_bucket_table(phase, outcomes, buckets=spectral.phase.buckets,
                                              source=name)
    projection = _stack(projections)
    out["phase_projection"] = (projection.with_columns(pl.lit(name).alias("source"))
                               if not projection.is_empty() else projection)
    if real:
        if source.has_pipeline:
            out["correlation"], out["redundancy"] = _redundancy(result, source, series, spectral)
        slice_ends = np.flatnonzero(result.valid)[-spectral.plots.sample_bars:]
        stamps = source.timestamps.to_numpy()
        out["plot_period"] = (stamps[slice_ends], result.dominant_period_bars[slice_ends])
        valid_idx = np.flatnonzero(result.valid)
        step = max(1, valid_idx.size // spectral.plots.max_series_points)
        thin = valid_idx[::step]
        out["plot_entropy"] = (stamps[thin], np.asarray(result.entropy[thin], dtype=np.float64))
        spec_ends = np.flatnonzero(result.valid)[-spectral.plots.spectrogram_bars:]
        out["plot_spectrogram"] = (stamps[spec_ends], bins / n / 0.5,
                                   window_spectra(values, spec_ends, n, spectral))
    del frame
    return out


# ---------------------------------------------------------------------------
# Verdicts and the null comparison
# ---------------------------------------------------------------------------
def _location(distribution: pl.DataFrame, metric: str,
              source: str) -> tuple[float, float, float]:
    """Median, inter-quartile range and 5-95 % range of one source's metric."""
    row = distribution.filter((pl.col("source") == source) & (pl.col("metric") == metric))
    if row.is_empty():
        return float("nan"), float("nan"), float("nan")
    r = row.row(0, named=True)
    return float(r["p50"]), float(r["p75"] - r["p25"]), float(r["p95"] - r["p5"])


def _effect(real: float, null: float, iqr: float, spread: float = float("nan")) -> float | None:
    """``(real - null median) / null IQR``; None only when a median is missing.

    A discrete metric (the dominant period) can have a zero IQR. Equal medians
    are then simply no difference; unequal ones are scaled by the null's
    5-95 % range instead, and by nothing (an infinite effect) when the null
    is a point mass elsewhere. Skipping such a null would let the verdict rest
    on the remaining ones alone - white noise, typically.
    """
    if not (np.isfinite(real) and np.isfinite(null)):
        return None
    if real == null:
        return 0.0
    for scale in (iqr, spread):
        if np.isfinite(scale) and scale > 0:
            return (real - null) / scale
    return float(np.copysign(np.inf, real - null))


def _verdict(effects: dict[str, float | None]) -> str:
    known = {k: v for k, v in effects.items() if v is not None}
    if not known:
        return "untested"
    if all(v > NULL_EFFECT_THRESHOLD for v in known.values()):
        return "exceeds_all_nulls"
    if all(v < -NULL_EFFECT_THRESHOLD for v in known.values()):
        return "below_all_nulls"
    if any(abs(v) <= NULL_EFFECT_THRESHOLD for v in known.values()):
        return "within_a_null"
    return "mixed"


def distribution_comparison(distribution: pl.DataFrame, nulls: list[str],
                            metrics: tuple[str, ...] = METRICS) -> list[dict[str, Any]]:
    """The median of every distributional metric, real beside each null, with effects."""
    rows = []
    for metric in metrics:
        real, _, _ = _location(distribution, metric, "real")
        row: dict[str, Any] = {"metric": f"median_{metric}", "real": real}
        for null in nulls:
            value, iqr, spread = _location(distribution, metric, null)
            row[null] = value
            row[f"effect_vs_{null}"] = _effect(real, value, iqr, spread)
        row["verdict"] = _verdict({n: row.get(f"effect_vs_{n}") for n in nulls})
        rows.append(row)
    return rows


def null_comparison(passes: dict[str, dict[str, Any]]) -> pl.DataFrame:
    """Every headline metric for the real series beside each null control."""
    if "real" not in passes:
        return pl.DataFrame()
    distribution = _stack([p["distribution"] for p in passes.values()])
    nulls = [s for s in passes if s != "real"]
    rows = distribution_comparison(distribution, nulls)
    scalar = {
        "switching_fraction_lag1": lambda p: p["stability"].get("switching_fraction_lag1"),
        "run_length_median": lambda p: p["stability"].get("run_length_median"),
        "same_dominant_bin_lag_N": lambda p: next(
            (v for k, v in p["stability"].items() if k.startswith("same_dominant_bin_lag_")
             and int(k.rsplit("_", 1)[1]) == p["fft_window"]), None),
        "top_k_overlap_lag_N": lambda p: next(
            (r["mean_overlap"] for r in p["persistence"] if r["lag"] == p["fft_window"]), None),
        "phase_error_resultant_lag_N": lambda p: next(
            (r.get("error_resultant_length") for r in p["phase_consistency"]
             if r["lag"] == p["fft_window"]), None),
        "extrapolation_skill_vs_mean_k3_h5": lambda p: _pick(
            p.get("extrapolation"), {"components": 3, "horizon": 5}, "skill_vs_mean"),
        "in_window_explained_variance_k3": lambda p: _pick(
            p.get("reconstruction"), {"components": 3}, "explained_variance_median"),
        "max_abs_rank_ic": lambda p: (float(p["ic"]["rank_ic"].abs().max())
                                      if isinstance(p.get("ic"), pl.DataFrame)
                                      and not p["ic"].is_empty() else None),
    }
    for metric, getter in scalar.items():
        row = {"metric": metric, "real": getter(passes["real"])}
        for null in nulls:
            value = getter(passes[null])
            row[null] = value
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def _pick(frame: Any, where: dict[str, Any], column: str) -> float | None:
    if not isinstance(frame, pl.DataFrame) or frame.is_empty():
        return None
    part = frame
    for k, v in where.items():
        if k not in part.columns:
            return None
        part = part.filter(pl.col(k) == v)
    return None if part.is_empty() else part[column][0]


def _ledger_entries(passes: dict[str, dict[str, Any]], comparison: pl.DataFrame, *,
                    timeframe: str, series: str, fft_window: int, dataset_version: str | None,
                    study: str) -> list[dict[str, Any]]:
    """Every hypothesis this study can speak to, with its a-priori verdict."""
    real = passes.get("real", {})
    entries: list[dict[str, Any]] = []
    base = {"timeframe": timeframe, "input_series": series, "window": fft_window,
            "dataset_version": dataset_version, "study": study}

    def add(hid: str, **values: Any) -> None:
        entries.append({**base, "hypothesis_id": hid, "definition": HYPOTHESES[hid], **values})

    def comp(metric: str) -> dict[str, Any]:
        part = comparison.filter(pl.col("metric") == metric)
        return part.row(0, named=True) if part.height else {}

    top3 = comp("median_top3_power_share")
    if top3:
        add("SPEC-H-002", feature="top3_power_share", metric="median", value=top3.get("real"),
            control_value=top3.get("random_walk"), verdict=top3.get("verdict"),
            details={k: v for k, v in top3.items() if k.startswith("effect_vs_")})
    persist = comp("same_dominant_bin_lag_N")
    if persist:
        nulls = {k: v for k, v in persist.items() if k not in ("metric", "real")}
        known = [v for v in nulls.values() if v is not None]
        value = persist.get("real")
        verdict = ("untested" if value is None or not known else
                   "exceeds_all_nulls" if all(value > v + 0.02 for v in known) else
                   "within_a_null")
        add("SPEC-H-003", feature="dominant_bin", metric="same_bin_fraction_lag_N", value=value,
            control_value=nulls.get("random_walk"), verdict=verdict, details=nulls)
    if series == "ou_innovation":
        effects = {}
        for metric in ("median_spectral_entropy", "median_spectral_flatness",
                       "median_top3_power_share"):
            row = comp(metric)
            for null in ("shuffled_returns", "block_bootstrap"):
                effects[f"{metric}|{null}"] = row.get(f"effect_vs_{null}")
        verdict = ("untested" if not any(v is not None for v in effects.values()) else
                   "exceeds_nulls" if all(v is not None and abs(v) > NULL_EFFECT_THRESHOLD
                                          for v in effects.values()) else "within_a_null")
        add("SPEC-H-005", feature="spectral_shape", metric="effect_vs_null_iqr",
            value=None, verdict=verdict, details=effects)
    extrap = comp("extrapolation_skill_vs_mean_k3_h5")
    if extrap:
        last = _pick(real.get("extrapolation"), {"components": 3, "horizon": 5}, "skill_vs_last")
        value, rw = extrap.get("real"), extrap.get("random_walk")
        beats = (value is not None and last is not None and value > 0 and last > 0
                 and (rw is None or value > rw))
        add("SPEC-H-007", feature="top3_components", target="self", horizon=5,
            metric="skill_vs_window_mean", value=value, control_value=rw,
            verdict="beats_baselines_and_null" if beats else "no_skill",
            details={"skill_vs_last": last})
    rw = passes.get("random_walk", {})
    if isinstance(real.get("outcomes"), pl.DataFrame) and isinstance(rw.get("outcomes"),
                                                                     pl.DataFrame):
        def spread(frame: pl.DataFrame) -> tuple[float | None, float | None]:
            if frame.is_empty() or "metric" not in frame.columns:
                return None, None
            part = frame.filter((pl.col("metric") == "spectral_entropy")
                                & (pl.col("extreme_z") == "residual_zscore")
                                & (pl.col("horizon") == 10))
            q = {r["bucket"]: r for r in part.iter_rows(named=True)}
            if "Q1" not in q or "Q4" not in q:
                return None, None
            d = q["Q1"]["prob_residual_shrinks"] - q["Q4"]["prob_residual_shrinks"]
            se = float(np.hypot(q["Q1"]["std_error"], q["Q4"]["std_error"]))
            return d, se
        d_real, se_real = spread(real["outcomes"])
        d_rw, se_rw = spread(rw["outcomes"])
        if d_real is not None and d_rw is not None:
            gap = d_real - d_rw
            se = float(np.hypot(se_real or 0.0, se_rw or 0.0))
            add("SPEC-H-001", feature="spectral_entropy", target="residual_shrinks", horizon=10,
                metric="P(shrink|Q1) - P(shrink|Q4)", value=d_real, control_value=d_rw,
                verdict="exceeds_null" if se > 0 and abs(gap) > 2 * se else "consistent_with_null",
                details={"difference_vs_null": gap, "std_error": se})
    if isinstance(real.get("phase_buckets"), pl.DataFrame) and isinstance(
            rw.get("phase_buckets"), pl.DataFrame):
        def phase_spread(frame: pl.DataFrame) -> float | None:
            if frame.is_empty() or "outcome" not in frame.columns:
                return None
            part = frame.filter((pl.col("outcome") == "residual_change_h5") & (pl.col("bucket") == -1))
            return part["spread_over_std"][0] if part.height else None
        a, b = phase_spread(real["phase_buckets"]), phase_spread(rw["phase_buckets"])
        if a is not None and b is not None:
            add("SPEC-H-004", feature="dominant_phase", target="residual_change", horizon=5,
                metric="spread_of_sector_means_over_std", value=a, control_value=b,
                verdict="exceeds_null" if a > 2 * b else "consistent_with_null")
    for hid, variable in (("SPEC-H-008", "volatility"), ("SPEC-H-009", "trend")):
        ranges: dict[str, float] = {}
        for src in ("real", "random_walk"):
            table = passes.get(src, {}).get("conditioning")
            if isinstance(table, pl.DataFrame) and not table.is_empty():
                part = table.filter(pl.col("variable") == variable)[
                    "median_spectral_entropy"].drop_nulls()
                if part.len():
                    ranges[src] = num(part.max()) - num(part.min())
        if "real" in ranges and "random_walk" in ranges:
            real_range, null_range = ranges["real"], ranges["random_walk"]
            verdict = ("exceeds_null" if real_range > 2 * null_range and real_range > 0.01
                       else "consistent_with_null")
            add(hid, feature="spectral_entropy", metric="range_of_bucket_medians",
                value=real_range, control_value=null_range, verdict=verdict)
    ic = real.get("ic")
    if isinstance(ic, pl.DataFrame) and not ic.is_empty():
        null_ics = [p["ic"]["rank_ic"].abs().max() for s, p in passes.items()
                    if s != "real" and isinstance(p.get("ic"), pl.DataFrame)
                    and not p["ic"].is_empty()]
        ceiling = float(max(null_ics)) if null_ics else None
        for r in ic.iter_rows(named=True):
            value = r.get("rank_ic")
            share = r.get("yearly_rank_ic_positive_share")
            stable = share is not None and (share >= 0.75 or share <= 0.25)
            verdict = ("untested" if value is None or ceiling is None else
                       "exceeds_null_max_with_stable_sign" if abs(value) > ceiling and stable
                       else "exceeds_null_max_unstable_sign" if abs(value) > ceiling
                       else "within_null_range")
            add("SPEC-H-006", feature=r["feature"], target=r["target"], horizon=r["horizon"],
                metric="rank_ic", value=value, control_value=ceiling, verdict=verdict,
                details={"yearly_mean": r.get("yearly_rank_ic_mean"),
                         "yearly_t": r.get("yearly_rank_ic_t"),
                         "yearly_positive_share": share})
    redundancy = real.get("redundancy")
    if isinstance(redundancy, pl.DataFrame):
        for r in redundancy.iter_rows(named=True):
            m = r.get("max_abs_spearman_existing")
            verdict = ("untested" if m is None else "distinct" if m < 0.3 else
                       "partially_redundant" if m < 0.7 else "redundant")
            add("SPEC-H-010", feature=r["feature"], metric="max_abs_spearman_existing", value=m,
                verdict=verdict, details={"most_similar_existing": r.get("most_similar_existing")})
    return entries


# ---------------------------------------------------------------------------
# A timeframe
# ---------------------------------------------------------------------------
def _write_features(result: RollingSpectrum, source: SourceData, spectral: SpectralConfig, *,
                    timeframe: str, series: str, lineage: dict[str, Any]) -> dict[str, Any]:
    """Per-bar features as Parquet partitioned by year, with a manifest."""
    frame = spectral_feature_frame(result, source.timestamps, spectral)
    root = (spectral.features_path / f"timeframe={timeframe}" / f"source={series}"
            / f"fft_window={result.fft_window}")
    ensure_dir(root)
    for stale in root.rglob("*.parquet"):
        stale.unlink()
    frame = frame.with_columns(pl.col("timestamp").dt.year().alias("__year"))
    written = 0
    for (year,), part in frame.group_by("__year", maintain_order=True):
        directory = ensure_dir(root / f"year={year}")
        tmp = directory / "features.parquet.partial"
        part.drop("__year").write_parquet(tmp, compression="zstd")
        tmp.replace(directory / "features.parquet")
        written += tmp.with_name("features.parquet").stat().st_size
    manifest = {
        "timeframe": timeframe, "input_series": series, "fft_window": result.fft_window,
        "rows": frame.height, "valid_rows": int(result.valid.sum()), "bytes": written,
        "columns": [c for c in frame.columns if c != "__year"],
        "engine_fingerprint": spectral.engine_fingerprint(),
        "tick_dataset_version": lineage.get("tick_dataset_version"),
        "bar_dataset_version": lineage.get("bar_dataset_version"),
        "regression_window": spectral.regression_window, "ou_window": spectral.ou_window,
        "generated_utc": utc_now_iso(),
    }
    atomic_write_text(root / "_manifest.json", json.dumps(manifest, indent=1) + "\n")
    return manifest


def generate_spectral_timeframe(
    config: Config, regression: RegressionConfig, research: ResearchConfig, ou: OUConfig,
    spectral: SpectralConfig, *, timeframe: str, series: list[str] | None = None,
    windows: list[int] | None = None, make_plots: bool = True,
    write_features: str | None = None,
) -> list[SpectralStudy]:
    """Run every requested (series, window) study for one timeframe.

    *write_features* is ``representative``, ``all`` or ``none``; by default
    the ``features_output.write`` policy of the configuration.
    """
    policy = write_features or spectral.features_output.write
    started = time.perf_counter()
    series = list(series or spectral.input_series)
    windows = sorted(windows or spectral.fft_windows)
    gate = integrity_gate(config, regression, ou, spectral, timeframe)
    if not gate["passed"]:
        raise RuntimeError(f"{timeframe}: dataset integrity gate failed - "
                           + "; ".join(gate["problems"]))
    lineage = gate["lineage"]
    LOGGER.info("[%s] gate passed: %s", timeframe, "; ".join(gate["notes"]))
    real = build_real_source(config, regression, ou, spectral, timeframe)
    passes: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    features_written: dict[tuple[str, int], dict[str, Any]] = {}
    timings: dict[str, float] = {}
    for source_name in ["real", *spectral.controls.enabled()]:
        t0 = time.perf_counter()
        source = real if source_name == "real" else build_control(
            source_name, real, regression=regression, ou=ou, spectral=spectral)
        for name in series:
            if name not in source.columns:
                continue
            values = source.series(name)
            for n in windows:
                t1 = time.perf_counter()
                result = rolling_spectrum(values, fft_window=n, config=spectral,
                                          bar_seconds=source.bar_seconds,
                                          missing_slots=source.missing_slots)
                analysis = analyse_pass(result, source, name, spectral, research,
                                        real=source_name == "real")
                analysis["fft_window"] = n
                passes.setdefault((name, n), {})[source_name] = analysis
                write = policy == "all" or (policy == "representative"
                                            and n == spectral.representative_fft_window)
                if write and source_name == "real":
                    features_written[(name, n)] = _write_features(
                        result, source, spectral, timeframe=timeframe, series=name,
                        lineage=lineage)
                del result
                gc.collect()
                LOGGER.info("[%s %s N=%d %s] %.1fs", timeframe, name, n, source_name,
                            time.perf_counter() - t1)
        timings[source_name] = time.perf_counter() - t0
        if source_name != "real":
            del source
            gc.collect()
    studies = []
    for (name, n), by_source in passes.items():
        studies.append(_write_study(by_source, spectral, config, regression, ou, gate,
                                    timeframe=timeframe, series=name, fft_window=n,
                                    real=real, make_plots=make_plots,
                                    features=features_written.get((name, n)),
                                    seconds=time.perf_counter() - started, timings=timings))
    return studies


def _write_study(by_source: dict[str, dict[str, Any]], spectral: SpectralConfig, config: Config,
                 regression: RegressionConfig, ou: OUConfig, gate: dict[str, Any], *,
                 timeframe: str, series: str, fft_window: int, real: SourceData,
                 make_plots: bool, features: dict[str, Any] | None, seconds: float,
                 timings: dict[str, float]) -> SpectralStudy:
    study = SpectralStudy(timeframe=timeframe, series=series, fft_window=fft_window,
                          output_dir=spectral.results_dir(timeframe, series, fft_window))
    writer = StudyWriter(study.output_dir, spectral.output, study.files)
    sources = list(by_source)

    def stack(key: str) -> pl.DataFrame:
        return _stack([frame for s in sources
                       if isinstance(frame := by_source[s].get(key), pl.DataFrame)])

    def rows(key: str) -> pl.DataFrame:
        items: list[dict[str, Any]] = []
        for s in sources:
            value = by_source[s].get(key)
            if isinstance(value, dict):
                items.append(value)
            elif isinstance(value, list):
                items.extend(value)
        return pl.DataFrame([_clean(i) for i in items], infer_schema_length=None) if items \
            else pl.DataFrame()

    distribution = stack("distribution")
    writer.table(distribution, "spectral_distribution")
    writer.table(distribution.filter(pl.col("metric").is_in(
        ["spectral_entropy", "spectral_flatness", "top1_power_share", "top3_power_share",
         "top5_power_share", "spectral_centroid_norm"])), "entropy_summary")
    writer.table(distribution.filter(pl.col("metric").is_in(
        [f"{b}_power_share" for b in spectral.power_bands.names])), "power_bands")
    writer.table(stack("mean_spectrum"), "mean_spectrum")
    writer.table(stack("dominant_histogram"), "dominant_period_distribution")
    writer.table(rows("stability"), "dominant_period_stability")
    writer.table(rows("persistence"), "frequency_persistence")
    writer.table(rows("phase_consistency"), "phase_consistency")
    writer.table(stack("changes"), "spectral_changes")
    for key, name in (("yearly", "yearly_stability"), ("quarterly", "quarterly_stability"),
                      ("monthly", "monthly_stability"), ("eras", "era_stability"),
                      ("conditioning", "conditioning"), ("intraday", "intraday"),
                      ("outcomes", "residual_outcomes"), ("phase_buckets", "phase_analysis"),
                      ("phase_projection", "phase_projection"),
                      ("reconstruction", "reconstruction_metrics"),
                      ("extrapolation", "extrapolation_metrics"), ("ic", "ic_analysis")):
        writer.table(stack(key), name)
    writer.table(stack("ic_periods"), "ic_by_period", csv=False)
    real_pass = by_source.get("real", {})
    writer.table(real_pass.get("correlation"), "feature_correlation")
    writer.table(real_pass.get("redundancy"), "feature_redundancy")
    comparison = null_comparison({s: {**p, "fft_window": fft_window}
                                  for s, p in by_source.items()})
    writer.table(comparison, "null_control_comparison")

    tests = {s: int(p.get("ic_tests", 0)) for s, p in by_source.items()}
    lineage = gate["lineage"]
    study_key = f"{timeframe}/{series}/fft_{fft_window}"
    if spectral.ledger.enabled:
        entries = _ledger_entries({s: {**p, "fft_window": fft_window} for s, p in by_source.items()},
                                  comparison, timeframe=timeframe, series=series,
                                  fft_window=fft_window,
                                  dataset_version=lineage.get("tick_dataset_version"),
                                  study=study_key)
        ResearchLedger(spectral.ledger_path).upsert(entries)
        verdicts = pl.DataFrame(entries).group_by("hypothesis_id", "verdict").len().sort(
            "hypothesis_id", "verdict") if entries else pl.DataFrame()
        writer.table(verdicts, "hypothesis_verdicts")
    if make_plots and spectral.plots.enabled:
        study.warnings += _plots(by_source, spectral, study, timeframe=timeframe, series=series,
                                 fft_window=fft_window)

    def dist(metric: str, q: str = "p50") -> float | None:
        part = distribution.filter((pl.col("source") == "real") & (pl.col("metric") == metric))
        return part[q][0] if part.height else None

    stability = real_pass.get("stability", {})
    summary = {
        "timeframe": timeframe, "input_series": series, "fft_window": fft_window,
        "regression_window": spectral.regression_window, "ou_window": spectral.ou_window,
        "observations": real.size,
        "valid_windows": real_pass.get("counts", {}).get("valid"),
        "valid_spectral_fraction": (real_pass.get("counts", {}).get("valid", 0)
                                    / max(real.size, 1)),
        "window_counts": real_pass.get("counts"),
        "frequency_resolution_cycles_per_bar": 1.0 / fft_window,
        "longest_period_bars": fft_window, "shortest_period_bars": 2,
        "median_spectral_entropy": dist("spectral_entropy"),
        "median_spectral_flatness": dist("spectral_flatness"),
        "median_top1_power_share": dist("top1_power_share"),
        "median_top3_power_share": dist("top3_power_share"),
        "median_spectral_centroid_norm": dist("spectral_centroid_norm"),
        "median_low_power_share": dist("low_power_share"),
        "median_high_power_share": dist("high_power_share"),
        "median_dominant_period_bars": dist("dominant_period_bars"),
        "median_dominant_period_seconds": dist("dominant_period_seconds"),
        "dominant_period_switching_fraction": stability.get("switching_fraction_lag1"),
        "dominant_period_persistence": stability.get(f"same_dominant_bin_lag_{fft_window}"),
        "phase_forward_relationship": _pick(real_pass.get("phase_projection"),
                                            {"horizon": 5, "target": "regression_residual"},
                                            "pearson"),
        "phase_forward_relationship_self": _pick(real_pass.get("phase_projection"),
                                                 {"horizon": 5, "target": "self"}, "pearson"),
        "reconstruction_rmse": _pick(real_pass.get("reconstruction"), {"components": 3},
                                     "rmse_median"),
        "reconstruction_explained_variance_k3": _pick(real_pass.get("reconstruction"),
                                                      {"components": 3},
                                                      "explained_variance_median"),
        "forward_extrapolation_rmse": _pick(real_pass.get("extrapolation"),
                                            {"components": 3, "horizon": 5}, "rmse"),
        "forward_extrapolation_skill_vs_mean": _pick(real_pass.get("extrapolation"),
                                                     {"components": 3, "horizon": 5},
                                                     "skill_vs_mean"),
        "forward_extrapolation_skill_vs_last": _pick(real_pass.get("extrapolation"),
                                                     {"components": 3, "horizon": 5},
                                                     "skill_vs_last"),
        "null_control_difference": {
            r["metric"]: {k: v for k, v in r.items() if k != "metric"}
            for r in comparison.filter(pl.col("metric").is_in(
                ["median_spectral_entropy", "median_spectral_flatness", "median_top3_power_share",
                 "same_dominant_bin_lag_N", "max_abs_rank_ic",
                 "extrapolation_skill_vs_mean_k3_h5"])).iter_rows(named=True)},
        "ic_tests": tests,
        "max_abs_rank_ic": _pick(comparison, {"metric": "max_abs_rank_ic"}, "real"),
        "features_written": features,
        "partial_dataset": bool(lineage.get("partial")),
        "caveat": CAVEAT,
    }
    study.summary = _clean(summary)
    study.duration_seconds = seconds
    writer.json({
        "summary": summary,
        "provenance": {
            "report_version": SPECTRAL_REPORT_VERSION,
            "generated_utc": utc_now_iso(), "git": git_info(config.project_root),
            "dataset": lineage, "integrity_gate": {k: v for k, v in gate.items()
                                                   if k != "lineage"},
            "regression_feature_version": gate.get("regression_feature_version"),
            "spectral_config_fingerprint": spectral.fingerprint(),
            "spectral_engine_fingerprint": spectral.engine_fingerprint(),
            "regression_config_fingerprint": regression.fingerprint(),
            "ou_config_fingerprint": ou.fingerprint(),
            "data_config_fingerprint": config.fingerprint(),
            "preprocessing": {"window_function": spectral.preprocessing.window_function,
                              "remove_mean": spectral.preprocessing.remove_mean,
                              "detrend_input": spectral.preprocessing.detrend_input,
                              "missing_bar_tolerance":
                                  spectral.preprocessing.missing_bar_tolerance,
                              "missing_value_policy": "a window with any missing value is "
                                                      "skipped; nothing is filled"},
            "controls": {s: by_source[s].get("counts") for s in sources},
            "source_seconds": timings,
            "sampling": {"reconstruction_windows": spectral.reconstruction.max_windows,
                         "ic_rows_real": spectral.ic.max_rows,
                         "ic_rows_control": spectral.ic.max_rows_control,
                         "note": "reconstruction, extrapolation, IC and persistence figures "
                                 "are estimates from evenly spaced samples"},
        },
        "hypotheses": HYPOTHESES,
        "warnings": study.warnings,
    }, "summary")
    return study


def _plots(by_source: dict[str, dict[str, Any]], spectral: SpectralConfig, study: SpectralStudy,
           *, timeframe: str, series: str, fft_window: int) -> list[str]:
    from . import spectral_plots as plots

    warnings: list[str] = []
    directory = spectral.plots_dir(timeframe, series, fft_window)
    tag = f"{timeframe}  {series}  N={fft_window}"
    bar_label = timeframe
    spectra = {s: p["mean_spectrum"] for s, p in by_source.items()}
    real = by_source.get("real", {})
    yearly = {s: p["yearly"] for s, p in by_source.items() if s in ("real", "random_walk")
              and isinstance(p.get("yearly"), pl.DataFrame)}
    jobs = [
        ("mean_spectrum", lambda p: plots.plot_mean_spectrum(
            spectra, path=p, config=spectral, title=f"{tag}  mean spectrum: XAUUSD vs nulls")),
        ("period_power", lambda p: plots.plot_period_power(
            spectra, path=p, config=spectral, title=f"{tag}  power by period",
            bar_label=bar_label)),
        ("null_distributions", lambda p: plots.plot_null_distributions(
            {s: q["histograms"] for s, q in by_source.items()}, path=p, config=spectral,
            title=f"{tag}  spectral shape: XAUUSD vs nulls")),
    ]
    if "plot_period" in real:
        period_times, periods = real["plot_period"]
        entropy_times, entropy = real["plot_entropy"]
        spec_times, spec_freq, spec_shares = real["plot_spectrogram"]
        jobs.append(("rolling_dominant_period", lambda p: plots.plot_dominant_period(
            period_times, periods, path=p, config=spectral,
            title=f"{tag}  dominant period, recent {spectral.plots.sample_bars:,} windows",
            bar_label=bar_label)))
        jobs.append(("rolling_entropy", lambda p: plots.plot_rolling_entropy(
            entropy_times, entropy, yearly, path=p, config=spectral,
            title=f"{tag}  spectral entropy through the full history")))
        jobs.append(("spectrogram", lambda p: plots.plot_spectrogram(
            spec_times, spec_freq, spec_shares, path=p, config=spectral,
            title=f"{tag}  spectrogram, recent {spectral.plots.spectrogram_bars:,} windows")))
    if isinstance(real.get("yearly"), pl.DataFrame) and spectral.power_bands.enabled:
        jobs.append(("band_power", lambda p: plots.plot_band_power(
            real["yearly"], spectral.power_bands.names, path=p, config=spectral,
            title=f"{tag}  power by band, yearly medians")))
    phase_tables = {s: p["phase_buckets"] for s, p in by_source.items()
                    if s in ("real", "random_walk", "white_noise")
                    and isinstance(p.get("phase_buckets"), pl.DataFrame)
                    and not p["phase_buckets"].is_empty()}
    outcome = "residual_change_h5"
    if phase_tables:
        jobs.append(("phase_outcomes", lambda p: plots.plot_phase_outcomes(
            phase_tables, outcome, path=p, config=spectral,
            title=f"{tag}  {outcome} by dominant-phase sector")))
    for name, job in jobs:
        path = directory / f"{name}.{spectral.plots.format}"
        try:
            job(path)
            study.files.append(f"plots/{path.name}")
        except Exception as exc:  # noqa: BLE001 - a failed plot must not lose the tables
            warnings.append(f"plot {name} failed: {type(exc).__name__}: {exc}")
    gc.collect()
    return warnings


# ---------------------------------------------------------------------------
# Reuse and comparison
# ---------------------------------------------------------------------------
def _read_summary(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) and payload.get("summary") else None


def reusable_spectral_study(config: Config, regression: RegressionConfig, ou: OUConfig,
                            spectral: SpectralConfig, *, timeframe: str, series: str,
                            fft_window: int) -> SpectralStudy | None:
    """The study on disk if it is current (versions, settings, report version)."""
    path = spectral.results_dir(timeframe, series, fft_window) / "summary.json"
    payload = _read_summary(path)
    if payload is None:
        return None
    recorded = payload.get("provenance") or {}
    if recorded.get("report_version") != SPECTRAL_REPORT_VERSION:
        return None
    lineage = dataset_lineage(config, timeframe)
    dataset = recorded.get("dataset") or {}
    for key in ("raw_fingerprint", "tick_dataset_version", "bar_dataset_version"):
        if lineage.get(key) is None or dataset.get(key) != lineage.get(key):
            return None
    expected = {
        "spectral_config_fingerprint": spectral.fingerprint(),
        "regression_config_fingerprint": regression.fingerprint(),
        "ou_config_fingerprint": ou.fingerprint(),
        "data_config_fingerprint": config.fingerprint(),
    }
    if any(recorded.get(k) != v for k, v in expected.items()):
        return None
    summary = payload["summary"]
    return SpectralStudy(timeframe=timeframe, series=series, fft_window=fft_window,
                         output_dir=path.parent, summary=summary,
                         warnings=list(payload.get("warnings") or []), reused=True)


def load_spectral_studies(spectral: SpectralConfig) -> list[SpectralStudy]:
    studies = []
    for path in sorted(spectral.results_path.glob("*/*/fft_*/summary.json")):
        payload = _read_summary(path)
        if payload is None:
            continue
        s = payload["summary"]
        studies.append(SpectralStudy(timeframe=s["timeframe"], series=s["input_series"],
                                     fft_window=int(s["fft_window"]), output_dir=path.parent,
                                     summary=s))
    return studies


#: Sources whose IC tests are like for like: those built through the full pipeline.
IC_SOURCES = ("real", "random_walk", "shuffled_returns", "block_bootstrap")


def _study_window(study: Any) -> int:
    """A study's analysis window: ``window`` (wavelet) or ``fft_window`` (spectral)."""
    return int(getattr(study, "window", None) or study.fft_window)


def ic_chance_rates(studies: list[Any]) -> pl.DataFrame:
    """How often each source's rank IC exceeds the largest |rank IC| of the others.

    SPEC-H-006 counts real tests above the nulls' maximum. If the real series
    were just one more draw, every source would be equally likely to hold the
    largest value, so each null's count against the other three sources (the
    real one included) is the chance rate to read the real count against.
    Real ICs use more rows than the nulls' (``ic.max_rows`` vs
    ``ic.max_rows_control``); less noise makes the real count, if anything, low.
    """
    rows: list[dict[str, Any]] = []
    for study in studies:
        path = study.output_dir / "ic_analysis.parquet"
        if not path.exists():
            continue
        table = (pl.read_parquet(path, columns=["source", "rank_ic",
                                                "yearly_rank_ic_positive_share"])
                 .filter(pl.col("source").is_in(IC_SOURCES) & pl.col("rank_ic").is_not_null()
                         & pl.col("rank_ic").is_not_nan())
                 .with_columns(pl.col("rank_ic").abs().alias("abs_rank_ic")))
        present = [s for s in IC_SOURCES if (table["source"] == s).any()]
        for source in present:
            others = table.filter(pl.col("source").is_in([s for s in present if s != source]))
            if others.is_empty():
                continue
            ceiling = num(others["abs_rank_ic"].max())
            own = table.filter(pl.col("source") == source)
            above = own.filter(pl.col("abs_rank_ic") > ceiling)
            share = pl.col("yearly_rank_ic_positive_share")
            rows.append({
                "timeframe": study.timeframe, "input_series": study.series,
                "window": _study_window(study), "source": source, "tests": own.height,
                "ceiling_from_other_sources": ceiling, "exceedances": above.height,
                "exceedances_stable_sign": above.filter((share >= 0.75) | (share <= 0.25)).height,
            })
    return pl.DataFrame(rows, schema={
        "timeframe": pl.Utf8, "input_series": pl.Utf8, "window": pl.Int64,
        "source": pl.Utf8, "tests": pl.Int64, "ceiling_from_other_sources": pl.Float64,
        "exceedances": pl.Int64, "exceedances_stable_sign": pl.Int64})


def hypothesis_test_counts(ledger: pl.DataFrame, prefix: str = "SPEC-") -> pl.DataFrame:
    """How many tests each hypothesis of one layer ran, and how they came out."""
    layer_rows = ledger.filter(pl.col("hypothesis_id").str.starts_with(prefix))
    return (layer_rows.group_by("hypothesis_id", "verdict")
            .agg(pl.len().cast(pl.Int64).alias("tests"),
                 pl.col("timeframe").n_unique().alias("timeframes"),
                 pl.col("input_series").n_unique().alias("input_series"),
                 pl.col("window").n_unique().alias("windows"))
            .with_columns((pl.col("tests") / pl.col("tests").sum().over("hypothesis_id"))
                          .alias("share_of_hypothesis_tests"))
            .sort("hypothesis_id", "verdict"))


def write_spectral_comparison(spectral: SpectralConfig, studies: list[SpectralStudy]) -> list[Path]:
    """Cross-study tables: timeframes, series, windows, null differences. No ranking.

    Also the multiple-testing record: every hypothesis's test and verdict
    counts from the ledger, and the IC chance rates (`ic_chance_rates`).
    """
    summaries = [s.summary for s in studies if s.summary]
    if not summaries:
        return []
    directory = ensure_dir(spectral.results_path)
    flat = []
    for s in summaries:
        row = {k: v for k, v in s.items() if not isinstance(v, (dict, list))
               and k != "caveat"}
        for metric, values in (s.get("null_control_difference") or {}).items():
            for key, value in values.items():
                if not isinstance(value, (dict, list)):
                    row[f"{metric}__{key}"] = value
        flat.append(row)
    table = pl.DataFrame(flat, infer_schema_length=None).sort("timeframe", "input_series",
                                                              "fft_window")
    written = []
    for name, frame in (("spectral_comparison", table),):
        frame.write_csv(directory / f"{name}.csv")
        frame.write_parquet(directory / f"{name}.parquet")
        written += [directory / f"{name}.csv", directory / f"{name}.parquet"]
    keep = ["timeframe", "input_series", "fft_window", "median_spectral_entropy",
            "median_spectral_flatness", "median_top3_power_share", "median_dominant_period_bars",
            "median_dominant_period_seconds", "dominant_period_switching_fraction",
            "dominant_period_persistence"]
    window = table.select([c for c in keep if c in table.columns])
    window.write_csv(directory / "window_comparison.csv")
    written.append(directory / "window_comparison.csv")
    chance = ic_chance_rates(studies)
    counts = hypothesis_test_counts(ResearchLedger(spectral.ledger_path).load())
    for name, frame in (("ic_chance_rates", chance), ("hypothesis_test_counts", counts)):
        if not frame.is_empty():
            frame.write_csv(directory / f"{name}.csv")
            written.append(directory / f"{name}.csv")
    atomic_write_text(directory / "spectral_comparison.json",
                      json.dumps(_clean(summaries), indent=1, default=str) + "\n")
    written.append(directory / "spectral_comparison.json")
    return written
