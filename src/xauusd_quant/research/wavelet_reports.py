"""Orchestration of the wavelet / time-frequency study (Prompt #6).

One timeframe at a time: the integrity gate, then every source (the real data
and each null control, one in memory at a time), every input series and every
rolling window. Per (timeframe, series, window) study the outputs go to
``results/wavelet_research/<tf>/<series>/window_<N>/``:

``summary.json``                        machine-readable digest + full provenance
``metric_distribution``                 quantiles of every wavelet metric, by source
``energy_summary``                      band energy shares with physical periods, by source
``entropy_summary``                     entropy and concentration quantiles, by source
``scale_persistence``, ``run_summary``  dominant-band persistence and runs
``drift_summary``, ``burst_summary``    frequency drift and localised bursts
``yearly/quarterly/era_stability``
``volatility/trend/ou_conditioning``, ``intraday``
``extreme_conditioning``                residual / OU extremes by wavelet (and OU) state
``ic_analysis``, ``ic_by_period``       information coefficients
``incremental_information``, ``incremental_summary``, ``incremental_by_feature``
``fft_comparison``, ``fft_determinism``, ``dyadic_comparison``
``feature_correlation``, ``feature_redundancy``, ``feature_quality``
``null_controls``                       every headline metric, real beside each null
``hypothesis_verdicts``, ``family_sensitivity`` (representative window)
``plots/``

Per (timeframe, series): ``offline_ridges`` and scalograms - NON-CAUSAL.

Every hypothesis is registered before any result exists
(``hypotheses_registered.json``) and every verdict is written to the research
ledger, failures included. The rules compare the real series with *every*
pipeline null - never the random walk alone.
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

from ..data.versioning import dataset_lineage
from ..features.config import RegressionConfig
from ..features.spectral import RollingSpectrum, rolling_spectrum
from ..features.spectral_config import SpectralConfig
from ..features.wavelet import NON_CAUSAL_LABEL
from ..features.wavelet_causal import (
    CAUSAL_FEATURES,
    RollingWavelet,
    feature_lookback,
    ic_feature_columns,
    rolling_wavelet,
    wavelet_feature_frame,
)
from ..features.wavelet_config import WaveletConfig
from ..models.config import OUConfig
from ..utils.clock import utc_now_iso
from ..utils.config import Config
from ..utils.logging import get_logger
from ..utils.paths import atomic_write_text, ensure_dir
from . import wavelet_plots as plots
from .config import ResearchConfig
from .ou_estimation import num
from .research_ledger import ResearchLedger
from .spectral_ic import FEATURES as FFT_FEATURES
from .spectral_ic import feature_columns as fft_feature_columns
from .spectral_nulls import SourceData, build_control, build_real_source
from .spectral_reports import (
    NULL_EFFECT_THRESHOLD,
    distribution_comparison,
    hypothesis_test_counts,
    ic_chance_rates,
)
from .spectral_reports import integrity_gate as spectral_integrity_gate
from .spectral_stability import conditioning_table, era_table, intraday_table, segment_table
from .study_io import (
    StudyWriter,
    clean_json,
    code_fingerprint,
    git_info,
    package_versions,
    stack_tables,
)
from .wavelet_analysis import (
    dyadic_comparison,
    fft_comparison,
    offline_ridges,
    offline_scalogram,
    select_slices,
)
from .wavelet_predictiveness import (
    BASELINE_EXTENSIONS,
    extreme_conditioning,
    incremental_study,
    incremental_summary,
    trailing_mean,
    wavelet_ic_study,
)
from .wavelet_stability import (
    WAVELET_METRICS,
    band_persistence,
    burst_summary,
    drift_summary,
    run_summary,
    wavelet_metrics_frame,
    wavelet_summaries,
)

__all__ = [
    "HYPOTHESES",
    "POST_HOC_HYPOTHESES",
    "WAVELET_REPORT_VERSION",
    "WaveletStudy",
    "generate_wavelet_timeframe",
    "integrity_gate",
    "load_wavelet_studies",
    "post_hoc_baseline_entries",
    "register_hypotheses",
    "reusable_wavelet_study",
    "source_roles",
    "write_wavelet_comparison",
]

LOGGER = get_logger("research.wavelet_reports")

#: Version of what a study writes; --resume reuses only studies of this version.
WAVELET_REPORT_VERSION = 2

CAVEAT = (
    "Descriptive research only. Multiscale structure can come from rolling detrending, "
    "volatility clustering, the trading-time axis, finite windows or chance; every "
    "statistic is read against white noise, a detrended random walk, shuffled and "
    "block-bootstrapped returns. Per-bar features are causal (one-sided MODWT, trailing "
    "windows); scalograms and ridges are NON-CAUSAL and never features. No signal, entry, "
    "exit, sizing or cost exists in this layer, and no parameter was chosen for performance."
)

HYPOTHESES: dict[str, str] = {
    "WAVE-H-001": "Low wavelet entropy is associated with stronger subsequent residual "
                  "normalisation: P(|eps_t+10| < |eps_t| | |Z| > 2) is higher in the lowest "
                  "than the highest entropy quartile by more than under every pipeline null "
                  "(gap > 2 standard errors).",
    "WAVE-H-002": "A rising fast/slow energy ratio (its change over N/4 bars) predicts "
                  "near-term residual dynamics: its rank IC with the future OU-innovation "
                  "magnitude over 5 bars exceeds every pipeline null's for the same test, with "
                  "a stable yearly sign.",
    "WAVE-H-003": "Wavelet dominant-scale persistence (run length) adds out-of-sample "
                  "information beyond the FFT features, FFT dominant-period persistence "
                  "included: adding it to model C raises out-of-sample R^2 for the future "
                  "absolute residual reduction (5 bars) in at least 3 of 4 chronological folds "
                  "and by more than on every incremental null.",
    "WAVE-H-004": "The OU innovation retains wavelet structure (entropy, top-3 share, fast/slow "
                  "log ratio) beyond shuffled and block-bootstrapped returns (|effect| > 0.25 "
                  "null IQR against both, for all three).",
    "WAVE-H-005": "The dominant wavelet band persists across disjoint windows (lag N) more "
                  "often than under every null control (by more than 0.02).",
    "WAVE-H-006": "A causal wavelet feature carries forward information (rank IC) larger than "
                  "any IC on the pipeline nulls of the same study, with a stable yearly sign.",
    "WAVE-H-007": "Wavelet features add out-of-sample explanatory power beyond statistical + OU "
                  "+ FFT features (model D vs C): better in at least 3 of 4 chronological folds "
                  "and by more than on every incremental null.",
    "WAVE-H-008": "Wavelet shape (median entropy) depends on the volatility regime more than on "
                  "every pipeline null (range across quartiles > 2x the largest null range and "
                  "> 0.01).",
    "WAVE-H-009": "Wavelet shape (median entropy) depends on trend strength more than on every "
                  "pipeline null (range across quintiles > 2x the largest null range and "
                  "> 0.01).",
    "WAVE-H-010": "Localised energy bursts (burst Z above the threshold) are more frequent than "
                  "under every null control (by more than 10 % relative).",
    "WAVE-H-011": "A wavelet feature is not a near-deterministic function of the same window's "
                  "FFT features (out-of-sample R^2 from FFT features < 0.8).",
    "WAVE-H-012": "A wavelet feature is distinct from existing regression, OU, volatility and "
                  "FFT features (max |Spearman| < 0.3).",
    "WAVE-H-013": "Offline CWT ridges last longer (median duration in cycles) in the real "
                  "series than in every null control, in at least 4 of 5 slices "
                  "(NON-CAUSAL, descriptive).",
}

#: Hypotheses formed *after* seeing registered results. They are tested and
#: recorded in the ledger like any other - every test counts toward the
#: multiple-testing record - but never written to hypotheses_registered.json,
#: and the ``WAVE-P-`` prefix keeps them apart from the ``WAVE-H-`` set. Such a
#: test can only qualify a registered finding, never promote one.
POST_HOC_HYPOTHESES: dict[str, str] = {
    "WAVE-P-001": "POST-HOC (not pre-registered; formed after WAVE-H-007): the wavelet "
                  "block's increment over model C survives a baseline widened with "
                  "longer-horizon volatility (ln RV and ln mean |OU innovation| over 64, "
                  "256 and 1024 bars) and/or hour-of-day indicators - mean D - C > 0 in at "
                  "least 3 of 4 chronological folds and above every incremental null under "
                  "the same widened baseline (the WAVE-H-007 rule).",
}

_WAVELET_SOURCES = (
    "features/wavelet.py", "features/wavelet_causal.py", "features/wavelet_energy.py",
    "features/wavelet_entropy.py", "features/wavelet_config.py",
    "research/wavelet_analysis.py", "research/wavelet_predictiveness.py",
    "research/wavelet_stability.py", "research/wavelet_reports.py",
    "research/spectral_nulls.py", "research/spectral_ic.py",
)
_PACKAGES = ("PyWavelets", "numpy", "scipy", "polars", "pyarrow")


@dataclass
class WaveletStudy:
    """Outcome of one (timeframe, input series, rolling window) study."""

    timeframe: str
    series: str
    window: int
    output_dir: Path
    summary: dict[str, Any] = field(default_factory=dict)
    files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    reused: bool = False


def source_roles(wavelet: WaveletConfig) -> dict[str, str]:
    """Which analyses each source gets.

    ``real`` everything; ``incremental`` nulls (random walk, bootstrap) also
    the FFT comparison and models A-D; ``core`` nulls the pipeline analyses
    (stability, conditioning, outcomes, IC); ``shape`` nulls (white noise,
    other block sizes) the distributions, persistence, drift and bursts.
    """
    roles = {"real": "real"}
    for name in wavelet.controls.enabled():
        roles[name] = ("incremental" if name in wavelet.incremental.controls
                       else "core" if name in wavelet.ic.controls else "shape")
    return roles


def _code_version() -> str:
    base = Path(__file__).resolve().parent.parent
    return code_fingerprint(base / rel for rel in _WAVELET_SOURCES)


def register_hypotheses(wavelet: WaveletConfig) -> Path:
    """Write the hypothesis definitions before any result (kept once registered)."""
    path = wavelet.results_path / "hypotheses_registered.json"
    ensure_dir(path.parent)
    if path.exists():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        if recorded.get("hypotheses") == HYPOTHESES:
            return path
        recorded.setdefault("previous_versions", []).append(
            {"registered_utc": recorded.get("registered_utc"),
             "hypotheses": recorded.get("hypotheses")})
    else:
        recorded = {}
    recorded.update({"registered_utc": utc_now_iso(), "hypotheses": HYPOTHESES,
                     "null_effect_threshold": NULL_EFFECT_THRESHOLD,
                     "note": "Verdict rules are fixed in wavelet_reports._ledger_entries."})
    atomic_write_text(path, json.dumps(recorded, indent=2) + "\n")
    return path


# ---------------------------------------------------------------------------
# Integrity gate
# ---------------------------------------------------------------------------
def integrity_gate(config: Config, regression: RegressionConfig, ou: OUConfig,
                   spectral: SpectralConfig, wavelet: WaveletConfig, timeframe: str, *,
                   require_spectral_features: bool = True) -> dict[str, Any]:
    """The Prompt #5 gate, plus: the stored FFT features must match the dataset.

    Ticks complete, bars built from them, regression features current (the
    spectral gate); the wavelet and spectral layers must read the same
    residual and innovation (same regression and OU windows); and every
    input series' Prompt #5 feature set for this timeframe must exist and
    carry the current tick, bar and engine versions.
    """
    gate = spectral_integrity_gate(config, regression, ou, spectral, timeframe)
    problems, notes = list(gate["problems"]), list(gate["notes"])
    if (spectral.regression_window, spectral.ou_window) != (wavelet.regression_window,
                                                            wavelet.ou_window):
        problems.append("the wavelet and spectral layers use different regression/OU windows")
    lineage = gate["lineage"]
    if require_spectral_features:
        for series in wavelet.input_series:
            manifest = (spectral.features_path / f"timeframe={timeframe}" / f"source={series}"
                        / f"fft_window={spectral.representative_fft_window}" / "_manifest.json")
            if not manifest.exists():
                problems.append(f"no Prompt #5 FFT features for {timeframe} {series}")
                continue
            recorded = json.loads(manifest.read_text(encoding="utf-8"))
            current = (recorded.get("tick_dataset_version") == lineage.get("tick_dataset_version")
                       and recorded.get("bar_dataset_version") == lineage.get(
                           "bar_dataset_version")
                       and recorded.get("engine_fingerprint") == spectral.engine_fingerprint())
            if current:
                notes.append(f"Prompt #5 FFT features {series}: current")
            else:
                problems.append(f"Prompt #5 FFT features for {timeframe} {series} are stale")
    else:
        notes.append("stored FFT features not checked (FFT features are recomputed in-process)")
    return {**gate, "passed": not problems, "problems": problems, "notes": notes}


# ---------------------------------------------------------------------------
# One pass: one source, one series, one window
# ---------------------------------------------------------------------------
def _lags(window: int, wavelet: WaveletConfig) -> list[int]:
    lags = {int(h) for h in wavelet.persistence.lags_in_bars}
    lags |= {max(1, int(round(f * window))) for f in wavelet.persistence.lags_in_windows}
    return sorted(lags)


def _distribution(frame: pl.DataFrame, source: str) -> pl.DataFrame:
    rows = []
    for metric in WAVELET_METRICS:
        if metric not in frame.columns:
            continue
        values = frame[metric].drop_nulls()
        if values.len() == 0:
            continue
        rows.append({"source": source, "metric": metric, "count": values.len(),
                     "mean": values.mean(), "std": values.std(),
                     **{f"p{q}": values.quantile(q / 100) for q in (5, 25, 50, 75, 95)}})
    return pl.DataFrame(rows, infer_schema_length=None)


def _energy_table(result: RollingWavelet, source: str) -> pl.DataFrame:
    layout = result.layout
    rows = []
    for i, band in enumerate(layout.describe()):
        values = result.shares[i][result.valid]
        if values.size == 0:
            continue
        rows.append({"source": source, **band,
                     "median_share": float(np.median(values)),
                     "mean_share": float(values.mean()),
                     "share_p25": float(np.quantile(values, 0.25)),
                     "share_p75": float(np.quantile(values, 0.75)),
                     "dominant_fraction": float(np.mean(result.dominant[result.valid] == i))})
    return pl.DataFrame(rows, infer_schema_length=None)


def _histograms(frame: pl.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for metric in ("wavelet_entropy", "wavelet_top3_scale_share",
                   "wavelet_fast_slow_log_ratio"):
        values = frame[metric].drop_nulls().to_numpy()
        if values.size:
            lo, hi = ((0.0, 1.0) if metric != "wavelet_fast_slow_log_ratio"
                      else tuple(np.quantile(values, [0.005, 0.995])))
            density, edges = np.histogram(values, bins=60, range=(lo, hi), density=True)
            out[metric] = ((edges[:-1] + edges[1:]) / 2, density)
    return out


def _existing_features(source: SourceData, series: str, window: int) -> dict[str, np.ndarray]:
    x = pl.Series(source.columns[series]).fill_nan(None)
    acf1 = pl.DataFrame({"x": x}).select(
        pl.rolling_corr(pl.col("x"), pl.col("x").shift(1), window_size=window,
                        min_samples=window)).to_series().to_numpy()
    c = source.columns
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "abs_residual_zscore": np.abs(c["residual_zscore"]),
            "regression_r_squared": c["r_squared"],
            "abs_regression_slope": np.abs(c["regression_slope"]),
            "abs_ou_zscore": np.abs(c["ou_zscore"]),
            "log_ou_half_life": np.log(np.where(c["ou_valid"] > 0, c["ou_half_life_bars"],
                                                np.nan)),
            "trailing_volatility": c["trailing_volatility"],
            "log_realized_vol_5": 0.5 * np.log(trailing_mean(c["log_return"] ** 2, 5)),
            "log_realized_vol_20": 0.5 * np.log(trailing_mean(c["log_return"] ** 2, 20)),
            "log_mean_abs_innovation_20": np.log(trailing_mean(np.abs(c["ou_innovation"]), 20)),
            f"rolling_acf1_{series}": acf1,
        }


def _redundancy(result: RollingWavelet, spectrum: RollingSpectrum | None, source: SourceData,
                series: str, wavelet: WaveletConfig, max_rows: int = 200_000
                ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Spearman / Pearson among wavelet, FFT and existing features; redundancy clusters."""
    rows = np.flatnonzero(result.valid)
    if rows.size > max_rows:
        rows = rows[np.linspace(0, rows.size - 1, max_rows).round().astype(np.int64)]
    wav = ic_feature_columns(result, wavelet.ic.features, rows)
    fft = ({f"fft_{k}": v for k, v in fft_feature_columns(spectrum, FFT_FEATURES, rows).items()}
           if spectrum is not None else {})
    existing = {k: np.asarray(v, dtype=np.float64)[rows]
                for k, v in _existing_features(source, series, result.window).items()}
    frame = pl.DataFrame({**wav, **fft, **existing}).with_columns(pl.all().fill_nan(None))
    names = frame.columns
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    values = frame.select(
        *[pl.corr(a, b).alias(f"p|{a}|{b}") for a, b in pairs],
        *[pl.corr(a, b, method="spearman").alias(f"s|{a}|{b}") for a, b in pairs],
    ).row(0, named=True)
    correlation = pl.DataFrame([{"a": a, "b": b, "pearson": values[f"p|{a}|{b}"],
                                 "spearman": values[f"s|{a}|{b}"]} for a, b in pairs],
                               infer_schema_length=None)
    groups = {"existing": list(existing), "fft": list(fft), "wavelet": list(wav)}
    strong = [(a, b) for a, b in pairs
              if values[f"s|{a}|{b}"] is not None and abs(values[f"s|{a}|{b}"]) >= 0.7]
    parent = {n: n for n in names}

    def root(n: str) -> str:
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    for a, b in strong:
        parent[root(a)] = root(b)
    clusters: dict[str, list[str]] = {}
    for n in names:
        clusters.setdefault(root(n), []).append(n)
    cluster_id = {n: i for i, members in enumerate(sorted(clusters.values(), key=len,
                                                          reverse=True)) for n in members}
    red = []
    for f in wav:
        entry: dict[str, Any] = {"feature": f, "cluster": cluster_id[f],
                                 "cluster_members": ",".join(sorted(
                                     clusters[root(f)]))}
        for group, members in groups.items():
            best, best_name = None, None
            for other in members:
                if other == f:
                    continue
                key = f"s|{f}|{other}" if f"s|{f}|{other}" in values else f"s|{other}|{f}"
                rho = values.get(key)
                if rho is not None and (best is None or abs(rho) > best):
                    best, best_name = abs(rho), other
            entry[f"max_abs_spearman_{group}"] = best
            entry[f"most_similar_{group}"] = best_name
        red.append(entry)
    return correlation, pl.DataFrame(red, infer_schema_length=None)


def analyse_pass(result: RollingWavelet, spectrum: RollingSpectrum | None, source: SourceData,
                 series: str, wavelet: WaveletConfig, spectral: SpectralConfig,
                 research: ResearchConfig, *, role: str) -> dict[str, Any]:
    """Every statistic of one (source, series, window) that the source's role allows."""
    name = source.name
    window = result.window
    out: dict[str, Any] = {"source": name, "role": role, "counts": result.counts}
    extra: dict[str, np.ndarray] | None = None
    if role != "shape" and source.has_pipeline:
        extra = {k: source.columns[k] for k in ("trailing_volatility", "regression_slope",
                                                "ou_half_life_bars", "ou_valid",
                                                "ou_state_code") if k in source.columns}
    frame = wavelet_metrics_frame(result, source.timestamps, extra)
    out["distribution"] = _distribution(frame, name)
    out["energy"] = _energy_table(result, name)
    out["persistence"] = band_persistence(result, _lags(window, wavelet),
                                          tolerance=wavelet.persistence.band_tolerance,
                                          source=name)
    out["runs"] = run_summary(result, source=name)
    out["drift"] = drift_summary(result, sorted({1, result.drift_lag, window}),
                                 wavelet.stability.change_quantiles, source=name)
    out["bursts"] = burst_summary(result, wavelet.causal_features.burst_z_threshold,
                                  source=name)
    out["histograms"] = _histograms(frame)
    drift_values = result.scale_drift[np.isfinite(result.scale_drift)]
    if drift_values.size > 50_000:
        drift_values = drift_values[np.linspace(0, drift_values.size - 1, 50_000)
                                    .round().astype(np.int64)]
    out["drift_values"] = drift_values
    summaries = wavelet_summaries()
    if role != "shape" and source.has_pipeline:
        years = pl.DataFrame({"timestamp": source.timestamps}).with_columns(
            pl.col("timestamp").dt.year().cast(pl.Utf8).alias("segment")).group_by(
            "segment").agg(pl.len().alias("bars"))
        out["yearly"] = segment_table(frame, by="year", source=name, total_bars=years,
                                      summaries=summaries)
        out["quarterly"] = segment_table(frame, by="quarter", source=name, summaries=summaries)
        out["era"] = era_table(frame, eras=wavelet.stability.eras, source=name,
                               summaries=summaries)
        out["conditioning"] = conditioning_table(frame, wavelet.conditioning, source=name,
                                                 summaries=summaries)
        out["intraday"] = intraday_table(frame, wavelet.conditioning, research, source=name,
                                         summaries=summaries)
        out["extremes"] = extreme_conditioning(result, source, wavelet, source_name=name)
        max_rows = wavelet.ic.max_rows if role == "real" else wavelet.ic.max_rows_control
        ic, periods, tests = wavelet_ic_study(result, source, wavelet, max_rows=max_rows,
                                              source_name=name)
        out["ic"], out["ic_periods"], out["ic_tests"] = ic, periods, tests
    del frame
    if role in ("real", "incremental") and spectrum is not None and source.has_pipeline:
        out["incremental"] = incremental_study(result, spectrum, source, wavelet,
                                               source_name=name)
        pairs, determinism = fft_comparison(result, spectrum, wavelet, source=name)
        out["fft_pairs"], out["fft_determinism"] = pairs, determinism
        out["dyadic"] = dyadic_comparison(source.columns[series], result, spectral, source=name)
    if role == "real":
        correlation, redundancy = _redundancy(result, spectrum, source, series, wavelet)
        out["correlation"], out["redundancy"] = correlation, redundancy
        valid_rows = np.flatnonzero(result.valid)
        features = ic_feature_columns(result, wavelet.ic.features, valid_rows)
        out["coverage"] = {k: float(np.isfinite(v).mean()) if v.size else None
                           for k, v in features.items()}
        out["coverage_of_bars"] = float(result.valid.mean()) if result.valid.size else None
        valid = np.flatnonzero(result.valid)
        recent = valid[-wavelet.plots.sample_bars:]
        out["plot_series"] = {
            "timestamps": source.timestamps.gather(pl.Series(recent)).to_numpy(),
            "dominant_period": result.dominant_period_bars[recent],
            "entropy": result.entropy[recent],
            "fast_slow": result.fast_slow_log_ratio[recent],
        }
        if spectrum is not None:
            both = np.flatnonzero(result.valid & spectrum.valid)
            if both.size > 50_000:
                both = both[np.linspace(0, both.size - 1, 50_000).round().astype(np.int64)]
            out["period_pairs"] = (spectrum.dominant_period_bars[both],
                                   result.dominant_period_bars[both])
    gc.collect()
    return out


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------
def _pick(frame: Any, where: dict[str, Any], column: str) -> Any:
    if not isinstance(frame, pl.DataFrame) or frame.is_empty():
        return None
    part = frame
    for k, v in where.items():
        if k not in part.columns:
            return None
        part = part.filter(pl.col(k) == v)
    return None if part.is_empty() else part[column][0]


def _known(value: Any) -> float | None:
    """A finite number, or None. Every verdict reads its inputs through this.

    An undefined statistic arrives as NaN as often as None (a fast/slow ratio
    with no slow band, a Spearman of an all-NaN column), and NaN fails every
    comparison silently: ``abs(nan) > ceiling`` is False, so the IC verdict
    became "within_null_range", and ``nan < 0.3`` / ``nan < 0.7`` are False,
    so redundancy became "redundant". Undefined is "untested", never a verdict.
    """
    number = num(value.item() if isinstance(value, np.generic) else value)
    return number if np.isfinite(number) else None


def null_comparison(passes: dict[str, dict[str, Any]]) -> pl.DataFrame:
    """Distributional metrics (effects in null IQRs) and scalar statistics, real beside nulls."""
    if "real" not in passes:
        return pl.DataFrame()
    distribution = stack_tables(p["distribution"] for p in passes.values())
    nulls = [s for s in passes if s != "real"]
    rows = distribution_comparison(distribution, nulls, WAVELET_METRICS)
    window = next(iter(passes.values()))["window"]

    def persistence(p: dict[str, Any], column: str) -> Any:
        return _pick(p.get("persistence"), {"lag": window}, column)

    scalar = {
        "same_band_lag_N": lambda p: persistence(p, "same_band"),
        "similar_band_lag_N": lambda p: persistence(p, "similar_band"),
        "switching_fraction_lag1": lambda p: (p.get("runs") or {}).get("switching_fraction_lag1"),
        "run_length_median": lambda p: (p.get("runs") or {}).get("run_length_median"),
        "burst_fraction": lambda p: _pick(p.get("bursts"), {"band": "any"}, "burst_fraction"),
        "drift_continuation_lag_N": lambda p: _pick(p.get("drift"), {"lag": window},
                                                    "continuation"),
        "max_abs_rank_ic": lambda p: (float(p["ic"]["rank_ic"].abs().max())
                                      if isinstance(p.get("ic"), pl.DataFrame)
                                      and not p["ic"].is_empty() else None),
    }
    for metric, getter in scalar.items():
        row = {"metric": metric, "real": getter(passes["real"])}
        for null in nulls:
            row[null] = getter(passes[null])
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def _stable(share: Any) -> bool:
    return share is not None and (share >= 0.75 or share <= 0.25)


def _increment_verdict(delta: float | None, folds_better: int | None,
                       null_deltas: list[float]) -> str:
    """The WAVE-H-007 rule (shared by WAVE-P-001): D beats C in >= 3 folds, above every null.

    *delta* is the fold-mean improvement of D over C (R^2 gain, or log-loss
    reduction); *null_deltas* the same on each incremental null.
    """
    if delta is None:
        return "untested"
    beats = delta > 0 and (folds_better or 0) >= 3 and all(delta > v for v in null_deltas)
    return "incremental_beyond_nulls" if beats else "no_incremental_information"


def post_hoc_baseline_entries(summary: pl.DataFrame, *, timeframe: str, series: str,
                              window: int, dataset_version: str | None,
                              study: str) -> list[dict[str, Any]]:
    """WAVE-P-001 ledger rows: the WAVE-H-007 rule under each widened baseline.

    *summary* is :func:`incremental_summary` output with a ``baseline``
    column naming a :data:`BASELINE_EXTENSIONS` entry; each real row is read
    against the nulls fitted with the *same* widened baseline. ``original``
    is WAVE-H-007 itself and gets no post-hoc row.
    """
    entries: list[dict[str, Any]] = []
    real = summary.filter((pl.col("source") == "real") & (pl.col("baseline") != "original")
                          & pl.col("metric").is_in(["oos_r2", "log_loss"]))
    for r in real.iter_rows(named=True):
        nulls = summary.filter((pl.col("source") != "real")
                               & (pl.col("baseline") == r["baseline"])
                               & (pl.col("target") == r["target"])
                               & (pl.col("horizon") == r["horizon"])
                               & (pl.col("metric") == r["metric"]))
        null_deltas = [v for v in (_known(d) for d in nulls["delta_D_C"].to_list())
                       if v is not None]
        delta = _known(r.get("delta_D_C"))
        entries.append({
            "hypothesis_id": "WAVE-P-001", "definition": POST_HOC_HYPOTHESES["WAVE-P-001"],
            "timeframe": timeframe, "input_series": series, "window": window,
            "feature": f"wavelet_block|baseline={r['baseline']}", "target": r["target"],
            "horizon": r["horizon"], "metric": f"mean_delta_{r['metric']}_D_vs_C",
            "value": delta, "control_value": max(null_deltas) if null_deltas else None,
            "verdict": _increment_verdict(delta, r.get("folds_d_beats_c"), null_deltas),
            "details": {"post_hoc": True, "baseline": BASELINE_EXTENSIONS[r["baseline"]],
                        "folds_d_beats_c": r.get("folds_d_beats_c"), "folds": r.get("folds"),
                        "C": r.get("C"), "D": r.get("D"),
                        "null_deltas": dict(zip(nulls["source"].to_list(),
                                                nulls["delta_D_C"].to_list(), strict=True))},
            "dataset_version": dataset_version, "study": study})
    return entries


def _ledger_entries(passes: dict[str, dict[str, Any]], comparison: pl.DataFrame, *,
                    timeframe: str, series: str, window: int, dataset_version: str | None,
                    study: str, summary: pl.DataFrame) -> list[dict[str, Any]]:
    """Every hypothesis this study can speak to, with its a-priori verdict."""
    real = passes.get("real", {})
    pipeline = [s for s, p in passes.items() if s != "real" and p["role"] in ("core",
                                                                              "incremental")]
    incremental = [s for s, p in passes.items() if p["role"] == "incremental"]
    entries: list[dict[str, Any]] = []
    base = {"timeframe": timeframe, "input_series": series, "window": window,
            "dataset_version": dataset_version, "study": study}

    def add(hid: str, **values: Any) -> None:
        entries.append({**base, "hypothesis_id": hid, "definition": HYPOTHESES[hid], **values})

    def comp(metric: str) -> dict[str, Any]:
        part = comparison.filter(pl.col("metric") == metric)
        return part.row(0, named=True) if part.height else {}

    # H-001: entropy quartiles vs residual normalisation after |Z| > 2
    def spread(source: str) -> tuple[float | None, float | None]:
        table = passes.get(source, {}).get("extremes")
        if not isinstance(table, pl.DataFrame) or table.is_empty():
            return None, None
        part = table.filter((pl.col("extreme") == "residual_z") & (pl.col("threshold") == 2.0)
                            & (pl.col("horizon") == 10)
                            & (pl.col("conditioning") == "wavelet_entropy"))
        q = {r["bucket"]: r for r in part.iter_rows(named=True)}
        if "Q1" not in q or "Q4" not in q:
            return None, None
        p1, p4 = _known(q["Q1"]["prob_shrinks"]), _known(q["Q4"]["prob_shrinks"])
        if p1 is None or p4 is None:
            return None, None
        return p1 - p4, _known(np.hypot(num(q["Q1"]["std_error"]), num(q["Q4"]["std_error"])))

    d_real, se_real = spread("real")
    nulls_h1 = {s: spread(s) for s in pipeline}
    known = {s: v for s, v in nulls_h1.items() if v[0] is not None}
    if d_real is not None and known:
        worst = max(known, key=lambda s: known[s][0] or -np.inf)
        d_null, se_null = known[worst]
        gap = d_real - (d_null or 0.0)
        se = float(np.hypot(se_real or 0.0, se_null or 0.0))
        add("WAVE-H-001", feature="wavelet_entropy", target="residual_shrinks", horizon=10,
            metric="P(shrink|Q1) - P(shrink|Q4)", value=d_real, control_value=d_null,
            verdict="exceeds_all_nulls" if se > 0 and gap > 2 * se else "within_null_range",
            details={s: v[0] for s, v in known.items()} | {"std_error": se})

    # H-002 and H-006: IC against the pipeline nulls
    ic = real.get("ic")
    if isinstance(ic, pl.DataFrame) and not ic.is_empty():
        null_tables = [passes[s]["ic"] for s in pipeline
                       if isinstance(passes[s].get("ic"), pl.DataFrame)
                       and not passes[s]["ic"].is_empty()]
        ceilings = [_known(t["rank_ic"].abs().max()) for t in null_tables]
        ceiling = max((c for c in ceilings if c is not None), default=None)
        for r in ic.iter_rows(named=True):
            value = _known(r.get("rank_ic"))
            share = r.get("yearly_rank_ic_positive_share")
            verdict = ("untested" if value is None or ceiling is None else
                       "exceeds_null_max_with_stable_sign" if abs(value) > ceiling
                       and _stable(share) else
                       "exceeds_null_max_unstable_sign" if abs(value) > ceiling
                       else "within_null_range")
            add("WAVE-H-006", feature=r["feature"], target=r["target"], horizon=r["horizon"],
                metric="rank_ic", value=value, control_value=ceiling, verdict=verdict,
                details={"yearly_mean": r.get("yearly_rank_ic_mean"),
                         "yearly_t": r.get("yearly_rank_ic_t"),
                         "yearly_positive_share": share})
        test = {"feature": "wavelet_fast_slow_change", "target": "future_innovation_magnitude",
                "horizon": 5}
        value = _known(_pick(ic, test, "rank_ic"))
        share = _pick(ic, test, "yearly_rank_ic_positive_share")
        null_values = [_known(_pick(t, test, "rank_ic")) for t in null_tables]
        null_max = max((abs(v) for v in null_values if v is not None), default=None)
        if value is not None and null_max is not None:
            add("WAVE-H-002", **test, metric="rank_ic", value=value, control_value=null_max,
                verdict=("exceeds_nulls_stable_sign" if abs(value) > null_max and _stable(share)
                         else "exceeds_nulls_unstable_sign" if abs(value) > null_max
                         else "within_null_range"),
                details={"yearly_positive_share": share})

    # H-003 and H-007: incremental information
    def feature_gain(source: str, model: str, target: str, horizon: int
                     ) -> tuple[float | None, int]:
        inc = passes.get(source, {}).get("incremental") or {}
        lin, feat = inc.get("linear"), inc.get("by_feature")
        if not isinstance(lin, pl.DataFrame) or lin.is_empty():
            return None, 0
        base_r2 = lin.filter((pl.col("model") == "C") & (pl.col("target") == target)
                             & (pl.col("horizon") == horizon)).select("fold", "oos_r2")
        if not isinstance(feat, pl.DataFrame) or feat.is_empty():
            return None, 0
        extra = feat.filter((pl.col("model") == model) & (pl.col("target") == target)
                            & (pl.col("horizon") == horizon)).select("fold", "oos_r2")
        joined = base_r2.join(extra, on="fold", suffix="_x").drop_nulls()
        if joined.is_empty():
            return None, 0
        gain = joined["oos_r2_x"] - joined["oos_r2"]
        return _known(gain.mean()), int((gain > 0).sum())

    g_real, folds_pos = feature_gain("real", "C+wavelet_log_run_length",
                                     "future_abs_residual_reduction", 5)
    if g_real is not None:
        g_nulls = {s: feature_gain(s, "C+wavelet_log_run_length",
                                   "future_abs_residual_reduction", 5)[0] for s in incremental}
        known_g = [v for v in g_nulls.values() if v is not None]
        beats = folds_pos >= 3 and g_real > 0 and all(g_real > v for v in known_g)
        add("WAVE-H-003", feature="wavelet_log_run_length", target="future_abs_residual_reduction",
            horizon=5, metric="mean_delta_oos_r2_vs_C", value=g_real,
            control_value=max(known_g) if known_g else None,
            verdict="incremental_beyond_nulls" if beats else "no_incremental_information",
            details={"folds_positive": folds_pos, **g_nulls})
    if isinstance(summary, pl.DataFrame) and not summary.is_empty() and "delta_D_C" in \
            summary.columns:
        for r in summary.filter((pl.col("source") == "real")
                                & (pl.col("metric").is_in(["oos_r2", "log_loss"]))
                                ).iter_rows(named=True):
            null_deltas = [v for v in (_known(d) for d in summary.filter(
                (pl.col("source") != "real") & (pl.col("target") == r["target"])
                & (pl.col("horizon") == r["horizon"]) & (pl.col("metric") == r["metric"])
            )["delta_D_C"].to_list()) if v is not None]
            delta = _known(r.get("delta_D_C"))
            add("WAVE-H-007", feature="wavelet_block", target=r["target"], horizon=r["horizon"],
                metric=f"mean_delta_{r['metric']}_D_vs_C", value=delta,
                control_value=max(null_deltas) if null_deltas else None,
                verdict=_increment_verdict(delta, r.get("folds_d_beats_c"), null_deltas),
                details={"folds_d_beats_c": r.get("folds_d_beats_c"), "folds": r.get("folds")})

    # H-004: OU innovation vs shuffled and bootstrapped returns
    if series == "ou_innovation":
        effects = {}
        for metric in ("median_wavelet_entropy", "median_wavelet_top3_scale_share",
                       "median_wavelet_fast_slow_log_ratio"):
            row = comp(metric)
            for null in ("shuffled_returns", "block_bootstrap"):
                effects[f"{metric}|{null}"] = row.get(f"effect_vs_{null}")
        verdict = ("untested" if not any(v is not None for v in effects.values()) else
                   "exceeds_nulls" if all(v is not None and abs(v) > NULL_EFFECT_THRESHOLD
                                          for v in effects.values()) else "within_a_null")
        add("WAVE-H-004", feature="wavelet_shape", metric="effect_vs_null_iqr", value=None,
            verdict=verdict, details=effects)

    # H-005 and H-010: persistence and bursts against every null
    for hid, metric, feature, margin in (("WAVE-H-005", "same_band_lag_N", "dominant_band",
                                          0.02),
                                         ("WAVE-H-010", "burst_fraction", "burst", None)):
        row = comp(metric)
        value = _known(row.get("real"))
        others = [v for k, v in row.items() if k not in ("metric", "real", "verdict")
                  and _known(v) is not None and not k.startswith("effect_vs_")]
        if value is None or not others:
            continue
        if margin is not None:
            exceeds = all(value > v + margin for v in others)
        else:
            exceeds = all(value > 1.1 * v for v in others)
        add(hid, feature=feature, metric=metric, value=value, control_value=max(others),
            verdict="exceeds_all_nulls" if exceeds else "within_a_null",
            details={k: v for k, v in row.items() if k not in ("metric", "real", "verdict")})

    # H-008 and H-009: conditioning ranges against every pipeline null
    for hid, variable in (("WAVE-H-008", "volatility"), ("WAVE-H-009", "trend")):
        ranges: dict[str, float] = {}
        for src in ["real", *pipeline]:
            table = passes.get(src, {}).get("conditioning")
            if isinstance(table, pl.DataFrame) and not table.is_empty():
                part = table.filter(pl.col("variable") == variable)[
                    "median_wavelet_entropy"].drop_nulls()
                spread_ = _known(num(part.max()) - num(part.min())) if part.len() else None
                if spread_ is not None:
                    ranges[src] = spread_
        if "real" in ranges and len(ranges) > 1:
            null_max = max(v for k, v in ranges.items() if k != "real")
            add(hid, feature="wavelet_entropy", metric="range_of_bucket_medians",
                value=ranges["real"], control_value=null_max,
                verdict=("exceeds_all_nulls" if ranges["real"] > 2 * null_max
                         and ranges["real"] > 0.01 else "consistent_with_nulls"),
                details=ranges)

    # H-011 and H-012: distinct from FFT / from everything existing
    det = real.get("fft_determinism")
    if isinstance(det, pl.DataFrame) and not det.is_empty():
        for r in det.iter_rows(named=True):
            r2 = _known(r.get("oos_r2_from_fft"))
            add("WAVE-H-011", feature=r["wavelet_feature"], metric="oos_r2_from_fft", value=r2,
                verdict=("untested" if r2 is None else "near_deterministic_of_fft" if r2 >= 0.8
                         else "distinct_from_fft"))
    red = real.get("redundancy")
    if isinstance(red, pl.DataFrame) and not red.is_empty():
        for r in red.iter_rows(named=True):
            values = [_known(r.get("max_abs_spearman_existing")),
                      _known(r.get("max_abs_spearman_fft"))]
            known_values = [v for v in values if v is not None]
            m = max(known_values) if known_values else None
            add("WAVE-H-012", feature=r["feature"], metric="max_abs_spearman_existing_or_fft",
                value=m, verdict=("untested" if m is None else "distinct" if m < 0.3 else
                                  "partially_redundant" if m < 0.7 else "redundant"),
                details={"most_similar_existing": r.get("most_similar_existing"),
                         "most_similar_fft": r.get("most_similar_fft")})
    return entries


def _ridge_entries(ridges: pl.DataFrame, wavelet: WaveletConfig, *, timeframe: str,
                   series: str, dataset_version: str | None) -> list[dict[str, Any]]:
    """WAVE-H-013 per CWT wavelet: slices where real ridges outlast every null."""
    entries: list[dict[str, Any]] = []
    if ridges.is_empty():
        return entries
    for (name,), part in ridges.group_by("wavelet"):
        wins, slices = 0, 0
        for (_,), sl in part.group_by("slice"):
            finite = sl.filter(pl.col("median_cycles").is_finite())
            real = finite.filter(pl.col("source") == "real")["median_cycles"]
            nulls = finite.filter(pl.col("source") != "real")["median_cycles"]
            if real.len() and nulls.len():
                slices += 1
                wins += int(num(real[0]) > num(nulls.max()))
        if slices:
            entries.append({
                "hypothesis_id": "WAVE-H-013", "definition": HYPOTHESES["WAVE-H-013"],
                "timeframe": timeframe, "input_series": series,
                "window": wavelet.cwt.slice_bars, "feature": f"cwt_ridges_{name}",
                "target": "", "horizon": -1, "metric": "slices_real_longer_than_all_nulls",
                "value": float(wins), "control_value": float(slices),
                "verdict": ("longer_than_all_nulls" if slices >= 5 and wins >= 4
                            else "within_null_range"),
                "details": {"label": NON_CAUSAL_LABEL}, "dataset_version": dataset_version,
                "study": f"{timeframe}/{series}/offline"})
    return entries


# ---------------------------------------------------------------------------
# Features on disk (Steps 50-51, 61)
# ---------------------------------------------------------------------------
def _write_features(result: RollingWavelet, source: SourceData, wavelet: WaveletConfig, *,
                    timeframe: str, series: str, lineage: dict[str, Any],
                    regression_feature_version: str | None, provenance: dict[str, Any]
                    ) -> dict[str, Any]:
    """Causal per-bar features as Parquet partitioned by year, with a provenance manifest."""
    frame = wavelet_feature_frame(result, source.timestamps,
                                  float32=wavelet.features_output.float32)
    root = (wavelet.features_path / f"timeframe={timeframe}" / f"source={series}"
            / f"window={result.window}")
    ensure_dir(root)
    for stale in root.rglob("*.parquet"):
        stale.unlink()
    frame = frame.with_columns(pl.col("timestamp").dt.year().alias("__year"))
    written = 0
    for (year,), part in frame.group_by("__year", maintain_order=True):
        directory = ensure_dir(root / f"year={year}")
        tmp = directory / "part-0.parquet.partial"
        part.drop("__year").write_parquet(tmp, compression="zstd")
        tmp.replace(directory / "part-0.parquet")
        written += (directory / "part-0.parquet").stat().st_size
    columns = [c for c in frame.columns if c not in ("__year", "timestamp")]
    manifest = {
        "timeframe": timeframe, "input_series": series, "rolling_window": result.window,
        "rows": frame.height, "valid_rows": int(result.valid.sum()), "bytes": written,
        "causal": True,
        "columns": {c: {"causal": True, "live_safe": True,
                        "lookback_bars": feature_lookback(c, result.window, result.drift_lag),
                        "description": (CAUSAL_FEATURES[c].description if c in CAUSAL_FEATURES
                                        else "band energy share")} for c in columns},
        "join_key": "timestamp",
        "wavelet": {"family": result.layout.wavelet, "method": wavelet.causal_features.method,
                    "levels": result.layout.levels, "padding": "none (one-sided filters)",
                    "bands": result.layout.describe(),
                    "drift_lag_bars": result.drift_lag,
                    "fast_slow_boundary_seconds": result.layout.fast_slow_boundary_seconds,
                    "band_edges_seconds": list(result.layout.band_edges_seconds)},
        "engine_fingerprint": wavelet.engine_fingerprint(),
        "tick_dataset_version": lineage.get("tick_dataset_version"),
        "bar_dataset_version": lineage.get("bar_dataset_version"),
        "source_series": {"regression_feature_version": regression_feature_version,
                          "regression_window": wavelet.regression_window,
                          "ou_window": wavelet.ou_window},
        **provenance,
        "generated_utc": utc_now_iso(),
    }
    atomic_write_text(root / "_manifest.json", json.dumps(clean_json(manifest), indent=1) + "\n")
    return {k: manifest[k] for k in ("rows", "valid_rows", "bytes", "rolling_window")}


# ---------------------------------------------------------------------------
# A timeframe
# ---------------------------------------------------------------------------
def generate_wavelet_timeframe(
    config: Config, regression: RegressionConfig, research: ResearchConfig, ou: OUConfig,
    spectral: SpectralConfig, wavelet: WaveletConfig, *, timeframe: str,
    series: list[str] | None = None, windows: list[int] | None = None, make_plots: bool = True,
    write_features: str | None = None, require_spectral_features: bool = True,
) -> list[WaveletStudy]:
    """Run every requested (series, window) study for one timeframe."""
    policy = write_features or wavelet.features_output.write
    started = time.perf_counter()
    series = list(series or wavelet.input_series)
    windows = sorted(windows or wavelet.windows)
    register_hypotheses(wavelet)
    gate = integrity_gate(config, regression, ou, spectral, wavelet, timeframe,
                          require_spectral_features=require_spectral_features)
    if not gate["passed"]:
        raise RuntimeError(f"{timeframe}: dataset integrity gate failed - "
                           + "; ".join(gate["problems"]))
    lineage = gate["lineage"]
    LOGGER.info("[%s] gate passed: %s", timeframe, "; ".join(gate["notes"]))
    provenance = {"code_fingerprint": _code_version(),
                  "git": git_info(config.project_root),
                  "packages": package_versions(_PACKAGES)}
    roles = source_roles(wavelet)
    real = build_real_source(config, regression, ou, wavelet, timeframe)
    slices = {name: select_slices(real, name, slice_bars=wavelet.cwt.slice_bars,
                                  eras=wavelet.stability.eras)[:wavelet.offline
                                                               .slices_per_timeframe]
              for name in series} if wavelet.analysis.cwt_enabled else {}
    slice_values: dict[str, dict[str, SourceData]] = {name: {} for name in series}
    passes: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    features_written: dict[tuple[str, int], dict[str, Any]] = {}
    sensitivity: dict[str, pl.DataFrame] = {}
    timings: dict[str, float] = {}
    for source_name, role in roles.items():
        t0 = time.perf_counter()
        source = real if source_name == "real" else build_control(
            source_name, real, regression=regression, ou=ou, spectral=wavelet)
        for name in series:
            if name not in source.columns:
                continue
            values = source.series(name)
            offline_source = role != "shape" or source_name == "white_noise"
            if slices.get(name) and offline_source:
                slice_values[name][source_name] = _slice_copy(source, name, slices[name])
            for n in windows:
                t1 = time.perf_counter()
                result = rolling_wavelet(values, window=n, config=wavelet,
                                         bar_seconds=source.bar_seconds,
                                         missing_slots=source.missing_slots)
                spectrum = None
                if role in ("real", "incremental"):
                    spectrum = rolling_spectrum(values, fft_window=n, config=spectral,
                                                bar_seconds=source.bar_seconds,
                                                missing_slots=source.missing_slots)
                analysis = analyse_pass(result, spectrum, source, name, wavelet, spectral,
                                        research, role=role)
                analysis["window"] = n
                analysis["layout"] = result.layout.describe()
                analysis["drift_lag"] = result.drift_lag
                passes.setdefault((name, n), {})[source_name] = analysis
                write = policy == "all" or (policy == "representative"
                                            and n == wavelet.representative_window)
                if write and source_name == "real":
                    features_written[(name, n)] = _write_features(
                        result, source, wavelet, timeframe=timeframe, series=name,
                        lineage=lineage,
                        regression_feature_version=gate.get("regression_feature_version"),
                        provenance=provenance)
                if source_name == "real" and n == wavelet.representative_window:
                    sensitivity[name] = _family_sensitivity(values, result, source, wavelet)
                del result, spectrum
                gc.collect()
                LOGGER.info("[%s %s N=%d %s] %.1fs", timeframe, name, n, source_name,
                            time.perf_counter() - t1)
        timings[source_name] = time.perf_counter() - t0
        if source_name != "real":
            del source
            gc.collect()
    studies = []
    ridge_tables = {}
    for name in series:
        if slices.get(name):
            ridge_tables[name] = _offline(name, slices[name], slice_values[name], wavelet,
                                          timeframe=timeframe, make_plots=make_plots,
                                          lineage=lineage)
    for (name, n), by_source in passes.items():
        studies.append(_write_study(
            by_source, wavelet, gate, timeframe=timeframe, series=name, window=n,
            features=features_written.get((name, n)), seconds=time.perf_counter() - started,
            timings=timings, make_plots=make_plots, provenance=provenance,
            sensitivity=sensitivity.get(name) if n == wavelet.representative_window else None))
    if wavelet.ledger.enabled:
        entries = []
        for name, ridges in ridge_tables.items():
            entries += _ridge_entries(ridges, wavelet, timeframe=timeframe, series=name,
                                      dataset_version=lineage.get("tick_dataset_version"))
        ResearchLedger(wavelet.ledger_path).upsert(entries)
    return studies


def _slice_copy(source: SourceData, series: str, slices: list[dict[str, Any]]) -> SourceData:
    """Just the slice positions of one series (the full source is not kept)."""
    n = source.size
    keep = np.full(n, np.nan)
    for spec in slices:
        keep[spec["start"]:spec["end"] + 1] = source.columns[series][spec["start"]:spec["end"] + 1]
    return SourceData(name=source.name, timestamps=source.timestamps, columns={series: keep},
                      missing_slots=source.missing_slots, bar_seconds=source.bar_seconds)


def _family_sensitivity(values: np.ndarray, base: RollingWavelet, source: SourceData,
                        wavelet: WaveletConfig) -> pl.DataFrame:
    """The representative window with the comparison wavelets, against the default."""
    rows = []
    names = ("wavelet_entropy", "wavelet_log_dominant_period", "wavelet_fast_slow_log_ratio",
             "wavelet_top3_scale_share")
    sample = np.flatnonzero(base.valid)
    if sample.size > 200_000:
        sample = sample[np.linspace(0, sample.size - 1, 200_000).round().astype(np.int64)]
    ref = ic_feature_columns(base, names, sample)
    for family in (base.layout.wavelet, *wavelet.dwt.compare_wavelets):
        other = base if family == base.layout.wavelet else rolling_wavelet(
            values, window=base.window, config=wavelet, bar_seconds=source.bar_seconds,
            missing_slots=source.missing_slots, wavelet=family)
        got = ic_feature_columns(other, names, sample)
        for metric in names:
            a, b = ref[metric], got[metric]
            ok = np.isfinite(a) & np.isfinite(b)
            rows.append({"wavelet": family, "metric": metric, "levels": other.layout.levels,
                         "median": float(np.median(b[np.isfinite(b)])) if np.isfinite(b).any()
                         else None,
                         "spearman_with_default": (float(pl.DataFrame({"a": a[ok], "b": b[ok]})
                                                         .select(pl.corr("a", "b",
                                                                         method="spearman"))
                                                         .item()) if ok.sum() > 10 else None)})
    return pl.DataFrame(rows, infer_schema_length=None)


def _offline(series: str, slices: list[dict[str, Any]], sources: dict[str, SourceData],
             wavelet: WaveletConfig, *, timeframe: str, make_plots: bool,
             lineage: dict[str, Any]) -> pl.DataFrame:
    """Ridge statistics and scalograms of the slices (NON-CAUSAL)."""
    directory = ensure_dir(wavelet.results_path / timeframe / series)
    ridges = offline_ridges(sources, series, slices, wavelet)
    if not ridges.is_empty():
        ridges = ridges.with_columns(pl.lit(timeframe).alias("timeframe"),
                                     pl.lit(series).alias("input_series"),
                                     pl.lit(lineage.get("tick_dataset_version"))
                                     .alias("dataset_version"))
        ridges.write_csv(directory / "offline_ridges.csv")
        ridges.write_parquet(directory / "offline_ridges.parquet")
    if make_plots and wavelet.plots.enabled and "real" in sources:
        real = sources["real"]
        for spec in slices:
            values = real.columns[series][spec["start"]:spec["end"] + 1]
            if not np.isfinite(values).all():
                continue
            scal = offline_scalogram(values, wavelet, bar_seconds=real.bar_seconds)
            times = real.timestamps.slice(spec["start"], spec["end"] - spec["start"] + 1
                                          ).to_numpy()
            plots.plot_scalogram(scal, times, path=directory / "plots" /
                                 f"scalogram_{spec['slice']}.png", config=wavelet,
                                 title=f"{timeframe} {series} - {spec['slice']}")
    return ridges


def _write_study(by_source: dict[str, dict[str, Any]], wavelet: WaveletConfig,
                 gate: dict[str, Any], *, timeframe: str, series: str, window: int,
                 features: dict[str, Any] | None, seconds: float, timings: dict[str, float],
                 make_plots: bool, provenance: dict[str, Any],
                 sensitivity: pl.DataFrame | None) -> WaveletStudy:
    study = WaveletStudy(timeframe=timeframe, series=series, window=window,
                         output_dir=wavelet.results_dir(timeframe, series, window))
    writer = StudyWriter(study.output_dir, wavelet.output, study.files)
    sources = list(by_source)

    def stack(key: str) -> pl.DataFrame:
        return stack_tables(by_source[s].get(key) for s in sources)

    for key, name in (("distribution", "metric_distribution"), ("energy", "energy_summary"),
                      ("persistence", "scale_persistence"), ("drift", "drift_summary"),
                      ("bursts", "burst_summary"), ("yearly", "yearly_stability"),
                      ("quarterly", "quarterly_stability"), ("era", "era_stability"),
                      ("intraday", "intraday"), ("extremes", "extreme_conditioning"),
                      ("ic", "ic_analysis"), ("fft_pairs", "fft_comparison"),
                      ("fft_determinism", "fft_determinism"), ("dyadic", "dyadic_comparison")):
        writer.table(stack(key), name)
    writer.table(stack("ic_periods"), "ic_by_period", csv=False)
    distribution = stack("distribution")
    writer.table(distribution.filter(pl.col("metric").is_in(
        ["wavelet_entropy", "wavelet_local_entropy", "wavelet_top3_scale_share",
         "wavelet_dominant_energy_share"])) if not distribution.is_empty() else None,
        "entropy_summary")
    runs = [by_source[s]["runs"] for s in sources if by_source[s].get("runs")]
    writer.table(pl.DataFrame(runs, infer_schema_length=None) if runs else None, "run_summary")
    conditioning = stack("conditioning")
    if not conditioning.is_empty():
        for variable, name in (("volatility", "volatility_conditioning"),
                               ("trend", "trend_conditioning"), ("ou_speed", "ou_conditioning")):
            writer.table(conditioning.filter(pl.col("variable") == variable), name)
    linear = stack_tables(by_source[s].get("incremental", {}).get("linear") for s in sources)
    by_feature = stack_tables(by_source[s].get("incremental", {}).get("by_feature")
                              for s in sources)
    binary = stack_tables(by_source[s].get("incremental", {}).get("binary") for s in sources)
    writer.table(stack_tables([linear, binary]), "incremental_information")
    writer.table(by_feature, "incremental_by_feature")
    summary_inc = incremental_summary(linear, binary)
    writer.table(summary_inc, "incremental_summary")
    real = by_source.get("real", {})
    writer.table(real.get("correlation"), "feature_correlation")
    writer.table(real.get("redundancy"), "feature_redundancy")
    comparison = null_comparison(by_source)
    writer.table(comparison, "null_controls")
    if sensitivity is not None:
        writer.table(sensitivity, "family_sensitivity")
    quality = _feature_quality(by_source, by_feature, wavelet, window)
    writer.table(quality, "feature_quality")

    tests = {s: int(p.get("ic_tests", 0)) for s, p in by_source.items()}
    lineage = gate["lineage"]
    study_key = f"{timeframe}/{series}/window_{window}"
    if wavelet.ledger.enabled:
        entries = _ledger_entries(by_source, comparison, timeframe=timeframe, series=series,
                                  window=window,
                                  dataset_version=lineage.get("tick_dataset_version"),
                                  study=study_key, summary=summary_inc)
        ResearchLedger(wavelet.ledger_path).upsert(entries)
        writer.table(pl.DataFrame([{k: (json.dumps(v, default=str) if isinstance(v, dict)
                                        else v) for k, v in e.items()} for e in entries],
                                  infer_schema_length=None) if entries else None,
                     "hypothesis_verdicts")

    counts = real.get("counts", {})

    def dist(metric: str, q: str = "p50") -> float | None:
        return _pick(real.get("distribution"), {"metric": metric}, q)

    ic = real.get("ic")
    best_ic = best_rank = None
    if isinstance(ic, pl.DataFrame) and not ic.is_empty():
        best_ic = num(ic["ic"].abs().max())
        best_rank = num(ic["rank_ic"].abs().max())
    fs_log = dist("wavelet_fast_slow_log_ratio")
    real_inc = (summary_inc.filter(pl.col("source") == "real")
                if not summary_inc.is_empty() else pl.DataFrame())
    redundancy = real.get("redundancy")
    summary: dict[str, Any] = {
        "timeframe": timeframe, "source": series, "input_series": series,
        "rolling_window": window, "wavelet": wavelet.dwt.wavelet,
        "method": wavelet.causal_features.method,
        "observations": counts.get("windows", 0) + window - 1,
        "valid_feature_fraction": (counts.get("valid", 0) / counts["windows"]
                                   if counts.get("windows") else None),
        "median_wavelet_entropy": dist("wavelet_entropy"),
        "median_dominant_period_bars": dist("wavelet_dominant_period_bars"),
        "median_dominant_period_seconds": dist("wavelet_dominant_period_seconds"),
        "median_fast_slow_energy_ratio": float(np.exp(fs_log)) if fs_log is not None else None,
        "median_top3_scale_share": dist("wavelet_top3_scale_share"),
        "dominant_scale_persistence": _pick(comparison, {"metric": "same_band_lag_N"}, "real"),
        "null_control_difference": {
            r["metric"]: {k: v for k, v in r.items() if k != "metric"}
            for r in comparison.filter(pl.col("metric").is_in(
                ["median_wavelet_entropy", "median_wavelet_top3_scale_share",
                 "median_wavelet_fast_slow_log_ratio", "same_band_lag_N", "burst_fraction",
                 "max_abs_rank_ic"])).iter_rows(named=True)} if not comparison.is_empty()
        else {},
        "best_absolute_ic": best_ic,
        "best_rank_ic": best_rank,
        "incremental_information_over_fft": {
            f"{r['target']}_h{r['horizon']}_{r['metric']}": {
                "delta_D_C": r.get("delta_D_C"), "folds_d_beats_c": r.get("folds_d_beats_c")}
            for r in real_inc.iter_rows(named=True)} if not real_inc.is_empty() else {},
        "feature_redundancy": ({
            "median_max_abs_spearman_fft": num(redundancy["max_abs_spearman_fft"]
                                               .drop_nulls().median()),
            "median_max_abs_spearman_existing": num(redundancy["max_abs_spearman_existing"]
                                                    .drop_nulls().median())}
            if isinstance(redundancy, pl.DataFrame) and not redundancy.is_empty() else {}),
        "bands": real.get("layout"),
        "ic_tests": tests,
        "features_written": features,
        "live_safe": True,
        "partial_dataset": bool(lineage.get("partial")),
        "caveat": CAVEAT,
    }
    study.summary = clean_json(summary)
    study.duration_seconds = seconds
    payload: dict[str, Any] = {
        "summary": study.summary,
        "provenance": {
            "report_version": WAVELET_REPORT_VERSION,
            "generated_utc": utc_now_iso(), **provenance,
            "dataset": lineage, "integrity_gate": {k: v for k, v in gate.items()
                                                   if k != "lineage"},
            "regression_feature_version": gate.get("regression_feature_version"),
            "wavelet_config_fingerprint": wavelet.fingerprint(),
            "wavelet_engine_fingerprint": wavelet.engine_fingerprint(),
            "roles": {s: by_source[s]["role"] for s in sources},
            "controls": {s: by_source[s].get("counts") for s in sources},
            "source_seconds": timings,
            "sampling": {"ic_rows_real": wavelet.ic.max_rows,
                         "ic_rows_control": wavelet.ic.max_rows_control,
                         "incremental_rows": wavelet.incremental.max_rows,
                         "note": "IC, incremental models, redundancy and persistence figures "
                                 "are estimates from evenly spaced samples"},
        },
        "hypotheses": HYPOTHESES,
        "warnings": study.warnings,
    }
    if lineage.get("label"):
        payload["provenance"]["label"] = lineage["label"]
    writer.json(payload, "summary")
    if make_plots and wavelet.plots.enabled:
        try:
            _plots(by_source, wavelet, study, stack("energy"), stack("persistence"),
                   stack("ic"))
        except Exception as exc:  # a figure must never lose the tables
            LOGGER.warning("plots for %s failed: %s", study_key, exc)
            study.warnings.append(f"plots failed: {exc}")
    return study


#: IC features that are logs of a stored causal feature.
_STORED_NAME = {
    "wavelet_log_dominant_period": "wavelet_dominant_period_bars",
    "wavelet_log_centroid_period": "wavelet_centroid_period_bars",
    "wavelet_log_run_length": "wavelet_dominant_run_length",
}


def _feature_quality(by_source: dict[str, dict[str, Any]], by_feature: pl.DataFrame,
                     wavelet: WaveletConfig, window: int) -> pl.DataFrame:
    """Step 48: one row per causal feature - definition, coverage, stability, nulls, IC,
    increment over FFT, redundancy, cost and live safety."""
    real = by_source.get("real", {})
    ic = real.get("ic")
    periods = real.get("ic_periods")
    red = real.get("redundancy")
    det = real.get("fft_determinism")
    names = wavelet.ic.features
    coverage = real.get("coverage") or {}
    rows = []
    for name in names:
        row: dict[str, Any] = {"feature": name, "live_safe": True, "causal": True}
        spec = CAUSAL_FEATURES.get(_STORED_NAME.get(name, name))
        row["definition"] = (("ln " if name in _STORED_NAME else "") + spec.description
                             if spec else name)
        row["coverage_of_valid_windows"] = coverage.get(name)
        row["coverage_of_bars"] = real.get("coverage_of_bars")
        if isinstance(ic, pl.DataFrame) and not ic.is_empty():
            part = ic.filter(pl.col("feature") == name)
            if part.height:
                a = part["rank_ic"].abs()
                row["max_abs_rank_ic"] = num(a.max())
                row["median_abs_rank_ic"] = num(a.median())
                row["max_abs_ic"] = num(part["ic"].abs().max())
                stable = part.filter((pl.col("yearly_rank_ic_positive_share") >= 0.75)
                                     | (pl.col("yearly_rank_ic_positive_share") <= 0.25))
                row["tests_with_stable_yearly_sign"] = stable.height
                row["ic_tests"] = int(part["rank_ic"].is_finite().sum())
        if isinstance(periods, pl.DataFrame) and not periods.is_empty():
            yearly = periods.filter((pl.col("feature") == name) & (pl.col("grouping") == "year"))
            if yearly.height:
                row["yearly_rank_ic_std_median"] = num(
                    yearly.group_by("target", "horizon").agg(pl.col("rank_ic").std())[
                        "rank_ic"].median())
        if isinstance(red, pl.DataFrame) and not red.is_empty():
            r = red.filter(pl.col("feature") == name)
            if r.height:
                row["max_abs_spearman_fft"] = r["max_abs_spearman_fft"][0]
                row["max_abs_spearman_existing"] = r["max_abs_spearman_existing"][0]
                row["redundancy_cluster"] = r["cluster"][0]
        if isinstance(det, pl.DataFrame) and not det.is_empty():
            r = det.filter(pl.col("wavelet_feature") == name)
            if r.height:
                row["oos_r2_from_fft"] = r["oos_r2_from_fft"][0]
        row["lookback_bars"] = (2 * window if name in ("wavelet_energy_log_change",
                                                       "wavelet_log_run_length")
                                else window + max(1, int(round(window * wavelet.causal_features
                                                               .drift_lag_fraction)))
                                if name in ("wavelet_scale_drift", "wavelet_fast_slow_change")
                                else window)
        rows.append(row)
    table = pl.DataFrame(rows, infer_schema_length=None)
    if isinstance(by_feature, pl.DataFrame) and not by_feature.is_empty():
        gains = _single_feature_gains(by_source)
        if not gains.is_empty():
            table = table.join(gains, on="feature", how="left")
    return table


def _single_feature_gains(by_source: dict[str, dict[str, Any]]) -> pl.DataFrame:
    """Mean out-of-sample R^2 gain of C + one feature over C, best target/horizon, real."""
    inc = by_source.get("real", {}).get("incremental") or {}
    lin, feat = inc.get("linear"), inc.get("by_feature")
    if not isinstance(lin, pl.DataFrame) or lin.is_empty() or not isinstance(
            feat, pl.DataFrame) or feat.is_empty():
        return pl.DataFrame()
    base = lin.filter(pl.col("model") == "C").select("target", "horizon", "fold",
                                                     pl.col("oos_r2").alias("c"))
    gains = (feat.join(base, on=["target", "horizon", "fold"])
             .with_columns((pl.col("oos_r2") - pl.col("c")).alias("gain"),
                           pl.col("model").str.strip_prefix("C+").alias("feature"))
             .group_by("feature", "target", "horizon")
             .agg(pl.col("gain").mean().alias("mean_gain"),
                  (pl.col("gain") > 0).sum().alias("folds_positive")))
    return (gains.sort("mean_gain", descending=True).group_by("feature", maintain_order=True)
            .agg(pl.col("mean_gain").first().alias("best_mean_oos_r2_gain_over_C"),
                 pl.col("target").first().alias("best_gain_target"),
                 pl.col("horizon").first().alias("best_gain_horizon"),
                 pl.col("folds_positive").first().alias("best_gain_folds_positive")))


def _plots(by_source: dict[str, dict[str, Any]], wavelet: WaveletConfig, study: WaveletStudy,
           energy: pl.DataFrame, persistence: pl.DataFrame, ic: pl.DataFrame) -> None:
    directory = wavelet.plots_dir(study.timeframe, study.series, study.window)
    tag = f"{study.timeframe} {study.series}, window {study.window}"
    real = by_source.get("real", {})
    if not energy.is_empty():
        plots.plot_energy_by_scale(energy, path=directory / "energy_by_scale.png",
                                   config=wavelet, title=f"Wavelet energy by scale - {tag}")
    series = real.get("plot_series")
    if series is not None:
        times = series["timestamps"]
        plots.plot_rolling_state(times, {"real": series["dominant_period"]},
                                 path=directory / "rolling_dominant_scale.png", config=wavelet,
                                 title=f"Rolling dominant scale (causal) - {tag}",
                                 ylabel="dominant band period (bars, log scale)", log=True)
        plots.plot_rolling_state(times, {"real": series["entropy"]},
                                 path=directory / "rolling_entropy.png", config=wavelet,
                                 title=f"Rolling wavelet entropy (causal) - {tag}",
                                 ylabel="normalised wavelet entropy")
        plots.plot_rolling_state(times, {"real": series["fast_slow"]},
                                 path=directory / "fast_slow_ratio.png", config=wavelet,
                                 title=f"Fast/slow energy ratio (causal) - {tag}",
                                 ylabel="ln(fast energy / slow energy)")
    drifts = {s: p["drift_values"] for s, p in by_source.items()
              if isinstance(p.get("drift_values"), np.ndarray)}
    if drifts:
        plots.plot_scale_drift(drifts, path=directory / "scale_drift.png", config=wavelet,
                               title=f"Frequency (centroid) drift over N/4 bars - {tag}")
    if not persistence.is_empty():
        plots.plot_scale_persistence(persistence, path=directory / "scale_persistence.png",
                                     config=wavelet, title=f"Dominant-scale persistence - {tag}")
    hist = {s: p["histograms"] for s, p in by_source.items() if p.get("histograms")}
    if hist:
        plots.plot_null_distributions(hist, path=directory / "null_comparison.png",
                                      config=wavelet, title=f"Real vs null controls - {tag}")
    pairs = real.get("period_pairs")
    if pairs is not None:
        plots.plot_fft_vs_wavelet_period(pairs[0], pairs[1],
                                         path=directory / "fft_vs_wavelet_period.png",
                                         config=wavelet,
                                         title=f"FFT vs wavelet dominant period - {tag}")
    if not ic.is_empty():
        plots.plot_ic_by_horizon(ic, path=directory / "ic_by_horizon.png", config=wavelet,
                                 title=f"Wavelet feature |rank IC| by horizon - {tag}")


# ---------------------------------------------------------------------------
# Resume, comparison
# ---------------------------------------------------------------------------
def _read_summary(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) and "summary" in payload else None


def reusable_wavelet_study(config: Config, wavelet: WaveletConfig, *, timeframe: str,
                           series: str, window: int) -> WaveletStudy | None:
    """The study on disk if it is current (versions, settings, report version)."""
    path = wavelet.results_dir(timeframe, series, window) / "summary.json"
    payload = _read_summary(path)
    if payload is None:
        return None
    recorded = payload.get("provenance") or {}
    if recorded.get("report_version") != WAVELET_REPORT_VERSION:
        return None
    lineage = dataset_lineage(config, timeframe)
    dataset = recorded.get("dataset") or {}
    for key in ("raw_fingerprint", "tick_dataset_version", "bar_dataset_version"):
        if lineage.get(key) is None or dataset.get(key) != lineage.get(key):
            return None
    if recorded.get("wavelet_config_fingerprint") != wavelet.fingerprint():
        return None
    return WaveletStudy(timeframe=timeframe, series=series, window=window,
                        output_dir=path.parent, summary=payload["summary"],
                        warnings=list(payload.get("warnings") or []), reused=True)


def load_wavelet_studies(wavelet: WaveletConfig) -> list[WaveletStudy]:
    studies = []
    for path in sorted(wavelet.results_path.glob("*/*/window_*/summary.json")):
        payload = _read_summary(path)
        if payload is None:
            continue
        s = payload["summary"]
        studies.append(WaveletStudy(timeframe=s["timeframe"], series=s["input_series"],
                                    window=int(s["rolling_window"]), output_dir=path.parent,
                                    summary=s))
    return studies


def _cross_timeframe(studies: list[WaveletStudy]) -> pl.DataFrame:
    """Band shares at their physical periods across timeframes: real, random walk, bootstrap."""
    parts = []
    for study in studies:
        path = study.output_dir / "energy_summary.parquet"
        if not path.exists():
            continue
        energy = pl.read_parquet(path)
        wide = energy.filter(pl.col("source").is_in(["real", "random_walk", "block_bootstrap"])
                             ).pivot(on="source", index=["band", "period_seconds", "period_bars"],
                                     values="median_share")
        if "real" not in wide.columns or "random_walk" not in wide.columns:
            continue
        parts.append(wide.with_columns(
            pl.lit(study.timeframe).alias("timeframe"), pl.lit(study.series).alias("input_series"),
            pl.lit(study.window).alias("window"),
            (pl.col("real") - pl.col("random_walk")).alias("excess_over_random_walk")))
    return stack_tables(parts)


def write_wavelet_comparison(wavelet: WaveletConfig, studies: list[WaveletStudy]) -> list[Path]:
    """Cross-study tables: every summary, windows, physical-time bands, test counts, chance."""
    summaries = [s.summary for s in studies if s.summary]
    if not summaries:
        return []
    directory = ensure_dir(wavelet.results_path)
    flat = []
    for s in summaries:
        row = {k: v for k, v in s.items() if not isinstance(v, (dict, list)) and k != "caveat"}
        for metric, values in (s.get("null_control_difference") or {}).items():
            for key, value in values.items():
                if not isinstance(value, (dict, list)):
                    row[f"{metric}__{key}"] = value
        flat.append(row)
    table = pl.DataFrame(flat, infer_schema_length=None).sort("timeframe", "input_series",
                                                              "rolling_window")
    written = []
    table.write_csv(directory / "wavelet_comparison.csv")
    table.write_parquet(directory / "wavelet_comparison.parquet")
    written += [directory / "wavelet_comparison.csv", directory / "wavelet_comparison.parquet"]
    keep = ["timeframe", "input_series", "rolling_window", "valid_feature_fraction",
            "median_wavelet_entropy", "median_dominant_period_seconds",
            "median_fast_slow_energy_ratio", "dominant_scale_persistence", "best_rank_ic"]
    table.select([c for c in keep if c in table.columns]).write_csv(
        directory / "window_comparison.csv")
    written.append(directory / "window_comparison.csv")
    bands = _cross_timeframe(studies)
    if not bands.is_empty():
        bands.write_csv(directory / "cross_timeframe_bands.csv")
        written.append(directory / "cross_timeframe_bands.csv")
        if wavelet.plots.enabled:
            for (name, window), part in bands.group_by("input_series", "window"):
                plots.plot_cross_timeframe(
                    part, path=directory / "plots" / f"cross_timeframe_{name}_window_{window}.png",
                    config=wavelet, title=f"Band shares in physical time, {name}, window "
                                          f"{window}: XAUUSD minus random walk")
    chance = ic_chance_rates(studies)
    counts = hypothesis_test_counts(ResearchLedger(wavelet.ledger_path).load(), prefix="WAVE-")
    for name, frame in (("ic_chance_rates", chance), ("hypothesis_test_counts", counts)):
        if not frame.is_empty():
            frame.write_csv(directory / f"{name}.csv")
            written.append(directory / f"{name}.csv")
    atomic_write_text(directory / "wavelet_comparison.json",
                      json.dumps(clean_json(summaries), indent=1, default=str) + "\n")
    written.append(directory / "wavelet_comparison.json")
    return written
