"""Orchestration of the regime-discovery study (Prompt #7).

One timeframe at a time, in resumable stages that write as they go:

``inputs``        the integrity gate, the cached causal feature table, the feature
                  manifest and redundancy (``<tf>/feature_manifest.json``)
``offline``       full-sample fits of every model and K, the chronological hold-out,
                  the baselines and the scaling comparison (``<tf>/offline/``) -
                  OFFLINE / NON-CAUSAL RESEARCH ONLY
``walk_forward``  causal refits per (model, K, scheme, refit frequency); the stored
                  live-safe features (``data/features/regime``) and the refit logs
                  (``<tf>/<model>/k<K>/walk_forward/<scheme>_<refit>/``)
``nulls``         HMM persistence on row-shuffled, row-block-bootstrapped and
                  pipeline-null features (``<tf>/nulls/``)
``analysis``      per (model, K) the Step 68 tables, plots, ``summary.json`` and
                  the hypothesis verdicts, from the causal states

``write_regime_comparison`` then builds the cross-timeframe tables: model
selection, the defensible number of states, cross-timeframe agreement, change
points and the multiple-testing record. Hypotheses are registered before
any result exists (``hypotheses_registered.json``); every verdict goes to the
shared research ledger, failures included.
"""

from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..features.config import RegressionConfig
from ..features.spectral_config import SpectralConfig
from ..features.wavelet_config import WaveletConfig
from ..models.config import OUConfig
from ..regimes.causal_inference import (
    WalkForwardResult,
    refit_schedule,
    run_walk_forward,
    write_regime_features,
)
from ..regimes.changepoints import pelt_mean, segments_table
from ..regimes.config import OFFLINE_LABEL, RegimeConfig
from ..regimes.dataset import (
    bar_seconds,
    feature_manifest,
    feature_matrix,
    integrity_gate,
    load_or_build_regime_inputs,
    null_regime_inputs,
    redundancy_table,
    source_from_frame,
)
from ..regimes.diagnostics import contingency, eta_squared, feature_contribution
from ..regimes.hmm import HMMModel
from ..regimes.registry import ModelRegistry
from ..regimes.transitions import duration_summary, run_lengths, transition_counts
from ..research.spectral_nulls import build_real_source
from ..utils.clock import utc_now_iso
from ..utils.config import Config
from ..utils.logging import get_logger
from ..utils.paths import atomic_write_text, ensure_dir
from . import regime_plots as plots
from .config import ResearchConfig
from .regime_analysis import (
    agreement,
    baseline_holdout,
    bucket_labels,
    describe_states,
    frequency_table,
    holdout_scores,
    intraday_frequency,
    offline_fit,
    offline_model_row,
    random_labels,
    scaling_comparison,
    standardized_means,
    state_profiles,
)
from .regime_nulls import block_bootstrap_rows, null_persistence, shuffle_rows, synthetic_controls
from .regime_outcomes import (
    calibration_tables,
    conditional_outcomes,
    entropy_analysis,
    future_volatility_change,
    incremental_information,
    incremental_summary,
    mean_reversion_by_state,
    mean_reversion_increment,
    ou_by_state,
    state_medians,
)
from .regime_stability import (
    defensible_states,
    era_definitions,
    finite_or_none,
    k_rule_verdict,
    refit_stability_summary,
    scheme_comparison,
)
from .research_ledger import ResearchLedger
from .spectral_reports import hypothesis_test_counts
from .study_io import StudyWriter, clean_json, code_fingerprint, git_info, package_versions

__all__ = [
    "HYPOTHESES",
    "REGIME_REPORT_VERSION",
    "RegimeContext",
    "RegimeStudy",
    "analyse_study",
    "benchmark_regimes",
    "estimate_seconds",
    "generate_regime_timeframe",
    "load_regime_studies",
    "null_increment_study",
    "null_inputs",
    "prepare_timeframe",
    "register_hypotheses",
    "run_nulls",
    "run_offline",
    "run_synthetic_controls",
    "walk_forward_stage",
    "write_regime_comparison",
]

LOGGER = get_logger("research.regime_reports")

#: Version of what a study writes; --resume reuses only artefacts of this version.
REGIME_REPORT_VERSION = 1

CAVEAT = (
    "Descriptive research only. States are numbered latent classes of a Gaussian model on "
    "trailing-window features; persistence can come from the windows themselves, and "
    "separation from volatility and the time of day, so every result is read against the "
    "single-state model, volatility buckets, random labels and null controls. Only walk-"
    "forward filtered probabilities (past data only, frozen scaler and model) are causal; "
    "full-sample fits, smoothed probabilities and Viterbi paths are OFFLINE / NON-CAUSAL. "
    "No signal, entry, exit, sizing or cost exists in this layer, and no K was chosen by "
    "performance."
)

HYPOTHESES: dict[str, str] = {
    "REG-H-001": "Regimes explain variation in residual mean reversion beyond volatility "
                 "quartiles: for P(|eps_t+h| < |eps_t|) after |Z| > 2, adding the filtered "
                 "state probabilities to a logistic model with |Z|, Z and volatility-quartile "
                 "indicators lowers out-of-sample log-loss in >= 3 of 4 chronological folds, "
                 "by more than adding random states does, and by more than the same increment "
                 "on the pipeline nulls (random walk, 1024-bar block bootstrap), each with its "
                 "own walk-forward model.",
    "REG-H-002": "HMM filtered states are more stable out of sample than static GMM "
                 "assignments (same timeframe and K): a lower bar-to-bar switching rate AND a "
                 "higher median walk-forward refit overlap (ARI).",
    "REG-H-003": "The state with the highest wavelet fast/slow energy ratio, if it is also the "
                 "state with the widest spread (percentile), has distinct future volatility: "
                 "its mean 20-bar change in ln realised volatility, within volatility deciles, "
                 "differs from the other states' by more than the 95th percentile of the same "
                 "statistic for random states with matched frequency and persistence.",
    "REG-H-004": "Discrete states are defensible: by the registered K rule (chronological "
                 "hold-out likelihood gain > 0.005 nats/bar at every step from K = 1, no "
                 "degenerate flag in the full-sample fit, median walk-forward refit overlap "
                 "ARI >= 0.5) the defensible K is at least 2.",
    "REG-H-005": "HMM states persist beyond mechanical smoothing: the median expected state "
                 "duration on the real features exceeds that on every pipeline null (random "
                 "walk, shuffled returns, 1024-bar block bootstrap) by more than 25 %, same K "
                 "and span.",
    "REG-H-006": "Regimes separate realised residual decay beyond volatility: the range across "
                 "states of the median |eps_t+10| / |eps_t| after |Z| > 2 exceeds the range "
                 "across same-K volatility buckets and the 95th percentile for random states.",
    "REG-H-007": "Filtered state probabilities add out-of-sample information beyond every "
                 "continuous feature (model A: volatility over 5-1024 bars, hour of day, "
                 "regression, OU, FFT, wavelet, spread, activity): A+soft beats A in >= 3 of 4 "
                 "chronological folds, by more than A+random states, and by more than A+soft "
                 "does on the pipeline nulls (random walk, 1024-bar block bootstrap), each with "
                 "its own walk-forward model.",
    "REG-H-008": "Soft state probabilities carry more information than hard labels: A+soft "
                 "beats A+hard in >= 3 of 4 chronological folds.",
    "REG-H-009": "The HMM's one-step leave probability is calibrated: logistic calibration "
                 "slope in [0.8, 1.25] and the largest reliability-bin gap below 0.05.",
    "REG-H-010": "Regimes at different timeframes agree beyond chance at equivalent "
                 "timestamps (HMM, same K): Cramer's V > 0.3.",
    "REG-H-011": "States are more than volatility buckets: normalised mutual information "
                 "between the state and same-K volatility buckets is below 0.5 AND the "
                 "non-volatility inputs' mean eta^2 is higher under the states than under the "
                 "buckets.",
    "REG-H-012": "Expanding-window training gives a higher out-of-sample likelihood per bar "
                 "than 5-year rolling training.",
    "REG-H-013": "State definitions are stable over 2008-2026: after alignment, no state's "
                 "Bhattacharyya distance from its first-refit definition exceeds the smallest "
                 "distance between two states of that first model.",
}

_MARGIN_K = 0.005
_MIN_OVERLAP_ARI = 0.5
_RANDOM_DRAWS = 20
_SOURCES = ("regimes/config.py", "regimes/dataset.py", "regimes/preprocessing.py",
            "regimes/emissions.py", "regimes/clustering.py", "regimes/gmm.py",
            "regimes/hmm.py", "regimes/state_alignment.py", "regimes/transitions.py",
            "regimes/diagnostics.py", "regimes/fitting.py", "regimes/causal_inference.py",
            "regimes/registry.py", "research/regime_analysis.py",
            "research/regime_outcomes.py", "research/regime_stability.py",
            "research/regime_nulls.py", "research/regime_reports.py")
_PACKAGES = ("numpy", "scipy", "polars", "pyarrow", "PyWavelets")
_NON_VOLATILITY = ("return_z_20", "slope_over_volatility", "r_squared", "log_ou_half_life",
                   "spectral_entropy", "fft_high_low_log_ratio", "wavelet_entropy",
                   "wavelet_fast_slow_log_ratio", "spread_percentile", "activity_percentile")
_ERAS = ((2008, 2012), (2013, 2017), (2018, 2022), (2023, 2026))


def code_version() -> str:
    base = Path(__file__).resolve().parent.parent
    return code_fingerprint(base / rel for rel in _SOURCES)


#: The code each cached stage's values depend on (reports and plots excluded, so an
#: edit to the analysis or the write-up never invalidates hours of fitting).
_STAGE_SOURCES: dict[str, tuple[str, ...]] = {
    "engine": ("regimes/dataset.py", "regimes/preprocessing.py", "regimes/emissions.py",
               "regimes/clustering.py", "regimes/gmm.py", "regimes/hmm.py",
               "regimes/state_alignment.py", "regimes/transitions.py",
               "regimes/diagnostics.py", "regimes/fitting.py", "regimes/causal_inference.py",
               "regimes/registry.py"),
    "offline": ("research/regime_analysis.py",),
    "nulls": ("research/regime_nulls.py",),
    "null_increment": ("research/regime_outcomes.py", "research/regime_analysis.py"),
}


def stage_code_version(stage: str) -> str:
    base = Path(__file__).resolve().parent.parent
    extra = _STAGE_SOURCES.get(stage, ())
    return code_fingerprint(base / rel for rel in (*_STAGE_SOURCES["engine"], *extra))


def register_hypotheses(regime: RegimeConfig) -> Path:
    """Write the hypothesis definitions before any result (kept once registered)."""
    path = regime.results_path / "hypotheses_registered.json"
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
                     "k_rule": {"holdout_margin_nats_per_bar": _MARGIN_K,
                                "min_refit_overlap_ari": _MIN_OVERLAP_ARI},
                     "random_label_draws": _RANDOM_DRAWS,
                     "note": "Verdict rules are fixed in regime_reports; nothing is chosen "
                             "by trading performance."})
    atomic_write_text(path, json.dumps(recorded, indent=2) + "\n")
    return path


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------
@dataclass
class RegimeContext:
    """Everything one timeframe's stages share."""

    config: Config
    regression: RegressionConfig
    ou: OUConfig
    spectral: SpectralConfig
    wavelet: WaveletConfig
    research: ResearchConfig
    regime: RegimeConfig
    timeframe: str
    gate: dict[str, Any]
    inputs: pl.DataFrame
    inputs_manifest: dict[str, Any]
    provenance: dict[str, Any]
    progress: Callable[[str], None] | None = None

    @property
    def features(self) -> tuple[str, ...]:
        return self.regime.features()

    @property
    def lineage(self) -> dict[str, Any]:
        return self.gate["lineage"]

    def log(self, message: str) -> None:
        LOGGER.info("[%s] %s", self.timeframe, message)
        if self.progress is not None:
            self.progress(message)

    def directory(self, *parts: str) -> Path:
        return ensure_dir(self.regime.timeframe_dir(self.timeframe).joinpath(*parts))


def prepare_timeframe(config: Config, regression: RegressionConfig, ou: OUConfig,
                      spectral: SpectralConfig, wavelet: WaveletConfig, research: ResearchConfig,
                      regime: RegimeConfig, timeframe: str, *,
                      progress: Callable[[str], None] | None = None) -> RegimeContext:
    """Gate, inputs, feature manifest; refuses to continue on any version mismatch."""
    register_hypotheses(regime)
    gate = integrity_gate(config, regression, ou, spectral, wavelet, regime, timeframe)
    if not gate["passed"]:
        raise RuntimeError(f"{timeframe}: dataset integrity gate failed - "
                           + "; ".join(gate["problems"]))
    inputs, manifest = load_or_build_regime_inputs(config, regression, ou, spectral, wavelet,
                                                   regime, timeframe, gate=gate)
    lineage = gate["lineage"]
    provenance = {
        "dataset_lineage": lineage,
        "tick_dataset_version": lineage.get("tick_dataset_version"),
        "bar_dataset_version": lineage.get("bar_dataset_version"),
        "feature_versions": {k: v for k, v in gate["versions"].items() if k != "recompute_check"},
        "regime_config_fingerprint": regime.fingerprint(),
        "regime_inputs_fingerprint": regime.inputs_fingerprint(),
        "code_fingerprint": code_version(),
        "git": git_info(config.project_root),
        "regime_model_version": (f"regime-{regime.fingerprint()}-{regime.inputs_fingerprint()}"
                                 f"-{code_version()}"),
    }
    ctx = RegimeContext(config, regression, ou, spectral, wavelet, research, regime, timeframe,
                        gate, inputs, manifest, provenance, progress)
    directory = ctx.directory()
    features = regime.features()
    redundancy = redundancy_table(inputs, features, sample_rows=regime.redundancy.sample_rows)
    manifest_out = feature_manifest(regime, regime.primary_feature_set, redundancy)
    coverage = {name: float(np.isfinite(inputs[name].to_numpy()).mean()) for name in features}
    manifest_out.update({"timeframe": timeframe, "coverage": coverage,
                         "complete_rows_share": float(np.isfinite(
                             feature_matrix(inputs, features)).all(axis=1).mean()),
                         "integrity_gate": {k: v for k, v in gate.items() if k != "lineage"},
                         "provenance": provenance, "generated_utc": utc_now_iso()})
    atomic_write_text(directory / "feature_manifest.json",
                      json.dumps(clean_json(manifest_out), indent=2, default=str) + "\n")
    redundancy.write_csv(directory / "feature_redundancy.csv")
    if not manifest_out["passes_redundancy_rule"]:
        LOGGER.warning("[%s] feature set holds a near-duplicate pair: %s", timeframe,
                       manifest_out["duplicate_pairs"])
    return ctx


def _stamp(ctx: RegimeContext, **extra: Any) -> dict[str, Any]:
    """What a stored stage artefact depends on (compared on --resume): data versions, the
    settings, and the code of the modules that compute that stage."""
    stage = str(extra.get("stage", ""))
    code_stage = {"offline": "offline", "scaling": "offline", "nulls": "nulls",
                  "null_inputs": "engine", "null_increment": "null_increment"}.get(stage,
                                                                                  "engine")
    return clean_json({"report_version": REGIME_REPORT_VERSION,
                       "tick_dataset_version": ctx.provenance["tick_dataset_version"],
                       "bar_dataset_version": ctx.provenance["bar_dataset_version"],
                       "regime_config_fingerprint": ctx.regime.fingerprint(),
                       "code_fingerprint": stage_code_version(code_stage), **extra})


def _reusable(path: Path, stamp: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        recorded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return all(recorded.get("stamp", {}).get(k) == v for k, v in stamp.items())


def _write_table(frame: pl.DataFrame | None, path: Path) -> None:
    if frame is None or frame.is_empty():
        return
    nulls = [c for c, d in frame.schema.items() if d == pl.Null]
    if nulls:
        frame = frame.with_columns(pl.col(nulls).cast(pl.Float64))
    listy = [c for c, d in frame.schema.items() if isinstance(d, (pl.List, pl.Struct))]
    frame.write_parquet(path.with_suffix(".parquet"))
    if listy:
        frame = frame.with_columns([pl.col(c).map_elements(
            lambda v: json.dumps(clean_json(v.to_list() if hasattr(v, "to_list") else v),
                                 default=str), return_dtype=pl.Utf8) for c in listy])
    frame.write_csv(path.with_suffix(".csv"))


# ---------------------------------------------------------------------------
# Stage: offline (full sample, non-causal)
# ---------------------------------------------------------------------------
def run_offline(ctx: RegimeContext, *, models: list[str], states: list[int],
                resume: bool = True, make_plots: bool = True) -> pl.DataFrame:
    """Full-sample fits of every (model, K) plus hold-out, baselines and scaling (labelled)."""
    regime = ctx.regime
    directory = ctx.directory("offline")
    features = ctx.features
    x = feature_matrix(ctx.inputs, features)
    times = ctx.inputs["timestamp"].to_numpy()
    split = np.datetime64(_holdout_split(regime))
    rows: list[dict[str, Any]] = []
    for model in models:
        for k in states:
            out_dir = ensure_dir(regime.results_dir(ctx.timeframe, model, k) / "offline")
            stamp = _stamp(ctx, model=model, states=k, stage="offline")
            marker = out_dir / "summary_offline.json"
            if resume and _reusable(marker, stamp):
                rows.append(json.loads(marker.read_text(encoding="utf-8"))["row"])
                continue
            seed = regime.seed + 101 * k
            ctx.log(f"offline {model} K={k}: full-sample fit")
            fitted, seconds = offline_fit(x, features, regime, family=model, k=k, seed=seed)
            row, labels = offline_model_row(fitted, x, regime, timeframe=ctx.timeframe,
                                            seconds=seconds)
            ctx.log(f"offline {model} K={k}: hold-out")
            row.update(holdout_scores(x, times, features, regime, family=model, k=k,
                                      split=split, seed=seed))
            _offline_tables(ctx, fitted, labels, x, out_dir, make_plots=make_plots)
            atomic_write_text(marker, json.dumps(clean_json(
                {"stamp": stamp, "label": OFFLINE_LABEL, "row": row,
                 "parameters": fitted.to_dict()}), indent=1, default=str) + "\n")
            rows.append(row)
            del fitted, labels
            gc.collect()
    # every current offline fit on disk joins the table, whichever run produced it
    done = {(r["model"], int(r["states"])) for r in rows}
    for marker in sorted(regime.timeframe_dir(ctx.timeframe).glob(
            "*/k*/offline/summary_offline.json")):
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        row = recorded.get("row") or {}
        key = (row.get("model"), int(row.get("states", 0)))
        stamp = _stamp(ctx, model=key[0], states=key[1], stage="offline")
        if key not in done and _reusable(marker, stamp):
            rows.append(row)
            done.add(key)
    table = pl.DataFrame([clean_json(r) for r in rows], infer_schema_length=None)
    all_states = sorted({int(r["states"]) for r in rows} | set(states))
    baselines = [baseline_holdout(x, times, features, regime, k=k, split=split)
                 for k in all_states]
    base = pl.DataFrame(baselines, infer_schema_length=None)
    if not base.is_empty():
        single = base["single_state_test_ll_per_obs"][0]
        if "test_ll_per_obs" not in table.columns:     # K-Means alone: no likelihood at all
            table = table.with_columns(pl.lit(None, dtype=pl.Float64).alias("test_ll_per_obs"))
        table = table.join(base.select("states", "volatility_buckets_test_ll_per_obs"),
                           on="states", how="left").with_columns(
            pl.lit(single).alias("single_state_test_ll_per_obs"),
            (pl.col("test_ll_per_obs") - pl.col("volatility_buckets_test_ll_per_obs"))
            .alias("gain_over_volatility_buckets"),
            (pl.col("test_ll_per_obs") - single).alias("gain_over_single_state"))
    _write_table(table.with_columns(pl.lit(OFFLINE_LABEL).alias("label")),
                 directory / "model_selection")
    _write_table(base, directory / "baselines_holdout")
    representative = 3 if 3 in states else states[0]
    for family in [m for m in ("gmm", "hmm") if m in models]:
        scaling_path = directory / f"scaling_comparison_{family}.json"
        stamp_s = _stamp(ctx, stage="scaling", model=family, states=representative)
        if resume and _reusable(scaling_path, stamp_s):
            continue
        ctx.log(f"scaling comparison {family} K={representative}")
        rows_s = scaling_comparison(x, features, regime, family=family, k=representative,
                                    seed=regime.seed)
        atomic_write_text(scaling_path, json.dumps(clean_json(
            {"stamp": stamp_s, "label": OFFLINE_LABEL, "rows": rows_s}),
            indent=1, default=str) + "\n")
    if make_plots and regime.plots.enabled and not table.is_empty():
        pdir = ensure_dir(directory / "plots")
        per_obs = table.with_columns((pl.col("bic") / pl.col("observations")).alias("bic_per_obs"))
        plots.plot_model_selection(per_obs, metric="bic_per_obs", ylabel="BIC / bars (lower is "
                                   "better)", path=pdir / "bic_by_k.png", config=regime,
                                   title=f"{ctx.timeframe}: BIC by K - {OFFLINE_LABEL}")
        plots.plot_model_selection(table, metric="test_ll_per_obs",
                                   ylabel="hold-out log-likelihood per bar (higher is better)",
                                   path=pdir / "holdout_ll_by_k.png", config=regime,
                                   title=f"{ctx.timeframe}: chronological hold-out likelihood")
    return table


def _holdout_split(regime: RegimeConfig) -> datetime:
    """The hold-out starts at the first test block of the information study (2011 by default)
    - a round date fixed before any result, the same for every model."""
    return datetime(regime.incremental.test_blocks[0][0], 1, 1)


def _offline_tables(ctx: RegimeContext, fitted: Any, labels: np.ndarray, x: np.ndarray,
                    out_dir: Path, *, make_plots: bool) -> None:
    regime = ctx.regime
    k = fitted.n_states
    profiles = state_profiles(ctx.inputs, labels, quantiles=regime.offline.profile_quantiles,
                              label=OFFLINE_LABEL)
    _write_table(profiles, out_dir / "state_profiles_offline")
    std = standardized_means(x, labels, ctx.features)
    _write_table(std, out_dir / "standardized_means_offline")
    runs = run_lengths(labels)
    expected = fitted.model.expected_durations() if isinstance(fitted.model, HMMModel) else None
    _write_table(duration_summary(runs, k, expected=expected), out_dir / "durations_offline")
    if isinstance(fitted.model, HMMModel):
        _write_table(_matrix_frame(fitted.model.transition), out_dir / "transition_matrix_offline")
    if make_plots and regime.plots.enabled and not std.is_empty():
        plots.plot_feature_profiles(std, path=out_dir / "plots" / "feature_profiles_offline.png",
                                    config=regime,
                                    title=f"{ctx.timeframe} {fitted.family} K={k}: standardised "
                                          f"state medians - {OFFLINE_LABEL}")


def _matrix_frame(matrix: np.ndarray) -> pl.DataFrame:
    k = matrix.shape[0]
    return pl.DataFrame({"from_state": list(range(k)),
                         **{f"to_state_{j}": matrix[:, j] for j in range(k)}})


# ---------------------------------------------------------------------------
# Stage: walk-forward (causal)
# ---------------------------------------------------------------------------
def _wf_dir(ctx: RegimeContext, model: str, k: int, scheme: str, refit: str) -> Path:
    return ensure_dir(ctx.regime.results_dir(ctx.timeframe, model, k) / "walk_forward"
                      / f"{scheme}_{refit}")


def walk_forward_stage(ctx: RegimeContext, *, model: str, k: int, scheme: str, refit: str,
                       resume: bool = True) -> WalkForwardResult:
    """One walk-forward run, reused from disk when current; writes the live-safe features."""
    regime = ctx.regime
    directory = _wf_dir(ctx, model, k, scheme, refit)
    stamp = _stamp(ctx, model=model, states=k, scheme=scheme, refit=refit, stage="walk_forward")
    marker = directory / "walk_forward.json"
    if resume and _reusable(marker, stamp) and (directory / "per_bar.parquet").exists():
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        return WalkForwardResult(ctx.timeframe, model, k, scheme, refit,
                                 regime.primary_feature_set,
                                 frame=pl.read_parquet(directory / "per_bar.parquet"),
                                 refits=pl.read_parquet(directory / "refits.parquet"),
                                 model_ids=list(recorded.get("model_ids") or []),
                                 seconds=float(recorded.get("seconds") or 0.0),
                                 anchor=recorded.get("anchor") or {})
    ctx.log(f"walk-forward {model} K={k} {scheme}/{refit}")
    registry = ModelRegistry(regime.models_path)
    result = run_walk_forward(ctx.inputs, regime, timeframe=ctx.timeframe, family=model,
                              states=k, scheme=scheme, refit=refit, registry=registry,
                              provenance={**{key: ctx.provenance[key] for key in (
                                  "tick_dataset_version", "bar_dataset_version",
                                  "feature_versions", "regime_config_fingerprint",
                                  "code_fingerprint", "git", "regime_model_version")}},
                              progress=ctx.progress)
    result.frame.write_parquet(directory / "per_bar.parquet")
    _write_table(result.refits, directory / "refits")
    features_root = write_regime_features(result, regime, manifest={
        "timeframe": ctx.timeframe, "model": model, "states": k, "scheme": scheme,
        "refit": refit, "feature_set": regime.primary_feature_set,
        "features": list(ctx.features), **ctx.provenance})
    atomic_write_text(marker, json.dumps(clean_json({
        "stamp": stamp, "model_ids": result.model_ids, "seconds": result.seconds,
        "anchor": result.anchor, "features_path": features_root.as_posix(),
        "rows": result.frame.height, "generated_utc": utc_now_iso()}), indent=1,
        default=str) + "\n")
    return result


# ---------------------------------------------------------------------------
# Stage: nulls
# ---------------------------------------------------------------------------
def run_nulls(ctx: RegimeContext, *, states: list[int], resume: bool = True) -> pl.DataFrame:
    """HMM persistence on the real features and on every null, same span, same code."""
    regime = ctx.regime
    directory = ctx.directory("nulls")
    cfg = regime.nulls
    stamp = _stamp(ctx, stage="nulls", states=states, row_shuffle=cfg.row_shuffle,
                   row_block_days=list(cfg.row_block_days), pipeline=list(cfg.pipeline),
                   seed=cfg.seed, max_rows=cfg.max_rows)
    marker = directory / "nulls.json"
    if resume and _reusable(marker, stamp) and (directory / "null_controls.parquet").exists():
        return pl.read_parquet(directory / "null_controls.parquet")
    features = ctx.features
    n = ctx.inputs.height
    start = max(0, n - regime.nulls.max_rows)
    span = slice(start, n)
    real = feature_matrix(ctx.inputs, features)[span]
    sources: dict[str, np.ndarray] = {"real": real}
    seed = regime.nulls.seed
    if regime.nulls.row_shuffle:
        sources["row_shuffle"] = shuffle_rows(real, seed)
    bars_day = regime.inputs.bars_per_day(bar_seconds(ctx.timeframe))
    for days in regime.nulls.row_block_days:
        sources[f"row_block_{days}d"] = block_bootstrap_rows(real, days * bars_day, seed + days)
    for name in regime.nulls.pipeline:
        frame = null_inputs(ctx, name)
        sources[name] = feature_matrix(frame, features)[span]
        del frame
        gc.collect()
    table = null_persistence(sources, features, regime, states=tuple(states), seed=seed,
                             progress=ctx.log)
    table = table.with_columns(pl.lit(ctx.timeframe).alias("timeframe"),
                               pl.lit(str(ctx.inputs["timestamp"][start])).alias("span_start"),
                               pl.lit(n - start).alias("span_bars"))
    _write_table(table, directory / "null_controls")
    atomic_write_text(marker, json.dumps(clean_json({"stamp": stamp, "sources": list(sources),
                                                     "generated_utc": utc_now_iso()}),
                                         indent=1, default=str) + "\n")
    return table


def _settings(regime: RegimeConfig) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(regression_window=regime.inputs.regression_window,
                           ou_window=regime.inputs.ou_window,
                           controls=SimpleNamespace(seed=regime.nulls.seed, block_size=256))


#: The pipeline nulls an increment must beat (REG-H-001, REG-H-007).
_INCREMENT_NULLS: tuple[str, ...] = ("random_walk", "block_bootstrap_1024")


def null_inputs(ctx: RegimeContext, name: str) -> pl.DataFrame:
    """One pipeline null's full-length regime inputs, cached per timeframe."""
    directory = ctx.directory("nulls")
    path = directory / f"inputs_{name}.parquet"
    stamp = _stamp(ctx, stage="null_inputs", null=name, seed=ctx.regime.nulls.seed,
                   inputs=ctx.regime.inputs_fingerprint())
    marker = directory / f"inputs_{name}.json"
    if _reusable(marker, stamp) and path.exists():
        return pl.read_parquet(path)
    ctx.log(f"building pipeline-null inputs: {name}")
    source = build_real_source(ctx.config, ctx.regression, ctx.ou, _settings(ctx.regime),
                               ctx.timeframe)
    frame = null_regime_inputs(name, source, ctx.inputs, ctx.regression, ctx.ou, ctx.spectral,
                               ctx.wavelet, ctx.regime)
    del source
    gc.collect()
    frame.write_parquet(path, compression="zstd")
    atomic_write_text(marker, json.dumps(clean_json({"stamp": stamp, "rows": frame.height,
                                                     "generated_utc": utc_now_iso()}),
                                         indent=1) + "\n")
    return frame


def _real_increment_passes(inc: pl.DataFrame, mr: dict[str, Any]) -> bool:
    """Whether any REG-H-001 / -007 case passes its fold and random-state tests (the only
    cases whose verdict the pipeline nulls can still change)."""
    for value in mr.values():
        if isinstance(value, dict):
            soft = value.get("log_loss_reduction_soft")
            rnd = value.get("log_loss_reduction_random") or 0.0
            if soft is not None and soft > 0 and soft > rnd and value.get(
                    "folds_soft_better", 0) >= 3:
                return True
    if isinstance(inc, pl.DataFrame) and not inc.is_empty():
        for r in inc.filter(pl.col("metric").is_in(["oos_r2", "log_loss"])).iter_rows(named=True):
            soft, rnd = r.get("delta_A+soft"), r.get("delta_A+random")
            if soft is not None and soft > 0 and (r.get("folds_better_A+soft") or 0) >= 3 and \
                    soft > (rnd if rnd is not None else 0):
                return True
    return False


def null_increment_study(ctx: RegimeContext, *, model: str, k: int, seed: int
                         ) -> tuple[pl.DataFrame, dict[str, dict[str, Any]]]:
    """The same walk-forward model and increment tests on each pipeline null's own features."""
    regime = ctx.regime
    directory = ensure_dir(regime.results_dir(ctx.timeframe, model, k) / "null_increments")
    summaries: list[pl.DataFrame] = []
    reversion: dict[str, dict[str, Any]] = {}
    for name in _INCREMENT_NULLS:
        stamp = _stamp(ctx, stage="null_increment", model=model, states=k, null=name)
        marker = directory / f"{name}.json"
        table_path = directory / f"{name}.parquet"
        if _reusable(marker, stamp) and table_path.exists():
            summaries.append(pl.read_parquet(table_path))
            reversion[name] = json.loads(marker.read_text(encoding="utf-8"))["reversion"]
            continue
        frame = null_inputs(ctx, name)
        ctx.log(f"{model} K={k}: walk-forward on {name}")
        result = run_walk_forward(frame, regime, timeframe=ctx.timeframe, family=model, states=k,
                                  scheme=regime.causal.primary_scheme,
                                  refit=regime.causal.primary_refit, registry=None)
        source = source_from_frame(frame, bar_seconds(ctx.timeframe), name=name)
        labels = _aligned_labels(frame, result.frame, f"{model}_state")
        empirical = _empirical_transition(labels[labels >= 0], k)
        tables = incremental_information(frame, result.frame, source, family=model, k=k,
                                         transition=empirical, cfg=regime.incremental,
                                         seed=seed)
        summary = incremental_summary(tables).with_columns(pl.lit(name).alias("source"))
        mr = mean_reversion_increment(frame, result.frame, source, model, k, regime, seed)
        mr.pop("rows", None)
        summary.write_parquet(table_path)
        atomic_write_text(marker, json.dumps(clean_json({"stamp": stamp, "reversion": mr,
                                                         "generated_utc": utc_now_iso()}),
                                             indent=1) + "\n")
        summaries.append(summary)
        reversion[name] = mr
        del frame, result, tables
        gc.collect()
    return (pl.concat(summaries, how="diagonal_relaxed") if summaries else pl.DataFrame(),
            reversion)


def run_synthetic_controls(regime: RegimeConfig) -> dict[str, Path]:
    """Steps 44-46: known HMM, one regime (Gaussian, Student-t), smooth continuum."""
    directory = ensure_dir(regime.results_path / "synthetic_controls")
    tables = synthetic_controls(regime)
    written = {}
    for name, frame in tables.items():
        _write_table(frame, directory / name)
        written[name] = directory / f"{name}.csv"
    return written


# ---------------------------------------------------------------------------
# Stage: analysis of one (model, K)
# ---------------------------------------------------------------------------
@dataclass
class RegimeStudy:
    """Outcome of one (timeframe, model, K) study."""

    timeframe: str
    model: str
    states: int
    output_dir: Path
    summary: dict[str, Any] = field(default_factory=dict)
    files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    reused: bool = False


def causal_volatility_buckets(times: np.ndarray, vol: np.ndarray, k: int,
                              regime: RegimeConfig) -> np.ndarray:
    """Volatility buckets with edges from all earlier bars, refreshed on the refit schedule."""
    labels = np.full(vol.size, -1, dtype=np.int64)
    first = datetime.fromisoformat(regime.causal.first_inference)
    ends = pl.Series(times[[0, -1]]).to_list()
    periods = refit_schedule(ends[0], ends[1], first_inference=first,
                             frequency=regime.causal.primary_refit, scheme="expanding",
                             rolling_years=regime.causal.rolling_training_years)
    for p in periods:
        lo = int(np.searchsorted(times, np.datetime64(p.infer_start), side="left"))
        hi = int(np.searchsorted(times, np.datetime64(p.infer_end), side="left"))
        past = vol[:lo][np.isfinite(vol[:lo])]
        if hi <= lo or past.size < 100:
            continue
        edges = np.quantile(past, np.linspace(0, 1, k + 1)[1:-1])
        labels[lo:hi], _ = bucket_labels(vol[lo:hi], k, edges=edges)
    return labels


def _aligned_labels(inputs: pl.DataFrame, frame: pl.DataFrame, column: str) -> np.ndarray:
    """A per-bar column of a walk-forward frame on the input table's rows (-1 elsewhere)."""
    joined = pl.DataFrame({"timestamp": inputs["timestamp"]}).join(
        frame.select("timestamp", column), on="timestamp", how="left")
    return joined[column].fill_null(-1).to_numpy().astype(np.int64)


def _empirical_transition(labels: np.ndarray, k: int) -> np.ndarray:
    counts = transition_counts(labels, k).astype(np.float64) + 1e-9
    return counts / counts.sum(axis=1, keepdims=True)


def analyse_study(ctx: RegimeContext, *, model: str, k: int, primary: WalkForwardResult,
                  others: dict[str, WalkForwardResult | None], nulls: pl.DataFrame | None,
                  offline_row: dict[str, Any] | None, make_plots: bool = True) -> RegimeStudy:
    """Every Step 68 table for one (model, K), from the causal states."""
    started = time.perf_counter()
    regime = ctx.regime
    inputs = ctx.inputs
    study = RegimeStudy(ctx.timeframe, model, k, regime.results_dir(ctx.timeframe, model, k))
    writer = StudyWriter(study.output_dir, regime.output, study.files)

    def put(table_: pl.DataFrame | None, name: str) -> None:
        """Parquet + CSV (nested columns as JSON in the CSV), recorded in the study."""
        if table_ is None or table_.is_empty():
            return
        _write_table(table_, study.output_dir / name)
        study.files += [f"{name}.parquet", f"{name}.csv"]

    frame = primary.frame
    state_col = f"{model}_state"
    labels = _aligned_labels(inputs, frame, state_col)
    covered = labels >= 0
    rows = np.flatnonzero(covered)
    sub = inputs[rows]
    lab = labels[rows]
    times = inputs["timestamp"].to_numpy()
    vol = inputs[regime.baselines.volatility_feature].to_numpy()
    vol_buckets = causal_volatility_buckets(times, vol, k, regime)[rows]
    empirical = _empirical_transition(lab, k)
    rng_seed = regime.baselines.random_label_seed + 17 * k
    chance = random_labels(lab.size, k, seed=rng_seed, transition=empirical)
    labelings = {"regime": lab, "volatility_buckets": vol_buckets, "random_labels": chance}
    x = feature_matrix(sub, ctx.features)
    # --- descriptions (Steps 21, 22, 62) ---
    profiles = state_profiles(sub, lab, quantiles=regime.offline.profile_quantiles,
                              label="causal walk-forward states")
    put(profiles, "state_profiles")
    std = standardized_means(x, lab, ctx.features)
    put(std, "standardized_state_medians")
    descriptions = describe_states(std)
    contribution = feature_contribution(sub, lab, list(ctx.features))
    bucket_contribution = feature_contribution(sub, vol_buckets, list(ctx.features))
    if not contribution.is_empty():
        contribution = contribution.join(bucket_contribution.rename(
            {"eta_squared": "eta_squared_volatility_buckets",
             "mutual_information_nats": "mutual_information_volatility_buckets"}),
            on="feature", how="left")
    put(contribution, "feature_contribution")
    # --- transitions and durations (Steps 16-17) ---
    runs = run_lengths(lab)
    refits = primary.refits.filter(pl.col("status") == "fitted") if "status" in \
        primary.refits.columns else primary.refits
    expected = None
    last_transition = None
    if "expected_durations" in refits.columns and refits.height:
        durations = [json.loads(v) for v in refits["expected_durations"].to_list() if v]
        expected = np.exp(np.median(np.log(np.clip(np.asarray(durations), 1.0, 1e9)), axis=0))
        last_transition = np.asarray(json.loads(refits["transition"][-1]))
    duration_table = duration_summary(runs, k, expected=expected)
    put(duration_table, "state_durations")
    counts = transition_counts(lab, k)
    matrices = [_matrix_frame(empirical).with_columns(pl.lit("empirical_filtered_states")
                                                      .alias("matrix"))]
    if last_transition is not None:
        matrices.append(_matrix_frame(last_transition).with_columns(
            pl.lit("model_last_refit").alias("matrix")))
    put(pl.concat(matrices, how="diagonal_relaxed").with_columns(
        pl.lit(json.dumps(counts.tolist())).alias("empirical_counts")), "transition_matrix")
    # --- frequency by time (Steps 38-39) ---
    stamps = sub["timestamp"]
    yearly = frequency_table(stamps, lab, by="year", k=k)
    put(yearly, "yearly_frequency")
    put(frequency_table(stamps, lab, by="quarter", k=k), "quarterly_frequency")
    intraday, intraday_stats = intraday_frequency(stamps, lab, ctx.research, k=k)
    put(intraday, "intraday_frequency")
    # --- stability (Steps 27, 40-41) ---
    put(primary.refits, "stability")
    stability = refit_stability_summary(primary.refits)
    schemes = scheme_comparison({f"{primary.scheme}_{primary.refit}": primary, **others}, model)
    put(pl.DataFrame([clean_json(r) for r in schemes], infer_schema_length=None),
                 "scheme_comparison")
    eras = era_definitions(primary.refits, ctx.features, eras=_ERAS)
    put(eras, "era_definitions")
    # --- volatility baseline (Steps 9, 43, question N) ---
    vol_agreement = agreement(lab, vol_buckets)
    eta_regime = [eta_squared(x[:, j], lab) for j, f in enumerate(ctx.features)
                  if f in _NON_VOLATILITY]
    eta_buckets = [eta_squared(x[:, j], vol_buckets) for j, f in enumerate(ctx.features)
                   if f in _NON_VOLATILITY]
    baseline = {**{f"agreement_{key}": v for key, v in vol_agreement.items()},
                "non_volatility_eta2_states": _mean(eta_regime),
                "non_volatility_eta2_volatility_buckets": _mean(eta_buckets),
                "offline_gain_over_volatility_buckets": (offline_row or {}).get(
                    "gain_over_volatility_buckets"),
                "offline_gain_over_single_state": (offline_row or {}).get(
                    "gain_over_single_state")}
    table = contingency(lab, vol_buckets)
    put(pl.DataFrame([baseline], infer_schema_length=None), "volatility_baseline")
    # --- outcomes (Steps 32-37, 63) ---
    horizons = regime.outcomes.horizons
    outcomes = conditional_outcomes(sub, labelings, k=k, horizons=horizons)
    put(outcomes, "regime_outcomes")
    reversion = mean_reversion_by_state(sub, labelings, k=k, horizons=horizons,
                                        extreme=regime.outcomes.residual_extreme)
    put(reversion, "mean_reversion_by_state")
    ou_table = ou_by_state(sub, lab, k=k, extreme=regime.outcomes.residual_extreme)
    put(ou_table, "ou_by_state")
    fft_n = regime.inputs.fft_window
    put(state_medians(sub, lab, k=k, columns=(
        "spectral_entropy", "fft_high_low_log_ratio", "fft_top3_power_share",
        "fft_dominant_period_bars"), lag_persistence={"fft_dominant_period_bars": fft_n}),
        "spectral_by_state")
    put(state_medians(sub, lab, k=k, columns=(
        "wavelet_entropy", "wavelet_fast_slow_log_ratio", "wavelet_top3_scale_share",
        "wavelet_dominant_period_bars", "wavelet_dominant_run_length", "wavelet_scale_drift"),
        lag_persistence={"wavelet_dominant_period_bars": regime.inputs.wavelet_window}),
        "wavelet_by_state")
    put(state_medians(sub, lab, k=k, columns=(
        "median_spread", "mean_spread", "spread_percentile", "tick_count",
        "activity_percentile")), "spread_activity_by_state")
    fut_vol = future_volatility_change(sub, labelings, k=k, horizons=(5, 20))
    put(fut_vol, "future_volatility_by_state")
    # --- calibration and uncertainty (Steps 31, 60, 61) ---
    calibration, calibration_summary = calibration_tables(frame, model, k,
                                                          bins=regime.calibration.bins)
    put(calibration, "calibration")
    # the scored bars only, row for row with `sub` (a GMM leaves unscored bars at -1)
    entropy = entropy_analysis(frame.filter(pl.col(state_col) >= 0), sub, model)
    put(entropy, "entropy_analysis")
    # --- incremental information (Steps 58-59) ---
    inc_summary = pl.DataFrame()
    mr_summary: dict[str, Any] = {}
    null_inc = pl.DataFrame()
    null_mr: dict[str, dict[str, Any]] = {}
    if regime.incremental.enabled:
        ctx.log(f"{model} K={k}: incremental information")
        source = source_from_frame(inputs, bar_seconds(ctx.timeframe))
        tables = incremental_information(inputs, frame, source, family=model, k=k,
                                         transition=empirical, cfg=regime.incremental,
                                         seed=rng_seed)
        linear, binary = tables["linear"], tables["binary"]
        put(pl.concat([linear.with_columns(pl.lit("continuous").alias("kind")),
                                binary.with_columns(pl.lit("binary").alias("kind"))],
                               how="diagonal_relaxed") if not (linear.is_empty()
                                                               and binary.is_empty())
                     else None, "incremental_information")
        inc_summary = incremental_summary(tables)
        put(inc_summary, "incremental_summary")
        mr_summary = mean_reversion_increment(inputs, frame, source, model, k, regime,
                                               rng_seed)
        put(pl.DataFrame(mr_summary.get("rows", []), infer_schema_length=None),
                     "mean_reversion_increment")
        if regime.nulls.enabled and model in regime.nulls.increment_models and \
                k in regime.nulls.increment_states and \
                _real_increment_passes(inc_summary, mr_summary):
            ctx.log(f"{model} K={k}: the real increment passes its first tests; pipeline nulls")
            null_inc, null_mr = null_increment_study(ctx, model=model, k=k, seed=rng_seed)
            put(null_inc, "incremental_summary_nulls")
            put(pl.DataFrame([{"source": name, "horizon": key, **value}
                              for name, per in null_mr.items() for key, value in per.items()
                              if isinstance(value, dict)], infer_schema_length=None),
                "mean_reversion_increment_nulls")
    # --- nulls (Steps 43, 47-48) ---
    null_rows = (nulls.filter(pl.col("states") == k) if isinstance(nulls, pl.DataFrame)
                 and not nulls.is_empty() and model == "hmm" else pl.DataFrame())
    random_stats = _random_label_nulls(sub, lab, empirical, k, rng_seed)
    put(pl.concat([null_rows, pl.DataFrame([random_stats], infer_schema_length=None)
                            .with_columns(pl.lit("random_labels").alias("source"))],
                           how="diagonal_relaxed") if not null_rows.is_empty()
                 else pl.DataFrame([random_stats], infer_schema_length=None).with_columns(
        pl.lit("random_labels").alias("source")), "null_controls")
    # --- summary (Step 69) ---
    counts_bars = np.bincount(lab, minlength=k)
    probs_cols = [f"{model}_p{j}" for j in range(k)]
    confidence = frame[f"{model}_state_confidence"].to_numpy() if \
        f"{model}_state_confidence" in frame.columns else None
    entropy_col = frame[f"{model}_entropy"].to_numpy() if f"{model}_entropy" in \
        frame.columns else None
    fitted_refits = refits
    ll = fitted_refits["oos_ll_per_obs"].cast(pl.Float64).to_numpy() if \
        "oos_ll_per_obs" in fitted_refits.columns else np.array([])
    weights = fitted_refits["oos_complete_rows"].cast(pl.Float64).to_numpy() if \
        "oos_complete_rows" in fitted_refits.columns else np.array([])
    ok = np.isfinite(ll) & (weights > 0) if ll.size else np.array([], dtype=bool)
    summary: dict[str, Any] = {
        "timeframe": ctx.timeframe, "model": model, "states": k,
        "feature_set": regime.primary_feature_set, "features": list(ctx.features),
        "scheme": f"{primary.scheme}_{primary.refit}",
        "observations": int(lab.size),
        "first_timestamp": str(stamps[0]) if stamps.len() else None,
        "last_timestamp": str(stamps[-1]) if stamps.len() else None,
        "refits": stability.get("refits"),
        # K-Means has no likelihood: its per-bar score is minus the squared distance
        "out_of_sample_log_likelihood": (float(np.average(ll[ok], weights=weights[ok]))
                                         if ok.any() and model != "kmeans" else None),
        "out_of_sample_mean_sq_distance": (float(-np.average(ll[ok], weights=weights[ok]))
                                           if ok.any() and model == "kmeans" else None),
        "state_frequencies": {f"state_{j}": float(counts_bars[j] / max(lab.size, 1))
                              for j in range(k)},
        "median_state_duration": {f"state_{r['state']}": r.get("median_duration")
                                  for r in duration_table.iter_rows(named=True)},
        "expected_state_duration": ({f"state_{j}": float(v) for j, v in enumerate(expected)}
                                    if expected is not None else None),
        "transition_matrix": last_transition.tolist() if last_transition is not None else None,
        "empirical_transition_matrix": empirical.tolist(),
        "mean_state_confidence": float(np.nanmean(confidence)) if confidence is not None
        else None,
        "mean_state_entropy": float(np.nanmean(entropy_col)) if entropy_col is not None
        else None,
        "state_alignment_stability": {"median_overlap_ari": stability.get(
            "overlap_ari__median"), "median_alignment_cost": stability.get(
            "alignment_cost__median"), "max_anchor_bhattacharyya": stability.get(
            "max_anchor_bhattacharyya")},
        "stability": stability,
        "volatility_baseline_improvement": baseline,
        "intraday": intraday_stats,
        "incremental_information": _incremental_digest(inc_summary),
        "mean_reversion_increment": {k2: v for k2, v in mr_summary.items() if k2 != "rows"},
        "calibration": calibration_summary,
        "state_descriptions": {f"state_{s}": d for s, d in descriptions.items()},
        "offline": offline_row,
        "probability_columns": probs_cols if probs_cols[0] in frame.columns else [],
        "live_safe": True,
        "live_features_path": (regime.features_path / f"timeframe={ctx.timeframe}"
                               / f"model={model}" / f"k={k}"
                               / f"scheme={primary.scheme}_{primary.refit}").as_posix(),
        "partial_dataset": bool(ctx.lineage.get("partial")),
        "caveat": CAVEAT,
    }
    study.summary = clean_json(summary)
    entries = _ledger_entries(ctx, model=model, k=k, summary=summary, inc=inc_summary,
                              mr=mr_summary, null_inc=null_inc, null_mr=null_mr, std=std,
                              nulls=null_rows, stability=stability, schemes=schemes,
                              calibration=calibration_summary, random_stats=random_stats,
                              sub=sub, lab=lab)
    if regime.ledger.enabled and entries:
        ResearchLedger(regime.ledger_path).upsert(entries)
    put(pl.DataFrame([{kk: (json.dumps(v, default=str) if isinstance(v, (dict, list))
                                     else v) for kk, v in e.items()} for e in entries],
                              infer_schema_length=None) if entries else None,
                 "hypothesis_verdicts")
    writer.json({"summary": study.summary, "provenance": {
        "report_version": REGIME_REPORT_VERSION, "generated_utc": utc_now_iso(),
        **ctx.provenance, "packages": package_versions(_PACKAGES),
        "integrity_gate": {kk: v for kk, v in ctx.gate.items() if kk != "lineage"},
        "sampling": {"incremental_rows": regime.incremental.max_rows,
                     "note": "incremental models use evenly spaced rows (estimates)"},
        "model_ids": primary.model_ids}, "hypotheses": HYPOTHESES,
        "warnings": study.warnings}, "summary")
    if make_plots and regime.plots.enabled:
        try:
            _plots(ctx, study, model=model, k=k, frame=frame, sub=sub, lab=lab, std=std,
                   runs=runs, yearly=yearly, table=table, reversion=reversion,
                   refits=primary.refits, last_transition=last_transition)
        except Exception as exc:  # a figure must never lose the tables
            LOGGER.warning("plots for %s/%s/k%d failed: %s", ctx.timeframe, model, k, exc)
            study.warnings.append(f"plots failed: {exc}")
    study.duration_seconds = time.perf_counter() - started
    return study


def _mean(values: list[float | None]) -> float | None:
    known = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(known)) if known else None


def _incremental_digest(summary: pl.DataFrame) -> dict[str, Any]:
    if summary.is_empty():
        return {}
    out = {}
    for r in summary.iter_rows(named=True):
        key = f"{r['target']}_h{r['horizon']}_{r['metric']}"
        out[key] = {m: r.get(m) for m in ("A", "delta_A+soft", "delta_A+hard",
                                           "delta_A+volatility", "delta_A+random",
                                           "folds_better_A+soft")}
    return out


def _random_label_nulls(sub: pl.DataFrame, lab: np.ndarray, transition: np.ndarray, k: int,
                        seed: int) -> dict[str, Any]:
    """95th percentiles of the REG-H-003 / -006 statistics over random state sequences."""
    eps = np.abs(sub["regression_residual"].to_numpy())
    z = np.abs(sub["residual_zscore"].to_numpy())
    later = np.full(eps.size, np.nan)
    later[:-10] = eps[10:]
    ratio = np.where((z > 2.0) & np.isfinite(later) & (eps > 0), later / eps, np.nan)
    vol_change = _future_vol_change(sub, 20)
    decile, _ = bucket_labels(sub["log_rv_20"].to_numpy(), 10)
    ranges, diffs = [], []
    for draw in range(_RANDOM_DRAWS):
        chance = random_labels(lab.size, k, seed=seed + 1000 + draw, transition=transition)
        ranges.append(_decay_range(ratio, chance, k))
        diffs.append(abs(_within_decile_difference(vol_change, decile, chance == 0) or 0.0))
    return {"random_decay_range_p95": float(np.nanpercentile(ranges, 95)),
            "random_vol_change_diff_p95": float(np.nanpercentile(diffs, 95)),
            "draws": _RANDOM_DRAWS}


def _future_vol_change(sub: pl.DataFrame, h: int) -> np.ndarray:
    r = sub["log_return"].to_numpy()
    sq = np.where(np.isfinite(r), r ** 2, np.nan)
    csum = np.concatenate(([0.0], np.cumsum(np.nan_to_num(sq))))
    out = np.full(r.size, np.nan)
    idx = np.arange(r.size - h)
    mean_sq = (csum[idx + h + 1] - csum[idx + 1]) / h
    with np.errstate(divide="ignore", invalid="ignore"):
        out[idx] = np.where(mean_sq > 0, 0.5 * np.log(mean_sq), np.nan)
    return out - sub["log_rv_20"].to_numpy()


def _decay_range(ratio: np.ndarray, labels: np.ndarray, k: int) -> float:
    medians = [np.nanmedian(ratio[labels == s]) for s in range(k)
               if np.isfinite(ratio[labels == s]).sum() >= 30]
    return float(max(medians) - min(medians)) if len(medians) >= 2 else np.nan


def _within_decile_difference(change: np.ndarray, decile: np.ndarray,
                              member: np.ndarray) -> float | None:
    diffs, weights = [], []
    for d in range(10):
        in_d = (decile == d) & np.isfinite(change)
        sel = in_d & member
        rest = in_d & ~member
        if sel.sum() >= 20 and rest.sum() >= 20:
            diffs.append(change[sel].mean() - change[rest].mean())
            weights.append(sel.sum())
    return float(np.average(diffs, weights=weights)) if diffs else None


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------
def increment_verdict(soft: Any, rnd: Any, folds: Any, null_values: dict[str, Any], *,
                      no_increment: str, beyond: str) -> tuple[str, float | None, float | None]:
    """REG-H-001 / -007: (verdict, value, control) for one target and horizon.

    The increment must be positive, win in >= 3 of 4 folds, beat random states
    (0 when their draws are unavailable) and then beat every pipeline null that
    was measured. A non-finite input is unavailable, never a comparison: an
    undefined increment is untested, an undefined null is left out.
    """
    soft, rnd = finite_or_none(soft), finite_or_none(rnd)
    known = [v for v in (finite_or_none(x) for x in null_values.values()) if v is not None]
    control = max(known) if known else rnd
    if soft is None:
        return "untested", None, control
    if not (soft > 0 and (folds or 0) >= 3 and soft > (rnd if rnd is not None else 0.0)):
        return no_increment, soft, control
    if not known:
        return "untested_against_pipeline_nulls", soft, control
    return (beyond if soft > max(known) else "within_pipeline_nulls"), soft, control


def _ledger_entries(ctx: RegimeContext, *, model: str, k: int, summary: dict[str, Any],
                    inc: pl.DataFrame, mr: dict[str, Any], null_inc: pl.DataFrame,
                    null_mr: dict[str, dict[str, Any]], std: pl.DataFrame,
                    nulls: pl.DataFrame, stability: dict[str, Any],
                    schemes: list[dict[str, Any]], calibration: dict[str, Any],
                    random_stats: dict[str, Any], sub: pl.DataFrame,
                    lab: np.ndarray) -> list[dict[str, Any]]:
    """Every registered hypothesis this study can speak to, with its a-priori verdict."""
    base = {"timeframe": ctx.timeframe, "input_series": ctx.regime.primary_feature_set,
            "window": k, "dataset_version": ctx.provenance["tick_dataset_version"],
            "study": f"{ctx.timeframe}/{model}/k{k}"}
    entries: list[dict[str, Any]] = []

    def add(hid: str, **values: Any) -> None:
        entries.append({**base, "hypothesis_id": hid, "definition": HYPOTHESES[hid], **values})

    # REG-H-001: mean reversion beyond volatility quartiles (and beyond the pipeline nulls)
    for key, value in mr.items():
        if not key.startswith("h") or not isinstance(value, dict):
            continue
        rnd = value.get("log_loss_reduction_random")
        null_values = {name: (per.get(key) or {}).get("log_loss_reduction_soft")
                       for name, per in null_mr.items()}
        verdict, soft, control = increment_verdict(
            value.get("log_loss_reduction_soft"), rnd, value.get("folds_soft_better", 0),
            null_values, no_increment="no_increment", beyond="explains_beyond_volatility")
        add("REG-H-001", feature=model, target="residual_shrinks", horizon=int(key[1:]),
            metric="mean_log_loss_reduction_V+soft", value=soft, control_value=control,
            verdict=verdict,
            details={**value, "random_states": rnd, "pipeline_nulls": null_values})
    # REG-H-003: fast-energy + wide-spread state
    if not std.is_empty():
        wide = std.pivot(on="feature", index="state", values="standardized_median")
        if {"wavelet_fast_slow_log_ratio", "spread_percentile"} <= set(wide.columns):
            fast = int(wide.sort("wavelet_fast_slow_log_ratio", descending=True)["state"][0])
            spread = int(wide.sort("spread_percentile", descending=True)["state"][0])
            if fast != spread:
                add("REG-H-003", feature=model, target="future_log_rv_change", horizon=20,
                    metric="within_decile_difference", value=None, verdict="no_such_state",
                    details={"highest_fast_slow_state": fast, "widest_spread_state": spread})
            else:
                change = _future_vol_change(sub, 20)
                decile, _ = bucket_labels(sub["log_rv_20"].to_numpy(), 10)
                diff = finite_or_none(_within_decile_difference(change, decile, lab == fast))
                ceiling = finite_or_none(random_stats.get("random_vol_change_diff_p95"))
                verdict = ("untested" if diff is None or ceiling is None else
                           "distinct_beyond_volatility" if abs(diff) > ceiling
                           else "not_distinct")
                add("REG-H-003", feature=f"{model}|state_{fast}", target="future_log_rv_change",
                    horizon=20, metric="within_decile_difference", value=diff,
                    control_value=ceiling, verdict=verdict, details={"state": fast})
    # REG-H-006: realised decay range
    eps = np.abs(sub["regression_residual"].to_numpy())
    zz = np.abs(sub["residual_zscore"].to_numpy())
    later = np.full(eps.size, np.nan)
    later[:-10] = eps[10:]
    ratio = np.where((zz > 2.0) & np.isfinite(later) & (eps > 0), later / eps, np.nan)
    real_range = finite_or_none(_decay_range(ratio, lab, k))
    vol_buckets = causal_volatility_buckets(sub["timestamp"].to_numpy(),
                                            sub["log_rv_20"].to_numpy(), k, ctx.regime)
    bucket_range = finite_or_none(_decay_range(ratio, vol_buckets, k))
    ceiling = finite_or_none(random_stats.get("random_decay_range_p95"))
    if real_range is not None:
        controls = [v for v in (bucket_range, ceiling) if v is not None]
        add("REG-H-006", feature=model, target="realised_decay_ratio", horizon=10,
            metric="range_of_state_medians", value=real_range,
            control_value=max(controls) if controls else None,
            verdict=("untested" if len(controls) < 2 else "separates_beyond_volatility"
                     if real_range > max(controls) else "within_controls"),
            details={"volatility_bucket_range": bucket_range, "random_p95": ceiling})
    # REG-H-007 / -008: incremental information
    for r in inc.iter_rows(named=True) if not inc.is_empty() else []:
        if r["metric"] not in ("oos_r2", "log_loss"):
            continue
        rnd, hard = r.get("delta_A+random"), finite_or_none(r.get("delta_A+hard"))
        folds = r.get("folds_better_A+soft") or 0
        null_deltas: dict[str, Any] = {}
        if isinstance(null_inc, pl.DataFrame) and not null_inc.is_empty():
            part = null_inc.filter((pl.col("target") == r["target"])
                                   & (pl.col("horizon") == r["horizon"])
                                   & (pl.col("metric") == r["metric"]))
            null_deltas = dict(zip(part["source"].to_list(), part["delta_A+soft"].to_list(),
                                   strict=True))
        verdict, soft, control = increment_verdict(
            r.get("delta_A+soft"), rnd, folds, null_deltas,
            no_increment="no_incremental_information", beyond="incremental_beyond_nulls")
        add("REG-H-007", feature=model, target=r["target"], horizon=r["horizon"],
            metric=f"mean_delta_{r['metric']}_soft_vs_A", value=soft,
            control_value=control, verdict=verdict,
            details={"folds_better": folds, "folds": r.get("folds"), "random_states": rnd,
                     "pipeline_nulls": null_deltas,
                     "delta_volatility_buckets": r.get("delta_A+volatility"),
                     "delta_hard": hard})
        soft_vs_hard = r.get("folds_soft_beats_hard")
        if soft is not None and hard is not None and soft_vs_hard is not None:
            add("REG-H-008", feature=model, target=r["target"], horizon=r["horizon"],
                metric=f"mean_delta_{r['metric']}_soft_minus_hard", value=soft - hard,
                control_value=float(soft_vs_hard),
                verdict="soft_better" if soft_vs_hard >= 3 else "soft_not_better",
                details={"soft": soft, "hard": hard, "folds_soft_beats_hard": soft_vs_hard,
                         "folds": r.get("folds")})
    # REG-H-005: persistence beyond pipeline nulls (HMM)
    if model == "hmm" and isinstance(nulls, pl.DataFrame) and not nulls.is_empty():
        real_row = nulls.filter(pl.col("source") == "real")
        pipeline = nulls.filter(pl.col("source").is_in(list(ctx.regime.nulls.pipeline)))
        if real_row.height and pipeline.height:
            value = finite_or_none(real_row["median_expected_duration"][0])
            measured = [v for v in (finite_or_none(x) for x in
                                    pipeline["median_expected_duration"].to_list())
                        if v is not None]
            ceiling_n = max(measured) if measured else None
            add("REG-H-005", feature=model, target="", horizon=-1,
                metric="median_expected_duration", value=value, control_value=ceiling_n,
                verdict=("untested" if value is None or ceiling_n is None else
                         "persists_beyond_nulls" if value > 1.25 * ceiling_n
                         else "within_null_persistence"),
                details={r["source"]: r["median_expected_duration"]
                         for r in nulls.iter_rows(named=True)})
    # REG-H-009: calibration (HMM)
    slope = finite_or_none(calibration.get("leave_calibration_slope"))
    if model == "hmm" and slope is not None:
        gap = finite_or_none(calibration.get("leave_max_reliability_gap"))
        add("REG-H-009", feature=model, target="state_change_next_bar", horizon=1,
            metric="leave_calibration_slope", value=slope, control_value=gap,
            verdict=("untested" if gap is None else "calibrated"
                     if 0.8 <= slope <= 1.25 and gap < 0.05 else "miscalibrated"),
            details=calibration)
    # REG-H-011: volatility in disguise
    baseline = summary["volatility_baseline_improvement"]
    nmi = finite_or_none(baseline.get("agreement_nmi"))
    eta_s, eta_b = (finite_or_none(baseline.get("non_volatility_eta2_states")),
                    finite_or_none(baseline.get("non_volatility_eta2_volatility_buckets")))
    if nmi is not None and eta_s is not None and eta_b is not None:
        verdict = ("more_than_volatility_buckets" if nmi < 0.5 and eta_s > eta_b else
                   "volatility_buckets_in_disguise" if nmi >= 0.5 and eta_s <= eta_b else
                   "mixed")
        add("REG-H-011", feature=model, target="", horizon=-1, metric="nmi_with_volatility_buckets",
            value=nmi, control_value=0.5, verdict=verdict,
            details={"eta2_states": eta_s, "eta2_buckets": eta_b})
    # REG-H-012: expanding vs rolling
    by_scheme = {r["scheme"]: r for r in schemes if " vs " not in str(r.get("scheme"))}
    exp = next((v for s, v in by_scheme.items() if s.startswith("expanding_")), None)
    roll = next((v for s, v in by_scheme.items() if s.startswith("rolling_")), None)
    exp_ll = finite_or_none((exp or {}).get("oos_ll_per_obs_weighted"))
    roll_ll = finite_or_none((roll or {}).get("oos_ll_per_obs_weighted"))
    if exp_ll is not None and roll_ll is not None:
        diff = exp_ll - roll_ll
        add("REG-H-012", feature=model, target="", horizon=-1,
            metric="oos_ll_per_obs_expanding_minus_rolling", value=diff,
            verdict="expanding_better" if diff > 0 else "rolling_better",
            details={"expanding": exp, "rolling": roll})
    # REG-H-013: definition drift over 2008-2026
    anchor = finite_or_none(stability.get("max_anchor_bhattacharyya"))
    offline = summary.get("offline") or {}
    min_sep = finite_or_none(offline.get("min_bhattacharyya"))
    if anchor is not None and min_sep is not None:
        add("REG-H-013", feature=model, target="", horizon=-1,
            metric="max_anchor_bhattacharyya", value=anchor, control_value=min_sep,
            verdict="stable_definitions" if anchor <= min_sep else "definitions_drift",
            details={"note": "control: smallest distance between two states of the "
                             "full-sample fit"})
    return entries


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _plots(ctx: RegimeContext, study: RegimeStudy, *, model: str, k: int, frame: pl.DataFrame,
           sub: pl.DataFrame, lab: np.ndarray, std: pl.DataFrame, runs: pl.DataFrame,
           yearly: pl.DataFrame, table: np.ndarray, reversion: pl.DataFrame,
           refits: pl.DataFrame, last_transition: np.ndarray | None) -> None:
    regime = ctx.regime
    directory = study.output_dir / "plots"
    tag = f"{ctx.timeframe} {model} K={k} (causal walk-forward)"
    bars = regime.plots.timeline_bars
    log_price = np.cumsum(np.nan_to_num(sub["log_return"].to_numpy()))
    times = sub["timestamp"].to_numpy()
    for name, start in _timeline_slices(sub, regime.offline.timeline_slices, bars):
        stop = min(start + bars, lab.size)
        plots.plot_state_timeline(times[start:stop], log_price[start:stop], lab[start:stop],
                                  k=k, path=directory / f"state_timeline_{name}.png",
                                  config=regime, title=f"{tag}: states over {name}")
    probs_cols = [f"{model}_p{j}" for j in range(k)]
    if all(c in frame.columns for c in probs_cols):
        tail = frame.tail(bars)
        plots.plot_state_probabilities(tail["timestamp"].to_numpy(),
                                       tail.select(probs_cols).to_numpy(),
                                       path=directory / "state_probabilities.png", config=regime,
                                       title=f"{tag}: P(S_t = k | X_<=t), last {bars} bars")
        entropy = frame[f"{model}_entropy"].to_numpy()
        plots.plot_entropy(entropy, k, path=directory / "state_entropy.png", config=regime,
                           title=f"{tag}: state probability entropy")
    if last_transition is not None:
        plots.plot_transition_matrix(last_transition, path=directory / "transition_matrix.png",
                                     config=regime, title=f"{tag}: transition matrix, last refit")
    if not std.is_empty():
        plots.plot_feature_profiles(std, path=directory / "state_feature_profiles.png",
                                    config=regime, title=f"{tag}: standardised state medians")
    plots.plot_duration_distribution(runs, k=k, path=directory / "state_durations.png",
                                      config=regime, title=f"{tag}: run lengths of the "
                                                           "filtered state")
    if not yearly.is_empty():
        plots.plot_yearly_frequency(yearly, k=k, path=directory / "regime_frequency_by_year.png",
                                    config=regime, title=f"{tag}: share of bars per state by year")
    if table.size:
        plots.plot_regime_vs_volatility(table, path=directory / "regime_vs_volatility.png",
                                        config=regime,
                                        title=f"{tag}: state vs causal volatility bucket")
    if not reversion.is_empty():
        plots.plot_state_outcomes(reversion, horizon=10, path=directory / "state_outcomes.png",
                                  config=regime, title=f"{tag}: residual shrinks within 10 bars "
                                                       "after |Z| > 2")
    plots.plot_refit_stability(refits, path=directory / "state_stability_across_refits.png",
                               config=regime, title=f"{tag}: stability across refits")


def _timeline_slices(sub: pl.DataFrame, count: int, bars: int) -> list[tuple[str, int]]:
    """Deterministic slices: per equal-length era, the window with median volatility."""
    n = sub.height
    if n <= bars:
        return [("all", 0)]
    vol = sub["log_rv_20"].to_numpy()
    out = []
    edges = np.linspace(0, n, count + 1).astype(int)
    for i in range(count):
        lo, hi = edges[i], edges[i + 1] - bars
        if hi <= lo:
            continue
        starts = np.arange(lo, hi, max(1, bars // 2))
        levels = np.array([np.nanmedian(vol[s:s + bars]) for s in starts])
        target = np.nanmedian(levels)
        start = int(starts[int(np.nanargmin(np.abs(levels - target)))])
        out.append((str(sub["timestamp"][start])[:10], start))
    return out


# ---------------------------------------------------------------------------
# A timeframe
# ---------------------------------------------------------------------------
def generate_regime_timeframe(config: Config, regression: RegressionConfig, research: ResearchConfig,
                              ou: OUConfig, spectral: SpectralConfig, wavelet: WaveletConfig,
                              regime: RegimeConfig, *, timeframe: str, models: list[str],
                              states: list[int], schemes: list[tuple[str, str]],
                              stages: tuple[str, ...] = ("offline", "walk_forward", "nulls",
                                                         "analysis"),
                              extra_schemes: dict[tuple[str, int], list[tuple[str, str]]]
                              | None = None, resume: bool = True, make_plots: bool = True,
                              progress: Callable[[str], None] | None = None
                              ) -> list[RegimeStudy]:
    """Every requested stage for one timeframe; each stage reuses current artefacts."""
    ctx = prepare_timeframe(config, regression, ou, spectral, wavelet, research, regime,
                            timeframe, progress=progress)
    offline_rows: dict[tuple[str, int], dict[str, Any]] = {}
    if "offline" in stages and regime.offline.enabled:
        table = run_offline(ctx, models=models, states=states, resume=resume,
                            make_plots=make_plots)
        for r in table.iter_rows(named=True):
            offline_rows[(r["model"], int(r["states"]))] = r
    nulls = None
    if "nulls" in stages and regime.nulls.enabled and "hmm" in models:
        null_states = regime.nulls.run_states or regime.nulls.states
        nulls = run_nulls(ctx, states=[s for s in states if s in null_states], resume=resume)
    studies = []
    for model in models:
        for k in states:
            results: dict[str, WalkForwardResult | None] = {}
            if "walk_forward" in stages or "analysis" in stages:
                primary_pair = (regime.causal.primary_scheme, regime.causal.primary_refit)
                plan = [pair for pair in schemes if pair == primary_pair
                        or model in regime.causal.secondary_scheme_models]
                plan += list((extra_schemes or {}).get((model, k), []))
                for scheme, refit in plan:
                    results[f"{scheme}_{refit}"] = walk_forward_stage(
                        ctx, model=model, k=k, scheme=scheme, refit=refit, resume=resume)
                    gc.collect()
            if "analysis" not in stages:
                continue
            primary_key = f"{regime.causal.primary_scheme}_{regime.causal.primary_refit}"
            primary = results.get(primary_key)
            if primary is None:
                LOGGER.warning("[%s] %s K=%d: no primary walk-forward; analysis skipped",
                               timeframe, model, k)
                continue
            if not offline_rows:
                offline_rows = _load_offline_rows(ctx)
            others = {name: r for name, r in results.items() if name != primary_key}
            ctx.log(f"analysis {model} K={k}")
            studies.append(analyse_study(ctx, model=model, k=k, primary=primary, others=others,
                                         nulls=nulls if nulls is not None else
                                         _load_nulls(ctx), offline_row=offline_rows.get(
                                             (model, k)), make_plots=make_plots))
            gc.collect()
    return studies


def _load_offline_rows(ctx: RegimeContext) -> dict[tuple[str, int], dict[str, Any]]:
    path = ctx.regime.timeframe_dir(ctx.timeframe) / "offline" / "model_selection.parquet"
    if not path.exists():
        return {}
    return {(r["model"], int(r["states"])): r for r in pl.read_parquet(path).iter_rows(named=True)}


def _load_nulls(ctx: RegimeContext) -> pl.DataFrame | None:
    path = ctx.regime.timeframe_dir(ctx.timeframe) / "nulls" / "null_controls.parquet"
    return pl.read_parquet(path) if path.exists() else None


# ---------------------------------------------------------------------------
# Cross-timeframe comparison
# ---------------------------------------------------------------------------
def load_regime_studies(regime: RegimeConfig) -> list[RegimeStudy]:
    studies = []
    for path in sorted(regime.results_path.glob("*/*/k*/summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        s = payload.get("summary") or {}
        if "timeframe" not in s:
            continue
        studies.append(RegimeStudy(s["timeframe"], s["model"], int(s["states"]), path.parent,
                                   summary=s, reused=True))
    return studies


def _cross_timeframe(regime: RegimeConfig) -> pl.DataFrame:
    """Agreement of HMM states across timeframes at equivalent (bar-close) timestamps."""
    rows = []
    primary = f"{regime.causal.primary_scheme}_{regime.causal.primary_refit}"
    for a, b in regime.cross_timeframe.pairs:
        for k in regime.models.hmm.states:
            paths = [regime.results_dir(tf, "hmm", k) / "walk_forward" / primary / "per_bar.parquet"
                     for tf in (a, b)]
            if not all(p.exists() for p in paths):
                continue
            frames = []
            for tf, path in zip((a, b), paths, strict=True):
                step = int(bar_seconds(tf) * 1_000_000)
                frames.append(pl.read_parquet(path, columns=["timestamp", "hmm_state"])
                              .with_columns((pl.col("timestamp") + pl.duration(microseconds=step))
                                            .alias("close_time")))
            fine, coarse = frames if bar_seconds(a) < bar_seconds(b) else frames[::-1]
            joined = coarse.select("close_time", pl.col("hmm_state").alias("coarse")).join(
                fine.select("close_time", pl.col("hmm_state").alias("fine")), on="close_time")
            stats = agreement(joined["fine"].to_numpy(), joined["coarse"].to_numpy())
            rows.append({"fine": a if bar_seconds(a) < bar_seconds(b) else b,
                         "coarse": b if bar_seconds(a) < bar_seconds(b) else a,
                         "states": k, **stats})
    return pl.DataFrame(rows, infer_schema_length=None)


def _changepoints(regime: RegimeConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    """PELT on daily median ln RV (from the coarsest timeframe with inputs), vs HMM switching."""
    rows, segments = [], []
    for tf in reversed(regime.timeframes):
        path = regime.inputs_path / f"timeframe={tf}" / "inputs.parquet"
        if not path.exists():
            continue
        frame = pl.read_parquet(path, columns=["timestamp", *regime.changepoints.series])
        daily = (frame.with_columns(pl.col("timestamp").dt.date().alias("day"))
                 .group_by("day").agg([pl.col(c).median() for c in regime.changepoints.series])
                 .sort("day"))
        for column in regime.changepoints.series:
            y = daily[column].cast(pl.Float64).to_numpy()
            ok = np.isfinite(y)
            days = daily["day"].filter(pl.Series(ok))
            points, info = pelt_mean(y[ok], min_size=regime.changepoints.min_segment_days,
                                     penalty_multiplier=regime.changepoints.penalty_multiplier)
            for seg in segments_table(y[ok], points):
                segments.append({"timeframe": tf, "series": column, **seg,
                                 "first_day": str(days[seg["start"]]),
                                 "last_day": str(days[seg["end"] - 1])})
            rows.append({"timeframe": tf, "series": column, "days": int(ok.sum()),
                         "change_points": len(points), **info,
                         "median_segment_days": float(np.median(np.diff([0, *points,
                                                                         int(ok.sum())])))})
        break
    return pl.DataFrame(rows, infer_schema_length=None), pl.DataFrame(segments,
                                                                        infer_schema_length=None)


def write_regime_comparison(regime: RegimeConfig) -> list[Path]:
    """Cross-study tables: model selection, the K rule, agreement, change points, tests."""
    directory = ensure_dir(regime.results_path)
    written: list[Path] = []
    selection = []
    for tf in regime.timeframes:
        path = regime.timeframe_dir(tf) / "offline" / "model_selection.parquet"
        if path.exists():
            selection.append(pl.read_parquet(path))
    studies = load_regime_studies(regime)
    ari = {(s.timeframe, s.model, s.states): (s.summary.get("state_alignment_stability") or {})
           .get("median_overlap_ari") for s in studies}
    if selection:
        table = pl.concat(selection, how="diagonal_relaxed")
        _write_table(table, directory / "model_selection")
        written.append(directory / "model_selection.csv")
        rule_rows, entries = [], []
        for (tf, model), part in table.group_by("timeframe", "model", maintain_order=True):
            if "test_ll_per_obs" not in part.columns or part["test_ll_per_obs"].null_count() \
                    == part.height:
                reason = "not applicable: no likelihood (K-Means)"
                rule_rows.append({"timeframe": tf, "model": model, "defensible_states": None,
                                  "stopped_because": reason,
                                  "margin": _MARGIN_K, "min_overlap_ari": _MIN_OVERLAP_ARI})
                # an explicit row, so a verdict an older version recorded here is replaced
                entries.append({"hypothesis_id": "REG-H-004",
                                "definition": HYPOTHESES["REG-H-004"], "timeframe": tf,
                                "input_series": regime.primary_feature_set, "window": -1,
                                "feature": model, "target": "", "horizon": -1,
                                "metric": "defensible_states", "value": None,
                                "verdict": "not_applicable", "details": {"reason": reason},
                                "dataset_version": None, "study": f"{tf}/{model}/k_rule"})
                continue
            candidates = [{"states": 1,
                           "test_ll_per_obs": part["single_state_test_ll_per_obs"][0]
                           if "single_state_test_ll_per_obs" in part.columns else None}]
            for r in part.iter_rows(named=True):
                candidates.append({"states": r["states"], "test_ll_per_obs": r.get(
                    "test_ll_per_obs"), "degenerate": r.get("degenerate"),
                    "flags": r.get("flags"), "overlap_ari": ari.get((tf, model, r["states"]))})
            verdict = defensible_states(candidates, margin=_MARGIN_K,
                                        min_overlap_ari=_MIN_OVERLAP_ARI)
            rule_rows.append({"timeframe": tf, "model": model, **verdict})
            entries.append({"hypothesis_id": "REG-H-004", "definition": HYPOTHESES["REG-H-004"],
                            "timeframe": tf, "input_series": regime.primary_feature_set,
                            "window": -1, "feature": model, "target": "", "horizon": -1,
                            "metric": "defensible_states",
                            "value": float(verdict["defensible_states"]),
                            "verdict": k_rule_verdict(verdict),
                            "details": verdict, "dataset_version": None,
                            "study": f"{tf}/{model}/k_rule"})
        rules = pl.DataFrame(rule_rows, infer_schema_length=None)
        _write_table(rules, directory / "defensible_states")
        written.append(directory / "defensible_states.csv")
        if regime.ledger.enabled and entries:
            ResearchLedger(regime.ledger_path).upsert(entries)
    # REG-H-002: HMM vs GMM stability
    by_key = {(s.timeframe, s.model, s.states): s.summary for s in studies}
    entries2 = []
    for (tf, model, k), hmm_summary in by_key.items():
        if model != "hmm" or (tf, "gmm", k) not in by_key:
            continue
        g = by_key[(tf, "gmm", k)]
        hs = finite_or_none(_switch_rate(hmm_summary))
        gs = finite_or_none(_switch_rate(g))
        ha = finite_or_none((hmm_summary.get("state_alignment_stability") or {})
                            .get("median_overlap_ari"))
        ga = finite_or_none((g.get("state_alignment_stability") or {}).get("median_overlap_ari"))
        if hs is None or gs is None or ha is None or ga is None:
            continue
        entries2.append({"hypothesis_id": "REG-H-002", "definition": HYPOTHESES["REG-H-002"],
                         "timeframe": tf, "input_series": regime.primary_feature_set,
                         "window": k, "feature": "hmm_vs_gmm", "target": "", "horizon": -1,
                         "metric": "switch_rate_hmm_minus_gmm", "value": hs - gs,
                         "control_value": ha - ga,
                         "verdict": "hmm_more_stable" if hs < gs and ha > ga else
                         "hmm_not_more_stable",
                         "details": {"switch_rate": {"hmm": hs, "gmm": gs},
                                     "overlap_ari": {"hmm": ha, "gmm": ga}},
                         "dataset_version": None, "study": f"{tf}/hmm_vs_gmm/k{k}"})
    cross = _cross_timeframe(regime)
    if not cross.is_empty():
        _write_table(cross, directory / "cross_timeframe_agreement")
        written.append(directory / "cross_timeframe_agreement.csv")
        for r in cross.iter_rows(named=True):
            v = finite_or_none(r.get("cramers_v"))
            entries2.append({"hypothesis_id": "REG-H-010", "definition": HYPOTHESES["REG-H-010"],
                             "timeframe": f"{r['fine']}|{r['coarse']}",
                             "input_series": regime.primary_feature_set, "window": r["states"],
                             "feature": "hmm", "target": "", "horizon": -1,
                             "metric": "cramers_v", "value": v, "control_value": 0.3,
                             "verdict": ("untested" if v is None else "agree_beyond_chance"
                                         if v > 0.3 else "weak_agreement"),
                             "details": r, "dataset_version": None,
                             "study": f"cross_timeframe/{r['fine']}_{r['coarse']}/k{r['states']}"})
    if regime.ledger.enabled and entries2:
        ResearchLedger(regime.ledger_path).upsert(entries2)
    if regime.changepoints.enabled:
        summary, segments = _changepoints(regime)
        cdir = ensure_dir(directory / "changepoints")
        _write_table(summary, cdir / "changepoint_summary")
        _write_table(segments, cdir / "changepoint_segments")
        written.append(cdir / "changepoint_summary.csv")
    flat = []
    for s in studies:
        row = {k: v for k, v in s.summary.items() if not isinstance(v, (dict, list)) and k
               != "caveat"}
        for key in ("state_alignment_stability", "volatility_baseline_improvement",
                    "calibration"):
            for k2, v in (s.summary.get(key) or {}).items():
                if not isinstance(v, (dict, list)):
                    row[f"{key}__{k2}"] = v
        flat.append(row)
    if flat:
        comparison = pl.DataFrame(flat, infer_schema_length=None).sort("timeframe", "model",
                                                                        "states")
        _write_table(comparison, directory / "regime_comparison")
        written.append(directory / "regime_comparison.csv")
    counts = hypothesis_test_counts(ResearchLedger(regime.ledger_path).load(), prefix="REG-")
    if not counts.is_empty():
        counts.write_csv(directory / "hypothesis_test_counts.csv")
        written.append(directory / "hypothesis_test_counts.csv")
    atomic_write_text(directory / "regime_comparison.json",
                      json.dumps(clean_json([s.summary for s in studies]), indent=1,
                                 default=str) + "\n")
    written.append(directory / "regime_comparison.json")
    return written


#: Rough seconds per million bars at K = 4 on the development PC (i7-6700), measured with
#: ``xq regime-research --benchmark`` and scaled linearly in K. Estimates only.
_COST = {
    "offline": {"kmeans": 20.0, "gmm_diag": 80.0, "gmm": 200.0, "hmm": 200.0},
    "walk_forward": {"kmeans": 60.0, "gmm_diag": 250.0, "gmm": 400.0, "hmm": 700.0},
    "analysis": {"kmeans": 30.0, "gmm_diag": 30.0, "gmm": 30.0, "hmm": 40.0},
}
_NULL_COST = 1500.0          # seconds per million span bars, all sources and K


def estimate_seconds(bars: int, model: str, k: int, *, stages: tuple[str, ...],
                     schemes: int) -> float:
    """A rough runtime estimate for --dry-run (never a promise)."""
    millions = bars / 1e6
    seconds = 0.0
    for stage in stages:
        if stage == "nulls":
            seconds += min(bars, 400_000) / 1e6 * _NULL_COST
            continue
        base = _COST.get(stage, {}).get(model, 0.0) * millions * max(k, 2) / 4.0
        seconds += base * (schemes if stage == "walk_forward" else 1)
    return seconds


def benchmark_regimes(config: Config, regression: RegressionConfig, research: ResearchConfig,
                      ou: OUConfig, spectral: SpectralConfig, wavelet: WaveletConfig,
                      regime: RegimeConfig) -> pl.DataFrame:
    """Time one full-sample fit and three walk-forward refits per timeframe (HMM and GMM, K=3)."""
    rows = []
    for timeframe in regime.timeframes:
        ctx = prepare_timeframe(config, regression, ou, spectral, wavelet, research, regime,
                                timeframe)
        x = feature_matrix(ctx.inputs, ctx.features)
        for model in ("gmm", "hmm"):
            t0 = time.perf_counter()
            offline_fit(x, ctx.features, regime, family=model, k=3, seed=regime.seed, n_init=1)
            offline_seconds = time.perf_counter() - t0
            t0 = time.perf_counter()
            result = run_walk_forward(ctx.inputs, regime, timeframe=timeframe, family=model,
                                      states=3, scheme="expanding",
                                      refit=regime.causal.primary_refit, limit_periods=3)
            rows.append({"timeframe": timeframe, "bars": ctx.inputs.height, "model": model,
                         "states": 3, "offline_fit_seconds_one_init": offline_seconds,
                         "walk_forward_3_periods_seconds": time.perf_counter() - t0,
                         "first_refit_seconds": float(result.refits["fit_seconds"][0])
                         if result.refits.height else None})
        del ctx, x
        gc.collect()
    table = pl.DataFrame(rows)
    ensure_dir(regime.results_path)
    table.write_csv(regime.results_path / "benchmark.csv")
    return table


def _switch_rate(summary: dict[str, Any]) -> float | None:
    matrix = summary.get("empirical_transition_matrix")
    freq = summary.get("state_frequencies")
    if not matrix or not freq:
        return None
    m = np.asarray(matrix, dtype=np.float64)
    p = np.asarray([freq[f"state_{j}"] for j in range(m.shape[0])], dtype=np.float64)
    return float((p * (1.0 - np.diag(m))).sum())
