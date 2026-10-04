"""Command-line interface.

    python -m xauusd_quant.cli <command> [options]
    xq <command> [options]                       # after `pip install -e .`

Commands
--------
``info``           show the effective configuration and dataset status
``inspect``        profile the raw file(s) without reading them fully
``detect-schema``  infer source columns and optionally save them to the config
``validate``       run every validation check over the raw input
``convert``        stream raw ticks into partitioned Parquet
``metadata``       compute dataset-level statistics from the Parquet dataset
``build-bars``     build OHLC bars for one or all timeframes
``diagnose``       bar coverage and gap analysis
``query``          ad-hoc range query against ticks or bars
``research-summary`` statistical research report for one or all timeframes
``regression-research`` rolling-regression residual study

Every command reads ``config/data.yaml`` by default; ``--config`` overrides it.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from .data import diagnostics as diag_mod
from .data import inspector as inspect_mod
from .data import metadata as meta_mod
from .data.converter import TickConverter
from .data.loader import DataStore
from .data.resampler import BarResampler, validate_timeframes
from .data.schema import SCHEMA_VERSION, detect_columns
from .utils.clock import utc_now_iso
from .utils.config import Config, ConfigError, load_config
from .utils.logging import get_logger, setup_logging
from .utils.paths import atomic_write_text, human_bytes

LOGGER = get_logger("cli")
console = Console()

RAW_REPORT_NAME = "raw_dataset_report.json"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xq",
        description="XAUUSD quantitative research platform - data pipeline (Prompt #1 scope).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  xq info\n"
            "  xq inspect\n"
            "  xq detect-schema --save\n"
            "  xq validate --max-rows 5000000\n"
            "  xq convert\n"
            "  xq convert --limit-rows 2000000 --no-resume   # quick smoke test\n"
            "  xq metadata\n"
            "  xq build-bars --timeframe 5m\n"
            "  xq build-bars --all\n"
            "  xq diagnose --all\n"
            '  xq query --sql "SELECT * FROM bars_1h LIMIT 5"\n'
        ),
    )
    parser.add_argument("--config", type=Path, default=None, help="path to data.yaml")
    parser.add_argument("--log-level", default=None,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="console log level (default INFO)")
    parser.add_argument("--raw-path", type=str, default=None,
                        help="override raw_data_path for this run")
    parser.add_argument("--version", action="version",
                        version=f"xauusd-quant (schema {SCHEMA_VERSION})")

    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    sub.add_parser("info", help="show the effective configuration and dataset status")

    p_inspect = sub.add_parser("inspect", help="profile the raw file(s) without a full read")
    p_inspect.add_argument("--path", type=Path, default=None, help="file, directory or glob")
    p_inspect.add_argument("--blocks", type=int, default=inspect_mod.DEFAULT_BLOCKS,
                           help="number of sampled blocks (default %(default)s)")
    p_inspect.add_argument("--block-kb", type=int, default=1024,
                           help="KiB per sampled block (default %(default)s)")
    p_inspect.add_argument("--head", type=int, default=5, help="head rows to show")
    p_inspect.add_argument("--tail", type=int, default=5, help="tail rows to show")
    p_inspect.add_argument("--out", type=Path, default=None,
                           help=f"JSON report path (default metadata_path/{RAW_REPORT_NAME})")
    p_inspect.add_argument("--no-report", action="store_true", help="print only, write nothing")

    p_detect = sub.add_parser("detect-schema", help="infer the source column mapping")
    p_detect.add_argument("--path", type=Path, default=None)
    p_detect.add_argument("--save", action="store_true",
                          help="write the detected mapping back into the YAML config")

    p_validate = sub.add_parser("validate", help="run every validation check over the raw input")
    p_validate.add_argument("--path", type=Path, default=None)
    p_validate.add_argument("--max-rows", type=int, default=None,
                            help="stop after N rows (default: whole file)")
    p_validate.add_argument("--out", type=Path, default=None, help="JSON report path")

    p_convert = sub.add_parser(
        "convert", help="convert raw ticks into partitioned Parquet (partition-safe, resumable)"
    )
    p_convert.add_argument("--limit-rows", type=int, default=None,
                           help="smoke test: first N source rows into <processed>.smoke "
                                "(never the canonical dataset)")
    p_convert.add_argument("--no-resume", action="store_true",
                           help="rebuild every partition (transactionally, like --overwrite)")
    p_convert.add_argument("--overwrite", action="store_true",
                           help="build a complete new dataset beside the current one and "
                                "swap it in only when it is complete and validated")
    p_convert.add_argument("--keep-previous", action="store_true",
                           help="after a swap, keep the replaced dataset as a backup")
    p_convert.add_argument("--rebuild-index", action="store_true",
                           help="re-scan the raw source even if the cached index is current")
    p_convert.add_argument("--skip-space-check", action="store_true",
                           help="do not refuse to start when free space looks insufficient")
    p_convert.add_argument("--out", type=Path, default=None, help="override processed_data_path")

    p_index = sub.add_parser(
        "source-index",
        help="index the raw source by partition and diagnose its time ordering",
    )
    p_index.add_argument("--rebuild", action="store_true",
                         help="re-scan even if the cached index is current")
    p_index.add_argument("--max-events", type=int, default=2000,
                         help="backward-jump events explained in the ordering report")

    p_verify = sub.add_parser(
        "verify-dataset",
        help="re-read every tick partition and verify integrity and coverage",
    )
    p_verify.add_argument("--quick", action="store_true",
                          help="skip the content-digest recomputation and derived-column checks")

    sub.add_parser("metadata", help="compute dataset-level statistics")

    p_bars = sub.add_parser("build-bars", help="build OHLC bars")
    group = p_bars.add_mutually_exclusive_group(required=True)
    group.add_argument("--timeframe", "-t", type=str, help="single timeframe, e.g. 5m")
    group.add_argument("--all", action="store_true", help="every configured timeframe")
    p_bars.add_argument("--keep-existing", action="store_true",
                        help="skip timeframes whose bars already match the current "
                             "tick dataset and settings")
    p_bars.add_argument("--allow-incomplete", action="store_true",
                        help="build even if the tick dataset is not complete "
                             "(development only: bars would cover part of history)")

    p_diag = sub.add_parser("diagnose", help="bar coverage and gap analysis")
    diag_group = p_diag.add_mutually_exclusive_group(required=True)
    diag_group.add_argument("--timeframe", "-t", type=str)
    diag_group.add_argument("--all", action="store_true")
    p_diag.add_argument("--out", type=Path, default=None)

    p_research = sub.add_parser(
        "research-summary", help="statistical research report (Prompt #2 scope)"
    )
    research_group = p_research.add_mutually_exclusive_group(required=True)
    research_group.add_argument("--timeframe", "-t", type=str,
                                help="single timeframe, e.g. 5m")
    research_group.add_argument("--all", action="store_true",
                                help="every timeframe in research.yaml")
    p_research.add_argument("--start", type=str, default=None,
                            help="inclusive, e.g. 2024-01-01")
    p_research.add_argument("--end", type=str, default=None,
                            help="exclusive, e.g. 2025-01-01")
    p_research.add_argument("--research-config", type=Path, default=None,
                            help="path to research.yaml")
    p_research.add_argument("--no-plots", action="store_true",
                            help="write tables only, skip figures")

    p_reg = sub.add_parser(
        "regression-research",
        help="rolling-regression residual study (Prompt #3 scope)",
    )
    reg_scope = p_reg.add_mutually_exclusive_group(required=True)
    reg_scope.add_argument("--timeframe", "-t", type=str,
                           help="single timeframe, e.g. 5m")
    reg_scope.add_argument("--all", action="store_true",
                           help="every timeframe x window in regression.yaml")
    p_reg.add_argument("--window", "-w", type=int, default=None,
                       help="single regression window, e.g. 128")
    p_reg.add_argument("--all-windows", action="store_true",
                       help="every window in regression.yaml for this timeframe")
    p_reg.add_argument("--start", type=str, default=None,
                       help="inclusive, e.g. 2024-01-01")
    p_reg.add_argument("--end", type=str, default=None,
                       help="exclusive, e.g. 2025-01-01")
    p_reg.add_argument("--regression-config", type=Path, default=None,
                       help="path to regression.yaml")
    p_reg.add_argument("--no-plots", action="store_true",
                       help="write tables only, skip figures")

    p_ou = sub.add_parser(
        "ou-research",
        help="Ornstein-Uhlenbeck study of the regression residual (Prompt #4 scope)",
    )
    ou_scope = p_ou.add_mutually_exclusive_group(required=True)
    ou_scope.add_argument("--timeframe", "-t", type=str, help="single timeframe, e.g. 5m")
    ou_scope.add_argument("--all", action="store_true",
                          help="every timeframe x regression window x OU window in ou.yaml")
    ou_scope.add_argument("--compare", action="store_true",
                          help="only rebuild the cross-model comparison from studies on disk")
    p_ou.add_argument("--regression-window", "-n", type=int, default=None,
                      help="regression window N whose residual is modelled")
    p_ou.add_argument("--ou-window", "-m", type=int, default=None,
                      help="OU estimation window M (residual observations)")
    p_ou.add_argument("--all-windows", action="store_true",
                      help="every N x M in ou.yaml for this timeframe")
    p_ou.add_argument("--start", type=str, default=None,
                      help="inclusive date filter (results are then labelled a subset)")
    p_ou.add_argument("--end", type=str, default=None, help="exclusive date filter")
    p_ou.add_argument("--write-parameters", action="store_true",
                      help="write parameters.parquet for every pair run, not just the "
                           "representative one")
    p_ou.add_argument("--no-plots", action="store_true", help="tables only")
    p_ou.add_argument("--in-process", action="store_true",
                      help="run every study in this process instead of a fresh child process "
                           "per study (debugging; memory then accumulates over a grid)")
    p_ou.add_argument("--resume", action="store_true",
                      help="reuse a study already on disk when it was computed from the same "
                           "raw file, tick/bar/feature versions and settings; recompute the rest")
    p_ou.add_argument("--ou-config", type=Path, default=None, help="path to ou.yaml")
    p_ou.add_argument("--regression-config", type=Path, default=None,
                      help="path to regression.yaml")
    p_ou.add_argument("--research-config", type=Path, default=None,
                      help="path to research.yaml (sessions)")

    p_spec = sub.add_parser(
        "spectral-research",
        help="Fourier / spectral study of the residual, OU innovation and returns (Prompt #5)",
    )
    spec_scope = p_spec.add_mutually_exclusive_group(required=True)
    spec_scope.add_argument("--timeframe", "-t", type=str, help="single timeframe, e.g. 5m")
    spec_scope.add_argument("--all", action="store_true",
                            help="every timeframe x series x FFT window in spectral.yaml")
    spec_scope.add_argument("--compare", action="store_true",
                            help="only rebuild the cross-study comparison from studies on disk")
    spec_scope.add_argument("--benchmark", action="store_true",
                            help="measure rolling-FFT throughput per window length")
    p_spec.add_argument("--source", "-s", type=str, default=None,
                        help="one input series (regression_residual, ou_innovation, "
                             "log_return, abs_ou_innovation); default: all configured")
    p_spec.add_argument("--fft-window", "-w", type=int, default=None, help="one FFT window N")
    p_spec.add_argument("--all-windows", action="store_true",
                        help="every FFT window in spectral.yaml")
    p_spec.add_argument("--allow-heavy", action="store_true",
                        help="run timeframes with more bars than grid.max_bars_without_override")
    p_spec.add_argument("--resume", action="store_true",
                        help="reuse studies already on disk that are current")
    p_spec.add_argument("--in-process", action="store_true",
                        help="run in this process instead of one child process per timeframe")
    p_spec.add_argument("--no-plots", action="store_true", help="tables only")
    p_spec.add_argument("--write-features", choices=("representative", "all", "none"),
                        default=None, help="override features_output.write")
    p_spec.add_argument("--spectral-config", type=Path, default=None,
                        help="path to spectral.yaml")
    p_spec.add_argument("--ou-config", type=Path, default=None, help="path to ou.yaml")
    p_spec.add_argument("--regression-config", type=Path, default=None,
                        help="path to regression.yaml")
    p_spec.add_argument("--research-config", type=Path, default=None,
                        help="path to research.yaml (sessions)")

    p_wave = sub.add_parser(
        "wavelet-research",
        help="wavelet / time-frequency study: causal MODWT features, offline scalograms "
             "(Prompt #6)",
    )
    wave_scope = p_wave.add_mutually_exclusive_group(required=True)
    wave_scope.add_argument("--timeframe", "-t", type=str, help="single timeframe, e.g. 5m")
    wave_scope.add_argument("--all", action="store_true",
                            help="every timeframe x series x rolling window in wavelet.yaml")
    wave_scope.add_argument("--compare", action="store_true",
                            help="only rebuild the cross-study comparison from studies on disk")
    wave_scope.add_argument("--benchmark", action="store_true",
                            help="measure causal-feature throughput and live update latency")
    wave_scope.add_argument("--controls", action="store_true",
                            help="synthetic positive controls and the boundary (padding) study")
    p_wave.add_argument("--source", "-s", type=str, default=None,
                        help="one input series (regression_residual, ou_innovation, "
                             "log_return, abs_ou_innovation); default: all configured")
    p_wave.add_argument("--window", "-w", type=int, default=None, help="one rolling window N")
    p_wave.add_argument("--all-windows", action="store_true",
                        help="every rolling window in wavelet.yaml")
    p_wave.add_argument("--allow-heavy", action="store_true",
                        help="run timeframes with more bars than grid.max_bars_without_override")
    p_wave.add_argument("--dry-run", action="store_true",
                        help="print the plan with runtime and disk estimates, run nothing")
    p_wave.add_argument("--resume", action="store_true",
                        help="reuse studies already on disk that are current")
    p_wave.add_argument("--in-process", action="store_true",
                        help="run in this process instead of one child process per timeframe")
    p_wave.add_argument("--no-plots", action="store_true", help="tables only")
    p_wave.add_argument("--write-features", choices=("representative", "all", "none"),
                        default=None, help="override features_output.write")
    p_wave.add_argument("--no-fft-feature-check", action="store_true",
                        help="do not require the stored Prompt #5 FFT features to be current "
                             "(development only; FFT comparisons are always recomputed)")
    p_wave.add_argument("--wavelet-config", type=Path, default=None,
                        help="path to wavelet.yaml")
    p_wave.add_argument("--spectral-config", type=Path, default=None,
                        help="path to spectral.yaml")
    p_wave.add_argument("--ou-config", type=Path, default=None, help="path to ou.yaml")
    p_wave.add_argument("--regression-config", type=Path, default=None,
                        help="path to regression.yaml")
    p_wave.add_argument("--research-config", type=Path, default=None,
                        help="path to research.yaml (sessions)")

    p_regime = sub.add_parser(
        "regime-research",
        help="unsupervised regime discovery: K-Means / GMM / HMM, offline and walk-forward "
             "(Prompt #7)",
    )
    regime_scope = p_regime.add_mutually_exclusive_group(required=True)
    regime_scope.add_argument("--timeframe", "-t", type=str, help="single timeframe, e.g. 5m")
    regime_scope.add_argument("--all", action="store_true",
                              help="every timeframe x model x K in regimes.yaml")
    regime_scope.add_argument("--compare", action="store_true",
                              help="only rebuild the cross-timeframe comparison from disk")
    regime_scope.add_argument("--controls", action="store_true",
                              help="synthetic controls: known HMM, one regime, smooth continuum")
    regime_scope.add_argument("--benchmark", action="store_true",
                              help="time one fit per timeframe and estimate the full grid")
    p_regime.add_argument("--model", "-m", type=str, default=None,
                          help="kmeans, gmm, gmm_diag or hmm (default: every enabled model)")
    for flag in ("--states", "--components", "--clusters"):
        p_regime.add_argument(flag, type=int, default=None, dest="states",
                              help="number of states K (default: every configured K)")
    p_regime.add_argument("--stages", type=str, default="offline,walk_forward,nulls,analysis",
                          help="comma-separated subset of offline,walk_forward,nulls,analysis")
    p_regime.add_argument("--schemes", type=str, default=None,
                          help="walk-forward schemes as scheme_refit, e.g. "
                               "expanding_quarterly,rolling_quarterly (default: every "
                               "configured scheme at the primary refit)")
    p_regime.add_argument("--monthly", action="store_true",
                          help="also run monthly expanding refits for causal.monthly_states")
    p_regime.add_argument("--null-increment-states", type=str, default=None,
                          help="K values whose passing increments are re-measured on the "
                               "pipeline nulls, e.g. 2,3 (default: nulls.increment_states)")
    p_regime.add_argument("--allow-heavy", action="store_true",
                          help="run timeframes with more bars than grid.max_bars_without_override")
    p_regime.add_argument("--dry-run", action="store_true",
                          help="print the plan with rough runtime estimates, run nothing")
    p_regime.add_argument("--resume", action="store_true",
                          help="reuse stage artefacts already on disk that are current")
    p_regime.add_argument("--in-process", action="store_true",
                          help="run in this process instead of one child process per timeframe")
    p_regime.add_argument("--no-plots", action="store_true", help="tables only")
    for flag, what in (("--regime-config", "regimes.yaml"), ("--wavelet-config", "wavelet.yaml"),
                       ("--spectral-config", "spectral.yaml"), ("--ou-config", "ou.yaml"),
                       ("--regression-config", "regression.yaml"),
                       ("--research-config", "research.yaml (sessions)")):
        p_regime.add_argument(flag, type=Path, default=None, help=f"path to {what}")

    p_rwf = sub.add_parser(
        "regime-walk-forward",
        help="causal walk-forward regime inference for one (timeframe, model, K) (Prompt #7)",
    )
    p_rwf.add_argument("--timeframe", "-t", type=str, required=True)
    p_rwf.add_argument("--model", "-m", type=str, required=True,
                       help="kmeans, gmm, gmm_diag or hmm")
    for flag in ("--states", "--components", "--clusters"):
        p_rwf.add_argument(flag, type=int, default=None, dest="states", help="K")
    p_rwf.add_argument("--scheme", choices=("expanding", "rolling"), default=None)
    p_rwf.add_argument("--refit", choices=("monthly", "quarterly"), default=None)
    p_rwf.add_argument("--resume", action="store_true",
                       help="reuse the run on disk if it is current")
    for flag, what in (("--regime-config", "regimes.yaml"), ("--wavelet-config", "wavelet.yaml"),
                       ("--spectral-config", "spectral.yaml"), ("--ou-config", "ou.yaml"),
                       ("--regression-config", "regression.yaml"),
                       ("--research-config", "research.yaml (sessions)")):
        p_rwf.add_argument(flag, type=Path, default=None, help=f"path to {what}")

    feature_config_flags = (
        ("--features-config", "features.yaml"), ("--targets-config", "targets.yaml"),
        ("--alpha-config", "alpha_research.yaml"), ("--regime-config", "regimes.yaml"),
        ("--wavelet-config", "wavelet.yaml"), ("--spectral-config", "spectral.yaml"),
        ("--ou-config", "ou.yaml"), ("--regression-config", "regression.yaml"),
        ("--research-config", "research.yaml (sessions)"))
    for name in ("build-feature-factory", "build-feature-matrix"):
        p_ff = sub.add_parser(
            name, help="versioned causal feature matrix + target table per timeframe (Prompt #8)"
            + (" (alias of build-feature-factory)" if name == "build-feature-matrix" else ""))
        ff_scope = p_ff.add_mutually_exclusive_group(required=True)
        ff_scope.add_argument("--timeframe", "-t", type=str, help="single timeframe, e.g. 5m")
        ff_scope.add_argument("--all", action="store_true",
                              help="every timeframe in features.yaml")
        p_ff.add_argument("--live-safe-only", action="store_true", default=True,
                          help="store only live-safe features (the default; non-causal "
                               "constructions are never computed)")
        p_ff.add_argument("--force", action="store_true",
                          help="rebuild even if the stored matrix is current")
        p_ff.add_argument("--dry-run", action="store_true",
                          help="run the integrity gate and print the plan, build nothing")
        for flag, what in feature_config_flags:
            p_ff.add_argument(flag, type=Path, default=None, help=f"path to {what}")

    p_fr = sub.add_parser(
        "feature-research",
        help="predictive feature research: IC, decay, stability, nulls, redundancy, "
             "statuses (Prompt #8)")
    fr_scope = p_fr.add_mutually_exclusive_group(required=True)
    fr_scope.add_argument("--timeframe", "-t", type=str, help="single timeframe, e.g. 5m")
    fr_scope.add_argument("--all", action="store_true",
                          help="every timeframe in features.yaml, one at a time")
    fr_scope.add_argument("--compare", action="store_true",
                          help="only rebuild the cross-timeframe comparison from disk")
    p_fr.add_argument("--stages", type=str, default=None,
                      help="comma-separated subset of factory,quality,ic,conditioning,mi,"
                           "redundancy,deciles,interactions,nulls,robustness,cost,pipeline")
    p_fr.add_argument("--feature", type=str, default=None,
                      help="show the stored evidence of one feature (runs nothing)")
    p_fr.add_argument("--target", type=str, default=None,
                      help="with --feature: one target family, e.g. realized_vol")
    p_fr.add_argument("--horizon", type=int, default=None,
                      help="with --feature: one horizon in bars")
    p_fr.add_argument("--report-only", action="store_true",
                      help="rebuild statuses, ledger and figures from the stage tables on disk")
    p_fr.add_argument("--no-resume", action="store_true",
                      help="recompute every stage even if its stamp is current")
    p_fr.add_argument("--no-ledger", action="store_true",
                      help="do not upsert the tests into the research ledger")
    p_fr.add_argument("--allow-heavy", action="store_true",
                      help="run timeframes below 5m (not configured by default)")
    p_fr.add_argument("--dry-run", action="store_true",
                      help="print the plan with rough runtime estimates, run nothing")
    p_fr.add_argument("--in-process", action="store_true",
                      help="run in this process instead of one child process per timeframe")
    for flag, what in feature_config_flags:
        p_fr.add_argument(flag, type=Path, default=None, help=f"path to {what}")

    p_red = sub.add_parser("feature-redundancy",
                           help="redundancy clusters / representatives of one timeframe "
                                "(Prompt #8; reads the stored research)")
    p_red.add_argument("--timeframe", "-t", type=str, required=True)
    p_red.add_argument("--feature", type=str, default=None,
                       help="show one feature's cluster and nearest features")
    for flag, what in feature_config_flags[:3]:
        p_red.add_argument(flag, type=Path, default=None, help=f"path to {what}")

    p_decay = sub.add_parser("alpha-decay",
                             help="IC decay curves and information half-life of one feature "
                                  "(Prompt #8; reads the stored research)")
    p_decay.add_argument("--timeframe", "-t", type=str, required=True)
    p_decay.add_argument("--feature", type=str, required=True)
    p_decay.add_argument("--target", type=str, default=None,
                         help="one target family (default: every family)")
    for flag, what in feature_config_flags[:3]:
        p_decay.add_argument(flag, type=Path, default=None, help=f"path to {what}")

    selection_flags = (("--selection-config", "feature_selection.yaml"), *feature_config_flags)
    p_fs = sub.add_parser(
        "feature-select",
        help="feature selection + dimensionality-reduction research: filters, clusters, "
             "mRMR, stability, L1 / elastic net, PCA, nested CV, sets, manifests (Prompt #9)")
    fs_scope = p_fs.add_mutually_exclusive_group(required=True)
    fs_scope.add_argument("--timeframe", "-t", type=str, help="single timeframe, e.g. 5m")
    fs_scope.add_argument("--all", action="store_true",
                          help="every timeframe in feature_selection.yaml, one at a time")
    p_fs.add_argument("--target", type=str, default=None,
                      help="show one target family's selection (e.g. residual_reduction, "
                           "future_return, realized_vol, abs_return); every target is "
                           "selected together, so this runs the shared stages if needed")
    p_fs.add_argument("--horizon", type=int, default=None, help="with --target: one horizon")
    p_fs.add_argument("--stages", type=str, default=None,
                      help="comma-separated subset of universe,redundancy,evidence,stability,"
                           "nested,sets,audit")
    p_fs.add_argument("--no-resume", action="store_true",
                      help="recompute every stage even if its stamp is current")
    p_fs.add_argument("--no-ledger", action="store_true",
                      help="do not upsert the selection experiments into the research ledger")
    p_fs.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    p_fs.add_argument("--in-process", action="store_true",
                      help="run in this process instead of one child process per timeframe")
    for flag, what in selection_flags:
        p_fs.add_argument(flag, type=Path, default=None, help=f"path to {what}")

    p_fsr = sub.add_parser("feature-selection-report",
                           help="summary, ledger rows and figures from the stored selection "
                                "(Prompt #9; runs no stage)")
    fsr_scope = p_fsr.add_mutually_exclusive_group(required=True)
    fsr_scope.add_argument("--timeframe", "-t", type=str)
    fsr_scope.add_argument("--all", action="store_true")
    fsr_scope.add_argument("--compare", action="store_true",
                           help="only the cross-timeframe comparison")
    p_fsr.add_argument("--no-ledger", action="store_true")
    p_fsr.add_argument("--no-plots", action="store_true")
    for flag, what in selection_flags[:1]:
        p_fsr.add_argument(flag, type=Path, default=None, help=f"path to {what}")

    p_bfm = sub.add_parser("build-feature-manifest",
                           help="write / verify the immutable manifest of a selected set and "
                                "check it reproduces the stored matrix order (Prompt #9)")
    p_bfm.add_argument("--set", type=str, required=True, dest="set_name",
                       help="minimal | standard | extended | general | target_<kind>")
    p_bfm.add_argument("--timeframe", "-t", type=str, default=None,
                       help="default: the first timeframe in feature_selection.yaml")
    p_bfm.add_argument("--output", type=Path, default=None,
                       help="also copy the verified manifest to this path")
    for flag, what in selection_flags[:2]:
        p_bfm.add_argument(flag, type=Path, default=None, help=f"path to {what}")

    ml_flags = (("--ml-config", "ml.yaml"), *selection_flags)
    p_mlr = sub.add_parser(
        "ml-research",
        help="supervised predictive-model research on the development period: baselines, "
             "linear models, RF / XGBoost / LightGBM / CatBoost, calibration, feature sets, "
             "searches, windows, decay, retraining, ablation, nulls (Prompt #10)")
    mlr_scope = p_mlr.add_mutually_exclusive_group(required=True)
    mlr_scope.add_argument("--timeframe", "-t", type=str, help="single timeframe, e.g. 5m")
    mlr_scope.add_argument("--all", action="store_true",
                           help="every timeframe in ml.yaml, one at a time")
    p_mlr.add_argument("--stages", type=str, default=None,
                       help="comma-separated subset of grid,trees,feature_sets,variants,search,"
                            "windows,learning_curve,decay,retraining,ablation,nulls,"
                            "pipeline_null,pipeline_null_posthoc")
    p_mlr.add_argument("--report", action="store_true",
                       help="build the tables, ledger rows and figures after the stages")
    p_mlr.add_argument("--no-ledger", action="store_true")
    p_mlr.add_argument("--dry-run", action="store_true",
                       help="print the unit plan (and how many are current), run nothing")
    p_mlr.add_argument("--in-process", action="store_true",
                       help="run in this process instead of one child process per timeframe")
    for flag, what in ml_flags:
        p_mlr.add_argument(flag, type=Path, default=None, help=f"path to {what}")

    p_mlt = sub.add_parser(
        "ml-train",
        help="walk-forward development training of one model configuration (Prompt #10; "
             "writes the same cached units as ml-research, never reads the reserved period)")
    p_mlt.add_argument("--timeframe", "-t", type=str, required=True)
    p_mlt.add_argument("--target", type=str, required=True,
                       help="a target of ml.yaml, e.g. mean_reversion, future_volatility")
    p_mlt.add_argument("--horizon", type=int, required=True)
    p_mlt.add_argument("--model", type=str, required=True,
                       help="constant | logistic_l2 | logistic_l1 | logistic_en | ols | ridge | "
                            "elastic_net | random_forest | xgboost | lightgbm | catboost")
    p_mlt.add_argument("--feature-set", type=str, default="standard",
                       help="standard | extended | minimal | target")
    p_mlt.add_argument("--folds", type=str, default=None,
                       help="comma-separated walk-forward blocks, 1-based (default: every block)")
    for flag, what in ml_flags:
        p_mlt.add_argument(flag, type=Path, default=None, help=f"path to {what}")

    p_mlw = sub.add_parser(
        "ml-walk-forward",
        help="every enabled model family on one target x horizon, walk-forward, with the "
             "comparison against the constant baseline and the reference linear model")
    p_mlw.add_argument("--timeframe", "-t", type=str, required=True)
    p_mlw.add_argument("--target", type=str, required=True)
    p_mlw.add_argument("--horizon", type=int, required=True)
    p_mlw.add_argument("--models", type=str, default=None,
                       help="comma-separated families (default: every enabled family)")
    p_mlw.add_argument("--feature-set", type=str, default="standard")
    for flag, what in ml_flags:
        p_mlw.add_argument(flag, type=Path, default=None, help=f"path to {what}")

    p_mlc = sub.add_parser(
        "ml-compare",
        help="compare the stored walk-forward results of one target (reads units; trains "
             "nothing)")
    p_mlc.add_argument("--timeframe", "-t", type=str, required=True)
    p_mlc.add_argument("--target", type=str, required=True)
    p_mlc.add_argument("--horizon", type=int, default=None)
    p_mlc.add_argument("--variants", action="store_true",
                       help="also list the research variants (windows, weights, searches, "
                            "nulls), not only the base configurations")
    p_mlc.add_argument("--ml-config", type=Path, default=None, help="path to ml.yaml")

    p_mlrep = sub.add_parser(
        "ml-report",
        help="tables, ML-H ledger rows and figures from the stored units (Prompt #10)")
    p_mlrep.add_argument("--timeframe", "-t", type=str, required=True)
    p_mlrep.add_argument("--no-ledger", action="store_true")
    p_mlrep.add_argument("--no-plots", action="store_true")
    p_mlrep.add_argument("--light", action="store_true",
                         help="skip the pooled-prediction, SHAP, shift and failure tables")
    for flag, what in ml_flags:
        p_mlrep.add_argument(flag, type=Path, default=None, help=f"path to {what}")

    p_mlf = sub.add_parser(
        "ml-freeze",
        help="apply the pre-registered freeze rules to development results and write the "
             "immutable MODEL_SPEC files (MODEL_SPEC_FROZEN)")
    p_mlf.add_argument("--timeframe", "-t", type=str, required=True)
    for flag, what in ml_flags:
        p_mlf.add_argument(flag, type=Path, default=None, help=f"path to {what}")

    p_mlfin = sub.add_parser(
        "ml-finalize",
        help="fit every frozen spec on the whole development span; artifacts, reload "
             "equality, live compatibility, streaming equality and latency")
    p_mlfin.add_argument("--timeframe", "-t", type=str, required=True)
    p_mlfin.add_argument("--no-streaming", action="store_true")
    for flag, what in ml_flags:
        p_mlfin.add_argument(flag, type=Path, default=None, help=f"path to {what}")

    p_mlft = sub.add_parser(
        "ml-final-test",
        help="ONE-TIME evaluation of explicitly frozen model specs on the reserved test "
             "period (refuses anything not frozen and hash-verified)")
    p_mlft.add_argument("--model-spec", type=str, nargs="+", required=True,
                        help="MODEL_SPEC ids (e.g. LGBM_REVERSION_5M_H5_V001) or spec paths")
    p_mlft.add_argument("--repeat-reason", type=str, default=None,
                        help="required to evaluate a spec a second time; logged")
    for flag, what in ml_flags:
        p_mlft.add_argument(flag, type=Path, default=None, help=f"path to {what}")

    ens_flags = (("--ensemble-config", "ensemble.yaml"), *ml_flags)
    p_er = sub.add_parser(
        "ensemble-research",
        help="ensembles, meta-models and predictive diversification on the Prompt #10 "
             "out-of-sample predictions: eligibility gate, meta walk-forward, diversity, "
             "stacking, conditional / dynamic weights, controls (Prompt #11)")
    er_scope = p_er.add_mutually_exclusive_group(required=True)
    er_scope.add_argument("--timeframe", "-t", type=str, help="single timeframe, e.g. 5m")
    er_scope.add_argument("--all", action="store_true",
                          help="every timeframe in ensemble.yaml, one at a time")
    p_er.add_argument("--target", type=str, default=None,
                      help="one target of ensemble.yaml (default: every pair)")
    p_er.add_argument("--horizon", type=int, default=None)
    p_er.add_argument("--no-pipeline-nulls", action="store_true",
                      help="skip the random-walk / sign-flip null labels (faster smoke runs)")
    p_er.add_argument("--report", action="store_true",
                      help="also build the timeframe tables, ENS-H ledger rows and figures")
    p_er.add_argument("--no-ledger", action="store_true")
    p_er.add_argument("--in-process", action="store_true",
                      help="run in this process instead of one child process per timeframe")
    for flag, what in ens_flags:
        p_er.add_argument(flag, type=Path, default=None, help=f"path to {what}")

    p_erep = sub.add_parser(
        "ensemble-report",
        help="cross-pair tables, ENS-H ledger rows, figures and the joint predictive state "
             "of one timeframe (from the stored ensemble research)")
    p_erep.add_argument("--timeframe", "-t", type=str, required=True)
    p_erep.add_argument("--no-ledger", action="store_true")
    p_erep.add_argument("--no-plots", action="store_true")
    for flag, what in ens_flags:
        p_erep.add_argument(flag, type=Path, default=None, help=f"path to {what}")

    p_eb = sub.add_parser(
        "ensemble-build",
        help="score one explicit combination on the meta walk-forward (research; freezes "
             "nothing): --method simple_average --models a b c, or --method stacking")
    p_eb.add_argument("--timeframe", "-t", type=str, default="5m")
    p_eb.add_argument("--target", type=str, required=True)
    p_eb.add_argument("--horizon", type=int, required=True)
    p_eb.add_argument("--method", type=str, required=True,
                      help="simple_average | median | performance_weighted | "
                           "diversity_weighted | stacking")
    p_eb.add_argument("--models", type=str, nargs="*", default=None,
                      help="constituents as '<family>|<set>' (default: the eligible universe)")
    for flag, what in ens_flags:
        p_eb.add_argument(flag, type=Path, default=None, help=f"path to {what}")

    p_ef = sub.add_parser(
        "ensemble-freeze",
        help="apply the pre-registered freeze rule to the development results and write the "
             "immutable ENSEMBLE_SPEC and constituent MODEL_SPEC files")
    p_ef.add_argument("--timeframe", "-t", type=str, required=True)
    for flag, what in ens_flags:
        p_ef.add_argument(flag, type=Path, default=None, help=f"path to {what}")

    p_efin = sub.add_parser(
        "ensemble-finalize",
        help="fit every frozen constituent on the development span; reload, version and "
             "feature checks, streaming = batch, latency / memory / size benchmark")
    p_efin.add_argument("--timeframe", "-t", type=str, required=True)
    p_efin.add_argument("--no-streaming", action="store_true")
    for flag, what in ens_flags:
        p_efin.add_argument(flag, type=Path, default=None, help=f"path to {what}")

    p_eft = sub.add_parser(
        "ensemble-final-test",
        help="ONE-TIME evaluation of frozen ensemble specs on the reserved period (a second "
             "look after Prompt #10; refuses anything not frozen and hash-verified)")
    p_eft.add_argument("--ensemble-spec", type=str, nargs="+", required=True,
                       help="ENSEMBLE_SPEC ids (e.g. ENS_REVERSION_5M_H5_V001) or spec paths")
    p_eft.add_argument("--repeat-reason", type=str, default=None,
                       help="required to evaluate a spec a second time; logged")
    for flag, what in ens_flags:
        p_eft.add_argument(flag, type=Path, default=None, help=f"path to {what}")

    for command in ("execution-smoke", "execution-readiness",
                    "execution-diagnostic", "execution-backtest"):
        p_exec = sub.add_parser(command, help="Stage 12 offline execution / evidence")
        p_exec.add_argument("--execution-config", type=Path, default=None)
        p_exec.add_argument("--run-id", required=True, help="new immutable id ending _V001")
        if command in ("execution-diagnostic", "execution-backtest"):
            p_exec.add_argument("--start", required=True, help="inclusive UTC ISO timestamp")
            p_exec.add_argument("--end", required=True, help="exclusive UTC ISO timestamp")
            p_exec.add_argument("--forecasts", type=Path)
            p_exec.add_argument("--forecast-metadata", type=Path)
            p_exec.add_argument("--historical-diagnostic", action="store_true")
            p_exec.add_argument("--quote-probe", action="store_true",
                                help="bounded resource probe; no forecast economics")
            p_exec.add_argument("--max-quotes", type=int, default=50_000)
            p_exec.add_argument("--scenario-set", type=Path)

    for command in ("strategy-plan", "strategy-readiness", "strategy-smoke", "strategy-validate"):
        p_strategy = sub.add_parser(command, help="Stage 13 offline chronological economic validation")
        p_strategy.add_argument("--plan", type=Path, default=None)
        p_strategy.add_argument("--execution-config", type=Path, default=None)
        p_strategy.add_argument("--evidence", type=Path, default=None)
        if command != "strategy-plan":
            p_strategy.add_argument("--run-id", required=True)
        if command == "strategy-validate":
            p_strategy.add_argument("--historical-diagnostic", action="store_true",
                                    help="record unresolved gates; diagnostics cannot be promoted")

    for command in ("econometric-plan", "econometric-readiness", "econometric-smoke", "econometric-evaluate"):
        p_econ = sub.add_parser(command, help="Stage 13.5 bounded offline econometric research")
        p_econ.add_argument("--plan", type=Path, default=None)
        p_econ.add_argument("--execution-config", type=Path, default=None)
        if command != "econometric-plan":
            p_econ.add_argument("--run-id", required=True)

    for command in ("robustness-plan", "robustness-inventory", "robustness-smoke", "robustness-evaluate"):
        p_robust = sub.add_parser(command, help="Stage 14 bounded offline robustness")
        p_robust.add_argument("--plan", type=Path, default=None)
        if command != "robustness-plan":
            p_robust.add_argument("--run-id", required=True)

    for command in ("portfolio-plan", "alpha-registry", "portfolio-smoke", "portfolio-evaluate"):
        p_portfolio = sub.add_parser(command, help="Stage 15 bounded offline alpha portfolio")
        p_portfolio.add_argument("--plan", type=Path, default=None)
        if command in ("portfolio-smoke", "portfolio-evaluate"):
            p_portfolio.add_argument("--run-id", required=True)

    for command in ("risk-plan", "risk-readiness", "risk-replay"):
        p_risk = sub.add_parser(command, help="Stage16 offline authoritative risk validation")
        p_risk.add_argument("--risk-config", type=Path, default=None)
        if command != "risk-plan":
            p_risk.add_argument("--run-id", required=True)

    for command in ("shadow-validate", "shadow-preflight", "shadow-capture", "shadow-run",
                    "shadow-replay", "shadow-compare"):
        p_shadow = sub.add_parser(command, help="Stage17 read-only demo feed/shadow diagnostics")
        p_shadow.add_argument("--shadow-config", type=Path, default=None)
        p_shadow.add_argument("--run-id", required=command != "shadow-validate")
        if command in ("shadow-capture", "shadow-run"):
            p_shadow.add_argument("--resume", type=Path, default=None)
        if command in ("shadow-replay", "shadow-compare"):
            p_shadow.add_argument("--recorded", type=Path, required=True)
        if command == "shadow-compare":
            p_shadow.add_argument("--replayed", type=Path, required=True)

    p_feat = sub.add_parser(
        "build-features",
        help="cache versioned rolling-regression features for full-history bars",
    )
    feat_scope = p_feat.add_mutually_exclusive_group(required=True)
    feat_scope.add_argument("--timeframe", "-t", type=str, help="single timeframe")
    feat_scope.add_argument("--all", action="store_true",
                            help="every timeframe x window in regression.yaml")
    p_feat.add_argument("--window", "-w", type=int, default=None,
                        help="single regression window (default: every configured window)")
    p_feat.add_argument("--force", action="store_true", help="recompute even if current")
    p_feat.add_argument("--regression-config", type=Path, default=None,
                        help="path to regression.yaml")

    p_query = sub.add_parser("query", help="ad-hoc range query")
    p_query.add_argument("--sql", type=str, default=None,
                         help="raw DuckDB SQL against views `ticks` / `bars_<tf>`")
    p_query.add_argument("--timeframe", "-t", type=str, default=None,
                         help="query bars of this timeframe instead of ticks")
    p_query.add_argument("--start", type=str, default=None, help="inclusive, e.g. 2024-05-01")
    p_query.add_argument("--end", type=str, default=None, help="exclusive, e.g. 2024-05-02")
    p_query.add_argument("--limit", type=int, default=20)
    p_query.add_argument("--columns", type=str, default=None, help="comma-separated")
    return parser


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_info(config: Config, _args: argparse.Namespace) -> int:
    store = DataStore(config)
    sources = inspect_mod.resolve_sources(config.raw_data_path)
    table = Table(title=f"{config.instrument} - configuration", show_header=False, box=None)
    table.add_column("key", style="bold cyan", no_wrap=True)
    table.add_column("value")
    rows = [
        ("config file", str(config.config_path)),
        ("config fingerprint", config.fingerprint()),
        ("schema version", SCHEMA_VERSION),
        ("raw_data_path", str(config.raw_data_path)),
        ("raw files found", f"{len(sources)} ({human_bytes(sum(s.stat().st_size for s in sources))})"
         if sources else "[red]none[/red]"),
        ("processed_data_path", str(config.processed_data_path)),
        ("bars_path", str(config.bars_path)),
        ("metadata_path", str(config.metadata_path)),
        ("source columns", ", ".join(f"{k}={v}" for k, v in config.input.source_columns.items())),
        ("timestamp format", str(config.input.timestamp_format)),
        ("timezone", f"{config.timezone.mode}: {config.timezone.describe()}"),
        ("emit timestamp_utc", str(config.timezone.emit_utc_column)),
        ("drop rules enabled", ", ".join(config.cleaning.enabled_drops()) or "none"),
        ("timeframes", ", ".join(config.resampling.timeframes)),
        ("bar convention", f"label={config.resampling.label}, closed={config.resampling.closed}, "
                           f"price={config.resampling.price_source}"),
    ]
    for key, value in rows:
        table.add_row(key, value)
    console.print(table)

    status = Table(title="dataset status", show_header=True, box=None)
    status.add_column("dataset", style="bold")
    status.add_column("state")
    if store.has_ticks():
        lo, hi = store.tick_span()
        size = human_bytes(sum(f.stat().st_size for f in config.processed_data_path.rglob("*.parquet")))
        status.add_row("ticks (parquet)", f"{lo} -> {hi}  ({size})")
    else:
        status.add_row("ticks (parquet)", "[yellow]not built - run `xq convert`[/yellow]")
    available = store.available_timeframes()
    status.add_row("bars", ", ".join(available) if available
                   else "[yellow]not built - run `xq build-bars --all`[/yellow]")
    store.close()
    console.print(status)
    return 0


def cmd_inspect(config: Config, args: argparse.Namespace) -> int:
    target = Path(args.path) if args.path else config.raw_data_path
    sources = inspect_mod.resolve_sources(target)
    if not sources:
        console.print(f"[red]No files found at[/red] {target}")
        return 2

    report = inspect_mod.inspect_sources(
        sources,
        blocks=args.blocks,
        block_bytes=args.block_kb * 1024,
        head=args.head,
        tail=args.tail,
        timestamp_format=config.input.timestamp_format,
        large_gap_seconds=config.validation.large_gap_seconds,
    )
    for entry in report["files"]:
        _print_inspection(entry)

    if not args.no_report:
        out = args.out or (config.metadata_path / RAW_REPORT_NAME)
        inspect_mod.write_report(report, out)
        console.print(f"\n[green]Report written:[/green] {out}")
    return 0


def _print_inspection(entry: dict[str, Any]) -> None:
    console.rule(f"[bold]{Path(entry['path']).name}")
    dialect = entry.get("dialect", {})
    facts = Table(show_header=False, box=None)
    facts.add_column("k", style="cyan", no_wrap=True)
    facts.add_column("v")
    for key, value in [
        ("path", entry["path"]),
        ("size", f"{entry['size_human']}  ({entry['size_bytes']:,} bytes)"),
        ("modified (UTC)", entry.get("modified_utc")),
        ("delimiter / encoding", f"{dialect.get('delimiter')!r} / {dialect.get('encoding')}"
                                 f"  BOM={dialect.get('has_bom')}"),
        ("line terminator", repr(dialect.get("line_terminator"))),
        ("header", ", ".join(entry.get("header", [])) or "(none)"),
        ("sampled", f"{entry['sampled_rows']:,} rows in {entry['sampled_blocks']} blocks"),
        ("mean bytes/line", f"{entry['mean_bytes_per_line']:.2f}"),
        ("estimated rows", f"~{entry['estimated_total_rows']:,}"
                           if entry.get("estimated_total_rows") else "n/a"),
        ("first timestamp", entry.get("first_timestamp")),
        ("last timestamp", entry.get("last_timestamp")),
        ("detected format", (entry.get("detection") or {}).get("timestamp_format")),
    ]:
        facts.add_row(key, str(value))
    console.print(facts)

    columns = Table(title="columns (sampled)", box=None)
    for name in ("name", "inferred dtype", "nulls", "null rate", "min", "max", "examples"):
        columns.add_column(name)
    for col in entry.get("columns", []):
        columns.add_row(
            col["name"], col["inferred_dtype"], f"{col['nulls']:,}", f"{col['null_rate']:.4%}",
            _fmt(col.get("min_value")), _fmt(col.get("max_value")),
            ", ".join(col.get("examples", [])[:2]),
        )
    console.print(columns)

    mapping = (entry.get("detection") or {}).get("mapping", {})
    console.print(f"[bold]detected mapping:[/bold] {mapping or '(none)'}")

    integrity = Table(title="integrity (sampled)", box=None, show_header=False)
    integrity.add_column("k", style="cyan")
    integrity.add_column("v")
    for key, value in [
        ("field-count mismatches", entry["field_count_mismatches"]),
        ("unparseable numeric cells", entry["unparseable_numeric_cells"]),
        ("duplicate rows", entry["sampled_duplicate_rows"]),
        ("duplicate timestamps", entry["sampled_duplicate_timestamps"]),
        ("non-monotonic timestamps", entry["sampled_non_monotonic"]),
        ("ask < bid", entry["sampled_ask_below_bid"]),
    ]:
        integrity.add_row(key, f"{value:,}")
    console.print(integrity)

    if spread := entry.get("spread_summary"):
        console.print("[bold]spread (sampled):[/bold] " + "  ".join(
            f"{k}={v:.4f}" for k, v in spread.items() if k != "sampled_count"
        ))
    if head := entry.get("head_lines"):
        console.print("[bold]head:[/bold]\n  " + "\n  ".join(head))
    if tail := entry.get("tail_lines"):
        console.print("[bold]tail:[/bold]\n  " + "\n  ".join(tail))
    if gaps := entry.get("observed_gaps"):
        console.print(f"[bold]large gaps seen inside sampled blocks:[/bold] {len(gaps)} "
                      f"(largest {max(g['gap_seconds'] for g in gaps) / 3600:.2f} h)")
    for warning in entry.get("warnings", []):
        console.print(f"[yellow]warning:[/yellow] {warning}")
    for note in entry.get("notes", []):
        console.print(f"[dim]note: {note}[/dim]")


def _fmt(value: Any) -> str:
    return "-" if value is None else (f"{value:,.6g}" if isinstance(value, float) else str(value))


def cmd_detect_schema(config: Config, args: argparse.Namespace) -> int:
    target = Path(args.path) if args.path else config.raw_data_path
    sources = inspect_mod.resolve_sources(target)
    if not sources:
        console.print(f"[red]No files found at[/red] {target}")
        return 2

    source = sources[0]
    dialect = inspect_mod.sniff_dialect(source)
    lines = inspect_mod.read_head_lines(source, 6, encoding=dialect.encoding)
    if not lines:
        console.print(f"[red]{source} is empty[/red]")
        return 2

    header = [h.strip() for h in lines[0].split(dialect.delimiter)]
    body = lines[1:] if dialect.has_header else lines
    detection = detect_columns(header, [r.split(dialect.delimiter)[0].strip() for r in body])

    table = Table(title=f"detected schema - {source.name}", box=None)
    for name in ("canonical", "source column", "index", "matched by"):
        table.add_column(name)
    for column in detection.columns:
        table.add_row(column.canonical, column.source, str(column.index), column.matched_pattern)
    console.print(table)
    console.print(f"[bold]timestamp format:[/bold] {detection.timestamp_format!r} "
                  f"(lexicographic: {detection.timestamp_lexicographic})")
    console.print(f"[bold]samples:[/bold] {list(detection.timestamp_samples)}")
    if detection.unmapped_headers:
        console.print(f"[yellow]unmapped headers:[/yellow] {list(detection.unmapped_headers)}")
    if missing := detection.missing():
        console.print(f"[red]missing required fields:[/red] {list(missing)}")

    fragment = detection.to_config_fragment()
    console.print("\n[bold]config fragment:[/bold]")
    console.print_json(json.dumps(fragment))

    if args.save:
        if not detection.is_usable:
            console.print("[red]Refusing to save: detection did not find timestamp + bid + ask.")
            return 2
        written = _save_schema_fragment(config, fragment)
        console.print(f"[green]Saved detected mapping to[/green] {written}")
    else:
        console.print("[dim]Re-run with --save to write this into the YAML config.[/dim]")
    return 0


def _save_schema_fragment(config: Config, fragment: dict[str, Any]) -> Path:
    """Rewrite only the mapping keys in the YAML, preserving comments elsewhere."""
    path = config.config_path
    if path is None:
        raise ConfigError("No config file path is known; cannot save.")
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    for key, value in fragment.items():
        rendered = "null" if value is None else (
            str(value) if isinstance(value, bool) else f'"{value}"'
        )
        replacement = f"{key}: {rendered}"
        for i, line in enumerate(lines):
            if line.split(":", 1)[0].strip() == key and not line.startswith((" ", "\t", "#")):
                comment = line.split("#", 1)
                suffix = f"  #{comment[1]}" if len(comment) > 1 else ""
                lines[i] = replacement + suffix
                break
        else:
            lines.append(replacement)
    backup = path.with_suffix(path.suffix + ".bak")
    backup.write_text(text, encoding="utf-8")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def cmd_validate(config: Config, args: argparse.Namespace) -> int:
    """Validate the raw input without writing any Parquet."""
    from .data.validator import StreamValidator

    target = Path(args.path) if args.path else config.raw_data_path
    sources = inspect_mod.resolve_sources(target)
    if not sources:
        console.print(f"[red]No files found at[/red] {target}")
        return 2

    converter = TickConverter(config)
    validator = StreamValidator(config)
    rows = 0
    with console.status("validating...") as status:
        for source in sources:
            for canonical in converter.iter_canonical_batches(source):
                validator.validate(canonical)
                rows += canonical.height
                status.update(f"validating... {rows:,} rows")
                if args.max_rows and rows >= args.max_rows:
                    break
            if args.max_rows and rows >= args.max_rows:
                break
    # Malformed rows are rejected before they reach a batch, so they are
    # counted by the converter's reader rather than by our validator.
    validator.note_malformed(converter.malformed_rows)

    report = validator.report()
    full_pass = not args.max_rows or rows < args.max_rows
    payload = report.to_dict()
    payload["complete_pass"] = full_pass
    payload["source_files"] = [str(s) for s in sources]
    payload["config_fingerprint"] = config.fingerprint()

    out = args.out or (config.metadata_path / "validation_report_standalone.json")
    atomic_write_text(out, json.dumps(payload, indent=2, default=str) + "\n")

    table = Table(title="validation", box=None)
    table.add_column("check", style="cyan")
    table.add_column("count", justify="right")
    table.add_row("rows checked", f"{report.rows_checked:,}")
    table.add_row("malformed rows", f"{report.malformed_rows:,}")
    for check, count in report.counts.items():
        table.add_row(check, f"{count:,}")
    table.add_row("large gaps", f"{report.gap_count:,}")
    table.add_row("largest gap (h)", f"{report.largest_gap_seconds / 3600:.2f}")
    console.print(table)
    console.print(f"span: {report.first_timestamp} -> {report.last_timestamp}")
    if not full_pass:
        console.print("[yellow]Partial pass (--max-rows): counts describe the sampled prefix "
                      "only, not the whole file.[/yellow]")
    console.print(f"[green]Report written:[/green] {out}")
    return 0


def cmd_convert(config: Config, args: argparse.Namespace) -> int:
    # `conversion.progress` shows a live counter when attached to a terminal.
    # The periodic log lines are emitted either way, so redirected output still
    # shows progress.
    live = config.conversion.progress and console.is_terminal
    with ExitStack() as stack:
        callback: Callable[[int, int], None] | None = None
        if live:
            status = stack.enter_context(console.status("converting..."))
            started = time.perf_counter()

            def report(rows: int, _batch: int) -> None:
                elapsed = time.perf_counter() - started
                rate = rows / elapsed if elapsed > 0 else 0.0
                status.update(f"converting... {rows:,} rows ({rate:,.0f} rows/s)")

            callback = report

        converter = TickConverter(config, progress_callback=callback)
        result = converter.run(
            limit_rows=args.limit_rows,
            resume=not args.no_resume,
            overwrite=args.overwrite,
            output_dir=args.out,
            keep_previous=True if args.keep_previous else None,
            rebuild_index=args.rebuild_index,
            check_space=not args.skip_space_check,
        )
    table = Table(title="conversion", box=None, show_header=False)
    table.add_column("k", style="cyan")
    table.add_column("v")
    complete = result.status == "complete" and not result.truncated_by_limit
    for key, value in [
        ("mode / status", f"{result.mode} / {result.status}"
                          + (" (swapped in)" if result.swapped else "")),
        ("dataset", result.dataset_dir),
        ("dataset version", result.dataset_version or "-"),
        ("partitions built this run", str(len(result.partitions))),
        ("partitions verified and reused", str(len(result.skipped_partitions))),
        ("partitions removed", str(len(result.removed_partitions))),
        ("rows read this run", f"{result.rows_read:,}"),
        ("rows written this run", f"{result.rows_written:,}"),
        ("rows dropped this run", f"{result.rows_dropped:,}"),
        ("malformed rows this run", f"{result.malformed_rows:,}"),
        ("dataset partitions / rows", f"{result.total_partitions:,} / {result.total_rows:,}"),
        ("output size", human_bytes(result.output_bytes)),
        ("compression", f"{result.source_bytes / result.output_bytes:.1f}x"
                        if result.output_bytes and complete else "n/a (partial)"),
        ("duration", f"{result.duration_seconds:.1f}s"),
        ("source sorted / output sorted", f"{result.input_sorted} / {result.output_sorted}"),
    ]:
        table.add_row(key, value)
    console.print(table)
    if result.truncated_by_limit:
        console.print(f"[yellow]Partial smoke output in {result.dataset_dir}; the "
                      "canonical dataset was not touched.[/yellow]")
    elif result.status != "complete":
        console.print("[red]The dataset is not complete; re-run `xq convert`.[/red]")
    console.print(f"[green]Reports written to:[/green] {config.metadata_path}")
    return 0


def cmd_source_index(config: Config, args: argparse.Namespace) -> int:
    """Index the raw source and write the ordering diagnosis."""
    from .data.source_index import ORDERING_REPORT_NAME, SourceIndexer, ordering_report

    sources = inspect_mod.resolve_sources(config.raw_data_path)
    if not sources:
        console.print(f"[red]No raw data found at[/red] {config.raw_data_path}")
        return 2
    converter = TickConverter(config)
    indexer = SourceIndexer(config, block_bytes=config.conversion.index_block_bytes)
    indexer.resolve_columns(converter.source_column_names(sources[0]))
    index = indexer.load_or_build(sources, rebuild=args.rebuild)
    report = ordering_report(index, sources, max_events=args.max_events)
    path = ensure_report_dir(config) / ORDERING_REPORT_NAME
    atomic_write_text(path, json.dumps(report, indent=1) + "\n")

    order = report["summary"]
    table = Table(title="source index", box=None, show_header=False)
    table.add_column("k", style="cyan")
    table.add_column("v")
    for key, value in [
        ("files", ", ".join(Path(f.path).name for f in index.files)),
        ("raw fingerprint", index.raw_fingerprint),
        ("lines", f"{index.total_lines:,} ({index.unassigned_lines:,} unassigned)"),
        ("partitions", f"{len(index.partitions)}: {index.keys[0]} .. {index.keys[-1]}"
                       if index.partitions else "0"),
        ("source sorted", str(order.get("input_sorted"))),
        ("backward jumps", f"{order.get('backward_jumps', 0):,}"),
        ("late lines", f"{order.get('late_lines', 0):,} "
                       f"({order.get('late_lines_in_earlier_partition', 0):,} in an "
                       "earlier partition)"),
        ("late lines repeated verbatim", f"{order.get('late_lines_repeated_verbatim', 0):,}"),
        ("late lines not seen before", f"{order.get('late_lines_not_seen_before', 0):,}"),
        ("backward jump size (s)", f"{order.get('backward_seconds_min')} .. "
                                   f"{order.get('backward_seconds_max')}"),
    ]:
        table.add_row(key, str(value))
    console.print(table)
    console.print(f"[green]Index:[/green] {indexer.path}")
    console.print(f"[green]Ordering report:[/green] {path}")
    return 0


def ensure_report_dir(config: Config) -> Path:
    config.metadata_path.mkdir(parents=True, exist_ok=True)
    return config.metadata_path


def cmd_verify_dataset(config: Config, args: argparse.Namespace) -> int:
    """Independent integrity and coverage verification of the tick dataset."""
    from .data.dataset_validation import verify_tick_dataset, write_verification

    with console.status("verifying every partition..."):
        result = verify_tick_dataset(config, deep=not args.quick)
    report, coverage = write_verification(config, result)

    table = Table(title="tick coverage by year", box=None)
    for name in ("year", "months", "tick rows", "start", "end", "status"):
        table.add_column(name)
    for row in result.coverage:
        table.add_row(
            str(row["year"]), f"{row['months_present']}/{row['months_expected']}",
            f"{row['tick_rows']:,}", str(row["start"])[:19], str(row["end"])[:19],
            row["status"],
        )
    console.print(table)

    facts = Table(title="integrity", box=None, show_header=False)
    facts.add_column("k", style="cyan")
    facts.add_column("v")
    recon = result.source_reconciliation
    for key, value in [
        ("verdict", "[green]PASSED[/green]" if result.passed else "[red]FAILED[/red]"),
        ("dataset version", f"{result.dataset_version} (recomputed "
                            f"{result.dataset_version_recomputed})"),
        ("span", f"{result.earliest_timestamp} -> {result.latest_timestamp}"),
        ("rows / partitions", f"{result.total_rows:,} / {result.partition_count}"),
        ("storage", human_bytes(result.storage_bytes)),
        ("non-monotonic output rows", f"{result.non_monotonic_rows:,}"),
        ("rows outside their month", f"{result.impure_rows:,}"),
        ("duplicate timestamps kept", f"{result.duplicate_timestamps:,}"),
        ("exact duplicate rows kept", f"{result.exact_duplicate_rows:,}"),
        ("ask < bid", f"{result.ask_below_bid:,}"),
        ("invalid prices (<=0 / non-finite / out of range)",
         f"{result.non_positive_prices:,} / {result.non_finite_prices:,} / "
         f"{result.prices_out_of_range:,}"),
        ("digest mismatches", str(len(result.digest_mismatches)) if result.deep else "not checked"),
        ("source lines = written + dropped + malformed",
         f"{recon.get('source_lines')} = {recon.get('rows_written')} + "
         f"{recon.get('rows_dropped')} + {recon.get('malformed_rows')} "
         f"({'balanced' if recon.get('balanced') else 'NOT balanced'})"),
        ("months missing from output (present in source)",
         ", ".join(result.months_missing_from_output) or "none"),
        ("months missing from the source itself",
         ", ".join(result.months_missing_from_source) or "none"),
        ("gaps > threshold by category", json.dumps(result.gaps_by_category)),
    ]:
        facts.add_row(key, value)
    console.print(facts)
    for failure in result.failures:
        console.print(f"[red]FAILURE:[/red] {failure}")
    console.print(f"[green]Report:[/green] {report}\n[green]Coverage:[/green] {coverage}")
    return 0 if result.passed else 1


def cmd_metadata(config: Config, _args: argparse.Namespace) -> int:
    meta = meta_mod.build_dataset_metadata(config)
    path = meta_mod.write_dataset_metadata(config, meta)

    table = Table(title="dataset metadata", box=None, show_header=False)
    table.add_column("k", style="cyan")
    table.add_column("v")
    for key, value in [
        ("total rows", f"{meta.total_rows:,}"),
        ("span", f"{meta.start_timestamp} -> {meta.end_timestamp}"),
        ("trading days", f"{meta.trading_days:,}"),
        ("calendar days", f"{meta.calendar_days_spanned:,}"),
        ("partitions / files", f"{meta.partitions} / {meta.files}"),
        ("storage", meta.storage_human),
        ("compression", f"{meta.compression_ratio:.1f}x" if meta.compression_ratio else "n/a"),
        ("price min / max", f"{meta.price_min} / {meta.price_max}"),
        ("spread mean / median", f"{meta.spread_mean:.5f} / {meta.spread_median:.5f}"
                                 if meta.spread_mean is not None else "n/a"),
        ("spread min / max", f"{meta.spread_min} / {meta.spread_max}"),
        ("quantile method", meta.quantile_method),
        ("duplicate timestamps", f"{meta.duplicate_timestamps:,}"
                                 if meta.duplicate_timestamps is not None else "n/a"),
        ("timezone", meta.timezone.get("description", "")),
    ]:
        table.add_row(key, str(value))
    console.print(table)
    if meta.spread_percentiles:
        console.print("[bold]spread percentiles:[/bold] " + "  ".join(
            f"{k}={v:.4f}" for k, v in meta.spread_percentiles.items()
        ))
    for warning in meta.warnings:
        console.print(f"[yellow]warning:[/yellow] {warning}")
    console.print(f"[green]Written:[/green] {path}")
    return 0


def cmd_build_bars(config: Config, args: argparse.Namespace) -> int:
    resampler = BarResampler(config)
    timeframes = list(config.resampling.timeframes) if args.all else [args.timeframe]
    validate_timeframes(timeframes)

    table = Table(title="bars built", box=None)
    for name in ("timeframe", "bars", "ticks", "years", "size", "span", "version", "seconds"):
        table.add_column(name)
    for timeframe in timeframes:
        result = resampler.build(
            timeframe, overwrite=not args.keep_existing,
            allow_incomplete=args.allow_incomplete,
        )
        table.add_row(
            timeframe, f"{result.bars:,}", f"{result.ticks_read:,}",
            str(len(result.bars_per_year)) if not result.skipped else "up to date",
            human_bytes(result.output_bytes),
            f"{result.first_timestamp} -> {result.last_timestamp}",
            str(result.bar_dataset_version), f"{result.duration_seconds:.1f}",
        )
    console.print(table)
    console.print(
        f"[dim]Convention: bars are stamped with the {config.resampling.label} of "
        f"their interval; intervals are "
        f"{'[t, t+D)' if config.resampling.closed == 'left' else '(t, t+D]'}; "
        f"OHLC from {config.resampling.price_source}.[/dim]"
    )
    return 0


def cmd_diagnose(config: Config, args: argparse.Namespace) -> int:
    timeframes = list(config.resampling.timeframes) if args.all else [args.timeframe]
    results = diag_mod.diagnose_all(config, timeframes)
    path = diag_mod.write_diagnostics(config, results)

    table = Table(title="bar diagnostics", box=None)
    for name in ("tf", "bars", "coverage", "missing", "gaps", "unexplained",
                 "mean ticks/bar", "low-tick bars"):
        table.add_column(name)
    for timeframe, report in results.items():
        table.add_row(
            timeframe, f"{report.bars:,}", f"{report.coverage_ratio:.2%}",
            f"{report.missing_slots:,}", f"{report.gap_count:,}",
            f"{report.gaps_by_category.get('holiday_or_unknown', 0):,}",
            f"{report.mean_ticks_per_bar:.1f}" if report.mean_ticks_per_bar else "-",
            f"{report.low_tick_count_bars:,}",
        )
    console.print(table)
    for timeframe, report in results.items():
        for warning in report.warnings:
            console.print(f"[yellow]{timeframe}:[/yellow] {warning}")
        if report.largest_unexplained_gaps:
            console.print(f"[bold]{timeframe} largest unexplained gaps:[/bold]")
            for gap in report.largest_unexplained_gaps[:5]:
                console.print(f"  {gap['from']} -> {gap['to']}  "
                              f"({gap['duration_hours']}h, {gap['missing_bars']:,} bars)")
    console.print(f"[green]Written:[/green] {path}")
    return 0


def cmd_query(config: Config, args: argparse.Namespace) -> int:
    columns = [c.strip() for c in args.columns.split(",")] if args.columns else None
    with DataStore(config) as store:
        if args.sql:
            frame = store.sql(args.sql)
        elif args.start and args.end:
            frame = (
                store.load_bars(args.timeframe, args.start, args.end, columns, limit=args.limit)
                if args.timeframe
                else store.load_ticks(args.start, args.end, columns, limit=args.limit)
            )
        else:
            console.print("[red]Provide --sql, or both --start and --end.[/red]")
            return 2
    console.print(frame.head(args.limit))
    console.print(f"[dim]{frame.height:,} row(s) x {frame.width} column(s)[/dim]")
    return 0


def _require_research_extra() -> None:
    """Turn a missing scipy/statsmodels/matplotlib into an actionable message.

    The research layers are an optional extra so that the data pipeline
    installs without them. A bare ``ModuleNotFoundError: scipy`` does not tell
    anyone that, so it is translated here.
    """
    missing = []
    for module in ("scipy", "statsmodels", "matplotlib"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise SystemExit(
            f"This command needs the research extra ({', '.join(missing)} "
            'missing). Install it with:\n\n    pip install -e ".[research]"\n'
        )


def cmd_research_summary(config: Config, args: argparse.Namespace) -> int:
    _require_research_extra()
    from .research.config import load_research_config
    from .research.reports import generate_multi_timeframe_summary, generate_report

    research = load_research_config(args.research_config)
    timeframes = list(research.timeframes) if args.all else [args.timeframe]

    console.print(
        f"[bold]Research configuration[/bold]: {research.config_path}\n"
        f"  fingerprint   : {research.fingerprint()}\n"
        f"  price source  : {research.returns.price_source} "
        f"({research.returns.price_column})\n"
        f"  return type   : {research.returns.primary}\n"
        f"  time basis    : {research.intraday.time_basis} "
        f"({config.timezone.describe()})\n"
        f"  output        : {research.results_path}"
    )

    reports = []
    for timeframe in timeframes:
        try:
            with console.status(f"analysing {timeframe}..."):
                report = generate_report(
                    config, research, timeframe=timeframe,
                    start=args.start, end=args.end,
                    make_plots=not args.no_plots,
                )
        except FileNotFoundError as exc:
            console.print(f"[yellow]{timeframe}: {exc}[/yellow]")
            continue
        reports.append(report)

    if not reports:
        console.print("[red]No timeframe could be analysed.[/red]")
        return 1

    table = Table(title="research summary", box=None)
    for name in ("tf", "bars", "span", "ret std", "kurtosis", "acf1",
                 "|r| acf1", "files", "sec"):
        table.add_column(name)
    for report in reports:
        m = report.metrics
        table.add_row(
            report.timeframe,
            f"{report.observations:,}",
            f"{str(report.first_timestamp)[:10]} -> {str(report.last_timestamp)[:10]}",
            f"{m.return_std:.3e}" if m else "-",
            f"{m.excess_kurtosis:.2f}" if m else "-",
            f"{m.acf_lag1:+.4f}" if m else "-",
            f"{m.abs_acf_lag1:+.4f}" if m else "-",
            str(len(report.files)),
            f"{report.duration_seconds:.1f}",
        )
    console.print(table)

    for report in reports:
        for warning in report.warnings[:4]:
            console.print(f"[yellow]{report.timeframe}:[/yellow] {warning}")

    if len(reports) > 1:
        path = generate_multi_timeframe_summary(config, research, reports)
        if path:
            console.print(f"[green]Multi-timeframe comparison:[/green] {path}")

    console.print(f"[green]Results written to:[/green] {research.results_path}")
    console.print(
        "[dim]Descriptive statistics only. No transaction costs are applied and "
        "nothing here is a trading signal.[/dim]"
    )
    return 0


def cmd_regression_research(config: Config, args: argparse.Namespace) -> int:
    _require_research_extra()
    from .features.config import load_regression_config
    from .research.regression_reports import (
        generate_regression_report,
        load_bars,
        write_comparison_tables,
    )

    regression = load_regression_config(args.regression_config)

    if args.all:
        timeframes = list(regression.timeframes)
        windows = list(regression.regression.windows)
    else:
        timeframes = [args.timeframe]
        if args.all_windows:
            windows = list(regression.regression.windows)
        elif args.window:
            windows = [int(args.window)]
        else:
            console.print(
                "[red]Give --window N, or --all-windows, with --timeframe.[/red]"
            )
            return 2

    console.print(
        f"[bold]Regression configuration[/bold]: {regression.config_path}\n"
        f"  fingerprint   : {regression.fingerprint()}\n"
        f"  transform     : {regression.regression.price_transform} of "
        f"{regression.regression.price_source} {regression.regression.price_column}\n"
        f"  windows       : {windows}\n"
        f"  timeframes    : {timeframes}\n"
        f"  time basis    : {regression.conditioning.time_basis} "
        f"({config.timezone.describe()})\n"
        f"  output        : {regression.results_path}"
    )

    reports = []
    for timeframe in timeframes:
        try:
            bars = load_bars(config, timeframe, args.start, args.end)
        except FileNotFoundError as exc:
            console.print(f"[yellow]{timeframe}: {exc}[/yellow]")
            continue
        for window in windows:
            with console.status(f"fitting {timeframe} window={window}..."):
                report = generate_regression_report(
                    config, regression, timeframe=timeframe, window=int(window),
                    start=args.start, end=args.end,
                    make_plots=not args.no_plots, bars=bars,
                )
            reports.append(report)

    if not reports:
        console.print("[red]No model could be analysed.[/red]")
        return 1

    table = Table(title="rolling-regression residual research", box=None)
    for name in ("tf", "win", "fits", "resid std", "decay b", "control b",
                 "ADF p", "P(->0) h5", "med cross", "sec"):
        table.add_column(name)
    for report in reports:
        m = report.model_summary
        if not m:
            table.add_row(report.timeframe, str(report.window), "-", "-", "-",
                          "-", "-", "-", "-", f"{report.duration_seconds:.1f}")
            continue
        table.add_row(
            m["timeframe"], str(m["regression_window"]), f"{m['observations']:,}",
            _fmt_metric(m.get("residual_std"), "{:.3e}"),
            _fmt_metric(m.get("decay_coefficient"), "{:+.4f}"),
            _fmt_metric(m.get("decay_coefficient_control"), "{:+.4f}"),
            _fmt_metric(m.get("adf_pvalue"), "{:.2e}"),
            _fmt_metric(m.get("move_toward_zero_probability_5"), "{:.3f}"),
            _fmt_metric(m.get("median_zero_crossing_bars"), "{:.0f}"),
            f"{report.duration_seconds:.1f}",
        )
    console.print(table)

    seen = set()
    for report in reports:
        for warning in report.warnings:
            if warning not in seen:
                seen.add(warning)
                console.print(f"[yellow]{report.timeframe} w={report.window}:[/yellow] "
                              f"{warning}")

    if len(reports) > 1:
        for path in write_comparison_tables(config, regression, reports):
            console.print(f"[green]Comparison:[/green] {path}")

    console.print(f"[green]Results written to:[/green] {regression.results_path}")
    console.print(
        "[dim]Descriptive only. The decay coefficient is not an OU speed and not a "
        "half-life; compare it against the random-walk control. No costs are applied "
        "and nothing here is a trading signal.[/dim]"
    )
    return 0


def cmd_ou_research(config: Config, args: argparse.Namespace) -> int:
    """Run the OU study for the requested (timeframe, N, M) combinations."""
    _require_research_extra()
    from .features.config import load_regression_config
    from .models.config import load_ou_config
    from .research.config import load_research_config
    from .research.ou_reports import (
        generate_ou_report,
        reusable_ou_report,
        write_ou_comparison,
    )

    ou = load_ou_config(args.ou_config)
    if args.compare:
        from .research.ou_reports import load_ou_reports

        existing = load_ou_reports(ou)
        for path in write_ou_comparison(config, ou, existing):
            console.print(f"[green]Comparison:[/green] {path}")
        console.print(f"{len(existing)} OU studies compared.")
        return 0 if existing else 1
    regression = load_regression_config(args.regression_config)
    research = load_research_config(args.research_config)
    if args.all:
        timeframes = list(ou.timeframes)
        pairs = [(n, m) for n in ou.regression_windows for m in ou.ou_estimation_windows]
    else:
        timeframes = [args.timeframe]
        if args.all_windows:
            pairs = [(n, m) for n in ou.regression_windows for m in ou.ou_estimation_windows]
        elif args.regression_window and args.ou_window:
            pairs = [(int(args.regression_window), int(args.ou_window))]
        else:
            console.print("[red]Give --regression-window N and --ou-window M, or "
                          "--all-windows, with --timeframe.[/red]")
            return 2
    console.print(
        f"[bold]OU configuration[/bold]: {ou.config_path}\n"
        f"  fingerprint : {ou.fingerprint()}\n"
        f"  timeframes  : {timeframes}\n"
        f"  (N, M)      : {pairs}\n"
        f"  controls    : {ou.controls.enabled()}\n"
        f"  output      : {ou.results_path}"
    )
    def run_study(**kwargs: Any) -> Any:
        if args.in_process:
            return generate_ou_report(config, regression, research, ou, **kwargs)
        # A fresh process per study, so nothing one study leaves in the
        # allocator accumulates over a grid (the first full grid was stopped
        # for low system memory on its third 1-minute study).
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_init_study_worker,
                                 initargs=(args.log_level,)) as pool:
            return pool.submit(generate_ou_report, config, regression, research, ou,
                               **kwargs).result()

    if not args.in_process:
        # Read by Polars' allocator (mimalloc) when the child starts: freed
        # pages go straight back to the system instead of being kept for
        # reuse, which cut a 15m study's peak working set by ~20% at no
        # measurable cost in time. An explicit setting wins.
        os.environ.setdefault("MIMALLOC_PURGE_DELAY", "0")
    reports = []
    failures: list[tuple[str, int, int, str]] = []
    resume = args.resume and args.start is None and args.end is None
    if args.resume and not resume:
        console.print("[yellow]--resume ignored: a date-filtered run always recomputes.[/yellow]")
    for timeframe in timeframes:
        for n, m in pairs:
            report = reusable_ou_report(
                config, regression, research, ou, timeframe=timeframe, regression_window=n,
                ou_window=m, need_parameters=args.write_parameters,
                need_plots=not args.no_plots and ou.plots.enabled,
            ) if resume else None
            if report is not None:
                reports.append(report)
                continue
            # One failing (timeframe, N, M) must not cost the rest of a long grid:
            # it is logged in full, the others run, and the exit code says so.
            try:
                with console.status(f"OU {timeframe} N={n} M={m}..."):
                    report = run_study(
                        timeframe=timeframe, regression_window=n, ou_window=m,
                        start=args.start, end=args.end, make_plots=not args.no_plots,
                        write_parameters=True if args.write_parameters else None,
                    )
            except MemoryError:
                raise
            except Exception as exc:
                LOGGER.exception("OU %s N=%d M=%d failed", timeframe, n, m)
                failures.append((timeframe, n, m, f"{type(exc).__name__}: {exc}"))
                continue
            reports.append(report)

    table = Table(title="Ornstein-Uhlenbeck residual research", box=None)
    for name in ("tf", "N", "M", "valid OU", "median b", "median HL", "HL p25-p75",
                 "control HL", "ref HL", "innov acf1", "sec"):
        table.add_column(name)
    for report in reports:
        s = report.model_summary
        if not s:
            table.add_row(report.timeframe, str(report.regression_window),
                          str(report.ou_window), *["-"] * 7, f"{report.duration_seconds:.0f}")
            continue
        control = s.get("control_random_walk") or {}
        reference = s.get("reference_ou") or {}
        table.add_row(
            s["timeframe"], str(s["regression_window"]), str(s["ou_window"]),
            _fmt_metric(s.get("valid_ou_fraction"), "{:.1%}"),
            _fmt_metric(s.get("median_b"), "{:.3f}"),
            _fmt_metric(s.get("median_half_life_bars"), "{:.1f}"),
            f"{_fmt_metric(s.get('half_life_p25'), '{:.1f}')}-"
            f"{_fmt_metric(s.get('half_life_p75'), '{:.1f}')}",
            _fmt_metric(control.get("median_half_life_bars"), "{:.1f}"),
            _fmt_metric(reference.get("median_half_life_bars"), "{:.1f}"),
            _fmt_metric(s.get("innovation_lag1_acf"), "{:+.3f}"),
            "reused" if report.reused else f"{report.duration_seconds:.0f}",
        )
    console.print(table)
    seen: set[str] = set()
    for report in reports:
        for warning in report.warnings:
            if warning not in seen:
                seen.add(warning)
                console.print(f"[yellow]{report.timeframe} N={report.regression_window} "
                              f"M={report.ou_window}:[/yellow] {warning}")
    if len(reports) > 1:
        for path in write_ou_comparison(config, ou, reports):
            console.print(f"[green]Comparison:[/green] {path}")
    console.print(f"[green]Results written to:[/green] {ou.results_path}")
    console.print(
        "[dim]Descriptive only. Read every half-life beside the detrended-random-walk "
        "control and the exact-OU reference. No costs, entries or exits exist here.[/dim]"
    )
    for timeframe, n, m, message in failures:
        console.print(f"[red]FAILED[/red] {timeframe} N={n} M={m}: {message}")
    if failures:
        console.print(f"[red]{len(failures)} of {len(failures) + len(reports)} studies failed; "
                      "the comparison above covers only those that ran. Fix and rerun with "
                      "--resume.[/red]")
    return 1 if failures else 0


def cmd_build_features(config: Config, args: argparse.Namespace) -> int:
    """Build the versioned regression-feature cache and its coverage report."""
    _require_research_extra()
    from .features.config import load_regression_config
    from .features.store import RegressionFeatureStore
    from .research.regression_reports import load_bars

    regression = load_regression_config(args.regression_config)
    store = RegressionFeatureStore(config, regression)
    timeframes = list(regression.timeframes) if args.all else [args.timeframe]
    windows = [int(args.window)] if args.window else list(regression.regression.windows)
    for timeframe in timeframes:
        bars = load_bars(config, timeframe, None, None)
        with console.status(f"features {timeframe} {windows}..."):
            store.build(timeframe, windows, bars=bars, force=args.force)
        del bars

    rows = store.coverage(list(regression.timeframes))
    table = Table(title="regression-feature coverage", box=None)
    for name in ("tf", "window", "rows", "valid fits", "first valid", "last", "years",
                 "current", "size"):
        table.add_column(name)
    for row in rows:
        table.add_row(
            row["timeframe"], str(row["window"]), f"{row['rows']:,}", f"{row['valid_fits']:,}",
            str(row["first_valid_timestamp"])[:19], str(row["last_timestamp"])[:19],
            str(row["years"]), "yes" if row["current"] else "[red]STALE[/red]",
            human_bytes(row["bytes"] or 0),
        )
    console.print(table)
    report = config.metadata_path / "regression_feature_coverage.json"
    atomic_write_text(report, json.dumps({
        "generated_utc": utc_now_iso(),
        "feature_store": str(regression.features_path),
        "coverage": rows,
    }, indent=1, default=str) + "\n")
    console.print(f"[green]Coverage report:[/green] {report}")
    return 0 if rows and all(r["current"] for r in rows) else 1


def cmd_spectral_research(config: Config, args: argparse.Namespace) -> int:
    """Run the spectral study for the requested timeframes, series and windows."""
    _require_research_extra()
    from .data.resampler import load_bar_manifest
    from .features.config import load_regression_config
    from .features.spectral_config import INPUT_SERIES, load_spectral_config
    from .models.config import load_ou_config
    from .research.config import load_research_config
    from .research.fft_analysis import benchmark_rolling_fft
    from .research.spectral_reports import (
        generate_spectral_timeframe,
        load_spectral_studies,
        reusable_spectral_study,
        write_spectral_comparison,
    )
    from .utils.paths import ensure_dir

    spectral = load_spectral_config(args.spectral_config)
    if args.compare:
        on_disk = load_spectral_studies(spectral)
        for path in write_spectral_comparison(spectral, on_disk):
            console.print(f"[green]Comparison:[/green] {path}")
        console.print(f"{len(on_disk)} spectral studies compared.")
        return 0 if on_disk else 1
    if args.benchmark:
        timings = benchmark_rolling_fft(spectral)
        out = ensure_dir(spectral.results_path) / "benchmark.csv"
        timings.write_csv(out)
        console.print(timings.pivot(on="stage", index="fft_window", values="windows_per_second"))
        console.print(f"[green]Benchmark:[/green] {out}")
        return 0
    regression = load_regression_config(args.regression_config)
    ou = load_ou_config(args.ou_config)
    research = load_research_config(args.research_config)
    if args.source is not None and args.source not in INPUT_SERIES:
        console.print(f"[red]--source must be one of {list(INPUT_SERIES)}[/red]")
        return 2
    series = [args.source] if args.source else list(spectral.input_series)
    if args.fft_window:
        windows = [int(args.fft_window)]
    elif args.all_windows or args.all:
        windows = list(spectral.fft_windows)
    else:
        console.print("[red]Give --fft-window N or --all-windows with --timeframe.[/red]")
        return 2
    timeframes = list(spectral.timeframes) if args.all else [args.timeframe]
    policy = args.write_features or spectral.features_output.write

    plan: list[tuple[str, list[str], list[int]]] = []
    for timeframe in timeframes:
        bars = int((load_bar_manifest(config.bars_dir(timeframe)) or {}).get("rows") or 0)
        if bars > spectral.grid.max_bars_without_override and not args.allow_heavy:
            console.print(f"[yellow]{timeframe}: {bars:,} bars exceeds "
                          f"grid.max_bars_without_override "
                          f"({spectral.grid.max_bars_without_override:,}); skipped. "
                          "Pass --allow-heavy to run it.[/yellow]")
            continue
        todo_series = [s for s in series if not (args.resume and all(
            reusable_spectral_study(config, regression, ou, spectral, timeframe=timeframe,
                                    series=s, fft_window=n) for n in windows))]
        if todo_series:
            plan.append((timeframe, todo_series, windows))
        # Rough upper bound for the per-bar feature files of this timeframe.
        written_windows = (windows if policy == "all" else [] if policy == "none" else
                           [n for n in windows if n == spectral.representative_fft_window])
        estimate = bars * len(todo_series) * len(written_windows) * 40 * 4 * 0.6
        console.print(f"[dim]{timeframe}: {bars:,} bars, {len(todo_series)} series x "
                      f"{len(windows)} windows x {1 + len(spectral.controls.enabled())} sources"
                      f"; features on disk <= ~{estimate / 2**30:.2f} GiB[/dim]")
    console.print(
        f"[bold]Spectral configuration[/bold]: {spectral.config_path}\n"
        f"  fingerprint : {spectral.fingerprint()}\n"
        f"  inputs      : N={spectral.regression_window} residual, M={spectral.ou_window} OU\n"
        f"  controls    : {spectral.controls.enabled()}\n"
        f"  output      : {spectral.results_path}"
    )

    def run(timeframe: str, names: list[str], lengths: list[int]) -> Any:
        if args.in_process:
            return generate_spectral_timeframe(
                config, regression, research, ou, spectral, timeframe=timeframe, series=names,
                windows=lengths, make_plots=not args.no_plots, write_features=policy)
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_init_study_worker,
                                 initargs=(args.log_level,)) as pool:
            return pool.submit(
                generate_spectral_timeframe, config, regression, research, ou, spectral,
                timeframe=timeframe, series=names, windows=lengths,
                make_plots=not args.no_plots, write_features=policy).result()

    if not args.in_process:
        os.environ.setdefault("MIMALLOC_PURGE_DELAY", "0")
    studies: list[Any] = []
    failures: list[tuple[str, str]] = []
    for timeframe, names, lengths in plan:
        try:
            with console.status(f"spectral {timeframe} {names} N={lengths}..."):
                studies += run(timeframe, names, lengths)
        except MemoryError:
            raise
        except Exception as exc:
            LOGGER.exception("spectral %s failed", timeframe)
            failures.append((timeframe, f"{type(exc).__name__}: {exc}"))

    table = Table(title="Spectral research (descriptive; read every value against the nulls)",
                  box=None)
    for name in ("tf", "series", "N", "valid", "entropy", "flatness", "top3", "dom. period",
                 "persist", "max|rIC|", "rIC null"):
        table.add_column(name)
    for study in studies:
        s = study.summary
        nulls = (s.get("null_control_difference") or {}).get("max_abs_rank_ic") or {}
        null_ic = max((v for k, v in nulls.items() if k != "real" and isinstance(v, float)),
                      default=None)
        table.add_row(
            s["timeframe"], s["input_series"], str(s["fft_window"]),
            _fmt_metric(s.get("valid_spectral_fraction"), "{:.0%}"),
            _fmt_metric(s.get("median_spectral_entropy"), "{:.3f}"),
            _fmt_metric(s.get("median_spectral_flatness"), "{:.3f}"),
            _fmt_metric(s.get("median_top3_power_share"), "{:.3f}"),
            _fmt_metric(s.get("median_dominant_period_bars"), "{:.1f}"),
            _fmt_metric(s.get("dominant_period_persistence"), "{:.3f}"),
            _fmt_metric(s.get("max_abs_rank_ic"), "{:.4f}"),
            _fmt_metric(null_ic, "{:.4f}"),
        )
    console.print(table)
    all_studies = load_spectral_studies(spectral)
    if len(all_studies) > 1:
        for path in write_spectral_comparison(spectral, all_studies):
            console.print(f"[green]Comparison:[/green] {path}")
    console.print(f"[green]Results written to:[/green] {spectral.results_path}")
    console.print("[dim]Descriptive only. A spectral peak is not a cycle until it survives "
                  "the null controls; nothing here is a trading signal.[/dim]")
    for timeframe, message in failures:
        console.print(f"[red]FAILED[/red] {timeframe}: {message}")
    return 1 if failures else 0


#: Rough cost model measured on the development machine (1h trial, all windows):
#: seconds per pass by source role, plus seconds per bar. For the estimate
#: printed before a run - not a guarantee.
_WAVELET_SECONDS_PER_BAR = {"real": 4.0e-5, "incremental": 3.0e-5, "core": 1.5e-5,
                            "shape": 2.0e-6}
_WAVELET_SECONDS_PER_PASS = {"real": 15.0, "incremental": 12.0, "core": 6.0, "shape": 0.5}


def cmd_wavelet_research(config: Config, args: argparse.Namespace) -> int:
    """Run the wavelet / time-frequency study (Prompt #6)."""
    _require_research_extra()
    from .data.resampler import load_bar_manifest
    from .features.config import load_regression_config
    from .features.spectral_config import INPUT_SERIES, load_spectral_config
    from .features.wavelet_config import load_wavelet_config
    from .models.config import load_ou_config
    from .research.config import load_research_config
    from .research.wavelet_analysis import (
        benchmark_wavelet,
        boundary_study,
        chirp_control,
        localized_oscillation_control,
        multiscale_control,
    )
    from .research.wavelet_reports import (
        generate_wavelet_timeframe,
        load_wavelet_studies,
        reusable_wavelet_study,
        source_roles,
        write_wavelet_comparison,
    )
    from .utils.paths import ensure_dir

    wavelet = load_wavelet_config(args.wavelet_config)
    spectral = load_spectral_config(args.spectral_config)
    if args.compare:
        on_disk = load_wavelet_studies(wavelet)
        for path in write_wavelet_comparison(wavelet, on_disk):
            console.print(f"[green]Comparison:[/green] {path}")
        console.print(f"{len(on_disk)} wavelet studies compared.")
        return 0 if on_disk else 1
    if args.benchmark:
        timings = benchmark_wavelet(wavelet)
        out = ensure_dir(wavelet.results_path / "offline") / "benchmark.csv"
        timings.write_csv(out)
        console.print(timings)
        console.print(f"[green]Benchmark:[/green] {out}")
        return 0
    if args.controls:
        from .research import wavelet_plots

        directory = ensure_dir(wavelet.results_path / "offline")
        tables = {
            "control_localized_oscillation": localized_oscillation_control(wavelet, spectral),
            "control_chirp": chirp_control(wavelet, spectral),
            "control_multiscale": multiscale_control(wavelet),
        }
        trials, summary = boundary_study(wavelet)
        tables["boundary_study"] = summary
        tables["boundary_trials"] = trials
        for name, frame in tables.items():
            frame.write_csv(directory / f"{name}.csv")
            console.print(f"[green]{name}[/green]: {directory / f'{name}.csv'}")
        if wavelet.plots.enabled and not args.no_plots:
            wavelet_plots.plot_boundary_study(
                summary, path=wavelet.results_path / "plots" / "boundary_study.png",
                config=wavelet)
        console.print(tables["control_localized_oscillation"])
        console.print(tables["control_chirp"])
        console.print(summary)
        return 0

    regression = load_regression_config(args.regression_config)
    ou = load_ou_config(args.ou_config)
    research = load_research_config(args.research_config)
    if args.source is not None and args.source not in INPUT_SERIES:
        console.print(f"[red]--source must be one of {list(INPUT_SERIES)}[/red]")
        return 2
    series = [args.source] if args.source else list(wavelet.input_series)
    if args.window:
        windows = [int(args.window)]
    elif args.all_windows or args.all:
        windows = list(wavelet.windows)
    else:
        console.print("[red]Give --window N or --all-windows with --timeframe.[/red]")
        return 2
    timeframes = list(wavelet.timeframes) if args.all else [args.timeframe]
    policy = args.write_features or wavelet.features_output.write
    roles = source_roles(wavelet)

    plan: list[tuple[str, list[str], list[int]]] = []
    total_seconds = 0.0
    for timeframe in timeframes:
        bars = int((load_bar_manifest(config.bars_dir(timeframe)) or {}).get("rows") or 0)
        if bars > wavelet.grid.max_bars_without_override and not args.allow_heavy:
            console.print(f"[yellow]{timeframe}: {bars:,} bars exceeds "
                          f"grid.max_bars_without_override "
                          f"({wavelet.grid.max_bars_without_override:,}); skipped. "
                          "Pass --allow-heavy to run it.[/yellow]")
            continue
        todo = [s for s in series if not (args.resume and all(
            reusable_wavelet_study(config, wavelet, timeframe=timeframe, series=s, window=n)
            for n in windows))]
        if todo:
            plan.append((timeframe, todo, windows))
        passes = len(todo) * len(windows)
        seconds = sum(passes * (bars * _WAVELET_SECONDS_PER_BAR[role]
                                + _WAVELET_SECONDS_PER_PASS[role]) for role in roles.values())
        total_seconds += seconds
        written = (windows if policy == "all" else [] if policy == "none" else
                   [n for n in windows if n == wavelet.representative_window])
        disk = bars * len(todo) * len(written) * 33 * 4 * 0.5
        console.print(f"[dim]{timeframe}: {bars:,} bars, {len(todo)} series x {len(windows)} "
                      f"windows x {len(roles)} sources; ~{seconds / 60:.0f} min, features on "
                      f"disk <= ~{disk / 2**30:.2f} GiB (rough estimates)[/dim]")
    console.print(
        f"[bold]Wavelet configuration[/bold]: {wavelet.config_path}\n"
        f"  fingerprint : {wavelet.fingerprint()}\n"
        f"  causal      : {wavelet.causal_features.method}, {wavelet.dwt.wavelet}, "
        f"windows {list(wavelet.windows)}\n"
        f"  controls    : {wavelet.controls.enabled()}\n"
        f"  estimate    : ~{total_seconds / 3600:.1f} h in total\n"
        f"  output      : {wavelet.results_path}"
    )
    if args.dry_run:
        return 0

    check = not args.no_fft_feature_check

    def run(timeframe: str, names: list[str], lengths: list[int]) -> Any:
        if args.in_process:
            return generate_wavelet_timeframe(
                config, regression, research, ou, spectral, wavelet, timeframe=timeframe,
                series=names, windows=lengths, make_plots=not args.no_plots,
                write_features=policy, require_spectral_features=check)
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_init_study_worker,
                                 initargs=(args.log_level,)) as pool:
            return pool.submit(
                generate_wavelet_timeframe, config, regression, research, ou, spectral, wavelet,
                timeframe=timeframe, series=names, windows=lengths, make_plots=not args.no_plots,
                write_features=policy, require_spectral_features=check).result()

    if not args.in_process:
        os.environ.setdefault("MIMALLOC_PURGE_DELAY", "0")
    studies: list[Any] = []
    failures: list[tuple[str, str]] = []
    for timeframe, names, lengths in plan:
        try:
            with console.status(f"wavelet {timeframe} {names} N={lengths}..."):
                studies += run(timeframe, names, lengths)
        except MemoryError:
            raise
        except Exception as exc:
            LOGGER.exception("wavelet %s failed", timeframe)
            failures.append((timeframe, f"{type(exc).__name__}: {exc}"))

    table = Table(title="Wavelet research (descriptive; read every value against the nulls)",
                  box=None)
    for name in ("tf", "series", "N", "valid", "entropy", "dom. period", "fast/slow",
                 "persist", "best|rIC|", "rIC null"):
        table.add_column(name)
    for study in studies:
        s = study.summary
        nulls = (s.get("null_control_difference") or {}).get("max_abs_rank_ic") or {}
        null_ic = max((v for k, v in nulls.items() if k != "real" and isinstance(v, float)),
                      default=None)
        table.add_row(
            s["timeframe"], s["input_series"], str(s["rolling_window"]),
            _fmt_metric(s.get("valid_feature_fraction"), "{:.0%}"),
            _fmt_metric(s.get("median_wavelet_entropy"), "{:.3f}"),
            _fmt_metric(s.get("median_dominant_period_bars"), "{:.1f}"),
            _fmt_metric(s.get("median_fast_slow_energy_ratio"), "{:.2f}"),
            _fmt_metric(s.get("dominant_scale_persistence"), "{:.3f}"),
            _fmt_metric(s.get("best_rank_ic"), "{:.4f}"),
            _fmt_metric(null_ic, "{:.4f}"),
        )
    console.print(table)
    all_studies = load_wavelet_studies(wavelet)
    if len(all_studies) > 1:
        for path in write_wavelet_comparison(wavelet, all_studies):
            console.print(f"[green]Comparison:[/green] {path}")
    console.print(f"[green]Results written to:[/green] {wavelet.results_path}")
    console.print("[dim]Descriptive only. Causal features are one-sided and live-safe; "
                  "scalograms are NON-CAUSAL. Nothing here is a trading signal.[/dim]")
    for timeframe, message in failures:
        console.print(f"[red]FAILED[/red] {timeframe}: {message}")
    return 1 if failures else 0


def _regime_configs(args: argparse.Namespace) -> tuple[Any, ...]:
    from .features.config import load_regression_config
    from .features.spectral_config import load_spectral_config
    from .features.wavelet_config import load_wavelet_config
    from .models.config import load_ou_config
    from .regimes.config import load_regime_config
    from .research.config import load_research_config

    return (load_regression_config(args.regression_config),
            load_research_config(args.research_config), load_ou_config(args.ou_config),
            load_spectral_config(args.spectral_config), load_wavelet_config(args.wavelet_config),
            load_regime_config(args.regime_config))


def cmd_regime_research(config: Config, args: argparse.Namespace) -> int:
    """Run the regime-discovery study (Prompt #7)."""
    _require_research_extra()
    from .data.resampler import load_bar_manifest
    from .regimes.config import MODEL_NAMES
    from .research.regime_reports import (
        benchmark_regimes,
        estimate_seconds,
        generate_regime_timeframe,
        load_regime_studies,
        run_synthetic_controls,
        write_regime_comparison,
    )

    regression, research, ou, spectral, wavelet, regime = _regime_configs(args)
    if args.null_increment_states:
        import dataclasses

        increment_states = tuple(int(v) for v in args.null_increment_states.split(",")
                                 if v.strip())
        regime = dataclasses.replace(regime, nulls=dataclasses.replace(
            regime.nulls, increment_states=increment_states))
    if args.compare:
        written = write_regime_comparison(regime)
        for path in written:
            console.print(f"[green]Comparison:[/green] {path}")
        return 0 if written else 1
    if args.controls:
        for name, path in run_synthetic_controls(regime).items():
            console.print(f"[green]{name}[/green]: {path}")
        return 0
    if args.benchmark:
        timings = benchmark_regimes(config, regression, research, ou, spectral, wavelet, regime)
        console.print(timings)
        return 0
    if args.model is not None and args.model not in MODEL_NAMES:
        console.print(f"[red]--model must be one of {list(MODEL_NAMES)}[/red]")
        return 2
    models = [args.model] if args.model else regime.enabled_models()
    stages = tuple(s.strip() for s in args.stages.split(",") if s.strip())
    unknown = [s for s in stages if s not in ("offline", "walk_forward", "nulls", "analysis")]
    if unknown:
        console.print(f"[red]unknown stage(s) {unknown}[/red]")
        return 2
    if args.schemes:
        schemes = [tuple(s.strip().split("_", 1)) for s in args.schemes.split(",") if s.strip()]
    else:
        schemes = [(s, regime.causal.primary_refit) for s in regime.causal.schemes]
    timeframes = list(regime.timeframes) if args.all else [args.timeframe]
    plan = []
    total = 0.0
    for timeframe in timeframes:
        bars = int((load_bar_manifest(config.bars_dir(timeframe)) or {}).get("rows") or 0)
        if bars > regime.grid.max_bars_without_override and not args.allow_heavy:
            console.print(f"[yellow]{timeframe}: {bars:,} bars exceeds "
                          f"grid.max_bars_without_override; skipped (use --allow-heavy).[/yellow]")
            continue
        seconds = 0.0
        for model in models:
            states = [args.states] if args.states else list(regime.states_for(model))
            extra = {}
            if args.monthly and model == "hmm":
                extra = {(model, k): [("expanding", "monthly")] for k in states
                         if k in regime.causal.monthly_states}
            for k in states:
                seconds += estimate_seconds(bars, model, k, stages=stages,
                                            schemes=len(schemes) + len(extra.get((model, k), [])))
            plan.append((timeframe, model, states, extra))
        if "nulls" in stages and "hmm" in models:
            seconds += estimate_seconds(bars, "hmm", 4, stages=("nulls",), schemes=0)
        total += seconds
        console.print(f"[dim]{timeframe}: {bars:,} bars, models {models}; ~{seconds / 60:.0f} "
                      "min (rough estimate from this PC's benchmark)[/dim]")
    console.print(f"[bold]Regime configuration[/bold]: {regime.config_path}\n"
                  f"  fingerprint : {regime.fingerprint()}\n"
                  f"  features    : {list(regime.features())}\n"
                  f"  stages      : {list(stages)}   schemes: {schemes}\n"
                  f"  estimate    : ~{total / 3600:.1f} h in total\n"
                  f"  output      : {regime.results_path}")
    if args.dry_run:
        return 0

    def run(timeframe: str, model_list: list[str], states: list[int], extra: dict) -> Any:
        kwargs = {"timeframe": timeframe, "models": model_list, "states": states,
                  "schemes": schemes, "stages": stages, "extra_schemes": extra,
                  "resume": args.resume, "make_plots": not args.no_plots}
        if args.in_process:
            return generate_regime_timeframe(config, regression, research, ou, spectral,
                                             wavelet, regime, progress=console.print, **kwargs)
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_init_study_worker,
                                 initargs=(args.log_level,)) as pool:
            return pool.submit(generate_regime_timeframe, config, regression, research, ou,
                               spectral, wavelet, regime, **kwargs).result()

    if not args.in_process:
        os.environ.setdefault("MIMALLOC_PURGE_DELAY", "0")
    failures: list[tuple[str, str]] = []
    studies: list[Any] = []
    for timeframe, model, states, extra in plan:
        try:
            studies += run(timeframe, [model], states, extra)
        except MemoryError:
            raise
        except Exception as exc:
            LOGGER.exception("regime %s %s failed", timeframe, model)
            failures.append((f"{timeframe} {model}", f"{type(exc).__name__}: {exc}"))
    table = Table(title="Regime research (descriptive; read every value against the baselines)",
                  box=None)
    for name in ("tf", "model", "K", "OOS ll/bar", "median dur.", "overlap ARI", "NMI vs vol",
                 "confidence"):
        table.add_column(name)
    for study in studies:
        s = study.summary
        durations = [v for v in (s.get("median_state_duration") or {}).values() if v is not None]
        table.add_row(s["timeframe"], s["model"], str(s["states"]),
                      _fmt_metric(s.get("out_of_sample_log_likelihood"), "{:.3f}"),
                      _fmt_metric(statistics.median(durations) if durations else None, "{:.0f}"),
                      _fmt_metric((s.get("state_alignment_stability") or {}).get(
                          "median_overlap_ari"), "{:.2f}"),
                      _fmt_metric((s.get("volatility_baseline_improvement") or {}).get(
                          "agreement_nmi"), "{:.2f}"),
                      _fmt_metric(s.get("mean_state_confidence"), "{:.2f}"))
    console.print(table)
    if len(load_regime_studies(regime)) > 1:
        for path in write_regime_comparison(regime):
            console.print(f"[green]Comparison:[/green] {path}")
    console.print(f"[green]Results written to:[/green] {regime.results_path}")
    console.print("[dim]Descriptive only. Walk-forward filtered probabilities are causal; "
                  "full-sample fits, smoothed probabilities and Viterbi paths are OFFLINE / "
                  "NON-CAUSAL. Nothing here is a trading signal.[/dim]")
    for what, message in failures:
        console.print(f"[red]FAILED[/red] {what}: {message}")
    return 1 if failures else 0


def cmd_regime_walk_forward(config: Config, args: argparse.Namespace) -> int:
    """One causal walk-forward regime run (Prompt #7)."""
    _require_research_extra()
    from .regimes.config import MODEL_NAMES
    from .research.regime_reports import prepare_timeframe, walk_forward_stage

    regression, research, ou, spectral, wavelet, regime = _regime_configs(args)
    if args.model not in MODEL_NAMES:
        console.print(f"[red]--model must be one of {list(MODEL_NAMES)}[/red]")
        return 2
    k = args.states or regime.states_for(args.model)[0]
    scheme = args.scheme or regime.causal.primary_scheme
    refit = args.refit or regime.causal.primary_refit
    ctx = prepare_timeframe(config, regression, ou, spectral, wavelet, research, regime,
                            args.timeframe, progress=console.print)
    result = walk_forward_stage(ctx, model=args.model, k=k, scheme=scheme, refit=refit,
                                resume=args.resume)
    fitted = sum(1 for status in (result.refits["status"].to_list()
                                  if "status" in result.refits.columns else [])
                 if status == "fitted")
    console.print(f"[green]{args.timeframe} {args.model} K={k} {scheme}/{refit}[/green]: "
                  f"{result.frame.height:,} bars, {fitted} refits, "
                  f"{result.seconds / 60:.1f} min; first model {result.model_ids[:1]}")
    console.print("[dim]Filtered probabilities only (live safe); stored under "
                  f"{regime.features_path}.[/dim]")
    return 0


def _feature_configs(config: Config, args: argparse.Namespace) -> Any:
    """Every configuration the feature factory and research read (Prompt #8)."""
    from .alpha.config import load_alpha_config
    from .features.config import load_regression_config
    from .features.factory_config import load_features_config
    from .features.spectral_config import load_spectral_config
    from .features.wavelet_config import load_wavelet_config
    from .models.config import load_ou_config
    from .regimes.config import load_regime_config
    from .research.config import load_research_config
    from .research.feature_research import Configs
    from .targets.config import load_targets_config

    return Configs(config=config, regression=load_regression_config(args.regression_config),
                   ou=load_ou_config(args.ou_config),
                   spectral=load_spectral_config(args.spectral_config),
                   wavelet=load_wavelet_config(args.wavelet_config),
                   research=load_research_config(args.research_config),
                   features=load_features_config(args.features_config),
                   targets=load_targets_config(args.targets_config),
                   alpha=load_alpha_config(args.alpha_config),
                   regime_features_path=load_regime_config(args.regime_config).features_path)


#: Rough seconds per bar of a full feature research run, measured at 1h on this PC.
_FEATURE_RESEARCH_SECONDS_PER_BAR = 290.0 / 141_569


def cmd_build_feature_factory(config: Config, args: argparse.Namespace) -> int:
    """Build (or confirm current) the versioned feature matrix and targets (Prompt #8)."""
    _require_research_extra()
    from .features.factory import factory_gate
    from .research.feature_research import ResearchContext, stage_factory

    cfg = _feature_configs(config, args)
    timeframes = list(cfg.features.timeframes) if args.all else [args.timeframe]
    failures = []
    for tf in timeframes:
        if args.dry_run:
            gate = factory_gate(config, cfg.regression, cfg.spectral, cfg.wavelet, cfg.features,
                                cfg.regime_features_path, tf)
            console.print(f"[bold]{tf}[/bold]: gate {'passed' if gate['passed'] else 'FAILED'}")
            for line in gate["problems"]:
                console.print(f"  [red]problem[/red] {line}")
            for line in gate["notes"]:
                console.print(f"  [dim]{line}[/dim]")
            continue
        ctx = ResearchContext(timeframe=tf, cfg=cfg, out_dir=cfg.features.results_path / tf,
                              progress=console.print)
        try:
            built = stage_factory(ctx, resume=not args.force)
        except Exception as exc:
            LOGGER.exception("feature factory %s failed", tf)
            failures.append((tf, f"{type(exc).__name__}: {exc}"))
            continue
        m = built["factory"]
        low = {k: round(v, 3) for k, v in (m.get("coverage") or {}).items() if v < 0.9}
        console.print(f"[green]{tf}[/green]: {m.get('features')} live-safe features x "
                      f"{m.get('rows'):,} bars, {m.get('factory_version')}; targets "
                      f"{built['targets'].get('target_version')}")
        if low:
            console.print(f"  [dim]coverage < 0.9: {low}[/dim]")
    for tf, message in failures:
        console.print(f"[red]FAILED[/red] {tf}: {message}")
    console.print("[dim]Features use bars <= t only; targets live in their own namespace "
                  f"({cfg.targets.targets_path}) and are never joined into the matrix.[/dim]")
    return 1 if failures else 0


def _research_and_report(cfg: Any, timeframe: str, stages: tuple[str, ...], resume: bool,
                         write_ledger: bool, report_only: bool) -> dict[str, Any]:
    """One timeframe: the stages, then statuses / ledger / figures (runs in a child)."""
    from .research.feature_reports import build_timeframe_report
    from .research.feature_research import run_timeframe

    info: dict[str, Any] = {}
    if not report_only:
        info = run_timeframe(cfg, timeframe, stages=stages, resume=resume)
    summary = build_timeframe_report(cfg.features, cfg.alpha, cfg.targets.targets_path,
                                     timeframe, write_ledger=write_ledger)
    return {"run": info, "summary": summary}


def _show_feature(cfg: Any, args: argparse.Namespace) -> int:
    import polars as pl

    base = cfg.features.results_path / args.timeframe
    status_path = base / "feature_status.parquet"
    if not status_path.exists():
        console.print(f"[red]no stored research for {args.timeframe}[/red] - run "
                      f"`xq feature-research --timeframe {args.timeframe}` first")
        return 1
    status = pl.read_parquet(status_path).filter(pl.col("feature") == args.feature)
    if status.is_empty():
        console.print(f"[red]{args.feature!r} is not a registered feature of "
                      f"{args.timeframe}[/red]")
        return 1
    row = status.row(0, named=True)
    console.print(f"[bold]{args.feature}[/bold] ({args.timeframe}): {row['status']} - "
                  f"{row['reason']}")
    ev = pl.read_parquet(base / "feature_evidence.parquet").filter(
        pl.col("feature") == args.feature)
    if args.target:
        ev = ev.filter(pl.col("target").str.starts_with(f"target_{args.target}_"))
    if args.horizon:
        ev = ev.filter(pl.col("horizon") == args.horizon)
    table = Table(box=None)
    for name in ("target", "rank IC", "SE", "q", "shift null", "pipeline", "mech.",
                 "yearly sign", "recent IC", "verdict"):
        table.add_column(name)
    for r in ev.sort("target").iter_rows(named=True):
        table.add_row(r["target"], _fmt_metric(r.get("value"), "{:+.4f}"),
                      _fmt_metric(r.get("se"), "{:.4f}"), _fmt_metric(r.get("q_value"), "{:.1e}"),
                      str(r.get("beyond_shift_null")), str(r.get("beyond_pipeline_null")),
                      _fmt_metric(r.get("mechanical_share"), "{:.2f}"),
                      _fmt_metric(r.get("sign_consistency"), "{:.0%}"),
                      _fmt_metric(r.get("recent_ic"), "{:+.4f}"), str(r.get("verdict")))
    console.print(table)
    return 0


def cmd_feature_research(config: Config, args: argparse.Namespace) -> int:
    """Predictive feature research of one or every timeframe (Prompt #8)."""
    _require_research_extra()
    from .data.resampler import load_bar_manifest
    from .research.feature_reports import write_comparison
    from .research.feature_research import STAGES

    cfg = _feature_configs(config, args)
    if args.compare:
        summary = write_comparison(cfg.features, list(cfg.features.timeframes))
        console.print(summary)
        return 0 if summary.get("timeframes") else 1
    if args.feature:
        if not args.timeframe:
            console.print("[red]--feature needs --timeframe[/red]")
            return 2
        return _show_feature(cfg, args)
    stages = (tuple(s.strip() for s in args.stages.split(",") if s.strip()) if args.stages
              else STAGES)
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        console.print(f"[red]unknown stage(s) {unknown}; expected from {list(STAGES)}[/red]")
        return 2
    timeframes = list(cfg.features.timeframes) if args.all else [args.timeframe]
    plan = []
    for tf in timeframes:
        bars = int((load_bar_manifest(config.bars_dir(tf)) or {}).get("rows") or 0)
        if tf not in cfg.features.timeframes and not args.allow_heavy:
            console.print(f"[yellow]{tf}: not in features.yaml timeframes; skipped "
                          "(use --allow-heavy).[/yellow]")
            continue
        seconds = bars * _FEATURE_RESEARCH_SECONDS_PER_BAR
        plan.append(tf)
        console.print(f"[dim]{tf}: {bars:,} bars; ~{seconds / 60:.0f} min for every stage "
                      "(rough, from the 1h benchmark; the factory build adds FFT / wavelet "
                      "windows that are not stored)[/dim]")
    console.print(f"[bold]Feature research[/bold]: stages {list(stages)}\n"
                  f"  features config : {cfg.features.config_path} ({cfg.features.fingerprint()})\n"
                  f"  alpha config    : {cfg.alpha.config_path} ({cfg.alpha.fingerprint()})\n"
                  f"  output          : {cfg.features.results_path}\n"
                  f"  ledger          : {cfg.alpha.ledger_path if not args.no_ledger else 'off'}")
    if args.dry_run:
        return 0

    def run(tf: str) -> dict[str, Any]:
        kwargs = {"stages": stages, "resume": not args.no_resume,
                  "write_ledger": not args.no_ledger, "report_only": args.report_only}
        if args.in_process:
            return _research_and_report(cfg, tf, **kwargs)
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_init_study_worker,
                                 initargs=(args.log_level,)) as pool:
            return pool.submit(_research_and_report, cfg, tf, **kwargs).result()

    if not args.in_process:
        os.environ.setdefault("MIMALLOC_PURGE_DELAY", "0")
    failures = []
    table = Table(title="Feature research (statuses are statistical information, not signals)",
                  box=None)
    for name in ("tf", "features", "strong", "candidate", "weak", "redundant", "unstable",
                 "failed", "tests", "noise cand."):
        table.add_column(name)
    for tf in plan:
        try:
            out = run(tf)
        except MemoryError:
            raise
        except Exception as exc:
            LOGGER.exception("feature research %s failed", tf)
            failures.append((tf, f"{type(exc).__name__}: {exc}"))
            continue
        s = out["summary"]
        c = s["status_counts"]
        table.add_row(tf, str(s["features"]), *[str(c.get(k, 0)) for k in (
            "strong_candidate", "candidate", "weak_candidate", "redundant", "unstable",
            "failed_null")], f"{s['multiple_testing']['tests']:,}",
            str(len((s.get("noise_control") or {}).get(
                "noise_features_with_candidate_status") or [])))
    console.print(table)
    if len(plan) > 1 or args.all:
        write_comparison(cfg.features, list(cfg.features.timeframes))
    console.print(f"[green]Results written to:[/green] {cfg.features.results_path}")
    console.print("[dim]Descriptive only: rank ICs against circular-shift, pipeline and "
                  "noise-feature nulls. No model, no signal, no cost, no PnL.[/dim]")
    for tf, message in failures:
        console.print(f"[red]FAILED[/red] {tf}: {message}")
    return 1 if failures else 0


def cmd_feature_redundancy(_config: Config, args: argparse.Namespace) -> int:
    """Clusters, redundancy groups and representatives from the stored research (Prompt #8)."""
    import json

    import polars as pl

    from .features.factory_config import load_features_config

    fcfg = load_features_config(args.features_config)
    base = fcfg.results_path / args.timeframe
    path = base / "redundancy/feature_clusters.json"
    if not path.exists():
        console.print(f"[red]no redundancy tables for {args.timeframe}[/red] - run `xq "
                      f"feature-research --timeframe {args.timeframe} --stages redundancy`")
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    status = (pl.read_parquet(base / "feature_status.parquet")
              if (base / "feature_status.parquet").exists() else pl.DataFrame())
    statuses = ({r["feature"]: (r["status"], r.get("representative"))
                 for r in status.iter_rows(named=True)} if not status.is_empty() else {})
    if args.feature:
        corr = pl.read_parquet(base / "redundancy/feature_correlations.parquet")
        near = corr.filter((pl.col("a") == args.feature) | (pl.col("b") == args.feature)).with_columns(
            pl.when(pl.col("a") == args.feature).then(pl.col("b")).otherwise(pl.col("a"))
            .alias("other")).sort(pl.col("spearman").abs(), descending=True).head(12)
        console.print(f"[bold]{args.feature}[/bold]: {statuses.get(args.feature)}")
        for r in near.iter_rows(named=True):
            console.print(f"  {r['other']:42s} spearman {r['spearman']:+.3f}  normalised MI "
                          f"{(r.get('normalized_mi') or 0):.3f}  {statuses.get(r['other'])}")
        return 0
    console.print(f"[bold]{args.timeframe}[/bold]: clusters at |rho| >= "
                  f"{data['threshold_abs_spearman']} ({data['linkage']} linkage, estimate on "
                  f"{data['sample_rows']:,} bars)")
    for cid, members in data["clusters"].items():
        if len(members) < 2:
            continue
        shown = ", ".join(f"{m} ({statuses.get(m, ('?', None))[0]})" for m in members)
        console.print(f"  cluster {cid}: {shown}")
    hidden = data.get("nonmonotone_pairs") or []
    if hidden:
        console.print("[bold]non-monotone near-duplicates[/bold] (normalised MI high, |rho| low):")
        for r in hidden[:15]:
            console.print(f"  {r['a']} ~ {r['b']}: NMI {r['normalized_mi']:.2f}, "
                          f"|rho| {r['abs_spearman']:.2f}")
    return 0


def cmd_alpha_decay(_config: Config, args: argparse.Namespace) -> int:
    """IC decay and information half-life of one feature from the stored research (Prompt #8)."""
    import polars as pl

    from .features.factory_config import load_features_config

    fcfg = load_features_config(args.features_config)
    path = fcfg.results_path / args.timeframe / "ic/alpha_decay.parquet"
    if not path.exists():
        console.print(f"[red]no decay table for {args.timeframe}[/red] - run `xq "
                      f"feature-research --timeframe {args.timeframe} --stages ic`")
        return 1
    decay = pl.read_parquet(path).filter((pl.col("feature") == args.feature)
                                         & (pl.col("method") == "spearman"))
    if args.target:
        decay = decay.filter(pl.col("target_family").str.ends_with(args.target))
    if decay.is_empty():
        console.print(f"[red]no decay rows for {args.feature!r}[/red]")
        return 1
    table = Table(title=f"{args.feature} ({args.timeframe}), rank IC by horizon / lag", box=None)
    for name in ("family", "curve", "ICs", "peak h", "half h", "zero h", "info half-life",
                 "fit"):
        table.add_column(name)
    for r in decay.sort("curve", "target_family").iter_rows(named=True):
        ics = " ".join(f"{h}:{v:+.3f}" for h, v in zip(r["horizons"], r["ics"], strict=True))
        table.add_row(str(r["target_family"]), str(r["curve"]), ics,
                      str(r.get("peak_horizon")), str(r.get("half_decay_horizon")),
                      str(r.get("zero_horizon")),
                      _fmt_metric(r.get("alpha_information_half_life"), "{:.1f} bars"),
                      str(r.get("fit") or ""))
    console.print(table)
    console.print("[dim]Horizon curves use cumulative targets (dilution ~ 1/sqrt(h)); the "
                  "half-life is fitted on the marginal curve only, when it is exponential.[/dim]")
    return 0


def _selection_configs(config: Config, args: argparse.Namespace) -> Any:
    """Every configuration the feature selection reads (Prompt #9)."""
    from .research.feature_selection_research import SelectionConfigs
    from .selection.config import load_selection_config

    base = _feature_configs(config, args)
    return SelectionConfigs(selection=load_selection_config(args.selection_config),
                            features=base.features, targets=base.targets, alpha=base.alpha,
                            config=config, regression=base.regression, ou=base.ou,
                            spectral=base.spectral, wavelet=base.wavelet, research=base.research)


#: Rough seconds per bar (all bars of the timeframe) of a full selection run, every stage,
#: measured at 1h on this PC.
_FEATURE_SELECTION_SECONDS_PER_BAR = 420.0 / 141_569

#: Target names accepted by ``feature-select --target`` -> the target family.
_TARGET_ALIASES = {
    "return": "return", "future_return": "return", "direction": "return",
    "residual_reduction": "residual_reduction", "reversion": "residual_reduction",
    "mean_reversion": "residual_reduction",
    "realized_vol": "realized_vol", "volatility": "realized_vol",
    "future_volatility": "realized_vol",
    "abs_return": "abs_return", "magnitude": "abs_return"}


def _selection_and_report(cfgs: Any, timeframe: str, stages: tuple[str, ...], resume: bool,
                          write_ledger: bool) -> dict[str, Any]:
    """One timeframe: the selection stages, then summary / ledger / figures (runs in a child)."""
    from .research.feature_selection_reports import build_selection_report
    from .research.feature_selection_research import run_selection

    info = run_selection(cfgs, timeframe, stages=stages, resume=resume)
    summary = build_selection_report(
        cfgs.selection.results_path, timeframe,
        ledger_path=cfgs.selection.ledger_path if write_ledger else None, cfg=cfgs.selection)
    return {"run": info, "summary": summary}


def _show_selection_target(results: Path, timeframe: str, family: str, horizon: int | None
                           ) -> int:
    import json

    import polars as pl

    out = results / timeframe
    if not (out / "sets.json").exists():
        console.print(f"[red]no stored selection for {timeframe}[/red]")
        return 1
    targets = [f"target_{family}_{h}" for h in ((horizon,) if horizon else (1, 5, 20))]
    freq = pl.read_parquet(out / "selection_frequency.parquet").filter(
        pl.col("target").is_in(targets) & ~pl.col("is_probe"))
    if freq.is_empty():
        console.print(f"[red]no selection rows for {targets}[/red] (horizons configured: see "
                      "feature_selection.yaml)")
        return 1
    stab = pl.read_parquet(out / "selection_stability.parquet").filter(
        pl.col("target").is_in(targets))
    sets = json.loads((out / "sets.json").read_text(encoding="utf-8"))
    kind = str(freq["kind"][0])
    for target in targets:
        part = freq.filter(pl.col("target") == target)
        if part.is_empty():
            continue
        s = stab.filter(pl.col("target") == target)
        depth = ", ".join(f"{r['selector']} {r['mean_selected']:.1f}"
                          for r in s.iter_rows(named=True))
        table = Table(title=f"{target} ({timeframe}): features selected before a noise probe "
                            f"(mean per resample: {depth})", box=None)
        for name in ("feature", "mRMR", "Lasso", "mean", "year subsets", "full dev"):
            table.add_column(name)
        wide = part.pivot(on="selector", index="feature", values="frequency").with_columns(
            ((pl.col("mrmr") + pl.col("lasso")) / 2).alias("mean")).sort(
            ["mean", "feature"], descending=[True, False]).filter(pl.col("mean") > 0)
        extra = part.group_by("feature").agg(pl.col("year_subsets_selected").max(),
                                             pl.col("in_full_development").any())
        for r in wide.join(extra, on="feature").sort(["mean", "feature"],
                                                     descending=[True, False]).head(20) \
                .iter_rows(named=True):
            table.add_row(r["feature"], f"{r['mrmr']:.2f}", f"{r['lasso']:.2f}",
                          f"{r['mean']:.2f}", str(r["year_subsets_selected"]),
                          str(r["in_full_development"]))
        console.print(table)
    name = f"target_{kind}"
    chosen = (sets.get("sets") or {}).get(name) or []
    why = (sets.get("justified") or {}).get(name) or {}
    console.print(f"[bold]{name}[/bold] set: {len(chosen)} features - "
                  f"{'justified' if why.get('justified') else 'not justified'} "
                  f"({why.get('reason')})")
    if chosen:
        console.print("  " + ", ".join(chosen))
    for r in (sets.get("kind_validation") or {}).get(kind, []):
        if r["target"] in targets:
            console.print(f"  validation rank IC {r['target']}: "
                          f"{_fmt_metric(r.get('rank_ic'), '{:+.4f}')} (read after selection)")
    if kind == "reversion":
        console.print("[dim]Residual targets: every candidate had to beat the random-walk "
                      "pipeline null on development (invariant 9); the set's validation IC "
                      "is not compared with a null model.[/dim]")
    return 0


def cmd_feature_select(config: Config, args: argparse.Namespace) -> int:
    """Feature selection + dimensionality-reduction research (Prompt #9)."""
    _require_research_extra()
    from .data.resampler import load_bar_manifest
    from .research.feature_selection_reports import write_selection_comparison
    from .research.feature_selection_research import STAGES

    cfgs = _selection_configs(config, args)
    scfg = cfgs.selection
    family = None
    if args.target:
        family = _TARGET_ALIASES.get(args.target.strip().lower())
        if family is None:
            console.print(f"[red]unknown target {args.target!r}[/red]; expected one of "
                          f"{sorted(_TARGET_ALIASES)}")
            return 2
    stages = (tuple(s.strip() for s in args.stages.split(",") if s.strip()) if args.stages
              else STAGES)
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        console.print(f"[red]unknown stage(s) {unknown}; expected from {list(STAGES)}[/red]")
        return 2
    timeframes = list(scfg.timeframes) if args.all else [args.timeframe]
    p = scfg.periods
    console.print(f"[bold]Feature selection[/bold]: stages {list(stages)}\n"
                  f"  selection config : {scfg.config_path} ({scfg.fingerprint()})\n"
                  f"  development      : {p.development.label()}\n"
                  f"  validation       : {p.validation.label()} (evaluates what development "
                  "chose; plateaus read here)\n"
                  f"  reserved test    : {p.reserved_test.label()} (outcomes never loaded)\n"
                  f"  output           : {scfg.results_path}\n"
                  f"  ledger           : {scfg.ledger_path if not args.no_ledger else 'off'}")
    for tf in timeframes:
        bars = int((load_bar_manifest(config.bars_dir(tf)) or {}).get("rows") or 0)
        seconds = bars * _FEATURE_SELECTION_SECONDS_PER_BAR
        console.print(f"[dim]{tf}: {bars:,} bars; ~{seconds / 60:.0f} min for every stage "
                      "(rough, from the 1h benchmark)[/dim]")
    if args.dry_run:
        return 0

    def run(tf: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"stages": stages, "resume": not args.no_resume,
                                  "write_ledger": not args.no_ledger}
        if args.in_process:
            return _selection_and_report(cfgs, tf, **kwargs)
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_init_study_worker,
                                 initargs=(args.log_level,)) as pool:
            return pool.submit(_selection_and_report, cfgs, tf, **kwargs).result()

    if not args.in_process:
        os.environ.setdefault("MIMALLOC_PURGE_DELAY", "0")
    failures = []
    table = Table(title="Feature selection (research sets, not a trading model)", box=None)
    for name in ("tf", "registered", "quality", "universe", "minimal", "standard", "extended",
                 "general", "direction", "reversion", "volatility", "magnitude", "live", "isolated"):
        table.add_column(name)
    for tf in timeframes:
        try:
            out = run(tf)
        except MemoryError:
            raise
        except Exception as exc:
            LOGGER.exception("feature selection %s failed", tf)
            failures.append((tf, f"{type(exc).__name__}: {exc}"))
            continue
        s = out["summary"]
        sets = s.get("sets") or {}

        def cell(name: str, sets: dict[str, Any] = sets) -> str:
            v = sets.get(name) or {}
            ok = (v.get("justified") or {}).get("justified")
            return f"{v.get('features', 0)}" + ("" if ok else " (-)")

        table.add_row(tf, str(s.get("registered_features")), str(s.get("quality_universe")),
                      str(s.get("selection_universe")),
                      *[cell(n) for n in ("minimal", "standard", "extended", "general",
                                          "target_direction", "target_reversion",
                                          "target_volatility", "target_magnitude")],
                      str((s.get("live_reconstruction") or {}).get("passed")),
                      str((s.get("reserved_isolation") or {}).get("passed")))
    console.print(table)
    if len(timeframes) > 1:
        write_selection_comparison(scfg.results_path, list(scfg.timeframes))
    console.print(f"[green]Results written to:[/green] {scfg.results_path}")
    console.print("[dim](-): not justified - see sets.json. Descriptive research: no final "
                  "model, no signal, no PnL; the reserved test period's outcomes were never "
                  "read.[/dim]")
    for tf, message in failures:
        console.print(f"[red]FAILED[/red] {tf}: {message}")
    if family and not failures:
        for tf in timeframes:
            _show_selection_target(scfg.results_path, tf, family, args.horizon)
    return 1 if failures else 0


def cmd_feature_selection_report(_config: Config, args: argparse.Namespace) -> int:
    """Summary, SEL-H ledger rows and figures from the stored selection (Prompt #9)."""
    _require_research_extra()
    from .research.feature_selection_reports import (
        build_selection_report,
        write_selection_comparison,
    )
    from .selection.config import load_selection_config

    scfg = load_selection_config(args.selection_config)
    if args.compare:
        summary = write_selection_comparison(scfg.results_path, list(scfg.timeframes))
        console.print(summary)
        return 0 if summary.get("timeframes") else 1
    timeframes = list(scfg.timeframes) if args.all else [args.timeframe]
    missing = []
    for tf in timeframes:
        if not (scfg.results_path / tf / "sets.json").exists():
            missing.append(tf)
            continue
        s = build_selection_report(scfg.results_path, tf,
                                   ledger_path=None if args.no_ledger else scfg.ledger_path,
                                   cfg=scfg, make_plots=not args.no_plots)
        console.print(f"[green]{tf}[/green]: universe {s['selection_universe']} of "
                      f"{s['registered_features']}; sets "
                      + ", ".join(f"{k} {v['features']}" for k, v in s["sets"].items())
                      + f"; {s.get('ledger_rows_upserted', 0)} ledger rows; "
                        f"{len(s.get('figures') or [])} figures")
    for tf in missing:
        console.print(f"[red]{tf}: no stored selection[/red] - run `xq feature-select "
                      f"--timeframe {tf}` first")
    if len(timeframes) > 1:
        write_selection_comparison(scfg.results_path, list(scfg.timeframes))
    return 1 if missing else 0


def cmd_build_feature_manifest(config: Config, args: argparse.Namespace) -> int:
    """Write (via the sets stage) / verify one immutable feature-set manifest (Prompt #9)."""
    import json

    from .features.factory_config import load_features_config
    from .selection.config import load_selection_config
    from .selection.manifest import load_manifest, matrix_from_manifest
    from .utils.paths import atomic_write_text

    scfg = load_selection_config(args.selection_config)
    fcfg = load_features_config(args.features_config)
    tf = args.timeframe or scfg.timeframes[0]
    out = scfg.results_path / tf
    sets_path = out / "sets.json"
    if not sets_path.exists():
        console.print(f"[red]no stored selection for {tf}[/red] - run `xq feature-select "
                      f"--timeframe {tf}` first")
        return 1
    sets = json.loads(sets_path.read_text(encoding="utf-8"))
    if args.set_name not in (sets.get("sets") or {}):
        console.print(f"[red]unknown set {args.set_name!r}[/red]; stored sets: "
                      f"{sorted(sets.get('sets') or {})}")
        return 2
    why = (sets.get("justified") or {}).get(args.set_name) or {}
    if not why.get("justified"):
        console.print(f"[yellow]{args.set_name} ({tf}) is not justified: {why.get('reason')} - "
                      "no manifest is written for it.[/yellow]")
        return 1
    path = out / "manifests" / f"{args.set_name}.json"
    if not path.exists():
        # manifests are written only by the sets stage, with the full selection evidence
        cfgs = _selection_configs(config, args)
        from .research.feature_selection_research import run_selection

        run_selection(cfgs, tf, stages=("sets",), progress=console.print)
    manifest = load_manifest(path)                       # hash + positions verified
    frame = matrix_from_manifest(manifest, fcfg)
    names = [r["name"] for r in manifest["features"]]
    if frame.columns != ["timestamp", *names]:
        console.print("[red]the stored matrix does not reproduce the manifest order[/red]")
        return 1
    console.print(f"[green]{manifest['feature_set_id']}[/green] ({manifest['content_hash']}): "
                  f"{manifest['feature_count']} features, max history "
                  f"{manifest['max_history_required']} bars, effective rank "
                  f"{_fmt_metric(manifest.get('effective_rank'), '{:.2f}')}; created from "
                  f"{manifest.get('created_from_period')}, reserved "
                  f"{manifest.get('reserved_test_period')}")
    console.print(f"  matrix reproduced: {frame.height:,} rows x {len(names)} columns in "
                  "manifest order")
    by_family: dict[str, list[str]] = {}
    for r in manifest["features"]:
        by_family.setdefault(str(r["family"]), []).append(r["name"])
    for fam, feats in by_family.items():
        console.print(f"  {fam:15s} {', '.join(feats)}")
    if args.output:
        atomic_write_text(args.output, json.dumps(manifest, indent=1, default=str) + "\n")
        console.print(f"  copied to {args.output}")
    return 0


# ---------------------------------------------------------------------------
# Supervised predictive research (Prompt #10)
# ---------------------------------------------------------------------------
def _require_ml_extra() -> None:
    missing = []
    for module in ("sklearn", "xgboost", "lightgbm", "catboost", "shap", "joblib"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise SystemExit(
            f"This command needs the ml extra ({', '.join(missing)} missing). Install it "
            'with:\n\n    pip install -e ".[ml]"\n')


def _ml_configs(config: Config, args: argparse.Namespace) -> Any:
    """Every configuration the supervised research reads (Prompt #10)."""
    from .ml.config import load_ml_config
    from .research.ml_research import MLConfigs
    from .selection.config import load_selection_config

    base = _feature_configs(config, args)
    return MLConfigs(ml=load_ml_config(args.ml_config),
                     selection=load_selection_config(args.selection_config),
                     features=base.features, targets=base.targets, config=config,
                     regression=base.regression, ou=base.ou, spectral=base.spectral,
                     wavelet=base.wavelet, research=base.research)


def _ml_research_child(cfgs: Any, timeframe: str, stages: tuple[str, ...], report: bool,
                       write_ledger: bool) -> dict[str, Any]:
    """One timeframe's stages (and report) - runs in a child process."""
    import warnings

    from .research.ml_reports import build_ml_report
    from .research.ml_research import load_context, run_ml_research

    # scikit-learn 1.9 + joblib 1.6 emit this once per forest prediction batch from
    # joblib's worker threads (153 MB of stderr in the first 5m run); it is harmless
    warnings.filterwarnings("ignore", message=r"`sklearn\.utils\.parallel\.delayed` should be "
                            r"used with `sklearn\.utils\.parallel\.Parallel`",
                            category=UserWarning)
    info = run_ml_research(cfgs, timeframe, stages=stages)
    if report:
        ctx = load_context(cfgs, timeframe)
        info["report"] = build_ml_report(ctx, ledger=write_ledger)
    return info


def _ml_target_check(cfg: Any, target: str, horizon: int | None, family: str | None
                     ) -> str | None:
    from .ml.config import CLASSIFIERS, REGRESSORS

    if target not in cfg.targets:
        return f"unknown target {target!r}; expected one of {sorted(cfg.targets)}"
    task = cfg.targets[target].task
    if family is not None:
        allowed = CLASSIFIERS if task == "classification" else REGRESSORS
        if family not in allowed:
            return f"{family!r} is not a {task} family; expected one of {list(allowed)}"
    if horizon is not None and horizon not in (1, 2, 3, 5, 10, 20, 50):
        return f"horizon {horizon} is not in the stored target table (1, 2, 3, 5, 10, 20, 50)"
    return None


_ML_COLUMNS = {"classification": (("auc", "{:.4f}"), ("log_loss_skill", "{:+.4f}"),
                                  ("brier_skill", "{:+.4f}"), ("ece", "{:.4f}"),
                                  ("rank_ic", "{:+.4f}")),
               "regression": (("rank_ic", "{:+.4f}"), ("mse_skill", "{:+.4f}"),
                              ("r2", "{:+.4f}"), ("rank_ic_raw", "{:+.4f}"),
                              ("mae", "{:.4f}"))}


def _print_ml_units(ctx: Any, units: list[Any]) -> None:
    """Block-by-block metrics of units already on disk (Platt for classification)."""
    import json

    import numpy as np

    from .research.ml_research import unit_paths

    by_family: dict[str, list[dict[str, Any]]] = {}
    for spec in units:
        js = unit_paths(ctx.out_dir, spec)[1]
        if not js.exists():
            continue
        d = json.loads(js.read_text(encoding="utf-8"))
        by_family.setdefault(spec.family, []).append(d)
    for fam, rows in by_family.items():
        task = ctx.cfg.targets[rows[0]["target"]].task
        calib = "platt" if task == "classification" else "raw"
        cols = _ML_COLUMNS[task]
        table = Table(title=f"{fam} / {rows[0]['feature_set']}: {rows[0]['target']} "
                            f"h{rows[0]['horizon']} ({calib}; skill vs the constant training "
                            "mean)", box=None)
        for name in ("block", "fit rows", "validation rows", *[c for c, _ in cols], "fit s"):
            table.add_column(name)
        values: dict[str, list[float]] = {c: [] for c, _ in cols}
        for d in sorted(rows, key=lambda d: d["fold"]["index"]):
            met = (d.get("metrics") or {}).get(calib) or {}
            for c, _ in cols:
                if met.get(c) is not None:
                    values[c].append(float(met[c]))
            table.add_row(d["fold"]["name"], f"{d['info'].get('fit_rows', 0):,}",
                          f"{d['info'].get('validation_rows', 0):,}",
                          *[_fmt_metric(met.get(c), f) for c, f in cols],
                          _fmt_metric(d["info"].get("fit_seconds"), "{:.1f}"))
        table.add_row("mean", "", "", *[_fmt_metric(np.mean(values[c]) if values[c] else None,
                                                    f) for c, f in cols], "")
        console.print(table)


def cmd_ml_research(config: Config, args: argparse.Namespace) -> int:
    """Supervised predictive-model research, development period only (Prompt #10)."""
    _require_research_extra()
    _require_ml_extra()
    from .research.ml_research import STAGES, load_context, pending_units, plan_stage

    cfgs = _ml_configs(config, args)
    cfg = cfgs.ml
    stages = (tuple(s.strip() for s in args.stages.split(",") if s.strip()) if args.stages
              else STAGES)
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        console.print(f"[red]unknown stage(s) {unknown}; expected from {list(STAGES)}[/red]")
        return 2
    timeframes = list(cfg.timeframes) if args.all else [args.timeframe]
    reserved = cfgs.selection.periods.reserved_test
    console.print(f"[bold]Supervised predictive research[/bold]: stages {list(stages)}\n"
                  f"  ml config     : {cfg.config_path} ({cfg.fingerprint()})\n"
                  f"  walk-forward  : {cfg.walk_forward.scheme}, blocks "
                  + ", ".join(f"{a.year}-{b.year - 1}" for a, b in
                              cfg.walk_forward.validation_blocks)
                  + f"; purge h + embargo {cfg.walk_forward.embargo_bars} bars\n"
                  f"  reserved test : {reserved.label()} (never loaded here)\n"
                  f"  output        : {cfg.results_path}\n"
                  f"  ledger        : {cfg.ledger_path if not args.no_ledger else 'off'}")
    if args.dry_run:
        for tf in timeframes:
            ctx = load_context(cfgs, tf)
            table = Table(title=f"{tf}: {ctx.data.n:,} development rows", box=None)
            for name in ("stage", "units", "current", "to run"):
                table.add_column(name)
            for stage in stages:
                if stage.startswith("pipeline_null"):
                    units = plan_stage(ctx, stage)
                    table.add_row(stage, str(len(units)), "?", "(null data built at run time)")
                    continue
                units = plan_stage(ctx, stage)
                todo = pending_units(ctx, units)
                table.add_row(stage, str(len(units)), str(len(units) - len(todo)),
                              str(len(todo)))
            console.print(table)
            console.print("[dim]'search' also runs the one-step neighbourhood of its best "
                          "LightGBM trial.[/dim]")
        return 0

    def run(tf: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"stages": stages, "report": args.report,
                                  "write_ledger": not args.no_ledger}
        if args.in_process:
            return _ml_research_child(cfgs, tf, **kwargs)
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_init_study_worker,
                                 initargs=(args.log_level,)) as pool:
            return pool.submit(_ml_research_child, cfgs, tf, **kwargs).result()

    if not args.in_process:
        os.environ.setdefault("MIMALLOC_PURGE_DELAY", "0")
    failures = []
    for tf in timeframes:
        try:
            info = run(tf)
        except MemoryError:
            raise
        except Exception as exc:
            LOGGER.exception("ml research %s failed", tf)
            failures.append((tf, f"{type(exc).__name__}: {exc}"))
            continue
        table = Table(title=f"{tf}: supervised research units", box=None)
        for name in ("stage", "units", "ran", "failed", "seconds"):
            table.add_column(name)
        for stage, r in info["stages"].items():
            table.add_row(stage, str(r.get("units")), str(r.get("ran")), str(r.get("failed")),
                          f"{r.get('seconds', 0):.0f}")
        console.print(table)
    console.print(f"[green]Results written to:[/green] {cfg.results_path}")
    console.print("[dim]Development period only; probabilities and expected values, never a "
                  "trade.[/dim]")
    for tf, message in failures:
        console.print(f"[red]FAILED[/red] {tf}: {message}")
    return 1 if failures else 0


def _ml_fold_indices(text: str | None) -> list[int] | None:
    if not text:
        return None
    return [int(s) - 1 for s in text.split(",") if s.strip()]


def cmd_ml_train(config: Config, args: argparse.Namespace) -> int:
    """Walk-forward training of one configuration (Prompt #10)."""
    _require_research_extra()
    _require_ml_extra()
    from .research.ml_research import load_context, plan_configuration, run_units

    cfgs = _ml_configs(config, args)
    problem = _ml_target_check(cfgs.ml, args.target, args.horizon, args.model)
    if problem:
        console.print(f"[red]{problem}[/red]")
        return 2
    ctx = load_context(cfgs, args.timeframe)
    units = plan_configuration(ctx, args.target, args.horizon, [args.model], args.feature_set,
                               fold_indices=_ml_fold_indices(args.folds))
    res = run_units(ctx, units, stage="ml-train")
    _print_ml_units(ctx, units)
    return 1 if res["failed"] else 0


def cmd_ml_walk_forward(config: Config, args: argparse.Namespace) -> int:
    """Every enabled family on one target x horizon, walk-forward (Prompt #10)."""
    _require_research_extra()
    _require_ml_extra()
    from .research.ml_reports import collect_units, paired_vs, summarise
    from .research.ml_research import load_context, plan_configuration, run_units

    cfgs = _ml_configs(config, args)
    cfg = cfgs.ml
    problem = _ml_target_check(cfg, args.target, args.horizon, None)
    if problem:
        console.print(f"[red]{problem}[/red]")
        return 2
    task = cfg.targets[args.target].task
    families = ([s.strip() for s in args.models.split(",") if s.strip()] if args.models
                else list(cfg.enabled_models(task)))
    for fam in families:
        problem = _ml_target_check(cfg, args.target, None, fam)
        if problem:
            console.print(f"[red]{problem}[/red]")
            return 2
    if "constant" not in families:
        families.insert(0, "constant")                  # every comparison needs the baseline
    ctx = load_context(cfgs, args.timeframe)
    units = plan_configuration(ctx, args.target, args.horizon, families, args.feature_set)
    res = run_units(ctx, units, stage="ml-walk-forward")
    folds, _ = collect_units(ctx.out_dir)
    part = folds.filter((folds["target"] == args.target) & (folds["horizon"] == args.horizon))
    _print_ml_summary(cfg, summarise(part, cfg), paired_vs(part, cfg, reference="constant"),
                      args.target, variants=False)
    return 1 if res["failed"] else 0


def _print_ml_summary(cfg: Any, summ: Any, paired: Any, target: str, *,
                      variants: bool) -> None:
    import polars as pl

    if summ.is_empty():
        console.print(f"[yellow]no stored units for {target}[/yellow]")
        return
    if not variants:
        summ = summ.filter(pl.col("variant") == "base")
    task = cfg.targets[target].task
    cols = _ML_COLUMNS[task]
    for (h,), part in summ.sort("horizon").group_by(["horizon"], maintain_order=True):
        table = Table(title=f"{target} h{h}: mean over walk-forward blocks "
                            f"({'Platt' if task == 'classification' else 'raw'}); primary "
                            f"{part['metric'][0]}", box=None)
        for name in ("family", "set", "variant", "blocks", "mean", "SE", "beats const",
                     "vs const (t)", *[f"mean {c}" for c, _ in cols[:3]], "recent"):
            table.add_column(name)
        pv = {(r["family"], r["feature_set"]): r for r in paired.filter(
            pl.col("horizon") == h).iter_rows(named=True)} if not paired.is_empty() else {}
        for r in part.sort("mean", descending=True, nulls_last=True).iter_rows(named=True):
            p = pv.get((r["family"], r["feature_set"])) if r["variant"] == "base" else None
            table.add_row(r["family"], r["feature_set"], r["variant"], str(r["blocks"]),
                          _fmt_metric(r["mean"], "{:+.4f}"), _fmt_metric(r["se"], "{:.4f}"),
                          f"{r['blocks_beating_baseline']}/{r['blocks']}",
                          _fmt_metric((p or {}).get("t"), "{:+.1f}"),
                          *[_fmt_metric(r.get(f"mean_{c}"), f) for c, f in cols[:3]],
                          _fmt_metric(r.get("recent_metric"), "{:+.4f}"))
        console.print(table)
    console.print("[dim]'beats const': blocks with positive skill against the constant "
                  "training mean (log-loss skill / MSE skill); t from the block-by-block "
                  "difference. Descriptive: no threshold, no trade.[/dim]")


def cmd_ml_compare(_config: Config, args: argparse.Namespace) -> int:
    """Compare stored walk-forward results of one target (Prompt #10; trains nothing)."""
    import polars as pl

    from .ml.config import load_ml_config
    from .research.ml_reports import collect_units, paired_vs, summarise

    cfg = load_ml_config(args.ml_config)
    problem = _ml_target_check(cfg, args.target, args.horizon, None)
    if problem:
        console.print(f"[red]{problem}[/red]")
        return 2
    out = cfg.results_path / args.timeframe
    if not (out / "units").exists():
        console.print(f"[red]no stored units under {out}[/red] - run `xq ml-research` or "
                      "`xq ml-walk-forward` first")
        return 1
    folds, _ = collect_units(out)
    part = folds.filter(pl.col("target") == args.target)
    if args.horizon is not None:
        part = part.filter(pl.col("horizon") == args.horizon)
    _print_ml_summary(cfg, summarise(part, cfg), paired_vs(part, cfg, reference="constant"),
                      args.target, variants=args.variants)
    return 0


def cmd_ml_report(config: Config, args: argparse.Namespace) -> int:
    """Tables, ML-H ledger rows and figures from the stored units (Prompt #10)."""
    _require_research_extra()
    _require_ml_extra()
    from .research.ml_reports import build_ml_report
    from .research.ml_research import load_context

    cfgs = _ml_configs(config, args)
    ctx = load_context(cfgs, args.timeframe)
    s = build_ml_report(ctx, ledger=not args.no_ledger, plots=not args.no_plots,
                        heavy=not args.light)
    console.print(f"[green]{args.timeframe}[/green]: {s.get('units', 0)} units, "
                  f"{s.get('configurations', 0)} configurations, "
                  f"{s.get('ledger_rows_upserted', 0)} ledger rows, "
                  f"{len(s.get('figures') or [])} figures -> {ctx.out_dir}")
    return 0


def cmd_ml_freeze(config: Config, args: argparse.Namespace) -> int:
    """Pre-registered freeze rules -> immutable MODEL_SPEC files (Prompt #10, Step 75)."""
    _require_research_extra()
    _require_ml_extra()
    from .research.ml_freeze import freeze_candidates
    from .research.ml_reports import collect_units
    from .research.ml_research import load_context

    cfgs = _ml_configs(config, args)
    ctx = load_context(cfgs, args.timeframe)
    folds, _ = collect_units(ctx.out_dir)
    if folds.is_empty():
        console.print("[red]no development results to freeze from[/red]")
        return 1
    report = freeze_candidates(ctx, folds)
    table = Table(title=f"{args.timeframe}: frozen model specifications (development "
                        "evidence only)", box=None)
    for name in ("target", "h", "eligible", "frozen", "window", "hyperparameters",
                 "calibration"):
        table.add_column(name)
    for d in report["decisions"]:
        table.add_row(d["target"], str(d["horizon"]), f"{d['eligible']}/{d['configurations']}",
                      str(d.get("frozen") or f"none - {d.get('reason')}"),
                      str(d.get("window_rule") or ""), str(d.get("hyperparameter_rule") or ""),
                      str(d.get("calibration_rule") or ""))
    console.print(table)
    console.print(f"MODEL_SPEC_FROZEN: {len(report['specs'])} specs under "
                  f"{ctx.out_dir / 'frozen'}")
    return 0


def cmd_ml_finalize(config: Config, args: argparse.Namespace) -> int:
    """Artifacts, reload / streaming equality and latency of the frozen specs (Steps 91-94)."""
    _require_research_extra()
    _require_ml_extra()
    from .research.ml_freeze import finalize
    from .research.ml_research import load_context

    cfgs = _ml_configs(config, args)
    ctx = load_context(cfgs, args.timeframe)
    out = finalize(ctx, streaming=not args.no_streaming)
    table = Table(title=f"{args.timeframe}: final development models", box=None)
    for name in ("model", "role", "artifact", "reload", "streaming", "size KB", "ms/row",
                 "ms/10k rows"):
        table.add_column(name)
    bad = 0
    for m in out["models"]:
        stream = m.get("streaming_safe")
        if m["reload_equal"] is False or stream is False:
            bad += 1
        table.add_row(m["model_id"], str(m.get("role")), m["artifact"], str(m["reload_equal"]),
                      "-" if stream is None else ("PASS" if stream else "FAIL"),
                      _fmt_metric((m.get("model_size_bytes") or 0) / 1e3, "{:.0f}"),
                      _fmt_metric(m.get("latency_ms_per_row"), "{:.3f}"),
                      _fmt_metric(m.get("latency_ms_per_10k_rows"), "{:.1f}"))
    console.print(table)
    if bad:
        console.print(f"[red]{bad} model(s) failed reload or streaming equality - not "
                      "live-ready[/red]")
    return 1 if bad else 0


def _resolve_model_specs(cfg: Any, items: list[str]) -> list[Path]:
    from .ml.final_test import FinalTestRefusedError

    out = []
    for item in items:
        p = Path(item)
        if p.suffix == ".json" and p.exists():
            out.append(p)
            continue
        name = item if item.startswith("MODEL_SPEC_") else f"MODEL_SPEC_{item}"
        hits = sorted(cfg.results_path.glob(f"*/frozen/{name}.json"))
        if not hits:
            raise FinalTestRefusedError(f"{item}: no frozen model spec - run `xq ml-freeze` first; "
                                   "the final test accepts frozen specs only")
        out.append(hits[0])
    return out


def cmd_ml_final_test(config: Config, args: argparse.Namespace) -> int:
    """One-time final test of explicitly frozen specs (Prompt #10, Steps 75-79)."""
    _require_research_extra()
    _require_ml_extra()
    from .ml.final_test import FinalTestRefusedError, run_final_test
    from .ml.registry import SpecError, load_frozen_spec
    from .research.ml_research import load_context

    cfgs = _ml_configs(config, args)
    cfg = cfgs.ml
    try:
        paths = _resolve_model_specs(cfg, list(args.model_spec))
        specs = [(p, load_frozen_spec(p)) for p in paths]
    except (FinalTestRefusedError, SpecError) as exc:
        console.print(f"[red]refused:[/red] {exc}")
        return 2
    by_tf: dict[str, list[Path]] = {}
    for p, s in specs:
        by_tf.setdefault(str(s["timeframe"]), []).append(p)
    console.print(f"[bold]Final test[/bold] of {len(specs)} frozen spec(s) on "
                  f"{cfgs.selection.periods.reserved_test.label()} - every access is logged")
    for tf, tf_paths in by_tf.items():
        ctx = load_context(cfgs, tf)
        try:
            results = run_final_test(cfgs, cfg, ctx.data, tf_paths,
                                     out_dir=cfg.results_path / tf / "final_test",
                                     repeat_reason=args.repeat_reason)
        except (FinalTestRefusedError, SpecError) as exc:
            console.print(f"[red]refused:[/red] {exc}")
            return 2
        table = Table(title=f"{tf}: reserved-period evaluation (development metrics, "
                            "nothing tuned)", box=None)
        for name in ("model", "rows", "AUC", "log-loss skill", "Brier", "rank IC",
                     "MSE skill", "R2", "recent 12m primary", "reload"):
            table.add_column(name)
        for r in results:
            ov, rc = r["overall"], r["recent_12_months"]
            cls = cfg.targets[r["target"]].is_classification
            table.add_row(r["model_id"], f"{r['rows']:,}", _fmt_metric(ov.get("auc"), "{:.4f}"),
                          _fmt_metric(ov.get("log_loss_skill"), "{:+.4f}"),
                          _fmt_metric(ov.get("brier"), "{:.4f}"),
                          _fmt_metric(ov.get("rank_ic"), "{:+.4f}"),
                          _fmt_metric(ov.get("mse_skill"), "{:+.4f}"),
                          _fmt_metric(ov.get("r2"), "{:+.4f}"),
                          _fmt_metric(rc.get("log_loss_skill" if cls else "rank_ic"),
                                      "{:+.4f}"),
                          str(r["reload_equal"]))
        console.print(table)
    console.print("[dim]Probabilities and expected values only - no threshold, signal or "
                  "PnL.[/dim]")
    return 0


# ---------------------------------------------------------------------------
# Prompt #11: ensembles, meta-models, predictive diversification
# ---------------------------------------------------------------------------
def _ensemble_configs(config: Config, args: argparse.Namespace) -> Any:
    from .ensemble.config import load_ensemble_config
    from .research.ensemble_research import EnsembleConfigs

    return EnsembleConfigs(ensemble=load_ensemble_config(args.ensemble_config),
                           ml=_ml_configs(config, args))


def _quiet_sklearn_parallel_warning() -> None:
    import warnings

    warnings.filterwarnings("ignore", message=r"`sklearn\.utils\.parallel\.delayed` should be "
                            r"used with `sklearn\.utils\.parallel\.Parallel`",
                            category=UserWarning)


def _ensemble_research_child(cfgs: Any, timeframe: str, pairs: list[tuple[str, int]] | None,
                             pipeline_nulls: bool, report: bool, write_ledger: bool
                             ) -> dict[str, Any]:
    """One timeframe's ensemble research (and report) - runs in a child process."""
    from .research.ensemble_research import run_ensemble_research

    _quiet_sklearn_parallel_warning()
    info = run_ensemble_research(cfgs, timeframe, pairs=pairs, pipeline_nulls=pipeline_nulls)
    if report:
        from .research.ensemble_reports import build_ensemble_report

        info["report"] = build_ensemble_report(cfgs, timeframe, ledger=write_ledger,
                                               plots=True)
    return info


def cmd_ensemble_research(config: Config, args: argparse.Namespace) -> int:
    """Ensemble research on the development period's out-of-sample predictions (#11)."""
    _require_research_extra()
    _require_ml_extra()
    cfgs = _ensemble_configs(config, args)
    ecfg = cfgs.ensemble
    pairs = None
    if args.target is not None:
        pairs = [(t, h) for t, h in ecfg.pairs if t == args.target
                 and (args.horizon is None or h == args.horizon)]
        if not pairs:
            console.print(f"[red]{args.target} h{args.horizon} is not a pair of "
                          f"{ecfg.config_path}[/red]")
            return 2
    timeframes = list(ecfg.timeframes) if args.all else [args.timeframe]
    reserved = cfgs.ml.selection.periods.reserved_test
    console.print(f"[bold]Ensemble research[/bold] (Prompt #11)\n"
                  f"  ensemble config : {ecfg.config_path} ({ecfg.fingerprint()})\n"
                  f"  inputs          : {ecfg.predictions_path}/<tf>/predictions (Prompt #10 "
                  "walk-forward out-of-sample predictions)\n"
                  f"  evaluation      : blocks {list(ecfg.evaluation_blocks)}, each fitted on "
                  f"earlier blocks only (purge h + {ecfg.embargo_bars} bars)\n"
                  f"  reserved test   : {reserved.label()} (never loaded here)\n"
                  f"  output          : {ecfg.results_path}")

    def run(tf: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"pairs": pairs, "pipeline_nulls": not args.no_pipeline_nulls,
                                  "report": args.report, "write_ledger": not args.no_ledger}
        if args.in_process:
            return _ensemble_research_child(cfgs, tf, **kwargs)
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_init_study_worker,
                                 initargs=(args.log_level,)) as pool:
            return pool.submit(_ensemble_research_child, cfgs, tf, **kwargs).result()

    if not args.in_process:
        os.environ.setdefault("MIMALLOC_PURGE_DELAY", "0")
    failures = []
    for tf in timeframes:
        try:
            info = run(tf)
        except MemoryError:
            raise
        except Exception as exc:
            LOGGER.exception("ensemble research %s failed", tf)
            failures.append((tf, f"{type(exc).__name__}: {exc}"))
            continue
        console.print(f"[green]{tf}[/green]: {len(info['pairs'])} pairs in "
                      f"{info['seconds']:.0f}s -> {ecfg.results_path / tf}")
        for f in info.get("failures") or []:
            failures.append((tf, f"{f['pair']}: {f['error']}"))
    for tf, message in failures:
        console.print(f"[red]FAILED[/red] {tf}: {message}")
    console.print("[dim]Development period only; probabilities, expected values and "
                  "uncertainty proxies - never a trade.[/dim]")
    return 1 if failures else 0


def cmd_ensemble_report(config: Config, args: argparse.Namespace) -> int:
    """Cross-pair tables, ENS-H ledger rows, figures, joint predictive state (#11)."""
    _require_research_extra()
    _require_ml_extra()
    from .research.ensemble_reports import build_ensemble_report

    cfgs = _ensemble_configs(config, args)
    s = build_ensemble_report(cfgs, args.timeframe, ledger=not args.no_ledger,
                              plots=not args.no_plots)
    console.print(f"[green]{args.timeframe}[/green]: {s.get('pairs', 0)} pairs, "
                  f"{s.get('experiments', 0)} ensemble experiments in the ledger, "
                  f"{len(s.get('figures') or [])} figures")
    return 0


def cmd_ensemble_build(config: Config, args: argparse.Namespace) -> int:
    """One explicit combination on the meta walk-forward (research; freezes nothing)."""
    _require_research_extra()
    _require_ml_extra()
    from .research.ensemble_research import score_combination

    cfgs = _ensemble_configs(config, args)
    models = [m for item in (args.models or []) for m in item.split(",") if m.strip()]
    try:
        table, used = score_combination(cfgs, args.timeframe, args.target, args.horizon,
                                        args.method, models or None)
    except (ValueError, KeyError) as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    task = cfgs.ml.ml.targets[args.target].task
    cols = _ML_COLUMNS[task]
    out = Table(title=f"{args.timeframe} {args.target} h{args.horizon}: {args.method} of "
                      f"{len(used)} model(s) - each block fitted on earlier blocks only",
                box=None)
    for name in ("block", "history rows", *[c for c, _ in cols]):
        out.add_column(name)
    for r in table.iter_rows(named=True):
        out.add_row(r["block"], f"{r['history_rows']:,}",
                    *[_fmt_metric(r.get(c), f) for c, f in cols])
    console.print(out)
    console.print(f"[dim]constituents: {', '.join(used)}. Research only - nothing frozen, no "
                  "trade.[/dim]")
    return 0


def _ensemble_tctx(cfgs: Any, timeframe: str) -> Any:
    from .research.ensemble_research import load_tf_context

    return load_tf_context(cfgs, timeframe, progress=lambda m: console.print(f"[dim]{m}[/dim]"))


def cmd_ensemble_freeze(config: Config, args: argparse.Namespace) -> int:
    """Pre-registered rule -> immutable ENSEMBLE_SPEC + constituent specs (#11, Step 60)."""
    _require_research_extra()
    _require_ml_extra()
    from .research.ensemble_freeze import freeze_ensembles

    cfgs = _ensemble_configs(config, args)
    report = freeze_ensembles(_ensemble_tctx(cfgs, args.timeframe))
    table = Table(title=f"{args.timeframe}: frozen ensembles (development evidence only)",
                  box=None)
    for name in ("target", "h", "frozen", "method", "role", "constituents", "calibration"):
        table.add_column(name)
    for d in report["decisions"]:
        table.add_row(d["target"], str(d["horizon"]), str(d.get("frozen") or "-"),
                      str(d.get("method") or d.get("reason") or ""), str(d.get("role") or ""),
                      str(len(d.get("constituents") or [])), str(d.get("calibration") or ""))
    console.print(table)
    console.print(f"ENSEMBLE_SPEC_FROZEN under {cfgs.ensemble.results_path / args.timeframe / 'frozen'}")
    return 0


def cmd_ensemble_finalize(config: Config, args: argparse.Namespace) -> int:
    """Constituent artifacts, reload / version / feature checks, streaming, benchmark."""
    _require_research_extra()
    _require_ml_extra()
    from .research.ensemble_freeze import finalize_ensembles

    cfgs = _ensemble_configs(config, args)
    _quiet_sklearn_parallel_warning()
    out = finalize_ensembles(_ensemble_tctx(cfgs, args.timeframe),
                             streaming=not args.no_streaming)
    table = Table(title=f"{args.timeframe}: frozen ensembles through the inference interface",
                  box=None)
    for name in ("ensemble", "method", "members", "reload", "streaming", "ms/row",
                 "ms/10k rows", "KB"):
        table.add_column(name)
    bad = 0
    for e in out["ensembles"]:
        stream = e.get("streaming_safe")
        if not e["reload_equal"] or not e["combination_equal"] or stream is False:
            bad += 1
        table.add_row(e["spec_id"], e["method"], str(e["constituents"]), str(e["reload_equal"]),
                      "-" if stream is None else ("PASS" if stream else "FAIL"),
                      _fmt_metric(e.get("latency_ms_per_row"), "{:.2f}"),
                      _fmt_metric(e.get("latency_ms_per_10k_rows"), "{:.0f}"),
                      _fmt_metric((e.get("artifact_bytes") or 0) / 1e3, "{:.0f}"))
    console.print(table)
    return 1 if bad else 0


def cmd_ensemble_final_test(config: Config, args: argparse.Namespace) -> int:
    """One-time final test of frozen ensemble specs (#11, Steps 62-63) - a second look."""
    _require_research_extra()
    _require_ml_extra()
    import gc

    from .ensemble.data import load_pair
    from .ensemble.final_test import (
        SECOND_LOOK_NOTE,
        EnsembleFinalTestRefusedError,
        run_ensemble_final_test,
        slim_development_data,
    )
    from .ensemble.registry import EnsembleSpecError, load_frozen_ensemble_spec

    cfgs = _ensemble_configs(config, args)
    ecfg, ml = cfgs.ensemble, cfgs.ml.ml
    paths = []
    try:
        for item in args.ensemble_spec:
            p = Path(item)
            if p.suffix != ".json" or not p.exists():
                name = item if item.startswith("ENSEMBLE_SPEC_") else f"ENSEMBLE_SPEC_{item}"
                hits = sorted(ecfg.results_path.glob(f"*/frozen/{name}.json"))
                if not hits:
                    raise EnsembleFinalTestRefusedError(
                        f"{item}: no frozen ensemble spec - run `xq ensemble-freeze` first")
                p = hits[0]
            paths.append((p, load_frozen_ensemble_spec(p)))
    except (EnsembleFinalTestRefusedError, EnsembleSpecError) as exc:
        console.print(f"[red]refused:[/red] {exc}")
        return 2
    console.print(f"[bold]Ensemble final test[/bold] of {len(paths)} frozen spec(s) on "
                  f"{cfgs.ml.selection.periods.reserved_test.label()} - logged.\n"
                  f"[yellow]{SECOND_LOOK_NOTE}[/yellow]")
    by_tf: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for p, s in paths:
        by_tf.setdefault(str(s["timeframe"]), []).append((p, s))
    for tf, items in by_tf.items():
        m = cfgs.ml
        # the labels at these horizons, the residual / volatility and two context columns -
        # not the development feature matrix the test never uses (memory, 2026-10-01)
        dev = slim_development_data(m, tf, sorted({int(s["horizon"]) for _, s in items}))
        pair_oos = {}
        for _, s in items:
            if s["method"] != "dynamic":
                continue
            tspec = ml.targets[s["target"]]
            pair = load_pair(ecfg.predictions_path / tf / "predictions"
                             / f"{s['target']}_h{s['horizon']}.parquet", timeframe=tf,
                             target=s["target"], horizon=int(s["horizon"]), task=tspec.task,
                             calibrated_inputs=ecfg.calibrated_only_for_classification,
                             reserved_start=dev.reserved_start)
            names = [c["name"] for c in s["constituents"]]
            pair_oos[s["spec_id"]] = {"p": pair.matrix(names), "y": pair.label,
                                      "base": pair.base, "rows": pair.rows,
                                      "timestamps": pair.timestamps}
        fin_path = ecfg.results_path / tf / "frozen" / "finalize.json"
        finalized = ({e["spec_id"]: e for e in json.loads(fin_path.read_text(
            encoding="utf-8")).get("ensembles", [])} if fin_path.exists() else {})
        try:
            results = run_ensemble_final_test(
                m, ml, ecfg, dev, [p for p, _ in items],
                out_dir=ecfg.results_path / tf / "final_test", repeat_reason=args.repeat_reason,
                pair_oos=pair_oos, finalized=finalized)
        except (EnsembleFinalTestRefusedError, EnsembleSpecError) as exc:
            console.print(f"[red]refused:[/red] {exc}")
            return 2
        table = Table(title=f"{tf}: reserved period (development metrics; nothing tuned)",
                      box=None)
        for name in ("ensemble", "method", "rows", "ensemble", "simple avg", "best indiv.",
                     "constant", "worst year", "invalid"):
            table.add_column(name)
        for r in results:
            task = ml.targets[r["summary"]["target"]].task
            metric = "log_loss_skill" if task == "classification" else "rank_ic"
            ov = r["overall"]
            table.add_row(r["spec_id"], r["summary"]["method"], f"{r['rows']:,}",
                          *[_fmt_metric((ov.get(k) or {}).get(metric), "{:+.4f}")
                            for k in ("ensemble", "simple_average", "best_individual",
                                      "constant")],
                          _fmt_metric(r["summary"].get("worst_year_metric"), "{:+.4f}"),
                          str(r["summary"].get("invalid_rows")))
        console.print(table)
        del dev, results
        gc.collect()
    console.print("[dim]Probabilities and expected values only - no threshold, signal or "
                  "PnL.[/dim]")
    return 0


def _init_study_worker(log_level: str | None) -> None:
    """Logging for a study's child process, which starts with none configured."""
    setup_logging(level=log_level)


def _fmt_metric(value: object, spec: str) -> str:
    """Format a research metric, or show a dash when missing or non-finite.

    Distinct from `_fmt`, which formats schema values and takes no spec.
    """
    if value is None:
        return "-"
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)
    return spec.format(number) if math.isfinite(number) else "-"


def cmd_execution(config: Config, args: argparse.Namespace) -> int:
    from .execution.runs import cli_command

    return cli_command(config, args)


def cmd_strategy_validation(config: Config, args: argparse.Namespace) -> int:
    """Lazy offline Stage 13 entry; prediction contracts remain unchanged."""
    from .strategy_validation.runs import cli_command
    return cli_command(config, args)


def cmd_robustness(config: Config, args: argparse.Namespace) -> int:
    from .robustness.runs import cli_command

    return cli_command(config, args)


def cmd_portfolio(config: Config, args: argparse.Namespace) -> int:
    from .alpha_portfolio.runs import cli_command

    return cli_command(config, args)


def cmd_risk(config: Config, args: argparse.Namespace) -> int:
    from .risk.runs import cli_command

    return cli_command(config, args)


def cmd_shadow(config: Config, args: argparse.Namespace) -> int:
    from .shadow.runs import cli_command

    return cli_command(config, args)


def cmd_econometrics(config: Config, args: argparse.Namespace) -> int:
    """Lazy Stage 13.5 entry; offline diagnostics, unchanged prediction contracts."""
    from .econometrics.runs import cli_command
    return cli_command(config, args)


COMMANDS = {
    "info": cmd_info,
    "inspect": cmd_inspect,
    "detect-schema": cmd_detect_schema,
    "validate": cmd_validate,
    "convert": cmd_convert,
    "source-index": cmd_source_index,
    "verify-dataset": cmd_verify_dataset,
    "metadata": cmd_metadata,
    "build-bars": cmd_build_bars,
    "diagnose": cmd_diagnose,
    "research-summary": cmd_research_summary,
    "regression-research": cmd_regression_research,
    "build-features": cmd_build_features,
    "ou-research": cmd_ou_research,
    "spectral-research": cmd_spectral_research,
    "wavelet-research": cmd_wavelet_research,
    "regime-research": cmd_regime_research,
    "regime-walk-forward": cmd_regime_walk_forward,
    "build-feature-factory": cmd_build_feature_factory,
    "build-feature-matrix": cmd_build_feature_factory,
    "feature-research": cmd_feature_research,
    "feature-redundancy": cmd_feature_redundancy,
    "alpha-decay": cmd_alpha_decay,
    "feature-select": cmd_feature_select,
    "feature-selection-report": cmd_feature_selection_report,
    "build-feature-manifest": cmd_build_feature_manifest,
    "ml-research": cmd_ml_research,
    "ml-train": cmd_ml_train,
    "ml-walk-forward": cmd_ml_walk_forward,
    "ml-compare": cmd_ml_compare,
    "ml-report": cmd_ml_report,
    "ml-freeze": cmd_ml_freeze,
    "ml-finalize": cmd_ml_finalize,
    "ml-final-test": cmd_ml_final_test,
    "ensemble-research": cmd_ensemble_research,
    "ensemble-report": cmd_ensemble_report,
    "ensemble-build": cmd_ensemble_build,
    "ensemble-freeze": cmd_ensemble_freeze,
    "ensemble-finalize": cmd_ensemble_finalize,
    "ensemble-final-test": cmd_ensemble_final_test,
    "robustness-plan": cmd_robustness,
    "robustness-inventory": cmd_robustness,
    "robustness-smoke": cmd_robustness,
    "robustness-evaluate": cmd_robustness,
    "portfolio-plan": cmd_portfolio,
    "alpha-registry": cmd_portfolio,
    "portfolio-smoke": cmd_portfolio,
    "portfolio-evaluate": cmd_portfolio,
    "risk-plan": cmd_risk,
    "risk-readiness": cmd_risk,
    "risk-replay": cmd_risk,
    "shadow-validate": cmd_shadow,
    "shadow-preflight": cmd_shadow,
    "shadow-capture": cmd_shadow,
    "shadow-run": cmd_shadow,
    "shadow-replay": cmd_shadow,
    "shadow-compare": cmd_shadow,
    "econometric-plan": cmd_econometrics,
    "econometric-readiness": cmd_econometrics,
    "econometric-smoke": cmd_econometrics,
    "econometric-evaluate": cmd_econometrics,
    "strategy-plan": cmd_strategy_validation,
    "strategy-readiness": cmd_strategy_validation,
    "strategy-smoke": cmd_strategy_validation,
    "strategy-validate": cmd_strategy_validation,
    "execution-smoke": cmd_execution,
    "execution-readiness": cmd_execution,
    "execution-diagnostic": cmd_execution,
    "execution-backtest": cmd_execution,
    "query": cmd_query,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(level=args.log_level)
    try:
        overrides = {"raw_data_path": args.raw_path} if args.raw_path else None
        config = load_config(args.config, overrides=overrides)
    except ConfigError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        return 2

    try:
        return COMMANDS[args.command](config, args)
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        LOGGER.error("%s: %s", type(exc).__name__, exc)
        console.print(f"[red]{type(exc).__name__}:[/red] {exc}")
        return 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Completed partitions are recorded in the "
                      "manifest; re-run the same command to resume.[/yellow]")
        return 130


if __name__ == "__main__":
    sys.exit(main())
