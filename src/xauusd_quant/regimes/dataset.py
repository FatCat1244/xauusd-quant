r"""The regime feature space: a compact, causal, documented set of inputs (Steps 2, 4, 5).

Every candidate feature is registered in :data:`FEATURE_SPECS` with its group,
definition, lookback and the reason it was included; features deliberately
left out are in :data:`EXCLUDED_FEATURES` with the reason. Both are written to
each study as ``feature_manifest.json``. Every value at bar ``t`` is computed
from bars ``<= t``:

========================  =====================================================
``return_z_20``           :math:`(\ln P_t - \ln P_{t-h}) / (\sigma_t \sqrt h)`,
                          ``h = 20`` bars, ``sigma`` the 50-bar trailing return sd
``log_rv_20``             :math:`\tfrac12 \ln \overline{r^2}` over 20 bars
``rv_percentile``         rank of ``log_rv_20`` among the last 20 trading days
``abs_return_acf1``       rolling lag-1 autocorrelation of :math:`|r|` (256 bars)
``slope_over_volatility`` Prompt #3 slope (N = 128) / trailing return sd
``r_squared``             Prompt #3 R^2 (N = 128)
``log_ou_half_life``      ln of the Prompt #4 rolling OU half-life (valid fits only)
``spectral_entropy``      Prompt #5 FFT entropy of log returns (N = 256), stored
``fft_high_low_log_ratio`` ln(high-band / low-band FFT power share), stored
``wavelet_entropy``       Prompt #6 wavelet entropy of log returns (N = 512), stored
``wavelet_fast_slow_log_ratio`` Prompt #6 ln(fast / slow energy), stored
``spread_percentile``     rank of the bar's median spread among the last 20 days
``activity_percentile``   rank of the bar's tick count among the last 20 days
========================  =====================================================

The FFT and wavelet features are **read from the stored, versioned Prompt #5
and #6 feature sets**, never recomputed: :func:`integrity_gate` refuses a
feature set whose tick, bar or engine version differs from the current ones,
and additionally recomputes the last few thousand bars and requires them to
match the stored values exactly. The pipeline null controls, which have no
stored features, run the same engines on their own price paths.
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl

from ..data.resampler import parse_timeframe
from ..data.versioning import dataset_lineage
from ..features.config import RegressionConfig
from ..features.spectral import rolling_spectrum, spectral_feature_frame
from ..features.spectral_config import SpectralConfig
from ..features.store import RegressionFeatureStore, StaleFeaturesError
from ..features.wavelet_causal import rolling_wavelet, wavelet_feature_frame
from ..features.wavelet_config import WaveletConfig
from ..models.config import OUConfig
from ..research.spectral_nulls import SourceData, build_control, build_real_source
from ..research.study_io import clean_json, code_fingerprint
from ..utils.clock import utc_now_iso
from ..utils.config import Config
from ..utils.logging import get_logger
from ..utils.paths import atomic_write_text, ensure_dir
from .config import RegimeConfig

__all__ = [
    "AUXILIARY_COLUMNS",
    "EXCLUDED_FEATURES",
    "FEATURE_SPECS",
    "FeatureSpec",
    "build_regime_inputs",
    "feature_manifest",
    "feature_matrix",
    "integrity_gate",
    "load_or_build_regime_inputs",
    "null_regime_inputs",
    "redundancy_table",
    "source_from_frame",
]

LOGGER = get_logger("regimes.dataset")

#: Bumped whenever a feature's definition changes (part of the cache key).
INPUTS_VERSION = 1


@dataclass(frozen=True)
class FeatureSpec:
    """One regime input: what it is, where it comes from, why it is in the model."""

    group: str
    description: str
    source: str
    lookback: str
    rationale: str
    causal: bool = True


FEATURE_SPECS: dict[str, FeatureSpec] = {
    "return_z_20": FeatureSpec(
        "returns", "(ln P_t - ln P_{t-20}) / (trailing 50-bar return sd x sqrt(20))",
        "bars (close), Prompt #3 trailing volatility", "50 bars",
        "Short-horizon directional drift in volatility units: the price/return phenomenon "
        "without a heavy-tailed single-bar return."),
    "log_rv_20": FeatureSpec(
        "volatility", "0.5 ln(mean r^2 over the last 20 bars)", "log returns", "20 bars",
        "The level of realised volatility; the variable the volatility-bucket baseline "
        "buckets. Replaces the 50-bar trailing sd (Spearman 0.80-0.88, Prompt #6)."),
    "rv_percentile": FeatureSpec(
        "volatility", "rank of log_rv_20 among the last 20 trading days, (rank - 0.5) / n",
        "log_rv_20", "20 trading days",
        "Volatility relative to its own recent history: separates 'high for this era' from "
        "'high in absolute terms', which drifts with the 2003-2026 eras."),
    "abs_return_acf1": FeatureSpec(
        "volatility", "rolling corr(|r_t|, |r_{t-1}|) over 256 bars", "log returns",
        "256 bars", "Absolute-return persistence: how strongly volatility clusters now "
        "(|eta| lag-1 ACF 0.28-0.40 was XAUUSD-specific in Prompt #4)."),
    "slope_over_volatility": FeatureSpec(
        "regression", "rolling-regression slope (N=128) / trailing return sd",
        "Prompt #3 feature store", "128 bars",
        "Trend strength in volatility units (model A of Prompt #6 uses the same ratio)."),
    "r_squared": FeatureSpec(
        "regression", "rolling-regression R^2 (N=128)", "Prompt #3 feature store", "128 bars",
        "How linear the recent path is; unsigned, so distinct from the slope (|Spearman| with "
        "|slope| 0.82-0.85: related, not a duplicate)."),
    "log_ou_half_life": FeatureSpec(
        "ou", "ln rolling OU half-life of the residual (M=256), valid fits only",
        "Prompt #4 rolling OU fit", "128 + 256 bars",
        "Mean-reversion speed of the residual. Theta = ln 2 / half-life is the same "
        "information and is not included twice. Largely mechanical (Prompt #4); included "
        "so the model can test whether states separate it."),
    "spectral_entropy": FeatureSpec(
        "spectral", "normalised FFT spectral entropy of log returns (N=256)",
        "stored Prompt #5 feature set", "256 bars",
        "Spectral concentration. Flatness (Spearman 0.86-0.88) and top-k shares "
        "(0.69-0.97) measure the same phenomenon and are left out."),
    "fft_high_low_log_ratio": FeatureSpec(
        "spectral", "ln(high-band / low-band FFT power share) of log returns (N=256)",
        "stored Prompt #5 feature set", "256 bars",
        "Where the spectral power sits (fast vs slow). The centroid duplicates the high share "
        "(Spearman 0.89-0.90) and is left out."),
    "wavelet_entropy": FeatureSpec(
        "wavelet", "normalised wavelet energy entropy of log returns (N=512)",
        "stored Prompt #6 feature set", "512 bars",
        "Multiscale energy concentration. Its centroid period is a near-duplicate "
        "(Spearman 0.995) and the top-3 share nearly so (-0.90); both are left out."),
    "wavelet_fast_slow_log_ratio": FeatureSpec(
        "wavelet", "ln(energy in bands < 4 h / energy in bands >= 4 h) of log returns (N=512)",
        "stored Prompt #6 feature set", "512 bars",
        "Fast vs slow energy: the one wavelet quantity with structure beyond the nulls "
        "(its intraday shift, Prompt #6)."),
    "spread_percentile": FeatureSpec(
        "spread_activity", "rank of the bar's median spread among the last 20 trading days",
        "bars (median_spread)", "20 trading days",
        "Execution environment relative to recent history (spreads are discrete and shift "
        "between eras, so a rank, not a level)."),
    "activity_percentile": FeatureSpec(
        "spread_activity", "rank of the bar's tick count among the last 20 trading days",
        "bars (tick_count)", "20 trading days",
        "Quote activity relative to recent history (tick counts rise ~50x from 2003 to 2026, "
        "so a rank, not a level)."),
}

#: Candidates considered and left out of the regime inputs, with the reason.
EXCLUDED_FEATURES: dict[str, str] = {
    "log_return": "a single-bar return is near-i.i.d. noise with kurtosis 24-34 (Prompt #4); "
                  "in a Gaussian mixture it produces tail-driven, flickering components. Its "
                  "information enters through return_z_20 and log_rv_20.",
    "residual_zscore": "a fast-oscillating position variable (half-life ~0.18 N bars), not a "
                       "state descriptor; it is the conditioning variable of the Step 33 "
                       "mean-reversion tables, so using it to define states would make them "
                       "circular.",
    "ou_zscore": "as residual_zscore: an oscillating position within the OU band.",
    "ou_theta": "ln 2 / half-life: the same information as log_ou_half_life.",
    "ou_valid": "binary; in a Gaussian state it gives a zero-variance coordinate (covariance "
                "collapse). Invalid fits are missing values of log_ou_half_life instead, and "
                "the valid share is reported per state.",
    "trailing_volatility": "duplicates log_rv_20 (Spearman 0.80-0.88).",
    "spectral_flatness": "duplicates spectral_entropy (Spearman 0.86-0.88).",
    "top3_power_share": "the same concentration phenomenon as spectral_entropy (-0.69 to "
                        "-0.72); kept as a descriptive column.",
    "spectral_centroid_norm": "duplicates the FFT high-band share (0.89-0.90).",
    "fft_dominant_period": "a discrete bin, fixed in bars by the window (Prompt #5: every "
                           "dominant period is a window artefact); kept as a descriptive column.",
    "wavelet_log_centroid_period": "near-duplicate of wavelet_entropy (Spearman 0.995).",
    "wavelet_top3_scale_share": "duplicates wavelet_entropy (-0.90); descriptive column.",
    "wavelet_dominant_period": "a discrete band that does not survive a change of wavelet "
                               "family (Spearman -0.07 to 0.35, Prompt #6); descriptive column.",
    "wavelet_dominant_run_length": "the persistence of that unstable dominant band; a capped "
                                   "count. Descriptive column.",
    "hour_of_day": "time is an explicit, known variable, not a market state: regimes are read "
                   "against it (Step 38) and the information test carries it in model A.",
}

#: Descriptive and outcome columns kept beside the inputs (never model inputs).
AUXILIARY_COLUMNS: tuple[str, ...] = (
    "log_return", "regression_residual", "residual_zscore", "regression_slope",
    "trailing_volatility", "ou_innovation", "ou_zscore", "ou_half_life_bars", "ou_valid",
    "ou_state_code", "ou_theta", "median_spread", "mean_spread", "tick_count",
    "fft_top3_power_share", "fft_dominant_period_bars", "wavelet_top3_scale_share",
    "wavelet_dominant_period_bars", "wavelet_dominant_run_length", "wavelet_scale_drift",
)

_INPUT_FILE = "inputs.parquet"
_MANIFEST = "_manifest.json"
_SOURCES = ("regimes/dataset.py", "research/spectral_nulls.py", "features/spectral.py",
            "features/wavelet_causal.py")


# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------
def _pipeline_settings(regime: RegimeConfig, block_size: int = 256) -> SimpleNamespace:
    """What the shared source builders read (regression/OU windows, null seed)."""
    return SimpleNamespace(regression_window=regime.inputs.regression_window,
                           ou_window=regime.inputs.ou_window,
                           controls=SimpleNamespace(seed=regime.nulls.seed,
                                                    block_size=block_size))


def _fft_root(spectral: SpectralConfig, regime: RegimeConfig, timeframe: str) -> Path:
    return (spectral.features_path / f"timeframe={timeframe}"
            / f"source={regime.inputs.fft_series}" / f"fft_window={regime.inputs.fft_window}")


def _wavelet_root(wavelet: WaveletConfig, regime: RegimeConfig, timeframe: str) -> Path:
    return (wavelet.features_path / f"timeframe={timeframe}"
            / f"source={regime.inputs.wavelet_series}" / f"window={regime.inputs.wavelet_window}")


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _code_version() -> str:
    base = Path(__file__).resolve().parent.parent
    return code_fingerprint(base / rel for rel in _SOURCES)


# ---------------------------------------------------------------------------
# Integrity gate (Step 2)
# ---------------------------------------------------------------------------
_FFT_COLUMNS = ("timestamp", "spectral_entropy", "low_power_share", "high_power_share",
                "top3_power_share", "fft_period_bars_1")
_WAVELET_COLUMNS = ("timestamp", "wavelet_entropy", "wavelet_fast_slow_log_ratio",
                    "wavelet_top3_scale_share", "wavelet_dominant_period_bars",
                    "wavelet_dominant_run_length", "wavelet_scale_drift")


def _stored(root: Path, columns: tuple[str, ...], pattern: str) -> pl.DataFrame:
    files = sorted(root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no stored features under {root}")
    return pl.read_parquet(files, columns=list(columns)).sort("timestamp")


def integrity_gate(config: Config, regression: RegressionConfig, ou: OUConfig,
                   spectral: SpectralConfig, wavelet: WaveletConfig, regime: RegimeConfig,
                   timeframe: str, *, recompute_check_bars: int = 3000,
                   source: SourceData | None = None) -> dict[str, Any]:
    """Refuse to fit regimes on anything but the current canonical dataset.

    Checks: the tick dataset is complete; the bars were built from it; the
    rolling-regression features (the configured window) are current; the
    stored FFT and wavelet feature sets carry the current tick and bar
    versions, the current engine fingerprints and every bar; and a fresh
    computation of the last *recompute_check_bars* bars reproduces the stored
    values exactly. The OU fit is recomputed from the current regression
    features, so it cannot be stale. Returns ``passed``, ``problems`` (what
    must be rebuilt), ``notes`` and the versions used.
    """
    lineage = dataset_lineage(config, timeframe)
    problems: list[str] = []
    notes: list[str] = []
    versions: dict[str, Any] = {}
    if lineage.get("tick_dataset_status") != "complete":
        problems.append(f"tick dataset is {lineage.get('tick_dataset_status')!r}, not complete "
                        "- run `xq convert` then `xq verify-dataset`")
    if not lineage.get("bars_match_ticks"):
        problems.append(f"{timeframe} bars were not built from the current tick dataset - run "
                        f"`xq build-bars --timeframe {timeframe}`")
    if lineage.get("partial"):
        problems.append(str(lineage.get("label")))
    store = RegressionFeatureStore(config, regression)
    window = regime.inputs.regression_window
    entry = ((store.manifest(timeframe) or {}).get("windows") or {}).get(str(window))
    if entry is None:
        problems.append(f"no regression features for {timeframe} N={window} - run "
                        f"`xq build-features --timeframe {timeframe} --window {window}`")
    else:
        try:
            store.load(timeframe, window, ["timestamp"])
            versions["regression_feature_version"] = entry.get("feature_version")
            notes.append(f"regression features N={window}: current "
                         f"({entry.get('feature_version')})")
        except StaleFeaturesError as exc:
            problems.append(f"regression features are stale ({exc}) - run `xq build-features "
                            f"--timeframe {timeframe} --window {window} --force`")
    for kind, root, engine, rebuild in (
        ("fft", _fft_root(spectral, regime, timeframe), spectral.engine_fingerprint(),
         f"`xq spectral-research --timeframe {timeframe} --source {regime.inputs.fft_series} "
         f"--fft-window {regime.inputs.fft_window}`"),
        ("wavelet", _wavelet_root(wavelet, regime, timeframe), wavelet.engine_fingerprint(),
         f"`xq wavelet-research --timeframe {timeframe} --source "
         f"{regime.inputs.wavelet_series} --window {regime.inputs.wavelet_window}`"),
    ):
        manifest = _read_manifest(root / _MANIFEST)
        if manifest is None:
            problems.append(f"no stored {kind} features at {root} - run {rebuild}")
            continue
        mismatches = [name for name, ok in (
            ("tick version", manifest.get("tick_dataset_version")
             == lineage.get("tick_dataset_version")),
            ("bar version", manifest.get("bar_dataset_version")
             == lineage.get("bar_dataset_version")),
            ("engine", manifest.get("engine_fingerprint") == engine),
            ("rows", manifest.get("rows") == lineage.get("bar_rows")),
        ) if not ok]
        if mismatches:
            problems.append(f"stored {kind} features are stale ({', '.join(mismatches)}) - "
                            f"run {rebuild}")
        else:
            versions[f"{kind}_features"] = {
                "path": root.as_posix(), "engine_fingerprint": engine,
                "generated_utc": manifest.get("generated_utc"), "rows": manifest.get("rows")}
            notes.append(f"stored {kind} features: current (engine {engine})")
    if not problems and recompute_check_bars > 0:
        check = _recompute_check(config, regression, ou, spectral, wavelet, regime, timeframe,
                                 bars=recompute_check_bars, source=source)
        versions["recompute_check"] = check
        if not check["matches"]:
            problems.append(f"stored FFT/wavelet values differ from a fresh computation of the "
                            f"last {recompute_check_bars} bars ({check['detail']}) - rebuild them")
        else:
            notes.append(f"last {recompute_check_bars} bars recomputed: identical to the stored "
                         "FFT and wavelet features")
    return {"passed": not problems, "problems": problems, "notes": notes,
            "lineage": lineage, "versions": versions}


def _recompute_check(config: Config, regression: RegressionConfig, ou: OUConfig,
                     spectral: SpectralConfig, wavelet: WaveletConfig, regime: RegimeConfig,
                     timeframe: str, *, bars: int, source: SourceData | None) -> dict[str, Any]:
    """Recompute the tail of the FFT and wavelet features and compare with the stored ones."""
    src = source or build_real_source(config, regression, ou, _pipeline_settings(regime),
                                      timeframe)
    n = src.size
    values = src.columns[regime.inputs.fft_series]
    n_fft, n_wav = regime.inputs.fft_window, regime.inputs.wavelet_window
    lo = max(0, n - bars - n_fft)
    spec = rolling_spectrum(values[lo:], fft_window=n_fft, config=spectral,
                            bar_seconds=src.bar_seconds, missing_slots=src.missing_slots[lo:])
    fresh_fft = spectral_feature_frame(spec, src.timestamps.slice(lo), spectral).tail(bars)
    values = src.columns[regime.inputs.wavelet_series]
    lo = max(0, n - bars - 2 * n_wav)
    wav = rolling_wavelet(values[lo:], window=n_wav, config=wavelet, bar_seconds=src.bar_seconds,
                          missing_slots=src.missing_slots[lo:])
    fresh_wav = wavelet_feature_frame(wav, src.timestamps.slice(lo),
                                      float32=wavelet.features_output.float32).tail(bars)
    first = fresh_fft["timestamp"][0]
    stored_fft = (_stored(_fft_root(spectral, regime, timeframe), _FFT_COLUMNS[:4],
                          "year=*/features.parquet").filter(pl.col("timestamp") >= first))
    stored_wav = (_stored(_wavelet_root(wavelet, regime, timeframe), _WAVELET_COLUMNS[:3],
                          "year=*/part-0.parquet").filter(pl.col("timestamp") >= first))
    detail = []
    ok = True
    worst = 0.0
    # FFT windows are computed independently: identical to the last bit. Wavelet
    # energies are prefix-sum differences, so starting the recomputation later
    # changes their rounding (never their value): compared to 1e-6.
    for fresh, stored, columns, tol in ((fresh_fft, stored_fft, _FFT_COLUMNS[1:4], 0.0),
                                        (fresh_wav, stored_wav, _WAVELET_COLUMNS[1:3], 1e-6)):
        if fresh.height != stored.height or not fresh["timestamp"].equals(stored["timestamp"]):
            ok = False
            detail.append("timestamps differ")
            continue
        for column in columns:
            a = fresh[column].cast(pl.Float64).fill_null(np.nan).to_numpy()
            b = stored[column].cast(pl.Float64).fill_null(np.nan).to_numpy()
            same_missing = bool(np.array_equal(np.isnan(a), np.isnan(b)))
            both = ~np.isnan(a) & ~np.isnan(b)
            diff = float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0
            worst = max(worst, diff)
            if not same_missing or diff > tol:
                ok = False
                detail.append(f"{column} (max |diff| {diff:.2e})")
    return {"bars": bars, "matches": ok, "max_abs_difference": worst,
            "detail": ",".join(detail) or "identical (FFT exactly, wavelet to 1e-6)"}


# ---------------------------------------------------------------------------
# Building the inputs
# ---------------------------------------------------------------------------
def _prefix_span_sum(values: np.ndarray, width: int) -> np.ndarray:
    """Sum of the last *width* values (NaN if any is NaN or fewer exist)."""
    x = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(x)
    total = np.concatenate(([0.0], np.cumsum(np.where(finite, x, 0.0))))
    bad = np.concatenate(([0], np.cumsum(~finite, dtype=np.int64)))
    out = np.full(x.size, np.nan)
    if width <= x.size:
        span = total[width:] - total[:-width]
        clean = (bad[width:] - bad[:-width]) == 0
        out[width - 1:] = np.where(clean, span, np.nan)
    return out


def _trailing_mean(values: np.ndarray, width: int) -> np.ndarray:
    from ..research.wavelet_predictiveness import trailing_mean

    return trailing_mean(values, width)


def _rolling_percentile(values: np.ndarray, window: int) -> np.ndarray:
    """(rank - 0.5) / n of each value among the last *window* values (average ties).

    Windows need 90 % of their values; missing values are ranked out. Polars'
    rolling rank is exact (checked against brute force in the tests).
    """
    s = pl.Series("x", np.asarray(values, dtype=np.float64)).fill_nan(None)
    need = max(2, int(0.9 * window))
    frame = pl.DataFrame({"x": s}).select(
        pl.col("x").rolling_rank(window_size=window, method="average",
                                 min_samples=need).alias("rank"),
        pl.col("x").is_not_null().cast(pl.Float64).rolling_sum(
            window_size=window, min_samples=1).alias("count"),
        pl.col("x").is_null().alias("missing"))
    rank = frame["rank"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    count = frame["count"].fill_null(np.nan).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (rank - 0.5) / count
    out[frame["missing"].to_numpy()] = np.nan
    out[:window - 1] = np.nan                    # only whole windows, never a partial start
    return out


def _rolling_abs_acf1(returns: np.ndarray, window: int) -> np.ndarray:
    a = pl.Series("a", np.abs(np.asarray(returns, dtype=np.float64))).fill_nan(None)
    return (pl.DataFrame({"a": a}).select(
        pl.rolling_corr(pl.col("a"), pl.col("a").shift(1), window_size=window,
                        min_samples=window)).to_series().fill_null(np.nan).to_numpy())


def _derive(columns: dict[str, np.ndarray], fft: pl.DataFrame, wav: pl.DataFrame,
            bars: pl.DataFrame, regime: RegimeConfig, bar_seconds: float) -> dict[str, np.ndarray]:
    """Every registered input and auxiliary column from one source's series (real or null)."""
    cfg = regime.inputs
    r = np.asarray(columns["log_return"], dtype=np.float64)
    sigma = np.asarray(columns["trailing_volatility"], dtype=np.float64)
    pct_window = cfg.percentile_window(bar_seconds)
    with np.errstate(divide="ignore", invalid="ignore"):
        move = _prefix_span_sum(r, cfg.return_horizon)
        return_z = np.where(sigma > 0, move / (sigma * np.sqrt(cfg.return_horizon)), np.nan)
        mean_sq = _trailing_mean(r ** 2, cfg.rv_window)
        log_rv = np.where(mean_sq > 0, 0.5 * np.log(mean_sq), np.nan)
        slope = np.asarray(columns["regression_slope"], dtype=np.float64)
        slope_vol = np.where(sigma > 0, slope / sigma, np.nan)
        valid = np.asarray(columns["ou_valid"], dtype=np.float64) > 0
        half_life = np.asarray(columns["ou_half_life_bars"], dtype=np.float64)
        log_hl = np.where(valid & (half_life > 0), np.log(half_life), np.nan)
        theta = np.where(valid & (half_life > 0), np.log(2.0) / half_life, np.nan)
        high = fft["high_power_share"].cast(pl.Float64).fill_null(np.nan).to_numpy()
        low = fft["low_power_share"].cast(pl.Float64).fill_null(np.nan).to_numpy()
        high_low = np.where((high > 0) & (low > 0), np.log(high) - np.log(low), np.nan)

    def f64(frame: pl.DataFrame, name: str) -> np.ndarray:
        return frame[name].cast(pl.Float64).fill_null(np.nan).to_numpy()

    median_spread = f64(bars, "median_spread")
    tick_count = f64(bars, "tick_count")
    out: dict[str, np.ndarray] = {
        "return_z_20": return_z,
        "log_rv_20": log_rv,
        "rv_percentile": _rolling_percentile(log_rv, pct_window),
        "abs_return_acf1": _rolling_abs_acf1(r, cfg.abs_return_acf_window),
        "slope_over_volatility": slope_vol,
        "r_squared": np.asarray(columns["r_squared"], dtype=np.float64),
        "log_ou_half_life": log_hl,
        "spectral_entropy": f64(fft, "spectral_entropy"),
        "fft_high_low_log_ratio": high_low,
        "wavelet_entropy": f64(wav, "wavelet_entropy"),
        "wavelet_fast_slow_log_ratio": f64(wav, "wavelet_fast_slow_log_ratio"),
        "spread_percentile": _rolling_percentile(median_spread, pct_window),
        "activity_percentile": _rolling_percentile(tick_count, pct_window),
        # auxiliary (descriptive / outcome) columns
        "log_return": r,
        "regression_residual": np.asarray(columns["regression_residual"], dtype=np.float64),
        "residual_zscore": np.asarray(columns["residual_zscore"], dtype=np.float64),
        "regression_slope": slope,
        "trailing_volatility": sigma,
        "ou_innovation": np.asarray(columns["ou_innovation"], dtype=np.float64),
        "ou_zscore": np.asarray(columns["ou_zscore"], dtype=np.float64),
        "ou_half_life_bars": np.where(valid, half_life, np.nan),
        "ou_valid": valid.astype(np.float64),
        "ou_state_code": np.asarray(columns["ou_state_code"], dtype=np.float64),
        "ou_theta": theta,
        "median_spread": median_spread,
        "mean_spread": f64(bars, "mean_spread"),
        "tick_count": tick_count,
        "fft_top3_power_share": f64(fft, "top3_power_share"),
        "fft_dominant_period_bars": f64(fft, "fft_period_bars_1"),
        "wavelet_top3_scale_share": f64(wav, "wavelet_top3_scale_share"),
        "wavelet_dominant_period_bars": f64(wav, "wavelet_dominant_period_bars"),
        "wavelet_dominant_run_length": f64(wav, "wavelet_dominant_run_length"),
        "wavelet_scale_drift": f64(wav, "wavelet_scale_drift"),
    }
    for name, values in out.items():                 # +-inf (e.g. ln 0) is missing, not a value
        if np.isinf(values).any():                   # copy: some arrays are the source's own
            out[name] = np.where(np.isinf(values), np.nan, values)
    return out


def _bars(config: Config, timeframe: str) -> pl.DataFrame:
    files = sorted(config.bars_dir(timeframe).rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {timeframe} bars under {config.bars_dir(timeframe)}")
    return (pl.scan_parquet(files).select("timestamp", "tick_count", "median_spread",
                                          "mean_spread").sort("timestamp").collect())


def _aligned(frame: pl.DataFrame, timestamps: pl.Series, what: str) -> pl.DataFrame:
    """*frame* in the exact row order of *timestamps* (refuses any mismatch)."""
    if frame.height != timestamps.len() or not frame["timestamp"].equals(timestamps):
        joined = pl.DataFrame({"timestamp": timestamps}).join(frame, on="timestamp", how="left")
        if joined.height != timestamps.len():
            raise RuntimeError(f"{what}: duplicate timestamps in the stored features")
        missing = int(joined.select(pl.col(frame.columns[1]).is_null().sum()).item())
        LOGGER.warning("%s: joined on timestamp; %d bars have no stored row", what, missing)
        return joined
    return frame


def build_regime_inputs(config: Config, regression: RegressionConfig, ou: OUConfig,
                        spectral: SpectralConfig, wavelet: WaveletConfig, regime: RegimeConfig,
                        timeframe: str, *, source: SourceData | None = None) -> pl.DataFrame:
    """Every regime input and auxiliary column for the real series, one row per bar."""
    src = source or build_real_source(config, regression, ou, _pipeline_settings(regime),
                                      timeframe)
    stamps = src.timestamps
    fft = _aligned(_stored(_fft_root(spectral, regime, timeframe), _FFT_COLUMNS,
                           "year=*/features.parquet"), stamps, "stored FFT features")
    wav = _aligned(_stored(_wavelet_root(wavelet, regime, timeframe), _WAVELET_COLUMNS,
                           "year=*/part-0.parquet"), stamps, "stored wavelet features")
    bars = _aligned(_bars(config, timeframe), stamps, "bars")
    columns = _derive(src.columns, fft, wav, bars, regime, src.bar_seconds)
    return pl.DataFrame({"timestamp": stamps, **columns})


def null_regime_inputs(name: str, real: SourceData, real_frame: pl.DataFrame,
                       regression: RegressionConfig, ou: OUConfig, spectral: SpectralConfig,
                       wavelet: WaveletConfig, regime: RegimeConfig) -> pl.DataFrame:
    """The same inputs for one pipeline null control (a price path with no real structure).

    The null's log returns go through the same regression, OU, FFT (N=256) and
    wavelet (N=512) engines as the stored real features, with the same float32
    storage precision. Spread and activity are exogenous to the price path and
    are taken from the real bars, timestamp for timestamp.
    """
    control = build_control(name, real, regression=regression, ou=ou,
                            spectral=_pipeline_settings(regime))
    values = control.columns[regime.inputs.fft_series]
    spec = rolling_spectrum(values, fft_window=regime.inputs.fft_window, config=spectral,
                            bar_seconds=control.bar_seconds, missing_slots=control.missing_slots)
    fft = spectral_feature_frame(spec, control.timestamps, spectral).select(_FFT_COLUMNS)
    del spec
    values = control.columns[regime.inputs.wavelet_series]
    wav_result = rolling_wavelet(values, window=regime.inputs.wavelet_window, config=wavelet,
                                 bar_seconds=control.bar_seconds,
                                 missing_slots=control.missing_slots)
    wav = wavelet_feature_frame(wav_result, control.timestamps,
                                float32=wavelet.features_output.float32).select(_WAVELET_COLUMNS)
    del wav_result
    gc.collect()
    bars = real_frame.select("timestamp", "tick_count", "median_spread", "mean_spread")
    columns = _derive(control.columns, fft, wav, bars, regime, control.bar_seconds)
    return pl.DataFrame({"timestamp": control.timestamps, **columns})


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------
def _cache_key(gate: dict[str, Any], regression: RegressionConfig, ou: OUConfig,
               regime: RegimeConfig) -> dict[str, Any]:
    lineage = gate["lineage"]
    versions = gate["versions"]
    return clean_json({
        "inputs_version": INPUTS_VERSION,
        "tick_dataset_version": lineage.get("tick_dataset_version"),
        "bar_dataset_version": lineage.get("bar_dataset_version"),
        "regression_feature_version": versions.get("regression_feature_version"),
        "regression_config_fingerprint": regression.fingerprint(),
        "ou_config_fingerprint": ou.fingerprint(),
        "fft_engine_fingerprint": (versions.get("fft_features") or {}).get("engine_fingerprint"),
        "fft_generated_utc": (versions.get("fft_features") or {}).get("generated_utc"),
        "wavelet_engine_fingerprint": (versions.get("wavelet_features") or {}).get(
            "engine_fingerprint"),
        "wavelet_generated_utc": (versions.get("wavelet_features") or {}).get("generated_utc"),
        "regime_inputs_fingerprint": regime.inputs_fingerprint(),
        "code_fingerprint": _code_version(),
    })


def load_or_build_regime_inputs(config: Config, regression: RegressionConfig, ou: OUConfig,
                                spectral: SpectralConfig, wavelet: WaveletConfig,
                                regime: RegimeConfig, timeframe: str, *,
                                gate: dict[str, Any]) -> tuple[pl.DataFrame, dict[str, Any]]:
    """The cached input table if its every version matches *gate*, else a fresh build."""
    if not gate.get("passed"):
        raise RuntimeError(f"{timeframe}: integrity gate failed - " + "; ".join(gate["problems"]))
    root = regime.inputs_path / f"timeframe={timeframe}"
    key = _cache_key(gate, regression, ou, regime)
    manifest = _read_manifest(root / _MANIFEST)
    path = root / _INPUT_FILE
    if manifest is not None and path.exists() and all(
            manifest.get(k) == v for k, v in key.items()):
        frame = pl.read_parquet(path)
        if frame.height == manifest.get("rows"):
            LOGGER.info("[%s] regime inputs: cached (%s rows)", timeframe, f"{frame.height:,}")
            return frame, manifest
    started = time.perf_counter()
    frame = build_regime_inputs(config, regression, ou, spectral, wavelet, regime, timeframe)
    ensure_dir(root)
    tmp = path.with_name(path.name + ".partial")
    frame.write_parquet(tmp, compression="zstd", compression_level=3)
    tmp.replace(path)
    coverage = {name: float(np.isfinite(frame[name].cast(pl.Float64).fill_null(np.nan)
                                        .to_numpy()).mean()) for name in FEATURE_SPECS}
    manifest = {**key, "timeframe": timeframe, "rows": frame.height,
                "first_timestamp": str(frame["timestamp"][0]),
                "last_timestamp": str(frame["timestamp"][-1]),
                "features": {name: {"group": spec.group, "description": spec.description,
                                    "source": spec.source, "lookback": spec.lookback,
                                    "causal": spec.causal} for name, spec in FEATURE_SPECS.items()},
                "coverage": coverage, "auxiliary_columns": list(AUXILIARY_COLUMNS),
                "causal": True, "join_key": "timestamp",
                "seconds": time.perf_counter() - started, "generated_utc": utc_now_iso()}
    atomic_write_text(root / _MANIFEST, json.dumps(clean_json(manifest), indent=1) + "\n")
    LOGGER.info("[%s] regime inputs built: %s rows in %.0fs", timeframe, f"{frame.height:,}",
                manifest["seconds"])
    return frame, manifest


# ---------------------------------------------------------------------------
# Helpers for the models
# ---------------------------------------------------------------------------
def feature_matrix(frame: pl.DataFrame, features: tuple[str, ...] | list[str]) -> np.ndarray:
    """(rows, features) float64 array; nulls and non-finite values become NaN."""
    columns = [frame[name].cast(pl.Float64).fill_null(np.nan).to_numpy() for name in features]
    x = np.column_stack(columns) if columns else np.empty((frame.height, 0))
    x[~np.isfinite(x)] = np.nan
    return x


def source_from_frame(frame: pl.DataFrame, bar_seconds: float, name: str = "real") -> SourceData:
    """A :class:`SourceData` view of the input table (for the shared outcome builders)."""
    keys = ("log_return", "regression_residual", "residual_zscore", "ou_innovation",
            "ou_zscore", "trailing_volatility", "regression_slope", "ou_half_life_bars",
            "ou_valid", "ou_state_code")
    columns = {k: frame[k].cast(pl.Float64).fill_null(np.nan).to_numpy() for k in keys
               if k in frame.columns}
    columns["r_squared"] = frame["r_squared"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    return SourceData(name=name, timestamps=frame["timestamp"], columns=columns,
                      missing_slots=np.zeros(frame.height), bar_seconds=bar_seconds)


def bar_seconds(timeframe: str) -> float:
    return parse_timeframe(timeframe).total_seconds()


def redundancy_table(frame: pl.DataFrame, features: tuple[str, ...] | list[str], *,
                     sample_rows: int) -> pl.DataFrame:
    """Spearman and Pearson of every feature pair on an evenly spaced sample (estimates)."""
    n = frame.height
    rows = np.arange(n) if n <= sample_rows else np.linspace(0, n - 1, sample_rows).round(
    ).astype(np.int64)
    sub = frame.select(list(features)).with_row_index("__i").filter(
        pl.col("__i").is_in(pl.Series(rows).cast(pl.UInt32))).drop("__i").with_columns(
        pl.all().cast(pl.Float64).fill_nan(None))
    pairs = [(a, b) for i, a in enumerate(features) for b in features[i + 1:]]
    if not pairs:
        return pl.DataFrame()
    values = sub.select(
        *[pl.corr(a, b).alias(f"p|{a}|{b}") for a, b in pairs],
        *[pl.corr(a, b, method="spearman").alias(f"s|{a}|{b}") for a, b in pairs],
    ).row(0, named=True)
    return pl.DataFrame([{"a": a, "b": b, "pearson": values[f"p|{a}|{b}"],
                          "spearman": values[f"s|{a}|{b}"]} for a, b in pairs],
                        infer_schema_length=None)


def feature_manifest(regime: RegimeConfig, feature_set: str,
                     redundancy: pl.DataFrame | None) -> dict[str, Any]:
    """Why each feature is in (or out of) the model, with the measured redundancy."""
    features = regime.features(feature_set)
    duplicates: list[dict[str, Any]] = []
    flagged: list[dict[str, Any]] = []
    if isinstance(redundancy, pl.DataFrame) and not redundancy.is_empty():
        for r in redundancy.iter_rows(named=True):
            rho = r.get("spearman")
            if rho is None or not np.isfinite(rho):
                continue
            entry = {"a": r["a"], "b": r["b"], "spearman": float(rho)}
            if abs(rho) >= regime.redundancy.duplicate_abs_spearman:
                duplicates.append(entry)
            elif abs(rho) >= regime.redundancy.flag_abs_spearman:
                flagged.append(entry)
    return clean_json({
        "feature_set": feature_set,
        "features": {name: {**FEATURE_SPECS[name].__dict__,
                            "group_in_config": regime.group_of(name)} for name in features},
        "excluded": EXCLUDED_FEATURES,
        "redundancy_rule": (f"no two features with |Spearman| >= "
                            f"{regime.redundancy.duplicate_abs_spearman} in one model; pairs "
                            f"above {regime.redundancy.flag_abs_spearman} are listed"),
        "duplicate_pairs": duplicates,
        "flagged_pairs": flagged,
        "passes_redundancy_rule": not duplicates,
        "causal": True,
    })
