r"""Do causal wavelet features carry forward information - beyond what we already have?

Three questions, all outcome research only - nothing here is a rule or a signal:

**IC** (Steps 29-30). :math:`IC(h) = Corr(S_t, Y_{t+h})` and rank IC for every
configured (feature, target, horizon), overall and by year, quarter and
volatility quartile - the Prompt #5 engine (:func:`spectral_ic.ic_table`),
fed wavelet features. Computed on the null controls too: the chance rate.

**Extreme conditioning** (Steps 27-28). After :math:`|Z_{residual}| > z`
(1, 2, 3), the probability that the residual has shrunk ``h`` bars later,
:math:`P(|\epsilon_{t+h}| < |\epsilon_t|)`, by quartile of a wavelet state;
after :math:`|Z_{OU}| > 2`, the probability that :math:`|Z_{OU}|` has shrunk,
by wavelet quartile *and* by OU half-life tercile and :math:`|Z_{OU}|` band -
so the differentiation a wavelet state adds can be set beside what the OU
state already gives.

**Incremental information** (Steps 31-32). Nested linear (continuous
targets) and logistic (binary targets) models:

====  ======================================================
A     statistical: residual Z, |Z|, ln trailing volatility, ln 5- and 20-bar
      realised volatility, slope / volatility, R^2, the latest return and its
      magnitude
B     A + OU: OU Z, |OU Z|, ln half-life, validity, near-unit-root flag, ln
      5- and 20-bar mean |innovation|
C     B + Fourier (Prompt #5, same window): entropy, flatness, centroid,
      top-1/3/5, bands, ln dominant period, phase sin/cos, ln dominant-bin run
D     C + wavelet (the causal features)
====  ======================================================

fitted on chronological, expanding training sets and evaluated on the next
block of years (``test_blocks``), with an embargo of the longest horizon
between them. Features are standardised with training statistics; a
missing value is set to the training mean and flagged by an indicator
column. The ridge penalty is fixed a priori. Out-of-sample :math:`R^2` is
measured against the training mean; binary models report log-loss, AUC,
Brier score and calibration slope. The same models run on null controls,
so an increment is read against the increment chance and overfitting give.
These are research models for measuring information, not the ML system.

:data:`BASELINE_EXTENSIONS` widens A and B with longer-horizon volatility
and the hour of day - a *post-hoc* check (WAVE-P-001), chosen after the
registered test and recorded as such, that the registered increments did not
survive (``scripts/wavelet_baseline_robustness.py``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from scipy.special import expit
from scipy.stats import rankdata

from ..features.spectral import RollingSpectrum
from ..features.wavelet_causal import RollingWavelet, ic_feature_columns
from ..features.wavelet_config import WaveletConfig
from ..models.ornstein_uhlenbeck import OU_STATES
from .ou_conditional import _bucket
from .ou_estimation import num
from .spectral_ic import FEATURES as FFT_FEATURES
from .spectral_ic import feature_columns as fft_feature_columns
from .spectral_ic import ic_table, sample_rows, target_columns
from .spectral_nulls import SourceData

__all__ = [
    "BLOCKS",
    "MODELS",
    "chronological_folds",
    "extreme_conditioning",
    "fit_linear_models",
    "fit_logistic",
    "incremental_study",
    "incremental_summary",
    "model_blocks",
    "trailing_mean",
    "roc_auc",
    "wavelet_ic_study",
]

BLOCKS: tuple[str, ...] = ("statistical", "ou", "fourier", "wavelet")
MODELS: dict[str, tuple[str, ...]] = {
    "A": BLOCKS[:1], "B": BLOCKS[:2], "C": BLOCKS[:3], "D": BLOCKS,
}
_NEAR_UNIT_ROOT = float(OU_STATES.index("near_unit_root"))


# ---------------------------------------------------------------------------
# IC
# ---------------------------------------------------------------------------
def wavelet_ic_study(result: RollingWavelet, source: SourceData, config: WaveletConfig, *,
                     max_rows: int, source_name: str) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    """IC and rank IC of every configured wavelet (feature, target, horizon)."""
    cfg = config.ic
    rows = sample_rows(result.valid, max_rows)
    features = ic_feature_columns(result, cfg.features, rows)
    targets = target_columns(source, cfg.targets, cfg.horizons, rows)
    volatility = (source.columns["trailing_volatility"][rows]
                  if "trailing_volatility" in source.columns else None)
    return ic_table(features, targets, timestamps=source.timestamps.gather(pl.Series(rows)),
                    volatility=volatility, source_name=source_name)


# ---------------------------------------------------------------------------
# Extreme conditioning
# ---------------------------------------------------------------------------
_STATE_METRICS: dict[str, Callable[[RollingWavelet], np.ndarray]] = {
    "wavelet_entropy": lambda r: r.entropy,
    "wavelet_log_dominant_period": lambda r: np.log(r.dominant_period_bars),
    "wavelet_fast_slow_log_ratio": lambda r: r.fast_slow_log_ratio,
    "wavelet_top3_scale_share": lambda r: r.top3_share,
}


def _shrink_rows(frame: pl.DataFrame, group: str, *, source: str, extreme: str,
                 threshold: float, horizon: int, conditioning: str) -> list[dict[str, Any]]:
    table = (frame.filter(pl.col(group).is_not_null() & pl.col("shrinks").is_not_null())
             .group_by(group).agg(pl.len().alias("observations"),
                                  pl.col("shrinks").mean().alias("prob_shrinks"))
             .sort(group))
    rows = []
    for record in table.iter_rows(named=True):
        p, n = record["prob_shrinks"], record["observations"]
        rows.append({"source": source, "extreme": extreme, "threshold": threshold,
                     "horizon": horizon, "conditioning": conditioning,
                     "bucket": str(record[group]), "observations": int(n),
                     "prob_shrinks": float(p),
                     "std_error": float(np.sqrt(max(p * (1 - p), 0.0) / n)) if n else None})
    return rows


def extreme_conditioning(result: RollingWavelet, source: SourceData, config: WaveletConfig, *,
                         source_name: str) -> pl.DataFrame:
    """Residual and OU-extreme outcomes by wavelet state (and by OU state, for comparison)."""
    if not source.has_pipeline:
        return pl.DataFrame()
    cfg = config.outcomes
    valid = result.valid
    n = valid.size
    residual = source.columns["regression_residual"]
    zres = source.columns["residual_zscore"]
    zou = source.columns["ou_zscore"]
    with np.errstate(divide="ignore", invalid="ignore"):
        states = {name: np.where(valid, fn(result), np.nan) for name, fn in _STATE_METRICS.items()}
    base = pl.DataFrame(states).with_columns(pl.all().fill_nan(None))
    buckets = {}
    for name in states:
        bucketed, _ = _bucket(base, name, cfg.buckets)
        buckets[name] = bucketed["bucket"]
    half_life = np.where(source.columns["ou_valid"] > 0, source.columns["ou_half_life_bars"],
                         np.nan)
    hl_frame = pl.DataFrame({"hl": half_life}).with_columns(pl.col("hl").fill_nan(None))
    hl_bucket, _ = _bucket(hl_frame, "hl", 3, ("fast", "medium", "slow"))
    ou_state = np.where(source.columns["ou_state_code"] == _NEAR_UNIT_ROOT, "near_unit_root",
                        np.where(source.columns["ou_valid"] > 0, "", "invalid"))
    hl_label = pl.Series(ou_state).zip_with(pl.Series(ou_state) != "",
                                            hl_bucket["bucket"].fill_null(""))
    rows: list[dict[str, Any]] = []
    for h in cfg.horizons:
        later = np.full(n, np.nan)
        later[:-h] = residual[h:]
        later_z = np.full(n, np.nan)
        later_z[:-h] = zou[h:]
        with np.errstate(invalid="ignore"):
            res_shrink = np.where(np.isfinite(later) & np.isfinite(residual),
                                  (np.abs(later) < np.abs(residual)).astype(np.float64), np.nan)
            ou_shrink = np.where(np.isfinite(later_z) & np.isfinite(zou),
                                 (np.abs(later_z) < np.abs(zou)).astype(np.float64), np.nan)
        for threshold in cfg.residual_extremes:
            mask = valid & (np.abs(np.nan_to_num(zres)) > threshold)
            frame = pl.DataFrame({"shrinks": res_shrink[mask],
                                  **{k: v.gather(pl.Series(np.flatnonzero(mask)))
                                     for k, v in buckets.items()}}).with_columns(
                pl.col("shrinks").fill_nan(None))
            for name in states:
                rows += _shrink_rows(frame, name, source=source_name, extreme="residual_z",
                                     threshold=threshold, horizon=h, conditioning=name)
        mask = valid & (np.abs(np.nan_to_num(zou)) > cfg.ou_extreme)
        idx = pl.Series(np.flatnonzero(mask))
        az = np.abs(zou[mask])
        frame = pl.DataFrame({
            "shrinks": ou_shrink[mask],
            **{k: v.gather(idx) for k, v in buckets.items()},
            "ou_half_life": hl_label.gather(idx),
            "ou_abs_z": np.where(az < 2.5, "2.0-2.5", np.where(az < 3.0, "2.5-3.0", ">3.0")),
        }).with_columns(pl.col("shrinks").fill_nan(None))
        for name in (*states, "ou_half_life", "ou_abs_z"):
            rows += _shrink_rows(frame, name, source=source_name, extreme="ou_z",
                                 threshold=cfg.ou_extreme, horizon=h, conditioning=name)
    return pl.DataFrame(rows, infer_schema_length=None)


# ---------------------------------------------------------------------------
# Incremental information: nested models A-D
# ---------------------------------------------------------------------------
def _dominant_run(dominant: np.ndarray, valid: np.ndarray) -> np.ndarray:
    n = dominant.size
    idx = np.arange(n)
    change = np.ones(n, dtype=bool)
    change[1:] = (dominant[1:] != dominant[:-1]) | ~valid[:-1]
    start = np.maximum.accumulate(np.where(change, idx, 0))
    return np.where(valid, (idx - start + 1).astype(np.float64), np.nan)


def trailing_mean(values: np.ndarray, width: int) -> np.ndarray:
    """Mean of the finite values among the last *width* bars (NaN if none); causal."""
    x = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(x)
    total = np.concatenate(([0.0], np.cumsum(np.where(finite, x, 0.0))))
    count = np.concatenate(([0], np.cumsum(finite, dtype=np.int64)))
    out = np.full(x.size, np.nan)
    if width <= x.size:
        n = count[width:] - count[:-width]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[width - 1:] = np.where(n > 0, (total[width:] - total[:-width]) / n, np.nan)
    return out


def model_blocks(result: RollingWavelet, spectrum: RollingSpectrum | None, source: SourceData,
                 config: WaveletConfig, rows: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """The four feature blocks at bars *rows* (NaN where a feature is undefined).

    The baselines include short-horizon volatility on purpose: fine-scale
    wavelet energy *is* a short-horizon realised volatility, so without 5-
    and 20-bar realised volatility (and 5- and 20-bar mean |OU innovation|)
    in A and B, a wavelet "increment" on a volatility target would only be
    the multi-horizon volatility effect.
    """
    c = source.columns
    returns = c["log_return"]
    innovation = np.abs(c["ou_innovation"])
    with np.errstate(divide="ignore", invalid="ignore"):
        vol = c["trailing_volatility"][rows]
        statistical = {
            "residual_zscore": c["residual_zscore"][rows],
            "abs_residual_zscore": np.abs(c["residual_zscore"][rows]),
            "log_trailing_volatility": np.log(vol),
            "log_realized_vol_5": 0.5 * np.log(trailing_mean(returns ** 2, 5)[rows]),
            "log_realized_vol_20": 0.5 * np.log(trailing_mean(returns ** 2, 20)[rows]),
            "slope_over_volatility": c["regression_slope"][rows] / vol,
            "r_squared": c["r_squared"][rows],
            "log_return": returns[rows],
            "abs_log_return": np.abs(returns[rows]),
        }
        ou = {
            "ou_zscore": c["ou_zscore"][rows],
            "abs_ou_zscore": np.abs(c["ou_zscore"][rows]),
            "log_ou_half_life": np.where(c["ou_valid"][rows] > 0,
                                         np.log(c["ou_half_life_bars"][rows]), np.nan),
            "ou_valid": c["ou_valid"][rows],
            "ou_near_unit_root": (c["ou_state_code"][rows] == _NEAR_UNIT_ROOT).astype(float),
            "log_mean_abs_innovation_5": np.log(trailing_mean(innovation, 5)[rows]),
            "log_mean_abs_innovation_20": np.log(trailing_mean(innovation, 20)[rows]),
        }
    fourier: dict[str, np.ndarray] = {}
    if spectrum is not None:
        fourier = {f"fft_{k}": v for k, v in fft_feature_columns(spectrum, FFT_FEATURES,
                                                                 rows).items()}
        run = _dominant_run(spectrum.dominant_bin.astype(np.int64), spectrum.valid)
        fourier["fft_log_run_length"] = np.log(np.minimum(run, spectrum.fft_window))[rows]
    wavelet = ic_feature_columns(result, config.ic.features, rows)
    return {"statistical": statistical, "ou": ou, "fourier": fourier, "wavelet": wavelet}


#: Post-hoc baseline extensions (``scripts/wavelet_baseline_robustness.py``).
#: Chosen *after* WAVE-H-007 was evaluated, so never part of models A-D: they
#: ask whether a wavelet increment is only volatility over horizons longer than
#: A's 50 bars (a window's wavelet energy is itself an N-bar realised
#: volatility) or the daily volatility cycle, which no baseline block carries.
BASELINE_EXTENSIONS: dict[str, dict[str, Any]] = {
    "original": {"volatility_windows": (), "hours": False},
    "hour_only": {"volatility_windows": (), "hours": True},
    "rv_only": {"volatility_windows": (64, 256, 1024), "hours": False},
    "rich": {"volatility_windows": (64, 256, 1024), "hours": True},
}


def extend_baseline(blocks: dict[str, dict[str, np.ndarray]], source: SourceData,
                    rows: np.ndarray, *, volatility_windows: tuple[int, ...] = (),
                    hours: bool = False) -> dict[str, dict[str, np.ndarray]]:
    """Blocks A and B widened with longer-horizon volatility and/or the hour of day.

    A gains ln realised volatility over each of *volatility_windows* bars and,
    with *hours*, indicators for hours 1-23 of the bar's timestamp (hour 0 is
    the reference; the broker clock is New York + 7 h, so the daily cycle
    sits at the same hours in every season); B gains ln mean |OU innovation|
    over the same windows. Everything is trailing, so causal. *blocks* is not
    modified.
    """
    c = source.columns
    out = {name: dict(block) for name, block in blocks.items()}
    with np.errstate(divide="ignore", invalid="ignore"):
        for width in volatility_windows:
            out["statistical"][f"log_realized_vol_{width}"] = 0.5 * np.log(
                trailing_mean(c["log_return"] ** 2, width)[rows])
            out["ou"][f"log_mean_abs_innovation_{width}"] = np.log(
                trailing_mean(np.abs(c["ou_innovation"]), width)[rows])
    if hours:
        hour = source.timestamps.gather(pl.Series(rows)).dt.hour().to_numpy()
        for h in range(1, 24):
            out["statistical"][f"hour_{h:02d}"] = (hour == h).astype(np.float64)
    return out


def chronological_folds(timestamps: pl.Series, rows: np.ndarray,
                        test_blocks: tuple[tuple[int, int], ...], embargo: int
                        ) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Expanding-window folds: train strictly before each test block, minus an embargo.

    *rows* are bar indices; a training row must end its longest outcome
    (``embargo`` bars) before the first test bar.
    """
    years = timestamps.dt.year().to_numpy()
    folds = []
    for first, last in test_blocks:
        test = np.flatnonzero((years >= first) & (years <= last))
        if test.size == 0:
            continue
        first_bar = rows[test[0]]
        train = np.flatnonzero(rows < first_bar - embargo)
        if train.size:
            folds.append((f"{first}-{last}", train, test))
    return folds


@dataclass
class _Design:
    """Standardised design matrix with missing indicators, and each column's block."""

    matrix: np.ndarray           # (rows, 1 + columns), first column the intercept
    names: list[str]
    blocks: list[str]


def _design(blocks: dict[str, dict[str, np.ndarray]], train: np.ndarray,
            order: tuple[str, ...] = BLOCKS) -> _Design:
    """Standardise every column on *train*; missing values -> 0 (the training mean).

    A column missing in more than 1 % (and fewer than 99 %) of training rows
    also gets a standardised missing indicator in the same block. *order* is
    the block order (models A-D by default; the regime layer passes its own).
    """
    size = next(len(v) for block in blocks.values() for v in block.values())
    columns: list[np.ndarray] = [np.ones(size)]
    names, owners = ["intercept"], ["intercept"]
    for block in order:
        for name, values in blocks[block].items():
            x = np.asarray(values, dtype=np.float64)
            finite = np.isfinite(x)
            tr = x[train][finite[train]]
            if tr.size < 2 or np.std(tr) == 0:
                continue
            mu, sd = float(tr.mean()), float(tr.std())
            columns.append(np.where(finite, (x - mu) / sd, 0.0))
            names.append(name)
            owners.append(block)
            missing = ~finite
            if missing[train].mean() > 0.01 and missing[train].mean() < 0.99:
                indicator = missing.astype(np.float64)
                m, s = indicator[train].mean(), indicator[train].std()
                columns.append((indicator - m) / s)
                names.append(f"{name}__missing")
                owners.append(block)
    return _Design(matrix=np.column_stack(columns), names=names, blocks=owners)


def _select(design: _Design, blocks: tuple[str, ...], extra: tuple[str, ...] = ()) -> np.ndarray:
    return np.array([i for i, (b, n) in enumerate(zip(design.blocks, design.names, strict=True))
                     if b == "intercept" or b in blocks or n in extra
                     or n.removesuffix("__missing") in extra])


def fit_linear_models(design: _Design, targets: dict[str, np.ndarray], train: np.ndarray,
                      test: np.ndarray, *, ridge: float,
                      subsets: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """Ridge OLS for every column subset and target, from one Gram matrix per fold."""
    x_train, x_test = design.matrix[train], design.matrix[test]
    names = list(targets)
    y = np.column_stack([targets[k] for k in names])
    y_train, y_test = y[train], y[test]
    gram = x_train.T @ x_train
    cross = x_train.T @ y_train
    penalty = np.full(gram.shape[0], ridge)
    penalty[0] = 0.0
    mean_train = y_train.mean(axis=0)
    sst = ((y_test - mean_train) ** 2).sum(axis=0)
    rows = []
    for label, cols in subsets.items():
        g = gram[np.ix_(cols, cols)] + np.diag(penalty[cols])
        beta = np.linalg.solve(g, cross[cols])
        sse = ((y_test - x_test[:, cols] @ beta) ** 2).sum(axis=0)
        for k, target in enumerate(names):
            rows.append({"model": label, "target": target, "train_rows": int(train.size),
                         "test_rows": int(test.size),
                         "oos_r2": float(1.0 - sse[k] / sst[k]) if sst[k] > 0 else None})
    return rows


def fit_logistic(x: np.ndarray, y: np.ndarray, *, ridge: float, max_iter: int = 50,
                 tol: float = 1e-9) -> np.ndarray:
    """L2-penalised logistic regression by Newton-IRLS (column 0 unpenalised)."""
    beta = np.zeros(x.shape[1])
    penalty = np.full(x.shape[1], ridge)
    penalty[0] = 0.0
    for _ in range(max_iter):
        p = expit(x @ beta)
        w = np.clip(p * (1 - p), 1e-12, None)
        grad = x.T @ (y - p) - penalty * beta
        hess = (x * w[:, None]).T @ x + np.diag(penalty)
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def roc_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    """Area under the ROC curve (Mann-Whitney, ties averaged)."""
    pos = y > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(score)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    logit = np.log(p / (1 - p))
    design = np.column_stack([np.ones_like(logit), logit])
    try:
        slope = float(fit_logistic(design, y, ridge=0.0)[1])
    except np.linalg.LinAlgError:
        slope = None
    return {"log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "auc": roc_auc(y, p), "brier": float(np.mean((p - y) ** 2)),
            "calibration_slope": slope, "base_rate": float(y.mean())}


def _binary_targets(source: SourceData, rows: np.ndarray, horizons: tuple[int, ...],
                    names: tuple[str, ...], extreme: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``name_h<h>`` -> (outcome 0/1 or NaN, in-scope mask) at *rows*."""
    n = source.size
    out = {}
    series = {"residual_shrinks": (source.columns["regression_residual"],
                                   source.columns["residual_zscore"]),
              "ou_zscore_shrinks": (source.columns["ou_zscore"], source.columns["ou_zscore"])}
    for name in names:
        values, zscore = series[name]
        scope = np.abs(np.nan_to_num(zscore[rows])) > extreme
        for h in horizons:
            ahead = rows + h
            inside = ahead < n
            later = np.where(inside, values[np.where(inside, ahead, 0)], np.nan)
            now = values[rows]
            with np.errstate(invalid="ignore"):
                y = np.where(np.isfinite(later) & np.isfinite(now),
                             (np.abs(later) < np.abs(now)).astype(np.float64), np.nan)
            out[f"{name}_h{h}"] = (y, scope & np.isfinite(y))
    return out


def incremental_study(result: RollingWavelet, spectrum: RollingSpectrum | None,
                      source: SourceData, config: WaveletConfig, *, source_name: str,
                      baseline: str = "original", per_feature: bool = True
                      ) -> dict[str, pl.DataFrame]:
    """Models A-D (and C + one wavelet feature at a time) on chronological folds.

    *baseline* names a :data:`BASELINE_EXTENSIONS` entry; anything but
    ``"original"`` is the post-hoc robustness check, not the registered test.
    ``per_feature=False`` skips the C + one-feature fits.
    """
    cfg = config.incremental
    if not source.has_pipeline:
        return {}
    rows = sample_rows(result.valid, cfg.max_rows)
    embargo = max((*cfg.horizons, 1))
    stamps = source.timestamps.gather(pl.Series(rows))
    blocks = extend_baseline(model_blocks(result, spectrum, source, config, rows), source, rows,
                             **BASELINE_EXTENSIONS[baseline])
    continuous = target_columns(source, cfg.targets, cfg.horizons, rows)
    complete = np.all([np.isfinite(v) for v in continuous.values()], axis=0)
    folds = chronological_folds(stamps, rows, cfg.test_blocks, embargo)
    linear_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    binary_rows: list[dict[str, Any]] = []
    binary = _binary_targets(source, rows, cfg.horizons, cfg.binary_targets, cfg.binary_extreme)
    for fold, train, test in folds:
        train_c, test_c = train[complete[train]], test[complete[test]]
        if train_c.size >= cfg.min_train_rows and test_c.size >= 100:
            design = _design(blocks, train_c)
            subsets = {label: _select(design, members) for label, members in MODELS.items()}
            for name in blocks["wavelet"] if per_feature else ():
                cols = _select(design, MODELS["C"], (name,))
                if cols.size > subsets["C"].size:
                    subsets[f"C+{name}"] = cols
            for record in fit_linear_models(design, continuous, train_c, test_c,
                                            ridge=cfg.ridge, subsets=subsets):
                target, horizon = record["target"].rsplit("_h", 1)
                record.update({"source": source_name, "fold": fold, "target": target,
                               "horizon": int(horizon)})
                (feature_rows if record["model"].startswith("C+") else linear_rows).append(
                    record)
        for key, (y, scope) in binary.items():
            tr, te = train[scope[train]], test[scope[test]]
            if tr.size < cfg.min_train_rows // 4 or te.size < 50:
                continue
            design = _design(blocks, tr)
            target, horizon = key.rsplit("_h", 1)
            for label, members in MODELS.items():
                cols = _select(design, members)
                try:
                    beta = fit_logistic(design.matrix[tr][:, cols], y[tr], ridge=cfg.ridge)
                except np.linalg.LinAlgError:
                    continue
                p = expit(design.matrix[te][:, cols] @ beta)
                binary_rows.append({"source": source_name, "fold": fold, "target": target,
                                    "horizon": int(horizon), "model": label,
                                    "train_rows": int(tr.size), "test_rows": int(te.size),
                                    **_binary_metrics(y[te], p)})
    return {"linear": pl.DataFrame(linear_rows, infer_schema_length=None),
            "by_feature": pl.DataFrame(feature_rows, infer_schema_length=None),
            "binary": pl.DataFrame(binary_rows, infer_schema_length=None)}


def incremental_summary(linear: pl.DataFrame, binary: pl.DataFrame) -> pl.DataFrame:
    """Per source, target and horizon: fold-mean scores of A-D and the increments.

    ``folds_d_beats_c`` counts folds where D improves on C (higher R^2, or
    lower log-loss for binary targets).
    """
    rows: list[dict[str, Any]] = []
    if not linear.is_empty():
        wide = linear.pivot(on="model", index=["source", "target", "horizon", "fold"],
                            values="oos_r2")
        for key, part in wide.group_by("source", "target", "horizon", maintain_order=True):
            rec: dict[str, Any] = {"source": key[0], "target": key[1], "horizon": key[2],
                                   "kind": "continuous", "metric": "oos_r2",
                                   "folds": part.height}
            for m in MODELS:
                if m in part.columns:
                    rec[m] = num(part[m].mean())
            for a, b in (("B", "A"), ("C", "B"), ("D", "C")):
                if a in part.columns and b in part.columns:
                    rec[f"delta_{a}_{b}"] = num((part[a] - part[b]).mean())
            if "D" in part.columns and "C" in part.columns:
                rec["folds_d_beats_c"] = int((part["D"] > part["C"]).sum())
            rows.append(rec)
    if not binary.is_empty():
        for metric in ("log_loss", "auc"):
            wide = binary.pivot(on="model", index=["source", "target", "horizon", "fold"],
                                values=metric)
            for key, part in wide.group_by("source", "target", "horizon", maintain_order=True):
                rec = {"source": key[0], "target": key[1], "horizon": key[2], "kind": "binary",
                       "metric": metric, "folds": part.height}
                for m in MODELS:
                    if m in part.columns:
                        rec[m] = num(part[m].mean())
                sign = -1.0 if metric == "log_loss" else 1.0
                for a, b in (("B", "A"), ("C", "B"), ("D", "C")):
                    if a in part.columns and b in part.columns:
                        rec[f"delta_{a}_{b}"] = sign * num((part[a] - part[b]).mean())
                if "D" in part.columns and "C" in part.columns:
                    better = (part["D"] < part["C"]) if metric == "log_loss" else (
                        part["D"] > part["C"])
                    rec["folds_d_beats_c"] = int(better.sum())
                rows.append(rec)
    return pl.DataFrame(rows, infer_schema_length=None)
