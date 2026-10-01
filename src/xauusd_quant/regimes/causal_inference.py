r"""Walk-forward regime inference: the only source of live-safe regime features (Steps 24-30, 54-56).

For each inference period (a calendar month or quarter from ``first_inference``):

1. train on past bars only - ``expanding`` (from the first bar) or ``rolling``
   (the last ``rolling_training_years``) - ending strictly before the period;
2. fit and **freeze** a scaler on those training bars;
3. fit and **freeze** the model (warm-started from the previous period, plus
   fresh initialisations every ``fresh_init_every`` refits; the higher training
   likelihood wins);
4. align its states to the previous period's (Hungarian, Bhattacharyya), so
   ``state_2`` keeps meaning the same state across refits - or say how far it
   drifted;
5. run the frozen model over ``burn_in_bars`` of past bars and the period:
   the HMM forward filter (:math:`P(S_t \mid X_{\le t})`), the GMM posterior
   or the nearest K-Means centre;
6. store every refit as an immutable model artefact and move on.

Nothing inside a period can see a later bar: the scaler and model were fitted
before it began, and the filter is a forward recursion. The per-bar table is
written only through :func:`regime_feature_frame`, whose schema
(:data:`LIVE_SAFE_COLUMNS`) refuses smoothed probabilities, Viterbi paths and
anything labelled offline.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..research.study_io import clean_json
from ..utils.clock import utc_now_iso
from ..utils.logging import get_logger
from ..utils.paths import atomic_write_text, ensure_dir
from .clustering import KMeansModel
from .config import RegimeConfig
from .dataset import feature_matrix
from .diagnostics import adjusted_rand, degenerate_flags, state_separation
from .emissions import bhattacharyya
from .fitting import (
    FittedRegimeModel,
    InferenceResult,
    fit_regime_model,
    missing_policy,
    warm_start,
)
from .hmm import HMMModel
from .preprocessing import fit_scaler
from .registry import ModelRegistry
from .state_alignment import align_states, canonical_order, cost_matrix
from .transitions import regime_age

__all__ = [
    "LIVE_SAFE_COLUMNS",
    "RefitPeriod",
    "WalkForwardResult",
    "enforce_live_safe",
    "model_occupancy",
    "refit_schedule",
    "regime_feature_frame",
    "run_walk_forward",
    "write_regime_features",
]

LOGGER = get_logger("regimes.causal_inference")

#: Column patterns a stored (live-safe) regime feature table may hold.
LIVE_SAFE_COLUMNS: dict[str, str] = {
    r"timestamp": "bar open time (join key)",
    r"regime_model_id": "immutable ID of the frozen model that produced the row",
    r"regime_model_version": "version of the regime pipeline (config, inputs, code)",
    r"(hmm|gmm|gmm_diag|kmeans)_state": "most likely state P(S_t | X_<=t) (argmax)",
    r"(hmm|gmm|gmm_diag)_p\d+": "P(S_t = k | X_<=t): HMM forward filter or GMM posterior",
    r"(hmm|gmm|gmm_diag)_state_confidence": "probability of the most likely state",
    r"(hmm|gmm|gmm_diag)_entropy": "entropy of the state probabilities (nats)",
    r"next_state_p\d+": "P(S_{t+1} = j | X_<=t) = sum_i p_i A_ij (HMM)",
    r"leave_probability": "P(S_{t+1} != S_t | X_<=t) = 1 - sum_i p_i A_ii (HMM)",
    r"regime_age_bars": "bars since the current most likely state began (past labels only)",
    r"regime_feature_coverage": "share of the regime inputs observed at t",
    r"regime_observed": "the bar was scored (enough coverage); else the HMM only predicted",
}
_FORBIDDEN = re.compile(r"(smooth|viterbi|offline|posterior_full|fwd_|future)", re.IGNORECASE)


def enforce_live_safe(columns: list[str] | tuple[str, ...]) -> None:
    """Refuse any column that is not a registered live-safe regime feature."""
    for name in columns:
        if _FORBIDDEN.search(name):
            raise ValueError(f"{name!r}: smoothed, Viterbi, offline or forward values can never "
                             "be stored as live regime features")
        if not any(re.fullmatch(pattern, name) for pattern in LIVE_SAFE_COLUMNS):
            raise ValueError(f"{name!r} is not a registered live-safe regime feature")


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RefitPeriod:
    """One refit: train on [train_start, train_end), infer on [infer_start, infer_end)."""

    index: int
    train_start: datetime
    train_end: datetime
    infer_start: datetime
    infer_end: datetime


def _period_starts(first: datetime, last: datetime, frequency: str) -> list[datetime]:
    step = 1 if frequency == "monthly" else 3
    month = first.month if frequency == "monthly" else 3 * ((first.month - 1) // 3) + 1
    cur = datetime(first.year, month, 1)
    out = []
    while cur <= last:
        out.append(cur)
        m = cur.month - 1 + step
        cur = datetime(cur.year + m // 12, m % 12 + 1, 1)
    out.append(cur)
    return out


def refit_schedule(first_timestamp: datetime, last_timestamp: datetime, *,
                   first_inference: datetime, frequency: str, scheme: str,
                   rolling_years: int) -> list[RefitPeriod]:
    """Calendar-aligned refit periods from *first_inference* to the end of the data."""
    starts = _period_starts(first_inference, last_timestamp, frequency)
    periods = []
    for i in range(len(starts) - 1):
        begin, end = starts[i], starts[i + 1]
        if scheme == "expanding":
            train_start = first_timestamp
        else:
            try:
                train_start = begin.replace(year=begin.year - rolling_years)
            except ValueError:                   # 29 February
                train_start = begin.replace(year=begin.year - rolling_years, day=28)
            train_start = max(train_start, first_timestamp)
        periods.append(RefitPeriod(i, train_start, begin, begin, end))
    return periods


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------
@dataclass
class WalkForwardResult:
    """Per-bar causal output and the per-refit log of one walk-forward run."""

    timeframe: str
    family: str
    states: int
    scheme: str
    refit: str
    feature_set: str
    frame: pl.DataFrame
    refits: pl.DataFrame
    model_ids: list[str] = field(default_factory=list)
    seconds: float = 0.0
    anchor: dict[str, Any] = field(default_factory=dict)


def _drift(prev: FittedRegimeModel, new: FittedRegimeModel) -> dict[str, Any]:
    """How much each aligned state changed between two refits (Steps 40-41)."""
    pm, pc = prev.raw_parameters()
    nm, nc = new.raw_parameters()
    scale = prev.scaler.scale
    mean_shift = np.sqrt((((nm - pm) / scale) ** 2).sum(axis=1))
    out: dict[str, Any] = {"mean_shift_scaled": mean_shift.tolist(),
                           "max_mean_shift_scaled": float(mean_shift.max())}
    if pc is not None and nc is not None:
        bd = [bhattacharyya(pm[i], pc[i], nm[i], nc[i]) for i in range(pm.shape[0])]
        out["bhattacharyya"] = bd
        out["max_bhattacharyya"] = float(max(bd))
        ratios = []
        for i in range(pm.shape[0]):
            _, a = np.linalg.slogdet(pc[i])
            _, b = np.linalg.slogdet(nc[i])
            ratios.append(float(b - a))
        out["log_det_change"] = ratios
    pt, nt = prev.transition(), new.transition()
    if pt is not None and nt is not None:
        out["max_transition_change"] = float(np.abs(nt - pt).max())
        with np.errstate(divide="ignore"):
            pd = 1.0 / (1.0 - np.diag(pt))
            nd = 1.0 / (1.0 - np.diag(nt))
        out["expected_duration_log_change"] = np.log(nd / pd).tolist()
    return out


def _anchor_distance(anchor: FittedRegimeModel, model: FittedRegimeModel) -> list[float] | None:
    am, ac = anchor.raw_parameters()
    mm, mc = model.raw_parameters()
    if ac is None or mc is None:
        return None
    return [bhattacharyya(am[i], ac[i], mm[i], mc[i]) for i in range(am.shape[0])]


def _occupancy(result: InferenceResult, k: int) -> np.ndarray:
    """Share of the period's bars per state (a state may rightly be rare in one period)."""
    if result.probs is not None:
        probs = result.probs[result.scored]
        return np.nansum(probs, axis=0) if probs.size else np.zeros(k)
    labels = result.state[result.state >= 0]
    return np.bincount(labels, minlength=k).astype(np.float64)


def model_occupancy(fitted: FittedRegimeModel, scaled_train: np.ndarray) -> np.ndarray:
    """The model's own state shares on its training data (for the degeneracy checks):
    GMM weights, the HMM's stationary distribution, K-Means cluster sizes."""
    model = fitted.model
    if isinstance(model, HMMModel):
        return model.stationary()
    if isinstance(model, KMeansModel):
        labels, _ = model.predict(scaled_train[np.isfinite(scaled_train).all(axis=1)])
        return np.bincount(labels[labels >= 0], minlength=model.n_states).astype(np.float64)
    return np.asarray(model.weights, dtype=np.float64)


def run_walk_forward(frame: pl.DataFrame, regime: RegimeConfig, *, timeframe: str,
                     family: str, states: int, scheme: str, refit: str,
                     feature_set: str | None = None, registry: ModelRegistry | None = None,
                     provenance: dict[str, Any] | None = None,
                     progress: Callable[[str], None] | None = None,
                     limit_periods: int | None = None) -> WalkForwardResult:
    """Walk-forward refits over the whole history; see the module docstring."""
    started = time.perf_counter()
    set_name = feature_set or regime.primary_feature_set
    features = regime.features(set_name)
    causal = regime.causal
    stamps = frame["timestamp"]
    x = feature_matrix(frame, features)
    times = stamps.to_numpy()
    first_inference = datetime.fromisoformat(causal.first_inference)
    periods = refit_schedule(stamps[0], stamps[-1], first_inference=first_inference,
                             frequency=refit, scheme=scheme,
                             rolling_years=causal.rolling_training_years)
    if limit_periods is not None:
        periods = periods[:limit_periods]
    policy = missing_policy(regime)
    pre = regime.preprocessing
    vol_index = list(features).index(regime.baselines.volatility_feature) if (
        regime.baselines.volatility_feature in features) else 0
    parts: list[dict[str, Any]] = []
    log_rows: list[dict[str, Any]] = []
    model_ids: list[str] = []
    previous: FittedRegimeModel | None = None
    anchor: FittedRegimeModel | None = None
    version = (provenance or {}).get("regime_model_version")
    for period in periods:
        lo = int(np.searchsorted(times, np.datetime64(period.train_start), side="left"))
        hi = int(np.searchsorted(times, np.datetime64(period.train_end), side="left"))
        i0 = hi
        i1 = int(np.searchsorted(times, np.datetime64(period.infer_end), side="left"))
        if i1 <= i0:
            continue
        train = x[lo:hi]
        complete = np.isfinite(train).all(axis=1)
        row: dict[str, Any] = {"period": period.index,
                               "train_start": str(period.train_start),
                               "train_end": str(period.train_end),
                               "infer_start": str(period.infer_start),
                               "infer_end": str(period.infer_end),
                               "train_rows": int(hi - lo), "train_complete_rows": int(complete.sum()),
                               "infer_rows": int(i1 - i0)}
        if int(complete.sum()) < causal.min_training_observations:
            row["status"] = "skipped: too few complete training rows"
            log_rows.append(row)
            continue
        t0 = time.perf_counter()
        scaler = fit_scaler(train, features, method=pre.scaling, clip=pre.clip,
                            training_start=str(stamps[lo]), training_end=str(stamps[hi - 1]))
        z = scaler.transform(train)
        seed = int(regime.seed + 1009 * period.index + states)
        fresh = previous is None or period.index % causal.fresh_init_every == 0
        if previous is None:
            model = fit_regime_model(family, z, states, regime, seed=seed)
        else:
            init = warm_start(previous, scaler) if causal.warm_start else None
            model = fit_regime_model(
                family, z, states, regime, seed=seed,
                n_init=causal.fresh_inits_per_refit if fresh else 0,
                max_iter=causal.refit_max_iter, tol=causal.refit_tol, init=init)
        fitted = FittedRegimeModel(family, model, scaler)
        means_raw, _ = fitted.raw_parameters()
        if previous is None:
            fitted = fitted.permuted(canonical_order(means_raw, vol_index))
            alignment_cost, ambiguity, order = 0.0, None, list(range(states))
        else:
            pm, pc = previous.raw_parameters()
            nm, nc = fitted.raw_parameters()
            alignment = align_states(cost_matrix(pm, pc, nm, nc, scale=previous.scaler.scale))
            fitted = fitted.permuted(alignment.order)
            alignment_cost = alignment.total_cost
            ambiguity = alignment.ambiguity.tolist()
            order = alignment.order.tolist()
        drift = _drift(previous, fitted) if previous is not None else {}
        if anchor is None:
            anchor = fitted
        fit_seconds = time.perf_counter() - t0
        # --- inference over burn-in + period (past bars only feed the filter) ---
        burn = max(0, i0 - causal.burn_in_bars) if family == "hmm" else i0
        result = fitted.infer(x[burn:i1], policy).tail(i0 - burn)
        # --- overlap with the previous model on this period's bars (definition change) ---
        overlap = None
        if previous is not None:
            before = previous.infer(x[burn:i1], policy).tail(i0 - burn)
            both = (before.state >= 0) & (result.state >= 0)
            if both.sum() > 10:
                overlap = {"ari": adjusted_rand(before.state[both], result.state[both]),
                           "agreement": float(np.mean(before.state[both] == result.state[both]))}
        k = states
        occupancy = _occupancy(result, k)
        means_raw, covs_raw = fitted.raw_parameters()
        flags = degenerate_flags(
            model_occupancy(fitted, z),
            covariances=(None if isinstance(fitted.model, KMeansModel)
                         else fitted.model.states.covariances),
            separation=state_separation(means_raw, covs_raw) if covs_raw is not None else None,
            transition=fitted.transition(), **asdict(regime.diagnostics))
        complete_scores = result.log_score[result.complete & np.isfinite(result.log_score)]
        metadata = {
            "timeframe": timeframe, "family": family, "states": states, "scheme": scheme,
            "refit": refit, "period": period.index, "feature_set": set_name,
            "features": list(features), "train_start": str(stamps[lo]),
            "train_end": str(stamps[hi - 1]), "infer_start": str(period.infer_start),
            "infer_end": str(period.infer_end), "scaler_version": scaler.fingerprint(),
            "hyperparameters": _hyperparameters(regime, family), "seed": seed,
            "warm_started": previous is not None and causal.warm_start,
            "fresh_initialisations": bool(fresh), "alignment_order": order,
            "alignment_cost": alignment_cost, "causal": True, "live_safe": True,
            "regime_model_version": version,
            **{k2: v for k2, v in (provenance or {}).items() if k2 != "regime_model_version"},
        }
        model_id = (registry.register(model=family, timeframe=timeframe, states=states,
                                      fitted=fitted.model, scaler=scaler, metadata=metadata)
                    if registry is not None else f"{family}-{timeframe}-k{states}-p{period.index}")
        model_ids.append(model_id)
        ic = fitted.information_criteria()
        train_ll = fitted.train_log_likelihood()
        row.update({
            "status": "fitted", "model_id": model_id, "scaler_version": scaler.fingerprint(),
            "fit_seconds": fit_seconds, "train_log_likelihood": train_ll,
            "train_ll_per_obs": (train_ll / int(complete.sum())
                                 if train_ll is not None else None),
            "n_iter": getattr(fitted.model, "n_iter", None),
            "converged": getattr(fitted.model, "converged", None),
            "fresh_initialisations": bool(fresh), **ic,
            "alignment_cost": alignment_cost, "alignment_order": json.dumps(order),
            "alignment_ambiguity": json.dumps(ambiguity),
            "oos_ll_per_obs": float(complete_scores.mean()) if complete_scores.size else None,
            "oos_complete_rows": int(complete_scores.size),
            "occupancy": json.dumps((occupancy / max(occupancy.sum(), 1e-300)).tolist()),
            "overlap_ari": None if overlap is None else overlap["ari"],
            "overlap_agreement": None if overlap is None else overlap["agreement"],
            "anchor_bhattacharyya": json.dumps(_anchor_distance(anchor, fitted)),
            "means_raw": json.dumps(means_raw.tolist()),
            **{f"drift_{k2}": (json.dumps(v) if isinstance(v, list) else v)
               for k2, v in drift.items()},
            **{f"flag_{k2}": v for k2, v in flags.items()},
        })
        if isinstance(fitted.model, HMMModel):
            row["transition"] = json.dumps(fitted.model.transition.tolist())
            row["expected_durations"] = json.dumps(fitted.model.expected_durations().tolist())
        log_rows.append(row)
        parts.append({"slice": (i0, i1), "result": result, "model_id": model_id})
        previous = fitted
        if progress is not None:
            progress(f"{timeframe} {family} K={states} {scheme}/{refit} period {period.index + 1}"
                     f"/{len(periods)} ({period.infer_start:%Y-%m}) fit {fit_seconds:.1f}s")
    out = _assemble(parts, stamps, family, states, version)
    return WalkForwardResult(timeframe=timeframe, family=family, states=states, scheme=scheme,
                             refit=refit, feature_set=set_name, frame=out,
                             refits=pl.DataFrame(log_rows, infer_schema_length=None),
                             model_ids=model_ids, seconds=time.perf_counter() - started,
                             anchor={"model_id": model_ids[0] if model_ids else None,
                                     "canonical_order_feature":
                                         regime.baselines.volatility_feature})


def _hyperparameters(regime: RegimeConfig, family: str) -> dict[str, Any]:
    m = regime.models
    if family == "kmeans":
        cfg: Any = m.kmeans
    elif family.startswith("gmm"):
        cfg = m.gmm
    else:
        cfg = m.hmm
    return {**asdict(cfg), "scaling": regime.preprocessing.scaling,
            "clip": regime.preprocessing.clip,
            "missing_policy": regime.preprocessing.missing_policy}


def _assemble(parts: list[dict[str, Any]], stamps: pl.Series, family: str, k: int,
              version: str | None) -> pl.DataFrame:
    """Concatenate the periods into one per-bar table (research columns included)."""
    if not parts:
        return pl.DataFrame()
    idx = np.concatenate([np.arange(*p["slice"]) for p in parts])
    state = np.concatenate([p["result"].state for p in parts])
    columns: dict[str, Any] = {
        "timestamp": stamps.gather(pl.Series(idx)),
        "regime_model_id": np.concatenate([np.full(p["slice"][1] - p["slice"][0], p["model_id"])
                                           for p in parts]),
        f"{family}_state": state,
        "regime_age_bars": regime_age(state),
        "regime_feature_coverage": np.concatenate([p["result"].coverage for p in parts]),
        "regime_observed": np.concatenate([p["result"].scored for p in parts]),
        "complete": np.concatenate([p["result"].complete for p in parts]),
        "log_score": np.concatenate([p["result"].log_score for p in parts]),
    }
    if parts[0]["result"].probs is not None:
        probs = np.concatenate([p["result"].probs for p in parts])
        for j in range(k):
            columns[f"{family}_p{j}"] = probs[:, j]
        columns[f"{family}_state_confidence"] = np.concatenate(
            [p["result"].confidence for p in parts])
        columns[f"{family}_entropy"] = np.concatenate([p["result"].entropy for p in parts])
    if parts[0]["result"].next_probs is not None:
        nxt = np.concatenate([p["result"].next_probs for p in parts])
        for j in range(k):
            columns[f"next_state_p{j}"] = nxt[:, j]
        columns["leave_probability"] = np.concatenate([p["result"].leave for p in parts])
    frame = pl.DataFrame(columns)
    if version is not None:
        frame = frame.with_columns(pl.lit(version).alias("regime_model_version"))
    return frame


# ---------------------------------------------------------------------------
# The stored feature table
# ---------------------------------------------------------------------------
def regime_feature_frame(result: WalkForwardResult, *, float32: bool = True) -> pl.DataFrame:
    """The live-safe per-bar columns of a walk-forward run, schema enforced."""
    frame = result.frame
    keep = [c for c in frame.columns if c not in ("complete", "log_score")]
    enforce_live_safe(keep)
    out = frame.select(keep)
    ftype = pl.Float32 if float32 else pl.Float64
    exprs = []
    for name, dtype in out.schema.items():
        if name.endswith("_state"):
            exprs.append(pl.col(name).cast(pl.Int8))
        elif name == "regime_age_bars":
            exprs.append(pl.col(name).cast(pl.Int32))
        elif dtype in (pl.Float64, pl.Float32):
            exprs.append(pl.col(name).cast(ftype))
    return out.with_columns(exprs) if exprs else out


def write_regime_features(result: WalkForwardResult, regime: RegimeConfig, *,
                          manifest: dict[str, Any]) -> Path:
    """Partition by year under data/features/regime/... with a provenance manifest."""
    table = regime_feature_frame(result, float32=regime.features_output.float32)
    root = (regime.features_path / f"timeframe={result.timeframe}"
            / f"model={result.family}" / f"k={result.states}"
            / f"scheme={result.scheme}_{result.refit}")
    ensure_dir(root)
    for stale in root.rglob("*.parquet"):
        stale.unlink()
    table = table.with_columns(pl.col("timestamp").dt.year().alias("__year"))
    written = 0
    for (year,), part in table.group_by("__year", maintain_order=True):
        directory = ensure_dir(root / f"year={year}")
        tmp = directory / "part-0.parquet.partial"
        part.drop("__year").write_parquet(tmp, compression="zstd")
        tmp.replace(directory / "part-0.parquet")
        written += (directory / "part-0.parquet").stat().st_size
    columns = [c for c in table.columns if c != "__year"]
    payload = {
        **manifest, "rows": table.height, "bytes": written, "causal": True, "live_safe": True,
        "columns": {c: next(desc for pattern, desc in LIVE_SAFE_COLUMNS.items()
                            if re.fullmatch(pattern, c)) for c in columns},
        "model_ids": result.model_ids, "join_key": "timestamp",
        "first_timestamp": str(table["timestamp"][0]) if table.height else None,
        "last_timestamp": str(table["timestamp"][-1]) if table.height else None,
        "generated_utc": utc_now_iso(),
    }
    atomic_write_text(root / "_manifest.json", json.dumps(clean_json(payload), indent=1,
                                                          default=str) + "\n")
    return root
