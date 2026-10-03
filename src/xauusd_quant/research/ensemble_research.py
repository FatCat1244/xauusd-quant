r"""Ensembles, meta-models and predictive diversification, one timeframe at a time
(Prompt #11).

For every target x horizon of ``config/ensemble.yaml`` (never across targets or
horizons) this module

1. loads the Prompt #10 out-of-sample prediction table and refuses one that does
   not describe the current data (:func:`~..ensemble.data.load_pair`);
2. gives every base model an **eligibility status** from its Prompt #10
   development evidence (Step 2): ``leakage_failed``, ``deprecated``,
   ``null_failed``, ``weak``, ``unstable``, ``calibration_failed`` or ``eligible``;
   the eligible models, minus near-duplicates, form the ensemble *universe*;
3. runs the **meta walk-forward**: every method is scored on the evaluation blocks
   (2013-2021), and anything a method fits - weights, a stacking meta-model, a
   calibrator, conditional or trailing weights, a subset - is fitted on the
   out-of-sample rows of *earlier* blocks only, purged by ``h + embargo`` bars
   (:meth:`~..ensemble.data.PairPredictions.history`). The "best individual" is
   chosen the same way, on earlier blocks;
4. measures diversity (prediction / error / calibration-residual correlation,
   Q-statistic, disagreement), stability (year, quarter, regime, volatility,
   spread, session), calibration orders, disagreement and OOD against error,
   failure and common-mode failure, decile shape and alpha decay;
5. runs the controls: ensembles of models trained on shifted labels and of models
   on the random-walk / sign-flip pipelines (invariant 9), an intentionally
   useless noise model and a duplicated model.

Ablations (ensemble size, greedy subsets, family and feature-set diversification,
weaker models, leave one model / family out) live in :mod:`.ensemble_ablation`;
tables, the freeze rule and ledger rows in :mod:`.ensemble_reports`. Nothing here
reads the reserved period, and nothing outputs a trade.
"""

from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..ensemble.averaging import median_combination, weighted_mean
from ..ensemble.calibration import calibration_variants
from ..ensemble.config import EnsembleConfig
from ..ensemble.data import (
    PairPredictions,
    ensemble_prediction_coverage,
    load_pair,
    load_unit_predictions,
)
from ..ensemble.diagnostics import (
    alpha_decay,
    block_ood_percentiles,
    common_mode,
    condition_codes,
    failure_table,
    loss_by_quantile,
    quantile_codes,
)
from ..ensemble.disagreement import binary_entropy, disagreement_measures, grouped_means
from ..ensemble.diversity import diversity_table
from ..ensemble.meta_model import fit_meta_model
from ..ensemble.stability import period_table, primary_metric, score, weight_stability
from ..ensemble.stacking import stack_block
from ..ensemble.weighting import (
    apply_state_weights,
    conditional_weights,
    diversity_weights,
    dynamic_weights,
    loss_rows,
    performance_weights,
    turnover,
    weight_entropy,
)
from ..ml.config import MLConfig
from ..ml.datasets import (
    SESSION_LABELS,
    LeakageError,
    MLData,
    MLIntegrityError,
    assert_no_leakage,
    derive_target,
    load_ml_data,
    read_years,
)
from ..ml.splits import walk_forward_folds
from ..utils.clock import utc_now_iso
from ..utils.logging import get_logger
from ..utils.paths import ensure_dir
from .feature_research import write_json, write_table

__all__ = ["EnsembleConfigs", "PairRun", "TFContext", "build_universe", "eligibility_table",
           "load_tf_context", "pair_context", "run_ensemble_research", "run_pair"]

LOGGER = get_logger("research.ensemble")
STATUS_ORDER = ("leakage_failed", "deprecated", "null_failed", "untested", "weak", "unstable",
                "calibration_failed")
_SET_KEYS = ("standard", "extended", "minimal")


@dataclass
class EnsembleConfigs:
    ensemble: EnsembleConfig
    ml: Any                            # research.ml_research.MLConfigs (every Prompt #10 config)


@dataclass
class TFContext:
    timeframe: str
    cfgs: EnsembleConfigs
    data: MLData
    out_dir: Path
    ood: np.ndarray                    # OOD percentile per development row (its fold's scorer)
    extra: dict[str, np.ndarray]       # context columns outside the manifests (rv_percentile, ...)
    progress: Callable[[str], None] | None = None
    tables10: dict[str, pl.DataFrame] = field(default_factory=dict)

    @property
    def cfg(self) -> EnsembleConfig:
        return self.cfgs.ensemble

    @property
    def ml(self) -> MLConfig:
        cfg: MLConfig = self.cfgs.ml.ml
        return cfg

    @property
    def ml_dir(self) -> Path:
        return self.cfg.predictions_path / self.timeframe

    def log(self, message: str) -> None:
        LOGGER.info("[%s] %s", self.timeframe, message)
        if self.progress is not None:
            self.progress(f"[{self.timeframe}] {message}")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_tf_context(cfgs: EnsembleConfigs, timeframe: str, *,
                    progress: Callable[[str], None] | None = None) -> TFContext:
    """Development rows of *timeframe* (context, targets at every decay horizon) and the
    per-row OOD score of the walk-forward fold that predicted each row."""
    from ..features.factory import current_version_dir

    m = cfgs.ml
    data = load_ml_data(m.ml, m.selection, m.features, m.targets, m.config, m.regression,
                        timeframe, extra_horizons=tuple(cfgs.ensemble.alpha_decay_horizons))
    base = current_version_dir(m.features.factory_path, timeframe)
    wanted: list[str] = [c for c in ("rv_percentile", "regime_confidence") if c in data.registry]
    extra: dict[str, np.ndarray] = {}
    if wanted:
        frame = read_years(base, wanted, data.reserved_start)
        if not frame["timestamp"].equals(data.timestamps):
            raise MLIntegrityError(f"{timeframe}: context columns are not on the development bars")
        extra = {c: frame[c].cast(pl.Float64).fill_null(np.nan).to_numpy() for c in wanted}
        del frame
    folds = walk_forward_folds(data.timestamps, m.ml.walk_forward, horizon=5)
    x = data.design(data.feature_set(m.ml.default_feature_set), key=m.ml.default_feature_set)
    ood = block_ood_percentiles(x, folds)
    data.drop_designs()
    gc.collect()
    out = ensure_dir(cfgs.ensemble.results_path / timeframe)
    return TFContext(timeframe=timeframe, cfgs=cfgs, data=data, out_dir=out, ood=ood,
                     extra=extra, progress=progress)


def pair_context(tctx: TFContext, pair: PairPredictions) -> dict[str, np.ndarray]:
    """Everything known at t that groups or conditions the pair's rows."""
    rows = pair.rows
    d = tctx.data
    ctx: dict[str, np.ndarray] = {}
    for name in ("year", "regime_state", "vol_quartile", "spread_quartile", "session"):
        if name in d.context:
            ctx[name] = np.asarray(d.context[name][rows])
    for f in ("regime_p0", "regime_p1", "regime_p2", "regime_entropy", "fft_entropy_256",
              "wav_entropy_512", "log_rv_20"):
        if f in d.features:
            ctx[f] = d.features[f][rows].astype(np.float64)
    for f, v in tctx.extra.items():
        ctx[f] = v[rows].astype(np.float64)
    ctx["ood_score"] = tctx.ood[rows]
    ts = pair.timestamps
    ctx["month"] = (ts.dt.year().cast(pl.Int64) * 100 + ts.dt.month().cast(pl.Int64)).to_numpy()
    ctx["quarter"] = (ts.dt.year().cast(pl.Int64) * 10 + ts.dt.quarter().cast(pl.Int64)
                      ).to_numpy()
    _, day = np.unique(ts.dt.date().to_numpy(), return_inverse=True)
    ctx["day"] = day.astype(np.int64)
    return ctx


def _tables10(tctx: TFContext) -> dict[str, pl.DataFrame]:
    if not tctx.tables10:
        base = tctx.ml_dir / "tables"
        for name in ("configuration_summary", "null_comparison", "skill_beyond_base_rate"):
            p = base / f"{name}.parquet"
            tctx.tables10[name] = pl.read_parquet(p) if p.exists() else pl.DataFrame()
    return tctx.tables10


def _unit_jsons(tctx: TFContext, target: str, h: int, family: str, fset: str
                ) -> list[dict[str, Any]]:
    d = tctx.ml_dir / "units" / target / f"h{h}" / family / fset / "base"
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))
            if not p.name.endswith(".failed.json")]


def _manifest_key(tctx: TFContext, target: str, fset: str) -> str:
    return fset if fset in _SET_KEYS else f"target_{tctx.ml.targets[target].kind}"


# ---------------------------------------------------------------------------
# Step 2: eligibility gate
# ---------------------------------------------------------------------------
def _null_rows(nc: pl.DataFrame, target: str, h: int, fam: str, fset: str,
               variants: tuple[str, ...]) -> list[dict[str, Any]]:
    if nc.is_empty():
        return []
    part = nc.filter((pl.col("target") == target) & (pl.col("horizon") == h)
                     & (pl.col("family") == fam) & (pl.col("feature_set") == fset))
    out = []
    for r in part.iter_rows(named=True):
        v = str(r["variant"])
        if any(v == want or v.startswith(want + "__") or (want == "shift"
                                                          and v.startswith("null-shift"))
               for want in variants):
            out.append(r)
    return out


def _null_verdict(rows: list[dict[str, Any]], expected: tuple[str, ...] = ()
                  ) -> tuple[str, list[str]]:
    """passed / failed / untested; failure: mean real - null <= 0, or > 1 losing block."""
    if not rows:
        return "untested", []
    notes, failed, tested, unknown = [], False, False, False
    for name in expected:
        if not any(str(r["variant"]) == name or str(r["variant"]).startswith(name + "__")
                   or (name == "shift" and str(r["variant"]).startswith("null-shift"))
                   for r in rows):
            unknown = True
            notes.append(f"{name}: unavailable")
    for r in rows:
        diff, blocks_raw, wins = r.get("real_minus_control"), r.get("blocks"), r.get("wins")
        known = all(v is not None and np.isfinite(v) for v in (diff, blocks_raw, wins))
        blocks = int(blocks_raw) if known and blocks_raw is not None else 0
        if not known or blocks <= 0:
            unknown = True
            notes.append(f"{r['variant']}: undefined")
            continue
        assert diff is not None and wins is not None
        tested = True
        bad = diff <= 0 or (blocks - int(wins)) > 1
        failed |= bad
        notes.append(f"{r['variant']}: {diff:+.4f} ({int(wins)}/{blocks} blocks)")
    if not tested:
        return "untested", notes
    return ("failed" if failed else "untested" if unknown else "passed"), notes


def eligibility_table(tctx: TFContext, pair: PairPredictions) -> pl.DataFrame:
    """One row per base model of the pair: evidence, every check and the status (Step 2)."""
    cfg, ml = tctx.cfg, tctx.ml
    el = cfg.eligibility
    tabs = _tables10(tctx)
    cs, nc, sb = tabs["configuration_summary"], tabs["null_comparison"], \
        tabs["skill_beyond_base_rate"]
    data = tctx.data
    tspec = ml.targets[pair.target]
    target_y = data.target(tspec, pair.horizon, ml.log_floor).y
    sample = np.unique(np.linspace(0, data.n - 1, 100_000).astype(np.int64))
    gap = pair.horizon + ml.walk_forward.embargo_bars
    min_folds = int(el.get("min_folds_beating_baseline", 4))
    registered = tuple(str(v) for v in el.get("registered_nulls") or ("shift",))
    reported = tuple(str(v) for v in el.get("reported_nulls") or ())
    rows = []
    for model in pair.model_names:
        fam, fset = pair.family(model), pair.feature_set(model)
        reasons: list[str] = []
        flags: dict[str, bool] = dict.fromkeys(STATUS_ORDER, False)
        summ = cs.filter((pl.col("target") == pair.target) & (pl.col("horizon") == pair.horizon)
                         & (pl.col("family") == fam) & (pl.col("feature_set") == fset)
                         & (pl.col("variant") == "base")) if not cs.is_empty() else cs
        s = summ.row(0, named=True) if summ.height else {}
        units = _unit_jsons(tctx, pair.target, pair.horizon, fam, fset)
        features = list(units[0]["features"]) if units else []
        # -- leakage: registered live-safe features, canary, purged folds, rows in their block
        try:
            assert_no_leakage(features, data.registry)
            xs = np.column_stack([data.features[f][sample] for f in features]) if features \
                else None
            if xs is not None:
                assert_no_leakage(features, data.registry, x=xs, targets=[target_y[sample]])
        except (LeakageError, KeyError) as exc:
            flags["leakage_failed"] = True
            reasons.append(f"leakage: {exc}")
        for u in units:
            f = u["fold"]
            fit, inner, val = f["fit"], f["inner"], f["validate"]
            if inner[1] > inner[0] and fit[1] > inner[0] - gap:
                flags["leakage_failed"] = True
                reasons.append(f"{f['name']}: fit rows reach the inner slice")
            if inner[1] > val[0] - gap:
                flags["leakage_failed"] = True
                reasons.append(f"{f['name']}: training outcomes reach the validation block")
            if f["name"] in pair.block_names:
                pos = pair.block_positions(pair.block_names.index(f["name"]))
                r = pair.rows[pos]
                if r.size and (r.min() < val[0] or r.max() >= val[1]):
                    flags["leakage_failed"] = True
                    reasons.append(f"{f['name']}: predictions outside the validation block")
        if len(units) != len(pair.block_names):
            flags["deprecated"] = True
            reasons.append(f"{len(units)} unit files for {len(pair.block_names)} blocks")
        # -- deprecated: features / feature-set id differ from the current manifest
        key = _manifest_key(tctx, pair.target, fset)
        cur_id = data.manifest_ids.get(key)
        side_id = (pair.models.get(model) or {}).get("feature_set_id")
        if key not in data.manifests or data.manifests[key] != features or side_id != cur_id:
            flags["deprecated"] = True
            reasons.append(f"feature set {side_id} is not the current {cur_id}")
        # -- nulls (registered gate; the post-hoc sign flip only reported)
        rep_fam = fam
        nrows = _null_rows(nc, pair.target, pair.horizon, fam, fset, registered)
        if not nrows:
            rep_fam = ("lightgbm" if fam in ml.tree_families
                       else ml.reference_linear(tspec.task))
            nrows = _null_rows(nc, pair.target, pair.horizon, rep_fam, fset, registered)
        required_nulls = tuple(n for n in registered
                               if tspec.kind == "reversion" or not n.startswith("pipeline-"))
        null_status, null_notes = _null_verdict(nrows, required_nulls)
        if null_status == "untested":
            flags["untested"] = True
            reasons.append("registered null evidence unavailable or incomplete")
        if null_status == "failed":
            flags["null_failed"] = True
            reasons.append("within a registered null: " + "; ".join(null_notes))
        posthoc_status, posthoc_notes = _null_verdict(
            _null_rows(nc, pair.target, pair.horizon, rep_fam, fset, reported))
        beyond = None
        if tspec.is_classification and el.get("beyond_base_rate", True) and not sb.is_empty():
            b = sb.filter((pl.col("target") == pair.target) & (pl.col("horizon") == pair.horizon)
                          & (pl.col("family") == fam) & (pl.col("feature_set") == fset))
            if b.height:
                beyond = b.row(0, named=True)
                if (beyond["skill_beyond_base_rate"] is None
                        or beyond["skill_beyond_base_rate"] <= 0
                        or int(beyond["wins"] or 0) < min_folds):
                    flags["null_failed"] = True
                    reasons.append("no skill beyond the recalibrated constant "
                                   f"({beyond['skill_beyond_base_rate']}, {beyond['wins']} wins)")
        # -- quality and stability (an undefined statistic is untested, never a pass or fail)
        def fnum(v: Any) -> float | None:
            return float(v) if v is not None and bool(np.isfinite(v)) else None

        mean, mn = s.get("mean"), s.get("min")
        mean_f, mn_f = fnum(mean), fnum(mn)
        beats = int(s.get("blocks_beating_baseline") or 0)
        if mean_f is None:
            flags["untested"] = True
            reasons.append("no defined development metric")
        elif mean_f <= 0 or beats < min_folds:
            flags["weak"] = True
            reasons.append(f"beats the constant in {beats} blocks (mean {mean_f:.4f})")
        elif mn_f is not None and mn_f < float(el.get("min_block_ratio", 0.5)) * mean_f:
            flags["unstable"] = True
            reasons.append(f"worst block {mn_f:.4f} < {el.get('min_block_ratio')} x "
                           f"mean {mean_f:.4f}")
        if tspec.is_classification:
            ece, bsk = s.get("mean_ece"), s.get("mean_brier_skill")
            ece_f, bsk_f = fnum(ece), fnum(bsk)
            if ece_f is None or bsk_f is None:
                flags["untested"] = True
                reasons.append(f"calibration evidence undefined (ECE {ece}, Brier skill {bsk})")
            else:
                if ece_f > float(el.get("max_mean_ece", 0.05)):
                    flags["calibration_failed"] = True
                    reasons.append(f"mean ECE {ece_f:.4f}")
                if bsk_f <= float(el.get("min_mean_brier_skill", 0.0)):
                    flags["calibration_failed"] = True
                    reasons.append(f"mean Brier skill {bsk_f:.4f}")
        status = next((st for st in STATUS_ORDER if flags[st]), "eligible")
        rows.append({
            "timeframe": pair.timeframe, "target": pair.target, "horizon": pair.horizon,
            "model": model, "family": fam, "feature_set": fset, "feature_set_id": side_id,
            "features": len(features), "metric": s.get("metric"), "mean": mean,
            "se": s.get("se"), "min": mn, "blocks": s.get("blocks"),
            "blocks_beating_baseline": beats, "mean_auc": s.get("mean_auc"),
            "mean_ece": s.get("mean_ece"), "mean_brier_skill": s.get("mean_brier_skill"),
            "mean_rank_ic": s.get("mean_rank_ic"), "mean_mse_skill": s.get("mean_mse_skill"),
            "recent_metric": s.get("recent_metric"),
            "null_evidence_from": f"{rep_fam}|{fset}", "registered_null": null_status,
            "registered_null_detail": "; ".join(null_notes),
            "posthoc_sign_flip": posthoc_status, "posthoc_detail": "; ".join(posthoc_notes),
            "skill_beyond_base_rate": None if beyond is None else beyond[
                "skill_beyond_base_rate"],
            **{f"flag_{k}": v for k, v in flags.items()},
            "status": status, "reasons": " | ".join(reasons)})
    return pl.DataFrame(rows, infer_schema_length=None)


def _simplicity(ml: MLConfig, fam: str, n_features: int, fset: str) -> tuple[int, int, int]:
    order = list(ml.simplicity_order)
    return (order.index(fam) if fam in order else len(order), n_features,
            0 if fset == ml.default_feature_set else 1)


def build_universe(tctx: TFContext, pair: PairPredictions, elig: pl.DataFrame
                   ) -> tuple[list[str], pl.DataFrame]:
    """Eligible models minus near-duplicates (simpler, then better, kept), capped."""
    cfg, ml = tctx.cfg, tctx.ml
    thr = float(cfg.universe.get("max_prediction_correlation", 0.995))
    ok = elig.filter(pl.col("status") == "eligible")
    if ok.is_empty():
        return [], pl.DataFrame()
    info = {r["model"]: r for r in ok.iter_rows(named=True)}
    order = sorted(info, key=lambda m: (*_simplicity(ml, info[m]["family"], int(info[m]["features"]),
                                                      info[m]["feature_set"]),
                                        -(info[m]["mean"] or 0.0)))
    mat = pair.matrix(order)
    cov = np.isfinite(mat).all(axis=1)
    sub = mat[cov]
    if sub.shape[0] > 400_000:
        sub = sub[np.unique(np.linspace(0, sub.shape[0] - 1, 400_000).astype(np.int64))]
    corr = np.corrcoef(sub, rowvar=False) if len(order) > 1 else np.ones((1, 1))
    kept: list[int] = []
    decision = []
    for i, m in enumerate(order):
        twin = next((order[j] for j in kept if corr[i, j] >= thr), None)
        if twin is None:
            kept.append(i)
        decision.append({"model": m, "redundant_with": twin,
                         "max_correlation_with_kept": float(max((corr[i, j] for j in kept
                                                                 if j != i), default=0.0))})
    universe = [order[i] for i in kept]
    universe.sort(key=lambda m: -(info[m]["mean"] or 0.0))
    cap = cfg.maximum_model_count
    dropped = universe[cap:]
    universe = universe[:cap]
    for d in decision:
        d["in_universe"] = d["model"] in universe
        d["capped"] = d["model"] in dropped
    return universe, pl.DataFrame(decision, infer_schema_length=None)


# ---------------------------------------------------------------------------
# The meta walk-forward of one pair
# ---------------------------------------------------------------------------
class PairRun:
    """Every series of one target x horizon x timeframe, scored block by block."""

    def __init__(self, tctx: TFContext, pair: PairPredictions, universe: list[str],
                 elig: pl.DataFrame) -> None:
        self.tctx = tctx
        self.cfg = tctx.cfg
        self.pair = pair
        self.universe = list(universe)
        self.elig = elig
        self.task = pair.task
        self.metric = primary_metric(pair.task)
        self.ctx = pair_context(tctx, pair)
        nb = len(pair.block_names)
        self.eval_blocks = [k for k in self.cfg.evaluation_blocks if k < nb]
        self.embargo = self.cfg.embargo_bars
        self.y, self.base, self.y_raw = pair.label, pair.base, pair.label_raw
        self.series: dict[str, np.ndarray] = {}
        self.kind: dict[str, str] = {}
        self.members: dict[str, list[str]] = {}
        self.block_weights: dict[str, dict[str, np.ndarray]] = {}
        self.weight_names: dict[str, list[str]] = {}
        self.notes: dict[str, Any] = {}
        self._scores: dict[tuple[str, int], dict[str, Any]] = {}
        self.cov = pair.covered(self.universe) if self.universe else np.isfinite(self.y)
        self.eval_mask = np.isin(pair.block, self.eval_blocks)

    # -- bookkeeping -----------------------------------------------------------
    def add(self, name: str, values: np.ndarray, kind: str, members: list[str]) -> None:
        self.series[name] = np.asarray(values, dtype=np.float64)
        self.kind[name] = kind
        self.members[name] = list(members)
        for key in [k for k in self._scores if k[0] in (name, f"prior:{name}")]:
            del self._scores[key]

    def empty(self) -> np.ndarray:
        return np.full(self.pair.n, np.nan)

    def block_score(self, name: str, k: int) -> dict[str, Any]:
        key = (name, k)
        if key not in self._scores:
            pos = self.pair.block_positions(k)
            s = self.series[name]
            self._scores[key] = score(self.task, s[pos], self.y[pos], self.base[pos],
                                      self.y_raw[pos])
        return self._scores[key]

    def primary(self, name: str, k: int) -> float | None:
        v = self.block_score(name, k).get(self.metric)
        return None if v is None or not np.isfinite(v) else float(v)

    def prior_scores(self, name: str, k: int) -> list[dict[str, Any]]:
        """Metrics of *name* on each earlier block, on its **purged** rows only: the rows
        of :meth:`~..ensemble.data.PairPredictions.history` for block *k* - the last
        ``h + embargo`` rows of block ``k - 1`` resolve inside block *k* and are left out."""
        key = (f"prior:{name}", k)
        if key not in self._scores:
            hist = self.pair.history(k, self.embargo)
            s = self.series[name]
            out = []
            for j in range(k):
                pos = hist[self.pair.block[hist] == j]
                if pos.size < 100:
                    continue
                out.append(score(self.task, s[pos], self.y[pos], self.base[pos],
                                 self.y_raw[pos]))
            self._scores[key] = {"blocks": out}
        return list(self._scores[key]["blocks"])

    def prior_mean(self, name: str, k: int, metric: str | None = None) -> float | None:
        got = [m.get(metric or self.metric) for m in self.prior_scores(name, k)]
        vals = [float(v) for v in got if v is not None and np.isfinite(v)]
        return float(np.mean(vals)) if vals else None

    def record_weights(self, scheme: str, block: int, w: np.ndarray, names: list[str]) -> None:
        self.block_weights.setdefault(scheme, {})[self.pair.block_names[block]] = \
            np.asarray(w, dtype=np.float64)
        self.weight_names[scheme] = list(names)

    def matrix(self, models: list[str] | None = None) -> np.ndarray:
        return self.pair.matrix(models if models is not None else self.universe)

    def history(self, k: int, models: list[str] | None = None) -> np.ndarray:
        hist = self.pair.history(k, self.embargo)
        cov = self.pair.covered(models) if models is not None else self.cov
        return hist[cov[hist]]

    # -- series ------------------------------------------------------------------
    def individuals(self) -> None:
        self.add("constant", self.base, "baseline", [])
        for m in self.pair.model_names:
            self.add(f"model:{m}", self.pair.prediction(m), "individual", [m])

    def best_individual(self) -> None:
        """The universe model with the best mean primary metric on the (purged) blocks before
        each evaluation block - a choice an analyst could have made at the time."""
        out = self.empty()
        picks = {}
        for k in self.eval_blocks:
            best, best_v = None, -np.inf
            for m in self.universe:
                v = self.prior_mean(f"model:{m}", k)
                if v is not None and v > best_v:
                    best, best_v = m, v
            if best is None:
                continue
            pos = self.pair.block_positions(k)
            out[pos] = self.pair.prediction(best)[pos]
            picks[self.pair.block_names[k]] = best
        self.add("best_individual", out, "method", self.universe)
        self.notes["best_individual_picks"] = picks

    def fixed_combinations(self, models: list[str], *, prefix: str = "",
                           kind: str = "method") -> None:
        p = self.matrix(models)
        mean = weighted_mean(p) if models else self.empty()
        med = median_combination(p) if models else self.empty()
        for name, values in (("simple_average", mean), ("median", med)):
            v = self.empty()
            v[self.eval_mask] = values[self.eval_mask]
            self.add(prefix + name, v, kind, models)

    def simple_average_all_blocks(self, models: list[str]) -> np.ndarray:
        return weighted_mean(self.matrix(models))

    def weighted_methods(self) -> None:
        cfg = self.cfg
        u = self.universe
        p = self.matrix()
        shrinks = [float(s) for s in cfg.weighting.get("shrink", [0.0, 0.5])]
        lambdas = [float(v) for v in cfg.diversity.get("lambdas", [0.5, 1.0])]
        dshrink = float(cfg.diversity.get("shrink", 0.5))
        dup = float(cfg.diversity.get("duplicate_correlation", 0.999))
        outs: dict[str, np.ndarray] = {}
        for k in self.eval_blocks:
            hist = self.history(k)
            pos = self.pair.block_positions(k)
            ph, yh, bh = p[hist], self.y[hist], self.base[hist]
            for s in shrinks:
                name = f"performance_weighted_s{int(round(s * 100))}"
                w, q = performance_weights(ph, yh, bh, self.task, shrink=s)
                outs.setdefault(name, self.empty())[pos] = weighted_mean(p[pos], w)
                self.record_weights(name, k, w, u)
                self.notes.setdefault("history_skill", {})[self.pair.block_names[k]] = \
                    dict(zip(u, [float(v) for v in q], strict=True))
            for lam in lambdas:
                name = f"diversity_weighted_l{int(round(lam * 100))}"
                w, info = diversity_weights(ph, yh, bh, self.task, lam=lam, shrink=dshrink,
                                            duplicate_correlation=dup)
                outs.setdefault(name, self.empty())[pos] = weighted_mean(p[pos], w)
                self.record_weights(name, k, w, u)
                self.notes.setdefault("diversity_clusters", {})[
                    f"{name}|{self.pair.block_names[k]}"] = [[u[i] for i in g]
                                                              for g in info["clusters"]]
        for name, values in outs.items():
            self.add(name, values, "method" if name in (self._candidate("performance_weighted"),
                                                        self._candidate("diversity_weighted"))
                     else "variant", u)

    def _candidate(self, method: str) -> str:
        if method == "performance_weighted":
            return f"performance_weighted_s{int(round(float(self.cfg.weighting.get('candidate_shrink', 0.5)) * 100))}"
        if method == "diversity_weighted":
            return f"diversity_weighted_l{int(round(float(self.cfg.diversity.get('candidate_lambda', 0.5)) * 100))}"
        return method

    def context_inputs(self) -> dict[str, np.ndarray]:
        """Step 26: the few context inputs of the context meta-model, each known at t."""
        wanted = list(self.cfg.stacking.get("context_features") or [])
        out = {}
        for name in wanted:
            if name == "disagreement":
                out[name] = disagreement_measures(self.matrix())["prediction_std"]
            elif name in self.ctx:
                out[name] = np.asarray(self.ctx[name], dtype=np.float64)
        return out

    def stacking_methods(self) -> None:
        params = {"logistic_C": self.cfg.stacking.get("logistic_C", 1.0),
                  "ridge_alpha": self.cfg.stacking.get("ridge_alpha", 1.0),
                  "max_rows": self.cfg.stacking.get("max_rows")}
        u = self.universe
        context = self.context_inputs()
        for name, ctx in (("stacking", None), ("stacking_context", context)):
            out = self.empty()
            for k in self.eval_blocks:
                pos = self.pair.block_positions(k)
                pred, meta, hist = stack_block(self.pair, u, k, embargo=self.embargo,
                                               context=ctx, params=params,
                                               seed=self.cfg.random_seed)
                out[pos] = pred
                shares = meta.weight_shares()
                self.record_weights(name, k, np.array([shares[m] for m in u]), u)
                self.notes.setdefault(f"{name}_coefficients", {})[self.pair.block_names[k]] = {
                    "inputs": meta.inputs, "coef": [float(c) for c in meta.coef],
                    "intercept": meta.intercept, "history_rows": int(hist.size)}
            self.add(name, out, "method" if name == "stacking" else "variant", u)

    def walk_forward_universe(self) -> None:
        """Robustness of the gate. The universe is chosen on Prompt #10's evidence over all
        five blocks (Step 2), so a model's later blocks influence whether it enters the
        early evaluation blocks' ensembles. Here the universe of block *k* is rebuilt from
        the purged earlier blocks only - the same quality, stability and calibration
        criteria, near-duplicates pruned on the history rows - and the simple average and
        the stack are scored on it. The structural statuses (leakage, deprecated, null,
        untested) stay those of the gate."""
        el = self.cfg.eligibility
        ml = self.tctx.ml
        ratio = float(el.get("min_block_ratio", 0.5))
        thr = float(self.cfg.universe.get("max_prediction_correlation", 0.995))
        cap = self.cfg.maximum_model_count
        skill_key = "log_loss_skill" if self.pair.is_classification else "mse_skill"
        rows = {r["model"]: r for r in self.elig.iter_rows(named=True)}
        candidates = [m for m, r in rows.items() if r["status"] not in (
            "leakage_failed", "deprecated", "null_failed", "untested")]
        params = {"logistic_C": self.cfg.stacking.get("logistic_C", 1.0),
                  "ridge_alpha": self.cfg.stacking.get("ridge_alpha", 1.0),
                  "max_rows": self.cfg.stacking.get("max_rows")}
        simple, stack = self.empty(), self.empty()
        picks: dict[str, list[str]] = {}

        def fin(vals: list[Any]) -> list[float]:
            return [float(v) for v in vals if v is not None and np.isfinite(v)]

        for k in self.eval_blocks:
            keep: dict[str, float] = {}
            for m in candidates:
                prior = self.prior_scores(f"model:{m}", k)
                prim = fin([p.get(self.metric) for p in prior])
                sk = fin([p.get(skill_key) for p in prior])
                if not prim or not sk:
                    continue
                mean = float(np.mean(prim))
                beats = sum(1 for v in sk if v > 1e-12)
                if mean <= 0 or beats < max(1, len(sk) - 1) or min(prim) < ratio * mean:
                    continue
                if self.pair.is_classification:
                    ece = fin([p.get("ece") for p in prior])
                    bsk = fin([p.get("brier_skill") for p in prior])
                    if not ece or not bsk or float(np.mean(ece)) > float(el.get(
                            "max_mean_ece", 0.05)) or float(np.mean(bsk)) <= float(el.get(
                            "min_mean_brier_skill", 0.0)):
                        continue
                keep[m] = mean
            order = sorted(keep, key=lambda m: (*_simplicity(ml, rows[m]["family"],
                                                             int(rows[m]["features"] or 0),
                                                             rows[m]["feature_set"]),
                                                -keep[m]))
            hist = self.history(k, order) if order else np.zeros(0, dtype=np.int64)
            chosen: list[str] = []
            if order and hist.size > 1:
                mat = self.pair.matrix(order)[hist]
                corr = np.corrcoef(mat, rowvar=False) if len(order) > 1 else np.ones((1, 1))
                for i, m in enumerate(order):
                    if not any(corr[i, order.index(c)] >= thr for c in chosen):
                        chosen.append(m)
            chosen = sorted(chosen, key=lambda m: -keep[m])[:cap]
            picks[self.pair.block_names[k]] = chosen
            if len(chosen) < self.cfg.minimum_model_count:
                continue
            pos = self.pair.block_positions(k)
            simple[pos] = weighted_mean(self.pair.matrix(chosen)[pos])
            pred, _, _ = stack_block(self.pair, chosen, k, embargo=self.embargo, params=params,
                                     seed=self.cfg.random_seed)
            stack[pos] = pred
        self.add("simple_average_wf_universe", simple, "variant", candidates)
        self.add("stacking_wf_universe", stack, "variant", candidates)
        self.notes["walk_forward_universe"] = picks

    def conditional_methods(self) -> None:
        cfg = self.cfg.conditional
        u = self.universe
        p = self.matrix()
        shrink = float(cfg.get("shrink", 0.5))
        min_rows = float(cfg.get("min_rows_per_state", 5000))
        states: dict[str, np.ndarray] = {}
        probs_names = [c for c in cfg.get("regime_probabilities") or [] if c in self.ctx]
        if probs_names:
            states["regime_conditioned"] = np.column_stack([self.ctx[c] for c in probs_names])
        if "vol_quartile" in self.ctx:
            nb = int(cfg.get("volatility_buckets", 4))
            q = np.asarray(self.ctx["vol_quartile"], dtype=np.int64)
            onehot = np.full((q.size, nb), np.nan)
            ok = q >= 0
            onehot[ok] = 0.0
            onehot[np.flatnonzero(ok), q[ok]] = 1.0
            states["volatility_conditioned"] = onehot
        for name, probs in states.items():
            out = self.empty()
            for k in self.eval_blocks:
                hist = self.history(k)
                pos = self.pair.block_positions(k)
                w_states, w0, own = conditional_weights(p[hist], self.y[hist], self.base[hist],
                                                        self.task, probs[hist], shrink=shrink,
                                                        min_rows=min_rows)
                rw = apply_state_weights(probs[pos], w_states, w0)
                out[pos] = weighted_mean(p[pos], rw)
                for s_i in range(w_states.shape[0]):
                    self.record_weights(f"{name}:state{s_i}", k, w_states[s_i], u)
                self.notes.setdefault(f"{name}_own_states", {})[self.pair.block_names[k]] = own
            self.add(name, out, "method", u)

    def dynamic_method(self) -> None:
        cfg = self.cfg.dynamic
        u = self.universe
        p = self.matrix()
        cpos = np.flatnonzero(self.cov)
        evaluate = np.flatnonzero(self.eval_mask[cpos])
        day = self.ctx["day"]
        # the first bar of each day on the full timeline (a covered subset may start later)
        first_bar = self.pair.rows[np.searchsorted(day, day, side="left")]
        dw = dynamic_weights(p[cpos], self.y[cpos], self.base[cpos], self.pair.rows[cpos],
                             day[cpos], evaluate, task=self.task,
                             horizon=self.pair.horizon,
                             window_days=int(cfg.get("window_days", 60)),
                             min_rows=int(cfg.get("min_rows", 2000)),
                             shrink=float(cfg.get("shrink", 0.5)),
                             first_bar=first_bar[cpos])
        out = self.empty()
        tgt = cpos[evaluate]
        out[tgt] = weighted_mean(p[tgt], dw.row_weights)
        self.add("dynamic", out, "method", u)
        to = turnover(dw.update_weights)
        ent = [weight_entropy(w)[0] for w in dw.update_weights]
        self.notes["dynamic"] = {"updates": int(dw.update_weights.shape[0]),
                                 "mean_turnover": float(to.mean()) if to.size else 0.0,
                                 "max_turnover": float(to.max()) if to.size else 0.0,
                                 "mean_entropy": float(np.mean(ent)) if ent else None,
                                 "updates_with_equal_weights_for_lack_of_rows": int(
                                     (dw.update_rows_used < int(cfg.get("min_rows", 2000))).sum())}
        stamps = self.pair.timestamps.gather(cpos[dw.update_positions])
        self.notes["dynamic_timeline"] = pl.DataFrame({
            "timestamp": stamps, **{m: dw.update_weights[:, j] for j, m in enumerate(u)},
            "rows_used": dw.update_rows_used,
            "turnover": np.concatenate([[np.nan], to]) if to.size else np.full(
                dw.update_weights.shape[0], np.nan)})
        for k in self.eval_blocks:
            sel = self.pair.block[cpos[dw.update_positions]] == k
            if sel.any():
                self.record_weights("dynamic", k, dw.update_weights[sel].mean(axis=0), u)

    def calibration_research(self) -> None:
        if not self.pair.is_classification:
            return
        methods = [str(m) for m in self.cfg.calibration.get("methods", ["sigmoid", "isotonic"])]
        min_rows = int(self.cfg.calibration.get("isotonic_min_rows", 20000))
        outs: dict[str, np.ndarray] = {}
        params: dict[str, Any] = {}
        for k in self.eval_blocks:
            preds, prm = calibration_variants(self.pair, self.universe, k, embargo=self.embargo,
                                              methods=methods, min_rows=min_rows)
            pos = self.pair.block_positions(k)
            for name, v in preds.items():
                outs.setdefault(name, self.empty())[pos] = v
            params[self.pair.block_names[k]] = prm
        for name, v in outs.items():
            self.add(f"calibration:{name}", v, "calibration", self.universe)
        self.notes["calibrators"] = params

    # -- tables ------------------------------------------------------------------
    def block_table(self, names: list[str] | None = None) -> pl.DataFrame:
        rows = []
        for name in names or list(self.series):
            for k in self.eval_blocks:
                met = self.block_score(name, k)
                rows.append({"series": name, "kind": self.kind[name],
                             "members": len(self.members[name]), "block": self.pair.block_names[k],
                             "block_index": k,
                             **{key: v for key, v in met.items()
                                if not isinstance(v, (list, dict))}})
        return pl.DataFrame(rows, infer_schema_length=None)

    def eval_positions(self, models: list[str] | None = None) -> np.ndarray:
        cov = self.pair.covered(models) if models is not None else self.cov
        return np.flatnonzero(self.eval_mask & cov)


# ---------------------------------------------------------------------------
# Controls (Steps 75-77)
# ---------------------------------------------------------------------------
def noise_and_duplicate_controls(run: PairRun) -> dict[str, Any]:
    """An intentionally useless model and a duplicated model, inserted into the universe:
    the weights each method gives them, and what they do to the ensemble."""
    pair, u = run.pair, run.universe
    rng = np.random.default_rng(run.cfg.random_seed + 76)
    p = run.matrix()
    z = rng.standard_normal(pair.n)
    best = u[0]
    out: dict[str, Any] = {"duplicated": best, "noise_scale": {}, "blocks": {}}
    labels = {"noise": [*u, "noise|synthetic"], "duplicate": [*u, f"{best}|copy"]}
    shrink = float(run.cfg.weighting.get("candidate_shrink", 0.5))
    lam = float(run.cfg.diversity.get("candidate_lambda", 0.5))
    dup = float(run.cfg.diversity.get("duplicate_correlation", 0.999))
    for tag in ("noise", "duplicate"):
        avg = run.empty()
        for k in run.eval_blocks:
            hist = run.history(k)
            pos = pair.block_positions(k)
            if tag == "noise":
                # amplitude of the useless model: the constituents' typical spread on the
                # history rows (nothing from block k or later)
                spread = float(np.nanmean(np.nanstd(p[hist], axis=0)))
                noise = run.base + spread * z
                if pair.is_classification:
                    noise = np.clip(noise, 1e-4, 1 - 1e-4)
                mat = np.column_stack([p, noise])
                out["noise_scale"][pair.block_names[k]] = spread
            else:
                mat = np.column_stack([p, pair.prediction(best)])
            ph, yh, bh = mat[hist], run.y[hist], run.base[hist]
            w_perf, q = performance_weights(ph, yh, bh, run.task, shrink=shrink)
            w_perf0, _ = performance_weights(ph, yh, bh, run.task, shrink=0.0)
            w_div, info = diversity_weights(ph, yh, bh, run.task, lam=lam, shrink=shrink,
                                            duplicate_correlation=dup)
            names = labels[tag]
            meta = fit_meta_model(ph, yh, names, constituents=len(names), task=run.task,
                                  logistic_c=float(run.cfg.stacking.get("logistic_C", 1.0)),
                                  ridge_alpha=float(run.cfg.stacking.get("ridge_alpha", 1.0)),
                                  max_rows=run.cfg.stacking.get("max_rows"),
                                  seed=run.cfg.random_seed)
            shares = meta.weight_shares()
            avg[pos] = weighted_mean(mat[pos])
            j = len(u)                                   # the inserted column
            entry: dict[str, Any] = {
                "equal_weight": 1.0 / len(names),
                "performance_weight": float(w_perf[j]), "performance_weight_raw": float(w_perf0[j]),
                "diversity_weight": float(w_div[j]),
                "stacking_share": float(shares[names[j]]),
                "skill_on_history": float(q[j]),
                "clusters": [[names[i] for i in g] for g in info["clusters"]]}
            if tag == "duplicate":
                # the original's weight without its copy: a duplicate-aware method gives the
                # pair together exactly that much (Step 77)
                alone_perf, _ = performance_weights(p[hist], yh, bh, run.task, shrink=shrink)
                alone_div, _ = diversity_weights(p[hist], yh, bh, run.task, lam=lam,
                                                 shrink=shrink, duplicate_correlation=dup)
                entry.update({
                    "original_performance_weight": float(w_perf[0]),
                    "pair_performance_weight": float(w_perf[0] + w_perf[j]),
                    "alone_performance_weight": float(alone_perf[0]),
                    "original_diversity_weight": float(w_div[0]),
                    "pair_diversity_weight": float(w_div[0] + w_div[j]),
                    "alone_diversity_weight": float(alone_div[0]),
                    "pair_stacking_share": float(shares[names[0]] + shares[names[j]])})
            out["blocks"].setdefault(tag, {})[pair.block_names[k]] = entry
        run.add(f"control:{tag}_simple_average", avg, "control", labels[tag])
    if len(u) >= 1:
        dmat = np.column_stack([p[run.cov], pair.prediction(best)[run.cov]])
        rows = diversity_table(dmat, run.y[run.cov], [*u, f"{best}|copy"], task=run.task,
                               max_rows=200_000)
        out["duplicate_diversity"] = [r for r in rows if r["b"].endswith("|copy")]
    return out


def _null_label_arrays(tctx: TFContext, name: str) -> dict[str, Any]:
    """Labels of the residual targets on a Prompt #8 pipeline null (the same null process,
    seed and engines Prompt #10 trained its null units on), development rows only.

    Prompt #10 built the null over the whole bar history; here it is built from the
    development bars alone, so no reserved price enters memory. The labels are the same:
    with the same seed every development row gets the same random draws, the sign-flip
    null uses each bar's own return, and the random walk's one scale (the close-to-close
    sd, which differs between the two) cancels in every label used here -
    ``|eps_{t+h}| < c |eps_t|`` and ``(|eps_t| - |eps_{t+h}|) / sigma_t``."""
    from ..ensemble.streaming import load_development_bars
    from ..features.factory import null_bar_series
    from ..features.joins import EngineProvider
    from ..ml.splits import first_row_at
    from ..targets.alignment import TargetInputs, build_targets

    m = tctx.cfgs.ml
    # development bars only (the reserved prices never enter memory); with the same seed
    # every development row gets the same random draws as Prompt #10's null, and the
    # random walk's scale - taken here from development bars, there from all bars - cancels
    # in every residual label (checked: shrink labels identical, residual reduction within
    # float32 rounding on 39 of 1.33M 5m rows, logs/codex/null_dev_bars_check.py)
    real = load_development_bars(m.config, tctx.timeframe, tctx.data.reserved_start)
    bars = null_bar_series(name, real, seed=tctx.ml.random_seed)
    del real
    provider = EngineProvider(bars=bars, regression_config=m.regression,
                              spectral_config=m.spectral, wavelet_config=m.wavelet)
    reg = provider.regression(m.targets.regression_window)
    tframe = build_targets(TargetInputs(bars=bars, residual=reg["residual"], sigma=reg["sigma"],
                                        ou_mu=None), m.targets)
    n = first_row_at(bars.timestamps, tctx.data.reserved_start)
    if not bars.timestamps.head(n).equals(tctx.data.timestamps):
        raise MLIntegrityError("the pipeline null is not on the development bars")
    raw = {}
    for c in tframe.columns:
        if c.startswith(("target_residual_shrinks_", "target_residual_reduction_")):
            v = np.array(tframe[c].cast(pl.Float64).fill_null(np.nan).to_numpy()[:n],
                         dtype=np.float64)
            h = int(c.rsplit("_", 1)[1])
            v[max(0, n - h):] = np.nan
            raw[c] = v
    out = {"raw": raw, "epsilon": np.asarray(reg["residual"][:n], dtype=np.float64),
           "sigma": np.asarray(reg["sigma"][:n], dtype=np.float64)}
    del tframe, reg, provider, bars
    gc.collect()
    return out


def null_ensembles(run: PairRun, null_labels: dict[str, dict[str, Any]] | None
                   ) -> list[dict[str, Any]]:
    """Step 75: ensembles of models that cannot know anything - shifted training labels
    (scored on the real labels) and the residual targets on pipeline nulls (scored on the
    null's own labels) - beside the same ensemble of the real models."""
    tctx, pair = run.tctx, run.pair
    ml = tctx.ml
    units = tctx.ml_dir / "units"
    tspec = ml.targets[pair.target]
    ref = ml.reference_linear(tspec.task)
    pos_of = {int(r): i for i, r in enumerate(pair.rows)}
    out: list[dict[str, Any]] = []

    def aligned(frame: pl.DataFrame) -> np.ndarray:
        v = np.full(pair.n, np.nan)
        if frame.is_empty():
            return v
        col = "cal_platt" if (pair.is_classification and run.cfg.calibrated_only_for_classification
                              and "cal_platt" in frame.columns) else "prediction"
        idx = np.array([pos_of.get(int(r), -1) for r in frame["row"].to_numpy()])
        ok = idx >= 0
        v[idx[ok]] = frame[col].cast(pl.Float64).to_numpy()[ok]
        return v

    def unit(fam: str, fset: str, variant: str) -> np.ndarray:
        return aligned(load_unit_predictions(units, pair.target, pair.horizon, fam, fset, variant,
                                             development_rows=tctx.data.n))

    def blocks_of(values: list[np.ndarray]) -> list[int]:
        have = np.isfinite(np.column_stack(values)).all(axis=1)
        return [k for k in range(len(pair.block_names)) if have[pair.block_positions(k)].any()]

    # shifted labels: features and labels no longer belong together; real labels scored
    for rep in (0, 1):
        variant = f"null-shift__repeat-{rep}"
        members = {f"{fam}|{ml.default_feature_set}": unit(fam, ml.default_feature_set, variant)
                   for fam in (ref, "lightgbm")}
        vals = [v for v in members.values() if np.isfinite(v).any()]
        if len(vals) < 2:
            continue
        null_avg = np.column_stack(vals).mean(axis=1)
        real_names = [m for m in members if m in pair.raw]
        real_avg = run.simple_average_all_blocks(real_names) if len(real_names) == len(members) \
            else None
        for k in blocks_of(vals):
            pos = pair.block_positions(k)
            nm = score(run.task, null_avg[pos], run.y[pos], run.base[pos], run.y_raw[pos])
            rm = (score(run.task, real_avg[pos], run.y[pos], run.base[pos], run.y_raw[pos])
                  if real_avg is not None else {})
            out.append({"control": "shifted_labels", "variant": variant,
                        "members": ",".join(members), "block": pair.block_names[k],
                        "null_metric": nm.get(run.metric), "real_metric": rm.get(run.metric),
                        "null_auc": nm.get("auc"), "null_rank_ic": nm.get("rank_ic"),
                        "rows": nm.get("n")})
    # pipeline nulls: the residual targets on a random walk / sign flip, their own labels
    if null_labels and tspec.kind == "reversion":
        sets = [s for s in ("standard", "extended") if f"lightgbm|{s}" in pair.raw]
        for null_name, lab in null_labels.items():
            variant = f"pipeline-{null_name}"
            members, real_names = {}, []
            for s in sets:
                for fam in (ref, "lightgbm"):
                    v = unit(fam, s, variant)
                    if np.isfinite(v).any():
                        members[f"{fam}|{s}"] = v
                        real_names.append(f"{fam}|{s}")
            const = unit("constant", ml.default_feature_set, variant)
            if len(members) < 2 or not np.isfinite(const).any():
                continue
            ta = derive_target(tspec, pair.horizon, lab["raw"], epsilon=lab["epsilon"],
                               sigma=lab["sigma"], spread_rel=None, log_floor=ml.log_floor)
            y_null = ta.y[pair.rows]
            y_null_raw = ta.y_raw[pair.rows]
            null_avg = np.column_stack(list(members.values())).mean(axis=1)
            real_avg = run.simple_average_all_blocks([m for m in real_names if m in pair.raw])
            for k in blocks_of(list(members.values())):
                pos = pair.block_positions(k)
                nm = score(run.task, null_avg[pos], y_null[pos], const[pos], y_null_raw[pos])
                rm = score(run.task, real_avg[pos], run.y[pos], run.base[pos], run.y_raw[pos])
                single = {}
                for mname, v in members.items():
                    single[mname] = score(run.task, v[pos], y_null[pos], const[pos],
                                          y_null_raw[pos]).get(run.metric)
                out.append({"control": f"pipeline_{null_name}", "variant": variant,
                            "members": ",".join(members), "block": pair.block_names[k],
                            "null_metric": nm.get(run.metric), "real_metric": rm.get(run.metric),
                            "null_auc": nm.get("auc"), "null_rank_ic": nm.get("rank_ic"),
                            "rows": nm.get("n"), "post_hoc": null_name != "random_walk",
                            "null_best_single": max((v for v in single.values() if v is not None),
                                                    default=None)})
    return out


# ---------------------------------------------------------------------------
# Disagreement, OOD, failures, deciles, decay, stability (Steps 8-9, 37-50)
# ---------------------------------------------------------------------------
def disagreement_views(run: PairRun, main: str, best: str | None) -> dict[str, Any]:
    pair = run.pair
    p = run.matrix()
    dis = disagreement_measures(p)
    ens = run.series[main]
    pos = run.eval_positions()
    frame = pl.DataFrame({"timestamp": pair.timestamps.gather(pos), "row": pair.rows[pos],
                          "block": [pair.block_names[b] for b in pair.block[pos]],
                          **{k: v[pos] for k, v in dis.items()},
                          "ensemble": ens[pos], "label": run.y[pos],
                          "ood_score": run.ctx["ood_score"][pos],
                          **({"ensemble_entropy": binary_entropy(ens[pos])}
                             if pair.is_classification else {})})
    by_group = []
    labels_by = {"session": dict(enumerate(SESSION_LABELS))}
    for g in ("vol_quartile", "regime_state", "spread_quartile", "session", "year"):
        if g in run.ctx:
            for r in grouped_means({k: v[pos] for k, v in dis.items()}, run.ctx[g][pos],
                                   labels=labels_by.get(g), min_rows=500):
                by_group.append({"grouping": g, **r})
    ens_loss = loss_rows(run.task, ens[pos], run.y[pos])
    losses = {"ensemble_loss": ens_loss}
    if best is not None:
        losses["best_individual_loss"] = loss_rows(run.task, run.series[best][pos], run.y[pos])
    for m in run.universe:
        losses[f"loss:{m}"] = loss_rows(run.task, pair.prediction(m)[pos], run.y[pos])
    vs_dis = loss_by_quantile(dis["prediction_std"][pos], losses, q=5)
    vs_ood = loss_by_quantile(run.ctx["ood_score"][pos], losses, q=10)
    ent_rows = (loss_by_quantile(binary_entropy(ens[pos]), {"ensemble_loss": ens_loss}, q=5)
                if pair.is_classification else [])
    # joint: OOD tercile x disagreement tercile
    joint = []
    oc = quantile_codes(run.ctx["ood_score"][pos], 3)
    dc = quantile_codes(dis["prediction_std"][pos], 3)
    for a in range(3):
        for b in range(3):
            msk = (oc == a) & (dc == b)
            if msk.sum() < 500:
                continue
            joint.append({"ood_tercile": a + 1, "disagreement_tercile": b + 1,
                          "rows": int(msk.sum()), "ensemble_loss": float(ens_loss[msk].mean())})
    corr = {}
    for name, proxy in (("disagreement", dis["prediction_std"][pos]),
                        ("ood_score", run.ctx["ood_score"][pos])):
        ok = np.isfinite(proxy) & np.isfinite(ens_loss)
        from ..alpha.information_coefficient import rank_scores

        corr[name] = (float(np.corrcoef(rank_scores(proxy[ok]), rank_scores(ens_loss[ok]))[0, 1])
                      if ok.sum() > 100 else None)
    return {"frame": frame, "by_group": by_group, "loss_vs_disagreement": vs_dis,
            "loss_vs_ood": vs_ood, "loss_vs_entropy": ent_rows, "joint": joint,
            "rank_corr_with_loss": corr}


def failure_views(run: PairRun, main: str) -> dict[str, Any]:
    pos = run.eval_positions()
    p = run.matrix()[pos]
    y = run.y[pos]
    ens_loss = loss_rows(run.task, run.series[main][pos], y)
    member_loss = loss_rows(run.task, p, y)
    dis = disagreement_measures(run.matrix())["prediction_std"][pos]
    ctx = {k: v[pos] for k, v in run.ctx.items()}
    codes = condition_codes(ctx, {"ood_score": ctx["ood_score"], "disagreement": dis})
    fq = float(run.cfg.failure.get("large_error_quantile", 0.99))
    cq = float(run.cfg.failure.get("common_mode_quantile", 0.90))
    f_rows, f_sum = failure_table(ens_loss, member_loss, run.universe, codes, quantile=fq)
    c_rows, c_sum = common_mode(member_loss, codes, ctx["month"], quantile=cq)
    return {"failure": f_rows, "failure_summary": f_sum, "common_mode": c_rows,
            "common_mode_summary": c_sum}


def decile_and_decay(run: PairRun, names: list[str]) -> tuple[list[dict[str, Any]],
                                                              list[dict[str, Any]]]:
    from ..ml.evaluation import decile_table

    tctx, pair = run.tctx, run.pair
    pos = run.eval_positions()
    dec_rows = []
    for name in names:
        t = decile_table(run.series[name][pos], run.y[pos])
        for r in t["rows"]:
            dec_rows.append({"series": name, **r, "spread": t["spread"],
                             "monotonicity": t["monotonicity"]})
    tspec = tctx.ml.targets[pair.target]
    outcomes = {}
    d = tctx.data
    for h in run.cfg.alpha_decay_horizons:
        try:
            ta = derive_target(tspec, int(h), d.targets_raw, epsilon=d.epsilon, sigma=d.sigma,
                               spread_rel=d.features.get("spread_rel"),
                               log_floor=tctx.ml.log_floor)
        except KeyError:
            continue
        outcomes[int(h)] = ta.y_raw[pair.rows][pos]
    decay_rows = []
    for name in names:
        for r in alpha_decay(run.series[name][pos], outcomes):
            decay_rows.append({"series": name, **r})
    return dec_rows, decay_rows


def stability_views(run: PairRun, names: list[str]) -> pl.DataFrame:
    pos = run.eval_positions()
    preds = {n: run.series[n][pos] for n in names}
    y, base, y_raw = run.y[pos], run.base[pos], run.y_raw[pos]
    st = run.cfg.stability
    rows = []
    groupings: list[tuple[str, np.ndarray, dict[int, str] | None, int]] = [
        ("year", run.ctx["year"][pos], None, int(st.get("min_rows_year", 5000))),
        ("quarter", run.ctx["quarter"][pos], None, int(st.get("min_rows_quarter", 3000)))]
    for g in ("regime_state", "vol_quartile", "spread_quartile", "session"):
        if g in run.ctx:
            labels = dict(enumerate(SESSION_LABELS)) if g == "session" else None
            groupings.append((g, run.ctx[g][pos], labels, int(st.get("min_rows_group", 2000))))
    block = run.pair.block[pos].astype(np.int64)
    groupings.append(("block", block, dict(enumerate(run.pair.block_names)), 0))
    # the last twelve months of development
    ts = run.pair.timestamps.gather(pos)
    last = ts.max()
    if isinstance(last, datetime):
        recent = (ts >= last - timedelta(days=365)).to_numpy()
        groupings.append(("recent_12_months", np.where(recent, 1, -1), {1: "last_12m"}, 0))
    for gname, codes, labels, min_rows in groupings:
        for r in period_table(run.task, preds, y, base, codes, labels=labels, min_rows=min_rows,
                              y_raw=y_raw):
            rows.append({"grouping": gname, **r})
    return pl.DataFrame(rows, infer_schema_length=None)


# ---------------------------------------------------------------------------
# One pair, one timeframe
# ---------------------------------------------------------------------------
def run_pair(tctx: TFContext, target: str, h: int, *,
             null_labels: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Everything of one target x horizon; tables under ``<tf>/<target>/h<h>/``."""
    from .ensemble_ablation import run_ablations

    cfg = tctx.cfg
    started = time.perf_counter()
    tspec = tctx.ml.targets[target]
    out = ensure_dir(tctx.out_dir / target / f"h{h}")
    versions = {k: tctx.data.versions.get(k) for k in ("tick_dataset_version",
                                                        "bar_dataset_version", "factory_version",
                                                        "target_version")}
    pair = load_pair(tctx.ml_dir / "predictions" / f"{target}_h{h}.parquet",
                     timeframe=tctx.timeframe, target=target, horizon=h, task=tspec.task,
                     expected_versions=versions,
                     calibrated_inputs=cfg.calibrated_only_for_classification,
                     reserved_start=tctx.data.reserved_start)
    write_table(ensemble_prediction_coverage(pair), out / "ensemble_prediction_coverage")
    elig = eligibility_table(tctx, pair)
    write_table(elig, out / "model_eligibility")
    universe, udec = build_universe(tctx, pair, elig)
    write_table(udec, out / "universe")
    summary: dict[str, Any] = {
        "timeframe": tctx.timeframe, "target": target, "horizon": h, "task": tspec.task,
        "kind": tspec.kind, "rows": pair.n, "blocks": pair.block_names,
        "evaluation_blocks": [pair.block_names[k] for k in cfg.evaluation_blocks
                              if k < len(pair.block_names)],
        "models": pair.model_names, "eligible": elig.filter(pl.col("status") == "eligible")[
            "model"].to_list(),
        "status_counts": {r["status"]: int(r["len"]) for r in
                          elig.group_by("status").len().iter_rows(named=True)},
        "universe": universe, "versions": versions, "metric": primary_metric(tspec.task)}
    tctx.log(f"{target} h{h}: {len(pair.model_names)} models, {len(summary['eligible'])} "
             f"eligible, universe {universe}")
    run = PairRun(tctx, pair, universe, elig)
    run.individuals()
    # diversity among every base model (eligible or not) and among the universe
    allm = pair.model_names
    cov_all = pair.covered(allm)
    write_table(pl.DataFrame(diversity_table(pair.matrix(allm)[cov_all], run.y[cov_all], allm,
                                             task=tspec.task), infer_schema_length=None),
                out / "diversity_all_models")
    if len(universe) < cfg.minimum_model_count:
        summary["ensemble"] = None
        summary["reason"] = (f"{len(universe)} eligible model(s) after the gate - an ensemble "
                             f"needs at least {cfg.minimum_model_count}")
        write_table(run.block_table([n for n in run.series if run.kind[n] != "baseline"
                                     or n == "constant"]), out / "block_metrics")
        write_json(out / "summary.json", summary)
        tctx.log(f"{target} h{h}: no ensemble - {summary['reason']}")
        return summary
    cov = run.cov
    div = diversity_table(pair.matrix(universe)[cov], run.y[cov], universe, task=tspec.task)
    write_table(pl.DataFrame(div, infer_schema_length=None), out / "diversity_matrix")
    write_table(pl.DataFrame([{k: v for k, v in r.items() if k in (
        "a", "b", "pearson", "spearman", "rows", "sampled")} for r in div],
        infer_schema_length=None), out / "prediction_correlation")
    write_table(pl.DataFrame([{k: v for k, v in r.items() if k in (
        "a", "b", "error_correlation", "error_covariance", "rows", "sampled")} for r in div],
        infer_schema_length=None), out / "error_correlation")
    # per-block diversity (does it change through time?)
    blk_rows = []
    for k in range(len(pair.block_names)):
        pos = pair.block_positions(k)
        pos = pos[cov[pos]]
        for r in diversity_table(pair.matrix(universe)[pos], run.y[pos], universe,
                                 task=tspec.task, max_rows=150_000):
            blk_rows.append({"block": pair.block_names[k], **r})
    write_table(pl.DataFrame(blk_rows, infer_schema_length=None), out / "diversity_by_block")
    # methods, in the Step 84 order
    run.best_individual()
    run.fixed_combinations(universe)
    if cfg.enabled("performance_weighted") or cfg.enabled("diversity_weighted"):
        run.weighted_methods()
    if cfg.enabled("stacking"):
        run.stacking_methods()
    run.walk_forward_universe()
    if cfg.enabled("regime_conditioned") or cfg.enabled("volatility_conditioned"):
        run.conditional_methods()
    if cfg.enabled("dynamic"):
        run.dynamic_method()
    run.calibration_research()
    tctx.log(f"{target} h{h}: methods done ({time.perf_counter() - started:.0f}s)")
    controls = noise_and_duplicate_controls(run) if (
        cfg.controls.get("noise_model") or cfg.controls.get("duplicate_model")) else {}
    nulls = null_ensembles(run, null_labels) if cfg.controls.get("null_ensemble") else []
    write_table(pl.DataFrame(nulls, infer_schema_length=None), out / "null_ensembles")
    write_json(out / "controls.json", controls)
    ablation_summary = run_ablations(run, out)
    tctx.log(f"{target} h{h}: ablations done ({time.perf_counter() - started:.0f}s)")
    # tables of every series
    blocks = run.block_table()
    write_table(blocks, out / "block_metrics")
    main_names = ["constant", *[f"model:{m}" for m in universe], "best_individual",
                  "simple_average", "median",
                  *[n for n in run.series if run.kind[n] in ("method", "variant")
                    and n not in ("best_individual", "simple_average", "median")]]
    main_names = list(dict.fromkeys(n for n in main_names if n in run.series))
    write_table(stability_views(run, main_names), out / "stability")
    dv = disagreement_views(run, "simple_average", "best_individual")
    write_table(dv["frame"], out / "disagreement", csv=False)       # one row per bar
    for key in ("by_group", "loss_vs_disagreement", "loss_vs_ood", "loss_vs_entropy", "joint"):
        write_table(pl.DataFrame(dv[key], infer_schema_length=None), out / f"disagreement_{key}")
    fv = failure_views(run, "simple_average")
    write_table(pl.DataFrame(fv["failure"], infer_schema_length=None), out / "failure_analysis")
    write_table(pl.DataFrame(fv["common_mode"], infer_schema_length=None), out / "common_mode")
    dec, decay = decile_and_decay(run, [n for n in main_names if n != "constant"])
    write_table(pl.DataFrame(dec, infer_schema_length=None), out / "deciles")
    write_table(pl.DataFrame(decay, infer_schema_length=None), out / "alpha_decay")
    # weights through time
    wrows: list[dict[str, Any]] = []
    wsum: dict[str, Any] = {}
    for scheme, by_block in run.block_weights.items():
        wtable, wsummary = weight_stability(
            by_block, run.weight_names[scheme],
            range_flag=float(cfg.stability.get("weight_range_flag", 0.4)))
        wrows += [{"scheme": scheme, **x} for x in wtable]
        wsum[scheme] = wsummary
    write_table(pl.DataFrame(wrows, infer_schema_length=None), out / "weights")
    if "dynamic_timeline" in run.notes:
        write_table(run.notes.pop("dynamic_timeline"), out / "dynamic_weights")
    from .ensemble_reports import pair_report

    report = pair_report(run, blocks)
    # the out-of-sample predictions of the frozen method and its references (evaluation
    # blocks; the input of the joint predictive state, Step 89)
    frozen_series = report["freeze"]["frozen_series"]
    pos = np.flatnonzero(run.eval_mask)
    dis = disagreement_measures(run.matrix())["prediction_std"]
    cols = {"timestamp": pair.timestamps.gather(pos), "row": pair.rows[pos],
            "block": [pair.block_names[b] for b in pair.block[pos]], "label": run.y[pos],
            "label_raw": run.y_raw[pos], "constant": run.base[pos],
            "best_individual": run.series["best_individual"][pos],
            "simple_average": run.series["simple_average"][pos],
            "frozen_method": run.series[frozen_series][pos], "disagreement": dis[pos],
            "ood_score": run.ctx["ood_score"][pos],
            "regime_entropy": run.ctx.get("regime_entropy", np.full(pair.n, np.nan))[pos],
            "sigma": (tctx.data.sigma[pair.rows[pos]] if tctx.data.sigma is not None
                      else np.full(pos.size, np.nan))}
    if pair.is_classification:
        cols["frozen_entropy"] = binary_entropy(run.series[frozen_series][pos])
    write_table(pl.DataFrame(cols), out / "oos_ensemble_predictions", csv=False)
    summary.update({"series": len(run.series), "notes": run.notes,
                    "weight_stability": wsum, "controls": {k: v for k, v in controls.items()
                                                          if k != "blocks"},
                    "disagreement_rank_corr_with_loss": dv["rank_corr_with_loss"],
                    "failure_summary": fv["failure_summary"],
                    "common_mode_summary": fv["common_mode_summary"],
                    "ablation": ablation_summary, **report,
                    "seconds": time.perf_counter() - started, "finished_utc": utc_now_iso()})
    write_json(out / "summary.json", summary)
    tctx.log(f"{target} h{h}: done in {time.perf_counter() - started:.0f}s")
    return summary


def score_combination(cfgs: EnsembleConfigs, timeframe: str, target: str, h: int, method: str,
                      models: list[str] | None) -> tuple[pl.DataFrame, list[str]]:
    """``xq ensemble-build``: one explicit combination on the meta walk-forward (research
    only - it freezes nothing). Default constituents: the pair's research universe."""
    ecfg = cfgs.ensemble
    ml: MLConfig = cfgs.ml.ml
    tspec = ml.targets[target]
    pair = load_pair(ecfg.predictions_path / timeframe / "predictions" / f"{target}_h{h}.parquet",
                     timeframe=timeframe, target=target, horizon=h, task=tspec.task,
                     calibrated_inputs=ecfg.calibrated_only_for_classification,
                     reserved_start=cfgs.ml.selection.periods.reserved_test.start)
    if not models:
        summ = ecfg.results_path / timeframe / target / f"h{h}" / "summary.json"
        models = (json.loads(summ.read_text(encoding="utf-8")).get("universe") if summ.exists()
                  else None) or pair.model_names
    unknown = [m for m in models if m not in pair.raw]
    if unknown:
        raise ValueError(f"unknown models {unknown}; the pair has {pair.model_names}")
    p = pair.matrix(models)
    cov = pair.covered(models)
    rows = []
    for k in ecfg.evaluation_blocks:
        if k >= len(pair.block_names):
            continue
        hist = pair.history(k, ecfg.embargo_bars)
        hist = hist[cov[hist]]
        pos = pair.block_positions(k)
        ph, yh, bh = p[hist], pair.label[hist], pair.base[hist]
        if method == "simple_average":
            pred = weighted_mean(p[pos])
        elif method == "median":
            pred = median_combination(p[pos])
        elif method == "performance_weighted":
            w, _ = performance_weights(ph, yh, bh, tspec.task, shrink=float(
                ecfg.weighting.get("candidate_shrink", 0.5)))
            pred = weighted_mean(p[pos], w)
        elif method == "diversity_weighted":
            w, _ = diversity_weights(ph, yh, bh, tspec.task, lam=float(
                ecfg.diversity.get("candidate_lambda", 0.5)), shrink=float(
                ecfg.diversity.get("shrink", 0.5)), duplicate_correlation=float(
                ecfg.diversity.get("duplicate_correlation", 0.999)))
            pred = weighted_mean(p[pos], w)
        elif method == "stacking":
            pred, _, _ = stack_block(pair, models, k, embargo=ecfg.embargo_bars, params={
                "logistic_C": ecfg.stacking.get("logistic_C", 1.0),
                "ridge_alpha": ecfg.stacking.get("ridge_alpha", 1.0),
                "max_rows": ecfg.stacking.get("max_rows")}, seed=ecfg.random_seed)
        else:
            raise ValueError(f"unknown method {method!r}")
        met = score(tspec.task, pred, pair.label[pos], pair.base[pos], pair.label_raw[pos])
        rows.append({"block": pair.block_names[k], "history_rows": int(hist.size),
                     **{key: v for key, v in met.items() if not isinstance(v, (list, dict))}})
    return pl.DataFrame(rows, infer_schema_length=None), models


def run_ensemble_research(cfgs: EnsembleConfigs, timeframe: str, *,
                          pairs: list[tuple[str, int]] | None = None,
                          progress: Callable[[str], None] | None = None,
                          pipeline_nulls: bool = True) -> dict[str, Any]:
    """Every configured pair of one timeframe (development period only)."""
    started = time.perf_counter()
    tctx = load_tf_context(cfgs, timeframe, progress=progress)
    tctx.log(f"{tctx.data.n:,} development rows (< {tctx.data.reserved_start}); OOD scores "
             f"for {int(np.isfinite(tctx.ood).sum()):,} validation rows")
    todo = pairs or list(cfgs.ensemble.pairs)
    null_labels: dict[str, dict[str, Any]] = {}
    if pipeline_nulls and any(tctx.ml.targets[t].kind == "reversion" for t, _ in todo):
        posthoc = [str(n) for n in tctx.ml.nulls.get("pipeline_posthoc") or []]
        for name in [str(tctx.ml.nulls.get("pipeline", "random_walk")), *posthoc]:
            t0 = time.perf_counter()
            null_labels[name] = _null_label_arrays(tctx, name)
            tctx.log(f"pipeline null {name}: residual labels in {time.perf_counter() - t0:.0f}s")
    results = {}
    failures = []
    for target, h in todo:
        try:
            results[f"{target}_h{h}"] = run_pair(tctx, target, h, null_labels=null_labels)
        except Exception as exc:                       # one pair must not stop the others
            LOGGER.exception("[%s] %s h%s failed", timeframe, target, h)
            failures.append({"pair": f"{target}_h{h}", "error": f"{type(exc).__name__}: {exc}"})
        gc.collect()
    from .ensemble_ablation import cross_horizon

    cross = {}
    for target, spec in cfgs.ensemble.cross_horizon.items():
        try:
            cross[target] = cross_horizon(tctx, target, spec)
        except Exception as exc:
            LOGGER.exception("[%s] cross-horizon %s failed", timeframe, target)
            failures.append({"pair": f"cross_horizon_{target}",
                             "error": f"{type(exc).__name__}: {exc}"})
    summary = {"timeframe": timeframe, "pairs": sorted(results), "failures": failures,
               "cross_horizon": cross, "development_rows": tctx.data.n,
               "reserved_start": str(tctx.data.reserved_start),
               "config_fingerprint": cfgs.ensemble.fingerprint(),
               "seconds": time.perf_counter() - started, "finished_utc": utc_now_iso()}
    write_json(tctx.out_dir / "run_summary.json", summary)
    return summary
