r"""Feature selection research, one timeframe at a time (Prompt #9).

Stages, each writing its tables when it finishes and a stamp (data versions,
selection config, code) that lets a rerun skip it:

``universe``    the quality filter (feature values only) and the development-period
                null screen: the default selection universe (Steps 2, 5-8)
``redundancy``  development correlations, clusters, representatives,
                parameter-family reduction, non-monotone duplicates (Steps 9-12)
``evidence``    effect stability, recent relevance, regime robustness, target
                specificity, alpha-decay diversity (Steps 41-44, 51)
``stability``   quarter-block resamples and year subsets of the development
                period: selection frequency, Jaccard, rank stability (Steps 16-19)
``nested``      nested chronological selection and evaluation: learning curves,
                mRMR / IC / Lasso / elastic net, PCA, family ablation, leave one
                family out, interactions, permutation importance (Steps 13-40)
``sets``        Minimal / Standard / Extended and target-specific sets from the
                development evidence, plateaus read on validation, collinearity,
                cost, warm-up, manifests (Steps 35-36, 45-59)
``audit``       live reconstruction, leakage audit, reserved-period isolation
                (Steps 53, 60-62)

The reserved test period's outcomes are never read (:mod:`..selection.data`);
the validation period is used only to evaluate what the development period
selected and to read plateaus. Nothing here is a model for trading.
"""

from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..alpha.config import AlphaConfig
from ..alpha.information_coefficient import (
    correlation_from_sums,
    ic_with_errors,
    month_moments,
    rank_scores,
)
from ..features.config import RegressionConfig
from ..features.factory import load_bar_series
from ..features.factory_config import FeatureFactoryConfig
from ..features.redundancy import (
    evenly_spaced_rows,
    feature_nmi_matrix,
    nonmonotone_pairs,
    pairwise_correlation,
)
from ..features.spectral_config import SpectralConfig
from ..features.wavelet_config import WaveletConfig
from ..models.config import OUConfig
from ..selection.ablation import (
    cumulative_groups,
    fit_and_score,
    leave_one_family_out,
    permutation_importance,
)
from ..selection.clustering import (
    EvidenceKey,
    cluster_members,
    parameter_family_reduction,
    representative_order,
)
from ..selection.config import FeatureSelectionConfig
from ..selection.data import SelectionData, load_selection_data, period_rank
from ..selection.filters import quality_filter
from ..selection.linear import Moments, lasso_path, logistic_l1_path, ridge
from ..selection.live import live_reconstruction, required_buffer
from ..selection.manifest import build_manifest, write_manifest
from ..selection.mrmr import effect_stability_score, mrmr
from ..selection.periods import chronological_folds
from ..selection.redundancy import effective_rank, set_collinearity
from ..selection.relevance import pipeline_veto, quarter_labels, span_evidence
from ..selection.selection_cv import (
    evaluate_split,
    method_orderings,
    pca_evaluation,
    prepare_split,
    split_evidence,
)
from ..selection.stability import (
    jaccard,
    mean_pairwise_jaccard,
    quarter_resamples,
    rank_stability,
    selection_frequency,
    until_first_probe,
    year_subsets,
)
from ..targets.alignment import target_kind
from ..targets.config import TargetConfig
from ..utils.clock import utc_now_iso
from ..utils.config import Config
from ..utils.logging import get_logger
from ..utils.paths import ensure_dir
from .feature_reports import pipeline_verdicts
from .feature_research import write_json, write_table
from .study_io import code_fingerprint, git_info

__all__ = ["STAGES", "SelectionConfigs", "SelectionContext", "run_selection"]

LOGGER = get_logger("research.feature_selection")
STAGES: tuple[str, ...] = ("universe", "redundancy", "evidence", "stability", "nested", "sets",
                           "audit")
_KINDS = ("direction", "reversion", "volatility", "magnitude")


@dataclass
class SelectionConfigs:
    selection: FeatureSelectionConfig
    features: FeatureFactoryConfig
    targets: TargetConfig
    alpha: AlphaConfig
    config: Config
    regression: RegressionConfig
    ou: OUConfig
    spectral: SpectralConfig
    wavelet: WaveletConfig
    research: Any


@dataclass
class SelectionContext:
    timeframe: str
    cfgs: SelectionConfigs
    out_dir: Path
    progress: Callable[[str], None] | None = None

    @property
    def cfg(self) -> FeatureSelectionConfig:
        return self.cfgs.selection

    def log(self, message: str) -> None:
        LOGGER.info("[%s] %s", self.timeframe, message)
        if self.progress is not None:
            self.progress(f"[{self.timeframe}] {message}")

    def path(self, name: str) -> Path:
        return self.out_dir / name


# ---------------------------------------------------------------------------
# Stamps
# ---------------------------------------------------------------------------
def _code() -> str:
    base = Path(__file__).resolve().parent.parent
    files = sorted((base / "selection").glob("*.py")) + [base / "research" / Path(__file__).name]
    return code_fingerprint(files)


def _stamp(ctx: SelectionContext, data: SelectionData, stage: str) -> dict[str, Any]:
    return {"stage": stage, "selection_config": ctx.cfg.fingerprint(),
            "factory_version": data.factory_manifest.get("factory_version"),
            "target_version": data.target_manifest.get("target_version"),
            "alpha_config": ctx.cfgs.alpha.fingerprint(), "code": _code()}


def _begin(ctx: SelectionContext, data: SelectionData, stage: str
           ) -> tuple[dict[str, Any], Path] | None:
    stamp = _stamp(ctx, data, stage)
    marker = ctx.path(f".{stage}.done.json")
    try:
        if json.loads(marker.read_text(encoding="utf-8")).get("stamp") == stamp:
            ctx.log(f"{stage}: current")
            return None
    except (OSError, ValueError):
        pass
    ctx.log(f"{stage}: running")
    return stamp, marker


def _mark(marker: Path, stamp: dict[str, Any], **extra: Any) -> None:
    write_json(marker, {"stamp": stamp, "generated_utc": utc_now_iso(), **extra})


def _read(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path) if path.exists() else pl.DataFrame()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Stage: universe (quality filter + development null screen)
# ---------------------------------------------------------------------------
def _veto_frame(ev: pl.DataFrame, data: SelectionData, quarters: list[str],
                ctx: SelectionContext) -> pl.DataFrame:
    """Attach the pipeline-null veto (Prompt #8's rule) on the chosen quarters."""
    veto = (pipeline_veto(data.research_dir, quarters, ctx.cfg, ctx.cfgs.alpha)
            if data.research_dir is not None else pl.DataFrame())
    if not veto.is_empty():
        wide = veto.pivot(on="null", index=["feature", "target"], values=["ic", "se"])
        rename = {c: (f"pipeline_{c[3:]}_ic" if c.startswith("ic_") else f"pipeline_{c[3:]}_se")
                  for c in wide.columns if c.startswith(("ic_", "se_"))}
        ev = ev.join(wide.rename(rename), on=["feature", "target"], how="left")
    selection_kind = ev["kind"]
    ev = pipeline_verdicts(ev.rename({"ic": "value"}).with_columns(
        pl.col("target").map_elements(target_kind, return_dtype=pl.Utf8).alias("kind")),
        ctx.cfgs.alpha).rename({"value": "ic"})
    return ev.with_columns(selection_kind.alias("kind"))


def stage_universe(ctx: SelectionContext, data: SelectionData) -> None:
    begun = _begin(ctx, data, "universe")
    if begun is None:
        return
    stamp, marker = begun
    cfg = ctx.cfg
    p8 = _read(data.research_dir / "feature_status.parquet") if data.research_dir else \
        pl.DataFrame()
    p8_status = ({r["feature"]: r["status"] for r in p8.iter_rows(named=True)}
                 if not p8.is_empty() else {})
    quality = quality_filter(data, cfg, prompt8_status=p8_status)
    ctx.log(f"quality filter: {int(quality['kept_quality'].sum())} of {quality.height} kept")
    lo, hi = data.rows["development"]
    started = time.perf_counter()
    ev = span_evidence(data.names, data.target_meta, data.features, data.targets,
                       data.timestamps, lo, hi, cfg,
                       recent_start_row=data.rows["recent_development"][0], seed=cfg.seed + 11)
    ctx.log(f"development evidence and max-T null screen in {time.perf_counter() - started:.0f}s")
    table = ev.table.join(ev.kind_null.select("feature", "kind", "beyond_max_t", "max_t_p",
                                              "best_studentized"),
                          on=["feature", "kind"], how="left")
    table = _veto_frame(table, data, quarter_labels(data.timestamps, lo, hi), ctx)
    fdr = cfg.null_screen.fdr_q
    pipeline_ok = (pl.when(pl.col("kind") == "reversion")
                   .then(pl.col("beyond_pipeline_null").fill_null(False))
                   .otherwise(pl.col("beyond_pipeline_null").fill_null(True)))
    table = table.with_columns(
        (pl.col("beyond_max_t").fill_null(False) & (pl.col("q") < fdr).fill_null(False)
         & pipeline_ok).alias("passes"),
        pl.lit("development").alias("period"))
    write_table(table, ctx.path("development_evidence"), csv=False)
    passing = (table.filter(pl.col("passes")).group_by("feature")
               .agg(pl.col("kind").unique().sort().alias("kinds_passing")))
    kinds = {r["feature"]: r["kinds_passing"] for r in passing.iter_rows(named=True)}
    quality = quality.with_columns(
        pl.col("feature").map_elements(lambda f: ",".join(kinds.get(f, [])),
                                       return_dtype=pl.Utf8).alias("kinds_passing_development"))
    quality = quality.with_columns(
        pl.when(pl.col("kept_quality") & (pl.col("kinds_passing_development") == ""))
        .then(pl.concat_str([pl.col("reasons"), pl.lit("EXCLUDE_FAILED_NULL")], separator=",")
              .str.strip_chars(","))
        .otherwise(pl.col("reasons")).alias("reasons"))
    quality = quality.with_columns(
        (pl.col("kept_quality") & (pl.col("kinds_passing_development") != ""))
        .alias("in_selection_universe"))
    write_table(quality, ctx.path("filtering"))
    probes_pass = sorted(p for p in data.probes if p in kinds)
    write_json(ctx.path("universe.json"), {
        "timeframe": ctx.timeframe,
        "registered": len(data.real_names()),
        "quality_universe": quality.filter(pl.col("kept_quality"))["feature"].to_list(),
        "selection_universe": quality.filter(pl.col("in_selection_universe"))["feature"]
        .to_list(),
        "probes": data.probes, "probes_passing_development_null": probes_pass,
        "prompt8_status_used_for_selection": False,
        "note": ("Prompt #8 statuses and the Prompt #5-#7 prior statuses saw 2022-2026 "
                 "outcomes: reported in filtering.csv, never used to select")})
    cache = ensure_dir(ctx.path("cache"))
    np.save(cache / "development_moments.npy", ev.moments)
    write_json(cache / "development_axes.json", {"features": data.names,
                                                 "targets": data.target_columns(),
                                                 "months": ev.months.labels,
                                                 "quarters": [f"{y}Q{q}" for y, q in zip(
                                                     ev.months.years, ev.months.quarters,
                                                     strict=True)]})
    _mark(marker, stamp)


# ---------------------------------------------------------------------------
# Stage: redundancy
# ---------------------------------------------------------------------------
def _evidence_keys(dev: pl.DataFrame, quality: pl.DataFrame, data: SelectionData,
                   names: list[str]) -> dict[str, EvidenceKey]:
    miss = {r["feature"]: r["missing_development"] for r in quality.iter_rows(named=True)}
    keys = {}
    for name in names:
        part = dev.filter(pl.col("feature") == name)
        passing = part.filter(pl.col("passes"))
        pool = passing if passing.height else part
        best = pool.sort(pl.col("ic").abs(), descending=True, nulls_last=True).head(1)
        row = best.row(0, named=True) if best.height else {}
        pooled, recent = row.get("ic"), row.get("recent_ic")
        rel = (abs(recent) if pooled is not None and recent is not None
               and np.sign(pooled) == np.sign(recent) else 0.0)
        spec = data.registry.get(name, {})
        keys[name] = EvidenceKey(name=name, passes_null=bool(passing.height),
                                 sign_consistency=row.get("sign_consistency"),
                                 median_yearly_ic=row.get("median_yearly_ic"),
                                 recent_relevance=rel,
                                 missing_development=float(miss.get(name) or 0.0),
                                 cost=spec.get("cost"),
                                 min_history=int(spec.get("min_history") or 0))
    return keys


def development_sample(data: SelectionData, names: list[str], size: int) -> np.ndarray:
    lo, hi = data.rows["development"]
    rows = lo + evenly_spaced_rows(hi - lo, size)
    idx = [data.names.index(n) for n in names]
    return data.features[rows][:, idx].astype(np.float64)


def stage_redundancy(ctx: SelectionContext, data: SelectionData) -> None:
    begun = _begin(ctx, data, "redundancy")
    if begun is None:
        return
    stamp, marker = begun
    cfg = ctx.cfg
    uni = _json(ctx.path("universe.json"))
    names = uni["quality_universe"]
    sample = development_sample(data, names, cfg.redundancy.sample_rows)
    pearson, _ = pairwise_correlation(sample)
    spearman, counts = pairwise_correlation(sample, rank=True)
    abs_s = np.abs(np.nan_to_num(spearman, nan=0.0))
    clusters, link = cluster_members(abs_s, names, threshold=cfg.redundancy.cluster_abs_corr)
    dev = _read(ctx.path("development_evidence.parquet"))
    quality = _read(ctx.path("filtering.parquet"))
    keys = _evidence_keys(dev, quality, data, names)
    out_clusters: list[dict[str, Any]] = []
    for cid, members in sorted(clusters.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        ordered = representative_order([keys[m] for m in members],
                                       cfg.representative.consistency_round)
        idx = [names.index(m) for m in members]
        sub = abs_s[np.ix_(idx, idx)]
        out_clusters.append({
            "cluster": int(cid), "size": len(members), "members": members,
            "representative": ordered[0].name,
            "order": [k.name for k in ordered],
            "mean_abs_spearman": float(sub[np.triu_indices(len(idx), 1)].mean())
            if len(idx) > 1 else 1.0,
            "why": ("passes the development null screen" if ordered[0].passes_null
                    else "no member passes the null screen: kept for reference only")})
    families = parameter_family_reduction(names, data.registry, abs_s, keys,
                                          threshold=cfg.redundancy.cluster_abs_corr,
                                          round_to=cfg.representative.consistency_round)
    nmi = feature_nmi_matrix(sample[evenly_spaced_rows(sample.shape[0], 100_000)],
                             bins=cfg.mrmr.mi_bins)
    hidden = nonmonotone_pairs(nmi, names, abs_s, nmi_threshold=cfg.redundancy.nonmonotone_nmi,
                               corr_ceiling=0.7)
    iu, ju = np.triu_indices(len(names), 1)
    hard = [{"a": names[i], "b": names[j], "abs_spearman": float(abs_s[i, j])}
            for i, j in zip(iu, ju, strict=True)
            if abs_s[i, j] >= cfg.redundancy.max_pairwise_correlation]
    sel = uni["selection_universe"]
    sel_idx = [names.index(n) for n in sel if n in names]
    write_json(ctx.path("redundancy_clusters.json"), {
        "threshold_abs_spearman": cfg.redundancy.cluster_abs_corr, "linkage": "average",
        "sample_rows": int(sample.shape[0]), "estimate": True, "period": "development",
        "clusters": out_clusters,
        "largest_clusters": [c for c in out_clusters if c["size"] > 1][:10],
        "hard_pairs": sorted(hard, key=lambda r: -r["abs_spearman"]),
        "nonmonotone_pairs": hidden,
        "effective_rank_quality_universe": effective_rank(spearman),
        "effective_rank_selection_universe": effective_rank(spearman[np.ix_(sel_idx, sel_idx)])
        if sel_idx else None,
        "features_quality_universe": len(names), "features_selection_universe": len(sel_idx)})
    write_table(pl.DataFrame([{**f, "members": json.dumps(f["members"]),
                               "adjacent_abs_spearman": json.dumps(f["adjacent_abs_spearman"]),
                               "kept": json.dumps(f["kept"]), "dropped": json.dumps(f["dropped"])}
                              for f in families], infer_schema_length=None),
                ctx.path("parameter_families"))
    cache = ensure_dir(ctx.path("cache"))
    np.save(cache / "spearman.npy", spearman)
    np.save(cache / "pearson.npy", pearson)
    np.save(cache / "linkage.npy", link)
    write_json(cache / "names.json", names)
    del counts
    _mark(marker, stamp)


def _spearman(ctx: SelectionContext) -> tuple[np.ndarray, list[str]]:
    return (np.load(ctx.path("cache/spearman.npy")), _json(ctx.path("cache/names.json")))


# ---------------------------------------------------------------------------
# Stage: evidence views
# ---------------------------------------------------------------------------
def stage_evidence(ctx: SelectionContext, data: SelectionData) -> None:
    begun = _begin(ctx, data, "evidence")
    if begun is None:
        return
    stamp, marker = begun
    cfg = ctx.cfg
    dev = _read(ctx.path("development_evidence.parquet")).filter(
        ~pl.col("feature").str.starts_with("probe_"))
    score = effect_stability_score(dev["median_yearly_ic"].fill_null(np.nan).to_numpy(),
                                   dev["sign_consistency"].fill_null(np.nan).to_numpy())
    views = dev.with_columns(pl.Series("effect_stability_score", score))
    write_table(views.select("feature", "target", "kind", "horizon", "ic", "q", "passes",
                             "median_yearly_ic", "sign_consistency", "worst_year_ic",
                             "recent_ic", "effect_stability_score").sort(
        ["target", "effect_stability_score"], descending=[False, True]),
        ctx.path("effect_stability"))
    fdr = cfg.null_screen.fdr_q
    recent = views.with_columns(
        (pl.col("recent_ic") / pl.col("ic")).alias("recent_over_pooled"),
        pl.when(pl.col("recent_ic").is_null()).then(pl.lit("undetermined"))
        .when((pl.col("q") < fdr) & (pl.col("recent_ic").sign() != pl.col("ic").sign()))
        .then(pl.lit("reversed_recently"))
        .when((pl.col("q") < fdr) & (pl.col("recent_ic").abs() < 0.25 * pl.col("ic").abs()))
        .then(pl.lit("vanished_recently"))
        .when(pl.col("q") < fdr).then(pl.lit("still_relevant"))
        .otherwise(pl.lit("not_significant")).alias("recent_relevance"))
    write_table(recent.select("feature", "target", "kind", "horizon", "ic", "recent_ic",
                              "recent_over_pooled", "q", "recent_relevance", "passes"),
                ctx.path("recent_relevance"))
    write_table(_regime_robustness(ctx, data, dev), ctx.path("regime_robustness"))
    spec = _target_specificity(dev)
    write_table(spec, ctx.path("target_specificity"))
    _mark(marker, stamp)


def _regime_robustness(ctx: SelectionContext, data: SelectionData,
                       dev: pl.DataFrame) -> pl.DataFrame:
    """IC inside Prompt #7 filtered states and causal volatility terciles (development rows)."""
    from ..alpha.conditioning import causal_quantile_buckets, month_codes
    from .feature_research import _partition

    lo, hi = data.rows["development"]
    stamps = data.timestamps.slice(lo, hi - lo)
    months = data.months
    k = ctx.cfgs.features.families.regime.states
    probs = [f"regime_p{j}" for j in range(k)]
    conds: dict[str, tuple[np.ndarray, list[str]]] = {}
    if all(p in data.names for p in probs):
        stack = np.column_stack([data.column(p)[lo:hi] for p in probs]).astype(np.float64)
        ok = np.isfinite(stack).all(axis=1)
        state = np.full(hi - lo, -1, dtype=np.int64)
        state[ok] = np.argmax(stack[ok], axis=1)
        conds["regime_state"] = (state, [f"state_{j}" for j in range(k)])
    if "log_rv_20" in data.names:
        bars_day = ctx.cfgs.features.bars_per_day(float(data.registry["log_rv_20"].get(
            "extra", {}).get("bar_seconds", 0)) or _bar_seconds(ctx.timeframe))
        codes = causal_quantile_buckets(data.column("log_rv_20")[lo:hi].astype(np.float64),
                                        stamps, 3, min_history=250 * bars_day)
        conds["volatility_tercile"] = (codes, ["low", "mid", "high"])
    passing = dev.filter(pl.col("passes"))
    if passing.is_empty() or not conds:
        return pl.DataFrame()
    pairs = list(passing.select("feature", "target", "kind", "ic").iter_rows(named=True))
    feats = sorted({p["feature"] for p in pairs})
    targets = data.target_columns()
    mstart = int(np.searchsorted(months.starts, lo, side="left"))
    mend = int(np.searchsorted(months.starts, hi, side="left"))
    local_starts = months.starts[mstart:mend] - lo
    mcode = month_codes(local_starts, hi - lo)
    ys = [period_rank(data.targets[:, j], lo, hi)[lo:hi] for j in range(len(targets))]
    xs = {f: period_rank(data.column(f), lo, hi)[lo:hi] for f in feats}
    rows = []
    for cname, (codes, labels) in conds.items():
        plan = _partition(codes, labels, mcode, local_starts.size)
        mom = month_moments([xs[f][plan.order] for f in feats], [y[plan.order] for y in ys],
                            plan.starts)
        mom = mom.reshape(len(feats), len(targets), len(labels), local_starts.size, 6)
        for pr in pairs:
            i, j = feats.index(pr["feature"]), targets.index(pr["target"])
            ics = []
            for c, label in enumerate(labels):
                stats = ic_with_errors(mom[i, j, c][:, None, :], hac_lags=2)
                n = float(stats["n"][0])
                ic = float(stats["ic"][0]) if n >= 2000 and np.isfinite(stats["ic"][0]) else None
                ics.append(ic)
                rows.append({"feature": pr["feature"], "target": pr["target"],
                             "kind": pr["kind"], "conditioning": cname, "condition": label,
                             "ic": ic, "n": n, "pooled_ic": pr["ic"]})
    frame = pl.DataFrame(rows, infer_schema_length=None)
    summary = frame.group_by(["feature", "target", "kind", "conditioning"]).agg(
        pl.col("pooled_ic").first(),
        pl.col("ic").min().alias("min_condition_ic"), pl.col("ic").max().alias("max_condition_ic"),
        ((pl.col("ic") * pl.col("pooled_ic").sign()) >= 0.5 * pl.col("pooled_ic").abs())
        .all().alias("holds_in_every_condition"),
        pl.col("ic").is_null().sum().alias("conditions_undetermined"))
    return summary.with_columns(
        pl.when(pl.col("holds_in_every_condition")).then(pl.lit("broad_regime"))
        .otherwise(pl.lit("regime_specific")).alias("regime_class")).sort(
        ["feature", "target", "conditioning"])


def _bar_seconds(timeframe: str) -> float:
    from ..data.resampler import parse_timeframe

    return parse_timeframe(timeframe).total_seconds()


_CLASS = {"direction": "directional", "reversion": "mean_reversion", "volatility": "volatility",
          "magnitude": "magnitude"}


def _target_specificity(dev: pl.DataFrame) -> pl.DataFrame:
    rows = []
    for (feature,), part in dev.group_by(["feature"]):
        passing = part.filter(pl.col("passes"))
        kinds = sorted(passing["kind"].unique().to_list())
        peak = {}
        for kind in kinds:
            kp = passing.filter(pl.col("kind") == kind).sort(pl.col("ic").abs(),
                                                              descending=True)
            peak[kind] = int(kp["horizon"][0])
        cls = ("none" if not kinds else "general" if len(kinds) >= 2 else _CLASS[kinds[0]])
        rows.append({"feature": feature, "kinds_passing": ",".join(kinds), "class": cls,
                     "peak_horizon_by_kind": json.dumps(peak),
                     "best_abs_ic": _max_abs(part["ic"])})
    return pl.DataFrame(rows, infer_schema_length=None).sort("feature")


def _max_abs(values: pl.Series) -> float:
    """Largest |value| of a column, 0 when it holds no value."""
    finite = values.drop_nulls().cast(pl.Float64).to_numpy()
    finite = finite[np.isfinite(finite)]
    return float(np.abs(finite).max()) if finite.size else 0.0


# ---------------------------------------------------------------------------
# Stage: stability selection (development quarters)
# ---------------------------------------------------------------------------
def _quarter_sums(mom: np.ndarray, quarters: list[str]) -> tuple[np.ndarray, list[str]]:
    """Month moments (F, T, M, 6) -> quarter moments (F, T, Q, 6) and the quarter labels."""
    labels = sorted(set(quarters), key=quarters.index)
    starts = np.array([quarters.index(q) for q in labels], dtype=np.int64)
    return np.add.reduceat(mom, starts, axis=2), labels


def _design_quarter_moments(data: SelectionData, columns: list[int], lo: int, hi: int
                            ) -> tuple[list[Moments], list[str]]:
    """Per development quarter: moments of the in-development CDF design and targets."""
    stamps = data.timestamps.slice(lo, hi - lo)
    q_codes = (stamps.dt.year() * 10 + stamps.dt.quarter()).to_numpy()
    bounds = np.concatenate(([0], np.flatnonzero(np.diff(q_codes) != 0) + 1, [hi - lo]))
    x = np.empty((hi - lo, len(columns)), dtype=np.float32)
    for i, c in enumerate(columns):
        u = rank_scores(data.features[lo:hi, c])
        x[:, i] = np.where(np.isfinite(u), u - 0.5, 0.0)
    y = np.empty((hi - lo, data.targets.shape[1]), dtype=np.float32)
    for j in range(data.targets.shape[1]):
        u = rank_scores(data.targets[lo:hi, j])
        y[:, j] = u - 0.5
    out, labels = [], []
    for b0, b1 in zip(bounds[:-1], bounds[1:], strict=True):
        ok = np.isfinite(y[b0:b1]).all(axis=1)
        xs = x[b0:b1][ok].astype(np.float64)
        ys = y[b0:b1][ok].astype(np.float64)
        out.append(Moments(float(ok.sum()), xs.sum(axis=0), xs.T @ xs, ys.sum(axis=0),
                           (ys * ys).sum(axis=0), xs.T @ ys))
        code = int(q_codes[b0])
        labels.append(f"{code // 10}Q{code % 10}")
    return out, labels


def stage_stability(ctx: SelectionContext, data: SelectionData) -> None:
    begun = _begin(ctx, data, "stability")
    if begun is None:
        return
    stamp, marker = begun
    cfg = ctx.cfg
    st = cfg.stability
    uni = _json(ctx.path("universe.json"))
    cand_names = list(uni["selection_universe"]) + list(data.probes)
    dev = _read(ctx.path("development_evidence.parquet"))
    passing = {t: set(dev.filter((pl.col("target") == t) & pl.col("passes"))["feature"].to_list())
               for t in data.target_columns()}
    axes = _json(ctx.path("cache/development_axes.json"))
    mom = np.load(ctx.path("cache/development_moments.npy"))       # (F, T, M, 6)
    qmom_all, qlabels = _quarter_sums(mom, axes["quarters"])
    del mom
    qmom = qmom_all[[axes["features"].index(n) for n in cand_names]]   # (F', T, Q, 6)
    del qmom_all
    spear, snames = _spearman(ctx)
    corr = np.zeros((len(cand_names), len(cand_names)))
    for a, na in enumerate(cand_names):
        for b, nb in enumerate(cand_names):
            if na in snames and nb in snames:
                corr[a, b] = abs(spear[snames.index(na), snames.index(nb)])
            elif a == b:
                corr[a, b] = 1.0
    lo, hi = data.rows["development"]
    design, dlabels = _design_quarter_moments(data, [data.names.index(n) for n in cand_names],
                                              lo, hi)
    if dlabels != qlabels:
        raise RuntimeError("quarter labels of the design and the moments differ")
    resamples = quarter_resamples(qlabels, count=st.resamples, fraction=st.fraction,
                                  seed=cfg.seed + 23)
    subsets = year_subsets(qlabels, st.year_subsets)
    runs = [("resample", f"r{i:02d}", q) for i, q in enumerate(resamples)] + \
           [("year_subset", k, v) for k, v in subsets.items()] + [("full", "development", qlabels)]
    targets = data.target_columns()
    freq_rows, stab_rows, subset_rows = [], [], []
    k = st.selector_k
    probe_set = frozenset(data.probes)
    labels_r = [f"r{i:02d}" for i in range(len(resamples))]
    started = time.perf_counter()
    for j, (kind, h, col) in enumerate(data.target_meta):
        orders: dict[str, dict[str, list[str]]] = {"mrmr": {}, "lasso": {}}
        rel_runs: list[np.ndarray] = []
        # target-specific pool: features passing this target's development screen (veto
        # included) - a mechanical residual feature that passes for volatility never
        # competes for a reversion target - plus every probe
        pool = [i for i, n in enumerate(cand_names) if n in passing[col] or n in data.probes]
        for run_kind, label, quarters in runs:
            idx = [qlabels.index(q) for q in quarters]
            sums = qmom[:, j][:, idx].sum(axis=1)                      # (F', 6)
            rel = np.abs(np.nan_to_num(correlation_from_sums(sums), nan=0.0))
            if run_kind == "resample":
                rel_runs.append(np.where(np.isin(np.arange(rel.size), pool), rel, 0.0))
            # k + 1 steps: enough to see whether a probe enters within the first k
            steps = mrmr(rel, corr, pool, k=k + 1, lam=1.0,
                         hard_limit=cfg.redundancy.max_pairwise_correlation)
            orders["mrmr"][label] = [cand_names[s.index] for s in steps]
            total = design[idx[0]]
            for q in idx[1:]:
                total = total + design[q]
            qg, c, _, _, _ = total.standardized(pool)
            path = lasso_path(qg, c[:, j], max_features=k + 1) if pool else {"order": []}
            orders["lasso"][label] = [cand_names[pool[i]] for i in path["order"]]
        for selector, ords in orders.items():
            stopped = {lab: until_first_probe(o, probe_set, k) for lab, o in ords.items()}
            sel = {lab: s for lab, (s, _) in stopped.items()}
            resample_sets = [sel[lab] for lab in labels_r]
            freq = selection_frequency(resample_sets, cand_names)
            freq_top = selection_frequency([ords[lab][:k] for lab in labels_r], cand_names)
            positions = [stopped[lab][1] for lab in labels_r]
            seen = [p for p in positions if p is not None]
            probe_top = max((freq_top[p] for p in data.probes), default=0.0)
            sub = {name: sel[name] for name in subsets}
            for name in cand_names:
                freq_rows.append({"feature": name, "target": col, "kind": kind, "horizon": h,
                                  "selector": selector, "frequency": freq[name],
                                  "frequency_top_k": freq_top[name],
                                  "year_subsets_selected": sum(name in v for v in sub.values()),
                                  "in_full_development": name in sel["development"],
                                  "probe_max_frequency_top_k": probe_top,
                                  "is_probe": name in data.probes})
            stab_rows.append({
                "target": col, "kind": kind, "horizon": h, "selector": selector,
                "resamples": len(resamples), "k": k,
                "real_candidates": sum(cand_names[i] not in probe_set for i in pool),
                "mean_selected": float(np.mean([len(s) for s in resample_sets])),
                "median_first_probe_position": float(np.median(seen)) if seen else None,
                "probe_first_share": float(np.mean([p == 1 for p in positions])),
                "no_probe_within_k_share": float(np.mean([p is None or p > k
                                                          for p in positions])),
                "full_development_selected": len(sel["development"]),
                "mean_pairwise_jaccard": mean_pairwise_jaccard(resample_sets),
                "rank_stability": (rank_stability(rel_runs) if selector == "mrmr" else None),
                "probe_max_frequency_top_k": probe_top})
            for name, chosen in sub.items():
                subset_rows.append({"target": col, "selector": selector, "subset": name,
                                    "jaccard_with_full_development": jaccard(
                                        chosen, sel["development"]),
                                    "selected": json.dumps(chosen)})
        ctx.log(f"stability: {col} ({time.perf_counter() - started:.0f}s)")
    write_table(pl.DataFrame(freq_rows, infer_schema_length=None),
                ctx.path("selection_frequency"))
    write_table(pl.DataFrame(stab_rows, infer_schema_length=None),
                ctx.path("selection_stability"))
    write_table(pl.DataFrame(subset_rows, infer_schema_length=None),
                ctx.path("year_subset_stability"))
    del design, qmom, targets
    gc.collect()
    _mark(marker, stamp, resamples=len(resamples), year_subsets=list(subsets),
          block="calendar quarter", selectors=["mrmr (|rank IC|, difference, lambda 1)",
                                               "lasso (entry order on the path)"],
          rule=f"a run selects the features entering before the first probe, at most {k}",
          note=("candidate pool = the features passing the target's development screen plus "
                "the noise probes; relevance, the mRMR ranking and the Lasso path are "
                "recomputed on each resample; ranks and CDFs are taken over the development "
                "period"))


# ---------------------------------------------------------------------------
# Stage: nested selection and evaluation
# ---------------------------------------------------------------------------
def stage_nested(ctx: SelectionContext, data: SelectionData) -> None:
    begun = _begin(ctx, data, "nested")
    if begun is None:
        return
    stamp, marker = begun
    cfg = ctx.cfg
    uni = _json(ctx.path("universe.json"))
    quality = uni["quality_universe"]
    columns = [data.names.index(n) for n in quality] + [data.names.index(p) for p in data.probes]
    splits = chronological_folds(data.timestamps, cfg)
    curve, orders_rows, path_rows, coef_rows = [], [], [], []
    pca_rows, pca_var, ablation_rows, lofo_rows, ix_rows, perm_rows, logit_rows = \
        [], [], [], [], [], [], []
    family_of = [str(data.registry.get(data.names[c], {}).get("family", "probe"))
                 for c in columns]
    universe_local = list(range(len(quality)))            # probes excluded from PCA / ablation
    spear, snames = _spearman(ctx)
    for split in splits:
        t0 = time.perf_counter()
        work = prepare_split(data, split, columns, cfg)
        ev = split_evidence(work, data, cfg, ctx.cfgs.alpha)
        write_table(ev, ctx.path(f"splits/evidence_{split.name}"), csv=False)
        ctx.log(f"nested {split.name}: prepared + evidence in {time.perf_counter() - t0:.0f}s")
        names = [data.names[c] for c in columns]
        for (_kind, _h, target) in data.target_meta:
            j = data.target_columns().index(target)
            orders = method_orderings(work, ev, data, target, cfg)
            curve += evaluate_split(work, orders, target, data, cfg)
            for method, order in orders.items():
                for rank, i in enumerate(order):
                    orders_rows.append({"split": split.name, "target": target, "method": method,
                                        "rank": rank + 1, "feature": names[i],
                                        "is_probe": names[i] in data.probes})
            rows, explained = pca_evaluation(work, universe_local, target, data, cfg)
            pca_rows += rows
            if target == data.target_columns()[0]:
                pca_var.append({"split": split.name,
                                **{f"explained_{k}": float(explained[:k].sum())
                                   for k in (5, 10, 20, 30, 50) if k <= explained.size},
                                "components": int(explained.size)})
            # coefficient / sign stability of the standard-size mRMR set and of the Lasso
            top = orders.get("mrmr_ic_difference_1", [])[:20]
            if top:
                _, beta = fit_and_score(work, top, j, cfg.linear.ridge_alpha)
                for i, b in zip(top, beta if beta is not None else [], strict=False):
                    coef_rows.append({"split": split.name, "target": target, "model": "ridge",
                                      "feature": names[i], "coefficient": float(b)})
            for l2 in cfg.linear.elastic_net_l2:
                method = "lasso" if l2 == 0 else f"elastic_net_{l2:g}"
                order = orders.get(method, [])
                if not order:
                    continue
                q, c, _, _, _ = work.moments.standardized(order)
                beta = ridge(q, c[:, [j]], cfg.linear.ridge_alpha)[:, 0]
                for rank, (i, b) in enumerate(zip(order, beta, strict=True)):
                    path_rows.append({"split": split.name, "target": target, "method": method,
                                      "entry_rank": rank + 1, "feature": names[i],
                                      "refit_coefficient": float(b)})
            # family ablation and leave one family out (quality universe, no probes)
            for label, fams in cumulative_groups():
                cols = [i for i in universe_local if family_of[i] in fams]
                m, _ = fit_and_score(work, cols, j, cfg.linear.ridge_alpha)
                ablation_rows.append({"split": split.name, "target": target, "step": label,
                                      "features": len(cols), **m})
            for r in leave_one_family_out(work, universe_local, family_of, j,
                                          cfg.linear.ridge_alpha):
                lofo_rows.append({"split": split.name, "target": target, **r})
            ix_rows += _interaction_review(work, names, data, j, target, split.name, cfg)
            if split.name == "development_to_validation" and top:
                groups = _clusters_within(top, names, spear, snames)
                block = max(1, int(round(cfg.evaluation.permutation_block_bars
                                         * 300.0 / _bar_seconds(ctx.timeframe))))
                for r in permutation_importance(work, top, j, cfg.linear.ridge_alpha,
                                                block=block,
                                                repeats=cfg.evaluation.permutation_repeats,
                                                seed=cfg.seed + j, groups=groups):
                    unit = r["unit"]
                    if unit.startswith("feature:"):
                        r["unit"] = "feature:" + names[int(unit.split(":", 1)[1])]
                    perm_rows.append({"target": target, **r})
        if split.name == "development_to_validation":
            logit_rows += _logistic_view(work, ev, data, cfg)
        del work
        gc.collect()
        ctx.log(f"nested {split.name}: done in {time.perf_counter() - t0:.0f}s")
    out = {"feature_count_curve": curve, "method_orderings": orders_rows,
           "lasso_elastic_net_selection": path_rows, "coefficient_stability_raw": coef_rows,
           "pca_predictive": pca_rows, "pca_analysis": pca_var, "family_ablation": ablation_rows,
           "leave_one_family_out": lofo_rows, "interactions_review": ix_rows,
           "permutation_importance": perm_rows, "logistic_l1": logit_rows}
    for name, rows in out.items():
        write_table(pl.DataFrame(rows, infer_schema_length=None), ctx.path(name))
    _coefficient_stability(ctx)
    _mark(marker, stamp, splits=[s.name for s in splits])


def _clusters_within(top: list[int], names: list[str], spear: np.ndarray, snames: list[str]
                     ) -> dict[str, list[int]]:
    """Clusters (|rho| >= 0.7) among the selected features: permuted together as one unit."""
    feats = [names[i] for i in top if names[i] in snames]
    if len(feats) < 2:
        return {}
    idx = [snames.index(f) for f in feats]
    clusters, _ = cluster_members(np.abs(np.nan_to_num(spear[np.ix_(idx, idx)], nan=0.0)),
                                  feats, threshold=0.7)
    return {str(c): [top[[names[i] for i in top].index(m)] for m in members]
            for c, members in clusters.items() if len(members) > 1}


def _interaction_review(work: Any, names: list[str], data: SelectionData, j: int, target: str,
                        split: str, cfg: FeatureSelectionConfig) -> list[dict[str, Any]]:
    rows = []
    fcfg_ix = [(n, data.registry[n]) for n in names
               if (data.registry.get(n) or {}).get("family") == "interaction"]
    for name, spec in fcfg_ix:
        parts = name.removeprefix("ix_").split("__x__")
        if len(parts) != 2 or not all(p in names for p in parts):
            rows.append({"split": split, "target": target, "interaction": name,
                         "hierarchy_ok": False, "delta_rank_ic": None, "delta_r2": None})
            continue
        comps = [names.index(p) for p in parts]
        base, _ = fit_and_score(work, comps, j, cfg.linear.ridge_alpha)
        full, _ = fit_and_score(work, [*comps, names.index(name)], j, cfg.linear.ridge_alpha)
        rows.append({"split": split, "target": target, "interaction": name, "hierarchy_ok": True,
                     "components_rank_ic": base.get("rank_ic"), "with_rank_ic": full.get("rank_ic"),
                     "delta_rank_ic": _delta(full.get("rank_ic"), base.get("rank_ic")),
                     "delta_r2": _delta(full.get("r2"), base.get("r2")),
                     "declared_cost": spec.get("cost")})
    return rows


def _delta(a: Any, b: Any) -> float | None:
    return None if a is None or b is None else float(a) - float(b)


def _logistic_view(work: Any, ev: pl.DataFrame, data: SelectionData,
                   cfg: FeatureSelectionConfig) -> list[dict[str, Any]]:
    """L1 logistic paths for the binary direction targets (sign of R_(t,h)), training rows."""
    rows: list[dict[str, Any]] = []
    names = [data.names[c] for c in work.columns]
    for (kind, _h, col) in data.target_meta:
        if kind != "direction":
            continue
        j = data.target_columns().index(col)
        cand = ev.filter((pl.col("target") == col) & pl.col("passes"))["feature"].to_list()
        idx = [names.index(f) for f in cand if f in names]
        if len(idx) < 2:
            rows.append({"target": col, "binary": "sign", "status": "fewer than 2 candidates"})
            continue
        lo, hi = work.train
        y = data.targets[lo:hi, j]                      # the raw return: its sign is the label
        ok = np.isfinite(y) & (y != 0)
        take = np.flatnonzero(ok)
        take = take[evenly_spaced_rows(take.size, cfg.linear.logistic_max_rows)]
        x = work.x_train[take][:, idx].astype(np.float64)
        x = (x - x.mean(axis=0)) / np.where(x.std(axis=0) > 0, x.std(axis=0), 1.0)
        yb = (y[take] > 0).astype(np.float64)
        path = logistic_l1_path(x, yb, cfg.linear.logistic_lambdas)
        for i, f in enumerate([names[t] for t in idx]):
            rows.append({"target": col, "binary": "sign", "feature": f,
                         "entry_lambda_index": None if not np.isfinite(path["entry"][i])
                         else int(path["entry"][i]),
                         "entry_lambda": None if not np.isfinite(path["entry"][i])
                         else float(path["lambdas"][int(path["entry"][i])]),
                         "final_coefficient": float(path["coefs"][-1, i])})
    return rows


def _coefficient_stability(ctx: SelectionContext) -> None:
    raw = _read(ctx.path("coefficient_stability_raw.parquet"))
    lasso = _read(ctx.path("lasso_elastic_net_selection.parquet"))
    frames = []
    if not raw.is_empty():
        frames.append(raw.group_by(["target", "feature", "model"]).agg(
            pl.len().alias("splits"), (pl.col("coefficient") > 0).sum().alias("positive"),
            (pl.col("coefficient") < 0).sum().alias("negative"),
            pl.col("coefficient").mean().alias("mean_coefficient")))
    if not lasso.is_empty():
        frames.append(lasso.rename({"method": "model", "refit_coefficient": "coefficient"})
                      .group_by(["target", "feature", "model"]).agg(
            pl.len().alias("splits"), (pl.col("coefficient") > 0).sum().alias("positive"),
            (pl.col("coefficient") < 0).sum().alias("negative"),
            pl.col("coefficient").mean().alias("mean_coefficient")))
    if not frames:
        return
    table = pl.concat(frames, how="diagonal_relaxed").with_columns(
        (pl.max_horizontal("positive", "negative") / pl.col("splits")).alias("sign_agreement"),
        ((pl.col("positive") > 0) & (pl.col("negative") > 0)).alias("sign_flips"))
    write_table(table.sort(["target", "model", "feature"]), ctx.path("coefficient_stability"))


# ---------------------------------------------------------------------------
# Stage: candidate sets and manifests
# ---------------------------------------------------------------------------
def _stability_scores(ctx: SelectionContext) -> pl.DataFrame:
    """Per feature x target: mean probe-stopped selection frequency over the two selectors."""
    freq = _read(ctx.path("selection_frequency.parquet"))
    return freq.group_by(["feature", "target", "kind", "horizon", "is_probe"]).agg(
        pl.col("frequency").mean().alias("score"),
        pl.col("year_subsets_selected").mean().alias("year_subsets"))


def _general_ranking(scores: pl.DataFrame, threshold: float, effect: pl.DataFrame
                     ) -> tuple[list[str], dict[str, Any]]:
    """Selection-stable features first (in more target kinds, more often), then the rest.

    Features no resample selected tie on the stability scores; they are ordered by
    their best development effect stability (|median yearly IC| x max(0, 2c - 1)
    over the targets they pass), never by name alone.
    """
    real = scores.filter(~pl.col("is_probe"))
    stable = real.with_columns((pl.col("score") >= threshold).alias("stable"))
    per = stable.group_by("feature").agg(
        pl.col("kind").filter(pl.col("stable")).unique().len().alias("kinds_stable"),
        pl.col("score").mean().alias("mean_score"), pl.col("score").max().alias("max_score"),
        pl.col("stable").any().alias("any_stable"))
    best = (effect.filter(pl.col("passes")).group_by("feature")
            .agg(pl.col("effect_stability_score").max().alias("effect_stability"))
            if not effect.is_empty() else pl.DataFrame(schema={"feature": pl.Utf8,
                                                               "effect_stability": pl.Float64}))
    per = per.join(best, on="feature", how="left").with_columns(
        pl.col("effect_stability").fill_null(0.0))
    per = per.sort(["kinds_stable", "mean_score", "max_score", "effect_stability", "feature"],
                   descending=[True, True, True, True, False])
    info = {r["feature"]: r for r in per.iter_rows(named=True)}
    return per["feature"].to_list(), info


def _constrained(order: list[str], k: int, corr: np.ndarray, names: list[str],
                 limit: float) -> list[str]:
    out: list[str] = []
    for f in order:
        if f not in names:
            continue
        i = names.index(f)
        if any(abs(corr[i, names.index(g)]) >= limit for g in out):
            continue
        out.append(f)
        if len(out) == k:
            break
    return out


def stage_sets(ctx: SelectionContext, data: SelectionData) -> None:
    begun = _begin(ctx, data, "sets")
    if begun is None:
        return
    stamp, marker = begun
    cfg = ctx.cfg
    ev_cfg = cfg.evaluation
    scores = _stability_scores(ctx)
    thr = cfg.stability.frequency_threshold
    order, info = _general_ranking(scores, thr, _read(ctx.path("effect_stability.parquet")))
    spear, snames = _spearman(ctx)
    limit = cfg.redundancy.max_pairwise_correlation
    uni = _json(ctx.path("universe.json"))
    order = [f for f in order if f in uni["selection_universe"]]
    sizes = [k for k in ev_cfg.candidate_set_sizes if k <= len(order)] or [len(order)]
    sets = {k: _constrained(order, k, spear, snames, limit) for k in sizes}
    # evaluate every size on development -> validation (the only use of validation outcomes)
    final = chronological_folds(data.timestamps, cfg)[-1]
    columns = [data.names.index(f) for f in order]
    work = prepare_split(data, final, columns, cfg)
    local = {data.names[c]: i for i, c in enumerate(columns)}
    curve = []
    for k, feats in sets.items():
        cols = [local[f] for f in feats]
        for (kind, h, target) in data.target_meta:
            j = data.target_columns().index(target)
            m, _ = fit_and_score(work, cols, j, cfg.linear.ridge_alpha)
            curve.append({"k": k, "features": len(feats), "target": target, "kind": kind,
                          "horizon": h, **m})
    kind_sets = {}
    for kind in _KINDS:
        part = scores.filter((pl.col("kind") == kind) & ~pl.col("is_probe")).group_by(
            "feature").agg(pl.col("score").max().alias("best"), pl.col("score").mean()
                           .alias("mean"))
        chosen = part.filter(pl.col("best") >= thr) \
            .sort(["best", "mean", "feature"], descending=[True, True, False])["feature"].to_list()
        kind_sets[kind] = _constrained([f for f in chosen if f in uni["selection_universe"]],
                                       40, spear, snames, limit)
    # Minimal, Standard and Extended are prefixes of ONE ranking (selection-stable features
    # first, then the rest of the development universe), so Minimal <= Standard <= Extended
    extended = _constrained(order, 75, spear, snames, limit)
    general = [f for f in order if info[f]["kinds_stable"] >= 2]
    curve_frame = pl.DataFrame(curve, infer_schema_length=None)
    write_table(curve_frame, ctx.path("set_curve"))
    plateau = _plateau(curve_frame, ev_cfg)
    plateau["selection_stable_features"] = int(sum(bool(info[f]["any_stable"]) for f in order))
    minimal = sets.get(plateau["minimal_k"], []) if plateau["minimal_k"] else []
    standard = sets.get(plateau["standard_k"], []) if plateau["standard_k"] else []
    kind_eval: dict[str, list[dict[str, Any]]] = {}
    for kind, feats in kind_sets.items():
        cols = [local[f] for f in feats if f in local]
        if not cols:
            continue
        kind_eval[kind] = []
        for (kk, _h, target) in data.target_meta:
            if kk != kind:
                continue
            m, _ = fit_and_score(work, cols, data.target_columns().index(target),
                                 cfg.linear.ridge_alpha)
            kind_eval[kind].append({"target": target, **m})
    vflags = _validation_flags(work, data, [*{*minimal, *standard, *extended}], local)
    del work
    gc.collect()
    all_sets = {"minimal": minimal, "standard": standard, "extended": extended,
                "general": general, **{f"target_{k}": v for k, v in kind_sets.items()}}
    union = sorted({f for feats in all_sets.values() for f in feats})
    write_table(_selected_yearly_ic(data, union), ctx.path("selected_yearly_ic"))
    justified = {name: _justified(name, feats, kind_eval, ev_cfg) for name, feats in
                 all_sets.items()}
    _describe_sets(ctx, data, all_sets, justified, spear, snames, info, vflags, plateau,
                   kind_eval)
    _mark(marker, stamp, plateau=plateau)


def _plateau(curve: pl.DataFrame, ev_cfg: Any) -> dict[str, Any]:
    """Smallest k reaching each configured share of the best validation rank IC.

    Residual (reversion) targets are left out: their raw validation IC measures
    how a residual is built as much as information (invariant 9).
    """
    if curve.is_empty():
        return {"minimal_k": None, "standard_k": None, "targets_used": []}
    curve = curve.filter(pl.col("kind") != "reversion")
    best = curve.group_by("target").agg(pl.col("rank_ic").max().alias("best"))
    meaningful = best.filter(pl.col("best") >= ev_cfg.min_meaningful_ic)["target"].to_list()
    if not meaningful:
        return {"minimal_k": None, "standard_k": None, "targets_used": [],
                "reason": "no target reaches the meaningful validation rank IC"}
    rel = (curve.filter(pl.col("target").is_in(meaningful)).join(best, on="target")
           .with_columns((pl.col("rank_ic") / pl.col("best")).alias("relative"))
           .group_by("k").agg(pl.col("relative").mean().alias("mean_relative")).sort("k"))
    out: dict[str, Any] = {"targets_used": meaningful,
                           "relative_by_k": dict(zip(rel["k"].to_list(),
                                                     rel["mean_relative"].to_list(),
                                                     strict=True))}
    for name, level in ev_cfg.plateau.items():
        hit = rel.filter(pl.col("mean_relative") >= level)
        out[f"{name}_k"] = int(hit["k"][0]) if hit.height else int(rel["k"][-1])
    return out


def _justified(name: str, feats: list[str], kind_eval: dict[str, list[dict[str, Any]]],
               ev_cfg: Any) -> dict[str, Any]:
    if not feats:
        return {"justified": False, "reason": "no feature is selection-stable for it"}
    if name.startswith("target_"):
        kind = name.removeprefix("target_")
        best = max((r.get("rank_ic") or -1.0 for r in kind_eval.get(kind, [])), default=-1.0)
        if best < ev_cfg.min_meaningful_ic:
            return {"justified": False, "reason": f"best validation rank IC {best:.4f} below "
                                                  f"{ev_cfg.min_meaningful_ic}"}
        return {"justified": True, "reason": f"best validation rank IC {best:.4f}"}
    return {"justified": True, "reason": "general-purpose set"}


def _validation_flags(work: Any, data: SelectionData, feats: list[str], local: dict[str, int]
                      ) -> dict[str, dict[str, Any]]:
    """Post-selection check (Step 42): each selected feature's validation rank IC vs development.

    Read *after* selection, for Prompt #10's information; it changes no set.
    """
    out: dict[str, dict[str, Any]] = {}
    for f in feats:
        if f not in local:
            continue
        i = local[f]
        for (_kind, _h, target) in data.target_meta:
            j = data.target_columns().index(target)
            y = work.y_eval_raw[:, j]
            x = work.x_eval[:, i]
            ok = np.isfinite(y)
            if ok.sum() < 1000:
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                r = float(np.corrcoef(rank_scores(x[ok]), rank_scores(y[ok]))[0, 1])
            out.setdefault(f, {})[target] = r
    return out


def _selected_yearly_ic(data: SelectionData, feats: list[str]) -> pl.DataFrame:
    """Rank IC of each selected feature in each calendar year (in-year ranks).

    Development and validation years only - the loader never holds a reserved
    outcome - and computed after the sets are fixed: a description for the
    report, not an input to any choice.
    """
    if not feats:
        return pl.DataFrame()
    years = data.timestamps.dt.year().to_numpy()
    cuts = np.flatnonzero(np.diff(years) != 0) + 1
    starts, ends = np.concatenate(([0], cuts)), np.concatenate((cuts, [data.n]))
    val_lo = data.rows["validation"][0]
    cols = [data.names.index(f) for f in feats]
    rows = []
    for a, b in zip(starts, ends, strict=True):
        ys = [rank_scores(data.targets[a:b, j]) for j in range(data.targets.shape[1])]
        for f, c in zip(feats, cols, strict=True):
            x = rank_scores(data.features[a:b, c])
            fin = np.isfinite(x)
            for j, (kind, h, col) in enumerate(data.target_meta):
                ok = fin & np.isfinite(ys[j])
                n = int(ok.sum())
                if n < 500:
                    continue
                with np.errstate(invalid="ignore", divide="ignore"):
                    r = float(np.corrcoef(x[ok], ys[j][ok])[0, 1])
                rows.append({"feature": f, "target": col, "kind": kind, "horizon": h,
                             "year": int(years[a]),
                             "period": "validation" if a >= val_lo else "development",
                             "ic": r if np.isfinite(r) else None, "n": n})
    return pl.DataFrame(rows, infer_schema_length=None)


def _describe_sets(ctx: SelectionContext, data: SelectionData, sets: dict[str, list[str]],
                   justified: dict[str, dict[str, Any]], spear: np.ndarray, snames: list[str],
                   info: dict[str, Any], vflags: dict[str, dict[str, Any]],
                   plateau: dict[str, Any], kind_eval: dict[str, Any]) -> None:
    cfg = ctx.cfg
    cost = _read(data.research_dir / "cost/cost_profile.parquet") if data.research_dir else \
        pl.DataFrame()
    fam_cost = ({r["family"]: r for r in cost.iter_rows(named=True)}
                if not cost.is_empty() else {})
    dev_ev = _read(ctx.path("development_evidence.parquet"))
    spec_tab = _read(ctx.path("target_specificity.parquet"))
    peak = ({r["feature"]: json.loads(r["peak_horizon_by_kind"])
             for r in spec_tab.iter_rows(named=True)} if not spec_tab.is_empty() else {})
    regime = _read(ctx.path("regime_robustness.parquet"))
    rows, coll, comp, cost_rows = [], [], [], []
    provenance = _provenance(ctx, data)
    for name, feats in sets.items():
        idx = [snames.index(f) for f in feats if f in snames]
        c = set_collinearity(spear, idx, snames)
        coll.append({"set": name, **{k: v for k, v in c.items() if k != "vif"}})
        fams: dict[str, int] = {}
        for f in feats:
            fam = str(data.registry[f].get("family"))
            fams[fam] = fams.get(fam, 0) + 1
        comp.append({"set": name, "features": len(feats), **fams})
        present = sorted(fams)
        ms = [fam_cost.get(fm, {}).get("live_recompute_ms") for fm in present]
        buffers = [int(data.registry[f].get("min_history") or 0) for f in feats]
        cost_rows.append({
            "set": name, "features": len(feats), "families": json.dumps(present),
            "live_recompute_ms_sum_of_families": float(np.nansum([m for m in ms if m is not None]))
            if ms else None,
            "max_required_history_bars": max(buffers, default=0),
            "expensive_or_moderate_features": sum(
                str(data.registry[f].get("cost")) in ("moderate", "expensive") for f in feats),
            "regime_features": sum(data.registry[f].get("family") == "regime" for f in feats),
            "float32_buffer_mb": (max(buffers, default=0) * len(feats) * 4) / 2 ** 20})
        for pos, f in enumerate(feats):
            rows.append({"set": name, "position": pos, "feature": f,
                         "family": data.registry[f].get("family"),
                         "selection_stable": bool(info.get(f, {}).get("any_stable")),
                         "kinds_stable": info.get(f, {}).get("kinds_stable"),
                         "mean_stability_score": info.get(f, {}).get("mean_score"),
                         "peak_horizons": json.dumps(peak.get(f, {})),
                         "validation_rank_ic": json.dumps(vflags.get(f, {}))})
        if justified[name]["justified"] and feats:
            evidence = {}
            for f in feats:
                part = dev_ev.filter((pl.col("feature") == f) & pl.col("passes"))
                reg = regime.filter(pl.col("feature") == f) if not regime.is_empty() else regime
                evidence[f] = {
                    "selection_stable": bool(info.get(f, {}).get("any_stable")),
                    "kinds_passing_development": sorted(part["kind"].unique().to_list()),
                    "best_development_rank_ic": _max_abs(part["ic"])
                    if part.height else None,
                    "stability_score_mean": info.get(f, {}).get("mean_score"),
                    "regime_class": (sorted(reg["regime_class"].unique().to_list())
                                     if reg.height else []),
                    "peak_horizon_by_kind": peak.get(f, {}),
                    "validation_rank_ic_post_selection": vflags.get(f, {})}
            manifest = build_manifest(
                name, ctx.timeframe, feats, data.registry, version=cfg.manifest.version,
                families_order=cfg.manifest.families_order, provenance=provenance,
                evidence=evidence, effective_rank=c.get("effective_rank"),
                note=justified[name]["reason"])
            path = ctx.path(f"manifests/{name}.json")
            ensure_dir(path.parent)
            write_manifest(path, manifest)
    write_table(pl.DataFrame(rows, infer_schema_length=None), ctx.path("candidate_sets"))
    write_table(pl.DataFrame(coll, infer_schema_length=None), ctx.path("collinearity"))
    write_table(pl.DataFrame(comp, infer_schema_length=None), ctx.path("family_composition"))
    write_table(pl.DataFrame(cost_rows, infer_schema_length=None), ctx.path("cost"))
    write_json(ctx.path("sets.json"), {"sets": sets, "justified": justified,
                                       "plateau": plateau, "kind_validation": kind_eval})


def _provenance(ctx: SelectionContext, data: SelectionData) -> dict[str, Any]:
    p = ctx.cfg.periods
    lineage = data.factory_manifest.get("dataset_lineage") or {}
    return {
        "dataset_version": data.factory_manifest.get("tick_dataset_version"),
        "bar_dataset_version": data.factory_manifest.get("bar_dataset_version"),
        "factory_version": data.factory_manifest.get("factory_version"),
        "target_version": data.target_manifest.get("target_version"),
        "partial_dataset": bool(lineage.get("partial")),
        "created_from_period": p.development.label(),
        "selection_validation_period": p.validation.label(),
        "reserved_test_period": p.reserved_test.label(),
        "selection_config": ctx.cfg.fingerprint(),
        "features_config": ctx.cfgs.features.fingerprint(),
        "source_feed": data.factory_manifest.get("source_feed"),
        "timestamp_convention": data.factory_manifest.get("timestamp_convention"),
        "git_commit": git_info(ctx.cfg.project_root).get("commit"),
        "created_utc": utc_now_iso(),
    }


# ---------------------------------------------------------------------------
# Stage: audit
# ---------------------------------------------------------------------------
_AUDIT_CHECKS = {
    "returns": "trailing log-price differences / moments over past bars",
    "volatility": "trailing sums, EWMA recursions and rolling ranks over past bars",
    "autocorrelation": "rolling correlations of past returns",
    "regression": "trailing-window OLS (Prompt #3 store, version-checked)",
    "ou": "rolling AR(1) fits of the trailing residual (block-shifted prefix sums)",
    "fft": "causal rolling FFT (Prompt #5); stored sets tail-recomputed by the factory gate",
    "wavelet": "one-sided causal MODWT (Prompt #6); stored sets tail-recomputed by the gate",
    "regime": "walk-forward filtered probabilities (Prompt #7), smoothed / Viterbi refused",
    "microstructure": "the bar's own spread / tick count and trailing ranks",
    "time": "the bar's open time only",
    "interaction": "product of trailing z-scores of two causal features",
}


def stage_audit(ctx: SelectionContext, data: SelectionData) -> None:
    begun = _begin(ctx, data, "audit")
    if begun is None:
        return
    stamp, marker = begun
    cfg = ctx.cfg
    manifests = {p.stem: _json(p) for p in sorted(ctx.path("manifests").glob("*.json"))}
    union = sorted({r["name"] for m in manifests.values() for r in m["features"]})
    cfgs = ctx.cfgs
    live: dict[str, Any] = {"features": 0, "passed": True, "note": "no manifest"}
    if union:
        bars = load_bar_series(cfgs.config, ctx.timeframe)
        files = sorted(data.matrix_dir.glob("year=*/part-0.parquet")) if data.matrix_dir else []
        stored = pl.read_parquet(files, columns=["timestamp", *union]).sort("timestamp")
        pct = cfgs.features.percentile_window(bars.bar_seconds)
        buffer = required_buffer(data.registry, union, pct)
        start = data.rows["validation"][1] - 150            # the last validation bars
        live = live_reconstruction(bars, stored, union, data.registry, fcfg=cfgs.features,
                                   ou=cfgs.ou, research=cfgs.research,
                                   regression=cfgs.regression, spectral=cfgs.spectral,
                                   wavelet=cfgs.wavelet, start=start, steps=60, buffer=buffer,
                                   stride=2)
        timestamps_ok = bool(stored["timestamp"].equals(bars.timestamps))
        del stored, bars
        gc.collect()
    else:
        timestamps_ok = True
    write_json(ctx.path("live_reconstruction.json"), live)
    audit = []
    for f in union:
        spec = data.registry[f]
        fam = str(spec.get("family"))
        recomputed = f in (live.get("features_checked") or [])
        audit.append({
            "feature": f, "family": fam, "definition": spec.get("definition"),
            "causal_computation": _AUDIT_CHECKS.get(fam, "registered causal family"),
            "live_reconstruction": ("matched" if recomputed and f not in live.get("failed", [])
                                    else "not recomputed (Prompt #7 prefix-invariance tests)"
                                    if not recomputed else "MISMATCH"),
            "normalization_scaling": "trailing statistics only; no full-sample scaling",
            "quantiles": ("trailing rolling rank" if "percentile" in f else "none"),
            "regime_fitting": ("walk-forward, filtered" if fam == "regime" else "not used"),
            "pca": "not used by any manifest (research comparison only, fitted on training)",
            "timestamp_alignment": "matrix timestamps equal the bars" if timestamps_ok
            else "MISMATCH",
            "synthetic_prefix_test": "tests/test_feature_leakage.py (every registered feature)",
            "verified": (recomputed and f not in live.get("failed", [])) or fam == "regime"
                        or f.startswith("ix_") and not recomputed})
    isolation = _reserved_isolation(ctx, data)
    write_json(ctx.path("leakage_audit.json"), {
        "features": audit, "all_verified": all(a["verified"] for a in audit),
        "live_reconstruction_passed": live.get("passed"),
        "timestamp_alignment_ok": timestamps_ok, "reserved_isolation": isolation})
    write_json(ctx.path("reserved_isolation.json"), isolation)
    del cfg
    _mark(marker, stamp)


def _reserved_isolation(ctx: SelectionContext, data: SelectionData) -> dict[str, Any]:
    from ..selection.periods import first_row_at

    reserved = ctx.cfg.periods.reserved_test.start
    last_target_ts = str(data.timestamps[-1])
    splits = chronological_folds(data.timestamps, ctx.cfg)
    checks = {
        "targets_loaded_before_reserved_start": bool(
            data.timestamps[-1] < _as_datetime(reserved)),
        "rows_loaded": data.n,
        "guard_reserved_start_row": data.guard.reserved_start_row,
        "no_split_evaluates_reserved_rows": all(s.evaluate[1] <= data.n for s in splits),
        "last_target_timestamp": last_target_ts,
        "reserved_start": reserved.isoformat(),
        "first_row_at_reserved_start": first_row_at(data.timestamps, reserved),
        "prompt8_statuses_used_for_selection": False,
    }
    checks["passed"] = bool(checks["targets_loaded_before_reserved_start"]
                            and checks["no_split_evaluates_reserved_rows"]
                            and checks["first_row_at_reserved_start"] == data.n)
    if not checks["passed"]:
        raise RuntimeError(f"reserved-test isolation FAILED: {checks}")
    return checks


def _as_datetime(d: Any) -> Any:
    from datetime import datetime

    return datetime(d.year, d.month, d.day)


# ---------------------------------------------------------------------------
# One timeframe
# ---------------------------------------------------------------------------
_RUNNERS: dict[str, Callable[[SelectionContext, SelectionData], None]] = {
    "universe": stage_universe, "redundancy": stage_redundancy, "evidence": stage_evidence,
    "stability": stage_stability, "nested": stage_nested, "sets": stage_sets,
    "audit": stage_audit}


def run_selection(cfgs: SelectionConfigs, timeframe: str, *, stages: tuple[str, ...] = STAGES,
                  resume: bool = True, progress: Callable[[str], None] | None = None
                  ) -> dict[str, Any]:
    unknown = sorted(set(stages) - set(STAGES))
    if unknown:
        raise ValueError(f"unknown stage(s) {unknown}; expected from {list(STAGES)}")
    out = ensure_dir(cfgs.selection.results_path / timeframe)
    ctx = SelectionContext(timeframe=timeframe, cfgs=cfgs, out_dir=out, progress=progress)
    started = time.perf_counter()
    ctx.log("loading development + validation features and outcome-safe targets")
    data = load_selection_data(cfgs.selection, cfgs.features, cfgs.targets, timeframe)
    ctx.log(f"{data.n:,} rows, {len(data.real_names())} features, {len(data.probes)} probes, "
            f"{len(data.target_meta)} targets")
    write_json(ctx.path("period_split.json"), _period_split(ctx, data))
    timings = {}
    for stage in STAGES:
        if stage not in stages:
            continue
        if not resume:
            ctx.path(f".{stage}.done.json").unlink(missing_ok=True)
        t0 = time.perf_counter()
        _RUNNERS[stage](ctx, data)
        gc.collect()
        timings[stage] = time.perf_counter() - t0
    info = {"timeframe": timeframe, "stages": list(stages), "timings": timings,
            "seconds": time.perf_counter() - started, "finished_utc": utc_now_iso()}
    write_json(ctx.path("run_info.json"), info)
    return info


def _period_split(ctx: SelectionContext, data: SelectionData) -> dict[str, Any]:
    p = ctx.cfg.periods
    out: dict[str, Any] = {"purge_bars": p.purge_bars, "embargo_bars": p.embargo_bars,
                           "boundaries": "calendar quarters, [start, end)"}
    for name in ("development", "validation", "recent_development"):
        lo, hi = data.rows[name]
        out[name] = {"period": getattr(p, name).label(), "rows": hi - lo,
                     "first_bar": str(data.timestamps[lo]) if hi > lo else None,
                     "last_bar": str(data.timestamps[hi - 1]) if hi > lo else None}
    out["reserved_test"] = {"period": p.reserved_test.label(),
                            "outcomes_loaded": False,
                            "note": "untouched by selection; feature values may be computed"}
    out["folds"] = [{"name": s.name, "train_rows": list(s.train),
                     "evaluate_rows": list(s.evaluate),
                     "train_period": f"{data.timestamps[s.train[0]]} .. "
                                     f"{data.timestamps[s.train[1] - 1]}",
                     "evaluate_period": f"{data.timestamps[s.evaluate[0]]} .. "
                                        f"{data.timestamps[s.evaluate[1] - 1]}"}
                    for s in chronological_folds(data.timestamps, ctx.cfg)]
    return out
