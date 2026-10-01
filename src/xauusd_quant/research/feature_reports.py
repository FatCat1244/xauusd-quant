r"""Multiple testing, feature statuses, the ledger and the summaries (Prompt #8, Steps 54-59, 67-80).

Reads the stage tables of :mod:`.feature_research` for one timeframe and writes,
under ``results/feature_research/<tf>/``:

``tests/all_tests.parquet``      every test: family, p, Benjamini-Hochberg q within its
                                 family and across the timeframe, Bonferroni, verdict,
                                 ``ALPHA-H`` id, preregistered or exploratory
``tests/family_summary.csv``     tests, discoveries and Bonferroni passes per family
``feature_evidence.parquet``     per feature x target (pooled rank IC): every condition
                                 the status reads, with its value
``feature_status.csv``           one row per feature: final status and the reason
``candidate_features.json``      the research candidates (strong / candidate / weak)
``ablation_manifests.json``      family groups A-G (+H) as feature-id lists
``noise_control.json``           the same tests and statuses applied to noise features:
                                 how many "discoveries" selection alone produces
``multiple_testing.json``        counts, effective number of features, inputs for a
                                 later deflated-statistic / PBO analysis
``summary.json``, ``feature_research_summary.md``

and upserts every test into the shared research ledger. A test is one
(timeframe, metric, feature, target, conditioning, condition); its ``ALPHA-H``
id is permanent (``results/feature_research/test_registry.parquet``).

Statuses follow :func:`..alpha.ranking.classify`; a feature's evidence for one
target kind is its best target of that kind, preferring targets that pass every
null and keep their sign. Undefined statistics are missing, never evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..alpha.config import AlphaConfig
from ..alpha.ranking import (
    PIPELINE_REQUIRED,
    STATUS_ORDER,
    KindEvidence,
    TestRegistry,
    benjamini_hochberg,
    classify,
    passes_nulls,
    status_rank,
)
from ..features.factory import current_version_dir
from ..features.factory_config import FeatureFactoryConfig
from ..targets.alignment import load_targets
from ..utils.clock import utc_now_iso
from ..utils.paths import ensure_dir
from .feature_candidates import ablation_sets, candidate_manifest, choose_representatives
from .feature_research import write_json, write_table
from .research_ledger import ResearchLedger
from .study_io import clean_json, code_fingerprint, git_info, package_versions

__all__ = ["KINDS", "Tables", "build_timeframe_report", "load_tables", "pipeline_verdicts",
           "write_comparison"]

KINDS = ("direction", "magnitude", "volatility", "residual", "excursion")
_CANDIDATES = ("strong_candidate", "candidate", "weak_candidate")


@dataclass
class Tables:
    timeframe: str
    out_dir: Path
    ic: pl.DataFrame
    consistency: pl.DataFrame
    nulls: pl.DataFrame
    pipeline: pl.DataFrame
    mi: pl.DataFrame
    conditioning: pl.DataFrame
    interactions: pl.DataFrame
    noise_ic: pl.DataFrame
    noise_consistency: pl.DataFrame
    noise_nulls: pl.DataFrame
    kind_null: pl.DataFrame
    noise_kind_null: pl.DataFrame
    clusters: pl.DataFrame
    correlations: pl.DataFrame
    cost: pl.DataFrame
    decay: pl.DataFrame
    registry: list[dict[str, Any]]
    manifest: dict[str, Any]
    target_manifest: dict[str, Any]
    stages: dict[str, Any]


def _read(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path) if path.exists() else pl.DataFrame()


def load_tables(fcfg: FeatureFactoryConfig, targets_path: Path, timeframe: str) -> Tables:
    out = fcfg.results_path / timeframe
    base = current_version_dir(fcfg.factory_path, timeframe)
    manifest = json.loads((base / "_manifest.json").read_text(encoding="utf-8"))
    registry = json.loads((base / "registry.json").read_text(encoding="utf-8"))
    _, tmanifest = load_targets(targets_path, timeframe, columns=["timestamp"])
    conditioning = [_read(out / f"conditioning/{n}.parquet")
                    for n in ("volatility_conditioning", "regime_conditioning", "intraday_ic")]
    stages = {}
    for marker in sorted(out.glob("*/.done.json")):
        try:
            stages[marker.parent.name] = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return Tables(
        timeframe=timeframe, out_dir=out,
        ic=_read(out / "ic/feature_ic.parquet"),
        consistency=_read(out / "ic/ic_consistency.parquet"),
        nulls=_read(out / "nulls/null_tests.parquet"),
        pipeline=_read(out / "pipeline/pipeline_null_ic.parquet"),
        mi=_read(out / "mi/mutual_information.parquet"),
        conditioning=pl.concat([c for c in conditioning if not c.is_empty()],
                               how="diagonal_relaxed") if any(not c.is_empty()
                                                              for c in conditioning)
        else pl.DataFrame(),
        interactions=_read(out / "interactions/interaction_increment.parquet"),
        noise_ic=_read(out / "nulls/synthetic_noise_ic.parquet"),
        noise_consistency=_read(out / "nulls/synthetic_noise_consistency.parquet"),
        noise_nulls=_read(out / "nulls/synthetic_noise_null_tests.parquet"),
        kind_null=_read(out / "nulls/kind_max_null.parquet"),
        noise_kind_null=_read(out / "nulls/synthetic_noise_kind_max_null.parquet"),
        clusters=_read(out / "redundancy/feature_clusters_table.parquet"),
        correlations=_read(out / "redundancy/feature_correlations.parquet"),
        cost=_read(out / "cost/cost_profile.parquet"),
        decay=_read(out / "ic/alpha_decay.parquet"),
        registry=registry, manifest=manifest, target_manifest=tmanifest, stages=stages)


# ---------------------------------------------------------------------------
# Tests, families, q-values
# ---------------------------------------------------------------------------
def _target_family(target: str) -> str:
    return target.removeprefix("target_").rsplit("_", 1)[0]


def collect_tests(t: Tables) -> pl.DataFrame:
    """Every test of the timeframe in one table: metric, value, p, family."""
    tf = t.timeframe
    parts = []
    if not t.ic.is_empty():
        metric = (pl.when(pl.col("target").str.starts_with("marginal_")).then(pl.lit("marginal_"))
                  .otherwise(pl.lit(""))
                  + pl.when(pl.col("method") == "spearman").then(pl.lit("rank_ic"))
                  .otherwise(pl.lit("pearson_ic")))
        parts.append(t.ic.select(
            "feature", "target", "horizon", "kind", metric.alias("metric"),
            pl.col("ic").alias("value"), "se", pl.col("n").alias("n_obs"),
            pl.col("p").alias("p_value"), pl.lit("pooled").alias("conditioning"),
            pl.lit("all").alias("condition")))
    if not t.conditioning.is_empty():
        parts.append(t.conditioning.select(
            "feature", "target", "horizon", "kind", pl.lit("conditional_rank_ic").alias("metric"),
            pl.col("ic").alias("value"), "se", pl.col("n").alias("n_obs"),
            pl.col("p").alias("p_value"), "conditioning", "condition"))
    if not t.mi.is_empty():
        parts.append(t.mi.select(
            "feature", "target", "horizon", "kind", pl.lit("copula_mi").alias("metric"),
            pl.col("mi_nats").alias("value"), pl.lit(None, dtype=pl.Float64).alias("se"),
            pl.col("pairs").cast(pl.Float64).alias("n_obs"),
            pl.col("p_shift_z").alias("p_value"), pl.lit("pooled").alias("conditioning"),
            pl.lit("all").alias("condition")))
    if not t.interactions.is_empty():
        parts.append(t.interactions.select(
            pl.col("interaction").alias("feature"), "target",
            pl.col("target").str.extract(r"_(\d+)$").cast(pl.Int64).alias("horizon"),
            pl.col("target").map_elements(_kind_of, return_dtype=pl.Utf8).alias("kind"),
            pl.lit("interaction_oos_r2_gain").alias("metric"),
            pl.col("mean_delta_oos_r2").alias("value"),
            pl.lit(None, dtype=pl.Float64).alias("se"), pl.lit(None, dtype=pl.Float64).alias("n_obs"),
            pl.lit(None, dtype=pl.Float64).alias("p_value"), pl.lit("pooled").alias("conditioning"),
            pl.lit("all").alias("condition")))
    tests = pl.concat(parts, how="diagonal_relaxed").with_columns(
        pl.col("n_obs").cast(pl.Float64), pl.col("horizon").cast(pl.Int64),
        pl.col("value").cast(pl.Float64).fill_nan(None),
        pl.col("p_value").cast(pl.Float64).fill_nan(None))
    return tests.with_columns(
        (pl.lit(f"{tf}|") + pl.col("metric") + "|" + pl.col("kind") + "|"
         + pl.col("conditioning")).alias("test_family"))


def _kind_of(target: str) -> str:
    from ..targets.alignment import target_kind

    return target_kind(target)


def add_q_values(tests: pl.DataFrame, *, bonferroni_alpha: float) -> pl.DataFrame:
    """BH q within each family and across the timeframe; Bonferroni within the family."""
    tests = tests.with_row_index("_row")
    parts = []
    for part in tests.partition_by("test_family", maintain_order=True):
        p = part["p_value"].fill_null(np.nan).to_numpy().astype(np.float64)
        m = int(np.isfinite(p).sum())
        parts.append(part.with_columns(
            pl.Series("q_value", benjamini_hochberg(p)).fill_nan(None),
            pl.lit(m, dtype=pl.Int64).alias("family_tests"),
            pl.when(pl.col("p_value").is_null()).then(None)
            .otherwise(pl.col("p_value") < bonferroni_alpha / max(m, 1))
            .alias("bonferroni_pass")))
    out = pl.concat(parts).sort("_row")
    p_all = out["p_value"].fill_null(np.nan).to_numpy().astype(np.float64)
    return out.with_columns(pl.Series("q_timeframe", benjamini_hochberg(p_all)).fill_nan(None)
                            ).drop("_row")


# ---------------------------------------------------------------------------
# Evidence for the pooled rank IC tests
# ---------------------------------------------------------------------------
def pipeline_wide(pipeline: pl.DataFrame) -> pl.DataFrame:
    """Per feature x target: every pipeline null's rank IC and SE as columns."""
    if pipeline.is_empty():
        return pl.DataFrame(schema={"feature": pl.Utf8, "target": pl.Utf8})
    pp = pipeline.filter(pl.col("method") == "spearman")
    ic = pp.pivot(on="null", index=["feature", "target"], values="ic")
    se = pp.pivot(on="null", index=["feature", "target"], values="se")
    ic = ic.rename({c: f"pipeline_{c}_ic" for c in ic.columns if c not in ("feature", "target")})
    se = se.rename({c: f"pipeline_{c}_se" for c in se.columns if c not in ("feature", "target")})
    return ic.join(se, on=["feature", "target"], how="full", coalesce=True)


def pipeline_verdicts(ev: pl.DataFrame, a: AlphaConfig) -> pl.DataFrame:
    """Veto per pooled test: a veto null of the target kind *reproduces* the real IC.

    Reproduces = same sign, itself distinguishable from zero (``veto_null_z``)
    and at least ``veto_share`` of the real |IC|. A null IC near zero never
    vetoes - weak-but-real information is judged by the other nulls, not
    confused with a mechanical artefact. ``mechanical_share`` is the largest
    same-sign share any veto null reproduces (0 when none does).
    """
    nl = a.nulls
    beyond: list[bool | None] = []
    share: list[float | None] = []
    matched_by: list[str | None] = []
    for r in ev.select("kind", "value", *[c for c in ev.columns
                                           if c.startswith("pipeline_")]).iter_rows(named=True):
        value = r.get("value")
        if value is None or not np.isfinite(value) or value == 0:
            beyond.append(None)
            share.append(None)
            matched_by.append(None)
            continue
        scored, best, who = False, 0.0, []
        for null in nl.veto_nulls(r["kind"]):
            ic, se = r.get(f"pipeline_{null}_ic"), r.get(f"pipeline_{null}_se")
            if ic is None or not np.isfinite(ic):
                continue
            scored = True
            same = ic * np.sign(value)
            best = max(best, same / abs(value))
            if (same > 0 and se is not None and np.isfinite(se) and abs(ic) > nl.veto_null_z * se
                    and abs(ic) >= nl.veto_share * abs(value)):
                who.append(null)
        beyond.append((not who) if scored else None)
        share.append(best if scored else None)
        matched_by.append(",".join(who) if who else None)
    return ev.with_columns(pl.Series("beyond_pipeline_null", beyond, dtype=pl.Boolean),
                           pl.Series("mechanical_share", share, dtype=pl.Float64),
                           pl.Series("matched_by_null", matched_by, dtype=pl.Utf8))


def evidence_table(t: Tables, tests: pl.DataFrame, a: AlphaConfig) -> pl.DataFrame:
    """Pooled rank-IC tests of the stored targets, with every condition the status reads."""
    fdr = a.multiple_testing.fdr_q
    ev = tests.filter((pl.col("metric") == "rank_ic") & (pl.col("conditioning") == "pooled")
                      & pl.col("target").str.starts_with("target_"))
    if not t.nulls.is_empty():
        keep = [c for c in t.nulls.columns if c in (
            "feature", "target", "rank_ic0", "shift_null_q", "shift_null_mean_abs", "perm_null_q",
            "null_percentile", "beyond_shift_null") or c.startswith("wrong_")]
        ev = ev.join(t.nulls.select(keep), on=["feature", "target"], how="left")
    else:
        ev = ev.with_columns(pl.lit(None, dtype=pl.Boolean).alias("beyond_shift_null"),
                             pl.lit(None, dtype=pl.Float64).alias("rank_ic0"))
    ev = ev.join(pipeline_wide(t.pipeline), on=["feature", "target"], how="left")
    ev = _join_kind_null(ev, t.kind_null)
    if not t.consistency.is_empty():
        cons = t.consistency.filter(pl.col("method") == "spearman").drop(
            "horizon", "kind", "method", strict=False)
        ev = ev.join(cons, on=["feature", "target"], how="left")
    ratio = a.nulls.slow_component_ratio
    ev = ev.with_columns(
        (pl.col("wrong_plus_60d").abs() >= ratio * pl.col("rank_ic0").abs())
        .alias("slow_component") if "wrong_plus_60d" in ev.columns
        else pl.lit(None, dtype=pl.Boolean).alias("slow_component"),
        pl.when(pl.col("recent_ic").is_null() | pl.col("value").is_null()).then(None)
        .otherwise(pl.col("recent_ic").sign() == pl.col("value").sign())
        .alias("recent_same_sign") if "recent_ic" in ev.columns
        else pl.lit(None, dtype=pl.Boolean).alias("recent_same_sign"))
    ev = pipeline_verdicts(ev, a)
    pipeline_ok = (pl.when(pl.col("kind").is_in(list(PIPELINE_REQUIRED)))
                   .then(pl.col("beyond_pipeline_null").fill_null(False))
                   .otherwise(pl.col("beyond_pipeline_null").fill_null(True)))
    return ev.with_columns(
        (pl.col("beyond_kind_max_null").fill_null(False)
         & (pl.col("q_value") < fdr).fill_null(False) & pipeline_ok).alias("passes_nulls"))


def _join_kind_null(frame: pl.DataFrame, kind_null: pl.DataFrame) -> pl.DataFrame:
    """Attach the feature x kind max-T verdict (False where the null stage has not run)."""
    if kind_null.is_empty():
        return frame.with_columns(pl.lit(None, dtype=pl.Boolean).alias("beyond_kind_max_null"),
                                  pl.lit(None, dtype=pl.Float64).alias("kind_max_null_q"),
                                  pl.lit(None, dtype=pl.Float64).alias("kind_max_null_p"))
    return frame.join(kind_null.select(
        "feature", "kind", "beyond_kind_max_null",
        pl.col("max_null_q").alias("kind_max_null_q"),
        pl.col("max_null_p").alias("kind_max_null_p")), on=["feature", "kind"], how="left")


def _verdicts(tests: pl.DataFrame, ev: pl.DataFrame, a: AlphaConfig) -> pl.DataFrame:
    """One verdict per test (pooled rank IC tests read their null evidence)."""
    fdr = a.multiple_testing.fdr_q
    wanted = {"beyond_shift_null": pl.Boolean, "beyond_pipeline_null": pl.Boolean,
              "slow_component": pl.Boolean, "shift_null_q": pl.Float64}
    ev = ev.with_columns([pl.lit(None, dtype=d).alias(c) for c, d in wanted.items()
                          if c not in ev.columns])
    joined = tests.join(ev.select("feature", "target", "metric", "conditioning", *wanted),
                        on=["feature", "target", "metric", "conditioning"], how="left")
    sig = (pl.col("q_value") < fdr).fill_null(False)
    rank_pooled = (pl.col("metric") == "rank_ic") & (pl.col("conditioning") == "pooled") \
        & pl.col("target").str.starts_with("target_")
    verdict = (
        pl.when(pl.col("metric") == "interaction_oos_r2_gain").then(
            pl.when(pl.col("value").is_null()).then(pl.lit("untested"))
            .when(pl.col("value") > 0).then(pl.lit("positive_mean_increment"))
            .otherwise(pl.lit("no_increment")))
        .when(pl.col("value").is_null() | pl.col("p_value").is_null()).then(pl.lit("untested"))
        .when(rank_pooled & sig & pl.col("beyond_shift_null").fill_null(False)
              & (pl.col("beyond_pipeline_null") == False).fill_null(False))  # noqa: E712
        .then(pl.lit("matched_by_pipeline_null"))
        .when(rank_pooled & sig & pl.col("beyond_shift_null").fill_null(False)
              & (pl.col("kind") == "direction") & pl.col("slow_component").fill_null(False))
        .then(pl.lit("slow_level_only"))
        .when(rank_pooled & sig & pl.col("beyond_shift_null").fill_null(False))
        .then(pl.lit("beyond_nulls"))
        .when(rank_pooled & sig).then(pl.lit("significant_within_shift_null"))
        .when(sig).then(pl.lit("significant"))
        .otherwise(pl.lit("not_significant")))
    return joined.with_columns(verdict.alias("verdict"))


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def _f(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def _kind_evidence(rows: list[dict[str, Any]], a: AlphaConfig) -> KindEvidence:
    """The best target of one kind: passing and stable first, then passing, then any."""
    cls = a.classification

    def as_ev(r: dict[str, Any]) -> KindEvidence:
        # the kind-level max-T verdict: the best target of a kind was chosen among several
        return KindEvidence(kind=r["kind"], target=r["target"], rank_ic=_f(r.get("value")),
                            q_value=_f(r.get("q_value")),
                            beyond_shift_null=bool(r.get("beyond_kind_max_null")),
                            beyond_pipeline_null=r.get("beyond_pipeline_null"),
                            sign_consistency=_f(r.get("sign_consistency")),
                            recent_same_sign=r.get("recent_same_sign"),
                            era_sign_flips=int(r.get("era_sign_flips") or 0),
                            slow_component=r.get("slow_component"))

    evs = [as_ev(r) for r in rows if np.isfinite(_f(r.get("value")))]
    if not evs:
        return as_ev(rows[0])
    fdr = a.multiple_testing.fdr_q

    def rank(e: KindEvidence) -> tuple[int, float]:
        passing = passes_nulls(e, fdr)
        stable = (passing and np.isfinite(e.sign_consistency)
                  and e.sign_consistency >= cls.candidate_sign_years
                  and e.recent_same_sign is not False
                  and not (e.kind == "direction" and e.slow_component))
        return (0 if stable else 1 if passing else 2, -abs(e.rank_ic))

    return min(evs, key=rank)


def classify_features(t: Tables, ev: pl.DataFrame, a: AlphaConfig
                      ) -> tuple[list[dict[str, Any]], dict[str, list[KindEvidence]]]:
    """Final status rows (with redundancy resolved) and the per-kind evidence used."""
    fdr = a.multiple_testing.fdr_q
    by_feature: dict[str, list[dict[str, Any]]] = {}
    for r in ev.iter_rows(named=True):
        by_feature.setdefault(r["feature"], []).append(r)
    specs = {r["name"]: r for r in t.registry}
    evidence: dict[str, list[KindEvidence]] = {}
    prelim: dict[str, dict[str, Any]] = {}
    for name, spec in specs.items():
        rows = by_feature.get(name, [])
        kinds = []
        for kind in KINDS:
            kr = [r for r in rows if r["kind"] == kind]
            if kr:
                kinds.append(_kind_evidence(kr, a))
        evidence[name] = kinds
        status, reason, best_kind = classify(
            name, kinds, live_safe=bool(spec.get("live_safe", True)),
            invalid_reason=spec.get("invalid_reason"), cluster_better=None,
            cfg=a.classification, fdr_q=fdr)
        top = next((e for e in kinds if e.kind == best_kind), None)
        prelim[name] = {"status": status, "reason": reason, "best_kind": best_kind,
                        "score": (abs(top.rank_ic) * top.sign_consistency
                                  if top is not None and np.isfinite(top.sign_consistency)
                                  else None),
                        "cost": spec.get("cost"), "min_history": spec.get("min_history")}
    clusters = ({r["feature"]: int(r["cluster"]) for r in t.clusters.iter_rows(named=True)}
                if not t.clusters.is_empty() else {})
    rho: dict[tuple[str, str], float] = {}
    if not t.correlations.is_empty():
        for r in t.correlations.select("a", "b", "spearman").iter_rows(named=True):
            if r["spearman"] is not None:
                rho[(r["a"], r["b"])] = abs(float(r["spearman"]))
    reps = choose_representatives(prelim, clusters, rho,
                                  threshold=a.classification.max_redundancy_rho)
    measured = ({r["family"]: r.get("measured_category") for r in t.cost.iter_rows(named=True)}
                if not t.cost.is_empty() else {})
    half_life = _half_lives(t.decay)
    rows_out = []
    for name, spec in specs.items():
        kinds = evidence[name]
        rep = reps.get(name)
        if rep is not None:
            status, reason, best_kind = classify(
                name, kinds, live_safe=bool(spec.get("live_safe", True)),
                invalid_reason=spec.get("invalid_reason"), cluster_better=rep,
                cfg=a.classification, fdr_q=fdr)
        else:
            p = prelim[name]
            status, reason, best_kind = p["status"], p["reason"], p["best_kind"]
        top = next((e for e in kinds if e.kind == best_kind), None)
        if top is None and kinds:
            top = max(kinds, key=lambda e: abs(e.rank_ic) if np.isfinite(e.rank_ic) else -1)
        row: dict[str, Any] = {
            "feature": name, "feature_id": spec.get("feature_id"), "family": spec.get("family"),
            "window": spec.get("window"), "min_history": spec.get("min_history"),
            "cost": spec.get("cost"), "measured_cost": measured.get(spec.get("family")),
            "incremental_update": spec.get("incremental_update"),
            "prior_status": spec.get("prior_status"), "status": status, "reason": reason,
            "best_kind": best_kind, "best_target": top.target if top else None,
            "best_rank_ic": _clean(top.rank_ic) if top else None,
            "best_abs_rank_ic": _clean(abs(top.rank_ic)) if top else None,
            "best_q_value": _clean(top.q_value) if top else None,
            "sign_consistency": _clean(top.sign_consistency) if top else None,
            "era_sign_flips": top.era_sign_flips if top else None,
            "beyond_pipeline_null": top.beyond_pipeline_null if top else None,
            "recent_same_sign": top.recent_same_sign if top else None,
            "cluster": clusters.get(name), "representative": rep,
            "kinds_passing": [e.kind for e in kinds if passes_nulls(e, fdr)],
            "live_safe": spec.get("live_safe"), "invalid_reason": spec.get("invalid_reason"),
            "excluded_from_default_candidates": spec.get("excluded_from_default_candidates"),
        }
        for e in kinds:
            row[f"{e.kind}_rank_ic"] = _clean(e.rank_ic)
            row[f"{e.kind}_target"] = e.target
            row[f"{e.kind}_passes"] = passes_nulls(e, fdr)
        row["information_half_life_return"] = half_life.get((name, "marginal_return"))
        row["information_half_life_abs_return"] = half_life.get((name, "marginal_abs_return"))
        best_row = next((r for r in by_feature.get(name, [])
                         if r["target"] == row["best_target"]), {})
        row["recent_ic"] = _clean(best_row.get("recent_ic"))
        row["mechanical_share"] = _clean(best_row.get("mechanical_share"))
        row["matched_by_null"] = best_row.get("matched_by_null")
        row["block_bootstrap_ic"] = _clean(next((best_row[c] for c in best_row
                                                 if c.startswith("pipeline_block_bootstrap")
                                                 and c.endswith("_ic")), None))
        rows_out.append(row)
    rows_out.sort(key=lambda r: (status_rank(r["status"]), -(r.get("best_abs_rank_ic") or 0.0)))
    return rows_out, evidence


def _clean(v: Any) -> float | None:
    x = _f(v)
    return x if np.isfinite(x) else None


def _half_lives(decay: pl.DataFrame) -> dict[tuple[str, str], float | None]:
    out: dict[tuple[str, str], float | None] = {}
    if decay.is_empty() or "alpha_information_half_life" not in decay.columns:
        return out
    part = decay.filter((pl.col("method") == "spearman") & (pl.col("curve") == "marginal"))
    for r in part.iter_rows(named=True):
        out[(r["feature"], r["target_family"])] = _clean(r.get("alpha_information_half_life"))
    return out


# ---------------------------------------------------------------------------
# Noise features: what selection alone produces
# ---------------------------------------------------------------------------
def noise_control(t: Tables, tests: pl.DataFrame, a: AlphaConfig) -> dict[str, Any]:
    """The research's own criteria applied to features that carry no information."""
    if t.noise_ic.is_empty():
        return {"available": False}
    fdr = a.multiple_testing.fdr_q
    nic = t.noise_ic.filter((pl.col("method") == "spearman")
                            & pl.col("target").str.starts_with("target_"))
    nic = nic.with_columns(pl.col("p").fill_nan(None), pl.col("ic").fill_nan(None))
    # q-values as if the noise tests had been in the real family
    real = tests.filter((pl.col("metric") == "rank_ic") & (pl.col("conditioning") == "pooled")
                        & pl.col("target").str.starts_with("target_"))
    qs = []
    for kind in nic["kind"].unique().to_list():
        rp = real.filter(pl.col("kind") == kind)["p_value"].fill_null(np.nan).to_numpy()
        part = nic.filter(pl.col("kind") == kind)
        npv = part["p"].fill_null(np.nan).to_numpy()
        q = benjamini_hochberg(np.concatenate((rp.astype(np.float64), npv.astype(np.float64))))
        qs.append(part.with_columns(pl.Series("q_value", q[rp.size:]).fill_nan(None)))
    nic = pl.concat(qs)
    if not t.noise_nulls.is_empty():
        nic = nic.join(t.noise_nulls.select("feature", "target", "beyond_shift_null",
                                            "rank_ic0", "wrong_plus_60d"),
                       on=["feature", "target"], how="left")
    nic = _join_kind_null(nic, t.noise_kind_null)
    cons = t.noise_consistency.filter(pl.col("method") == "spearman").select(
        "feature", "target", "sign_consistency", "recent_ic", "era_sign_flips")
    nic = nic.join(cons, on=["feature", "target"], how="left").with_columns(
        pl.col("feature").str.replace(r"_\d+$", "").alias("noise_type"))
    ratio = a.nulls.slow_component_ratio
    statuses: dict[str, str] = {}
    for name, part in nic.group_by("feature"):
        kinds = []
        for kind in KINDS:
            kr = [dict(r, value=r["ic"], beyond_pipeline_null=None,
                       recent_same_sign=(None if r.get("recent_ic") is None or r.get("ic") is None
                                         else bool(np.sign(r["recent_ic"]) == np.sign(r["ic"]))),
                       slow_component=(None if r.get("wrong_plus_60d") is None
                                       or r.get("rank_ic0") is None
                                       else abs(r["wrong_plus_60d"]) >= ratio * abs(r["rank_ic0"])))
                  for r in part.filter(pl.col("kind") == kind).iter_rows(named=True)]
            if kr:
                kinds.append(_kind_evidence(kr, a))
        status, _, _ = classify(str(name[0]), kinds, live_safe=True, invalid_reason=None,
                                cluster_better=None, cfg=a.classification, fdr_q=fdr)
        statuses[str(name[0])] = status
    by_type = nic.group_by("noise_type").agg(
        pl.len().alias("tests"),
        (pl.col("p") < 0.05).mean().alias("share_p_below_0_05"),
        (pl.col("p") < 0.01).mean().alias("share_p_below_0_01"),
        (pl.col("q_value") < fdr).sum().alias("fdr_discoveries"),
        ((pl.col("q_value") < fdr) & pl.col("beyond_shift_null").fill_null(False)).sum()
        .alias("fdr_and_beyond_shift_null_per_test"),
        pl.col("beyond_shift_null").fill_null(False).mean().alias("share_beyond_shift_per_test"),
        pl.col("ic").abs().max().alias("max_abs_rank_ic")).sort("noise_type")
    kind_rate = (t.noise_kind_null.group_by("kind").agg(
        pl.col("beyond_kind_max_null").mean().alias("share_beyond_kind_max_null"),
        pl.len().alias("feature_kinds")).sort("kind").to_dicts()
        if not t.noise_kind_null.is_empty() else [])
    counts: dict[str, int] = {}
    for s in statuses.values():
        counts[s] = counts.get(s, 0) + 1
    return {"available": True, "features": len(statuses), "tests": nic.height,
            "status_counts": counts,
            "noise_features_with_candidate_status": sorted(
                f for f, s in statuses.items() if s in _CANDIDATES),
            "by_noise_type": by_type.to_dicts(),
            "kind_max_null_rate": kind_rate,
            "reading": ("share_p_below_0_05 ~ 0.05 means the batch-means p-values are calibrated "
                        "for that kind of noise; a larger share (persistent noise against "
                        "persistent targets) is why the circular-shift null is required on top "
                        "of FDR. A noise feature with a candidate status would be a false "
                        "discovery of the full procedure.")}


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------
def _definition(r: dict[str, Any]) -> str:
    what = {"rank_ic": "rank IC", "pearson_ic": "Pearson IC",
            "marginal_rank_ic": "rank IC with the one-bar outcome k bars ahead",
            "marginal_pearson_ic": "Pearson IC with the one-bar outcome k bars ahead",
            "conditional_rank_ic": "rank IC within a condition",
            "copula_mi": "copula mutual information (vs circular-shift null)",
            "interaction_oos_r2_gain": "out-of-sample R^2 gain of the interaction over its "
                                       "components"}.get(r["metric"], r["metric"])
    cond = "" if r["conditioning"] == "pooled" else f" | {r['conditioning']} = {r['condition']}"
    return f"{what} of {r['feature']} with {r['target']}{cond}"


def ledger_rows(t: Tables, tests: pl.DataFrame, a: AlphaConfig, registry_path: Path
                ) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Assign ALPHA-H ids (preregistered or exploratory) and build the ledger entries."""
    tf = t.timeframe
    specs = {r["name"]: r for r in t.registry}
    keys = [f"{tf}|{m}|{f}|{tg}|{c}|{cd}" for m, f, tg, c, cd in zip(
        tests["metric"], tests["feature"], tests["target"], tests["conditioning"],
        tests["condition"], strict=True)]
    prereg = []
    for m, f, tg, c, h in zip(tests["metric"], tests["feature"], tests["target"],
                              tests["conditioning"], tests["horizon"], strict=True):
        hid = None
        if m in ("rank_ic", "pearson_ic") and c == "pooled" and tg.startswith("target_"):
            hid = a.preregistration(f, _target_family(tg), int(h))
        prereg.append(hid)
    registry = TestRegistry(registry_path)
    families = tests["test_family"].to_list()
    ids: list[str | None] = [None] * len(keys)
    for flag in (True, False):
        idx = [i for i, p in enumerate(prereg) if (p is not None) == flag]
        assigned = registry.assign([keys[i] for i in idx], [families[i] for i in idx],
                                   preregistered=flag)
        for i, tid in zip(idx, assigned, strict=True):
            ids[i] = tid
    first = registry.preregistered([str(i) for i in ids])
    tests = tests.with_columns(pl.Series("test_id", ids), pl.Series("preregistered", first),
                               pl.Series("preregistration", prereg, dtype=pl.Utf8))
    version = t.manifest.get("factory_version")
    entries = []
    for r in tests.iter_rows(named=True):
        spec = specs.get(r["feature"], {})
        control = r.get("shift_null_q") if r["metric"] == "rank_ic" else None
        entries.append({
            "hypothesis_id": r["test_id"], "definition": _definition(r), "timeframe": tf,
            "input_series": "feature_matrix", "window": spec.get("window"),
            "feature": r["feature"], "target": r["target"], "horizon": r["horizon"],
            "metric": r["metric"], "value": r["value"], "control_value": control,
            "verdict": r["verdict"],
            "details": {k: r.get(k) for k in ("se", "q_timeframe", "family_tests",
                                              "bonferroni_pass", "preregistration",
                                              "beyond_shift_null", "beyond_pipeline_null")},
            "dataset_version": version, "study": f"feature_research/{tf}",
            "preregistered": r["preregistered"], "test_family": r["test_family"],
            "p_value": r["p_value"], "q_value": r["q_value"],
            "conditioning": ("pooled" if r["conditioning"] == "pooled"
                             else f"{r['conditioning']}={r['condition']}"),
            "feature_version": spec.get("feature_version"),
            "source_feed": spec.get("source_feed") or t.manifest.get("source_feed"),
            "n_obs": int(r["n_obs"]) if r.get("n_obs") is not None else None})
    return tests, entries


# ---------------------------------------------------------------------------
# Multiple-testing summary (and inputs for deflated statistics / PBO later)
# ---------------------------------------------------------------------------
def _effective_features(t: Tables) -> dict[str, Any]:
    """Nyholt / Li-Ji effective number of independent features from the |Spearman| matrix."""
    path = t.out_dir / "redundancy/cache/spearman.npy"
    if not path.exists():
        return {}
    r = np.load(path)
    r = np.nan_to_num(r, nan=0.0)
    np.fill_diagonal(r, 1.0)
    lam = np.clip(np.linalg.eigvalsh((r + r.T) / 2.0), 0.0, None)
    m = lam.size
    nyholt = 1.0 + (m - 1) * (1.0 - np.var(lam, ddof=1) / m) if m > 1 else float(m)
    li_ji = float(np.sum((lam >= 1.0) + (lam - np.floor(lam))))
    return {"features": m, "nyholt_meff": float(nyholt), "li_ji_meff": li_ji,
            "estimate": "from the evenly spaced redundancy sample"}


def multiple_testing_summary(t: Tables, tests: pl.DataFrame, a: AlphaConfig,
                             noise: dict[str, Any]) -> dict[str, Any]:
    fdr = a.multiple_testing.fdr_q
    fam = tests.group_by("test_family").agg(
        pl.len().alias("tests"), pl.col("p_value").is_not_null().sum().alias("with_p"),
        (pl.col("q_value") < fdr).sum().alias("bh_discoveries"),
        pl.col("bonferroni_pass").fill_null(False).sum().alias("bonferroni_passes"),
        pl.col("preregistered").sum().alias("preregistered"),
        pl.col("p_value").min().alias("min_p")).sort("test_family")
    write_table(fam, t.out_dir / "tests/family_summary")
    pooled = tests.filter((pl.col("metric") == "rank_ic") & (pl.col("conditioning") == "pooled")
                          & pl.col("target").str.starts_with("target_"))
    best = pooled.group_by("feature").agg(pl.col("value").abs().max().alias("best_abs_ic"))
    vals = best["best_abs_ic"].drop_nulls().to_numpy()
    months = (int(np.nanmax(t.ic["months"].to_numpy())) if "months" in t.ic.columns
              and t.ic.height else None)
    return {
        "timeframe": t.timeframe, "tests": tests.height,
        "tests_with_p": int(tests["p_value"].is_not_null().sum()),
        "families": fam.height,
        "preregistered_tests": int(tests["preregistered"].sum()),
        "exploratory_tests": int((~tests["preregistered"]).sum()),
        "bh_discoveries_within_family": int((tests["q_value"] < fdr).sum()),
        "bh_discoveries_across_timeframe": int((tests["q_timeframe"] < fdr).sum()),
        "bonferroni_passes": int(tests["bonferroni_pass"].fill_null(False).sum()),
        "by_metric": tests.group_by("metric").agg(
            pl.len().alias("tests"), (pl.col("q_value") < fdr).sum().alias("bh_discoveries")
        ).sort("metric").to_dicts(),
        "verdicts": tests.group_by("verdict").len().sort("verdict").to_dicts(),
        "effective_number_of_features": _effective_features(t),
        "selection_bias_inputs": {
            "what": "for a later deflated statistic / probability of backtest overfitting: "
                    "number of trials, the spread of the best |rank IC| per feature and the "
                    "effective number of independent features",
            "trials_pooled_rank_ic": pooled.height,
            "features": int(best.height),
            "best_abs_ic_mean": _clean(vals.mean()) if vals.size else None,
            "best_abs_ic_var": _clean(vals.var(ddof=1)) if vals.size > 1 else None,
            "best_abs_ic_max": _clean(vals.max()) if vals.size else None,
            "months": months},
        "noise_control": {k: noise.get(k) for k in ("features", "tests", "status_counts",
                                                    "noise_features_with_candidate_status")},
        "fdr_q": fdr, "bonferroni_alpha": a.multiple_testing.bonferroni_alpha,
    }


# ---------------------------------------------------------------------------
# One timeframe
# ---------------------------------------------------------------------------
def build_timeframe_report(fcfg: FeatureFactoryConfig, a: AlphaConfig, targets_path: Path,
                           timeframe: str, *, write_ledger: bool = True) -> dict[str, Any]:
    t = load_tables(fcfg, targets_path, timeframe)
    if t.ic.is_empty():
        raise RuntimeError(f"{timeframe}: no IC tables - run `xq feature-research` first")
    tests = add_q_values(collect_tests(t), bonferroni_alpha=a.multiple_testing.bonferroni_alpha)
    ev = evidence_table(t, tests, a)
    tests = _verdicts(tests, ev, a)
    tests, entries = ledger_rows(t, tests, a, fcfg.results_path / "test_registry.parquet")
    write_table(tests, t.out_dir / "tests/all_tests", csv=False)
    ev = ev.join(tests.select("feature", "target", "metric", "conditioning", "test_id",
                              "preregistered", "verdict"),
                 on=["feature", "target", "metric", "conditioning"], how="left")
    write_table(ev, t.out_dir / "feature_evidence", csv=False)
    rows, _ = classify_features(t, ev, a)
    write_table(pl.DataFrame(rows, infer_schema_length=None), t.out_dir / "feature_status")
    noise = noise_control(t, tests, a)
    write_json(t.out_dir / "noise_control.json", noise)
    mt = multiple_testing_summary(t, tests, a, noise)
    write_json(t.out_dir / "multiple_testing.json", mt)
    provenance = _provenance(t, a, fcfg)
    write_json(t.out_dir / "candidate_features.json",
               candidate_manifest(timeframe, rows, provenance))
    statuses = {r["feature"]: r["status"] for r in rows}
    write_json(t.out_dir / "ablation_manifests.json",
               ablation_sets(t.registry, statuses, dict(fcfg.ablation)))
    ledger_count = None
    if write_ledger:
        ledger_count = ResearchLedger(a.ledger_path).upsert(entries)
    summary = _summary(t, rows, mt, noise, provenance, ledger_count)
    if a.plots.enabled:
        from .feature_plots import write_feature_figures

        summary["figures"] = write_feature_figures(t.out_dir,
                                                   top_features=a.plots.top_features)
    write_json(t.out_dir / "summary.json", summary)
    (t.out_dir / "feature_research_summary.md").write_text(_markdown(t, rows, summary),
                                                         encoding="utf-8")
    return summary


def _provenance(t: Tables, a: AlphaConfig, fcfg: FeatureFactoryConfig) -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    lineage = t.manifest.get("dataset_lineage") or {}
    return {
        "generated_utc": utc_now_iso(), "timeframe": t.timeframe,
        "factory_version": t.manifest.get("factory_version"),
        "target_version": t.target_manifest.get("target_version"),
        "dataset_lineage": lineage,
        "partial": bool(lineage.get("partial")),
        "label": lineage.get("label"),
        "source_feed": fcfg.source_feed,
        "timestamp_convention": t.manifest.get("timestamp_convention"),
        "features_config": fcfg.fingerprint(), "alpha_config": a.fingerprint(),
        "stages": {k: v.get("stamp") for k, v in t.stages.items()},
        "code": code_fingerprint(sorted((root / "alpha").glob("*.py"))
                                 + sorted((root / "research").glob("feature_*.py"))),
        "git": git_info(a.project_root),
        "packages": package_versions(("numpy", "polars", "scipy", "pyarrow")),
    }


def _summary(t: Tables, rows: list[dict[str, Any]], mt: dict[str, Any], noise: dict[str, Any],
             provenance: dict[str, Any], ledger_count: int | None) -> dict[str, Any]:
    counts: dict[str, int] = {}
    fam_counts: dict[str, dict[str, int]] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        fc = fam_counts.setdefault(str(r["family"]), {})
        fc[r["status"]] = fc.get(r["status"], 0) + 1
    best_by_kind = {}
    for kind in KINDS:
        cand = [r for r in rows if r.get(f"{kind}_passes")]
        cand.sort(key=lambda r: -abs(r.get(f"{kind}_rank_ic") or 0.0))
        best_by_kind[kind] = [{"feature": r["feature"], "target": r.get(f"{kind}_target"),
                               "rank_ic": r.get(f"{kind}_rank_ic"), "status": r["status"]}
                              for r in cand[:8]]
    return {"timeframe": t.timeframe, "features": len(rows),
            "status_counts": dict(sorted(counts.items(), key=lambda kv: status_rank(kv[0]))),
            "status_by_family": fam_counts, "best_passing_by_kind": best_by_kind,
            "multiple_testing": {k: mt[k] for k in (
                "tests", "families", "preregistered_tests", "exploratory_tests",
                "bh_discoveries_within_family", "bonferroni_passes",
                "effective_number_of_features")},
            "noise_control": {k: noise.get(k) for k in ("status_counts",
                                                        "noise_features_with_candidate_status")},
            "ledger_rows_after_upsert": ledger_count, "provenance": provenance}


def _fmt(v: Any, spec: str = "+.4f") -> str:
    if v is None:
        return "-"
    try:
        return format(float(v), spec)
    except (TypeError, ValueError):
        return str(v)


def _markdown(t: Tables, rows: list[dict[str, Any]], s: dict[str, Any]) -> str:
    p = s["provenance"]
    lines = [f"# Feature research - {t.timeframe}", "",
             f"Factory `{p['factory_version']}`, targets `{p['target_version']}`, source feed "
             f"`{p['source_feed']}`. {p['timestamp_convention']}.", ""]
    if p.get("partial"):
        lines += [f"**{p.get('label')}**", ""]
    lines += ["## Statuses", "", "| status | features |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in s["status_counts"].items()]
    lines += ["", "## Candidates", "",
              "| feature | status | best kind | target | rank IC | q | yearly sign | eras flipping |",
              "|---|---|---|---|---:|---:|---:|---:|"]
    for r in rows:
        if r["status"] in _CANDIDATES:
            lines.append(f"| {r['feature']} | {r['status']} | {r['best_kind']} | "
                         f"{r['best_target']} | {_fmt(r['best_rank_ic'])} | "
                         f"{_fmt(r['best_q_value'], '.1e')} | "
                         f"{_fmt(r['sign_consistency'], '.0%')} | {r['era_sign_flips']} |")
    lines += ["", "## Best passing feature per outcome kind", ""]
    for kind, items in s["best_passing_by_kind"].items():
        shown = ", ".join(f"{i['feature']} ({_fmt(i['rank_ic'])}, {i['status']})"
                          for i in items[:5]) or "none"
        lines.append(f"- **{kind}**: {shown}")
    mt = s["multiple_testing"]
    lines += ["", "## Multiple testing", "",
              f"{mt['tests']:,} tests in {mt['families']} families "
              f"({mt['preregistered_tests']:,} preregistered, {mt['exploratory_tests']:,} "
              f"exploratory); {mt['bh_discoveries_within_family']:,} BH discoveries within "
              f"family, {mt['bonferroni_passes']:,} Bonferroni passes.",
              f"Noise features: {s['noise_control']}", "",
              "## Failed or vetoed (examples)", ""]
    for r in [r for r in rows if r["status"] in ("failed_null", "unstable")][:25]:
        lines.append(f"- `{r['feature']}` - {r['status']}: {r['reason']}")
    lines += ["", "Descriptive research only: no signal, no trade, no cost, no PnL.", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Across timeframes
# ---------------------------------------------------------------------------
def write_comparison(fcfg: FeatureFactoryConfig, timeframes: list[str]) -> dict[str, Any]:
    """Status and best-IC matrices across the timeframes that have a report."""
    frames = []
    for tf in timeframes:
        path = fcfg.results_path / tf / "feature_status.parquet"
        if path.exists():
            frames.append(pl.read_parquet(path).with_columns(pl.lit(tf).alias("timeframe")))
    if not frames:
        return {"timeframes": []}
    allr = pl.concat(frames, how="diagonal_relaxed")
    out = ensure_dir(fcfg.results_path / "comparison")
    status = allr.pivot(on="timeframe", index=["feature", "family"], values="status")
    write_table(status.sort(["family", "feature"]), out / "status_by_timeframe")
    ic = allr.pivot(on="timeframe", index=["feature", "family"], values="best_rank_ic")
    write_table(ic.sort(["family", "feature"]), out / "best_rank_ic_by_timeframe")
    counts = allr.group_by(["timeframe", "status"]).len().pivot(
        on="status", index="timeframe", values="len").fill_null(0)
    write_table(counts, out / "status_counts")
    per_kind = []
    for kind in KINDS:
        col = f"{kind}_passes"
        if col in allr.columns:
            per_kind.append(allr.group_by("timeframe").agg(
                pl.lit(kind).alias("kind"), pl.col(col).fill_null(False).sum().alias("features_passing")))
    if per_kind:
        write_table(pl.concat(per_kind), out / "passing_by_kind")
    everywhere = (allr.filter(pl.col("status").is_in(list(_CANDIDATES)))
                  .group_by("feature").agg(pl.col("timeframe").alias("timeframes"),
                                           pl.len().alias("n")).sort("n", descending=True))
    write_table(everywhere, out / "candidates_across_timeframes")
    summary = {"timeframes": sorted(allr["timeframe"].unique().to_list()),
               "status_counts": counts.to_dicts(),
               "candidates_in_every_timeframe": everywhere.filter(
                   pl.col("n") == len(frames))["feature"].to_list(),
               "generated_utc": utc_now_iso(), "status_order": list(STATUS_ORDER)}
    write_json(out / "comparison.json", clean_json(summary))
    return summary
