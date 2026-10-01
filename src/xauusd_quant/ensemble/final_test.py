r"""The one-time evaluation of frozen ensembles on the reserved period (Prompt #11,
Steps 62-65, 82).

The only ensemble code that reads reserved-period outcomes - through
:func:`xauusd_quant.ml.final_test.load_reserved_rows`, which itself refuses
anything that is not a frozen spec. It refuses to run without frozen,
hash-verified ``ENSEMBLE_SPEC`` files whose constituents load with the exact
spec versions and live feature manifests they were frozen with
(:class:`~.inference.EnsembleModel`), and every run is appended to
``final_test/access_log.jsonl``.

**A second look.** Prompt #10 already evaluated 37 single models on this period;
the access log's first line says so, every result carries the note, and nothing
here is tuned afterwards (Step 63: a failure is reported, not repaired).

For each frozen ensemble: the constituents' final models (fitted by
``xq ensemble-finalize`` on development rows only), the frozen combination and
calibrator, and beside them - in the same pass, on the same rows - the simple
average of the same constituents, the frozen best individual model and the
constant (the development training mean). Metrics overall, per year, per quarter
(flagged when thin), for the last twelve months and by regime / volatility /
spread / session; the ensemble's trailing weights (if dynamic) use only labels
resolved before each day's first bar.
"""

from __future__ import annotations

import gc
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl

from ..alpha.conditioning import causal_quantile_buckets
from ..ml.config import MLConfig
from ..ml.datasets import SESSION_ORDER, MLData, derive_target
from ..ml.final_test import load_reserved_rows, prior_evaluations
from ..ml.training import final_fold
from ..utils.clock import utc_now_iso
from ..utils.paths import ensure_dir
from .inference import EnsembleModel, current_manifests
from .registry import load_frozen_ensemble_spec
from .stability import period_table, score
from .weighting import dynamic_weights

__all__ = ["EnsembleFinalTestRefusedError", "SECOND_LOOK_NOTE", "run_ensemble_final_test",
           "slim_development_data"]

SECOND_LOOK_NOTE = ("second look at the reserved period: Prompt #10 evaluated 37 single-model "
                    "specs on it on 2026-10-01 08:30; the ensembles were frozen on development "
                    "data by pre-registered rules, and nothing is tuned after this evaluation")


class EnsembleFinalTestRefusedError(RuntimeError):
    """The ensemble final test was asked for without a frozen, loadable spec."""


def _midnight(d: Any) -> datetime:
    return datetime(d.year, d.month, d.day)


def slim_development_data(cfgs: Any, timeframe: str, horizons: list[int]) -> MLData:
    """The few development columns the final test needs - labels at *horizons*, the
    residual and trailing volatility, ``spread_rel`` and ``log_rv_20`` - read like
    :func:`~..ml.datasets.load_ml_data` reads them (rows before the reserved start, later
    year files never opened, outcomes purged at the reserved start by their horizon), but
    without the feature matrix (~1.2 GB at 5m that this test never uses)."""
    from ..features.factory import current_version_dir
    from ..features.store import RegressionFeatureStore
    from ..ml.datasets import MLIntegrityError, read_years

    reserved = cfgs.selection.periods.reserved_test.start
    fbase = current_version_dir(cfgs.features.factory_path, timeframe)
    registry = {r["name"]: r for r in json.loads((fbase / "registry.json").read_text(
        encoding="utf-8"))}
    cols = [c for c in ("spread_rel", "log_rv_20") if c in registry]
    frame = read_years(fbase, cols, reserved)
    stamps = frame["timestamp"]
    features = {c: frame[c].cast(pl.Float32).fill_null(np.nan).to_numpy() for c in cols}
    tptr = cfgs.targets.targets_path / f"timeframe={timeframe}" / "CURRENT"
    tbase = cfgs.targets.targets_path / f"timeframe={timeframe}" / \
        f"version={tptr.read_text().strip()}"
    tman = json.loads((tbase / "_manifest.json").read_text(encoding="utf-8"))
    tcols = [f"target_{fam}_{h}" for fam in ("abs_return", "realized_vol", "residual_reduction",
                                             "residual_shrinks", "return") for h in horizons
             if f"target_{fam}_{h}" in tman["targets"]]
    tframe = read_years(tbase, tcols, reserved)
    if not tframe["timestamp"].equals(stamps):
        raise MLIntegrityError(f"{timeframe}: targets and features are not on the same bars")
    n = int(stamps.len())
    raw = {}
    for c in tcols:
        h = int(c.rsplit("_", 1)[1])
        v = np.array(tframe[c].cast(pl.Float64).fill_null(np.nan).to_numpy(), dtype=np.float64)
        v[max(0, n - h):] = np.nan                     # outcome windows crossing the start
        raw[c] = v
    store = RegressionFeatureStore(cfgs.config, cfgs.regression)
    rframe = store.load(timeframe, cfgs.targets.regression_window,
                        ["timestamp", "residual", "trailing_volatility"])
    rframe = rframe.filter(pl.col("timestamp") < pl.lit(_midnight(reserved))
                           .cast(rframe["timestamp"].dtype))
    if not rframe["timestamp"].equals(stamps):
        raise MLIntegrityError(f"{timeframe}: regression store and matrix differ in bars")
    return MLData(timeframe=timeframe, timestamps=stamps, reserved_start=reserved,
                  features=features, manifests={}, manifest_ids={}, manifest_hashes={},
                  registry=registry, targets_raw=raw,
                  epsilon=rframe["residual"].cast(pl.Float64).fill_null(np.nan).to_numpy(),
                  sigma=rframe["trailing_volatility"].cast(pl.Float64).fill_null(np.nan)
                  .to_numpy(), versions={})


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _prior_starts(log: Path, hashes: set[str]) -> list[dict[str, Any]]:
    """Earlier ``final_test_started`` entries naming any of these spec hashes - a run that
    read reserved rows and stopped before logging ``evaluated`` was still a look."""
    if not log.exists():
        return []
    out = []
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if entry.get("event") == "final_test_started" and \
                hashes & set((entry.get("specs") or {}).values()):
            out.append(entry)
    return out


def _check_frozen_config(spec: dict[str, Any], cfg: MLConfig, ecfg: Any) -> None:
    """Refuse a live configuration that would compute other labels, folds or constants
    than the ones the ensemble was frozen with."""
    td = spec.get("target_definition") or {}
    tspec = cfg.targets.get(spec["target"])
    if tspec is None:
        raise EnsembleFinalTestRefusedError(f"{spec['spec_id']}: target {spec['target']} is no "
                                            "longer configured")
    for key in ("source", "task", "kind", "transform", "c", "cost_multiple"):
        if key in td and td[key] != getattr(tspec, key):
            raise EnsembleFinalTestRefusedError(
                f"{spec['spec_id']}: target {key} is {getattr(tspec, key)!r} now, "
                f"{td[key]!r} when frozen")
    frozen_ml = spec.get("ml_config_fingerprint")
    if frozen_ml is not None and frozen_ml != cfg.fingerprint():
        raise EnsembleFinalTestRefusedError(f"{spec['spec_id']}: config/ml.yaml changed since the "
                                            "ensemble was frozen")
    frozen_ens = spec.get("ensemble_config_fingerprint")
    if frozen_ens is not None and hasattr(ecfg, "fingerprint") \
            and frozen_ens != ecfg.fingerprint():
        raise EnsembleFinalTestRefusedError(f"{spec['spec_id']}: config/ensemble.yaml changed "
                                            "since the ensemble was frozen")


def _context(model: EnsembleModel, dev: MLData, res: MLData, pair_oos: dict[str, Any] | None,
             p_res: np.ndarray, y_res: np.ndarray, base: float) -> dict[str, np.ndarray] | None:
    names = model.required_context()
    if not names:
        return None
    ctx: dict[str, np.ndarray] = {}
    for n in names:
        if n == "vol_quartile":
            # causal quartiles of log_rv_20 over the whole expanding history (development
            # then reserved bars), exactly as the development context was built
            v = np.concatenate([dev.features["log_rv_20"], res.features["log_rv_20"]])
            ts = pl.concat([dev.timestamps, res.timestamps])
            per_day = max(1, int(round(dev.n / max(1, dev.timestamps.dt.date().n_unique()))))
            q = causal_quantile_buckets(v.astype(np.float64), ts, 4, min_history=250 * per_day)
            ctx[n] = q[dev.n:].astype(np.float64)
        elif n == "dynamic_weights":
            if pair_oos is None:
                raise EnsembleFinalTestRefusedError("dynamic weights need the development "
                                                    "out-of-sample rows")
            comb = model.combination
            p_all = np.vstack([pair_oos["p"], p_res])
            y_all = np.concatenate([pair_oos["y"], y_res])
            b_all = np.concatenate([pair_oos["base"], np.full(y_res.size, base)])
            rows_all = np.concatenate([pair_oos["rows"], dev.n + np.arange(y_res.size)])
            ts = pl.concat([pair_oos["timestamps"], res.timestamps])
            _, day = np.unique(ts.dt.date().to_numpy(), return_inverse=True)
            # a row without a prediction or a label keeps its place in time and adds no loss;
            # a reserved label enters only once it has resolved (row + h < the day's first bar)
            evaluate = np.arange(pair_oos["y"].size, y_all.size)
            dw = dynamic_weights(p_all, y_all, b_all, rows_all, day.astype(np.int64), evaluate,
                                 task=model.task, horizon=int(model.spec["horizon"]),
                                 window_days=int(comb["window_days"]),
                                 min_rows=int(comb["min_rows"]), shrink=float(comb["shrink"]))
            ctx[n] = dw.row_weights
        elif n in res.features:
            ctx[n] = res.features[n].astype(np.float64)
        else:
            raise EnsembleFinalTestRefusedError(f"{model.spec_id}: no reserved-period input "
                                                f"{n!r}")
    return ctx


def run_ensemble_final_test(cfgs: Any, cfg: MLConfig, ecfg: Any, dev: MLData,
                            spec_paths: list[Path], *, out_dir: Path,
                            repeat_reason: str | None = None,
                            pair_oos: dict[str, dict[str, Any]] | None = None,
                            finalized: dict[str, dict[str, Any]] | None = None
                            ) -> list[dict[str, Any]]:
    """Evaluate each frozen ensemble once on the reserved period (and log it).

    Everything that can refuse does so before the first reserved row is read: unfrozen or
    edited specs, an earlier look (an ``evaluated`` *or* an interrupted
    ``final_test_started`` entry) without a logged reason, a live configuration that
    differs from the frozen one, a constituent of another version or feature manifest, an
    artifact manifest changed since ``ensemble-finalize``, a missing best individual.
    """
    specs = [load_frozen_ensemble_spec(p) for p in spec_paths]   # refuses unfrozen / edited
    out_dir = ensure_dir(out_dir)
    log = out_dir / "access_log.jsonl"
    hashes = {s["content_hash"] for s in specs}
    prior = prior_evaluations(log, hashes) + _prior_starts(log, hashes)
    if prior and not repeat_reason:
        seen = sorted({str(p.get("spec_id") or p.get("utc")) for p in prior})
        raise EnsembleFinalTestRefusedError(
            f"already looked at the reserved period: {seen}. The final test is one-time; a "
            "repeat needs an explicit reason, which is logged")
    for s in specs:
        _check_frozen_config(s, cfg, ecfg)
        if s.get("companions", {}).get("best_individual") not in [c["name"] for c in
                                                                  s["constituents"]]:
            raise EnsembleFinalTestRefusedError(f"{s['spec_id']}: the frozen best individual is "
                                                "not one of its constituents")
    manifests = current_manifests(cfgs.selection.results_path / dev.timeframe / "manifests")

    def load(p: Path, s: dict[str, Any]) -> EnsembleModel:
        try:
            return EnsembleModel.load(p, models_path=ecfg.models_path,
                                      constituent_dir=p.parent.parent / "constituents",
                                      live_manifests=manifests)
        except Exception as exc:
            raise EnsembleFinalTestRefusedError(f"{s['spec_id']}: {exc}") from exc

    # every spec must load (versions, feature manifests, artifact hashes) before the first
    # reserved row is read; the models are then reloaded one at a time (memory)
    for p, s in zip(spec_paths, specs, strict=True):
        load(p, s)                                        # refuses on any mismatch
        recorded =((finalized or {}).get(s["spec_id"]) or {}).get("constituent_manifests") or {}
        for c in s["constituents"]:
            want = recorded.get(c["model_id"])
            if want is not None and _sha256(ecfg.models_path / c["model_id"] / "manifest.json") \
                    != want:
                raise EnsembleFinalTestRefusedError(
                    f"{s['spec_id']}: the artifact manifest of {c['model_id']} changed after "
                    "ensemble-finalize")
    with log.open("a", encoding="utf-8") as fh:
        if prior:
            fh.write(json.dumps({"utc": utc_now_iso(), "event": "repeat_authorised",
                                 "reason": repeat_reason,
                                 "previous": [x.get("utc") for x in prior]}) + "\n")
        fh.write(json.dumps({"utc": utc_now_iso(), "event": "final_test_started",
                             "note": SECOND_LOOK_NOTE,
                             "specs": {s["spec_id"]: s["content_hash"] for s in specs}}) + "\n")
    results = []
    for spec_path, spec in zip(spec_paths, specs, strict=True):
        model = load(spec_path, spec)
        tspec = cfg.targets[spec["target"]]
        h = int(spec["horizon"])
        feats = model.required_features()
        ctx_feats = [n for n in model.required_context() if n not in ("vol_quartile",
                                                                      "dynamic_weights")]
        if "vol_quartile" in model.required_context():
            ctx_feats.append("log_rv_20")             # the causal buckets are built from it
        pseudo = {"frozen": True, "features": sorted(set(feats) | set(ctx_feats)), "horizon": h,
                  "feature_set_id": f"ENSEMBLE:{spec['spec_id']}",
                  "feature_set_hash": spec["content_hash"]}
        res = load_reserved_rows(cfgs, dev.timeframe, pseudo)        # the one reserved read
        ta = derive_target(tspec, h, res.targets_raw, epsilon=res.epsilon, sigma=res.sigma,
                           spread_rel=res.features.get("spread_rel"), log_floor=cfg.log_floor)
        fold = final_fold(dev, cfg, horizon=h, scheme="expanding",
                          rolling_years=cfg.walk_forward.rolling_years)
        y_dev = dev.target(tspec, h, cfg.log_floor).y
        fit = y_dev[fold.fit[0]:fold.fit[1]]
        base = float(np.nanmean(fit))                               # the constant
        inner = y_dev[fold.inner[0]:fold.inner[1]]
        recent_base = float(np.nanmean(inner)) if np.isfinite(inner).any() else None
        x = {f: res.features[f] for f in feats}
        p_members = model.constituent_matrix(x)
        ctx = _context(model, dev, res, (pair_oos or {}).get(spec["spec_id"]), p_members,
                       ta.y, base)
        out = model.predict_batch(x, ctx)
        pred = out["prediction"]
        names = model.names
        series = {"ensemble": pred, "simple_average": p_members.mean(axis=1),
                  "constant": np.full(res.n, base)}
        best = spec["companions"]["best_individual"]
        series["best_individual"] = p_members[:, names.index(best)]
        for j, nm in enumerate(names):
            series[f"constituent:{nm}"] = p_members[:, j]
        # one set of rows for every series compared: a labelled bar where the ensemble is
        # ENSEMBLE_INVALID is left out for all of them (and counted), never for one only
        labelled = np.isfinite(ta.y)
        ok = labelled & np.isfinite(pred)
        excluded_invalid = int((labelled & ~np.isfinite(pred)).sum())
        bvec = np.full(res.n, base)
        overall = {k: score(tspec.task, v[ok], ta.y[ok], bvec[ok], ta.y_raw[ok])
                   for k, v in series.items()}
        vs_recent = ({k: score(tspec.task, v[ok], ta.y[ok], np.full(int(ok.sum()), recent_base),
                               ta.y_raw[ok]) for k, v in series.items()}
                     if recent_base is not None else {})
        years = res.timestamps.dt.year().to_numpy()
        quarters = (res.timestamps.dt.year().cast(pl.Int64) * 10
                    + res.timestamps.dt.quarter().cast(pl.Int64)).to_numpy()
        main = {k: v for k, v in series.items() if not k.startswith("constituent:")}
        by_year = period_table(tspec.task, {k: v[ok] for k, v in series.items()}, ta.y[ok],
                               bvec[ok], years[ok], min_rows=int(ecfg.stability.get(
                                   "min_rows_year", 5000)), y_raw=ta.y_raw[ok])
        by_quarter = period_table(tspec.task, {k: v[ok] for k, v in main.items()}, ta.y[ok],
                                  bvec[ok], quarters[ok], min_rows=int(ecfg.stability.get(
                                      "min_rows_quarter", 3000)), y_raw=ta.y_raw[ok])
        last = cast(datetime, res.timestamps.max())
        recent = ok & (res.timestamps >= last - timedelta(days=365)).to_numpy()
        recent_12m = {k: score(tspec.task, v[recent], ta.y[recent], bvec[recent],
                               ta.y_raw[recent]) for k, v in main.items()}
        groups = []
        labels_by = {"session": {i: s.removeprefix("session_") for i, s in
                                 enumerate(SESSION_ORDER)}}
        vq = ctx.get("vol_quartile") if ctx else None
        gctx = dict(res.context)
        if vq is not None:
            gctx["vol_quartile"] = np.where(np.isfinite(vq), vq, -1).astype(np.int64)
        for g in ("regime_state", "vol_quartile", "spread_quartile", "session"):
            if g in gctx:
                for r in period_table(tspec.task, {k: v[ok] for k, v in main.items()},
                                      ta.y[ok], bvec[ok], np.asarray(gctx[g])[ok],
                                      labels=labels_by.get(g), min_rows=2000,
                                      y_raw=ta.y_raw[ok]):
                    groups.append({"grouping": g, **r})
        metric = "log_loss_skill" if tspec.is_classification else "rank_ic"
        yearly: dict[str, list[float]] = {r["series"]: [] for r in by_year}
        for r in by_year:
            v = r.get(metric)
            if v is not None and np.isfinite(v):
                yearly[r["series"]].append(float(v))
        step64 = {}
        for k in ("ensemble", "simple_average", "best_individual"):
            if k not in overall:
                continue
            yv = yearly.get(k) or []
            step64[k] = {"overall": overall[k].get(metric),
                         "worst_year": min(yv) if yv else None,
                         "year_sd": float(np.std(yv, ddof=1)) if len(yv) > 1 else None,
                         "ece": overall[k].get("ece"),
                         "recent_12_months": recent_12m[k].get(metric)}
        audit = pl.DataFrame({
            "timestamp": res.timestamps, "ensemble_id": spec["spec_id"],
            "target": spec["target"], "horizon": h, "prediction": pred.astype(np.float32),
            "status": out["status"], "disagreement": out["disagreement"].astype(np.float32),
            "simple_average": series["simple_average"].astype(np.float32),
            "best_individual": series["best_individual"].astype(np.float32),
            "label": ta.y.astype(np.float32), "fold": "final_test",
            "dataset_version": (spec.get("dataset_versions") or {}).get("tick_dataset_version"),
            "note": SECOND_LOOK_NOTE})
        audit.write_parquet(out_dir / f"{spec['spec_id']}_predictions.parquet")
        fin = (finalized or {}).get(spec["spec_id"], {})
        dev_ev = (spec.get("development") or {}).get("frozen_series") or {}
        summary = {
            "ensemble_id": spec["spec_id"], "target": spec["target"], "horizon": h,
            "timeframe": spec["timeframe"], "method": spec["method"], "role": spec["role"],
            "constituent_models": [c["model_id"] for c in spec["constituents"]],
            "development_metric": dev_ev.get("metric"), "development_mean": dev_ev.get("mean"),
            "development_worst_block": dev_ev.get("worst_block"),
            "development_auc": dev_ev.get("mean_auc"),
            "development_brier_skill": dev_ev.get("mean_brier_skill"),
            "development_rank_ic": dev_ev.get("mean_rank_ic"),
            "final_test_metric": overall["ensemble"].get(metric),
            "final_test_auc": overall["ensemble"].get("auc"),
            "final_test_brier": overall["ensemble"].get("brier"),
            "final_test_ece": overall["ensemble"].get("ece"),
            "final_test_rank_ic": overall["ensemble"].get("rank_ic"),
            "final_test_minus_best_individual": (
                None if "best_individual" not in overall or overall["best_individual"].get(
                    metric) is None or overall["ensemble"].get(metric) is None
                else overall["ensemble"][metric] - overall["best_individual"][metric]),
            "worst_year_metric": step64.get("ensemble", {}).get("worst_year"),
            "streaming_safe": fin.get("streaming_safe"),
            "inference_latency_ms": fin.get("latency_ms_per_row"),
            "invalid_rows": int((out["status"] != "ok").sum()),
            "labelled_rows_excluded_as_invalid": excluded_invalid, "rows": int(ok.sum()),
            "note": SECOND_LOOK_NOTE}
        result = {"spec_id": spec["spec_id"], "spec_hash": spec["content_hash"],
                  "first": str(res.timestamps[0]), "last": str(res.timestamps[-1]),
                  "rows": int(ok.sum()), "base": base, "recent_base": recent_base,
                  "overall": overall, "overall_vs_recent_base": {
                      k: {m: v.get(m) for m in ("log_loss_skill", "brier_skill", "mse_skill")}
                      for k, v in vs_recent.items()},
                  "step64": step64, "by_year": by_year, "by_quarter": by_quarter,
                  "recent_12_months": recent_12m, "by_group": groups, "summary": summary,
                  "evaluated_utc": utc_now_iso(), "note": SECOND_LOOK_NOTE}
        (out_dir / f"{spec['spec_id']}.json").write_text(
            json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8")
        results.append(result)
        with log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"utc": utc_now_iso(), "event": "evaluated",
                                 "spec_id": spec["spec_id"], "spec_hash": spec["content_hash"],
                                 "rows": int(ok.sum())}) + "\n")
        del model, res, out, p_members, series, x, ctx, audit       # one ensemble at a time
        gc.collect()
    path = out_dir / "ensemble_summaries.json"
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    current.update({r["spec_id"]: r["summary"] for r in results})
    path.write_text(json.dumps(current, indent=1, default=str) + "\n", encoding="utf-8")
    return results
