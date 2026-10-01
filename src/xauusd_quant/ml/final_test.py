r"""The one-time evaluation of frozen models on the reserved test period (Prompt #10,
Steps 75-79).

This is the **only** code that reads reserved-period outcomes. It refuses to run
without a frozen, hash-verified model spec (:func:`~.registry.load_frozen_spec`),
and every run is appended to ``final_test/access_log.jsonl`` with the spec
hashes, so a second look at the final test is visible forever.

For each frozen spec: fit the model on its development training window
(:func:`~.training.train_final`; no reserved row is loaded at that point),
save the artifact and check that the reloaded model reproduces its
predictions, then load the reserved rows, predict, and report the development
metrics on them - overall, per calendar year, for the most recent twelve months
and by regime / volatility / session. Nothing is tuned here.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl

from ..features.factory import current_version_dir
from ..features.store import RegressionFeatureStore
from ..utils.clock import utc_now_iso
from ..utils.paths import ensure_dir
from .config import MLConfig, TargetSpec
from .datasets import (
    CONTEXT_FEATURES,
    SESSION_ORDER,
    MLData,
    MLIntegrityError,
    build_context,
    derive_target,
)
from .evaluation import classification_metrics, grouped_metrics, regression_metrics
from .registry import load_artifact, load_frozen_spec, save_artifact
from .training import final_fold, train_final

__all__ = ["FinalTestRefusedError", "load_reserved_rows", "model_summary", "prior_evaluations",
           "recent_base", "run_final_test"]


class FinalTestRefusedError(RuntimeError):
    """The final test was asked for without a frozen specification."""


def _midnight(d: date) -> datetime:
    return datetime(d.year, d.month, d.day)


def load_reserved_rows(cfgs: Any, timeframe: str, spec: dict[str, Any]) -> MLData:
    """Features, targets and context of the reserved period - only for a frozen spec."""
    if spec.get("frozen") is not True:
        raise FinalTestRefusedError("the reserved period is read only for a frozen model spec")
    reserved = cfgs.selection.periods.reserved_test.start
    fcfg, tcfg = cfgs.features, cfgs.targets
    base = current_version_dir(fcfg.factory_path, timeframe)
    registry = {r["name"]: r for r in json.loads((base / "registry.json").read_text(
        encoding="utf-8"))}
    wanted = sorted(set(spec["features"]) | {c for c in CONTEXT_FEATURES if c in registry})
    files = [p for p in sorted(base.glob("year=*/part-0.parquet"))
             if int(p.parent.name.split("=")[1]) >= reserved.year]
    frame = (pl.scan_parquet(files).select("timestamp", *wanted)
             .filter(pl.col("timestamp") >= pl.lit(_midnight(reserved))).collect()
             .sort("timestamp"))
    stamps = frame["timestamp"]
    features = {c: frame[c].cast(pl.Float32).fill_null(np.nan).to_numpy() for c in wanted}
    tptr = tcfg.targets_path / f"timeframe={timeframe}" / "CURRENT"
    tbase = tcfg.targets_path / f"timeframe={timeframe}" / f"version={tptr.read_text().strip()}"
    h = int(spec["horizon"])
    tfiles = [p for p in sorted(tbase.glob("year=*/part-0.parquet"))
              if int(p.parent.name.split("=")[1]) >= reserved.year]
    stored = set(pl.read_parquet_schema(tfiles[0]))
    tcols = [c for c in (f"target_{fam}_{h}" for fam in (
        "return", "abs_return", "realized_vol", "residual_reduction", "residual_shrinks"))
        if c in stored]
    tframe = (pl.scan_parquet(tfiles).select("timestamp", *tcols)
              .filter(pl.col("timestamp") >= pl.lit(_midnight(reserved))).collect()
              .sort("timestamp"))
    if not tframe["timestamp"].equals(stamps):
        raise MLIntegrityError("reserved targets and features are not on the same bars")
    targets_raw = {c: tframe[c].cast(pl.Float64).fill_null(np.nan).to_numpy() for c in tcols}
    store = RegressionFeatureStore(cfgs.config, cfgs.regression)
    rframe = store.load(timeframe, tcfg.regression_window,
                        ["timestamp", "residual", "trailing_volatility"])
    rframe = rframe.filter(pl.col("timestamp") >= pl.lit(_midnight(reserved))
                           .cast(rframe["timestamp"].dtype))
    if not rframe["timestamp"].equals(stamps):
        raise MLIntegrityError("reserved regression rows and features differ")
    data = MLData(timeframe=timeframe, timestamps=stamps, reserved_start=reserved,
                  features=features, manifests={"frozen": list(spec["features"])},
                  manifest_ids={"frozen": spec["feature_set_id"]},
                  manifest_hashes={"frozen": spec["feature_set_hash"]}, registry=registry,
                  targets_raw=targets_raw,
                  epsilon=rframe["residual"].cast(pl.Float64).fill_null(np.nan).to_numpy(),
                  sigma=rframe["trailing_volatility"].cast(pl.Float64).fill_null(np.nan)
                  .to_numpy(),
                  versions={})
    data.context = build_context(data)
    return data


def _metrics(task: str, pred: np.ndarray, y: np.ndarray, y_raw: np.ndarray, base: float
             ) -> dict[str, Any]:
    if task == "classification":
        return classification_metrics(pred, y, base_rate=base)
    return regression_metrics(pred, y, y_raw=y_raw, base_value=base)


def recent_base(dev: MLData, spec: dict[str, Any], tspec: TargetSpec, cfg: MLConfig
                ) -> float | None:
    """Mean label of the final model's inner slice (the latest 10 % of development).

    The skills are measured against the training mean. For a label whose base rate
    drifts (the one-spread direction labels move with spreads and volatility), a
    constant at the recent rate earns skill without any feature, so the final test
    also reports each model against it. Development rows only.
    """
    policy = spec.get("training_policy") or {}
    fold = final_fold(dev, cfg, horizon=int(spec["horizon"]),
                      scheme=str(policy.get("scheme", "expanding")),
                      rolling_years=int(policy.get("rolling_years", 5)))
    y = dev.target(tspec, int(spec["horizon"]), cfg.log_floor).y[fold.inner[0]:fold.inner[1]]
    ok = np.isfinite(y)
    return float(y[ok].mean()) if ok.any() else None


def prior_evaluations(log: Path, hashes: set[str]) -> list[dict[str, Any]]:
    """Earlier ``evaluated`` entries of the access log for any of these spec hashes."""
    if not log.exists():
        return []
    out = []
    for line in log.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if entry.get("event") == "evaluated" and entry.get("spec_hash") in hashes:
            out.append(entry)
    return out


def model_summary(spec: dict[str, Any], result: dict[str, Any],
                  finalized: dict[str, Any] | None) -> dict[str, Any]:
    """The Step 100 summary of one frozen model: development vs final-test metrics."""
    dev = spec.get("development") or {}
    ov = result.get("overall") or {}
    fin = finalized or {}
    policy = spec.get("training_policy") or {}
    size = fin.get("model_size_bytes")
    return {
        "model_id": spec["spec_id"], "role": spec.get("role"), "model_family": spec["family"],
        "target": spec["target"], "horizon": int(spec["horizon"]),
        "timeframe": spec["timeframe"], "feature_set": spec["feature_set_id"],
        "training_policy": str(policy.get("scheme")) + (f", half-life {policy['weighting']} y"
                                                        if policy.get("weighting") else ""),
        "development_auc": dev.get("mean_auc"), "development_brier": dev.get("mean_brier"),
        "development_log_loss_skill": dev.get("mean_log_loss_skill"),
        "development_rank_ic": dev.get("mean_rank_ic"),
        "development_mse_skill": dev.get("mean_mse_skill"),
        "final_test_auc": ov.get("auc"), "final_test_brier": ov.get("brier"),
        "final_test_log_loss_skill": ov.get("log_loss_skill"),
        "final_test_rank_ic": ov.get("rank_ic"), "final_test_mse_skill": ov.get("mse_skill"),
        "final_test_r2": ov.get("r2"), "final_test_rows": result.get("rows"),
        "final_test_skill_vs_recent_base": (result.get("overall_vs_recent_base") or {}).get(
            "log_loss_skill" if spec.get("target_definition", {}).get("task") == "classification"
            else "mse_skill"),
        "calibrated": spec.get("calibration") not in (None, "none"),
        "calibration": spec.get("calibration"),
        "streaming_safe": fin.get("streaming_safe"),
        "model_size_mb": None if size is None else float(size) / 1e6,
        "inference_latency_ms": fin.get("latency_ms_per_row")}


def run_final_test(cfgs: Any, cfg: MLConfig, dev: MLData, spec_paths: list[Path], *,
                   out_dir: Path, repeat_reason: str | None = None) -> list[dict[str, Any]]:
    """Fit, save, verify and evaluate each frozen spec once on the reserved period.

    A spec already evaluated (its hash in the access log) is refused unless a
    *repeat_reason* is given; the reason is logged beside the second look.
    """
    specs = [load_frozen_spec(p) for p in spec_paths]       # refuses unfrozen / edited specs
    out_dir = ensure_dir(out_dir)
    log = out_dir / "access_log.jsonl"
    prior = prior_evaluations(log, {s["content_hash"] for s in specs})
    if prior and not repeat_reason:
        seen = sorted({str(p.get("spec_id")) for p in prior})
        raise FinalTestRefusedError(f"already evaluated on the reserved period: {seen}. The final "
                               "test is one-time; a repeat needs an explicit reason, which is "
                               "logged")
    with log.open("a", encoding="utf-8") as fh:
        if prior:
            fh.write(json.dumps({"utc": utc_now_iso(), "event": "repeat_authorised",
                                 "reason": repeat_reason,
                                 "previous": [p.get("utc") for p in prior]}) + "\n")
        fh.write(json.dumps({"utc": utc_now_iso(), "event": "final_test_started",
                             "specs": {s["spec_id"]: s["content_hash"] for s in specs}}) + "\n")
    finalized: dict[str, dict[str, Any]] = {}
    fin_path = out_dir.parent / "frozen" / "finalize.json"
    if fin_path.exists():
        finalized = {m["model_id"]: m for m in json.loads(fin_path.read_text(
            encoding="utf-8")).get("models", [])}
    results = []
    for spec in specs:
        tspec: TargetSpec = cfg.targets[spec["target"]]
        art_dir = cfg.models_path / spec["spec_id"]
        if (art_dir / "manifest.json").exists():
            model, pre, cal, man = load_artifact(art_dir)          # file hashes verified
            if man.get("spec_hash") != spec["content_hash"]:
                raise FinalTestRefusedError(f"{spec['spec_id']}: the artifact was trained from "
                                       "another spec")
            info = dict(man.get("training") or {})
        else:
            model, pre, cal, info = train_final(dev, spec, tspec, cfg)
            save_artifact(art_dir, model=model, preprocessor=pre, calibrator=cal, spec=spec,
                          extra={"training": info, "training_policy": spec["training_policy"]})
        # reload equality on development rows before anything reserved is touched
        check_rows = np.linspace(0, dev.n - 1, 2000).astype(np.int64)
        xd = dev.design(list(spec["features"]))[check_rows]
        first = cal.apply(model.predict(pre.transform(xd, spec["features"])))
        m2, p2, c2, _ = load_artifact(art_dir)
        again = c2.apply(m2.predict(p2.transform(xd, spec["features"])))
        reload_ok = bool(np.allclose(first, again, rtol=1e-7, atol=1e-9, equal_nan=True))
        rbase = recent_base(dev, spec, tspec, cfg)                  # development rows only
        res = load_reserved_rows(cfgs, dev.timeframe, spec)
        ta = derive_target(tspec, int(spec["horizon"]), res.targets_raw, epsilon=res.epsilon,
                           sigma=res.sigma, spread_rel=res.features.get("spread_rel"),
                           log_floor=cfg.log_floor)
        x = np.column_stack([res.features[f] for f in spec["features"]]).astype(np.float32)
        raw = model.predict(pre.transform(x, spec["features"]))
        pred = cal.apply(raw) if tspec.is_classification else raw
        years = res.timestamps.dt.year().to_numpy()
        ok = np.isfinite(ta.y)
        overall = _metrics(tspec.task, pred[ok], ta.y[ok], ta.y_raw[ok], info["base"])
        vs_recent: dict[str, Any] = {}
        if rbase is not None:
            m_recent = _metrics(tspec.task, pred[ok], ta.y[ok], ta.y_raw[ok], rbase)
            vs_recent = {k: m_recent.get(k) for k in ("log_loss_skill", "brier_skill",
                                                      "mse_skill")}
        by_year = {}
        for yr in np.unique(years):
            m = ok & (years == yr)
            by_year[int(yr)] = _metrics(tspec.task, pred[m], ta.y[m], ta.y_raw[m], info["base"])
        last = cast(datetime, res.timestamps.max())
        cutoff = last - timedelta(days=365)
        recent_mask = ok & (res.timestamps >= cutoff).to_numpy()
        recent = _metrics(tspec.task, pred[recent_mask], ta.y[recent_mask],
                          ta.y_raw[recent_mask], info["base"])
        groups = {}
        for key in ("regime_state", "vol_quartile", "session"):
            if key in res.context:
                labels = ({i: s.removeprefix("session_") for i, s in enumerate(SESSION_ORDER)}
                          if key == "session" else None)
                groups[key] = grouped_metrics(pred[ok], ta.y[ok], res.context[key][ok],
                                              task=tspec.task, base=info["base"],
                                              y_raw=ta.y_raw[ok], labels=labels)
        audit = pl.DataFrame({
            "timestamp": res.timestamps, "model_id": spec["spec_id"], "target": spec["target"],
            "horizon": int(spec["horizon"]), "prediction": raw.astype(np.float32),
            "calibrated": pred.astype(np.float32), "label": ta.y.astype(np.float32),
            "fold": "final_test", "feature_set_id": spec["feature_set_id"],
            "dataset_version": (spec.get("dataset_versions") or {}).get("tick_dataset_version")})
        audit.write_parquet(out_dir / f"{spec['spec_id']}_predictions.parquet")
        result = {"model_id": spec["spec_id"], "spec_hash": spec["content_hash"],
                  "family": spec["family"], "target": spec["target"],
                  "horizon": int(spec["horizon"]), "feature_set": spec["feature_set_id"],
                  "training": info, "reload_equal": reload_ok,
                  "rows": int(ok.sum()), "first": str(res.timestamps[0]),
                  "last": str(res.timestamps[-1]), "overall": overall,
                  "recent_base": rbase, "overall_vs_recent_base": vs_recent, "by_year": by_year,
                  "recent_12_months": recent, "by_group": groups,
                  "evaluated_utc": utc_now_iso()}
        result["summary"] = model_summary(spec, result, finalized.get(spec["spec_id"]))
        (out_dir / f"{spec['spec_id']}.json").write_text(json.dumps(result, indent=1,
                                                                   default=str) + "\n",
                                                        encoding="utf-8")
        results.append(result)
        with log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"utc": utc_now_iso(), "event": "evaluated",
                                 "spec_id": spec["spec_id"], "spec_hash": spec["content_hash"],
                                 "rows": int(ok.sum())}) + "\n")
    summaries_path = out_dir / "model_summaries.json"
    summaries = (json.loads(summaries_path.read_text(encoding="utf-8"))
                 if summaries_path.exists() else {})
    summaries.update({r["model_id"]: r["summary"] for r in results})
    summaries_path.write_text(json.dumps(summaries, indent=1, default=str) + "\n",
                              encoding="utf-8")
    return results
