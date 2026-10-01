r"""Freezing ensembles, fitting their constituents, live checks and benchmark (Prompt
#11, Steps 59-61, 66-72).

:func:`freeze_ensembles` reads the development results only (each pair's
``freeze_decision.json``, written by the pre-registered rule) and writes, per
target x horizon with an ensemble universe:

* one ``MODEL_SPEC`` per constituent (``results/ensemble_research/<tf>/
  constituents``): the Prompt #10 base configuration that produced the
  out-of-sample predictions the ensemble was studied on - family defaults,
  manifest features, fold-fitted preprocessing, Platt calibration for
  probabilities, expanding training to the reserved start;
* one immutable ``ENSEMBLE_SPEC`` (``<tf>/frozen``): constituents and their spec
  hashes, the combination and its parameters **fitted on every out-of-sample row
  of 2011-2021** (weights, stacked meta-model, state weights, the trailing-weight
  rule), the final calibrator chosen by the registered rule, the companions the
  final test evaluates beside it (simple average, best individual, constant), the
  training and refit policy and the development evidence. A single model retained
  by the rule still freezes the simple average, labelled *benchmark*.

:func:`finalize_ensembles` then fits every constituent on the whole development
span (no reserved row exists in memory), saves and reloads it, loads each
ensemble through :class:`~..ensemble.inference.EnsembleModel` (spec hashes, model
versions and live feature manifests verified), checks batch = reload = streaming,
and measures loading memory, latency and artifact size - all before
``xq ensemble-final-test`` may touch the reserved period. Nothing is retrained
automatically: an approved version stays frozen (Step 72).
"""

from __future__ import annotations

import gc
import json
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..ensemble.calibration import fit_ensemble_calibrator
from ..ensemble.data import load_pair
from ..ensemble.inference import EnsembleModel, current_manifests
from ..ensemble.registry import (
    constituent_id,
    ensemble_id,
    freeze_ensemble_spec,
    load_frozen_ensemble_spec,
)
from ..ensemble.stacking import fit_stack
from ..ensemble.weighting import (
    apply_state_weights,
    conditional_weights,
    diversity_weights,
    performance_weights,
)
from ..ml.datasets import MLData
from ..ml.models import preprocessing_kind
from ..ml.registry import freeze_spec, load_artifact, load_frozen_spec, save_artifact
from ..ml.training import train_final
from ..utils.clock import utc_now_iso
from ..utils.paths import atomic_write_text, ensure_dir
from .feature_research import write_json

__all__ = ["finalize_ensembles", "freeze_ensembles", "write_ensemble_registry"]

SECOND_LOOK = ("The reserved period (2022-01-01 ->) was evaluated once before, by Prompt #10 "
               "(37 single-model specs, 2026-10-01 08:30). This ensemble was frozen by rules "
               "fixed in config/ensemble.yaml before any ensemble result and applied to every "
               "pair alike on development data only; its final test is a second look at a "
               "period whose single-model results were already known - informative, not "
               "untouched.")


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _set_key(ml: Any, target: str, fset: str) -> str:
    return fset if fset in ("standard", "extended", "minimal") else f"target_{ml.targets[target].kind}"


def constituent_spec(ml: Any, data: MLData, timeframe: str, target: str, h: int, model: str,
                     evidence: dict[str, Any]) -> dict[str, Any]:
    """The frozen Prompt #10 base configuration of one constituent."""
    fam, fset = model.split("|", 1)
    tspec = ml.targets[target]
    key = _set_key(ml, target, fset)
    return {
        "spec_id": constituent_id(fam, fset, target, timeframe, h, ml.registry_version),
        "role": "ensemble_constituent", "family": fam, "target": target,
        "target_definition": {"source": tspec.source, "task": tspec.task, "kind": tspec.kind,
                              "transform": tspec.transform, "c": tspec.c,
                              "cost_multiple": tspec.cost_multiple},
        "horizon": int(h), "timeframe": timeframe, "feature_set": fset,
        "feature_set_id": data.manifest_ids.get(key),
        "feature_set_hash": data.manifest_hashes.get(key),
        "features": data.feature_set(key),
        "params": ml.model_params(fam),
        "preprocessing": {"kind": preprocessing_kind(fam, str(ml.preprocessing.get(
            "tree_missing", "native"))), "clip": float(ml.preprocessing.get("clip", 8.0))},
        # the ensemble's inputs are the constituents' Platt probabilities (Prompt #10: fitted
        # on the fold's inner slice; here on the final model's inner slice)
        "calibration": "platt" if tspec.is_classification else "none",
        "training_policy": {"scheme": "expanding",
                            "rolling_years": ml.walk_forward.rolling_years, "weighting": None,
                            "inner_fraction": ml.walk_forward.inner_fraction,
                            "embargo_bars": ml.walk_forward.embargo_bars, "purge_bars": int(h),
                            "train_end": str(data.reserved_start)},
        "seed": ml.random_seed,
        "dataset_versions": {k: data.versions.get(k) for k in (
            "tick_dataset_version", "bar_dataset_version", "factory_version", "target_version")},
        "development": evidence, "config_fingerprint": ml.fingerprint(), "git_commit": None,
        "outputs": "probability" if tspec.is_classification else "expected value",
        "no_trading_output": True}


def _state_inputs(ecfg: Any, ctx: dict[str, np.ndarray], method: str, rows: np.ndarray
                  ) -> tuple[list[str], np.ndarray]:
    if method == "regime_conditioned":
        names = [c for c in ecfg.conditional.get("regime_probabilities") or [] if c in ctx]
        return names, np.column_stack([ctx[c][rows] for c in names])
    nb = int(ecfg.conditional.get("volatility_buckets", 4))
    q = np.asarray(ctx["vol_quartile"][rows], dtype=np.int64)
    probs = np.full((q.size, nb), np.nan)
    ok = q >= 0
    probs[ok] = 0.0
    probs[np.flatnonzero(ok), q[ok]] = 1.0
    return ["vol_quartile"], probs


def freeze_ensembles(tctx: Any) -> dict[str, Any]:
    """ENSEMBLE_SPEC + constituent MODEL_SPEC files of one timeframe (development only)."""
    from .ensemble_research import pair_context

    ecfg, ml, data = tctx.cfg, tctx.ml, tctx.data
    out = tctx.out_dir
    frozen_dir = ensure_dir(out / "frozen")
    const_dir = ensure_dir(out / "constituents")
    decisions = []
    for target, h in ecfg.pairs:
        d = out / target / f"h{h}"
        if not (d / "summary.json").exists():
            decisions.append({"target": target, "horizon": h, "frozen": None,
                              "reason": "no research results"})
            continue
        summ = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        universe = list(summ.get("universe") or [])
        if summ.get("ensemble", "ok") is None or len(universe) < ecfg.minimum_model_count \
                or not (d / "freeze_decision.json").exists():
            decisions.append({"target": target, "horizon": h, "frozen": None,
                              "reason": summ.get("reason") or "no ensemble universe"})
            continue
        fd = json.loads((d / "freeze_decision.json").read_text(encoding="utf-8"))
        tspec = ml.targets[target]
        pair = load_pair(tctx.ml_dir / "predictions" / f"{target}_h{h}.parquet",
                         timeframe=tctx.timeframe, target=target, horizon=h, task=tspec.task,
                         calibrated_inputs=ecfg.calibrated_only_for_classification,
                         reserved_start=data.reserved_start)
        ctx = pair_context(tctx, pair)
        cov = pair.covered(universe)
        rows = np.flatnonzero(cov)                     # every out-of-sample row, 2011-2021
        p, y, base = pair.matrix(universe)[rows], pair.label[rows], pair.base[rows]
        method = str(fd["frozen_method"])
        comb: dict[str, Any]
        context_inputs: list[str] = []
        if method == "best_individual":
            member = fd.get("best_individual_for_final_test") or universe[0]
            comb = {"kind": "single", "member": member,
                    "rule": "no ensemble earned its complexity in development: the best "
                            "individual model (Prompt #10's frozen candidate when it is in the "
                            "universe, else the best development mean)"}
            pred = p[:, universe.index(member)]
        elif method in ("simple_average", "median"):
            comb = {"kind": method}
            pred = p.mean(axis=1) if method == "simple_average" else np.median(p, axis=1)
        elif method == "performance_weighted":
            s = float(ecfg.weighting.get("candidate_shrink", 0.5))
            w, q = performance_weights(p, y, base, tspec.task, shrink=s)
            comb = {"kind": "weights", "rule": "performance", "shrink": s,
                    "weights": w.tolist(), "skill": q.tolist()}
            pred = p @ w
        elif method == "diversity_weighted":
            lam = float(ecfg.diversity.get("candidate_lambda", 0.5))
            w, info = diversity_weights(p, y, base, tspec.task, lam=lam,
                                        shrink=float(ecfg.diversity.get("shrink", 0.5)),
                                        duplicate_correlation=float(ecfg.diversity.get(
                                            "duplicate_correlation", 0.999)))
            comb = {"kind": "weights", "rule": "diversity", "lambda": lam,
                    "weights": w.tolist(), "skill": info["quality"].tolist(),
                    "redundancy": info["redundancy"].tolist(),
                    "clusters": [[universe[i] for i in g] for g in info["clusters"]]}
            pred = p @ w
        elif method == "stacking":
            meta = fit_stack(pair, universe, params={
                "logistic_C": ecfg.stacking.get("logistic_C", 1.0),
                "ridge_alpha": ecfg.stacking.get("ridge_alpha", 1.0),
                "max_rows": ecfg.stacking.get("max_rows")}, seed=ecfg.random_seed)
            comb = {"kind": "stacking", "meta_model": meta.to_dict()}
            pred = meta.predict(p)
        elif method in ("regime_conditioned", "volatility_conditioned"):
            names, probs = _state_inputs(ecfg, ctx, method, rows)
            w_states, w0, own = conditional_weights(
                p, y, base, tspec.task, probs,
                shrink=float(ecfg.conditional.get("shrink", 0.5)),
                min_rows=float(ecfg.conditional.get("min_rows_per_state", 5000)))
            comb = {"kind": "state_weights",
                    "states": "regime" if method == "regime_conditioned" else "volatility",
                    "state_inputs": names, "weights": w_states.tolist(), "fallback": w0.tolist(),
                    "states_estimated_on_their_own": own}
            if method == "volatility_conditioned":
                comb["bucket_rule"] = ("causal quartiles of log_rv_20 over the expanding "
                                       "history (ml.datasets.build_context)")
            context_inputs = names
            pred = np.einsum("ij,ij->i", p, apply_state_weights(probs, w_states, w0))
        elif method == "dynamic":
            comb = {"kind": "dynamic", "window_days": int(ecfg.dynamic.get("window_days", 60)),
                    "min_rows": int(ecfg.dynamic.get("min_rows", 2000)),
                    "shrink": float(ecfg.dynamic.get("shrink", 0.5)),
                    "update": "once per trading day from rows whose label resolved before the "
                              "day's first bar"}
            context_inputs = ["dynamic_weights"]
            pred = p.mean(axis=1)                       # (calibration below is skipped)
        else:                                           # pragma: no cover - rule-checked
            raise ValueError(f"unknown frozen method {method!r}")
        cal_choice = (fd.get("calibration") or {}).get("calibration", "none")
        calibration: dict[str, Any] = {"method": "none"}
        # a probability average may need recalibrating; a stacked logistic model and a single
        # (Platt-calibrated) constituent do not; trailing weights have no fixed predictions
        if tspec.is_classification and method not in ("stacking", "dynamic", "best_individual") \
                and cal_choice != "none":
            c = fit_ensemble_calibrator(pred, y, cal_choice, min_rows=int(
                ecfg.calibration.get("isotonic_min_rows", 20000)))
            calibration = c.to_dict()
        # constituents (and their own frozen specs)
        elig = pl.read_parquet(d / "model_eligibility.parquet")
        constituents = []
        for m in universe:
            ev = elig.filter(pl.col("model") == m).row(0, named=True)
            body = constituent_spec(ml, data, tctx.timeframe, target, h, m,
                                    {k: ev.get(k) for k in ("status", "mean", "se", "min",
                                                            "blocks_beating_baseline",
                                                            "mean_auc", "mean_ece",
                                                            "registered_null",
                                                            "posthoc_sign_flip")})
            path = freeze_spec(body, const_dir)
            cs = load_frozen_spec(path)
            constituents.append({"name": m, "model_id": cs["spec_id"],
                                 "spec_hash": cs["content_hash"], "family": cs["family"],
                                 "feature_set": cs["feature_set"],
                                 "feature_set_id": cs["feature_set_id"],
                                 "feature_set_hash": cs["feature_set_hash"],
                                 "features": cs["features"], "calibration": cs["calibration"]})
        best = fd.get("best_individual_for_final_test")
        series = pl.read_parquet(d / "series_summary.parquet")

        def ev_of(name: str, table: pl.DataFrame = series) -> dict[str, Any]:
            r = table.filter(pl.col("series") == name)
            return {k: v for k, v in r.row(0, named=True).items() if not isinstance(v, str)
                    or k in ("series", "kind", "metric")} if r.height else {}

        spec = {
            "spec_id": ensemble_id(target, tctx.timeframe, h, ecfg.registry_version),
            "timeframe": tctx.timeframe, "target": target,
            "target_definition": {"source": tspec.source, "task": tspec.task,
                                  "kind": tspec.kind, "transform": tspec.transform,
                                  "c": tspec.c, "cost_multiple": tspec.cost_multiple},
            "horizon": int(h), "task": tspec.task, "method": method, "role": fd["role"],
            "input_kind": ("Platt probabilities of the constituents" if tspec.is_classification
                           else "expected values of the constituents"),
            "constituents": constituents, "combination": comb, "calibration": calibration,
            "companions": {"simple_average": universe, "best_individual": best,
                           "constant": "mean label of the development fit rows"},
            "context_inputs": context_inputs,
            "training_policy": {"constituents": "expanding, every development row before "
                                f"{data.reserved_start} (purge h + embargo), inner slice "
                                f"{ml.walk_forward.inner_fraction} for early stopping and "
                                "calibration",
                                "combination": "fitted on every walk-forward out-of-sample "
                                               "row 2011-2021 (blocks wf1-wf5)"},
            "refit_policy": "frozen between approved versions; no automatic retraining; a "
                            "constituent or combination refresh is a new ensemble version",
            "dataset_versions": {k: data.versions.get(k) for k in (
                "tick_dataset_version", "bar_dataset_version", "factory_version",
                "target_version")},
            "missing_constituent_policy": "fail_closed (ENSEMBLE_INVALID)",
            "development": {"frozen_series": ev_of(fd["frozen_series"]),
                            "best_individual": ev_of("best_individual"),
                            "simple_average": ev_of("simple_average"),
                            "decision_steps": fd.get("steps"),
                            "single_model_retained": fd.get("single_model_retained"),
                            "calibration_rule": fd.get("calibration")},
            "ensemble_config_fingerprint": ecfg.fingerprint(),
            "ml_config_fingerprint": ml.fingerprint(), "git_commit": None,
            "reserved_period_note": SECOND_LOOK, "no_trading_output": True}
        path = freeze_ensemble_spec(spec, frozen_dir)
        decisions.append({"target": target, "horizon": h, "frozen": spec["spec_id"],
                          "method": method, "role": fd["role"],
                          "constituents": [c["model_id"] for c in constituents],
                          "calibration": calibration.get("method"), "spec": path.name})
        tctx.log(f"frozen {spec['spec_id']}: {method} of {len(universe)} ({fd['role']})")
    report = {"timeframe": tctx.timeframe, "decisions": decisions, "frozen_utc": utc_now_iso(),
              "ENSEMBLE_SPEC_FROZEN": True, "note": SECOND_LOOK}
    write_json(frozen_dir / "freeze_report.json", report)
    return report


# ---------------------------------------------------------------------------
# Finalize: constituent artifacts, live checks, benchmark
# ---------------------------------------------------------------------------
def _context_for(model: EnsembleModel, data: MLData, rows: np.ndarray, tctx: Any
                 ) -> dict[str, np.ndarray] | None:
    names = model.required_context()
    if not names:
        return None
    ctx: dict[str, np.ndarray] = {}
    for n in names:
        if n == "vol_quartile":
            ctx[n] = data.context["vol_quartile"][rows].astype(np.float64)
        elif n == "dynamic_weights":
            # the latest trailing weights of the research (the model-health service's
            # state at the end of development), the same for batch and stream
            dw = tctx.out_dir / model.spec["target"] / f"h{model.spec['horizon']}" \
                / "dynamic_weights.parquet"
            last = pl.read_parquet(dw).tail(1)
            w = np.array([float(last[c][0]) for c in model.names])
            ctx[n] = np.tile(w, (rows.size, 1))
        elif n in data.features:
            ctx[n] = data.features[n][rows].astype(np.float64)
        elif n in tctx.extra:
            ctx[n] = tctx.extra[n][rows]
        elif n == "ood_score":
            ctx[n] = tctx.ood[rows]
        else:
            raise KeyError(f"no context input {n!r} for {model.spec_id}")
    return ctx


def finalize_ensembles(tctx: Any, *, streaming: bool = True) -> dict[str, Any]:
    ecfg, ml, data = tctx.cfg, tctx.ml, tctx.data
    frozen_dir = tctx.out_dir / "frozen"
    const_dir = tctx.out_dir / "constituents"
    manifests = current_manifests(tctx.cfgs.ml.selection.results_path / tctx.timeframe
                                  / "manifests")
    out: dict[str, Any] = {"timeframe": tctx.timeframe, "constituents": [], "ensembles": []}
    trained: set[str] = set()
    bars = None
    specs = sorted(frozen_dir.glob("ENSEMBLE_SPEC_*.json"))
    for spec_path in specs:
        spec = load_frozen_ensemble_spec(spec_path)
        tspec = ml.targets[spec["target"]]
        for c in spec["constituents"]:
            mid = c["model_id"]
            if mid in trained:
                continue
            cspec = load_frozen_spec(const_dir / f"MODEL_SPEC_{mid}.json")
            art = ecfg.models_path / mid
            entry: dict[str, Any] = {"model_id": mid}
            if (art / "manifest.json").exists():
                _, _, _, man = load_artifact(art)
                if man.get("spec_hash") != cspec["content_hash"]:
                    raise RuntimeError(f"{mid}: the stored artifact was built from another spec")
                entry["artifact"] = "existing"
            else:
                # a linear fit on the Extended set at 5m makes ~1.3 GB of float64 copies: start
                # each fit with no cached design and no garbage left from the previous one
                data.drop_designs()
                gc.collect()
                t0 = time.perf_counter()
                cmodel, cpre, ccal, info = train_final(data, cspec, tspec, ml)
                save_artifact(art, model=cmodel, preprocessor=cpre, calibrator=ccal, spec=cspec,
                              extra={"training": info, "training_policy":
                                     cspec["training_policy"]})
                entry.update({"artifact": "trained", "train_seconds": time.perf_counter() - t0,
                              "fit_rows": info.get("fit_rows"),
                              "best_iteration": info.get("best_iteration")})
                del cmodel, cpre, ccal
            m1, p1, c1, man1 = load_artifact(art)
            check = np.linspace(0, data.n - 1, 3000).astype(np.int64)
            xd = data.design(list(cspec["features"]))[check]
            first = c1.apply(m1.predict(p1.transform(xd, cspec["features"])))
            m2, p2, c2, _ = load_artifact(art)
            second = c2.apply(m2.predict(p2.transform(xd, cspec["features"])))
            entry["reload_equal"] = bool(np.allclose(first, second, rtol=1e-7, atol=1e-9,
                                                     equal_nan=True))
            entry["model_size_bytes"] = man1.get("model_size_bytes")
            out["constituents"].append(entry)
            trained.add(mid)
            data.drop_designs()
            tctx.log(f"constituent {mid}: {entry['artifact']}, reload {entry['reload_equal']}")
        # the ensemble through its inference interface
        tracemalloc.start()
        t0 = time.perf_counter()
        model = EnsembleModel.load(spec_path, models_path=ecfg.models_path,
                                   constituent_dir=const_dir, live_manifests=manifests)
        load_seconds = time.perf_counter() - t0
        load_peak_mb = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
        feats = model.required_features()
        rows = np.linspace(0, data.n - 1, 3000).astype(np.int64)
        x = {f: data.features[f][rows] for f in feats}
        ctx = _context_for(model, data, rows, tctx)
        a = model.predict_batch(x, ctx)
        again = EnsembleModel.load(spec_path, models_path=ecfg.models_path,
                                   constituent_dir=const_dir,
                                   live_manifests=manifests).predict_batch(x, ctx)
        # the combination recomputed by hand from the constituents' own outputs
        manual = model.combine(a["constituents"], ctx)
        e: dict[str, Any] = {
            "spec_id": model.spec_id, "method": spec["method"], "role": spec["role"],
            "constituents": len(model.constituents), "features": len(feats),
            "reload_equal": bool(np.allclose(a["prediction"], again["prediction"], rtol=1e-9,
                                             atol=1e-12, equal_nan=True)),
            "combination_equal": bool(np.allclose(a["prediction"], manual, rtol=1e-12,
                                                  atol=1e-12, equal_nan=True)),
            "invalid_rows_of_3000": int((a["status"] != "ok").sum()),
            "load_seconds": load_seconds, "load_python_peak_mb": load_peak_mb}
        one_row = {f: float(x[f][0]) for f in feats}
        one_ctx = None if ctx is None else {k: v[0] for k, v in ctx.items()}
        t0 = time.perf_counter()
        for _ in range(100):
            model.predict(one_row, one_ctx)
        e["latency_ms_per_row"] = (time.perf_counter() - t0) / 100 * 1000
        big = np.linspace(0, data.n - 1, 10_000).astype(np.int64)
        xb = {f: data.features[f][big] for f in feats}
        cb = _context_for(model, data, big, tctx)
        tracemalloc.start()
        t0, c0 = time.perf_counter(), time.process_time()
        model.predict_batch(xb, cb)
        wall, cpu = time.perf_counter() - t0, time.process_time() - c0
        e["latency_ms_per_10k_rows"] = wall * 1000
        # process CPU time over wall time: > 1 means the boosters / forests used several cores
        e["cpu_utilisation_10k_rows"] = cpu / wall if wall > 0 else None
        e["prediction_python_peak_mb_10k_rows"] = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
        e["artifact_bytes"] = int(sum((ecfg.models_path / c.model_id / name).stat().st_size
                                      for c in model.constituents for name in c.manifest["files"])
                                  + spec_path.stat().st_size)
        # the artifact manifests as finalized: the final test refuses one changed afterwards
        e["constituent_manifests"] = {c.model_id: _sha256(ecfg.models_path / c.model_id
                                                          / "manifest.json")
                                      for c in model.constituents}
        e["cpu_note"] = ("single-threaded per constituent call; random forests dominate the "
                         "latency (Prompt #10: 19-28 ms per row)")
        if streaming:
            from ..ensemble.streaming import load_development_bars, stream_ensemble
            from ..selection.live import required_buffer

            if bars is None:
                # development bars only: the reserved period's prices never enter memory
                bars = load_development_bars(tctx.cfgs.ml.config, tctx.timeframe,
                                             data.reserved_start)
                if not bars.timestamps.equals(data.timestamps):
                    raise RuntimeError("development bars and feature rows differ")
            st = ecfg.streaming
            steps, stride = int(st.get("steps", 40)), int(st.get("stride", 7))
            srows = [data.n - 1 - stride * k for k in range(steps)][::-1]
            cfgs = tctx.cfgs.ml
            pct = cfgs.features.percentile_window(bars.bar_seconds)
            buffer = required_buffer(data.registry, feats, pct)
            sctx = _context_for(model, data, np.asarray(srows, dtype=np.int64), tctx)
            res = stream_ensemble(model, data.registry, bars, data.features, srows,
                                  buffer=buffer, fcfg=cfgs.features, ou=cfgs.ou,
                                  research=cfgs.research, regression=cfgs.regression,
                                  spectral=cfgs.spectral, wavelet=cfgs.wavelet, context=sctx,
                                  rtol=float(st.get("rtol", 1e-4)),
                                  atol=float(st.get("atol", 1e-6)))
            e["streaming"] = res
            e["streaming_safe"] = bool(res["passed"])
        out["ensembles"].append(e)
        tctx.log(f"finalize {model.spec_id}: reload {e['reload_equal']}, streaming "
                 f"{e.get('streaming_safe')}, {e['latency_ms_per_row']:.2f} ms/row")
        del model
        data.drop_designs()
        gc.collect()
    out["finished_utc"] = utc_now_iso()
    write_json(frozen_dir / "finalize.json", out)
    write_ensemble_registry(ecfg, frozen_dir, out)
    return out


def write_ensemble_registry(ecfg: Any, frozen_dir: Path, finalized: dict[str, Any]) -> Path:
    """``config/ensemble_registry.yaml``: one entry per frozen ensemble (generated)."""
    import yaml

    path = ecfg.registry_path
    current: dict[str, Any] = {}
    if path.exists():
        current = dict((yaml.safe_load(path.read_text(encoding="utf-8")) or {})
                       .get("ensembles") or {})
    root = ecfg.project_root

    def rel(p: Path) -> str:
        try:
            return p.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            return p.as_posix()

    by_id = {e["spec_id"]: e for e in finalized.get("ensembles", [])}
    for sp in sorted(frozen_dir.glob("ENSEMBLE_SPEC_*.json")):
        spec = load_frozen_ensemble_spec(sp)
        e = by_id.get(spec["spec_id"], {})
        current[spec["spec_id"]] = {
            "role": spec.get("role"), "method": spec["method"], "target": spec["target"],
            "horizon": int(spec["horizon"]), "timeframe": spec["timeframe"],
            "outputs": "probability" if spec["task"] == "classification" else "expected value",
            "constituents": [c["model_id"] for c in spec["constituents"]],
            "constituent_spec_hashes": [c["spec_hash"] for c in spec["constituents"]],
            "calibration": (spec.get("calibration") or {}).get("method"),
            "spec": rel(sp), "spec_hash": spec["content_hash"],
            "reload_equal": e.get("reload_equal"), "streaming_safe": e.get("streaming_safe"),
            "latency_ms_per_row": e.get("latency_ms_per_row"),
            "artifact_bytes": e.get("artifact_bytes"), "no_trading_output": True}
    header = ("# Registry of frozen ensembles - generated by `xq ensemble-finalize`; do not "
              "edit.\n# Each entry points to its frozen, content-hashed ENSEMBLE_SPEC; "
              "ensembles output\n# probabilities / expected values only (Prompt #11).\n")
    body = yaml.safe_dump({"schema_version": 1, "updated_utc": utc_now_iso(),
                           "ensembles": dict(sorted(current.items()))}, sort_keys=False,
                          allow_unicode=True)
    atomic_write_text(path, header + body)
    return path
