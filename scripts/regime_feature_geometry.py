#!/usr/bin/env python
"""POST HOC, DESCRIPTIVE - NOT A REGISTERED TEST: are the trend "regimes" feature geometry?

    python scripts/regime_feature_geometry.py              # 1h and 30m
    python scripts/regime_feature_geometry.py -t 1h

Reading the Prompt #7 results, the full-covariance GMM and HMM states at 30m-1h
turned out to be trend states (upward drift and a linear path / downward drift
/ a less linear path). Two of the thirteen inputs are the rolling regression's
slope / volatility and R^2 over the same 128 bars, and R^2 is close to a
monotone function of |slope / volatility|: a V-shaped dependence that the
registered redundancy rule (|Spearman| >= 0.9 on the signed values) cannot
see and that one full-covariance Gaussian cannot follow - so a mixture spends
a component on each arm and one on the vertex, and the 128-bar windows make
the components persistent.

This checks that reading on the real series and on the three pipeline nulls
(random walk, shuffled returns, 1024-bar block bootstrap: the same engines, no
regimes by construction), over the same bars:

    geometry  Spearman(R^2, slope/vol), Spearman(R^2, |slope/vol|), and the share
              of R^2's variance a 20-bin step function of |slope/vol| explains
    pair      a full GMM on (slope/vol, R^2) alone, K = 1..4: hold-out
              log-likelihood per bar (fitted before 2011-01-01, as the registered
              hold-out) and the K = 3 state descriptions
    hmm       an HMM, K = 3, fitted over the whole span with the core features and
              with the core features minus R^2: expected durations, state
              descriptions, adjacency of transitions in the slope ordering, and
              agreement (NMI) with volatility terciles

Nothing is chosen from these numbers and nothing enters the research ledger:
they qualify how the registered results are read. Outputs:
``<regime results>/post_hoc/feature_geometry_<tf>.json`` and ``.csv`` files,
with the dataset lineage. Measured beside nothing else: 1h ~11 min, 30m
~17 min, peak 0.5-0.75 GB.

    python scripts/regime_feature_geometry.py --adjacency

reads the stored walk-forward HMM studies instead (no fitting) and writes
``post_hoc/transition_adjacency.csv``: for K >= 3, the share of each last
refit's off-diagonal transition mass that goes to a neighbouring state when
the states are ordered by their median slope / volatility (chance: 2/K).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from xauusd_quant.regimes.config import RegimeConfig, load_regime_config  # noqa: E402
from xauusd_quant.regimes.dataset import feature_matrix  # noqa: E402
from xauusd_quant.regimes.emissions import log_densities  # noqa: E402
from xauusd_quant.regimes.fitting import missing_policy  # noqa: E402
from xauusd_quant.regimes.hmm import HMMModel  # noqa: E402
from xauusd_quant.regimes.preprocessing import fit_scaler  # noqa: E402
from xauusd_quant.research.regime_analysis import (  # noqa: E402
    agreement,
    bucket_labels,
    describe_states,
    holdout_scores,
    offline_fit,
    single_gaussian,
    standardized_means,
)
from xauusd_quant.research.study_io import (  # noqa: E402
    clean_json,
    code_fingerprint,
    package_versions,
)
from xauusd_quant.utils.clock import utc_now_iso  # noqa: E402
from xauusd_quant.utils.paths import atomic_write_bytes, ensure_dir  # noqa: E402

LABEL = "POST HOC, DESCRIPTIVE - NOT A REGISTERED TEST"
PAIR = ("slope_over_volatility", "r_squared")
SPLIT = np.datetime64("2011-01-01")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pl.Series(a).rank().to_numpy()
    rb = pl.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def _sources(regime: RegimeConfig, tf: str) -> tuple[dict[str, pl.DataFrame], dict[str, Any]]:
    """The real inputs and each pipeline null's inputs, over the bars they share."""
    frames = {"real": pl.read_parquet(regime.inputs_path / f"timeframe={tf}" / "inputs.parquet")}
    stamps: dict[str, Any] = {}
    for name in regime.nulls.pipeline:
        path = regime.timeframe_dir(tf) / "nulls" / f"inputs_{name}.parquet"
        if path.exists():
            frames[name] = pl.read_parquet(path)
            meta = path.with_suffix(".json")
            if meta.exists():
                stamps[name] = json.loads(meta.read_text(encoding="utf-8")).get("stamp")
    n = min(f.height for f in frames.values())
    frames = {k: f.tail(n) for k, f in frames.items()}
    first = frames["real"]["timestamp"]
    for name, f in frames.items():
        if not f["timestamp"].equals(first):
            raise RuntimeError(f"{tf}: {name} does not cover the real series' bars")
    return frames, stamps


def _geometry(frame: pl.DataFrame) -> dict[str, Any]:
    s = frame["slope_over_volatility"].cast(pl.Float64).to_numpy()
    r2 = frame["r_squared"].cast(pl.Float64).to_numpy()
    ok = np.isfinite(s) & np.isfinite(r2)
    s, r2 = s[ok], r2[ok]
    edges = np.quantile(np.abs(s), np.linspace(0, 1, 21))
    idx = np.clip(np.searchsorted(edges, np.abs(s), side="right") - 1, 0, 19)
    medians = np.array([np.median(r2[idx == b]) for b in range(20)])
    return {"rows": int(ok.sum()),
            "spearman_r2_slope": _spearman(r2, s),
            "spearman_r2_abs_slope": _spearman(r2, np.abs(s)),
            "r2_share_explained_by_abs_slope_bins": float(1 - (r2 - medians[idx]).var()
                                                          / r2.var()),
            "median_r2_by_abs_slope_ventile": [round(float(v), 3) for v in medians]}


def _pair_holdout(frame: pl.DataFrame, regime: RegimeConfig) -> list[dict[str, Any]]:
    x = feature_matrix(frame, PAIR)
    times = frame["timestamp"].to_numpy()
    cut = int(np.searchsorted(times, SPLIT, side="left"))
    scaler = fit_scaler(x[:cut], PAIR, method=regime.preprocessing.scaling,
                        clip=regime.preprocessing.clip)
    z_train, z_test = scaler.transform(x[:cut]), scaler.transform(x[cut:])
    one = single_gaussian(z_train, regime.models.gmm.reg_covar)
    test = z_test[np.isfinite(z_test).all(axis=1)]
    rows = [{"states": 1, "test_ll_per_obs": float(log_densities(test, one.states)[:, 0].mean())}]
    for k in (2, 3, 4):
        out = holdout_scores(x, times, PAIR, regime, family="gmm", k=k, split=SPLIT,
                             seed=regime.seed)
        rows.append({"states": k, "test_ll_per_obs": out.get("test_ll_per_obs")})
    base = rows[0]["test_ll_per_obs"]
    for r in rows:
        r["gain_over_one_state"] = (r["test_ll_per_obs"] - base
                                    if r["test_ll_per_obs"] is not None else None)
    fitted, _ = offline_fit(x, PAIR, regime, family="gmm", k=3, seed=regime.seed)
    labels = fitted.infer(x, missing_policy(regime)).state
    described = describe_states(standardized_means(x, labels, PAIR))
    rows[2]["states_described"] = [described[j] for j in sorted(described)]
    return rows


def _adjacency(transition: np.ndarray, order_values: np.ndarray) -> float | None:
    k = transition.shape[0]
    if k < 3:
        return None
    rank = np.empty(k, dtype=np.int64)
    rank[np.argsort(order_values)] = np.arange(k)
    off = transition.copy()
    np.fill_diagonal(off, 0.0)
    adjacent = np.abs(rank[:, None] - rank[None, :]) == 1
    return float(off[adjacent].sum() / off.sum()) if off.sum() > 0 else None


def _hmm(frame: pl.DataFrame, features: tuple[str, ...], regime: RegimeConfig) -> dict[str, Any]:
    x = feature_matrix(frame, features)
    started = time.perf_counter()
    fitted, _ = offline_fit(x, features, regime, family="hmm", k=3, seed=regime.seed)
    model = fitted.model
    assert isinstance(model, HMMModel)
    labels = fitted.infer(x, missing_policy(regime), exact=False).state
    std = standardized_means(x, labels, features)
    described = describe_states(std)
    vol = frame["log_rv_20"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    terciles, _ = bucket_labels(vol, 3)
    slope = frame["slope_over_volatility"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    slope_medians = np.array([np.nanmedian(slope[labels == j]) if (labels == j).any() else np.nan
                              for j in range(3)])
    durations = model.expected_durations()
    return {"expected_durations": [float(v) for v in durations],
            "median_expected_duration": float(np.median(durations)),
            "states_described": [described[j] for j in sorted(described)],
            "adjacency_share_slope_order": _adjacency(model.transition, slope_medians),
            "nmi_with_volatility_terciles": agreement(labels, terciles).get("nmi"),
            "fit_seconds": time.perf_counter() - started}


def adjacency_table(regime: RegimeConfig) -> pl.DataFrame:
    """Neighbour share of the stored walk-forward HMMs' transitions (slope ordering)."""
    rows = []
    for tf in regime.timeframes:
        for k in range(3, 7):
            directory = regime.results_dir(tf, "hmm", k)
            summary_path = directory / "summary.json"
            medians_path = directory / "standardized_state_medians.csv"
            if not (summary_path.exists() and medians_path.exists()):
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))["summary"]
            transition = np.asarray(summary["transition_matrix"], dtype=np.float64)
            slope = (pl.read_csv(medians_path)
                     .filter(pl.col("feature") == "slope_over_volatility").sort("state")
                     ["standardized_median"].to_numpy())
            order = np.argsort(slope)
            rows.append({"timeframe": tf, "states": k, "chance": 2 / k,
                         "adjacency_share_slope_order": _adjacency(transition, slope),
                         "lowest_to_highest_slope_state": float(transition[order[0], order[-1]]),
                         "highest_to_lowest_slope_state": float(transition[order[-1], order[0]]),
                         "scheme": summary.get("scheme"), "label": LABEL})
    return pl.DataFrame(rows, infer_schema_length=None)


def run(regime: RegimeConfig, tf: str) -> dict[str, Any]:
    started = time.perf_counter()
    frames, stamps = _sources(regime, tf)
    core = regime.features()
    sets = {"core": core, "core_minus_r_squared": tuple(f for f in core if f != "r_squared")}
    geometry, pair, hmm = [], [], []
    for name, frame in frames.items():
        print(f"[{tf}] {name}: geometry, pair GMM, HMM", flush=True)
        geometry.append({"timeframe": tf, "source": name, **_geometry(frame)})
        for r in _pair_holdout(frame, regime):
            pair.append({"timeframe": tf, "source": name, **r})
        for set_name, features in sets.items():
            hmm.append({"timeframe": tf, "source": name, "feature_set": set_name,
                        **_hmm(frame, features, regime)})
    study = regime.results_dir(tf, "hmm", 3) / "summary.json"
    lineage = (json.loads(study.read_text(encoding="utf-8"))["provenance"].get("dataset_lineage")
               if study.exists() else None)
    manifest = regime.inputs_path / f"timeframe={tf}" / "_manifest.json"
    inputs = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    here = Path(__file__).resolve()
    return {"label": LABEL, "timeframe": tf, "generated_utc": utc_now_iso(),
            "bars": frames["real"].height,
            "first_timestamp": str(frames["real"]["timestamp"][0]),
            "last_timestamp": str(frames["real"]["timestamp"][-1]),
            "geometry": geometry, "pair_gmm_holdout": pair, "hmm_k3": hmm,
            "provenance": {"dataset_lineage": lineage,
                           "inputs_version": inputs.get("inputs_version"),
                           "regime_inputs_fingerprint": inputs.get("regime_inputs_fingerprint"),
                           "regime_config_fingerprint": regime.fingerprint(),
                           "null_input_stamps": stamps,
                           "script_fingerprint": code_fingerprint([here]),
                           "packages": package_versions(("numpy", "scipy", "polars"))},
            "seconds": time.perf_counter() - started}


def _write(result: dict[str, Any], directory: Path) -> None:
    tf = result["timeframe"]
    atomic_write_bytes(directory / f"feature_geometry_{tf}.json",
                       json.dumps(clean_json(result), indent=2).encode("utf-8"))
    for key in ("geometry", "pair_gmm_holdout", "hmm_k3"):
        frame = pl.DataFrame(result[key], infer_schema_length=None)
        frame = frame.with_columns([pl.col(c).map_elements(lambda v: json.dumps(v.to_list()),
                                                           return_dtype=pl.Utf8)
                                    for c, t in frame.schema.items() if isinstance(t, pl.List)])
        frame.with_columns(pl.lit(LABEL).alias("label")).write_csv(
            directory / f"feature_geometry_{tf}_{key}.csv")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-t", "--timeframes", nargs="+", default=["1h", "30m"])
    parser.add_argument("--adjacency", action="store_true",
                        help="only the transition-adjacency table from the stored HMM studies")
    args = parser.parse_args()
    regime = load_regime_config()
    directory = ensure_dir(regime.results_path / "post_hoc")
    if args.adjacency:
        table = adjacency_table(regime)
        table.write_csv(directory / "transition_adjacency.csv")
        print(table)
        return 0
    for tf in args.timeframes:
        result = run(regime, tf)
        _write(result, directory)
        print(f"[{tf}] written in {result['seconds']:.0f} s -> {directory}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
