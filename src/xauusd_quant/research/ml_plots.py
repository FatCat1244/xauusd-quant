"""Figures of the supervised predictive research (Prompt #10, Step 97).

Each figure reads a table written by :mod:`.ml_reports` and writes the plotted
numbers beside it as a CSV, under ``results/ml_research/<tf>/plots/``. Style as
in Prompts #8-#9: the project's categorical order for model families (never
cycled past six - a figure shows the six families of one comparison at most),
a sequential blue ramp for magnitudes, the blue-gray-red diverging pair for
signed values, one y-scale per axis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402,F401
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from ..ml.config import MLConfig  # noqa: E402
from ..utils.paths import ensure_dir  # noqa: E402
from .feature_plots import _BLUE, _DIVERGING, COLOURS, _figure, _finish, _legend  # noqa: E402
from .ou_plots import BASELINE, INK_SECONDARY, MUTED  # noqa: E402

__all__ = ["write_ml_figures"]

_FAMILY_COLOURS = {"logistic_l2": COLOURS[0], "ridge": COLOURS[0], "random_forest": COLOURS[1],
                   "xgboost": COLOURS[2], "lightgbm": COLOURS[3], "catboost": COLOURS[4],
                   "logistic_l1": COLOURS[5], "elastic_net": COLOURS[5], "ols": COLOURS[5],
                   "logistic_en": MUTED, "constant": BASELINE}


def _read(tables: Path, name: str) -> pl.DataFrame:
    p = tables / f"{name}.parquet"
    return pl.read_parquet(p) if p.exists() else pl.DataFrame()


def _label(t: str, h: int) -> str:
    return f"{t} h{h}"


def _groups(frame: pl.DataFrame, keys: list[str]) -> list[tuple[Any, pl.DataFrame]]:
    """``group_by`` in order, or nothing when the table was not written (a stage not run)."""
    if frame.is_empty() or not set(keys) <= set(frame.columns):
        return []
    return list(frame.group_by(keys, maintain_order=True))


def _grid_axes(n: int, cols: int = 3, width: float = 12.0, row_h: float = 3.6
               ) -> tuple[Any, list[Any]]:
    rows = max(1, int(np.ceil(n / cols)))
    fig, axes = _figure(rows, cols, width=width, height=row_h * rows)
    flat = list(axes.ravel())
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def plot_roc_pr(plots: Path, t: pl.DataFrame, cfg: MLConfig, tf: str) -> Path | None:
    if t.is_empty():
        return None
    pairs = [p for p in cfg.comparison_pairs() if cfg.targets[p[0]].is_classification]
    part = t.filter(pl.col("feature_set") == cfg.default_feature_set)
    fig, axes = _grid_axes(2 * len(pairs), cols=4, width=15, row_h=3.4)
    k = 0
    for target, h in pairs:
        for curve in ("roc", "pr"):
            ax = axes[k]
            k += 1
            sub = part.filter((pl.col("target") == target) & (pl.col("horizon") == h)
                              & (pl.col("curve") == curve))
            for fam in sub["family"].unique().sort().to_list():
                c = sub.filter(pl.col("family") == fam)
                ax.plot(c["x"], c["y"], lw=1.3, color=_FAMILY_COLOURS.get(fam, MUTED), label=fam)
            if curve == "roc":
                ax.plot([0, 1], [0, 1], color=BASELINE, lw=0.8, ls="--")
                ax.set_xlabel("false positive rate")
                ax.set_ylabel("true positive rate")
            else:
                ax.set_xlabel("recall")
                ax.set_ylabel("precision")
            ax.set_title(f"{_label(target, h)} {curve.upper()}", fontsize=9)
            _legend(ax, fontsize=6)
    return _finish(fig, plots / "roc_pr.png", f"{tf}: ROC and PR curves, pooled 2011-2021 "
                   "walk-forward predictions (platt)", t)


def plot_reliability(plots: Path, t: pl.DataFrame, groups: pl.DataFrame, cfg: MLConfig,
                     tf: str) -> list[Path]:
    out = []
    if not t.is_empty():
        pairs = [p for p in cfg.comparison_pairs() if cfg.targets[p[0]].is_classification]
        fig, axes = _grid_axes(len(pairs), cols=3)
        for ax, (target, h) in zip(axes, pairs, strict=True):
            sub = t.filter((pl.col("target") == target) & (pl.col("horizon") == h)
                           & (pl.col("feature_set") == cfg.default_feature_set))
            for fam in ("logistic_l2", "lightgbm", "xgboost", "catboost", "random_forest"):
                for calib, ls in (("raw", ":"), ("platt", "-")):
                    c = sub.filter((pl.col("family") == fam) & (pl.col("calibration") == calib))
                    if c.height:
                        ax.plot(c["mean_predicted"], c["observed"], marker="o", ms=2.5, lw=1.1,
                                ls=ls, color=_FAMILY_COLOURS.get(fam, MUTED),
                                label=f"{fam} {calib}")
            mp = sub["mean_predicted"].cast(pl.Float64).drop_nulls().to_numpy()
            lo, hi = (float(mp.min()), float(mp.max())) if mp.size else (0.0, 1.0)
            ax.plot([lo, hi], [lo, hi], color=BASELINE, lw=0.8, ls="--")
            ax.set_xlabel("mean predicted probability")
            ax.set_ylabel("observed frequency")
            ax.set_title(_label(target, h), fontsize=9)
            _legend(ax, fontsize=5)
        out.append(_finish(fig, plots / "reliability.png", f"{tf}: reliability diagrams "
                           "(quantile bins; dotted raw, solid Platt)", t))
    if not groups.is_empty() and "brier_skill" in groups.columns:
        sub = groups.filter(pl.col("grouping").is_in(["regime_state", "vol_quartile"])
                            & pl.col("brier_skill").is_not_null()
                            & (pl.col("feature_set") == cfg.default_feature_set))
        if sub.height:
            fig, axes = _figure(1, 2, width=12, height=4.2)
            for ax, g in zip(axes.ravel(), ("regime_state", "vol_quartile"), strict=True):
                part = sub.filter((pl.col("grouping") == g) & (pl.col("family") == "lightgbm"))
                for i, (t_, h_) in enumerate(p for p in cfg.comparison_pairs()
                                             if cfg.targets[p[0]].is_classification):
                    c = part.filter((pl.col("target") == t_) & (pl.col("horizon") == h_)).sort(
                        "group")
                    if c.height:
                        ax.plot(c["label"], c["ece"], marker="o", ms=3, lw=1.4,
                                color=COLOURS[i % 6], label=_label(t_, h_))
                ax.set_title(f"expected calibration error by {g} (LightGBM, Platt)", fontsize=9)
                ax.set_ylabel("ECE")
                _legend(ax, fontsize=6)
            out.append(_finish(fig, plots / "calibration_by_regime.png",
                               f"{tf}: calibration by regime state and volatility quartile",
                               sub))
    return out


def plot_probability_hist(plots: Path, t: pl.DataFrame, cfg: MLConfig, tf: str) -> Path | None:
    if t.is_empty():
        return None
    pairs = [p for p in cfg.comparison_pairs() if cfg.targets[p[0]].is_classification]
    fig, axes = _grid_axes(len(pairs), cols=3)
    for ax, (target, h) in zip(axes, pairs, strict=True):
        sub = t.filter((pl.col("target") == target) & (pl.col("horizon") == h)
                       & (pl.col("feature_set") == cfg.default_feature_set))
        for fam in ("logistic_l2", "lightgbm", "catboost"):
            c = sub.filter(pl.col("family") == fam).sort("lo")
            if c.height:
                ax.step(c["lo"], c["share"], where="post", lw=1.3,
                        color=_FAMILY_COLOURS.get(fam, MUTED), label=fam)
        ax.axvline(0.5, color=BASELINE, lw=0.7, ls="--")
        ax.set_xlabel("predicted probability (Platt)")
        ax.set_ylabel("share of bars")
        ax.set_title(_label(target, h), fontsize=9)
        _legend(ax, fontsize=6)
    return _finish(fig, plots / "probability_distribution.png",
                   f"{tf}: distribution of predicted probabilities (no thresholds chosen)", t)


def plot_deciles(plots: Path, t: pl.DataFrame, cfg: MLConfig, tf: str) -> Path | None:
    if t.is_empty():
        return None
    pairs = cfg.comparison_pairs()
    fig, axes = _grid_axes(len(pairs), cols=4, width=15, row_h=3.2)
    for ax, (target, h) in zip(axes, pairs, strict=True):
        sub = t.filter((pl.col("target") == target) & (pl.col("horizon") == h)
                       & (pl.col("feature_set") == cfg.default_feature_set))
        for fam in ("logistic_l2", "ridge", "lightgbm"):
            c = sub.filter(pl.col("family") == fam).sort("decile")
            if c.height:
                ax.plot(c["decile"], c["mean_outcome"], marker="o", ms=3, lw=1.4,
                        color=_FAMILY_COLOURS.get(fam, MUTED), label=fam)
        ax.set_xlabel("prediction decile")
        ax.set_ylabel("mean outcome")
        ax.set_title(_label(target, h), fontsize=9)
        _legend(ax, fontsize=6)
    return _finish(fig, plots / "prediction_deciles.png",
                   f"{tf}: outcome by prediction decile (predictive separation, not PnL)", t)


def plot_ic_decay(plots: Path, t: pl.DataFrame, cfg: MLConfig, tf: str) -> Path | None:
    if t.is_empty():
        return None
    pairs = cfg.comparison_pairs()
    fig, axes = _grid_axes(len(pairs), cols=4, width=15, row_h=3.2)
    for ax, (target, h) in zip(axes, pairs, strict=True):
        sub = t.filter((pl.col("target") == target) & (pl.col("horizon") == h)
                       & (pl.col("feature_set") == cfg.default_feature_set))
        for fam in ("logistic_l2", "ridge", "lightgbm"):
            c = sub.filter(pl.col("family") == fam).sort("outcome_horizon")
            if c.height:
                ax.plot(c["outcome_horizon"], c["rank_ic"], marker="o", ms=3, lw=1.4,
                        color=_FAMILY_COLOURS.get(fam, MUTED), label=fam)
        ax.axvline(h, color=BASELINE, lw=0.7, ls=":")
        ax.axhline(0, color=BASELINE, lw=0.7)
        ax.set_xscale("log")
        ax.set_xlabel("outcome horizon (bars, log)")
        ax.set_ylabel("rank IC of the model score")
        ax.set_title(f"{_label(target, h)} (trained at h={h})", fontsize=9)
        _legend(ax, fontsize=6)
    return _finish(fig, plots / "model_ic_decay.png",
                   f"{tf}: model-level alpha decay - one score against later outcomes", t)


def plot_yearly(plots: Path, t: pl.DataFrame, cfg: MLConfig, tf: str) -> Path | None:
    if t.is_empty():
        return None
    pairs = cfg.comparison_pairs()
    fig, axes = _grid_axes(len(pairs), cols=4, width=15, row_h=3.2)
    for ax, (target, h) in zip(axes, pairs, strict=True):
        task = cfg.targets[target].task
        metric = "log_loss_skill" if task == "classification" else "rank_ic"
        sub = t.filter((pl.col("target") == target) & (pl.col("horizon") == h)
                       & (pl.col("feature_set") == cfg.default_feature_set))
        for fam in ("logistic_l2", "ridge", "lightgbm", "catboost"):
            c = sub.filter(pl.col("family") == fam).sort("year")
            if c.height and metric in c.columns:
                ax.plot(c["year"], c[metric], marker="o", ms=3, lw=1.3,
                        color=_FAMILY_COLOURS.get(fam, MUTED), label=fam)
        ax.axhline(0, color=BASELINE, lw=0.7)
        ax.set_ylabel(metric)
        ax.set_title(_label(target, h), fontsize=9)
        _legend(ax, fontsize=6)
    return _finish(fig, plots / "yearly_performance.png",
                   f"{tf}: year-by-year out-of-sample performance (skill vs constant, or IC)", t)


def plot_folds(plots: Path, t: pl.DataFrame, cfg: MLConfig, tf: str) -> Path | None:
    if t.is_empty():
        return None
    pairs = cfg.comparison_pairs()
    fig, axes = _grid_axes(len(pairs), cols=4, width=15, row_h=3.2)
    rows = []
    for ax, (target, h) in zip(axes, pairs, strict=True):
        task = cfg.targets[target].task
        metric = "log_loss_skill" if task == "classification" else "rank_ic"
        calib = "platt" if task == "classification" else "raw"
        sub = t.filter((pl.col("target") == target) & (pl.col("horizon") == h)
                       & (pl.col("variant") == "base") & (pl.col("calibration") == calib)
                       & (pl.col("feature_set") == cfg.default_feature_set)
                       & (pl.col("family") != "constant"))
        fams = sub["family"].unique().sort().to_list()
        width = 0.8 / max(1, len(fams))
        for i, fam in enumerate(fams):
            c = sub.filter(pl.col("family") == fam).sort("fold_index")
            ax.bar(c["fold_index"].to_numpy() + i * width, c[metric].to_numpy(), width=width,
                   color=_FAMILY_COLOURS.get(fam, MUTED), label=fam)
            rows.append(c.select("target", "horizon", "family", "fold", metric))
        ax.axhline(0, color=BASELINE, lw=0.7)
        names = sub.select("fold_index", "fold").unique().sort("fold_index")
        ax.set_xticks(names["fold_index"].to_numpy() + 0.4,
                      ["-".join(n.split("_")[1:]) for n in names["fold"].to_list()], fontsize=6)
        ax.set_ylabel(metric)
        ax.set_title(_label(target, h), fontsize=9)
        _legend(ax, fontsize=5)
    return _finish(fig, plots / "fold_performance.png",
                   f"{tf}: walk-forward block-by-block performance",
                   pl.concat(rows, how="diagonal_relaxed") if rows else None)


def plot_importance(plots: Path, perm: pl.DataFrame, shap: pl.DataFrame, cfg: MLConfig,
                    tf: str) -> list[Path]:
    out = []
    for name, t, value, title in (("feature_importance", perm, "importance",
                                   "block permutation importance (rise in loss)"),
                                  ("shap_summary", shap, "mean_abs_shap",
                                   "mean |SHAP| (model output units)")):
        if t.is_empty():
            continue
        sub = t.filter((pl.col("family") == "lightgbm") & (pl.col("variant") == "base")
                       & (pl.col("feature_set") == cfg.default_feature_set))
        if sub.is_empty():
            continue
        agg = sub.group_by(["target", "horizon", "feature"]).agg(pl.col(value).mean())
        targets = sorted(agg.select("target", "horizon").unique().rows())
        feats = (agg.group_by("feature").agg(pl.col(value).max()).sort(value, descending=True)
                 ["feature"].head(20).to_list())
        grid = np.full((len(feats), len(targets)), np.nan)
        for r in agg.iter_rows(named=True):
            if r["feature"] in feats:
                j = targets.index((r["target"], r["horizon"]))
                col_max = agg.filter((pl.col("target") == r["target"])
                                     & (pl.col("horizon") == r["horizon"]))[value].max()
                grid[feats.index(r["feature"]), j] = (r[value] / col_max
                                                      if col_max else np.nan)
        fig, axes = _figure(width=10, height=0.3 * len(feats) + 2.2)
        ax = axes[0, 0]
        im = ax.imshow(grid, aspect="auto", cmap=_BLUE, vmin=0, vmax=1, interpolation="nearest")
        ax.grid(False)
        ax.set_yticks(range(len(feats)), feats, fontsize=7)
        ax.set_xticks(range(len(targets)), [_label(t_, h_) for t_, h_ in targets], rotation=45,
                      ha="right", fontsize=7)
        cb = fig.colorbar(im, ax=ax, fraction=0.03)
        cb.ax.tick_params(labelsize=7, colors=INK_SECONDARY)
        cb.set_label("share of the target's largest", color=INK_SECONDARY, fontsize=8)
        out.append(_finish(fig, plots / f"{name}.png", f"{tf}: LightGBM {title}, mean over "
                           "blocks (descriptive, not causal)", agg))
    return out


def plot_hyper(plots: Path, search: pl.DataFrame, neigh: pl.DataFrame, comp: pl.DataFrame,
               tf: str) -> Path | None:
    """One row per searched target (each with its own metric): the randomized trials, the
    one-step neighbours of the best LightGBM trial and the num_leaves complexity curve."""
    frames = [f for f in (search, neigh, comp) if not f.is_empty()]
    if not frames:
        return None
    pairs = sorted({(str(t), int(h)) for f in frames
                    for t, h in f.select("target", "horizon").unique().rows()})
    fig, axes = _figure(len(pairs), 3, width=15, height=3.6 * len(pairs))
    for r, (target, h) in enumerate(pairs):
        s_all = _pair(search, target, h)
        metric = next((str(f["metric"][0]) for f in (s_all, _pair(comp, target, h),
                                                     _pair(neigh, target, h))
                       if not f.is_empty() and "metric" in f.columns), "metric")
        ax = axes[r, 0]
        for (fam,), part in _groups(s_all, ["family"]):
            s = part.sort("mean", descending=True)
            ax.errorbar(np.arange(s.height), s["mean"], yerr=s["se"].fill_null(0), fmt="o",
                        ms=3, color=_FAMILY_COLOURS.get(str(fam), MUTED), label=str(fam))
        ax.set_xlabel("trial (sorted)")
        ax.set_ylabel(f"{_label(target, h)}\nmean {metric} (2018-2021)", fontsize=8)
        ax.set_title("randomized search trials (+ SE over blocks)", fontsize=9)
        _legend(ax, fontsize=6)
        ax = axes[r, 1]
        s = _pair(neigh, target, h)
        if not s.is_empty():
            s = s.sort("mean")
            ax.errorbar(s["mean"], np.arange(s.height), xerr=s["se"].fill_null(0), fmt="o",
                        ms=3, color=_FAMILY_COLOURS.get("lightgbm", MUTED))
            ax.set_yticks(np.arange(s.height), [v.removeprefix("neighbour-") for v in
                                                s["variant"].to_list()], fontsize=6)
            best = _pair(search, target, h).filter(pl.col("family") == "lightgbm")
            top = best["mean"].drop_nulls().to_list() if not best.is_empty() else []
            if top:
                ax.axvline(max(float(v) for v in top), color=INK_SECONDARY, lw=0.8, ls="--")
        ax.set_xlabel(f"mean {metric}; dashed: the best LightGBM trial", fontsize=7)
        ax.set_title("one-step neighbours of the best LightGBM trial", fontsize=9)
        ax = axes[r, 2]
        s = _pair(comp, target, h)
        if not s.is_empty():
            s = s.with_columns(pl.col("variant").str.extract(r"leaves-(\d+)").cast(pl.Int64)
                               .alias("leaves")).sort("leaves")
            ax.errorbar(s["leaves"], s["mean"], yerr=s["se"].fill_null(0), marker="o", ms=3,
                        color=_FAMILY_COLOURS.get("lightgbm", MUTED))
            ax.set_xscale("log", base=2)
        ax.set_xlabel("num_leaves (log2)")
        ax.set_title("complexity curve (LightGBM)", fontsize=9)
    return _finish(fig, plots / "hyperparameter_stability.png",
                   f"{tf}: hyperparameter search, neighbourhood and complexity",
                   pl.concat([search, neigh, comp], how="diagonal_relaxed"))


def _pair(frame: pl.DataFrame, target: str, h: int) -> pl.DataFrame:
    if frame.is_empty() or "target" not in frame.columns:
        return pl.DataFrame()
    return frame.filter((pl.col("target") == target) & (pl.col("horizon") == h))


def plot_windows(plots: Path, windows: pl.DataFrame, lc: pl.DataFrame, retrain: pl.DataFrame,
                 decay: pl.DataFrame, cfg: MLConfig, tf: str) -> Path | None:
    """One row per focus pair (one metric per axis): windows / weights, the learning
    curve, the refit frequency and the decay of a model frozen in 2011."""
    researched = windows.filter(pl.col("variant") != "base") if not windows.is_empty() \
        else windows
    pairs = [(t, h) for t, h in cfg.focus
             if not _pair(researched, t, h).is_empty() or not _pair(lc, t, h).is_empty()]
    if not pairs:
        return None
    fig, axes = _figure(len(pairs), 4, width=17, height=3.4 * len(pairs))
    variants = ["base", "window-rolling5", "weighting-2p0", "weighting-5p0"]
    for r, (target, h) in enumerate(pairs):
        metric = "log_loss_skill" if cfg.targets[target].is_classification else "rank_ic"
        fams = [cfg.reference_linear(cfg.targets[target].task), "lightgbm"]
        ax = axes[r, 0]
        w = _pair(windows, target, h)
        for k, fam in enumerate(fams):              # colour = family, as in the other panels
            vals = []
            for var in variants:
                v = w.filter((pl.col("family") == fam) & (pl.col("variant") == var)) \
                    if not w.is_empty() else w
                vals.append(float(v["mean"][0]) if v.height and v["mean"][0] is not None
                            else np.nan)
            ax.bar(np.arange(len(variants)) + 0.4 * k, vals, width=0.38,
                   color=_FAMILY_COLOURS.get(fam, MUTED), label=fam)
        ax.set_xticks(np.arange(len(variants)) + 0.2,
                      ["expanding", "rolling 5y", "half-life 2y", "half-life 5y"], fontsize=7)
        ax.axhline(0, color=BASELINE, lw=0.7)
        ax.set_ylabel(f"{_label(target, h)}\nmean {metric}", fontsize=8)
        ax.set_title("expanding vs rolling vs time-decay weights", fontsize=9)
        _legend(ax, fontsize=6)
        ax = axes[r, 1]
        for (fam,), part in _groups(_pair(lc, target, h), ["family"]):
            s = part.with_columns(pl.col("variant").str.extract(r"history-([0-9p]+)")
                                  .str.replace("p", ".").cast(pl.Float64).alias("share")) \
                .sort("share")
            ax.plot(s["share"], s["mean"], marker="o", ms=3, color=_FAMILY_COLOURS.get(fam, MUTED), label=fam)
        ax.set_xlabel("share of the most recent history (last block)")
        ax.set_title("learning curve", fontsize=9)
        _legend(ax, fontsize=6)
        ax = axes[r, 2]
        refits = _pair(retrain, target, h)
        for (fam,), part in _groups(refits, ["family"]):
            s = part.with_columns(pl.col("refit_every").str.replace("m", "").cast(pl.Int64)
                                  .alias("months")).sort("months")
            if metric in s.columns:
                ax.plot(s["months"], s[metric], marker="o", ms=3, color=_FAMILY_COLOURS.get(fam, MUTED), label=fam)
        if refits.is_empty():                       # the refit study covers the first pairs only
            ax.text(0.5, 0.5, "not run for this pair", transform=ax.transAxes, ha="center",
                    va="center", fontsize=8, color=INK_SECONDARY)
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_xlabel("refit every N months (2018-2021)")
        ax.set_title("retraining frequency", fontsize=9)
        _legend(ax, fontsize=6)
        ax = axes[r, 3]
        for (fam,), part in _groups(_pair(decay, target, h), ["family"]):
            s = part.sort("year")
            if metric in s.columns:
                ax.plot(s["years_after_training"], s[metric], marker="o", ms=3,
                        color=_FAMILY_COLOURS.get(fam, MUTED), label=fam)
        ax.axhline(0, color=BASELINE, lw=0.7)
        ax.set_xlabel("years after the 2011-01-01 training end")
        ax.set_title("decay of a frozen model", fontsize=9)
        _legend(ax, fontsize=6)
    return _finish(fig, plots / "training_window.png",
                   f"{tf}: training windows, weights, history, refits and decay "
                   "(log-loss skill for probabilities, rank IC for expected values)",
                   pl.concat([windows, lc], how="diagonal_relaxed"))


def plot_correlation(plots: Path, corr: pl.DataFrame, dis: pl.DataFrame, tf: str
                     ) -> list[Path]:
    out = []
    if not corr.is_empty():
        pairs = sorted(corr.select("target", "horizon").unique().rows())
        fig, axes = _grid_axes(len(pairs), cols=4, width=15, row_h=3.6)
        for ax, (t_, h_) in zip(axes, pairs, strict=True):
            sub = corr.filter((pl.col("target") == t_) & (pl.col("horizon") == h_))
            fams = sorted(set(sub["a"].to_list()) | set(sub["b"].to_list()))
            m = np.eye(len(fams))
            for r in sub.iter_rows(named=True):
                i, j = fams.index(r["a"]), fams.index(r["b"])
                m[i, j] = m[j, i] = r["spearman"]
            im = ax.imshow(m, cmap=_DIVERGING, vmin=-1, vmax=1)
            ax.grid(False)
            ax.set_xticks(range(len(fams)), fams, rotation=45, ha="right", fontsize=6)
            ax.set_yticks(range(len(fams)), fams, fontsize=6)
            for i in range(len(fams)):
                for j in range(len(fams)):
                    ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=5)
            ax.set_title(_label(t_, h_), fontsize=9)
            del im
        out.append(_finish(fig, plots / "prediction_correlation.png",
                           f"{tf}: Spearman correlation of out-of-sample predictions", corr))
    if not dis.is_empty():
        fig, axes = _figure(1, 2, width=12, height=4.2)
        for i, ((t_, h_), part) in enumerate(_groups(dis, ["target", "horizon"])):
            s = part.sort("disagreement_quintile")
            col = "consensus_auc" if "consensus_auc" in s.columns and s["consensus_auc"] \
                .is_not_null().any() else "consensus_rank_ic"
            axes[0, 0].plot(s["disagreement_quintile"], s[col], marker="o", ms=3,
                            color=COLOURS[i % 6], label=f"{_label(str(t_), int(h_))} ({col})")
            if "mean_vol_quartile" in s.columns:
                axes[0, 1].plot(s["disagreement_quintile"], s["mean_vol_quartile"], marker="o",
                                ms=3, color=COLOURS[i % 6], label=_label(str(t_), int(h_)))
        axes[0, 0].set_xlabel("model disagreement quintile")
        axes[0, 0].set_title("consensus quality by disagreement", fontsize=9)
        axes[0, 1].set_xlabel("model disagreement quintile")
        axes[0, 1].set_title("mean volatility quartile by disagreement", fontsize=9)
        _legend(axes[0, 0], fontsize=6)
        _legend(axes[0, 1], fontsize=6)
        out.append(_finish(fig, plots / "model_disagreement.png",
                           f"{tf}: model disagreement as an uncertainty proxy", dis))
    return out


_CONTROL_LABELS = {"null-shift__repeat-0": "shifted target 1",
                   "null-shift__repeat-1": "shifted target 2",
                   "pipeline-sign_flip": "sign-flip pipeline (post-hoc)"}


def _control_label(variant: str) -> str:
    if variant in _CONTROL_LABELS:
        return _CONTROL_LABELS[variant]
    if variant.startswith("noise-"):
        return f"+{variant.removeprefix('noise-')} noise cols"
    if variant.startswith("permute_features-"):
        return "permuted features"
    if variant.startswith("pipeline-"):
        return variant.removeprefix("pipeline-").replace("_", " ") + " pipeline"
    return variant


def plot_controls(plots: Path, nulls: pl.DataFrame, ablation: pl.DataFrame, tf: str
                  ) -> list[Path]:
    """Real minus control, block by block (one panel and one metric per target), and the
    ablation deltas against the full Extended model."""
    out = []
    if not nulls.is_empty():
        pairs = sorted(nulls.select("target", "horizon").unique().rows())
        fig, axes = _grid_axes(len(pairs), cols=3, width=15, row_h=3.8)
        for ax, (target, h) in zip(axes, pairs, strict=True):
            sub = nulls.filter((pl.col("target") == target) & (pl.col("horizon") == h))
            controls = list(dict.fromkeys(sub.sort("variant")["variant"].to_list()))
            fams = sorted(sub["family"].unique().to_list())
            width = 0.8 / max(1, len(fams))
            for k, fam in enumerate(fams):
                vals, errs = [], []
                for var in controls:
                    r = sub.filter((pl.col("family") == fam) & (pl.col("variant") == var))
                    ok = r.height and r["real_minus_control"][0] is not None
                    vals.append(float(r["real_minus_control"][0]) if ok else np.nan)
                    errs.append(float(r["se"][0] or 0.0) if ok and r["se"][0] is not None
                                else 0.0)
                ax.bar(np.arange(len(controls)) + k * width, vals, width=width, yerr=errs,
                       color=_FAMILY_COLOURS.get(fam, MUTED), label=fam,
                       error_kw={"elinewidth": 0.8, "ecolor": INK_SECONDARY})
            ax.set_xticks(np.arange(len(controls)) + 0.4 - width / 2,
                          [_control_label(v) for v in controls], rotation=30, ha="right",
                          fontsize=6)
            ax.axhline(0, color=BASELINE, lw=0.7)
            ax.set_ylabel(f"real - control ({sub['metric'][0]})", fontsize=8)
            ax.set_title(_label(str(target), int(h)), fontsize=9)
            _legend(ax, fontsize=6)
        out.append(_finish(fig, plots / "null_controls.png",
                           f"{tf}: the real model minus its controls on the same blocks "
                           "(+ SE); above zero = beats the control", nulls))
    if not ablation.is_empty():
        pairs = sorted(ablation.select("target", "horizon").unique().rows())
        fig, axes = _grid_axes(len(pairs), cols=3, width=15, row_h=4.0)
        for ax, (target, h) in zip(axes, pairs, strict=True):
            sub = ablation.filter((pl.col("target") == target) & (pl.col("horizon") == h)) \
                .sort(["kind", "set"])
            colours = [COLOURS[0] if k == "leave_one_out" else COLOURS[2]
                       for k in sub["kind"].to_list()]
            ax.bar(np.arange(sub.height), sub["delta_vs_extended"].fill_null(np.nan),
                   yerr=sub["se"].fill_null(0.0), color=colours,
                   error_kw={"elinewidth": 0.8, "ecolor": INK_SECONDARY})
            ax.set_xticks(np.arange(sub.height), sub["set"].to_list(), rotation=45, ha="right",
                          fontsize=6)
            ax.axhline(0, color=BASELINE, lw=0.7)
            ax.set_ylabel(f"delta vs Extended ({sub['metric'][0]})", fontsize=8)
            ax.set_title(f"{_label(str(target), int(h))}: blue leave-one-out, green forward",
                         fontsize=9)
        out.append(_finish(fig, plots / "family_ablation.png",
                           f"{tf}: LightGBM family ablation - change against the full "
                           "Extended model on the same blocks (+ SE)", ablation))
    return out


def write_ml_figures(out_dir: Path, cfg: MLConfig) -> list[str]:
    tf = out_dir.name
    tables = out_dir / "tables"
    plots = ensure_dir(out_dir / "plots")
    written: list[Path | None] = [
        plot_roc_pr(plots, _read(tables, "roc_pr"), cfg, tf),
        plot_probability_hist(plots, _read(tables, "probability_hist"), cfg, tf),
        plot_deciles(plots, _read(tables, "deciles"), cfg, tf),
        plot_ic_decay(plots, _read(tables, "alpha_decay"), cfg, tf),
        plot_yearly(plots, _read(tables, "yearly"), cfg, tf),
        plot_folds(plots, _read(tables, "fold_metrics"), cfg, tf),
        plot_hyper(plots, _read(tables, "search"), _read(tables, "neighbourhood"),
                   _read(tables, "complexity"), tf),
        plot_windows(plots, _read(tables, "windows"), _read(tables, "learning_curve"),
                     _read(tables, "retraining_pooled"), _read(tables, "decay_by_year"), cfg, tf),
    ]
    written += plot_reliability(plots, _read(tables, "reliability"), _read(tables, "groups"),
                                cfg, tf)
    written += plot_importance(plots, _read(tables, "permutation"), _read(tables, "shap"), cfg,
                               tf)
    written += plot_correlation(plots, _read(tables, "prediction_correlation"),
                                _read(tables, "disagreement"), tf)
    written += plot_controls(plots, _read(tables, "null_comparison"),
                             _read(tables, "ablation_deltas"), tf)
    return [p.name for p in written if p is not None]
