"""Figures of the feature selection (Prompt #9, Step 65).

Each figure reads the tables of :mod:`.feature_selection_research` and writes
the plotted numbers beside it as a CSV, into ``<timeframe>/plots/``. The style
is Prompt #8's (:mod:`.feature_plots`): the project's categorical order for up
to six categories, never cycled (horizons, splits, sets); a sequential blue
ramp for magnitudes (selection frequency, |rho|, counts); the blue-gray-red
diverging pair around zero for signed ICs and coefficients; one y-scale per
axis. The validation period appears only where the evaluation read it, after
the development period chose; the reserved test period appears nowhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from ..selection.redundancy import set_collinearity  # noqa: E402
from ..utils.paths import ensure_dir  # noqa: E402
from .feature_plots import _BLUE, _DIVERGING, COLOURS, _figure, _finish, _legend  # noqa: E402
from .ou_plots import BASELINE, INK, INK_SECONDARY, MUTED  # noqa: E402

__all__ = ["effective_rank_curve", "write_selection_figures"]

_KINDS = ("direction", "reversion", "volatility", "magnitude")


def _read(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path) if path.exists() else pl.DataFrame()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _short(target: str) -> str:
    return target.removeprefix("target_")


def _heat(ax: Any, fig: Any, grid: np.ndarray, *, cmap: Any, vmin: float, vmax: float,
          label: str) -> None:
    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025)
    cb.ax.tick_params(labelsize=7, colors=INK_SECONDARY)
    cb.set_label(label, color=INK_SECONDARY, fontsize=8)


# ---------------------------------------------------------------------------
# 1. Selection frequency
# ---------------------------------------------------------------------------
def plot_selection_frequency(out: Path, freq: pl.DataFrame, stability: pl.DataFrame,
                             threshold: float, tf: str) -> Path | None:
    if freq.is_empty() or "frequency_top_k" not in freq.columns:
        return None
    mean = freq.filter(~pl.col("is_probe")).group_by(["feature", "target"]).agg(
        pl.col("frequency").mean().alias("frequency"),
        pl.col("frequency_top_k").mean().alias("frequency_top_k"))
    top = (mean.group_by("feature").agg(pl.col("frequency").max().alias("best"))
           .filter(pl.col("best") > 0).sort(["best", "feature"], descending=[True, False]))
    feats = top["feature"].head(50).to_list()
    if not feats:
        return None
    targets = sorted(mean["target"].unique().to_list(),
                     key=lambda t: (_KINDS.index(_kind(t)) if _kind(t) in _KINDS else 9,
                                    int(t.rsplit("_", 1)[1])))
    grid = np.full((len(feats), len(targets)), np.nan)
    for r in mean.iter_rows(named=True):
        if r["feature"] in feats:
            grid[feats.index(r["feature"]), targets.index(r["target"])] = r["frequency"]
    depth = ({r["target"]: r["mean_selected"] for r in stability.group_by("target").agg(
        pl.col("mean_selected").mean()).iter_rows(named=True)}
        if "mean_selected" in stability.columns else {})
    fig, axes = _figure(width=10.5, height=0.2 * len(feats) + 2.4)
    ax = axes[0, 0]
    _heat(ax, fig, grid, cmap=_BLUE, vmin=0.0, vmax=1.0,
          label="probe-stopped selection frequency (mean of mRMR and Lasso)")
    ax.set_yticks(range(len(feats)), feats, fontsize=6)
    for tick, name in zip(ax.get_yticklabels(), feats, strict=True):
        stable = bool(np.nanmax(grid[feats.index(name)]) >= threshold)
        tick.set_color(INK if stable else MUTED)
    ax.set_xticks(range(len(targets)),
                  [f"{_short(t)}\n{depth.get(t, 0):.1f} before a probe" for t in targets],
                  rotation=90, fontsize=6)
    return _finish(fig, out / "selection_frequency.png",
                   f"{tf}: probe-stopped selection frequency, development quarter resamples\n"
                   f"(a run keeps the features entering before the first noise probe; dark "
                   f"labels reach {threshold:g} for some target)",
                   mean.filter(pl.col("feature").is_in(feats)).sort(["feature", "target"]))


def _kind(target: str) -> str:
    fam = target.removeprefix("target_").rsplit("_", 1)[0]
    return {"return": "direction", "residual_reduction": "reversion",
            "realized_vol": "volatility", "abs_return": "magnitude"}.get(fam, fam)


# ---------------------------------------------------------------------------
# 2. Dendrogram
# ---------------------------------------------------------------------------
def plot_dendrogram(out: Path, cache: Path, selected: list[str], threshold: float,
                    tf: str) -> Path | None:
    from scipy.cluster.hierarchy import dendrogram

    try:
        link = np.load(cache / "linkage.npy")
        names = json.loads((cache / "names.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    fig, axes = _figure(width=14, height=5.4)
    ax = axes[0, 0]
    cut = 1.0 - threshold
    with plt.rc_context({"lines.linewidth": 0.8}):
        tree = dendrogram(link, labels=names, ax=ax, color_threshold=cut,
                          above_threshold_color=MUTED, leaf_font_size=5)
    for tick in ax.get_xticklabels():
        chosen = tick.get_text() in selected
        tick.set_color(COLOURS[0] if chosen else INK_SECONDARY)
        tick.set_fontweight("bold" if chosen else "normal")
    ax.axhline(cut, color=COLOURS[1], lw=0.9, ls="--", label=f"cluster cut: |rho| = {threshold}")
    ax.set_ylabel("1 - |Spearman| (average linkage)")
    ax.grid(False)
    _legend(ax, loc="upper right")
    leaves = [names[i] for i in tree["leaves"]]
    return _finish(fig, out / "dendrogram.png",
                   f"{tf}: quality-universe dendrogram, development sample (blue, bold: the "
                   "Standard set)",
                   pl.DataFrame({"leaf_order": list(range(len(leaves))), "feature": leaves,
                                 "in_standard": [f in selected for f in leaves]}))


# ---------------------------------------------------------------------------
# 3. Feature count vs validation metric
# ---------------------------------------------------------------------------
def plot_feature_count(out: Path, curve: pl.DataFrame, plateau: dict[str, Any], tf: str
                       ) -> Path | None:
    if curve.is_empty():
        return None
    fig, axes = _figure(2, 2, width=11, height=7.0)
    for ax, kind in zip(axes.ravel(), _KINDS, strict=True):
        part = curve.filter(pl.col("kind") == kind)
        for i, h in enumerate(sorted(part["horizon"].unique().to_list())):
            p = part.filter(pl.col("horizon") == h).sort("k")
            ax.plot(p["k"], p["rank_ic"], marker="o", ms=4, lw=1.8, color=COLOURS[i],
                    label=f"h = {h}")
        for key, style in (("minimal_k", ":"), ("standard_k", "--")):
            if plateau.get(key):
                ax.axvline(plateau[key], color=BASELINE, lw=0.9, ls=style,
                           label=key.replace("_k", ""))
        ax.axhline(0, color=BASELINE, lw=0.7)
        ax.set_title(kind + (" (not used for the plateau: invariant 9)" if kind == "reversion"
                             else ""), fontsize=9)
        ax.set_xlabel("features (prefix of the development stability ranking)")
        ax.set_ylabel("validation rank IC (ridge)")
        _legend(ax, loc="lower right")
    return _finish(fig, out / "feature_count_curve.png",
                   f"{tf}: feature count vs validation rank IC (sets fixed on development)",
                   curve.select("k", "features", "target", "kind", "horizon", "rank_ic", "r2", "n"))


def plot_method_curves(out: Path, curve: pl.DataFrame, tf: str) -> Path | None:
    """Nested learning curves: mean evaluation rank IC over folds, per selection method."""
    if curve.is_empty():
        return None
    methods = ["ic_rank", "mrmr_ic_difference_1", "mrmr_mi_difference_1", "lasso",
               "elastic_net_0.1", "cluster_representatives"]
    part = (curve.filter(pl.col("method").is_in(methods) & pl.col("rank_ic").is_not_null())
            .with_columns(pl.col("target").map_elements(_kind, return_dtype=pl.Utf8)
                          .alias("kind"))
            .group_by(["kind", "method", "k"]).agg(pl.col("rank_ic").mean().alias("mean_ic"),
                                                   pl.len().alias("evaluations")))
    fig, axes = _figure(2, 2, width=11, height=7.0)
    for ax, kind in zip(axes.ravel(), _KINDS, strict=True):
        for i, m in enumerate(methods):
            p = part.filter((pl.col("kind") == kind) & (pl.col("method") == m)).sort("k")
            if p.height:
                ax.plot(p["k"], p["mean_ic"], marker="o", ms=3, lw=1.5, color=COLOURS[i], label=m)
        ax.axhline(0, color=BASELINE, lw=0.7)
        ax.set_xscale("log")
        ax.set_title(kind, fontsize=9)
        ax.set_xlabel("features selected inside the fold (log)")
        ax.set_ylabel("mean evaluation rank IC over folds and horizons")
        _legend(ax, loc="lower right", fontsize=6)
    return _finish(fig, out / "method_curves.png",
                   f"{tf}: nested chronological selection, evaluation rank IC by method",
                   part.sort(["kind", "method", "k"]))


# ---------------------------------------------------------------------------
# 4. Effective rank vs feature count
# ---------------------------------------------------------------------------
def effective_rank_curve(cache: Path, ranking: list[str]) -> pl.DataFrame:
    """Effective rank, condition number and max |rho| of every prefix of *ranking*."""
    try:
        rho = np.load(cache / "spearman.npy")
        names = json.loads((cache / "names.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return pl.DataFrame()
    idx = [names.index(f) for f in ranking if f in names]
    rows = []
    for k in range(1, len(idx) + 1):
        c = set_collinearity(rho, idx[:k], names)
        rows.append({"k": k, "feature": names[idx[k - 1]],
                     "effective_rank": c.get("effective_rank"),
                     "condition_number": c.get("condition_number"),
                     "max_vif": c.get("max_vif"), "max_abs_spearman": c.get("max_abs_spearman")})
    return pl.DataFrame(rows, infer_schema_length=None)


def plot_effective_rank(out: Path, curve: pl.DataFrame, plateau: dict[str, Any], tf: str
                        ) -> Path | None:
    if curve.is_empty():
        return None
    fig, axes = _figure(1, 2, width=11, height=4.2)
    ax = axes[0, 0]
    ax.plot(curve["k"], curve["k"], color=BASELINE, lw=0.9, ls="--",
            label="independent features (rank = k)")
    ax.plot(curve["k"], curve["effective_rank"], color=COLOURS[0], lw=1.8,
            label="effective rank (entropy of |rho| eigenvalues)")
    for key, style in (("minimal_k", ":"), ("standard_k", "--")):
        if plateau.get(key):
            ax.axvline(plateau[key], color=MUTED, lw=0.9, ls=style, label=key.replace("_k", ""))
    ax.set_xlabel("features (Extended ranking prefix)")
    ax.set_ylabel("effective rank")
    _legend(ax, loc="upper left")
    ax = axes[0, 1]
    ax.plot(curve["k"], curve["condition_number"], color=COLOURS[1], lw=1.8)
    ax.set_yscale("log")
    ax.set_xlabel("features (Extended ranking prefix)")
    ax.set_ylabel("condition number of the |Spearman| matrix (log)")
    return _finish(fig, out / "effective_rank.png",
                   f"{tf}: effective rank and conditioning vs feature count (development "
                   "sample, estimate)", curve)


# ---------------------------------------------------------------------------
# 5. Family composition
# ---------------------------------------------------------------------------
def plot_family_composition(out: Path, comp: pl.DataFrame, tf: str) -> Path | None:
    if comp.is_empty():
        return None
    fams = [c for c in comp.columns if c not in ("set", "features")]
    sets = comp["set"].to_list()
    grid = np.array([[float(r.get(f) or 0) for f in fams] for r in comp.iter_rows(named=True)])
    fig, axes = _figure(width=10, height=0.45 * len(sets) + 2.0)
    ax = axes[0, 0]
    _heat(ax, fig, grid, cmap=_BLUE, vmin=0.0, vmax=max(1.0, float(grid.max())),
          label="features")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j]:
                ax.text(j, i, f"{int(grid[i, j])}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if grid[i, j] > 0.6 * grid.max() else INK)
    ax.set_yticks(range(len(sets)),
                  [f"{s} ({int(n)})" for s, n in zip(sets, comp["features"], strict=True)],
                  fontsize=8)
    ax.set_xticks(range(len(fams)), fams, rotation=45, ha="right", fontsize=8)
    return _finish(fig, out / "family_composition.png",
                   f"{tf}: family composition of the candidate sets", comp)


# ---------------------------------------------------------------------------
# 6. Coefficient stability
# ---------------------------------------------------------------------------
def plot_coefficient_stability(out: Path, raw: pl.DataFrame, tf: str) -> Path | None:
    if raw.is_empty():
        return None
    ridge = raw.filter(pl.col("model") == "ridge")
    splits = sorted(ridge["split"].unique().to_list(),
                    key=lambda s: (s == "development_to_validation", s))
    fig, axes = _figure(2, 2, width=12, height=9.0)
    rows = []
    for ax, kind in zip(axes.ravel(), _KINDS, strict=True):
        targets = sorted({t for t in ridge["target"].unique().to_list() if _kind(t) == kind},
                         key=lambda t: int(t.rsplit("_", 1)[1]))
        if not targets:
            ax.set_visible(False)
            continue
        target = targets[len(targets) // 2]
        part = ridge.filter(pl.col("target") == target)
        feats = (part.group_by("feature").agg(pl.col("coefficient").abs().mean().alias("m"))
                 .sort(["m", "feature"], descending=[True, False])["feature"].head(20).to_list())
        for i, s in enumerate(splits):
            p = part.filter((pl.col("split") == s) & pl.col("feature").is_in(feats))
            ys = [feats.index(f) + (i - (len(splits) - 1) / 2) * 0.16 for f in p["feature"]]
            ax.scatter(p["coefficient"], ys, s=16, color=COLOURS[i % 6], label=s,
                       edgecolors="#ffffff", linewidths=0.4)
        ax.axvline(0, color=BASELINE, lw=0.8)
        ax.set_yticks(range(len(feats)), feats, fontsize=6)
        ax.invert_yaxis()
        ax.set_title(f"{kind}: {_short(target)} (ridge on the fold's first 20 mRMR features)",
                     fontsize=8)
        ax.set_xlabel("standardised coefficient")
        _legend(ax, loc="lower right", fontsize=6)
        rows.append(part)
    return _finish(fig, out / "coefficient_stability.png",
                   f"{tf}: coefficient signs across chronological folds (a feature absent from a "
                   "fold was not selected there)",
                   pl.concat(rows, how="diagonal_relaxed") if rows else None)


# ---------------------------------------------------------------------------
# 7. Selected feature yearly IC
# ---------------------------------------------------------------------------
def plot_selected_yearly_ic(out: Path, yearly: pl.DataFrame, standard: list[str],
                            dev: pl.DataFrame, tf: str) -> Path | None:
    if yearly.is_empty() or not standard:
        return None
    best: dict[str, str] = {}
    for f in standard:
        part = dev.filter((pl.col("feature") == f) & pl.col("passes")) if not dev.is_empty() \
            else dev
        if part.height:
            best[f] = str(part.sort(pl.col("ic").abs(), descending=True)["target"][0])
    keep = [f for f in standard if f in best]
    if not keep:
        return None
    wanted = pl.DataFrame({"feature": keep, "target": [best[f] for f in keep]})
    part = yearly.join(wanted, on=["feature", "target"])
    years = sorted(part["year"].unique().to_list())
    grid = np.full((len(keep), len(years)), np.nan)
    for r in part.iter_rows(named=True):
        grid[keep.index(r["feature"]), years.index(r["year"])] = r["ic"] if r["ic"] is not None \
            else np.nan
    fig, axes = _figure(width=11.5, height=0.3 * len(keep) + 1.8)
    ax = axes[0, 0]
    lim = float(np.nanmax(np.abs(grid))) if np.isfinite(grid).any() else 1.0
    _heat(ax, fig, grid, cmap=_DIVERGING, vmin=-lim, vmax=lim, label="rank IC in the year")
    ax.set_yticks(range(len(keep)), [f"{f} | {_short(best[f])}" for f in keep], fontsize=6)
    ax.set_xticks(range(len(years)), [str(y) for y in years], rotation=90, fontsize=7)
    val_years = sorted(part.filter(pl.col("period") == "validation")["year"].unique().to_list())
    if val_years:
        ax.axvline(years.index(val_years[0]) - 0.5, color=INK, lw=1.2)
        ax.text(years.index(val_years[0]) - 0.4, -0.8, "validation", fontsize=7, color=INK,
                va="bottom")
    return _finish(fig, out / "selected_yearly_ic.png",
                   f"{tf}: Standard set, yearly rank IC with each feature's best development "
                   "target (reserved years never loaded)", part.sort(["feature", "year"]))


# ---------------------------------------------------------------------------
# 8-9. PCA
# ---------------------------------------------------------------------------
def plot_pca_variance(out: Path, var: pl.DataFrame, tf: str) -> Path | None:
    if var.is_empty():
        return None
    cols = [c for c in var.columns if c.startswith("explained_")]
    ks = [int(c.split("_")[1]) for c in cols]
    fig, axes = _figure(width=8, height=4.2)
    ax = axes[0, 0]
    for i, r in enumerate(var.iter_rows(named=True)):
        ax.plot(ks, [r[c] for c in cols], marker="o", ms=4, lw=1.6, color=COLOURS[i % 6],
                label=f"{r['split']} ({r['components']} features)")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("principal components (fitted on the fold's training rows)")
    ax.set_ylabel("cumulative explained variance (rank design)")
    _legend(ax, loc="lower right")
    return _finish(fig, out / "pca_explained_variance.png",
                   f"{tf}: PCA explained variance by fold", var)


def plot_pca_vs_raw(out: Path, pca: pl.DataFrame, curve: pl.DataFrame, tf: str) -> Path | None:
    if pca.is_empty() or curve.is_empty():
        return None
    final = "development_to_validation"
    p = pca.filter(pl.col("split") == final)
    raw = curve.filter((pl.col("split") == final) & (pl.col("method") == "mrmr_ic_difference_1"))
    fig, axes = _figure(2, 2, width=11, height=7.0)
    rows = []
    for ax, kind in zip(axes.ravel(), _KINDS, strict=True):
        targets = sorted({t for t in p["target"].unique().to_list() if _kind(t) == kind},
                         key=lambda t: int(t.rsplit("_", 1)[1]))
        for i, t in enumerate(targets):
            a = p.filter(pl.col("target") == t).sort("k")
            b = raw.filter(pl.col("target") == t).sort("k")
            h = t.rsplit("_", 1)[1]
            ax.plot(a["k"], a["rank_ic"], marker="s", ms=3, lw=1.4, ls="--", color=COLOURS[i],
                    label=f"PCA, h = {h}")
            if b.height:
                ax.plot(b["k"], b["rank_ic"], marker="o", ms=3, lw=1.6, color=COLOURS[i],
                        label=f"raw mRMR, h = {h}")
            rows.append(a.select("target", "k", pl.lit("pca").alias("design"), "rank_ic"))
            rows.append(b.select("target", "k", pl.lit("raw_mrmr").alias("design"), "rank_ic"))
        ax.axhline(0, color=BASELINE, lw=0.7)
        ax.set_xscale("log")
        ax.set_title(kind, fontsize=9)
        ax.set_xlabel("components / features (log)")
        ax.set_ylabel("validation rank IC (ridge)")
        _legend(ax, loc="lower right", fontsize=6)
    return _finish(fig, out / "pca_vs_raw.png",
                   f"{tf}: PCA components vs raw features, development -> validation",
                   pl.concat(rows, how="diagonal_relaxed") if rows else None)


# ---------------------------------------------------------------------------
# 10. Candidate-set correlation matrix
# ---------------------------------------------------------------------------
def plot_set_correlation(out: Path, cache: Path, features: list[str], set_name: str, tf: str
                         ) -> Path | None:
    try:
        rho = np.load(cache / "spearman.npy")
        names = json.loads((cache / "names.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    feats = [f for f in features if f in names]
    if len(feats) < 2:
        return None
    idx = [names.index(f) for f in feats]
    grid = rho[np.ix_(idx, idx)]
    fig, axes = _figure(width=9.5, height=8.6)
    ax = axes[0, 0]
    _heat(ax, fig, grid, cmap=_DIVERGING, vmin=-1.0, vmax=1.0, label="Spearman rho")
    ax.set_xticks(range(len(feats)), feats, rotation=90, fontsize=6)
    ax.set_yticks(range(len(feats)), feats, fontsize=6)
    rows = [{"a": a, "b": b, "spearman": float(grid[i, j])}
            for i, a in enumerate(feats) for j, b in enumerate(feats) if i < j]
    return _finish(fig, out / f"{set_name}_set_correlation.png",
                   f"{tf}: {set_name} set, Spearman correlation (development sample, manifest "
                   "order)", pl.DataFrame(rows))


# ---------------------------------------------------------------------------
# All figures of one timeframe
# ---------------------------------------------------------------------------
def write_selection_figures(out_dir: Path, *, threshold: float = 0.6) -> list[str]:
    tf = out_dir.name
    plots = ensure_dir(out_dir / "plots")
    sets = _json(out_dir / "sets.json")
    plateau = sets.get("plateau") or {}
    standard = (sets.get("sets") or {}).get("standard") or []
    extended = (sets.get("sets") or {}).get("extended") or []
    manifest = _json(out_dir / "manifests" / "standard.json")
    ordered = [f["name"] for f in manifest.get("features", [])] or standard
    clusters = _json(out_dir / "redundancy_clusters.json")
    cache = out_dir / "cache"
    er = effective_rank_curve(cache, extended)
    written = [
        plot_selection_frequency(plots, _read(out_dir / "selection_frequency.parquet"),
                                 _read(out_dir / "selection_stability.parquet"), threshold, tf),
        plot_dendrogram(plots, cache, standard,
                        float(clusters.get("threshold_abs_spearman") or 0.9), tf),
        plot_feature_count(plots, _read(out_dir / "set_curve.parquet"), plateau, tf),
        plot_method_curves(plots, _read(out_dir / "feature_count_curve.parquet"), tf),
        plot_effective_rank(plots, er, plateau, tf),
        plot_family_composition(plots, _read(out_dir / "family_composition.parquet"), tf),
        plot_coefficient_stability(plots, _read(out_dir / "coefficient_stability_raw.parquet"),
                                   tf),
        plot_selected_yearly_ic(plots, _read(out_dir / "selected_yearly_ic.parquet"), standard,
                                _read(out_dir / "development_evidence.parquet"), tf),
        plot_pca_variance(plots, _read(out_dir / "pca_analysis.parquet"), tf),
        plot_pca_vs_raw(plots, _read(out_dir / "pca_predictive.parquet"),
                        _read(out_dir / "feature_count_curve.parquet"), tf),
        plot_set_correlation(plots, cache, ordered, "standard", tf),
    ]
    return [p.name for p in written if p is not None]
