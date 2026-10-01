"""Figures of the predictive feature research (Prompt #8, Step 76).

Every figure is a view of a table written by :mod:`.feature_research` /
:mod:`.feature_reports`, and the plotted numbers are written beside it as a
CSV. Categories (statuses, target kinds, features) take the project's
categorical order (blue, orange, aqua, yellow, magenta, green), never cycled
past six: a figure that would need more shows the six features with the best
status and evidence ("top" is a display choice, not a selection). Magnitudes
use one blue ramp, signed ICs the blue-gray-red diverging pair around zero.
One y-scale per axis; two measures are two panels.
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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from ..alpha.ranking import status_rank  # noqa: E402
from ..utils.paths import ensure_dir  # noqa: E402
from .ou_plots import BASELINE, GRID, INK, INK_SECONDARY, MUTED, SURFACE  # noqa: E402

__all__ = ["write_feature_figures"]

COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
STATUS_COLOURS = {"strong_candidate": COLOURS[0], "candidate": COLOURS[2],
                  "weak_candidate": COLOURS[3], "redundant": COLOURS[4],
                  "unstable": COLOURS[1], "failed_null": MUTED}
KIND_COLOURS = {"direction": COLOURS[0], "magnitude": COLOURS[1], "volatility": COLOURS[2],
                "residual": COLOURS[3], "excursion": COLOURS[4]}
_BLUE = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#f0efec", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"])
_DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8f8e", "#b52f2f"])
_DPI = 130


def _figure(rows: int = 1, cols: int = 1, *, width: float = 10.0, height: float = 4.2
            ) -> tuple[Any, Any]:
    fig, axes = plt.subplots(rows, cols, figsize=(width, height), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.ravel():
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for name, spine in ax.spines.items():
            spine.set_visible(name in ("left", "bottom"))
            spine.set_color(BASELINE)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8)
        ax.xaxis.label.set_color(INK_SECONDARY)
        ax.yaxis.label.set_color(INK_SECONDARY)
        ax.title.set_color(INK)
    return fig, axes


def _legend(ax: Any, **kwargs: Any) -> None:
    handles, _ = ax.get_legend_handles_labels()
    if not handles:
        return
    legend = ax.legend(frameon=False, **({"loc": "best", "fontsize": 7} | kwargs))
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)


def _finish(fig: Any, path: Path, title: str, data: pl.DataFrame | None = None) -> Path:
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=_DPI, facecolor=SURFACE)
    plt.close(fig)
    if data is not None and not data.is_empty():
        listy = [c for c, d in data.schema.items() if isinstance(d, (pl.List, pl.Struct))]
        if listy:
            data = data.with_columns([pl.col(c).map_elements(
                lambda v: json.dumps(v.to_list() if hasattr(v, "to_list") else v),
                return_dtype=pl.Utf8) for c in listy])
        data.write_csv(path.with_suffix(".csv"))
    return path


def _read(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path) if path.exists() else pl.DataFrame()


def _top(status: pl.DataFrame, k: int) -> list[dict[str, Any]]:
    """The *k* features with the best status, then the largest |rank IC| (display only)."""
    rows = [r for r in status.iter_rows(named=True)
            if r["status"] in ("strong_candidate", "candidate", "weak_candidate")]
    rows.sort(key=lambda r: (status_rank(r["status"]), -(r.get("best_abs_rank_ic") or 0.0)))
    return rows[:k]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_ic_decay(out: Path, top: list[dict[str, Any]], ic: pl.DataFrame, tf: str) -> Path | None:
    if not top or ic.is_empty():
        return None
    fig, axes = _figure(1, 2)
    rows = []
    for i, r in enumerate(top[:6]):
        fam = str(r["best_target"]).rsplit("_", 1)[0]
        part = ic.filter((pl.col("feature") == r["feature"]) & (pl.col("method") == "spearman")
                         & pl.col("target").str.starts_with(fam + "_")).sort("horizon")
        if part.height:
            axes[0, 0].plot(part["horizon"], part["ic"], marker="o", ms=3, lw=1.6,
                            color=COLOURS[i], label=f"{r['feature']} ({fam.removeprefix('target_')})")
            rows.append(part.select("feature", "target", "horizon", "ic", "se"))
        marg = ("marginal_abs_return" if r["best_kind"] in ("magnitude", "volatility", "excursion")
                else "marginal_return")
        part = ic.filter((pl.col("feature") == r["feature"]) & (pl.col("method") == "spearman")
                         & pl.col("target").str.starts_with(marg)).sort("horizon")
        if part.height:
            axes[0, 1].plot(part["horizon"], part["ic"], marker="o", ms=3, lw=1.6,
                            color=COLOURS[i], label=f"{r['feature']} ({marg})")
            rows.append(part.select("feature", "target", "horizon", "ic", "se"))
    for ax, xl in ((axes[0, 0], "horizon h (bars): cumulative target"),
                   (axes[0, 1], "lag k (bars): one-bar outcome k bars ahead")):
        ax.axhline(0, color=BASELINE, lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel("rank IC")
        _legend(ax)
    return _finish(fig, out / "ic_decay.png", f"{tf}: IC decay of the leading candidates",
                   pl.concat(rows) if rows else None)


def plot_yearly_ic(out: Path, top: list[dict[str, Any]], yearly: pl.DataFrame, tf: str
                   ) -> Path | None:
    if not top or yearly.is_empty():
        return None
    keep = [(r["feature"], r["best_target"]) for r in top[:12]]
    wanted = pl.DataFrame(keep, schema={"feature": pl.Utf8, "target": pl.Utf8}, orient="row")
    part = yearly.filter(pl.col("method") == "spearman").join(wanted, on=["feature", "target"])
    if part.is_empty():
        return None
    years = sorted(part["year"].unique().to_list())
    labels = [f"{f} | {t.removeprefix('target_')}" for f, t in keep]
    grid = np.full((len(keep), len(years)), np.nan)
    for r in part.iter_rows(named=True):
        grid[keep.index((r["feature"], r["target"])), years.index(r["year"])] = r["ic"]
    fig, axes = _figure(width=11, height=0.35 * len(keep) + 1.6)
    ax = axes[0, 0]
    lim = float(np.nanmax(np.abs(grid))) if np.isfinite(grid).any() else 1.0
    im = ax.imshow(grid, aspect="auto", cmap=_DIVERGING, vmin=-lim, vmax=lim,
                   interpolation="nearest")
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xticks(range(len(years)), [str(y) for y in years], rotation=90, fontsize=7)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025)
    cb.ax.tick_params(labelsize=7, colors=INK_SECONDARY)
    cb.set_label("rank IC in the year", color=INK_SECONDARY, fontsize=8)
    return _finish(fig, out / "yearly_ic.png", f"{tf}: yearly rank IC, leading candidates "
                   "(best target)", part.select("feature", "target", "year", "ic", "n"))


def plot_rolling_ic(out: Path, top: list[dict[str, Any]], rolling: pl.DataFrame, tf: str
                    ) -> Path | None:
    if not top or rolling.is_empty():
        return None
    fig, axes = _figure(height=4.6)
    ax = axes[0, 0]
    rows = []
    if "method" in rolling.columns:
        rolling = rolling.filter(pl.col("method") == "spearman")
    for i, r in enumerate(top[:6]):
        part = rolling.filter((pl.col("feature") == r["feature"])
                              & (pl.col("target") == r["best_target"])).sort("month")
        if part.is_empty():
            continue
        x = np.arange(part.height)
        ax.plot(x, part["rolling_ic"], lw=1.4, color=COLOURS[i],
                label=f"{r['feature']} | {str(r['best_target']).removeprefix('target_')}")
        rows.append(part)
        ticks = x[:: max(1, part.height // 10)]
        ax.set_xticks(ticks, [part["month"][int(j)] for j in ticks], rotation=45, fontsize=7)
    ax.axhline(0, color=BASELINE, lw=0.8)
    ax.set_ylabel("rank IC over the trailing 24 months")
    _legend(ax, loc="upper left")
    return _finish(fig, out / "rolling_ic.png", f"{tf}: causal rolling IC (window ends the "
                   "month before the date)", pl.concat(rows) if rows else None)


def plot_deciles(out: Path, top: list[dict[str, Any]], curves: pl.DataFrame, tf: str
                 ) -> Path | None:
    if not top or curves.is_empty():
        return None
    fig, axes = _figure(2, 3, width=12, height=6.4)
    rows = []
    for i, r in enumerate(top[:6]):
        ax = axes.ravel()[i]
        part = curves.filter(pl.col("feature") == r["feature"])
        if part.is_empty():
            ax.set_visible(False)
            continue
        target = (r["best_target"] if part.filter(pl.col("target") == r["best_target"]).height
                  else part.sort("target")["target"][0])
        part = part.filter(pl.col("target") == target).sort("decile")
        ax.errorbar(part["decile"], part["mean"],
                    yerr=[part["mean"] - part["ci_low"], part["ci_high"] - part["mean"]],
                    color=COLOURS[0], marker="o", ms=3, lw=1.4, capsize=2)
        ax.set_title(f"{r['feature']}\n{str(target).removeprefix('target_')}", fontsize=8)
        ax.set_xlabel("feature decile (full-sample edges, descriptive)")
        ax.set_ylabel("mean outcome")
        rows.append(part)
    for j in range(len(top[:6]), 6):
        axes.ravel()[j].set_visible(False)
    return _finish(fig, out / "deciles.png", f"{tf}: outcome by feature decile (95 % CI)",
                   pl.concat(rows, how="diagonal_relaxed") if rows else None)


def plot_stability(out: Path, status: pl.DataFrame, tf: str) -> Path | None:
    part = status.filter(pl.col("best_abs_rank_ic").is_not_null()
                         & pl.col("sign_consistency").is_not_null())
    if part.is_empty():
        return None
    fig, axes = _figure(height=4.8)
    ax = axes[0, 0]
    for st, colour in STATUS_COLOURS.items():
        p = part.filter(pl.col("status") == st)
        if p.height:
            ax.scatter(p["best_abs_rank_ic"], p["sign_consistency"], s=18, color=colour,
                       label=f"{st} ({p.height})", edgecolors=SURFACE, linewidths=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("|rank IC| of the best target (log)")
    ax.set_ylabel("share of years with the pooled sign")
    _legend(ax, loc="lower right")
    return _finish(fig, out / "stability.png", f"{tf}: effect size vs yearly sign stability",
                   part.select("feature", "status", "best_kind", "best_target",
                               "best_abs_rank_ic", "sign_consistency"))


def plot_dendrogram_and_clusters(out: Path, cache: Path, tf: str) -> list[Path]:
    from scipy.cluster.hierarchy import dendrogram

    try:
        link = np.load(cache / "linkage.npy")
        rho = np.load(cache / "spearman.npy")
        names = json.loads((cache / "names.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    paths = []
    fig, axes = _figure(width=14, height=5.2)
    ax = axes[0, 0]
    with plt.rc_context({"lines.linewidth": 0.8}):
        tree = dendrogram(link, labels=names, ax=ax, color_threshold=0.1,
                          above_threshold_color=MUTED, leaf_font_size=5)
    ax.axhline(0.1, color=COLOURS[1], lw=0.9, ls="--", label="cut: |rho| = 0.9")
    ax.set_ylabel("1 - |Spearman| (average linkage)")
    ax.grid(False)
    _legend(ax, loc="upper right")
    paths.append(_finish(fig, out / "dendrogram.png", f"{tf}: feature dendrogram (estimate, "
                         "evenly spaced sample)"))
    order = tree["leaves"]
    grid = np.abs(rho[np.ix_(order, order)])
    fig, axes = _figure(width=10, height=9.2)
    ax = axes[0, 0]
    im = ax.imshow(grid, cmap=_BLUE, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(order)), [names[i] for i in order], rotation=90, fontsize=4)
    ax.set_yticks(range(len(order)), [names[i] for i in order], fontsize=4)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03)
    cb.ax.tick_params(labelsize=7, colors=INK_SECONDARY)
    cb.set_label("|Spearman|", color=INK_SECONDARY, fontsize=8)
    paths.append(_finish(fig, out / "cluster_map.png", f"{tf}: |Spearman| between features, "
                         "dendrogram order"))
    return paths


def plot_null_distribution(out: Path, maxima: pl.DataFrame, nulls: pl.DataFrame, tf: str
                           ) -> Path | None:
    if maxima.is_empty() or nulls.is_empty():
        return None
    fig, axes = _figure(height=4.2)
    ax = axes[0, 0]
    vals = maxima["max_abs_rank_ic0_over_all_pairs"].to_numpy()
    ax.hist(vals, bins=30, color=COLOURS[0], alpha=0.85, label="null: largest |IC| over every "
            "feature x target in one circular shift")
    real = (nulls.select(pl.col("rank_ic0").abs().alias("a")).sort("a", descending=True)
            ["a"].head(5).to_list())
    for j, v in enumerate(real):
        ax.axvline(v, color=COLOURS[1], lw=1.0, ls="-" if j == 0 else ":",
                   label="largest real |IC| (top 5)" if j == 0 else None)
    ax.set_xlabel("|rank IC| (missing = 0 estimator)")
    ax.set_ylabel("shift draws")
    _legend(ax)
    return _finish(fig, out / "null_ic_distribution.png",
                   f"{tf}: what the largest IC looks like with no information", maxima)


def plot_real_vs_null(out: Path, kind_null: pl.DataFrame, tf: str) -> Path | None:
    if kind_null.is_empty():
        return None
    fig, axes = _figure(width=7.5, height=6.2)
    ax = axes[0, 0]
    for kind, colour in KIND_COLOURS.items():
        p = kind_null.filter(pl.col("kind") == kind)
        if p.height:
            ax.scatter(p["max_null_q"], p["best_studentized"], s=14, color=colour,
                       label=kind, edgecolors=SURFACE, linewidths=0.5)
    lim = max(float(np.nanmax(kind_null["max_null_q"].to_numpy())), 1.0)
    ax.plot([0, lim], [0, lim], color=BASELINE, lw=1.0, label="real = null quantile")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("99 % quantile of the feature's max-T shift null")
    ax.set_ylabel("best studentized |IC| of the kind")
    _legend(ax, loc="upper left")
    return _finish(fig, out / "real_vs_null.png", f"{tf}: real vs max-T circular-shift null "
                   "(above the line: beyond the null)", kind_null)


def plot_parameter_surface(out: Path, surface: pl.DataFrame, tf: str) -> Path | None:
    if surface.is_empty():
        return None
    targets = ("target_realized_vol_20", "target_return_5")
    fig, axes = _figure(1, 2, height=4.4)
    rows = []
    for a_i, target in enumerate(targets):
        ax = axes[0, a_i]
        part = surface.filter((pl.col("target") == target) & pl.col("window").is_not_null())
        fams = (part.group_by("parameter_family").len().filter(pl.col("len") > 1)
                .sort("parameter_family")["parameter_family"].to_list())[:6]
        for i, fam in enumerate(fams):
            p = part.filter(pl.col("parameter_family") == fam).sort("window")
            ax.plot(p["window"], p["ic"], marker="o", ms=3, lw=1.4, color=COLOURS[i], label=fam)
            rows.append(p)
        ax.axhline(0, color=BASELINE, lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("window (bars)")
        ax.set_ylabel(f"rank IC with {target.removeprefix('target_')}")
        _legend(ax, fontsize=6)
    return _finish(fig, out / "parameter_surface.png", f"{tf}: IC across each parameter "
                   "family's windows (first six families)",
                   pl.concat(rows, how="diagonal_relaxed") if rows else None)


def plot_family_coverage(out: Path, status: pl.DataFrame, tf: str) -> Path | None:
    if status.is_empty():
        return None
    counts = status.group_by(["family", "status"]).len()
    fams = sorted(counts["family"].unique().to_list())
    fig, axes = _figure(height=0.32 * len(fams) + 1.8)
    ax = axes[0, 0]
    left = np.zeros(len(fams))
    for st, colour in STATUS_COLOURS.items():
        vals = np.array([counts.filter((pl.col("family") == f) & (pl.col("status") == st))
                         ["len"].sum() for f in fams], dtype=float)
        if vals.sum() == 0:
            continue
        ax.barh(fams, vals, left=left, color=colour, label=st, edgecolor=SURFACE, linewidth=1.5)
        left += vals
    ax.set_xlabel("features")
    ax.invert_yaxis()
    _legend(ax, loc="lower right")
    return _finish(fig, out / "family_coverage.png", f"{tf}: status by feature family",
                   counts.sort(["family", "status"]))


def plot_mechanical(out: Path, evidence: pl.DataFrame, tf: str) -> Path | None:
    """Residual targets: the real IC against the random-walk pipeline null's (invariant 9)."""
    col = "pipeline_random_walk_ic"
    if evidence.is_empty() or col not in evidence.columns:
        return None
    part = evidence.filter((pl.col("kind") == "residual") & pl.col(col).is_not_null()
                           & pl.col("value").is_not_null())
    if part.is_empty():
        return None
    best = part.sort(pl.col("value").abs(), descending=True).group_by(
        "feature", maintain_order=True).first()
    fig, axes = _figure(width=7.5, height=6.2)
    ax = axes[0, 0]
    for flag, colour, label in ((True, COLOURS[0], "beyond the pipeline nulls"),
                                (False, COLOURS[1], "reproduced by a null (mechanical)")):
        p = best.filter(pl.col("beyond_pipeline_null") == flag)
        if p.height:
            ax.scatter(p["value"], p[col], s=16, color=colour, label=f"{label} ({p.height})",
                       edgecolors=SURFACE, linewidths=0.5)
    lim = float(best.select(pl.max_horizontal(pl.col("value").abs(), pl.col(col).abs()))
                .max().item() or 1.0)
    ax.plot([-lim, lim], [-lim, lim], color=BASELINE, lw=1.0, label="null = real")
    ax.axhline(0, color=GRID, lw=0.8)
    ax.axvline(0, color=GRID, lw=0.8)
    ax.set_xlabel("rank IC with a residual target (real XAUUSD)")
    ax.set_ylabel("same feature and target on a random walk")
    _legend(ax, loc="upper left")
    return _finish(fig, out / "residual_real_vs_random_walk.png",
                   f"{tf}: residual predictability vs the random-walk pipeline null",
                   best.select("feature", "target", "value", col, "beyond_pipeline_null",
                               "mechanical_share"))


def write_feature_figures(out_dir: Path, *, top_features: int = 12) -> list[str]:
    """Every figure of one timeframe's research, into ``<out_dir>/figures/``."""
    tf = out_dir.name
    figs = ensure_dir(out_dir / "figures")
    status = _read(out_dir / "feature_status.parquet")
    if status.is_empty():
        return []
    top = _top(status, top_features)
    written: list[Path | None] = [
        plot_ic_decay(figs, top, _read(out_dir / "ic/feature_ic.parquet"), tf),
        plot_yearly_ic(figs, top, _read(out_dir / "ic/yearly_ic.parquet"), tf),
        plot_rolling_ic(figs, top, _read(out_dir / "ic/rolling_ic.parquet"), tf),
        plot_deciles(figs, top, _read(out_dir / "deciles/decile_curves.parquet"), tf),
        plot_stability(figs, status, tf),
        plot_null_distribution(figs, _read(out_dir / "nulls/shift_null_maxima.parquet"),
                               _read(out_dir / "nulls/null_tests.parquet"), tf),
        plot_real_vs_null(figs, _read(out_dir / "nulls/kind_max_null.parquet"), tf),
        plot_parameter_surface(figs, _read(out_dir / "robustness/parameter_surface.parquet"),
                               tf),
        plot_family_coverage(figs, status, tf),
        plot_mechanical(figs, _read(out_dir / "feature_evidence.parquet"), tf),
    ]
    written.extend(plot_dendrogram_and_clusters(figs, out_dir / "redundancy/cache", tf))
    return [p.name for p in written if p is not None]
