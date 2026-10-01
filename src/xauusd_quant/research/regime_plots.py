"""Figures for the regime study (Step 66). Every figure is a view of a table written beside it.

States are *identities*, so they take the project's categorical order
(slots 1-6 of the reference palette, fixed order, never cycled: blue,
orange, aqua, yellow, magenta, green - the documented adjacent-pair
validated order; unchanged hex, so no re-validation was needed and none was
possible here without node). Three slots sit below 3:1 contrast on the light
surface, so every figure has a legend and its numbers are in the table beside
it. Magnitudes use one blue ramp (transition probabilities, contingency
shares); signed deviations (standardised state medians) use the blue-red
diverging pair around a gray midpoint. One y-scale per axis, always:
two measures are two panels. Offline figures carry the offline label in
their title; long histories are shown as selected slices or aggregates, never
23 years of bars at once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from ..regimes.config import RegimeConfig  # noqa: E402
from ..utils.paths import ensure_dir  # noqa: E402
from .ou_plots import BASELINE, GRID, INK, INK_SECONDARY, SURFACE  # noqa: E402

__all__ = [
    "STATE_COLOURS",
    "plot_duration_distribution",
    "plot_entropy",
    "plot_feature_profiles",
    "plot_model_selection",
    "plot_refit_stability",
    "plot_regime_vs_volatility",
    "plot_state_outcomes",
    "plot_state_probabilities",
    "plot_state_timeline",
    "plot_transition_matrix",
    "plot_yearly_frequency",
]

#: Categorical slots 1-6 of the reference palette, in their fixed order.
STATE_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
_BLUE = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
_DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8f8e", "#b52f2f"])
_MODEL_STYLE = {"kmeans": ("#e87ba4", ":"), "gmm_diag": ("#eda100", "-."),
                "gmm": ("#eb6834", "--"), "hmm": ("#2a78d6", "-")}


def _colour(state: int) -> str:
    return STATE_COLOURS[state % len(STATE_COLOURS)]


def _figure(config: RegimeConfig, rows: int = 1, cols: int = 1, *,
            height: float | None = None, width: float | None = None) -> tuple[Any, Any]:
    w, base = config.plots.figsize
    fig, axes = plt.subplots(rows, cols, figsize=(width or w, height or base), squeeze=False)
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
    return fig, axes


def _legend(ax: Any, **kwargs: Any) -> None:
    legend = ax.legend(fontsize=8, frameon=False, **({"loc": "best"} | kwargs))
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)


def _finish(fig: Any, path: Path, config: RegimeConfig, title: str) -> Path:
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=config.plots.dpi, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_state_timeline(times: np.ndarray, log_price: np.ndarray, states: np.ndarray, *,
                        k: int, path: Path, config: RegimeConfig, title: str) -> Path:
    """Log price over one slice with the inferred state as the background colour."""
    fig, axes = _figure(config)
    ax = axes[0, 0]
    lo, hi = np.nanmin(log_price), np.nanmax(log_price)
    pad = 0.05 * (hi - lo or 1.0)
    for s in range(k):
        ax.fill_between(times, lo - pad, hi + pad, where=states == s, step="post",
                        color=_colour(s), alpha=0.22, linewidth=0, label=f"state_{s}")
    ax.plot(times, log_price, color=INK, linewidth=1.0, label="ln mid price")
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_ylabel("ln price")
    _legend(ax, loc="upper left", ncol=min(k + 1, 4))
    return _finish(fig, path, config, title)


def plot_state_probabilities(times: np.ndarray, probs: np.ndarray, *, path: Path,
                             config: RegimeConfig, title: str) -> Path:
    """Stacked P(S_t = k | X_<=t) over one slice (causal filter)."""
    fig, axes = _figure(config)
    ax = axes[0, 0]
    k = probs.shape[1]
    ax.stackplot(times, probs.T, colors=[_colour(s) for s in range(k)],
                 labels=[f"state_{s}" for s in range(k)], edgecolor=SURFACE, linewidth=0.3)
    ax.set_ylim(0, 1)
    ax.set_ylabel("filtered probability")
    _legend(ax, loc="upper left", ncol=min(k, 6))
    return _finish(fig, path, config, title)


def _annotated_heatmap(ax: Any, values: np.ndarray, *, cmap: Any, vmin: float, vmax: float,
                       fmt: str, xlabels: list[str], ylabels: list[str]) -> Any:
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=45, ha="right")
    ax.set_yticks(range(len(ylabels)), ylabels)
    mid = 0.5 * (vmin + vmax)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isfinite(v):
                dark = (v > mid) if cmap is _BLUE else abs(v) > 0.6 * max(abs(vmin), abs(vmax))
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=7,
                        color="#ffffff" if dark else INK)
    return image


def plot_transition_matrix(transition: np.ndarray, *, path: Path, config: RegimeConfig,
                           title: str) -> Path:
    k = transition.shape[0]
    fig, axes = _figure(config, width=4 + 0.9 * k, height=3 + 0.8 * k)
    ax = axes[0, 0]
    image = _annotated_heatmap(ax, transition, cmap=_BLUE, vmin=0.0, vmax=1.0, fmt=".3f",
                               xlabels=[f"to state_{j}" for j in range(k)],
                               ylabels=[f"from state_{i}" for i in range(k)])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
    return _finish(fig, path, config, title)


def plot_feature_profiles(standardized: pl.DataFrame, *, path: Path, config: RegimeConfig,
                          title: str) -> Path:
    """Standardised state medians (feature x state), diverging around the overall median."""
    features = standardized["feature"].unique(maintain_order=True).to_list()
    states = sorted(standardized["state"].unique().to_list())
    values = np.full((len(features), len(states)), np.nan)
    for r in standardized.iter_rows(named=True):
        values[features.index(r["feature"]), states.index(r["state"])] = r["standardized_median"]
    limit = float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 1.0
    limit = min(max(limit, 0.5), 3.0)
    fig, axes = _figure(config, width=4 + 0.9 * len(states), height=2 + 0.35 * len(features))
    ax = axes[0, 0]
    image = _annotated_heatmap(ax, np.clip(values, -limit, limit), cmap=_DIVERGING,
                               vmin=-limit, vmax=limit, fmt="+.2f",
                               xlabels=[f"state_{s}" for s in states], ylabels=features)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04,
                 label="(state median - overall median) / IQR").ax.tick_params(labelsize=7)
    return _finish(fig, path, config, title)


def plot_duration_distribution(runs: pl.DataFrame, *, k: int, path: Path, config: RegimeConfig,
                               title: str) -> Path:
    """Survival of uncensored run lengths per state (log x): P(run >= d)."""
    fig, axes = _figure(config)
    ax = axes[0, 0]
    for s in range(k):
        lengths = np.sort(runs.filter((pl.col("state") == s) & ~pl.col("censored"))[
            "length"].to_numpy())
        if lengths.size:
            survival = 1.0 - np.arange(lengths.size) / lengths.size
            ax.step(lengths, survival, where="post", color=_colour(s), linewidth=2,
                    label=f"state_{s} (median {np.median(lengths):.0f} bars)")
    ax.set_xscale("log")
    ax.set_xlabel("run length (bars)")
    ax.set_ylabel("share of runs at least this long")
    _legend(ax)
    return _finish(fig, path, config, title)


def plot_yearly_frequency(frequency: pl.DataFrame, *, k: int, path: Path, config: RegimeConfig,
                          title: str) -> Path:
    fig, axes = _figure(config)
    ax = axes[0, 0]
    periods = frequency["period"].to_list()
    bottom = np.zeros(len(periods))
    x = np.arange(len(periods))
    for s in range(k):
        column = f"state_{s}_share"
        if column not in frequency.columns:
            continue
        share = frequency[column].to_numpy()
        ax.bar(x, share, bottom=bottom, color=_colour(s), edgecolor=SURFACE, linewidth=1.0,
               width=0.8, label=f"state_{s}")
        bottom += share
    ax.set_xticks(x, periods, rotation=90, fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of bars")
    _legend(ax, loc="upper left", ncol=min(k, 6), bbox_to_anchor=(0, 1.12))
    return _finish(fig, path, config, title)


def plot_entropy(entropy: np.ndarray, k: int, *, path: Path, config: RegimeConfig,
                 title: str) -> Path:
    fig, axes = _figure(config)
    ax = axes[0, 0]
    values = entropy[np.isfinite(entropy)]
    ax.hist(values, bins=60, range=(0, np.log(k)), color=STATE_COLOURS[0],
            edgecolor=SURFACE, linewidth=0.5)
    ax.set_yscale("log")
    ax.set_xlabel(f"entropy of the state probabilities (nats; max ln {k} = {np.log(k):.2f})")
    ax.set_ylabel("bars (log scale)")
    return _finish(fig, path, config, title)


def plot_regime_vs_volatility(table: np.ndarray, *, path: Path, config: RegimeConfig,
                              title: str) -> Path:
    """Share of each state's bars in each volatility bucket (rows sum to 1)."""
    k, b = table.shape
    fig, axes = _figure(config, width=4 + 0.9 * b, height=3 + 0.7 * k)
    ax = axes[0, 0]
    rows = table / np.maximum(table.sum(axis=1, keepdims=True), 1)
    image = _annotated_heatmap(ax, rows, cmap=_BLUE, vmin=0.0, vmax=1.0, fmt=".2f",
                               xlabels=[f"vol bucket {j + 1}" for j in range(b)],
                               ylabels=[f"state_{i}" for i in range(k)])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
    return _finish(fig, path, config, title)


def plot_state_outcomes(table: pl.DataFrame, *, horizon: int, path: Path, config: RegimeConfig,
                        title: str) -> Path:
    """P(residual shrinks | |Z| > z) by group for regimes, volatility buckets and random labels."""
    part = table.filter((pl.col("horizon") == horizon) & (pl.col("within") == "all"))
    labelings = part["labeling"].unique(maintain_order=True).to_list()
    fig, axes = _figure(config, 1, len(labelings), width=4 * max(len(labelings), 1))
    for ax, labeling in zip(axes[0], labelings, strict=False):
        rows = part.filter(pl.col("labeling") == labeling).sort("group")
        groups = rows["group"].to_numpy()
        p = rows["prob_shrinks"].cast(pl.Float64).to_numpy()
        se = rows["std_error"].cast(pl.Float64).fill_null(0.0).to_numpy()
        colours = [_colour(int(g)) if labeling == "regime" else INK_SECONDARY for g in groups]
        ax.bar(groups, p, color=colours, edgecolor=SURFACE, linewidth=1.0, width=0.7)
        ax.errorbar(groups, p, yerr=2 * se, fmt="none", ecolor=INK, elinewidth=1.0, capsize=3)
        ax.set_title(labeling, fontsize=9, color=INK_SECONDARY)
        ax.set_xticks(groups, [str(int(g)) for g in groups])
        finite = p[np.isfinite(p)]
        if finite.size:
            ax.set_ylim(max(0.0, finite.min() - 0.1), min(1.0, finite.max() + 0.1))
        ax.set_xlabel("group")
    axes[0, 0].set_ylabel("P(|resid| smaller h bars later), +-2 SE")
    return _finish(fig, path, config, title)


def plot_refit_stability(refits: pl.DataFrame, *, path: Path, config: RegimeConfig,
                         title: str) -> Path:
    """Two panels (never two y-scales): refit overlap ARI, and the largest state drift."""
    fitted = refits.filter(pl.col("status") == "fitted") if "status" in refits.columns else refits
    fig, axes = _figure(config, 2, 1, height=7)
    x = np.array([str(v)[:7] for v in fitted["infer_start"].to_list()])
    idx = np.arange(x.size)
    if "overlap_ari" in fitted.columns:
        axes[0, 0].plot(idx, fitted["overlap_ari"].cast(pl.Float64).to_numpy(),
                        color=STATE_COLOURS[0], linewidth=2)
    axes[0, 0].set_ylim(-0.05, 1.05)
    axes[0, 0].set_ylabel("ARI previous vs new model")
    column = ("drift_max_bhattacharyya" if "drift_max_bhattacharyya" in fitted.columns
              else "drift_max_mean_shift_scaled")
    if column in fitted.columns:
        axes[1, 0].plot(idx, fitted[column].cast(pl.Float64).to_numpy(), color=STATE_COLOURS[1],
                        linewidth=2)
    axes[1, 0].set_ylabel(column.replace("drift_", "").replace("_", " "))
    step = max(1, x.size // 20)
    for ax in axes[:, 0]:
        ax.set_xticks(idx[::step], x[::step], rotation=90, fontsize=7)
    return _finish(fig, path, config, title)


def plot_model_selection(table: pl.DataFrame, *, metric: str, ylabel: str, path: Path,
                         config: RegimeConfig, title: str) -> Path:
    """One line per model family across K (a lower BIC / higher likelihood is better)."""
    fig, axes = _figure(config)
    ax = axes[0, 0]
    for model, part in table.group_by("model", maintain_order=True):
        name = str(model[0])
        rows = part.filter(pl.col(metric).is_not_null()).sort("states")
        if rows.is_empty():
            continue
        colour, style = _MODEL_STYLE.get(name, (INK_SECONDARY, "-"))
        ax.plot(rows["states"].to_numpy(), rows[metric].cast(pl.Float64).to_numpy(),
                color=colour, linestyle=style, marker="o", markersize=5, linewidth=2, label=name)
    ax.set_xlabel("number of states K")
    ax.set_ylabel(ylabel)
    _legend(ax)
    return _finish(fig, path, config, title)
