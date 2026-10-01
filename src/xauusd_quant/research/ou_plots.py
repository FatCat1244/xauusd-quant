"""Figures for the Ornstein-Uhlenbeck research.

Every figure is a view of a table written beside it, so nothing exists only
as pixels. Colours follow the entity, never its rank, and are the validated
reference palette (first three categorical slots pass every colour-vision
check across all pairs):

* the real XAUUSD residual - blue
* the detrended random-walk control - orange
* the exact-OU reference - aqua

One y-axis per panel, always: two quantities of different scale go in two
panels, never on a twin axis. Reference lines (``b = 1``, ``|Z| = 2``) are
visual aids, not thresholds, entries or exits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy import stats  # noqa: E402

from ..models.config import OUConfig  # noqa: E402
from ..utils.paths import ensure_dir  # noqa: E402

__all__ = [
    "plot_estimated_vs_realized",
    "plot_half_life_distribution",
    "plot_innovation_acf",
    "plot_innovation_distribution",
    "plot_ou_equilibrium",
    "plot_ou_zscore",
    "plot_parameter_stability",
    "plot_rolling_b",
    "plot_rolling_half_life",
    "plot_rolling_theta",
    "plot_window_matrix",
]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = {
    "residual": "#2a78d6",
    "control_random_walk": "#eb6834",
    "reference_ou": "#1baf7a",
}
LABELS = {
    "residual": "XAUUSD residual",
    "control_random_walk": "detrended random walk (control)",
    "reference_ou": "exact OU, fitted parameters (reference)",
}
_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def _figure(config: OUConfig, rows: int = 1, cols: int = 1, *, height: float | None = None,
            sharex: bool = False) -> tuple[Any, Any]:
    width, base = config.plots.figsize
    fig, axes = plt.subplots(rows, cols, figsize=(width, height or base), sharex=sharex,
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.ravel():
        _style(ax)
    return fig, axes


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    for name, spine in ax.spines.items():
        spine.set_visible(name in ("left", "bottom"))
        spine.set_color(BASELINE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.title.set_color(INK)


def _finish(fig: Any, path: Path, config: OUConfig, title: str) -> Path:
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=config.plots.dpi, facecolor=SURFACE)
    plt.close(fig)
    return path


def _legend(ax: Any) -> None:
    legend = ax.legend(fontsize=8, frameon=False, loc="best")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)


def _thin(frame: pl.DataFrame, limit: int) -> pl.DataFrame:
    if frame.height <= limit:
        return frame
    step = int(np.ceil(frame.height / limit))
    return frame.gather_every(step)


# ---------------------------------------------------------------------------
# Rolling estimates through time
# ---------------------------------------------------------------------------
def _rolling_panel(
    frame: pl.DataFrame, profile: pl.DataFrame | None, column: str, *, path: Path,
    config: OUConfig, title: str, ylabel: str, log: bool = False,
    reference: list[tuple[float, str]] | None = None,
) -> Path:
    data = frame.select("timestamp", column).drop_nulls()
    if data.is_empty():
        raise ValueError(f"no values of {column} to plot")
    thin = _thin(data, config.plots.max_series_points)
    fig, axes = _figure(config)
    ax = axes[0, 0]
    ax.plot(thin["timestamp"].to_numpy(), thin[column].to_numpy(), lw=0.5,
            color=SERIES["residual"], alpha=0.35, label=f"rolling estimate (thinned to "
            f"{thin.height:,} of {data.height:,})")
    if profile is not None and not profile.is_empty():
        line = _thin(profile.drop_nulls(), config.plots.max_series_points)
        ax.plot(line["timestamp"].to_numpy(), line["rolling_median"].to_numpy(), lw=1.5,
                color=SERIES["residual"], label="rolling median")
    for value, label in reference or []:
        ax.axhline(value, color=INK_SECONDARY, lw=1.0, ls="--", label=label)
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("time (broker clock)")
    _legend(ax)
    return _finish(fig, path, config, title)


def plot_rolling_half_life(frame: pl.DataFrame, profile: pl.DataFrame, *, path: Path,
                           config: OUConfig, title: str) -> Path:
    return _rolling_panel(frame, profile, "ou_half_life_bars", path=path, config=config,
                          title=title, ylabel="half-life (bars, log scale)", log=True)


def plot_rolling_theta(frame: pl.DataFrame, *, path: Path, config: OUConfig, title: str) -> Path:
    window = config.stability.rolling_window
    profile = frame.select(
        "timestamp",
        pl.col("ou_theta").rolling_median(window_size=window, min_samples=window // 4)
        .alias("rolling_median"),
    )
    return _rolling_panel(frame, profile, "ou_theta", path=path, config=config, title=title,
                          ylabel="theta (per bar, log scale)", log=True)


def plot_rolling_b(frame: pl.DataFrame, *, path: Path, config: OUConfig, title: str) -> Path:
    window = config.stability.rolling_window
    profile = frame.select(
        "timestamp",
        pl.col("ou_b").rolling_median(window_size=window, min_samples=window // 4)
        .alias("rolling_median"),
    )
    refs = [(1.0, "b = 1 (unit root)")]
    return _rolling_panel(frame, profile, "ou_b", path=path, config=config, title=title,
                          ylabel="AR(1) coefficient b", reference=refs)


def plot_ou_equilibrium(frame: pl.DataFrame, *, path: Path, config: OUConfig,
                        title: str) -> Path:
    recent = (frame.select("timestamp", "residual", "ou_mu", "ou_stationary_std", "ou_valid")
              .filter(pl.col("ou_valid")).tail(config.plots.sample_bars))
    if recent.is_empty():
        raise ValueError("no valid OU fits to plot")
    fig, axes = _figure(config)
    ax = axes[0, 0]
    t = recent["timestamp"].to_numpy()
    mu = recent["ou_mu"].to_numpy()
    band = recent["ou_stationary_std"].to_numpy()
    ax.fill_between(t, mu - band, mu + band, color=SERIES["control_random_walk"], alpha=0.12,
                    lw=0, label="mu +/- stationary std")
    ax.plot(t, recent["residual"].to_numpy(), lw=1.0, color=SERIES["residual"],
            label="residual X_t")
    ax.plot(t, mu, lw=1.5, color=SERIES["control_random_walk"], label="rolling equilibrium mu_t")
    ax.axhline(0.0, color=BASELINE, lw=0.8)
    ax.set_ylabel("residual (log-price units)")
    ax.set_xlabel("time (broker clock)")
    _legend(ax)
    return _finish(fig, path, config, title)


def plot_ou_zscore(frame: pl.DataFrame, *, path: Path, config: OUConfig, title: str) -> Path:
    recent = frame.select("timestamp", "ou_zscore").drop_nulls().tail(config.plots.sample_bars)
    if recent.is_empty():
        raise ValueError("no OU Z-scores to plot")
    fig, axes = _figure(config)
    ax = axes[0, 0]
    ax.plot(recent["timestamp"].to_numpy(), recent["ou_zscore"].to_numpy(), lw=1.0,
            color=SERIES["residual"], label="OU Z-score")
    for level in (-2.0, 2.0):
        ax.axhline(level, color=INK_SECONDARY, lw=0.9, ls="--")
    ax.axhline(0.0, color=BASELINE, lw=0.8)
    ax.set_ylabel("(X_t - mu_t) / stationary std")
    ax.set_xlabel("time (broker clock); dashed lines at +/-2 are references, not thresholds")
    _legend(ax)
    return _finish(fig, path, config, title)


# ---------------------------------------------------------------------------
# Model against reality
# ---------------------------------------------------------------------------
def plot_estimated_vs_realized(events: dict[str, pl.DataFrame], *, path: Path,
                               config: OUConfig, title: str) -> Path:
    """Estimated half-life against realised half-decay, real beside the reference."""
    level = config.extremes.primary_abs_z
    panels = [(name, frame) for name, frame in events.items()
              if frame is not None and not frame.is_empty()]
    if not panels:
        raise ValueError("no extreme events to plot")
    fig, axes = _figure(config, 1, len(panels), height=config.plots.figsize[1])
    limit = config.plots.max_scatter_points
    top = 1.0
    for ax, (name, frame) in zip(axes[0], panels, strict=True):
        part = frame.filter((pl.col("abs_z") > level) & (pl.col("fp_50") > 0))
        part = _thin(part, limit)
        x = part["half_life_estimated"].to_numpy()
        y = part["fp_50"].to_numpy().astype(np.float64)
        if x.size:
            top = max(top, float(np.nanmax(x)), float(np.nanmax(y)))
        ax.scatter(x, y, s=8, alpha=0.25, color=SERIES.get(name, MUTED), linewidths=0,
                   label=f"{LABELS.get(name, name)} (n={part.height:,})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("estimated half-life at start (bars)")
        ax.set_ylabel("realised bars to halve the deviation")
        _legend(ax)
    for ax in axes[0]:
        ax.plot([0.5, top], [0.5, top], color=INK_SECONDARY, lw=1.0, ls="--")
    return _finish(fig, path, config, title + f"  (|Z_OU| > {level:g}; dashed line: equal)")


def plot_half_life_distribution(frames: dict[str, pl.DataFrame], *, path: Path,
                                config: OUConfig, title: str) -> Path:
    fig, axes = _figure(config)
    ax = axes[0, 0]
    drawn = 0
    all_values = [f["ou_half_life_bars"].drop_nulls().to_numpy() for f in frames.values()
                  if f is not None and not f.is_empty()]
    positive = np.concatenate([v[v > 0] for v in all_values]) if all_values else np.empty(0)
    if positive.size == 0:
        raise ValueError("no half-lives to plot")
    bins = np.logspace(np.log10(max(positive.min(), 0.1)), np.log10(positive.max()), 80)
    for name, frame in frames.items():
        if frame is None or frame.is_empty():
            continue
        values = frame["ou_half_life_bars"].drop_nulls().to_numpy()
        if values.size == 0:
            continue
        ax.hist(values, bins=bins, histtype="step", lw=1.5, density=True,
                color=SERIES.get(name, MUTED),
                label=f"{LABELS.get(name, name)}: median {np.median(values):.1f} bars")
        drawn += 1
    ax.set_xscale("log")
    ax.set_xlabel("rolling half-life (bars, log scale)")
    ax.set_ylabel("density")
    if drawn:
        _legend(ax)
    return _finish(fig, path, config, title)


# ---------------------------------------------------------------------------
# Innovations
# ---------------------------------------------------------------------------
def plot_innovation_acf(acf: pl.DataFrame, *, series: str, path: Path, config: OUConfig,
                        title: str) -> Path:
    """ACF of the innovations, |innovations| and squared innovations; control as a line."""
    names = [series, f"abs_{series}", f"squared_{series}"]
    fig, axes = _figure(config, 3, 1, height=config.plots.figsize[1] * 1.5, sharex=True)
    for ax, name in zip(axes[:, 0], names, strict=True):
        real = acf.filter((pl.col("source") == "residual") & (pl.col("series") == name)).sort("lag")
        if real.is_empty():
            continue
        ax.bar(real["lag"].to_numpy(), real["autocorrelation"].to_numpy(), width=0.7,
               color=SERIES["residual"], label=LABELS["residual"])
        control = acf.filter((pl.col("source") == "control_random_walk")
                             & (pl.col("series") == name)).sort("lag")
        if not control.is_empty():
            ax.plot(control["lag"].to_numpy(), control["autocorrelation"].to_numpy(), lw=1.5,
                    marker="o", ms=3, color=SERIES["control_random_walk"],
                    label=LABELS["control_random_walk"])
        bound = float(real["conf_upper"][0])
        ax.axhspan(-bound, bound, color=GRID, alpha=0.8, lw=0, label="white-noise band")
        ax.axhline(0.0, color=BASELINE, lw=0.8)
        ax.set_ylabel(name.replace("_", " "), fontsize=8)
        _legend(ax)
    axes[-1, 0].set_xlabel("lag (bars)")
    return _finish(fig, path, config, title)


def plot_innovation_distribution(values: np.ndarray, *, path: Path, config: OUConfig,
                                 title: str) -> Path:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size < 100:
        raise ValueError("too few innovations to plot")
    fig, axes = _figure(config, 1, 2)
    left, right = axes[0, 0], axes[0, 1]
    grid = np.linspace(np.quantile(data, 0.0005), np.quantile(data, 0.9995), 400)
    left.hist(data, bins=200, range=(grid[0], grid[-1]), density=True,
              color=SERIES["residual"], alpha=0.8, label=f"standardised innovations "
              f"(n={data.size:,})")
    left.plot(grid, stats.norm.pdf(grid, data.mean(), data.std(ddof=1)), lw=1.5,
              color=INK_SECONDARY, label="Gaussian, same mean and std")
    left.set_yscale("log")
    left.set_xlabel("innovation / window innovation std")
    left.set_ylabel("density (log scale: tails visible)")
    _legend(left)
    sample = np.sort(data)
    if sample.size > config.plots.max_scatter_points:
        sample = sample[:: int(np.ceil(sample.size / config.plots.max_scatter_points))]
    theoretical = stats.norm.ppf((np.arange(1, sample.size + 1) - 0.5) / sample.size)
    right.scatter(theoretical, sample, s=6, color=SERIES["residual"], alpha=0.5, linewidths=0)
    lim = max(abs(theoretical).max(), 1.0)
    right.plot([-lim, lim], [-lim * data.std(ddof=1), lim * data.std(ddof=1)],
               color=INK_SECONDARY, lw=1.0, ls="--")
    right.set_xlabel("standard normal quantile")
    right.set_ylabel("observed quantile")
    right.set_title("normal Q-Q", fontsize=9)
    return _finish(fig, path, config, title)


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------
def plot_parameter_stability(years: pl.DataFrame, *, path: Path, config: OUConfig,
                             title: str) -> Path:
    """Per-year rolling half-life (median, IQR) and valid-OU share, in two panels."""
    data = years.filter(pl.col("rolling_median_half_life").is_not_null()) if (
        "rolling_median_half_life" in years.columns) else pl.DataFrame()
    if data.is_empty():
        raise ValueError("no yearly half-life summary to plot")
    fig, axes = _figure(config, 2, 1, height=config.plots.figsize[1] * 1.3, sharex=True)
    x = data["segment"].cast(pl.Int32).to_numpy()
    median = data["rolling_median_half_life"].to_numpy()
    lo = data["rolling_half_life_p25"].to_numpy()
    hi = data["rolling_half_life_p75"].to_numpy()
    top, bottom = axes[0, 0], axes[1, 0]
    top.fill_between(x, lo, hi, color=SERIES["residual"], alpha=0.18, lw=0,
                     label="interquartile range of rolling half-life")
    top.plot(x, median, color=SERIES["residual"], lw=1.5, marker="o", ms=4,
             label="median rolling half-life")
    if "static_half_life_bars" in data.columns:
        top.plot(x, data["static_half_life_bars"].to_numpy(), lw=0, marker="D", ms=4,
                 color=SERIES["control_random_walk"], label="whole-year static fit")
    top.set_ylabel("half-life (bars)")
    _legend(top)
    bottom.plot(x, data["valid_ou_fraction"].to_numpy(), color=SERIES["residual"], lw=1.5,
                marker="o", ms=4, label="share of windows with a valid OU fit")
    bottom.set_ylim(0, 1.02)
    bottom.set_ylabel("valid OU share")
    bottom.set_xlabel("year")
    _legend(bottom)
    return _finish(fig, path, config, title)


def plot_window_matrix(matrix: pl.DataFrame, *, metric: str, path: Path, config: OUConfig,
                       title: str) -> Path:
    """Regression window x OU window heatmap of one metric (values annotated)."""
    rows = sorted(matrix["regression_window"].unique().to_list())
    cols = sorted(matrix["ou_window"].unique().to_list())
    grid = np.full((len(rows), len(cols)), np.nan)
    for rec in matrix.iter_rows(named=True):
        value = rec.get(metric)
        if value is not None:
            grid[rows.index(rec["regression_window"]), cols.index(rec["ou_window"])] = value
    fig, axes = _figure(config, height=config.plots.figsize[1] * 0.9)
    ax = axes[0, 0]
    ax.grid(False)
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blues", _BLUES)
    image = ax.imshow(grid, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(cols)), [str(c) for c in cols])
    ax.set_yticks(range(len(rows)), [str(r) for r in rows])
    ax.set_xlabel("OU estimation window M (residual observations)")
    ax.set_ylabel("regression window N (bars)")
    finite = grid[np.isfinite(grid)]
    mid = (finite.min() + finite.max()) / 2 if finite.size else 0.0
    for i in range(len(rows)):
        for j in range(len(cols)):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.3g}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if grid[i, j] > mid else INK)
    fig.colorbar(image, ax=ax, fraction=0.04)
    return _finish(fig, path, config, title)
