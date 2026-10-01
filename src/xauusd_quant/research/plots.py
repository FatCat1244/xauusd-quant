"""Reusable Matplotlib figures for the research reports.

Every function takes already-computed results and returns the path it wrote, so
notebooks never have to assemble a chart by hand and no calculation lives in a
plotting function.

Plots are a *view* of the results, never the results themselves: each figure
drawn here has a corresponding Parquet/CSV/JSON table written alongside it, so
nothing important exists only as pixels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: reports are generated from the CLI

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy import stats  # noqa: E402

from ..utils.paths import ensure_dir  # noqa: E402
from .config import ResearchConfig  # noqa: E402

__all__ = [
    "plot_acf",
    "plot_by_hour",
    "plot_conditional_returns",
    "plot_qq",
    "plot_return_histogram",
    "plot_rolling_volatility",
    "plot_yearly_comparison",
    "use_style",
]


def use_style(config: ResearchConfig) -> None:
    """Apply the configured Matplotlib style, tolerating an unknown name."""
    try:
        plt.style.use(config.plots.style)
    except (OSError, ValueError):
        plt.style.use("default")


def _finish(fig: Any, path: Path, config: ResearchConfig) -> Path:
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=config.plots.dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _array(values: Any) -> np.ndarray:
    if isinstance(values, pl.Series):
        values = values.drop_nulls().to_numpy()
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


# ---------------------------------------------------------------------------
# Distribution
# ---------------------------------------------------------------------------
def plot_return_histogram(
    returns: Any, *, path: Path, config: ResearchConfig, timeframe: str, gaussian_overlay: bool = True
) -> Path:
    """Return histogram, optionally against a fitted Gaussian.

    The log-scale y-axis is deliberate: on a linear axis the tails - the part
    that actually distinguishes a return distribution from a Gaussian - are
    invisible.
    """
    array = _array(returns)
    fig, axes = plt.subplots(1, 2, figsize=(config.plots.figsize[0] * 1.4,
                                            config.plots.figsize[1]))
    for ax, log_scale in zip(axes, (False, True), strict=True):
        ax.hist(array, bins=config.plots.histogram_bins
                if hasattr(config.plots, "histogram_bins")
                else config.distribution.histogram_bins,
                density=True, alpha=0.75, color="#4C72B0",
                label=f"observed (n={array.size:,})")
        if gaussian_overlay and array.size > 1:
            mu, sd = float(np.mean(array)), float(np.std(array, ddof=1))
            grid = np.linspace(array.min(), array.max(), 500)
            ax.plot(grid, stats.norm.pdf(grid, mu, sd), "r-", lw=1.4,
                    label=f"Gaussian(mu={mu:.2e}, sd={sd:.2e})")
        if log_scale:
            ax.set_yscale("log")
            ax.set_title("log density - tails visible")
        else:
            ax.set_title("linear density")
        ax.set_xlabel("log return")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    fig.suptitle(f"{timeframe} return distribution vs Gaussian reference")
    return _finish(fig, path, config)


def plot_qq(returns: Any, *, path: Path, config: ResearchConfig, timeframe: str) -> Path:
    """Normal Q-Q plot.

    Points bending away from the line at both ends are the visual signature of
    heavy tails; the straight middle is what makes the bends meaningful.
    """
    array = _array(returns)
    if array.size > config.plots.max_scatter_points:
        step = int(np.ceil(array.size / config.plots.max_scatter_points))
        array = np.sort(array)[::step]
    fig, ax = plt.subplots(figsize=config.plots.figsize)
    stats.probplot(array, dist="norm", plot=ax)
    ax.set_title(f"{timeframe} normal Q-Q plot (n={array.size:,} shown)")
    ax.get_lines()[0].set_markersize(2.0)
    ax.get_lines()[0].set_alpha(0.5)
    return _finish(fig, path, config)


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------
def plot_acf(acf_results: list[Any], *, path: Path, config: ResearchConfig, timeframe: str) -> Path:
    """Stacked ACF panels for returns, |r| and r^2.

    Sharing the x-axis makes the usual contrast obvious at a glance: raw
    returns hugging zero while |r| and r^2 stay well above the band for many
    lags.
    """
    results = [r for r in acf_results if r is not None]
    if not results:
        raise ValueError("no ACF results to plot")
    fig, axes = plt.subplots(
        len(results), 1, figsize=(config.plots.figsize[0], 3.0 * len(results)),
        sharex=True,
    )
    if len(results) == 1:
        axes = [axes]
    for ax, result in zip(axes, results, strict=True):
        ax.bar(result.lags, result.values, width=0.8, color="#4C72B0")
        ax.axhline(0, color="black", lw=0.8)
        ax.axhline(result.confidence_bound, color="red", ls="--", lw=0.9,
                   label=f"{result.confidence_level:.0%} white-noise band")
        ax.axhline(-result.confidence_bound, color="red", ls="--", lw=0.9)
        ax.set_ylabel("rho")
        ax.set_title(f"{result.series_name}  (n={result.observations:,})", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("lag")
    fig.suptitle(f"{timeframe} autocorrelation")
    return _finish(fig, path, config)


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------
def plot_rolling_volatility(
    frame: pl.DataFrame, *, path: Path, config: ResearchConfig, timeframe: str,
    time_column: str = "timestamp", columns: list[str] | None = None,
) -> Path:
    """Rolling volatility through time, one line per window."""
    candidates = columns or [
        c for c in frame.columns if c.startswith("roll") and c.endswith("_volatility")
    ]
    fig, ax = plt.subplots(figsize=config.plots.figsize)
    times = frame[time_column].to_numpy()
    for column in candidates:
        series = frame[column].to_numpy().astype(np.float64)
        ax.plot(times, series, lw=0.7, label=column)
    ax.set_xlabel(f"time ({time_column})")
    ax.set_ylabel("rolling std of log returns (per bar)")
    ax.set_title(f"{timeframe} rolling volatility")
    ax.legend(fontsize=8)
    return _finish(fig, path, config)


# ---------------------------------------------------------------------------
# Intraday
# ---------------------------------------------------------------------------
def plot_by_hour(
    table: pl.DataFrame, *, path: Path, config: ResearchConfig, timeframe: str, timezone_description: str,
) -> Path:
    """Volatility, spread, activity and mean return by hour of day."""
    panels = [
        ("return_std", "return std", "#4C72B0"),
        ("mean_spread", "mean spread (price units)", "#DD8452"),
        ("mean_tick_count", "mean quote updates", "#55A868"),
        ("mean_return", "mean return", "#C44E52"),
    ]
    available = [p for p in panels if p[0] in table.columns]
    if not available:
        raise ValueError("hourly table has none of the expected columns")

    fig, axes = plt.subplots(
        len(available), 1, figsize=(config.plots.figsize[0], 2.6 * len(available)),
        sharex=True,
    )
    if len(available) == 1:
        axes = [axes]
    hours = table["hour"].to_numpy()
    for ax, (column, label, colour) in zip(axes, available, strict=True):
        ax.bar(hours, table[column].to_numpy(), color=colour, width=0.8)
        ax.set_ylabel(label, fontsize=9)
        if column == "mean_return":
            ax.axhline(0, color="black", lw=0.8)
    axes[-1].set_xlabel(f"hour of day ({timezone_description})")
    axes[-1].set_xticks(range(0, 24, 2))
    fig.suptitle(f"{timeframe} intraday profile")
    return _finish(fig, path, config)


# ---------------------------------------------------------------------------
# Conditional / reversal
# ---------------------------------------------------------------------------
def plot_conditional_returns(
    table: pl.DataFrame, *, path: Path, config: ResearchConfig, timeframe: str
) -> Path:
    """Mean forward return by horizon, one line per extreme-return bucket."""
    if table.is_empty():
        raise ValueError("conditional table is empty")
    fig, axes = plt.subplots(1, 2, figsize=(config.plots.figsize[0] * 1.5,
                                            config.plots.figsize[1]))
    for ax, direction, title in zip(
        axes, ("lower", "upper"),
        ("after extreme NEGATIVE returns", "after extreme POSITIVE returns"),
        strict=True,
    ):
        subset = table.filter(pl.col("direction") == direction)
        for quantile in sorted(subset["quantile"].unique().to_list()):
            block = subset.filter(pl.col("quantile") == quantile).sort("horizon")
            ax.plot(block["horizon"].to_numpy(),
                    block["mean_forward_return"].to_numpy(),
                    marker="o", ms=3.5, lw=1.2, label=f"q={quantile:g}")
        ax.axhline(0, color="black", lw=0.9)
        ax.set_xlabel("forward horizon (bars)")
        ax.set_ylabel("mean forward log return")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle(
        f"{timeframe} mean forward return after extreme moves "
        "(descriptive; no costs applied)"
    )
    return _finish(fig, path, config)


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------
def plot_yearly_comparison(
    table: pl.DataFrame, *, path: Path, config: ResearchConfig, timeframe: str
) -> Path:
    """Headline statistics per chronological segment."""
    usable = table.filter(pl.col("observations") > 0)
    if "sufficient_data" in usable.columns:
        usable = usable.filter(pl.col("sufficient_data"))
    if usable.is_empty():
        raise ValueError("stability table has no populated segments")

    panels = [
        ("std", "return std"),
        ("excess_kurtosis", "excess kurtosis"),
        ("acf1", "lag-1 autocorrelation"),
        ("abs_acf1", "|r| lag-1 autocorrelation"),
    ]
    available = [p for p in panels if p[0] in usable.columns]
    fig, axes = plt.subplots(
        len(available), 1, figsize=(config.plots.figsize[0], 2.4 * len(available)),
        sharex=True,
    )
    if len(available) == 1:
        axes = [axes]
    segments = usable["segment"].to_list()
    positions = np.arange(len(segments))
    for ax, (column, label) in zip(axes, available, strict=True):
        ax.bar(positions, usable[column].to_numpy(), color="#4C72B0", width=0.7)
        ax.set_ylabel(label, fontsize=9)
        if column in ("acf1", "abs_acf1"):
            ax.axhline(0, color="black", lw=0.8)
    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels(segments, rotation=45, ha="right")
    axes[-1].set_xlabel("segment")
    fig.suptitle(f"{timeframe} statistics through time")
    return _finish(fig, path, config)
