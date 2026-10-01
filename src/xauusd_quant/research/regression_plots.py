"""Figures for the rolling-regression research.

Every function takes already-computed results and returns the path it wrote.
Plots are a *view*: each one has a Parquet/CSV table beside it, so no result
exists only as pixels.

The reference lines at -3/-2/0/+2/+3 on the Z-score panel are visual aids for
reading the chart. They are **not** thresholds, entries or exits.
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

from ..features.config import RegressionConfig  # noqa: E402
from ..utils.paths import ensure_dir  # noqa: E402

__all__ = [
    "plot_decay_curve",
    "plot_price_and_fit",
    "plot_residual_acf",
    "plot_residual_histogram",
    "plot_residual_qq",
    "plot_residual_series",
    "plot_residual_zscore",
    "plot_zero_crossing_curve",
]


def _finish(fig: Any, path: Path, config: RegressionConfig) -> Path:
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=config.plots.dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _tail(frame: pl.DataFrame, config: RegressionConfig) -> pl.DataFrame:
    """The most recent slice, for the time-series panels."""
    return frame.tail(config.plots.sample_bars)


def _finite(values: Any) -> np.ndarray:
    if isinstance(values, pl.Series):
        values = values.drop_nulls().to_numpy()
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


# ---------------------------------------------------------------------------
# Price and fit
# ---------------------------------------------------------------------------
def plot_price_and_fit(
    frame: pl.DataFrame, *, path: Path, config: RegressionConfig,
    timeframe: str, window: int,
) -> Path:
    """Log price with the rolling fitted trend overlaid."""
    sample = _tail(frame.filter(pl.col("fitted_log_price").is_not_null()), config)
    if sample.is_empty():
        raise ValueError("no fitted values to plot")
    fig, ax = plt.subplots(figsize=config.plots.figsize)
    times = sample["timestamp"].to_numpy()
    ax.plot(times, sample["log_price"].to_numpy(), lw=0.9, color="#4C72B0",
            label="log price")
    ax.plot(times, sample["fitted_log_price"].to_numpy(), lw=1.1, color="#C44E52",
            label=f"rolling {window}-bar fit (endpoint)")
    ax.set_xlabel("time (broker clock)")
    ax.set_ylabel("log price")
    ax.set_title(f"{timeframe}  window={window}  price and rolling fit "
                 f"(last {sample.height:,} bars)")
    ax.legend(fontsize=8)
    return _finish(fig, path, config)


# ---------------------------------------------------------------------------
# Residual series and Z-score
# ---------------------------------------------------------------------------
def plot_residual_series(
    frame: pl.DataFrame, *, path: Path, config: RegressionConfig,
    timeframe: str, window: int,
) -> Path:
    """The residual through time."""
    sample = _tail(frame.filter(pl.col("residual").is_not_null()), config)
    if sample.is_empty():
        raise ValueError("no residuals to plot")
    fig, ax = plt.subplots(figsize=config.plots.figsize)
    ax.plot(sample["timestamp"].to_numpy(), sample["residual"].to_numpy(),
            lw=0.8, color="#4C72B0")
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xlabel("time (broker clock)")
    ax.set_ylabel("residual (log price units)")
    ax.set_title(f"{timeframe}  window={window}  regression residual "
                 f"(last {sample.height:,} bars)")
    return _finish(fig, path, config)


def plot_residual_zscore(
    frame: pl.DataFrame, *, path: Path, config: RegressionConfig,
    timeframe: str, window: int, column: str = "residual_zscore_fit",
) -> Path:
    """Residual Z-score with reference lines.

    The lines at +/-2 and +/-3 are visual references for reading the chart, not
    trading thresholds.
    """
    sample = _tail(frame.filter(pl.col(column).is_not_null()), config)
    if sample.is_empty():
        raise ValueError("no Z-scores to plot")
    fig, ax = plt.subplots(figsize=config.plots.figsize)
    ax.plot(sample["timestamp"].to_numpy(), sample[column].to_numpy(),
            lw=0.8, color="#4C72B0")
    for level, style, colour in (
        (0, "-", "black"), (2, "--", "#DD8452"), (-2, "--", "#DD8452"),
        (3, ":", "#C44E52"), (-3, ":", "#C44E52"),
    ):
        ax.axhline(level, ls=style, lw=0.9, color=colour)
    ax.set_xlabel("time (broker clock)")
    ax.set_ylabel("residual Z-score")
    ax.set_title(f"{timeframe}  window={window}  {column} "
                 "(reference lines, not thresholds)")
    return _finish(fig, path, config)


# ---------------------------------------------------------------------------
# Distribution
# ---------------------------------------------------------------------------
def plot_residual_histogram(
    frame: pl.DataFrame, *, path: Path, config: RegressionConfig,
    timeframe: str, window: int, column: str = "residual_zscore_fit",
) -> Path:
    """Residual Z histogram against a standard normal."""
    values = _finite(frame[column])
    if values.size < 50:
        raise ValueError("too few values for a histogram")
    fig, axes = plt.subplots(1, 2, figsize=(config.plots.figsize[0] * 1.3,
                                            config.plots.figsize[1]))
    for ax, log_scale in zip(axes, (False, True), strict=True):
        ax.hist(values, bins=200, density=True, alpha=0.75, color="#4C72B0",
                label=f"observed (n={values.size:,})")
        grid = np.linspace(values.min(), values.max(), 500)
        ax.plot(grid, stats.norm.pdf(grid, values.mean(), values.std(ddof=1)),
                "r-", lw=1.3, label="Gaussian, same mean/sd")
        if log_scale:
            ax.set_yscale("log")
            ax.set_title("log density - tails visible")
        else:
            ax.set_title("linear density")
        ax.set_xlabel(column)
        ax.legend(fontsize=8)
    fig.suptitle(f"{timeframe}  window={window}  residual Z distribution")
    return _finish(fig, path, config)


def plot_residual_qq(
    frame: pl.DataFrame, *, path: Path, config: RegressionConfig,
    timeframe: str, window: int, column: str = "residual_zscore_fit",
) -> Path:
    """Normal Q-Q plot of the residual Z-score."""
    values = _finite(frame[column])
    if values.size < 50:
        raise ValueError("too few values for a Q-Q plot")
    if values.size > config.plots.max_scatter_points:
        step = int(np.ceil(values.size / config.plots.max_scatter_points))
        values = np.sort(values)[::step]
    fig, ax = plt.subplots(figsize=config.plots.figsize)
    stats.probplot(values, dist="norm", plot=ax)
    ax.get_lines()[0].set_markersize(2.0)
    ax.get_lines()[0].set_alpha(0.5)
    ax.set_title(f"{timeframe}  window={window}  residual Z Q-Q "
                 f"(n={values.size:,} shown)")
    return _finish(fig, path, config)


def plot_residual_acf(
    acf_table: pl.DataFrame, *, path: Path, config: RegressionConfig,
    timeframe: str, window: int,
) -> Path:
    """ACF panels for the residual, its difference and |residual|."""
    if acf_table.is_empty():
        raise ValueError("empty ACF table")
    series_names = acf_table["series"].unique(maintain_order=True).to_list()
    fig, axes = plt.subplots(
        len(series_names), 1,
        figsize=(config.plots.figsize[0], 2.9 * len(series_names)), sharex=True,
    )
    if len(series_names) == 1:
        axes = [axes]
    for ax, name in zip(axes, series_names, strict=True):
        block = acf_table.filter(pl.col("series") == name).sort("lag")
        ax.bar(block["lag"].to_numpy(), block["autocorrelation"].to_numpy(),
               width=0.8, color="#4C72B0")
        bound = float(block["conf_upper"][0])
        ax.axhline(0, color="black", lw=0.8)
        ax.axhline(bound, color="red", ls="--", lw=0.9, label="white-noise band")
        ax.axhline(-bound, color="red", ls="--", lw=0.9)
        ax.set_ylabel("rho")
        ax.set_title(f"{name}  (n={int(block['observations'][0]):,})", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("lag")
    fig.suptitle(f"{timeframe}  window={window}  residual autocorrelation "
                 "(raw ACF is inflated by window overlap)")
    return _finish(fig, path, config)


# ---------------------------------------------------------------------------
# Decay and crossing
# ---------------------------------------------------------------------------
def plot_decay_curve(
    decay_table: pl.DataFrame, *, path: Path, config: RegressionConfig,
    timeframe: str, window: int,
) -> Path:
    r"""Mean :math:`Z_{t+h}` versus horizon, one line per starting Z bin.

    Lines converging toward zero indicate decay; flat lines indicate none.
    """
    if decay_table.is_empty():
        raise ValueError("empty decay table")
    fig, axes = plt.subplots(1, 2, figsize=(config.plots.figsize[0] * 1.45,
                                            config.plots.figsize[1]))
    for bin_label in decay_table["z_bin"].unique(maintain_order=True).to_list():
        block = decay_table.filter(pl.col("z_bin") == bin_label).sort("horizon")
        axes[0].plot(block["horizon"].to_numpy(), block["mean_z_future"].to_numpy(),
                     marker="o", ms=3.5, lw=1.2, label=bin_label)
        axes[1].plot(block["horizon"].to_numpy(),
                     block["prob_move_toward_zero"].to_numpy(),
                     marker="o", ms=3.5, lw=1.2, label=bin_label)
    axes[0].axhline(0, color="black", lw=0.9)
    axes[0].set_ylabel("mean Z at t+h")
    axes[0].set_title("conditional Z path", fontsize=10)
    axes[1].axhline(0.5, color="black", lw=0.9, ls="--")
    axes[1].set_ylabel("P(|eps_{t+h}| < |eps_t|)")
    axes[1].set_title("movement toward zero (0.5 = no tendency)", fontsize=10)
    for ax in axes:
        ax.set_xlabel("forward horizon (bars)")
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle(f"{timeframe}  window={window}  residual decay by starting Z bin "
                 "(descriptive; no costs applied)")
    return _finish(fig, path, config)


def plot_zero_crossing_curve(
    crossing_table: pl.DataFrame, *, path: Path, config: RegressionConfig,
    timeframe: str, window: int,
) -> Path:
    """Cumulative probability of a sign change versus bars elapsed."""
    if crossing_table.is_empty():
        raise ValueError("empty crossing table")
    checkpoints = list(config.extremes.crossing_checkpoints)
    fig, ax = plt.subplots(figsize=config.plots.figsize)
    for row in crossing_table.iter_rows(named=True):
        raw = [row.get(f"crossed_within_{c}") for c in checkpoints]
        if any(v is None for v in raw):
            continue
        values = [float(v) for v in raw if v is not None]
        ax.plot(checkpoints, values, marker="o", ms=4, lw=1.2,
                label=f"{row['condition']} (n={row['observations']:,})")
    ax.set_xlabel("bars elapsed")
    ax.set_ylabel("cumulative P(residual changed sign)")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{timeframe}  window={window}  zero-crossing "
                 "(a sign change, not a trade exit)")
    ax.legend(fontsize=8)
    return _finish(fig, path, config)
