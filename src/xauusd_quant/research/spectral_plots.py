"""Figures for the spectral study. Every figure is a view of a table written beside it.

Series colours follow the entity, in the validated categorical order of the
project palette: the real series is always slot 1 (blue), the detrended
random walk slot 2 (orange, as in the OU figures), then white noise, shuffled
and block-bootstrapped returns. Null series are drawn dashed as well, so
identity never rests on colour alone. Ordered bands use one blue ramp.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from ..features.spectral_config import SpectralConfig  # noqa: E402
from ..utils.paths import ensure_dir  # noqa: E402
from .ou_plots import BASELINE, GRID, INK, INK_SECONDARY, SURFACE  # noqa: E402

__all__ = [
    "LABELS",
    "SERIES",
    "plot_band_power",
    "plot_dominant_period",
    "plot_mean_spectrum",
    "plot_null_distributions",
    "plot_period_power",
    "plot_phase_outcomes",
    "plot_rolling_entropy",
    "plot_spectrogram",
]

SERIES = {
    "real": "#2a78d6",
    "random_walk": "#eb6834",
    "white_noise": "#1baf7a",
    "shuffled_returns": "#eda100",
    "block_bootstrap": "#e87ba4",
}
LABELS = {
    "real": "XAUUSD",
    "random_walk": "detrended random walk (null)",
    "white_noise": "white noise (null)",
    "shuffled_returns": "shuffled returns (null)",
    "block_bootstrap": "block-bootstrapped returns (null)",
}
_RAMP = ["#86b6ef", "#2a78d6", "#104281"]     # ordinal: low, mid, high band


def _figure(config: SpectralConfig, rows: int = 1, cols: int = 1, *,
            height: float | None = None) -> tuple[Any, Any]:
    width, base = config.plots.figsize
    fig, axes = plt.subplots(rows, cols, figsize=(width, height or base), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.ravel():
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6, linestyle="-")
        ax.set_axisbelow(True)
        for name, spine in ax.spines.items():
            spine.set_visible(name in ("left", "bottom"))
            spine.set_color(BASELINE)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8)
        ax.xaxis.label.set_color(INK_SECONDARY)
        ax.yaxis.label.set_color(INK_SECONDARY)
    return fig, axes


def _finish(fig: Any, path: Path, config: SpectralConfig, title: str) -> Path:
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


def _style(source: str) -> dict[str, Any]:
    return {"color": SERIES.get(source, INK_SECONDARY),
            "linestyle": "-" if source == "real" else "--",
            "linewidth": 2.0 if source == "real" else 1.4,
            "label": LABELS.get(source, source)}


def plot_mean_spectrum(spectra: dict[str, pl.DataFrame], *, path: Path, config: SpectralConfig,
                       title: str) -> Path:
    """Average power share per bin against normalised frequency, real beside nulls."""
    fig, axes = _figure(config)
    ax = axes[0, 0]
    for source, frame in spectra.items():
        ax.plot(frame["normalised_frequency"].to_numpy(), frame["mean_power_share"].to_numpy(),
                **_style(source))
    ax.set_yscale("log")
    ax.set_xlabel("frequency / Nyquist (1 = a 2-bar cycle)")
    ax.set_ylabel("mean share of window power (log scale)")
    _legend(ax)
    return _finish(fig, path, config, title)


def plot_period_power(spectra: dict[str, pl.DataFrame], *, path: Path, config: SpectralConfig,
                      title: str, bar_label: str) -> Path:
    """The same spectra against period in bars (log scale)."""
    fig, axes = _figure(config)
    ax = axes[0, 0]
    for source, frame in spectra.items():
        ax.plot(frame["period_bars"].to_numpy(), frame["mean_power_share"].to_numpy(),
                **_style(source))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"period (bars of {bar_label}, log scale)")
    ax.set_ylabel("mean share of window power (log scale)")
    _legend(ax)
    return _finish(fig, path, config, title)


def plot_dominant_period(timestamps: np.ndarray, period: np.ndarray, *, path: Path,
                         config: SpectralConfig, title: str, bar_label: str) -> Path:
    """The dominant period bar by bar over a recent slice, with its rolling median."""
    fig, axes = _figure(config)
    ax = axes[0, 0]
    ax.plot(timestamps, period, lw=0.0, marker=".", markersize=2.5, color=SERIES["real"],
            alpha=0.5, label="dominant period, window ending at each bar")
    median = pl.Series(period).rolling_median(window_size=max(5, period.size // 30),
                                              min_samples=3).to_numpy()
    ax.plot(timestamps, median, color=SERIES["real"], lw=1.8, label="rolling median")
    ax.set_yscale("log")
    ax.set_ylabel(f"period (bars of {bar_label}, log scale)")
    ax.set_xlabel("time (broker clock)")
    _legend(ax)
    return _finish(fig, path, config, title)


def plot_rolling_entropy(timestamps: np.ndarray, entropy: np.ndarray,
                         yearly: dict[str, pl.DataFrame], *, path: Path, config: SpectralConfig,
                         title: str) -> Path:
    """Spectral entropy through the whole history, with yearly medians of real and nulls."""
    fig, axes = _figure(config)
    ax = axes[0, 0]
    ax.plot(timestamps, entropy, lw=0.4, color=SERIES["real"], alpha=0.25,
            label=f"XAUUSD, thinned to {entropy.size:,} windows")
    for source, frame in yearly.items():
        if frame.is_empty():
            continue
        years = frame["segment"].cast(pl.Int32).to_numpy()
        mids = np.array([np.datetime64(f"{y}-07-01") for y in years])
        style = _style(source)
        style["label"] = f"{style['label']}: yearly median"
        ax.plot(mids, frame["median_spectral_entropy"].to_numpy(), marker="o", markersize=3,
                **style)
    ax.set_ylabel("normalised spectral entropy")
    ax.set_xlabel("time (broker clock)")
    _legend(ax)
    return _finish(fig, path, config, title)


def plot_band_power(yearly: pl.DataFrame, band_names: tuple[str, ...], *, path: Path,
                    config: SpectralConfig, title: str) -> Path:
    """Yearly median power share of each frequency band."""
    fig, axes = _figure(config)
    ax = axes[0, 0]
    years = yearly["segment"].cast(pl.Int32).to_numpy()
    for colour, band in zip(_RAMP, band_names, strict=False):
        ax.plot(years, yearly[f"median_{band}_power_share"].to_numpy(), color=colour, lw=1.8,
                marker="o", markersize=3, label=f"{band} band")
    ax.set_ylabel("median share of window power")
    ax.set_xlabel("year")
    _legend(ax)
    return _finish(fig, path, config, title)


def plot_spectrogram(times: np.ndarray, normalised_frequency: np.ndarray, shares: np.ndarray, *,
                     path: Path, config: SpectralConfig, title: str) -> Path:
    """Power share by frequency (rows) and window end (columns) for a short recent slice."""
    fig, axes = _figure(config)
    ax = axes[0, 0]
    ax.grid(False)
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#86b6ef", "#2a78d6", "#0d366b"])
    image = ax.imshow(shares.T, aspect="auto", origin="lower", cmap=cmap,
                      extent=(0, shares.shape[0], normalised_frequency[0],
                              normalised_frequency[-1]), interpolation="nearest")
    ticks = np.linspace(0, shares.shape[0] - 1, 5).astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(times[i])[:16] for i in ticks], fontsize=7)
    ax.set_ylabel("frequency / Nyquist")
    ax.set_xlabel("window end (broker clock)")
    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.01)
    bar.set_label("share of window power", color=INK_SECONDARY, fontsize=8)
    bar.ax.tick_params(labelsize=7, colors=INK_SECONDARY)
    return _finish(fig, path, config, title)


def plot_phase_outcomes(tables: dict[str, pl.DataFrame], outcome: str, *, path: Path,
                        config: SpectralConfig, title: str) -> Path:
    """Mean outcome by dominant-phase sector, with one standard error, real beside nulls."""
    fig, axes = _figure(config)
    ax = axes[0, 0]
    offset = {name: (i - (len(tables) - 1) / 2) * 0.06 for i, name in enumerate(tables)}
    for source, table in tables.items():
        part = table.filter((pl.col("outcome") == outcome) & (pl.col("bucket") >= 0)).sort("bucket")
        if part.is_empty():
            continue
        centre = ((part["phase_lower"] + part["phase_upper"]) / 2).to_numpy() + offset[source]
        style = _style(source)
        ax.errorbar(centre, part["mean"].to_numpy(), yerr=part["std_error"].fill_null(0).to_numpy(),
                    fmt="o", markersize=4, capsize=2, color=style["color"], label=style["label"])
    ax.axhline(0.0, color=BASELINE, lw=0.8)
    ax.set_xlabel("phase of the dominant component at the window's last bar (radians)")
    ax.set_ylabel(outcome.replace("_", " "))
    _legend(ax)
    return _finish(fig, path, config, title)


def plot_null_distributions(histograms: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]], *,
                            path: Path, config: SpectralConfig, title: str) -> Path:
    """Distributions of spectral metrics for the real series and every null."""
    metrics = list(next(iter(histograms.values())).keys()) if histograms else []
    fig, axes = _figure(config, 1, max(len(metrics), 1), height=config.plots.figsize[1] * 0.8)
    for ax, metric in zip(axes[0], metrics, strict=False):
        for source, per_metric in histograms.items():
            if metric not in per_metric:
                continue
            edges, density = per_metric[metric]
            ax.step(edges[:-1], density, where="post", **_style(source))
        ax.set_xlabel(metric.replace("_", " "))
        ax.set_ylabel("density")
    if metrics:
        _legend(axes[0, 0])
    return _finish(fig, path, config, title)
