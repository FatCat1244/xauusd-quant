"""Figures for the wavelet study. Every figure is a view of a table written beside it.

Colours follow the entity in the project's validated categorical order
(:data:`spectral_plots.SERIES`): the real series slot 1 (blue), the detrended
random walk slot 2 (orange), then white noise, shuffled and bootstrapped
returns. Block-bootstrap variants keep the bootstrap hue and differ by dash
pattern - a block size is not a new entity. Nulls are dashed, so identity
never rests on colour alone. Offline (centred) figures carry the
NON-CAUSAL label in their title.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from ..features.wavelet import NON_CAUSAL_LABEL, OfflineScalogram  # noqa: E402
from ..features.wavelet_config import WaveletConfig  # noqa: E402
from ..utils.paths import ensure_dir  # noqa: E402
from .ou_plots import BASELINE, GRID, INK, INK_SECONDARY, SURFACE  # noqa: E402
from .spectral_plots import LABELS as _SPECTRAL_LABELS  # noqa: E402
from .spectral_plots import SERIES as _SPECTRAL_SERIES  # noqa: E402

__all__ = [
    "plot_boundary_study",
    "plot_chirp_control",
    "plot_cross_timeframe",
    "plot_energy_by_scale",
    "plot_fft_vs_wavelet_period",
    "plot_ic_by_horizon",
    "plot_localized_control",
    "plot_null_distributions",
    "plot_rolling_state",
    "plot_scale_drift",
    "plot_scale_persistence",
    "plot_scalogram",
]

_DASH = {"block_bootstrap_64": (0, (5, 2, 1, 2)), "block_bootstrap_1024": (0, (1, 2))}


def _colour(source: str) -> str:
    base = "block_bootstrap" if source.startswith("block_bootstrap") else source
    return _SPECTRAL_SERIES.get(base, INK_SECONDARY)


def _label(source: str) -> str:
    if source.startswith("block_bootstrap_"):
        return f"block bootstrap, {source.rsplit('_', 1)[1]}-bar blocks (null)"
    return _SPECTRAL_LABELS.get(source, source)


def _style(source: str) -> dict[str, Any]:
    return {"color": _colour(source),
            "linestyle": "-" if source == "real" else _DASH.get(source, "--"),
            "linewidth": 2.0 if source == "real" else 1.3, "label": _label(source)}


def _figure(config: WaveletConfig, rows: int = 1, cols: int = 1, *,
            height: float | None = None) -> tuple[Any, Any]:
    width, base = config.plots.figsize
    fig, axes = plt.subplots(rows, cols, figsize=(width, height or base), squeeze=False)
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


def _legend(ax: Any) -> None:
    legend = ax.legend(fontsize=8, frameon=False, loc="best")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)


def _finish(fig: Any, path: Path, config: WaveletConfig, title: str) -> Path:
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=config.plots.dpi, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_energy_by_scale(energy: pl.DataFrame, *, path: Path, config: WaveletConfig,
                         title: str) -> Path:
    """Median band energy share against the band's period in time, real beside nulls."""
    fig, axes = _figure(config)
    ax = axes[0, 0]
    for (source,), part in energy.group_by("source", maintain_order=True):
        part = part.sort("period_seconds")
        hours = part["period_seconds"].to_numpy() / 3600
        ax.plot(hours, part["median_share"].to_numpy(), marker="o", markersize=4,
                **_style(str(source)))
        if source == "real":
            ax.fill_between(hours, part["share_p25"].to_numpy(), part["share_p75"].to_numpy(),
                            color=_colour("real"), alpha=0.15, linewidth=0,
                            label="XAUUSD inter-quartile range")
    ax.set_xscale("log")
    ax.set_xlabel("band period (hours, log scale; last point = approximation band)")
    ax.set_ylabel("share of window energy (median)")
    _legend(ax)
    return _finish(fig, path, config, title)


def plot_rolling_state(timestamps: np.ndarray, values: dict[str, np.ndarray], *, path: Path,
                       config: WaveletConfig, title: str, ylabel: str, log: bool = False
                       ) -> Path:
    """A causal wavelet state through a recent slice, real beside a null."""
    fig, axes = _figure(config)
    ax = axes[0, 0]
    for source, series in values.items():
        style = _style(source)
        ax.plot(timestamps, series, lw=0.0, marker=".", markersize=2.0, color=style["color"],
                alpha=0.35)
        median = pl.Series(series).rolling_median(window_size=max(5, series.size // 30),
                                                  min_samples=3).to_numpy()
        style["label"] = f"{style['label']} (rolling median)"
        ax.plot(timestamps, median, **style)
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("bar (broker clock), window ending at each bar")
    _legend(ax)
    return _finish(fig, path, config, title)


def plot_scale_persistence(persistence: pl.DataFrame, *, path: Path, config: WaveletConfig,
                           title: str) -> Path:
    """P(same dominant band h bars later) against h in windows; >= 1 is beyond overlap."""
    fig, axes = _figure(config)
    ax = axes[0, 0]
    for (source,), part in persistence.group_by("source", maintain_order=True):
        part = part.sort("lag_in_windows")
        ax.plot(part["lag_in_windows"].to_numpy(), part["same_band"].to_numpy(), marker="o",
                markersize=4, **_style(str(source)))
    ax.axvline(1.0, color=BASELINE, lw=1.0, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel("lag (windows, log scale; right of the dotted line the windows are disjoint)")
    ax.set_ylabel("share of pairs with the same dominant band")
    _legend(ax)
    return _finish(fig, path, config, title)


def plot_scale_drift(drift: dict[str, np.ndarray], *, path: Path, config: WaveletConfig,
                     title: str) -> Path:
    """Distribution of the centroid-period drift over N/4 bars, real beside nulls."""
    fig, axes = _figure(config)
    ax = axes[0, 0]
    finite = [v[np.isfinite(v)] for v in drift.values() if np.isfinite(v).any()]
    if finite:
        lo, hi = np.quantile(np.concatenate(finite), [0.005, 0.995])
        bins = np.linspace(lo, hi, 80)
        for source, values in drift.items():
            v = values[np.isfinite(values)]
            if v.size:
                hist, edges = np.histogram(v, bins=bins, density=True)
                ax.plot((edges[:-1] + edges[1:]) / 2, hist, **_style(source))
    ax.set_xlabel("change in ln(centroid period) over N/4 bars")
    ax.set_ylabel("density")
    _legend(ax)
    return _finish(fig, path, config, title)


def plot_null_distributions(histograms: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]], *,
                            path: Path, config: WaveletConfig, title: str) -> Path:
    """Per-bar distributions of the headline wavelet metrics, real beside every null."""
    metrics = list(next(iter(histograms.values())).keys()) if histograms else []
    fig, axes = _figure(config, 1, max(1, len(metrics)), height=4.2)
    for ax, metric in zip(axes[0], metrics, strict=False):
        for source, per_metric in histograms.items():
            if metric in per_metric:
                centres, density = per_metric[metric]
                ax.plot(centres, density, **_style(source))
        ax.set_xlabel(metric.removeprefix("wavelet_").replace("_", " "))
        ax.set_ylabel("density")
    if metrics:
        _legend(axes[0][0])
    return _finish(fig, path, config, title)


def plot_fft_vs_wavelet_period(fft_period: np.ndarray, wavelet_period: np.ndarray, *,
                               path: Path, config: WaveletConfig, title: str) -> Path:
    """Joint distribution of the FFT dominant period and the wavelet dominant period."""
    fig, axes = _figure(config)
    ax = axes[0, 0]
    ok = np.isfinite(fft_period) & np.isfinite(wavelet_period) & (fft_period > 0)
    if ok.any():
        x, y = np.log2(fft_period[ok]), np.log2(wavelet_period[ok])
        h = ax.hist2d(x, y, bins=[60, max(8, len(np.unique(y)))], cmap="Blues", cmin=1)
        cbar = fig.colorbar(h[3], ax=ax)
        cbar.set_label("windows", color=INK_SECONDARY)
    ax.set_xlabel("FFT dominant period (log2 bars)")
    ax.set_ylabel("wavelet dominant band period (log2 bars)")
    return _finish(fig, path, config, title)


def plot_ic_by_horizon(ic: pl.DataFrame, *, path: Path, config: WaveletConfig,
                       title: str) -> Path:
    """Largest |rank IC| at each horizon, real beside each null (all features and targets)."""
    fig, axes = _figure(config)
    ax = axes[0, 0]
    if not ic.is_empty():
        table = (ic.with_columns(pl.col("rank_ic").abs().alias("a"))
                 .group_by("source", "horizon").agg(pl.col("a").max().alias("max_abs"),
                                                    pl.col("a").median().alias("median_abs"))
                 .sort("horizon"))
        for (source,), part in table.group_by("source", maintain_order=True):
            style = _style(str(source))
            ax.plot(part["horizon"].to_numpy(), part["max_abs"].to_numpy(), marker="o",
                    markersize=4, **style)
    ax.set_xscale("log")
    ax.set_xlabel("horizon (bars, log scale)")
    ax.set_ylabel("largest |rank IC| over features and targets")
    _legend(ax)
    return _finish(fig, path, config, title)


def plot_scalogram(scalogram: OfflineScalogram, times: np.ndarray, *, path: Path,
                   config: WaveletConfig, title: str) -> Path:
    """Log power of an offline slice with the cone of influence hatched."""
    fig, axes = _figure(config, height=5.2)
    ax = axes[0, 0]
    power = np.log10(np.maximum(scalogram.power, np.finfo(np.float32).tiny))
    hours = scalogram.periods_seconds / 3600
    x = np.arange(times.size)
    mesh = ax.pcolormesh(x, hours, power, shading="auto", cmap="Blues")
    coi = scalogram.coi_period_bars * scalogram.bar_seconds / 3600
    ax.fill_between(x, np.maximum(coi, hours.min()), hours.max(), facecolor="none",
                    edgecolor=INK_SECONDARY, hatch="///", linewidth=0.0,
                    label="cone of influence (edge-affected)")
    ax.set_yscale("log")
    ax.set_ylim(hours.min(), hours.max())
    ticks = np.linspace(0, times.size - 1, 6).round().astype(int)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(np.datetime_as_string(times[i], unit="D")) for i in ticks],
                       fontsize=7)
    ax.set_ylabel("period (hours, log scale)")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("log10 power", color=INK_SECONDARY)
    _legend(ax)
    return _finish(fig, path, config, f"{title} - {NON_CAUSAL_LABEL}")


def plot_localized_control(t: np.ndarray, signal: np.ndarray, series: dict[str, np.ndarray],
                           active: tuple[int, int], *, path: Path, config: WaveletConfig
                           ) -> Path:
    """The localised-cycle control: the signal and each method's normalised response."""
    fig, axes = _figure(config, 2, 1, height=6.5)
    axes[0, 0].plot(t, signal, color=INK_SECONDARY, lw=0.5)
    axes[0, 0].set_ylabel("signal")
    colours = {"causal": _colour("real"), "fft": _colour("random_walk"),
               "offline": _colour("white_noise")}
    for name, values in series.items():
        v = values - np.nanmin(values)
        v = v / np.nanmax(v) if np.nanmax(v) > 0 else v
        kind = "offline" if name.startswith("offline") else "fft" if "fft" in name else "causal"
        axes[1, 0].plot(t, v, color=colours[kind], lw=1.4,
                        linestyle="-" if kind == "causal" else "--", label=name)
    for ax in axes[:, 0]:
        ax.axvspan(active[0], active[1], color=GRID, alpha=0.6)
    axes[1, 0].set_ylabel("response (0-1)")
    axes[1, 0].set_xlabel("bar")
    _legend(axes[1, 0])
    return _finish(fig, path, config, "Positive control: a cycle active only in the shaded "
                   "span (offline CWT is NON-CAUSAL)")


def plot_chirp_control(t: np.ndarray, true_period: np.ndarray, estimates: dict[str, np.ndarray],
                       *, path: Path, config: WaveletConfig) -> Path:
    """The chirp control: the true period and what each method reports."""
    fig, axes = _figure(config)
    ax = axes[0, 0]
    ax.plot(t, true_period, color=INK, lw=2.0, label="true period")
    for i, (name, values) in enumerate(estimates.items()):
        ax.plot(t, values, lw=1.2, linestyle=["--", ":", "-."][i % 3], label=name,
                color=list(_SPECTRAL_SERIES.values())[1 + i % 4])
    ax.set_yscale("log")
    ax.set_ylabel("period (bars, log scale)")
    ax.set_xlabel("bar")
    _legend(ax)
    return _finish(fig, path, config, "Positive control: a chirp (offline CWT ridge is "
                   "NON-CAUSAL)")


def plot_boundary_study(summary: pl.DataFrame, *, path: Path, config: WaveletConfig) -> Path:
    """Latest-state energy error by method, stationary and after a recent step."""
    fig, axes = _figure(config)
    ax = axes[0, 0]
    methods = summary.filter(~pl.col("step")).sort("median_abs_log_error")["method"].to_list()
    x = np.arange(len(methods))
    for offset, step, colour in ((-0.2, False, _colour("real")), (0.2, True,
                                                                 _colour("random_walk"))):
        part = {r["method"]: r["median_abs_log_error"]
                for r in summary.filter(pl.col("step") == step).iter_rows(named=True)}
        ax.bar(x + offset, [part.get(m, np.nan) for m in methods], width=0.38, color=colour,
               label="amplitude stepped 16-128 bars ago" if step else "stationary")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("windowed_dwt_", "windowed DWT, ").replace("_", " ")
                        for m in methods], rotation=20, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("median |ln(estimate / target)| (log scale)")
    _legend(ax)
    return _finish(fig, path, config, "Boundary study: latest-state energy at the right edge")


def plot_cross_timeframe(bands: pl.DataFrame, *, path: Path, config: WaveletConfig,
                         title: str) -> Path:
    """Excess band share over the random walk against physical period, one line per timeframe."""
    fig, axes = _figure(config)
    ax = axes[0, 0]
    ramp = ["#86b6ef", "#4f94e3", "#2a78d6", "#1a5aa8", "#104281"]
    order = {"1m": 0, "5m": 1, "15m": 2, "30m": 3, "1h": 4}
    for (timeframe,), part in sorted(bands.group_by("timeframe"),
                                     key=lambda g: order.get(str(g[0][0]), 9)):
        part = part.sort("period_seconds")
        ax.plot(part["period_seconds"].to_numpy() / 3600, part["excess_over_random_walk"]
                .to_numpy(), marker="o", markersize=4, lw=1.6,
                color=ramp[order.get(str(timeframe), 2)], label=str(timeframe))
    ax.axhline(0.0, color=BASELINE, lw=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("band period (hours, log scale)")
    ax.set_ylabel("median share, XAUUSD minus detrended random walk")
    _legend(ax)
    return _finish(fig, path, config, title)
