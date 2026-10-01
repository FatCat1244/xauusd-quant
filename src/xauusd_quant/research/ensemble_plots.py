"""Descriptive figures of the ensemble predictive research (Prompt #11).

Figures read the per-pair summary, diversity, prediction, stability, weight
and leave-out tables written under ``results/ensemble_research/<tf>/``.
Only existing research results are drawn; no models are fitted or selected.
Per-bar reads project the required columns and exclude the reserved period
before collection. Missing tables or required columns skip their figures.
Each axis shows one metric; missing statistics remain unavailable.
"""

from __future__ import annotations

import gc
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from .feature_plots import _DIVERGING, COLOURS, _figure, _legend  # noqa: E402
from .ou_plots import BASELINE, INK_SECONDARY, MUTED  # noqa: E402

__all__ = ["write_ensemble_figures"]

_RESERVED_START = datetime(2022, 1, 1)
_METHODS = (
    "simple_average",
    "median",
    "performance_weighted_s50",
    "diversity_weighted_l50",
    "stacking",
    "regime_conditioned",
    "volatility_conditioned",
    "dynamic",
)
_SCHEMES = ("performance_weighted_s50", "diversity_weighted_l50", "stacking")
_STATES = ("regime_state", "vol_quartile", "spread_quartile", "session")


@dataclass(frozen=True)
class _Pair:
    directory: Path
    title: str
    task: str
    metric: str
    universe: tuple[str, ...]
    frozen: str | None

    @property
    def series(self) -> list[str]:
        names = ["best_individual", "simple_average"]
        if self.frozen:
            names.append(self.frozen)
        return list(dict.fromkeys(names))


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _pair(directory: Path) -> _Pair:
    summary = _json(directory / "summary.json")
    freeze = _json(directory / "freeze_decision.json")
    target = str(summary.get("target", directory.parent.name))
    horizon = str(summary.get("horizon", directory.name.removeprefix("h")))
    task = str(summary.get("task", ""))
    metric = str(summary.get("metric", ""))
    universe = summary.get("universe", [])
    frozen = freeze.get("frozen_series")
    return _Pair(
        directory=directory,
        title=f"{target} h{horizon}",
        task=task,
        metric=metric,
        universe=tuple(dict.fromkeys(str(v) for v in universe))
        if isinstance(universe, list) else (),
        frozen=frozen if isinstance(frozen, str) and frozen else None,
    )


def _table(pair: _Pair, name: str) -> Path | None:
    for root in (pair.directory / "tables", pair.directory):
        path = root / f"{name}.parquet"
        if path.is_file():
            return path
    return None


def _scan(
    pair: _Pair,
    name: str,
    columns: list[str],
    *,
    per_bar: bool = False,
) -> pl.LazyFrame | None:
    path = _table(pair, name)
    if path is None:
        return None
    frame = pl.scan_parquet(path)
    schema = frame.collect_schema()
    required = set(columns)
    if per_bar:
        required.add("timestamp")
    if not required <= set(schema.names()):
        return None
    if per_bar:
        # Apply the research-period predicate before materializing any rows.
        frame = frame.filter(
            pl.col("timestamp") < pl.lit(_RESERVED_START).cast(schema["timestamp"])
        )
    return frame.select(columns)


def _read(pair: _Pair, name: str, columns: list[str]) -> pl.DataFrame:
    frame = _scan(pair, name, columns)
    return frame.collect() if frame is not None else pl.DataFrame()


def _values(frame: pl.DataFrame, column: str) -> NDArray[np.float64]:
    return np.asarray(frame[column].cast(pl.Float64).to_numpy(), dtype=np.float64)


def _finite(frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    return frame.filter(
        pl.all_horizontal([pl.col(c).cast(pl.Float64).is_finite() for c in columns])
    )


def _natural(value: Any) -> tuple[tuple[int, int | str], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"(\d+)", str(value))
        if part
    )


def _grid(n: int, cols: int = 3) -> tuple[Any, list[Any]]:
    cols = min(cols, n)
    rows = (n + cols - 1) // cols
    fig, axes = _figure(rows, cols, width=5.2 * cols, height=4.2 * rows)
    flat = list(axes.ravel())
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def _save(fig: Any, path: Path, title: str) -> str:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.suptitle(title, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(path, dpi=110)
    finally:
        plt.close(fig)
    return str(path)


def _destination(pair: _Pair, name: str) -> Path:
    return pair.directory / "plots" / f"{name}.png"


def _series_label(pair: _Pair, series: str) -> str:
    return f"{series} (frozen)" if series == pair.frozen else series


def _correlation(pair: _Pair, column: str, name: str) -> str | None:
    table = _read(pair, "diversity_matrix", ["a", "b", column])
    models = list(pair.universe)
    if table.is_empty() or not models:
        return None
    matrix = np.full((len(models), len(models)), np.nan)
    index = {model: i for i, model in enumerate(models)}
    for a, b, value in table.iter_rows():
        if a in index and b in index and value is not None:
            matrix[index[a], index[b]] = matrix[index[b], index[a]] = float(value)
    np.fill_diagonal(matrix, 1.0)
    size = max(6.0, 0.65 * len(models) + 2.0)
    fig, axes = _figure(width=size, height=size)
    ax = axes[0, 0]
    image = ax.imshow(
        np.ma.masked_invalid(matrix), cmap=_DIVERGING, vmin=-1, vmax=1,
        interpolation="nearest",
    )
    ax.grid(False)
    ax.set_xticks(range(len(models)), models, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(models)), models, fontsize=7)
    for i in range(len(models)):
        for j in range(len(models)):
            value = matrix[i, j]
            text = f"{value:.2f}" if np.isfinite(value) else "NA"
            ax.text(j, i, text, ha="center", va="center", fontsize=6)
    fig.colorbar(image, ax=ax, fraction=0.04, label=column)
    title = "Spearman prediction correlation" if column == "spearman" else "Error correlation"
    return _save(fig, _destination(pair, name), f"{pair.title}: {title}")


def _disagreement(pair: _Pair) -> str | None:
    frame = _scan(
        pair, "disagreement", ["timestamp", "block", "prediction_std"], per_bar=True,
    )
    if frame is None:
        return None
    frame = frame.filter(pl.col("timestamp").is_not_null())
    daily = (
        frame.with_columns(
            pl.col("timestamp").dt.date().alias("date"),
            pl.when(pl.col("prediction_std").is_finite())
            .then(pl.col("prediction_std")).otherwise(None).alias("prediction_std"),
        )
        .group_by("date")
        .agg(pl.col("prediction_std").mean())
        .sort("date")
        .collect()
    )
    if daily.is_empty() or not np.isfinite(_values(daily, "prediction_std")).any():
        return None
    spans = (
        frame.group_by("block")
        .agg(
            pl.col("timestamp").min().alias("start"),
            pl.col("timestamp").max().alias("end"),
        )
        .sort("start")
        .collect()
    )
    fig, axes = _figure(width=11, height=4)
    ax = axes[0, 0]
    for i, (_, start, end) in enumerate(spans.iter_rows()):
        if i % 2 == 0:
            ax.axvspan(start, end + timedelta(microseconds=1), color=MUTED, alpha=0.12)
    ax.plot(daily["date"].to_list(), _values(daily, "prediction_std"), color=COLOURS[0])
    ax.set_xlabel("date")
    ax.set_ylabel("daily mean prediction_std")
    return _save(
        fig, _destination(pair, "disagreement_through_time"),
        f"{pair.title}: daily model disagreement; alternating walk-forward blocks",
    )


def _bars(
    ax: Any,
    table: pl.DataFrame,
    value: str,
    error: str,
    labels: list[str],
    colours: list[str],
) -> None:
    """Draw available errors only; an unavailable SE is never replaced by zero."""
    y = np.arange(table.height)
    values = _values(table, value)
    errors = _values(table, error)
    ax.barh(y, values, color=colours)
    available = np.isfinite(values) & np.isfinite(errors) & (errors >= 0)
    if available.any():
        ax.errorbar(
            values[available], y[available], xerr=errors[available],
            fmt="none", ecolor=INK_SECONDARY, elinewidth=0.8, capsize=2,
        )
    ax.set_yticks(y, labels, fontsize=7)


def _individual(pair: _Pair) -> str | None:
    table = _read(pair, "series_summary", ["series", "kind", "metric", "mean", "se"])
    if table.is_empty():
        return None
    table = _finite(
        table.filter(pl.col("kind").is_in(["individual", "method"])), ["mean"],
    ).sort("mean")
    metrics = table["metric"].drop_nulls().unique().to_list()
    if table.is_empty() or len(metrics) != 1:
        return None
    fig, axes = _figure(width=10, height=max(4, 0.28 * table.height + 1.8))
    ax = axes[0, 0]
    colours = [COLOURS[0] if kind == "individual" else COLOURS[2]
               for kind in table["kind"].to_list()]
    _bars(ax, table, "mean", "se", table["series"].to_list(), colours)
    ax.set_xlabel(str(metrics[0]))
    return _save(
        fig, _destination(pair, "individual_vs_ensemble"),
        f"{pair.title}: individual (blue) and ensemble (green) means, ± SE",
    )


def _reliability(pair: _Pair) -> str | None:
    if pair.task != "classification":
        return None
    columns = ["label", "best_individual", "simple_average", "frozen_method"]
    frame = _scan(pair, "oos_ensemble_predictions", columns, per_bar=True)
    if frame is None:
        return None
    table = frame.collect()
    curves: list[tuple[str, NDArray[np.float64], NDArray[np.float64]]] = []
    for column in columns[1:]:
        part = _finite(table.select(column, "label"), [column, "label"])
        if part.is_empty():
            continue
        predictions = _values(part, column)
        outcomes = _values(part, "label")
        # Unique quantile boundaries keep tied predictions together.
        edges = np.unique(np.quantile(predictions, np.linspace(0, 1, 21)))
        bins = np.searchsorted(edges[1:-1], predictions, side="right")
        occupied = np.unique(bins)
        x = np.array([predictions[bins == b].mean() for b in occupied])
        y = np.array([outcomes[bins == b].mean() for b in occupied])
        label = column
        if column == "frozen_method" and pair.frozen:
            label = f"frozen_method ({pair.frozen})"
        curves.append((label, x, y))
    if not curves:
        return None
    fig, axes = _figure(width=6, height=5)
    ax = axes[0, 0]
    for i, (label, x, y) in enumerate(curves):
        ax.plot(x, y, marker="o", ms=3, lw=1.3, color=COLOURS[i], label=label)
    ax.plot([0, 1], [0, 1], color=BASELINE, lw=0.8, ls="--")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("mean label")
    _legend(ax, fontsize=7)
    return _save(
        fig, _destination(pair, "reliability"),
        f"{pair.title}: reliability, 20 quantile bins (fewer for tied predictions)",
    )


def _curve(pair: _Pair, *, decay: bool) -> str | None:
    name = "alpha_decay" if decay else "deciles"
    x = "horizon" if decay else "decile"
    y = "rank_ic" if decay else "mean_outcome"
    table = _read(pair, name, ["series", x, y])
    if table.is_empty():
        return None
    selected = pl.col("series").is_in(pair.series)
    if decay:
        selected = selected | pl.col("series").str.starts_with("model:")
    table = table.filter(selected)
    if decay:
        table = table.filter(pl.col(x) > 0)
    if table.is_empty() or not np.isfinite(_values(table, y)).any():
        return None
    fig, axes = _figure(width=7, height=4.5)
    ax = axes[0, 0]
    if decay:
        models = table.filter(pl.col("series").str.starts_with("model:"))
        for model in models["series"].unique().sort().to_list():
            part = models.filter(pl.col("series") == model).sort(x)
            ax.plot(part[x].to_list(), _values(part, y), color=MUTED, lw=0.6, alpha=0.6)
    for i, series in enumerate(pair.series):
        part = table.filter(pl.col("series") == series).sort(x)
        if not part.is_empty():
            ax.plot(
                part[x].to_list(), _values(part, y), marker="o", ms=3, lw=1.4,
                color=COLOURS[i], label=_series_label(pair, series),
            )
    if decay:
        ax.set_xscale("log")
        ax.axhline(0, color=BASELINE, lw=0.7)
    ax.set_xlabel("outcome horizon (bars, log)" if decay else "prediction decile")
    ax.set_ylabel(y)
    _legend(ax, fontsize=7)
    title = "rank IC by outcome horizon; grey: individual models" if decay else (
        "mean outcome by prediction decile"
    )
    return _save(
        fig, _destination(pair, "ic_decay" if decay else "deciles"),
        f"{pair.title}: {title}",
    )


def _primary(pair: _Pair) -> str | None:
    if pair.task == "classification":
        return "log_loss_skill"
    if pair.task == "regression":
        return "rank_ic"
    return None


def _yearly(pair: _Pair) -> str | None:
    metric = _primary(pair)
    if metric is None:
        return None
    table = _read(
        pair, "stability", ["grouping", "period", "series", "few_rows", metric],
    )
    if table.is_empty():
        return None
    table = table.filter(
        (pl.col("grouping") == "year") & pl.col("series").is_in(pair.series)
    )
    if table.is_empty() or not np.isfinite(_values(table, metric)).any():
        return None
    years = sorted(table["period"].drop_nulls().unique().to_list(), key=_natural)
    positions = {year: i for i, year in enumerate(years)}
    fig, axes = _figure(width=9, height=4.5)
    ax = axes[0, 0]
    for i, series in enumerate(pair.series):
        part = table.filter(pl.col("series") == series).sort("period")
        if part.is_empty():
            continue
        x = np.array([positions[year] for year in part["period"].to_list()])
        y = _values(part, metric)
        ax.plot(x, y, color=COLOURS[i], lw=1.2, label=_series_label(pair, series))
        flags = part["few_rows"].to_list()
        for flag, face in ((False, COLOURS[i]), (True, "none"), (None, "none")):
            mask = np.array([value is flag for value in flags]) & np.isfinite(y)
            if mask.any():
                ax.scatter(x[mask], y[mask], s=20, facecolors=face, edgecolors=COLOURS[i])
    ax.set_xticks(range(len(years)), [str(year) for year in years], fontsize=7)
    ax.axhline(0, color=BASELINE, lw=0.7)
    ax.set_xlabel("year")
    ax.set_ylabel(metric)
    _legend(ax, fontsize=7)
    return _save(
        fig, _destination(pair, "performance_by_year"),
        f"{pair.title}: yearly {metric}; hollow: few rows or unavailable row flag",
    )


def _states(pair: _Pair) -> str | None:
    metric = _primary(pair)
    if metric is None:
        return None
    table = _read(pair, "stability", ["grouping", "period", "label", "series", metric])
    if table.is_empty():
        return None
    table = table.filter(pl.col("series").is_in(pair.series))
    groups = [
        group for group in _STATES
        if not table.filter(pl.col("grouping") == group).is_empty()
    ]
    if not groups or not np.isfinite(_values(table, metric)).any():
        return None
    fig, axes = _grid(len(groups), cols=2)
    for ax, group in zip(axes, groups, strict=True):
        part = table.filter(pl.col("grouping") == group)
        categories = part.select("period", "label").unique().sort("period")
        periods = categories["period"].to_list()
        x = np.arange(len(periods))
        width = 0.8 / len(pair.series)
        for i, series in enumerate(pair.series):
            sub = part.filter(pl.col("series") == series)
            lookup = dict(sub.select("period", metric).rows())
            values = [
                np.nan if lookup.get(period) is None else float(lookup[period])
                for period in periods
            ]
            ax.bar(
                x + (i - (len(pair.series) - 1) / 2) * width,
                values, width=width, color=COLOURS[i],
                label=_series_label(pair, series),
            )
        ax.set_xticks(x, categories["label"].to_list(), rotation=30, ha="right", fontsize=7)
        ax.axhline(0, color=BASELINE, lw=0.7)
        ax.set_title(group, fontsize=9)
        ax.set_ylabel(metric)
        _legend(ax, fontsize=6)
    return _save(
        fig, _destination(pair, "performance_by_state"),
        f"{pair.title}: {metric} by state",
    )


def _model_styles(models: list[str]) -> dict[str, tuple[str, str]]:
    """Use family colours and line styles for feature-set variants."""
    families = list(dict.fromkeys(model.split("|", 1)[0] for model in models))
    styles = ("-", "--", ":", "-.")
    result: dict[str, tuple[str, str]] = {}
    counts: dict[str, int] = {}
    for model in models:
        family = model.split("|", 1)[0]
        index = families.index(family)
        colour = COLOURS[index] if index < len(COLOURS) else MUTED
        variant = counts.get(family, 0)
        result[model] = (colour, styles[variant % len(styles)])
        counts[family] = variant + 1
    return result


def _dynamic(pair: _Pair) -> str | None:
    models = list(pair.universe)
    if not models:
        return None
    frame = _scan(pair, "dynamic_weights", ["timestamp", *models], per_bar=True)
    if frame is None:
        return None
    table = frame.filter(pl.col("timestamp").is_not_null()).sort("timestamp").collect()
    if table.is_empty():
        return None
    values = np.vstack([_values(table, model) for model in models])
    if not np.isfinite(values).all(axis=0).any():
        return None
    # Mask whole unavailable stacks rather than silently treating missing weights as zero.
    values[:, ~np.isfinite(values).all(axis=0)] = np.nan
    styles = _model_styles(models)
    fig, axes = _figure(width=11, height=5)
    ax = axes[0, 0]
    areas = ax.stackplot(
        table["timestamp"].to_list(), *values,
        labels=models, colors=[styles[model][0] for model in models], alpha=0.8,
    )
    hatches = ("", "/", "\\", ".")
    for i, area in enumerate(areas):
        area.set_hatch(hatches[i % len(hatches)])
    ax.set_xlabel("timestamp")
    ax.set_ylabel("weight")
    _legend(ax, fontsize=6)
    return _save(
        fig, _destination(pair, "dynamic_weights"),
        f"{pair.title}: dynamic ensemble weights through time",
    )


def _weight_stability(pair: _Pair) -> str | None:
    table = _read(pair, "weights", ["scheme", "block", "model", "weight"])
    if table.is_empty():
        return None
    schemes = [
        scheme for scheme in _SCHEMES
        if not table.filter(pl.col("scheme") == scheme).is_empty()
    ]
    if not schemes:
        return None
    models = sorted(table["model"].drop_nulls().unique().to_list())
    styles = _model_styles(models)
    fig, axes = _grid(len(schemes))
    for ax, scheme in zip(axes, schemes, strict=True):
        part = table.filter(pl.col("scheme") == scheme)
        blocks = sorted(part["block"].drop_nulls().unique().to_list(), key=_natural)
        x = np.arange(len(blocks))
        for model in models:
            sub = part.filter(pl.col("model") == model)
            if sub.is_empty():
                continue
            lookup = dict(sub.select("block", "weight").rows())
            values = [
                np.nan if lookup.get(block) is None else float(lookup[block])
                for block in blocks
            ]
            colour, linestyle = styles[model]
            ax.plot(
                x, values, marker="o", ms=3, lw=1.1,
                color=colour, ls=linestyle, label=model,
            )
        ax.set_xticks(x, blocks, rotation=35, ha="right", fontsize=6)
        ax.set_xlabel("block")
        ax.set_ylabel("weight")
        ax.set_title(scheme, fontsize=9)
        _legend(ax, fontsize=5)
    return _save(
        fig, _destination(pair, "weight_stability"),
        f"{pair.title}: ensemble weights by walk-forward block",
    )


def _leave_out(pair: _Pair) -> str | None:
    table = _read(pair, "ablation/leave_out", ["removed", "series", "contribution", "se"])
    if table.is_empty() or not pair.metric:
        return None
    table = _finite(table, ["contribution"]).sort("contribution")
    if table.is_empty():
        return None
    labels = [
        f"{removed} ({'group' if str(series).startswith('lofo:') else 'model'})"
        for removed, series in table.select("removed", "series").rows()
    ]
    fig, axes = _figure(width=9, height=max(4, 0.3 * table.height + 1.8))
    ax = axes[0, 0]
    _bars(
        ax, table, "contribution", "se", labels,
        [COLOURS[0]] * table.height,
    )
    ax.axvline(0, color=BASELINE, lw=0.7)
    ax.set_xlabel(f"contribution ({pair.metric}; full minus without)")
    return _save(
        fig, _destination(pair, "leave_one_out"),
        f"{pair.title}: leave-out contributions, ± SE",
    )


def _overview(tf_dir: Path, pairs: list[_Pair]) -> str | None:
    panels: list[tuple[_Pair, pl.DataFrame]] = []
    for pair in pairs:
        table = _read(
            pair, "series_summary",
            ["series", "metric", "minus_best_individual", "se_vs_best_individual"],
        )
        if table.is_empty():
            continue
        table = _finite(
            table.filter(pl.col("series").is_in(_METHODS)), ["minus_best_individual"],
        )
        if table.is_empty():
            continue
        metrics = table["metric"].drop_nulls().unique().to_list()
        if len(metrics) != 1:
            continue
        order = {series: i for i, series in enumerate(_METHODS)}
        table = table.with_columns(
            pl.col("series").replace_strict(order, return_dtype=pl.Int64).alias("_order")
        ).sort("_order", descending=True)
        panels.append((pair, table))
    if not panels:
        return None
    fig, axes = _grid(len(panels), cols=2)
    for ax, (pair, table) in zip(axes, panels, strict=True):
        _bars(
            ax, table, "minus_best_individual", "se_vs_best_individual",
            table["series"].to_list(), [COLOURS[0]] * table.height,
        )
        ax.axvline(0, color=BASELINE, lw=0.7)
        ax.set_title(pair.title, fontsize=9)
        ax.set_xlabel(f"minus best individual ({table['metric'][0]}), ± SE", fontsize=8)
    return _save(
        fig, tf_dir / "plots" / "overview_methods.png",
        f"{tf_dir.name}: ensemble methods relative to the best individual",
    )


def write_ensemble_figures(tf_dir: Path, pair_dirs: list[Path]) -> list[str]:
    """Write available pair figures and the timeframe overview; return full paths.

    Inputs are existing ensemble research outputs, never raw or final-test
    datasets. Figures are saved independently and closed immediately.
    """
    written: list[str] = []
    pairs: list[_Pair] = []
    try:
        for directory in dict.fromkeys(pair_dirs):
            pair = _pair(directory)
            pairs.append(pair)
            for column, name in (
                ("spearman", "prediction_correlation"),
                ("error_correlation", "error_correlation"),
            ):
                path = _correlation(pair, column, name)
                if path is not None:
                    written.append(path)
            for plotter in (_disagreement, _individual, _reliability):
                path = plotter(pair)
                if path is not None:
                    written.append(path)
            for decay in (False, True):
                path = _curve(pair, decay=decay)
                if path is not None:
                    written.append(path)
            for plotter in (_yearly, _states, _dynamic, _weight_stability, _leave_out):
                path = plotter(pair)
                if path is not None:
                    written.append(path)
        path = _overview(tf_dir, pairs)
        if path is not None:
            written.append(path)
        return written
    finally:
        # Matplotlib figures retain reference cycles after close().
        gc.collect()
