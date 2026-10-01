r"""The series a spectral study runs on: the real data and its null controls.

Every source provides the same columns, built by the same code, so any
spectral statistic can be computed identically on each:

``regression_residual``  :math:`\epsilon_t` of the rolling regression (window N)
``ou_innovation``        :math:`\eta_t = X_t - (a_{t-1} + b_{t-1} X_{t-1})`,
                         the causal one-step error of the rolling OU fit (window M)
``abs_ou_innovation``    :math:`|\eta_t|`
``log_return``           :math:`r_t`

plus, for outcome and conditioning studies, the residual's rolling Z-score
(Prompt #3), the OU Z-score and equilibrium, the regression slope and R^2,
trailing volatility and the OU half-life and validity.

Null controls, all with the real bar count and fixed seeds:

``white_noise``      independent Gaussian noise *per series*, with that
                     series' own standard deviation. What spectra of pure
                     noise look like; no pipeline, so no outcome studies.
``random_walk``      a Gaussian random walk (step = the data's return std)
                     through the *same* rolling regression and rolling OU fit.
                     Rolling detrending alone can make a spectrum look
                     oscillatory; this is the control for that.
``shuffled_returns`` the real returns in random order, cumulated and run
                     through the same pipeline: the exact return distribution,
                     fat tails included, with every serial dependence gone.
``block_bootstrap``  the real returns resampled in blocks of ``block_size``
                     bars (circularly), cumulated and run through the same
                     pipeline: short-range dependence and volatility
                     clustering survive inside blocks, long-range structure
                     does not.

``block_bootstrap_<k>`` is the same bootstrap with blocks of ``k`` bars, for
block-size sensitivity; plain ``block_bootstrap`` keeps its own seed stream,
so it draws the identical series in every layer that uses the same seed.

The controls use the real timestamps position by position, and the real
missing-bar mask, so every source is cut into the same windows, years and
hours. The builders need only a regression window, an OU window and the
control settings (:class:`PipelineSettings`), so any research layer can
share them - and with the same seed, the same null series.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import polars as pl

from ..data.diagnostics import _classify
from ..data.resampler import parse_timeframe
from ..features.config import RegressionConfig
from ..features.rolling_regression import _rolling_zscore, rolling_ols
from ..features.store import RegressionFeatureStore
from ..models.config import OUConfig
from ..models.ornstein_uhlenbeck import rolling_ou
from ..utils.config import Config

__all__ = [
    "PIPELINE_COLUMNS",
    "ControlSettings",
    "PipelineSettings",
    "SourceData",
    "build_control",
    "bootstrap_block_size",
    "build_real_source",
    "missing_bar_slots",
]

#: Columns a source built through the full pipeline carries besides the inputs.
PIPELINE_COLUMNS: tuple[str, ...] = (
    "residual_zscore", "ou_zscore", "ou_mu", "regression_slope", "r_squared",
    "trailing_volatility", "ou_half_life_bars", "ou_valid", "ou_state_code",
)


class ControlSettings(Protocol):
    """The null-control settings a builder reads (seed and default block size)."""

    @property
    def seed(self) -> int: ...

    @property
    def block_size(self) -> int: ...


class PipelineSettings(Protocol):
    """What building a source needs from a layer's configuration."""

    @property
    def regression_window(self) -> int: ...

    @property
    def ou_window(self) -> int: ...

    @property
    def controls(self) -> ControlSettings: ...
_INPUTS = ("regression_residual", "ou_innovation", "abs_ou_innovation", "log_return")


@dataclass
class SourceData:
    """One source's series, aligned bar by bar with the real timestamps."""

    name: str
    timestamps: pl.Series
    columns: dict[str, np.ndarray]
    missing_slots: np.ndarray
    bar_seconds: float
    notes: list[str] = field(default_factory=list)

    @property
    def has_pipeline(self) -> bool:
        """Built through the regression and OU pipeline (outcome studies possible)."""
        return "residual_zscore" in self.columns

    def series(self, name: str) -> np.ndarray:
        return self.columns[name]

    @property
    def size(self) -> int:
        return self.timestamps.len()


def missing_bar_slots(timestamps: pl.Series, bar_seconds: float, config: Config) -> np.ndarray:
    """Expected-but-absent bar slots before each bar, counting unscheduled gaps only.

    Weekends, holiday weekends and the daily break are market closures on the
    trading-time axis and count as zero; every other gap longer than one bar
    (``holiday_or_unknown``) contributes its missing slots.
    """
    session = config.diagnostics.session
    step_us = int(bar_seconds * 1_000_000)
    frame = pl.DataFrame({"curr": timestamps}).with_columns(
        pl.col("curr").shift(1).alias("prev")).with_columns(
        ((pl.col("curr") - pl.col("prev")).dt.total_microseconds() // step_us - 1)
        .fill_null(0).alias("missing"))
    slots = frame["missing"].to_numpy().astype(np.float64)
    gaps = np.flatnonzero(slots > 0)
    if gaps.size:
        # Only the gap rows become Python datetimes (7.9M of them at 1m otherwise).
        index = pl.Series(gaps)
        prev = frame["prev"].gather(index).to_list()
        curr = frame["curr"].gather(index).to_list()
        for j, i in enumerate(gaps):
            if _classify(prev[j], curr[j], session) != "holiday_or_unknown":
                slots[i] = 0.0
    return slots


def _ou_columns(residual: np.ndarray, ou: OUConfig, ou_window: int) -> dict[str, np.ndarray]:
    """The rolling OU fit's causal innovation, Z-score, equilibrium and half-life."""
    fit = rolling_ou(residual, ou_window, rules=ou.validity_rules(),
                     chunk_rows=ou.numerical.chunk_rows)
    return {
        "ou_innovation": fit.innovation,
        "abs_ou_innovation": np.abs(fit.innovation),
        "ou_zscore": fit.zscore,
        "ou_mu": fit.mapping.mu,
        "ou_half_life_bars": fit.mapping.half_life_bars,
        "ou_valid": fit.mapping.valid.astype(np.float64),
        "ou_state_code": fit.mapping.state_code.astype(np.float64),
    }


def _pipeline(log_price: np.ndarray, *, regression: RegressionConfig,
              spectral: PipelineSettings, ou: OUConfig) -> dict[str, np.ndarray]:
    """Residual, Z-score, slope, R^2, volatility and OU columns of one log-price path."""
    n_reg = spectral.regression_window
    fit = rolling_ols(log_price, n_reg,
                      min_residual_std=regression.numerical.min_residual_std,
                      min_total_variance=regression.numerical.min_total_variance)
    z_window = regression.zscore.resolve_primary(n_reg)
    min_periods = max(3, int(round(z_window * regression.zscore.min_periods_ratio)))
    vol_window = regression.conditioning.volatility_window
    frame = pl.DataFrame({"residual": fit.residual, "log_price": log_price}).with_columns(
        pl.col("residual").fill_nan(None)).with_columns(
        _rolling_zscore("residual", z_window, min_periods=min_periods,
                        min_std=regression.numerical.min_residual_std).alias("z"),
        pl.col("log_price").diff().alias("log_return"),
    ).with_columns(
        pl.col("log_return").rolling_std(window_size=vol_window, min_samples=vol_window, ddof=1)
        .alias("trailing_volatility"))
    columns = {
        "regression_residual": fit.residual,
        "residual_zscore": frame["z"].fill_null(np.nan).to_numpy(),
        "log_return": frame["log_return"].fill_null(np.nan).to_numpy(),
        "trailing_volatility": frame["trailing_volatility"].fill_null(np.nan).to_numpy(),
        "regression_slope": fit.slope,
        "r_squared": fit.r_squared,
    }
    columns.update(_ou_columns(fit.residual, ou, spectral.ou_window))
    return columns


def build_real_source(
    config: Config, regression: RegressionConfig, ou: OUConfig, spectral: PipelineSettings,
    timeframe: str,
) -> SourceData:
    """The real XAUUSD series from the versioned feature store, plus the OU fit."""
    store = RegressionFeatureStore(config, regression)
    features = store.load_or_build(
        timeframe, spectral.regression_window,
        columns=["timestamp", "residual", "residual_zscore_rolling", "regression_slope",
                 "r_squared", "trailing_volatility", "log_return"])
    bar_seconds = parse_timeframe(timeframe).total_seconds()
    residual = features["residual"].fill_null(np.nan).to_numpy().astype(np.float64)
    columns = {
        "regression_residual": residual,
        "residual_zscore": features["residual_zscore_rolling"].fill_null(np.nan).to_numpy(),
        "log_return": features["log_return"].fill_null(np.nan).to_numpy(),
        "trailing_volatility": features["trailing_volatility"].fill_null(np.nan).to_numpy(),
        "regression_slope": features["regression_slope"].fill_null(np.nan).to_numpy(),
        "r_squared": features["r_squared"].fill_null(np.nan).to_numpy(),
    }
    columns.update(_ou_columns(residual, ou, spectral.ou_window))
    timestamps = features["timestamp"]
    slots = missing_bar_slots(timestamps, bar_seconds, config)
    return SourceData(name="real", timestamps=timestamps, columns=columns,
                      missing_slots=slots, bar_seconds=bar_seconds)


def _block_bootstrap(returns: np.ndarray, n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Circular block bootstrap of *returns* to length *n*."""
    starts = rng.integers(0, returns.size, size=int(np.ceil(n / block)))
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % returns.size
    return returns[idx]


def bootstrap_block_size(name: str, default: int) -> int | None:
    """Block size of a ``block_bootstrap`` / ``block_bootstrap_<k>`` control, else None."""
    if name == "block_bootstrap":
        return default
    prefix = "block_bootstrap_"
    if name.startswith(prefix) and name[len(prefix):].isdigit():
        return int(name[len(prefix):])
    return None


def build_control(
    name: str, real: SourceData, *, regression: RegressionConfig, ou: OUConfig,
    spectral: PipelineSettings,
) -> SourceData:
    """One null control, the length of *real*, from a fixed seed."""
    controls = spectral.controls
    order = {"white_noise": 1, "random_walk": 2, "shuffled_returns": 3, "block_bootstrap": 4}
    block = bootstrap_block_size(name, controls.block_size)
    if name in order:
        rng = np.random.default_rng(controls.seed + order[name])
    elif block is not None:
        if block < 2:
            raise ValueError(f"{name}: blocks must be at least 2 bars")
        rng = np.random.default_rng([controls.seed + order["block_bootstrap"], block])
    else:
        raise ValueError(f"unknown control {name!r}")
    n = real.size
    returns = real.columns["log_return"]
    finite = returns[np.isfinite(returns)]
    notes: list[str] = []
    if name == "white_noise":
        columns = {}
        for series in _INPUTS:
            values = real.columns[series]
            std = float(np.nanstd(values)) if np.isfinite(values).any() else 1.0
            columns[series] = rng.normal(0.0, std or 1.0, n)
        notes.append("independent Gaussian noise per series, matched standard deviation")
        return SourceData(name=name, timestamps=real.timestamps, columns=columns,
                          missing_slots=real.missing_slots, bar_seconds=real.bar_seconds,
                          notes=notes)
    if name == "random_walk":
        steps = rng.normal(0.0, float(np.std(finite, ddof=1)), n)
        notes.append("Gaussian steps with the data's return standard deviation")
    elif name == "shuffled_returns":
        steps = rng.permutation(finite)
        steps = np.resize(steps, n) if steps.size < n else steps[:n]
        notes.append("the real returns in random order")
    elif block is not None:
        steps = _block_bootstrap(finite, n, block, rng)
        notes.append(f"the real returns in circular blocks of {block} bars")
    else:
        raise ValueError(f"unknown control {name!r}")
    log_price = np.log(1000.0) + np.concatenate(([0.0], np.cumsum(steps[1:])))
    columns = _pipeline(log_price, regression=regression, spectral=spectral, ou=ou)
    return SourceData(name=name, timestamps=real.timestamps, columns=columns,
                      missing_slots=real.missing_slots, bar_seconds=real.bar_seconds,
                      notes=notes)
