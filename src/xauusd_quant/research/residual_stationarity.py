r"""Stationarity testing of rolling-regression residuals.

This module reuses :mod:`xauusd_quant.research.stationarity` and attaches the
interpretation a residual series specifically requires.

Why "ADF p < 0.05" is not enough here
-------------------------------------
The residual is not a free-standing time series. Each point is the deviation
from a *different* locally fitted line, and the fitting itself removes
low-frequency variation:

* **Detrending is mechanical.** Subtracting a local linear fit from any series
  - random walk included - produces a residual that looks far more stationary
  than the original. Applying ADF to a rolling-regression residual and
  rejecting the unit root therefore tells you mostly that the detrending
  worked, not that the market mean-reverts.
* **Windows overlap.** Consecutive residuals share ``N-1`` bars, so the
  effective sample size is far below the nominal one and the test's standard
  errors are too small.
* **The residual is bounded by construction.** OLS forces the in-window
  residuals to sum to zero, which pins the series near zero whether or not any
  economic force does.

:func:`residual_stationarity` therefore runs the same test on a **random-walk
control** detrended identically. If the real residual and the control give
similar verdicts, the verdict is a property of the method rather than of
XAUUSD, and the report says exactly that.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import polars as pl

from ..features.config import RegressionConfig, StationarityConfig
from ..features.rolling_regression import rolling_ols
from .stationarity import adf_test, kpss_test

__all__ = [
    "ResidualStationarity",
    "random_walk_control",
    "residual_stationarity",
    "segmented_residual_stationarity",
]

INTERPRETATION = (
    "ADF H0 = unit root; KPSS H0 = stationarity. For a ROLLING-REGRESSION "
    "RESIDUAL these tests are weak evidence about the market: local detrending "
    "makes almost any series look stationary, overlapping windows understate the "
    "standard errors, and OLS pins the in-window residuals to a zero mean. "
    "Compare the verdict against the random-walk control before reading anything "
    "into it."
)


@dataclass
class ResidualStationarity:
    """Stationarity of one residual series, beside its random-walk control."""

    timeframe: str
    window: int
    observations: int
    adf: dict[str, Any] | None = None
    kpss: dict[str, Any] | None = None
    verdict: str = "not evaluated"
    control_adf: dict[str, Any] | None = None
    control_kpss: dict[str, Any] | None = None
    control_verdict: str = "not evaluated"
    control_matches: bool | None = None
    interpretation: str = INTERPRETATION
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def random_walk_control(
    n: int, window: int, *, seed: int = 20260924, sigma: float = 0.001
) -> np.ndarray:
    """Residuals of the same rolling regression applied to a random walk.

    The control has no mean reversion whatsoever. Whatever stationarity verdict
    it produces is attributable purely to the detrending, which is the point.
    """
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.normal(0.0, sigma, n)) + np.log(400.0)
    fit = rolling_ols(walk, window)
    residual = fit.residual
    return residual[np.isfinite(residual)]


def residual_stationarity(
    frame: pl.DataFrame,
    *,
    timeframe: str,
    window: int,
    config: RegressionConfig,
    with_control: bool = True,
) -> ResidualStationarity:
    """Run ADF and KPSS on the residual, and on a matched random-walk control."""
    cfg = config.stationarity
    values = _finite(frame["residual"])
    result = ResidualStationarity(
        timeframe=timeframe, window=window, observations=int(values.size)
    )
    if values.size < 100:
        result.warnings.append(f"Only {values.size} residuals; tests skipped.")
        return result

    adf_result, kpss_result = _run_pair(values, cfg)
    result.adf = adf_result.to_dict() if adf_result else None
    result.kpss = kpss_result.to_dict() if kpss_result else None
    result.verdict = _verdict(adf_result, kpss_result)

    if with_control:
        control = random_walk_control(values.size + window, window)
        c_adf, c_kpss = _run_pair(control, cfg)
        result.control_adf = c_adf.to_dict() if c_adf else None
        result.control_kpss = c_kpss.to_dict() if c_kpss else None
        result.control_verdict = _verdict(c_adf, c_kpss)
        result.control_matches = result.verdict == result.control_verdict
        if result.control_matches:
            result.notes.append(
                "The random-walk control reaches the SAME verdict. The result is "
                "therefore consistent with an artefact of local detrending and is not "
                "by itself evidence of mean reversion in XAUUSD."
            )
        else:
            result.notes.append(
                f"The control verdict ({result.control_verdict}) differs from the "
                f"residual's ({result.verdict}), so the residual behaves differently "
                "from a detrended random walk. That is worth following up, but still "
                "is not a statement about tradability."
            )

    result.notes.append(
        f"Overlapping windows: consecutive residuals share {window - 1} of {window} "
        "bars, so the effective sample is far smaller than "
        f"{values.size:,} and the p-values are optimistic."
    )
    return result


def segmented_residual_stationarity(
    frame: pl.DataFrame, *, timeframe: str, window: int, config: RegressionConfig
) -> pl.DataFrame:
    """Re-run the tests within each chronological segment."""
    cfg = config.stationarity
    usable = frame.select("timestamp", "residual").drop_nulls()
    if usable.is_empty():
        return pl.DataFrame()

    label = (
        pl.col("timestamp").dt.year().cast(pl.Utf8)
        if cfg.segmented_by == "year"
        else pl.col("timestamp").dt.year().cast(pl.Utf8)
        + "Q" + pl.col("timestamp").dt.quarter().cast(pl.Utf8)
    )
    tagged = usable.with_columns(label.alias("segment"))

    rows = []
    for segment in sorted(tagged["segment"].unique().to_list()):
        chunk = _finite(tagged.filter(pl.col("segment") == segment)["residual"])
        row: dict[str, Any] = {
            "timeframe": timeframe, "window": window, "segment": segment,
            "observations": int(chunk.size),
        }
        if chunk.size < cfg.segment_min_observations:
            row["verdict"] = "skipped (too few observations)"
            rows.append(row)
            continue
        adf_result, kpss_result = _run_pair(chunk, cfg)
        row.update({
            "adf_statistic": adf_result.statistic if adf_result else None,
            "adf_p_value": adf_result.p_value if adf_result else None,
            "adf_rejects_unit_root": adf_result.reject_at_5pct if adf_result else None,
            "kpss_statistic": kpss_result.statistic if kpss_result else None,
            "kpss_p_value": kpss_result.p_value if kpss_result else None,
            "kpss_rejects_stationarity": kpss_result.reject_at_5pct if kpss_result else None,
            "verdict": _verdict(adf_result, kpss_result),
        })
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _run_pair(
    values: np.ndarray, cfg: StationarityConfig
) -> tuple[Any, Any]:
    """ADF and KPSS, each tolerating the other's failure."""
    adf_result = kpss_result = None
    if cfg.adf_enabled:
        try:
            adf_result = adf_test(
                values, regression=cfg.adf_regression, autolag=cfg.adf_autolag,
                max_observations=cfg.max_observations,
            )
        except Exception:  # noqa: BLE001 - one failed test must not lose the other
            adf_result = None
    if cfg.kpss_enabled:
        try:
            kpss_result = kpss_test(
                values, regression=cfg.kpss_regression, nlags=cfg.kpss_nlags,
                max_observations=cfg.max_observations,
            )
        except Exception:  # noqa: BLE001
            kpss_result = None
    return adf_result, kpss_result


def _verdict(adf_result: Any, kpss_result: Any) -> str:
    """Combine the two opposite nulls without collapsing them."""
    if adf_result is None and kpss_result is None:
        return "not evaluated"
    if kpss_result is None:
        return "stationary (ADF only)" if adf_result.reject_at_5pct else "unit root (ADF only)"
    if adf_result is None:
        return (
            "non-stationary (KPSS only)" if kpss_result.reject_at_5pct
            else "stationary (KPSS only)"
        )
    adf_rejects, kpss_rejects = adf_result.reject_at_5pct, kpss_result.reject_at_5pct
    if adf_rejects and not kpss_rejects:
        return "stationary"
    if not adf_rejects and kpss_rejects:
        return "non-stationary (unit root)"
    if not adf_rejects and not kpss_rejects:
        return "inconclusive"
    return "conflicting"


def _finite(series: pl.Series) -> np.ndarray:
    array = series.drop_nulls().to_numpy().astype(np.float64, copy=False)
    return array[np.isfinite(array)]
