"""Stationarity testing.

ADF and KPSS are run together, and the reports keep them apart, because their
null hypotheses are opposites:

* **ADF** — :math:`H_0`: the series *has a unit root* (non-stationary).
  A small p-value is evidence **against** a unit root.
* **KPSS** — :math:`H_0`: the series *is stationary*.
  A small p-value is evidence **against** stationarity.

So the informative cases are the agreements, and the honest ones are the
disagreements:

===================  ==================  =================================
ADF rejects?         KPSS rejects?       Reading
===================  ==================  =================================
yes                  no                  consistent with stationarity
no                   yes                 consistent with a unit root
no                   no                  inconclusive; often too little data
yes                  yes                 conflicting; consider a trend, a
                                         structural break, or fractional
                                         integration
===================  ==================  =================================

Reducing this to a single boolean throws away exactly the information that
matters, so :class:`StationarityReport` carries both tests and a verdict drawn
from the table above.

Cost control: these tests converge long before tens of millions of points, and
ADF with ``autolag`` is superlinear. Above ``stationarity.max_observations`` a
deterministic, evenly-spaced subsample is used, and the report records that it
happened.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import polars as pl
from statsmodels.tsa.stattools import adfuller, kpss

from .config import ResearchConfig


# statsmodels 0.16 will switch these to returning a result object. Pin the
# current tuple shape explicitly where supported, so the upgrade cannot
# silently change what we unpack.
def _accepts_result_object(fn: Any) -> bool:
    """Whether this statsmodels version takes the `result_object` kwarg."""
    import inspect

    try:
        return "result_object" in inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return False


__all__ = [
    "SegmentedStationarity",
    "StationarityReport",
    "TestResult",
    "adf_test",
    "kpss_test",
    "segmented_stationarity",
    "stationarity_report",
    "subsample",
]


@dataclass
class TestResult:
    """One stationarity test, with everything needed to reproduce the reading."""

    test: str
    null_hypothesis: str
    statistic: float
    p_value: float
    observations: int
    lags_used: int | None
    critical_values: dict[str, float]
    reject_at_5pct: bool
    p_value_is_bounded: bool = False
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def subsample(values: np.ndarray, max_observations: int) -> tuple[np.ndarray, bool]:
    """Evenly-spaced deterministic subsample, if the series is too long.

    Even spacing (rather than a random draw) preserves the series' ordering and
    autocorrelation structure, which is the whole point of the test. Returns
    the array and whether subsampling occurred.
    """
    if values.size <= max_observations:
        return values, False
    step = int(np.ceil(values.size / max_observations))
    return values[::step], True


def adf_test(
    values: pl.Series | np.ndarray,
    *,
    regression: str = "c",
    autolag: str | None = "AIC",
    max_observations: int = 500_000,
) -> TestResult:
    r"""Augmented Dickey-Fuller test.

    :math:`H_0`: a unit root is present, i.e. the series is non-stationary.
    A **small p-value rejects the unit root**, which is evidence *for*
    stationarity.
    """
    array, was_subsampled = subsample(_as_array(values), max_observations)
    if array.size < 20:
        raise ValueError(f"ADF needs at least 20 observations, got {array.size}")

    kwargs: dict[str, Any] = {"regression": regression, "autolag": autolag}
    if _accepts_result_object(adfuller):
        kwargs["result_object"] = False
    stat, p_value, lags, nobs, crit, _ = adfuller(array, **kwargs)
    note = None
    if was_subsampled:
        note = (
            f"Evenly-spaced subsample of {array.size:,} used (cap "
            f"{max_observations:,}); ordering preserved."
        )
    return TestResult(
        test="adf",
        null_hypothesis="unit root present (non-stationary)",
        statistic=float(stat),
        p_value=float(p_value),
        observations=int(nobs),
        lags_used=int(lags),
        critical_values={k: float(v) for k, v in crit.items()},
        reject_at_5pct=bool(p_value < 0.05),
        note=note,
    )


def kpss_test(
    values: pl.Series | np.ndarray,
    *,
    regression: str = "c",
    nlags: str | int = "auto",
    max_observations: int = 500_000,
) -> TestResult:
    r"""KPSS test.

    :math:`H_0`: the series **is** stationary (around a level or a trend).
    A **small p-value rejects stationarity** - the opposite direction to ADF.

    ``statsmodels`` interpolates the p-value from a small table and warns when
    the statistic falls outside it; the result records that the p-value is a
    bound rather than an exact figure.
    """
    array, was_subsampled = subsample(_as_array(values), max_observations)
    if array.size < 20:
        raise ValueError(f"KPSS needs at least 20 observations, got {array.size}")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kwargs: dict[str, Any] = {"regression": regression, "nlags": nlags}
        if _accepts_result_object(kpss):
            kwargs["result_object"] = False
        stat, p_value, lags, crit = kpss(array, **kwargs)
        bounded = any("p-value is" in str(w.message).lower() for w in caught)

    notes = []
    if was_subsampled:
        notes.append(
            f"Evenly-spaced subsample of {array.size:,} used (cap {max_observations:,})."
        )
    if bounded:
        notes.append(
            "p-value is outside the interpolation table and is reported as a bound, "
            "not an exact value."
        )
    return TestResult(
        test="kpss",
        null_hypothesis="series is stationary",
        statistic=float(stat),
        p_value=float(p_value),
        observations=int(array.size),
        lags_used=int(lags),
        critical_values={k: float(v) for k, v in crit.items()},
        reject_at_5pct=bool(p_value < 0.05),
        p_value_is_bounded=bounded,
        note="; ".join(notes) or None,
    )


@dataclass
class StationarityReport:
    """ADF and KPSS for one series, plus a verdict that respects both."""

    series_name: str
    observations: int
    adf: dict[str, Any] | None = None
    kpss: dict[str, Any] | None = None
    verdict: str = "not evaluated"
    interpretation: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _verdict(adf: TestResult | None, kp: TestResult | None) -> tuple[str, str]:
    """Combine the two tests without collapsing them into one boolean."""
    if adf is None and kp is None:
        return "not evaluated", "Neither test was enabled."
    if adf is not None and kp is None:
        return (
            "stationary (ADF only)" if adf.reject_at_5pct else "unit root (ADF only)",
            "Only ADF was run; KPSS would provide complementary evidence.",
        )
    if adf is None and kp is not None:
        return (
            "non-stationary (KPSS only)" if kp.reject_at_5pct else "stationary (KPSS only)",
            "Only KPSS was run; ADF would provide complementary evidence.",
        )

    assert adf is not None and kp is not None
    adf_rejects, kpss_rejects = adf.reject_at_5pct, kp.reject_at_5pct
    if adf_rejects and not kpss_rejects:
        return (
            "stationary",
            "ADF rejects a unit root and KPSS does not reject stationarity: both "
            "tests point the same way.",
        )
    if not adf_rejects and kpss_rejects:
        return (
            "non-stationary (unit root)",
            "ADF fails to reject a unit root and KPSS rejects stationarity: both "
            "tests point the same way.",
        )
    if not adf_rejects and not kpss_rejects:
        return (
            "inconclusive",
            "Neither test rejects its null. Usually a sign of limited data or low "
            "power rather than a meaningful finding.",
        )
    return (
        "conflicting",
        "Both tests reject their nulls, which are opposites. Often indicates a "
        "deterministic trend, a structural break, or fractional integration; "
        "inspect the series before drawing a conclusion.",
    )


def stationarity_report(
    values: pl.Series | np.ndarray,
    *,
    series_name: str,
    config: ResearchConfig,
) -> StationarityReport:
    """Run the configured tests on one series and combine them honestly."""
    array = _as_array(values)
    report = StationarityReport(series_name=series_name, observations=int(array.size))
    if array.size < 20:
        report.verdict = "not evaluated"
        report.interpretation = f"Only {array.size} observations; too few to test."
        return report

    cfg = config.stationarity
    adf_result = kpss_result = None
    if cfg.adf_enabled:
        try:
            adf_result = adf_test(
                array, regression=cfg.adf_regression, autolag=cfg.adf_autolag,
                max_observations=cfg.max_observations,
            )
            report.adf = adf_result.to_dict()
        except Exception as exc:  # noqa: BLE001 - one failed test must not lose the other
            report.notes.append(f"ADF failed: {type(exc).__name__}: {exc}")
    if cfg.kpss_enabled:
        try:
            kpss_result = kpss_test(
                array, regression=cfg.kpss_regression, nlags=cfg.kpss_nlags,
                max_observations=cfg.max_observations,
            )
            report.kpss = kpss_result.to_dict()
        except Exception as exc:  # noqa: BLE001
            report.notes.append(f"KPSS failed: {type(exc).__name__}: {exc}")

    report.verdict, report.interpretation = _verdict(adf_result, kpss_result)
    report.notes.append(
        "ADF H0 = unit root; KPSS H0 = stationarity. The two nulls are opposites, "
        "so agreement between the tests is what carries information."
    )
    return report


@dataclass
class SegmentedStationarity:
    """Stationarity re-tested within chronological segments."""

    series_name: str
    segment_by: str
    segments: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_frame(self) -> pl.DataFrame:
        if not self.segments:
            return pl.DataFrame()
        rows = []
        for seg in self.segments:
            adf, kp = seg.get("adf") or {}, seg.get("kpss") or {}
            rows.append({
                "segment": seg["segment"],
                "observations": seg["observations"],
                "adf_statistic": adf.get("statistic"),
                "adf_p_value": adf.get("p_value"),
                "adf_rejects_unit_root": adf.get("reject_at_5pct"),
                "kpss_statistic": kp.get("statistic"),
                "kpss_p_value": kp.get("p_value"),
                "kpss_rejects_stationarity": kp.get("reject_at_5pct"),
                "verdict": seg["verdict"],
            })
        return pl.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def segmented_stationarity(
    frame: pl.DataFrame,
    *,
    series_column: str,
    time_column: str,
    series_name: str,
    config: ResearchConfig,
) -> SegmentedStationarity:
    """Re-run the tests within each year or quarter.

    A market can change. Testing the whole history at once answers a different
    question from testing each year, and a verdict that flips between segments
    is itself a finding about structural stability.
    """
    cfg = config.stationarity.segmented
    result = SegmentedStationarity(series_name=series_name, segment_by=cfg.by)
    if not cfg.enabled:
        result.notes.append("Segmented stationarity is disabled in configuration.")
        return result

    usable = frame.select(time_column, series_column).drop_nulls()
    if usable.is_empty():
        result.notes.append("No usable observations.")
        return result

    label = (
        pl.col(time_column).dt.year().cast(pl.Utf8)
        if cfg.by == "year"
        else pl.col(time_column).dt.year().cast(pl.Utf8)
        + "Q"
        + pl.col(time_column).dt.quarter().cast(pl.Utf8)
    )
    tagged = usable.with_columns(label.alias("__segment"))

    for segment in sorted(tagged["__segment"].unique().to_list()):
        chunk = tagged.filter(pl.col("__segment") == segment)[series_column]
        if chunk.len() < cfg.min_observations:
            result.segments.append({
                "segment": segment,
                "observations": int(chunk.len()),
                "verdict": "skipped (too few observations)",
                "interpretation": (
                    f"{chunk.len():,} observations is below the configured minimum of "
                    f"{cfg.min_observations:,}."
                ),
                "adf": None,
                "kpss": None,
            })
            continue
        sub = stationarity_report(chunk, series_name=f"{series_name}[{segment}]", config=config)
        payload = sub.to_dict()
        payload["segment"] = segment
        result.segments.append(payload)

    result.notes.append(
        f"Each segment tested independently ({cfg.by}); a verdict that changes between "
        "segments is evidence of structural change, not a contradiction."
    )
    return result


def _as_array(values: pl.Series | np.ndarray) -> np.ndarray:
    array = (
        values.drop_nulls().to_numpy()
        if isinstance(values, pl.Series)
        else np.asarray(values)
    )
    array = array.astype(np.float64, copy=False)
    return array[np.isfinite(array)]
