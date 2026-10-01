"""Autocorrelation and Ljung-Box testing.

The ACF of raw returns, of ``|r|`` and of ``r^2`` are computed side by side
because they answer different questions. Raw-return autocorrelation speaks to
predictability of *direction*; the absolute and squared series speak to
volatility clustering, which is usually far stronger and far more persistent.

On reading these numbers
------------------------
With millions of bars the confidence band around zero is tiny -
:math:`1.96/\\sqrt{n}` at n = 4,000,000 is about 0.001 - so almost any
autocorrelation is "statistically significant". That is a statement about
sample size, not about tradability. Every row therefore carries the band, the
sample count, and a plain-language effect-size label, and
:func:`ljung_box_test` reports the mean absolute autocorrelation alongside the
p-value so the magnitude is never lost behind the significance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import polars as pl
from scipy import stats
from scipy.stats import chi2
from statsmodels.tsa.stattools import acf as sm_acf

__all__ = [
    "AcfResult",
    "LjungBoxResult",
    "acf_absolute_returns",
    "acf_frame",
    "acf_returns",
    "acf_squared_returns",
    "autocorrelation",
    "effect_size_label",
    "ljung_box_test",
]

#: Rough bands for talking about |rho| without implying tradability.
#: Deliberately conservative: the top band says "large for a financial return
#: series", not "profitable".
_EFFECT_BANDS: tuple[tuple[float, str], ...] = (
    (0.01, "negligible"),
    (0.05, "very small"),
    (0.10, "small"),
    (0.20, "moderate"),
    (float("inf"), "large"),
)


def effect_size_label(rho: float) -> str:
    """Plain-language magnitude for an autocorrelation coefficient."""
    if not np.isfinite(rho):
        return "undefined"
    magnitude = abs(float(rho))
    for threshold, label in _EFFECT_BANDS:
        if magnitude < threshold:
            return label
    return "large"  # pragma: no cover - the final band is unbounded


@dataclass
class AcfResult:
    """Autocorrelation function of one series, lag by lag."""

    series_name: str
    observations: int
    max_lag: int
    confidence_level: float
    lags: list[int]
    values: list[float]
    confidence_bound: float
    notes: list[str]

    def to_frame(self) -> pl.DataFrame:
        bound = self.confidence_bound
        return pl.DataFrame({
            "lag": self.lags,
            "autocorrelation": self.values,
            "conf_lower": [-bound] * len(self.lags),
            "conf_upper": [bound] * len(self.lags),
            "significant": [abs(v) > bound for v in self.values],
            "effect_size": [effect_size_label(v) for v in self.values],
            "observations": [self.observations] * len(self.lags),
        }).with_columns(pl.lit(self.series_name).alias("series"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def autocorrelation(
    values: pl.Series | np.ndarray,
    *,
    max_lag: int = 100,
    series_name: str = "returns",
    confidence_level: float = 0.95,
) -> AcfResult:
    """Sample ACF for lags ``1..max_lag``.

    Lag 0 is dropped: it is identically 1 and only clutters plots and tables.

    The confidence bound is the standard large-sample band
    :math:`z_{\\alpha/2}/\\sqrt{n}`, which assumes the series is white noise.
    For returns whose *volatility* clusters - which is the normal case - this
    band is optimistic, so a marginal crossing should not be over-read.
    """
    array = _as_array(values)
    n = array.size
    if n < 3:
        raise ValueError(f"{series_name}: need at least 3 observations, got {n}")
    usable_lag = int(min(max_lag, n - 2))
    if usable_lag < 1:
        raise ValueError(f"{series_name}: too few observations for any lag")

    raw = sm_acf(array, nlags=usable_lag, fft=True, missing="drop")
    values_out = [float(v) for v in raw[1:]]
    lags = list(range(1, len(values_out) + 1))
    z = float(stats.norm.ppf(0.5 + confidence_level / 2.0))
    bound = z / np.sqrt(n)

    notes = [
        f"Confidence bound is +/-{bound:.6g} (z={z:.3f}/sqrt(n), n={n:,}), the "
        "white-noise band.",
        "With a large n almost any non-zero autocorrelation crosses this band. "
        "Read `effect_size`, not `significant`, for practical magnitude.",
    ]
    if usable_lag < max_lag:
        notes.append(f"max_lag reduced from {max_lag} to {usable_lag} by sample size.")

    return AcfResult(
        series_name=series_name,
        observations=int(n),
        max_lag=usable_lag,
        confidence_level=confidence_level,
        lags=lags,
        values=values_out,
        confidence_bound=float(bound),
        notes=notes,
    )


def acf_returns(returns: pl.Series | np.ndarray, **kwargs: Any) -> AcfResult:
    """ACF of raw returns - speaks to directional predictability."""
    kwargs.setdefault("series_name", "returns")
    return autocorrelation(returns, **kwargs)


def acf_absolute_returns(returns: pl.Series | np.ndarray, **kwargs: Any) -> AcfResult:
    """ACF of ``|r|`` - a standard volatility-clustering diagnostic."""
    kwargs.setdefault("series_name", "abs_returns")
    return autocorrelation(np.abs(_as_array(returns)), **kwargs)


def acf_squared_returns(returns: pl.Series | np.ndarray, **kwargs: Any) -> AcfResult:
    """ACF of ``r^2`` - the other standard volatility-clustering diagnostic."""
    kwargs.setdefault("series_name", "squared_returns")
    return autocorrelation(_as_array(returns) ** 2, **kwargs)


def acf_frame(results: list[AcfResult]) -> pl.DataFrame:
    """Stack several ACF results into one long-format table."""
    if not results:
        return pl.DataFrame()
    return pl.concat([r.to_frame() for r in results], how="vertical").select(
        "series", "lag", "autocorrelation", "conf_lower", "conf_upper",
        "significant", "effect_size", "observations",
    )


@dataclass
class LjungBoxResult:
    """Ljung-Box portmanteau test at one lag, with effect size attached."""

    series_name: str
    lag: int
    statistic: float
    p_value: float
    observations: int
    reject_at_5pct: bool
    mean_abs_autocorrelation: float
    max_abs_autocorrelation: float
    effect_size: str
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ljung_box_from_acf(
    rho: np.ndarray, *, n: int, lags: list[int]
) -> dict[int, tuple[float, float]]:
    r"""Ljung-Box statistic and p-value from a precomputed ACF.

    .. math::

        Q(h) = n(n+2) \sum_{k=1}^{h} rac{ho_k^2}{n-k}

    with :math:`Q(h) \sim \chi^2_h` under the null. This is the definition
    statsmodels implements; only the route to :math:`ho` differs.
    """
    weights = np.asarray(rho, dtype=np.float64) ** 2 / (
        n - np.arange(1, rho.size + 1, dtype=np.float64)
    )
    cumulative = np.cumsum(weights) * n * (n + 2)
    return {
        lag: (float(cumulative[lag - 1]), float(chi2.sf(cumulative[lag - 1], lag)))
        for lag in lags
    }


def ljung_box_test(
    values: pl.Series | np.ndarray,
    *,
    lags: tuple[int, ...] = (5, 10, 20, 50),
    series_name: str = "returns",
    large_sample_threshold: int = 100_000,
) -> list[LjungBoxResult]:
    r"""Ljung-Box test of :math:`H_0: \rho_1 = \dots = \rho_k = 0`.

    Each result carries the mean and maximum ``|rho|`` over the tested lags, so
    a rejection can be read together with how large the underlying correlations
    actually are. A test on four million bars rejects for a mean ``|rho|`` of
    0.002; the rejection is real and the effect is negligible, and both facts
    belong in the report.
    """
    array = _as_array(values)
    n = array.size
    if n < 10:
        raise ValueError(f"{series_name}: need at least 10 observations, got {n}")

    usable = [int(k) for k in lags if 0 < int(k) < n - 1]
    if not usable:
        return []

    # statsmodels' acorr_ljungbox recomputes the autocorrelations with a direct
    # O(n^2) correlation, which costs ~25 s per call at n = 10^5. The statistic
    # is a closed form in rho, and rho is already available by FFT, so it is
    # evaluated here instead. `tests/test_autocorrelation.py` pins the result
    # against statsmodels.
    rho = sm_acf(array, nlags=max(usable), fft=True, missing="drop")[1:]
    table = _ljung_box_from_acf(rho, n=n, lags=usable)

    warning = None
    if n >= large_sample_threshold:
        warning = (
            f"n = {n:,}: the test has very high power here and will reject for "
            "economically trivial autocorrelation. Compare "
            "`mean_abs_autocorrelation` against transaction costs before reading "
            "anything into a rejection."
        )

    out: list[LjungBoxResult] = []
    for lag in usable:
        statistic, p_value = table[lag]
        window = np.abs(rho[:lag])
        mean_abs = float(np.mean(window)) if window.size else float("nan")
        out.append(LjungBoxResult(
            series_name=series_name,
            lag=lag,
            statistic=statistic,
            p_value=p_value,
            observations=int(n),
            reject_at_5pct=bool(p_value < 0.05),
            mean_abs_autocorrelation=mean_abs,
            max_abs_autocorrelation=float(np.max(window)) if window.size else float("nan"),
            effect_size=effect_size_label(mean_abs),
            warning=warning,
        ))
    return out


def ljung_box_frame(results: list[LjungBoxResult]) -> pl.DataFrame:
    """Tabulate Ljung-Box results for file output."""
    if not results:
        return pl.DataFrame()
    return pl.DataFrame([r.to_dict() for r in results])


def _as_array(values: pl.Series | np.ndarray) -> np.ndarray:
    array = (
        values.drop_nulls().to_numpy()
        if isinstance(values, pl.Series)
        else np.asarray(values)
    )
    array = array.astype(np.float64, copy=False)
    return array[np.isfinite(array)]
