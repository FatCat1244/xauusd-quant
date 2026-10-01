r"""Is the OU model complete? Diagnostics of its innovations.

After fitting :math:`X_{t+1} = a + bX_t + \eta_t`, a well-specified model
leaves innovations that are close to white noise with a constant variance.
This module measures how far that is from true, and never hides it:

* distribution - mean, std, skewness, excess kurtosis, normality tests and
  Gaussian tail ratios (reusing :mod:`.distributions`);
* serial structure - ACF of :math:`\eta` for lags 1..50 with Ljung-Box
  (reusing :mod:`.autocorrelation`);
* volatility clustering - ACF of :math:`|\eta|` and :math:`\eta^2`, the
  McLeod-Li test (Ljung-Box on :math:`\eta^2`) and Engle's ARCH-LM;
* *structural* lags - single autocorrelations at the regression window
  ``N``, the OU window ``M`` and one trading day of bars. A rolling-regression
  residual carries mechanical structure at the window length; a periodicity
  that shows up at exactly ``N`` is the detrending, not the market. Measuring
  it here, in the time domain, is what lets Prompt #5 tell the two apart
  before any spectral method is applied.

Three innovation series are examined: the in-sample residuals of the
whole-sample static fit, and the one-step-ahead errors of the rolling fits
(raw and standardised by each window's own innovation std). Standardisation
removes the part of the volatility clustering the rolling window already
tracks, so comparing raw and standardised series shows how much is left.

Nothing here is modelled away: no GARCH, no stochastic volatility. The
evidence is reported and that is all.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from ..models.config import OUConfig
from .autocorrelation import acf_frame, autocorrelation, ljung_box_test
from .distributions import describe_distribution, gaussian_tail_comparison, normality_tests

__all__ = ["innovation_diagnostics", "lag_correlation"]


def lag_correlation(values: np.ndarray, lag: int) -> float | None:
    """Plain Pearson autocorrelation at one lag, computed directly."""
    x = values[np.isfinite(values)]
    if lag < 1 or x.size <= lag + 2:
        return None
    a, b = x[lag:], x[:-lag]
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _subsample(values: np.ndarray, limit: int) -> tuple[np.ndarray, bool]:
    if values.size <= limit:
        return values, False
    step = int(np.ceil(values.size / limit))
    return values[::step], True


def _arch_lm(values: np.ndarray, lags: int) -> tuple[float | None, float | None]:
    import warnings

    try:
        from statsmodels.stats.diagnostic import het_arch

        with warnings.catch_warnings():
            # statsmodels announces a future change of acorr_lm's return type;
            # only the statistic and p-value are used here, so either works.
            warnings.simplefilter("ignore", FutureWarning)
            stat, p_value, _, _ = het_arch(values, nlags=lags)
        return float(stat), float(p_value)
    except Exception:  # noqa: BLE001 - a failed test must not lose the rest
        return None, None


def innovation_diagnostics(
    series: dict[str, np.ndarray],
    *,
    source: str,
    config: OUConfig,
    structural_lags: dict[str, int] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """One summary row per innovation series, plus the long ACF table."""
    cfg = config.diagnostics
    rows: list[dict[str, Any]] = []
    acf_results = []
    for name, raw in series.items():
        values = np.asarray(raw, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size < max(100, cfg.max_lag * 3):
            rows.append({"source": source, "series": name, "observations": int(values.size),
                         "sufficient_data": False})
            continue
        summary = describe_distribution(values, name=name, quantiles=(0.01, 0.5, 0.99))
        row: dict[str, Any] = {
            "source": source,
            "series": name,
            "observations": int(values.size),
            "sufficient_data": True,
            "mean": summary.mean,
            "std": summary.std,
            "skewness": summary.skewness,
            "excess_kurtosis": summary.excess_kurtosis,
        }
        tails = {t["sigma"]: t for t in gaussian_tail_comparison(values, sigma_levels=(3, 4, 5))}
        for level, tail in tails.items():
            row[f"tail_ratio_{int(level)}sigma"] = tail["ratio_observed_to_gaussian"]
        sample, thinned = _subsample(values, cfg.max_observations)
        normality = normality_tests(sample)
        for test in normality["tests"]:
            row[f"{test['test']}_p"] = test["p_value"]
        row["tests_subsampled"] = thinned

        derived = {
            name: values,
            f"abs_{name}": np.abs(values),
            f"squared_{name}": values ** 2,
        }
        for label, data in derived.items():
            result = autocorrelation(data, max_lag=cfg.max_lag, series_name=label,
                                     confidence_level=cfg.confidence_level)
            acf_results.append(result)
            prefix = {name: "", f"abs_{name}": "abs_", f"squared_{name}": "sq_"}[label]
            rho = np.asarray(result.values)
            row[f"{prefix}acf1"] = float(rho[0])
            if not prefix:
                row["acf2"] = float(rho[1]) if rho.size > 1 else None
                row["acf5"] = float(rho[4]) if rho.size > 4 else None
                row["max_abs_acf"] = float(np.max(np.abs(rho)))
                row["lag_of_max_abs_acf"] = int(np.argmax(np.abs(rho)) + 1)
                row["acf_band"] = result.confidence_bound
                row["significant_lags"] = int(np.sum(np.abs(rho) > result.confidence_bound))
            else:
                row[f"{prefix}acf_mean_1_{cfg.max_lag}"] = float(np.mean(rho))
        for test in ljung_box_test(values, lags=cfg.ljung_box_lags, series_name=name):
            row[f"ljung_box_p_{test.lag}"] = test.p_value
            row[f"ljung_box_mean_abs_rho_{test.lag}"] = test.mean_abs_autocorrelation
        mcleod = ljung_box_test(values ** 2, lags=(10,), series_name=f"squared_{name}")
        row["mcleod_li_p_10"] = mcleod[0].p_value if mcleod else None
        row["arch_lm_stat"], row["arch_lm_p"] = _arch_lm(sample, cfg.arch_lags)
        if cfg.structural_lags and structural_lags:
            for label, lag in structural_lags.items():
                row[f"acf_at_{label}"] = lag_correlation(values, lag)
                row[f"abs_acf_at_{label}"] = lag_correlation(np.abs(values), lag)
        rows.append(row)

    acf = acf_frame(acf_results)
    if not acf.is_empty():
        acf = acf.with_columns(pl.lit(source).alias("source")).select(
            "source", pl.exclude("source")
        )
    return pl.DataFrame(rows, infer_schema_length=None), acf
