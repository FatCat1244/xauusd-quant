"""Statistical research layer.

Characterises the XAUUSD dataset. It answers "what are the statistical
properties of this series?" and deliberately stops there - no entries, exits,
sizing, PnL or optimisation live here or anywhere else in the project yet.

Module map::

    config.py           typed research configuration (config/research.yaml)
    returns.py          backward returns (features) and forward returns (outcomes)
    distributions.py    moments, quantiles, Gaussian tail comparison, normality
    autocorrelation.py  ACF of r, |r|, r^2 plus Ljung-Box, with effect sizes
    volatility.py       rolling / realized / EWMA volatility, regime buckets
    stationarity.py     ADF and KPSS, whole-sample and segmented
    rolling.py          strictly backward-looking rolling statistics
    conditional.py      forward returns after extreme moves; reversal study
    intraday.py         hour, weekday and session behaviour
    spread.py           spread distribution and quote activity
    comparison.py       cross-timeframe and year-over-year comparison
    plots.py            reusable Matplotlib figures
    reports.py          orchestration and machine-readable output

Three rules hold throughout:

1. A feature at time ``t`` uses only data at or before ``t``. Forward returns
   exist solely as outcomes, always named ``fwd_*``, and never feed back into a
   feature, threshold or rolling statistic.
2. Statistical significance is reported next to effect size. With millions of
   bars almost anything is significant, so magnitude is what carries meaning.
3. Nothing here is a trading edge. No transaction cost, spread or slippage is
   applied, so no result should be read as tradable.
"""

from __future__ import annotations

from .autocorrelation import (
    acf_absolute_returns,
    acf_returns,
    acf_squared_returns,
    autocorrelation,
    ljung_box_test,
)
from .comparison import multi_timeframe_table, stability_analysis, timeframe_metrics
from .conditional import conditional_return_analysis, reversal_summary
from .config import ResearchConfig, load_research_config
from .distributions import (
    describe_distribution,
    gaussian_tail_comparison,
    normality_tests,
)
from .intraday import hourly_analysis, session_analysis, weekday_analysis
from .reports import ResearchReport, generate_report, provenance
from .returns import (
    cumulative_returns,
    forward_returns,
    log_returns,
    multi_period_returns,
    prepare_returns,
    simple_returns,
)
from .rolling import rolling_statistics
from .spread import activity_analysis, spread_analysis
from .stationarity import adf_test, kpss_test, segmented_stationarity, stationarity_report
from .volatility import (
    ewma_volatility,
    realized_volatility,
    rolling_volatility,
    volatility_regimes,
)

__all__ = [
    "ResearchConfig",
    "ResearchReport",
    "acf_absolute_returns",
    "acf_returns",
    "acf_squared_returns",
    "activity_analysis",
    "adf_test",
    "autocorrelation",
    "conditional_return_analysis",
    "cumulative_returns",
    "describe_distribution",
    "ewma_volatility",
    "forward_returns",
    "gaussian_tail_comparison",
    "generate_report",
    "hourly_analysis",
    "kpss_test",
    "ljung_box_test",
    "load_research_config",
    "log_returns",
    "multi_period_returns",
    "multi_timeframe_table",
    "normality_tests",
    "prepare_returns",
    "provenance",
    "realized_volatility",
    "reversal_summary",
    "rolling_statistics",
    "rolling_volatility",
    "segmented_stationarity",
    "session_analysis",
    "simple_returns",
    "spread_analysis",
    "stability_analysis",
    "stationarity_report",
    "timeframe_metrics",
    "volatility_regimes",
    "weekday_analysis",
]
