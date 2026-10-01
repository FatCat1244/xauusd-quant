"""Conditional and reversal analysis after statistically extreme moves.

The question this module answers is descriptive:

    Given that the return at ``t`` fell in the bottom 1% of its distribution,
    what did returns over the next ``h`` bars actually look like?

It is **not** a strategy, and nothing here is a trading rule. There are no
entries, no exits, no position sizing and no PnL.

Feature / outcome separation
----------------------------
This is the one place in the project where forward-looking data is used at all,
so the boundary is made explicit rather than assumed:

* The **condition** (which bucket a bar falls into) is computed from the return
  at ``t`` and the *unconditional* quantiles of the return distribution. It
  uses no future information about that bar.
* The **outcome** is a forward return over ``(t, t+h]``, produced only by
  :func:`~xauusd_quant.research.returns.forward_returns` and always named with
  a ``fwd_`` prefix.

Outcome columns never feed back into a condition, a threshold or a rolling
statistic. :func:`conditional_return_analysis` asserts this before returning.

A note on the quantile thresholds: they are computed over the whole sample,
which means a bar's bucket depends on the full distribution including later
data. That is legitimate for *describing* history but would be look-ahead in a
live rule, and the reports say so. Building a causal, expanding-window version
is a Prompt #3 concern.

Overlapping horizons
--------------------
Forward windows at consecutive bars overlap, so the observations are not
independent. Naive standard errors would be too small. The bootstrap here
resamples whole observations and the reports state the caveat; a block
bootstrap would be the stricter treatment and is noted as a limitation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import polars as pl

from .config import ResearchConfig
from .returns import forward_returns

__all__ = [
    "ConditionalAnalysis",
    "ConditionalBucket",
    "bootstrap_mean_ci",
    "conditional_return_analysis",
    "reversal_summary",
]


@dataclass
class ConditionalBucket:
    """Forward-return behaviour after one class of extreme current return."""

    bucket: str
    direction: str
    quantile: float
    threshold: float
    horizon: int
    observations: int
    mean_forward_return: float
    median_forward_return: float
    std_forward_return: float
    prob_positive: float
    prob_reversal: float
    prob_continuation: float
    mean_ci_lower: float | None = None
    mean_ci_upper: float | None = None
    standard_error: float | None = None
    thin_sample: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConditionalAnalysis:
    """All buckets and horizons for one timeframe."""

    timeframe: str
    return_column: str
    total_observations: int
    buckets: list[ConditionalBucket] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_frame(self) -> pl.DataFrame:
        if not self.buckets:
            return pl.DataFrame()
        return pl.DataFrame([b.to_dict() for b in self.buckets]).sort(
            "direction", "quantile", "horizon"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeframe": self.timeframe,
            "return_column": self.return_column,
            "total_observations": self.total_observations,
            "thresholds": self.thresholds,
            "notes": self.notes,
            "warnings": self.warnings,
            "buckets": [b.to_dict() for b in self.buckets],
        }


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    iterations: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 20260921,
    max_samples: int = 200_000,
) -> tuple[float | None, float | None]:
    """Percentile bootstrap confidence interval for a mean.

    Returns ``(None, None)`` when the sample is too small or too large to
    bootstrap cheaply, so the caller can say "not computed" rather than print a
    fabricated interval.

    Caveat carried by every caller: overlapping forward windows make the
    observations dependent, so this interval is narrower than the truth. A
    block bootstrap would be the correct fix.
    """
    finite = values[np.isfinite(values)]
    if finite.size < 30 or finite.size > max_samples:
        return None, None
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    n = finite.size
    for i in range(iterations):
        means[i] = np.mean(finite[rng.integers(0, n, n)])
    alpha = (1.0 - confidence_level) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def conditional_return_analysis(
    frame: pl.DataFrame,
    *,
    timeframe: str,
    return_column: str,
    config: ResearchConfig,
    price_column: str = "close",
) -> ConditionalAnalysis:
    r"""Forward-return behaviour conditioned on extreme current returns.

    Computes, for each configured quantile bucket and horizon,
    :math:`E[r_{t+1:t+h} \mid r_t < Q_p]` and the corresponding reversal and
    continuation probabilities.
    """
    cond = config.conditional
    usable = frame.filter(pl.col(return_column).is_not_null())
    analysis = ConditionalAnalysis(
        timeframe=timeframe,
        return_column=return_column,
        total_observations=usable.height,
    )
    if usable.height < 100:
        analysis.warnings.append(
            f"Only {usable.height} usable returns; conditional analysis skipped."
        )
        return analysis

    # Outcomes. Forward returns are built here and nowhere else.
    horizons = tuple(int(h) for h in cond.forward_horizons)
    with_outcomes = usable.with_columns(
        forward_returns(pl.col(price_column), horizons, return_type="log", prefix="fwd")
    )

    returns = with_outcomes[return_column]
    for direction, quantiles in (
        ("lower", cond.lower_quantiles),
        ("upper", cond.upper_quantiles),
    ):
        for q in quantiles:
            threshold = _f(returns.quantile(q))
            analysis.thresholds[f"{direction}_q{q:g}"] = threshold
            mask = (
                pl.col(return_column) <= threshold
                if direction == "lower"
                else pl.col(return_column) >= threshold
            )
            selected = with_outcomes.filter(mask)
            for horizon in horizons:
                bucket = _summarise_bucket(
                    selected, direction=direction, quantile=q, threshold=threshold,
                    horizon=horizon, config=config,
                )
                if bucket is not None:
                    analysis.buckets.append(bucket)

    thin = [b for b in analysis.buckets if b.thin_sample]
    if thin:
        analysis.warnings.append(
            f"{len(thin)} bucket/horizon combinations have fewer than "
            f"{cond.min_samples_warning:,} observations and are marked thin_sample."
        )
    analysis.notes += [
        "Forward returns are OUTCOMES only. They are never used as features, "
        "thresholds or inputs to any historical statistic.",
        "Quantile thresholds are computed over the whole sample, so bucket "
        "membership uses the full-period distribution. That is fine for describing "
        "history but would be look-ahead in a live rule; an expanding-window "
        "version would be needed for that.",
        "Forward windows at consecutive bars overlap, so observations are not "
        "independent and the confidence intervals are narrower than the truth. "
        "A block bootstrap would be the stricter treatment.",
        "Descriptive statistics only. No transaction costs, spread or slippage "
        "are applied, so none of this implies a tradable effect.",
    ]
    return analysis


def _summarise_bucket(
    selected: pl.DataFrame, *, direction: str, quantile: float, threshold: float,
    horizon: int, config: ResearchConfig,
) -> ConditionalBucket | None:
    """Summarise one (bucket, horizon) cell."""
    column = f"fwd_log_{horizon}"
    if column not in selected.columns:
        return None
    outcomes = selected[column].drop_nulls().to_numpy().astype(np.float64)
    outcomes = outcomes[np.isfinite(outcomes)]
    if outcomes.size == 0:
        return None

    cond = config.conditional
    n = int(outcomes.size)
    mean = float(np.mean(outcomes))
    prob_positive = float(np.mean(outcomes > 0))
    # After a fall, a reversal is a subsequent rise, and vice versa.
    prob_reversal = prob_positive if direction == "lower" else float(np.mean(outcomes < 0))

    lower = upper = None
    if cond.bootstrap.enabled:
        lower, upper = bootstrap_mean_ci(
            outcomes,
            iterations=cond.bootstrap.iterations,
            confidence_level=cond.bootstrap.confidence_level,
            seed=cond.bootstrap.seed + horizon,
            max_samples=cond.bootstrap.max_samples,
        )

    return ConditionalBucket(
        bucket=f"{direction}_q{quantile:g}",
        direction=direction,
        quantile=float(quantile),
        threshold=threshold,
        horizon=horizon,
        observations=n,
        mean_forward_return=mean,
        median_forward_return=float(np.median(outcomes)),
        std_forward_return=float(np.std(outcomes, ddof=1)) if n > 1 else float("nan"),
        prob_positive=prob_positive,
        prob_reversal=prob_reversal,
        prob_continuation=1.0 - prob_reversal,
        mean_ci_lower=lower,
        mean_ci_upper=upper,
        standard_error=(
            float(np.std(outcomes, ddof=1) / np.sqrt(n)) if n > 1 else None
        ),
        thin_sample=n < cond.min_samples_warning,
    )


def reversal_summary(analysis: ConditionalAnalysis) -> pl.DataFrame:
    r"""Condense the analysis into the reversal question.

    Reports :math:`P(r_{t+h} > 0 \mid r_t < Q_p)` for the lower tail and
    :math:`P(r_{t+h} < 0 \mid r_t > Q_{1-p})` for the upper, next to the mean
    forward return and whether the bootstrap interval excludes zero.

    "Excludes zero" is evidence of a non-zero mean, nothing more. It is not a
    claim of profitability: no spread, slippage or cost is applied anywhere in
    this module.
    """
    table = analysis.to_frame()
    if table.is_empty():
        return table
    return table.select(
        "bucket", "direction", "quantile", "horizon", "observations",
        "threshold", "mean_forward_return", "median_forward_return",
        "prob_reversal", "prob_continuation",
        "mean_ci_lower", "mean_ci_upper", "standard_error", "thin_sample",
    ).with_columns(
        pl.when(pl.col("mean_ci_lower").is_null())
        .then(None)
        .otherwise(
            (pl.col("mean_ci_lower") > 0) | (pl.col("mean_ci_upper") < 0)
        )
        .alias("ci_excludes_zero")
    ).sort("direction", "quantile", "horizon")


def _f(value: Any) -> float:
    """Coerce a Polars aggregate to float.

    Polars aggregates are Optional in the type stubs because an empty frame
    yields null. Callers here guard against that, but an empty group would give
    NaN rather than a crash, which is the right failure mode for a statistic.
    """
    return float("nan") if value is None else float(value)


def _i(value: Any) -> int:
    """Integer counterpart of :func:`_f`; zero when the aggregate is null."""
    return 0 if value is None else int(value)
