"""Return mathematics.

Two ideas are kept strictly apart here, because conflating them is how
look-ahead bias gets into research:

**Backward returns** (:func:`simple_returns`, :func:`log_returns`,
:func:`multi_period_returns`) are what is *known* at time ``t``. The value at
``t`` uses only bars at or before ``t``. These are safe to use as features.

**Forward returns** (:func:`forward_returns`) are what *happens after* ``t``.
They are outcomes, never features. Every function that produces them names
them with a ``fwd_`` prefix and says so in its docstring, so a forward column
cannot be mistaken for a historical one at a glance.

Session gaps
------------
A return spanning a weekend or the daily break is not a market move; it is the
market being shut. With ``returns.drop_session_gap_returns`` enabled, those
returns become null rather than being silently treated as ordinary observations
- which matters most for exactly the extreme-move analysis this project cares
about, since gap returns would otherwise dominate the tails.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import polars as pl

from ..data.resampler import parse_timeframe
from .config import ResearchConfig

__all__ = [
    "PRICE_COLUMN_FOR_SOURCE",
    "ReturnSeries",
    "cumulative_returns",
    "forward_returns",
    "log_returns",
    "mark_session_gaps",
    "multi_period_returns",
    "prepare_returns",
    "resolve_price_column",
    "simple_returns",
]

#: Bar column supplying each configured research price source.
PRICE_COLUMN_FOR_SOURCE: dict[str, dict[str, str]] = {
    "mid": {"close": "close", "open": "open", "high": "high", "low": "low"},
    "bid": {"close": "last_bid", "open": "first_bid"},
    "ask": {"close": "last_ask", "open": "first_ask"},
}

ReturnType = Literal["log", "simple"]


def resolve_price_column(price_source: str, price_column: str) -> str:
    """Map ``(price_source, price_column)`` onto an actual bar column.

    Bars carry OHLC built from mid, plus the bid/ask at each bar's edges. Asking
    for ``bid``/``close`` therefore means ``last_bid``, not ``close``.
    """
    try:
        mapping = PRICE_COLUMN_FOR_SOURCE[price_source]
    except KeyError as exc:
        raise KeyError(
            f"Unknown price_source {price_source!r}; expected one of "
            f"{sorted(PRICE_COLUMN_FOR_SOURCE)}"
        ) from exc
    if price_column not in mapping:
        raise KeyError(
            f"price_source={price_source!r} has no {price_column!r} column; "
            f"available: {sorted(mapping)}"
        )
    return mapping[price_column]


# ---------------------------------------------------------------------------
# Backward-looking returns
# ---------------------------------------------------------------------------
def simple_returns(prices: pl.Expr, periods: int = 1) -> pl.Expr:
    r"""Simple return over *periods* bars: :math:`(P_t - P_{t-n}) / P_{t-n}`.

    Backward-looking: the value at ``t`` needs nothing after ``t``. The first
    *periods* values are null.
    """
    if periods < 1:
        raise ValueError(f"periods must be >= 1, got {periods}")
    previous = prices.shift(periods)
    return (prices - previous) / previous


def log_returns(prices: pl.Expr, periods: int = 1) -> pl.Expr:
    r"""Log return over *periods* bars: :math:`\ln(P_t / P_{t-n})`.

    Backward-looking, like :func:`simple_returns`. Log returns are the default
    for analysis because they add across time, which keeps multi-period and
    aggregation arithmetic consistent.
    """
    if periods < 1:
        raise ValueError(f"periods must be >= 1, got {periods}")
    return (prices / prices.shift(periods)).log()


def cumulative_returns(
    returns: pl.Expr, *, return_type: ReturnType = "log"
) -> pl.Expr:
    """Cumulative return of a return series, as a growth factor minus one.

    Log returns cumulate by summing then exponentiating; simple returns
    cumulate multiplicatively. Both are expressed as simple total return so the
    two are directly comparable.
    """
    if return_type == "log":
        return returns.fill_null(0.0).cum_sum().exp() - 1.0
    if return_type == "simple":
        return (returns.fill_null(0.0) + 1.0).cum_prod() - 1.0
    raise ValueError(f"return_type must be 'log' or 'simple', got {return_type!r}")


def multi_period_returns(
    prices: pl.Expr,
    horizons: tuple[int, ...] | list[int],
    *,
    return_type: ReturnType = "log",
    prefix: str = "ret",
) -> list[pl.Expr]:
    r"""Backward multi-period returns :math:`r_{t-n \to t}` for each horizon.

    Named ``{prefix}_{type}_{n}``. These look *back* n bars from ``t``, so they
    are legitimate features. For the forward-looking counterpart, which is an
    outcome rather than a feature, see :func:`forward_returns`.
    """
    fn = log_returns if return_type == "log" else simple_returns
    return [
        fn(prices, periods=int(n)).alias(f"{prefix}_{return_type}_{int(n)}")
        for n in horizons
    ]


# ---------------------------------------------------------------------------
# Forward-looking returns - OUTCOMES ONLY
# ---------------------------------------------------------------------------
def forward_returns(
    prices: pl.Expr,
    horizons: tuple[int, ...] | list[int],
    *,
    return_type: ReturnType = "log",
    prefix: str = "fwd",
) -> list[pl.Expr]:
    r"""Forward returns :math:`r_{t \to t+h}`, as **outcomes, never features**.

    The value at ``t`` describes what happened *after* ``t``, so using one in a
    feature, a rolling statistic or a filter would be look-ahead bias. They
    exist for conditional and reversal analysis, where the question is
    explicitly "given what was known at ``t``, what followed?".

    The ``fwd_`` prefix is deliberate: it makes a forward column obvious in any
    dataframe, and :func:`assert_no_forward_columns` can enforce their absence
    where features are built.
    """
    exprs: list[pl.Expr] = []
    for horizon in horizons:
        h = int(horizon)
        if h < 1:
            raise ValueError(f"forward horizon must be >= 1, got {horizon}")
        ahead = prices.shift(-h)
        expr = (ahead / prices).log() if return_type == "log" else (ahead - prices) / prices
        exprs.append(expr.alias(f"{prefix}_{return_type}_{h}"))
    return exprs


def assert_no_forward_columns(frame: pl.DataFrame, *, context: str = "feature set") -> None:
    """Raise if *frame* carries forward-looking columns.

    Used at the boundaries where features are assembled, so a leak fails loudly
    instead of quietly improving a result.
    """
    leaked = [c for c in frame.columns if c.startswith("fwd_")]
    if leaked:
        raise ValueError(
            f"{context} contains forward-looking column(s) {leaked}. Forward returns "
            "are outcomes and must never enter a feature or historical statistic."
        )


# ---------------------------------------------------------------------------
# Session gaps
# ---------------------------------------------------------------------------
def mark_session_gaps(
    timestamps: pl.Expr, timeframe: str, *, gap_bar_multiple: float = 1.5
) -> pl.Expr:
    """True where the bar at ``t`` does not directly follow the bar at ``t-1``.

    A return computed across such a boundary spans a weekend, a holiday or the
    daily settlement break, so it measures a closure rather than a market move.
    """
    step = parse_timeframe(timeframe)
    threshold_us = int(step.total_seconds() * 1_000_000 * gap_bar_multiple)
    delta = (timestamps - timestamps.shift(1)).dt.total_microseconds()
    return (delta > threshold_us).fill_null(True)  # noqa: FBT003


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ReturnSeries:
    """A bar frame with return columns attached, plus what was done to it."""

    frame: pl.DataFrame
    price_column: str
    return_column: str
    return_type: str
    timeframe: str
    rows_input: int
    rows_with_return: int
    session_gap_returns_dropped: int
    time_basis: str

    def to_dict(self) -> dict[str, object]:
        return {
            "price_column": self.price_column,
            "return_column": self.return_column,
            "return_type": self.return_type,
            "timeframe": self.timeframe,
            "time_basis": self.time_basis,
            "rows_input": self.rows_input,
            "rows_with_return": self.rows_with_return,
            "session_gap_returns_dropped": self.session_gap_returns_dropped,
        }

    @property
    def returns(self) -> pl.Series:
        """The primary return series with nulls removed."""
        return self.frame[self.return_column].drop_nulls()


def prepare_returns(
    bars: pl.DataFrame,
    timeframe: str,
    config: ResearchConfig,
    *,
    time_basis: str | None = None,
) -> ReturnSeries:
    """Attach the configured return columns to a sorted bar frame.

    Produces every configured return type, the configured backward multi-period
    horizons, and ``abs``/``squared`` transforms of the primary series. No
    forward column is produced here - that is :func:`forward_returns`, called
    explicitly by the conditional analysis.
    """
    if bars.is_empty():
        raise ValueError(f"No bars supplied for timeframe {timeframe!r}")

    basis = time_basis or config.intraday.time_basis
    if basis not in bars.columns:
        basis = "timestamp"
    if basis not in bars.columns:
        raise KeyError(f"Bar frame has no timestamp column; columns: {bars.columns}")

    price_col = resolve_price_column(
        config.returns.price_source, config.returns.price_column
    )
    if price_col not in bars.columns:
        raise KeyError(
            f"Bars for {timeframe} have no {price_col!r} column "
            f"(price_source={config.returns.price_source}); available: {bars.columns}"
        )

    frame = bars.sort(basis)
    rows_input = frame.height
    price = pl.col(price_col)

    exprs: list[pl.Expr] = []
    for return_type in config.returns.types:
        fn = log_returns if return_type == "log" else simple_returns
        exprs.append(fn(price).alias(f"ret_{return_type}"))
    primary_type = cast(ReturnType, config.returns.primary)
    exprs += multi_period_returns(
        price, config.returns.multi_period_horizons, return_type=primary_type
    )
    exprs.append(
        mark_session_gaps(
            pl.col(basis), timeframe,
            gap_bar_multiple=config.returns.session_gap_bar_multiple,
        ).alias("is_session_gap")
    )
    frame = frame.with_columns(exprs)

    primary = f"ret_{config.returns.primary}"
    dropped = 0
    if config.returns.drop_session_gap_returns:
        return_cols = [c for c in frame.columns if c.startswith("ret_")]
        before = frame.select(pl.col(primary).is_not_null().sum()).item()
        frame = frame.with_columns(
            [
                pl.when(pl.col("is_session_gap")).then(None).otherwise(pl.col(c)).alias(c)
                for c in return_cols
            ]
        )
        after = frame.select(pl.col(primary).is_not_null().sum()).item()
        dropped = int(before - after)

    frame = frame.with_columns(
        pl.col(primary).abs().alias("abs_return"),
        (pl.col(primary) ** 2).alias("squared_return"),
    )

    return ReturnSeries(
        frame=frame,
        price_column=price_col,
        return_column=primary,
        return_type=config.returns.primary,
        timeframe=timeframe,
        rows_input=rows_input,
        rows_with_return=int(frame.select(pl.col(primary).is_not_null().sum()).item()),
        session_gap_returns_dropped=dropped,
        time_basis=basis,
    )
