"""Synthetic bars for the Prompt #8 tests (nothing here reads the real dataset).

A heavy-tailed random walk with persistent log volatility, OHLC bars with
wicks, a spread and a tick count that follow the volatility, hourly on a
five-day week (weekends are gaps) - enough structure for every family.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from functools import cache

import numpy as np
import polars as pl

from xauusd_quant.features.families import BarSeries


def hourly_stamps(n: int, start: datetime = datetime(2019, 1, 7, 1)) -> pl.Series:
    """Hourly bar opens 01:00-23:00 Monday-Friday (broker clock), weekends skipped."""
    out = []
    t = start
    while len(out) < n:
        if t.weekday() < 5 and t.hour >= 1:
            out.append(t)
        t += timedelta(hours=1)
    return pl.Series("timestamp", out, dtype=pl.Datetime("us"))


@cache
def _paths(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    lv = np.zeros(n)
    shocks = rng.normal(0.0, 0.15, n)
    for i in range(1, n):
        lv[i] = 0.985 * lv[i - 1] + shocks[i]
    sigma = 0.002 * np.exp(lv)
    r = sigma * rng.standard_t(5, n) / np.sqrt(5 / 3)
    r[0] = 0.0
    close = 1500.0 * np.exp(np.cumsum(r))
    open_ = np.concatenate(([close[0]], close[:-1])) * np.exp(rng.normal(0, 0.1, n) * sigma)
    up = np.abs(rng.normal(0, 0.6, n)) * sigma
    dn = np.abs(rng.normal(0, 0.6, n)) * sigma
    high = np.maximum(open_, close) * np.exp(up)
    low = np.minimum(open_, close) * np.exp(-dn)
    spread = 0.25 + 40.0 * sigma + 0.05 * rng.random(n)
    ticks = rng.poisson(200.0 * (sigma / 0.002)).astype(np.float64)
    return {"open": open_, "high": high, "low": low, "close": close, "spread": spread,
            "ticks": ticks}


def synthetic_bars(n: int = 4000, *, seed: int = 0, cut: int | None = None) -> BarSeries:
    """The first *cut* (default *n*) bars of one fixed synthetic path of length *n*."""
    p = _paths(n, seed)
    m = n if cut is None else cut
    return BarSeries(name="synthetic", timeframe="1h", timestamps=hourly_stamps(n).head(m),
                     close=p["close"][:m].copy(), bar_seconds=3600.0, open=p["open"][:m].copy(),
                     high=p["high"][:m].copy(), low=p["low"][:m].copy(),
                     median_spread=p["spread"][:m].copy(), tick_count=p["ticks"][:m].copy())


def with_wild_future(bs: BarSeries, keep: int, *, seed: int = 99) -> BarSeries:
    """The first *keep* bars followed by absurd bars (Cauchy jumps, huge spreads)."""
    rng = np.random.default_rng(seed)
    n = bs.size
    extra = n - keep
    r = rng.standard_cauchy(extra) * 0.05
    close = np.concatenate((bs.close[:keep], bs.close[keep - 1] * np.exp(np.cumsum(r))))
    open_ = np.concatenate(([close[0]], close[:-1]))
    open_[:keep] = bs.open[:keep]
    high = np.maximum(open_, close) * np.exp(np.abs(rng.standard_cauchy(n)) * 0.01)
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.standard_cauchy(n)) * 0.01)
    high[:keep], low[:keep] = bs.high[:keep], bs.low[:keep]
    spread = np.concatenate((bs.median_spread[:keep], 50.0 * (1 + rng.random(extra))))
    ticks = np.concatenate((bs.tick_count[:keep], rng.integers(0, 100_000, extra)
                            .astype(np.float64)))
    return BarSeries(name="wild", timeframe=bs.timeframe, timestamps=bs.timestamps,
                     close=close, bar_seconds=bs.bar_seconds, open=open_, high=high, low=low,
                     median_spread=spread, tick_count=ticks)
