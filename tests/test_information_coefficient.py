"""Information coefficients from monthly moments (Prompt #8, Steps 21-27, 32-34, 62).

The pooled IC equals the pairwise-complete correlation exactly; the matrix
formulation equals the column loop; every temporal slice is a sum of months;
the rolling IC at a month never reads that month; the batch-means standard
error is near the i.i.d. one for independent data and grows (Newey-West) for
persistent pairs.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import polars as pl
import pytest
from scipy.stats import spearmanr

from xauusd_quant.alpha.information_coefficient import (
    block_moments,
    correlation_from_sums,
    ic_with_errors,
    month_index,
    month_moments,
    rank_scores,
)
from xauusd_quant.alpha.stability import (
    alpha_health,
    consistency,
    era_ics,
    grouped_ics,
    rolling_ics,
    subsample_ics,
)


def _stamps(n: int, per_month: int = 500) -> pl.Series:
    months = np.arange(n) // per_month
    return pl.Series("ts", [datetime(2010 + int(m) // 12, int(m) % 12 + 1, 1) for m in months],
                     dtype=pl.Datetime("us"))


def _pairs(n: int, rho: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = rho * x + np.sqrt(1 - rho ** 2) * rng.normal(size=n)
    return x, y


def test_pooled_ic_is_the_pairwise_complete_correlation() -> None:
    x, y = _pairs(12_000, 0.3, 1)
    x[::13] = np.nan
    y[::29] = np.nan
    idx = month_index(_stamps(x.size))
    mom = month_moments([x], [y], idx.starts)[0, 0]
    ok = np.isfinite(x) & np.isfinite(y)
    assert correlation_from_sums(mom.sum(axis=0)) == pytest.approx(
        np.corrcoef(x[ok], y[ok])[0, 1], abs=1e-12)
    assert mom[:, 0].sum() == ok.sum()


def test_matrix_moments_equal_the_column_loop() -> None:
    rng = np.random.default_rng(2)
    n = 6000
    xs = [rng.normal(size=n).astype(np.float32) * 5 - 7 for _ in range(4)]
    ys = [rng.normal(size=n).astype(np.float32) for _ in range(3)]
    xs[1][::7] = np.nan
    ys[2][:200] = np.nan
    starts = month_index(_stamps(n)).starts
    a = month_moments(xs, ys, starts)
    b = np.stack([np.moveaxis(block_moments(x, ys, starts), 0, 1) for x in xs])
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-7)


def test_rank_ic_is_close_to_spearman() -> None:
    x, y = _pairs(8000, 0.2, 3)
    y = np.exp(y)                                           # monotone: Spearman unchanged
    idx = month_index(_stamps(x.size))
    mom = month_moments([rank_scores(x)], [rank_scores(y)], idx.starts)[0, 0]
    assert correlation_from_sums(mom.sum(axis=0)) == pytest.approx(
        spearmanr(x, y).statistic, abs=1e-9)


def test_rank_scores_average_ties_and_keep_missing() -> None:
    u = rank_scores(np.array([3.0, 1.0, np.nan, 1.0, 2.0]))
    np.testing.assert_allclose(u[[0, 1, 3, 4]], [(4 - 0.5) / 4, (1.5 - 0.5) / 4,
                                                 (1.5 - 0.5) / 4, (3 - 0.5) / 4])
    assert np.isnan(u[2])


def test_batch_means_se_is_calibrated_for_independent_data() -> None:
    x, y = _pairs(40_000, 0.1, 4)
    idx = month_index(_stamps(x.size, per_month=400))
    stats = ic_with_errors(np.moveaxis(month_moments([x], [y], idx.starts), 2, 0))
    ratio = float(stats["se"][0, 0] / stats["se_iid"][0, 0])
    assert 0.75 < ratio < 1.3


def test_newey_west_widens_the_error_of_persistent_pairs() -> None:
    rng = np.random.default_rng(5)
    n = 30_000
    e1, e2 = rng.normal(size=n), rng.normal(size=n)
    x, y = np.empty(n), np.empty(n)
    x[0], y[0] = e1[0], e2[0]
    for i in range(1, n):                                   # two independent AR(1), phi 0.999
        x[i] = 0.999 * x[i - 1] + e1[i]
        y[i] = 0.999 * y[i - 1] + e2[i]
    idx = month_index(_stamps(n, per_month=300))
    mom = np.moveaxis(month_moments([x], [y], idx.starts), 2, 0)
    plain = ic_with_errors(mom, hac_lags=0)["se"][0, 0]
    hac = ic_with_errors(mom, hac_lags=2)["se"][0, 0]
    assert hac > 1.2 * plain


def test_slices_are_sums_of_months_and_rolling_ic_never_reads_its_month() -> None:
    x, y = _pairs(24 * 500, 0.2, 6)
    idx = month_index(_stamps(x.size))
    mom = month_moments([x], [y], idx.starts)
    m = np.moveaxis(mom, 2, 0)                              # (M, 1, 1, 6)
    labels, yic, yn = grouped_ics(m, idx.years, min_obs=100)
    for i, year in enumerate(labels):
        sel = np.isin(np.arange(x.size) // 500, np.flatnonzero(idx.years == year))
        assert yic[i, 0, 0] == pytest.approx(np.corrcoef(x[sel], y[sel])[0, 1], abs=1e-12)
    roll = rolling_ics(m, window=6, min_months=3, min_obs=100)
    y2 = y.copy()
    y2[10 * 500:11 * 500] *= -1                             # rewrite month 10 only
    roll2 = rolling_ics(np.moveaxis(month_moments([x], [y2], idx.starts), 2, 0), window=6,
                        min_months=3, min_obs=100)
    assert roll["ic"][10, 0, 0] == roll2["ic"][10, 0, 0]    # month 10's IC ignores month 10
    assert roll["ic"][11, 0, 0] != roll2["ic"][11, 0, 0]
    eras = era_ics(m, 3, min_obs=100)
    subs = subsample_ics(m, idx, min_obs=100)
    assert np.isfinite(eras).all() and set(subs) >= {"odd_years", "first_half", "odd_quarters"}
    health = alpha_health(roll["ic"], roll["z"], ic_with_errors(m)["ic"])
    assert health["rolling_windows"][0, 0] > 0


def test_consistency_counts_years_with_the_pooled_sign() -> None:
    yearly = np.array([0.1, 0.2, -0.05, 0.3, np.nan])[:, None]
    out = consistency(yearly, np.array([0.15]), np.array([0.2]))
    assert out["years"][0] == 4
    assert out["sign_consistency"][0] == pytest.approx(0.75)
    assert out["worst_year_ic"][0] == pytest.approx(-0.05)
    assert out["best_year_ic"][0] == pytest.approx(0.3)


def test_an_undefined_correlation_is_nan_never_a_number() -> None:
    x = np.ones(3000)
    y = np.random.default_rng(7).normal(size=3000)
    idx = month_index(_stamps(3000))
    stats = ic_with_errors(np.moveaxis(month_moments([x], [y], idx.starts), 2, 0))
    assert np.isnan(stats["ic"][0, 0]) and np.isnan(stats["se"][0, 0])
    assert np.isnan(stats["p"][0, 0])
