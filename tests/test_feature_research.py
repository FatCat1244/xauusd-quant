"""The research stages on synthetic data (Prompt #8, Steps 21-59).

A planted feature must pass the studentized max-T circular-shift null and
noise features must (almost always) not; the segmented conditioning moments
must equal the direct per-condition sums; the IC tables must carry every
column the statuses read.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from xauusd_quant.alpha.conditioning import condition_moments, month_codes
from xauusd_quant.alpha.config import load_alpha_config
from xauusd_quant.alpha.information_coefficient import month_index, month_moments, rank_scores
from xauusd_quant.alpha.null_tests import standardized_scores
from xauusd_quant.features.factory_config import load_features_config
from xauusd_quant.research import feature_research as fr
from xauusd_quant.targets.config import load_targets_config


def _ctx(tmp_path: Path) -> fr.ResearchContext:
    cfg = fr.Configs(config=None, regression=None, ou=None, spectral=None, wavelet=None,
                     research=None, features=load_features_config(),
                     targets=load_targets_config(), alpha=load_alpha_config(),
                     regime_features_path=tmp_path)
    return fr.ResearchContext(timeframe="1h", cfg=cfg, out_dir=tmp_path)


def _stamps(n: int) -> pl.Series:
    start = datetime(2015, 1, 5)
    return pl.Series("timestamp", [start + timedelta(hours=i) for i in range(n)],
                     dtype=pl.Datetime("us"))


def _targets(n: int, seed: int) -> tuple[list[str], list[np.ndarray], np.ndarray]:
    """Two target kinds with a few horizons each, from one persistent driver."""
    rng = np.random.default_rng(seed)
    driver = np.convolve(rng.normal(size=n + 50), np.ones(50) / np.sqrt(50), mode="valid")[:n]
    names, cols = [], []
    for h in (1, 5, 20):
        names.append(f"target_return_{h}")
        cols.append(np.convolve(rng.normal(size=n + h), np.ones(h), mode="valid")[:n])
        names.append(f"target_realized_vol_{h}")
        cols.append(np.abs(driver) + 0.5 * rng.normal(size=n))
    return names, cols, driver


@pytest.mark.slow
def test_max_t_null_passes_a_planted_feature_and_rarely_noise(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    n = 30_000
    names, cols, driver = _targets(n, 1)
    rng = np.random.default_rng(2)
    feats = {"planted": (np.abs(driver) + 0.8 * rng.normal(size=n)).astype(np.float32)}
    for i in range(30):
        ar = np.zeros(n)
        e = rng.normal(size=n)
        for t in range(1, n):
            ar[t] = 0.99 * ar[t - 1] + e[t]
        feats[f"noise_{i:02d}"] = ar.astype(np.float32)
    zy = standardized_scores([rank_scores(c) for c in cols])
    frames, kinds, maxima = fr.null_rows(ctx, list(feats), lambda f: feats[f], zy, names,
                                         seed=7)
    kind = pl.concat(kinds)
    planted = kind.filter(pl.col("feature") == "planted")
    assert planted.filter(pl.col("kind") == "volatility")["beyond_kind_max_null"][0]
    noise = kind.filter(pl.col("feature").str.starts_with("noise"))
    per_feature = noise.group_by("feature").agg(pl.col("beyond_kind_max_null").any())
    assert per_feature["beyond_kind_max_null"].sum() <= 3          # ~1 % per feature expected
    tests = pl.concat(frames)
    assert set(tests.columns) >= {"rank_ic0", "shift_null_q", "beyond_shift_null",
                                  "wrong_plus_60d", "wrong_minus_5d"}
    assert maxima.shape == (ctx.cfg.alpha.nulls.circular_shifts,)


def test_segmented_condition_moments_equal_the_direct_sums() -> None:
    n = 20_000
    rng = np.random.default_rng(3)
    stamps = _stamps(n)
    months = month_index(stamps)
    mcode = month_codes(months.starts, n)
    x = rng.normal(size=n)
    x[::13] = np.nan
    ys = [rng.normal(size=n) for _ in range(3)]
    codes = rng.integers(-1, 4, size=n)                          # -1 = left out
    labels = ["a", "b", "c", "d"]
    plan = fr._partition(codes, labels, mcode, months.size)
    seg = month_moments([x[plan.order]], [y[plan.order] for y in ys], plan.starts)
    seg = seg.reshape(1, 3, 4, months.size, 6)[0]                # (T, C, M, 6)
    direct = condition_moments(x, ys, mcode, codes, months.size, 4)   # (M, C, T, 6)
    np.testing.assert_allclose(np.transpose(seg, (2, 1, 0, 3))[..., 0],
                               direct[..., 0])                   # the pair counts
    from xauusd_quant.alpha.information_coefficient import correlation_from_sums

    a = correlation_from_sums(seg.sum(axis=2))                   # (T, C)
    b = correlation_from_sums(direct.sum(axis=0)).T              # (T, C)
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_ic_tables_carry_what_the_statuses_read(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    n = 24 * 700
    names, cols, driver = _targets(n, 4)
    rng = np.random.default_rng(5)
    base = np.empty((n, len(names)), dtype=np.float32, order="F")
    for j, c in enumerate(cols):
        base[:, j] = c
    ranks = np.empty_like(base)
    for j in range(base.shape[1]):
        ranks[:, j] = rank_scores(base[:, j])
    tb = fr.TargetBlocks(names=names, blocks=[("all", list(range(len(names))))],
                         horizons={t: int(t.rsplit("_", 1)[1]) for t in names}, base=base,
                         extra=np.empty((n, 0), dtype=np.float32, order="F"), ranks=ranks)
    feats = {"f": (np.abs(driver) + rng.normal(size=n)).astype(np.float32),
             "g": rng.normal(size=n).astype(np.float32)}
    months = month_index(_stamps(n))
    raw, rnk = fr.compute_moments(list(feats), lambda k: feats[k], tb, months)
    tables = fr.ic_tables(ctx, list(feats), tb, months, raw, rnk)
    ic = tables["ic"]
    assert set(ic.columns) >= {"feature", "target", "horizon", "kind", "method", "ic", "se",
                               "se_iid", "z", "p", "ci_low", "ci_high", "months"}
    cons = tables["consistency"]
    assert set(cons.columns) >= {"sign_consistency", "recent_ic", "era_1_ic", "era_sign_flips",
                                 "odd_years_ic", "first_half_ic", "last_rolling_ic"}
    assert set(tables["rolling"]["method"].unique()) == {"pearson", "spearman"}
    row = ic.filter((pl.col("feature") == "f") & (pl.col("target") == "target_realized_vol_5")
                    & (pl.col("method") == "spearman"))
    assert row["ic"][0] > 0.2 and row["p"][0] < 1e-6
    decay = fr.decay_table(ctx, ic)
    assert {"peak_horizon", "half_decay_horizon", "curve"} <= set(decay.columns)


def test_write_table_turns_nan_into_missing(tmp_path: Path) -> None:
    frame = pl.DataFrame({"a": [1.0, float("nan")], "b": [None, None]})
    fr.write_table(frame, tmp_path / "t")
    back = pl.read_parquet(tmp_path / "t.parquet")
    assert back["a"].null_count() == 1 and back["b"].dtype == pl.Float64
    assert (tmp_path / "t.csv").exists()
