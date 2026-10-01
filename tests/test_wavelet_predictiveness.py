"""The incremental-information machinery: folds, OLS by Gram matrix, logistic, AUC."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm

from xauusd_quant.research.wavelet_predictiveness import (
    _design,
    _select,
    chronological_folds,
    fit_linear_models,
    fit_logistic,
    roc_auc,
)


def test_folds_are_chronological_and_embargoed():
    stamps = pl.Series("t", np.datetime64("2009-01-01", "us")
                       + np.arange(0, 20 * 365, 2) * np.timedelta64(1, "D"))
    rows = np.arange(stamps.len()) * 10                      # bar indices of a sample
    folds = chronological_folds(stamps, rows, ((2015, 2018), (2019, 2022)), embargo=25)
    assert [f[0] for f in folds] == ["2015-2018", "2019-2022"]
    for _, train, test in folds:
        assert rows[train].max() < rows[test].min() - 25
        years = stamps.dt.year().to_numpy()
        assert years[train].max() <= years[test].min()


def test_a_block_with_real_information_raises_out_of_sample_r2_and_noise_does_not():
    rng = np.random.default_rng(0)
    n = 20_000
    a = rng.normal(size=n)
    useful = rng.normal(size=n)
    noise = rng.normal(size=n)
    y = 0.5 * a + 0.3 * useful + rng.normal(size=n)
    blocks = {"statistical": {"a": a}, "ou": {}, "fourier": {"noise": noise},
              "wavelet": {"useful": useful}}
    idx = np.arange(n)
    train, test = idx[:15_000], idx[15_000:]
    design = _design(blocks, train)
    subsets = {"A": _select(design, ("statistical",)),
               "C": _select(design, ("statistical", "ou", "fourier")),
               "D": _select(design, ("statistical", "ou", "fourier", "wavelet"))}
    r2 = {r["model"]: r["oos_r2"] for r in fit_linear_models(
        design, {"y": y}, train, test, ridge=1.0, subsets=subsets)}
    assert r2["D"] - r2["C"] > 0.05
    assert abs(r2["C"] - r2["A"]) < 0.005


def test_missing_values_get_an_indicator_and_the_training_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(size=1000)
    x[::10] = np.nan
    design = _design({"statistical": {"x": x}, "ou": {}, "fourier": {}, "wavelet": {}},
                     np.arange(800))
    assert design.names == ["intercept", "x", "x__missing"]
    assert np.isfinite(design.matrix).all()


def test_logistic_matches_statsmodels_without_penalty():
    rng = np.random.default_rng(2)
    x = np.column_stack([np.ones(3000), rng.normal(size=(3000, 2))])
    p = 1 / (1 + np.exp(-(0.3 + x[:, 1] - 0.5 * x[:, 2])))
    y = (rng.random(3000) < p).astype(float)
    ours = fit_logistic(x, y, ridge=0.0)
    reference = sm.Logit(y, x).fit(disp=0).params
    assert ours == pytest.approx(reference, abs=1e-6)


def test_auc_of_known_orderings():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5
    assert roc_auc(np.zeros(4), np.arange(4.0)) is None


def test_trailing_mean_is_causal_and_skips_missing_values():
    from xauusd_quant.research.wavelet_predictiveness import trailing_mean

    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 100.0])
    out = trailing_mean(x, 3)
    assert np.isnan(out[:2]).all()
    assert out[2] == pytest.approx(1.5) and out[3] == pytest.approx(3.0)
    assert out[4] == pytest.approx(4.5)
    assert np.array_equal(trailing_mean(x[:5], 3), out[:5], equal_nan=True), "no look-ahead"


def test_baseline_extensions_widen_a_and_b_causally_and_leave_the_input_alone():
    """The post-hoc WAVE-P-001 baselines: longer-horizon volatility and the hour of day."""
    from xauusd_quant.research.spectral_nulls import SourceData
    from xauusd_quant.research.wavelet_predictiveness import (
        BASELINE_EXTENSIONS,
        extend_baseline,
    )

    rng = np.random.default_rng(5)
    n = 3000
    stamps = pl.Series("timestamp", np.datetime64("2020-01-01T00:00", "us")
                       + np.arange(n) * np.timedelta64(1, "h"))
    columns = {"log_return": rng.normal(0, 1e-3, n), "ou_innovation": rng.normal(size=n)}

    def source(length: int) -> SourceData:
        return SourceData(name="real", timestamps=stamps[:length],
                          columns={k: v[:length] for k, v in columns.items()},
                          missing_slots=np.zeros(length), bar_seconds=3600.0)

    blocks = {"statistical": {"x": rng.normal(size=1000)}, "ou": {"y": rng.normal(size=1000)},
              "fourier": {}, "wavelet": {"w": rng.normal(size=1000)}}
    rows = np.arange(2000, 3000)
    rich = extend_baseline(blocks, source(n), rows, volatility_windows=(64, 256), hours=True)
    assert set(blocks["statistical"]) == {"x"} and set(blocks["ou"]) == {"y"}, "input unchanged"
    assert set(rich["statistical"]) == {"x", "log_realized_vol_64", "log_realized_vol_256",
                                        *(f"hour_{h:02d}" for h in range(1, 24))}
    assert set(rich["ou"]) == {"y", "log_mean_abs_innovation_64", "log_mean_abs_innovation_256"}
    assert rich["wavelet"] == blocks["wavelet"] and rich["fourier"] == {}

    hour = stamps.gather(pl.Series(rows)).dt.hour().to_numpy()
    dummies = np.sum([rich["statistical"][f"hour_{h:02d}"] for h in range(1, 24)], axis=0)
    assert np.array_equal(dummies, (hour != 0).astype(float)), "one-hot, hour 0 the reference"
    r = rows[10]
    expected = 0.5 * np.log(np.mean(columns["log_return"][r - 63:r + 1] ** 2))
    assert rich["statistical"]["log_realized_vol_64"][10] == pytest.approx(expected)

    early = rows[rows < 2500]
    cut = extend_baseline({k: {} for k in blocks}, source(2500), early,
                          volatility_windows=(64, 256), hours=True)
    for block, name in (("statistical", "log_realized_vol_256"),
                        ("ou", "log_mean_abs_innovation_256")):
        assert np.array_equal(cut[block][name], rich[block][name][:early.size]), "no look-ahead"
    assert BASELINE_EXTENSIONS["original"] == {"volatility_windows": (), "hours": False}
