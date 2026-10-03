"""Aligned known processes, hand recursion and explicit econometric failures."""

from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import statsmodels.api as sm

from econometric_synth import at, grid, returns_process
from xauusd_quant.econometrics.data import GridData
from xauusd_quant.econometrics.diagnostics import segmented_hac, series_diagnostics
from xauusd_quant.econometrics.regression import fit_regression
from xauusd_quant.econometrics.volatility import GarchFit, VarianceState, fit_garch, variance_path
from xauusd_quant.research.residual_stationarity import random_walk_control


def test_targets_use_contiguous_grid_and_actual_information_intervals() -> None:
    data = grid(250)
    sample = data.samples(at(20), at(100), 3, 300000)[0]
    i = sample.point.index
    expected = np.diff(data.logs[i : i + 4])
    assert sample.outcome.log_return == pytest.approx(expected.sum())
    assert sample.outcome.realized_variance == pytest.approx(np.square(expected).sum())
    assert sample.outcome.end_utc - sample.outcome.start_utc == timedelta(minutes=15)
    assert all(s.outcome.matured_utc < at(100) for s in data.samples(at(20), at(100), 3, 300000))
    altered = [t if i < 50 else t + timedelta(days=1) for i, t in enumerate(data.opens)]
    gap = GridData(altered, list(np.exp(data.logs)), 300)
    assert not any(50 <= p.index < 62 for p in gap.points(at(0), at(600)))
    assert data.opens[49].isoformat() not in gap.outcomes(3, 0, at(600))


def test_fixed_arx_recovers_ar_structure_with_predictive_hac_uncertainty() -> None:
    data = grid(2400)
    fit = fit_regression(data.samples(at(0), at(2000), 1, 0), at(2000), "ARX")
    assert fit.diagnostics["coefficients_raw_units"][0] == pytest.approx(0.35, abs=0.1)
    assert len(fit.diagnostics["hac_standard_errors"]) == 4
    assert all(np.isfinite(fit.diagnostics["hac_standard_errors"]))
    point = data.points(at(2000), at(2100))[0]
    assert np.isfinite(fit.predict(point, point.available_utc))
    with pytest.raises(ValueError, match="unavailable"):
        fit.predict(point, point.available_utc - timedelta(seconds=1))


def test_fit_rejects_crossing_labels_and_mixed_horizons() -> None:
    data = grid(500)
    samples = data.samples(at(0), at(400), 3, 0)
    leaking = replace(samples[-1], outcome=replace(samples[-1].outcome, matured_utc=at(400)))
    with pytest.raises(ValueError, match="future or crossing"):
        fit_regression([*samples[:-1], leaking], at(400), "ARX")
    mixed = replace(
        samples[-1],
        outcome=replace(
            samples[-1].outcome,
            horizon_bars=1,
            end_utc=samples[-1].outcome.start_utc + timedelta(minutes=5),
        ),
    )
    with pytest.raises(ValueError, match="mixed target"):
        fit_regression([*samples[:-1], mixed], at(400), "ARX")


def test_har_is_nonnegative_variance_levels_with_declared_intraday_windows() -> None:
    data = grid(800)
    fit = fit_regression(data.samples(at(0), at(600), 3, 0), at(600), "HAR")
    assert min(fit.coefficients) >= 0 and fit.intercept >= 0
    assert fit.horizon_bars == 3 and fit.names == ("rv_1", "rv_3", "rv_12")
    assert fit.predict(data.points(at(600), at(650))[0], at(610)) > 0
    assert "unknown" in fit.diagnostics["coefficient_uncertainty"]
    invalid = replace(fit, coefficients=(0.0, 0.0, 0.0), intercept=0.0)
    with pytest.raises(ValueError, match="no fallback/clipping"):
        invalid.predict(data.points(at(600), at(650))[0], at(610))


def test_white_noise_return_coefficients_are_not_forced_to_show_skill() -> None:
    rng = np.random.default_rng(135014)
    prices = 2000 * np.exp(np.cumsum(rng.normal(0, 0.0004, 3000)))
    data = GridData([at(i) for i in range(len(prices))], list(prices), 300)
    fitted = fit_regression(data.samples(at(0), at(2800), 1, 0), at(2800), "ARX")
    assert max(abs(v) for v in fitted.diagnostics["coefficients_raw_units"][:2]) < 0.08


def test_gap_aware_hac_matches_installed_statsmodels_on_regular_grid() -> None:
    rng = np.random.default_rng(18)
    x = np.column_stack((np.ones(200), rng.normal(size=200)))
    y = x @ [0.1, 0.3] + rng.normal(size=200)
    ordinary = sm.OLS(y, x).fit()
    expected = ordinary.get_robustcov_results(
        cov_type="HAC", maxlags=3, use_correction=True
    ).cov_params()
    actual = segmented_hac(x, ordinary.resid, [at(i) for i in range(200)], 300, 3)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    # A wide gap must not create a lag-1 score pair across the closure.
    separated = [at(i) if i < 100 else at(i + 1000) for i in range(200)]
    gap_cov = segmented_hac(x, ordinary.resid, separated, 300, 3)
    assert not np.allclose(gap_cov, actual, atol=1e-12, rtol=1e-12)


def test_garch_variance_recursion_and_multistep_second_moments_by_hand() -> None:
    parameters = GarchFit(0.01, 0.1, 0.8, 0.1, {})
    state = VarianceState("GARCH", 0.1, 3, garch=parameters)
    state.observe(0.2, at(1), contiguous=True)
    # Next h=.01+.1*.04+.8*.1=.094; later expectation .01+.9*h.
    expected = 0.094 + 0.0946 + 0.09514
    assert state.forecast() == pytest.approx(expected)
    np.testing.assert_allclose(
        variance_path(np.asarray([0.2, 0.1]), 0.01, 0.1, 0.8, 0.1), [0.1, 0.094]
    )
    state.observe(None, at(100), contiguous=False)
    assert state.next_variance == 0.1 and not state.history


def test_garch_fit_has_positive_finite_forecasts_on_conditional_variance_fixture() -> None:
    returns = returns_process(1800)
    fit = fit_garch([returns])
    assert fit.omega > 0 and fit.alpha + fit.beta < 0.995
    state = VarianceState("GARCH", fit.initialization_variance, garch=fit)
    for i, value in enumerate(returns):
        state.observe(float(value), at(i), contiguous=True)
        assert np.isfinite(state.forecast()) and state.forecast() > 0


def test_garch_nonconvergence_and_invalid_history_do_not_become_benchmarks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="degenerate"):
        fit_garch([np.zeros(200)])
    with pytest.raises(ValueError, match="insufficient"):
        fit_garch([np.ones(20)])
    monkeypatch.setattr(
        "xauusd_quant.econometrics.volatility.minimize",
        lambda *a, **k: SimpleNamespace(success=False, message="forced nonconvergence"),
    )
    with pytest.raises(ValueError, match="nonconvergence.*no fallback"):
        fit_garch([returns_process(200)])


def test_existing_stationarity_and_detrended_walk_controls_are_diagnostics_only() -> None:
    control = random_walk_control(640, 32, seed=135013)
    report = series_diagnostics(control, label="detrended_random_walk")
    assert "unit root" in report["adf"]["null_hypothesis"]
    assert "stationary" in report["kpss"]["null_hypothesis"]
    assert report["status"] == "diagnostics_only"
    assert "not economic reversion" in report["interpretation"]
    assert series_diagnostics(np.zeros(100), label="degenerate")["status"] == "untested"


@pytest.mark.parametrize("prices", [[1, -1], [1, float("nan")]])
def test_invalid_midpoint_measurements_are_explicit(prices: list[float]) -> None:
    with pytest.raises(ValueError, match="invalid/nonpositive"):
        GridData([at(0), at(1)], prices, 300)
