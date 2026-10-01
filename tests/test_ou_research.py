"""The OU research layer: outcome analyses, conditioning, stability, report.

Outcome analyses are checked on hand-built paths whose answers are known, so
a censoring or off-by-one error shows up as a wrong number rather than as a
plausible-looking table. The end-to-end test runs the whole report on
synthetic bars and checks every promised file and summary key exists.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from xauusd_quant.models import load_ou_config
from xauusd_quant.models.ornstein_uhlenbeck import simulate_ou
from xauusd_quant.research.ou_conditional import (
    attach_conditioning,
    trend_conditioning,
    volatility_conditioning,
)
from xauusd_quant.research.ou_decay import (
    extreme_events,
    first_passage_summary,
    forecast_evaluation,
    realized_decay_comparison,
)
from xauusd_quant.research.ou_estimation import ou_parameter_frame
from xauusd_quant.research.ou_stability import (
    era_table,
    half_life_stability,
    recent_versus_history,
    segment_table,
)


@pytest.fixture(scope="module")
def ou_config():
    return load_ou_config()


def handmade(values: list[float], *, mu: float = 0.0, b: float = 0.5,
             stat: float = 1.0) -> pl.DataFrame:
    """A parameter frame with fixed, known OU parameters at every bar."""
    n = len(values)
    theta = -math.log(b)
    return pl.DataFrame({
        "residual": values,
        "ou_mu": [mu] * n,
        "ou_b": [b] * n,
        "ou_theta": [theta] * n,
        "ou_stationary_std": [stat] * n,
        "ou_half_life_bars": [math.log(2) / theta] * n,
        "ou_zscore": [(v - mu) / stat for v in values],
        "ou_valid": [True] * n,
    })


def test_first_passage_times_on_a_known_path(ou_config):
    # Start at 4 (Z=4); then 3, 2.5, 1.9, -0.1 ... : 75% of 4 = 3 at h=1,
    # 50% = 2 at h=3, 25% = 1 at h=4, crossing at h=4.
    path = [4.0, 3.0, 2.5, 1.9, -0.1] + [0.0] * 600
    events = extreme_events(handmade(path), config=ou_config)
    first = events.filter(pl.col("index") == 0).row(0, named=True)
    assert (first["fp_75"], first["fp_50"], first["fp_25"], first["cross"]) == (1, 3, 4, 4)
    assert first["remaining_h1"] == pytest.approx(0.75)
    assert first["toward_h1"] is True


def test_an_adverse_excursion_is_measured_in_stationary_std(ou_config):
    path = [2.5, 3.5, 4.0, 1.0, -0.5] + [0.0] * 600
    events = extreme_events(handmade(path, stat=0.5), config=ou_config)
    first = events.filter(pl.col("index") == 0).row(0, named=True)
    assert first["max_adverse_z"] == pytest.approx((4.0 - 2.5) / 0.5)


def test_a_start_that_never_halves_is_censored_not_dropped(ou_config):
    path = [3.0] * 700
    events = extreme_events(handmade(path), config=ou_config)
    assert events.height > 0
    assert (events["fp_50"] == 0).all() and (events["cross"] == 0).all()
    comparison = realized_decay_comparison(events, source="t", config=ou_config)
    row = comparison.filter(pl.col("subset") == "all starts").row(0, named=True)
    assert row["censored_fraction"] == 1.0
    assert row["median_ratio"] is None, "an unknowable median must not be invented"


def test_starts_without_a_full_horizon_are_excluded(ou_config):
    path = [0.0] * 100 + [5.0] + [0.0] * 50
    events = extreme_events(handmade(path), config=ou_config)
    assert events.is_empty(), "a start 50 bars from the end has no 500-bar future"


def test_episode_starts_keep_one_start_per_excursion(ou_config):
    path = ([0.0] * 5 + [2.5, 2.6, 2.4, 0.0]) * 3 + [0.0] * 600
    events = extreme_events(handmade(path), config=ou_config)
    passage = first_passage_summary(events, source="t", config=ou_config)
    level2 = passage.filter((pl.col("abs_z_level") == 2.0)
                            & (pl.col("target") == "50% of D0"))
    counts = dict(zip(level2["subset"].to_list(), level2["observations"].to_list(), strict=True))
    assert counts == {"all starts": 9, "episode starts": 3}


def test_forecast_evaluation_on_an_exact_ou_beats_persistence(ou_config):
    x = simulate_ou(0.2, 0.0, 1.0, 50_000, seed=1)
    frame = ou_parameter_frame(x, None, ou_window=256, config=ou_config)
    table = forecast_evaluation(frame, source="t", config=ou_config)
    h10 = table.filter((pl.col("horizon") == 10) & (pl.col("subset") == "all")).row(0,
                                                                                   named=True)
    # Closed form for an exact OU: MSE_ou = V(1-b^2h), MSE_persist = 2V(1-b^h).
    b = math.exp(-0.2)
    theory = 1.0 - (1.0 - b ** 20) / (2.0 * (1.0 - b ** 10))
    assert h10["skill_vs_persistence"] == pytest.approx(theory, abs=0.03)
    assert abs(h10["bias"]) < 0.05


def test_realized_half_decay_of_an_exact_ou_is_close_to_its_half_life(ou_config):
    """Reference behaviour: a correct OU halves large deviations in about one HL."""
    x = simulate_ou(-math.log(0.95), 0.0, 1.0, 200_000, seed=2)       # HL 13.5 bars
    frame = ou_parameter_frame(x, None, ou_window=512, config=ou_config)
    events = extreme_events(frame, config=ou_config)
    table = realized_decay_comparison(events, source="t", config=ou_config)
    row = table.filter((pl.col("abs_z_level") == 2.0)
                       & (pl.col("subset") == "all starts")).row(0, named=True)
    assert 0.5 < row["median_ratio"] < 1.3
    assert row["censored_fraction"] < 0.01


def test_a_statistic_undefined_for_a_whole_source_still_stacks(ou_config):
    """Regression: the first full grid died stacking a fully censored source.

    When more than a tenth of a source's starts never halve, every ratio p90 is
    None and its table carries a dtype-Null column. The real residual is stacked
    first; where it was the censored one and a control was not (1h N=256 M=128,
    5m N=512 M=256, ...), a strict diagonal concat refused the control's Float64.
    """
    from xauusd_quant.research.ou_reports import _stack, _typed

    censored = realized_decay_comparison(
        extreme_events(handmade([3.0] * 700), config=ou_config),
        source="residual", config=ou_config)
    x = simulate_ou(-math.log(0.95), 0.0, 1.0, 20_000, seed=2)
    frame = ou_parameter_frame(x, None, ou_window=256, config=ou_config)
    control = realized_decay_comparison(extreme_events(frame, config=ou_config),
                                        source="reference_ou", config=ou_config)
    assert censored.schema["ratio_p90"] == pl.Null, "the precondition of the failure"
    assert control.schema["ratio_p90"] == pl.Float64
    stacked = _stack([censored, pl.DataFrame(), control])      # the report's order
    assert stacked.height == censored.height + control.height
    assert stacked.schema["ratio_p90"] == pl.Float64
    assert stacked.filter(pl.col("source") == "residual")["ratio_p90"].is_null().all()
    assert all(dtype != pl.Null for dtype in _typed(censored).schema.values())


def synthetic_frame(ou_config, n: int = 40_000) -> pl.DataFrame:
    x = simulate_ou(0.05, 0.0, 1e-3, n, seed=4)
    start = datetime(2021, 1, 4)
    stamps = pl.Series("timestamp", [start + timedelta(hours=i) for i in range(n)])
    frame = ou_parameter_frame(x, stamps, ou_window=128, config=ou_config)
    rng = np.random.default_rng(0)
    features = pl.DataFrame({
        "regression_slope": rng.normal(0, 1e-4, n),
        "r_squared": rng.uniform(0, 1, n),
        "trailing_volatility": rng.uniform(1e-4, 5e-4, n),
        "log_return": rng.normal(0, 3e-4, n),
    })
    return attach_conditioning(frame, features, ou_window=128)


def test_conditioning_buckets_are_ordered_and_complete(ou_config):
    frame = synthetic_frame(ou_config)
    vol = volatility_conditioning(frame, source="t", config=ou_config)
    window = vol.filter(pl.col("variable") == "ou_window_volatility")
    assert window["bucket"].to_list() == ["Q1", "Q2", "Q3", "Q4"]
    assert window["upper_edge"][:3].is_sorted()
    trend = trend_conditioning(frame, source="t", config=ou_config)
    labels = trend.filter(pl.col("variable") == "regression_slope")["bucket"].to_list()
    assert labels == ["strongly_negative", "weak_negative", "near_flat", "weak_positive",
                      "strongly_positive"]
    sizes = trend.filter(pl.col("variable") == "regression_slope")["windows"].to_list()
    assert max(sizes) - min(sizes) <= 0.02 * sum(sizes), "quintiles should be near-equal"


def test_segments_eras_and_recent_comparison(ou_config):
    frame = synthetic_frame(ou_config)
    years = segment_table(frame, by="year", config=ou_config)
    assert years["segment"].to_list() == ["2021", "2022", "2023", "2024", "2025"]
    assert years.filter(pl.col("sufficient_data"))["static_ou_valid"].all()
    quarters = segment_table(frame, by="quarter", config=ou_config)
    assert quarters.height >= 18
    eras = era_table(frame, config=ou_config)
    assert eras["segment"].to_list() == ["1_early", "2_middle", "3_recent"]
    spans = [datetime.fromisoformat(e) - datetime.fromisoformat(s)
             for s, e in zip(eras["block_start"], eras["block_end"], strict=True)]
    assert max(spans) - min(spans) < timedelta(hours=2), "blocks must be equal length"
    recent = recent_versus_history(frame)
    assert recent["sufficient_data"] and "half_life_bars_ks_statistic" in recent


def test_half_life_stability_of_a_constant_ou(ou_config):
    x = simulate_ou(0.05, 0.0, 1.0, 50_000, seed=6)
    frame = ou_parameter_frame(x, None, ou_window=256, config=ou_config)
    out = half_life_stability(frame, ou_window=256, source="t", config=ou_config)
    assert out["disjoint_pairs"] > 1000
    assert out["adjacent_median_abs_log_change"] < out["disjoint_median_abs_log_change"]
    assert 0 <= out["state_change_fraction"] < 0.05


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------
def test_the_report_writes_every_promised_output(tmp_path, monkeypatch):
    from xauusd_quant.features import load_regression_config
    from xauusd_quant.features.rolling_regression import rolling_regression_features
    from xauusd_quant.research.config import load_research_config
    from xauusd_quant.research.ou_reports import generate_ou_report, write_ou_comparison
    from xauusd_quant.utils.config import load_config

    monkeypatch.setenv("XAUUSD_OU_RESULTS", str(tmp_path / "ou").replace("\\", "/"))
    ou = load_ou_config()
    regression = load_regression_config()
    research = load_research_config()
    config = load_config()
    rng = np.random.default_rng(3)
    n = 6_000
    prices = 2000.0 * np.exp(np.cumsum(rng.normal(0, 5e-4, n)))
    bars = pl.DataFrame({
        "timestamp": [datetime(2021, 1, 4) + timedelta(minutes=5 * i) for i in range(n)],
        "open": prices, "high": prices, "low": prices, "close": prices,
        "first_bid": prices, "first_ask": prices, "last_bid": prices, "last_ask": prices,
    })
    features, _ = rolling_regression_features(bars, window=64, config=regression,
                                              timeframe="5m")
    report = generate_ou_report(config, regression, research, ou, timeframe="5m",
                                regression_window=64, ou_window=128, features=features,
                                make_plots=True, write_parameters=True)
    out = report.output_dir
    for name in ("summary.json", "parameters.parquet", "parameter_distribution.csv",
                 "half_life.csv", "yearly_stability.csv", "quarterly_stability.csv",
                 "volatility_conditioning.csv", "trend_conditioning.csv", "r2_conditioning.csv",
                 "innovation_diagnostics.csv", "realized_decay_comparison.csv",
                 "forecast_evaluation.csv", "extreme_deviations.csv", "first_passage.csv",
                 "b_classification.csv", "equilibrium.csv", "static_fit.csv",
                 "intraday_conditioning.csv", "era_stability.csv", "zscore_comparison.csv"):
        assert (out / name).exists(), name
    for plot in ("rolling_half_life", "rolling_theta", "rolling_b", "ou_equilibrium",
                 "ou_zscore", "estimated_vs_realized_half_decay", "innovation_acf",
                 "innovation_distribution", "half_life_distribution"):
        assert (out / "plots" / f"{plot}.png").exists(), plot
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    model = summary["model_summary"]
    for key in ("timeframe", "regression_window", "ou_window", "observations",
                "valid_ou_fraction", "median_b", "median_theta", "median_mu", "median_sigma",
                "median_half_life_bars", "half_life_p25", "half_life_p75",
                "near_unit_root_fraction", "median_ou_r_squared", "innovation_lag1_acf",
                "estimated_vs_realized_half_life_correlation"):
        assert key in model, key
    assert model["control_random_walk"]["median_half_life_bars"] is not None
    assert model["reference_ou"] is not None
    assert "dataset" in summary["provenance"]
    assert summary["provenance"]["report_version"] == 2
    for name in ("volatility_conditioning", "trend_conditioning", "r2_conditioning"):
        sources = set(pl.read_csv(out / f"{name}.csv")["source"].to_list())
        assert sources == {"residual", "control_random_walk"}, (name, sources)
    written = write_ou_comparison(config, ou, [report, report])
    assert any(p.name == "window_matrix.csv" for p in written)


def test_resume_reuses_only_a_current_study(world, tmp_path, monkeypatch):
    """--resume must never pass off a study from other data or settings as current."""
    import yaml

    from xauusd_quant.features.store import RegressionFeatureStore
    from xauusd_quant.research.config import load_research_config
    from xauusd_quant.research.ou_reports import generate_ou_report, reusable_ou_report

    config, regression = world
    monkeypatch.setenv("XAUUSD_OU_RESULTS", str(tmp_path / "ou").replace("\\", "/"))
    ou, research = load_ou_config(), load_research_config()
    RegressionFeatureStore(config, regression).build("1m", [32])
    study = {"timeframe": "1m", "regression_window": 32, "ou_window": 64}

    def reuse(settings=ou, **need):
        return reusable_ou_report(config, regression, research, settings, **study, **need)

    assert reuse() is None, "nothing on disk yet"
    fresh = generate_ou_report(config, regression, research, ou, **study,
                               make_plots=False, write_parameters=False)
    again = reuse()
    assert again is not None and again.reused
    assert (again.observations, again.first_timestamp) == (fresh.observations,
                                                           fresh.first_timestamp)
    assert reuse(need_parameters=True) is None, "parameters.parquet was not written"
    assert reuse(need_plots=True) is None, "no plots were drawn"

    raw = yaml.safe_load(ou.config_path.read_text(encoding="utf-8"))
    raw["controls"]["seed"] = int(raw["controls"]["seed"]) + 1
    changed = tmp_path / "ou_changed.yaml"
    changed.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert reuse(load_ou_config(changed)) is None, "other OU settings"

    path = fresh.output_dir / "summary.json"
    original = path.read_text(encoding="utf-8")
    summary = json.loads(original)
    summary["provenance"]["report_version"] = 1
    path.write_text(json.dumps(summary), encoding="utf-8")
    assert reuse() is None, "a study written by an older report version"
    summary = json.loads(original)
    summary["provenance"]["dataset"]["tick_dataset_version"] = "ticks-0000000000000000"
    path.write_text(json.dumps(summary), encoding="utf-8")
    assert reuse() is None, "other tick data"


def test_one_failing_study_does_not_stop_the_grid(monkeypatch, tmp_path):
    from xauusd_quant import cli
    from xauusd_quant.research import ou_reports

    monkeypatch.setenv("XAUUSD_OU_RESULTS", str(tmp_path / "ou").replace("\\", "/"))
    ran: list[tuple[int, int]] = []

    def fake(config, regression, research, ou, *, timeframe, regression_window, ou_window,
             **_):
        if (regression_window, ou_window) == (64, 128):
            raise pl.exceptions.SchemaError("synthetic failure")
        ran.append((regression_window, ou_window))
        return ou_reports.OUReport(timeframe=timeframe, regression_window=regression_window,
                                   ou_window=ou_window, output_dir=tmp_path)

    monkeypatch.setattr(ou_reports, "generate_ou_report", fake)
    code = cli.main(["ou-research", "--timeframe", "1h", "--all-windows", "--no-plots",
                     "--in-process"])
    ou = load_ou_config()
    grid = [(n, m) for n in ou.regression_windows for m in ou.ou_estimation_windows]
    assert code == 1
    assert ran == [pair for pair in grid if pair != (64, 128)], "every other study still ran"


def test_a_study_runs_in_its_own_process_and_comes_back_whole(world, tmp_path, monkeypatch):
    """The default path: one fresh child process per study, the report returned."""
    from xauusd_quant import cli
    from xauusd_quant.features.store import RegressionFeatureStore

    config, regression = world
    monkeypatch.setenv("XAUUSD_OU_RESULTS", str(tmp_path / "ou").replace("\\", "/"))
    RegressionFeatureStore(config, regression).build("1m", [32])
    args = ["--config", str(config.config_path), "ou-research", "--timeframe", "1m",
            "--regression-window", "32", "--ou-window", "64", "--no-plots"]
    assert cli.main(args) == 0
    summary = tmp_path / "ou" / "1m" / "regression_32" / "ou_64" / "summary.json"
    model = json.loads(summary.read_text(encoding="utf-8"))["model_summary"]
    assert model["observations"] > 0 and model["control_random_walk"] is not None
    assert cli.main([*args, "--resume"]) == 0, "and a current study is then reused"


def test_the_conditioned_control_is_the_same_walk_with_the_same_definitions():
    """random_walk_features must add conditioning variables without changing the control."""
    from datetime import datetime, timedelta

    from xauusd_quant.features import load_regression_config
    from xauusd_quant.features.rolling_regression import rolling_regression_features
    from xauusd_quant.models.ornstein_uhlenbeck import simulate_random_walk
    from xauusd_quant.research.ou_estimation import _random_walk, random_walk_features

    regression = load_regression_config()
    window = regression.conditioning.volatility_window
    n, step, seed = 5_000, 7e-4, 11
    features = random_walk_features(n, regression_window=64, step_std=step, seed=seed,
                                    volatility_window=window)
    plain = _random_walk(n, regression_window=64, step_std=step, seed=seed)
    got = features["residual"].fill_null(np.nan).to_numpy()
    assert np.array_equal(np.isnan(got), np.isnan(plain))
    assert np.array_equal(got[~np.isnan(got)], plain[~np.isnan(plain)]), "control changed"

    price = np.exp(simulate_random_walk(n, step, x0=float(np.log(1000.0)), seed=seed))
    bars = pl.DataFrame({
        "timestamp": [datetime(2021, 1, 4) + timedelta(minutes=i) for i in range(n)],
        "open": price, "high": price, "low": price, "close": price,
        "first_bid": price, "first_ask": price, "last_bid": price, "last_ask": price,
    })
    reference, _ = rolling_regression_features(bars, window=64, config=regression,
                                               timeframe="1m")
    for column in ("regression_slope", "r_squared", "log_return", "trailing_volatility"):
        a = features[column].fill_null(np.nan).to_numpy()
        b = reference[column].fill_null(np.nan).to_numpy()
        assert np.array_equal(np.isnan(a), np.isnan(b)), column
        # exp then log in the reference costs a few ulps; the definitions agree.
        assert np.allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=1e-7, atol=1e-12), column
