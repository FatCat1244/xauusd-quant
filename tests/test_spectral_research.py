"""The spectral study end to end, on a small synthetic but fully versioned pipeline."""

from __future__ import annotations

import json

import polars as pl
import pytest

from xauusd_quant.features.spectral_config import load_spectral_config
from xauusd_quant.features.store import RegressionFeatureStore
from xauusd_quant.models import load_ou_config
from xauusd_quant.research.config import load_research_config
from xauusd_quant.research.research_ledger import ResearchLedger
from xauusd_quant.research.spectral_reports import (
    HYPOTHESES,
    IC_SOURCES,
    SpectralStudy,
    _effect,
    distribution_comparison,
    generate_spectral_timeframe,
    ic_chance_rates,
    integrity_gate,
    load_spectral_studies,
    reusable_spectral_study,
    write_spectral_comparison,
)

SUMMARY_KEYS = (
    "timeframe", "input_series", "fft_window", "observations", "valid_spectral_fraction",
    "median_spectral_entropy", "median_spectral_flatness", "median_top1_power_share",
    "median_top3_power_share", "median_dominant_period_bars", "median_dominant_period_seconds",
    "dominant_period_persistence", "phase_forward_relationship", "null_control_difference",
    "reconstruction_rmse", "forward_extrapolation_rmse",
)


@pytest.fixture
def study(world, tmp_path, monkeypatch):
    config, regression = world
    for var, sub in (("XAUUSD_SPECTRAL_RESULTS", "spectral"), ("XAUUSD_SPECTRAL_FEATURES",
                                                              "spectral_features"),
                     ("XAUUSD_RESEARCH_LEDGER", "ledger.parquet"), ("XAUUSD_OU_RESULTS", "ou")):
        monkeypatch.setenv(var, str(tmp_path / sub).replace("\\", "/"))
    spectral = load_spectral_config(overrides={
        "regression_window": 32, "ou_window": 64, "fft_windows": [32, 64],
        "representative_fft_window": 32,
        "reconstruction": {"max_windows": 300},
        "ic": {"features": ["spectral_entropy", "top3_power_share", "dominant_phase_sin"],
               "horizons": [1, 5], "max_rows": 5000, "max_rows_control": 5000},
        "outcomes": {"horizons": [1, 5], "extreme_abs_z": 1.0, "buckets": 2},
        "controls": {"block_size": 16},
        "plots": {"sample_bars": 200, "spectrogram_bars": 100, "max_series_points": 500},
    })
    ou, research = load_ou_config(), load_research_config()
    RegressionFeatureStore(config, regression).build("1m", [32])
    studies = generate_spectral_timeframe(config, regression, research, ou, spectral,
                                          timeframe="1m", make_plots=True)
    return config, regression, ou, spectral, studies


def test_every_study_writes_its_outputs_and_summary(study):
    _, _, _, spectral, studies = study
    assert {(s.series, s.fft_window) for s in studies} == {
        (series, n) for series in spectral.input_series for n in (32, 64)}
    for s in studies:
        out = s.output_dir
        for name in ("summary.json", "spectral_distribution.csv", "entropy_summary.csv",
                     "power_bands.csv", "mean_spectrum.csv", "dominant_period_distribution.csv",
                     "dominant_period_stability.csv", "frequency_persistence.csv",
                     "yearly_stability.csv", "quarterly_stability.csv", "era_stability.csv",
                     "phase_analysis.csv", "reconstruction_metrics.csv",
                     "extrapolation_metrics.csv", "null_control_comparison.csv",
                     "ic_analysis.csv", "conditioning.csv", "intraday.csv"):
            assert (out / name).exists(), (s.series, s.fft_window, name)
        assert (out / "plots" / "mean_spectrum.png").exists()
        payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        for key in SUMMARY_KEYS:
            assert key in payload["summary"], key
        assert payload["provenance"]["dataset"]["tick_dataset_version"]
        sources = set(pl.read_csv(out / "spectral_distribution.csv")["source"].to_list())
        assert sources == {"real", *spectral.controls.enabled()}


def test_the_representative_window_is_stored_as_partitioned_features(study):
    _, _, _, spectral, _ = study
    root = spectral.features_path / "timeframe=1m" / "source=regression_residual" / \
        "fft_window=32"
    parts = list(root.rglob("features.parquet"))
    assert parts and (root / "_manifest.json").exists()
    frame = pl.read_parquet(parts[0])
    assert "timestamp" in frame.columns and "residual" not in frame.columns, "join by timestamp"
    assert not (spectral.features_path / "timeframe=1m" / "source=regression_residual"
                / "fft_window=64").exists(), "only the representative window by default"


def test_every_hypothesis_lands_in_the_ledger_with_a_verdict(study):
    _, _, _, spectral, _ = study
    ledger = ResearchLedger(spectral.ledger_path).load()
    assert ledger.height > 0
    assert set(ledger["hypothesis_id"].unique().to_list()) <= set(HYPOTHESES)
    assert ledger["verdict"].null_count() == 0
    assert ledger["dataset_version"].null_count() == 0
    ic_rows = ledger.filter(pl.col("hypothesis_id") == "SPEC-H-006")
    # features x targets x horizons x series x windows, every one kept
    assert ic_rows.height == 3 * 3 * 2 * 3 * 2


def test_resume_reuses_only_current_studies_and_comparison_is_written(study):
    config, regression, ou, spectral, studies = study
    s = studies[0]
    again = reusable_spectral_study(config, regression, ou, spectral, timeframe="1m",
                                    series=s.series, fft_window=s.fft_window)
    assert again is not None and again.reused
    other = load_spectral_config(overrides={
        "regression_window": 32, "ou_window": 64, "fft_windows": [32, 64],
        "representative_fft_window": 32, "controls": {"seed": 1}})
    assert reusable_spectral_study(config, regression, ou, other, timeframe="1m",
                                   series=s.series, fft_window=s.fft_window) is None
    written = write_spectral_comparison(spectral, load_spectral_studies(spectral))
    assert any(p.name == "spectral_comparison.csv" for p in written)
    counts = pl.read_csv(spectral.results_path / "hypothesis_test_counts.csv")
    assert counts["tests"].sum() == ResearchLedger(spectral.ledger_path).load().height
    chance = pl.read_csv(spectral.results_path / "ic_chance_rates.csv")
    assert set(chance["source"].to_list()) == set(IC_SOURCES), "white noise has no IC tests"
    assert (chance["exceedances"] <= chance["tests"]).all()


def test_a_null_with_zero_iqr_still_counts_in_the_verdict():
    # The residual's dominant period sits on bin 1 in almost every window, real
    # and random walk alike: that null's IQR is zero, and the verdict must still
    # hear it rather than rest on white noise alone.
    def row(source, p5, p25, p50, p75, p95):
        return {"source": source, "metric": "dominant_period_bars", "p5": p5, "p25": p25,
                "p50": p50, "p75": p75, "p95": p95}

    distribution = pl.DataFrame([
        row("real", 64, 64, 64, 64, 64), row("white_noise", 4, 8, 12, 16, 40),
        row("random_walk", 64, 64, 64, 64, 64)])
    result = {r["metric"]: r for r in distribution_comparison(
        distribution, ["white_noise", "random_walk"])}["median_dominant_period_bars"]
    assert result["effect_vs_white_noise"] == 6.5
    assert result["effect_vs_random_walk"] == 0.0
    assert result["verdict"] == "within_a_null"
    assert _effect(64.0, 128.0, 0.0, 64.0) == -1.0, "zero IQR: the 5-95 % range scales it"
    assert _effect(64.0, 128.0, 0.0, 0.0) == float("-inf"), "a point mass elsewhere"
    assert _effect(float("nan"), 1.0, 1.0) is None


def test_ic_chance_rates_compare_each_source_with_the_other_three(tmp_path):
    study_dir = tmp_path / "fft_32"
    study_dir.mkdir()
    rows = [("real", 0.5, 0.9), ("real", -0.1, 0.5), ("real", float("nan"), 1.0),
            ("random_walk", 0.2, 0.9), ("shuffled_returns", -0.3, 0.1),
            ("block_bootstrap", 0.4, 0.5), ("white_noise", 0.9, 1.0)]
    pl.DataFrame(rows, schema=["source", "rank_ic", "yearly_rank_ic_positive_share"],
                 orient="row").write_parquet(study_dir / "ic_analysis.parquet")
    study = SpectralStudy(timeframe="5m", series="ou_innovation", fft_window=32,
                          output_dir=study_dir)
    rates = {r["source"]: r for r in ic_chance_rates([study]).iter_rows(named=True)}
    assert set(rates) == set(IC_SOURCES), "white noise is not one of the IC sources"
    real = rates["real"]
    assert real["tests"] == 2, "a NaN IC is not a test"
    assert real["ceiling_from_other_sources"] == 0.4
    assert (real["exceedances"], real["exceedances_stable_sign"]) == (1, 1)
    for null in ("random_walk", "shuffled_returns", "block_bootstrap"):
        assert rates[null]["ceiling_from_other_sources"] == 0.5, "the real IC counts too"
        assert rates[null]["exceedances"] == 0


def test_the_integrity_gate_refuses_stale_features(study, monkeypatch):
    config, regression, ou, spectral, _ = study
    assert integrity_gate(config, regression, ou, spectral, "1m")["passed"]
    import xauusd_quant.research.spectral_reports as reports

    monkeypatch.setattr(reports, "dataset_lineage", lambda *a, **k: {
        "tick_dataset_status": "complete", "bars_match_ticks": False,
        "tick_dataset_version": "ticks-new", "bar_built_from_tick_version": "ticks-old"})
    gate = reports.integrity_gate(config, regression, ou, spectral, "1m")
    assert not gate["passed"] and "bars were not built" in gate["problems"][0]
