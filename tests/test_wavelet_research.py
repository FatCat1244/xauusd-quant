"""The wavelet study end to end, on a small synthetic but fully versioned pipeline."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from xauusd_quant.features.spectral import rolling_spectrum
from xauusd_quant.features.spectral_config import load_spectral_config
from xauusd_quant.features.store import RegressionFeatureStore
from xauusd_quant.features.wavelet import NON_CAUSAL_LABEL
from xauusd_quant.features.wavelet_causal import rolling_wavelet
from xauusd_quant.features.wavelet_config import load_wavelet_config
from xauusd_quant.models import load_ou_config
from xauusd_quant.research.config import load_research_config
from xauusd_quant.research.research_ledger import ResearchLedger
from xauusd_quant.research.spectral_nulls import SourceData
from xauusd_quant.research.wavelet_predictiveness import incremental_study, incremental_summary
from xauusd_quant.research.wavelet_reports import (
    HYPOTHESES,
    POST_HOC_HYPOTHESES,
    _ledger_entries,
    _ridge_entries,
    generate_wavelet_timeframe,
    integrity_gate,
    load_wavelet_studies,
    post_hoc_baseline_entries,
    reusable_wavelet_study,
    write_wavelet_comparison,
)

SUMMARY_KEYS = (
    "timeframe", "source", "rolling_window", "wavelet", "observations", "valid_feature_fraction",
    "median_wavelet_entropy", "median_dominant_period_seconds", "median_fast_slow_energy_ratio",
    "dominant_scale_persistence", "null_control_difference", "best_absolute_ic", "best_rank_ic",
    "incremental_information_over_fft", "feature_redundancy",
)


def _configs(tmp_path, monkeypatch):
    for var, sub in (("XAUUSD_WAVELET_RESULTS", "wavelet"),
                     ("XAUUSD_WAVELET_FEATURES", "wavelet_features"),
                     ("XAUUSD_RESEARCH_LEDGER", "ledger.parquet"), ("XAUUSD_OU_RESULTS", "ou"),
                     ("XAUUSD_SPECTRAL_RESULTS", "spectral"),
                     ("XAUUSD_SPECTRAL_FEATURES", "spectral_features")):
        monkeypatch.setenv(var, str(tmp_path / sub).replace("\\", "/"))
    wavelet = load_wavelet_config(overrides={
        "regression_window": 32, "ou_window": 64,
        "causal_features": {"rolling_windows": [64, 128], "representative_window": 64},
        "cwt": {"slice_bars": 256},
        "ic": {"horizons": [1, 5], "max_rows": 5000, "max_rows_control": 5000},
        "incremental": {"horizons": [1, 5], "test_blocks": [[2021, 2021]]},
        "null_controls": {"block_sizes": [16, 256]},
        "plots": {"sample_bars": 200},
        "offline": {"slices_per_timeframe": 3},
    })
    spectral = load_spectral_config(overrides={
        "regression_window": 32, "ou_window": 64, "fft_windows": [64, 128],
        "representative_fft_window": 64})
    return wavelet, spectral


@pytest.fixture
def study(world, tmp_path, monkeypatch):
    config, regression = world
    wavelet, spectral = _configs(tmp_path, monkeypatch)
    ou, research = load_ou_config(), load_research_config()
    RegressionFeatureStore(config, regression).build("1m", [32])
    studies = generate_wavelet_timeframe(config, regression, research, ou, spectral, wavelet,
                                         timeframe="1m", make_plots=True,
                                         require_spectral_features=False)
    return config, regression, ou, spectral, wavelet, studies


def test_every_study_writes_its_outputs_and_summary(study):
    *_, wavelet, studies = study
    assert {(s.series, s.window) for s in studies} == {
        (series, n) for series in wavelet.input_series for n in (64, 128)}
    for s in studies:
        out = s.output_dir
        for name in ("summary.json", "metric_distribution.csv", "energy_summary.csv",
                     "entropy_summary.csv", "scale_persistence.csv", "run_summary.csv",
                     "drift_summary.csv", "burst_summary.csv", "yearly_stability.csv",
                     "quarterly_stability.csv", "era_stability.csv",
                     "volatility_conditioning.csv", "trend_conditioning.csv",
                     "ou_conditioning.csv", "intraday.csv", "extreme_conditioning.csv",
                     "ic_analysis.csv", "fft_comparison.csv", "fft_determinism.csv",
                     "dyadic_comparison.csv", "feature_redundancy.csv", "feature_quality.csv",
                     "null_controls.csv", "hypothesis_verdicts.csv"):
            assert (out / name).exists(), (s.series, s.window, name)
        assert (out / "plots" / "energy_by_scale.png").exists()
        payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        for key in SUMMARY_KEYS:
            assert key in payload["summary"], key
        provenance = payload["provenance"]
        assert provenance["dataset"]["tick_dataset_version"]
        assert provenance["packages"]["PyWavelets"]
        assert provenance["code_fingerprint"]
        sources = set(pl.read_csv(out / "metric_distribution.csv")["source"].to_list())
        assert sources == {"real", *wavelet.controls.enabled()}
        ou_buckets = set(pl.read_csv(out / "ou_conditioning.csv")["bucket"].to_list())
        assert ou_buckets <= {"fast", "medium", "slow", "near_unit_root", "invalid"}


def test_the_representative_window_is_stored_as_causal_features_with_provenance(study):
    *_, wavelet, _ = study
    root = wavelet.features_path / "timeframe=1m" / "source=regression_residual" / "window=64"
    parts = list(root.rglob("part-0.parquet"))
    assert parts and (root / "_manifest.json").exists()
    manifest = json.loads((root / "_manifest.json").read_text(encoding="utf-8"))
    assert manifest["causal"] is True and manifest["join_key"] == "timestamp"
    assert all(c["causal"] and c["live_safe"] for c in manifest["columns"].values())
    assert manifest["wavelet"]["family"] == "db4" and manifest["wavelet"]["bands"]
    assert manifest["wavelet"]["padding"].startswith("none")
    for key in ("tick_dataset_version", "bar_dataset_version", "engine_fingerprint",
                "code_fingerprint", "packages", "generated_utc", "source_series"):
        assert key in manifest, key
    frame = pl.read_parquet(parts[0])
    assert "timestamp" in frame.columns
    assert not any(c.startswith("offline_") or c in ("residual", "mid", "close")
                   for c in frame.columns), "no offline values, no repeated source columns"
    assert not (wavelet.features_path / "timeframe=1m" / "source=regression_residual"
                / "window=128").exists(), "only the representative window by default"


def test_hypotheses_are_registered_and_every_verdict_lands_in_the_ledger(study):
    *_, wavelet, _ = study
    registered = json.loads((wavelet.results_path / "hypotheses_registered.json")
                            .read_text(encoding="utf-8"))
    assert registered["hypotheses"] == HYPOTHESES and registered["registered_utc"]
    ledger = ResearchLedger(wavelet.ledger_path).load()
    wave = ledger.filter(pl.col("hypothesis_id").str.starts_with("WAVE-"))
    assert wave.height > 0 and set(wave["hypothesis_id"].unique()) <= set(HYPOTHESES)
    assert wave["verdict"].null_count() == 0 and wave["dataset_version"].null_count() == 0
    ic_rows = wave.filter(pl.col("hypothesis_id") == "WAVE-H-006")
    # features x targets x horizons x series x windows, every one kept
    assert ic_rows.height == len(wavelet.ic.features) * 5 * 2 * 3 * 2


def test_offline_scalograms_and_ridges_are_labelled_non_causal(study):
    *_, wavelet, _ = study
    series_dir = wavelet.results_path / "1m" / "regression_residual"
    ridges = pl.read_csv(series_dir / "offline_ridges.csv")
    assert ridges.height > 0 and set(ridges["label"].unique()) == {NON_CAUSAL_LABEL}
    assert {"real", "random_walk"} <= set(ridges["source"].unique())
    assert list((series_dir / "plots").glob("scalogram_*.png"))
    assert not list(wavelet.features_path.rglob("*ridge*")), "offline output is never a feature"


def test_resume_reuses_only_current_studies_and_comparison_is_written(study, tmp_path,
                                                                      monkeypatch):
    config, _, _, _, wavelet, studies = study
    s = studies[0]
    again = reusable_wavelet_study(config, wavelet, timeframe="1m", series=s.series,
                                   window=s.window)
    assert again is not None and again.reused
    other = load_wavelet_config(overrides={
        "regression_window": 32, "ou_window": 64, "null_controls": {"seed": 1},
        "causal_features": {"rolling_windows": [64, 128], "representative_window": 64}})
    assert reusable_wavelet_study(config, other, timeframe="1m", series=s.series,
                                  window=s.window) is None
    written = {p.name for p in write_wavelet_comparison(wavelet, load_wavelet_studies(wavelet))}
    assert {"wavelet_comparison.csv", "window_comparison.csv", "cross_timeframe_bands.csv",
            "hypothesis_test_counts.csv", "ic_chance_rates.csv"} <= written


def test_the_gate_requires_current_fft_features(study):
    config, regression, ou, spectral, wavelet, _ = study
    gate = integrity_gate(config, regression, ou, spectral, wavelet, "1m")
    assert not gate["passed"] and any("no Prompt #5 FFT features" in p for p in gate["problems"])
    lineage = gate["lineage"]
    for series in wavelet.input_series:
        root = (spectral.features_path / "timeframe=1m" / f"source={series}"
                / f"fft_window={spectral.representative_fft_window}")
        root.mkdir(parents=True, exist_ok=True)
        (root / "_manifest.json").write_text(json.dumps({
            "tick_dataset_version": lineage["tick_dataset_version"],
            "bar_dataset_version": lineage["bar_dataset_version"],
            "engine_fingerprint": spectral.engine_fingerprint()}), encoding="utf-8")
    assert integrity_gate(config, regression, ou, spectral, wavelet, "1m")["passed"]
    (root / "_manifest.json").write_text(json.dumps({
        "tick_dataset_version": "ticks-old", "bar_dataset_version": "bars-old",
        "engine_fingerprint": spectral.engine_fingerprint()}), encoding="utf-8")
    gate = integrity_gate(config, regression, ou, spectral, wavelet, "1m")
    assert not gate["passed"] and any("stale" in p for p in gate["problems"])


def test_incremental_models_on_a_multi_year_source():
    """Models A-D on a synthetic source whose wavelet state does carry information."""
    rng = np.random.default_rng(3)
    n = 24_000
    stamps = pl.Series("timestamp", np.datetime64("2008-01-01T00:00", "us")
                       + np.arange(n) * np.timedelta64(4, "h"))            # 2008 .. ~2019
    vol = np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    residual = np.cumsum(rng.normal(size=n) * vol)
    residual -= np.convolve(residual, np.ones(50) / 50, mode="same")
    returns = np.diff(residual, prepend=np.nan)
    columns = {
        "regression_residual": residual, "log_return": returns,
        "ou_innovation": rng.normal(size=n) * vol, "abs_ou_innovation": np.abs(rng.normal(size=n)),
        "residual_zscore": residual / np.std(residual), "trailing_volatility": vol,
        "regression_slope": rng.normal(size=n), "r_squared": rng.random(n),
        "ou_zscore": rng.normal(size=n), "ou_mu": np.zeros(n),
        "ou_half_life_bars": rng.uniform(5, 50, n), "ou_valid": np.ones(n),
        "ou_state_code": np.zeros(n),
    }
    source = SourceData(name="real", timestamps=stamps, columns=columns,
                        missing_slots=np.zeros(n), bar_seconds=14_400.0)
    wavelet = load_wavelet_config(overrides={
        "incremental": {"horizons": [1, 5], "test_blocks": [[2014, 2015], [2016, 2017]],
                        "max_rows": 20_000}})
    spectral = load_spectral_config()
    result = rolling_wavelet(residual, window=128, config=wavelet, bar_seconds=14_400.0)
    spectrum = rolling_spectrum(residual, fft_window=128, config=spectral, bar_seconds=14_400.0)
    tables = incremental_study(result, spectrum, source, wavelet, source_name="real")
    linear = tables["linear"]
    assert set(linear["model"].unique()) == {"A", "B", "C", "D"}
    assert set(linear["fold"].unique()) == {"2014-2015", "2016-2017"}
    assert tables["by_feature"]["model"].str.starts_with("C+").all()
    summary = incremental_summary(linear, tables["binary"])
    assert {"delta_B_A", "delta_C_B", "delta_D_C", "folds_d_beats_c"} <= set(summary.columns)
    assert (summary.filter(pl.col("kind") == "continuous")["folds"] == 2).all()

    same = incremental_study(result, spectrum, source, wavelet, source_name="real",
                             baseline="original", per_feature=False)
    assert same["linear"].equals(linear), "'original' is the registered model, bit for bit"
    assert same["by_feature"].is_empty()
    rich = incremental_study(result, spectrum, source, wavelet, source_name="real",
                             baseline="rich", per_feature=False)["linear"]
    key = ["model", "target", "horizon", "fold"]
    both = linear.join(rich, on=key, suffix="_rich").filter(pl.col("model") == "C")
    assert both.height and not (both["oos_r2"] == both["oos_r2_rich"]).all(), "C is widened"


def test_an_undefined_statistic_is_untested_never_a_verdict():
    """NaN fails every comparison silently; each rule must read it as missing.

    Before the fix a NaN rank IC (a fast/slow ratio with no slow band) was
    recorded as "within_null_range" and a NaN redundancy as "redundant".
    """
    nan = float("nan")
    ic = pl.DataFrame({
        "feature": ["wavelet_local_fast_slow_log_ratio", "wavelet_burst_z_max",
                    "wavelet_entropy"],
        "target": ["future_volatility"] * 3, "horizon": [5] * 3,
        "rank_ic": [nan, 0.5, 0.05], "yearly_rank_ic_positive_share": [nan, 0.9, 0.5]})
    passes = {
        "real": {"role": "real", "ic": ic,
                 "fft_determinism": pl.DataFrame({
                     "wavelet_feature": ["wavelet_local_fast_slow_log_ratio",
                                         "wavelet_entropy"],
                     "oos_r2_from_fft": [nan, 0.9]}),
                 "redundancy": pl.DataFrame({
                     "feature": ["wavelet_local_fast_slow_log_ratio", "wavelet_entropy"],
                     "max_abs_spearman_existing": [nan, nan],
                     "max_abs_spearman_fft": [nan, 0.2],
                     "most_similar_existing": [None, "x"], "most_similar_fft": [None, "y"]})},
        "shuffled_returns": {"role": "core", "ic": ic.with_columns(
            pl.Series("rank_ic", [nan, 0.1, -0.08]))},
    }
    entries = _ledger_entries(passes, pl.DataFrame(schema={"metric": pl.String}),
                              timeframe="5m", series="log_return", window=128,
                              dataset_version="v", study="s", summary=pl.DataFrame())
    verdicts = {(e["hypothesis_id"], e["feature"]): (e["verdict"], e.get("value"))
                for e in entries}
    assert verdicts["WAVE-H-006", "wavelet_local_fast_slow_log_ratio"] == ("untested", None)
    assert verdicts["WAVE-H-006", "wavelet_burst_z_max"][0] == "exceeds_null_max_with_stable_sign"
    assert verdicts["WAVE-H-006", "wavelet_entropy"][0] == "within_null_range"
    assert all(e["control_value"] == 0.1 for e in entries if e["hypothesis_id"] == "WAVE-H-006")
    assert verdicts["WAVE-H-011", "wavelet_local_fast_slow_log_ratio"][0] == "untested"
    assert verdicts["WAVE-H-011", "wavelet_entropy"][0] == "near_deterministic_of_fft"
    assert verdicts["WAVE-H-012", "wavelet_local_fast_slow_log_ratio"] == ("untested", None)
    assert verdicts["WAVE-H-012", "wavelet_entropy"] == ("distinct", 0.2)

    ridges = pl.DataFrame({
        "wavelet": ["cmor"] * 10, "slice": [f"s{i // 2}" for i in range(10)],
        "source": ["real", "white_noise"] * 5,
        "median_cycles": [nan, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0]})
    (ridge,) = _ridge_entries(ridges, load_wavelet_config(), timeframe="5m",
                              series="log_return", dataset_version="v")
    assert (ridge["value"], ridge["control_value"]) == (4.0, 4.0), "the NaN slice is not a test"
    assert ridge["verdict"] == "within_null_range", "fewer than five testable slices"


def test_post_hoc_baseline_verdicts_use_the_registered_rule_against_the_same_baseline():
    """WAVE-P-001: each widened baseline is read against nulls fitted with that baseline."""
    nan = float("nan")
    rows = []
    for baseline, real, null, folds in (("original", 0.03, 0.001, 4), ("hour_only", 0.02, 0.001, 4),
                                        ("rv_only", 0.002, 0.004, 4), ("rich", nan, 0.0, 0)):
        rows.append({"baseline": baseline, "source": "real", "delta_D_C": real,
                     "folds_d_beats_c": folds})
        rows.append({"baseline": baseline, "source": "random_walk", "delta_D_C": null,
                     "folds_d_beats_c": 1})
    summary = pl.DataFrame(rows).with_columns(
        pl.lit("future_volatility").alias("target"), pl.lit(20).alias("horizon"),
        pl.lit("oos_r2").alias("metric"), pl.lit(4).alias("folds"),
        pl.lit(0.3).alias("C"), pl.lit(0.31).alias("D"))
    entries = post_hoc_baseline_entries(summary, timeframe="15m", series="log_return",
                                        window=512, dataset_version="v", study="s")
    verdicts = {e["feature"]: e["verdict"] for e in entries}
    assert verdicts == {"wavelet_block|baseline=hour_only": "incremental_beyond_nulls",
                        "wavelet_block|baseline=rv_only": "no_incremental_information",
                        "wavelet_block|baseline=rich": "untested"}, "no row for 'original'"
    assert {e["hypothesis_id"] for e in entries} == set(POST_HOC_HYPOTHESES)
    assert not set(POST_HOC_HYPOTHESES) & set(HYPOTHESES), "never among the registered set"
    rv = next(e for e in entries if e["feature"].endswith("rv_only"))
    assert rv["control_value"] == 0.004 and rv["details"]["post_hoc"] is True
