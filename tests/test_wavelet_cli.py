"""`xq wavelet-research`: the guards that keep an unattended grid sane."""

from __future__ import annotations

import yaml

from xauusd_quant import cli
from xauusd_quant.features.wavelet_config import load_wavelet_config


def _env(monkeypatch, tmp_path):
    for var, sub in (("XAUUSD_WAVELET_RESULTS", "wavelet"),
                     ("XAUUSD_WAVELET_FEATURES", "features"),
                     ("XAUUSD_RESEARCH_LEDGER", "ledger.parquet")):
        monkeypatch.setenv(var, str(tmp_path / sub).replace("\\", "/"))


def test_a_window_choice_is_required(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    assert cli.main(["wavelet-research", "--timeframe", "5m"]) == 2


def test_raw_price_and_unknown_series_are_refused(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    for source in ("price", "log_price", "nonsense"):
        assert cli.main(["wavelet-research", "--timeframe", "5m", "--source", source,
                         "--window", "256"]) == 2


def test_compare_with_nothing_on_disk_fails_cleanly(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    assert cli.main(["wavelet-research", "--compare"]) == 1


def test_dry_run_estimates_and_runs_nothing(world, monkeypatch, tmp_path, capsys):
    config, _ = world
    _env(monkeypatch, tmp_path)
    args = ["--config", str(config.config_path), "wavelet-research", "--timeframe", "1m",
            "--window", "64", "--dry-run"]
    assert cli.main(args) == 0
    out = capsys.readouterr().out
    assert "estimate" in out and "min" in out
    assert not (tmp_path / "wavelet" / "1m").exists()


def test_a_heavy_timeframe_is_skipped_without_the_override(world, monkeypatch, tmp_path):
    config, _ = world
    _env(monkeypatch, tmp_path)
    raw = yaml.safe_load(load_wavelet_config().config_path.read_text(encoding="utf-8"))
    raw["grid"] = {"max_bars_without_override": 10}
    path = tmp_path / "wavelet.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    args = ["--config", str(config.config_path), "wavelet-research", "--timeframe", "1m",
            "--window", "64", "--in-process", "--no-plots", "--wavelet-config", str(path)]
    assert cli.main(args) == 0
    assert not (tmp_path / "wavelet" / "1m").exists(), "900 bars > 10: skipped, not run"


def test_controls_write_the_positive_controls_and_the_boundary_study(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    raw = yaml.safe_load(load_wavelet_config().config_path.read_text(encoding="utf-8"))
    raw["boundary"] = {"trials": 20}
    path = tmp_path / "wavelet.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert cli.main(["wavelet-research", "--controls", "--no-plots",
                     "--wavelet-config", str(path)]) == 0
    offline = tmp_path / "wavelet" / "offline"
    for name in ("control_localized_oscillation", "control_chirp", "control_multiscale",
                 "boundary_study", "boundary_trials"):
        assert (offline / f"{name}.csv").exists(), name
