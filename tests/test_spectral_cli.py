"""`xq spectral-research`: the guards that keep an unattended grid sane."""

from __future__ import annotations

import yaml

from xauusd_quant import cli
from xauusd_quant.features.spectral_config import load_spectral_config


def _env(monkeypatch, tmp_path):
    for var, sub in (("XAUUSD_SPECTRAL_RESULTS", "spectral"),
                     ("XAUUSD_SPECTRAL_FEATURES", "features"),
                     ("XAUUSD_RESEARCH_LEDGER", "ledger.parquet")):
        monkeypatch.setenv(var, str(tmp_path / sub).replace("\\", "/"))


def test_a_window_choice_is_required(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    assert cli.main(["spectral-research", "--timeframe", "5m"]) == 2


def test_raw_price_and_unknown_series_are_refused(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    for source in ("price", "log_price", "nonsense"):
        assert cli.main(["spectral-research", "--timeframe", "5m", "--source", source,
                         "--fft-window", "256"]) == 2


def test_compare_with_nothing_on_disk_fails_cleanly(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    assert cli.main(["spectral-research", "--compare"]) == 1


def test_a_heavy_timeframe_is_skipped_without_the_override(world, monkeypatch, tmp_path):
    config, _ = world
    _env(monkeypatch, tmp_path)
    raw = yaml.safe_load(load_spectral_config().config_path.read_text(encoding="utf-8"))
    raw["grid"] = {"max_bars_without_override": 10}
    raw["regression_window"], raw["ou_window"] = 32, 64
    raw["fft_windows"], raw["representative_fft_window"] = [32], 32
    path = tmp_path / "spectral.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    args = ["--config", str(config.config_path), "spectral-research", "--timeframe", "1m",
            "--fft-window", "32", "--in-process", "--no-plots", "--spectral-config", str(path)]
    assert cli.main(args) == 0
    assert not (tmp_path / "spectral" / "1m").exists(), "900 bars > 10: skipped, not run"
