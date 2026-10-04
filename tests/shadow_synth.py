"""Hand-checkable fake MT5 metadata/feed; no real account or vendor package."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from risk_synth import configuration
from xauusd_quant.alpha_portfolio.allocation import Allocation
from xauusd_quant.alpha_portfolio.intents import ObservedBarPolicy
from xauusd_quant.alpha_portfolio.portfolio import WeightUpdate
from xauusd_quant.alpha_portfolio.registry import Alpha
from xauusd_quant.execution.config import ExecutionConfig, content_hash
from xauusd_quant.execution.engine import MemoryRecorder
from xauusd_quant.features.factory_config import load_features_config
from xauusd_quant.risk.portfolio import RiskPortfolio
from xauusd_quant.shadow.config import ShadowConfig
from xauusd_quant.shadow.ingestion import Tick
from xauusd_quant.shadow.pipeline import Binding, ShadowPipeline

ROOT = Path(__file__).resolve().parents[1]
T = datetime(2021, 1, 4, 12, tzinfo=UTC)


def row(second: float, bid: float = 1800, **extra: Any) -> dict[str, Any]:
    return {"time_msc": int((T + timedelta(seconds=second)).timestamp() * 1000),
            "bid": bid, "ask": bid + .2, "flags": 6, **extra}


def tick(second: float, sequence: int, *, backfill: bool = False, bid: float = 1800) -> Tick:
    r = row(second, bid)
    return Tick(r["time_msc"], r["bid"], r["ask"], content_hash(r), sequence,
                T + timedelta(seconds=second + .2), backfill)


class Native:
    ACCOUNT_TRADE_MODE_DEMO = 0
    COPY_TICKS_ALL = 0

    def __init__(self, terminal: Path) -> None:
        self.terminal = SimpleNamespace(connected=True, path=str(terminal.parent))
        self.account = SimpleNamespace(login=123456, server="SYNTHETIC_DEMO_SERVER",
            company="Exness Synthetic Fixture", trade_mode=0)
        self.symbol = SimpleNamespace(name="SYNTHETIC_GOLD", visible=True, digits=2,
            trade_contract_size=100., trade_tick_size=.01, point=.01,
            volume_min=.01, volume_step=.01, volume_max=1.,
            currency_profit="USD", currency_margin="USD")
        self.calls: list[str] = []
        self.rows = [row(0)]

    def initialize(self, path: str, **kwargs: Any) -> bool:
        self.calls.append("initialize")
        assert "login" not in kwargs and "password" not in kwargs and path
        return True

    def terminal_info(self) -> Any:
        return self.terminal

    def account_info(self) -> Any:
        return self.account

    def symbol_info(self, symbol: str) -> Any:
        assert symbol == self.symbol.name
        return self.symbol

    def symbol_info_tick(self, symbol: str) -> Any:
        return SimpleNamespace(**self.rows[-1])

    def copy_ticks_from(self, symbol: str, start: datetime, count: int, flags: int) -> Any:
        assert start.tzinfo is UTC
        self.calls.append("copy_ticks_from")
        rows = [r for r in self.rows if r["time_msc"] >= start.timestamp() * 1000][:count]
        dtype = [("time_msc", "i8"), ("bid", "f8"), ("ask", "f8"), ("flags", "i4")]
        return np.array([tuple(r[n] for n, _ in dtype) for r in rows], dtype=dtype)

    def shutdown(self) -> None:
        self.calls.append("shutdown")

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"forbidden or unsupported native operation: {name}")


def local_config(terminal: Path) -> ShadowConfig:
    terminal.write_text("synthetic executable fixture", encoding="utf-8")
    return ShadowConfig(terminal_path=str(terminal), expected_login=123456,
        expected_server="SYNTHETIC_DEMO_SERVER", expected_company="Exness Synthetic Fixture",
        symbol="SYNTHETIC_GOLD", timeframes=("1m",), backfill_seconds=0, duration_seconds=1)


class FixtureModel:
    model_id = "MODEL_A_V001"
    feature_id = "FEATURES_A_V001"
    features = ("ret_1",)
    target = "future_return"
    units = "log_mid_return"
    timeframe = "1m"
    horizon = 1
    identity = "SYNTHETIC_FIXED_RETURN_MAPPING_V001"

    def predict(self, values: Any) -> float:
        return float(values["ret_1"]) * .2


class FixtureDiagnostics:
    identity = "SYNTHETIC_DIAGNOSTICS_V001"

    def evaluate(self, available: datetime, values: Any) -> dict[str, Any]:
        return {"available_utc": available, "calibration_healthy": True,
            "diagnostic_healthy": True, "volatility": .001, "uncertainty": .001}

    def state(self) -> dict[str, Any]:
        return {"identity": self.identity, "calibration": "fixed synthetic fixture; stateless"}

    def restore(self, body: dict[str, Any]) -> None:
        if body != self.state():
            raise ValueError("incompatible diagnostic checkpoint")


def governed(*, diagnostics: bool = True) -> tuple[ShadowPipeline, MemoryRecorder]:
    cfg = ShadowConfig(timeframes=("1m",), max_gap_seconds=30)
    sink = MemoryRecorder()
    execution = ExecutionConfig(timeframe="1m", quantity_lots=.02)
    # This fixture publishes once/minute, unlike Stage16's seconds-long tests.
    # The explicit synthetic health window supports that cadence; no live value.
    portfolio = RiskPortfolio(execution, {"A": "A_V001"}, .02, sink,
                              configuration("A", max_health_age_seconds=120))
    alpha = Alpha("A", "A_V001", "1m", 1, {}, {}, "BLOCKED", "BLOCKED", None,
                  ("synthetic fixture only",), synthetic=True)
    binding = Binding(alpha, FixtureModel(), ObservedBarPolicy("A", "A_V001", execution, 300),
                      None, FixtureDiagnostics() if diagnostics else None)
    pipeline = ShadowPipeline(cfg, load_features_config(root=ROOT), sink, bindings=(binding,),
        portfolio=portfolio, cutoff=T, offline_synthetic=True)
    portfolio.consume([WeightUpdate(Allocation("EQUAL_V001", T,
        T - timedelta(seconds=1), {"A": 1.}, "equal", {}))])
    return pipeline, sink


def feed(pipeline: ShadowPipeline, ticks: list[Tick]) -> None:
    for t in ticks:
        pipeline.consume([t], lambda t=t: t.received_utc + timedelta(milliseconds=10))


def trend(count: int = 26) -> list[Tick]:
    return [tick(i * 10, i + 1, bid=1800 + (i // 6) * .01) for i in range(count)]
