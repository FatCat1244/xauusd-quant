"""Explicit offline synthetic units/health; never a claim of market eligibility."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from xauusd_quant.risk.contracts import AccountSnapshot, HealthSnapshot, MarketSnapshot, RiskRequest
from xauusd_quant.risk.policy import RiskConfiguration, load_configuration

T = datetime(2021, 1, 4, 12, tzinfo=UTC)
ROOT = Path(__file__).resolve().parents[1]


def configuration(*alphas: str, **changes: object) -> RiskConfiguration:
    cfg = load_configuration(ROOT / "config/risk_synthetic.yaml")
    assert cfg.policy is not None
    policy = replace(cfg.policy, alpha_contracts={a: cfg.policy.alpha_contracts[a] for a in alphas or ("A", "B")}, **changes)
    return replace(cfg, policy=policy)


def account(second: float = 0, **changes: object) -> AccountSnapshot:
    at = T + timedelta(seconds=second)
    return replace(AccountSnapshot(f"ACCOUNT_{second}", at, at, "SYNTHETIC_ACCOUNT_V001", 10000,
                                   10000, 10000, 0, True, True, open_estimated_loss=0), **changes)


def market(second: float = 0, **changes: object) -> MarketSnapshot:
    at = T + timedelta(seconds=second)
    return replace(MarketSnapshot(at, at, 1799.9, 1800.1), **changes)


def health(alpha: str = "A", second: float = 0, **changes: object) -> HealthSnapshot:
    at = T + timedelta(seconds=second)
    return replace(HealthSnapshot(alpha, f"{alpha}_V001", f"MODEL_{alpha}_V001", f"FEATURES_{alpha}_V001",
        at, at, at, at + timedelta(seconds=300), .001, "log_mid_return", 300,
        True, True, True, True, True, True, .01, .01), **changes)


def request(identity: str = "INTENT_1", target: float = .02, second: float = 0,
            **changes: object) -> RiskRequest:
    at = T + timedelta(seconds=second)
    return replace(RiskRequest(identity, "SYNTHETIC_PORTFOLIO_V001", "EQUAL_V001", at, at,
                               target, {"A": target}, "horizon_stress"), **changes)
