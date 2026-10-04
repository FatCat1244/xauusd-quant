"""Bounded chronological synthetic comparisons; no market alpha is promoted."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from ..execution.config import ExecutionConfig, content_hash
from ..execution.engine import Quote
from ..execution.io import DiskRecorder, write_new_json
from ..robustness.reports import concentration
from ..strategy_validation.metrics import fold_metrics, snapshot
from ..strategy_validation.plan import Fold
from ..strategy_validation.runs import TrialLedger
from .allocation import fit
from .intents import Intent
from .plan import PortfolioPlan, freeze
from .portfolio import Portfolio, WeightUpdate, event_key

SPECS = {"SYNTH_5M": "SYNTH_SIGN_5M_V001", "SYNTH_15M": "SYNTH_CONTRARIAN_15M_V001"}


def fixture(seed: int) -> tuple[list[Quote], list[Intent], datetime, datetime]:
    """IID price increments; signals use completed own-frequency returns only.

    A deterministic synthetic mechanism exercises opposing exposure and unequal
    frequency. It supplies no population predictive edge; no profitability test.
    """
    origin = datetime(2021, 1, 4, tzinfo=UTC)
    start = origin + timedelta(seconds=64 * 300)
    end = start + timedelta(hours=1)
    rng = np.random.default_rng(seed)
    mids = 1800 + np.cumsum(rng.normal(0, 0.08, 76 * 30 + 1))
    quotes = [
        Quote(
            origin + timedelta(seconds=i * 10),
            (origin + timedelta(seconds=i * 10)).replace(tzinfo=None),
            float(mid - 0.05),
            float(mid + 0.05),
            i,
        )
        for i, mid in enumerate(mids)
    ]
    intents = []
    for alpha, seconds, sign in (("SYNTH_5M", 300, 1), ("SYNTH_15M", 900, -1)):
        for step in range(seconds // 10, len(quotes), seconds // 10):
            at = quotes[step].timestamp_utc + timedelta(seconds=1)
            value = float(np.sign(mids[step] - mids[step - seconds // 10])) * sign
            intents.append(
                Intent(
                    alpha,
                    SPECS[alpha],
                    quotes[step].timestamp_utc,
                    at,
                    at + timedelta(seconds=seconds),
                    value,
                    "active" if value else "flat",
                    "synthetic fixed sign of completed return",
                    "5m" if seconds == 300 else "15m",
                    1,
                )
            )
    return quotes, intents, start, end


class Capture:
    """Only bounded sampled marks/trades held, all engine events streamed."""

    def __init__(self, sink: DiskRecorder) -> None:
        self.sink = sink
        self.equity: dict[datetime, float | None] = {}
        self.trades: list[dict[str, Any]] = []

    def __call__(self, table: str, row: dict[str, Any]) -> None:
        if table == "equity":
            stamp = row["timestamp_utc"]
            if stamp.minute % 5 == 0 and stamp.second == 0:
                self.equity[stamp] = row["equity_account"]
        if table == "trades":
            self.trades.append(row.copy())
        self.sink(table, row)


def synthetic_studies(
    plan: PortfolioPlan, directory: Path, ledger: TrialLedger, check_budget: Callable[[], None]
) -> dict[str, Any]:
    quotes, intents, start, end = fixture(plan.seed)
    execution = ExecutionConfig(quantity_lots=plan.budget_lots, equity_sample_ms=1000)
    before = start - timedelta(days=1)
    training_times = [quotes[0].timestamp_utc + timedelta(seconds=i * 300) for i in range(1, 64)]
    columns = []
    training_identity: dict[str, Any] = {}
    for alpha, spec in SPECS.items():

        def train(alpha: str = alpha, spec: str = spec) -> list[float]:
            path = directory / f"training_{alpha}"
            path.mkdir()
            events: list[Quote | Intent | WeightUpdate] = [
                q for q in quotes if q.timestamp_utc < start
            ]
            events.extend(i for i in intents if i.alpha_id == alpha and i.available_utc < start)
            events.sort(key=event_key)
            if len(events) > plan.maximum_events:
                raise RuntimeError("frozen training event budget exhausted")
            allocation = fit((alpha,), "equal", quotes[0].timestamp_utc, [], [], np.empty((0, 1)))
            with DiskRecorder(path) as sink:
                capture = Capture(sink)
                portfolio = Portfolio(execution, {alpha: spec}, plan.budget_lots, capture)
                portfolio.consume([WeightUpdate(allocation), *events])
                result = portfolio.finish(start)
            marks = [capture.equity[t] for t in training_times]
            changes = [
                float("nan") if a is None or b is None else (b - a) / execution.initial_cash_account
                for a, b in zip(marks[:-1], marks[1:], strict=True)
            ]
            write_new_json(path / "summary.json", result)
            training_identity[alpha] = {
                "policy": spec,
                "execution": execution.resolved(),
                "source": "synthetic prior-only shared-engine standalone",
            }
            return changes

        column = ledger.call(
            "synthetic_prior_sleeve_returns", {"alpha": alpha, "cutoff": start}, train
        )
        if column is None:
            raise RuntimeError("required risk input study failed")
        columns.append(column)
        check_budget()
    returns = np.asarray(columns).T
    stamps = training_times[1:]
    matured = stamps.copy()  # marks known at that quote; never future price targets
    # Dependence views are deliberately distinct. No actual forecast values are
    # supplied by this fixture, so forecast correlation is unavailable.
    aligned_signals = []
    for stamp in stamps[-32:]:
        at = stamp + timedelta(seconds=2)
        values = []
        for alpha in SPECS:
            available = [
                i
                for i in intents
                if i.alpha_id == alpha and i.available_utc <= at < i.valid_until_utc
            ]
            values.append(available[-1].target if available else 0)
        aligned_signals.append(values)
    write_new_json(
        directory / "dependence_V001.json",
        {
            "forecast_correlation": {
                "status": "unavailable",
                "reason": "fixture emits policy intents, not forecasts",
            },
            "signal_correlation": np.corrcoef(np.asarray(aligned_signals).T).tolist(),
            "standalone_policy_return_correlation": np.corrcoef(returns[-32:].T).tolist(),
            "negative_coloss_fraction": float((returns[-32:] < 0).all(axis=1).mean()),
            "tail_coloss_status": "unavailable: only32 dependent synthetic observations",
            "alignment": "preceding32 5-minute grid; own-frequency intent carried only until explicit expiry; same fixed-capital return convention",
            "shared_exposure": "both sleeves trade the same XAUUSD account; model names do not establish independence",
            "uncertainty": "short dependent synthetic sample; no effective independent observations or IID significance; half-window covariance sensitivity in each allocation",
        },
    )
    freeze(
        directory / "normalization_V001.json",
        {
            "rule": "sign of own-frequency completed return -> signed unit budget fraction",
            "fit": "none",
            "software_fixture_only": True,
            "specifications": SPECS,
            "training_identity": training_identity,
            "data_sha256": content_hash(
                {
                    "quotes": [(q.timestamp_utc, q.bid, q.ask) for q in quotes],
                    "intents": [i.resolved() for i in intents],
                }
            ),
        },
    )
    combinations = [
        ("no_trading", (), "equal"),
        ("standalone_5m", ("SYNTH_5M",), "equal"),
        ("standalone_15m", ("SYNTH_15M",), "equal"),
        ("equal", tuple(SPECS), "equal"),
        ("minimum_variance_shrunk", tuple(SPECS), "minimum_variance_shrunk"),
        ("equal_remove_5m", ("SYNTH_15M",), "equal"),
        ("equal_remove_15m", ("SYNTH_5M",), "equal"),
    ]
    summaries = []
    folds = [
        Fold(
            "F01",
            before.isoformat(),
            start.isoformat(),
            (start + timedelta(minutes=30)).isoformat(),
            (),
        ),
        Fold(
            "F02",
            before.isoformat(),
            (start + timedelta(minutes=30)).isoformat(),
            end.isoformat(),
            (),
        ),
    ]
    for name, universe, method in combinations:
        for scenario in ("BASE", "ADVERSE_COST"):

            def evaluate(
                name: str = name,
                universe: tuple[str, ...] = universe,
                method: str = method,
                scenario: str = scenario,
            ) -> dict[str, Any]:
                path = directory / f"{name}_{scenario}"
                path.mkdir()
                config = (
                    execution
                    if scenario == "BASE"
                    else replace(
                        execution,
                        scenario_id="SYNTH_ADVERSE_COST_V001",
                        commission_account_per_lot_per_leg=execution.commission_account_per_lot_per_leg
                        * 2,
                        slippage_usd_per_ounce_per_leg=execution.slippage_usd_per_ounce_per_leg
                        + 0.02,
                    )
                )
                freeze(path / "execution_assumptions_V001.json", config.resolved())
                indices = [list(SPECS).index(alpha) for alpha in universe]
                risk = returns[:, indices]
                allocations = [
                    fit(universe, method, fold.start, stamps, matured, risk) for fold in folds
                ]
                for fold, allocation in zip(folds, allocations, strict=True):
                    freeze(path / f"{fold.fold_id}_allocation_V001.json", allocation.resolved())
                events: list[Quote | Intent | WeightUpdate] = [WeightUpdate(a) for a in allocations]
                events.extend(q for q in quotes if start <= q.timestamp_utc < end)
                events.extend(
                    i for i in intents if i.alpha_id in universe and start <= i.available_utc < end
                )
                events.sort(key=event_key)
                if len(events) > plan.maximum_events:
                    raise RuntimeError("frozen event budget exhausted")
                with DiskRecorder(path) as sink:
                    capture = Capture(sink)
                    portfolio = Portfolio(
                        config, {a: SPECS[a] for a in universe}, plan.budget_lots, capture
                    )
                    fold_records = []
                    for fold in folds:
                        a = snapshot(portfolio.engine, fold.start)
                        portfolio.consume(
                            event
                            for event in events
                            if fold.start <= event_key(event)[0] < fold.end
                        )
                        portfolio._expire_until(fold.end)
                        b = snapshot(portfolio.engine, fold.end)
                        record = fold_metrics(fold.fold_id, a, b)
                        sink("fold_metrics", record)
                        fold_records.append(record)
                    result = portfolio.finish(end)
                    result["concentration"] = concentration(capture.trades)
                    result["max_holding_seconds"] = max(
                        ((t["exit_utc"] - t["entry_utc"]).total_seconds() for t in capture.trades),
                        default=0,
                    )
                    result["folds"] = fold_records
                    result["recent_behavior"] = (
                        "last synthetic chronological fold; no market decay inference"
                    )
                    write_new_json(path / "final_state_V001.json", portfolio.state())
                result.update(
                    {
                        "name": name,
                        "scenario": scenario,
                        "method": method,
                        "universe": universe,
                        "classification": "software_correctness",
                        "diversification": "not established; common XAUUSD exposure",
                        "selection": "all declared configurations retained; no winner selected",
                    }
                )
                write_new_json(path / "summary.json", result)
                return result

            result = ledger.call(
                "synthetic_portfolio",
                {"name": name, "universe": universe, "method": method, "scenario": scenario},
                evaluate,
            )
            summaries.append({"name": name, "scenario": scenario, "result": result})
            if result is None:
                raise RuntimeError("declared portfolio configuration failed; retained in ledger")
            check_budget()
    write_new_json(directory / "comparisons_V001.json", summaries)
    return {
        "synthetic_comparisons": len(summaries),
        "risk_inputs": training_identity,
        "result": "NO_ELIGIBLE_ALPHAS",
        "market_candidates_eligible": [],
        "interpretation": "synthetic infrastructure only; no selected trading strategy",
    }
