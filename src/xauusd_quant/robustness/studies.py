"""Bounded synthetic discovery controls, fold-local perturbations and engine stress."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from ..execution.config import ExecutionConfig, content_hash
from ..execution.engine import BarClose, Quote
from ..execution.io import write_new_json
from ..execution.policy import Forecast
from ..strategy_validation.pipeline import Selection, TrainingRow, fit_fold, select_features
from ..strategy_validation.plan import Candidate, ExperimentPlan, Fold, Scenario
from ..strategy_validation.runs import TrialLedger, execute_trial
from ..strategy_validation.source import SyntheticSource
from .plan import RobustnessPlan
from .reports import concentration, read_records
from .resampling import bootstrap
from .stress import OperationalScenario, matched_cost_check, monte_carlo_scenarios


def synthetic_fixture(
    seed: int, relationship: float
) -> tuple[
    ExperimentPlan, ExecutionConfig, SyntheticSource, Callable[[ExecutionConfig], Iterable[Quote]]
]:
    """IID innovations (null) or AR(.5) increments; same construction/fitting decisions."""
    origin = datetime(2021, 5, 1, tzinfo=UTC)
    times = [origin + timedelta(seconds=300 * i) for i in range(951)]
    rng = np.random.default_rng(seed)
    innovations = rng.normal(0, 0.0005, len(times))
    returns = np.zeros(len(times))
    for i in range(1, len(times)):
        returns[i] = relationship * returns[i - 1] + innovations[i]
    prices = list(2000 * np.exp(np.cumsum(returns)))
    fold = Fold(
        "F01",
        times[0].isoformat(),
        times[700].isoformat(),
        times[940].isoformat(),
        (
            (times[200].isoformat(), times[250].isoformat()),
            (times[400].isoformat(), times[450].isoformat()),
        ),
    )
    plan = ExperimentPlan(
        "ROBUST_SYNTH_V001",
        "synthetic conditional discovery control",
        "5m",
        1,
        (fold,),
        (Candidate("RIDGE", "ridge", "fixed causal return pool"),),
        (Scenario("BASE"),),
        policy_thresholds=(0.0, 1e-5),
        evaluation_history_classification="software_correctness",
    )
    config = ExecutionConfig(max_gap_ms=300000, max_quote_age_ms=1000)
    source = SyntheticSource(times, prices, config, 1)

    def quotes(execution: ExecutionConfig) -> Iterable[Quote]:
        del execution
        sequence = 0
        for i in range(699, 940):
            # At open, only the preceding close is available. No future close quote.
            midpoint = prices[i - 1]
            for offset in (0, 1, 3, 299):
                at = times[i] + timedelta(seconds=offset)
                if plan.start <= at < plan.end:
                    yield Quote(
                        at, at.replace(tzinfo=None), midpoint - 0.1, midpoint + 0.1, sequence
                    )
                    sequence += 1

    return plan, config, source, quotes


def without_momentum(rows: Sequence[TrainingRow], count: int) -> Selection:
    chosen = select_features(rows, 3)
    return replace(chosen, names=tuple(n for n in chosen.names if n != "momentum_3")[:count])


def fit_predictions(
    plan: ExperimentPlan,
    source: SyntheticSource,
    ledger: TrialLedger,
    *,
    drop_momentum: bool = False,
) -> tuple[list[Forecast], list[BarClose], dict[str, Any]]:
    fold = plan.folds[0]
    selector = without_momentum if drop_momentum else select_features
    fit = ledger.call(
        "synthetic_outer_fit",
        {
            "seed_identity": source.identity,
            "alpha": plan.ridge_alpha,
            "drop_momentum": drop_momentum,
            "fold": fold.fold_id,
        },
        lambda: fit_fold(source, plan, fold, plan.candidates[0], ledger.call, selector=selector),
    )
    if fit is None:
        raise RuntimeError("synthetic fit failed; preserve ledger and stop")
    rows, closes = source.evaluation(plan.start, plan.end)
    forecasts = [
        Forecast.from_bar(
            forecast_id=r.row_id,
            bar_open_utc=r.bar_open_utc,
            bar_index=r.bar_index,
            value=fit.predict(r, prediction_at=r.available_at_utc),
            config=source.execution,
            provenance_id=content_hash(fit.resolved()),
        )
        for r in rows
    ]
    position = {t: i for i, t in enumerate(source.opens)}
    effects = []
    for r, f in zip(rows, forecasts, strict=True):
        index = position[r.bar_open_utc]
        if (
            index + 1 < len(source.opens)
            and source.opens[index + 1] + timedelta(seconds=300) < plan.end
        ):
            actual = math.log(source.closes[index + 1] / source.closes[index])
            assert f.value is not None
            effects.append(actual**2 - (actual - f.value) ** 2)
    return (
        forecasts,
        closes,
        {
            "fit": fit.resolved(),
            "effects": effects,
            "source_accesses": source.accesses,
            "selection_steps": [
                "causal feature construction",
                "label maturity purge",
                "inner feature identity ranking",
                "inner count MSE choice",
                "training scaling",
                "outer fit",
                "fixed sign policy Stage12",
            ],
        },
    )


def synthetic_studies(
    plan: RobustnessPlan, directory: Path, ledger: TrialLedger, check_budget: Callable[[], None]
) -> dict[str, Any]:
    controls = []
    representative = None
    for relationship in (0.0, 0.5):
        for i in range(plan.discovery_replicates):
            seed = plan.seed + i
            design, execution, source, quotes = synthetic_fixture(seed, relationship)
            forecasts, closes, record = fit_predictions(design, source, ledger)
            replay = ledger.call(
                "synthetic_null_policy",
                {"relationship": relationship, "seed": seed},
                partial(
                    execute_trial,
                    design,
                    execution,
                    Scenario("BASE"),
                    forecasts,
                    closes,
                    quotes,
                    0.0,
                    "forecast",
                    directory / f"control_{relationship}_{i:03}",
                ),
            )
            effects = record.pop("effects")
            record.update(
                relationship=relationship,
                seed=seed,
                mean_loss_improvement=float(np.mean(effects)),
                economic=replay["aggregate"] if replay else None,
            )
            controls.append(record)
            if relationship == 0.5 and i == 0:
                representative = (design, execution, source, quotes, forecasts, closes)
            check_budget()
    write_new_json(
        directory / "null_controls.json",
        {
            "records": controls,
            "scope": "synthetic pipeline includes registered adaptive steps; not matched market controls or historical-search reconstruction",
            "finite_monte_carlo": [
                {
                    "relationship": rho,
                    "replicates": plan.discovery_replicates,
                    "positive_effect_fraction": float(
                        np.mean(
                            [
                                r["mean_loss_improvement"] > 0
                                for r in controls
                                if r["relationship"] == rho
                            ]
                        )
                    ),
                    "maximum_binomial_standard_error": 0.5 / math.sqrt(plan.discovery_replicates),
                }
                for rho in (0.0, 0.5)
            ],
        },
    )
    assert representative is not None
    design, execution, source, quotes, forecasts, closes = representative
    perturbations = []
    for alpha, ablation in ((0.5, False), (1.0, False), (2.0, False), (1.0, True)):
        fresh = SyntheticSource(source.opens, source.closes, execution, 1)
        _, _, record = fit_predictions(
            replace(design, ridge_alpha=alpha), fresh, ledger, drop_momentum=ablation
        )
        record.update(
            alpha=alpha,
            drop_momentum=ablation,
            mean_loss_improvement=float(np.mean(record.pop("effects"))),
        )
        perturbations.append(record)
        check_budget()
    write_new_json(
        directory / "perturbations.json",
        {
            "records": perturbations,
            "scope": "whole registered neighborhood; fold-local refits; no profitable neighbor selection; econometric market refit perturbations blocked",
        },
    )
    scenarios = [
        OperationalScenario("BASE"),
        OperationalScenario("COMMISSION", commission_multiplier=2),
        OperationalScenario("SLIPPAGE", extra_slippage=0.02),
        OperationalScenario("LATENCY", latency_ms=2000),
        OperationalScenario("SPREAD", spread_multiplier=2),
        OperationalScenario("FUNDING", financing_extra=2),
        OperationalScenario("MISSED", miss_probability=0.1, seed=plan.seed),
        OperationalScenario(
            "FEED_INTERRUPTION",
            interruption_start=design.start + timedelta(seconds=300),
            interruption_end=design.start + timedelta(seconds=330),
        ),
    ]
    stressed: list[dict[str, Any]] = []
    mc_results: list[dict[str, Any]] = []
    base_trades: list[dict[str, Any]] = []
    for i, scenario in enumerate(
        [*scenarios, *monte_carlo_scenarios(plan.monte_carlo_paths, plan.seed)]
    ):
        adjusted = scenario.configuration(execution)
        target = directory / f"stress_{i:03}"

        def operation(
            scenario: OperationalScenario = scenario,
            adjusted: ExecutionConfig = adjusted,
            target: Path = target,
        ) -> dict[str, Any]:
            return execute_trial(
                design,
                adjusted,
                Scenario(scenario.scenario_id),
                scenario.forecasts(forecasts),
                closes,
                lambda c: scenario.quotes(quotes(c)),
                0.0,
                "forecast",
                target,
            )

        replay = ledger.call("execution_scenario", scenario.resolved(), operation)
        row: dict[str, Any] = {"scenario": scenario.resolved(), "result": replay}
        if replay:
            trades = (
                read_records(target / "trades.jsonl", plan.maximum_records)
                if (target / "trades.jsonl").exists()
                else []
            )
            if i == 0:
                base_trades = trades
            row["matched_cost_diagnostic"] = matched_cost_check(
                base_trades, trades, execution, adjusted
            )
            row["concentration"] = concentration(trades)
        (stressed if i < len(scenarios) else mc_results).append(row)
        check_budget()
    # Registered threshold neighbor repeats decisions, not an adjustment to final PnL.
    threshold = ledger.call(
        "policy_perturbation",
        {"threshold": 1e-5},
        lambda: execute_trial(
            design,
            execution,
            Scenario("BASE"),
            forecasts,
            closes,
            quotes,
            1e-5,
            "forecast",
            directory / "threshold_neighbor",
        ),
    )
    write_new_json(
        directory / "execution_stress.json",
        {
            "records": stressed,
            "threshold_neighbor": threshold,
            "scope": "same Stage12 accounting; hypothetical paths; different policies/fills can change population and aggregate cost monotonicity",
        },
    )
    nets = [
        r["result"]["aggregate"]["marked_equity_change_account"]
        for r in mc_results
        if r["result"] and r["result"]["aggregate"]["marked_equity_change_account"] is not None
    ]
    write_new_json(
        directory / "monte_carlo.json",
        {
            "records": mc_results,
            "marked_pnl_quantiles": list(np.quantile(nets, [0.05, 0.5, 0.95])) if nets else None,
            "scope": "conditional scenario model on this synthetic quote path, not future forecast; no trade-order shuffle",
        },
    )
    # Known positive dependence versus singleton blocks is an explicitly synthetic diagnostic.
    rng = np.random.default_rng(plan.seed)
    correlated = np.zeros(2400)
    for i in range(1, len(correlated)):
        correlated[i] = 0.8 * correlated[i - 1] + rng.normal()
    times = [datetime(2021, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(len(correlated))]
    reports = [
        bootstrap(
            correlated,
            times,
            ["synthetic"] * len(times),
            seconds=3600,
            horizon=1,
            block=b,
            replicates=plan.bootstrap_replicates,
            seed=plan.seed,
            minimum_days=5,
            minimum_rows=200,
            minimum_blocks=8,
        )
        for b in (1, 4, 12, 24)
    ]
    write_new_json(
        directory / "bootstrap.json",
        {
            "records": reports,
            "singleton_role": "diagnostic showing IID sensitivity, not allowed market inference",
        },
    )
    if plan.econometric_controls:
        from .econometric_controls import controls as econometric_controls

        write_new_json(
            directory / "econometric_sensitivity.json", econometric_controls(plan, ledger)
        )
        check_budget()
    return {
        "candidate_verdicts": {
            "synthetic": {
                "status": "INCONCLUSIVE",
                "reason": "software/scenario checks, no market candidate evidence",
            }
        },
        "stage15_candidates": [],
        "reserved_outcomes_read": False,
    }
