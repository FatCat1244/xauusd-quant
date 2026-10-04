# AGENTS.md - working rules for coding agents (Codex and others)

This repository is a research codebase for XAUUSD (gold) tick data. Read
`CLAUDE.md` before changing anything: it is the authoritative working notes
(scope, invariants, pitfalls, measured dataset facts, commands). This file is
the short version for an agent asked to draft or review a piece of it. Where
the two differ, `CLAUDE.md` wins.

## Scope - research forecasts and authorized offline Stages #12-#16

Prompt #16 (2026-10-04) authorizes an authoritative risk boundary, explicit unit
and sizing contracts, persistent fail-safe state, and bounded synthetic offline
replay. Stage #15 has no eligible alphas; actual operation remains inactive.
No broker, demo/live orders, deployment, commits, pushes or automatic Stage #17.
See `docs/stage16_risk_engine.md`. Earlier stopping rules are historical.

Prompt #15 (2026-10-04) authorizes bounded offline alpha registries, causal policy
combination and one shared Stage #12 account. Missing eligibility keeps the
portfolio inactive. Historical stopping rules below are superseded only for
this extension. No broker, demo/live orders, deployment, commits, pushes or
automatic Stage #16. See `docs/stage15_alpha_portfolio.md`.

Prompt #14 (2026-10-04) authorizes bounded offline robustness: immutable plans,
selection inventories, assumption-aware statistics, dependence-aware uncertainty,
synthetic discovery controls and Stage #12 execution stress. It supersedes the
earlier stop-before-#14 rules only for this work. Stop before Stage #15. No broker,
demo/live orders, deployment, commits or pushes. See `docs/stage14_robustness.md`.

- Allowed outputs: descriptive statistics, research tables, probabilities,
  expected values, calibrations, uncertainty proxies.
- Prompt #12 (2026-10-03) explicitly authorizes offline execution/backtesting,
  a separate fixed decision policy, position/cost accounting, PnL and equity.
  ML and ensemble prediction contracts remain forecasts only. No model,
  feature set or policy variant is selected by trading performance.
- Broker/MT5 connectivity, demo/live trading, deployment and neural networks
  remain prohibited. Prompt #13 authorizes offline nested strategy validation,
  small prespecified policy families, historical diagnostics and robustness.
  The historical Stage #13 stopping rule is superseded by Prompt #14 above.
  Historical 2022+ outcomes have already been inspected;
  Stage #12/#13 execution and validation commands refuse reserved access.

Prompt #13.5 (2026-10-04) adds bounded offline econometric benchmarks, causal
state estimation, delayed outcome intervals and health diagnostics. Fixed
model hypotheses can fail. Prompt #14 adds offline robustness only; broker access,
demo/live orders, deployment, commits and pushes remain prohibited. See
`docs/stage13_5_econometrics.md`.

## Invariants you must not break

1. The raw CSV is read-only. Never write to `raw_data_path`.
2. No look-ahead: a feature at bar `t` uses bars `<= t` only. Forward values
   are outcomes (`target_*`, `fwd_*`) and never enter a feature table or a
   design matrix.
3. The reserved test period (2022-01-01 onward) stays out of memory, not just
   out of the metrics. Only `ml/final_test.py` (and its ensemble counterpart)
   may read reserved rows, only for a frozen, hash-verified spec, and every
   access is logged. Nothing that selects or tunes a model may import it.
4. Walk-forward splits are chronological with purge (horizon) + embargo;
   never shuffle rows across time; never fit anything on the rows it is
   scored on.
5. Every mean-reversion / residual claim is read beside the random-walk and
   sign-flip pipeline nulls (invariant 9 in `CLAUDE.md`).
6. Frozen specs and manifests are immutable: a different content under an
   existing id must raise, never overwrite. Bump the version instead.
7. Missing is not failing: a NaN or unavailable statistic makes a verdict
   "untested", never pass or fail (`abs(nan) > x` is silently False).

## Machine and process rules

- 16 GB Windows machine, often ~4-5 GB free. Do not run the full test suite
  (~4.3 GB, ~21 min) or any `xq ... -research` study unless explicitly told
  to; run only the targeted test files for the code you touched.
- Do not edit `src/xauusd_quant/ml/{datasets,splits,preprocessing,models,
  training,calibration,explainability}.py`, `selection/*.py`, `regimes/*.py`
  or the research stage modules while a run is going: their source code is
  part of cached-result stamps, and an edit makes every cached unit stale.
- Write files as UTF-8 **without** a BOM (PowerShell 5.1 `Set-Content
  -Encoding utf8` adds one; it broke `cli.py` once).
- `pytest` runs with `-W error::DeprecationWarning` for this package: a
  deprecated call in our code fails the suite (scikit-learn 1.9:
  `LogisticRegression(l1_ratio=...)`, not `penalty=`; LightGBM 4.7:
  `fit(eval_X=..., eval_y=...)`, not `eval_set=`).
- Polars NaN is not null (`fill_nan(None)` before null filters); Polars
  `to_numpy()` may return a read-only view (copy before writing).

## Style

- Python 3.12+, `ruff check src tests scripts` and `mypy` must stay clean
  (line length 100, typed defs).
- Match the surrounding code: module docstrings that say what is computed and
  why it is causal, relative imports inside the package, `polars` + `numpy`,
  small pure functions, results written part by part so a stopped run loses
  one part.
- Tests: synthetic data (`tests/ml_synth.py` and friends), deterministic seeds.
  Where a test guards a leakage or isolation property, make sure it fails
  when the property is broken (re-break it once and watch it fail).

## Working on a task

- Change only what the task asks; report every file you touched.
- Do not commit; do not delete results or data.
- When reviewing, look first for look-ahead, leakage into 2022+, fits on
  scored rows, NaN reaching a verdict, and silent fills of missing values.
