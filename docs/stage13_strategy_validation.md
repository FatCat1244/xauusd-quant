# Stage #13: purged walk-forward strategy validation

Prompt #13 (2026-10-03) authorizes offline strategy validation using Stage #12.
Broker/MT5 connections, demo/live orders, deployment and Stage #14 are excluded.
Forecast contracts stay prediction-only. Historical research and reserved-access
logs are preserved. The committed Stage #12 implementation was inspected and its
154 targeted tests rerun successfully before implementation.

The question is: **Does a prespecified forecast-driven policy produce repeatable
after-cost economic value under chronological evaluation and declared assumptions?**
Forecast accuracy, an equity curve and a chosen outer-fold winner cannot answer it.

## Interfaces and frozen plans

The new package is `xauusd_quant.strategy_validation`. Use `xq` or
`.venv\Scripts\python.exe -m xauusd_quant.cli`. New run IDs are required; existing
directories and plans are immutable, including interrupted/failed attempts.

```powershell
xq strategy-plan --plan config/strategy_validation.yaml
xq strategy-plan --plan config/strategy_validation_15m.yaml
xq strategy-readiness --plan config/strategy_validation.yaml --run-id STRATEGY_READINESS_5M_V002
xq strategy-smoke --plan config/strategy_validation.yaml --run-id STRATEGY_SMOKE_5M_V002
xq strategy-smoke --plan config/strategy_validation_15m.yaml --run-id STRATEGY_SMOKE_15M_V002
xq strategy-validate --plan config/strategy_validation.yaml --run-id STRATEGY_5M_DIAGNOSTIC_V003 --historical-diagnostic
xq strategy-validate --plan config/strategy_validation_15m.yaml --run-id STRATEGY_15M_DIAGNOSTIC_V003 --historical-diagnostic
```

The original V001 representative designs were frozen before their Stage #13
outcomes were read. Their inner windows were Saturdays (May 15/22); the 5m
attempt recorded all 36 failed model/fitting configurations and eight completed
controls. Its plans/results remain immutable. The unexecuted 15m V001 plan is
also retained. V002 moves both inner windows back exactly two days to May 13/20
weekdays, preserving models, policies, outer periods and budgets. Both amended
plans were frozen before their corrected model economics, **after** V001 control
outcomes. This is explicitly a post-attempt revision, not original registration.
Registration is **retrospective**, after historical research and Stage #12's quote
inspection. It does not make June 2021 untouched. The current two plans share
May 2021 training, two chronological prior inner weekdays, and contiguous
outer periods June 1, 2021, 00:00-00:30 and 00:30-01:00 UTC. The only target is
expected future log mid-close return over **one subsequent observed bar**.
The identical horizon count therefore has different clock-time meaning at 5m/15m.

The bounded family contains ridge and a causal historical-mean forecast benchmark.
Ridge uses the fixed pool `return_1`, `momentum_3`, `volatility_3`; each is computed
from at most the current and three preceding completed mid bars. This pool is a
new retrospective design; analyst exposure to earlier results is documented.
No Stage #9 feature identity/count or Stage #11 ensemble universe is imported.
No tree-model search or full-history retraining is a dependency.

Only feature count (1 or 2) is selected by mean inner validation MSE, with smallest
count breaking ties. Feature identities are ranked by training-only absolute
correlation with the target. Ridge penalty is fixed at 1. Training is expanding;
an explicitly registered rolling-day option exists but is not compared here.
Models and policies are reported as a family; no model winner is selected on outer
PnL. Historical-mean feature-count trials are redundant and retained in the attempt
count; its fitted prediction actually uses zero features.

Fixed policy thresholds are 0 and neighboring 0.00001 absolute log-return units.
They are separate variants, with zero the primary sign policy. They are never
chosen from outer economics. Four scenarios are base hypothetical Stage #12 terms,
double commission/slippage, 500ms latency, and 1.5x spread. Widening is centered on
each observed midpoint, preserves timestamp/sequence and leaves invalid inputs
invalid. It is a counterfactual stress, not another broker's measured execution.
All cost/funding/contract/currency units retain Stage #12 semantics.

There are 24 economic variants (two models x two thresholds x four scenarios,
plus two controls x four scenarios), and 20 nested fitting attempts: **44 planned
attempts per timeframe**, within a frozen budget of 64. Failed trials consume
budget and remain in the ledger. No disappointing result increases the search.
Historical prior search exposure is unknown in total; existing inspections are
recorded. Synthetic fixture trials are separate from historical searches.

`ExperimentPlan`, `Fold`, `Candidate`, `Scenario`, `freeze_plan`, `fit_fold` and
`run_framework` provide programmatic interfaces. Plans serialize resolved design
and execution assumptions, identities, question, eligible target/units, inner and
outer schedules, training policy, primary metric, evidence rules, stopping rule,
history classification, revision reason and seeds. A changed plan or execution
configuration under the same plan ID raises. New designs need a new plan version,
`supersedes_plan` and a reason, especially after outcome inspection.

## Information intervals and the nested path

Each outer fold's start is its information cutoff. Training feature values are
available at bar close; a label spans that close through the h-th subsequent
observed bar close. Label publication may be later. Both interval end and label
publication must be **strictly before** the fitting cutoff. Crossing labels and
labels published at the boundary are purged. This accounts for overlapping target
windows and closures, without replacing an observed-bar horizon with minutes.

Each inner split repeats feature selection, preprocessing and model fitting on
its own purged preceding data. Inner validation labels are fully observed before
inner end, itself before outer start. No call from inner selection requests outer
outcomes. The final outer model repeats selection/preprocessing/fitting using the
chosen count and all purged permitted training rows. Calibration is the declared
identity operation; model penalty and policy thresholds are fixed. Ensemble
membership, weights and screening are not fitted in this restricted new path.
External adaptive/ensemble audit records, if supplied, must prove every Stage #12
adaptive choice precedes its fold; later or missing choices fail/remain unknown.

No future-side training exists, so embargo is zero. The strict information-end
cut supplies the separation this split structure needs. An arbitrary percentage
embargo is not used. This does not change earlier Stage #10 purge/embargo semantics.

Development bar scans use broker-local and derived UTC predicates **before
collecting closing values**. A collection limit enforces the registered row cap
before allocating an unbounded frame. The current bar/tick lineage, convention,
configuration fingerprint, actual partition row counts/sizes and file coverage
are checked. Footer checks do not reverify all historical content digests.
Training labels and evaluation feature-only interfaces are separate. Evaluation
features are causal rolling values; future label values are never attached to
them. Synthetic append tests verify earlier frozen models and features stay fixed.

Every per-fold specification is written before that fold's evaluation features
are scored. It includes selected identities, actual count, fitted means/scales,
coefficients/intercept, label-information maximum, training count and training-ID
hash, inner trials and cutoff. Forecast records carry the spec-file SHA-256.
This is offline historical reconstruction: later-fold training can use fully
available earlier-fold labels, and fitting may be calculated ahead during replay.
As-of checks enforce the simulated chronology; no wall-clock prospective fit is
claimed. Historical creation paths remain recorded with their limitations.

## Policies, benchmarks and execution across folds

Policies consume raw expected log-return forecasts separately. Above a frozen
absolute threshold, sign supplies direction; otherwise abstain. No direction is
inferred from a volatility forecast or residual score. Other targets are excluded.
Raw predictions and policy consumption are both recorded. Invalid availability,
units, status or nonfinite predictions remain explicit rejections.

All evaluations use `ExecutionEngine`, observed quote sides, computation/order
delay, strict subsequent-quote fills, expiry, stale/invalid quote handling,
commission/slippage and declared continuous funding. Quantity is fixed lots;
initial cash is fixed, without resetting or reinvesting across folds. The engine
allows one net position and rejects overlaps; no pyramiding/reversal exposure.
Overnight/weekend exposure is financed. Stops, limits, partial fills, margin and
production risk management remain outside this stage.

One engine carries pending orders, positions, clocks and cash from the first fold
through the last. `ExecutionEngine.checkpoint` advances to the left boundary
without finishing the run. Events exactly at the boundary belong to the following
fold. An old position exits on its original observed-bar horizon; the new fold
cannot retrospectively close it or replace its policy. New signals during old
exposure are rejected. At final cutoff pending orders are unfilled, exposure is
retained and liquidation marking follows Stage #12. No curve concatenation or
hidden reset occurs. Fold equity/cash increments and closed-by-exit accounting
are distinct; a trade can begin in one fold and close in another.

Benchmarks are no trading and a fixed-seed random direction held for 16 decision
opportunities per random block. The historical mean is a model-free causal
forecast benchmark within the candidate family. Controls use the same valid
feature-row opportunities, horizon, size and engine, but trade populations/exposure
can differ. Individual observations are not shuffled. The random control tests a
narrow directional diagnostic, not the entire discovery process. It neither
reselects models/features nor replaces random-walk/sign-flip pipeline null evidence.
No best seed or benchmark is selected afterward.

## Metrics, uncertainty and verdicts

Primary metric: account-currency **marked-equity change over the declared outer
interval**. Return is that change divided by fixed initial cash. Complete UTC-day
increments use the same denominator; there is no annualization or Sharpe.
Liquidation marks exclude prospective exit fees/slippage exactly as Stage #12
declares. Unknown/stale open marks stay unknown. Final open exposure prevents
passing economic criteria.

Each fold and aggregate report gross/closed net price economics, cost components,
cash/equity changes, orders/fills/rejections/expiry, closed trades, turnover and
lot-second exposure. Gross spread/slippage costs already enter quote-side PnL
and are not subtracted twice. Fold drawdown is tracked across all known actual
quote/boundary marks; aggregate drawdown is the continuous running measure.
Unknown intervals limit coverage. Duplicate closed-position IDs or inconsistent
boundary marks raise. Daily marks, chronological fold variation, exit-period
attribution and the full cost/latency/spread/threshold family are retained.

Concentration reports the contribution of the largest three positive closed
trades and the largest positive fold, and closed net without those three trades.
These descriptive removals do not turn trades into independent samples.

A noncircular moving-block bootstrap uses a prespecified five complete UTC-day
block and 199 replicates, seed 130013. At least 20 complete daily observations
and four blocks are needed. Missing marks/calendar spacing are not imputed;
holdings at least as long as the block prevent an interval. Five days is a declared
dependence sensitivity, not an empirically optimal block or proof that all dependence
was captured. The interval conditions on the frozen policy and fitted sequence;
it does not refit the whole selection process or correct unknown earlier searches.
No decorative multiple-testing statistic is produced.

Evidence statuses:

| Status | Meaning |
|---|---|
| BLOCKED | Missing/failed scientific, identity, specification or trial inputs |
| INCONCLUSIVE | Software-only evidence, insufficient folds/trades/days, unknown marks or unresolved exposure |
| REJECTED | Adequate permitted evidence fails one or more frozen criteria |
| PASSES DECLARED HISTORICAL CRITERIA | Meets the entire frozen family within retrospective scope and assumptions; no prospective-profitability or trading claim |

Passing requires gates, at least three folds, 30 closed trades, complete marks,
positive net equity, at least 60% positive folds, positive lower conditional interval,
largest-three trade share <=50%, largest-fold share <=75%, positive closed net after
removing the three largest positive trades, and positive equity for **all** registered
policy neighbors/scenarios. Insufficient evidence is inconclusive. The representative
one-hour/two-fold designs cannot meet these minima, regardless of favorable PnL.

Stage #12 readiness/promotion semantics are unchanged. Stage #13 separately requires
an honestly declared historical classification; passing historical criteria never
requires or establishes that inspected history is prospective. No prospective mode
is implemented, and all real intervals remain before broker-local 2022-01-01.
Existing reserved loaders, repeat reasons and interrupted-attempt guards are intact.

`--evidence` may point to a separately reviewed JSON audit: `plan_sha256` (the
frozen artifact's content hash), `dataset_version`, current `pipeline_source_identity`,
repository-relative `files` mapped to SHA-256, and `folds` keyed by fold ID. Each
fold needs random_walk/sign_flip records with `status`, preceding `as_of_utc`,
verified `file` and matching `plan_sha256`. Optional `external_adaptive_choices`
must include all Stage #12 adaptive as-of fields. An execution-specification record
needs verified `file` and `execution_sha256`, plus verified-supplied execution config.
Hashes prove identity, not scientific truth or contract terms. Missing inputs remain
unknown. No matching null or verified broker/account evidence is currently supplied.
Overriding readiness programmatically is restricted to declared synthetic fixtures.

## Artifacts and actual verification

Plans live in `results/strategy_validation/plans`; runs live in versioned directories
under `results/strategy_validation/runs`. A run preserves preflight requests and
failures, readiness, manifest, source identity, fold specifications, raw predictions,
attempt/completion/failure ledger, every trial's Stage #12 records and fold metrics,
aggregate summaries, robustness family and verdict. Invariant/accounting failures
abort instead of becoming a passing partial result. Source/config/data/spec/policy
and assumption identities are recorded, including uncommitted source hashes.

Actual measurements, tests, guard mutations, diagnostic outcomes and remaining
blockers are recorded in `WORKLOG.md`. Stage #12's previous reports are preserved;
earlier Stage #9-#11 accuracy/ensemble findings are not reproduced or relabelled as
clean evidence. Historical selected manifests/universes still need a new fold-local
creation path before reuse; this small new pipeline does not repair them implicitly.


## Bounded diagnostic outcome (2026-10-03)

Both amended V002 real runs completed all 44 declared attempts. All 16 forecast
variants per timeframe lost money in the one-hour June 2021 interval; no-trading
controls earned zero. Primary zero-threshold/base hypothetical results:

| Path | Closed trades | Gross executable PnL USD | Net USD | Evidence status |
|---|---:|---:|---:|---|
| 5m ridge | 6 | -2.5430 | -2.903416 | BLOCKED |
| 5m historical mean | 6 | -2.8620 | -3.222416 | BLOCKED |
| 15m ridge | 2 | -1.6970 | -1.817416 | BLOCKED |
| 15m historical mean | 2 | -1.6970 | -1.817416 | BLOCKED |

Sizing was 0.01 hypothetical lots x 100 oz/lot, initial USD 10,000. Spread and
adverse slippage are already included in gross executable PnL; commissions and
declared financing bridge gross to net. These are hypothetical terms, not broker
profit estimates. Both paths lack matching prior pipeline null evidence and
verified execution specifications. Even with these gates supplied, two folds,
6/2 trades and zero complete daily observations would be insufficient. No numeric
uncertainty interval or full-history economic conclusion was produced.

The actual corrected 5m/15m wall times were 41.78/39.78 seconds; peak own-process
working sets were 125.96/117.75 MiB and private memory 450.08/430.59 MiB. Full
history can require more resources. 186 focused tests passed; all 26 guard
mutations were detected, original source unchanged. Saved records passed
chronology/spec-identity/accounting checks across 48 economic trials.

See the Stage #13 WORKLOG entry and local immutable outputs
`results/strategy_validation/DIAGNOSTIC_SUMMARY_V001.json` and
`FINAL_VERIFICATION_V001.json` for measurements, all variants, known historical
attempt counts (132, including 36 preserved failures), exact commands and the
post-attempt V002 revision. Existing Stage #9-#11 results were not reproduced.
CLI examples above use unused diagnostic IDs; choose a new ID for every run.
