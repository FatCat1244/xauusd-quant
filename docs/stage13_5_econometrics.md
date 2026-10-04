# Stage #13.5: bounded econometrics, causal adaptation and uncertainty

This is the preserved Stage #13.5 record. Prompt #14 supersedes its earlier
stopping rule only for [bounded offline robustness](stage14_robustness.md).

Prompt #13.5 authorizes offline implementation/research only. Stop before Stage
#14. No broker, demo/live orders, deployment, commit or push. Forecast contracts
and earlier results remain unchanged. Nothing here establishes profitability.

The extension asks whether interpretable econometrics, recursive state and error
measurements add information beyond aligned simple forecasts. A failed hypothesis
is retained. It reuses Stage #13 chronological schedules, ridge selection, label
purge, trial ledger and metadata readiness, plus Stage #12 immutable records,
code identities, resource measurements, timestamps and reserved-period checks.
Existing ADF/KPSS, Ljung-Box, ARCH-LM and detrended random-walk routines are reused.
No cache-stamped feature/ML/selection/regime modules are changed.

## Plan, scope and CLI

```powershell
xq econometric-plan --plan config/econometrics.yaml
xq econometric-plan --plan config/econometrics_15m.yaml
xq econometric-readiness --plan config/econometrics.yaml --run-id ECON_READY_5M_V001
xq econometric-smoke --plan config/econometrics.yaml --run-id ECON_SMOKE_5M_V001
xq econometric-evaluate --plan config/econometrics.yaml --run-id ECON_5M_DIAGNOSTIC_V001
xq econometric-evaluate --plan config/econometrics_15m.yaml --run-id ECON_15M_DIAGNOSTIC_V001
```

Equivalent entry: `.venv\Scripts\python.exe -m xauusd_quant.cli`. Every run uses a
new versioned ID. Plans are immutable/idempotent for identical resolved content;
changed design/assumptions require a new version and revision reason. Readiness
can inspect metadata without reading regression values or new evaluation prices.
Missing inputs retain a preflight failure and block dependent conclusions.

The initial designs fit expanding May 2021 data and evaluate three contiguous
eight-hour folds on June 1 UTC. Model fitting ends 24 hours before each outer
start, leaving a separate preceding calibration holdout. Only the preserved
ridge reference selects feature count 1/2 via May 13/20 inner weekdays, both
preceding that holdout. The added models have fixed identities/specifications;
no model/lag/distribution/policy search. Eight model/benchmark fits, four interval
calibrations and one health initialization per fold, plus twelve inner fits,
give **51 declared attempts** within budget 64. Every attempted fit/calibration,
including failures, is retained. Negative comparisons do not enlarge the family.

Resource limits: 20,000 permitted bars per bounded scan, 240 seconds per path,
1 GiB own-process private memory at checked fold boundaries, optimizer cap 200
iterations, no raw-tick econometric fit or concurrent heavy study. The 15m design
may run after the 5m software, chronology/output and resource checks, regardless
of performance sign. Reports give actual measurements, not a memory guarantee.
The wall limit is checked between operations/folds; it cannot interrupt an
individual library call. Reader state is carried through 64-row replay batches.

All history is retrospective. Stage #9 global identities/counts and Stage #11
eligibility/universe/correlation/null decisions remain unsuitable for early
fold-local reuse. Their historical results are preserved, not repaired by this
extension. 2022+ was inspected; it is not untouched and the guarded source refuses
it. No reserved-period reader or repeat-reason guard is weakened.

The representative designs cover one evaluation day and cannot satisfy the
minimum five-day forecast evidence rule. These are infrastructure/effect-size
diagnostics, not a claimed full-history test. Existing boosted-model/ensemble
accuracy findings are not reproduced: their chronology/target units/horizons do
not permit an honest matched comparison here. The comparison with existing
models is specifically a new aligned refit of Stage #13's restricted ridge path.

## Measurements and target alignment

Bar timestamps are opening times. A close becomes available at open + 5m/15m;
forecasts add the declared computation delay, which must be shorter than one
grid interval. Source broker-local timestamps are converted by the existing
documented New York + seven-hour convention, including source DST rules.

Use completed midpoint closes on the existing regular grid, not every tick.
Invalid/nonpositive closes reject; no interpolation, resampling across closures,
or fabricated returns. A return needs consecutive grid bars. Feature windows
need twelve contiguous returns and therefore thirteen closes; after a gap there
is a new warm-up. Future targets need every intervening grid close; missing or
boundary outcomes are explicitly unscored/censored. Forecasts are still emitted
without looking at whether future outcomes will exist.

| Object | Meaning / units |
|---|---|
| Price/log price | Midpoint USD/oz or natural log thereof; not return/PnL |
| Future return | Sum of next h grid log-close returns; `log_mid_return` |
| Realized variance | Sum of squared next h grid log returns; `log_return_squared` |
| Conditional variance | Model's conditional forecast of future squared-return sum under stated mean/innovation assumptions |
| Standard deviation | Square root of variance, a different quantity; no directional signal |
| Forecast error | Realized target minus its preceding prediction; usable only after publication |
| Regression residual | Descriptive fitted-error diagnostic, not executable price profit |
| Economic PnL | Stage #12 quote-side cash flow; not inferred from forecast MSE or a log interval |

Target horizon h defaults to one and may be 1..12 contiguous grid intervals.
Identical h at 5m and 15m has different clock meaning. Target publication is the
last required close plus configured outcome delay. Fits require both interval
end and publication **strictly before** cutoff. Return fits reuse Stage #13's
information-interval purge. No future-side fit exists; extra embargo is zero.

Training-only measurement diagnostics report within-segment lag-1 correlation
and the ratio of RV from disjoint three-grid aggregated returns to fine-grid RV.
Coarse groups are anchored at each contiguous segment, never chosen by outcomes.
They investigate sensitivity to quote noise without claiming to identify noise
alone. Existing early-feed noise findings remain limitations, not rerun claims.
Observed Bid/Ask stays intact in the existing tick store for eligible economics.

## Fixed model equations and assumptions

**ARX:** an OLS intercept, last two completed returns, and square root of mean
past twelve squared returns predict the future h-grid log return. Means/scales
are fitted on preceding training only. Fixed predictors, no price-level R²
claim. Coefficients and chronological raw-unit summaries are predictive
associations, never causal effects.

**Reference forecasts:** zero future return and the Stage #13 ridge family.
The ridge selector, transforms and count choice repeat inside its chronological
inner splits. Its training labels are aligned to this extension's contiguous
grid targets; this is a new refit, not reuse of an old selected manifest.

**GARCH(1,1):** conditional mean zero, `h[t+1] = omega + alpha*r[t]^2 + beta*h[t]`.
Gaussian quasi-likelihood uses returns x10,000 for numerical scale, then restores
variance units. `omega>0`, `alpha,beta>=0`, sum <=0.995. Stationary variance
initializes each contiguous training segment. One fixed optimizer start, SLSQP,
declared iteration cap. Nonconvergence, insufficient/degenerate history, invalid
variance, persistence at the constraint or implausibly large unconditional
variance explicitly fails. There is no replacement forecast or favorable floor.
Parameter diagnostics retain convergence and persistence.

For h>1, expected subsequent variances recur with `alpha+beta` and are summed.
This assumes zero conditional mean and uncorrelated innovations; it is not a
forecast of arithmetic price profit or volatility-based direction. Rolling
twelve squared returns and fixed EWMA lambda 0.94 are variance benchmarks.
EWMA multi-step expected variance is held constant; gaps reset declared state.

**HAR-style:** nonnegative least squares of future RV on an intercept and mean
past squared returns over 1/3/12 contiguous grid intervals. These are intraday
short/medium/long components, **not** conventional daily/weekly/monthly HAR.
At 5m they mean 5/15/60 minutes; at 15m, 15/45/180 minutes. Fit in arithmetic
variance levels, with numerical response scaling restored afterward. No log
transformation/exponentiation/Jensen correction. A nonpositive/invalid prediction
rejects; it is not silently clipped. Ordinary unconstrained OLS/HAC inference is
not applied to NNLS coefficients.

**Local-level Kalman:** `x[t]=x[t-1]+w[t]`, `y[t]=x[t]+v[t]` on log midpoint.
Independent zero-mean Gaussian process/observation noise is the model assumption.
Pre-holdout Gaussian likelihood estimates positive q/r via L-BFGS-B. Bounds are
declared relative to preceding difference second moment; boundary estimates
are flagged for weak identification. Initialization conditions on the first
segment observation, covariance r; gaps reset to the new observed log price.
Joseph covariance update maintains positive stable covariance. Only filtering
is implemented, no full-sample smoothing or future observations.

The forecast is filtered expected future log observation minus current observed
log price under a no-drift local level. This mapping is specific to that model,
not a claim that the latent level is true fair value. Initial training-state
replay conditions the fit for subsequent inference; it is not labeled historical
out-of-sample state. Emitted filtered states begin after parameter fitting.
Chunking and deterministic JSON reload retain state without resetting at a batch.

## Econometric diagnostics and coefficient uncertainty

Reuse existing ADF (unit-root null), KPSS (stationarity null), Ljung-Box (specified
lags have zero autocorrelation) and ARCH-LM (no lagged conditional-variance
dependence). Tests operate on the last <=512 observations of the longest
contiguous preceding segment, never sparse subsampling or gap concatenation.
In-sample ARX residuals and standardized GARCH innovations are explicitly
descriptive diagnostics. Undefined/constant/insufficient series remain untested.
Failure to reject is not proof of a null; stationarity is not economic reversion.
The existing detrended random-walk control is retained in synthetic reports;
mechanical detrending is not presented as a market mechanism.

ARX uses Bartlett Newey-West score covariance with lags max(3,h-1), finite-sample
correction and actual timestamp-separated score pairs. Closure gaps cannot
manufacture adjacent covariance pairs. Tests compare the regular-grid case with
the installed statsmodels API. `hac_standard_errors` applies to the intercept
and **standardized design coefficients**; raw-unit coefficient summaries are
separately named. This is conditional fixed-regression parameter uncertainty,
not an outcome interval, causal effect, leakage repair or multiple-testing fix.
No automatic stability test chooses the most favorable coefficient period.

## Delayed outcome prediction intervals

Separate sidecars wrap ARX and the ridge reference. Their raw forecast contracts
stay unchanged. Rolling absolute residuals use a bounded 128-observation deque,
minimum 32, target coverage 80%. Quantile rank is `ceil((n+1)*(1-alpha))`; a rank
beyond available observations is uncalibrated, not silently capped. The symmetric
range is forecast +/- that empirical quantile. State records window, count,
publication cutoff, pending forecasts, coverage, widths and actual updates.

One additional fixed variant is **bounded ACI-inspired adaptation**:
`alpha <- clip(alpha + .01*(.2-miss), .01, .5)` after a matured scored interval.
It uses the same residual window; no alternate gamma/seed/window search. Clipping,
financial dependence and short coverage mean **no exchangeable-data or original
unmodified ACI theorem is claimed**. [Gibbs and Candès' original method](https://arxiv.org/abs/2106.00170)
motivates the scalar update; this bounded delayed implementation reports empirical
coverage only.

A forecast enters a pending queue without its outcome value. Deliveries require
actual target maturity strictly before update and before the next decision.
Overlapping horizons do not accelerate publication. Unknown gap/end outcomes are
expired/censored explicitly without error imputation. Update and decision clocks
cannot move backward. Calibration state is frozen before outer values are opened.
Online interval updates may use newly matured outer errors under the fixed rule;
they never refit predictive model coefficients or change a strategy.

Artifacts label the range `future_outcome_prediction_interval`. It is neither a
confidence interval for a conditional mean, a parameter interval nor a measure
of all model uncertainty. Its lower endpoint is not a conservative expected-return
bound and cannot justify a supposedly profitable trade.

## Health monitoring and forecast comparison

Two-sided Page CUSUM monitors matured ARX errors standardized by preceding holdout
RMSE, fixed zero reference, drift 0.5 and threshold 8. Initialization follows
adequate preceding calibration; missing/degenerate scale is untested. Constant
memory, one update per matured error; accumulator resets after alarm. Fixed
Gaussian unchanged and shifted (+3 sigma halfway) fixtures report false alarms
and detection delay. These do not estimate market false-alarm rates or substitute
for pipeline nulls. [NIST's CUSUM description](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc323.htm)
supports the interpretation; thresholds here are declared research scenarios.
Health outputs never switch models, retrain on outer outcomes or alter risk rules.

Compare exact matching targets, horizons, observations, availability and folds.
Primary return loss is MSE, with absolute and relative improvement. Primary
variance loss is QLIKE `log(forecast variance)+RV/forecast variance`; report its
matched difference, never divide by negative QLIKE. Each added forecast is
compared with both corresponding benchmarks; the full family remains visible.
Duplicates or incompatible records raise. Failed fits/rejected forecasts and
missing targets retain their coverage limitations rather than disappearing.

A prespecified within-contiguous-fold moving-block bootstrap uses 60-minute
blocks (at least h observations), 199 replicates, seed 135013. Five distinct
evaluation days, 200 matched rows and four eligible contiguous blocks are required.
Short samples have no numerical confidence interval/p-value. The procedure
conditions on the fitted forecast sequence; it does not refit all selections or
repair unknown historical search. No Diebold-Mariano statistic is produced for
these short, nested/adaptive, dependent and potentially heavy-tailed comparisons.

Passing forecast criteria requires matching provenance/null evidence, adequate
coverage, >=1% relative MSE or >=0.01 absolute QLIKE improvement, positive lower
conditional block interval and >=60% positive folds against both benchmarks.
Statuses distinguish blocked inputs, inconclusive evidence, no demonstrated
incremental value and promising declared historical evidence. Synthetic correctness
never becomes market evidence. Interval coverage and health have their own
descriptive roles; neither substitutes for predictive or economic superiority.

## Economics, artifacts and audit history

No new economic candidate is registered in this plan. The preserved Stage #13
frozen baseline remains unchanged. Matching pipeline random-walk/sign-flip
evidence, audited provenance and verified supplied execution terms are absent;
economic comparisons remain blocked and are **not executed**. Once a separate
eligible experiment is registered, it must use Stage #12 Bid/Ask, latency, fees,
slippage/funding and reconciliation, retaining quote-side costs exactly once.
No broker fees, fill probabilities or measured execution are invented here.
Covariance allocation/CVaR/robust control remain future Stage #15/#16 subjects.

Public package: `xauusd_quant.econometrics`; immutable plans/runs under
`results/econometrics`. `EconometricPlan`, `GridData`, `RegressionFit`,
`GarchFit`, `VarianceState`, `LocalLevelFit/State`, `DelayedIntervals`,
`ErrorCUSUM`, `compare` and `evaluate` provide programmatic interfaces.
`BarDevelopmentSource.completed_bars` exposes the existing bounded guarded reader;
opening timestamps are derived UTC, close values strictly before the cutoff.

Run artifacts include request/readiness/manifest, code and source/config identities,
fold/cutoff definitions, trial ledger, immutable model specifications and hashes,
pre-outer calibration states, raw forecasts, separate uncertainty, filtered-state
and health records, outcomes/scores, gap/end censoring, fold coverage/counts,
coefficient stability, matched comparisons, final state and evidence verdict.
New namespace/versioned sidecars do not mutate ML/ensemble historical contracts.
Resource/invariant failures stop the bounded design, preserving partial records.

During testing, an initial missing-input fixture replaced `project_root` but kept
absolute real data paths. Without an explicit synthetic reader, it accessed May
4, 2021 development outcomes while labeled software correctness. Its 34 attempts
(19 completed, 15 failed) are preserved verbatim under
`results/econometrics/audits/UNINTENDED_REAL_FIXTURE_V001`, with a separate
`ACCESS_AUDIT.json` flagging the original classification. No reserved access.
The input guard now refuses any software-correctness run without `SyntheticBars`;
its deliberately disabled regression test stops at a forbidden metadata fixture,
before any real values. This exposure is recorded before registering the June
plans; it did not select/change candidate specifications. Original misleading
artifacts were not rewritten as clean evidence.

Installed APIs inspected: SciPy 1.18.1 `minimize`/`nnls`, statsmodels 0.15.0.
See [SciPy optimizer](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html),
[statsmodels HAC API](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_robustcov_results.html)
and [GARCH multi-step assumptions](https://arch.readthedocs.io/en/latest/univariate/forecasting.html).
The `arch` library is not required; the small constrained recursion uses existing
SciPy. No new dependency was installed.

Actual commands, tests, measurements, diagnostic findings and changed files are
recorded in the Stage #13.5 WORKLOG entry after the bounded runs.


## Executed bounded findings (2026-10-04)

The commands above were executed with those exact V001 IDs. Repeating a run
requires a fresh ID; existing directories raise rather than overwrite. Both
real plans were registered before these June evaluation values were read.
The 15m path followed successful saved-record/resource checks of the 5m path,
without changing specifications after seeing the 5m comparisons.

Data inventory confirmed metadata for `ticks-2e173ef8e61bd240`: 281 monthly
partitions, 729,244,369 footer rows, stored broker-local timestamps
2003-05-05 03:01:03.421 through 2026-09-18 23:59:59.079. Manifest SHA256
`da8d0bfe17a33ca95080e872ac98f8474721616309234e29ce339e2236f4318c`.
This is a metadata/identity check, **not** a fresh full-content/raw hash audit.
The raw CSV remained unchanged. Neither the old 2003-2004 partial dataset nor
the complete tick history was used as the evaluation sample.

Evaluation: June 1, 2021 UTC, three eight-hour folds, preceding May fits and
holdouts. Each of eight forecasts scored 260 matched observations at 5m and
77 at 15m. Grid gaps require fresh feature warm-up; missing/end targets are
censored. Per-model fold scored counts were 95/95/70 at 5m, 26/31/20 at 15m;
censored counts were 1/1/2 and 1/1/1 respectively. No invalid forecasts or
fitting failures in either planned real run. Both had 51 completed attempts,
including inner fits, wrapper calibrations and monitor initialization.

Positive numbers below mean lower matched loss than the stated benchmark.
Return entries are relative MSE improvements (%); variance entries are absolute
QLIKE improvements. Different quantities/horizons are separate comparisons.

| Candidate / benchmark | 5m improvement | 15m improvement |
| --- | ---: | ---: |
| ARX / zero return | -1.3198% | -4.1167% |
| ARX / aligned Stage 13 ridge | +0.1426% | -0.2998% |
| Kalman / zero return | -0.7563% | -2.1588% |
| Kalman / aligned Stage 13 ridge | +0.6979% | +1.5864% |
| GARCH / rolling variance | +0.02046 | +0.29499 |
| GARCH / EWMA variance | +0.08268 | +0.11594 |
| HAR-style / rolling variance | -0.18568 | +0.17974 |
| HAR-style / EWMA variance | -0.12347 | +0.00070 |

All four model verdicts are **INCONCLUSIVE**, promotion blocked. Return models
did not improve on zero in aggregate. GARCH's favorable aggregate variance
loss does not establish repeatability: it beat rolling in 1/3 folds at 5m
and 2/3 at 15m; relative to EWMA, 3/3 and 2/3. HAR-style was worse against
both benchmarks at 5m and mixed across 15m folds. No method earns a clean
superiority claim. All comparisons fail the five-day evidence minimum; 15m
also lacks 200 matched rows. No bootstrap interval, DM statistic or profitability
estimate was emitted. Full fold effect sizes remain in `comparisons.json`.

ARX rolling/adaptive outcome-interval coverage was 76.15%/79.23% at 5m and
75.32%/77.92% at 15m, against nominal 80%. Mean widths in log-return units
were 0.0009359/0.0010459 and 0.0014544/0.0015884. Aligned ridge coverage was
75.77%/79.23% and 75.32%/76.62%. Adaptive coverage used wider intervals; this
short sample cannot establish a superior uncertainty method. Coverage varies
substantially by fold (15m ARX rolling 76.92%/61.29%/95.00%). Keep both as
calibration benchmarks with **INCONCLUSIVE** market evidence.

CUSUM produced four and one outer-forecast error alarms at 5m/15m respectively;
these have no automatically assigned economic/structural-break meaning. The
fixed 500-observation unchanged Gaussian control had zero alarms; the +3 sigma
shift control first alarmed three matured updates after the change. Keep it
as an implemented health diagnostic; market false-alarm/detection evidence
is **INCONCLUSIVE**. Existing detrended-random-walk evidence remains separate.

| Run | Wall / CPU seconds | Peak working set bytes | Private bytes at completion |
| --- | ---: | ---: | ---: |
| 5m synthetic smoke | 21.0564 / 18.5469 | 270,352,384 | 763,867,136 |
| 5m historical diagnostic | 12.8210 / 12.6406 | 278,241,280 | 824,905,728 |
| 15m historical diagnostic | 16.2982 / 5.7344 | 255,897,600 | 793,100,288 |

These are measured own-process Windows values, including imports/preflight;
they are neither total machine memory nor guarantees for expanded studies.
Sequential paths stayed within declared checked budgets.

`FIVE_MINUTE_CHECK_V001.json` and `FINAL_VERIFICATION_V001.json` verify saved
records without reading new outcomes. Across smoke plus the two real runs:
72 frozen fits, 162 serialized states, 18,688 raw forecast records and 4,976
scored records checked for cutoffs, label maturity, availability, identities,
duplicate prevention and exact state reload. All passed. Source/config identity
matched the run manifests. No reserved outcome access or economic evaluation.

Focused verification: **271 tests passed in 33.44 seconds**, including 35 new
tests and existing Stage 12/13, reserved-period, stationarity and autocorrelation
tests. Ruff clean; mypy clean (219 source files). Mutation audit V001 detected
35 broken guards; V002 detected all **36**, including the additional calibration
update-clock guard; original source hashes unchanged. Exact commands are in
WORKLOG.md. Full test suite and research pipeline were not run.

Known historical exposure after this extension: 268 attempts (217 completed,
51 failed): prior Stage 13's 132, the preserved misclassified fixture's 34,
and these two runs' 102. Actual prior counts before the real runs were 166/217;
the immutable plans retain registration-time exposure 166. The verification
sidecar records the later actual count without rewriting the 15m manifest.
The 51 synthetic smoke attempts are separate; older research searches unknown.

Economics remains **BLOCKED BY MISSING INPUTS OR EVIDENCE**: matching prior-only
pipeline null/evidence files and verified supplied execution terms are absent.
For a stronger forecast conclusion, register a new bounded plan with adequate
chronological coverage and matched audited evidence. Do not reuse old global
feature/ensemble choices without repairing their full fold-local creation path.
No existing Stage 9-13 historical accuracy/economics claims were reproduced,
and no period was treated as untouched prospective evidence. Stop at Stage 13.5.
