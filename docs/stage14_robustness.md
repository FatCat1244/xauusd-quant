# Stage #14: bounded offline robustness

Prompt #14 authorizes this stage on 2026-10-04 and supersedes earlier stopping
rules only for offline research. No broker, demo/live orders, deployment, commit,
push or Stage #15 is authorized. Forecast contracts, accounting, historical
artifacts and reserved readers are preserved. No market candidate survives.

## What was actually present

The working tree was clean before edits. Read the working notes, prior-stage
documentation, code/configuration and saved records before implementation.
Metadata audit reused Stage #12: 281 monthly partitions, 729,244,369 footer rows,
`ticks-2e173ef8e61bd240`, broker-local 2003-05-05 through 2026-09-18. The manifest
SHA256 is `da8d0bfe17a33ca95080e872ac98f8474721616309234e29ce339e2236f4318c`.
Raw size is 36,233,955,746 bytes; unchanged. No raw or full quote-content rehash.
Frozen/fitted/feature hashes were audited through the existing metadata adapter.

Stage #12 has a functioning engine, synthetic round trips, metadata readiness and
bounded quote probes. It does not establish verified broker execution terms.
Stage #13 has ridge and historical-mean forecasts, a fixed sign policy and one
threshold neighbor, four execution scenarios, no-trading/random-direction controls.
V001 attempted weekend inner windows and failed; V002 repaired the dates. All
attempts remain. Chronological fits cannot erase prior analyst outcome inspection.
Stage #13.5 has eight fixed return/variance models and benchmarks, causal Kalman
states, matured-residual intervals and CUSUM. Its two real June 1 studies scored
260/77 rows per model at 5m/15m, with three folds. One day is insufficient for
the declared five-day inference rule. Previously documented broad ML/ensemble
accuracy is historical documentation here, not a result reproduced by Stage #14.

The inventory finds **268 recorded historical attempts: 217 completed, 51 failed**.
This includes 34 real May 2021 fixture attempts originally misclassified as
software tests and subsequently preserved in the audit directory. It separately
finds 139 upstream synthetic attempts. The view retains attempts, terminals,
failures and interrupted/abandoned attempts; folds identify repeated fits, rather
than automatically creating independent candidate trials. Specification keys
reflect the recorded fields only, not an inferred complete model-search universe.
Scenario/seed/resample fields are separate where recorded.

Registry metadata: 684 ML registered tests, 565 ensemble tests, 330,790 feature
tests and 389,480 shared research-ledger rows. These are **not effective independent
trials**. Older searches, abandoned unrecorded trials and adaptive revisions remain
unknown. Correlated trials are not assumed independent; a clean recent ledger
does not reset historical exposure.

Four access logs remain unchanged. ML: 20/17 evaluated specs, one start per
timeframe. Ensembles: 7/6 evaluated specs; 5m has two starts and one authorized
repeat, including its interrupted attempt. No reserved values or final-test
tables were read. The 2022+ period was already inspected. Global Stage #9 feature
identities/counts and Stage #11 eligibility/universe/correlation/null choices remain
unsuitable for earlier fold-local deployment evidence.

## Plans and commands

```powershell
xq robustness-plan --plan config/robustness.yaml
xq robustness-inventory --run-id ROBUST_INVENTORY_V002
xq robustness-smoke --run-id ROBUST_SMOKE_V003
xq robustness-evaluate --run-id ROBUST_HISTORICAL_V002
python scripts/summarize_robustness.py --run-id ROBUST_SMOKE_V002
python scripts/check_robustness_guards.py --output results/robustness/GUARD_MUTATIONS_V003.json
```

Equivalent CLI entry: `.venv\Scripts\python.exe -m xauusd_quant.cli`.
Examples with new IDs are interfaces, not claims those specific IDs were run.
Saved source score/metadata/fit byte hashes are additionally bound in immutable
`*_SOURCES.json` sidecars before opening outcome values; changed sources reject
before materialization. These hashes establish identity, not a fresh period.
Every run directory is exclusive; identical frozen plans are idempotent, changed
content under the same plan ID raises. Failed/interrupted outputs cannot be reused.
Actual executed IDs and commands are listed in WORKLOG.md.

`ROBUSTNESS_V001` froze before new diagnostics. `ROBUSTNESS_V002` explicitly adds
fixed synthetic econometric sensitivity after V001 smoke/diagnostics, retaining
market candidates, evaluation periods, acceptance rules and earlier failures.
No market search is expanded and no passing neighbor replaces the primary.
The plan records hypotheses, benchmarks, prior inspection, periods, losses,
families, dependence methods, scenarios/distributions, perturbations, null steps,
concentration definitions, budgets, stopping rules and unsupported methods.

Bounded work: two saved development runs; eight benchmark comparisons each;
block lengths 4/12/24; 199 replicates; 12 null and 12 controlled AR discovery
pipelines; eight deterministic execution scenarios and 16 Monte Carlo paths;
ridge alpha .5/1/2 plus removal of momentum and one threshold neighbor. V002
adds Kalman q multipliers .5/1/2, residual windows64/128 x gamma0/.01, and CUSUM
thresholds6/8/10 over the same 12 synthetic seeds. Hard caps: 256 recorded
operations, 25,000 saved records/file, 240 seconds and 1 GiB private memory checked
between operations. Library calls are not forcibly interrupted. No parallel heavy
studies or full historical training pipeline.

## Statistics and assumptions

`ReturnDefinition` requires sampling, capital, exposure, compounding, missingness,
overnight treatment, overlap, dependence and annualization. Supported portfolio
increments are regular equity changes divided by fixed initial capital, no
compounding or annualization; irregular/missing marks reject. Trades remain
attribution records, never a Sharpe input. Zero variance makes the descriptive
mean/SD ratio unavailable; no IID standard error is produced.

Prediction losses reuse Stage #13.5's MSE and QLIKE. The adapter verifies immutable
plan and fit identities, periods before opening scores, fold availability, label
maturity, training cutoffs, target units/horizons, duplicates and loss arithmetic.
Saved sources never use a new reserved reader. Matching uses actual identical
target/availability/observation rows; unmatched rows are counted.

Holm is implemented for a **complete predeclared family with valid marginal
p-values**. For sorted p-values, adjusted p[i] is
`min(1, max_{j<=i} (m-j+1)*p[j])` (one-based indices); reject when adjusted<=alpha.
Its critical value is alpha divided by the number of unrejected hypotheses,
verified in [Goeman and Solari (2010), section 3](https://arxiv.org/html/1211.3313#S3).
Dependence between tests does not require independence for this procedure;
valid marginal p-values and the specified fixed analysis remain prerequisites.
Repeated adaptive analysis and invalid p-values are not repaired. Missing tests
retain the whole family and make correction unavailable. A hand-calculated
.01/.03/.04 example produces .03/.06/.06. No FDR or individual-strategy truth
claim is made. **Actual market corrections are unavailable**, because valid
marginal p-values are unsupported; intervals are not converted to p-values.

The moving-block definition is random sampling with replacement of the n-b+1
overlapping consecutive blocks, concatenation and final truncation. See the
[author-uploaded Künsch (1989) paper](https://www.researchgate.net/publication/2355926_The_Jackknife_And_The_Bootstrap_For_General_Stationary_Observations)
and [Politis/Romano's primary technical report, introduction](https://www.stat.purdue.edu/docs/research/tech-reports/1991/tr91-03.pdf).
Independent index arithmetic is tested. Our additional fold/session stratification
is explicitly an implementation restriction, not a claim of the original global
stationary theorem for adaptive financial forecasts. Segments split on fold
boundaries or grid gaps; each retains its original weight. No circular session
wrap, dropped short segment or IID fallback. Blocks must cover target overlap.
Each block choice needs >=5 observed UTC dates, >=200 rows, >=8 nonoverlapping
block capacity and no segment shorter than that block. Zero variance is unavailable.
Within-segment weak dependence/local stationarity is assumed, with approximate
percentile mean intervals. All block choices are reported; no favorable choice.

These intervals address **conditional uncertainty of a frozen forecast sequence**.
They do not refit discovery or correct prior searches. Separate synthetic null
controls repeat causal features, matured-label purge, inner identities/count
selection, scaling, outer fit and fixed sign decisions through Stage #12. The
Gaussian IID-increment control asks a narrower question than the unknown real
historical search; AR(.5) increments provide a controlled relationship. Neither
replaces matching prior-only random-walk/sign-flip evidence for a market candidate.
DSR, PBO, Reality Check and SPA are explicitly unavailable, rather than forced
onto unknown trial exposure or incompatible data. Statistics cannot cure leakage,
accounting errors or reused holdouts.

## Scenario and perturbation semantics

All execution uses Stage #12 `ExecutionEngine` through Stage #13 `execute_trial`.
Deterministic stress doubles commission, adds .02 USD/oz adverse slippage,
uses 2000ms latency, doubles valid spread, adds2 account/lot/day funding,
misses10% of opportunities or declares a30s feed interruption. Missing forecasts
are explicit rejected opportunities; interrupted quotes disappear, with no
invented fills. Invalid original quotes stay invalid; extreme widening that
would make nonpositive executable Bid rejects. UTC clocks, sequence, strict
arrival<fill<expiry, quote sides, overlap rules and retained exposure remain.

Monte Carlo samples path-level commission U(1,2)x, extra slippage U(0,.04),
integer latency U{100,...,2000}ms, spread U(1,2)x, funding extra U(0,4), and
Bernoulli(.1) missed opportunities using declared per-path seeds. These are
hypothetical independent assumptions, **not measured OANDA/Exness execution**.
Every path reruns decisions and fills. No random multiplier is applied to final
PnL and no trade-order shuffle supplies drawdown evidence. Reported distributions
are conditional on the synthetic quote path and scenario model, not forecasts.

Matched closed-population arithmetic checks both commissions, embedded adverse
slippage/spread and accrued funding without subtracting spread twice. Break-even
extra commission is net closed PnL/(2*total lots), only for the identical signed
quantity, timing and matched midpoint population. A negative value means it is
already losing. It excludes open positions. Changing latency/opportunity policy
can change fills and population; aggregate economics need not be monotone in a
cost parameter. Engine summaries and audit streams retain orders/fills/exposure,
cash flows, PnL, drawdown and reconciliation for every path.

Perturbed ridge models refit each prior-only inner selector/scaler and outer
model. Removing momentum restricts identities while preserving fold-local count
selection. The full neighborhood is reported. Synthetic Kalman q sensitivity
fits parameters on the preceding300 observations, then filters only; no smoothing.
Residual windows/gamma use matured labels with a30s publication delay and declared
scale shift; CUSUM thresholds retain false alarms and detection delays. No neighbor
is selected. Market refit perturbations remain blocked on the missing evidence.

Concentration uses top-three positive closed-trade share, largest positive-month
share, monthly closed PnL, and closed PnL without the three largest gains. These
are retrospective attribution, not executable trade removal. Loss reports retain
all fold effects and first/last chronological halves. A single day/hour cannot
support month/year decay, recovery stability, causal-regime economics or recent
full-history superiority; no retrospective regime is presented as a policy input.

## Executed bounded findings

V001 synthetic smoke: 189 operations, 16.115 seconds, peak working set92,315,648
bytes, private339,288,064. V002: 192 operations, 18.573 seconds, peak121,200,640,
private633,004,032. Saved historical V001 analysis:18 operations, 2.992 seconds,
peak98,283,520/private311,095,296. Inventory V001:17.219 seconds,
peak93,507,584/private308,080,640. These are own Windows process measurements,
including imports; not total-system memory or guarantees for larger runs.

Final source-bound replay V003:192 completed operations, 17.009 seconds,
peak121,458,688/private633,528,320 bytes. Final historical V002:18 completed,
8.651 seconds, peak98,332,672/private311,144,448 bytes. The replay verified the
source identity guard and final provenance changes; original numerical inputs
and frozen acceptance rules stayed fixed. Saved verification checked22,606 order
events, 11,302 fills and5,646 closed trades for causal timing and cost arithmetic.
Both final manifests match current source/config hashes; reserved access logs
match their inventoried hashes.

Synthetic pipelines: 3/12 null paths had positive matched loss improvement;
12/12 AR(.5) paths did. These are **effect signs, not significance/false-discovery
rates**. Maximum binomial standard error with12 paths is0.144; finite simulations
can produce false positives. Dependent AR(.8) mean bootstrap SD was .03492 for
singletons, .06171/.08248/.09068 for4/12/24 blocks. Singleton resampling is a
diagnostic contrast only, never the market inference method.

Synthetic Monte Carlo marked-PnL quantiles5/50/95% were -12.206/-2.826/+15.500
account units, conditional on the declared synthetic inputs. The median loss is
preserved. Narrow economic inference from16 hypothetical paths is unsupported.
Kalman q neighborhoods had synthetic MSE1.750/1.688/1.682e-7; none selected.
Residual coverage was78.26/77.21% for rolling64/128 versus80.34/80.21% adaptive,
with adaptive widths larger (.002008/.001978 versus .001894/.001840). This is a
synthetic calibration tradeoff, not prospective/economic superiority.
CUSUM threshold6 had any null alarm in3/12 paths; thresholds8/10 had0/12, which
does not prove zero false-alarm probability. +3sigma detection delays were2-5
observations. Outputs remain health diagnostics without automatic actions.

Saved June 1 effect sizes (positive means benchmark loss minus candidate loss):

| Comparison | 5m | 15m |
|---|---:|---:|
| ARX / zero, relative MSE | -1.3198% | -4.1167% |
| ARX / ridge, relative MSE | +0.1426% | -0.2998% |
| Kalman / zero, relative MSE | -0.7563% | -2.1588% |
| Kalman / ridge, relative MSE | +0.6979% | +1.5864% |
| GARCH / rolling, QLIKE difference | +0.02046 | +0.29499 |
| GARCH / EWMA, QLIKE difference | +0.08268 | +0.11594 |
| HAR / rolling, QLIKE difference | -0.18568 | +0.17974 |
| HAR / EWMA, QLIKE difference | -0.12347 | +0.00070 |

These independently recalculated saved losses match Stage #13.5; this is not a
new market run. Return improvements against zero are negative. Favorable aggregate
GARCH comparisons do not establish stable superiority: rolling comparison is
positive in1/3 and2/3 folds. All48 block configurations are insufficient; no
market bootstrap intervals or p-values are printed. Existing coefficient,
coverage, width and filtered-state diagnostics are linked by hashes and retained.

ARX, Kalman, GARCH and HAR at both timeframes: **BLOCKED** for promotion due to
missing audited evidence identity and matching pipeline nulls, with conditional
forecast evidence **INCONCLUSIVE** due to short coverage. Ridge/historical mean:
**BLOCKED** by matching evidence and verified execution assumptions. Legacy
ML/ensembles: **BLOCKED** additionally by global feature/count/universe chronology.
No candidate is suitable for Stage #15 on this evidence. Search exposure is only
partially inventoried; discovery uncertainty is unknown. No profitability or
prospective validity claim follows. Stop after Stage #14.

## Verification and artifacts

Targeted tests cover independent arithmetic, dependence sensitivity, zero/invalid
variance, missing inputs, complete families, chronology, matured labels, ledger
failures, null selection steps, reserved refusal, quote-side costs and operational
ordering. Existing Stage12/13/13.5 tests are reused. Full suite/heavy research is
not run. The mutation audit breaks guards in isolated copies, restores each,
checks original hashes and preserves unsuccessful audit attempts. Exact commands,
counts and any sandbox recovery are in WORKLOG.md.

Actual checks:221 targeted regression tests passed in48.12 seconds; final18
Stage14 tests passed in2.15 seconds, including the later source identity and
econometric fixture tests (225 distinct targeted tests across these commands).
Ruff clean; mypy clean, 229 source files; `git diff --check` clean. Mutation audit
V002 detected all 41 guards; V003 detected all 42, including the source hash guard.

Artifacts under `results/robustness`: immutable plans; run manifest/config/code
fingerprints; selection inventory; assumption audit; statistics/corrections;
bootstrap, Monte Carlo, execution, perturbation, concentration/decay and
econometric sensitivity reports; candidate verdicts; incremental trial ledger;
Stage12 engine streams per execution path; mutation reports. Failed and interrupted
runs remain. Source/config changes affect later fingerprints, not old artifacts.
