# Stage #15: offline alpha portfolio

**NO_ELIGIBLE_ALPHAS.** The framework is implemented; no evidence-supported
trading strategy has been selected. The portfolio remains inactive. Prompt #15
authorizes offline research only, superseding earlier stopping rules for this
extension. Stop before Stage #16; no broker, orders, deployment, commits or pushes.

The complete architecture and worked examples are in
[strategy specification V001](stage15_strategy_specification.md).

## What the audit actually found

The working tree was clean at `fde1c79` before edits. The saved Stage #14
`ROBUST_HISTORICAL_V002/verdict.json` names no Stage #15 candidates. All its
ridge/mean, econometric and legacy entries are blocked; short econometric
conditional evidence is inconclusive. Saved manifests, source identities,
ledgers, fitted spec files and access logs were inspected and hash-bound to the
new registry. Earlier performance is recorded evidence, not reproduced research.

The initial 5m Stage #13 V001 ledger has 44 attempts: 36 failed, eight completed;
its insufficient inner validation produced no frozen fits. The corrected 5m and
15m V002 runs each completed all 44 attempts and contain four frozen ridge/mean
fits over two folds. Registry V003 binds those actual fit and feature identities,
while retaining the failed V001 history. Their recorded one-hour economics are
negative for all 16 forecast variants per timeframe. These historical results
were inspected, not rerun; 6/2 trades are inadequate uncertainty evidence.
Existing synthetic checks do not supply market evidence. Stage #13.5
has actual saved 5m/15m econometric fits, but matching null/evidence identity and
economic policy validation are missing. GARCH/HAR, intervals and CUSUM remain
conditioning/health diagnostics. The previous ensemble and model forecasts are
retained; globally adaptive feature/count/universe chronology remains unresolved.

Registry V003 preserves 66 records: 16 directional policy hypotheses and 50
forecast/diagnostic entries. None is scientifically eligible. The core threshold
neighbors are the existing Stage #13 thresholds, not new searches. ARX/Kalman
and legacy return-to-sign mappings record an economic interpretation but no
evaluated economic evidence. Non-return forecasts receive no invented direction.
Software readiness refers to the available contract/adapter, not available fits
or passing science. Incomplete policy clocks, prior-only nulls and matching
economic/execution evidence cannot produce eligibility. Current policies have
no selected market staleness bound; this remains an explicit unknown.

Dataset identity is the saved audited `ticks-2e173ef8e61bd240`, 729,244,369 rows
and 281 partitions. No full content verification or market forecast replay was
performed here. Data lineage, feature/model identities and existing artifact
hashes are retained in the registry and source manifests. Historical upstream
exposure remains partially recorded (217 completed, 51 failed recorded attempts);
older adaptive searches and effective independent trials remain unknown.
Four reserved-access logs, including interruption/repeat records, are preserved
and checked unchanged. 2022+ has already been inspected; no final-test values
were loaded, and no new genuinely uninspected period was established.

## Frozen design and implementation

`ALPHA_PORTFOLIO_V001` was frozen before synthetic outcomes. V002 refines source
metadata after the smoke. V003 corrects V001-only references to include actual
Stage #13 V002 fits and adds explicit policy completeness validation under the
original eligibility rule. Universe, periods, methods, budget and acceptance
remain unchanged. All versions
and all results remain on disk. Registry updates require new versions; hashes
and existing content must match before reuse.

The package `alpha_portfolio` separates evidence registry, normalized exposure
intents, prior-only allocations, causal streaming decisions and bounded studies.
The scientific constructor refuses candidates whose eligibility was not known
strictly before the fold; the raw constructor is a software interface and does
not confer scientific eligibility. Per-alpha frequency/horizon contracts remain
explicit. Existing observed-bar policy adapters do not know future bar timestamps.

The small comparison family is no trading, each standalone, equal allocation,
one nonnegative minimum-variance allocation and both removal-of-one descriptions,
under base and one adverse-cost scenario. Research budget .02 hypothetical lots;
no leverage optimization. Both sleeves share XAUUSD market exposure. No winning
portfolio or subset is selected. Allocation choices and their information cutoff
are persisted before the scored fold. The fixture uses two contiguous 30-minute
folds and the same preceding training risk cache at both fold boundaries, with
no outer outcomes used for fitting. Real chronological evaluation is blocked.

The covariance option explicitly computes `C=.5*S+.5*diag(S)` on the last 32
aligned fixed-capital 5-minute standalone net returns, complete cases, at least
16 observations. First/last half-window covariances and co-loss frequency are
descriptive sensitivity, not IID standard errors. Convex covariance shrinkage
is discussed in the primary [Ledoit and Wolf paper](https://www.ledoit.net/Well-conditioned2004.pdf);
our fixed diagonal target/intensity is a declared scenario, not their estimated
optimal intensity or a guarantee of improved risk. The bounded minimum-variance
solver enumerates active sets for 1..3 sleeves; invalid/nonpositive covariance or
solver failure records equal fallback. No return forecasts or volatility logs
are supplied as optimizer expected returns.

Forecast correlation is explicitly unavailable in the synthetic intent-only
fixture. Signal correlation, standalone return correlation and shared exposure
are distinct report fields. Simultaneous negative-return frequency uses zero,
not a tail threshold; tail inference is explicitly unavailable on this short
sample. These sleeve counterfactual returns
are not independent assets or actual shared-account sleeve fills.

Stage #12 has the only fills, fees, financing, marks and P&L engine. Its new target
API supports variable quantities while closing fully before resizing/reversing.
Original standalone forecast behavior is regression tested. Actual time in a
position is tracked separately from lot-seconds so variable size cannot inflate
the reported time-exposure fraction. Shared attribution is entry-owner accounting,
not a causal allocation of alpha skill; every cash/cost metric reconciles.
Checkpoint serialization binds execution settings, budget, alpha specifications,
contracts, active intents, pending orders, owners, compensated cash and clocks.

## Actual bounded studies

`PORTFOLIO_SMOKE_V001`: 16 completed operations (two prior standalone return
caches + 14 portfolio comparisons), 2.5006 seconds, 338,440,192 private bytes.
After provenance and time-exposure repairs, V002 replay: 16 completed,
2.7878 seconds, 338,169,856 private bytes. Independent verification caught an
allocation-envelope defect: timestamps were hashed with a different representation
from the persisted JSON. Earlier runs and failed verification remain; the shared
writer now hashes its exact serialized representation. V003 smoke replay after
repair took 2.9110 seconds and 338,219,008 private bytes, with all cash/fill and
immutable identity checks passing. Final source-bound V004 also includes the
corrected registry (measurements below and in WORKLOG). Measured own-process
Windows counters, not total machine memory; no market search expansion.

Final `PORTFOLIO_SMOKE_V004`: 16 completed operations, 4.7103 seconds,
338,960,384 private bytes. `PORTFOLIO_INACTIVE_V003`: one inactive-account check,
.5652 seconds, 302,485,504 private bytes; no market study. Independent final
verification checked 334 fills, 162 closed trades, 7,582 cash flows and 40
matched cost comparisons, current source/immutable artifacts, 123 upstream
evidence references, tick-manifest identity and unchanged reserved access logs.
Final broad regression: 237 tests passed in 36.77 seconds; final Stage15-only
47 passed in 1.05 seconds (overlapping). Ruff and mypy are clean (237 source files).
Final isolated mutation audit detected all 54 guards, original source unchanged.
Earlier obstructed and superseded audit attempts remain recorded.

The synthetic market has IID Gaussian price increments, with fixed causal
5m momentum and 15m contrarian sleeve intentions. It does not establish an edge.
Base closed net P&L: standalone 5m -6.3712, standalone 15m -3.3553, equal and
minimum variance both -4.2864 account units. Adverse commission/slippage:
-8.5712, -3.9553, -5.2864 respectively. These are closed-account synthetic
numbers, not market findings. The risk alternative weights are approximately
.535887/.464113; lot rounding produces the same actual fills as equal allocation.
Removal results reproduce the corresponding standalone; no favored subset is chosen.

Some standalone cutoff marks are unknown because open exposure outlives the last
fresh synthetic quote; this is retained, not filled forward or converted to zero.
Fold mark uncertainty, open costs and known-mark drawdown are reported explicitly.
Negative realized results do not replace an unavailable primary marked-equity
statistic. Exact cash and matched-trade cost verification is separate from claims
of future portfolio performance.

## Commands and artifacts

Use `xq` or `.venv\Scripts\python.exe -m xauusd_quant.cli`:

```powershell
xq portfolio-plan
xq alpha-registry
xq portfolio-smoke --run-id PORTFOLIO_SMOKE_V004
xq portfolio-evaluate --run-id PORTFOLIO_INACTIVE_V003
```

Run IDs above identify actual studies; replay uses a new version. `portfolio-evaluate`
records the inactive account when eligibility is empty and opens no ticks or
forecast values. With future nonempty eligibility it refuses market study until
the missing fold/input audit is supplied; there is no permissive bypass.
This is not a zero-return market backtest. Programmatic intent/account/fold
machinery is validated on synthetic data independently of that scientific gate.

Versioned `results/alpha_portfolio` artifacts include plans and source hashes,
registries, exposure contract, normalization, allocations and assumptions,
decisions and sleeve intents, orders/fills/positions/cash/equity, entry ownership,
reconciled sleeve accounting, dependence, concentration, fold comparisons,
final state, verdicts, human specification, trial ledger and audit verification.
Files are created exclusively and event rows flushed incrementally; interrupted
or failed attempts remain. Data and research results remain ignored by Git.

Targeted tests cover empty/single/multiple universes, opposite intents, hand
cost arithmetic, pending/rejected orders, timing/expiry, unequal frequencies,
prior weights, matured returns, covariance/fallbacks, scientific eligibility,
future-prefix invariance, checkpoint equality, attribution and reserved guards.
The guard script disables checks in isolated copies, restores them and verifies
original source bytes. Full suite and large research runs are not invoked.
Exact final test, mutation and verification outcomes are recorded in WORKLOG.md.

Remaining blockers are matching prior-only pipeline nulls, audited economic and
execution terms, complete declared policy clocks, adequate chronological coverage,
full fold-local eligibility chronology and unknown historical selection exposure.
No candidate is suitable for an evidence-supported portfolio or automatic next
stage. Stage #16 risk management remains separate and unauthorized.
