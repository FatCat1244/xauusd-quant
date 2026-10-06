# Stage #18: demo-only execution and reconciliation

Stage #18 authorizes bounded execution on the explicitly configured DEMO account
and committing/pushing completed implementation. It does not authorize real money,
deployment, unattended operation, or Stage #19. The Stage #17 application remains
read-only: it does not import the demo adapter and has no trading switch.

## Actual upstream evidence

The Stage #15 V003 alpha registry contains 66 records and zero eligible alphas.
Stage #14 produced no survivor. Null, chronology, economics and sufficient-coverage
gates remain unresolved; historical 2022+ outcomes were previously inspected.
Stage #16's actual risk configuration is UNCONFIGURED. Its synthetic replay is
software evidence, not a supplied account policy.

Stage #17 connected to the configured demo terminal, but the bounded native capture
contained zero accepted ticks and bars. Its empty replay comparison did not verify
full model or pipeline equality. Feed transfer, compatible frozen feature/context
and model artifacts, calibrated health inputs and prospective strategy evidence
remain missing. Historical performance was not reproduced during Stage #18 and
reserved market data was not opened.

Accordingly, strategy-driven demo operation is BLOCKED. The coordinator has a
typed Stage #16 RiskRequest/HealthSnapshot boundary for future portfolio integration;
the native adapter deliberately rejects STRATEGY requests. A functioning native
strategy streaming binding cannot be validated against the current empty eligible
universe and is not claimed as completed. No arbitrary alpha or fake health inputs
are substituted. A separate mechanical smoke test can assess an order lifecycle
without predictive inputs or a profitability claim.

## Configuration: what the user needs to choose

Terminal identity and execution permission are different from risk permission.
The existing ignored `config/local/shadow.yaml` supplies terminal executable,
expected account/server/company and exact symbol. It can be reused by reference;
no password or account number should be copied into the public demo template.
Use plain local YAML; an `.env` is unnecessary for an already authenticated terminal.

Copy `config/demo.yaml` to `config/local/demo.yaml`, and `config/risk_demo.yaml`
to `config/local/risk_demo.yaml`, without overwriting an existing local file.
Keep all private identity and financial settings in that ignored directory.
Null values intentionally block execution. These templates are not configured
risk policies; do not copy numeric synthetic fixtures and relabel them verified.

The smoke-test choices include:

| Setting | Meaning and units |
| --- | --- |
| `side`, `quantity_lots` | User-chosen BUY or SELL and requested lot quantity for one mechanical entry. No strategy signal is implied. |
| `max_quantity_lots`, `max_net_lots`, `max_gross_lots` | Hard request and exposure caps in lots; the requested quantity must fit all three. |
| `max_duration_seconds`, `hold_seconds`, `cleanup_seconds` | Bounded runtime, intended short holding time and time reserved for exit/reconciliation. Total runtime cannot exceed 300 seconds. |
| `entry_budget`, `entry_request_budget` | Exactly one for a smoke test. A failed entry is not resubmitted. |
| `cleanup_request_budget` | User-chosen capacity, 1–3 requests. A second close is allowed only after a terminal partial close and verified remaining owned quantity. |
| `deviation_points` | Native symbol points, not pips, ounces or a guaranteed slippage ceiling. |
| `filling_policy` | FOK or IOC, supported by the actual symbol's execution mode and filling flags. No fallback search. |
| `expected_account_mode`, `expected_account_currency` | Verified NETTING or HEDGING and the exact native ledger currency. Cent denomination must be explicit in risk metadata. |
| `dedicated_account_confirmed` | Explicit acknowledgement that the account is dedicated. Startup still checks it is flat with no pending orders. |
| `project_tag`, `magic`, versioned IDs | Stable local project/run/specification identities; these do not alone prove position ownership. |
| `intent_ttl_seconds` | Explicit approval lifetime, at most 60 seconds. |
| `shutdown_policy`, `protection` | Smoke requires CLOSE_OWNED and supported PROCESS_EXIT. Broker-hosted stop policies are rejected, not fabricated. |

The dedicated risk policy uses the exact Stage #16 schema in
`src/xauusd_quant/risk/policy.py` and `risk/contracts.py`. `demo-readiness` reports
every missing field. Replace each null block with a complete supplied structure:

- **Policy:** account, position and sleeve loss budgets; daily loss and drawdown
  caps in ledger currency; net/gross lot caps; one shared position; pending/order
  limits and measurement windows; turnover capacity; minimum free margin; quote,
  account and health freshness; maximum spread in USD per troy ounce; session and
  overnight rules; limit response and emergency-reduction behavior; warm-up count;
  holding limit; hypothetical horizon stress fraction and adverse exit allowance.
  All schema fields remain required, including diagnostic limits that the separate
  no-model smoke route does not consult.
- **Account:** identified currency, `major` or `cent` denomination, ledger units
  per major currency, conversion from USD, source/provenance and specification ID.
  A balance number does not establish denomination. Conversion must be verified.
- **Instrument:** canonical XAUUSD mapping to the exact configured native symbol;
  USD per troy ounce prices, lots, ounces per lot, minimum/maximum/step, tick size,
  precision, conservative margin fraction, commission per lot per leg, adverse
  execution allowance, long/short financing and provenance. Symbol metadata alone
  does not establish commission or financing. Native profit/margin calculations
  must agree with supplied units and the risk reservation.

Native policy/account/instrument blocks require `status: verified_supplied`.
For the separate smoke policy use `expected_portfolio_id: EXECUTION_SMOKE_V001`,
`allowed_allocation_ids: [SMOKE_V001]`, `alpha_contracts: {}`,
`require_diagnostic: false` and `sizing_method: horizon_stress`; these identify a
mechanical test, not an eligible trading strategy. Use a distinct risk-policy ID
and reference it in `demo.yaml`. Loss estimates under horizon stress are conditional
scenario estimates, not guaranteed maximum losses. Quantity rounds down; a broker
minimum above the budget causes rejection.

No monetary value is selected on the user's behalf. Once complete supplied
settings and readiness gates pass, the Stage #18 request already authorizes the
bounded demo smoke run; no additional discretionary permission is needed.

## Interfaces and authoritative risk boundary

`demo/config.py` defines strict DEMO-only configuration. `demo/broker.py` wraps the
official vendor package lazily, separately from Stage #17. `demo/worker.py` isolates
native IPC behind fixed verbs and a 10-second call deadline; ordinary offline tests
inject fake brokers/vendor APIs. No REAL mode, enum override, account switching,
login request or terminal setting change exists.

`demo/coordinator.py` accepts proposed exposure through Stage #16's RiskEngine,
refreshes verified account/quote data, reserves risk, and binds an opaque capability
for the exact approved request. Raw strategy requests cannot use that adapter
instance. This is an architectural boundary, not protection against malicious
Python code modifying the interpreter. Risk/model checks for actual strategies
remain intact; only the dedicated mechanical smoke namespace omits predictive
health requirements. It still enforces units, budgets, sizing, sessions, margin,
loss/drawdown and pending reservations.

The capability also carries an expiry, maximum quote age and approved economic
state through IPC. The native boundary independently compares current balances,
currency/mode, exposure, instrument terms and entry quote with that state; it
rejects missing/expired approvals and stale quotes. Verified MARKET reductions
permit price/mark changes while binding currency, position identity and quantity.
This final recheck cannot make the subsequent broker call atomic with market state.

The existing Stage #12 simulator remains authoritative for shadow/counterfactual
fills and Stage #15 shared-account economics. Actual demo accounting instead uses
broker-reported deals; it does not add standalone curves, substitute requested
prices or use a competing simulation engine.

## Native API semantics and supported capabilities

Primary references verified for this implementation:

- [Official Python integration](https://www.mql5.com/en/docs/python_metatrader5),
  [account_info](https://www.mql5.com/en/docs/python_metatrader5/mt5accountinfo_py),
  [terminal_info](https://www.mql5.com/en/docs/python_metatrader5/mt5terminalinfo_py),
  [account enums](https://www.mql5.com/en/docs/constants/environment_state/accountinformation).
- [symbol_info](https://www.mql5.com/en/docs/python_metatrader5/mt5symbolinfo_py),
  [symbol_info_tick](https://www.mql5.com/en/docs/python_metatrader5/mt5symbolinfotick_py),
  [symbol capabilities](https://www.mql5.com/en/docs/constants/environment_state/marketinfoconstants).
- [order_calc_margin](https://www.mql5.com/en/docs/python_metatrader5/mt5ordercalcmargin_py),
  [order_calc_profit](https://www.mql5.com/en/docs/python_metatrader5/mt5ordercalcprofit_py),
  [order_check](https://www.mql5.com/en/docs/python_metatrader5/mt5ordercheck_py),
  [order_send](https://www.mql5.com/en/docs/python_metatrader5/mt5ordersend_py),
  [return codes](https://www.mql5.com/en/docs/constants/errorswarnings/enum_trade_return_codes).
- [orders_get](https://www.mql5.com/en/docs/python_metatrader5/mt5ordersget_py),
  [positions_get](https://www.mql5.com/en/docs/python_metatrader5/mt5positionsget_py),
  [history_orders_get](https://www.mql5.com/en/docs/python_metatrader5/mt5historyordersget_py),
  [history_deals_get](https://www.mql5.com/en/docs/python_metatrader5/mt5historydealsget_py).

`order_check` success uses retcode 0. An order-send DONE response uses 10009 and
still requires deal/account reconciliation. PLACED 10008, partial 10010 and uncertain
timeout/connection/API outcomes are distinguished. An accepted request is not a
verified fill. Filling flags FOK=1 and IOC=2 map to request enums 0 and 1. MARKET
execution requires the advertised flag and omits request price. REQUEST/INSTANT
execution uses the executable quote side and tick precision; exchange execution,
RETURN, pending-order types, stop modifications and cancellation are unsupported.
Unsupported capabilities halt/reject rather than cycling alternatives.

Demo type is checked through the native enumeration, alongside exact terminal,
login, server, company, currency/mode and permissions before arming, check/send,
recovery and closure. Identity is checked again after sending. A terminal can
change account between API calls; the interface cannot provide an atomic server-side
identity guarantee. Such a race becomes unresolved and must not cause cleanup on
the newly connected account.

Both netting and a single dedicated hedging position are supported. An unrelated
position anywhere in the dedicated account or an unexplained deal/order blocks
execution. Closure references the current broker position ticket, linked through
the position identifier to durable requests, unique orders and actual deals.
Order, deal, position ticket and position identifier are not interchangeable.
Magic/comment matching alone cannot adopt or close a position.

## Persistence, failure and shutdown

`demo/journal.py` provides an account-level OS single-writer lock and bounded,
append-only hash-chained JSONL with flush/fsync. An intent, reservation and
SUBMISSION_ATTEMPTED checkpoint are durable before calling the broker. A crash
around the call is UNKNOWN even if no response was stored. Repeated intent IDs
cannot allocate or submit twice. No exactly-once broker guarantee is asserted.

Lifecycle: CREATED → RISK_APPROVED → CHECKED → SUBMISSION_ATTEMPTED →
ACCEPTED_OR_PENDING/PARTIALLY_FILLED → FILLED or CLOSED_OR_CANCELLED. Rejections,
UNKNOWN and RECONCILIATION_REQUIRED retain auditable reasons. A precheck failure
releases a reservation only after a verified fresh account reconciliation. Partial
execution preserves remaining risk until terminal status and the actual account
are known. Unknown outcomes are never retried because a timer elapsed.

Reconciliation polls orders, positions and bounded order/deal history. Missing,
delayed, conflicting or amended economic records halt exposure increases and retain
reservations. Actual deal quantity/side/price and profit/commission/swap/fee reconcile
to dedicated-account balance and net position, with the native currency precision
as the cash tolerance. External balance flows are reconciled but excluded from risk
profit/loss/high-water marks. Unsupported credit/separate charge/correction types
remain unresolved rather than guessed.

Any changed economic state before entry causes abstention. There is no entry retry,
price chasing, filling-policy fallback or P&L tuning. Verified risk reductions may
proceed during loss halts; MARKET reductions can use a refreshed quote while still
checking position identity and remaining quantity. Close requests cannot reverse
the verified position.

At shutdown, entries stop and only verified owned exposure may be closed within
the declared request/time budget. Pending or unknown outcomes remain unresolved;
this version does not cancel broker orders. API/storage failure can prevent cleanup.
An emitted close is not proof of flatness. The worker is stopped, the journal is
closed and unresolved exposure is reported. PROCESS_EXIT protection relies on the
running process; disconnection/crash can leave exposure without a broker stop.

Restarts preserve halts, high-water marks, losses, reservations, intent history and
code/config identities. Incompatible or corrupt checkpoints and uncheckpointed
tails block restoration. Recovery after a submission cannot rearm entries. A separately invoked
bounded recovery close gets a new cleanup time window while retaining the original
history boundary and all risk state. There is no automatic reset or automatic
operator rearming tool in Stage #18. An unresolved journal must not be deleted or
edited to obtain a READY status; checkpoint migration/manual reconciliation after
code changes remains an operational prerequisite.

## Commands

From the repository, prefix each command with
`.venv\Scripts\python.exe -m xauusd_quant.cli`. Every run ID must be new and versioned.
Results are immutable under ignored `results/demo/runs/<id>`. Account-private
journals are under ignored `runtime/demo/<hashed-account-identity>`.

```powershell
# No broker execution:
demo-plan --run-id DEMO_PLAN_V001
demo-readiness --demo-config config/local/demo.yaml --run-id DEMO_READINESS_V002
demo-preflight --terminal-config config/local/shadow.yaml --run-id DEMO_NATIVE_PREFLIGHT_V002

# Only after complete verified settings; precheck itself does not submit:
demo-precheck --demo-config config/local/demo.yaml --run-id DEMO_PRECHECK_V001
demo-smoke --demo-config config/local/demo.yaml --run-id DEMO_SMOKE_V001

# Read-only reconciliation or explicitly bounded verified owned reduction:
demo-reconcile --demo-config config/local/demo.yaml --run-id DEMO_RECONCILE_V001
demo-recover-close --demo-config config/local/demo.yaml --run-id DEMO_RECOVER_V001

# Remains blocked by upstream evidence and absent native strategy binding:
demo-strategy --demo-config config/local/demo.yaml --run-id DEMO_STRATEGY_V001
```

A successful check-only journal is retained. An explicitly invoked subsequent smoke
command may arm it only after fresh reconciliation, READY risk state and verified
permissions, with no prior submission attempt or halt and all check-only intents
terminal. After any submission attempt, restart permits reconciliation/reduction
only. Repeated execution runs require a future audited rearming procedure; do not
change identity/path or erase risk history to evade that restriction.

Ordinary tests never initialize native MT5 or submit orders. Fake vendor tests cover
identity races, enum rejection, permissions, filling rules, precheck/send distinctions,
quantities and ownership. Coordinator tests cover risk expiry, changed account/quote,
cent units, costs, duplicate/partial/late fills, timeouts, external activity, crashes,
failed exits, restoration and durable submission ordering. Mutation checks disable
guards in isolated source copies and restore each copy; reserved guards are included.

Actual validation results and code identities are recorded in WORKLOG.md and local
versioned verification artifacts. Passing offline tests establishes neither live
safety nor profitability. Stage #19 economics validation and further operational
testing remain separate; a strategy-driven demo run requires the missing scientific,
feed/equality/artifact/risk gates first.

## Executed validation (2026-10-05)

357 targeted tests passed; full-repository Ruff and typing (260 source files)
passed. Nine isolated mutation canaries detected their disabled guards, with
original source unchanged. Earlier failed guard/study attempts are preserved;
WORKLOG.md records their defects and corrected version identities.

DEMO_SYNTHETIC_V003 ran the six frozen fake-broker cases sequentially in 7.8792
seconds. Its representative case took 1.3158 seconds. Peak own working set was
230731776 bytes, private memory 662831104 bytes; worker memory was not measured.
Accepted/unknown cases retained reservations. The full-fill known-price case's
3.48 ledger-unit cash change is synthetic arithmetic, not a strategy finding.

Actual read-only preflights V001/V002/V003 connected and verified the configured
demo account, HEDGING mode and empty current positions/orders. V001/V002 observed
fresh quote snapshots; V003's quote was stale under the configured freshness rule.
No continuous-feed or full-model equality was established. Execution permissions
were disabled: V002/V003 isolated `terminal_trade_allowed` as the failed flag;
Python API/account permissions passed. To execute a later fully configured smoke
test, the user would need to enable the intended terminal's **Algo Trading /
AutoTrading** control manually. This can affect other EAs, so it belongs in a
dedicated terminal/account setup. It is unnecessary for read-only data collection.
The application did not change this setting.

Final synthetic/native validation code identity:
`7b33d23c30f78900d67f31c8cdbc534397e97718f29f6c3bfbf835fb52c8650c`.
Workers stopped before subsequent documentation/Git operations. No source edits
occurred during a bounded native or synthetic study.

Actual order checks, submissions, fills and closures: **zero**. There is no
project-created broker exposure or uncertain submission. Empty account state was
observed at preflight, not inferred from a close intent. SMOKE and PRECHECK commands
blocked before native adapter construction because the local quantity, limits,
account/instrument terms and risk policy are incomplete. STRATEGY also reports
NO_ELIGIBLE_ALPHAS, full live equality/feed-transfer and native-binding blockers.
Ignored local templates were prepared without choosing financial values. No
evidence-supported trading strategy has been activated. Stop after Stage #18.

## Follow-up: explicit recovery of a never-submitted quote abort

The user subsequently supplied local limits, confirmed dedicated account use and
enabled the intended demo terminal's Algo Trading permission. Read-only preflights
verified demo identity, fresh quotes, all execution permission flags and an empty
current book. Local risk/account/instrument configurations now load; these facts
do not resolve the zero eligible alpha universe or strategy streaming gates.

A configured `demo-precheck` attempt was halted when the quote changed between
approval and submission. No broker order_check or order_send was called. Read-only
reconciliation retired the unsubmitted intent, released its reservation and
verified flatness while retaining the halt. The exact-state guard is conservative:
even an ordinary small quote movement can abort entry. This is not a broker rejection.

`scripts/run_unsubmitted_demo_smoke.py` adds a narrow explicit operator workflow:

```powershell
.venv\Scripts\python.exe scripts/run_unsubmitted_demo_smoke.py --demo-config config/local/demo.yaml --run-id DEMO_OPERATOR_SMOKE_V001 --operator USER_REQUEST
```

Use a new run identity for new artifacts. This is a broker-changing command for
a fully configured DEMO SMOKE only; it is not an ordinary test or a shadow command.
It must be explicitly invoked under existing bounded demo authorization. The
script freezes DEMO_UNSUBMITTED_REARM_PLAN_V001 before observations. It obtains
the normal per-account single-writer lock and restores the existing coordinator
without any checkpoint migration or source-identity exception.

Rearming requires quote-abort halt reasons only, fresh verified demo identity,
valid quote and permissions, flat reconciled broker state, terminal unsubmitted
intents, no reservations, and recovered loss/drawdown limits. The full journal is
checked as well as current intents. Any submission_utc or broker-response evidence
blocks this path, including a rejected, filled or uncertain request. Other halts,
unavailable history, incompatible checkpoints, external activity or unknown
ownership remain blocked.

The existing RiskEngine.rearm records prior halts and operator/reason; an additional
execution_rearm record identifies the plan and script hash. Loss, high-water mark,
daily baseline, decisions, order-rate/turnover and old intent history remain intact.
Only this same process gains permission to arm. A restart still cannot auto-arm.
The old intent is never resubmitted: a new intent must obtain a new risk approval.
Native request construction, ownership, exact quote/account binding, expiry and
precheck/send guards are unchanged. Another quote abort stops this invocation.

The workflow permits one entry send at most and only the configured bounded owned
cleanup. Unknown ownership or execution retains reservations rather than guessing
an exit. Cleanup is attempted in the finally path after an interrupted/failed run;
verified closure and actual deals, not emitted intents, determine success. Broker
API deadlines and configured cleanup duration remain finite and cannot guarantee
closure. This tool cannot retry any account with a previous submission.

Validation: 142 targeted tests passed, full-repository Ruff passed, and typing
passed for 261 source/script files. Seventeen added synthetic tests cover history
preservation, full-journal submission exclusion, stale/invalid inputs, unrelated
halts, restart isolation, unchanged quote rejection, verified fake lifecycle and
failed exit with retained exposure/reservation. A deliberately disabled submission
guard in an isolated in-memory function caused the relevant guard test to fail;
the deployed function was unchanged.

Actual DEMO_OPERATOR_SMOKE_V001 was explicitly invoked once. It recorded a valid
operator rearm, then another quote change aborted before broker precheck or send.
Result: NO_VERIFIED_LIFECYCLE, zero checks/submissions/deals, zero positions/orders/
reservations, no unresolved intents and verified flat broker state. The quote halt
and EXPLICIT_REARM_SMOKE_ABORTED remain persisted. No automatic second attempt or
weakening of the quote rule was performed. Repeated prompt progression cannot
turn this negative operational result into strategy or execution readiness.

Native run source identity remained
`7b33d23c30f78900d67f31c8cdbc534397e97718f29f6c3bfbf835fb52c8650c`.
Operator-script identity used for that run:
`84474a9a4858915c8f0050b435c3d049d07dadc46bd9b14f011749515a6c82e3`.
The broker worker shut down before final documentation/Git operations. Private
configuration, observations and journals remain ignored. Demo strategy trading and
Stage #19 remain blocked; actual demo lifecycle success has not been demonstrated.

## Follow-up: bounded fresh-quote revalidation

The user authorized correcting the quote-only abort and another single bounded
mechanical demo lifecycle. `--fresh-quotes` explicitly selects
DEMO_FRESH_QUOTE_PLAN_V002. Ordinary CLI/default operator runs retain the prior
exact-price binding. The new mode supports MARKET smoke entries only, with no
strategy activation, expiry extension, quantity increase or automatic retry.

Before checking an order, the coordinator persists a reservation for the smaller
of the supplied position, sole-sleeve and account risk allowances, and the available
margin headroom above the configured floor. Only one flat-account entry can be
reserved. These are conservative reservations within existing budgets, not new
financial settings or guaranteed loss ceilings. Actual open estimated risk retains
the conservative allowance until verified closure.

At each of two parent snapshots and each of the native precheck/send boundaries,
the existing Stage16 engine is restored into an isolated calculation copy. Only
that copy's unsubmitted reservation is replaced for the calculation. The actual
reservation, loss/high-water marks, order-rate and turnover history remain intact.
The new decision must approve the identical signed quantity and fit within the
reserved allowance and margin. Revalidation counts conservatively as an extra
approval in the calculation copy; it never clears actual rate/turnover history.
Native margin/profit units are checked again using the observed executable quote.
Account, positions, orders, instrument capabilities, identity, permissions,
freshness, session, limits and expiry still gate execution. There is no quote loop
or broker resubmission. Quote observation and execution are not atomic, so this
does not guarantee a fill price, stop loss or realized loss ceiling.

Example for an explicitly authorized configured DEMO smoke invocation:

```powershell
.venv\Scripts\python.exe scripts/run_unsubmitted_demo_smoke.py --demo-config config/local/demo.yaml --run-id DEMO_FRESH_QUOTE_SMOKE_V001 --operator USER_REQUEST --fresh-quotes
```

The operator tool can migrate only the documented predecessor source
`7b33d23c30f78900d67f31c8cdbc534397e97718f29f6c3bfbf835fb52c8650c`.
The hash-chained journal must end in a complete checkpoint; configuration and
terminal identities and risk checkpoint validation must match. Every historical
checkpoint is inspected for submissions, and any broker response prohibits this
path. Fresh identity, permissions, quotes and verified flat reconciliation are
required. Migration records old/new source identities and operator before the
explicit risk rearm audit. Prior loss/drawdown/HWM/order/intent history survives;
no journal deletion, unconfirmed exposure adoption or generic code bypass exists.
Any non-quote halt remains blocked. The prior wrapper abort is admitted only
alongside a recorded never-submitted quote halt in this opt-in mode.

Native quote receipt time is sampled after tick retrieval; ticks genuinely dated
after that receipt remain rejected. A hand-checkable test models a tick arriving
during API reads, and a final age check rejects a quote that ages out during native
risk calculations. Worker errors now expose only controlled local validation codes
and distinguish boundary rejection from a check/send API exception; arbitrary
vendor messages or private identity cannot cross this diagnostic path. Operator
verdicts retain the last phase and safe failure code. These additions were made
after the bounded worker stopped; they were not used in that native attempt.

Actual `DEMO_FRESH_QUOTE_PREFLIGHT_V001` verified demo identity, HEDGING mode, all
execution permissions, a fresh quote and an empty book. Actual
`DEMO_FRESH_QUOTE_SMOKE_V001` migrated the documented unsubmitted predecessor with
audit, rearmed explicitly and obtained an APPROVE on the first full fresh-quote
Stage16 evaluation. It then returned a native-worker ValueError in the precheck
path. No successful precheck result was recorded. The old worker did not preserve
the exact boundary/API failure phase, so order_check invocation itself cannot be
certified from this evidence. There were **zero broker-changing submissions**, zero
entry/close deals and zero broker-reported cash change. Final reconciliation
verified positions, orders, reservations and unresolved intents all zero. The
EXPLICIT_REARM_SMOKE_ABORTED halt remains. Status: NO_VERIFIED_LIFECYCLE; wall time
4.595128 seconds. No second invocation or entry retry was made.

A subsequent read-only margin/profit calculation at the recorded quote matched
the supplied linear CFD profit units within currency precision; it did not check,
submit or fill an order. The subsequently reproduced early-clock bug is not a
confirmed cause of the native failure. Full native diagnostics remain a blocker.
After source changes, the journal's run identity remains its actual validated
source identity; it is not relabeled as the final code. The generic runtime
rejects that mismatch, and the operator tool cannot migrate this new halt as an
original quote-only abort. No checkpoint deletion or automatic rearm is permitted.

Native run source identity:
`220270ae2d4849cc636fa5b6a48b9a66e1154eedf95a19cd5581a0eeedd6e0e9`.
Operator script used:
`3e2b2f2199866232c6bc667769eac1963a9b47b63d32e5ff55a0839234361f17`.
The post-run diagnostic/clock changes have only offline test evidence. Strategy
eligibility, feed-transfer/full-model equality and Stage #19 remain unresolved;
this work establishes neither a successful demo lifecycle nor trading readiness.

Final verification:211 targeted tests passed (13.08seconds), full Ruff passed,
typing passed263 source/script files, and all11 isolated disabled-guard canaries
were detected with deployed source unchanged. The initial sandbox-denied mutation
run is preserved; the shared runner now requires JUnit assertion failures without
fixture errors rather than matching a word in console output. The local evidence
verifier is broker-free:

```powershell
.venv\Scripts\python.exe scripts/verify_fresh_quote_artifacts.py --demo-config config/local/demo.yaml --output results/demo/FINAL_FRESH_QUOTE_VERIFICATION_V001.json
```

Use a new output version to rerun. It validates frozen reports, risk checkpoint,
full journal chain, migration/revalidation, zero submissions and flat cash
reconciliation. Its successful verification preserves the failed lifecycle status.
Final offline-tested source:
`caece6af9b565f7b1450a01fecf2046b6c1929bc4e854b0a76077db82b5c2e28`.
Final artifact verification was repeated as V003 after the final age guard, without
broker connectivity or alteration of the preserved halt/checkpoint.

## Continuation: diagnostic and fresh feed observation

The user requested completion of the remaining work. The continuation preserves
supplied local financial settings and the dedicated mechanical SMOKE specification.
No evidence-supported trading strategy has been selected: ALPHA_REGISTRY_V003
contains 66 BLOCKED entries, including 16 directional candidates and 50 diagnostics.
Saved metadata lists 37 model and 13 ensemble specifications; those counts are
not live-compatible fitted artifacts or qualified alphas. The model ensemble is
unchanged. No search expansion or historical training study was run.

Missing scientific prerequisites remain matched prior-only random-walk/sign-flip
pipeline controls, verified historical execution/account assumptions, audited full
legacy fold-local feature/count/universe chronology, adequate chronological coverage
and independently established uninspected evaluation evidence. Current demo terms
do not retroactively verify historical execution economics. Previously inspected
2022+ outcomes remain reserved and were not loaded. Strategy feed transfer, full
feature/model streaming equality and strategy-specific risk settings are also
unresolved. The configured demo risk policy governs SMOKE only.

The operator tool now offers `--recover-validation-abort` and `--precheck-only`.
Exceptional recovery requires the original frozen failed-run/configuration hashes,
its intact full journal, zero historical submission evidence and fresh permitted
flat reconciliation. Only the documented unsubmitted native wrapper halt may be
rearmed; severe or unrelated halts block. Known source migration and explicit
operator rearm are audited. No loss/HWM/order/turnover history is reset.

V001 diagnostic/recovery plans were frozen before observation. After its session
rejection, V002 plans were frozen with explicit session-rejection recovery and a
mandatory successful current-source precheck gate before smoke. This is a recorded
workflow change, not a revision of execution/economic acceptance criteria. V001
artifacts remain immutable. A session-only rejection must have zero approved
change, no broker check/send, identical configuration/source and frozen diagnostic
evidence. It never counts as a passing precheck. A marker alone is insufficient:
the latest diagnostic needs its durable approved intent, successful check code 0
and zero submission. The tool refuses premature recovery smoke before connection.
One diagnostic attempt is persisted before entering the check path; interruption
or failure consumes it and does not create an automatic diagnostic retry.

Diagnostic mode cannot submit entry or cleanup orders. Unexpected exposure retains
a halt and unresolved state; it cannot be adopted or closed by a check-only run.
Actual smoke cleanup remains bounded and requires verified project ownership.
Ordinary shadow paths remain read-only.

Actual continuation results:

| Observation | Result |
|---|---|
| DEMO_CONTINUE_PREFLIGHT_V001 | Configured demo identity, HEDGING, permissions, fresh quote and empty book verified |
| DEMO_NATIVE_DIAGNOSTIC_V001 | Risk REJECT: SESSION_CLOSED and OVERNIGHT_RESTRICTION, 4.506309 seconds |
| Broker calls from that diagnostic | Zero order_check calls, zero submissions, zero entry/close deals |
| Final reconciliation | Flat verified; no orders/positions/reservations/unresolved intents; cash discrepancy 0 |
| Original native ValueError | Unresolved: this diagnostic never reached the native boundary |
| EXNESS_CONTINUE_CAPTURE_V001 | Read-only 59.802941 seconds, 2507 ticks: 229 fresh, 2278 backfill; no halts |
| EXNESS_CONTINUE_REPLAY_V001 | 48 committed batches, 2507 ticks, four bars |
| EXNESS_CONTINUE_EQUALITY_V001 | Four bars/features/NO_ACTION records matched, zero mismatches; predictions0 |

All four emitted bars were reconstructed backfill (three 5m, one 15m); two partial
startup bars were invalid. The capture emitted no newly completed live bar.
Features were monitoring ret_1 only. Full model and portfolio/risk equality remain
untested. Peak own-process working set 243789824 bytes, private bytes 694235136;
worker memory excluded. This was a fresh-feed observation, not strategy evaluation.

The supplied session is weekdays 07:00-17:00 UTC (14:00-00:00 Bangkok). The diagnostic
occurred at 2026-10-06 17:21:41 UTC, outside that window. Its original generic
NO_VERIFIED_LIFECYCLE verdict is preserved; a derived verified classification is
RISK_BLOCKED_NO_ORDERS. Future operator reports include the risk decision/rules
directly. The original wrapper halt was explicitly rearmed and retained in audit;
after the session rejection the checkpoint had READY risk and no active halt.
Restoration still requires reconciliation and explicit arming. No second native
attempt was made, no limits were altered, and no unattended wait/run was scheduled.

Actual observation source identity:
`caece6af9b565f7b1450a01fecf2046b6c1929bc4e854b0a76077db82b5c2e28`.
Actual diagnostic operator-script identity:
`8eeb66faaa1ff9847518cc4ad0afb5c99fc023a760d3ccf79d6902d81f883f69`.
Subsequent script safeguards have offline validation only. All bounded native
processes stopped before final edits and Git operations.

### Commands for the next permitted session

The next permitted entry window after this attempt starts 7 October 2026 at 14:00
Bangkok (07:00 UTC). A later invocation must still satisfy fresh identity, session,
spread, funds, risk, ownership and configured duration/budget checks. No additional
financial values are required for this configured SMOKE; strategy remains blocked.
Run from the repository and retain new run identities:

```powershell
.venv\Scripts\python.exe scripts/run_unsubmitted_demo_smoke.py --demo-config config/local/demo.yaml --run-id DEMO_NATIVE_DIAGNOSTIC_V002 --operator USER_REQUEST --fresh-quotes --recover-validation-abort --precheck-only
```

Proceed only on BROKER_PRECHECK_PASSED_NO_ORDERS with verified flatness. A successful
diagnostic consumes approval-rate/turnover history despite submitting nothing.
Let the configured 300-second measurement window expire naturally before smoke;
do not delete checkpoints, reset counters or loosen limits. Then explicitly invoke
one bounded mechanical lifecycle:

```powershell
.venv\Scripts\python.exe scripts/run_unsubmitted_demo_smoke.py --demo-config config/local/demo.yaml --run-id DEMO_NATIVE_RECOVERY_SMOKE_V001 --operator USER_REQUEST --fresh-quotes --recover-validation-abort
```

This second command may submit actual demo entry and verified owned exit requests.
It tests execution lifecycle only, not the trading strategy. Failure or unresolved
exposure stops progression; no blind resubmission. The supplied risk budget is a
scenario allowance, not a guaranteed realized loss ceiling.

Broker-free verification of the completed observation:

```powershell
.venv\Scripts\python.exe scripts/verify_demo_continuation.py --demo-config config/local/demo.yaml --output results/demo/DEMO_CONTINUATION_VERIFICATION_V003.json
```

V001/V002 already exist. The verifier is deliberately bound to this completed
observation and its current checkpoint; after a new native run, a new matching
verification specification is needed. It verifies frozen records, full journal,
risk checkpoint, retained financial history across rearm, capture manifest,
recomputed observed-table equality and saved registry evidence. Its limited
before/after access-log comparison covers verification only. It never connects
to MT5 or reads historical market partitions. It preserves negative conclusions.

Offline validation passed 288 targeted tests, full Ruff and typing for 264 files.
Two deliberately disabled guards were detected in isolated in-memory tests.
Actual command records are in WORKLOG.md. No Stage19 work,
strategy activation, real-money execution or deployment occurred.

The subsequent user-authorized continuation performed a longer read-only capture
and independent inference checks on two saved Stage13 fits. No broker precheck or
submission was attempted outside the configured session. Nine reconstructed 5m
forecasts matched calculations; 15m remained untested. The complete model ensemble,
eligible portfolio and live trading path remain blocked. See
[actual forecast diagnostic evidence](stage18_forecast_diagnostics.md).
