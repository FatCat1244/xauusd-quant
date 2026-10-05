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
