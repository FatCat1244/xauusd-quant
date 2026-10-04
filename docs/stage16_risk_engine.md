# Stage #16: authoritative offline risk engine

Human-readable risk specification: RISK_SPECIFICATION_V001.

Actual operation is **UNCONFIGURED / BLOCKED / NO_ELIGIBLE_ALPHAS**. The software
governs synthetic Stage #15 portfolios through the existing Stage #12 account.
There is no evidence-supported trading strategy, supplied production risk policy,
broker connection, live specification lookup, deployment or order transport.
Prompt #16 authorizes this offline extension only. Stop before Stage #17.

## Evidence inspected

The tree was clean on `main` at `967fdf3` before editing. Read AGENTS.md, CLAUDE.md,
WORKLOG.md and Stage #12-#15 implementations, documentation and saved reports.
Stage #15 registry `ALPHA_REGISTRY_V003` retains 66 entries, none scientifically
eligible. Its saved inactive verdict says `NO_ELIGIBLE_ALPHAS`; the saved final
verification reports 334 synthetic fills, 162 trades and reconciled costs.
Those prior studies were inspected, not reproduced. Stage #14 still supplies
no eligible candidates. Stage #13 V002 has actual ridge/mean fits; their short
negative historical diagnostics do not establish eligibility. Stage #13.5 has
causal estimates, intervals and matured-error CUSUM diagnostics, not validated
directional trading policies. No earlier historical conclusion was rewritten.

Saved metadata, result identities and the four existing reserved access-log hashes
are bound into each risk manifest. Previously inspected 2022+ is not fresh evidence.
No tick/forecast/feature values or reserved outcomes are read by these commands.
Real replay requires matched prior-only nulls, adequate economic coverage,
fold-local eligibility, actual policy clocks and supplied execution/account terms.
Unknown historical selection exposure remains unknown; risk software cannot fix it.

## Boundary and interfaces

`risk/contracts.py` supplies explicit account, instrument, market, account,
health, request and execution-update contracts. `risk/policy.py` strictly validates
configuration. `risk/engine.py` separates policy, risk state, snapshots, decisions
and audit. `risk/portfolio.py` overrides Stage #15 target routing and installs the
mandatory boundary in Stage #12's engine. Existing research APIs remain historical
baseline tools; a governed account refuses their raw target calls.

Strategies propose signed lots and reconciled sleeve intentions. The risk layer
owns final quantity rounding. Each decision carries an immutable request identity,
portfolio/allocation/alpha identities, event and receipt times, policy/configuration
hash, input hash, measured actual/reserved/gross exposure, configured limits,
approved target/change, reason codes, transitions and unresolved actions.

Only a current exact approval with a matching reserved quantity can submit an order.
The target call requires the adapter capability; the submission guard checks the
reservation again. Fills recheck quote validity, current loss state, health,
position absence, margin and conditional loss/notional reservation. An adverse
price jump requires a new approval, rather than enlarging a previous reservation.
This is an application boundary against accidental bypass, not security against
malicious Python code modifying private process attributes. There is no broker API.

Supported decisions: APPROVE, RESIZE, REJECT, HALT_NEW_EXPOSURE, REQUEST_CANCEL and
REQUEST_REDUCE_OR_FLATTEN. Priority is missing/uncertain account and configuration,
pending exposure, hard state/health/session gates, sizing/exposure/margin/rate caps,
then model preferences. Verified closes use separate quote/spread rules. An entry
rejection never supplies a guessed exit quantity.

Event time and receipt time are distinct aware timestamps normalized to UTC;
future and stale snapshots are rejected. Out-of-order receipts or account/health
events retain a halt. Stage #15's deterministic weight/quote/alpha ordering remains.
No later 15-minute input revises an earlier 5-minute decision. Warm-up, eligible
status, actual forecast value/units/horizon/availability, matching model/feature
and allocation versions, calibration, data and feature health, volatility and
uncertainty must pass. A registry flag alone cannot satisfy these checks.
The diagnostic health field can carry a causally available Stage #13.5 matured
CUSUM result; missing required or negative health blocks. No diagnostic retrains
or changes strategies. Actual health adapters remain later-stage work.

## Units, budgets and sizing

`config/risk.yaml` explicitly leaves account, instrument and policy null. Missing
values never inherit Stage #12's hypothetical defaults. `config/risk_synthetic.yaml`
is a labeled test fixture, not recommended risk values or Exness terms.
Risk and execution units, quantity increments, contract size, conversion and all
costs must match exactly. Supplied terms cannot be paired with hypothetical terms.

Prices are USD per troy ounce; quantities are lots; contract size is ounces/lot.
Ledger currency and major currency are distinct. USD major has scale 1; an explicit
USD cent ledger has scale 100 and label USD_CENT. Non-USD conversion is a declared
major-currency/USD factor multiplied by denomination scale. This static offline
scenario is not a verified live FX feed. Unknown conversion blocks increases.
Only linear quote-side CFD P&L and absolute-notional-fraction margin are supported.
Broker tiered/dynamic margin, hedging credits and multi-instrument portfolios are
unsupported. The Stage #12 accountant remains authoritative; margin is a separate
conservative risk estimate, not invented broker margin accounting.

For long entry use Ask; for short entry use Bid. Stop-distance sizing requires a
supplied stop: future executable Bid for a long, Ask for a short. Price distance,
two adverse slippage allowances, exit stress, both commissions and declared
positive financing allowance produce conditional loss per lot in ledger units.
The stop must be correctly sided, finite and tick aligned. No stop is manufactured.
Stage #12 does not gain a stop-order execution model; the core sizing calculation
is independently tested, while current Stage #15 horizon policies use horizon sizing.

Horizon sizing uses a **declared hypothetical price-stress fraction**, current
spread, both slippages/commissions, extra adverse exit and maximum holding financing.
It describes a scenario loss, not a statistical VaR or a guaranteed maximum loss.
Budgets are absolute ledger amounts for account, position and alpha conditional
loss. Net and gross-intended caps are lots. Margin uses executable-side notional
times the declared fraction and conversion, plus entry commission and immediate
spread/slippage liquidation-mark loss. Quantities round toward zero to the
instrument step, then all sizing caps are checked again. A minimum lot that would
exceed budget is rejected. There is no leverage or P&L optimization.

Gross intended sleeve exposure is measured before opposing intentions cancel.
Netting occurs before orders. Sleeve budgets scale the whole proposed target;
no eligible sleeve is substituted to use freed capacity. Stage #12 has one net
position and one order, full fills, no pyramiding. Resizes and reversals close
fully before a separately approved later entry. Original attribution conventions
and all cash/fee/financing/turnover records remain in the Stage #15/12 engines.

## Reservations, emergencies and limits

Approval reserves exposure, scenario loss and margin **before** acknowledgement.
The full pending quantity is retained through retries, unknown outcomes and
unconfirmed cancellation; opposite orders receive no netting credit. Unchanged
healthy pending intentions retain their original approval without a second
reservation. Changed intentions request cancellation. The simulator confirms its
synchronous terminal outcome, reconciles the account, then makes at most one
immediate replacement attempt. A broker adapter must await actual acknowledgements.

Execution updates have idempotent event and intent IDs, cumulative filled quantity,
event/receipt times and order identity. Duplicate identical messages do nothing;
collisions, backward fills, overfills, late post-terminal events and unknown
outcomes halt for reconciliation. Terminal reports retain reservations until a
verified account agrees with the fills. Partial-fill reports are recognized but
the Stage #12 simulator does not generate partial fills. A partial report keeps
the full reservation and halts; a later terminal report plus verified residual
position can reconcile it. This is conservative offline interface validation.

Daily loss is the decline from the last verified adjusted mark at the daily
boundary; drawdown uses the persistent adjusted-equity high-water mark. Equity
includes cash commissions/financing and executable quote-side unrealized price
P&L, excluding unincurred future exit fees. Subtract verified cumulative external
cash flows before measuring trading changes. Where no midnight quote exists,
the prior verified mark is an explicitly approximate daily baseline; never use
the first post-loss new-day mark to erase the loss. The configured IANA daily
timezone determines reset. Midnight never clears a drawdown or manual halt.

Limits can block, request cancellation or request flattening. Sessions/weekdays,
overnight entry horizon, actual exposure and conditional open-risk breaches also
gate the account. Order and turnover windows are rolling receipt-time seconds.
Turnover is lots of approved change, conservatively booked even if cancelled,
until the window expires; it is distinct from actual filled account notional.
The order/rate/turnover limits also bound close attempts. A constrained exit stays
unresolved rather than violating the configured hard limit.

A reduction uses the verified actual position, cannot cross zero, and cannot
duplicate a pending close. Pending or uncertain fills block a guessed close.
Fresh valid quote sides are always required. The configured emergency rule may
permit a wide spread, but never an invalid/stale quote or unavailable account.
An exit request does not make the account flat. Gaps, failed execution or stale
data can exceed loss thresholds; the halt and unresolved exposure remain recorded.

## Persistent state and rearming

States are UNCONFIGURED, WARMING_UP, READY, HALTED, REDUCING and
RECONCILIATION_REQUIRED. Start unconfigured or requiring fresh reconciliation;
warm-up precedes READY. Limits/kill retain halt reasons. A permitted close enters
REDUCING; an uncertain order/failed account enters RECONCILIATION_REQUIRED.

Checksummed checkpoints retain configuration identity, daily baseline/key/loss,
HWM/drawdown, halts, reservations, terminal/order/event identities, decisions,
rate/turnover windows, last verified account, pending actions and audit sequence.
Governed checkpoints also retain the complete Stage #15/12 state and health clocks.
They do not mutate when restored runtime dictionaries change. Changed configuration,
bad hashes or incompatible state is refused. A restored engine always requires
reconciliation; it never defaults to READY. Preexisting severe halts additionally
need explicit operator identity and reason, fresh account state, resolved orders
and recovered limits. Rearming never resets HWM or loss history. Persistent
drawdown cannot be cleared by restarting or simply increasing a limit.

The simulator's internal account can reconcile synthetic replay; live account
reconciliation has not been validated. State is bounded to 10,000 decisions;
capacity exhaustion halts and stops instead of silently evicting idempotency IDs.
Long-running transport/storage, operator authentication and policy migration are
remaining operational validation requirements, not implemented broker capabilities.

## Reproducibility and actual commands

```powershell
xq risk-plan
xq risk-readiness --run-id RISK_READINESS_V001
xq risk-replay --risk-config config/risk_synthetic.yaml --run-id RISK_SMOKE_V002
```

Use a new version for each run. Existing IDs, configurations and plans cannot be
overwritten. Risk-readiness returns exit code 1 for BLOCKED; this is expected.
The frozen seven-scenario plan compares agreement, opposition, pending changes,
adverse price jump, loss halt, invalid feed and invalid health, each baseline/risk:
14 trials, at most 16 events each, 120 seconds and 1 GiB private-memory budget.
No threshold/model searches or coverage expansion. JSONL records flush per event;
failed and interrupted ledgers remain. Artifacts include the plan, schema, policy,
unit contracts, source/data/evidence hashes, configuration, requests/decisions,
account marks, actions, transitions, reservations, checkpoints, shared execution,
reconciled attribution, comparisons and readiness verdict.

The first completed smoke, RISK_SMOKE_V002, executed 14 trials in .8036 seconds,
304,599,040 own-process private bytes. RISK_SMOKE_V001 failed because a trial
output directory had not been created; its failed ledger remains. Initial tests
also exposed test-import/config-construction errors; these were repaired, not
counted as passing checks. Initial mutation audit detected 13/15; independent
checks exposed and repaired restored-checkpoint aliasing and reliance on secondary
submission checks. Its negative record remains alongside subsequent audits.
Final run/check measurements are recorded in WORKLOG.md.

Final source-bound `RISK_SMOKE_V004`: 14 completed trials, .7783 seconds,
303,198,208 private bytes. `RISK_READINESS_V003`: .3145 seconds, 302,190,592
private bytes, expected exit code 1 with UNCONFIGURED/BLOCKED. Final verification
reconciles 13 fills, 34 cash-flow rows, five closed trades and 92 risk decisions,
current source/frozen identities and unchanged access logs. The final targeted
regression has 270 passing tests in 11.91 seconds; all 19 disabled guards are
detected in isolated restored copies. Ruff and mypy are clean (243 source files).
No full suite or market-data research was invoked.

The synthetic agreement round trip earns approximately 3.4 ledger units after
costs in both variants; opposition is inactive. The adverse-jump baseline loses
approximately 16.6; risk refuses the pending fill. The loss-halt gap exceeds the
40-unit synthetic drawdown limit by much more: the risk exit realizes approximately
-200.6, including additional exit costs. In the invalid-feed case exposure remains
open and flattening unresolved. Rejecting invalid health also rejects a profitable
synthetic baseline trade. These hand-constructed examples validate behavior;
they establish neither market profitability nor improved unbiased performance.

Actual real-data replay is **not executed**. Required values remain the complete
policy fields reported by `risk-readiness`, verified account currency/denomination/
conversion, instrument contract/tick/quantity/margin/P&L conventions and commission/
financing/execution assumptions. No production limits are recommended here.
Stage #17 requires separate authorization for Exness feed/shadow validation with
no orders; Stage #18 requires separate demo readiness and authorization. Offline
software tests establish no live safety, profitability or deployment permission.
