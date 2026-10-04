# Stage #15 architecture specification V001

**NO_ELIGIBLE_ALPHAS. No evidence-supported trading strategy has been selected.**
The market portfolio remains inactive. This document specifies tested offline
architecture and explicitly synthetic examples, not proposed profitable entries
or permission to trade. No forecast ensemble is changed.

The registry binds each policy to its actual model, features, data, horizon, validation
and Stage #14 evidence. Missing matching prior-only nulls, economic evidence or
execution terms or incomplete policy clocks blocks eligibility. Current blocked
market hypotheses have no selected maximum staleness bound. A later scientific verdict cannot be used as
an earlier fold's known universe. Forecast-only variance, interval, health and
residual probability outputs supply no direction. Historical 2022+ results were
already inspected; this layer has no final-test reader.

A directional sleeve consumes an expected future log mid-return at its own
frequency, after its bar closes and declared computation/publication delay.
The existing fixed sign policy emits +1, -1 or zero unit-budget intent, retaining
timeframe and forecast/holding horizon. Each sleeve exits on its own h-th
subsequent observed bar. Its maximum wall-clock age is separately required; a
session gap does not imply knowledge of future observed bar timestamps. The
adapter rejects inconsistent units, horizons, incomplete bars and invalid data.
Overlapping valid forecasts do not reopen an active intent. Invalid information
withdraws the intent immediately; expired intent contributes zero. Health
diagnostics may be supplied as explicit invalidation events, not trading direction.
Policies and the account carry state across chunks and serialized restarts.

Equal allocation assigns 1/N of a fixed research budget to each frozen eligible
sleeve. An inactive sleeve contributes zero; its budget is not redistributed.
Single-alpha allocation is 100% of that budget and makes no diversification claim.
An empty allocation is zero. There is no fitted signal scaling. The alternative
minimizes w'Cw with nonnegative weights summing to one, for at most three sleeves.
It uses aligned 5-minute fixed-capital standalone net returns known strictly
before the fold. The preceding 32 training observations, with at least 16
complete rows, use a fixed half-sample/half-diagonal covariance. Invalid,
insufficient, singular or failed estimation uses recorded equal fallback.
There are no expected-return optimizer inputs, leverage optimization or Kelly
sizing. Covariance describes sleeve counterfactual risk, not independent assets.

The frozen research exposure budget is .02 hypothetical lots. Actual contract
size and costs come from the explicitly supplied Stage #12 execution configuration;
current repository terms remain hypothetical. Weighted signed intents are summed
before orders, then rounded toward zero to the declared .01 lot increment.
There is one shared CFD account, cash ledger and net position. A change in size
or sign closes the whole position before opening the replacement. This conservative
convention adds turnover compared with a partial resize; it is fully recorded.
Production partial-fill and margin support are absent.

At identical timestamps: expired intents are withdrawn, the new frozen allocation
is applied, quotes retain sequence order, then intents use stable alpha-ID order.
An order needs a strictly later quote than its latency-adjusted arrival and before
its exclusive expiry. Signals at a quote timestamp cannot fill on that quote.
An entry waiting for a changed target is cancelled; an exit already pending is
retained. After expiry or a gap rejection, the adapter retries the current target
at the next event, with a new arrival/TTL. After an exit fill, reentry cannot use
the same quote. Missing quotes never create fills. Actual exposure can persist
after intent expiry until liquidation succeeds, with financing still accruing.
At the run cutoff, orders are unfilled and existing exposure is retained and
marked under Stage #12's liquidation-side convention.

Portfolio accounting is authoritative. Spread/slippage are already included in
executable-price P&L and are decomposed once. Sleeve attribution assigns each
actual entry to same-direction supporting weighted intents, proportional to their
absolute contribution at the fill. Those owner shares remain frozen for all
position cash flows, costs, turnover and unrealized P&L until close. Opposing
intents and cancelled exposure remain visible but receive no share of that trade.
This convention is non-unique and is not a causal contribution estimate. Every
allocated metric and cash change reconciles to the account. Standalone comparisons
are separate counterfactual engine runs, never additive portfolio equity.

Worked examples below are **synthetic** with equal allocations and .02 lot budget:

| Inputs/event | Combined target | Shared-account behavior |
|---|---:|---|
| A=+1, B=+1 | +.02 lots | One net long request; both owners share actual trade equally |
| A=+1, B=-1 | 0 | No executed offsetting sleeve orders; pending entry cancelled |
| A=+1, B expires or is invalid | +.01 lots | A half budget remains; existing .02 long closes before .01 reentry |
| Pending +.02 entry, both turn -1 | -.02 lots | Cancel old entry, submit short with new latency; use only a later quote |
| Existing +.02 long, exit pending, target turns -.02 | -.02 lots | Keep exit; request replacement short only after actual exit |

There is no deployable system selected on current evidence. Consideration for
Stage #16 requires matched prior-only nulls, adequate chronological economic
coverage, valid fold-local eligibility and explicit execution/account evidence.
Stage #16 must supply production position/risk limits, loss/margin controls,
operational kill behavior and sizing validation. Stage #17 supplies broker and
shadow validation only if separately authorized; demo orders require Stage #18.
No broker connection, deployment, commit, push or later-stage work is authorized
by this Stage #15 implementation.
