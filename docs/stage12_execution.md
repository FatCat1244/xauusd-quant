# Stage #12: offline execution and trading economics

This is the preserved Stage #12 record. Later prompts authorize the offline
Stage #13/#13.5 and [Stage #14](stage14_robustness.md) extensions only.

Prompt #12 explicitly authorized this offline layer on 2026-10-03. No broker,
MT5, demo/live trading or deployment is implemented or authorized. Stop before
Stage #13. Nothing selects features, models or policies by trading performance.
ML and ensemble prediction contracts remain unchanged; decisions live in
`xauusd_quant.execution` only.

## Interfaces and commands

Use `xq` or `.venv\Scripts\python.exe -m xauusd_quant.cli`:

```powershell
xq execution-smoke --run-id EXEC_SMOKE_V003
xq execution-readiness --run-id EXEC_READINESS_V003
xq execution-diagnostic --run-id EXEC_QUOTE_PROBE_V003 --quote-probe --start 2021-06-01T00:00:00Z --end 2021-06-02T00:00:00Z --max-quotes 50000
```

The smoke is a declared synthetic long round trip. Readiness returns exit code
1 when evidence is incomplete, with its report on disk. A quote probe exercises
bounded streaming and measures resources; it contains no forecasts or economic
conclusion. These commands create new directories under `results/execution`;
an existing run ID is refused, including an interrupted run. Use a new version.
They do not change raw data, historical results, manifests or fitted artifacts.

Configured backtests require a separately supplied forecast provenance sidecar.
The command below is an interface example, **not a run executed for this stage**;
`forecast_metadata.json` does not currently exist in the project:

```powershell
xq execution-backtest --run-id HISTORICAL_5M_V001 --execution-config config/execution.yaml --forecasts results/ensemble_research/5m/joint_predictive_state.parquet --forecast-metadata forecast_metadata.json --historical-diagnostic --scenario-set config/execution_scenarios.yaml --start 2021-06-01T00:00:00Z --end 2021-06-02T00:00:00Z
```

`execution-diagnostic` exposes the same configured engine as `execution-backtest`.
Without `--historical-diagnostic`, unresolved gates refuse execution before tick
or forecast values are opened. The flag permits a labelled historical diagnostic,
never promotion or a broker-specific conclusion. The resource probe is a separate
explicit path and takes no forecasts. All real input intervals are bounded to
development before broker-local 2022-01-01; **no new reserved outcome loader** is
introduced. Configured economic intervals are capped at 366 days. State survives
all chunks and months within one run; splitting runs creates separate simulations
and must not be presented as a continuous multi-year portfolio.

The adapter reads only `timestamp`, `expected_return` and
`status_expected_return_vol_scaled` from the existing flattened contract. It
filters the selected development interval before collecting forecast values.
The bounded forecast interval may be materialized; ticks are streamed from
selected Parquet row groups in `batch_rows` batches. Timestamp-only bar scans
are per year, also bounded. No whole tick history is required in memory.

Required sidecar keys:

| Key | Meaning |
|---|---|
| `forecast_id`, `forecast_sha256` | Declared forecast identity and SHA-256 of the actual Parquet file |
| `timeframe`, `target`, `units`, `horizon_bars` | Must match the fixed policy: `future_return`, `log_mid_return`, configured timeframe/horizon |
| `timestamp_convention` | Exactly `broker_local_bar_open`; close availability is derived, never bar-open availability |
| `dataset_version`, `bar_dataset_version` | Must match the independently inspected local manifest identities |
| `feature_set_ids` | Actual feature manifest IDs; historical V001 identities cannot pass chronology through a new assertion |
| `model_provenance`, `feature_provenance` | Actual frozen specifications, fitted artifacts, feature identities and hashes; incomplete references stay diagnostic |
| `evaluation_history_classification` | `historical_inspected`, `untouched_verified` with supporting evidence, or unknown |
| `evidence_files` | Optional repository-relative path -> SHA-256 mapping, verified before execution |
| `evidence_roles` | References into `evidence_files` for `model`, `features`, `chronology`, `nulls`, `specification`, `evaluation_history`; each is a nonempty list required for evidence promotion |
| `folds` | Optional per-fold audit records: `evaluation_start_utc`, `evaluation_end_utc`, `adaptive_as_of_utc`, `nulls` |

`adaptive_as_of_utc` must include feature identities/count, model fitting, preprocessing, tuning,
calibration, policy, eligibility, universe, correlation screening and weights.
Each choice must predate its scored fold; missing records are unknown. Random-walk
and sign-flip null verdicts must both be passed, with prior-information provenance.
Validating supplied records is not an independent reconstruction of an unpublished
scientific audit. All current local development outcomes have already been used;
existing access history cannot be erased by a sidecar claiming untouched data.
Audited fold intervals must cover the entire requested run without gaps. File
hashes establish identity, not the scientific truth of the supplied records;
roles identify the evidence requiring review rather than accepting an arbitrary
hashed file as sufficient proof. Manifests include both resolved execution and
data configurations with hashes, dependency versions, and source hashes including
uncommitted files.

Programmatic interfaces are `Forecast.from_bar`, `Quote`, `BarClose`,
`ExecutionEngine.consume`, `ExecutionEngine.finish`, and `MemoryRecorder` for
small fixtures / `DiskRecorder` for incremental output. `bar_index` is a contiguous
observed-bar ordinal shared by forecasts and close events. A chunk must call
`consume`, not `finish`; finish is the declared run cutoff.

## Forecast meaning and reference policy

Every target's horizon is its configured number of **subsequent observed bars**,
not necessarily the same amount of wall-clock time across closures/missing bars.
Availability is the input bar's close plus declared computation delay.

| Target | Meaning / fitted units | Reference policy support |
|---|---|---|
| `future_return` | Expected future log mid-close return; raw model predictions can be volatility-scaled by trailing sigma and sqrt(h) | Consume only the contract's reconstructed `expected_return` in log-return units |
| `mean_reversion`, `mean_reversion_half` | Probability residual magnitude shrinks / halves | Rejected as a direction input |
| `residual_reduction` | Residual reduction divided by trailing volatility | Rejected; not executable price profit |
| `direction_up_cost`, `direction_down_cost` | Probability return exceeds a trailing spread proxy | Rejected; proxy does not model an actual filled round trip |
| `future_volatility` | Expected ln(realized volatility + floor) | Rejected; no direction; exponentiation is not an arithmetic mean |
| `future_abs_move` | Expected ln(abs(log return) + floor) | Rejected; magnitude without direction |

`SIGNED_EXPECTED_RETURN_V001` is fixed: positive expected log return -> long,
negative -> short, zero -> neutral, fixed lot quantity, exit after that forecast's
h subsequent observed bar closes. No cost threshold, direction inferred from
volatility, residual-to-profit mapping, fitted threshold or policy search exists.
Overlapping signals are explicitly rejected. Delay beyond an already completed
horizon is rejected. This sign policy exercises execution; it is not an optimal
policy or evidence of positive net economics. Forecasts refer to mid closes;
the delayed executable prices can have very different economics.

## Event timing, quotes and gaps

Bars use the actual `[t,t+D)` / open-label / left-closed repository convention.
Quote events precede bar-close events at identical timestamps. Equal-time quotes
retain the converter's stable partition order and their stream ordinal. A
close-time forecast becomes usable after that bar completes. Computation delay
and venue latency use an explicit UTC clock.

An order fills only when `arrival < quote_time < expiry`, using the **first**
eligible fresh valid quote. A quote at arrival, even a later sequence with the
same timestamp, cannot fill. No favorable window search, interpolation or use
of an earlier cached quote for execution occurs. Expiry is exclusive and its
clock advances across missing data. Invalid/crossed/nonpositive/nonfinite quotes
and stale/future-dated quote observations are rejected and counted.

A gap above `max_gap_ms` cancels pending orders as unfilled, including exits.
An open position is retained, financed and marked when fresh quotes resume;
the engine does not invent a gap-closing fill or retry an expired exit. Bar
closes count only nonempty observed bars. Future bar prices/labels never enter
this schedule. At an interval cutoff, pending orders are unfilled and exposure
is retained. The final quote may be too old to mark: equity is then unknown,
not filled forward. Gap types are conservatively grouped as session/data gaps,
not attributed to a measured session classifier.

Source timestamps remain broker-local wall time. Stored `timestamp_utc` is checked
against the repository timezone conversion, not reinterpreted by attaching UTC
to the local clock. The actual convention is America/New_York wall time +7 h,
UTC+2/+3 on US DST dates. Winter/summer and US/Europe DST-disagreement dates are
tested. Real DST transitions fall in the documented weekend closure; unknown
derived timestamps fail clearly.

## Costs, positions and reconciliation

One net CFD-like position at a time; market orders with assumed full fills.
No order-book depth, measured fill probabilities, margin calls, partial fills,
limit orders, stops or take-profits are implemented. They are deferred, not
approximated by guaranteed prices. No portfolio allocation or risk engine exists.

Long entry pays Ask, exit receives Bid. Short entry receives Bid, exit pays Ask.
Adverse deterministic slippage increases a buy price and decreases a sell price.
Prices are USD per troy ounce; contract size is ounces per lot; quantity is lots.
Price PnL is multiplied by lots x ounces/lot x account-currency/USD conversion.
Commission is account currency per lot **per leg**, independent of that conversion.
Bounds and quantity increments are validated. Non-USD conversion is a declared
constant scenario, not a historical FX series or a measured conversion fee.

`config/execution.yaml` contains completely declared **hypothetical** quantities,
contract size, commission, slippage and funding. It is not Exness terms or an
account specification. Missing required cost/unit/funding fields in a supplied
configuration fail; no financing rate silently defaults to zero. Funding is a
continuous elapsed-UTC-day charge per lot, separately declared for long and short,
including overnight/weekend exposure. Negative rates are credits. This is not
broker rollover/triple-swap accounting. A verified supplied specification is
required before a broker-specific economic conclusion is eligible for review.

The three registered sensitivity scenarios are base, a commission/slippage-free
cost control, and longer latency. They share the same sign policy and observed
Bid/Ask. The cost control still pays spread. Every scenario is recorded before
any quote/forecast values are consumed, and runs sequentially. No winner is selected.
Do not infer monotonic aggregate PnL from latency changing which trades fill.
Matched-trade cost arithmetic has its own deterministic test.

For a closed trade:

```text
gross_price_pnl = signed executable-price difference x contract/quantity/FX
net_pnl = gross_price_pnl - both commissions - accrued financing
matched_midpoint_pnl - two half-spread costs - adverse slippage = gross_price_pnl
```

Spread and slippage are already in executable-price PnL: decomposition columns
must not be subtracted from it a second time. Midpoints use the actual matched
fill times and are non-executable counterfactuals. An additional immediate-midpoint
counterfactual uses the last fresh quote at each decision/exit-request time; if
either is unavailable, it is null. Its difference from the matched midpoint
benchmark measures delay for that matched population, not a strategy optimized
with immediate execution.

Cash changes by fees, financing and realized price PnL; entry/exit notional is not
credited/debited as if buying physical gold. Realized net PnL includes all closed
trade costs. For an open trade, cash already includes entry commission/funding,
and unrealized price PnL uses Bid for long liquidation or Ask for short liquidation.
Cash = initial cash + closed net PnL - paid open costs. Equity = cash + unrealized
price PnL. Marks exclude prospective exit fees/slippage; they are explicit quote
liquidation marks, not guaranteed final net proceeds. Reconciliation is checked
at every quote mark and at the cutoff.

Records: `decisions`, order submitted/terminal events, `fills`, `positions`,
`cash_flows`, sampled `equity`, `trades`, and `quote_rejections` JSONL files.
Files are created once and each row is flushed; completed records survive an
interruption. Full-run restart/resume of persisted engine state is deferred.
New run IDs are required for a replay; historical directories are not overwritten.

Reports contain orders/fills/rejections/expiry, gross/net PnL, matched midpoint and
cost decomposition, turnover in account currency, lot-seconds exposure, open
exposure, monthly closed-trade PnL and drawdown over known marks. Equity samples
use a fixed millisecond interval, while drawdown is tracked at all quote marks.
Stale unpriced intervals are unknown and limit drawdown coverage. Return is cutoff
marked-equity change / initial cash, without leverage assumptions, annualization
or Sharpe. Regime attribution is deferred because no verified causal regime
provenance was supplied to this execution path.

## Audit and actual local evidence

The Stage #12 metadata audit verified **281 actual monthly files**, matching
729,244,369 footer rows and manifest sizes, calendar-month completeness, every
partition's footer timestamp extrema, and reconstructed tick-manifest identity
`ticks-2e173ef8e61bd240`. Actual first/last rows were also inspected:
2003-05-05 03:01:03.421 -> 2026-09-18 23:59:59.079 broker-local. This is the full
dataset, not the archived 2003-2004 partial export. The 36,233,955,746-byte raw
file exists; it was not changed or hashed again. This metadata/edge check is
**not** a new full scan of all quote values or partition content digests.

50 primary frozen specs (37 ML +13 ensemble), 119 fitted artifact manifests with
their recorded payload hashes, and 32 feature-manifest content hashes checked
successfully. Four existing final-test access logs were inspected as history,
without rerunning a final test or reading reserved regression/target values.

Audit repairs:

1. `RegressionFeatureStore.load(..., before=...)` cuts the base scan and slices
   the row-aligned timestamp-free window scan **before collecting**. Both ML
   development loading and ensemble slim development loading pass the boundary.
   Unbounded consumers retain their existing interface and values.
2. ML prior-access detection includes matching `final_test_started` entries,
   including interruptions. Repeat attempts need the existing logged reason.
   The ensemble already had a started-event guard; it now uses the common
   matching-attempt helper without double-counting starts.
3. An unavailable/undefined registered null makes ensemble eligibility untested;
   a passing null cannot conceal an undefined or missing required null variant.
   Historical eligibility tables and frozen specs are not rewritten.

Unresolved scientific prerequisites:

* Stage #9 identities use development through 2017; Minimal/Standard counts use
  2018-2021 validation plateaus. Those choices are later than early ML folds.
  Per-fold model preprocessing/fitting/calibration being chronological does not
  repair globally selected features. The existing ML scores are historical
  diagnostics, not clean end-to-end OOS selection evidence.
* Stage #11 global eligibility and correlation screening use all five blocks.
  Its `*_wf_universe` variant retains global null/structural statuses, so it is
  also not a complete chronology repair. Weight fitting on earlier blocks does
  not undo outcome-dependent constituent selection.
* 2022+ was inspected by feature research, ML and ensembles, including an
  interrupted attempt. It is not untouched. No fresh uninspected outcome period
  is independently established, and no 2022+ test was rerun for Stage #12.
* Current costs are hypothetical. No verified broker/account specifications or
  audited execution forecast sidecars were supplied. Existing fitted artifacts
  are present; absence is not being inferred from documentation.

Readiness `EXECUTION_READINESS_V001` marks these unknown/failed gates and assigns
`historical_diagnostic_only`. Minimum repair is fold-local feature identities
**and counts**, with nested preprocessing/tuning/calibration, plus eligibility,
null evidence, universe/correlation screening and weights using only prior
permitted data. Reconstruct with new versions; preserve existing records. Any
new period needs proof that its outcomes remained uninspected. These research
reruns are not dependencies for the implemented synthetic execution engine.

Actual runs and final verification are recorded in `WORKLOG.md` and the local
versioned `results/execution` outputs. No real forecast economics or profitability
conclusion was produced for this stage. Previously documented accuracy and
ensemble results were not reproduced.
