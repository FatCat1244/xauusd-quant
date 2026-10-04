# Stage #17: read-only Exness demo feed and shadow architecture

Prompt #17 authorizes an explicitly configured Exness MT5 **demo** terminal,
read-only capture, causal streaming and local shadow simulation. It authorizes
commit/push of this stage after checks and records the same preference for future
completed implementation stages. Broker orders, deployment and automatic Stage
#18 remain prohibited. This is software validation, not demonstrated live safety
or profitability.

## Actual evidence and operating status

The initial working tree was clean on `main`, origin was the intended private
GitHub repository, and HEAD was Stage16 `898cae8452bef9499e589ae6d7576174b1c5f796`.
Stages12–16 documentation, source/interfaces and saved evidence were inspected;
market studies were not reproduced. The Stage15 `ALPHA_REGISTRY_V003` contains
66 candidates and **zero eligible alphas**. The actual risk configuration remains
`RISK_UNCONFIGURED_V001`. Stage14 matching null, chronology, adequate coverage
and supplied execution terms remain unresolved. No strategy has been selected.

The metadata inventory finds 37 frozen model specs and 13 frozen ensemble specs.
Historical finalized-artifact and streaming reports exist. Those historical
streaming checks copied stored regime/context values; they do not establish the
complete new-feed path. Saved `future_return` model definitions use volatility
scaling: a target name is insufficient to identify an output as log return.
Stage17 rejects such a directional binding without a compatible declared inverse
transformation. No existing market-model binary was loaded or promoted here.
Stage13.5 filters, intervals and change detectors remain forecast/health services;
they supply no automatic direction or green health flag.

`shadow-run` currently captures and runs **blocked feature/health diagnostics**.
It does not activate an unqualified model or invent a portfolio. Frozen-model,
ensemble and multi-alpha/risk binding interfaces are validated offline. Native
CLI activation of an eligible strategy is unavailable in this version because
eligibility, compatible live features/context, health and supplied risk/account
terms are absent. No complete actual-model streaming comparison has run.

## Read-only architecture

`shadow/adapter.py` exposes initialize/verify/ticks/latest/shutdown only. It
lazily imports the official vendor package, always initializes the configured
executable, and uses the already-authenticated terminal. It never calls login,
selects a different account, changes Market Watch or enables automated trading.
The exact symbol must already be visible. Identity verification matches terminal
directory, login, server and company and independently requires the vendor's
`ACCOUNT_TRADE_MODE_DEMO` value. Identity is rechecked on every retrieval and
reconnection. Server-name text alone never establishes demo status.

There is no broker-order method, generic native dispatcher, trading configuration
switch or order adapter. Tests trap unsupported vendor methods, including order
submission. Account metadata stays in memory; reports contain only an allowlisted
verification result and instrument metadata. Broker balances and positions never
initialize or masquerade as the simulated account. Other terminal software may
trade independently; this project cannot attribute that activity to itself.

`shadow/worker.py` runs native reads in an owned child with five fixed verbs. A
stalled read times out, stops that worker, and halts processing. Shutdown requests
detach the API; fallback termination affects only the owned Python worker, never
the user's terminal. No indefinite background service is started. The configured
observation duration is checked between calls; a call can add at most its ten-second
deadline plus bounded worker cleanup. Identity failures do not trigger account search.

## Time, ordering and completeness

MT5 source seconds/milliseconds are UTC. The historical OANDA CSV's broker-local
wall-clock conversion is not applied. Record source event milliseconds, local UTC
receipt, ingestion sequence, bar open/close and actual post-computation publication
time separately. Quote-age and receipt-lag measurements are diagnostics; without
clock synchronization they are not measured network or broker execution latency.

Retrieval uses `copy_ticks_from(..., COPY_TICKS_ALL)` with a bounded count and
overlap starting at the last source **second**. The cursor persists the ordered
hash list of every occurrence already accepted in that second. Identical quotes
and distinct ticks at the same millisecond survive. A repeated batch adds no new
occurrences. A changed/missing overlap prefix, reordered source sequence, invalid
price/time, or saturated single-second batch blocks processing rather than skipping
uncertain data. The API exposes no unique tick identifier or proof of completeness;
stable server occurrence order is an explicit assumption, not a proven guarantee.
Missing identical observations cannot always be detected from API fields alone.

The canonical `data.resampler.resample_ticks` builds open-labelled `[t,t+D)` bars
from recorded UTC ticks. Only the event-time watermark minus declared lateness
completes a bar; no wall-clock flush invents completion. Quotes on a boundary enter
the next bar. Empty intervals stay absent. Partial startup and gap-affected bars are
invalid; feature windows clear across invalid/missing slots. A late tick after
emission raises and retains a correction/halt indication; prior outputs remain
immutable. Source gaps, restart continuity and disconnects retain halts. This
conservative policy also halts across closures; resuming strategy decisions would
require a separately validated session/continuity policy and explicit rearming.

Backfill warms state only. Its original event time does not imply original live
availability. Backfilled quotes never enter local fill simulation; reconstructed
bars cannot issue retrospective live intentions. Excess publication delay and
already matured horizons block decisions. Receipt batches process available quotes
before forecast publications; a local order requires a strictly later eligible
receipt-time quote. Recorded publication clocks make replay compare the same
information sequence rather than replay-machine processing speed.

## Features, models, risk and local accounting

The bounded feature service reuses canonical returns, autocorrelation and
microstructure computations with registry-derived `min_history`. Selected finite
trailing windows are serialized; the default monitoring feature is `ret_1`, requiring
two consecutive complete bars. Time/regime, recursive volatility, spectral,
wavelet and interactions requiring unsupported live state are explicitly blocked,
not replaced with historical stores. This restriction prevents a false full-model
equality claim. Feed transformations alone do not prove Exness/OANDA transfer.

`FrozenModel` verifies pinned artifact manifest, frozen spec identity and existing
artifact hashes before loading preprocessing/calibration/model. Missing live
features are blocked before fitted imputers could make them appear available.
`FrozenEnsemble.load` delegates to the existing constituent/spec/feature-manifest
checks, leaves the existing ensemble intact, and refuses unavailable context.
Output units and horizon must match the alpha contract.

`Binding` connects an eligible alpha, frozen predictor, observed-bar policy and
explicit health provider. Synthetic bindings require the offline fixture path;
native commands cannot select it. Actual constructor gates also require prior
eligibility, feed-evidence identity and supplied risk terms. An evidence identity
is a reference requiring independent scientific audit; it is not proof of transfer.
Unavailable calibration, uncertainty, volatility or diagnostic inputs stay missing,
and the Stage16 health rules can reject them. Health providers must persist their
calibration/state and publish no later information.

The existing `RiskPortfolio` alone routes accepted intents to the Stage12 local
account; unvalidated raw targets remain rejected. Five- and fifteen-minute sleeve
intents retain their own horizons. Equal-time publications use stable alpha order
and one final target per sleeve. A holding-end and fresh same-direction forecast
can renew net intended exposure without a fictional filled round trip. Opposing
sleeves net before orders. Shared fills, costs and attribution remain authoritative
in Stage12/15; these are **local hypothetical fills**, never broker fills.

A direct Stage16 integration defect was repaired: newly available quote metadata
now reaches same-event risk checks before target reevaluation, after earlier expiry
timers. Previously a sparse stream could cancel a pending order using its previous
stale quote. The account mark uses that currently available quote. Strict later-fill
ordering, risk reservations and capability guards remain. Historical artifacts are
preserved; this repair does not rewrite their conclusions.

No assumed costs, commission, financing or slippage are labelled measured Exness
execution. Native metadata reports contract/volume/price fields in their native
units; it does not establish ounces, cent denomination, currency conversion, margin
or financing conventions. Actual terms must be supplied and matched before any
economic shadow account is enabled. Missing terms do not receive synthetic defaults.

## Persistence, health and recovery

Accepted batch commits include tick-occurrence and publication-clock hashes.
Append-only records flush and fsync before immutable numbered checkpoints. A
checkpoint binds code, full local configuration, cursor, bars, bounded feature
windows, frozen model identities, policies, diagnostic/calibration state and
governed portfolio/risk/execution state when present. Its audit-prefix byte count
and hash must reconcile to the preceding records. Cursor and pipeline sequences
must agree. Changed code/configuration/model state or truncated audit blocks restore.

Resume uses a new run ID and a committed checkpoint; it never overwrites the old
run. It adds a persistent restart-continuity halt. Risk restoration retains loss,
reservations and halts and requires reconciliation; reconnect never rearms it.
Storage failure, overload, invalid clocks and stream failures stop processing.
Interrupted/uncommitted tails cannot support equality. Graceful shutdown detaches
the read adapter and closes records. No emitted reduction request proves a fill
or a flat account.

Buffers, event counts, batch sizes, retry counts, duration and backfill are bounded.
Records are capped at 64 MiB and checkpoint-directory storage at 256 MiB. Monitoring
records include empty/repeated polls, quote freshness, retrieval counts, processing
clocks, gaps, halts and shutdown status. Identity details, passwords and API-native
exception text are excluded. Config identity hashes and captures/checkpoints stay
local and must not be published as anonymized account data.

## Windows runbook

Use the existing project environment; no unrelated dependency upgrades are needed.
Official PyPI metadata checked on 2026-10-04 provides
`metatrader5-5.0.6231-cp314-cp314-win_amd64.whl`; this machine uses Python 3.14.7,
64-bit Windows. Vendor package availability does not prove terminal compatibility.
The optional `mt5` extra pins that version on Windows; offline tests do not require it.

If native support is needed, install only that package into this environment:

```powershell
.venv\Scripts\python.exe -m pip install --no-deps MetaTrader5==5.0.6231
```

Copy `config/shadow.yaml` to ignored `config/local/shadow.yaml`, then locally fill
`terminal_path` (the intended `terminal64.exe`, not an MQL5/Experts directory),
`expected_login`, `expected_server`, `expected_company`, and the exact `symbol`.
Use an already authenticated demo account with the intended symbol visible.
Do not enter passwords in YAML, source, CLI arguments or reports. This version
does not implement credential login; authenticate locally in the terminal.

```powershell
xq shadow-validate --shadow-config config/local/shadow.yaml --run-id SHADOW_READY_V001
xq shadow-preflight --shadow-config config/local/shadow.yaml --run-id EXNESS_PREFLIGHT_V002
xq shadow-capture --shadow-config config/local/shadow.yaml --run-id EXNESS_CAPTURE_V002
xq shadow-run --shadow-config config/local/shadow.yaml --run-id EXNESS_SHADOW_V001
xq shadow-replay --shadow-config config/local/shadow.yaml --recorded results/shadow/runs/EXNESS_CAPTURE_V002 --run-id EXNESS_REPLAY_V002
xq shadow-compare --shadow-config config/local/shadow.yaml --recorded results/shadow/runs/EXNESS_CAPTURE_V002 --replayed results/shadow/runs/EXNESS_REPLAY_V002 --run-id EXNESS_EQUALITY_V002
```

Set duration/event limits in the local configuration before starting. Compare
requires the capture's unchanged configuration and code identity. For explicit
resume, add `--resume <committed checkpoint file>` to capture/run and use a new
run ID; the persistent continuity halt remains. Neither successful preflight nor
capture enables a strategy. On a closed market, stale quotes and historical
retrieval remain correctly labelled and do not count as a fresh live observation.

## Validation record

`SHADOW_SYNTHETIC_V002` executed the final frozen bounded study. Its fake read adapter
captured 36 ticks over 180 **synthetic** seconds, yielding two one-minute bars.
Recorded-feed replay matched those bars, monitoring features and blocked actions;
this is not an Exness recording. A separate fixed synthetic return mapping ran
26 ticks, four completed bars, four prediction records, 58 risk decisions and one
local hypothetical entry fill. Cash flows and sleeve attribution reconciled;
the position remained simulated, with no broker action. Flat-price and missing-
uncertainty controls produced no fills. No test requires profitability improvement.
Measured wall time was 2.36 seconds; peak working set 245,497,856 bytes and private
bytes 690,302,976. The earlier source-bound V001 remains preserved. No coverage
expansion or model/parameter search followed.

244 targeted Stage17/execution/portfolio/risk/reserved-isolation tests passed in
18.33 seconds on the final implementation. Eight deliberately broken guards were
detected in isolated copies; the original source was unchanged. The initial failed
mutation attempt remains recorded, including its temporary-directory errors and
masked availability canary. The strengthened canary independently checks the first
publication. Final checks are recorded in WORKLOG.md and versioned local artifacts.
Ruff and mypy cover the repository; the full historical test suite was not run.

No native connection or live observation is established by the synthetic study.
The user supplied an Experts directory, which does not specify expected account
identity or the executable. An ignored unconfigured local template was created;
at initial delivery native observation remained dependent on that configuration
and optional-package availability. Actual model equality, eligible strategy replay,
broker-specific economics and Stage18 readiness remain blocked. Reserved historical
access logs and prior failed research are preserved.

### Subsequent configured-terminal check, 2026-10-05 local time

After the user supplied local identity values, they were moved from the tracked
template to ignored `config/local/shadow.yaml` and the public template restored.
The exact supplied company identity passed native verification; private account
values are not included here. The pinned optional package was installed without
changing other dependencies. Initial installation was blocked by sandbox socket
permissions; the authorized retry succeeded.

`EXNESS_PREFLIGHT_V001` verified the configured terminal/account/symbol and vendor
demo mode. `EXNESS_CAPTURE_V001` then completed the configured 60-second budget
(57.78 seconds in the measured capture section), with 57 retrieval polls, no
reconnections, **zero ticks and zero bars**. All polls reported a non-fresh quote
and empty/repeated history. The adapter shut down at completion. Own-process peak
working set was 232,964,096 bytes and private bytes 666,083,328; worker memory is
not included. Source identity remained
`4b1b03de3cfbc58b657770ad3e2383af73a663c6d7e554913b276b5c236b0184`.

`EXNESS_REPLAY_V001` and `EXNESS_EQUALITY_V001` completed on that recording. The
compared observation tables are empty: equality is therefore uninformative about
actual tick/bar/feature/model processing. Fresh live observation, actual model
and full pipeline equality remain **unvalidated**. No market-closure cause was
established from this result. `SHADOW_TERMINAL_READINESS_V001` records configured
identity and installed native support while strategy readiness stays BLOCKED:
NO_ELIGIBLE_ALPHAS, RISK_UNCONFIGURED and FEED_TRANSFER_UNASSESSED. No shadow fill
or broker order occurred. All runtime artifacts remain ignored/local.

The next data check is another bounded capture with a fresh run ID when quotes
are updating. Even non-empty capture/replay equality will not qualify an alpha or
supply missing risk/account/cost settings. Strategy activation and Stage18 remain
separate blocked gates; do not weaken them to generate signals.

Official primary references checked for API/time/account semantics:
[MetaQuotes Python integration](https://www.mql5.com/en/docs/python_metatrader5),
[initialize](https://www.mql5.com/en/docs/python_metatrader5/mt5initialize_py),
[tick retrieval](https://www.mql5.com/en/docs/python_metatrader5/mt5copyticksfrom_py),
[UTC tick ranges](https://www.mql5.com/en/docs/python_metatrader5/mt5copyticksrange_py),
[account information](https://www.mql5.com/en/docs/python_metatrader5/mt5accountinfo_py),
[account trade modes](https://www.mql5.com/en/docs/constants/environment_state/accountinformation),
[terminal info](https://www.mql5.com/en/docs/python_metatrader5/mt5terminalinfo_py),
[symbol info](https://www.mql5.com/en/docs/python_metatrader5/mt5symbolinfo_py),
[vendor package metadata](https://pypi.org/pypi/MetaTrader5/json).
