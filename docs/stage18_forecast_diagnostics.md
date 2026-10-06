# Stage18: bounded recorded-feed forecast diagnostics

This continuation responds to the user's request to complete the remaining work.
The configured entry session was closed, so no execution attempt was consumed.
The original native ValueError remains unresolved. Existing demo quantity, financial
limits, session and risk checkpoint were unchanged; shadow remained read-only.

The study improves evidence about a small existing forecast path. It does not select
a trading strategy or establish predictive value. All 66 Stage15 candidates remain
BLOCKED, and strategy-specific risk configuration remains unavailable.

## Frozen software study

`scripts/diagnose_recorded_forecasts.py` reuses Stage13's `source.features` and
`FrozenPipeline.predict`; it never fits, creates policy intents, opens MT5 or reads
historical market partitions. Both existing CAUSAL_RIDGE fits from F02 were pinned:
5m and 15m, last chronological fold rather than performance-selected fold. Original
run manifests and parameter-file hashes are retained. Their historical scientific
limitations and earlier outcome inspection remain unchanged.

Features are Stage13 return_1, momentum_3 and volatility_3 from four consecutive
valid completed midpoint closes. Output is one-subsequent-observed-bar log return.
The loader validates finite coefficients/positive scales, feature identities,
training-label maturity and inner-fit cutoffs. Invalid, duplicate, unavailable or
gap-affected bars reject or reset warm-up; future fitting cannot score earlier bars.
Four-close state is bounded, serialized and identity-checked on restoration.

The V001 plan was frozen before the new capture: two fixed fits, no refitting or
model search, one 300-second read-only observation, at most 30000 ticks/256 forecast
bars, one hour of bounded warm-up retrieval. No financial settings were copied into
the diagnostic. An ignored `config/local/shadow_forecast.yaml` copies configured
demo identity with those observation limits; original shadow/demo/risk files remain
unchanged. Raw quotes and all reports stay local under ignored results/runtime.

Reconstruction happens after capture, not inside the live forecast pipeline.
Rows preserve recorded input availability and separately report actual calculation
time. Forecasts on backfill are labeled reconstructed diagnostics and cannot issue
retrospective live actions. No outcomes, forecast losses, P&L or significance claims
are calculated. An actual live-inference binding still needs validation.

## Actual observations and calculations

| Check | Actual result |
|---|---|
| EXNESS_FORECAST_CAPTURE_V001 | 299.835640 seconds; 8502 committed ticks; 7968 backfill, 534 nonbackfill, 529 fresh |
| Retrieval | 264 polls, 249 committed batches, no reconnects or continuity halts |
| Bars | 13 at 5m,4 at 15m; all retained backfill flags, 2 invalid partial startup bars |
| Health | 15 empty/repeated-batch notices, 3 non-fresh-quote notices; no processing halt |
| Observed baseline replay | All 17 bar, monitoring-feature and NO_ACTION records matched; zero mismatches |
| Saved 5m ridge inference | Nine finite reconstructed forecasts; batch/incremental maximum difference 3.3881317890172014e-21 |
| Saved 15m ridge inference | Zero forecasts; only three consecutive valid closes, requires four; UNTESTED |
| Serialization/prefix checks | Exact equality across two-bar chunks/restarts and prefix replay |
| Orders/local fills | Zero; no portfolio or risk intent generated |

One 5m bar completed during observation contained both retrieved history and newly
received ticks and retained its backfill classification. The short segment does not
prove full live feature/model/ensemble equality. Baseline comparison used existing
float32 monitoring tolerances; independent Stage13 prediction arithmetic used
rtol1e-12/atol1e-14. Batch inference computes the same canonical features on each
full available contiguous prefix, with vectorized scaling/coefficient arithmetic;
incremental inference uses only four closes and the original predictor.

Own-process peak working set 245764096 bytes, private 709836800 bytes; worker memory
excluded. Capture/replay and inference used native source identity
`caece6af9b565f7b1450a01fecf2046b6c1929bc4e854b0a76077db82b5c2e28`.
No source module changed. Diagnostic script hashes are pinned separately.
All native workers stopped before final code/documentation/Git operations.

V001 called the observed software result FORECAST_DIAGNOSTICS_PASSED while leaving
15m absent from its finite-count mapping. Reporting plans V002/V003 explicitly
retain zero counts and UNTESTED status per timeframe, and classify the result
PARTIAL_FORECAST_DIAGNOSTICS. This revision followed V001 inspection; it changes
coverage disclosure, without changing fits, inputs, numerical acceptance or any
scientific gate. V002 also exposed a relative-path CLI display error after its plan
had already been saved; V003 fixes the display. All original plans/reports are
preserved, along with the actual V001/V002 script sources. No second capture ran.

The final V003 report verifies unchanged scientific metadata and reserved access-log
hashes across the study. This is an identity/access-log check, not proof that earlier
history was uninspected. No historical reserved rows or model binaries were loaded.

## Model readiness and remaining work

The audit inventories 37 frozen ML specs and 13 ensembles. All 50 require feature
services outside the current validated live returns/autocorrelation/microstructure
service. Maximum declared feature history is 7520 bars. The four directional-return
specifications use volatility-scaled units, needing a separately justified inverse
transformation; other targets are not automatically directional policies. Thirty-seven
model manifests are present at the standard artifact paths. Presence is not proof
of compatible preprocessing, context, live warm-up or feed transfer.

The two Stage13 fits offer software diagnostics with explicit log-return units;
their successful arithmetic cannot substitute for missing pipeline null controls,
historical execution terms, audited selection chronology or adequate evaluation.
Their parameters were trained on 2021 OANDA data; Exness transfer and current
predictive/economic usefulness remain unassessed. Nothing was promoted to alpha
eligibility or chosen based on these outputs.

Remaining gates: native broker precheck and a verified bounded demo entry/closure;
adequate 15m and other required warm-up coverage; actual live model/ensemble and
portfolio/risk equality; complete scientific evidence and independently established
evaluation coverage; frozen strategy-specific risk configuration. If no candidate
passes, trading stays inactive. Longer observation alone does not qualify a policy.
Use the existing next-session demo runbook; no new financial inputs are needed for
the configured mechanical smoke test. Stage19 and continuous operation did not run.

## Commands and checks

Actual initial preparation and observation:

```powershell
.venv\Scripts\python.exe scripts/diagnose_recorded_forecasts.py prepare --shadow-config config/local/shadow.yaml --local-capture-config config/local/shadow_forecast.yaml
.venv\Scripts\python.exe -m xauusd_quant.cli shadow-capture --shadow-config config/local/shadow_forecast.yaml --run-id EXNESS_FORECAST_CAPTURE_V001
.venv\Scripts\python.exe -m xauusd_quant.cli shadow-replay --shadow-config config/local/shadow_forecast.yaml --recorded results/shadow/runs/EXNESS_FORECAST_CAPTURE_V001 --run-id EXNESS_FORECAST_REPLAY_V001
.venv\Scripts\python.exe -m xauusd_quant.cli shadow-compare --shadow-config config/local/shadow_forecast.yaml --recorded results/shadow/runs/EXNESS_FORECAST_CAPTURE_V001 --replayed results/shadow/runs/EXNESS_FORECAST_REPLAY_V001 --run-id EXNESS_FORECAST_EQUALITY_V001
```

Those identities exist and cannot be overwritten. The study is bounded to this
prespecified family/observation; future data studies need a new frozen plan/budget.
Broker-free reconstruction using the final frozen reporting plan, with a new run ID:

```powershell
.venv\Scripts\python.exe scripts/diagnose_recorded_forecasts.py evaluate --plan results/shadow/forecast_diagnostics/plans/EXNESS_FROZEN_FORECAST_DIAGNOSTIC_V003.json --recorded results/shadow/runs/EXNESS_FORECAST_CAPTURE_V001 --replayed results/shadow/runs/EXNESS_FORECAST_REPLAY_V001 --run-id EXNESS_FORECAST_DIAGNOSTIC_V004
```

V001-V003 results already exist. Changes to diagnostic code or fitted files require
a new version and preserved provenance. `amend-reporting --original-plan <path>
--plan-id <new-version>` freezes an explicitly retrospective coverage-reporting
revision; it cannot change the native source, models or economic eligibility.

Final validation: 72 targeted offline tests passed, full Ruff passed, typing passed
262 source/script files. Tests include independent hand arithmetic, warm-up/gaps,
future fit and bar availability rejection, invalid scales/maturity/artifact hashes,
bounded serialization, prefix invariance, immutable end-to-end output and partial
coverage reporting. A deliberately disabled availability guard caused its assertion
test to fail; deployed code remained intact. No native order integration tests,
full test suite or unrestricted historical research study ran. WORKLOG.md records
the actual commands, preserved failures and remaining blockers.
