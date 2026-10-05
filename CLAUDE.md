# xauusd-quant — working notes

## Scope discipline

**Stage #18 authorization (2026-10-05):** separate DEMO-only execution, durable
intent/risk reservations, actual deal reconciliation and bounded smoke/strategy
runs when explicit configuration and all gates pass. Stage17 remains read-only.
No real account, account-type override, unrestricted unattended operation,
deployment or automatic Stage19. Scientific gates are unchanged; strategy remains
blocked without an eligible portfolio, complete fresh-feed validation and supplied
risk terms. Smoke tests are execution evidence, not forecasts or alpha validation.
Commit/push intended source/tests/public examples/docs after checks; private
identity, captures and execution journals remain local. Earlier restrictions are
superseded only for this scope. See `docs/stage18_demo_execution.md`.

Stage #18 follow-up: `scripts/run_unsubmitted_demo_smoke.py` is an explicitly
invoked operator tool for never-submitted quote aborts only. It uses the existing
risk rearm audit, keeps all loss/order/intent history, and never changes the native
quote/identity guard. Any prior submission or non-quote halt blocks it. Its own
hash is recorded alongside the unchanged `src` identity. The bounded native attempt
again aborted before order_check/order_send; final account state verified flat.
No automatic rearm/retry, strategy activation or Stage #19 is authorized.

**Stage #17 authorization (2026-10-04):** explicit Exness MT5 demo terminal
identity, official read-only market/account metadata APIs, bounded capture,
causal UTC streaming and local shadow diagnostics are authorized. The shadow
adapter cannot send, modify, cancel or close broker orders. Keep the actual
empty eligible universe and unconfigured risk policy inactive. Historical
stopping rules below are superseded only for this extension; reserved historical
data isolation remains. New Exness captures are a separate feed, never access
to the historical reserved dataset. Stop after Stage #17, no deployment or
automatic Stage #18. The user explicitly authorizes commit/push and prefers
commit/push of future completed implementation stages after appropriate checks;
this supersedes earlier Git prohibitions, not scientific or broker safeguards.
See `docs/stage17_shadow.md` for actual validation and missing prerequisites.

**Stage #16 authorization (2026-10-04):** offline production-oriented risk
software, explicit account/instrument units, conservative reservations, health
and loss gates, persisted halts and bounded synthetic replay through Stage #12.
No eligible Stage #15 portfolio exists; actual settings stay unconfigured and
actual strategy operation inactive. Earlier stops are superseded only for this
extension. No broker, orders, deployment, commit/push or automatic Stage #17.
Offline tests do not establish live safety. See `docs/stage16_risk_engine.md`.

**Stage #15 authorization (2026-10-04):** bounded offline alpha registration,
causal normalized exposure intents, frozen allocations and shared Stage #12
execution/accounting. Stage #14's actual empty eligible universe stays inactive;
synthetic fixtures are software evidence only. Reuse chronological/evidence
guards; no reserved 2022+ outcomes, broker, demo/live orders, deployment,
commit/push or automatic Stage #16. Earlier stopping rules are historical.
See `docs/stage15_alpha_portfolio.md` for actual runs and strategy limitations.

**Stage #14 authorization (2026-10-04):** Prompt #14 authorizes bounded offline
robustness, selection-exposure reconstruction, complete-family corrections when
valid inputs exist, dependence-aware conditional uncertainty, synthetic discovery
controls, fixed perturbations and scenario replay through Stage #12. Reuse the
existing accounting, chronological fitting and econometric APIs. Freeze plans
before diagnostic outcomes; failures remain recorded. Earlier stop-before-#14
rules below are historical and superseded only for this extension. No broker,
demo/live orders, deployment, commits/pushes or automatic Stage #15. Reserved
2022+ remains inspected and inaccessible to this layer. Unknown search history
and missing scientific gates cannot become a passing statistic. Details and
actual bounded results: `docs/stage14_robustness.md`.

**Stage #13.5 authorization (2026-10-04):** Prompt #13.5 authorizes a bounded
offline econometric extension: fixed ARX/GARCH/HAR-style/local-level benchmarks,
causal filtered state, matured-residual outcome intervals and a health-only
CUSUM. Reuse Stage #13 fit/split/ledger and guarded sources. Every target,
transformation and available time is explicit; no selection by outer outcomes.
Forecast superiority is separate from economics. Existing null/provenance/
execution-specification gates and final-test guards remain; 2022+ is inspected.
No broker/demo/live/deployment, commit/push or Stage #14. Details in
`docs/stage13_5_econometrics.md`. Earlier stage stopping rules are superseded
only to permit this extension; historical limitations remain.


**Stage #13 authorization (2026-10-03):** Prompt #13 authorizes offline
purged nested strategy validation and after-cost robustness using Stage #12.
A new small fold-local return model path may select features/counts only
inside preceding inner data. Fixed policy neighbors are reported together,
never chosen by outer PnL. This supersedes the earlier stop-before-#13 rule.
No broker, MT5, demo/live trading, deployment or automatic Stage #14.
Historical reconstruction remains retrospective; 2022+ is inspected and
reserved readers/attempt guards remain unchanged. Missing null/specification
evidence blocks promotion. See `docs/stage13_strategy_validation.md`.

**Stage #12 authorization (2026-10-03):** Prompt #12 explicitly permits an
offline execution and trading-economics layer, including a separate fixed
decision policy, market-order simulation, position accounting, costs, PnL
and equity. Earlier research-only restrictions below describe Stages #1-#11
and are superseded only for this offline layer. Forecast contracts stay
prediction-only; no model/features/policy choice uses trading performance.
Broker/MT5 connections, demo/live trading, deployment, neural networks and
automatic progression to Stage #13 remain prohibited. See
`docs/stage12_execution.md` for timing, units, gates and actual verification.
2022+ has already been inspected (including feature research, ML and
ensembles). It is not untouched. Stage #12 commands read development
forecasts/ticks only; no new reserved-outcome access path is introduced.


The historical Stages #1-#11 scope was **research forecasts only** - nine
descriptive layers and two predictive ones, preserved below. Stage #12 adds
the explicitly authorized offline layer described above:

1. the data foundation (ticks -> Parquet -> bars -> queries);
2. the statistical research layer (returns, volatility, ACF, stationarity,
   sessions);
3. the rolling-regression layer (trailing-window OLS on log mid price, and the
   behaviour of its residual);
4. the Ornstein-Uhlenbeck layer (rolling AR(1)/OU fits of that residual, and
   whether the fitted decay resembles the realised one);
5. the spectral layer (causal rolling FFTs of the residual, the OU innovation
   and log returns, always against four null controls);
6. the wavelet layer (causal one-sided-MODWT rolling features of the same
   series, offline scalograms labelled non-causal, six null controls, and
   nested models A-D measuring information beyond regression + OU + FFT);
7. the regime layer (Prompt #7: K-Means, Gaussian mixtures and a Gaussian
   HMM, K = 2..6, on a compact causal feature set - offline fits labelled
   non-causal, walk-forward filtered probabilities as the only stored
   features - against one state, volatility buckets, random labels and nulls);
8. the feature factory and predictive feature research (Prompt #8: one
   versioned causal matrix per timeframe, a separate `target_*` table, rank
   IC / decay / stability / conditioning / MI / redundancy against max-T
   circular-shift, pipeline and noise-feature nulls - statuses, not signals);
9. feature selection and dimensionality reduction (Prompt #9: selection on
   development 2003-2017 only, evaluation on validation 2018-2021, the
   reserved test period 2022- never loaded with outcomes; immutable ordered
   feature-set manifests for the next stage);
10. supervised predictive-model research (Prompt #10: constant / linear
    baselines, random forest, XGBoost, LightGBM, CatBoost on the manifests;
    chronological walk-forward 2011-2021 with purge + embargo; calibration on
    a purged inner slice; nulls; every trial in the ledger; frozen, hashed
    model specs; one logged evaluation on 2022-). Models output probabilities
    and expected values only;
11. ensembles, meta-models and predictive diversification (Prompt #11:
    eligibility gate on the Prompt #10 evidence, a meta walk-forward over the
    Prompt #10 out-of-sample predictions - every weight / meta-model /
    calibrator fitted on earlier blocks only - simple / median / weighted /
    diversity-aware / stacked / conditional / trailing combinations, nulls,
    frozen hashed ENSEMBLE_SPECs, an inference interface that fails closed,
    and the standardized prediction contract; the final test is a logged
    second look at 2022-).

Earlier stages excluded strategy, signal, sizing, PnL and execution code.
Prompt #12 supplies the explicit authorization for offline execution only.
Optimization by trading performance, neural networks and live trading remain
prohibited; the forecast/research modules retain their original boundaries.
See the "Non-goals" section of `README.md`.

Note the distinction that list draws: regression, OU, spectral, wavelet and
regime *strategies* are non-goals, while the research above is implemented.
The wavelet layer's models A-D and the regime layer's A / A+states are small
ridge / logistic research models that measure incremental information - they
are not the ML system. The regime layer's walk-forward makes its state
estimates causal; it evaluates no trade and never picks K by performance. The OU
layer estimates rolling reversion speeds purely descriptively, beside two
references (invariant 9); the spectral layer measures frequency structure
beside white noise, a detrended random walk, and shuffled and block-bootstrapped
returns, and records every hypothesis it tests in the research ledger. The
feature selection's ridge / Lasso / elastic-net / L1-logistic / PCA models are
diagnostics of forward information on rank designs. The supervised layer
(`ml/`, `research/ml_*`) and the ensemble layer (`ensemble/`, `research/ensemble_*`)
are prediction research: they may output a probability, an
expected value, a calibration or an uncertainty proxy - never BUY / SELL, LONG /
SHORT, a threshold, a position size, a stop / take-profit or strategy PnL.
Ensemble weights come from out-of-sample predictive skill and error diversity,
never from a trading result; the prediction contract refuses decision fields.
Nothing here defines a trade or applies a cost, and no feature set or model is
chosen by trading performance.

## Work log

`WORKLOG.md` is the history of everything done here (asked / built / run /
fixed / found / open). Append to it after every stage, run, fix or decision,
in the same turn, and keep its "At a glance" and "Open items" current.

## Environment

- Python 3.12+ (developed on 3.14), virtualenv at `.venv/`.
- `pip install -e ".[dev]"`, then the CLI is `xq` (or `python -m xauusd_quant.cli`).
- `tzdata` is a hard requirement on Windows: the timezone policy needs an IANA
  database and Windows does not ship one.
- The development machine has 16 GB RAM shared with a browser, often with
  only ~4.5 GB free. Conversion holds one month in memory (peak ~3.5 GB on
  the 9.4M-row months). One OU study peaks at ~0.6 GB (1h), ~0.7 GB (15m),
  ~1.25 GB (5m), and an extrapolated ~4.5 GB at 1m (7.9M bars). A spectral
  timeframe (all series, windows and nulls) peaks at ~2 GB (5m) and ~1.3 GB
  (15m-1h); the 1m spectral study passed 3.7 GB and was stopped for low
  memory before finishing - run it only with the browser closed. Do not
  parallelise partitions or studies here without measuring memory first.
- Stopping a background shell on Windows can leave its Python children
  running (a spawned study child survives its parent) - or the whole tree,
  wrapper shell and logging loop included (2026-09-27). After a stop, check
  `Get-CimInstance Win32_Process` for `bash`/`xq`/`python` and end the tree
  from its root (`taskkill /PID <root> /T /F`). Long runs should write their
  results part by part so a stop loses one part, not the run.
- Git: `main` -> private GitHub `FatCat1244/xauusd-quant` (Git for Windows and
  `gh` installed 2026-10-02). Keep the repository's `core.autocrlf false`: the
  tree mixes CRLF and LF, and every cache stamp hashes source files as raw bytes
  (`study_io.code_fingerprint`), so a checkout that rewrote line endings would
  make every cached unit and stage stale. Commit only when asked.

## Commands

```
pytest                                   # synthetic unit tests + real-data tests
ruff check src tests scripts
mypy
xq info | inspect | detect-schema | validate | metadata | diagnose | query
xq source-index [--rebuild]              # byte-level partition index + ordering report
xq convert [--overwrite] [--limit-rows N] [--keep-previous] [--rebuild-index]
xq verify-dataset [--quick]              # independent integrity + coverage check
xq build-bars --all [--keep-existing]
xq build-features --all | --timeframe 5m [--window 128]
xq research-summary --timeframe 5m | --all
xq regression-research --timeframe 1h --window 128 | --all-windows | --all
xq ou-research --timeframe 5m --regression-window 128 --ou-window 256
             | --timeframe 5m --all-windows | --all  [--resume] [--in-process]
xq ou-research --compare                 # comparison tables from studies on disk
xq spectral-research --timeframe 5m --source regression_residual --fft-window 256
                   | --timeframe 5m --all-windows | --all [--allow-heavy] [--resume]
xq spectral-research --compare | --benchmark
xq wavelet-research --timeframe 5m --source ou_innovation --window 512
                  | --timeframe 5m --all-windows | --all [--dry-run] [--allow-heavy]
xq wavelet-research --controls | --compare | --benchmark
python scripts/wavelet_baseline_robustness.py [-t 5m ..] [-w 512 ..] [--resume] [--no-ledger]
                                         # post-hoc WAVE-P-001: A/B + longer RV + hour of day
xq regime-research --timeframe 1h --model hmm --states 3 [--stages offline,walk_forward,nulls,analysis]
                 [--schemes expanding_quarterly,rolling_quarterly] [--monthly] [--resume]
xq regime-research --all --dry-run | --controls | --compare | --benchmark
xq regime-walk-forward --timeframe 5m --model hmm --states 4 [--scheme rolling] [--refit monthly]
powershell -File scripts\regime_grid.ps1 -Timeframes 1h,30m,15m   # resumable grid; 5m: -Heavy
xq build-feature-factory --timeframe 5m [--force] [--dry-run]     # alias build-feature-matrix
xq feature-research --timeframe 1h | --all [--stages ic,nulls] [--report-only] [--no-ledger]
xq feature-research --timeframe 1h --feature log_rv_20 [--target realized_vol] | --compare
xq feature-redundancy --timeframe 1h [--feature x];  xq alpha-decay --timeframe 1h --feature x
xq feature-select --timeframe 5m | --all [--stages universe,...,audit] [--no-resume] [--no-ledger]
xq feature-select --timeframe 5m --target residual_reduction | future_return --horizon 5
xq feature-selection-report --timeframe 5m | --all | --compare [--no-plots]
xq build-feature-manifest --set standard [--timeframe 5m] [--output path]
xq ml-research --timeframe 5m [--stages grid,trees,...] [--dry-run] [--report] [--in-process]
xq ml-train --timeframe 5m --target mean_reversion --horizon 5 --model xgboost [--feature-set standard] [--folds 4,5]
xq ml-walk-forward --timeframe 5m --target mean_reversion --horizon 5 [--models ...]
xq ml-compare --timeframe 5m --target mean_reversion [--horizon 5] [--variants]
xq ml-report --timeframe 5m [--light] [--no-plots] [--no-ledger]
xq ml-freeze --timeframe 5m;  xq ml-finalize --timeframe 5m [--no-streaming]
xq ml-final-test --model-spec LGBM_REVERSION_5M_H5_V001 [...]   # frozen specs only, logged
xq ensemble-research --timeframe 5m | --all [--target T --horizon H] [--report] [--no-pipeline-nulls]
xq ensemble-build -t 5m --target mean_reversion --horizon 5 --method stacking [--models "a|set" ...]
xq ensemble-report -t 5m [--no-ledger] [--no-plots]   # tables, ENS-H ledger rows, joint predictive state
xq ensemble-freeze -t 5m;  xq ensemble-finalize -t 5m [--no-streaming]
xq ensemble-final-test --ensemble-spec ENS_VOL_5M_H5_V001 [...]  # frozen specs only, logged, second look
powershell -File scripts\run_logged.ps1 -Name <log> -ArgList "-m xauusd_quant.cli ..."  # memory-logged run
```

`ensemble-research` is not cached: a run recomputes every pair (5m ~30 min, ~3.8 GB
private; 15m ~12 min). It reads only the Prompt #10 prediction tables, the unit files
(null and cross-horizon units) and the development rows; nothing it imports can read
2022-. The pipeline-null labels rebuild the null from the development bars only
(`ensemble.streaming.load_development_bars`, the cut is in the scan): the same seed gives
every development row the draws of Prompt #10's null, and the random walk's scale - the
one quantity that differs - cancels in every residual label (checked: identical, or
within float32 rounding). `features/factory.py` is part of the feature-matrix build key:
do not add parameters there.

`ml-research` runs each timeframe in a spawned child and caches every *unit*
(target x horizon x family x feature set x variant x block) as it finishes,
stamped with the data versions, the unit spec, the config sections it reads
and the code of `ml/{datasets,splits,preprocessing,models,training,calibration,
explainability}.py`. Editing one of those files makes every unit stale (a
rerun refits everything) - never edit them while a run is going. `research/ml_*`
edits do not invalidate units.

`feature-research` and `feature-select` run each timeframe in a spawned child
(`MIMALLOC_PURGE_DELAY=0`) and cache every stage with a stamp of the data
versions, the config fingerprint and the code of that layer's modules
(`feature_research._STAGE_CODE`; for the selection, all of `selection/*.py`
plus `research/feature_selection_research.py`) - editing those files reruns
the stages, so never edit them while a run is going.

`regime-research` caches every expensive stage (offline fits, walk-forward runs,
null fits, null inputs, pipeline-null increments) with a stamp of the data
versions, the config fingerprint and the code of *that stage's* modules
(`regime_reports._STAGE_SOURCES`); the analysis is always recomputed. Editing
`regimes/*.py` or `research/regime_{analysis,nulls,outcomes}.py` invalidates
cached stages - never do it while a grid is running.

`ou-research` runs every study in a fresh child process (`--in-process` turns
that off for debugging) and keeps going past a failed study, exiting non-zero
with the list. `--resume` reuses a study only when its `summary.json`
provenance matches the current raw file, tick/bar/feature versions and all
four config fingerprints; it cannot see code changes, so after a change that
alters values, rerun without it.

Pipeline order after a new export: `convert` -> `verify-dataset` ->
`build-bars --all` -> `diagnose --all` -> `build-features --all` -> research.
Bars refuse an incomplete tick dataset; features refuse unverified bars.

## Invariants to preserve

1. The raw file is read-only. Never write to `raw_data_path`.
2. `timestamp` is the broker-local wall clock exactly as recorded, never
   rewritten. `timestamp_utc` is a separate derived column.
3. `bid` and `ask` pass through untouched; `mid` and `spread` are derived.
4. The validator flags; only the cleaner drops, and only what `cleaning.drop_*`
   enables. Every drop is counted and attributed.
5. Bars aggregate `[t, t+D)` with `closed: left`. No look-ahead, no filling.
6. Nothing may require the whole dataset in memory (one month at a time is
   the unit).
7. Sampled figures must be labelled as estimates; full-pass figures as exact.
8. A feature at `t` uses only bars at or before `t`. Forward values are
   *outcomes*, are named `fwd_*` or live only in outcome tables, and never
   enter a feature table.
9. Every mean-reversion or stationarity claim about the regression residual is
   reported beside the same statistic computed on a detrended random walk.
   For OU statistics, also beside an exact OU simulation with the fitted
   parameters (`reference_ou`). Without those the number is not interpretable.
10. Conversion never assumes a sorted source. Each partition is read from its
    indexed byte runs, validated as a whole month, stably sorted, written as
    `*.parquet.partial`, re-opened and verified, then renamed. Its manifest
    (`_manifest.json`) lives inside the dataset directory.
11. `--overwrite` / `--no-resume` build beside the live dataset
    (`<processed>.building`) and swap only when the new one is complete and
    validated. The live dataset is never deleted first. `--limit-rows` writes
    only to `<processed>.smoke`.
12. Every research artefact embeds `dataset_lineage` (raw fingerprint -> tick
    version -> bar version -> feature version, with coverage). A result from an
    incomplete dataset carries `PARTIAL DEVELOPMENT SAMPLE — NOT FULL RESEARCH
    RESULT` in its own provenance.

## Things that are easy to get wrong here

- **Hive partition typing.** `month=01` is read by DuckDB as VARCHAR while
  `year=2021` is BIGINT. Predicates must cast both (`loader._partition_predicate`).
- **DuckDB session timezone.** Set to UTC in `DataStore.connect()`, otherwise
  `timestamp_utc` renders in the machine's local zone.
- **The source is NOT sorted, by design of the export.** From 2021-01-04 the
  file repeats the opening 54 min - 2 h of every trading week verbatim: 301
  backward jumps, 2,683,507 rows behind the running maximum, none in an
  earlier month, none new (all byte-identical repeats). Each repeat ends on a
  row that *ties* the running maximum, so exact-duplicate drops are 2,683,808
  = late rows + 1 per splice. `drop_exact_duplicate_rows: true` removes them;
  the source index diagnoses them (`xq source-index`). Do not "fix" the
  disorder by assuming it away - the converter never needs sorted input.
- **The old converter could destroy a month.** It closed a partition when a
  later month appeared and *recreated* the file if another row for it arrived.
  It never fired on this export only because no late row crosses a month.
  `test_a_row_arriving_after_its_month_closed_does_not_clobber_the_month`
  keeps it that way.
- **Partition digests** use *two* hashers per column - one for the dense NumPy
  values, one for the null mask - combined in schema order at close. The rule
  is: every hash stream gets only its own bytes, in row order. Three attempts
  got this wrong before it was right:
    1. hashing Arrow buffers (chunk-level padding leaks in);
    2. one hasher per column, values and null mask interleaved per flush;
    3. same, after "fixing" it - the interleaving was the bug all along.
  The digest is now fed one row group at a time;
  `test_digest_is_invariant_to_chunking` varies `row_group_rows` and asserts
  the runs really chunked differently and the data has nulls before comparing.
  If you touch the digest, re-break it deliberately and confirm that test
  fails (last done 2026-09-24).
- **Parquet writer.** Partitions are written with Polars (parallel column
  compression, ~7x faster than pyarrow's writer here) and must pass
  `arrow_schema=` so the `xq_*` provenance lands where `pq.read_schema` finds
  it. Encodings are the writer's choice; content is what the digest checks.
- **Source index positions** come from `timestamp_format`: `%Y`/`%m` must sit
  at fixed character offsets. Lines without a readable year-month are
  *unassigned*, still parsed, and attributed to a drop reason - never lost.
- **Polars NaN is not null.** `filter(pl.col(x).is_not_null())` *keeps* NaN.
  Both `rolling_regression_features` and `ou_parameter_frame` convert every
  float column with `fill_nan(None)`. Without it warm-up rows survive every
  filter, inflate counts and drag quantile edges upward (measured once: q25
  0.336 against a true 0.288).
- **Bucket labels built by a reversed `when/otherwise` chain are easy to get
  off by one.** The chain is evaluated outermost-first, so iterating edges in
  reverse means edge `i` carries label `Q{buckets-1-i}`, not `Q{...+1}`.
- **`acorr_ljungbox` recomputes the ACF with an O(n^2) correlation.**
  `ljung_box_test` evaluates the closed form from an FFT-based ACF instead;
  `test_ljung_box_matches_statsmodels_exactly` pins them together.
- **The bootstrap of a proportion is exactly `Binomial(n, p)/n`.**
- **`P(|eps_{t+h}| < |eps_t|)` has a high baseline, not a zero one** (~0.97 for
  an i.i.d. series conditioned on `|Z| > 2`). Read it against the control.
- **Newey-West does not always inflate** (one-step decay score ACF ~0.02).
- **A windowed OU half-life of a random walk is finite.** In-window OLS has a
  downward (Dickey-Fuller / Kendall) bias, so every window reports `b < 1`.
  The tell is scaling: a random walk's median half-life grows roughly in
  proportion to the OU window M; a genuine OU's converges. Always compare the
  same M against `reference_ou` and `control_random_walk`.
- **Rolling AR(1) sums are block-shifted prefix sums.** Exact, bit-for-bit
  equality holds only between runs with the same block structure (which is
  what no-look-ahead needs). A series that drifts far from a block's shift
  loses precision ~ eps*(range/local std)^2; stationary residuals are fine.
- **Censored half-lives.** Windows with `0 < b < 1` but a half-life above
  `max_reportable_bars` are `near_unit_root`: counted, never averaged, and a
  quantile landing among them is null with `p.._above_cap = True`.
- **An all-null column has dtype `Null`, and a strict concat refuses it.** A
  statistic undefined for every row of one source (a censored ratio p90, say)
  builds a `Null` column; `pl.concat(how="diagonal")` then fails on the next
  source's Float64 - but only when the `Null` table comes *first*, so a test
  that stacks in the other order passes against the bug. This stopped the
  first full grid at 1h N=256 M=128. Stack per-source tables with
  `ou_reports._stack` (`diagonal_relaxed`); `_Writer.table` writes any `Null`
  column as Float64.
- **Memory at 1-minute scale (7.9M bars).** Every column is ~63 MB, and a
  filter copies every column it keeps: select the columns a step reads
  *before* filtering (the full OU frame is ~30 columns, ~2 GB). Never turn a
  column into a NumPy array of Python strings (`Enum` codes compare the same).
  The feature store loads only the columns asked for (`load(..., columns=)`).
- **matplotlib leaves a reference cycle behind** after `tight_layout`/`savefig`:
  the plotting calls' execution frames stay alive, and with them whatever they
  drew. The ~0.5 GB (5m) real OU frame stayed resident through every control
  until `generate_ou_report` began calling `gc.collect()` after dropping it.
- **Spectral components are spectral peaks, not the largest bins.** A Hann
  window leaks half the amplitude (a quarter of the power) into each
  neighbouring bin, so the second-largest bin is usually the first one's
  leakage. `batch_spectrum` ranks local maxima; concentration (top-1/3/5
  share) is still over raw sorted bins, as defined. An on-bin tone keeps 2/3
  of its Hann power in one bin.
- **Phase is exact only on the bin grid.** A cycle `delta` bins off the grid is
  reported at the nearest bin with a phase biased by about `pi * delta` and up
  to ~15 % less amplitude (Hann scalloping). The positive-control table shows
  an off-bin case on purpose; do not "fix" it by quietly interpolating.
- **Overlapping windows fake phase consistency.** Windows `h < N` bars apart
  share `N - h` values, so their phases advance consistently for any series,
  white noise included. Persistence and phase continuation are read at lags
  `>= N` and always against the nulls.
- **The FFT axis is trading time.** Weekends and the daily break are closures,
  not missing data (the regression and OU layers use the same bar sequence);
  only `holiday_or_unknown` gaps count toward `missing_bar_tolerance`.
- **Spectral memory at 1m.** A rolling pass keeps ~0.1 KB per bar of float32
  features, plus the source (~0.8 GB at 1m); reconstruction samples are capped
  at 20M values; IC and redundancy use evenly spaced samples (labelled as
  estimates). `--all` skips 1m unless `--allow-heavy`.
- **A null with zero IQR is still a null.** Discrete metrics (the dominant
  period) often have a zero inter-quartile range - the residual's dominant bin
  is bin 1 in almost every window, real and null alike. `_effect` once
  returned None there, `_verdict` then ignored that null, and the verdict
  rested on white noise alone: the residual's dominant period read
  "exceeds_all_nulls" while being identical to the random walk's. Equal
  medians are now 0, unequal ones are scaled by the 5-95 % range
  (`test_a_null_with_zero_iqr_still_counts_in_the_verdict`).
- **"Above the largest null IC" happens by chance.** SPEC-H-006 compares each
  real IC with the maximum over the nulls' ICs; with thousands of tests some
  real ICs exceed it. `ic_chance_rates.csv` (written by `--compare`) gives
  each null's own count against the other sources - read the real count
  against those, never on its own.
- **Polars' allocator (mimalloc) keeps freed pages** unless
  `MIMALLOC_PURGE_DELAY=0` is set before Polars loads; `ou-research` sets it for
  its child processes (15m peak -20%, no measurable slowdown). The package
  itself does not set it, because conversion was never measured with it.
- **Gap classification at tick level.** Ticks straddle the 00:00-01:00 break by
  seconds, so `verify-dataset` uses a 5-minute edge tolerance; the bar-level
  classifier does not need one.
- **Bar timeframes must divide a day evenly** because bars are built one month
  at a time. `validate_timeframes` enforces it.
- `datetime.utcnow()` is deprecated; use `utils/clock.py`.
- `pytest` runs with `-W error::DeprecationWarning` for this package, so a
  deprecation in our own code fails the suite. That is intentional.
- **PyYAML reads `1.0e8` as a string** (YAML 1.1 wants a signed exponent:
  `1.0e+8`). The regime config coerces its numeric fields; write `e+`/`e-`.
- **Regime features are prefix-invariant only because every product has one
  shape.** The HMM filter uses fixed 512-bar blocks anchored at the first bar
  and multiplies in fixed groups of 128 blocks; emissions and K-Means
  distances pad their last chunk to 16,384 / 65,536 rows; next-state and leave
  probabilities are summed elementwise. A plain `probs @ A` over n rows can
  change the last bit when n changes (BLAS picks kernels by shape).
  `test_walk_forward_outputs_are_prefix_invariant` (HMM, GMM, K-Means, wild
  appended future) pins it.
- **Smoothed HMM probabilities use later bars** (hmmlearn's `predict_proba` is
  smoothed too). Only `HMMModel.filter` feeds features; `enforce_live_safe`
  refuses `smooth*`, `viterbi*`, `offline*` columns in the stored table.
- **Any HMM on trailing-window features looks persistent.** Consecutive bars
  share most of their windows, so the smoothness alone gives long state
  durations: at 1h K=3 the real features' median expected duration (30 bars)
  sits among the pipeline nulls' (random walk 27, shuffled returns 58, block
  bootstrap 62) and a row shuffle collapses it to 1.5. Read durations and the
  transition matrix's likelihood gain only against the nulls.
- **States need not be volatility.** The 1h K=3 HMM split on trend (slope / vol)
  and R^2, NMI with volatility buckets 0.001. Describe states from
  `standardized_state_medians`, never from the canonical order (ascending mean
  ln RV is only a tie-breaking convention).
- **Those trend states are feature geometry, and the redundancy rule cannot see
  it.** R^2 of the 128-bar regression is nearly a monotone function of
  |slope / vol| (Spearman 0.93; 86 % of its variance at 1h), a V that one
  Gaussian cannot follow, so every full-covariance mixture puts a component on
  each arm and one on the vertex - on a random walk too (0.94, same states,
  same likelihood gain; `scripts/regime_feature_geometry.py`). The
  |Spearman| >= 0.9 rule looks at the signed pair (0.13-0.19) and let it
  through. Screen new inputs for non-monotone dependence (|x| vs y, a step
  function of one on the other), not only rank correlation. Rank features
  (`*_percentile`) are uniform by construction and invite the same banding.
- **A likelihood K rule cannot tell regimes from a continuum.** A smoothly
  drifting volatility improves with every K (synthetic control, +3.4 nats at
  K = 5) exactly as the real features do; only a known discrete HMM plateaus
  (at K = 3). The hold-out fits also use one initialisation, so a step can fail
  on an EM optimum (1h HMM: K = 5 worse than 4 and 6).
- **A GMM or K-Means leaves unscored bars at -1; the HMM filter predicts through
  them.** Walk-forward frames can therefore hold more rows than the scored
  inputs; filter `state >= 0` (or join on timestamp) before mixing the two.
  Grid run 1 failed on exactly this.
- **The venv's `python.exe` is a launcher** that spawns the real interpreter:
  measure the child (or all `python.exe`) when logging memory.
- **Feature selection must never see the reserved period's outcomes.** The
  loader (`selection/data.py`) never opens a target year file that starts at or
  after 2022-01-01 and filters rows at the scan; a guard refuses outcome rows
  at or after the reserved start; ranks and CDFs are taken inside a period or a
  training fold, never over the whole history. Prompt #8 statuses were computed
  on 2003-2026 outcomes, so they are *reported*, never used to select - the
  universe is re-screened on development. `tests/test_selection_isolation.py`
  corrupts the reserved target files; re-broken 2026-09-30 (a loader opening
  them, training rows reaching the evaluation block) and seen to fail.
- **A plain top-k stability count saturates.** When a target's candidate pool
  is not much larger than k (direction at h >= 5, reversion), every candidate -
  noise probes included - is chosen in every resample. Stability selection
  keeps the features that enter *before the first noise probe*
  (`stability.until_first_probe`); top-k frequencies are only a diagnostic.
- **Selection candidates are per target.** A residual feature that passes
  for volatility must not compete for a reversion target; a reversion claim
  also needs the pipeline veto (Prompt #8 calls that kind `residual`, the
  selection `reversion` - map with `targets.alignment.target_kind`).
- **Minimal / Standard / Extended are prefixes of one ranking**, so Minimal is
  inside Standard inside Extended; sizes come from the validation plateau with
  residual targets left out (their raw validation IC is partly mechanical).
- **Manifests are immutable.** Writing a different feature set under an
  existing `FEATURESET_<TF>_<SET>_V<nnn>` raises `ManifestConflictError`: delete
  development-run manifests before a final run, or bump `manifest.version`.
- **The selection layer's linear models are ours; scikit-learn is only the
  `ml` extra.** The Lasso / naive elastic net (coordinate descent on the Gram)
  and the L1 logistic path (FISTA) in `selection/linear.py` do not use it and
  stay that way (tested against least squares and known supports). The
  supervised layer uses scikit-learn 1.9, XGBoost 3.4, LightGBM 4.7, CatBoost
  1.2 and SHAP 0.52 (`pip install -e ".[ml]"`).
- **API drift in the ML libraries fails the suite** (`-W error::Deprecation`
  catches our calls): scikit-learn 1.9 deprecates `LogisticRegression(penalty=)`
  - use `l1_ratio` (0 = L2 with lbfgs, 1 = L1, 0.5 = elastic net, with saga);
  LightGBM 4.7 deprecates `fit(eval_set=)` - use `eval_X=(x,), eval_y=(y,)`.
- **Polars `to_numpy()` can return a read-only view.** Copy
  (`np.array(..., dtype=np.float64)`) before writing into it (the target purge
  did, and failed with "assignment destination is read-only").
- **The reserved period stays out of memory, not just out of the metrics.**
  `ml.datasets.read_years` never opens a year file starting at or after
  2022-01-01 and filters at the scan; targets are purged at the reserved start
  by their own horizon; only `ml/final_test.py` reads reserved rows, only for a
  frozen hash-verified spec, and nothing that selects models imports it
  (`test_final_test_isolation.py`, re-broken 2026-09-30 and seen to fail).
- **An SE of exactly zero is a consistent difference, not a missing SE.**
  `paired_blocks` gives SE 0 when every block differs by the same amount; the
  freeze rules once read `not se` as "no SE", so a method that was worse in
  every block counted as "within one SE" and a window that was better in every
  block as "not beyond". `ml_freeze._better_by_one_se` (needs >= 2 blocks,
  `se is not None`); `test_ml_freeze.py` pins it with dyadic values.
- **PowerShell 5.1 `Set-Content -Encoding utf8` writes a BOM.** It did, into
  `cli.py`. Edit with the editor tools, or write with
  `[IO.File]::WriteAllText(p, s, (New-Object Text.UTF8Encoding($false)))`.
- **The family-ablation pool has no pipeline veto**, so residual-target
  ablation rows include mechanical effects; read them beside invariant 9.
- **`MLData.design` caches one matrix per key until the stage ends**, and a
  feature list that is not a named manifest (an ablation subset, a `target`
  set) is keyed by its names. Fifteen 60-column 5m subsets would hold ~4.8 GB;
  `run_units` drops the cache after every such unit (2026-10-01, found at 3.8 GB
  and rising, beside the 15m run).
- **The one-spread direction labels have a drifting base rate.** P(R > spread)
  moves with spreads and volatility, so calibrating on the inner slice (the
  recent 10 % of training) earns log-loss skill against the training rate with
  no feature at all: the Platt-calibrated *constant* scores +0.018 / +0.021 /
  +0.011 (h1 / h5 / h20, 5m), and a shifted-label model +0.016 or -0.134 at AUC
  0.50. Read direction skill beside `skill_beyond_base_rate` (and the final
  test's skill against the recent base rate); the reversion labels' base rate is
  stable (~0).
- **A circularly shifted volatility target keeps the daily cycle.** A shift that
  lands near a whole number of days leaves the time-of-day alignment intact, so
  the shifted-label control for `future_volatility` has rank IC from -0.16 to
  +0.53 by block. It still collapses for reversion and return; for volatility
  read the real IC against every shifted control, not their mean.
- **Residual targets need the sign-flip null, not only the random walk.** The
  random walk is homoscedastic, so real minus random walk can hold volatility
  dynamics (a falling volatility shrinks |eps| too). `nulls.pipeline_posthoc:
  [sign_flip]` (stage `pipeline_null_posthoc`, post-hoc 2026-10-01) keeps the
  real volatility path with random signs. 15m mean_reversion h5 LightGBM: real
  +0.131 log-loss skill, random walk +0.112, sign flip +0.126; residual
  reduction is *less* predictable on real data than on either null.
- **A pipeline null build costs ~2.8 KB of private memory per bar** on top of
  the process (15m: +1.6 GB, 50 s). At 5m (~1.66M bars) that is ~4.7 GB over a
  ~3 GB base: run it alone. The stage skips the build when all of a null's
  units are current.
- **Search trials are compared with the defaults on the searched set only.**
  With base units of Standard and Extended on the same blocks, a per-block
  lookup took whichever set's row came last; `_search_choice` now filters the
  set, and a model frozen on another set keeps the defaults.
- **At 15m, Standard = Extended (72 features)**, so the 15m Standard set carries
  the residual's own z-scores and every 15m reversion number is mostly
  mechanical - read it against both pipeline nulls.
- **Every ensemble choice for block k must come from purged earlier rows.**
  `PairPredictions.history(k, embargo)` drops the last `h + embargo` rows of block
  `k - 1` (their labels resolve inside block k). The first version ranked the "best
  individual" and the size orders on *whole* earlier blocks - a few bars of block k
  leaked in (found by the Codex review, fixed with `PairRun.prior_scores`). Trailing
  weights take the day's first bar from the full timeline, not the first covered row.
  `tests/test_stacking_no_leakage.py` and the dynamic-weight tests were re-broken
  (no purge: 6 fail; unresolved labels: 3 fail).
- **The eligibility gate reads all five Prompt #10 blocks**, so a model's later
  blocks decide whether it enters the early blocks' ensembles. `*_wf_universe`
  rebuilds the universe of each block from the earlier blocks only - read the
  methods beside it.
- **An average of calibrated probabilities is under-confident** when the
  constituents differ in sharpness (boosters averaged with linear models pull
  every probability toward the middle), and weak eligible models dilute it; the
  reliability diagram shows it, a final calibrator (fitted on earlier blocks)
  repairs the calibration but not the dilution.
- **Target-set models have no pipeline-null units**, so their null status is
  `untested` and they pass the gate (missing is not failing) - a reversion
  ensemble that contains them cannot be read against the random-walk null.
- **The ensemble final test is a second look at 2022-.** Prompt #10 evaluated 37
  single models on it; the ensembles were frozen by rules fixed before any ensemble
  result, and every final-test output carries that note. The access log counts an
  interrupted run (`final_test_started` without `evaluated`) as a look too.
- **`ensemble-finalize` at 5m fits 41 constituents in one process.** A linear model
  on the Extended set makes ~1.3 GB of float64 copies (142 design columns with the
  missing indicators x 1.2M rows); with the cached designs of earlier fits that ran
  out of memory once (peak 5.6 GB, 2026-10-01 21:36). Designs are now dropped and
  garbage collected before each fit; a rerun reuses every saved artifact (only the
  missing constituents train), so a stop loses one constituent.
- **`ensemble-final-test` must not load the whole ML data set.** The first run (all
  13 specs in one process) stopped with a `MemoryError` while the first ensemble
  predicted the reserved rows (~0.4 GB commit free beside a 4 GB application), and
  the interrupted look still counted - the rerun needed a logged `--repeat-reason`.
  `final_test.slim_development_data` reads only spread, `log_rv_20`, the specs'
  targets and the residual / trailing volatility; run one timeframe per process
  (5m peak 1.75 GB, 15m 1.46 GB).

## Dataset facts (measured, not assumed)

Current export: `2026.9.18XAUUSD_oanda-TICK-No Session.csv`, 33.75 GiB
(36,233,955,746 bytes), whole-file blake2b-256 `e15005ce…222c`.

- One CSV, `DateTime,Bid,Ask,Volume,Spread`, `YYYYMMDD HH:MM:SS.mmm`, LF, UTF-8.
- 731,928,177 data lines, 0 malformed, 0 without a readable timestamp.
- Converted: **729,244,369 rows, 281 monthly partitions, 2003-05-05 03:01:03.421
  -> 2026-09-18 23:59:59.079, 9.13 GiB**, every month present, dataset version
  `ticks-2e173ef8e61bd240`. Independently verified: 0 backward steps, 0 kept
  duplicates, 0 crossed quotes, 0 invalid prices, all digests match.
- Bars (version-chained to those ticks): 1m 7,937,007; 5m 1,663,606;
  15m 561,817; 30m 282,256; 1h 141,569.
- The source `Spread` equals `Ask - Bid` everywhere, but is still kept as
  `spread_source` so the pipeline can *check* that rather than assume it.
- Broker server time = America/New_York wall clock + 7 h (UTC+2 winter,
  UTC+3 summer, on **US** DST dates) — for the whole 23-year span.
- **Re-derive the timezone before trusting it on a new export.** The release-
  spike method resolves 2011+ cleanly but is too noisy for 2003-2010. What
  settles the early era is the CME settlement lull (17:00-18:00 New York, the
  quietest hour of the gold day): it sits at local hour 0 in every era and both
  seasons. Probe scripts for both methods are described in `README.md`.
- **January 2012 structural change, not a clock change.** Before 2012 the
  00:00-01:00 hour carries ~1.2% of each day's ticks; from 2012-01 the provider
  stops exporting it, so the lull becomes a hard daily gap and the weekend goes
  from 48 h (Mon 00:00 open) to 49 h (Mon 01:00 open).
- Early years are far sparser than recent ones (2003: 104k 1m bars for May-Dec;
  ~350k/year from 2007). Expect a large `holiday_or_unknown` count at 1m in
  that era; it is real, not a bug.
- **The early feed carries heavy quote noise.** Lag-1 autocorrelation of bar
  mid returns in 2003-2005 is -0.40 (1m), -0.29 (5m), -0.19 (15m), -0.11 (30m),
  -0.06 (1h); 2006-2007 about half that; from 2008 it is within +/-0.03 at
  every timeframe. Pooled 23-year high-frequency statistics are dominated by
  it (1m pooled: -0.17). Split any 1m-15m result by era before reading it -
  the OU layer's "faster than the random-walk control" at 5m is mostly this.
- `Volume` is broker liquidity units, not traded volume.
- The modern feed is throttled to ~50 ms; the early feed is seconds-grained.
  Neither is true full-depth tick data.
- **Wavelet features must be causal, and the transform is ours on purpose.**
  `pywt.swt` is circular: its newest coefficients wrap around to the start of
  the series, and a windowed `pywt.wavedec` makes the newest coefficient
  depend on its padding mode (by e^0.6 - e^13 in energy in the boundary
  study). `features/wavelet.causal_modwt` builds the MODWT filters from
  `pywt.Wavelet` and applies them one-sided with `np.convolve`, which is bit
  for bit prefix-invariant; band energies use only coefficients whose whole
  filter lies inside the window. Anything computed from both sides of a
  point (CWT scalograms, ridges, `dwt_components`) is offline and labelled
  `NON_CAUSAL_LABEL`; `wavelet_feature_frame` refuses any column that is not
  in `CAUSAL_FEATURES` (and anything named `offline_*`).
- **Raw MODWT band energy is not flat for white noise.** Band j of an
  orthogonal MODWT holds 2^-j of white noise, so "the band with the most
  energy" is always the finest one and looks perfectly persistent. The
  dominant band is the one whose share most exceeds `white_noise_shares`
  (exact expectations, including the approximation band's within-window
  variance). Entropy and top-k shares stay on raw shares (the standard
  relative wavelet energy).
- **Drift "continuation" near -0.5 is no persistence.** Consecutive disjoint
  changes of a stationary quantity share an endpoint, so their correlation is
  -0.5 with no structure at all; a continuing drift would be positive.
- **The 256-block bootstrap's IC ceiling is inflated for volatility targets.**
  Blocks drawn from different eras put volatility jumps at their joins, which
  a trailing energy feature predicts well. Read real against each null
  separately (and against `ic_chance_rates`), not only against the maximum.
- **The research ledger is shared across layers.** Its window column is
  `window` (it was `fft_window`; old files are renamed on load and rewritten
  on the next upsert). The hypothesis-id prefix (`SPEC-`, `WAVE-`) names the
  layer.
- **The full test suite peaks at ~3.9 GB resident, ~5 GB committed** (1003
  tests, 22.5 min on the PC, 2026-09-30; 887 tests / ~4.2 GB before Prompts
  #8-#9; 1,120 tests 5.5 GB private in one process, 2026-10-01). Do not run it
  beside a timeframe study on this machine; with the usual desktop apps open,
  commit headroom fell to ~1.2 GB during it. When memory is short, split it into
  processes (`-m "not realdata"` by file range, then `-m realdata`) - but a
  runner *script* that calls `pytest.main` must sit under
  `if __name__ == "__main__":`. Some tests start a spawned child (the CLIs run
  studies or timeframes in one), which re-imports the main script; unguarded,
  the child reran the whole part instead of its job and the run never finished
  (2026-10-01). `python -m pytest` is safe. Split that way (1,259 tests,
  2026-10-01) the unit parts peak at ~1.1-1.3 GB, but
  `test_realdata.py::test_dataset_is_sorted_and_free_of_duplicate_timestamps`
  (DuckDB `count(DISTINCT timestamp)` over 729M rows) needs ~4.3 GB private on
  its own: with ~2.2 GB commit free it failed with "Out of Memory Error:
  Allocation failure", and passed alone in 9.4 min.
- **Feature-selection runs** (`feature-select`, one timeframe per child):
  1h 5 min / 1.1 GB, 30m 8 min / 1.4 GB, 15m 13 min / 1.8 GB, 5m ~35 min /
  3.0 GB (nested stage ~26 min). The nested stage writes its tables only when
  all four splits finish - a stop inside it loses the whole stage.
- **NaN fails every comparison silently, so a verdict must never see one.**
  `abs(nan) > ceiling` is False and `nan < 0.3` is False: an undefined IC
  (a fast/slow ratio with no slow band, 5m N <= 256) was recorded as
  "within_null_range" and an undefined redundancy as "redundant" (186 rows,
  corrected 2026-09-27 with a note in each row). Every wavelet verdict reads
  its inputs through `wavelet_reports._known` (non-finite -> None ->
  "untested"), and IC test counts include only defined ICs
  (`test_an_undefined_statistic_is_untested_never_a_verdict`, re-broken and
  seen to fail). The regime verdicts do the same through
  `regime_stability.finite_or_none`. Missing is not failing either: the K rule
  once read "no walk-forward" (a model run offline only, like the 5m GMM) as a
  refit ARI below 0.5, i.e. "no discrete states"; an unavailable step now
  leaves the rule incomplete and REG-H-004 "untested"
  (`test_the_k_rule_reads_an_unavailable_statistic_as_untested`).
- **Post-hoc tests are recorded, but apart.** `POST_HOC_HYPOTHESES`
  (`WAVE-P-`) go to the ledger like any test - they count toward multiple
  testing - but never into `hypotheses_registered.json`, and they can only
  qualify a registered result. `incremental_study(baseline="original")` is
  the registered model bit for bit.
- **Spectrally, the residual, the OU innovation and returns are
  indistinguishable from their nulls (5m-1h, 2003-2026).** The one periodic
  structure is the one-trading-day volatility cycle, visible in |eta|
  (`abs_ou_innovation`): dominant in 23-73 % of 15m-1h windows vs < 1 % for
  the nulls, at the same 23.3 h at every timeframe. Every other dominant
  period is fixed in bars (a window or rolling-filter artefact).
- **Wavelets add nothing beyond the nulls, longer-horizon volatility and the
  hour of day (5m-1h, 2003-2026).** Shape, persistence, bursts, ridges and IC
  are all within the nulls; the only structure beyond them is the intraday
  shift of energy to fine scales (the same daily cycle). The registered
  model increment (WAVE-H-007: 229 of 1,008, all volatility-type targets)
  falls to 1 of 108 once A/B carry ln RV over 64-1024 bars and hour-of-day
  indicators (post-hoc WAVE-P-001, N = 512) - the wavelet block was a proxy
  for volatility over its own window and for the daily cycle.
- **No discrete regimes beyond volatility (5m-1h; causal 2008-2026).**
  Full-covariance GMM / HMM states are trend / linearity bands - the mixture's
  way of fitting the R^2 vs |slope/vol| V, which a random walk reproduces
  (same states, same likelihood gain) - and their persistence is within the
  pipeline nulls at every K tested (REG-H-005: 0 of 12). K-Means states are
  volatility level and the daily cycle (Cramer's V with the hour up to 0.33).
  Beyond the continuous features the filtered probabilities add at most
  +0.001 R^2 or nats (REG-H-007: 14 of 825 beyond the nulls); a one-hot of
  volatility buckets adds more (up to +0.039). The likelihood K rule passes
  (HMM K* 4-6) exactly as it does for the smooth-continuum control.
- **Supervised models predict volatility; the rest is mechanical or tiny (5m /
  15m; walk-forward 2011-2021, one final test 2022-01-03 -> 2026-09-18).**
  Volatility h5: rank IC ~0.69 / 0.65 in development, 0.74 / 0.70 in the final
  test (boosters +0.03 over ridge; time of day is the only family beyond the
  statistics block that matters; 2026 is the weakest year). Reversion skill
  (up to +0.14 log-loss skill, AUC 0.74) is reproduced by a random walk and by
  the sign-flip null through the same pipeline - beyond the latter +0.002-0.008.
  Direction labels are mostly magnitude plus a drifting base rate; future
  return rank IC 0.046 in development fell to 0.021 in the final test with
  negative MSE skill. Families' predictions correlate 0.92-1.00 on volatility.
- **Ensembles add little because the models' errors are nearly the same (5m /
  15m; meta walk-forward 2013-2021, one logged final test 2022-).** Error
  correlation between eligible models is 0.90-1.00; simple averaging never beat
  the best single model by more than one SE and dilutes reversion badly (+0.05
  vs +0.14 log-loss skill, ECE ~0.07). The freeze rule kept an ensemble in 5 of
  13 pairs (stacking: 5m volatility, abs move, reversion c = 0.5, 15m residual
  reduction; regime-conditioned weights: 5m residual reduction), each +0.0005 to
  +0.009 over the best model, and those margins held on 2022- (5m volatility
  0.745 vs 0.741). At 15m the simple average beat the frozen single models for
  volatility and abs move on 2022- (+0.006 / +0.010). Disagreement and OOD do not
  flag errors beyond volatility (|rank corr| <= 0.04 for volatility, abs move,
  return, direction); for reversion, high disagreement means *lower* loss.
