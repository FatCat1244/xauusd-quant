# Work log - xauusd-quant

A running record of everything done in this repository: what was asked, what
was built and run, what came out, what broke and how it was fixed, and what is
still open. Newest stage last. Times are local (UTC+7).

The *how* lives elsewhere and is not repeated here: `README.md` (every layer,
its commands and its findings), `CLAUDE.md` (working notes, invariants,
pitfalls, measured dataset facts). This file is the *history*.

---

## At a glance (2026-10-04)

| Item | State |
|---|---|
| Raw file | `2026.9.18XAUUSD_oanda-TICK-No Session.csv`, 33.75 GiB, read-only, never modified |
| Tick dataset | `ticks-2e173ef8e61bd240`: 729,244,369 rows, 281 monthly partitions, 2003-05-05 03:01 -> 2026-09-18 23:59, 9.13 GiB, independently verified |
| Bars | 1m 7,937,007 / 5m 1,663,606 / 15m 561,817 / 30m 282,256 / 1h 141,569 |
| Layers done | 1 data, 2 statistics, 3 rolling regression, 4 OU, 5 spectral, 6 wavelets, 7 regimes, 8 feature factory + predictive research, 9 feature selection (5m, 15m, 30m, 1h), 10 supervised predictive models (5m, 15m), 11 ensembles / meta-models / prediction contract (5m, 15m) |
| Manifests | `results/feature_selection/<tf>/manifests/*.json` (FEATURESET_<TF>_<SET>_V001, immutable) |
| Frozen models | 37 specs (`results/ml_research/<tf>/frozen/`, artifacts `data/models/<id>/`, `config/model_registry.yaml`); final test done once (2026-10-01 08:30) |
| Frozen ensembles | 13 ENSEMBLE_SPECs + 82 constituent specs (`results/ensemble_research/<tf>/frozen/`, `config/ensemble_registry.yaml`); final test once per spec (2026-10-01 22:22-22:30, a logged second look at 2022-) |
| Tests | 1,259 passed, 0 failed (three processes 22:48-23:07, one real-data test rerun alone after a DuckDB allocation failure, 2026-10-01); ruff and mypy clean |
| Research ledger | ~389,480 rows (+565 ENS-H for Prompt #11) |
| Git | branch `main`, pushed to the private GitHub repo `FatCat1244/xauusd-quant` (first commit 2026-10-02); code, config, tests, notebooks and docs only - data, results, logs and `.venv` stay local |
| Current extension | Stage #17 read-only MT5/shadow framework; 244 targeted tests passed, eight guard breaks detected; synthetic equality and native demo identity verified; bounded native capture returned zero ticks/bars, fresh live path unvalidated; zero eligible alphas/unconfigured risk; no broker orders; stop before #18 |

**Open decisions (yours):** 1m spectral, wavelet and regime studies; the
all-window post-hoc wavelet run; regenerating the Prompt #2/#3 results on the
full dataset. Details at the end.

---

## Timeline

| When | What |
|---|---|
| 2026-09-20 19:48 | Prompt #1: data foundation |
| 2026-09-21 00:12 | Dataset replaced with the 2003-2026 export; "redo everything" |
| 2026-09-21 12:32 | Prompt #2: statistical research layer |
| 2026-09-24 19:55 | Prompt #3: rolling regression and residual research |
| 2026-09-24 21:37 | Claude Code updated (`npm i -g @anthropic-ai/claude-code`) |
| 2026-09-24 21:53 | Prompt #4: Ornstein-Uhlenbeck layer |
| 2026-09-24 22:06 | Scope correction: restore the full 2003-2026 dataset first (Phases A-J) |
| 2026-09-25 00:22 | OU grid stopped by Claude Code for low memory |
| 2026-09-25 03:48 | You chose "Resume all 34"; the grid completed |
| 2026-09-25 14:00 | Prompt #5: Fourier / FFT spectral research |
| 2026-09-25 ~18:00 | Spectral 1m study stopped for memory; not restarted |
| 2026-09-27 01:13 | Prompt #6: wavelet / time-frequency research |
| 2026-09-27 04:28-06:13 | Post-hoc wavelet run; "stopped" for memory, tree found alive and ended |
| 2026-09-27 morning | Prompt #6 final report delivered |
| 2026-09-27 22:42 | This log started |
| 2026-09-27 evening | Prepared the move from the laptop to the PC (sizes, checksum, pinned packages) |
| 2026-09-28 | Raw CSV on the PC: SHA-256 and blake2b-256 both match the references |
| 2026-09-28 ~16:40 | Prompt #7: unsupervised regime discovery; 795 tests pass on the PC, gate passes |
| 2026-09-28 19:31 | Regime grid run 1 (1h); stopped by Claude Code for low memory ~20:00 |
| 2026-09-28 22:04 | You chose "Resume all, then 5m"; grid run 2: 1h 22:04-23:51, 30m -03:39, 15m from 03:39 |
| 2026-09-29 04:44 | 15m K-Means and GMM-diag done; full GMM in its walk-forward |
| 2026-09-29 04:52-05:14 | Synthetic controls; post-hoc feature-geometry check at 1h |
| 2026-09-29 07:45 | 15m done (GMM 68 min, HMM 140 min); 5m -Heavy started |
| 2026-09-29 ~13:38 | Grid run 2 stopped by Claude Code for low memory (idle session) during 5m HMM K = 6 |
| 2026-09-29 15:39 | You chose "Resume the 5m remainder"; grid complete 16:54 |
| 2026-09-29 16:57-17:47 | Post-grid steps, 30m post-hoc check, full suite (887 passed) |
| 2026-09-29 evening | Prompt #7 final report delivered; Prompt #8 not started |
| 2026-09-29 18:10 | Prompt #8: feature factory + predictive feature research |
| 2026-09-29 ~22:00 | Usage limit before the Prompt #8 report; you sent Prompt #9 |
| 2026-09-29 23:04-23:45 | Prompt #8 clean runs 1h, 30m, 15m (one at a time) |
| 2026-09-29 23:46 | Prompt #8 5m run started; Prompt #9 built beside it |
| 2026-09-30 00:34 | Prompt #8 5m done (47.6 min, 3.1 GB); `--compare` |
| 2026-09-30 00:35 | Prompt #9 dev outputs deleted; final selection runs 1h -> 30m -> 15m -> 5m started (`logs/fs9_*`) |
| 2026-09-30 00:40-01:01 | Selection 1h (5.0 min, 1.13 GB), 30m (7.9 min, 1.39 GB), 15m (13.4 min, 1.76 GB) complete |
| 2026-09-30 ~01:19 | 5m selection stopped by Claude Code for low memory (session idle at the usage limit) in the nested stage; no process survived; not restarted - waits for you |
| 2026-09-30 16:01 | You chose "Resume 5m now"; resumed (`logs/fs9_5m_resume.*`), stamped stages skipped |
| 2026-09-30 16:31 | 5m selection complete (30.8 min resumed, peak 3,005 MB; free memory 1.3-2.5 GB throughout); `feature-selection-report --compare` |
| 2026-09-30 16:32-16:55 | Full suite: 1003 passed (22.5 min, 3.9 GB); ruff, mypy clean; README / CLAUDE.md updated |
| 2026-09-30 ~17:05 | Prompt #9 report and compact Prompt #8 report delivered; Prompt #10 not started |
| 2026-09-30 ~17:10 | Prompt #10 received (supervised predictive research); status check; `ml` extra installed |
| 2026-09-30 17:30-21:05 | `ml/` package, stages, reports, freeze, final test, figures, CLI; 102 new tests pass; three checks re-broken and seen to fail |
| 2026-09-30 21:04-21:26 | Full suite: 1105 passed (21.7 min, 4.4 GB) |
| 2026-09-30 21:30-23:05 | End-to-end synthetic test found the unit-cache miss (JSON stamps), fixed; figures redrawn as small multiples; ablation / registry / per-target views added |
| 2026-09-30 23:09 | 5m supervised run started (all stages, `logs/ml10_5m.*`) |
| 2026-10-01 01:39-01:46 | One search unit failed (MemoryError, system memory pressure); the run's process tree ended ~01:41-01:46 during the search stage (4 stages + 25 search units done, all kept) |
| 2026-10-01 01:49 | New session: "continue prompt#10"; 107 ML tests pass; dry run shows every finished unit current |
| 2026-10-01 01:54 | 5m search stage resumed (`logs/ml10_5m_r2_search.*`, memory logged) |
| 2026-10-01 02:04 | 15m supervised run started in parallel (`logs/ml10_15m.*`) |
| 2026-10-01 04:30 | 5m ablation stopped (design cache growing) and fixed; resumed 04:31 |
| 2026-10-01 05:55 | Post-hoc sign-flip pipeline null added (before any pipeline-null result); 15m split into processes |
| 2026-10-01 07:27 | Every 15m unit done; 15m report, freeze (17 specs), finalize |
| 2026-10-01 08:03 | Every 5m unit done (pipeline nulls alone); 5m report, freeze (20 specs), finalize |
| 2026-10-01 08:30 | **One-time final test** of all 37 frozen specs on 2022-01-03 -> 2026-09-18 |
| 2026-10-01 08:33-08:55 | Full suite: 1,118 passed |
| 2026-10-01 09:0x | Ledger duplicate and two figures fixed; reports rerun; final suite rerun |
| 2026-10-01 afternoon | Prompt #11 (with `AGENTS.md`); Step 1 suite 1,120 passed (15:36-16:02); ensemble package, tests, two Codex reviews |
| 2026-10-01 16:20-16:43 | Provisional ensemble run; stopped, outputs deleted after the packaging decision |
| 2026-10-01 16:44-17:16 | Final ensemble research run (14 pairs, 565 ENS-H rows, 151 figures) |
| 2026-10-01 21:1x-22:14 | Freeze (13 ENSEMBLE_SPECs); finalize 5m (MemoryError, rerun) and 15m |
| 2026-10-01 22:16 | First ensemble final-test attempt: MemoryError before any metric; counted as a look |
| 2026-10-01 22:22-22:30 | **Ensemble final test**, once per spec (5m with a logged repeat reason, 15m) |
| 2026-10-01 22:36-23:17 | Full suite in three processes: 1,259 passed (one real-data test rerun alone); docs; Prompt #11 report |
| 2026-10-01 23:5x-10-02 00:1x | "commit and push to github": Git for Windows and GitHub CLI installed; private repo `FatCat1244/xauusd-quant`; first commit pushed |

---

## Standing rules (from your instructions)

- The work arrives as numbered prompts; each ends with "Do NOT proceed
  automatically to Prompt #N+1". Finish, report, stop.
- The raw file is never deleted, modified, overwritten, cleaned in place,
  renamed or moved. Recovery works on generated data only. Deletions are
  targeted, never of ambiguous paths, never "because disk is high", and each
  is logged (path, size, reason, reproducible?) - see
  `data/metadata/cleanup_log.json`.
- No conclusions from a partial dataset. Anything computed on one carries
  `PARTIAL DEVELOPMENT SAMPLE — NOT FULL RESEARCH RESULT`.
- Stages #1-#11 were descriptive/predictive only. Prompt #12 explicitly
  authorizes offline execution, a separate fixed reference policy, cost/PnL
  and equity accounting. Broker access, demo/live trading, deployment and
  choosing features/models/policy variants by profitability remain prohibited.
- Every claim beside null controls; negative results kept; hypotheses in the
  research ledger with multiple-testing counts (from #5), registered before
  evaluation (from #6). "Do not force a positive conclusion."
- Features at t use only data up to t; offline pictures are labelled
  `NON-CAUSAL — VISUALIZATION ONLY` and never stored as features (#6).
- A run stopped by Claude Code for memory is restarted only when you say so.
- Commit only when asked.

---

## Prompt #1 - data foundation (2026-09-20)

**Asked:** the repository and data pipeline only - inspect the raw file
without loading it into RAM, canonical tick schema, validation, conservative
cleaning with an audit trail, streaming resumable Parquet conversion
(ZSTD, Hive `year=/month=`), metadata, bars 1m-1h, diagnostics, DuckDB query
layer, CLI, logging, tests, README. Polars first, pandas only when justified.
No look-ahead, never modify raw data, never invent missing data, keep bid
and ask.

**Built (32 modules):** `config/{data,logging}.yaml`, `cli.py`,
`data/{schema,timezones,inspector,validator,cleaner,converter,resampler,
diagnostics,metadata,loader}.py`, `utils/{config,logging,paths,clock}.py`,
tests, scripts, README, CLAUDE.md, pyproject.

**Six bugs found by my own verification, each fixed with a regression test:**
a resume gap that lost a middle partition; Hive `month=01` read as VARCHAR
against BIGINT, so every tick query returned 0 rows; `timestamp_utc`
rendered in the machine's zone; `--overwrite` deleting non-pipeline files;
`datetime.utcnow()` deprecation; the partition digest depended on chunk
layout - three attempts before it was right, confirmed by re-breaking it.

**Timezone:** broker time = New York wall clock + 7 h (UTC+2/+3 on US DST
dates) for the whole span - release spikes settle 2011+, the CME settlement
lull settles 2003-2010. From January 2012 the provider stops exporting the
00:00-01:00 hour (an export change, not a clock change).

## Dataset switch (2026-09-21)

**Asked:** "I replaced the dataset ... from like 2003 till 2026 ... it has a
spread column, can we redo everything again with the new dataset".

- The new export hid 2,683,808 duplicate rows that the validator missed (it
  compared only adjacent rows; the repeats sit ~8,600 rows apart). Fixed with
  a windowed duplicate check. I told you plainly that two earlier "verified
  reproducible" claims had been wrong.
- The background conversion was stopped by Claude Code for low memory with
  19 of 281 months converted (2003-05 -> 2004-11). Prompts #2 and #3 ran on
  that partial sample.

## Prompt #2 - statistical research layer (2026-09-21)

**Asked:** returns, volatility, autocorrelation, distributions,
stationarity, time of day and sessions, multi-timeframe diagnostics - no
leakage, no parameter search, significance is not profitability.

**Built:** `config/research.yaml`, `research/{config,returns,distributions,
autocorrelation,volatility,stationarity,rolling,conditional,intraday,spread,
comparison,plots,reports}.py`, notebooks 01-06, `xq research-summary`.

**Fixed:** an Int8 overflow in session assignment (`hour * 60` wrapped, so
every bar landed in `off_hours` - it would have corrupted every session
statistic); rolling skewness vectorised (25 s -> 0.5 s per window, exact).

**Results:** computed on the 19-month partial sample only, now archived under
`results/_archive/partial_dev_sample_2003-05_2004-11/`. **Not regenerated
on the full dataset** (open item: `xq research-summary --all`).

## Prompt #3 - rolling regression and residual research (2026-09-24)

**Asked:** trailing-window OLS on log mid price, and the residual's
distribution, ACF, stationarity, decay, Z-scores, slope, R^2 and stability -
measured, never assumed; no window picked; not cointegration.

**Built:** `config/regression.yaml`, `features/{config,rolling_regression}.py`,
`research/{residuals,residual_stationarity,residual_decay,residual_extremes,
regression_plots,regression_reports}.py`, notebooks 07-10,
`xq regression-research`.

**Fixed:** `add_quantile_buckets` never emitted Q1 and merged the top two
quartiles - a production bug in every conditioning table; one of my own tests
assumed a centre outlier moves the OLS slope (it does not; leverage is lowest
there) and was rewritten.

**Results:** the 5 x 5 grid ran on the partial sample (archived with #2). On
it, residual decay was b = -0.0335 against -0.0320 for a detrended random walk
(mostly mechanical), and the reversal effect was 0.18-0.23 of the round-trip
spread. **Not regenerated on the full dataset** (open item:
`xq regression-research --all`). The regression *features* were rebuilt on
the full data and feed every later layer.

## Prompt #4 - Ornstein-Uhlenbeck layer, and the full-dataset recovery (2026-09-24 -> 09-25)

**Asked:** exact AR(1) -> OU mapping with validity cases, rolling causal
estimation, half-lives, fitted vs realised decay, innovation diagnostics,
stability, conditioning, controls. **Then corrected (22:06):** no final
research on the 19 months - restore the whole 2003-2026 dataset first
(Phases A-J), under strict raw-data safety rules, with dataset versioning in
every artefact, full-history reporting, and three cases (real residual,
detrended random walk, exact simulated OU).

**Recovery:**
- The source is not sorted: from 2021-01-04 the export repeats the opening
  54 min - 2 h of every trading week verbatim (301 splices, 2,683,507 late
  rows, all byte-identical; the dedup policy drops 2,683,808).
- New byte-level source index (`xq source-index`); converter rewritten to be
  partition-safe (per month: read -> hash -> parse -> validate -> clean ->
  stable sort -> `.partial` -> verify -> rename; manifest inside the dataset;
  transactional rebuild with `.building` / `.previous` swap; smoke mode).
- `xq verify-dataset`, dataset lineage in every result (raw fingerprint ->
  tick -> bar -> feature versions), versioned feature store.
- Parquet written with Polars (`arrow_schema=`), ~7x faster than pyarrow.
- Outcome: the full dataset above, verified (0 backward steps, 0 kept
  duplicates, 0 crossed quotes, 0 invalid prices, all digests match); bars
  and regression features (windows 32-512, 4.2 GB) rebuilt; stale generated
  files removed by targeted, logged deletes.

**OU layer built:** `config/ou.yaml`, `models/{config,ornstein_uhlenbeck}.py`,
`research/{ou_estimation,ou_decay,ou_diagnostics,ou_stability,
ou_conditional,ou_plots,ou_reports}.py`, notebooks 11-14, `xq ou-research`
(one child process per study, `--resume`, `--compare`).

**Runs:** 100 studies (5 timeframes x regression N x OU window M). The first
grid was stopped for memory; you chose "Resume all 34" and it completed.

**Fixed on the way:** an all-null column (dtype `Null`) broke a strict concat
and stopped the grid at 1h N=256 M=128 (`diagonal_relaxed` stacking);
memory - select columns before filtering, a matplotlib reference cycle kept
frames alive (`gc.collect()`), mimalloc kept freed pages
(`MIMALLOC_PURGE_DELAY=0` for children).

**Findings:** the residual's OU behaviour is what detrending a random walk
produces - valid fits in 95-100 % of windows for both, half-life ~0.18 N bars
set by the regression window, realised decay matching the fit on the control
too. Faster reversion at 1m-15m is the noisy 2003-2007 feed. What is specific
to XAUUSD is in the innovations: kurtosis 24-34, volatility clustering (|eta|
lag-1 ACF 0.28-0.40), a daily cycle in their size (ACF at one day 0.20-0.27).

## Prompt #5 - Fourier / FFT spectral research (2026-09-25 -> 09-26)

**Asked:** causal rolling FFTs of the residual, the OU innovation and returns
(not raw price); entropy, flatness, bands, top-k, phase; persistence,
reconstruction, extrapolation, phase outcomes, IC; four nulls and positive
controls; a hypothesis ledger (`SPEC-H-xxx`); questions A-L, report items
1-33.

**Built:** `config/spectral.yaml`, `features/{spectral_config,spectral,
spectral_bands,spectral_entropy}.py`, `research/{fft_analysis,spectral_nulls,
spectral_stability,spectral_phase,spectral_reconstruction,spectral_ic,
research_ledger,spectral_plots,spectral_reports}.py`, notebooks 15-19,
`xq spectral-research`.

**Runs:** 60 studies (5m-1h x 3 series x N 64-1024) plus 15 on |eta|
(15m-1h). The 1m study was stopped for memory; its orphaned children were
ended; not restarted.

**Fixed:** components were leakage bins, now spectral peaks (a Hann window
leaks half the amplitude into each neighbour); later, during #6, a zero-IQR
null was ignored by the verdict (`_effect`), giving false "exceeds all nulls"
- fixed with a test, 16 ledger rows re-derived.

**Findings:** nothing in the signed series survives the nulls; every
dominant period is fixed in bars (a window artefact); phase, persistence and
extrapolation are the nulls'. The one real structure is the one-trading-day
volatility cycle in |eta| - dominant in 23-73 % of 15m-1h windows against
< 1 % for the nulls, at 23.3 h at every timeframe. My recommendation then:
drop Prompt #6 or narrow it to |eta|. You sent Prompt #6.

## Prompt #6 - wavelet / time-frequency research (2026-09-27)

**Asked:** when scales become strong or weak, how long they last, and
whether that carries information beyond regression + OU + FFT + volatility
+ nulls. Strict causality; offline pictures labelled non-causal; hypotheses
registered before evaluation; models A-D; questions A-N, report items 1-40.

**Built:**
- `config/wavelet.yaml`; `features/{wavelet_config,wavelet,wavelet_energy,
  wavelet_entropy,wavelet_causal}.py`; `research/{study_io,wavelet_stability,
  wavelet_predictiveness,wavelet_analysis,wavelet_plots,wavelet_reports}.py`;
  notebooks 20-25; 9 test modules; `xq wavelet-research`;
  `scripts/wavelet_baseline_robustness.py`.
- A one-sided MODWT (`db4`) of our own: PyWavelets has none (`pywt.swt` is
  circular), so filters come from `pywt.Wavelet` and are applied by direct
  convolution - bit for bit prefix-invariant, no padding.
- Six nulls (white noise, random walk, shuffled, block bootstrap 64/256/1024),
  13 registered hypotheses `WAVE-H-001..013`, nested models A-D on
  chronological folds.

**Fixed on the way:** the raw dominant band is always the finest one for
white noise, so it is now the band most above its white-noise share; model A
lacked short-horizon volatility (added 5/20-bar realised volatility and mean
|eta|; report version 2); runtime estimates recalibrated (0.8 h -> 1.7 h;
actual 1 h 45 m); **undefined statistics (NaN) were recorded as verdicts** -
180 IC tests as "within null range", 6 redundancy tests as "redundant" - now
"untested", each corrected row with a dated note, test added and re-broken to
confirm it fails; IC test counts now count only defined ICs.

**Runs:** 48 studies (5m-1h x 3 series x N 128-1024), 1 h 45 m, peak ~3.2 GB.
1m not run (heavy; its #5 FFT features lack log returns).

**Findings:** apart from the daily volatility cycle (energy moving to fine
scales in active sessions), nothing survives the nulls - shape, dominant
scale, persistence, drift, bursts, ridges, conditioning, extremes, IC. The
registered improvement of model D over C (229 of 1,008 tests, volatility-type
targets only) is volatility over longer horizons and the time of day:

| Baseline A/B (post-hoc WAVE-P-001, N = 512, 108 cases) | Beyond every null |
|---|---|
| registered | 52 |
| + hour of day | 43 |
| + ln RV and ln mean abs(eta) over 64, 256, 1024 bars | 9 |
| + both | 1 (adds 0.0004 R^2) |

No wavelet feature is worth carrying forward. Full numbers: README,
"Wavelet / time-frequency research -> Findings".

**The stopped run:** the post-hoc script over every window was "stopped" by
Claude Code for memory while the session sat idle at the usage limit
(04:33). On Windows nothing died: the wrapper, its memory logger and Python
were still running at 06:12, 85 % done. I ended the tree. The script wrote
only at the end, so nothing was kept; it now writes each timeframe to
`post_hoc/parts/` as it finishes and `--resume` continues. The ledger holds
its 1h N = 512 rows; the 5m-30m numbers in the table come from a scratch run
of the same code (identical to 4e-16 where they overlap).

**Delivered:** the final report (items 1-40, questions A-N). Not started:
Prompt #7.

## Moving to the PC (2026-09-27, evening)

**Asked:** can the folder be copied to a flash drive and from there to the
PC's HDD, to work on the PC instead of the laptop? Yes - prepared:

- **Size without `.venv`:** ~54 GB (raw CSV 33.75, `data/parquet` 9.14,
  `data/features` 6.56, `results` 3.49, `data/bars` 0.61 GB; ~16,000 files).
  The raw CSV alone exceeds FAT32's 4 GB file limit: the drive must be exFAT
  or NTFS, and 64 GB is tight - 128 GB is comfortable.
- **Nothing is tied to the path:** `config/data.yaml` paths are relative to
  the repo root; the raw fingerprint and every dataset version are computed
  from file *content* (and name), so the folder can live anywhere (e.g.
  `D:\XAUUSD`). The byte index `data/metadata/source_index.json` also notes
  path and mtime, and rebuilds itself (one read of the CSV) the next time
  `convert` or `source-index` runs; research commands and `--resume` do not
  depend on it.
- **Do not copy `.venv`** (0.77 GB, 18,462 files; it points at
  `C:\Python314` by absolute path). Recreate it on the PC.
- **Reference checksum (laptop):** SHA-256 of
  `2026.9.18XAUUSD_oanda-TICK-No Session.csv` =
  `FCFB5E9D894711BC2F4722C50C7BC55372B290D15136252826C90179CBE8BAA9`
  (36,233,955,746 bytes).
- **Exact environment:** Python 3.14.4; `requirements-lock.txt` pins all 43
  packages (numpy 2.5.3, scipy 1.18.1, polars 1.44.2, pyarrow 25.0.1, duckdb
  1.5.5, PyWavelets 1.10.0, statsmodels 0.15.0 ...), so the PC reproduces the
  laptop's numbers.

On the PC, after copying:

```powershell
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-lock.txt
.venv\Scripts\python -m pip install -e . --no-deps
Get-FileHash -Algorithm SHA256 "data\2026.9.18XAUUSD_oanda-TICK-No Session.csv"   # must match above
.venv\Scripts\xq verify-dataset          # partition digests: detects a corrupted copy
.venv\Scripts\python -m pytest            # 795 tests, incl. 15 on the real data
```

Keep one working copy at a time (laptop *or* PC) so the two never diverge.
CLAUDE.md's memory notes describe the laptop (16 GB); update them if the PC
differs. Claude Code's own notes live outside the repo
(`%USERPROFILE%\.claude\projects\<project>\memory\`); `CLAUDE.md`,
`README.md` and this log carry the essentials either way.

---

## Raw-file check after the copy (2026-09-28)

**Asked:** SHA-256 of the raw CSV, then the blake2b fingerprint too. Both
read-only, run on this machine (`E:\XAUUSD`):

- SHA-256 `FCFB5E9D894711BC2F4722C50C7BC55372B290D15136252826C90179CBE8BAA9`
  - **matches** the laptop reference above.
- blake2b-256 (whole file, as `source_index.py` computes it)
  `e15005cea920ac95ed2b0dc46981e0790153fb5ec29f2dd6c5e7edeb8800222c`
  - **matches** `data/metadata/source_index.json`, i.e. the raw fingerprint
  every dataset version and lineage chains from.

The raw file arrived byte-identical. Still to run from the checklist above:
`xq verify-dataset` (partition digests) and `pytest`.

---

## Prompt #7 - unsupervised regime discovery (2026-09-28/29)

**Asked:** K-Means / GMM / Gaussian HMM regimes (K = 2..6), offline vs strictly
causal walk-forward inference, baselines (single state, volatility buckets,
random labels), nulls, outcomes, incremental information, questions A-O.

**Checked first:** 795 existing tests pass on the PC (20 min); integrity gate
passes for 5m-1h (bars, regression N=128, stored FFT/wavelet log-return
features current; last 3000 bars recomputed = stored).

**Built:** `config/regimes.yaml`; `regimes/` (config, dataset, preprocessing,
emissions, clustering, gmm, hmm, state_alignment, transitions, diagnostics,
fitting, causal_inference, registry, synthetic, changepoints);
`research/regime_{analysis,outcomes,stability,nulls,plots,reports}.py`;
`xq regime-research`, `xq regime-walk-forward`; notebooks 26-31;
`scripts/regime_grid.ps1`; 6 test modules (88 tests, all pass; ruff, mypy
clean). Own NumPy HMM (no sklearn/hmmlearn here): blocked forward filter,
verified against textbook recursions and brute force, bit-for-bit
prefix-invariant (walk-forward too). `wavelet_predictiveness._design` gained an
optional block order (default unchanged).

**Disclosed rule change:** after one smoke run (1h HMM K=3) REG-H-001/-007 were
made *stricter* - an increment must also beat the same model on the pipeline
nulls - because the smoke run's states were trend/R^2 states, which shape
residual decay mechanically.

**First result (1h HMM K=3, smoke):** states are trend/linearity states, not
volatility (NMI with vol buckets 0.001); persistence is the nulls' (median
expected duration 30 bars vs random walk 27, shuffled returns 58, block
bootstrap 62; row shuffle 1.5); increments ~0.1-0.2 % of log-loss.

**Grid run 1 (19:32-~20:00), stopped by Claude Code for low memory while the
session was idle** (not a fault of the run: its peak was 1.7 GB). This time
the whole tree was gone (checked). It had written, and `--resume` reuses: all
20 offline 1h fits (K-Means, GMM full/diag, HMM x K 2-6), walk-forward GMM
K=2 and HMM K=2-3 (expanding and rolling). Two bugs it exposed, both fixed
(in `regime_reports.py`, which no cache key depends on): the offline table
assumed a likelihood column K-Means lacks; the entropy analysis mixed the
full walk-forward frame (with a GMM's unscored bars) and the scored rows. Not
restarted - waiting for the user. **22:04: the user chose "Resume all, then
5m"**; grid run 2 started (1h, 30m, 15m, then 5m -Heavy). Its 1h K-Means step
failed on a Windows "Access is denied" renaming the model registry's
`_index.json` (a scanner holding the just-written file); `atomic_write_bytes`
now retries the rename for ~3 s (`utils/paths.py`, no cache key touched).
The K-Means 1h step is re-run after the grid. **Cost decision (23:05):** the
pipeline-null increment check (two null walk-forwards per K) costs ~5-10 min
per K at 1h and ~12x that at 5m; 1h ran it for every K, 30m-5m run it for
K = 3 only (`nulls.increment_states`, not part of any cache key); elsewhere
REG-H-001/-007 read "untested_against_pipeline_nulls". **01:00:** the 30m GMM
walk-forward took 48 min (1h: 17 min), so at 5m (-Heavy) the full GMM runs
offline only; HMM vs GMM walk-forward (REG-H-002) comes from 15m-1h. Measured
peaks so far: 1.2 GB (1h), 1.9 GB (30m). **01:30:** persistence-null fits take
~3 min each at 30m (35 per timeframe); 1h and 30m run K = 2..6, 15m and 5m
run K = 3 (`nulls.run_states`, kept out of the config fingerprint - checked
unchanged, 28333752d232503c - so no cache went stale). Still to run: the rest of 1h, 30m, 15m
(`scripts\regime_grid.ps1 -Timeframes 1h,30m,15m`), 5m (`-Timeframes 5m
-Heavy`), `--controls`, `--compare`, the full suite, README / CLAUDE.md, the
final report. Do not edit engine files while the grid runs (stage caches are
keyed by their code).

**Grid run 2 progress (measured, 2026-09-29):** 1h done 22:04-23:51 (K-Means
failed on the rename above; HMM 86.8 min, GMM 16.8 min); 30m done -03:39
(K-Means 11.7, GMM-diag 9.1, GMM 48.1, HMM 159.6 min, peak 1.9 GB); 15m
K-Means 18.8 min (peak 2.2 GB), GMM-diag 18.6 min (1.9 GB), full GMM
04:17-05:25, 68.1 min (offline K = 2..6 24 min, K = 6 alone 11 min; walk-forward
3-13 min per K); its logged peak 2.76 GB includes the controls and the
post-hoc check below, which ran beside it (0.28 + 0.50 GB). 15m HMM 05:25-07:45,
139.8 min, peak 2.5 GB (offline 20 min; persistence nulls K = 3 ~1-2 min each;
walk-forwards 8 min at K = 2 to 28 min at K = 6; monthly K = 3 15 min;
pipeline-null increments for K = 3 13 min). No step failed at 30m or 15m.
**5m (-Heavy) started 07:45.** K-Means 61.9 min (peak 2.9 GB), GMM-diag
offline 58.3 min (2.6 GB), GMM offline 65.9 min (2.5 GB); HMM from 10:51:
offline 47 min, persistence nulls K = 3, walk-forwards and analyses K = 2..5
done (K = 3 with its pipeline-null increments, 12:00-12:40; free memory fell
to 2.3 GB there, python 3.2 GB).

**Grid run 2 stopped by Claude Code for low memory, ~13:38:45, while the
session sat idle** (the PowerShell safety classifier had failed six times in a
row, so monitoring stopped and the turn ended; the stop came within minutes).
Checked 13:38:49: no python process left, whole tree gone, 5.1 GB free.
Lost: the 5m HMM K = 6 expanding walk-forward in progress (29 of 75 refits
registered as HMM_5M_K6_00001-00029, ~14 min; the stage writes its cache only
at the end, so it restarts from refit 1 - identical refits dedup by content
hash in the registry). Not run: the K = 6 analysis, the step's `--compare`,
and the two rolling steps (5m HMM K = 2 and 3, expanding + rolling). Everything
else at 5m is on disk and reusable with `--resume`. **~13:50: you chose "Resume
the 5m remainder"** - the three missing steps only (HMM K = 6 expanding, then
K = 2 and 3 expanding + rolling), with the session kept active meanwhile.
Started 15:39 (the answer arrived then); stopped by me at 15:41 and restarted
at 15:41: with `--states 6` the offline stage (cached) would have redone the
robust-vs-standard scaling comparison at K = 6 - its representative K is the
first requested - over the stored K = 3 file (checked intact, stamp K = 3).
The restart skips the offline stage (`--stages walk_forward,analysis`).
**Grid complete 16:54:** 5m HMM K = 6 expanding 54.8 min (peak 2.9 GB),
K = 2 expanding + rolling 7.3 min (3.2 GB), K = 3 10.9 min (3.3 GB), all exit
code 0. The re-run K = 6 refits registered as HMM_5M_K6_00030-00104: the
first 29 IDs are the stopped run's orphans (the model metadata carries the
code fingerprint, which the 10:55 verdict fix changed, so the content hashes
differ and nothing deduplicated); the per-bar features name the IDs they used.
Post-grid steps started 16:57: 1h K-Means, analysis refresh (1h GMM/HMM, 30m
K-Means), `--compare`, post-hoc geometry at 30m. 1h K-Means 3.8 min (0.9 GB),
1h GMM / 1h HMM / 30m K-Means analyses 1.3 / 1.4 / 2.5 min (1h HMM read the
cached pipeline-null increments for every K, `--null-increment-states
2,3,4,5,6`), `--compare` 10 s: the 5m GMM's REG-H-004 now "untested".

**Fixed (17:06): two stale REG-H-004 rows.** 1h and 30m K-Means kept
"no_discrete_states_defensible" rows from 2026-09-28 17:02 UTC, written before
K-Means was made "not applicable" (no likelihood); the current code wrote no
row for K-Means, so the upsert never replaced them. `write_regime_comparison`
now records an explicit `not_applicable` row for a family without a
likelihood, which replaces them at the next `--compare`.

**Synthetic controls (04:52-04:56, beside the 15m GMM step; 4.3 min, own peak
280 MB, so the grid log's 15m GMM peak includes up to 0.3 GB of it):**
`xq regime-research --controls` -> `results/regime_research/synthetic_controls/`.
The machinery passes: a known 3-state HMM is recovered (hold-out
log-likelihood plateaus exactly at K = 3 for HMM and GMM, BIC minimum at 3;
self-transitions 0.988/0.981/0.972 vs 0.990/0.980/0.975; filtered accuracy
0.980, smoothed 0.994; K = 4-5 flagged flickering). One Gaussian regime: no
gain for any K, every HMM flickers. The two false positives the real data must
be read against: a single **Student-t** regime gives a mixture +0.26-0.28
nats/bar for K = 2-4 (tails, not states) but its HMM states flicker (expected
duration ~1.5 bars, temporal gain ~0); a **smooth continuum** (slow AR(1) log
volatility) improves with every K without a plateau (HMM K = 5: +3.4 nats),
with persistent states (durations 61-86 bars) that move only between
neighbours (adjacency share 0.97-0.99). Rows of the 3-state HMM shuffled: the
mixture still finds 3 components, the HMM's persistence vanishes.

**Found while reading 1h/30m (post hoc, descriptive):** the GMM/HMM "trend"
states (up drift + linear / down drift / less linear) sit on a near-functional,
V-shaped dependence between two inputs: R^2 is a monotone function of
|slope / volatility| (Spearman with |slope/vol| 0.926 at 1h, 0.914 at 30m;
a 20-bin step function of |slope/vol| explains 86 % / 84 % of R^2's variance),
while the signed pair has Spearman 0.13-0.15 - which is all the registered
redundancy rule (|Spearman| >= 0.9) looks at, so it let the pair through. A
full-covariance Gaussian cannot follow a V, so the mixture spends one
component on each arm and one on the vertex. In that trend ordering the HMM's
transition mass goes almost only between neighbours (adjacency share 0.996-
0.999 at K = 3-5, chance 2/K), as for the smooth-continuum control. Scratch
scripts: `vshape.py`, `report_numbers.py adjacency` (session scratchpad).

**Post-hoc check `scripts/regime_feature_geometry.py` (new; descriptive, not a
registered test, not in the ledger), 1h, 05:03-05:14 beside the grid (10.8
min, own peak 0.5 GB; a CSV-writer bug after the JSON was written, fixed and
the CSVs rewritten from the JSON):** the real series and the three pipeline
nulls over the same 141,442 bars, `results/regime_research/post_hoc/`.
- The V is the regression's own geometry: Spearman(R^2, |slope/vol|) real
  0.926, random walk 0.941, shuffled 0.927, block bootstrap 0.923.
- A GMM on (slope/vol, R^2) alone gains +0.60/+0.78/+0.94 nats/bar over one
  Gaussian on the hold-out for K = 2/3/4 - and +0.67/+0.86/+0.99 on the random
  walk, +0.54-0.93 on the other nulls - with the same three K = 3 states
  (downward drift / less linear / upward drift) in every source.
- HMM K = 3, full span: core features, median expected duration 29.6 bars vs
  27.4-32.7 for the nulls. Without R^2 the states turn into volatility states
  (real: "volatility high for its recent history (+0.7 IQR)"; NMI with
  volatility terciles 0.17 real, 0.47 random walk) and their persistence
  (44 bars) exceeds the random walk and shuffled nulls (14-18) but not the
  1024-block bootstrap (46), which keeps volatility clustering.
- The core-feature HMM K = 3 has more than one optimum: this seed's full-span
  fit (= the null-control "real" fit) is flat / upward drift / high recent
  volatility, the walk-forward's is upward / less linear / downward.

The same check at **30m** (17:05-17:22, alone, 17 min, peak 0.74 GB):
Spearman(R^2, |slope/vol|) real 0.913, random walk 0.942, shuffled 0.919,
bootstrap 0.909; pair-GMM gains K = 2/3/4 real +0.54/+0.71/+0.89, random walk
+0.69/+0.91/+1.06, the other nulls +0.53-0.90, same three states. HMM K = 3
core: 57.5 bars vs nulls 49-56. Without R^2: real 42.1 bars ("faster residual
reversion", median, "high volatility for its recent history") vs random walk
13.7, shuffled 18.7 and - unlike 1h - the block bootstrap 24.6. Not pursued;
a plausible reading (unverified) is the 2003-2007 quote-noise era, one long
"faster reversion" stretch that the bootstrap scatters.

**Fixed (10:50-11:04, during the 5m HMM step): the K rule read an unavailable
statistic as a failure.** The 5m GMM runs offline only (cost decision), so it
has no walk-forward refit overlap; `defensible_states` treated `ARI is None` as
"< 0.5" and REG-H-004 recorded "no_discrete_states_defensible" for it (and a
NaN would have *passed* both of the rule's comparisons). Now an unavailable or
non-finite step stops the rule as incomplete and `k_rule_verdict` records
"untested" unless K >= 2 was already reached (`regime_stability.py`). The
same hardening for every regime verdict: all compared numbers go through
`finite_or_none`, REG-H-001/-007 share `increment_verdict`, and REG-H-005/-006
/-009 now say "untested" when a control is missing instead of a verdict
(`regime_reports.py`). Checked first: no REG verdict so far rested on a NaN or
missing control (1,801 rows), so only the 5m GMM's REG-H-004 row changes, at
the next `--compare`. Tests: `test_the_k_rule_reads_an_unavailable_statistic_
as_untested`, `test_an_undefined_increment_is_untested_never_a_verdict`, both
re-broken and seen to fail (None and NaN). Neither file is in a stage cache
key.

`regime_feature_geometry.py --adjacency` (17:55, reads the stored studies, no
fitting) now reproduces the scratch adjacency numbers:
`post_hoc/transition_adjacency.csv`. 30m and 1h K = 3-4 0.998-0.999; 15m and 5m
~chance, because there the states split the slope / R^2 plane in two
dimensions.

**Verified at the end (17:24-17:48):** full suite **887 passed** (871 unit +
16 real-data; 88 regime tests + 4 new today) in 22.5 min, alone, peak 4.2 GB
(was ~3.3 GB before the regime tests); ruff and mypy (101 files) clean. The
ledger holds 2,115 REG rows, none resting on a NaN or missing input.
Storage: live-safe per-bar features 0.70 GB (`data/features/regime/`), inputs
0.39 GB, model registry 0.12 GB (6,507 files), results 3.0 GB (walk-forward
caches 1.73, null inputs 1.17, tables 0.03, 728 plots 0.05).

**Results (55 studies, 2,115 REG-H tests; detail in README "Unsupervised
regime discovery"):** no discrete regimes beyond volatility.
- GMM / HMM states are trend / linearity bands - the mixture's fit to the R^2
  vs |slope/vol| V, which a random walk reproduces (same states, same
  likelihood gain; post hoc at 1h and 30m). Their persistence is the
  pipeline nulls' at every K tested (REG-H-005: 0 of 12). NMI with
  volatility buckets <= 0.02 (HMM), <= 0.05 (GMM).
- K-Means states are volatility level and the daily cycle (NMI 0.14-0.33 for
  K >= 3; Cramer's V with the hour up to 0.33).
- K rule (REG-H-004): HMM 4 / 6 / 5 / 6 at 1h / 30m / 15m / 5m, GMM 5 / 3 /
  4 / untested, diagonal GMM 1, K-Means not applicable - but the same rule
  passes for the smooth-continuum control, so it cannot show discreteness.
- Refits are stable (ARI 0.97-0.996); definitions drift at K >= 4 (REG-H-013
  14 of 35); expanding beats rolling 16 of 17; confidence 0.95-0.98 but the
  leave probability is miscalibrated in 13 of 20.
- Mean-reversion differences by state exist but the nulls reproduce 52-97 %
  of them (REG-H-001: 13 beyond); OU half-life, spectra, wavelets, spread and
  activity do not differ across HMM states.
- Incremental information beyond model A: 14 of 825 beyond the nulls
  (REG-H-007), all <= +0.001 R^2 or nats, none for returns; a volatility-bucket
  one-hot adds up to +0.039. Soft beats hard in 206 of 825 (REG-H-008).
- Across timeframes the states disagree (30m-1h Cramer's V 0.21-0.39, the
  rest <= 0.11).

## Prompt #8 - quant feature factory + predictive feature research (2026-09-29, in progress)

**Asked:** a versioned, causal feature registry and factory (every family from
#1-#7), a separate target factory (future returns, residual / OU reversion,
volatility, absolute move, excursions), and alpha research: IC / rank IC,
decay and information half-life, yearly / quarterly / recent stability,
volatility / regime / intraday conditioning, mutual information, redundancy
clusters, interactions, deciles, null controls (shuffled, time-shift,
synthetic noise, random walks), every test counted in the ledger with FDR,
parameter and perturbation robustness, cost and live-buffer profiling,
candidate statuses and manifests for Prompt #9, ablation manifests A-G.
No supervised ML, no PnL. Report items 1-43 and questions A-S; do not start #9.

**Status before coding (Step 1, 18:10):** 887 tests passed at 17:47 on this
code; dataset `ticks-2e173ef8e61bd240` complete; 5m-1h bars, regression store
(windows 32-512), stored FFT (N = 256 only) and wavelet (N = 512 only) sets and
regime features current. Prior evidence: only volatility (level, clustering)
and the daily cycle survived the nulls in #5-#7; FFT / wavelet shape, regime
persistence and increments failed; residual reversion is largely mechanical.
Prompt #2/#3 result tables exist only for the archived partial sample - #8
re-measures those features on the full dataset. 1m excluded (no FFT / wavelet
log-return features at 1m; 7.9M bars).

**Built (2026-09-29, evening):** configs `features.yaml`, `targets.yaml`,
`alpha_research.yaml` (with 9 preregistered hypotheses ALPHA-PR-01..09 and a
ledger path); packages `alpha/` (IC, stability, decay, conditioning, MI, nulls,
ranking, interactions), `targets/`, and in `features/` the registry, families,
joins, manifest, factory, interactions, quality, redundancy, transformations;
`research/feature_research.py` (12 cached stages), `feature_reports.py`,
`feature_selection_research.py`, `feature_plots.py`; CLI `build-feature-factory`
(alias `build-feature-matrix`), `feature-research`, `feature-redundancy`,
`alpha-decay`; 9 new test files (84 tests, all pass, incl. real-data 1h).

**Found and fixed while building (all at 1h):**
- Noise features exposed a lenient procedure: 16 of 50 got a candidate status.
  Fixes: studentized Westfall-Young max-T circular-shift null over all targets
  of a feature (per-kind claims), 200 shifts, Newey-West (2 months) on the
  monthly batch sums -> 0 of 50.
- Range features escaped the random-walk control (nulls had no highs / lows):
  nulls are now whole OHLC paths; a residual claim without the pipeline
  control fails (invariant 9). Veto = a null reproduces >= half the real IC
  (same sign, significant); a sign-flip null (real bars, random signs) vetoes
  direction / residual claims only.
- The first IC loop would take ~30 min at 5m: per-month matrix products
  (identical to 1e-11); conditioning via segmented sorts; quality drift on a
  quantile grid (180 s -> 12 s at 1h).
- Centring sums by a full-sample mean let a trailing-window IC depend on later
  data in the last bit: now centred by each column's first value.
- The leakage test was re-broken deliberately (centred rolling mean): caught
  14 features.

**1h result (development run, before the last two code changes):** 27 strong /
33 candidate / 6 weak / 15 redundant / 2 unstable / 46 failed; 86 of 101
residual relations reproduced on the random walk; FFT / wavelet shape mostly
failed; volatility, activity, intraday session strong. Not final - rerun.

**Left to do (as of the usage limit):** 24 ruff findings and mypy; delete the dev
test registry and rerun 1h cleanly with the ledger on; measure memory at 15m,
then run 30m, 15m, 5m; notebooks 32-38; README / CLAUDE.md / memory; final report.

**Done since (2026-09-29, 22:00 - 09-30):** ruff and mypy clean (164 mypy errors
fixed); the module `feature_selection_research.py` of the first draft (Prompt #8
candidates / ablation manifests) renamed `feature_candidates.py` so the name
could go to Prompt #9. Clean runs, one at a time, ledger on
(`scratchpad/run_fr8.ps1` via `run_measured.ps1`):

| tf | time | peak | strong | candidate | weak | redundant | unstable | failed_null | tests | noise candidates |
|---|---|---|---|---|---|---|---|---|---|---|
| 1h | 6.9 min | 1,316 MB | 27 | 33 | 6 | 15 | 2 | 46 | 83,019 | 0 of 50 |
| 30m | 13.0 min | 1,414 MB | 29 | 35 | 11 | 19 | 2 | 33 | 83,019 | 0 of 50 |
| 15m | 21.2 min | 1,686 MB | 38 | 26 | 16 | 12 | 0 | 37 | 83,019 | 0 of 50 |
| 5m | 47.6 min | 3,081 MB | 43 | 30 | 20 | 11 | 11 | 11 | 81,733 | 0 of 50 |

5m ran 2026-09-29 23:46 - 09-30 00:34 (`logs/fr8_5m.*`; 5m matrix 127 features
x 1,663,606 bars built in 63 s). `feature-research --compare` written 00:35.
Ledger: 370,771 rows (39,981 before + ALPHA-H 3 x 83,019 + 81,733). Notebooks 32-38 written
(`research/`, no calculations; executed headlessly against the 1h results:
all cells run). Still open: `--compare` after 5m, README / CLAUDE.md, the
Prompt #8 report (items 1-43, questions A-S) - it was never delivered because
of the usage limit; you then sent Prompt #9.

## Prompt #9 - feature selection + dimensionality reduction (2026-09-29/30, in progress)

**Asked:** which compact, causal, live-safe, low-redundancy, time-robust,
reproducible subset of the Prompt #8 features should enter supervised ML in
Prompt #10. Chronological development / validation / reserved-test split
(never inspect the reserved period while selecting), quality filter with
reason codes, redundancy clusters and representatives by broad evidence,
parameter-family reduction, mRMR, multi-target and multi-horizon selection,
stability selection on chronological blocks, L1 / elastic net with train-only
scaling, PCA fitted on training only, family ablation and leave-one-family-out,
permutation importance, nested chronological CV with purging and embargo,
Minimal / Standard / Extended and target-specific sets, collinearity, cost,
warm-up, streaming reconstruction, immutable manifests, leakage audit,
reserved-test isolation, every experiment in the ledger, plots, CLI, tests,
notebooks 39-44; report items 1-40 and questions A-Q. No XGBoost / LightGBM /
CatBoost / random forest / neural network, no PnL. Do not start Prompt #10.

**Design decisions:** periods development 2003-2017, validation 2018-2021,
reserved 2022- (quarter-aligned; `config/feature_selection.yaml`); nested
folds validate 2011-12, 2013-14, 2015-17 inside development; purge 50 bars
(>= the longest horizon, 20) plus a 12-bar embargo. The loader never opens a
reserved-year target file and filters at the scan; every target is purged at
the period boundaries by its own horizon; a guard refuses any outcome row at or
after the reserved start. Prompt #8 statuses saw 2022-2026 outcomes, so they
are reported but never used to select: the default universe is re-screened on
development only (Prompt #8's studentized max-T circular-shift null + the
pipeline veto on development quarters). Targets: direction (return),
reversion (residual_reduction, needs the random-walk pipeline control),
volatility (realized_vol), magnitude (abs_return), h = 1, 5, 20. Diagnostic
models are linear on training-CDF rank designs from sufficient statistics
(ridge, Lasso / naive elastic net by coordinate descent, L1 logistic by FISTA -
scikit-learn is not installed, so these are our own, tested against least
squares and known supports).

**Built:** package `selection/` (config, periods, data, relevance, filters,
redundancy, clustering, mrmr, linear, pca, stability, selection_cv, ablation,
manifest, live); `research/feature_selection_research.py` (7 cached stages:
universe, redundancy, evidence, stability, nested, sets, audit),
`feature_selection_reports.py` (summary, markdown, SEL-H ledger rows, trial
counts, Step 67 tables), `feature_selection_plots.py` (11 figures); CLI
`feature-select`, `feature-selection-report`, `build-feature-manifest`;
8 test files (40 tests, all pass; two guards re-broken deliberately and seen
to fail: a loader opening reserved years, training rows reaching evaluation);
notebooks 39-44 (run headlessly against the 1h dev outputs).

**Found and fixed while building (1h development runs):**
- Stability selection was uninformative for thin targets: with a pool not much
  larger than k = 20 (direction h = 5 / 20, reversion), every candidate - noise
  probes included - was selected in every resample, so the probe maximum was
  1.00 and those target sets came out empty for a mechanical reason. Now a run
  keeps the features that enter before the first noise probe (random-probe
  rule, Stoppiglia et al. 2003), at most k; top-k frequencies kept as a
  diagnostic. Result at 1h: 9-12 features enter before a probe for 1-bar
  direction, ~4.5 for h = 5, 1-2 for h = 20, 2-4.5 for reversion, 13-17 for
  volatility / magnitude.
- Stability candidates were one pool for all targets, so a mechanical residual
  feature that passes for volatility could compete for a reversion target: the
  pool is now each target's own development pass list plus the probes.
- Standard (30) came out larger than Extended (27): all general sets are now
  prefixes of one ranking (stable features first).
- The pipeline veto read the selection's kind names; Prompt #8's are
  "residual", not "reversion" - mapped.
- The synthetic test generator wrote warm-up NaNs into the arrays that drive
  its targets, which made one purge assertion pass vacuously; fixed and rerun.
- Features no resample selected tied at score 0 and were ordered by name, so a
  size-50 prefix held an alphabetical choice; the tail is now ordered by the
  best development effect stability.
- The loader read the 2022 target file and filtered afterwards (reserved values
  briefly in memory, never used); it now skips every year file starting at or
  after the reserved date and filters at the scan.
- `live_reconstruction` copied every stored column to float64 in full (~0.9 GB
  at 5m) to compare 60 rows; it now gathers those rows only.
- Disclosure: the probe-stopping rule, the per-target pools, the prefix rule and
  the tie-break were changed after looking at 1h development-run outputs
  (stability frequencies, set sizes, 1h validation curves). None used the
  reserved period; 5m, 15m and 30m were not looked at before their final runs.
- Config: `timeframes` now [5m, 15m, 30m, 1h] (5m primary; the others show
  whether a selection holds across bar lengths).

**1h development result (not final - to be rerun clean):** 129 registered ->
115 quality universe (FFT / wavelet 512-1024 windows fail validation
missingness, `ou_valid_256` near-constant) -> 70 pass the development screen;
Minimal 20, Standard 30, Extended 64, general 15; target sets direction 9
(validation rank IC 0.045), reversion 4, volatility 15 (0.60), magnitude 17
(0.36). Plateau: mean relative validation IC 0.74 (5) / 0.76 (10) / 0.94 (20)
/ 0.99 (30) / 0.88 (50). Leave one family out: statistics -0.097, time -0.027,
regression -0.011, OU -0.010, the rest ~0. PCA with 30 components ~ raw
features, 5 components clearly worse. Live reconstruction 60 steps passed;
reserved isolation passed.

**Final runs (2026-09-30, `scratchpad/run_fs9.ps1`, one at a time):** dev
outputs deleted first (no SEL-H rows existed). 1h 5.0 min / 1.13 GB, 30m 7.9 min
/ 1.39 GB, 15m 13.4 min / 1.76 GB: every stage, report, manifests, SEL-H ledger
rows (1h 4,356; 30m 4,368; ledger 379,495 after 30m), live reconstruction and
reserved isolation passed. 5m: universe, redundancy, evidence and stability
done (01:01-01:05); the nested stage finished 3 of its 4 splits (01:18:45) and
was stopped by Claude Code for low memory ~01:19, while the session was idle at
the usage limit. The nested stage writes its tables only at the end, so the
three finished splits (~13 min) are lost; a resume skips the four stamped
stages. No process survived. The compare step did not run.

| tf | quality | universe | Minimal | Standard | Extended | general | direction | reversion | volatility | magnitude |
|---|---|---|---|---|---|---|---|---|---|---|
| 1h | 115 | 70 | 20 | 30 | 64 | 15 | 9 | 4 | 15 | 17 |
| 30m | 123 | 88 | 50 | 50 | 75 | 16 | 15 | 8 | 15 | 17 |
| 15m | 126 | 78 | 20 | 72 | 72 | 17 | 11 | 6 | 15 | 18 |
| 5m | 124 | 86 | 20 | 20 | 75 | 17 | 19 | 13 | 17 | 17 |

5m was resumed on your go-ahead at 16:01 (nested 4 splits 26 min, sets 2 min,
audit 2 min; 30.8 min, peak 3,005 MB) and completed at 16:31.

The registered plateau rule (mean over targets of IC / best IC) swings with the
weak direction targets: volatility and magnitude reach >= 97 % of their best
validation IC with the first 20 features at every timeframe, direction needs
30-75 (post-hoc sensitivity, `scratchpad/plateau_sensitivity.py`, not
registered, changes no set).

**Tests (2026-09-30 16:32-16:55, alone):** 1003 passed (984 unit + 19
real-data; 887 before #8, +76 Prompt #8, +40 Prompt #9) in 22.5 min, peak
3,856 MB resident (up to 5.1 GB committed; commit headroom fell to 1.2 GB with
the desktop apps open). ruff and mypy clean after the last edits.

**Post-hoc reading (descriptive, changes no set):** `scratchpad/set_flags.py` -
the "regime_specific" list in `summary.json` flags a feature if *any* of its
(target, conditioning) rows fails, so nearly every feature is on it; read per
feature's strongest relation instead: at 5m the short-term reversal features
(ret_1..ret_10, reg_resid_z_64) are regime-state specific, the volatility
features only tercile-specific (conditioning on a volatility tercile removes
most of what they rank). The mRMR "rank stability" (0.97-1.00) is inflated by
out-of-pool candidates fixed at zero in every run; the resample Jaccard is the
informative number. Probes passing the development screen: 2 of 32 probe x
timeframe screens (5m AR(0.9) for direction with IC 0.002-0.006 at n ~ 1M;
15m AR(0.999) for magnitude, IC -0.05); no probe reaches a set.

**Reports:** Prompt #9 report (items 1-40, questions A-Q) and a compact Prompt
#8 report (status, findings, questions A-S) delivered 2026-09-30. Prompt #10
not started.

## Prompt #10 - supervised predictive-model research (2026-09-30 -> 10-01)

**Asked:** can supervised models extract *stable out-of-sample* predictive
information from the Prompt #9 feature sets? Targets: mean-reversion
probability, residual reduction, future return, future volatility, future
absolute move (plus return-beyond-one-spread labels). Baselines first
(constant, logistic / linear, ridge, elastic net), then random forest, XGBoost,
LightGBM, CatBoost; chronological walk-forward with purge and embargo,
fold-specific preprocessing, calibration on separate rows, nulls, frozen specs,
then one final test on the reserved period (2022-). Outputs are probabilities
and expected values only - no BUY/SELL, thresholds, sizing, PnL, execution; no
neural networks; stop after the report (no Prompt #11).

**Status check (Step 1):** 1003 tests passing; dataset `ticks-2e173ef8e61bd240`
verified; Prompt #9 manifests present for 5m / 15m / 30m / 1h; target table
horizons 1, 2, 3, 5, 10, 20, 50; reserved period 2022-01-01 onward never loaded
with outcomes; ledger 388,231 rows.

**Built (2026-09-30):**
- `config/ml.yaml`, fixed before any result: targets and horizons, the five
  validation blocks (2011-12, 2013-14, 2015-17, 2018-19, 2020-21), purge h +
  embargo 12 bars, a purged inner slice (last 10 % of the training span) for
  early stopping and calibration only, model defaults, search grids, freeze
  rules (>= 4 of 5 blocks beating the constant, one-SE simplicity rule, window /
  weights / search / calibration changes only beyond one SE).
- `pip` extra `ml`: scikit-learn 1.9.1, xgboost 3.4.1, lightgbm 4.7.0,
  catboost 1.2.10, shap 0.52.0, joblib 1.6.0 (`requirements-lock.txt`, 57 pins).
- `src/xauusd_quant/ml/`: config, splits, datasets (development rows only;
  leakage gate), preprocessing (fold-fitted, order-checked), models (11
  families, native serialization), calibration, evaluation, training (units and
  variants), explainability (TreeSHAP, block permutation, ALE), diagnostics
  (PSI / KS / Wasserstein, robust-Mahalanobis OOD), registry (model ids, frozen
  specs, hashed artifacts), streaming, final_test.
- `research/ml_research.py` (12 cached stages of units), `ml_reports.py`
  (tables, pooled predictions, ML-H ledger rows), `ml_freeze.py` (freeze rules,
  final development models, streaming, latency), `ml_plots.py` (Step 97 figures).
- CLI: `ml-research`, `ml-train`, `ml-walk-forward`, `ml-compare`, `ml-report`,
  `ml-freeze`, `ml-finalize`, `ml-final-test` (frozen, hash-verified specs only;
  every access logged; a second look refused without a logged reason).
- Tests: 9 files, 102 tests (splits, preprocessing, training, calibration,
  serialization, leakage, final-test isolation, streaming, freeze rules).

**Fixed while building:** read-only NumPy views from Polars; LightGBM 4.7
`eval_set` and scikit-learn 1.9 `penalty` deprecations; the pooled baseline held
as dicts (~3.6 GB) -> dense float32; the freeze rules read an SE of exactly 0 as
"no SE", so a perfectly consistent difference counted as within one SE
(calibration) or not beyond it (window / search) -> `_better_by_one_se`,
pinned with dyadic test values; `Set-Content -Encoding utf8` (PowerShell 5.1)
wrote a BOM into `cli.py` / `ml/final_test.py` -> stripped.

**Verified by breaking (2026-09-30):** purge gap set to 0 (12 split tests
fail), the development loader opening reserved year files (2 isolation tests
fail), the old SE rule (the freeze test fails). Reverted; 102 pass; ruff and
mypy clean.

**Smoke (real 5m, block 2020-21, mean_reversion h5):** constant AUC 0.500;
logistic L2 AUC 0.551, log-loss skill +0.006; LightGBM AUC 0.602, +0.025 (707
trees, 36 s). Ridge on log volatility h5: rank IC 0.70, R^2 0.51. Development
rows 1,329,347 (< 2022-01-01), 80 features loaded in 5 s.

**5m plan (`xq ml-research -t 5m --dry-run`):** 1,250 units + the search
neighbourhood - grid 310, trees 160, feature_sets 120, variants 15, search 104,
windows 90, learning_curve 24, decay 6, retraining 208, ablation 90, nulls 78,
pipeline_null 45.

**Full suite (2026-09-30 21:04-21:26, alone):** 1105 passed (1003 + 102) in
21.7 min, peak 4,420 MB.

**Found by the end-to-end synthetic test (`test_ml_report.py`, written after the
suite had collected):**
- the unit cache never hit: a stamp read back from JSON (lists) never equalled a
  fresh one (tuples), so every rerun would have refitted every unit - a stopped
  5m run could not have resumed. `_stamp` now returns its JSON form;
- the OOD-decile table needed 500 rows per decile (fine for 5m blocks, empty for
  small ones) -> min(500, max(50, block / 20));
- the decile table of a constant model (all scores tied) averaged empty buckets;
- the training-window, control and ablation figures mixed log-loss skill and
  rank IC on one axis -> small multiples, one target and one metric per panel;
  controls are now read as real minus control on the same blocks
  (`null_comparison`), ablation as the change against the full Extended model
  (`research/model_ablation.py`).
Also added: `config/model_registry.yaml` (generated by `ml-finalize`), per-target
folders `<tf>/<target>/h<h>/...` (Step 99), `load_reserved_rows` reads only the
target columns that exist, tested on the synthetic stores. Real-data smoke of
the other tree families (5m, block 2011-12, mean_reversion h5): XGBoost AUC
0.594 / log-loss skill +0.021, CatBoost 0.596 / +0.022, random forest 0.581 /
+0.015 (30-40 s per unit with SHAP).

**5m run started 23:09** (`logs/ml10_5m.*`, all 12 stages, one child process).
Grid 309 units in 21 min, no failures; trees ~40 s per unit with explanations.

**While it runs (code outside the unit stamps):**
- `build_pipeline_null` smoke-tested on real 1h: built in 9-12 s on the same
  bars; mean_reversion h5 base rate 0.528 real vs 0.502 on the random walk. The
  interaction features were all-NaN in the null (interactions are computed after
  the families) -> computed like the live path now. The 5m Standard set has no
  interaction (the running 5m stage is unaffected); 15m has four.
- Reports: the constant is read raw (a Platt-calibrated constant is the inner
  slice's rate, another model); "beats the constant" and the paired comparison
  with it use the skill that defines the baseline (log-loss / MSE skill), above
  float noise; `ml/hyperparameters.py` holds the search trials (identical draws,
  pinned by `test_ml_hyperparameters.py`); notebooks renamed to the names asked
  for (45 baselines, 46 trees, 47 probability calibration, 48 model stability,
  49 feature importance, 50 final OOS evaluation).

**First read of the grid (development blocks 2011-2021, 5m, Standard set):**
mean-reversion logistic skill grows with the horizon (+0.002 at h1 ... +0.026 at
h20, 5/5 blocks; residual targets still to be read against the random-walk
pipeline); future volatility ridge rank IC 0.41 / 0.66 / 0.71 (h1 / h5 / h20);
future return rank IC +0.049 at h1 but MSE skill negative in 5/5 blocks - the
ranking carries a little information, the expected values are worse than the
training mean.

**Trees (5m, development):** mean_reversion h5 boosters log-loss skill +0.024
(AUC 0.60), random forest +0.016, logistic +0.008 (5/5 blocks each); h20
LightGBM +0.055 (AUC 0.65); residual_reduction h5 random forest rank IC 0.16,
ridge 0.11 - all still to be read against the random-walk pipeline.
direction_up_cost and direction_down_cost reach the same skill (h1 logistic
+0.049, AUC 0.62): post-hoc, their predictions correlate +0.87-0.93 (h1),
+0.69-0.81 (h5), +0.48-0.63 (h20) across blocks - the cost-buffered labels are
mostly "will the move exceed one spread" (magnitude vs spread), not direction.
future_return: trees shrink to the mean (MSE skill ~0), rank IC 0.036-0.046 at
h1.

**Feature sets (5m):** the Extended set lifts mean_reversion h5 LightGBM from
+0.024 to +0.143 log-loss skill (AUC 0.60 -> 0.74), logistic +0.008 -> +0.034;
volatility barely moves (0.689 -> 0.691). Extended carries `reg_resid_z_128` and
`ou_zscore_128/256` - z-scores of the very residual the reversion targets are
built from - so "|eps| shrinks" is partly mechanical (true of a rolling
regression on a random walk too). The pipeline null was planned on the Standard
set only -> extended (`pipeline_null_sets`): reference linear + LightGBM on
Extended for the three residual targets, run after the main 5m run
(`--stages pipeline_null`; Standard null units stay cached). Invariant 9: every
reversion number is reported beside it. The freeze rules stay as registered.
- Null builder, two more fixes found on real 1h: families ran in alphabetical
  order, so FFT ran before OU and the |eta| spectrum (15m+) was all-NaN on the
  null -> canonical order, as the live path; regime features stay NaN on the
  null by design (no regime null).

**The first 5m run stopped (2026-10-01 ~01:41):** grid (309 units, 21 min),
trees (157, 85 min), feature_sets (120, 28 min) and variants (15, 3 min)
finished; the search stage reached 25 of 104 units. At 01:39 one unit
(mean_reversion h5 LightGBM search t10, block 2020-21) failed with an empty
`MemoryError` (a C-level allocation under system memory pressure; the run went
on), and the process tree was gone by 01:46 - no END line, no survivor. Nothing
completed was lost (every unit is written when it finishes).
`logs/ml10_5m.err` had grown to 153 MB of one scikit-learn 1.9 / joblib 1.6
`UserWarning` ("`sklearn.utils.parallel.delayed` should be used with
`sklearn.utils.parallel.Parallel`"), emitted from the random forest's worker
threads - harmless; now filtered in the ML child process (`cli.py`, outside
the unit stamps).

**Session of 2026-10-01 ("continue prompt#10"):** 107 Prompt #10 tests pass
(46 s). `--dry-run`: every finished unit still current (grid 310, trees 160,
feature_sets 120, variants 15, search 26 of 104). Resumed 01:54 with the search
stage alone (`logs/ml10_5m_r2_search.*`, memory of the process tree every 30 s
in `.mem.csv`: ~1.9 GB working set, ~3.0 GB private); the failed unit re-ran in
38 s. Remaining after it: windows 90, learning_curve 24, decay 6, retraining
208, ablation 90, nulls 78, pipeline_null 75 (Standard + Extended).
- Search done 03:11 (78 units + 28 neighbourhood units, 77.6 min, peak 1.95 GB,
  0 failures). Flat plateau: mean_reversion h5 trials 0.0198-0.0245 log-loss
  skill (defaults ~0.0245), future_volatility h5 0.687-0.700 rank IC; num_leaves
  plateaus from 15 (reversion) / 7 (volatility), 3 leaves is clearly worse.
- 15m run started 02:04 in parallel (`logs/ml10_15m.*`, all 12 stages; 15m
  Standard = Extended = 72 features, so its reversion rows need the pipeline
  null as much as 5m Extended's). Combined private memory ~5.8 GB, free commit
  ~4.2-4.9 GB, stable. Its grid is slow (saga L1 / elastic-net logistic 60-110 s
  each on 110 design columns).
- Remaining 5m stages started 03:12 (`logs/ml10_5m_r3_rest.*`).
- **Freeze fix (before anything was frozen):** `_search_choice` compared the
  search trials (run on Standard) with the default-parameter units of *every*
  feature set of the family, so on the search blocks the per-block baseline was
  whichever set's row came last (Standard or Extended). Now trials, neighbours
  and defaults share the searched set, and a model frozen on another set keeps
  the defaults (the search says nothing about that set).
  `test_search_trials_are_read_against_defaults_on_the_searched_set_only`
  fails with the old filter (re-broken and seen to fail), passes with the fix.
- 5m windows (90 units, 14 min), learning_curve (24), decay (6) and retraining
  (208, 56 min) done by 04:26.
- **Ablation memory (stopped and fixed, 04:30):** a design matrix is cached per
  key until the stage ends, and an ablation subset is not a named manifest, so
  each of its 15 subsets (40-72 columns x 1.33M rows) got its own cache entry -
  ~4.8 GB by the end of the first target, beside the 15m run. The private
  memory had reached 3.8 GB two units in; the 5m tree was ended from its root
  (`taskkill /T`, 3 ablation units kept, none lost). `run_units` now drops the
  cached designs after any unit whose feature list is not a named manifest
  (rebuilding one costs < 1 s). Restarted 04:31 with `--stages ablation,nulls`
  (`logs/ml10_5m_r4_ablation_nulls.*`): flat at ~3.4 GB private; done 05:50
  (78.7 min, 0 failures).
- **Base-rate recalibration (found reading the 5m controls):** a shifted-label
  LightGBM on `direction_up_cost` h5 scored +0.016 / -0.134 log-loss skill at
  AUC 0.50 - calibration on the inner slice moves a model to the *recent* base
  rate, and the one-spread direction labels' base rate drifts with spreads and
  volatility. The constant itself, Platt-calibrated, earns +0.018 (h1), +0.020-
  0.023 (h5), +0.009-0.014 (h20) on the direction labels, ~0 on the reversion
  labels. Skill against the training rate stays the registered baseline; added
  beside it (reporting only): `skill_beyond_base_rate` (model minus the
  recalibrated constant, block by block) in the report, and in the final test
  the skill against the recent base rate (the final model's inner-slice mean,
  development rows, computed before any reserved row is read;
  `final_test.recent_base`, tested).
- **Post-hoc pipeline null (decided 05:55, before any pipeline-null result):**
  the registered random walk is homoscedastic, so real minus random walk can hold
  volatility dynamics (a falling volatility shrinks |eps| too). Added
  `nulls.pipeline_posthoc: [sign_flip]` - the Prompt #8 sign-flip null, the real
  bars in real order with random signs, a martingale with the real volatility
  path - as stage `pipeline_null_posthoc`, labelled POST-HOC in the config, the
  ledger rows and the figures; a harder control, never used to select or freeze.
  The pipeline stages now skip the null build when all of a null's units are
  current (`test_ml_pipeline_null.py`, re-broken and seen to fail).
- **Scheduling:** 15m ablation + nulls moved to a second process at 05:50
  (`logs/ml10_15m_b_ablation_nulls.*`) while the first reached them only ~2 h
  later; the first 15m process was ended at 05:55 in its search stage (50 of 104
  search units done, none lost) and restarted with `--stages search,windows,
  learning_curve,decay,retraining` (`logs/ml10_15m_a2.*`), so that no two
  processes build a pipeline null at once. (One launch at 05:56 was detached
  from the session by mistake; ended after 10 s and relaunched normally.)
- 15m runs: ablation + nulls 05:50-06:27 (36.8 min), pipeline nulls (random walk
  + sign flip) 06:27-06:40 (13 min; the null build 50 s, +1.6 GB), search
  remainder / windows / learning curve / decay 06:41-07:08, retraining (split off
  at 06:41) -07:27. 0 failures anywhere.
- 15m report 07:30-07:36 (1,331 units, 338 configurations -> 338 ML-H ledger
  rows, 15 figures, 1.1 GB). **15m freeze 07:36**: 17 specs (CAT reversion h5,
  LGBM rev-half h5, RF residual-reduction h5, LOGL2 direction-up h5 on the target
  set, LGBM volatility h5 on the target set, RF abs-move h5, each with its
  constant and, where different, its reference linear model); future_return h1:
  none (no model beats the constant in >= 4 of 5 blocks). **15m finalize** 07:37-
  07:48: every model reloads identically, streaming = batch (max difference
  <= 1e-10), latency 0.14-1.0 ms/row (random forests 19-26 ms/row).
- 5m pipeline nulls 07:27-08:03 alone (35.9 min; null builds peaked at 6.8 GB
  private, free commit never below 2.6 GB). Every 5m and 15m unit current.
- Leakage audit (development rows, 200k sampled - an estimate): every manifest
  feature registered and live-safe; max |Spearman| of any feature with any target
  0.748 at 5m (`log_parkinson_20` vs future volatility h20), 0.549 at 15m; none
  above 0.95.
- 5m report 08:04-08:16 (1,383 units, 346 configurations -> ML-H rows, 16
  figures). **5m freeze 08:17**: 20 specs (LGBM reversion h5 on Extended, LGBM
  rev-half h5, LGBM residual-reduction h5 on Extended, XGB return h1, LOGL2
  direction-up h5 on Extended, LGBM volatility h5 on the target set, RF abs-move
  h5; constants and references beside). **5m finalize** 08:17-08:28: all 20
  reload identically, streaming = batch, 0.15-0.55 ms/row (RF 27.7 ms/row).
- 15m ablation table was empty: Extended = Standard at 15m, so no unit is named
  "extended" there -> `ablation_table` reads the default set when Extended has no
  units of its own (`reference_set` column; `test_ml_ablation.py`, re-broken and
  seen to fail); 15m report rerun 08:18-08:25.
- Before the one look: an end-to-end test of `run_final_test` on the synthetic
  stores (`test_the_final_test_end_to_end_on_synthetic_stores`: artifacts, reserved
  rows only, labels aligned with their bars, recent base rate, audit trail,
  access log, refusal of a second look). A one-bar label shift in the reserved
  loader drops its rank IC from > 0.3 to 0.11 (re-broken, seen to fail, restored).
- **One-time final test 08:30-08:32** on all 37 frozen specs (5m 20, 15m 17;
  `results/ml_research/<tf>/final_test/`, access logs: 1 start + 20 / 17
  evaluations; nothing tuned afterwards). Reserved period 2022-01-03 ->
  2026-09-18 (5m 334,254 labelled rows, 15m 111,421). Development -> final:
  - volatility h5 LightGBM (target set): 5m rank IC 0.688 -> 0.741 (ridge
    0.658 -> 0.716), 15m 0.647 -> 0.702 (ridge 0.678); per year 0.62-0.71
    (5m), weakest in 2026 (0.62; 15m 0.53);
  - abs move h5 random forest: 5m 0.392 -> 0.433 (ridge 0.417), 15m 0.372 ->
    0.411 (ridge 0.402);
  - mean reversion h5: 5m LightGBM/Extended +0.143 -> +0.136 log-loss skill (AUC
    0.736), 15m CatBoost +0.132 -> +0.134 - held, but mostly mechanical (beyond
    the sign-flip null +0.002-0.008 in development);
  - residual reduction h5: 5m 0.459 -> 0.465, 15m 0.427 -> 0.437 (mechanical);
  - reversion c = 0.5: 5m +0.005 -> +0.007, 15m +0.064 -> +0.061;
  - direction up h5 logistic: 5m +0.035 -> +0.039 but only +0.007 beyond the
    recent base rate (AUC 0.552, falling to 0.527 in the last 12 months); 15m
    +0.016 -> +0.019, +0.000 beyond it;
  - future return h1 XGBoost: rank IC 0.046 -> 0.021, MSE skill +0.0001 ->
    -0.0006 (ridge 0.049 -> 0.022) - the one target that clearly decayed.
- ruff and mypy clean; full suite started 08:33 (alone). The ledger fix below
  came after the suite collected, so it reruns at the end.
- **Ledger duplicate (fixed):** the ledger's upsert key includes `window`, where
  ML-H rows keep the block count; the first 15m report (before retraining
  finished) wrote `refit` rows with fewer blocks, and the later report added a
  second row for the same id (ML-H-000246). `ResearchLedger.upsert(replace_by=)`
  - the ML layer now replaces by `hypothesis_id` (each id is one
  configuration); other layers keep the full key (`test_research_ledger.py`).
- Full suite 08:33-08:55 (alone): **1,118 passed** (21.6 min, peak 4.4 GB) -
  before the ledger fix and the figure fixes below, so it is rerun at the end.
- Step 59, model vs the best single feature (2011-2021 pooled, 300k sampled
  rows - an estimate): volatility h5 0.73 vs 0.63 (`log_parkinson_20`) at 5m,
  0.69 vs 0.50 at 15m; abs move 0.42 vs 0.37 / 0.40 vs 0.29; future return
  0.048 vs -0.045 (`ret_2`, short-term reversal) - the model adds nothing over
  one feature; reversion 0.42 vs 0.15 (the |z| nonlinearity, mechanical).
- **Two figures were misleading (fixed, reports rerun):** the training-window
  panel's "expanding" bar was the first base row of the family - any feature
  set, ablation subsets included (5m reversion LightGBM showed 0.135, Standard's
  is 0.024) -> the `windows` table now holds the default set only (the window
  and weight variants exist only there); the hyperparameter figure put log-loss
  skill (~0.02) and rank IC (~0.70) on one axis and labelled the neighbourhood
  ticks for the first target only -> one row per searched target, neighbours
  as points with their SE beside the best trial. The empty retraining panel of
  future_return says "not run for this pair".
- Reports rerun 09:04-09:17 (5m 13.4 min, 15m 5.3 min); ledger 346 + 338 ML-H
  rows, one per configuration; per-target `final_test/` copies present.
- **Final full suite 09:18-09:39 (alone): 1,120 passed** (21.1 min, peak 4.3 GB);
  ruff and mypy clean.
- **Steps 61 / 74 gap closed afterwards:** the development out-of-sample
  predictions lived only per unit (rows, with the spec in a sidecar JSON). Added
  `export_oos_predictions` (in the heavy report): `predictions/<target>_h<h>
  .parquet` per comparison pair - timestamp, row, fold, target, horizon, dataset
  version, label, label_raw, one column per `<family>|<set>` (+ `|platt`) - with
  a sidecar JSON of model ids, feature-set ids and versions. Tested against the
  unit files (`test_ml_report.py`, 1 passed); ruff, mypy clean; reports rerun
  09:41-09:56 to write them (5m 7 tables ~300 MB, e.g. mean_reversion h5
  782,390 rows x 12 models; 15m ~100 MB). Ledger still 346 + 338 ML-H rows.
- **Final report delivered (2026-10-01 ~10:00); Prompt #11 not started.**

## Prompt #11 - ensembles, meta-models and predictive diversification (2026-10-01)

**Asked:** does combining genuinely distinct models give predictions that are
more stable, better calibrated, less fragile and more informative out of sample
than one model? Eligibility gate; aligned out-of-sample inputs only; prediction /
error diversity; disagreement; simple / median / performance- / diversity-
weighted ensembles; stacking (simple meta-models, chronological, purged);
limited context; regime-, volatility-conditioned and trailing weights; ensemble
calibration; size, subsets, families, feature sets, cross-horizon; stability by
year / quarter / regime / volatility / spread / session; failure and common-mode
failure; leave-one-out; null, noise and duplicate controls; frozen ENSEMBLE_SPECs;
inference interface with version / feature checks and fail-closed behaviour;
streaming equality; benchmark; one-time final test; the standardized prediction
contract for Prompt #12. Still prediction research (no BUY / SELL, thresholds,
sizing, PnL, execution, Exness, MT5). Also asked: add `AGENTS.md` and work with
Codex.

**Step 1 status:** full suite 1,120 passed (25.3 min, `logs/suite11a.*`);
data versions current (ticks-2e173ef8e61bd240; 5m factory-5m-0541c6a9 /
targets-5m-4e259797; 15m factory-15m-2ce812a5 / targets-15m-64285f53); Prompt #9
manifests V001 hash-verified; Prompt #10 prediction tables for 7 pairs x 2
timeframes on exactly those versions. Flagged to you: the reserved period was
already evaluated once (Prompt #10, 08:30), so an ensemble final test is a second
look - planned as one logged, labelled evaluation of rules fixed on development
data.

**Built:**
- `AGENTS.md` (rules for Codex and other agents; CLAUDE.md stays authoritative);
  `scripts/run_logged.ps1` (any command with START / END and process-tree memory
  logging - the earlier runner lived in a temporary scratchpad).
- `config/ensemble.yaml` (pairs, gate thresholds, meta walk-forward blocks
  2013-2021, methods and their few parameters, freeze rule, streaming), fixed
  before any ensemble result.
- `src/xauusd_quant/ensemble/`: config, data (load + verify + coverage; history
  = purged earlier blocks; refuses any row at or after the reserved start),
  averaging, weighting (performance, diversity with duplicate clusters,
  conditional, trailing daily weights that wait for labels to resolve, entropy,
  turnover), meta_model + stacking (logistic / ridge on out-of-sample predictions
  only), calibration (A / B / C orders), diversity (prediction, error,
  calibration-residual correlation, Q, covariance), disagreement, stability,
  diagnostics (OOD per fold, failure, common mode, alpha decay), registry
  (ENS ids, constituent ids, immutable ENSEMBLE_SPEC), inference
  (`EnsembleModel.predict / predict_proba`, `ENSEMBLE_INVALID`,
  `ModelVersionMismatchError`, `FeatureVersionMismatchError`), streaming,
  final_test (the only ensemble code reading 2022-; second-look note in every
  output), contract (prediction record schema; refuses decision-like keys).
- `research/ensemble_research.py` (gate, universe, `PairRun` meta walk-forward,
  controls incl. pipeline-null labels, disagreement / OOD / failure views),
  `ensemble_ablation.py` (size, greedy, families, feature sets, weaker models,
  leave-out, contribution, cross-horizon), `ensemble_reports.py` (series summary,
  freeze rule, calibration rule, ENS-H ledger rows, timeframe tables, joint
  predictive state), `ensemble_freeze.py` (freeze, finalize, registry yaml),
  `ensemble_plots.py` (13 figures; drafted by Codex).
- CLI: `ensemble-research`, `ensemble-report`, `ensemble-build`,
  `ensemble-freeze`, `ensemble-finalize`, `ensemble-final-test`.
- Tests: 10 files (139 tests at the end, with the review-2 and slim-loader
  additions; `ensemble_synth.py` helper; alignment, contract, diversity /
  weighting drafted by Codex and reviewed). Re-broken and seen to
  fail: no purge in `history()` (6 fail), unresolved labels in trailing weights
  (3 fail), a selection module importing the reserved loader (1 fails).
- Notebooks 51-56 (no calculations; generated by `logs/codex/make_notebooks.py`).

**Codex:** CLI 0.159.3 is installed and logged in, but its read-only sandbox
rejects every shell command on this Windows setup, so prompts inline the source
files (stdin) and Codex returns code blocks that are reviewed, linted and run
here (`logs/codex/`). A leakage review of the new code found four real issues,
all fixed before any reported result:
1. the "best individual" and the size orders ranked on *whole* earlier blocks -
   the last h + embargo rows of block k-1 resolve inside block k
   (`PairRun.prior_scores` now uses the purged history);
2. trailing weights took the first *covered* row of a day as the update bar
   (now the day's first bar on the full timeline);
3. the loaders did not refuse reserved rows themselves (now `reserved_start` /
   `development_rows` guards);
4. the noise control's scale came from all blocks (now from each block's
   history); plus smaller ones (exact-duplicate columns, "untested" instead of
   pass / fail for undefined statistics, a walk-forward-universe robustness
   variant because the gate reads all five blocks).

**Decision after the first development results (before any freeze or final
test):** the first freeze code froze the simple average as a pair's output even
when the rule had *rejected* it (single model retained). Now the retained
predictor is frozen - the single model as a one-member spec - with the simple
average evaluated beside it. Packaging only: the rule's decisions are unchanged;
marked in `config/ensemble.yaml`. The provisional run (16:20-16:43, stopped
during 5m abs move) and earlier smoke outputs were deleted
(`results/ensemble_research/{5m,15m,plots}`, 337 MB, regenerated by the final
run).

**Runs:** final research run 16:44-17:16 (`logs/ens11_research_final.*`, `--all
--report`; 5m 23 min, 15m 7.4 min, peak 3.8 GB private, exit 0): 14 pairs, 565
`ENS-H` ledger rows (5m 290, 15m 275; ledger 389,480 rows), 151 figures, joint
predictive state 5m 637,400 rows (every field ok), 15m 212,799 rows (future return:
no eligible ensemble).

**Development findings (meta walk-forward, blocks 2013-2021):**
- Gate: 5m - every model eligible for reversion, revhalf, direction, volatility,
  abs move; residual reduction 6 null_failed (5 eligible incl. target-set models
  whose null is untested); future return 9 weak (2 eligible). 15m - future return
  0 eligible (no ensemble); residual reduction 7 null_failed; reversion 1
  unstable.
- Diversity: prediction Spearman 0.16-0.99, but error correlation 0.90-1.00
  (medians 0.94-0.999) - little independent error to average away.
- Freeze rule (development only): ensembles retained for 5m volatility (stacking,
  +0.0018 rank IC over the best model, 4/4 blocks), 5m abs move (stacking), 5m
  revhalf (stacking, +0.0002), 5m residual reduction (regime-conditioned, +0.003;
  universe includes target-set models with untested nulls), 15m residual reduction
  (stacking, minimal / target sets, untested nulls); the single model retained for
  the other 8 pairs (5m reversion LightGBM/Extended +0.142 vs simple average
  +0.053 - weaker models dilute; averaged probabilities are under-confident, ECE
  0.065-0.078).
- Controls: noise model gets 0 raw weight everywhere (stacking share 0.0003-0.06);
  a duplicated model gets double weight from naive performance weights and exactly
  its single weight from the diversity-aware weights; shifted-label null ensembles
  ~0 (volatility 0.13-0.15 rank IC from the daily cycle vs 0.64-0.68 real);
  reversion beside the sign-flip null +0.006; residual-reduction ensemble of the
  null-tested models below both pipeline nulls.
- Disagreement / OOD vs loss: rank correlation within +-0.04 (OOD +-0.09) except
  reversion, where it is negative (-0.21 / -0.32: disagreement marks bars where the
  strong models are confidently right). Common-mode failure 0.4-9.9 % of rows vs
  10^-8 if independent and 10 % if identical.
- Size: the best average uses 1-4 models in 12 of 13 pairs; weights stable (no
  scheme flagged), near-equal; trailing-weight turnover 0.0003-0.04 per day;
  cross-horizon inputs add +0.0003-0.001 (reversion LightGBM) or hurt (return).

**Codex review 2 (final test, freeze, inference), all fixed before any freeze /
final-test run:** finalize's streaming loaded the whole bar history (now
development bars only: `ensemble.streaming.load_development_bars`, cut in the
scan - a copy on purpose, `features/factory.py` is in the feature-matrix build
key); an interrupted look (`final_test_started` without `evaluated`) now counts
as a look; the live config must match the frozen one (target definition, ml and
ensemble fingerprints); invalid rows of the trailing-weight history no longer get
made-up losses (`dynamic_weights(valid=)`); `log_rv_20` requested explicitly for
the volatility buckets; one shared row mask for every compared series (invalid
ensemble rows excluded and counted); a missing best individual refused; the
second-look note in the audit parquet; artifact manifests must hash every
prediction-bearing file and are bound by their sha256 at finalize. The research's
pipeline-null labels now come from development bars too - checked identical
(shrink labels exactly; residual reduction within float32 rounding on 39 of
1.33M 5m rows), so the stored results stand. Tests: 6 more; the interrupted-look
guard re-broken and seen to fail.

**Freeze 21:1x:** 13 ENSEMBLE_SPECs (5m 7, 15m 6) + 82 constituent MODEL_SPECs.
**Finalize 5m** 21:15-21:36 (`logs/ens11_finalize_5m.*`): 27 constituents fitted
and 5 ensembles verified (reload equal, streaming = batch; 1.4-37 ms per row), then
a `MemoryError` (1.27 GiB for the 142-column float64 design of the Extended-set
logistic model for direction; peak 5.6 GB private, ~2.5 GB commit free). Nothing
lost (artifacts are saved per constituent and reused). Fix: drop the cached
designs and collect before each fit and after each ensemble. Rerun 21:39-21:54
(`logs/ens11_finalize_5m_r2.*`, 15.1 min, exit 0): all 41 constituents saved,
every 5m ensemble reloads identically, recombines exactly and streams equal to
batch (40 bars each); 1.2-1.5 ms per row without random-forest constituents,
31-36 ms with them (a single-model spec still loads the universe for its
companions - live, only the selected model is needed); 33-542 ms per 10k rows,
artifacts 77 KB - 10 MB, load 0.04-0.36 s.
**Finalize 15m** 21:55-22:14 (`logs/ens11_finalize_15m.*`, 18.6 min, peak 2.9 GB
private, exit 0): all 41 constituents trained and reloaded equal; all 6 ensembles
reload identically, recombine exactly and stream equal to batch; 1.8 ms per row
(the stacked residual-reduction ensemble, LightGBM / ridge members) to 34-37 ms
(with a random forest); 101-507 ms per 10k rows; artifacts 0.7-6.5 MB.

**One-time final test (a second look at 2022-, logged):**
- 22:16, first attempt with all 13 specs in one process (`logs/ens11_final_test.*`):
  `final_test_started` was logged for 5m, then the first ensemble (5m abs move)
  stopped with a `MemoryError` while its constituents predicted the reserved rows
  (191 MiB for a 334,259 x 75 float64 array; system commit ~0.4 GB free - another
  application held 4 GB). No metric was computed, written or seen; the 15m log was
  not touched. By the interrupted-look rule this counts as a look, so the rerun
  needed `--repeat-reason` (logged as `repeat_authorised`, with the reason).
- Fixed before the rerun (final-test path and CLI only, no cache-stamped module):
  `slim_development_data` reads only what the final test needs from development
  (spread, `log_rv_20`, the specs' targets purged at the reserved start, the
  regression residual and trailing volatility) instead of the whole ML data; one
  timeframe per process; collection after each ensemble.
- 5m 22:22-22:27 (`logs/ens11_final_test_5m.*`, 4.6 min, peak 1.75 GB) and 15m
  22:28-22:30 (`logs/ens11_final_test_15m.*`, 2.0 min, peak 1.46 GB), exit 0.
  Access logs: 5m started (interrupted) -> repeat_authorised -> started -> 7
  evaluated; 15m started -> 6 evaluated. 2022-01-03 -> 2026-09-18, 334,254
  labelled rows at 5m (334,258 at h1) and 111,421 at 15m, 0 invalid.

| Spec | Frozen method | Metric | Ensemble | Simple avg | Best indiv. | Dev. | Worst year (ens / avg) |
|---|---|---|---|---|---|---|---|
| VOL 5m h5 | stacking | rank IC | 0.7450 | 0.7412 | 0.7411 | 0.6853 | 2026: 0.626 / 0.622 |
| ABSMOVE 5m h5 | stacking | rank IC | 0.4419 | 0.4395 | 0.4329 | 0.3956 | 2026: 0.342 / 0.337 |
| RESIDRED 5m h5 | regime-conditioned | rank IC | 0.0927 | 0.0916 | 0.0887 | 0.1142 | 0.091 / 0.087 |
| REVHALF 5m h5 | stacking | log-loss skill | 0.0073 | 0.0071 | 0.0068 | 0.0055 | 2025: 0.0049 / 0.0049 |
| REVERSION 5m h5 | single LightGBM / Extended (its own Platt calibration) | log-loss skill | 0.1371 | 0.0500 | = | 0.1423 | 2025: 0.127 / 0.048 |
| UPCOST 5m h5 | single logistic / Extended | log-loss skill | 0.0391 | 0.0382 | = | 0.0294 | 2022: 0.024 / 0.026 |
| RETURN 5m h1 | single XGBoost | rank IC | 0.0207 | 0.0200 | = | 0.0469 | 2025: 0.012 / 0.015 |
| VOL 15m h5 | single LightGBM / target | rank IC | 0.7023 | 0.7087 | = | 0.6432 | 2026: 0.526 / 0.543 |
| ABSMOVE 15m h5 | single random forest | rank IC | 0.4113 | 0.4216 | = | 0.3741 | 2026: 0.274 / 0.292 |
| RESIDRED 15m h5 | stacking | rank IC | 0.2686 | 0.2427 | 0.2627 | 0.2510 | 2026: 0.249 / 0.220 |
| REVERSION 15m h5 | single CatBoost | log-loss skill | 0.1345 | 0.1062 | = | 0.1331 | 2025: 0.122 / 0.100 |
| REVHALF 15m h5 | single LightGBM / Standard | log-loss skill | 0.0605 | 0.0592 | = | 0.0679 | 2025: 0.047 / 0.050 |
| UPCOST 15m h5 | single random forest | log-loss skill | 0.0199 | 0.0210 | = | 0.0157 | 2022: 0.010 / 0.010 |

"Best indiv." is Prompt #10's frozen model for the pair (Step 62's frozen best
individual candidate); "=" means the frozen spec is that model. Reading:
- Where the rule kept an ensemble, it beat both its simple average and the best
  individual on the reserved period, by the same small margins as in development
  (+0.0005 to +0.009 over the best individual), and its worst year was no worse.
- Where it kept one model, the simple average was far worse for reversion
  (diluted, under-confident: ECE 0.074 / 0.063 vs 0.013 / 0.014), within 0.001
  for 5m direction / return and 15m reversion c = 0.5, and *better* at 15m for
  volatility (+0.006), abs move (+0.010) and direction (+0.001). The 15m volatility
  and abs-move singles are Prompt #10's frozen choices (LightGBM target set,
  random forest), which ranked below CatBoost and LightGBM / Standard on 2022-
  (0.714 / 0.712 and 0.425 / 0.423). In development they differed by ~0.01 or
  less. Reported; nothing retuned.
- Direction beyond the recent base rate: 5m +0.0071, 15m +0.0008 (nothing).
  Future return h1 decayed as in Prompt #10 (0.047 -> 0.021). 2026 is the weakest
  year for every volatility-type target. Reversion (both thresholds) and residual
  reduction stay mostly mechanical (sign-flip null, untested target-set nulls).

**Verification at the end (22:36-23:17):**
- New test for the final test's light loader
  (`test_the_slim_development_loader_reads_development_rows_only`: development
  rows only with the reserved target files corrupted, every outcome window
  crossing the start purged, every target's labels derivable). Re-broken twice
  and seen to fail (purge removed; reserved files opened), restored byte for
  byte.
- Full suite in three processes (free commit 0.5-3 GB at times; another
  application held 4 GB): A unit tests `test_[a-l]*` **481 passed** (1.5 min,
  1.1 GB); B unit tests `test_[m-z]*` **759 passed** (10.4 min, 1.3 GB); C
  realdata **18 passed, 1 failed** (6.0 min, 1.6 GB). The failure was
  `test_realdata.py::test_dataset_is_sorted_and_free_of_duplicate_timestamps`:
  DuckDB "Out of Memory Error: Allocation failure" on `count(DISTINCT timestamp)`
  over 729M rows with ~2.2 GB commit free. Rerun alone it **passed** (9.4 min,
  peak 4.3 GB; `logs/suite11_final2_C_rerun.*`). The rerun (23:07-23:17) was
  started by a peer Claude session in this repository (`xauusd-81`, opened
  23:04); we coordinated by message and it left the docs and runs here. **Total 1,259 passed, 0 failed** (1,120
  before Prompt #11 + 139 new). ruff and mypy clean.
- The first split attempt (22:36) never finished part B: the temporary runner
  called `pytest.main` without a `__main__` guard, so a test's spawned child
  re-imported it and reran the whole part. Stopped (no process survived), fixed,
  and every part rerun (`logs/suite11_final2_*`; the stopped attempt is
  `logs/suite11_final_*`). CLAUDE.md notes the guard.
- Notebooks 51-56: every code cell executed against the real outputs (Jupyter is
  not installed, so with `IPython.display` stubbed): 0 failures, every figure they
  show exists.
- **Final report delivered (2026-10-01 ~23:30); Prompt #12 not started.**

## Git and GitHub (2026-10-02)

**Asked:** "commit and push to github".

**Done:**
- Git was not installed on the PC (the repository was `git init`ed on the laptop,
  2026-09-20, never committed). Installed with winget, by your choice: Git for
  Windows 2.55.0.windows.5 (`C:\Program Files\Git`) and GitHub CLI 2.102.0
  (`C:\Program Files\GitHub CLI`). Processes started before the install (this
  Claude Code session among them) do not have them on PATH; new shells do.
- `gh auth login` (device flow, account `FatCat1244`, HTTPS, token in the Windows
  keyring), `gh auth setup-git`; private repository
  https://github.com/FatCat1244/xauusd-quant created with `origin` pointing at it.
- Repository-local config: branch renamed `master` -> `main` (no commits existed);
  `user.name Nontapat`, `user.email aa5205238@gmail.com`; **`core.autocrlf
  false`**. The installer sets `autocrlf true` system-wide, and the working tree
  mixes line endings (131 CRLF files, 254 LF). Every cache stamp hashes source
  files as raw bytes (`research/study_io.code_fingerprint`), so a checkout that
  rewrote line endings would make every cached unit and stage stale. With
  `false`, git stores and restores the bytes exactly.
- First commit: 393 files, 4.4 MB - `src/`, `tests/`, `config/` (including
  `model_registry.yaml` / `ensemble_registry.yaml`), `scripts/`, `research/`
  notebooks 01-56, README, CLAUDE.md, AGENTS.md, WORKLOG.md, `pyproject.toml`,
  `requirements-lock.txt`, `.env.example` and the `.gitkeep` placeholders.
  Excluded by `.gitignore`: `data/` (54 GB, raw CSV included), `results/`
  (11.8 GB), `logs/`, `.venv/` and tool caches. Before committing, the files were
  scanned for credentials (none) and sizes (largest `cli.py`, 179 KB).
- Research artefacts record the commit (`study_io.git_info` and the older
  `_git_commit` helpers run `git rev-parse HEAD`) when they are produced from a
  shell that has git on PATH. Earlier artefacts carry a null commit.

## Open items

Stage #15: no evidence-supported alpha portfolio. Matching prior-only pipeline
nulls, economic/execution evidence, complete policy clocks, adequate coverage and
fold-local eligibility remain prerequisites. Corrected Stage #13 fits are present;
unknown legacy selection chronology and previously inspected periods remain.
Stage #16 offline risk software is authorized; actual settings and scientific
eligibility remain missing. Broker/shadow/demo work requires separate authorization.

Stage #16: supply explicit account denomination/currency/conversion, instrument
contract/tick/quantity/margin/cost/financing terms and the complete risk policy
listed by `risk-readiness`. Synthetic values are not production recommendations.
Async transport, live reconciliation, persistence/operator procedures, broker
margin and feed/shadow validation remain later-stage requirements. No live safety
or profitability follows from passing offline tests.

Stage #14: no candidate currently eligible for Stage #15. Required matching
prior-only random-walk/sign-flip evidence, audited provenance and verified supplied
execution terms remain missing. Legacy feature/count/universe chronology remains
unresolved. Saved June 1 development samples cannot support the five-day evidence
minimum. A larger study needs a new frozen plan; do not expand searches to get a
passing result. 2022+ is previously inspected, and no uninspected period is proved.

| # | Item | How |
|---|---|---|
| 1 | 1m spectral (#5), wavelet (#6) and regime (#7) studies | Heavy (7.9M bars); run alone with the browser closed; wavelet and regimes also need the 1m log-return FFT features first |
| 2 | Post-hoc wavelet check over all windows (~2 h) | `python scripts/wavelet_baseline_robustness.py --resume`, alone |
| 3 | Prompt #2 and #3 results on the full dataset | `xq research-summary --all`, `xq regression-research --all` (currently only the archived partial sample) |
| 4 | 28 unexplained tick gaps in 2014 | Flagged 2026-09-24, not inspected yet |
| 5 | Prompt #10 follow-ups (not run, optional) | 30m / 1h supervised runs (`timeframes` in `ml.yaml` holds 5m, 15m); a sign-flip pipeline null registered from the start; windows / weights tested on the target sets the freeze picked (they ran on Standard only, so the volatility models stayed expanding) |
| 8 | Plateau rule for the general Standard set | The registered mean-relative rule is driven by weak direction targets (20 / 72 / 50 / 30 at 5m / 15m / 30m / 1h); a per-kind or IC-weighted rule would need registering *before* it is used |
| 6 | Regime follow-ups (not run, optional) | the configured `volatility_only` feature set; a non-monotone redundancy screen for new inputs; the 30m no-R^2 HMM persistence above the block bootstrap (era split); 29 orphaned models HMM_5M_K6_00001-00029 in the registry (harmless, unreferenced) |
| 9 | Prompt #11 follow-ups (not run, optional) | 30m / 1h ensembles; a pipeline null for the target / minimal feature sets (the residual-reduction ensembles built on them are untested against nulls); a lighter live path for single-model specs (a one-member spec still loads its companions: 31-37 ms per row with a random forest among them, ~1.5 ms without) |
| 10 | The reserved period 2022- has been looked at twice (Prompt #10, Prompt #11) | Any new research direction needs a genuinely untouched evaluation period (Prompt #11 Step 63), e.g. data after 2026-09-18 |

## Lessons (kept in CLAUDE.md where they affect code)

- Verify by trying to break things: the digest, the NaN verdicts and the
  zero-IQR effect were all caught or confirmed by re-breaking a fix.
- Every finding so far that looked like structure was either a pipeline
  artefact (fixed in bars, reproduced by a null) or volatility. The regime
  layer added a third kind: feature geometry (a near-functional, non-monotone
  link between two inputs) that a mixture turns into "states".
- A likelihood rule rewards finer discretisation of anything continuous; only
  the nulls and a continuum control can tell regimes from bands.
- Missing is not failing: an unavailable statistic must make a verdict
  "untested", never pass or fail it.
- This 16 GB machine: heavy runs one at a time, memory logged, results written
  part by part; a Claude Code memory "stop" can leave the whole Windows process
  tree running - check and end it from the root. The stops come when the
  session is idle: stay active (poll) while a background run is going.
- Ensembles cannot add what the models do not disagree about: with error
  correlations of 0.90-1.00, averaging only dilutes the strongest model.
  Read error correlations first; the simplest model that holds out of sample
  is often the answer.
- A one-time evaluation must be light enough to finish: an interrupted look
  still counts, so the reserved-period run should load only what it scores.

---

## How this log is kept

Appended at the end of every stage, and whenever something is built, run,
fixed or decided. Keep entries factual: what was asked, what was done, what
came out, what is open. Dates local (UTC+7).


## Prompt #12 - offline execution and economics (2026-10-03)

Explicit authorization: build and test an offline execution/backtesting layer,
not broker access, MT5, demo/live trading, deployment or Stage #13. Initial
working tree was clean. No research process was running when stamped source
was repaired. No commit or push. Existing raw data, historical results,
manifests, frozen specifications and fitted payloads were preserved. Existing
source line endings were preserved; files are UTF-8 without BOM.

Implemented the `execution` package, four CLI commands, declared hypothetical
execution configuration and three prespecified sensitivity scenarios. Forecast
contracts are unchanged. A separate fixed signed-log-return policy feeds a
single net CFD-like position engine, market orders and first eligible subsequent
Bid/Ask quotes. Completed-bar availability, computation delay, venue latency,
expiry, invalid/stale quotes and conservative session-gap cancellation are
explicit. State persists across batches/months. No stops, limits, partial fills,
depth, measured fill probabilities, margin calls or full-run restart are claimed.
Continuous UTC-day funding is declared, including overnight/weekend exposure.

Structured ledgers reconcile closed and open costs, cash, realized/unrealized
PnL and liquidation-side equity. Spread/slippage decomposition is not subtracted
twice. Matched-fill and immediate-midpoint counterfactuals are labelled.
Unknown marks remain unknown. Reports provide counts, gross/net, turnover,
lot-second exposure, known-mark drawdown and monthly closed-position summaries;
return is cutoff equity change / initial cash, without annualization/Sharpe.
Manifests freeze every variant before input values and record resolved configs,
hashes, source identity including uncommitted files, data/forecast/provenance,
assumptions, dependency versions and versioned readiness status. Existing IDs
cannot be overwritten. Real execution access stays before broker-local 2022.

Audit repairs completed:
- RegressionFeatureStore.load has an exclusive optional `before`; both base
  filtering and row-aligned window slicing happen before materialization. Both
  ML development and ensemble slim development callers supply the cutoff.
  Unbounded consumers retain their behavior.
- Matching final_test_started events count as attempted access, including
  interruption. Repeat attempts use the existing explicit logged reason.
  Ensemble common detection avoids adding the same started event twice.
- Missing or undefined registered null evidence cannot become passing ensemble
  eligibility; missing remains untested. Historical result files are untouched.
- EXECUTION_READINESS_V001 blocks globally selected historical features and
  outcome-dependent eligibility/correlation. Every adaptive choice, including
  model fit, must predate each fold; audit intervals must cover the entire run.
  Missing scientific/specification evidence blocks promotion. Hash roles identify
  which supplied evidence needs review; hashes alone do not establish truth.

Actual local inventory (metadata + edge inspection, not a new full-content audit):
- ticks-2e173ef8e61bd240: all 281 monthly files present; 729,244,369 actual footer
  rows; manifest file sizes, per-partition extrema, month completeness and
  reconstructed manifest identity match. Broker-local first/last timestamps:
  2003-05-05 03:01:03.421 through 2026-09-18 23:59:59.079; UTC edge rows:
  2003-05-05 00:01:03.421 through 2026-09-18 20:59:59.079.
- Raw file exists, 36,233,955,746 bytes; unchanged, no fresh full raw hash or
  full quote-content/digest verification. This is not the old 2003-2004 subset.
- 50 primary frozen hashes checked (37 ML, 13 ensemble), 119 fitted manifests
  and their payload hashes checked, 32 feature-manifest content hashes checked.
  No spec/artifact identity errors found. Four prior final-test logs inspected
  as history only. No new final-test evaluation or reserved feature/target load.
- Bounded 2021-06-01 contract availability inspection: 5m had 276 rows and 276
  available expected_return values; 15m had 92 rows and zero available values
  (no_eligible_ensemble). This sample is not a full-history availability claim.

Actual runs (all outputs local, versioned under results/execution):
- EXEC_SMOKE_V001/V002: declared synthetic execution only, two fills and one
  closed long; accounting reconciled. Synthetic fixture PnL is not market evidence.
- EXEC_READINESS_V001/V002: expected exit 1; data metadata passed, chronology
  failed, null evidence unknown, untouched evaluation failed, broker terms and
  audited execution forecast evidence unknown. Status historical_diagnostic_only.
- EXEC_QUOTE_PROBE_V001/V002: no forecasts, decisions, orders or market economics.
  Final V002 streamed 50,000 quotes from 2021-06-01T00:00:00.091Z through
  2021-06-01T07:25:29.755Z (declared request ends next midnight; limit reached).
  Actual V002 wall time 3.4203555 s, CPU 3.140625 s, own-process peak working set
  105,754,624 bytes (100.9 MiB), private/peak commit 353,665,024 bytes (337.3 MiB).
  Counters include imports/preflight, not a RAM guarantee for a full study.
- GUARD_MUTATIONS_V001 detected 11 broken guards; final V002 detected all 16.
  Copies were deliberately broken, selected regression tests failed, copies
  restored, original source hashes unchanged. No actual research source mutation.

Final commands actually run (earlier V001 runs retained):
```powershell
.venv\Scripts\python.exe -m pytest tests/test_execution_engine.py tests/test_execution_readiness.py tests/test_execution_io.py tests/test_stage12_audit_repairs.py tests/test_feature_store.py tests/test_final_test_isolation.py tests/test_ensemble_final_test.py tests/test_prediction_contract.py -q -p no:cacheprovider
.venv\Scripts\ruff.exe check src tests scripts
.venv\Scripts\mypy.exe
.venv\Scripts\python.exe scripts/check_execution_guards.py --output results/execution/GUARD_MUTATIONS_V002.json
.venv\Scripts\python.exe -m xauusd_quant.cli execution-smoke --run-id EXEC_SMOKE_V002
.venv\Scripts\python.exe -m xauusd_quant.cli execution-readiness --run-id EXEC_READINESS_V002
.venv\Scripts\python.exe -m xauusd_quant.cli execution-diagnostic --run-id EXEC_QUOTE_PROBE_V002 --quote-probe --start 2021-06-01T00:00:00Z --end 2021-06-02T00:00:00Z --max-quotes 50000
```
Focused suite: **154 passed in 14.87 s**. Ruff clean; mypy clean, 200 source
files. Targeted suite includes existing feature-store, final-test and forecast
contract regressions. Full suite and unrelated research studies were not run.
Tests hand-check long/short sides, units, costs, financing, clocks/ties/expiry,
gaps, marks, reconciliation, chunk/month equivalence, future-append causality,
DST convention, explicit invalid-forecast rejection and readiness refusal.

Economic conclusions remain blocked. Historical V001 selection uses 2003-2017
development identities and 2018-2021 size selection, later than early ML scores.
Ensemble global eligibility/correlation and wf_universe null/structural statuses
are outcome-dependent across scored blocks. Minimum repair: new versioned,
fold-local feature identities AND counts; nested preprocessing/tuning/calibration;
and prior-only eligibility, nulls, universe, correlation and weights. Keep earlier
records. 2022+ was already inspected in feature research, ML and ensembles;
neither it nor hypothetical future data is automatically untouched. A new period
requires independently established uninspected outcomes. Verified broker/account
terms and an audited, hash-linked execution forecast sidecar are not supplied.
Existing fitted artifacts are present. No real forecast economic simulation was
executed, and earlier research accuracy/ensemble results were not reproduced.
Stage #12 software is complete; no profitability or production claim, no Stage #13.

Files changed (excluding generated immutable results):
- Documentation: AGENTS.md, CLAUDE.md, README.md, WORKLOG.md,
  docs/stage12_execution.md.
- Config: config/execution.yaml, config/execution_scenarios.yaml.
- New package: src/xauusd_quant/execution/{__init__,config,policy,engine,io,
  readiness,runs}.py.
- Integration/repairs: src/xauusd_quant/cli.py, features/store.py, ml/datasets.py,
  ml/final_test.py, ensemble/final_test.py, research/ensemble_research.py
  (the shorter paths share src/xauusd_quant/).
- Tests: tests/test_execution_engine.py, test_execution_readiness.py,
  test_execution_io.py, test_stage12_audit_repairs.py,
  test_final_test_isolation.py (all under tests/).
- Verification: scripts/check_execution_guards.py. Temporary edit helpers removed.

Additional final verification: strengthened the three-scenario integration test
to check that all variants exist in the immutable manifest before either
quote or forecast input opens. Ran `.venv\Scripts\python.exe -m pytest
tests/test_execution_io.py -q -p no:cacheprovider`: **9 passed in 1.31 s**.
All unchanged source lines retained their exact original bytes/line endings.
`git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol
diff --check` passed (CRLF is intentional). Final ruff and mypy checks clean.


## 2026-10-03 - Stage #13 offline strategy-validation framework

Prompt #13 explicitly authorizes this offline stage and supersedes the previous
stop-before-13 boundary. Stop before Stage #14. No broker/MT5, demo/live orders,
deployment, commit or push occurred. Initial working tree was clean at
b9d4596c9abe76f6c808454a713a16f6924e5147; no research process was running. Read
AGENTS.md, CLAUDE.md, this log, README and Stage #12 implementation, configuration,
tests, frozen/spec/artifact inventory, actual manifests and reports first. Reran
Stage #12's eight relevant test files: 154 passed in 17.12 s. Its streaming
execution, costs, availability, accounting, immutable outputs and readiness gates
were verified; previous economic/scientific restrictions were not waived.

Implemented software (details and runnable interfaces in docs/stage13_strategy_validation.md):
- Versioned immutable experiment plans; small prespecified ridge/historical-mean
  family, constant sizing, policy neighbors, four execution scenarios and two
  controls. Register all 24 economic variants before reading evaluation values;
  20 nested fitting attempts plus 24 economic attempts, budget 64 per design.
- New restricted 5m/15m expected future log mid-return path. Three causal features
  from completed bars; training-only identity selection, inner chronological count
  selection, fitted scaling and fixed ridge penalty. Actual label intervals and
  publication are strictly before every fit cutoff. No globally selected Stage
  #9 identities/counts or Stage #11 ensemble universe is reused. External adaptive
  provenance still needs preceding evidence, including ensemble choices/nulls.
- Bounded lazy development scans apply local/derived UTC predicates and row caps
  before collecting price values. Feature-only outer interfaces never attach
  outcomes. Reserved regression/final-test guards remain intact.
- Separate forecast consumers and fixed sign/abstention policy; prediction
  contracts unchanged. Same Stage #12 engine for every primary/benchmark/stress
  trial. Public checkpoint preserves pending orders, positions and cash across
  folds; no artificial resets, compounding, duplicate positions or boundary exits.
- Streamed fold specifications, predictions, decisions, orders, fills, positions,
  costs/equity, fold/aggregate metrics, daily marks and attribution. Trial ledger
  retains failures; invariant failures abort. Per-run code/config/data/model/policy
  and assumption identities include uncommitted source hashes.
- Readiness before evaluation; historical-diagnostic override remains BLOCKED for
  promotion. Matching frozen-plan/source/data identities and prior pipeline null
  evidence plus verified broker/account terms are required for a historical pass.
  Unknown remains unknown. No prospective mode or fresh-test claim.
- Complete-family verdict, fold variation, cost sensitivity, fixed parameter
  neighborhood, concentration/removal diagnostics and conditional moving-block
  bootstrap of complete UTC-day PnL. Fixed five-day blocks, 199 replicates/seed
  130013, minimum 20 daily observations; no Sharpe/annualization or search correction.

Actual data/provenance: current tick manifest remains ticks-2e173ef8e61bd240, all
281 monthly files, 729,244,369 footer rows, broker-local 2003-05-05 03:01:03.421
through 2026-09-18 23:59:59.079. Raw 36,233,955,746-byte input is unchanged.
Stage #13 checks bar/tick lineage, conventions, configuration identity, actual bar
partition row counts/sizes and coverage; it does not rehash every historical value.
This is not the old 2003-2004 partial history. Stage #12's verified 50 primary
frozen specs, 119 fitted manifests/payloads and 32 feature manifests are present,
but presence does not make their creation chronology clean. Existing Stage #9-#11
forecast-accuracy/ensemble claims were not reproduced. Four final-access logs
remain history; no new 2022+ feature, target or quote evaluation was performed.

Plans and the preserved failed attempt:
- STRATEGY_5M_RECONSTRUCTION_V001 and STRATEGY_15M_RECONSTRUCTION_V001 were frozen
  before new Stage #13 economics. The chosen inner dates (May 15/22, 2021) were
  Saturdays. This avoidable schedule error yielded insufficient training/labels.
- STRATEGY_5M_DIAGNOSTIC_V001 recorded every one of 44 attempts: 36 failed nested
  fits/frozen models/candidate executions and eight completed controls. No partial
  model economics were promoted. Wall 18.0925 s; own-process peak working set
  128,274,432 bytes and private memory 464,883,712 bytes. 15m V001 was not executed.
- V002 moves both inner windows back exactly two days to May 13/20 weekdays;
  models, outer schedule, policies, costs and budget unchanged. Both V002 plans
  were frozen after V001 control outcomes, before corrected model economics,
  with supersedes/revision reason and prior attempts. This is a post-attempt
  revision, not original preregistration; all V001 artifacts remain immutable.
- V002 plan-file SHA-256: 5m
  973bd47e2a1ea0c180a95e48254be49ab81c464b64f6e0e871c61d6cf7a1101a;
  15m a9a2b982667c6b8ba4a12603d164006757e113b3c5ff8bcfece0ebb8e1cda6d4.

Actual bounded historical diagnostics (each sequential; no expansion):
- Train May 2021; two chronological preceding inner weekdays; contiguous outer
  folds June 1, 2021 00:00-00:30 and 00:30-01:00 UTC. Each economic trial saw
  4,707 actual quotes. Fixed 0.01 lots x hypothetical 100 oz/lot = 1 oz; USD
  account, initial cash USD 10,000, declared continuous financing. Terms and
  slippage are hypothetical scenarios, not verified Exness execution.
- STRATEGY_5M_DIAGNOSTIC_V002 and STRATEGY_15M_DIAGNOSTIC_V002 each completed all
  44 attempts. Every one of the 16 forecast-policy/scenario variants per timeframe
  had negative marked-equity change. No-trading controls were zero; fixed-seed
  directional controls were negative, and happened to match historical-mean
  decisions in this short sample. All variants were retained; no winner chosen.

| Primary zero-threshold / base scenario | Closed trades | Gross executable PnL USD | Commission USD | Financing USD | Net USD | Fold net USD |
|---|---:|---:|---:|---:|---:|---|
| 5m ridge | 6 | -2.5430 | 0.3600 | 0.000416 | -2.903416 | -0.805208, -2.098208 |
| 5m historical mean | 6 | -2.8620 | 0.3600 | 0.000416 | -3.222416 | -2.544208, -0.678208 |
| 15m ridge | 2 | -1.6970 | 0.1200 | 0.000416 | -1.817416 | -1.610208, -0.207208 |
| 15m historical mean | 2 | -1.6970 | 0.1200 | 0.000416 | -1.817416 | -1.610208, -0.207208 |

For 5m ridge: 12 submitted/filled legs, six accepted decisions and six rejected
overlaps. For 15m ridge: four submitted/filled legs, two accepted and two rejected
overlaps. No base-scenario expiry/unfilled orders or final open exposure. Closed
spread/slippage components USD 2.1485/0.24 (5m ridge) and 0.6815/0.08 (15m ridge)
already enter gross executable PnL; subtract only commission and financing from
that gross. This avoids counting spread/slippage twice. Full forecast-family net
ranges: 5m [-4.296666, -2.145277], 15m [-2.158166, -1.817416]; these are coverage
summaries, not a choice of the best variant. Threshold/latency scenarios can change
trade populations, so aggregate cost sensitivity is not matched-trade inference.
Detailed turnover, exposure, drawdown and all period/scenario records are saved.

Resource measurements, own Windows process including imports/preflight:
5m V002 wall 41.7828 s, CPU 42.8906 s, peak working set 132,075,520 bytes (125.96
MiB), private 471,945,216 bytes, peak commit 474,120,192 bytes. 15m V002 wall
39.7844 s, CPU 40.4531 s, peak working set 123,465,728 bytes (117.75 MiB), private
451,510,272 bytes, peak commit 452,636,672 bytes. These bounded measurements are
not a full-history memory guarantee. Both had zero complete daily observations:
no numeric uncertainty interval. Two folds and 6/2 trades cannot meet frozen
minima (three folds, 30 trades, 20 daily observations), even with all gates passed.

Verdict: both candidates BLOCKED at both timeframes. Data metadata, restricted
fold-local implementation and honest history declaration pass; matching prior
random-walk/sign-flip pipeline null evidence, verified account/contract/fee terms
and corresponding evidence identities are unknown. Real runs used the explicit
historical-diagnostic override, never passing promotion. No validated edge,
prospective profitability or production readiness is established. The negative
one-hour sample does not establish full-history rejection either. Before any
economic conclusion: obtain independently reviewed, plan/pipeline-linked prior
null evidence and supplied execution specifications, then freeze adequate
historical coverage in a new version. Reusing old Stage #9-#11 models additionally
requires a genuine fold-local selection/universe creation path for every adaptive
choice. 2022+ was inspected before; no period is assumed untouched just because
new timestamps exist. Older research search counts remain unknown.

Known Stage #13 historical attempts: 132 total, 36 failed and 96 completed,
chronological prior counts 0/44/88. Synthetic smoke attempts (88) are separate.
V002 registration records 44 known preceding attempts; the 15m V002 registration
preceded the 5m V002 run, so its original manifest retains that registration-time
count. DIAGNOSTIC_SUMMARY_V001.json records the actual 88 attempts before its run
without editing either immutable plan/manifest.

Commands actually executed (CLI via `.venv\Scripts\python.exe -m xauusd_quant.cli`):
- `strategy-plan --plan config/strategy_validation.yaml` and the 15m counterpart,
  for V001 then V002; immutable registration succeeded each time.
- `strategy-readiness --plan config/strategy_validation.yaml --run-id
  STRATEGY_READINESS_5M_V001`: exit 1 expected, missing scientific/spec gates.
- `strategy-smoke --plan config/strategy_validation.yaml --run-id
  STRATEGY_SMOKE_5M_V001` and 15m counterpart STRATEGY_SMOKE_15M_V001: 44 attempts
  each, INCONCLUSIVE software evidence. Wall 2.1428/1.0811 s; peak working sets
  88,928,256/89,169,920 bytes.
- `strategy-validate --plan config/strategy_validation.yaml --run-id
  STRATEGY_5M_DIAGNOSTIC_V001 --historical-diagnostic`, then V002, then the 15m
  config with STRATEGY_15M_DIAGNOSTIC_V002. Expected exit 1: BLOCKED verdicts.
- `.venv\Scripts\python.exe -m pytest tests/test_strategy_chronology.py
  tests/test_strategy_execution.py tests/test_strategy_framework.py
  tests/test_strategy_source.py tests/test_execution_engine.py
  tests/test_execution_readiness.py tests/test_execution_io.py
  tests/test_stage12_audit_repairs.py tests/test_feature_store.py
  tests/test_final_test_isolation.py tests/test_ensemble_final_test.py
  tests/test_prediction_contract.py -q -p no:cacheprovider`: initially 182 passed
  in 15.62 s; after adding favorable full-family/unknown-economic verdict tests,
  final **186 passed in 15.38 s**. No full suite or full research rerun.
- `.venv\Scripts\python.exe scripts/check_strategy_guards.py --output
  results/strategy_validation/GUARD_MUTATIONS_V001.json`: all **26 deliberately
  broken guards detected**, exit 0. Isolated copies restored between mutations;
  original source hashes unchanged. Includes 16 Stage #12 guards and 10 Stage #13
  chronology, pre-materialization, ensemble, null, readiness and accounting guards.
- `.venv\Scripts\ruff.exe check src tests scripts` clean; `.venv\Scripts\mypy.exe`
  clean, 208 source files. `git -c core.whitespace=blank-at-eol,blank-at-eof,
  space-before-tab,cr-at-eol diff --check` clean; original matching source lines
  retained exact bytes/line endings, UTF-8 without BOM. Final checks rerun.
- Read-only saved-record validation: eight specs, 32 predictions, 48 economic
  trials, 304 fills and 72,528 marks checked for preceding labels/inner cutoffs,
  bar-close availability, frozen-spec hashes, subsequent-arrival fills, duplicate
  trades, preserved fold marks and PnL/accounting reconciliation. Passed; written
  immutably to FINAL_VERIFICATION_V001.json. No new outcome evaluation.

Outputs (local ignored results, never replacing historical artifacts): plans and
runs under results/strategy_validation/, GUARD_MUTATIONS_V001.json,
DIAGNOSTIC_SUMMARY_V001.json and FINAL_VERIFICATION_V001.json. No results/data
were deleted. Temporary edit/verification helpers removed.

Changed files (23 source/config/documentation/test files):
- AGENTS.md, CLAUDE.md, README.md, WORKLOG.md,
  docs/stage13_strategy_validation.md.
- config/strategy_validation.yaml, config/strategy_validation_15m.yaml.
- src/xauusd_quant/cli.py; src/xauusd_quant/execution/engine.py.
- src/xauusd_quant/strategy_validation/__init__.py, plan.py, pipeline.py,
  source.py, policies.py, readiness.py, metrics.py, runs.py (all eight in that
  new package).
- tests/strategy_synth.py, tests/test_strategy_chronology.py,
  tests/test_strategy_execution.py, tests/test_strategy_framework.py,
  tests/test_strategy_source.py.
- scripts/check_strategy_guards.py.


## 2026-10-04 - Stage #13.5 bounded econometrics, causal state and uncertainty

Completed the explicitly authorized offline extension. No broker/demo/live
connection or order, deployment, commit, push or Stage #14 progression.
Initially clean main at 45ff2fd; inspected working tree, AGENTS.md, CLAUDE.md,
README/WORKLOG, stages 9-13 code/config/reports, data manifests, frozen/fitted
artifacts, source APIs and existing diagnostics before editing. No research
process was running and no cache-stamped ML/selection/regime module was edited.

Implemented `xauusd_quant.econometrics` (11 modules). Reused Stage 13 label
purge, chronological inner count selection for its ridge reference, TrialLedger,
bounded guarded bar source and readiness; Stage 12 immutable records, code
identities, resource measurement and timestamp/reserved-period semantics.
Existing ADF/KPSS, Ljung-Box, ARCH-LM and detrended-random-walk routines reused.
New public `BarDevelopmentSource.completed_bars` wraps that existing bounded
reader. Forecast contracts and the execution engine were not changed.

Fixed ARX (r1, r2, prior volatility12) with segmented Bartlett HAC; constrained
zero-mean Gaussian quasi-MLE GARCH(1,1), explicit convergence/variance failures;
nonnegative arithmetic-level HAR-style variance1/3/12; prior-only Gaussian
local log-level Kalman MLE/filter with Joseph covariance and gap resets.
Zero return, aligned fold-local Stage 13 ridge, rolling12 and EWMA0.94 remain
benchmarks. No coefficient is claimed causal and filtered level is not fair value.
GARCH h-step forecasts sum expected variances under stated zero-mean assumptions;
HAR uses intraday contiguous grid windows, not purported daily components.
No logarithmic variance retransformation or silent favorable fallback.

Return target is future h-grid log midpoint return; variance target is the sum
of future h contiguous grid squared returns. No gap interpolation or return
across closures; 12-return features require 13 closes. Bar opening timestamps,
bar-close availability and computation/publication delays explicit. Stored
broker-local times use the existing New York +7h DST convention. Statistical
midpoints do not replace executable Bid/Ask in economics. Training-only quote
noise signature diagnostics are descriptive, not proof of microstructure noise.

Separate rolling residual and clipped ACI-inspired prediction intervals on
ARX/ridge: preceding 128 matured errors, minimum 32, nominal 80%, gamma0/.01.
Strict outcome maturity plus monotonic update/decision clocks, bounded pending
state and explicit missing/outcome censoring. Exact serialized reload and batch
carry. Intervals concern outcomes, not confidence bounds on expected profit.
No exchangeability guarantee for dependent data. Two-sided bounded CUSUM on
preceding-scale standardized matured ARX errors is health only; drift 0.5 / threshold 8,
no strategy change, risk rule or evaluation-driven retraining.

Frozen plans ECONOMETRICS_5M_V001 / ECONOMETRICS_15M_V001 registered before June
extension outcome inspection. Plan-file SHA256:
- 5m: 9fba86e897c38bf2bfbe614c15e61fe07dbc45c0f632b5b554ad9960bf024856
- 15m: fec05fbb8d4b3a8241ec763896a7b5f379348d5ef3baf85259c2290862b46395
Eight model/benchmark fits + four wrappers + monitor per fold, twelve ridge
inner fits: 51 attempts per path within 64. May expanding fit; preceding 24h
calibration; three June 1 UTC eight-hour outer folds. Inner May 13/20 exclusively
prior. No added-model search, economic policy or parameter expansion. Cap 20,000
bars/scan, 240 sec/path, checked 1 GiB own-private memory, optimizer 200 iterations.
15m followed 5m saved-record/resource checks regardless of outcome sign.

Data metadata actually audited: ticks-2e173ef8e61bd240, 281 monthly partitions,
729,244,369 footer rows, broker-local 2003-05-05 03:01:03.421 through
2026-09-18 23:59:59.079; manifest SHA256
da8d0bfe17a33ca95080e872ac98f8474721616309234e29ce339e2236f4318c.
Raw CSV exists at 36,233,955,746 bytes, unchanged. Footer/config/path/identity
checks passed; full-content and raw hashes were not independently recomputed.
Actual frozen/fitted/feature manifests and final-test access logs inventoried.
Old global feature identities/counts and ensemble outcome-dependent universes
remain invalid for earlier fold-local reuse. 2022+ was already inspected; no
new reserved evaluation or bypass. This is not the old 2003-2004 partial dataset.

Audit incident preserved: an initial missing-input test changed project_root
but retained absolute real paths. A software-classified run without SyntheticBars
therefore evaluated May 4 2021 development data before its intended input error.
34 attempts, 19 completed/15 failed. Original records copied verbatim to
results/econometrics/audits/UNINTENDED_REAL_FIXTURE_V001, with separate
ACCESS_AUDIT.json and ORIGINAL_PLAN.json, hashes and explicit incorrect original
classification. No reserved access. No candidate specification changed from
those outcomes. Software correctness now requires an explicit synthetic reader
before even metadata access; its mutation regression uses forbidden metadata
and cannot repeat real reads. Original records were not rewritten as clean evidence.

Bounded runs executed:
- ECON_READY_5M_V001: expected exit1. Data/fold-local path/history gates passed;
  matching null evidence, audited promotion identity and supplied execution terms
  unknown. No model fits or outcome-value study in this readiness-only run.
- ECON_SMOKE_5M_V001: 51/51 completed, zero failed models. Wall 21.0563891 sec,
  CPU 18.546875 sec, peak working set 270,352,384 bytes, private 763,867,136 bytes.
- ECON_5M_DIAGNOSTIC_V001: 51/51 completed, zero failed/rejected forecasts,
  260 scored observations/model, folds 95/95/70; censored 1/1/2 per model.
  Wall 12.8210345 sec / CPU 12.640625 sec, peak working set 278,241,280 bytes,
  private 824,905,728 bytes. June 1 only; gaps/feature warm-up reduce coverage.
- ECON_15M_DIAGNOSTIC_V001: 51/51 completed, zero failed/rejected forecasts,
  77 scored observations/model, folds 26/31/20; censored 1/1/1 per model.
  Wall 16.2982309 sec / CPU 5.734375 sec, peak working set 255,897,600 bytes,
  private 793,100,288 bytes. These are measured Windows own-process values;
  no invented memory guarantee or concurrent heavy runs.

Matched forecast effect sizes (positive = loss reduction):

| Candidate/benchmark | 5m | 15m |
| --- | ---: | ---: |
| ARX/zero, relative MSE | -1.3198% | -4.1167% |
| ARX/aligned ridge, relative MSE | +0.1426% | -0.2998% |
| Kalman/zero, relative MSE | -0.7563% | -2.1588% |
| Kalman/aligned ridge, relative MSE | +0.6979% | +1.5864% |
| GARCH/rolling, absolute QLIKE | +0.02046 | +0.29499 |
| GARCH/EWMA, absolute QLIKE | +0.08268 | +0.11594 |
| HAR/rolling, absolute QLIKE | -0.18568 | +0.17974 |
| HAR/EWMA, absolute QLIKE | -0.12347 | +0.00070 |

All four additions: INCONCLUSIVE with promotion blocked. Keep ARX/GARCH/HAR as
interpretable benchmarks and Kalman as a causal state/forecast benchmark.
GARCH favorable aggregate variance loss is not stable superiority (rolling
comparison positive in 1/3 5m and 2/3 15m folds). Negative return/HAR findings
preserved. One evaluation day is below frozen 5-day inference minimum, and 77
15m rows below 200. No bootstrap interval or decorative DM/multiple-testing
statistic, Sharpe, annualization or economic superiority assertion. Full fold
loss effects/coefficient stability persist. Simple moving-block uncertainty
implemented for eligible larger declared studies, conditional on fitted forecasts;
not a bootstrap of the whole discovery process. No such larger study executed.

Uncertainty evidence INCONCLUSIVE; role = calibration diagnostic. ARX rolling/
adaptive empirical coverage 76.15%/79.23% at 5m and 75.32%/77.92% at 15m;
log-return mean widths .0009359/.0010459 and .0014544/.0015884. Nominal 80%.
Aligned ridge coverage 75.77%/79.23% and75.32%/76.62%. Wider adaptive intervals
are not proof of superiority. Per-fold variation retained (15m ARX rolling
76.92%/61.29%/95.00%). CUSUM role = health diagnostic, market evidence INCONCLUSIVE:
four/one outer-forecast alarms. Fixed 500-observation unchanged Gaussian control
zero alarms; +3 sigma shift first detected after 3 matured updates. No measured
market false-alarm rate or automatic action. Existing detrended-RW nulls preserved.

Economics BLOCKED BY MISSING INPUTS OR EVIDENCE, no new economic candidate
registered/evaluated. Original Stage 13 frozen baseline preserved. Required
matching prior-only random-walk/sign-flip evidence, audit-file identities and
verified supplied execution assumptions absent. Adequate coverage and those
forecast gates require a new frozen design before any expanded study. Legacy
boosted/ensemble reuse additionally needs feature identities/counts, preprocessing,
tuning/calibration and universe/null/correlation/weight choices reconstructed
within each actual preceding fold. Earlier research/economics claims were not
reproduced. Historical reconstruction remains retrospective, not prospective.

Known historical ledger after this extension: 268 attempts, 217 completed, 51 failed
(132 Stage 13 + 34 incident + 102 declared real econometric). Actual prior counts
before real runs 166/217. Both plans froze before 5m, so immutable 15m manifest
retains registration count 166; FINAL_VERIFICATION_V001.json records actual 217
without rewriting it. Synthetic smoke 51 attempts separate. Older searches unknown.

Commands actually run (CLI via `.venv\Scripts\python.exe -m xauusd_quant.cli`):
- `econometric-plan --plan config/econometrics.yaml`, and 15m counterpart.
- `econometric-readiness --plan config/econometrics.yaml --run-id ECON_READY_5M_V001`.
- `econometric-smoke --plan config/econometrics.yaml --run-id ECON_SMOKE_5M_V001`.
- `econometric-evaluate --plan config/econometrics.yaml --run-id ECON_5M_DIAGNOSTIC_V001`.
- `econometric-evaluate --plan config/econometrics_15m.yaml --run-id ECON_15M_DIAGNOSTIC_V001`.
- `.venv\Scripts\python.exe -m pytest tests/test_econometric_models.py
  tests/test_econometric_state.py tests/test_econometric_uncertainty.py
  tests/test_econometric_framework.py tests/test_econometric_source.py
  tests/test_strategy_chronology.py tests/test_strategy_execution.py
  tests/test_strategy_framework.py tests/test_strategy_source.py
  tests/test_execution_engine.py tests/test_execution_readiness.py
  tests/test_execution_io.py tests/test_stage12_audit_repairs.py
  tests/test_feature_store.py tests/test_final_test_isolation.py
  tests/test_ensemble_final_test.py tests/test_prediction_contract.py
  tests/test_stationarity.py tests/test_autocorrelation.py -q -p no:cacheprovider`:
  **271 passed in 33.44 sec**. 35 new econometric tests; synthetic white-noise/AR,
  conditional variance, drift, shifts/RW/delayed outcomes. Earlier pre-change
  Stage 12/13 focused suite 186 passed in 18.16 sec. No full suite/research pipeline.
- `.venv\Scripts\python.exe scripts/check_econometric_guards.py --output
  results/econometrics/GUARD_MUTATIONS_V001.json`: 35 broken guards detected.
  Same command with V002: all 36 detected, includes new calibration update-clock
  guard plus prior 26 Stage 12/13 guards. Isolated copies; original source unchanged.
- `.venv\Scripts\python.exe -m ruff check src tests scripts`:clean.
  `.venv\Scripts\python.exe -m mypy src`:clean, 219 source files.
- Read-only temporary `_verify_econometric_runs.py` wrote immutable
  FIVE_MINUTE_CHECK_V001.json (before 15m) and FINAL_VERIFICATION_V001.json:
  72 frozen fits, 162 reloadable states, 18,688 raw forecasts, 4,976 scored records
  across smoke and two real runs. Verified prior fit labels/parameters, calibration
  maturation/update clocks, availability/target alignment, spec hashes, duplicate
  exclusion, exact state reload, terminal trial records, checked budgets and
  blocked promotion/economics. Passed. No new outcomes read.

Outputs are local ignored results/econometrics/plans, runs, audits, mutation and
verification sidecars; never overwrote existing research files or data. Temporary
editing/verification helpers removed after use. Relevant source line endings and
UTF-8 without BOM preserved. Dependency APIs verified (SciPy 1.18.1,
statsmodels 0.15.0); no new dependency, no arch package required.

Changed files (27 source/config/documentation/test files):
- AGENTS.md, CLAUDE.md, README.md, WORKLOG.md, docs/stage13_5_econometrics.md.
- config/econometrics.yaml, config/econometrics_15m.yaml.
- src/xauusd_quant/cli.py, src/xauusd_quant/strategy_validation/source.py.
- src/xauusd_quant/econometrics/__init__.py, plan.py, data.py, diagnostics.py,
  regression.py, volatility.py, state_space.py, uncertainty.py, monitor.py,
  comparison.py, runs.py (11 modules in that new package).
- tests/econometric_synth.py, tests/test_econometric_models.py,
  tests/test_econometric_state.py, tests/test_econometric_uncertainty.py,
  tests/test_econometric_framework.py, tests/test_econometric_source.py.
- scripts/check_econometric_guards.py.

## 2026-10-04 - Stage #14 bounded robustness and selection exposure

Implemented Prompt #14's explicitly authorized offline extension. Initial tree
clean; read AGENTS/CLAUDE/WORKLOG, prior-stage docs/code/configs and actual saved
artifacts before editing. Preserved all prior results, failures and data. No raw
changes, new market outcome study, reserved outcome values, broker access,
demo/live orders, deployment, commits/pushes or Stage #15. No cached ML/selection/
regime source was edited. No research process was active before changes.

Reused Stage12 inventory/hash/resource APIs and accounting, Stage13 immutable
records, TrialLedger, nested selectors/fits and execute_trial, Stage13.5 losses,
Kalman/interval/CUSUM APIs. Ten new robustness modules implement immutable plans
and source-byte identity sidecars; recorded selection inventory; explicit return
sampling/capital/exposure/compounding/duration/missing/overnight/overlap/dependence/
annualization conventions; zero-variance unavailable states; complete-family Holm;
segmented moving blocks; saved fit/maturity/target/loss audit; whole fixed synthetic
discovery and perturbation controls; causal operational replay, matched costs and
fixed-population break-even; historical attribution and candidate verdicts.
No competing engine, IID time-series errors, trade-PnL Sharpe, fabricated effective
trial count or unsupported named selection statistic. Holm formula checked against
Goeman/Solari (2010) section3; block resampling against primary author paper/report
and explicit independent index arithmetic. Primary references/assumptions in docs.

Metadata audit passed: actual281 partition footer files and729,244,369 rows,
ticks-2e173ef8e61bd240, source size36,233,955,746; same tick-manifest SHA256
da8d0bfe17a33ca95080e872ac98f8474721616309234e29ce339e2236f4318c.
Raw/full quote content hashes not recomputed. Current primary frozen/fitted/feature
identities checked by existing readiness adapter. Actual registry metadata684 ML,
565 ensemble,330,790 feature tests and389,480 shared ledger rows. Counts are
registered tests, not independent candidate trials. Historical upstream268 attempts:
217 completed/51 failed, including the preserved34 misclassified real fixture
attempts. Upstream synthetic139 attempts separate. Four reserved-access logs hashed,
including the interrupted ensemble start and authorized repeat; no reserved values.
Old feature/count/universe chronology unresolved; older searches unknown.

ROBUSTNESS_V001 frozen before new diagnostics. V002 explicitly added fixed synthetic
Kalman q/window/gamma/CUSUM sensitivities after V001 results to complete the requested
infrastructure. Market candidates, periods and passing criteria unchanged; no market
search expansion. Both plans/artifacts retained. V002 content digest:
e69b2a65819130c0d438b82beb1907934af8968c61dfc99f6a31d7ad7dda7910.
Source byte hashes are immutable *_SOURCES sidecars, registered before the final
source-bound analysis. They do not retroactively claim historical outcomes were
uninspected. All final model/forecast source identities and code/config/dependency
versions recorded. Resource limits256 operations,25,000 records/file,240sec/1GiB
checked between operations. Sequential bounded studies; no full suite/heavy research.

Actual CLI commands (`.venv\Scripts\python.exe -m xauusd_quant.cli`):
- `robustness-plan` initially V001, later V002 through config/robustness.yaml.
- `robustness-inventory --run-id ROBUST_INVENTORY_V001`:17.2186sec,
  peak93,507,584/private308,080,640 bytes; metadata/ledgers only.
- `robustness-smoke --run-id ROBUST_SMOKE_V001`:189 completed operations,
  16.1150sec, peak92,315,648/private339,288,064 bytes.
- `robustness-evaluate --run-id ROBUST_HISTORICAL_V001`:18 completed operations,
  2.9923sec, peak98,283,520/private311,095,296 bytes; saved development losses only.
- `robustness-smoke --run-id ROBUST_SMOKE_V002`:192 completed,
  18.5730sec, peak121,200,640/private633,004,032 bytes.
- Final `robustness-smoke --run-id ROBUST_SMOKE_V003`:192 completed,
  17.0089sec, peak121,458,688/private633,528,320 bytes. Necessary final replay after
  strengthening source binding and provenance; no changed market inputs/criteria.
- Final `robustness-evaluate --run-id ROBUST_HISTORICAL_V002`:18 completed,
  8.6510sec, peak98,332,672/private311,144,448 bytes. Reuses saved fits/scores; no
  retraining or new market outcomes. The final manifests match current source.
These are measured own-process Windows counters, not total machine memory.

Executed bounded findings:12 IID-increment null pipelines and12 controlled AR(.5)
pipelines repeat causal features, label purge, inner count/identity/scaling/fit and
outer fit, followed by fixed sign decisions in Stage12. Positive loss-effect signs
3/12 null versus12/12 AR, not significance/false-discovery rates. Maximum binomial
SE .144; finite null false positives allowed. Dependent AR(.8) block mean SD
.03492 singleton versus .06171/.08248/.09068 for4/12/24. Singleton is a fixture
sensitivity only. Synthetic execution8 deterministic adverse cases +16 MC paths;
all cash/cost/fill records retained. MC PnL q5/50/95=-12.206/-2.826/+15.500
account units, hypothetical assumptions on a synthetic path, not broker measurement
or future forecast. Median loss preserved. Ridge .5/1/2 penalty, momentum ablation
and threshold neighbor retained whole; no selected profitable replacement.

Econometric controls: synthetic Kalman parameters fitted on prior300 observations,
q multipliers .5/1/2, positive Joseph covariances. Delayed intervals64/128 x
gamma0/.01,30sec publication delay and fixed doubled scale after400. Coverage
78.26/77.21% rolling versus80.34/80.21% adaptive, with wider adaptive intervals.
CUSUM threshold6 any null alarm3/12; thresholds8/10 zero/12, not a zero-rate proof;
+3sigma detection2-5 observations. Whole neighborhoods retained; no health action.

Saved forecast audit checked2,696 scores across48 fitted specifications. Recomputed
June1 effects match Stage13.5: ARX/zero relative MSE -1.3198%/-4.1167%; Kalman/zero
-0.7563%/-2.1588%; GARCH/rolling QLIKE +.02046/+.29499, but only1/3 and2/3 positive
folds. Negative5m HAR differences preserved. All48 block configurations insufficient
(one day;15m77 rows). No market CI, p-value or selection-aware significance claim.
Holm unavailable for full16 predictive hypotheses; economic inference unavailable.
Coefficient/state/coverage/width records retained by hashes; no market perturbation
refits claimed. Concentration is retrospective closed-trade/month attribution;
single-day/hour evidence cannot support long-term decay or regime economics.

Verdicts: ARX/Kalman/GARCH/HAR at both timeframes BLOCKED for promotion by missing
matching prior-only null/evidence identities; conditional evidence INCONCLUSIVE.
Ridge/historical mean BLOCKED by matching evidence and verified supplied execution
terms. Legacy ML/ensembles additionally BLOCKED by global adaptive chronology.
No Stage15 candidate. Search exposure partially recorded, effective trials and
whole historical discovery uncertainty unknown. No new uninspected period proved.
Do not enlarge the search until something passes; any expanded study needs a new
frozen plan, adequate coverage and required scientific prerequisites.

Verification commands/outcomes:
- `.venv\Scripts\python.exe -m pytest tests/test_robustness_statistics.py
  tests/test_robustness_framework.py tests/test_robustness_execution.py
  tests/test_econometric_models.py tests/test_econometric_state.py
  tests/test_econometric_uncertainty.py tests/test_econometric_framework.py
  tests/test_econometric_source.py tests/test_strategy_chronology.py
  tests/test_strategy_execution.py tests/test_strategy_framework.py
  tests/test_strategy_source.py tests/test_execution_engine.py
  tests/test_execution_readiness.py tests/test_execution_io.py
  tests/test_stage12_audit_repairs.py tests/test_final_test_isolation.py
  tests/test_prediction_contract.py -q -p no:cacheprovider`:221 passed/48.12sec.
- Final Stage14-only set adds `tests/test_robustness_econometric_controls.py`
  and `tests/test_robustness_source_identity.py` to the three robustness test files:
 18 passed/2.15sec. Total225 distinct targeted tests over these two commands;
  no full suite. Earlier17-only verification passed/2.01sec.
- `.venv\Scripts\python.exe -m ruff check src tests scripts`:clean.
  `.venv\Scripts\python.exe -m mypy src`:clean,229 source files.
  `git diff --check`:clean.
- `.venv\Scripts\python.exe scripts/check_robustness_guards.py --output
  results/robustness/GUARD_MUTATIONS_V001.json`:preserved obstructed audit.
  Pytest child processes could not access their Windows temporary directory;
  setup errors were correctly not counted as detected guard failures.
  Approved outside-sandbox rerun V002 detected all41 guards, source unchanged.
  Final V003 detected all42 guards, including saved source identity; original
  source hashes unchanged. Disabled guards were restored in the isolated copy.
  A later sandbox test invocation likewise had11 pass/6 setup errors; approved
  rerun passed all17, and final18 passed. No permission errors treated as passes.
- `.venv\Scripts\python.exe scripts/verify_robustness.py --run-id
  ROBUST_SMOKE_V003 --run-id ROBUST_HISTORICAL_V002 --output
  results/robustness/FINAL_VERIFICATION_V001.json`:passed. Checked22,606 order events,
  11,302 fills,5,646 closed trades for strict arrival/expiry, cost decomposition,
  cash reconciliation,192+18 terminal attempts, immutable plan/source identities,
  current source fingerprints and unchanged reserved access logs. Saved records only.
- `.venv\Scripts\python.exe scripts/summarize_robustness.py --run-id
  ROBUST_SMOKE_V002`, and ROBUST_HISTORICAL_V001:read-only summaries verified above.

Changed files (28 source/config/documentation/test files; UTF-8 without BOM):
- AGENTS.md, CLAUDE.md, README.md, WORKLOG.md.
- docs/stage12_execution.md, docs/stage13_strategy_validation.md,
  docs/stage13_5_econometrics.md, docs/stage14_robustness.md.
- config/robustness.yaml; src/xauusd_quant/cli.py.
- src/xauusd_quant/robustness/__init__.py, plan.py, inventory.py, statistics.py,
  resampling.py, reports.py, stress.py, studies.py, econometric_controls.py, runs.py.
- tests/test_robustness_statistics.py, tests/test_robustness_framework.py,
  tests/test_robustness_execution.py, tests/test_robustness_econometric_controls.py,
  tests/test_robustness_source_identity.py.
- scripts/check_robustness_guards.py, scripts/summarize_robustness.py,
  scripts/verify_robustness.py.
Local ignored results/robustness holds plans, inventories, immutable versioned
studies, detailed engine streams, failed audit, mutation/verification reports.
No results or data deleted or overwritten. Stop after Stage14.

## 2026-10-04 - Stage #14 commit and push authorization

The user separately requested `commit and push` after Stage #14 completion.
This authorizes the Git commit and push of the 28 Stage #14 source, configuration,
test and documentation files. Earlier no-commit/no-push instructions describe
the research implementation request; broker, deployment and Stage #15 restrictions
remain. Pre-commit inspection confirms `main`, the expected GitHub origin, no
unrelated working-tree changes, and a clean `git diff --check`. Data and local
research artifacts remain ignored. Prior targeted tests, guard mutations, lint
and typing results above remain the validation evidence for this unchanged code.

## 2026-10-04 - Stage #15 offline alpha portfolio

Prompt #15 authorizes bounded offline alpha combination and shared accounting,
superseding prior stage stopping rules for this extension. The tree was clean at
fde1c79 before edits. No commit/push, broker connection, demo/live order, deployment,
raw-data change, reserved outcome access or Stage #16 work. No cached ML, selection,
regime or ensemble source was changed. Historical artifacts and failed runs remain.

Actual evidence audited: Stage12/13/13.5 documentation, source, ledgers/manifests,
Stage14 ROBUST_HISTORICAL_V002 verdict/inventory and ROBUST_SMOKE_V003 verdict.
Stage14 has no eligible Stage15 candidate. Initial 5m Stage13 V001 has 44 attempts,
36 failed/8 completed, no frozen fits. Corrected 5m/15m V002 each has 44 completed
attempts and four actual frozen fits over two folds. The first registry audit
incorrectly referenced only V001; the directory audit found the V002 files, and
registry/plan V003 corrects that reference without erasing earlier versions.
Saved Stage13 economics are negative across all16 forecast variants/timeframe;
6/2 trades in one hour cannot support robust uncertainty. These recorded market
results were inspected, not reproduced or used to select portfolio methods.
Stage13.5 actual 5m/15m fits remain forecast evidence, with matching null/evidence
and economic gates missing. Legacy feature/count/universe chronology unresolved.
217 completed/51 failed recorded upstream historical attempts; older discovery
searches and effective independent trials unknown. No new untouched period proved.

Registry ALPHA_REGISTRY_V003 holds66 records:16 directional policy hypotheses,
50 diagnostic/forecast entries, zero scientific eligibility. Actual model/feature/
data/validation/specification identities and123 source references are hash-bound.
Variance, uncertainty, health and residual probabilities supply no forced direction.
Software readiness is separate from eligibility. The scientific constructor
requires complete policy units/horizons/clocks, matching gates and evidence known
strictly before the fold. Current missing market maximum-age choice stays unknown.
Ticks identity remains ticks-2e173ef8e61bd240; saved metadata729,244,369 rows/281
partitions, current tick-manifest hash independently checked; no full data scan.
Four reserved access logs including interruption/repeat records remain unchanged.

ALPHA_PORTFOLIO_V001 frozen before portfolio outcomes. V002 improved source binding
but still used older Stage13 IDs. V003 corrects actual upstream fit references and
explicit policy completeness under the original eligibility rule. All versions
remain. Candidate family, methods, research lot budget, evaluation periods and
acceptance unchanged; no search expansion after losses. Replays are repeated
software studies, not independent model trials or new selection significance.

Implemented8 modules: versioned registry/plan, comparable signed unit-budget
intents, observed-bar sign policy adapters, prior-only covariance and constrained
allocation, streaming timing/expiry, shared Stage12 target account, reconciled
entry-owner attribution, serialized restart and bounded sequential reports.
Equal and one minimum-variance alternative only; fixed half-sample/half-diagonal
shrinkage, preceding32 aligned 5-minute standalone fixed-capital returns, >=16
complete rows, recorded equal fallback. No expected-return/Kelly/leverage optimizer.
First/last half-window covariance sensitivity is descriptive. Forecast correlation
unavailable in the intent-only fixture; signal/return correlation and negative
co-loss frequency are distinct, with tail inference explicitly unavailable.
Shared XAUUSD exposure does not establish diversification. No IID errors/Sharpe.

Stage12 remains the only execution/accounting engine. Added public target API and
variable full-fill quantities: opposing intents net before orders; resize/reversal
fully closes then opens on a later quote. Pending entry changes cancel, pending
exits remain; expiry/gap failures retry at the next event. All quote-side, latency,
TTL, financing and cost arithmetic retained. Time-in-position is separate from
lot-seconds for valid variable-size time-exposure fractions. All costs/cash/marks
attribute under frozen entry-supporter shares and reconcile; counterfactual
standalone sleeves are never summed as executable equity. No margin/partial-fill
or production-risk support implied. Synthetic worked examples are in the strategy
specification; no evidence-supported trading strategy has been selected.

Executed CLI commands (`.venv\Scripts\python.exe -m xauusd_quant.cli`):
- `portfolio-plan` and `alpha-registry`:V001/V002/V003 retained, currentV003.
- `portfolio-smoke --run-id PORTFOLIO_SMOKE_V001`:16 completed operations,
  2.5006sec,338,440,192 private bytes; representative smoke before further replay.
- V002 smoke:16 completed,2.7878sec,338,169,856 private bytes.
- V003 smoke:16 completed,2.9110sec,338,219,008 private bytes, after hash repair.
- Final `portfolio-smoke --run-id PORTFOLIO_SMOKE_V004`:16 completed,
  4.7103sec,338,960,384 private bytes; final source/registry identities.
- `portfolio-evaluate --run-id PORTFOLIO_INACTIVE_V001`:1 completed,
  1.5178sec,302,276,608 private bytes. V002:.4610sec,302,706,688 bytes.
- Final `portfolio-evaluate --run-id PORTFOLIO_INACTIVE_V003`:1 completed,
  .5652sec,302,485,504 private bytes; NO_ELIGIBLE_ALPHAS, no market loader called.
Own Windows process counters, not total machine memory. Each smoke caches two
prior standalone return streams and retains all14 two-fold comparisons: no trade,
each standalone, equal, minimum variance, both equal removal-of-one descriptions,
base/adverse costs. Same prior training cache at both weight updates, no outer
outcomes fit weights. Sequential, <=24 operations, <=10,000 events/operation,
120sec/1GiB declared budgets; no full suite or historical training rerun.

Synthetic IID-increment price findings: all active policies have negative closed
net PnL. 5m/15m/equal base=-6.3712/-3.3553/-4.2864 account units; adverse
commission/slippage=-8.5712/-3.9553/-5.2864. Minimum-variance weights
.535887/.464113 produce the same fills as equal after lot rounding; no diversification
claim or chosen subset. Some open standalone cutoff marks are unknown from quote
age, not zero or filled forward. Synthetic losses cannot reject/promote a market
candidate. Conditional covariance is descriptive; portfolio discovery uncertainty
and prospective validity unassessed. Real eligible candidates: none.

Independent verification caught a new allocation-envelope bug: datetime hashes
used str() while the JSON writer used ISO format. V001/V002 metadata and failed
FINAL_VERIFICATION_V001.json preserved. Freeze now hashes exactly the common
writer representation; regression verifies read-back/reuse and mutation catches
its removal. Later source-bound replays did not change outcomes or acceptance.
An early test invocation had2 pass/4 setup errors because its explicit temp parent
was absent; creating the parent yielded6 pass. Workspace scratch was preserved
under results/alpha_portfolio/test_scratch_preserved_V001; no unrelated files moved.

Final verification commands/outcomes:
- `.venv\Scripts\python.exe -m pytest tests/test_alpha_portfolio_account.py
  tests/test_alpha_portfolio_causality.py tests/test_alpha_portfolio_framework.py
  tests/test_execution_engine.py tests/test_execution_io.py
  tests/test_execution_readiness.py tests/test_stage12_audit_repairs.py
  tests/test_final_test_isolation.py tests/test_prediction_contract.py
  tests/test_strategy_execution.py tests/test_strategy_chronology.py
  tests/test_strategy_framework.py tests/test_strategy_source.py
  tests/test_robustness_statistics.py tests/test_robustness_framework.py
  tests/test_robustness_execution.py tests/test_robustness_econometric_controls.py
  tests/test_robustness_source_identity.py -q -p no:cacheprovider
  --basetemp results/alpha_portfolio/test_scratch_V006`:237 passed/36.77sec.
  Earlier broad235 passed/63.29sec; latest Stage15-only47 passed/1.05sec with
  `--basetemp results/alpha_portfolio/test_scratch_V007` (overlapping tests).
- `.venv\Scripts\python.exe -m ruff check src tests scripts`:clean.
  `.venv\Scripts\python.exe -m mypy src`:clean,237 source files.
- `.venv\Scripts\python.exe scripts/check_portfolio_guards.py --output
  results/alpha_portfolio/GUARD_MUTATIONS_V001.json`:34/51 detected;17 child
  setups obstructed by Windows temp permissions, correctly not called passes.
  Approved outside-sandbox V002 detected51/51; V003 detected52/52, including
  serialized timestamp identity. Final V004 detected54/54, including complete
  policy contract and corrected-fit inventory. Source restored/unchanged each.
- `.venv\Scripts\python.exe scripts/verify_portfolio.py --run-id
  PORTFOLIO_SMOKE_V004 --run-id PORTFOLIO_INACTIVE_V003 --output
  results/alpha_portfolio/FINAL_VERIFICATION_V003.json`:passed.334 fills,
  162 closed trades,7,582 cash flows,40 matched cost comparisons; cash/cost/sleeve
  reconciliation, strictly subsequent fills, prior information, complete ledgers,
  frozen JSON identities, current source,123 upstream evidence hashes, actual
  tick manifest and unchanged access logs. V002 verification on smokeV003/inactiveV002
  also passed; failedV001 retained. Saved records only, no new market outcomes.

Changed22 source/config/documentation/test files (UTF-8 without BOM):
- AGENTS.md, CLAUDE.md, README.md, WORKLOG.md.
- config/alpha_portfolio.yaml.
- docs/stage15_alpha_portfolio.md, docs/stage15_strategy_specification.md.
- src/xauusd_quant/cli.py, src/xauusd_quant/execution/engine.py.
- src/xauusd_quant/alpha_portfolio/__init__.py, plan.py, registry.py, allocation.py,
  intents.py, portfolio.py, studies.py, runs.py.
- tests/test_alpha_portfolio_account.py, tests/test_alpha_portfolio_causality.py,
  tests/test_alpha_portfolio_framework.py.
- scripts/check_portfolio_guards.py, scripts/verify_portfolio.py.
Local ignored results/alpha_portfolio contains all immutable artifacts, streamed
records, specifications, source versions, failed attempts and verification.
No result/data deletion or historical conclusion overwrite. Stop after Stage15.
Final `git diff --check` clean; all22 source/config/doc/test changes remain
uncommitted, HEAD unchanged at fde1c79. Final Ruff and mypy checks remained clean.

## 2026-10-04 - Stage #15 commit and push authorization

The user separately requested `commit and push` after Stage #15 completion.
This authorizes committing and pushing the 22 Stage #15 source, configuration,
test and documentation files. Broker, deployment and Stage #16 restrictions remain.
Pre-commit inspection confirms `main`, the expected GitHub origin, no unrelated
changes and a clean `git diff --check`. Data and local research artifacts remain
ignored. The preceding targeted tests, guard audits, lint, typing and independent
record verification remain the validation evidence for the unchanged code.

## 2026-10-04 - Stage #16 authoritative offline risk engine

Prompt #16 authorizes offline implementation and bounded replay only. No broker,
credentials, MT5 session, demo/live orders, deployment, commit, push or Stage17.
Tree was clean on main at 967fdf3 before edits; no unrelated changes were present.
Read scope/working notes, predecessor documentation, portfolio/intent/allocation
contracts, shared target/order lifecycle, accounting, health and chronology guards,
actual Stage13 V002/13.5/14 result metadata and Stage15 registry/verdict/verification.
Actual registry ALPHA_REGISTRY_V003 has66 entries and zero eligibility. Stage13
V002 fits exist; earlier short negative economics and Stage13.5 forecasts do not
pass missing scientific gates. Stage14 names no Stage15 candidate. These saved
results were inspected, not reproduced. Previously inspected 2022+ stays reserved;
the four access-log hashes are unchanged. No market outcome/value loader invoked.

Implemented typed account/instrument/snapshot/health/intent/ack contracts, strict
versioned risk settings, conditional horizon-stress and explicit stop-distance
sizing, conservative gross/net/sleeve/cash/margin/rate budgets, idempotent pending
reservations, health/version/time/session gates, daily adjusted-equity loss and
persistent HWM/drawdown, explicit halts/reconciliation/rearming, checkpoints and
audit summaries. Actual config/risk.yaml leaves policy/account/instrument null:
UNCONFIGURED/BLOCKED/NO_ELIGIBLE_ALPHAS. All synthetic values are labeled; no live
values inferred from Stage12 defaults or a cent account inferred from its balance.

RiskPortfolio reuses Stage15 netting/attribution and Stage12 fills/cash/costs;
no competing accountant. Its target capability, submission reservation and actual
fill checks are mandatory. Raw strategy calls cannot submit, enlarge a permit or
restore a governed account without the boundary. Verified reductions close fully,
never cross zero; replacement is a separate later approval. Simulator cancellation
is explicitly synchronous confirmed execution, followed by account reconciliation
and at most one immediate replacement attempt. Unknown/unconfirmed orders retain
reservations; partial reports halt until terminal residual-position reconciliation.
The simulator still has one position/order and full fills; no broker partial fills.
Margin reserves entry commission and immediate spread/slippage liquidation-mark
loss as well as conservative Ask-side notional. Costs and financing remain in
the authoritative equity record. A gap can exceed every configured loss threshold.

RISK_REPLAY_PLAN_V001 frozen before new synthetic outcomes:7 hand-constructed
scenarios, baseline plus risk,14 operations, at most16 events each,120sec/1GiB,
sequential. Acceptance is software invariants and retained unresolved exposures;
no PnL-based parameter changes/search or real-candidate promotion. Configuration
identities are globally immutable as well as saved per run. State/schema/unit
contracts, audit JSONL, persistent checkpoints and copied human specification
are versioned artifacts. Restarts require reconciliation and retained severe
halts require explicit operator/reason; neither HWM nor loss is silently reset.

Actual commands/studies, using `.venv\Scripts\python.exe -m xauusd_quant.cli`:
- `risk-plan`: froze results/risk/plans/RISK_REPLAY_PLAN_V001.json.
- `risk-replay --risk-config config/risk_synthetic.yaml --run-id RISK_SMOKE_V001`:
  failed first baseline because the trial directory was not created; failed ledger
  and manifest retained. No completed result claimed. Fixed directory creation.
- Same replay with RISK_SMOKE_V002:14 completed,.803554sec,304599040 private bytes.
  V003:14,.828047sec,304762880. Final V004:14,.778314sec,303198208 private bytes,
  peak working set88936448. Measured own process, not whole-machine memory.
- `risk-readiness --run-id RISK_READINESS_V001`:UNCONFIGURED/BLOCKED, exit1,
  .337509sec,302145536 private bytes. V002:.312300sec,302272512. Final V003:
  .314472sec,302190592; precise missing field lists saved; no real-data replay.

Synthetic findings retained: agreement earns ~3.4 ledger units in both variants;
opposition/pending-net cancellation inactive. Adverse-jump baseline loses ~16.6;
risk refuses fill outside reservation. Loss-halt gap: risk realizes ~-200.6,
exceeding synthetic40 drawdown threshold; its exit costs slightly worsen marked
economics compared with holding. Invalid next feed leaves the account exposed
with unresolved flattening. Invalid health blocks even the profitable fixture
trade. None establishes improved unbiased performance, market edge or loss ceiling.

Actual validation/failures:
- Initial two test modules failed collection due test-relative imports; fixed
  to the repository's absolute fixture convention. Then43 core tests passed.
- Initial portfolio tests7 passed/2 failed: pending target cancellation needed
  confirmed reconciliation then bounded replacement before a later quote. Fixed,
  nine passed; additions cover stronger independent boundary checks.
- Framework first2 passed/2 failed from an incorrect load_config keyword. Fixed
  with the existing dataclass replacement convention; subsequent full framework
  tests pass. Scratch versions retained under ignored results/risk.
- Mutation GUARD_MUTATIONS_V001:13/15 detected, source unchanged. Two checks were
  masked by secondary protection. Added independent submission checks; repaired
  checkpoint dictionary aliasing that could mutate its frozen input. V002:15/15;
  V003:18/18; final V004:19/19, all original source bytes unchanged. Sources are
  disabled only in isolated copies and restored; not a claim based on unbroken tests.
- Final targeted command (no full suite):
  `.venv\Scripts\python.exe -m pytest tests/test_risk_decisions.py
  tests/test_risk_state.py tests/test_risk_portfolio.py tests/test_risk_framework.py
  tests/test_alpha_portfolio_account.py tests/test_alpha_portfolio_causality.py
  tests/test_alpha_portfolio_framework.py tests/test_execution_engine.py
  tests/test_execution_io.py tests/test_execution_readiness.py
  tests/test_stage12_audit_repairs.py tests/test_final_test_isolation.py
  tests/test_prediction_contract.py tests/test_strategy_execution.py
  tests/test_strategy_chronology.py -q -p no:cacheprovider
  --basetemp results/risk/test_scratch_V007`:270 passed/11.91sec.
  Previous targeted268/269 and risk-only passes are overlapping checks, not added totals.
- `.venv\Scripts\python.exe scripts/check_risk_guards.py --output
  results/risk/GUARD_MUTATIONS_V004.json`:19/19 detected; unchanged originals.
- `.venv\Scripts\python.exe scripts/verify_risk.py --replay RISK_SMOKE_V004
  --readiness RISK_READINESS_V003 --output results/risk/FINAL_VERIFICATION_V002.json`:
  passed.13 fills,34 cash flows,five closed trades,92 risk decisions; quantities,
  executable quote sides, strict timing, cash/fee/PnL/sleeve reconciliation,
  immutable JSON/source identities and unchanged reserved logs. V001 verification
  on prior source-bound V003/V002 also passed and remains on disk.
- `.venv\Scripts\ruff.exe check src tests scripts`:clean.
  `.venv\Scripts\mypy.exe src`:clean,243 source files. `git diff --check`:clean.

Changed23 source/config/test/doc files:
- AGENTS.md, CLAUDE.md, README.md, WORKLOG.md.
- config/risk.yaml, config/risk_synthetic.yaml, docs/stage16_risk_engine.md.
- src/xauusd_quant/cli.py, execution/engine.py, alpha_portfolio/portfolio.py.
- src/xauusd_quant/risk/__init__.py, contracts.py, policy.py, engine.py,
  portfolio.py, runs.py.
- tests/risk_synth.py, test_risk_decisions.py, test_risk_state.py,
  test_risk_portfolio.py, test_risk_framework.py.
- scripts/check_risk_guards.py, scripts/verify_risk.py.
Ignored results/risk retains all versions, failures, streamed accounting/audits,
plans/configuration schemas, state machines/checkpoints, readiness/comparisons,
human specification snapshots and verification. No historical artifact deleted
or overwritten. No heavy study, full test suite, actual broker/reconciliation
claim, real-data portfolio replay, deployment, commit or push. Stop after Stage16.

## 2026-10-04 - Stage #16 commit and push authorization

The user separately requested `commit and push` after Stage #16 completion.
This authorizes committing and pushing the 23 reviewed Stage #16 source,
configuration, test and documentation files. Broker, deployment and Stage #17
restrictions remain. Pre-commit inspection confirms main, the expected GitHub
origin, no unrelated changes and a clean `git diff --check`. Data and research
artifacts remain ignored. The preceding 270 targeted tests, 19 guard mutations,
lint, typing and saved-record verification validate the unchanged code.

## 2026-10-04/05 - Stage #17 read-only Exness demo integration and shadow framework

Prompt #17 explicitly authorizes read-only connectivity to a configured Exness
demo terminal, bounded capture, causal shadow operation and commit/push after
checks. This supersedes older connectivity/Git prohibitions for this scope only.
The user also requests future completed implementation stages be committed and
pushed after appropriate checks. Broker orders, deployment and automatic Stage18
remain prohibited. Initial main/898cae8452bef9499e589ae6d7576174b1c5f796 and the
expected private origin were verified; the working tree was clean.

Inspected actual Stage12-16 source, contracts, saved manifests and documentation.
Stage15 ALPHA_REGISTRY_V003 contains66 candidates, none eligible; actual Stage16
risk is RISK_UNCONFIGURED_V001 with missing policy/account/instrument terms.
Stage14 null/chronology/coverage/execution gates remain unresolved. Inventory
records37 frozen model and13 ensemble specifications and their metadata hashes;
no historical market-model binary was loaded. Previously recorded metrics were
not reproduced. Volatility-scaled future_return outputs are not silently treated
as raw returns. Historical streaming reports using stored regime/context do not
prove full Exness feature/model compatibility. Historical2022+ remains previously
inspected and reserved guards/access logs are preserved.

Implemented an optional official MT5 adapter with five fixed read-only verbs,
explicit authenticated executable/account/server/company/symbol matching and
independent vendor demo-mode verification. No login/account selection, symbol
selection, terminal-setting change or broker-order method exists. An owned worker
bounds vendor reads and cleanup without terminating the user's terminal. Native
errors and identity fields are sanitized. Official MetaQuotes API/time/account
documentation and current PyPI wheel availability were checked; only the optional
Windows MetaTrader5==5.0.6231 dependency is added, not installed here.

Tick ingestion persists ordered full-field occurrence hashes for overlapping UTC
retrievals, retaining same-time distinct/identical observations. Ambiguous overlap,
saturated timestamp batches, invalid quotes, late corrections and gaps halt.
Canonical completed bars retain [open,close) semantics; missing bars are not
fabricated. Bounded canonical return/ACF/microstructure feature windows restore;
unsupported recursive/context features remain blocked. Frozen model/ensemble
load interfaces reuse pinned artifact and specification checks. The causal binding
supports multi-frequency alphas, delayed availability, explicit diagnostic state,
portfolio netting and authoritative Stage16 risk through Stage12 local execution.
Native CLI remains capture/blocked diagnostics because actual strategy inputs
are unqualified; synthetic bindings are confined to offline fixtures.

Direct integration repair: publish the incoming quote to same-event risk checks
before target reevaluation, after prior expiry timers. Previously sparse ticks
could cancel a pending order against a stale previous quote. Current available
quotes now mark the account; strictly later fill timing and risk capability guards
are preserved. Prior artifacts/conclusions are retained, not silently revised.

Records/checkpoints bind code/configuration, occurrence cursor, bar/feature/model/
policy/diagnostic state, governed account/risk state when present and audit-prefix
hash. New-run resume requires compatible state and retains continuity/risk halts.
Storage failures, clock errors, overload and failed continuity stop processing.
Backfill never creates retrospective live actions or local fills. Broker snapshots
remain separate from synthetic accounting. Shadow fills are hypothetical and
assumed costs are not measured Exness execution. Streaming/replay comparison uses
the same recorded publication clocks with declared feature/prediction tolerances.

Actual bounded studies/checks (local results are ignored and versioned):

- `.venv\Scripts\python.exe scripts/verify_shadow.py --run-id SHADOW_SYNTHETIC_V001`:
  earlier source-bound study preserved;2.59sec,245309440 peak working-set bytes.
- Same command with `SHADOW_SYNTHETIC_V002`: final code identity
  `4b1b03de3cfbc58b657770ad3e2383af73a663c6d7e554913b276b5c236b0184`;
  synthetic adapter36 ticks/180 synthetic seconds/two1m completed bars;
  recorded replay equals monitoring bars/features/blocked actions. Separate fixed
  governed fixture26 ticks/four bars/four predictions/58 risk decisions/one local
  hypothetical entry fill; cash and sleeve attribution reconcile. The position
  remains simulated; no completed round trip is claimed. Flat and missing-health
  controls produce no fills.2.3645337sec actual wall;245497856 peak working-set,
  690302976 private,691593216 peak-commit bytes for the own process. No native
  connectivity, actual-model equality, market study or profitability claim.
- `.venv\Scripts\python.exe -m xauusd_quant.cli shadow-validate --run-id
  SHADOW_READINESS_V001`: configuration syntax valid, scientifically BLOCKED,
  NO_ELIGIBLE_ALPHAS, risk unconfigured, native package absent; saved readiness,
  frozen validation plan and metadata inventory. Public configuration has all
  five terminal identity fields null. A blank ignored config/local/shadow.yaml
  was created for local configuration; the user supplied an MQL5/Experts folder,
  which is neither terminal64.exe nor an explicit identity configuration.
- `.venv\Scripts\python.exe scripts/check_shadow_guards.py --output
  results/shadow/GUARD_MUTATIONS_V001.json`: first attempt failed; two isolated
  tests hit temporary-directory permissions, and the publication canary was masked
  by another clock guard. The failed artifact remains. Strengthened independent
  first-publication canary. V002 and final V003 each detect8/8 deliberate breaks:
  demo identity, forbidden order trap, occurrence overlap, publication timing,
  alpha eligibility, authoritative risk, reserved-period access and current-quote
  ordering. Original source bytes remain unchanged; mutations occur in isolated
  copies. V003 uses the final implementation.
- Final targeted test command:
  `.venv\Scripts\python.exe -m pytest tests/test_shadow_adapter.py
  tests/test_shadow_ingestion.py tests/test_shadow_streaming.py
  tests/test_shadow_models.py tests/test_shadow_pipeline.py
  tests/test_shadow_runs.py tests/test_shadow_worker.py
  tests/test_alpha_portfolio_account.py tests/test_alpha_portfolio_causality.py
  tests/test_alpha_portfolio_framework.py tests/test_risk_decisions.py
  tests/test_risk_framework.py tests/test_risk_portfolio.py tests/test_risk_state.py
  tests/test_execution_engine.py tests/test_execution_io.py
  tests/test_execution_readiness.py tests/test_final_test_isolation.py
  -q -p no:cacheprovider`:244 passed in18.33sec. Earlier244/20.88sec is an
  overlapping run, not another244 distinct tests. Full suite was not run.
- `.venv\Scripts\ruff.exe check src tests scripts`:clean.
  `.venv\Scripts\mypy.exe src/xauusd_quant`:clean,253 source files.
  `git diff --check`:clean. Final verification records source/study/guard hashes,
  checks and unchanged reserved-access evidence under results/shadow.
- `.venv\Scripts\python.exe results/shadow/verify_final_local.py`: first local
  helper incorrectly indexed the already-unwrapped immutable body and failed
  with KeyError before writing a verdict. Corrected the helper; final invocation
  passed and wrote FINAL_VERIFICATION_V001.json. Study/source and fixture hashes
  match, all8 guard breaks are detected, reserved logs and upstream evidence hashes
  are unchanged; local terminal configuration remains incomplete. The helper and
  verification artifact remain ignored local research records.

Changed files:

- .gitignore, AGENTS.md, CLAUDE.md, README.md, WORKLOG.md, pyproject.toml.
- config/shadow.yaml; docs/stage17_shadow.md.
- src/xauusd_quant/cli.py; alpha_portfolio/portfolio.py; risk/portfolio.py.
- src/xauusd_quant/shadow/__init__.py, config.py, adapter.py, worker.py,
  ingestion.py, bars.py, features.py, models.py, pipeline.py, runs.py.
- scripts/check_shadow_guards.py, scripts/verify_shadow.py.
- tests/shadow_synth.py, test_shadow_adapter.py, test_shadow_ingestion.py,
  test_shadow_streaming.py, test_shadow_models.py, test_shadow_pipeline.py,
  test_shadow_runs.py, test_shadow_worker.py.

No terminal initialization, broker connection, actual Exness recording or live
observation occurred. Local explicit executable/demo identity/exact symbol and
optional package are required for read-only observation. Eligibility, compatible
features/context, feed transfer, supplied risk/account/instrument/cost settings,
diagnostic health and full recorded-feed equality are separate strategy blockers.
No Stage18 readiness or evidence-supported strategy is asserted. All failed and
historical artifacts remain local. Git review includes only explicit intended
source/config/test/doc paths, excluding local identity config, ticks, logs, model
binaries and data. Commit/push is authorized for this completed framework only;
final delivery records the commit and verified remote branch. Stop after Stage17.

## 2026-10-05 - Stage #17 configured demo preflight and bounded native observation

Initial Stage17 framework was committed as9629f23d713fc00a2bba28ee35022a072dbf88d3
and pushed to main; GitHub branch hash was verified and the tree was clean.
The user then supplied terminal identity values and asked about company matching
and the next step. They had edited config/shadow.yaml. The values were moved into
ignored config/local/shadow.yaml, the supplied company identity used there, and
the tracked public template restored. No private values entered a commit or log.
On a subsequent question the on-disk local configuration was rechecked as complete;
only the public template retained null fields. No environment substitution added.

Installed only MetaTrader5==5.0.6231 with `--no-deps`; the first sandbox invocation
failed with forbidden socket access, and the explicitly approved retry succeeded.
No unrelated dependencies were upgraded. Read-only preflight used the existing
owned worker and verified exact configured terminal/account/company/symbol plus
vendor demo mode. No broker execution method was called. Preflight disconnected
after completion; successful connection does not mean a background session remains.

Actual commands, prefixed `.venv\Scripts\python.exe -m xauusd_quant.cli`:

- `shadow-preflight --shadow-config config/local/shadow.yaml --run-id
  EXNESS_PREFLIGHT_V001`:exit0, identity_verified and demo_mode_verified true.
- `shadow-capture --shadow-config config/local/shadow.yaml --run-id
  EXNESS_CAPTURE_V001`:exit0, completed configured60-second budget, measured
  capture section57.7818274sec,57 polls, zero reconnections, zero accepted ticks,
  zero fresh ticks and zero bars.57 NO_FRESH_LIVE_QUOTE and57
  EMPTY_OR_REPEATED_BATCH health records. No source coverage exists. Peak own
  working set232964096/private666083328 bytes, worker memory not measured.
  This establishes terminal connectivity and bounded shutdown only; fresh live
  data observation remains false. Market-closure cause was not established.
- `shadow-replay --shadow-config config/local/shadow.yaml --recorded
  results/shadow/runs/EXNESS_CAPTURE_V001 --run-id EXNESS_REPLAY_V001`:exit0,
  zero batches/ticks/bars. Captured audit SHA256
  abebde0cab58cb2611cbc977fd8e505441ba45e0ceae6dd053f7aec67a2ae978.
- `shadow-compare --shadow-config config/local/shadow.yaml --recorded
  results/shadow/runs/EXNESS_CAPTURE_V001 --replayed
  results/shadow/runs/EXNESS_REPLAY_V001 --run-id EXNESS_EQUALITY_V001`:exit0,
  equal empty observation tables, zero mismatches, full_model_equality and
  full_pipeline_equality false. Empty equality is not processing validation.
- `shadow-validate --shadow-config config/local/shadow.yaml --run-id
  SHADOW_TERMINAL_READINESS_V001`:exit0, no missing terminal fields, native
  package installed; strategy BLOCKED/NO_ELIGIBLE_ALPHAS/risk unconfigured.

Code identity4b1b03de3cfbc58b657770ad3e2383af73a663c6d7e554913b276b5c236b0184
matches the final synthetic validation, and no source was changed while capture
ran. No forecasts, sleeve intents, risk-approved actions, shadow fills or broker
orders were produced. Run artifacts, local configuration and checkpoints remain
ignored; no captured values are published. Existing failed/synthetic artifacts
remain intact. Native manifest and source identities, health counts and unchanged
reserved-access evidence were checked in NATIVE_VERIFICATION_V001 locally.

Only README.md, WORKLOG.md and docs/stage17_shadow.md change for this validation update.
CLI examples use fresh versioned native run IDs. `git diff --check` passed;
no code change warrants repeating the already-passed244 tests, lint or typing.
Documentation is committed/pushed under the user's Stage17 authorization and
future-stage Git preference. Next: bounded non-empty capture during updating
quotes, followed by actual recording replay equality; scientific eligibility,
feed compatibility and supplied risk/account/cost terms remain separate blockers.
No broker orders or automatic Stage18. Stop after Stage17.

## 2026-10-05 - Stage #18 demo-only execution infrastructure and blocked native execution

The user explicitly authorized implementation, bounded DEMO orders only after all
readiness/configuration gates, and commit/push. Main/origin were inspected before
editing: clean tree, HEAD dc3ed400aa7c32a6137652eb3b51046ff9a6a8dc,
origin https://github.com/FatCat1244/xauusd-quant.git. The established main workflow
is retained. No unrelated changes were present or staged. The user's preference
to commit/push completed stages is retained; it does not authorize merge/deployment,
real accounts or automatic Stage19. Scope notes supersede historical broker/Git
restrictions only for the explicitly bounded DEMO stage.

Actual audit: Stage15 ALPHA_REGISTRY_V003 has66 records/zero eligible alphas;
Stage14 has no survivor. Null, complete fold-local chronology, economic terms,
adequate coverage and an independently uninspected period remain missing. Actual
Stage16 risk.yaml is unconfigured. Stage17 native capture had zero accepted ticks
or bars and empty equality, not full model/pipeline validation. Frozen model/ensemble
metadata does not supply complete compatible new-feed artifacts or scientific
eligibility. No historical market study/performance was reproduced; no reserved
market rows were read. Selection history and previously inspected2022+ limitations
remain unchanged. Read-only metadata/access-log hashes agree with initial evidence.

Implemented: separate optional native DEMO adapter and fixed-verb bounded worker;
no shadow import/trading switch, REAL mode or account override. Exact terminal/demo
enum/account/server/company/currency/mode/permissions are verified at changing
boundaries, after submission and during recovery. MarketFOK/IOC requests validate
actual execution mode, flags/enums, lot increments/caps, tick/price and correct
position-ticket semantics. order_check retcode0 differs from order_send10009;
acceptance/completion/partial/unknown responses are not fills.

Stage16 RiskEngine remains the authority. A narrowly identified mechanical smoke
route omits predictive inputs only; all units, budgets, rounding, margin, session,
loss/drawdown and reservation controls still apply. Standard strategy health checks
remain unchanged. Risk authorization binds exact request, expiry, permitted quote
age and economic snapshot through IPC, then independently validates them natively.
Material entry-state changes abstain. Native margin/profit are checked against the
supplied reservation/conversion. Synthetic terms cannot arm native execution.
Actual strategy-native binding is deliberately blocked: independent lifecycle
infrastructure does not establish the missing portfolio/feed/scientific prerequisites.

An account-level OS single-writer lock and bounded fsynced hash-chain journal persist
intents/reservations/SUBMISSION_ATTEMPTED before send. Unknown outcomes never cause
blind resend. Orders/deals/positions/history establish ownership; magic/comments
alone cannot adopt exposure. Terminal partial quantities, immutable deal records,
actual charges, external balance flows, current net lots and balance cash reconcile.
No simulated/requested price becomes a broker fill. Existing Stage12 shadow/accounting
engine remains unchanged. The durable submission count denotes attempts (including
crash uncertainty), cumulatively across the account journal; events/intent records
retain run_id for run-specific attribution. Exactly-once execution is not claimed.

Shutdown stops entries and only permits bounded verified owned closure. Uncertain
orders are retained; pending cancellation, exchange execution, broker-stop policies,
multiple hedged positions, account credit and unsupported charge/correction deal
types are blocked, not guessed. Restart preserves risk history/halts/reservations,
requires reconciliation, and cannot rearm after a submission. A verified successful
check-only history can arm only on an explicit subsequent smoke invocation. Recovery
closing has its own bounded runtime while retaining original history/risk. Interrupts
around submission persist unresolved exposure and stop the worker. There is no
automatic rearming/migration or guarantee that a requested close succeeds.

Public templates retain null financial values. Ignored config/local/demo.yaml and
config/local/risk_demo.yaml were created only after confirming neither existed;
the former references the existing local terminal identity and a separate mechanical
smoke namespace, one entry attempt and close-owned/process-exit behavior. Quantity,
exposure, loss limits, timing and account/instrument/risk terms remain unset. No
credentials or private identifiers were printed or copied into tracked files.
The user's request to explain settings is addressed in the Stage18 configuration
guide. No synthetic budgets were reused as actual settings.

Actual checks (prefix `.venv\Scripts\python.exe`):

- `-m pytest tests/test_demo_lifecycle.py tests/test_demo_guards.py
  tests/test_demo_native.py tests/test_demo_state.py tests/test_demo_worker.py
  tests/test_demo_reconciliation.py tests/test_demo_runs.py
  tests/test_shadow_adapter.py tests/test_shadow_ingestion.py
  tests/test_shadow_streaming.py tests/test_shadow_models.py
  tests/test_shadow_pipeline.py tests/test_shadow_runs.py tests/test_shadow_worker.py
  tests/test_alpha_portfolio_account.py tests/test_alpha_portfolio_causality.py
  tests/test_alpha_portfolio_framework.py tests/test_risk_decisions.py
  tests/test_risk_framework.py tests/test_risk_portfolio.py tests/test_risk_state.py
  tests/test_execution_engine.py tests/test_execution_io.py
  tests/test_execution_readiness.py tests/test_final_test_isolation.py
  -q -p no:cacheprovider`:final357 passed in24.80seconds. This is targeted, not the
  full suite. Ordinary tests use fake vendor/broker adapters, never actual orders.
- `-m ruff check src tests scripts`:passed. `-m mypy src`:passed260 source files.
  `git diff --check`:passed before staging; staged review repeats the whitespace check.
- `scripts/check_demo_guards.py --output results/demo/DEMO_GUARDS_V002.json`:
  exit0,9/9 independent canaries detected disabled guards in isolated source copies;
  originals unchanged. Includes final demo enum, capability, order_check semantics,
  durable send ordering, material state, recovery arming, filling enum mapping,
  native state binding and reserved access. V001 aborted on a nonexistent final
  mutation anchor and exposed a masked account-change canary; preserve
  DEMO_GUARDS_V001_FAILURE.json. The corrected quote-change canary independently
  exercises revalidation. All guards restored; no broker calls in mutation tests.

Earlier targeted runs exposed same-account-mark terminal reservation release and
Windows lock-file reading defects (9 failed/74 passed); fixed with risk-sequence
mark IDs and size-based lock initialization. Two new recovery/durability tests first
failed (original deadline prevented later cleanup; fixture checkpoint key wrong),
then60 passed after correction. A precheck-restore fixture initially named the HWM
field incorrectly; corrected test subsequently passed. The guard failure and all
versioned failed/successful study records remain intact; no favorable result was
selected or existing record overwritten.

Actual bounded commands, prefix `-m xauusd_quant.cli` unless script shown:

- `demo-plan --demo-config config/local/demo.yaml --run-id DEMO_PLAN_V001`:
  immutable plan/readiness saved, exit1 because settings intentionally unconfigured.
- `demo-readiness --demo-config config/local/demo.yaml --run-id DEMO_READINESS_V001`:
  exit1/BLOCKED; lists17 missing execution fields and all missing risk/account/
  instrument schema fields. Existing terminal reference is complete.
- `scripts/verify_demo.py --run-id DEMO_SYNTHETIC_V001`:exit1 before representative
  run because the fake case directory was missing. Plan/failure retained. Corrected
  V002 completed six fixed cases; final V003 reran after final source edits.
- `scripts/verify_demo.py --run-id DEMO_SYNTHETIC_V003`:exit0; frozen six-case study
  full/partial/timeout-after-fill/accepted/None-without-fill/reject; representative
 1.3157936seconds, whole study7.8791913seconds, peak own working set230731776 and
  private662831104/peak commit664117248bytes. Sequential, no native worker, below
  declared120seconds/1GiB budget. Full/partial/late-discovery cases reconcile known
  fake entry/closure prices and fees; full synthetic cash change3.48 ledger units
  is hand-checkable arithmetic, not predictive/economic evidence. Accepted/unknown
  cases remain unresolved with reservations; both failures are retained.
- `demo-smoke --demo-config config/local/demo.yaml --run-id DEMO_BLOCKED_SMOKE_V001`:
  exit1/BLOCKED before native construction;zero submissions/fills/closures.
- `demo-strategy --run-id DEMO_BLOCKED_STRATEGY_V001`:initial exit1/unconfigured.
  `demo-strategy --demo-config config/local/demo.yaml --run-id
  DEMO_BLOCKED_STRATEGY_V002`:exit1, explicit NO_ELIGIBLE_ALPHAS, full-live equality,
  feed-transfer/native-binding and configuration blockers. No arbitrary signals.
- `demo-precheck --demo-config config/local/demo.yaml --run-id
  DEMO_BLOCKED_PRECHECK_V001`:exit1/BLOCKED; no actual order_check/order_send.
- `demo-preflight --terminal-config config/local/shadow.yaml --run-id
  DEMO_NATIVE_PREFLIGHT_V001` and V002 and V003:exit0/read-only connectivity only,
  DEMO verified, HEDGING account mode, no current positions/orders. V001/V002 quote
  snapshots fresh; V003 stale under configured freshness rule. Execution permissions
  false; V002/V003 isolate terminal_trade_allowed false, Python/account flags true.
  MT5 Algo Trading/AutoTrading must be enabled manually in the intended dedicated
  terminal before a later configured execution run; no setting/EA was changed.
  These snapshots do not establish continuous ingestion or full pipeline equality.
- `scripts/verify_demo_artifacts.py --output results/demo/FINAL_VERIFICATION_V001.json`:
  exit0; verifies frozen hashes, final synthetic/native code identity,9 detected
  mutations, unchanged upstream/reserved-access history, blocked actual execution
  and preserved synthetic unknowns. No market values read.

Native source identities:V00181721b6744f3e8d83633a13d4aea7586d11f5627a1b43495f9edc69be5c3dd5e;
V002a102ccfdf8742d91805765dc74c07afc9b35f3c0d004c75a24d1edd196ba8d85;
finalV003/syntheticV0037b33d23c30f78900d67f31c8cdbc534397e97718f29f6c3bfbf835fb52c8650c.
Workers stopped before edits/Git operations; no cache-stamped source changed during
active native/synthetic studies. Broker checks/submissions/fills/closures:all0.
No project broker position/order or uncertain submission was created. No unrelated
account activity was managed. Actual demo lifecycle and strategy trading remain
BLOCKED. Passing offline tests does not establish real-money safety or profitability.

Tracked changed files (explicit Git staging only):

- AGENTS.md, CLAUDE.md, README.md, WORKLOG.md;
- config/demo.yaml, config/risk_demo.yaml; docs/stage18_demo_execution.md;
- src/xauusd_quant/cli.py, src/xauusd_quant/risk/engine.py;
- src/xauusd_quant/demo/__init__.py, config.py, broker.py, worker.py, journal.py,
  coordinator.py, runs.py;
- scripts/check_demo_guards.py, scripts/verify_demo.py, scripts/verify_demo_artifacts.py;
- tests/demo_synth.py, tests/test_demo_lifecycle.py, test_demo_guards.py,
  test_demo_native.py, test_demo_reconciliation.py, test_demo_runs.py,
  test_demo_state.py, test_demo_worker.py.

Local ignored additions: the two unconfigured config/local templates, immutable
results/demo plans/readiness/studies/failures/native snapshots/verification records,
and synthetic runtime journals within study directories. Data, captures, private
account snapshots, model binaries, environments and journals are excluded from Git.
Final delivery records the commit and verified origin/main hash. Stop after Stage18;
Stage19 validation is separate and no real-money/deployment authorization is implied.

### Stage #18 follow-up (2026-10-05): explicit never-submitted quote-abort recovery

User supplied local demo risk limits, confirmed exclusive demo-account use and
enabled Algo Trading. Configured preflights verified exact demo identity, fresh
quotes, all execution permissions and no current positions/orders. No private
account or financial settings are included in tracked files. The local position
budget update was versioned and earlier local configurations preserved.

The first configured precheck attempt, DEMO_USER_SMOKE_PRECHECK_V001, aborted on
MATERIAL_STATE_CHANGED_REVALIDATION_REQUIRED before actual order_check/order_send.
DEMO_USER_SMOKE_RECONCILE_V001 verified flat broker state and retired the unused
reservation while preserving the halt. No actual demo submission was created.

Added scripts/run_unsubmitted_demo_smoke.py and tests/test_demo_operator_rearm.py.
The explicit operator workflow uses the existing coordinator, risk rearm and
account journal/lock; no competing execution/accounting or shadow order path.
Frozen plan DEMO_UNSUBMITTED_REARM_PLAN_V001 permits a single explicit invocation,
one entry send maximum and configured bounded owned cleanup. Full-journal prior
submission detection, quote-only halts, fresh identity/account/quote and verified
flatness gate recovery. All loss, HWM, intent/order/turnover history remains.
Runtime arming authorization is not inherited on restart. Quote/native/risk guards
remain unchanged; strategy operation still has zero eligible alphas.

Actual checks:

- Initial new tests:7 failed/7 passed because the fixture's risk kill reasons use
  a KILL: prefix. Narrow allowlist corrected without permitting other risk halts.
- Next run:3 failed/11 passed; corrected fixture run identity for a distinct
  operator invocation and expected identity exception type. Then14 passed.
- Final targeted command: `.venv\Scripts\python.exe -m pytest
  tests/test_demo_operator_rearm.py tests/test_demo_guards.py tests/test_demo_state.py
  tests/test_demo_lifecycle.py tests/test_demo_reconciliation.py tests/test_demo_native.py
  tests/test_demo_runs.py tests/test_risk_state.py -q -p no:cacheprovider`:
  **142 passed in10.11seconds**. Includes known fake full lifecycle, failed exit,
  reservation retention and isolated disabled-submission-guard detection. No native
  vendor operations occur in ordinary tests. Full suite not run.
- `.venv\Scripts\python.exe -m ruff check src tests scripts`:passed.
- `.venv\Scripts\python.exe -m mypy src scripts/run_unsubmitted_demo_smoke.py`:
  passed261 source/script files.
- Native command: `.venv\Scripts\python.exe scripts/run_unsubmitted_demo_smoke.py
  --demo-config config/local/demo.yaml --run-id DEMO_OPERATOR_SMOKE_V001
  --operator USER_REQUEST`:exit1/NO_VERIFIED_LIFECYCLE in4.577114seconds after
  another quote change. Operator rearm was audited; no guard bypass or retry.
  Actual prechecks/submissions/fills/closures all0. Final reconciliation verified
  positions/orders/reservations/unresolved intents all0 and cash change0. Quote
  halt and EXPLICIT_REARM_SMOKE_ABORTED retained; no successful demo lifecycle.

Native `src` identity unchanged:
7b33d23c30f78900d67f31c8cdbc534397e97718f29f6c3bfbf835fb52c8650c.
Operator-script hash used:
84474a9a4858915c8f0050b435c3d049d07dadc46bd9b14f011749515a6c82e3.
Worker stopped before documentation/Git. Changed tracked paths:README.md,
CLAUDE.md, WORKLOG.md, docs/stage18_demo_execution.md,
scripts/run_unsubmitted_demo_smoke.py, tests/test_demo_operator_rearm.py.
Private local configs, backup drafts, native observations, run artifacts and
account journal remain ignored. No reserved market access or Stage #19 work.

### Stage #18 follow-up (2026-10-05): bounded fresh-quote risk revalidation

User explicitly requested continuing the bounded smoke with fresh-quote risk
revalidation. Initial branch main/origin FatCat1244/xauusd-quant and working tree
were verified; there were no unrelated changes. Existing V003 local settings were
retained unchanged. No strategy qualifies; this remains mechanical SMOKE only.

Implemented opt-in --fresh-quotes under DEMO_FRESH_QUOTE_PLAN_V002; an earlier
draft plan V001 is preserved locally. Two parent and two native evaluations reuse
the Stage16 engine in isolated calculation copies. Identical approved quantity,
hard limits, expiry, native identity, account/book/capabilities and units are
required. The actual unsubmitted reservation persists a conservative allowance
within supplied budgets and margin headroom; copies retain rate/turnover history.
Actual open estimated risk retains that allowance. No execution/accounting engine
was replaced. Default exact-quote handling and shadow read-only isolation remain.

The explicit operator path admits only the documented predecessor source and
intact same-config journal with no historical submission evidence. It requires
fresh verified flat reconciliation and records code migration before explicit
risk rearm. Loss/HWM/daily baseline/order/intent history survives. Subsequent
non-quote halts and incompatible source are blocked; no generic migration or retry.

Actual commands and outcomes:

- Initial existing targeted execution tests:130 passed in9.62seconds.
- Initial new fresh-quote tests:10 failed/19 passed. Corrected checkpoint fixture
  body/checksum construction and chose a loss-cap canary that did not already fail
  the independent native-margin guard. Then29 passed; extended workflow/clock
  and worker diagnostic coverage subsequently passed.
- Final `.venv\Scripts\python.exe -m pytest tests/test_demo_fresh_quotes.py
  tests/test_demo_operator_rearm.py tests/test_demo_lifecycle.py tests/test_demo_guards.py
  tests/test_demo_native.py tests/test_demo_reconciliation.py tests/test_demo_runs.py
  tests/test_demo_state.py tests/test_demo_worker.py tests/test_risk_decisions.py
  tests/test_risk_state.py -q -p no:cacheprovider`:211 passed in12.80seconds.
  Full suite/research studies were not run. Tests cover same-quantity moving quotes,
  loss/margin/turnover/expiry/staleness/cash/permission/metadata rejection, unknown
  reservations, loss history, restart isolation, guarded migration, native rerisk,
  quote receipt timing, safe IPC diagnostics and hand-checkable fake closure/cash.
- `.venv\Scripts\python.exe -m ruff check src tests scripts`:passed.
- `.venv\Scripts\python.exe -m mypy src scripts/run_unsubmitted_demo_smoke.py
  scripts/verify_fresh_quote_artifacts.py`:passed263 source/script files.
- `scripts/check_demo_guards.py --output results/demo/guards/DEMO_FRESH_QUOTE_GUARDS_V001.json`
  failed because sandbox-denied pytest temporary folders caused fixture errors.
  The shared mutation runner's substring detector could mistake "failed" in a
  test name/error output for detection. Fixed it to require JUnit assertion
  failures and zero testcase errors. Retained V001 rather than overwriting.
  Escalated offline runs V002/V003 were allowed to access temporary folders.
  Final V003 detected all11 deliberately disabled guards, including full fresh
  risk and post-retrieval quote clock. Original source remained unchanged.
- `demo-preflight --demo-config config/local/demo.yaml --terminal-config
  config/local/shadow.yaml --run-id DEMO_FRESH_QUOTE_PREFLIGHT_V001`:verified DEMO
  identity, HEDGING, fresh quote, execution permissions, no current orders/positions.
- `.venv\Scripts\python.exe scripts/run_unsubmitted_demo_smoke.py --demo-config
  config/local/demo.yaml --run-id DEMO_FRESH_QUOTE_SMOKE_V001 --operator USER_REQUEST
  --fresh-quotes`:exit1/NO_VERIFIED_LIFECYCLE,4.595128seconds. Explicit migration
  and rearm were audited; parent fresh-risk evaluation APPROVE. The native-worker
  precheck path returned ValueError before any submission. No completed precheck
  result was recorded; the exact native boundary/API failure phase was not logged,
  so actual order_check invocation count is unknown. Broker-changing submissions,
  entry/close deals and cash change all0. Final reconciliation verified flat,
  positions/orders/reservations/unresolved intents all0. Wrapper halt retained.
  No second invocation or blind retry. No strategy/Stage19 operation.
- A subsequent read-only margin/profit calculation at the recorded quote returned
  compatible linear-CFD profit within currency precision; no check/send/fill.
  Independently reproduced an early quote-clock sample that could classify a tick
  arriving during API reads as future. Fixed clock sampling after retrieval and
  added the canary; this is not a confirmed explanation of the actual failure.
  Added controlled worker error codes/API phases and operator last-phase reporting
  after the worker stopped. These final additions have offline evidence only.
- `scripts/verify_fresh_quote_artifacts.py --demo-config config/local/demo.yaml
  --output results/demo/FINAL_FRESH_QUOTE_VERIFICATION_V001.json`:passed, status
  FAILED_NATIVE_LIFECYCLE_PRESERVED. Initial import/typing failure was corrected to
  reuse shadow.runs.read_frozen; verifier checks immutable artifacts, full journal
  hash chain, risk checkpoint, migration, fresh decision, zero submissions and
  flat/cash reconciliation without connecting to MT5. It does not promote failure.
- Read-only Windows process inspection, escalated after sandbox denied CIM access,
  confirmed no smoke/spawn/demo worker remained before final documentation/Git.

Actual native source identity:
220270ae2d4849cc636fa5b6a48b9a66e1154eedf95a19cd5581a0eeedd6e0e9.
Actual native operator script:
3e2b2f2199866232c6bc667769eac1963a9b47b63d32e5ff55a0839234361f17.
Final offline-tested source identity:
caece6af9b565f7b1450a01fecf2046b6c1929bc4e854b0a76077db82b5c2e28.
Source changes occurred only after the bounded native process stopped. The
preserved native checkpoint is not relabeled as final-source validation. Its
post-run source mismatch and unresolved non-quote halt block automatic recovery.

Changed public paths:README.md, CLAUDE.md, WORKLOG.md,
docs/stage18_demo_execution.md; scripts/check_demo_guards.py,
scripts/check_execution_guards.py, scripts/run_unsubmitted_demo_smoke.py,
scripts/verify_fresh_quote_artifacts.py; src/xauusd_quant/demo/broker.py,
coordinator.py, worker.py, revalidation.py; tests/test_demo_fresh_quotes.py,
tests/test_demo_worker.py. Private configuration/notes, journals, native snapshots,
plans, failures and verification artifacts remain ignored and local. No reserved
historical data access, financial-limit tuning, real trading or Stage19 work.

Remaining blockers:unclassified native validation failure, no verified demo
entry/closure, post-run diagnostic/clock changes not native-validated, retained
halt/source compatibility gate, zero eligible strategy alphas and missing full
live feature/model equality/feed transfer. Commit/push is authorized; no merge or
deployment is implied. Stop after Stage18.

Final artifact verification was repeated as FINAL_FRESH_QUOTE_VERIFICATION_V002
after strengthening native source/script consistency and requiring exactly one
recorded smoke intent; it again preserved FAILED_NATIVE_LIFECYCLE_PRESERVED.

Final review added a post-calculation native quote-age check; a deliberately slow
fake margin calculation verifies that no precheck occurs after the quote ages out.
A test-edit indentation error was caught by pytest collection and Ruff, corrected,
then the final targeted command passed211 tests in13.08seconds. Ruff and typing
passed. FINAL_FRESH_QUOTE_VERIFICATION_V003 verifies the final source identity above
while retaining the actual native source separately. No further broker invocation.

## 2026-10-07: Stage18 continuation, diagnostic gate and fresh-feed replay

User request: "let's do the rest" after discussing demo readiness. Continued
authorized Stage18 mechanical diagnostics and read-only observation. No Stage19,
strategy activation, financial-limit changes or unrestricted run. Initial branch
main, origin FatCat1244/xauusd-quant, clean working tree at644b3a2 were verified.
The configured account remains dedicated; private configuration stays ignored.

Reviewed actual Stage14/15 metadata, Stage16 risk state and Stage17/18 evidence.
ALPHA_REGISTRY_V003 retains66 BLOCKED entries (16 directional,50 diagnostics).
Metadata-only inventory found37 model and13 ensemble specs, without loading binaries.
Missing matched prior-only pipeline nulls, verified historical execution/account
terms, full legacy fold-local chronology, adequate chronological coverage and
independent uninspected evidence remain scientific blockers. Full live model/feature
equality/feed transfer and strategy-specific risk configuration remain unresolved.
Supplied local risk is SMOKE only; no candidate was promoted or search expanded.

Implemented an opt-in recovery of the documented unsubmitted native ValueError:
hash-verified frozen failed-run/provenance, matching settings, intact full journal,
fresh demo identity/flat reconciliation and wrapper-only halt. Audit precedes
known-source migration and explicit risk rearm. History/HWM/loss/order/turnover
records survive. Added precheck-only mode with no entry or cleanup sends. Default
exact-quote paths and shadow isolation are unchanged.

V001 plans were frozen before the native diagnostic. After its risk rejection,
V002 plans explicitly admit frozen session-only zero-approval diagnostic recovery
and require a durable successful current-source precheck before recovery smoke.
A marker alone cannot pass. The last diagnostic's approved intent and check code0
must support it. One attempted diagnostic is persisted before its check path;
failure/interruption consumes its bounded budget. Neither restart nor repeated
invocation permits a blind diagnostic/submission retry. Premature smoke stops
before connection. Native V001 failure artifacts were preserved, not relabeled.

Actual native/read-only commands and outcomes:

- `demo-preflight --demo-config config/local/demo.yaml --terminal-config
  config/local/shadow.yaml --run-id DEMO_CONTINUE_PREFLIGHT_V001`:configured demo
  identity, HEDGING, execution permissions, fresh quote and empty book passed.
- `.venv\Scripts\python.exe scripts/run_unsubmitted_demo_smoke.py --demo-config
  config/local/demo.yaml --run-id DEMO_NATIVE_DIAGNOSTIC_V001 --operator USER_REQUEST
  --fresh-quotes --recover-validation-abort --precheck-only`:exit1,4.506309 seconds.
  Audited original wrapper recovery/source migration succeeded. Risk rejected
  SESSION_CLOSED and OVERNIGHT_RESTRICTION at2026-10-06 17:21:41 UTC; the supplied
  weekday session is07:00-17:00 UTC. No order_check or order_send reached, zero
  submissions/entry deals/close deals/cash change. Final broker reconciliation
  verified flat, zero positions/orders/reservations/unresolved intents. Risk READY,
  no current halt; original halt/rearm records remain audited. Original native
  ValueError is still unexplained because this run never reached that boundary.
  No second native attempt, time fabrication, session widening or unattended wait.
- `.venv\Scripts\python.exe -m xauusd_quant.cli shadow-capture --shadow-config
  config/local/shadow.yaml --run-id EXNESS_CONTINUE_CAPTURE_V001`:59.802941 seconds,
  49 polls/no reconnects,2507 committed ticks (2278 backfill,229 fresh live).
  No halts, simulated fills or broker orders. Peak own-process working set243789824
  bytes/private694235136 bytes; worker memory excluded. Four emitted backfill bars
  (three5m/one15m), two invalid partial startup bars. No newly completed live bar.
  Four monitoring ret_1 feature rows, four blocked NO_ACTION records; no predictions.
- `shadow-replay --shadow-config config/local/shadow.yaml --recorded
  results/shadow/runs/EXNESS_CONTINUE_CAPTURE_V001 --run-id EXNESS_CONTINUE_REPLAY_V001`:
  replayed48 committed batches/2507 ticks/four bars.
- `shadow-compare --shadow-config config/local/shadow.yaml --recorded
  results/shadow/runs/EXNESS_CONTINUE_CAPTURE_V001 --replayed
  results/shadow/runs/EXNESS_CONTINUE_REPLAY_V001 --run-id EXNESS_CONTINUE_EQUALITY_V001`:
  zero mismatches/four bar,feature,blocked-action rows. Float rtol1e-5/atol1e-7,
  exact integer/clocks. Predictions0:full model/pipeline equality unavailable.
- Broker-free `scripts/verify_demo_continuation.py --demo-config config/local/demo.yaml
  --output results/demo/DEMO_CONTINUATION_VERIFICATION_V001.json`:passed,
  RISK_BLOCKED_NO_ORDERS. Repeated asV002 after adding direct before/after financial
  history checks across rearm. It validates immutable original reports, full journal,
  risk checkpoint, capture manifest and recomputed comparison, registry hashes and
  limited before/after access-log stability. No historical dataset/model/broker read.
  The original generic NO_VERIFIED_LIFECYCLE is preserved alongside its derived
  session-rejection classification. Flatness refers to run-end evidence, not a new read.

Actual native/capture/replay source identity:
caece6af9b565f7b1450a01fecf2046b6c1929bc4e854b0a76077db82b5c2e28.
Actual diagnostic operator-script hash:
8eeb66faaa1ff9847518cc4ad0afb5c99fc023a760d3ccf79d6902d81f883f69.
Later operator safeguards have offline evidence; source modules were not edited.
Windows process inspection confirmed no smoke/spawn/demo worker remained before
final documentation/Git. Fresh read-only capture is separate from strategy evidence.

Validation:

- Initial new tests:11 passed; existing recovery/fresh tests59 passed. Extension
  initially failed synthetic fixture identity/session assumptions and the isolated
  mutation's dependency binding; corrected to unique intent IDs, actual synthetic
  weekday session, connected-only cleanup and patched dependencies in the mutation.
  Then23 recovery tests passed, including hand-checkable precheck->bounded fake
  entry/closure after naturally expired counters and session rejection without check.
- Deliberately disabled full-history submission and precheck-before-smoke guards
  in isolated in-memory copies; both relevant assertion tests failed as required.
  Deployed functions stayed intact. No broker-changing integration tests ran.
- `.venv\Scripts\python.exe -m pytest tests/test_demo_native_recovery.py
  tests/test_demo_operator_rearm.py tests/test_demo_fresh_quotes.py
  tests/test_demo_lifecycle.py tests/test_demo_guards.py tests/test_demo_native.py
  tests/test_demo_reconciliation.py tests/test_demo_runs.py tests/test_demo_state.py
  tests/test_demo_worker.py tests/test_risk_decisions.py tests/test_risk_state.py
  tests/test_shadow_adapter.py tests/test_shadow_ingestion.py tests/test_shadow_pipeline.py
  tests/test_shadow_streaming.py tests/test_shadow_runs.py -q -p no:cacheprovider`:
  earlier286 passed in31.57 seconds before the last two added cases; final result
  recorded below. No full suite or research study.
- `.venv\Scripts\python.exe -m ruff check src tests scripts`:passed.
- `.venv\Scripts\python.exe -m mypy src scripts/run_unsubmitted_demo_smoke.py
  scripts/verify_fresh_quote_artifacts.py scripts/verify_demo_continuation.py`:
  passed264 source/script files. `git diff --check`:passed.

Changed public paths:README.md,CLAUDE.md,WORKLOG.md,docs/stage17_shadow.md,
docs/stage18_demo_execution.md,scripts/run_unsubmitted_demo_smoke.py,
scripts/verify_demo_continuation.py,tests/test_demo_native_recovery.py. Captures,
execution journals, plans/results and supplied local account/risk configuration
remain ignored. Commit/push completed intended paths is authorized; no merge/deploy.

Remaining bounded lifecycle step is a diagnostic in a permitted session, then
one explicitly invoked smoke only if it passes and configured approval counters
expire naturally. Next session after this attempt:7 October2026,14:00 Bangkok.
No completed broker lifecycle or evidence-supported trading strategy exists.
Strategy operation and Stage19 remain blocked. Exact Windows commands and honest
negative/partial validation are documented in docs/stage18_demo_execution.md.

Final targeted command above:288 passed in26.15 seconds. The last added cases
verify persisted diagnostic-budget exhaustion before connection and explicit
session-rejection reporting with forbidden check/send traps. Ruff and typing passed
after those changes; no subsequent code changes. Verification V002 remains the
final bounded-observation classification. No further broker activity.

## 2026-10-07: saved forecast diagnostics while the execution session is closed

User: "do the rest then". Initial working tree clean at 04ff687/main; verified
origin FatCat1244/xauusd-quant. Read AGENTS/CLAUDE and actual Stage13-18 sources,
saved fits and evidence. UTC clock was 17:58 on 6 October (Bangkok 7 October), outside
the supplied 07:00-17:00 entry window. No precheck/smoke attempt, session expansion,
financial-limit change or unattended wait. Original native ValueError remains
unresolved, with zero verified broker lifecycle and zero eligible strategy alphas.

Implemented scripts/diagnose_recorded_forecasts.py using existing Stage13 feature
construction and FrozenPipeline.predict, without fitting, model binaries, policy
intents, execution/accounting replacements or historical dataset reads. Selected
both CAUSAL_RIDGE saved F02 fits by last chronological fold (5m/15m), not performance.
Pinned fitted files, historical manifests, script/source identity and evidence-log
hashes. Four completed consecutive valid closes supply Stage13 return_1/momentum_3/
volatility_3. Loader validates numerical inputs, features, scales and training/inner
cutoffs. Bars guard availability/order, invalidity/gaps reset warm-up. Checkpoints,
chunk/restart and prefix checks stay bounded. Outputs are reconstructed forecasts
computed after capture, not actual live predictions; no predictive/economic claim.

V001 plan frozen before capture: two fixed fits, one <=300-second read-only capture,
<=30000 ticks/256 forecast bars, 3600s backfill, no search/refit. Created ignored
config/local/shadow_forecast.yaml from supplied identity plus these observation
limits; original shadow/demo/risk configs and risk journal unchanged. Native source
remained caece6af9b565f7b1450a01fecf2046b6c1929bc4e854b0a76077db82b5c2e28.

Actual commands/results:

- `.venv\Scripts\python.exe scripts/diagnose_recorded_forecasts.py prepare
  --shadow-config config/local/shadow.yaml --local-capture-config config/local/shadow_forecast.yaml`:
  froze EXNESS_FROZEN_FORECAST_DIAGNOSTIC_V001 before new observations.
- `.venv\Scripts\python.exe -m xauusd_quant.cli shadow-capture --shadow-config
  config/local/shadow_forecast.yaml --run-id EXNESS_FORECAST_CAPTURE_V001`:
  299.835640 seconds; 8502 committed ticks (7968 backfill, 534 nonbackfill, 529 fresh).
  264 polls/249 committed batches/no reconnects or continuity halts. 17 bars: 13 at 5m,
  4 at 15m; all retained backfill flags, 2 invalid partial startup bars. One 5m bar
  completed during observation with mixed history/new ticks. No broker orders or
  simulated fills. Health notices: 15 empty/repeated batches, 3 non-fresh quotes.
  Own-process peak working set 245764096 bytes/private 709836800; worker excluded.
- `shadow-replay --shadow-config config/local/shadow_forecast.yaml --recorded
  results/shadow/runs/EXNESS_FORECAST_CAPTURE_V001 --run-id EXNESS_FORECAST_REPLAY_V001`:
  249 committed batches/8502 ticks/17 bars replayed. Captured records digest
  84cb93bfff97f11db2343cd02bc8db083a1acf80f7257f1b8de9628422d90aae.
- `shadow-compare --shadow-config config/local/shadow_forecast.yaml --recorded
  results/shadow/runs/EXNESS_FORECAST_CAPTURE_V001 --replayed
  results/shadow/runs/EXNESS_FORECAST_REPLAY_V001 --run-id EXNESS_FORECAST_EQUALITY_V001`:
  all 17 bar/monitoring-feature/NO_ACTION rows matched; zero mismatches. Baseline
  produced no model predictions; full pipeline/risk equality remains unavailable.
- `scripts/diagnose_recorded_forecasts.py evaluate --plan
  results/shadow/forecast_diagnostics/plans/EXNESS_FROZEN_FORECAST_DIAGNOSTIC_V001.json
  --recorded results/shadow/runs/EXNESS_FORECAST_CAPTURE_V001 --replayed
  results/shadow/runs/EXNESS_FORECAST_REPLAY_V001 --run-id EXNESS_FORECAST_DIAGNOSTIC_V001`:
  nine reconstructed 5m forecasts matched canonical batch/incremental arithmetic,
  max prediction difference 3.3881317890172014e-21 (rtol1e-12/atol1e-14). Exact
  two-bar chunk/restart and prefix equality. 15m produced zero forecasts because
  only three valid consecutive closes were available; four required.
- The V001 software status could obscure absent 15m coverage. Preserved actual V001
  script source beside the original report. Reporting amendment V002 explicitly
  adds zero counts/per-timeframe UNTESTED and overall PARTIAL_FORECAST_DIAGNOSTICS.
  Its CLI exited 1 while displaying a relative plan path after successfully freezing
  the artifact; V002 evaluation itself passed. Preserved V002 script/report/plan.
  Fixed relative-path display, then froze V003 with explicit post-V001 reporting
  revision reason. No model, inputs, tolerance, acceptance or scientific gate changed.
  No new capture. Final command:
  `scripts/diagnose_recorded_forecasts.py amend-reporting --original-plan
  results/shadow/forecast_diagnostics/plans/EXNESS_FROZEN_FORECAST_DIAGNOSTIC_V001.json
  --plan-id EXNESS_FROZEN_FORECAST_DIAGNOSTIC_V003`, followed by the same evaluate
  command with plan V003/run EXNESS_FORECAST_DIAGNOSTIC_V003: exit 0,
  PARTIAL_FORECAST_DIAGNOSTICS, 5m 9 / 15m 0, same arithmetic;strategy_active false/orders 0.
  Model manifests/capture hashes and reserved access-log hashes verified unchanged.

Artifact feasibility audit: 37 ML specs / 13 ensembles, all 50 require unsupported live
feature services; maximum required history 7520 bars. 37 manifests exist at standard
model paths, but binaries were not opened and compatibility is not inferred from
presence. Four directional-return specs use volatility-scaled outputs; 46 others
are unsupported/unknown for this directional live binding. No inversion, feature
substitution, strategy promotion or false feed-transfer evidence was invented.

Tests/checks:

- Initial 14 new tests: 9 passed, 4 fixture errors from sandbox-denied Windows pytest
  temp access, 1 canary failure. Escalated rerun: 13 passed/1 canary failure. The
  original canary was masked by FrozenPipeline's independent availability guard.
  Changed it to test an unavailable first warm-up bar, so disabling the bar guard
  alone causes the assertion test to fail; then 14 passed. Deployed code intact.
- Added partial-coverage and end-to-end immutable-output tests plus a regression
  for relative-path CLI display: 17 new tests passed in 2.78 seconds.
- Final `.venv\Scripts\python.exe -m pytest tests/test_recorded_forecast_diagnostics.py
  tests/test_strategy_chronology.py tests/test_shadow_streaming.py
  tests/test_shadow_ingestion.py tests/test_demo_native_recovery.py -q -p no:cacheprovider`:
  72 passed in 6.16 seconds. Offline synthetic adapters only. No full suite/research grid.
- `.venv\Scripts\python.exe -m ruff check src tests scripts`: passed.
- `.venv\Scripts\python.exe -m mypy src scripts/diagnose_recorded_forecasts.py`:
  passed 262 files. Windows process inspection confirmed no owned worker/smoke process
  remained before final edits/Git. No native source modules were edited.

Changed public paths: CLAUDE.md,README.md,WORKLOG.md,docs/stage18_demo_execution.md,
docs/stage18_forecast_diagnostics.md,scripts/diagnose_recorded_forecasts.py,
tests/test_recorded_forecast_diagnostics.py. Private identity, capture/config,
checkpoints, numerical reports and diagnostic source snapshots remain ignored/local.
Commit/push of completed intended work remains authorized; no merge or deployment.

Remaining blockers: permitted-session native diagnostic and verified demo entry/
closure; 15m/full legacy live services/warm-up/unit/context compatibility; actual
live model/ensemble and portfolio/risk equality; matched prior-only pipeline nulls,
audited historical selection/execution assumptions, adequate independent evidence
coverage and strategy-specific risk configuration. This study repairs software
evidence about a small saved forecast path, not strategy eligibility. All 66 alphas
remain BLOCKED. No Stage19 or continuous strategy operation.
