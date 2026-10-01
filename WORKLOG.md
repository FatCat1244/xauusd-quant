# Work log - xauusd-quant

A running record of everything done in this repository: what was asked, what
was built and run, what came out, what broke and how it was fixed, and what is
still open. Newest stage last. Times are local (UTC+7).

The *how* lives elsewhere and is not repeated here: `README.md` (every layer,
its commands and its findings), `CLAUDE.md` (working notes, invariants,
pitfalls, measured dataset facts). This file is the *history*.

---

## At a glance (2026-10-01)

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
| Next stage | Prompt #11 complete; Prompt #12 not started (waits for you) |

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
- Descriptive research only: no signals, PnL, sizing, stops, execution, live
  trading, ML strategies or optimisation; nothing (window, band, threshold,
  wavelet family) is chosen by profitability.
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
