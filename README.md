# xauusd-quant

A quantitative research platform for **XAUUSD** (spot gold).

Twelve layers: nine descriptive, two predictive, and explicitly authorized
offline execution research. Forecast contracts remain prediction-only:

1. **Data foundation** - raw-file inspection and a byte-level source index,
   validation, conservative cleaning, partition-safe conversion to Parquet,
   independent dataset verification, tick-to-bar resampling, data-quality
   diagnostics, a DuckDB query layer, and dataset versioning end to end.
2. **Statistical research** - returns, distributions, autocorrelation,
   volatility, stationarity, intraday and session behaviour, spread, extremes.
3. **Rolling regression** - a trailing-window OLS on log mid price and the
   behaviour of its residual, always beside a detrended-random-walk control.
4. **Ornstein-Uhlenbeck research** - rolling AR(1)/OU fits of that residual,
   and whether the fitted decay resembles the realised one.
5. **Spectral research** - causal rolling FFTs of the residual, the OU
   innovation and log returns, against four null controls, with every
   hypothesis tested recorded in a research ledger.
6. **Wavelet / time-frequency research** - causal (one-sided MODWT) rolling
   wavelet features of the same series and offline scalograms clearly
   labelled non-causal, against six null controls, tested for information
   beyond the regression, OU, volatility and Fourier features.
7. **Unsupervised regime discovery** - K-Means, Gaussian mixtures and a
   Gaussian HMM (K = 2..6) on a compact causal feature set, fitted offline
   (labelled non-causal) and walk-forward (live-safe filtered probabilities),
   against volatility buckets, random labels and null controls.
8. **Feature factory and predictive feature research** - one versioned,
   causal feature matrix per timeframe from every earlier layer, a separate
   target table, and rank-IC research (decay, stability, conditioning, mutual
   information, redundancy, interactions) against circular-shift, pipeline and
   noise-feature nulls, every test in the ledger.
9. **Feature selection and dimensionality reduction** - which compact,
   low-redundancy, time-robust subset should enter supervised learning:
   selection on a development period only, evaluation on a validation period,
   a reserved test period whose outcomes are never loaded; immutable, ordered
   feature-set manifests.
10. **Supervised predictive-model research** - can baselines, linear models,
    random forests, XGBoost, LightGBM and CatBoost extract *stable
    out-of-sample* information from those feature sets? Chronological
    walk-forward with purge and embargo, calibration, nulls, every trial in
    the ledger, frozen model specifications and one evaluation on the reserved
    period. Models output probabilities and expected values only.

11. **Ensemble research** - eligibility, diversification, chronological weight
    fitting, frozen specifications and forecast contracts. Historical selection
    limitations are retained and block clean out-of-sample promotion.
12. **Offline execution and economics** - streaming Bid/Ask market-order
    simulation, a fixed separate policy, costs, accounting and readiness gates.
    No broker access or live trading.

The data set is 23 years of OANDA XAUUSD ticks (2003-05 to 2026-09,
729 million rows after cleaning).

> **Prompt #12 authorizes offline backtesting only.** The separate execution
> layer has a fixed reference policy and hypothetical cost scenarios. Scientific
> and broker-specification gates currently block economic conclusions. Forecasts
> remain probabilities and expected values; broker connectivity, demo/live trading
> and deployment remain prohibited. See [Stage #12](docs/stage12_execution.md).

---

## Table of contents

1. [Purpose](#purpose)
2. [Current scope](#current-scope)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Quick start](#quick-start)
6. [Inspecting the raw dataset](#1-inspect-the-raw-dataset)
7. [Validating it](#2-validate-it)
8. [Converting to Parquet](#3-convert-to-parquet)
9. [Dataset metadata](#4-dataset-metadata)
10. [Building bars](#5-build-bars)
11. [Diagnostics](#6-diagnose-bar-quality)
12. [Querying](#7-query-the-data)
13. [Results on the real dataset](#results-on-the-real-dataset)
14. [Statistical research layer](#statistical-research-layer)
15. [Rolling regression and residual research](#rolling-regression-and-residual-research)
16. [Ornstein-Uhlenbeck research](#ornstein-uhlenbeck-research)
17. [Spectral research](#spectral-research)
18. [Wavelet / time-frequency research](#wavelet--time-frequency-research)
19. [Unsupervised regime discovery](#unsupervised-regime-discovery)
20. [Feature factory and predictive feature research](#feature-factory-and-predictive-feature-research)
21. [Feature selection and dimensionality reduction](#feature-selection-and-dimensionality-reduction)
22. [Supervised predictive-model research](#supervised-predictive-model-research)
23. [Dataset versioning](#dataset-versioning)
24. [Directory structure](#directory-structure)
25. [Data assumptions](#data-assumptions)
26. [Design guarantees](#design-guarantees)
27. [Known limitations](#known-limitations)
28. [Non-goals](#non-goals)
29. [Development](#development)

---

## Purpose

Build a trustworthy, reproducible, scalable base for mathematical and
statistical research on XAUUSD — regression-residual mean reversion, Z-scores,
Ornstein–Uhlenbeck processes, spectral analysis, volatility and regime
detection, machine learning, and realistic bid/ask backtesting with
walk-forward validation.

None of that research is credible if the data underneath it is not. So this
stage exists to answer, with evidence rather than assumption:

- What exactly is in the raw file?
- What timezone are the timestamps in?
- Where are the gaps, and which ones are real market closures?
- What was dropped, and why?
- Can the whole pipeline be re-run and produce the same thing?

## Current scope

```
Raw XAUUSD tick CSV (34 GB, read-only)
   -> inspection       (profile the file without reading it all)
   -> source index     (byte runs per month, content hashes, ordering report)
   -> validation       (14 checks, exact counts, nothing deleted)
   -> cleaning         (conservative, audited, configurable)
   -> Parquet storage  (one validated, sorted, atomically written file per month)
   -> verification     (independent re-check of every partition and the source)
   -> dataset metadata (full-pass statistics)
   -> time bars        (1m / 5m / 15m / 30m / 1h, no look-ahead, versioned)
   -> diagnostics      (coverage, gap classification)
   -> DuckDB queries   (range-scoped, guarded against full loads)
   -> statistics       (returns, volatility, ACF, stationarity, sessions)
   -> rolling OLS      (trailing-window fit on log mid price, cached, versioned)
   -> residual study   (distribution, decay, extremes - descriptive only)
   -> OU study         (rolling AR(1)/OU fits, half-lives, controls - descriptive)
   -> spectral study   (rolling FFT of residual, OU innovation, returns vs 4 nulls)
   -> wavelet study    (causal MODWT features, offline scalograms, vs FFT and 6 nulls)
   -> regime study     (K-Means / GMM / HMM, offline vs walk-forward filtered states)
   -> feature factory  (one versioned causal matrix + a separate target table)
   -> alpha research   (rank IC, decay, stability, nulls, redundancy - statuses, not signals)
   -> feature selection (historical selection, validation, immutable manifests)
   -> supervised models (walk-forward research, frozen specs, logged final attempts)
   -> ensembles         (forecast combinations, historical chronology limitations)
   -> offline execution (fixed separate policy, streaming quotes, costs, accounting)
```

Stage #12 applies declared execution costs in an offline simulator. Current
scientific evidence and unknown broker terms prevent a tradable-edge conclusion.
No broker connectivity or demo/live trading is implemented. Stop before #13.

## Installation

Requires **Python 3.12+** (developed and tested on 3.14).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -e .                 # data pipeline only
pip install -e ".[research]"     # + the statistical and regression layers
pip install -e ".[notebooks]"    # + JupyterLab for research/
pip install -e ".[dev]"          # + pytest, ruff, mypy
```

Core dependencies: `polars`, `pyarrow`, `duckdb`, `numpy`, `PyYAML`, `rich`,
and `tzdata` on Windows (Windows ships no IANA timezone database, and the
timezone policy needs one).

The research layers add `scipy`, `statsmodels` and `matplotlib`. They are an
extra rather than a core dependency because the pipeline itself does not need
them: `xauusd_quant.research` is imported lazily, so `xq convert` and
`xq build-bars` work on a bare install.

Verify:

```bash
xq --version
xq info
```

`xq` and `python -m xauusd_quant.cli` are equivalent.

## Configuration

Everything lives in [`config/data.yaml`](config/data.yaml); logging lives in
[`config/logging.yaml`](config/logging.yaml). Relative paths resolve against
the repository root, and values support `${VAR}` / `${VAR:-default}`
environment expansion (see [`.env.example`](.env.example)).

The dataset does **not** need to live inside the repository:

```yaml
raw_data_path: "D:/market-data/xauusd/ticks.csv"   # or a directory, or a glob
processed_data_path: "D:/market-data/xauusd/parquet"
```

or per-run:

```bash
xq convert --raw-path "D:/market-data/xauusd/ticks.csv"
```

Nothing dataset-specific is hard-coded. Column names, the timestamp format,
thresholds, the cleaning policy, bar conventions and the timezone rule are all
configuration.

### Key settings

| Setting | Meaning |
|---|---|
| `timestamp_column`, `bid_column`, `ask_column`, `volume_column` | source column names; `xq detect-schema` fills these in |
| `timestamp_format` | chrono/strptime format for the raw timestamp string |
| `timezone.*` | how naive timestamps are interpreted (see [assumptions](#timezone)) |
| `cleaning.drop_*` | which validation failures remove rows; everything else is flagged and kept |
| `parquet.*` | compression, partitioning, row-group size |
| `resampling.label` / `closed` | bar timestamp convention (default: interval **open**, `[t, t+Δ)`) |
| `duckdb.max_rows_without_override` | guard against accidentally loading everything |

## Quick start

```bash
xq info                                   # what is configured, what exists
xq inspect                                # profile the raw file(s)
xq detect-schema --save                   # confirm/persist the column mapping
xq source-index                           # index months, diagnose ordering
xq convert --limit-rows 2000000           # smoke test (writes <processed>.smoke)
xq convert                                # the real run; resumes per month
xq verify-dataset                         # independent integrity + coverage check
xq metadata
xq build-bars --all                       # refuses an incomplete tick dataset
xq diagnose --all
xq build-features --all                   # rolling-regression cache, versioned
xq research-summary --all                 # statistical layer
xq regression-research --all              # rolling-regression layer
xq ou-research --all --resume             # OU layer, one process per study
xq spectral-research --all --resume       # spectral layer (1m needs --allow-heavy)
```

Each stage refuses input that an earlier stage has not completed and
verified: bars need a complete tick dataset, features need current bars.

---

### 1. Inspect the raw dataset

```bash
xq inspect
xq inspect --path "D:/other/file.csv" --blocks 80 --block-kb 2048
```

Reads only the head, the tail, and a stratified set of blocks — the 34 GB file
profiles in seconds. Reports file size, estimated row count, delimiter,
encoding, BOM, line terminator, header, per-column inferred dtypes, null rates,
min/max, timestamp and bid/ask examples, exact first/last timestamps, sampled
duplicate/ordering/crossed-quote counts, a sampled spread distribution, and
large gaps observed inside sampled blocks.

Writes `data/metadata/raw_dataset_report.json`.

Everything derived from sampling is labelled as an estimate; only the first and
last timestamps are exact (they come from the head and tail directly).

### 2. Validate it

```bash
xq validate                      # full pass over the raw input
xq validate --max-rows 5000000   # quick prefix pass
```

Runs all 14 checks and reports **exact** counts:

| | |
|---|---|
| `missing_timestamp` | empty timestamp cell |
| `unparseable_timestamp` | non-empty but does not match `timestamp_format` |
| `missing_bid_or_ask` | null quote |
| `non_finite` | NaN / ±Inf |
| `non_positive_price` | bid ≤ 0 or ask ≤ 0 |
| `price_out_of_range` | outside `validation.min_price`..`max_price` |
| `ask_below_bid` | crossed quote |
| `negative_spread` | spread < 0 |
| `extreme_spread` | spread > `validation.max_spread` |
| `negative_volume`, `invalid_volume` | impossible volumes |
| `exact_duplicate_rows` | identical consecutive rows |
| `duplicate_timestamps` | repeated timestamp |
| `non_monotonic` | timestamp goes backwards |

plus malformed-row counts, large gaps, and bounded example rows for each check.
Writes `data/metadata/validation_report_standalone.json`.

Checks that need the previous row carry state across batches, so the results do
not depend on batch size.

### 3. Convert to Parquet

```bash
xq source-index                    # index the source first (convert does it too)
xq convert                         # resumes per month
xq convert --overwrite             # rebuild beside the live dataset, then swap
xq convert --no-resume             # same: every partition, transactionally
xq convert --limit-rows 2000000    # smoke test into <processed>.smoke
```

**The converter never assumes a sorted source** (this export is not sorted -
see [the ordering finding](#what-the-data-turned-out-to-be)). Conversion runs in
two passes:

1. **Source index** (`xq source-index`, cached in `data/metadata/source_index.json`).
   One streaming pass reads the year-month of every line from fixed character
   offsets given by `timestamp_format`, and records, per month, the byte runs
   holding its lines, a blake2b hash of their content, and ordering statistics:
   backward jumps, rows behind the running maximum, and rows that land in an
   *earlier* month. Lines without a readable year-month are *unassigned*: still
   parsed later and attributed to a drop reason, never lost. The index is
   refused and rebuilt if the file's size, mtime or format changes.
2. **One partition at a time.** For each month: read exactly its byte runs,
   check them against the indexed hash, parse, run the validator over the
   *whole month* (so duplicate detection sees every row of it, however
   scattered in the file), clean, **stable-sort** by timestamp (and assert
   it), compute the content digest, write `*.parquet.partial`, re-open its
   footer to check row count and schema, then rename. A crash leaves at most
   one `.partial` file, which is deleted on the next run. Digests are
   recomputed from the files themselves by `xq verify-dataset`.

Memory is one month at a time: peak ~3.5 GB on the busiest 9.4M-row months.

Output layout, with the manifest inside the dataset directory:

```
data/parquet/_manifest.json
             year=2003/month=05/xauusd-ticks-2003-05.parquet
             ...
             year=2026/month=09/xauusd-ticks-2026-09.parquet
```

`_manifest.json` records, per partition: status, rows, content digest, the
validation and cleaning counts, the source-run hash it was built from and the
schema version; and for the dataset: the raw-file fingerprint, a
`tick_fingerprint` of the settings that change content (not operational ones
such as block sizes) and the resulting `dataset_version`.

**Resume** is per partition: a month is reused only if its manifest entry is
complete and validated, its source lines (hash and count), tick settings and
schema version are unchanged, and its file exists with the recorded size, row
count and schema. Everything else is rebuilt, and the reason is logged.

**Overwrite is transactional.** `--overwrite` / `--no-resume` build a complete
new dataset in `<processed>.building`, validate it, and only then swap it into
place (`--keep-previous` keeps the old one as `<processed>.previous`). An
interrupted rebuild never touches the live dataset, and a half-finished swap is
repaired on the next run. `--limit-rows` writes only to `<processed>.smoke`, so
a smoke test can never be mistaken for the canonical dataset. A disk-space
preflight refuses to start a build that cannot fit (`--skip-space-check` to
override).

Reports written: `conversion_manifest.json` (a copy of the manifest),
`validation_report.json`, `cleaning_report.json`, `source_ordering_report.json`.

### 3b. Verify the dataset

```bash
xq verify-dataset            # full check, ~4 minutes on 729M rows
xq verify-dataset --quick    # skip digest recomputation and derived columns
```

An independent re-check that does not trust the converter: every partition's
content digest recomputed and compared, files on disk against the manifest
(unknown files, missing files), partition purity, sortedness, duplicates,
crossed quotes, invalid prices, `mid`/`spread` recomputed, the dataset version
recomputed, and the source reconciliation `source lines = rows written + rows
dropped + malformed`. Months the source lacks are reported as missing
*history*, not as failures. Gaps over an hour are classified per year.

Writes `data/metadata/dataset_validation.json` and `tick_coverage.csv`.

### 4. Dataset metadata

```bash
xq metadata
```

Computes, over the **entire** Parquet dataset via DuckDB: total rows, start/end
timestamps, trading days, calendar days, partitions and files, storage size and
compression ratio, price min/max, spread mean/median/min/max and percentiles,
volume totals, per-column null counts, duplicate-timestamp count, and the bar
timeframes present.

Validation and cleaning counts are folded in from the conversion pass (the only
stage that saw the raw rows), and the `coverage` block states exactly what each
figure covers — including whether quantiles are exact or t-digest
approximations, and whether the validation counts came from a full raw pass or a
resumed/truncated one.

Writes `data/metadata/dataset_metadata.json`.

### 5. Build bars

```bash
xq build-bars --timeframe 5m
xq build-bars --all
xq build-bars --all --keep-existing    # skip timeframes already current
```

Bars are built only from a **complete** tick dataset (`--allow-incomplete`
exists for development and labels the result). A rebuild is transactional,
like conversion, and each timeframe's `_manifest.json` records the tick
`dataset_version` it was built from and its own `bar_dataset_version`, so bars
from one tick version can never be read against another.

Per bar: `timestamp`, `open`, `high`, `low`, `close`, `tick_count`, `volume`,
`mean_spread`, `median_spread`, `min_spread`, `max_spread`, `first_bid`,
`first_ask`, `last_bid`, `last_ask`, `first_tick_timestamp`,
`last_tick_timestamp`.

OHLC comes from **mid** = `(bid + ask) / 2` by default (`resampling.price_source`).

**Timestamp convention — default `label: open`.** A bar stamped `10:05:00` on
the 5m grid covers ticks in `[10:05:00, 10:10:00)`. With `closed: left`, a tick
landing exactly on `10:10:00.000` belongs to the *next* bar. No bar can contain
information that did not exist by its own close. Set `resampling.label: close`
to stamp bars with their interval end instead; membership is unaffected.

Intervals with no ticks produce **no bar**. Nothing is filled or forward-filled.

Output:

```
data/bars/1m/_manifest.json
data/bars/1m/year=2003/xauusd-bars-1m-2003.parquet
data/bars/5m/year=2003/...
```

Bar files carry their conventions in Parquet key-value metadata
(`xq_bar_label`, `xq_bar_interval`, `xq_price_source`, `xq_time_basis`,
`xq_timezone_description`), so a file is self-describing even out of context.

### 6. Diagnose bar quality

```bash
xq diagnose --all
xq diagnose --timeframe 1m
```

Per timeframe: bar count, first/last timestamp, expected vs. missing grid slots,
coverage ratio, gap count, mean/median/min ticks per bar, unusually low
tick-count bars, spread distribution, zero-range bars, and bars per year.

Gaps are classified against the configured session schedule as `weekend`,
`daily_break`, or `holiday_or_unknown`. Only the last category needs
investigation, and the largest such gaps are listed explicitly. Missing bars are
reported, never filled.

Writes `data/metadata/bar_diagnostics.json`.

### 7. Query the data

```bash
xq query --start 2024-05-01 --end 2024-05-02 --limit 10
xq query -t 1h --start 2024-05-01 --end 2024-06-01
xq query --sql "SELECT date_trunc('month', timestamp) m, count(*) FROM bars_1h GROUP BY 1 ORDER BY 1"
```

From Python:

```python
from xauusd_quant import load_config
from xauusd_quant.data import DataStore

config = load_config()
with DataStore(config) as store:
    ticks = store.load_ticks("2024-05-01", "2024-05-02")
    bars  = store.load_bars("5m", "2024-01-01", "2024-02-01")

    # Ranges too large to materialise:
    for chunk in store.iter_ticks("2024-01-01", "2024-02-01", chunk="1d"):
        ...
    lazy = store.scan_ticks("2024-01-01", "2025-01-01")   # polars LazyFrame
```

`start` and `end` are **required** on every loader and the range is half-open
(`start <= timestamp < end`), so consecutive calls tile without overlap. A query
that would materialise more than `duckdb.max_rows_without_override` rows raises
`TooManyRowsError` unless you pass `allow_large=True`. Generated SQL constrains
the Hive `year`/`month` columns so DuckDB prunes whole files.

---

## Results on the real dataset

Export: `2026.9.18XAUUSD_oanda-TICK-No Session.csv`, **33.75 GiB**
(36,233,955,746 bytes), one file, never modified. Every figure below is a full
pass over it, reproducible with the commands in [Quick start](#quick-start).

| | |
|---|---|
| Source lines | 731,928,177 (0 malformed, 0 without a readable timestamp) |
| Rows after cleaning | **729,244,369** in **281** monthly partitions, every month present |
| Span | 2003-05-05 03:01:03.421 → 2026-09-18 23:59:59.079 (broker clock) |
| Dropped | 2,683,808 exact duplicate rows - all verbatim weekly repeats (below); nothing else |
| Parquet on disk | 9.13 GiB (3.7× smaller than the CSV) |
| Dataset version | `ticks-2e173ef8e61bd240` |
| `verify-dataset` | **passed** in 4 min: 0 backward steps, 0 rows in the wrong partition, 0 kept duplicates, 0 crossed quotes, 0 invalid prices, all 281 content digests match, source reconciliation balanced |

### What the data turned out to be

**The source is not time-sorted, by design of the export.** From 2021-01-04
onward, every trading week's opening 54 minutes to 2 hours appears *twice*: the
file reaches some point into the week, then jumps back to the week's open and
repeats it verbatim. `xq source-index` finds **301 backward jumps**, all at a
week open (2021-01-04 → 2026-09-07), with **2,683,507 rows** behind the running
maximum. Every one of them is a byte-identical repeat of a row already seen;
none is new data and none lands in an earlier month. Each repeat ends on a row
that ties the running maximum, so exact-duplicate drops come to 2,683,808 =
late rows + 1 per splice. The converter needs no sorted input, so the disorder
is diagnosed (`source_ordering_report.json`, with example context lines around
each jump) rather than assumed away.

Otherwise the file is clean: no malformed rows, no unparseable timestamps, no
missing or non-positive prices, no crossed quotes. The source `Spread` column
equals `Ask - Bid` on every row checked.

**History is uneven.** Ticks per year, from the verified dataset:

| Years | Ticks per year | Note |
|---|---|---|
| 2003 (May-Dec) - 2006 | 2.8M - 12.8M | seconds-grained, frequent intraday gaps |
| 2007 - 2009 | 3.8M - 5.4M | the sparsest full years |
| 2010 - 2015 | 13.1M - 25.8M | |
| 2016 - 2022 | 35.9M - 55.8M | |
| 2023 - 2026 (to Sep) | 36.4M - 71.4M | ~50 ms throttled modern feed |

So 1-minute statistics pooled over 2003-2026 mix very different data regimes.
Every research layer reports its results by year and by era for that reason.

### Bars and coverage

| Timeframe | Bars | Size | Coverage | Weekend | Holiday wknd | Daily break | Unexplained |
|---|---|---|---|---|---|---|---|
| 1m | 7,937,007 | 450 MiB | 64.6 % | 1,192 | 17 | 27,470 | 174,377 |
| 5m | 1,663,606 | 108 MiB | 67.7 % | 1,192 | 17 | 5,131 | 12,770 |
| 15m | 561,817 | 40 MiB | 68.5 % | 1,192 | 17 | 3,361 | 1,728 |
| 30m | 282,256 | 22 MiB | 68.9 % | 1,192 | 17 | 3,034 | 246 |
| 1h | 141,569 | 13 MiB | 69.1 % | 1,192 | 17 | 2,442 | 91 |

Weekend closures total 1,209 (1,192 regular, 17 extended by a holiday) over
~1,218 week boundaries; the few others fall in the irregular early feed and are
counted as unexplained. The `unexplained` column is the one that matters, and it falls steeply with the timeframe: at 1h it is 91
gaps, mostly real holiday closures the configured schedule cannot express; at
1m it is 174,377 because individual minutes in the thin 2003-2010 feed (and in
modern overnight liquidity) genuinely contain no ticks. None of these gaps
are filled.

**Earlier results.** Statistical-layer and regression-layer results computed
while only 19 months had been converted are archived under
`results/_archive/partial_dev_sample_2003-05_2004-11/` and labelled `PARTIAL DEVELOPMENT SAMPLE — NOT FULL RESEARCH RESULT`. They
describe 2003-05 to 2004-11 only and are not results about XAUUSD.

---

## Statistical research layer

Built on top of the data foundation, this layer *characterises* the series. It
answers "what are the statistical properties of XAUUSD?" and stops there.

```bash
xq research-summary --timeframe 5m
xq research-summary --all
xq research-summary -t 5m --start 2024-01-01 --end 2025-01-01
xq research-summary --all --no-plots        # tables only
```

Each run writes `results/statistical_research/<timeframe>/` containing
`summary.json`, `distribution`, `autocorrelation`, `ljung_box`, `volatility`,
`volatility_regimes`, `stationarity`, `stationarity_segments`, `intraday`,
`weekday`, `session`, `spread_analysis`, `spread_by_hour`, `activity`,
`extreme_return_analysis`, `stability` and `rolling_statistics` — each as both
Parquet and CSV — plus `plots/`. Running `--all` also writes
`multi_timeframe_comparison.{csv,parquet,json}` at the top level.

Configuration lives in [`config/research.yaml`](config/research.yaml).
Notebooks in [`research/`](research/) are deliberately thin: they load, call
and display, and contain no calculations.

> This layer's results on the full 2003-2026 dataset have not been regenerated
> yet; the tables from the 19-month development sample are archived (see
> [Results](#results-on-the-real-dataset)). Rerun `xq research-summary --all`.

### What the layer guarantees

**Features never see the future.** A value at `t` uses only data at or before
`t`. Forward returns exist solely as *outcomes* for conditional analysis, are
always named `fwd_*`, and `assert_no_forward_columns()` enforces their absence
where features are assembled. `rolling.include_current_bar` controls whether
`t` itself is in the window, and which was used is recorded in every report.

**Significance is always reported next to effect size.** At these sample sizes
a Ljung-Box test rejects for a mean |rho| of 0.002. Every ACF row carries a
plain-language effect-size label, and every Ljung-Box result carries the mean
and max |rho| beside its p-value.

**ADF and KPSS are never collapsed into one flag.** Their nulls are opposites,
so the verdict names the combination — including `conflicting` and
`inconclusive` — rather than reducing it to a boolean.

**Annualisation is opt-in.** XAUUSD trades ~23 h/day, 5 days/week, so the
equity convention of 252 × 6.5 h is simply wrong. Volatility is per-bar unless
`volatility.annualization.enabled` is set, and the assumption is then written
into the report.

**Session boundaries are configuration, not fact**, and are not DST-adjusted.
The reports say so.

**Nothing here is a trading signal.** No transaction cost, spread or slippage
is applied anywhere in this layer.

---

## Rolling regression and residual research

The third layer fits a rolling ordinary-least-squares line to a trailing window
of log mid price and studies what is left over. It asks "how does the detrended
series behave?" and stops there.

```bash
xq regression-research --timeframe 1h --window 128
xq regression-research --timeframe 5m --all-windows
xq regression-research --all                        # every timeframe x window
xq regression-research -t 5m -w 128 --start 2024-01-01 --end 2025-01-01
xq regression-research --all --no-plots             # tables only
```

### The model

For each bar `t`, OLS is fitted to the window **ending at `t`**:

```
y_i = alpha_t + beta_t * x_i + eps_i        i = 0 .. N-1,   x_i = i
```

`y` is the log of the mid close, and the residual recorded at `t` is that of
the window's last point. Because `x` is a fixed integer ramp, `x_bar = (N-1)/2`
and `Sxx = N(N^2-1)/12` are constants, so each fit costs two dot products; the
series is centred before any sum of squares is formed, which is what keeps
`R^2` from losing precision on log prices near 6.0.

Default windows are `[32, 64, 128, 256, 512]`, set in
[`config/regression.yaml`](config/regression.yaml).

### Two Z-scores, kept apart

| Column | Divides by | Question it answers |
|---|---|---|
| `residual_zscore_fit` | the standard error of its own fit | how unusual is the last point relative to the line drawn through it |
| `residual_zscore_rolling` | the rolling std of the residual series | how unusual is this residual relative to recent residuals |

They routinely disagree. Both are computed, they live in separate columns, and
a test asserts they are never silently interchanged.

### What the layer guarantees

**No look-ahead, checked column by column.** Appending future bars must not
change a single historical value. `tests/test_regression_no_leakage.py` verifies
this to *exact* equality - not `approx` - for every feature column, at several
windows, including a test that drops a 5x price spike into the future and
confirms nothing before it moves. Warm-up bars stay null rather than being
back-filled.

**Every reversion claim is fitted against a control.** Detrending any series -
including one with no mean reversion at all - produces something bounded and
zero-centred, so `b < 0` in `delta eps = a + b * eps` appears mechanically. The
same regression is therefore also fitted to a random walk detrended the same
way, and both coefficients are reported. The gap between them is the finding;
`b` alone is not.

**"Moves toward zero" is reported against its arithmetic baseline.** For an
independent series with no predictability whatsoever, conditioning on
`|Z| > 2` makes `P(|eps_{t+h}| < |eps_t|)` land near **0.97**, purely because a
large draw is usually followed by a smaller one. The tests pin this down
explicitly so the headline probability cannot be read as evidence on its own.

**Censoring is explicit.** Starts whose residual never crosses zero inside the
search horizon are counted, not dropped, and the median crossing time is
reported as null when more than half the starts are censored. Starts too close
to the end of the sample to have a full horizon are excluded entirely, so
position alone never masquerades as censoring.

**Stationarity gets the same control treatment.** ADF and KPSS keep their
opposite nulls, and each verdict is paired with the verdict the detrended
random walk produced. `control_matches = true` means the result says nothing
about XAUUSD.

**No window is selected.** Every configured window is reported side by side and
none is marked best. Choosing the window with the most attractive statistic
would be full-sample parameter selection, which is out of scope here.

**A rolling regression is not cointegration.** There is one asset and no
economic relationship under test. A price series detrended against its own
recent history is a single series, not a spread.

**Nothing here is a trading signal.** No entry, exit, size, stop or cost exists
in this layer. Its decay coefficient is not an Ornstein-Uhlenbeck speed; the
[OU layer](#ornstein-uhlenbeck-research) estimates those, separately.

### Output

Each model writes `results/regression_research/<timeframe>/window_<N>/`:

| Group | Tables |
|---|---|
| fit | `slope_statistics`, `r_squared_statistics`, `slope_by_year` |
| residual | `residual_distribution`, `residual_acf`, `residual_ljung_box` |
| stationarity | `stationarity.json`, `stationarity_segments` |
| decay | `decay_analysis` |
| extremes | `zero_crossing`, `extreme_persistence`, `adverse_excursion` |
| conditioning | `volatility_conditioning`, `slope_conditioning`, `r_squared_conditioning`, `intraday`, `yearly_stability` |

Each table is written as both Parquet and CSV, alongside `summary.json` and
`plots/`. `summary.json` carries a `model_summary` block - a compact,
machine-readable digest of the model - plus full provenance: config
fingerprints, git state, package versions and the resolved timezone.

Fitted features are cached by `xq build-features` under
`data/features/regression/timeframe=<tf>/`: the window-independent columns once
in `base.parquet`, each window's own columns in `window=<N>/features.parquet`,
and a `_manifest.json` with the bar version and settings fingerprint they were
computed from. A loaded frame is bit-identical to a fresh computation (tested),
a cache from other bars or settings is refused (`StaleFeaturesError`), and a
caller can load only the columns it needs.

> As with the statistical layer, full-dataset results for this layer have not
> been regenerated yet. Rerun `xq regression-research --all`.

Running more than one model also writes `model_comparison.{csv,parquet,json}`
and `model_comparison_matrix.csv` at the top level.

Notebooks [`07_rolling_regression`](research/07_rolling_regression.ipynb),
[`08_residual_stationarity`](research/08_residual_stationarity.ipynb),
[`09_residual_decay`](research/09_residual_decay.ipynb) and
[`10_residual_extremes`](research/10_residual_extremes.ipynb) load, call and
display. Like the earlier ones they contain no calculations.

---

## Ornstein-Uhlenbeck research

The fourth layer asks whether the regression residual behaves like an
Ornstein-Uhlenbeck process, how fast its fitted reversion is, how stable that
estimate is, and - above all - whether the realised decay resembles the fitted
one. It describes; it does not trade.

```bash
xq ou-research --timeframe 5m --regression-window 128 --ou-window 256
xq ou-research --timeframe 5m --all-windows        # every N x M
xq ou-research --all --resume                      # the whole grid
xq ou-research --compare                           # comparison tables only
```

Configuration: [`config/ou.yaml`](config/ou.yaml). The grid is five timeframes
× regression windows N ∈ {32, 64, 128, 256, 512} × OU windows
M ∈ {64, 128, 256, 512}: 100 studies.

### The model

`X_t` is the rolling-regression residual at window N (log-price units). Over a
trailing window of M residuals, OLS fits the exact discretisation of an OU
process, an AR(1):

```
X_{t+1} = a + b X_t + eta_t

theta = -ln(b) / dt          mu = a / (1 - b)          half-life = ln 2 / theta
sigma = sqrt( Var(eta) * 2 theta / (1 - exp(-2 theta dt)) )
stationary std = sigma / sqrt(2 theta)                  dt = 1 bar
```

The mapping is only defined for `0 < b < 1`, and nothing is forced:

| State | Condition | Treatment |
|---|---|---|
| `valid` | 0 < b < 1, half-life ≤ 10,000 bars | mapped |
| `near_unit_root` | 0 < b < 1, half-life above the cap | counted, **censored** - never averaged |
| `unit_root` | \|b - 1\| ≤ 1e-9 | counted, not mapped |
| `explosive` | b > 1 | counted, not mapped |
| `non_positive` | b ≤ 0 | counted separately; `-ln b` is never applied |
| `degenerate` | flat window, zero regressor variance | counted |
| `insufficient_data` | warm-up, too few pairs | null |

Rolling fits are causal (the window ends at `t`), computed with block-shifted
prefix sums in O(n), and checked against statsmodels OLS window by window.
Appending future bars changes no past estimate, to exact equality. The static
whole-sample fit carries Newey-West standard errors, a confidence interval for
`mu` (estimated, never assumed zero) and the Dickey-Fuller statistic.

The OU Z-score `(X_t - mu_t) / stationary std_t` is kept beside, never in place
of, the regression layer's `residual_zscore_rolling`; they are compared, not
substituted.

### Every number beside two references

A windowed AR(1) fit has a downward (Dickey-Fuller / Kendall) bias, so **a
random walk also reports a finite "half-life"**, and it grows with M. So every
statistic is computed three times, by the same code:

| Source | What it is | What it shows |
|---|---|---|
| `residual` | the XAUUSD regression residual | the finding, if any |
| `control_random_walk` | a Gaussian random walk, same bar count, step std = the data's return std, detrended by the same rolling regression | what detrending plus windowed estimation produce *with no reversion at all* |
| `reference_ou` | an exact OU with the residual's own whole-sample θ, μ, σ, same length, estimated the same way | what estimation noise alone does to a *true* OU |

A residual indistinguishable from `control_random_walk` says nothing about
XAUUSD. The tell of a random walk is scaling: its median half-life grows
roughly in proportion to M, while a genuine OU's converges. A third control,
`control_shuffled_returns`, is registered and off by default; adding another
(block bootstrap, volatility-matched walk) is one function in
`CONTROL_BUILDERS`.

### What is measured

| Group | Tables |
|---|---|
| fit | `static_fit`, `parameter_distribution`, `b_classification`, `equilibrium` |
| half-life | `half_life` (censoring stated), `half_life_stability`, `rolling_half_life_profile`, `expected_decay` |
| model vs reality | `forecast_evaluation` (h = 1-50; skill against persistence and against the equilibrium), `extreme_deviations` (\|Z\| > 1, 2, 3; all starts and episode starts), `realized_decay_comparison`, `first_passage` (75 / 50 / 25 % of the deviation, and crossing) |
| innovations | `innovation_diagnostics` (moments, tails, normality, ACF of η, \|η\| and η², Ljung-Box, McLeod-Li, ARCH-LM), `innovation_acf` |
| stability | `yearly_stability`, `quarterly_stability`, `era_stability` (three equal-length eras), `recent_versus_history` in `summary.json` |
| conditioning | `volatility_conditioning`, `trend_conditioning` (five slope buckets), `r2_conditioning`, `intraday_conditioning` (hour and session) |

Each study writes `results/ou_research/<tf>/regression_<N>/ou_<M>/` with those
tables as Parquet and CSV, `summary.json` (a machine-readable `model_summary`
plus provenance and dataset lineage), `extreme_events.parquet`, `plots/`, and
for the representative pair (N = 128, M = 256) the per-bar
`parameters.parquet`. The top level holds `ou_comparison.{csv,parquet,json}`,
`timeframe_comparison.csv` and `window_matrix.csv`. **No column ranks
anything**: the question is whether a finding survives neighbouring settings,
not where it is largest.

Notebooks [`11_ou_estimation`](research/11_ou_estimation.ipynb),
[`12_ou_half_life`](research/12_ou_half_life.ipynb),
[`13_ou_stability`](research/13_ou_stability.ipynb) and
[`14_ou_diagnostics`](research/14_ou_diagnostics.ipynb) load and display.

### Running it

Every study runs in a fresh child process, so nothing one study leaves in the
allocator accumulates over a grid; one failing study is logged and the rest
continue. `--resume` reuses a study only if its `summary.json` records the
same raw file, tick, bar and feature versions, the same four configuration
fingerprints and the current `report_version`. Measured per study: ~7 s and
0.6 GB at 1h, ~22 s and 0.7 GB at 15m, ~65 s and 1.25 GB at 5m, 3.5-5.5 min
and 4.4 GB at 1m.

### Findings on the full 2003-2026 history

All 100 studies (5 timeframes x N x M), dataset `ticks-2e173ef8e61bd240`. The
representative pair is N = 128, M = 256; `results/ou_research/window_matrix.csv`
holds the rest.

**The residual's OU behaviour is what detrending a random walk produces.**

| Timeframe | Median half-life, bars (real / random-walk control / exact OU) | Wall clock | Static fit: half-life real / control, Dickey-Fuller real / control |
|---|---|---|---|
| 1h | 20.1 / 19.6 / 15.8 | 20 h | 23.3 / 22.8, -45.9 / -46.4 |
| 30m | 19.7 / 19.6 / 15.9 | 9.8 h | 23.3 / 22.6, -64.8 / -65.7 |
| 15m | 17.5 / 19.7 / 14.6 | 4.4 h | 21.0 / 22.8, -96.3 / -92.5 |
| 5m | 16.2 / 20.2 / 14.3 | 81 min | 19.9 / 23.2, -170 / -158 |
| 1m | 15.9 / 20.1 / 11.8 | 16 min | 15.2 / 23.2, -425 / -344 |

- 95-100 % of rolling windows give a valid OU fit - and so do 95.5-100 % of
  the random-walk control's. The unit-root rejection is as strong on the
  control; it comes from detrending.
- The half-life is set by the regression window: about 0.18 N bars at every
  timeframe (static fit), the same number of *bars* whatever the bar size, and
  it grows with the OU window exactly as the control's does.
- At 1m-15m the residual reverts faster than the control, but that is the
  2003-2007 feed: lag-1 autocorrelation of 1-minute returns is -0.40 in
  2003-2005 and about 0 from 2008; yearly median half-lives there fall to
  1.7 (1m) and 5.4 (5m) bars and recover to 16-20 from 2008.
- Realised half-decay after |Z| > 2 matches the fitted half-life (median
  ratio 1.0-1.2) - as it does on the control. The OU forecast beats "no
  change" at 5 bars by at most 0.02 of MSE, less than on the control.
- Half-lives are as unstable between disjoint windows as pure estimation
  noise, and vary from year to year no more than a constant-parameter process
  does (2008+). Equal-length eras differ only where the early feed does.
- Trend and R^2 dependence are mechanical (the control shows them). At
  30m-1h, reversion is ~15 % slower in the highest volatility quartile while
  the control's is faster - the one conditional effect beyond the mechanics.
- The equilibrium is zero within its confidence interval everywhere.

**What is specific to XAUUSD is in the innovations**: kurtosis 24-34 (4-sigma
events 110-130 times the Gaussian rate), strong volatility clustering (|eta|
lag-1 autocorrelation 0.28-0.40), and a daily cycle in their magnitude (|eta|
autocorrelation at exactly one day 0.20-0.27 at every timeframe) - none of
which the controls show. Negative excursions revert slightly more often but
overshoot further first (90th-percentile adverse move 2.1 vs 1.5 stationary
standard deviations at 1h).

---

## Spectral research

The fifth layer asks whether frequency-domain structure remains in XAUUSD
after local trend (the regression) and first-order mean reversion (the OU
fit) are removed - and, above all, whether any of it survives null controls.
It never looks at raw price, whose FFT is dominated by non-stationarity.

```bash
xq spectral-research --timeframe 5m --source regression_residual --fft-window 256
xq spectral-research --timeframe 5m --source ou_innovation --all-windows
xq spectral-research --all                     # 5m-1h; 1m needs --allow-heavy
xq spectral-research --compare | --benchmark
```

Configuration: [`config/spectral.yaml`](config/spectral.yaml). Inputs are the
residual of the N = 128 regression, the causal one-step innovation of the
M = 256 rolling OU fit, and log returns; windows N = 64 ... 1024 bars.

### The engine

For each bar `t`, the window `x[t-N+1 .. t]` - never anything later - is
checked, mean-removed, Hann-tapered and transformed with a real FFT
(`features/spectral.py`). Conventions are fixed in one place:

| Quantity | Definition |
|---|---|
| frequency | cycles per bar, `k/N`; period `N/k` bars and seconds; nothing shorter than 2 bars (Nyquist) is representable |
| power | raw FFT power `|X_k|^2` of the tapered window (DC excluded, Nyquist optional) |
| power share | `P_k / sum P` - not a PSD; a periodogram PSD in physical units exists separately (`fft_analysis.power_spectral_density`) |
| component | a local maximum of the spectrum, ranked by power (a bin beside a stronger one is its Hann leakage) |
| amplitude | `2|X_k| / sum(w)`, the cosine amplitude corrected for the taper's gain |
| phase | the component's phase at the window's **last** bar; null when it holds under 2 % of the power |
| entropy, flatness | `-sum p ln p / ln K`; geometric over arithmetic mean power, in log space |
| centroid, bands | power-weighted mean frequency; shares of power in configurable normalised-frequency bands |
| concentration | top-1 / top-3 / top-5 share of raw bins |

A window is skipped (null, never filled) when any value is missing or its
bars span unscheduled gaps of more than 5 % of its slots; weekends and the
daily break are market closures on the trading-time axis, as in every other
layer. Estimates sit on the bin grid: a cycle off the grid is reported at the
nearest bin with some amplitude lost and a biased phase - the positive
controls show that case explicitly rather than hiding it.

### Every number beside four nulls

| Null | Built from | What it shows |
|---|---|---|
| `white_noise` | Gaussian noise per series, same length and scale | what spectra of pure noise look like, peaks included |
| `random_walk` | a Gaussian walk through the same rolling regression and OU fit | the spectral shape that rolling detrending creates by itself |
| `shuffled_returns` | the real returns in random order, same pipeline | the real return distribution with no serial dependence |
| `block_bootstrap` | the real returns resampled in 256-bar blocks, same pipeline | short-range dependence and volatility clustering, no long-range structure |

Every distributional metric is compared with each null as a difference of
medians in units of the null's inter-quartile range; the verdict rule
(|effect| <= 0.25 counts as no difference) was fixed before any real result.

### What is measured

| Group | Tables |
|---|---|
| shape | `spectral_distribution`, `entropy_summary`, `power_bands`, `mean_spectrum`, `null_control_comparison` |
| dominant period | `dominant_period_distribution`, `dominant_period_stability` (switching, runs, persistence at lags up to 2N), `frequency_persistence` (top-K overlap with 10 % frequency tolerance) |
| time | `yearly_stability`, `quarterly_stability`, `monthly_stability`, `era_stability`, `spectral_changes` |
| regimes | `conditioning` (volatility quartile, trend quintile, OU speed tercile + invalid), `intraday` (hour, session) |
| phase | `phase_analysis` (8 circular sectors vs later changes), `phase_consistency` (disjoint windows, circular statistics), `phase_projection` |
| model vs reality | `reconstruction_metrics` (in-window, exact by Parseval), `extrapolation_metrics` (components carried forward, against last-value and window-mean baselines) |
| information | `residual_outcomes` (entropy / concentration buckets vs residual decay after |Z| > 2), `ic_analysis` + `ic_by_period` (IC and rank IC by year, quarter, volatility), `feature_correlation`, `feature_redundancy` |

Per-bar features for later work are written for the representative window
only, as Parquet partitioned by timeframe, series, window and year under
`data/features/spectral/` (join key: `timestamp`), with a manifest recording
the dataset versions and engine fingerprint. Every tested hypothesis
(`SPEC-H-001` ... `SPEC-H-010`, one row per timeframe, series, window,
feature, target and horizon) is upserted into `results/research_ledger.parquet`
with its verdict and dataset version - failures included.

### What the layer guarantees

**No look-ahead**: appending bars changes no earlier feature, to exact
equality, whatever the chunking (`tests/test_spectral_no_leakage.py`).
**Positive controls**: known sinusoids, alone and in pairs, are recovered in
frequency, amplitude and phase; two close frequencies separate only when the
window allows it (`tests/test_fft.py`). **An integrity gate**: a study refuses
to run if the ticks, bars and feature cache do not form one dataset version.
**No selection**: no window, band or threshold is chosen by any outcome, and
nothing here is a signal.

`--compare` (and every `--all`) also writes the multiple-testing record across
all studies on disk: `hypothesis_test_counts.csv` (every hypothesis, how often
it was tested, every verdict) and `ic_chance_rates.csv`. The latter matters
because "an IC above the largest null IC" happens by chance: each null's own
count against the other three sources (the real one included) is the rate
chance produces, and the real count is read against those.

### Findings on the full 2003-2026 history

75 studies, dataset `ticks-2e173ef8e61bd240`: 5m, 15m, 30m and 1h x residual,
OU innovation and log returns x N = 64 ... 1024 (60), plus `abs_ou_innovation`
(|eta|) at 15m-1h (15). 14,795 hypothesis tests are in the ledger. The 1m
study (N = 256) was stopped for memory before it finished, so no 1m result is
reported; the 5m-1h results are consistent with each other.

**Nothing in the signed series survives the nulls.** Medians at N = 256,
ranges over 5m-1h:

| Series | Spectral entropy: real / pipeline nulls / white noise | Top-3 power share: real / pipeline nulls | Dominant period |
|---|---|---|---|
| regression residual | 0.41-0.43 / 0.40-0.44 / 0.92 | 0.73-0.76 / 0.73-0.77 | 128-256 bars at every timeframe, nulls the same |
| OU innovation | 0.915-0.917 / 0.913-0.918 / 0.915 | 0.10 / 0.10-0.11 | ~4 bars, as for noise |
| log return | 0.915-0.917 / 0.914-0.919 / 0.915 | 0.10 / 0.10-0.11 | 2-4 bars, as for noise |

- All 660 distributional comparisons for these three series (11 metrics x 60
  studies) are within a null. The residual's apparent concentration is the
  rolling regression's: it matches the detrended random walk and shuffled
  returns to |effect| <= 0.37 null IQRs and the block bootstrap to <= 0.16.
  Returns and innovations differ from the random walk and shuffled nulls by up
  to 0.5 (the early feed's negative autocorrelation and volatility
  clustering), and the bootstrap, which keeps both, matches them to <= 0.22.
- Every dominant period is fixed in *bars*, not in time: the residual's sits
  at 64-171 bars (the window, then the 128-bar regression's edge: 5-14 h at
  5m, 64-171 h at 1h); the innovation's low-frequency bump at 114-146 bars
  (N = 1024) is the rolling pipeline's and appears in every pipeline null.
- Persistence, phase and extrapolation are the nulls' (N = 256): the same
  dominant bin recurs a window later in 39-40 % of residual windows (nulls
  35-40 %) and 1-2 % for returns and innovations; the dominant component
  carried 5 bars forward correlates with the future at r <= 0.006 (returns,
  innovations) and 0.05-0.07 (residual; nulls 0.05-0.07). Three components
  explain 74-77 % of a residual window, as for the random walk; carried
  forward, their MSE is 3.3-12 times that of "no change" (h = 5 and 1) and
  0.04-0.06 below the window mean's at 5 bars (nulls 0.03-0.05). For returns
  and innovations they are 10 % worse than the mean at every horizon -
  exactly as for the nulls.
- 10,800 IC tests on these series: 23 exceed the largest IC of the other
  sources, where each null manages 6-34 on its own. The largest |rank IC|
  (0.28, residual phase vs later residual change) is mechanical: the random
  walk reaches it too.
- Year-to-year and quarter-to-quarter variation (2008+) is the nulls'; the
  2003-2007 feed raises the residual's entropy and lowers the returns' at
  5m-15m. Dependence on OU speed is mechanical (random walk: entropy 0.50 fast
  vs 0.34 slow; real 0.48-0.52 vs 0.34-0.35); on volatility and trend it is
  <= 0.004 for returns and innovations. One small effect beyond the random
  walk: at 30m-1h the residual spectrum concentrates in high volatility (30m
  top-3 share 0.747 -> 0.776 from the lowest to the highest quartile) - the
  direction of the OU layer's slower reversion there. (Conditioning tables
  carry the random-walk control only.)
- The a-priori rules that compare with the random walk alone (SPEC-H-001,
  -004, -008, -009) flagged 25, 18, 7 and 3 of 75 studies. Against the
  shuffled and bootstrapped returns most of those vanish: residual decay by
  entropy quartile is reproduced by the bootstrap (e.g. 15m N = 128: real
  0.169, bootstrap 0.174), and phase-sector spreads sit inside the nulls'
  range. The ledger keeps the verdicts as recorded.
- Redundancy (median of the max |Spearman| with existing features): the
  residual's shape metrics 0.52-0.76, mostly with its rolling lag-1
  autocorrelation; the returns' and innovations' centroid and high-band share
  0.66-0.74; their entropy, flatness and top-k only 0.08-0.16, but 0.87-0.97
  with each other. Phase sin/cos is distinct (<= 0.07) - and carries nothing.

**The one structure beyond the nulls is the daily volatility cycle, in
|eta|.** At 15m-1h the dominant period of |eta| is one trading day (23.3 h -
the same *time* at every timeframe, unlike every period above) in 23-73 % of
windows at N >= 256, against < 1 % for white noise, the random walk and
shuffled returns (6-35 % for the block bootstrap, whose 256-bar blocks contain
whole days). It recurs across disjoint windows (dominant bin kept 25-61 %;
bootstrap 14-19 %, other nulls ~1 %), with a stable phase at 15m-30m, and it
is the only source of IC beyond chance: 46 of 2,700 tests (nulls 0-3), mostly
the daily component's phase against later absolute residual reduction
(|rank IC| <= 0.115) - a clock.
It is the intraday volatility seasonality the OU layer found in the
innovations (|eta| autocorrelation at one day 0.20-0.27); it says when
volatility changes, not which way price goes.

---

## Wavelet / time-frequency research

The sixth layer asks *when* timescales become strong or weak, how long they
persist, and whether that time-varying structure carries information beyond
the regression, OU, volatility and Fourier features - read, as always,
against null controls.

```bash
xq wavelet-research --controls                 # positive controls + boundary (padding) study
xq wavelet-research --timeframe 5m --source ou_innovation --window 512
xq wavelet-research --timeframe 5m --all-windows
xq wavelet-research --all [--dry-run]          # 5m-1h; 1m needs --allow-heavy
xq wavelet-research --compare | --benchmark
```

Configuration: [`config/wavelet.yaml`](config/wavelet.yaml). The inputs are
the Prompt #5 series (residual of the N = 128 regression, innovation of the
M = 256 OU fit, log returns); rolling windows N = 128, 256, 512, 1024 bars.

### Causal features and offline pictures, kept apart

| | Causal features | Offline analysis |
|---|---|---|
| transform | one-sided MODWT (`db4`), trailing window only | continuous transform (analytic Morlet, Mexican hat) of a whole slice |
| uses bars after t | never - verified bit for bit, at random cut-offs, and from a 2N live buffer | yes - centred on every point |
| stored | `data/features/wavelet/`, schema-checked, `live_safe` per column | never; labelled `NON-CAUSAL — VISUALIZATION ONLY` |
| used for | features, IC, models A-D, stability, conditioning | scalograms, ridges, positive controls |

**The engine** (`features/wavelet.py`, `wavelet_causal.py`). PyWavelets has no
one-sided MODWT (`pywt.swt` is circular, so its newest coefficients wrap
around to the start), so the level-j filters are built from `pywt.Wavelet`
and applied by direct convolution: the coefficient at bar t reads bars
t-L_j+1 .. t and nothing later, with no padding at the right edge. Its price
is a delay of about half a filter length, which is recorded. For a window of
N bars the energy of each band is the mean square of the coefficients whose
whole filter lies inside the window - the boundary-free MODWT wavelet-variance
estimator - so no padding is needed at either end. The boundary study
(`--controls`) measures the alternative: the newest coefficient of a windowed
DWT depends on its padding mode by a factor of e^0.6 to e^13 in energy, while
the causal estimate has the smallest error and no padding dependence.

| Quantity | Definition |
|---|---|
| bands | detail band j (periods ~2^j-2^(j+1) bars, nominal 1.4 x 2^j for `db4`) and the approximation band; J = 4, 5, 6, 7 for N = 128 ... 1024 |
| physical period | every band's nominal period in seconds; fast / slow at 4 h, high / mid / low at 1 h and 8 h - groups empty at a timeframe are undefined, not zero |
| energy share, entropy | p_j = E_j / sum E; H = -sum p ln p / ln(J+1) |
| dominant band | the band whose share most exceeds its exact white-noise share (band j of an orthogonal MODWT holds 2^-j of white noise, so the raw argmax is always the finest band) |
| concentration | top-1 / top-3 band shares; bands above the uniform share |
| drift, change | ln centroid period vs N/4 bars earlier; ln total energy vs the previous window |
| persistence | run of the dominant band (capped at N); P(same band h bars later) |
| latest state | local energies over the newest 2^j coefficients: local entropy, local fast/slow, burst Z against the band's own earlier values inside the window |

A window is valid when complete, with unscheduled gaps within 5 % of its
slots (the trading-time axis of Prompt #5), and with non-zero energy;
constant input yields no entropy and no dominant scale rather than a
misleading one.

### Nulls, hypotheses and incremental information

Six null series, all from the Prompt #5 builders and seeds (so the random
walk, shuffled and 256-block bootstrap are the identical series): white noise,
the detrended random walk, shuffled returns and block-bootstrapped returns in
blocks of 64, 256 and 1024 bars. Thirteen hypotheses (`WAVE-H-001` ...
`WAVE-H-013`) are written to `results/wavelet_research/hypotheses_registered.json`
before any result, and every verdict goes to the research ledger. Every rule
compares the real series with *every* pipeline null, not with the random
walk alone.

Incremental information (Steps 31-32) uses nested research models on
chronological, expanding training sets (test blocks 2011-14, 2015-18, 2019-22,
2023-26, embargo of the longest horizon): **A** statistical, **B** + OU,
**C** + Fourier (same window), **D** + wavelet - ridge OLS for continuous
outcomes, logistic for residual / OU-Z normalisation after |Z| > 2. The same
models run on the random walk and the bootstrap, so an increment is read
against what chance and overfitting give.

`scripts/wavelet_baseline_robustness.py` is a **post-hoc** check, chosen after
the registered results were seen and recorded as such (`WAVE-P-001`, never in
`hypotheses_registered.json`): the same models, rows and folds with A and B
widened by ln realised volatility and ln mean |innovation| over 64, 256 and
1024 bars and by hour-of-day indicators, on the real series and on the nulls
alike. Each timeframe is written to `post_hoc/parts/` as it finishes;
`--resume` continues a stopped run.

### What is measured

| Group | Tables (per timeframe / series / window) |
|---|---|
| energy and shape | `energy_summary`, `entropy_summary`, `metric_distribution`, `null_controls` |
| persistence and change | `scale_persistence`, `run_summary`, `drift_summary`, `burst_summary` |
| time | `yearly_stability`, `quarterly_stability`, `era_stability` |
| regimes | `volatility_conditioning`, `trend_conditioning`, `ou_conditioning` (fast / medium / slow / near-unit-root / invalid), `intraday` |
| outcomes | `extreme_conditioning` (|Z| > 1, 2, 3; OU extremes by wavelet and by OU state), `ic_analysis`, `ic_by_period` |
| beyond FFT | `fft_comparison`, `fft_determinism` (wavelet feature predicted from FFT features, out of sample), `dyadic_comparison`, `incremental_information`, `incremental_summary`, `incremental_by_feature` |
| candidates | `feature_correlation`, `feature_redundancy` (clusters at |rho| >= 0.7), `feature_quality` (Step 48), `family_sensitivity` (haar, sym4) |

Per timeframe and series: `offline_ridges` and scalograms of five slices
chosen by volatility and era. Across studies (`--compare`):
`wavelet_comparison`, `cross_timeframe_bands` (band shares at their physical
periods), `hypothesis_test_counts`, `ic_chance_rates`.

### Findings on the full 2003-2026 history

48 studies, dataset `ticks-2e173ef8e61bd240`: 5m, 15m, 30m and 1h x residual,
OU innovation and log returns x N = 128, 256, 512, 1024. The ledger holds
22,696 registered wavelet tests, 186 rows recorded as *untested* (a statistic
undefined there - no slow band at 5m for N <= 256) and 189 post-hoc tests. 1m
was not run (heavy, and its Prompt #5 FFT feature set lacks log returns).

**Apart from the daily volatility cycle, nothing in the time-varying scale
structure survives the nulls.** Medians at N = 512, ranges over 5m-1h:

| Series | Wavelet entropy: real / pipeline nulls / white noise | Top-3 band share: real / pipeline nulls / white noise | Whitened dominant band |
|---|---|---|---|
| regression residual | 0.73-0.75 / 0.73-0.76 / 0.67-0.68 | 0.83-0.85 / 0.82-0.85 / 0.89 | the approximation band (256 bars, the window) at every timeframe, nulls the same |
| OU innovation | 0.68-0.69 / 0.66-0.70 / 0.67 | 0.875-0.88 / 0.87-0.89 / 0.89 | d6 (90 bars), nulls the same |
| log return | 0.65-0.66 / 0.62-0.67 / 0.67-0.68 | 0.89-0.90 / 0.89-0.91 / 0.89 | 22-45 bars, nulls the same |

- All 741 testable distributional comparisons (entropy, top-k, centroid,
  fast/slow, dominant excess, local entropy, burst Z; 48 studies) are within
  a null. The residual is far from white noise but matches the random walk and
  shuffled returns to <= 0.36 null IQRs and the block bootstraps to <= 0.15;
  returns and innovations differ from the random walk and shuffled nulls by up
  to 0.55 (the early feed's quote noise, volatility clustering) and from the
  1024-block bootstrap, which keeps both, by <= 0.17.
- Energy by band follows the white-noise halving for returns and innovations
  (d1 0.51-0.52, d2 0.25, ...). The only departure, +0.012-0.018 in d1 against
  the random walk, sits in the finest band at *every* timeframe (0.23 h at 5m,
  2.8 h at 1h): fixed in bars, the bars' small negative lag-1 autocorrelation,
  reproduced by the bootstrap. Every dominant band is fixed in bars too.
- Persistence is the nulls': the dominant band recurs one window later (lag N)
  in 54-56 % of residual windows (nulls 52-56 %), 29-30 % for innovations
  (26-31 %) and 18-21 % for returns (16-23 %); returns at 15m-1h sit above
  every null by < 0.02, which passes the registered margin once in 48
  (WAVE-H-005). Drift "continuation" is -0.5, i.e. none. Bursts (a band at
  Z > 3 against its own history) occur in 0.6-1.2 % of windows, as in the
  block bootstraps (0.7-1.3 %) and ten times white noise: volatility
  clustering, not time-frequency events (WAVE-H-010: 0 of 48). Offline CWT
  ridges last no longer than the nulls' (WAVE-H-013: 2 of 24).
- Time and regimes: yearly variation (2008+) is the nulls' except a marginal
  excess for returns and innovations at 5m-15m (SD 0.006-0.008 vs <= 0.006);
  quarterly variation is the block bootstrap's. The 2003-2007 feed lowers the
  returns' entropy (5m: 0.636 vs 0.653-0.657 later). For returns and
  innovations entropy falls from the lowest to the highest volatility
  quartile by 0.006-0.023, as for shuffled returns (0.015-0.020), and trend
  moves it by <= 0.006; the residual's ranges (up to 0.031) stay below the
  block bootstrap's (0.018-0.048). OU-speed dependence is mechanical
  (reproduced by the shuffled null). WAVE-H-008/-009: 0 of 48.
- The one structure beyond every null is **intraday**: across sessions the
  fast/slow energy ratio moves 11-19 times as much as in the nulls at 5m-30m
  (5m returns: 0.62 vs <= 0.035), ~4 times at 1h - the daily volatility cycle
  the spectral layer found in |eta|, seen as energy moving to fine scales in
  active sessions.
- Outcomes: after |Z| > 2 the wavelet quartiles spread the probability that
  the residual has shrunk 10 bars later by 0.02-0.04, as for shuffled and
  bootstrapped returns (WAVE-H-001: 1 of 48); after |Z_OU| > 2 they spread it
  by 0.013-0.023, against 0.058 for the |Z_OU| band and 0.074 for the
  half-life tercile - the OU state differentiates more.
- IC: 19,980 defined tests on the real series. For direction (future return,
  future residual change) the median |rank IC| is 0.003, the 99th percentile
  0.023, the maximum 0.034. Volatility-type targets reach 0.30 (burst Z vs
  future |eta| at 5m, N = 256 - identical for all three series: it is
  volatility). None exceeds the largest IC of the other sources (WAVE-H-006:
  0 of 19,980; the random walk and shuffled nulls also 0; the block
  bootstrap 800, its block joins being volatility jumps a trailing energy
  predicts). The fast/slow change (WAVE-H-002) beats its nulls in 26 of 48
  studies with a sign that flips with the window (5m returns: +0.12 at
  N = 256, -0.12 at N = 512).
- Wavelets vs Fourier: a window's band shares equal its FFT octave shares on
  average (per-window Spearman 0.65-0.75 for returns and innovations,
  0.48-0.62 for the residual); whole-window shape features are half-determined
  by the FFT features (out-of-sample R^2 0.43-0.55 for returns and
  innovations, 0.17-0.30 for the residual), the latest-state ones hardly at
  all (median R^2 <= 0.012 for returns and innovations), so all 666 WAVE-H-011
  verdicts read "distinct from FFT". Distinct is not informative: the
  latest-state features resemble 5- and 20-bar realised volatility
  (|Spearman| 0.31-0.49), entropy and centroid the rolling lag-1
  autocorrelation (0.61-0.73); WAVE-H-012: 260 distinct, 360 partially
  redundant, 46 redundant. db4 and sym4 agree (Spearman 0.85-0.98 on returns
  and innovations); the dominant band does not survive a change of family
  (-0.07 to 0.35).

**The registered "increment" is volatility over longer horizons and the time
of day.** Model D beats C beyond both incremental nulls in 229 of 1,008 tests
(WAVE-H-007) - never for future returns or residual changes (D - C < 0
everywhere; C's R^2 for returns is <= 0), mostly for future volatility (93 of
144) and future |eta| (84 of 144), median +0.011 R^2 at 20 bars, at most
+0.05. The post-hoc check (WAVE-P-001) at N = 512, 5m-1h, the three
volatility-type targets (108 real cases):

| Baseline A/B | Cases beyond every null | Median D - C | Largest D - C |
|---|---|---|---|
| registered | 52 | +0.0012 | +0.041 |
| + hour of day | 43 | +0.0007 | +0.012 |
| + ln RV, ln mean \|eta\| over 64, 256, 1024 bars | 9 | -0.0002 | +0.0034 |
| + both | 1 | -0.0003 | +0.0016 |

The survivor adds 0.0004 R^2 (1h residual, future |residual| reduction, 5
bars). Meanwhile the baseline itself explains far more (15m, 20 bars: C's R^2
for future volatility 0.31 -> 0.46): the wavelet block had been standing in
for volatility over its own window and for the daily cycle. WAVE-H-003 (run
length) passes 11 of 48 with gains <= 0.0003 R^2 - its rule set no minimum
effect size. The script's run over every window was stopped for memory; the
ledger's WAVE-P-001 rows cover 1h, N = 512 (a scratch run of the same code
produced the 5m-30m rows of the table and matches the script to 4e-16 where
they overlap).

**The engine does what it should.** A localised cycle is found 77 bars after
it starts (rolling FFT 261, offline CWT 17); a chirp is tracked by the causal
centroid (r = 0.995, slope 0.92) and the offline ridge (r = 0.999); a
fast burst raises local fast energy 24-fold while the slow bands keep 90 % of
the window. The causal energy estimate has median |log error| 0.20 (0.79 just
after a volatility step) and no padding dependence, against 0.64-10.1 for a
windowed DWT depending on its padding. It runs at 0.5-0.8 M bars/s, 1.0-3.1 ms
per live update; a 2048-bar CWT takes 7 ms.

Conclusion: wavelets describe the series faithfully and cheaply, and every
description is either reproduced by a null or explained by volatility and the
time of day. No wavelet feature is a candidate beyond what trailing
volatility at several horizons and the hour already give.

---

## Unsupervised regime discovery

The seventh layer asks whether XAUUSD repeatedly enters statistically
distinct states that can be identified *from information available at the
time* - and whether those states are more than volatility buckets, the time
of day, or the smoothness of trailing-window features. A regime is a latent
state S_t in {0..K-1} shaping the distribution of the features X_t; the layer
estimates P(S_t = k | X_<=t) and never names a state before its profile says
what it is.

```bash
xq regime-research --timeframe 1h --model hmm --states 3        # one study, all stages
xq regime-research --timeframe 15m --model gmm --components 4
xq regime-walk-forward --timeframe 5m --model hmm --states 4    # causal features only
xq regime-research --all --dry-run                              # the plan and a rough runtime
xq regime-research --controls                                   # synthetic controls
xq regime-research --compare                                    # cross-timeframe tables
powershell -File scripts\regime_grid.ps1 -Timeframes 1h,30m,15m # the resumable grid
powershell -File scripts\regime_grid.ps1 -Timeframes 5m -Heavy
```

Configuration: [`config/regimes.yaml`](config/regimes.yaml). Models: K-Means
(geometric baseline), Gaussian mixtures (full and diagonal covariance) and a
Gaussian hidden Markov model, K = 2..6 - a deliberately limited family, all
implemented in NumPy/SciPy (`src/xauusd_quant/regimes/`): scikit-learn and
hmmlearn are not dependencies, and hmmlearn's public API offers only smoothed
(future-aware) probabilities.

### Offline and causal, kept apart

| | Causal / live-safe | Offline / non-causal |
|---|---|---|
| fitted on | bars before each quarterly (or monthly) refit, expanding from 2003 or the last 5 years | the whole 2003-2026 sample |
| scaler | frozen with the model, from its training bars | full sample |
| state estimate | HMM **forward filter** P(S_t \| X_<=t), GMM posterior, nearest K-Means centre | smoothed P(S_t \| X_1:T), Viterbi path |
| after t | never used - bit for bit, through the whole walk-forward, at many cut-offs | used |
| stored | `data/features/regime/` (schema refuses `smooth*`, `viterbi*`, `offline*`); every refit an immutable model (`data/regime_models/HMM_5M_K4_00017.json`) | results tables only, labelled `OFFLINE / NON-CAUSAL RESEARCH ONLY` |

**Walk-forward** (`regimes/causal_inference.py`): per period, train on the
past only, freeze the robust scaler (median / IQR of the training bars), fit
(warm-started from the previous period, a fresh initialisation every 8th
refit, best training likelihood wins), align the states to the previous
period's (Hungarian matching on Bhattacharyya distance, which sees means and
covariances and is invariant to the scaler), then run the frozen model over a
2000-bar burn-in and the period. Per bar: state probabilities, most likely
state, confidence, entropy, next-state probabilities `P(S_t+1 = j | X_<=t) =
sum_i p_i A_ij`, the leave probability, regime age (bars since the current
state began, from past labels) and the feature coverage.

**The HMM engine** (`regimes/hmm.py`). The scaled forward and backward
recursions are linear, so the sequence is cut into fixed 512-bar blocks
anchored at its first bar; per block the product of the step operators is
accumulated with row renormalisation, a short pass over the blocks gives each
block's incoming distribution, and the within-block recursions run for every
block at once. It equals the textbook recursion (tested against it and
against brute-force enumeration of every path), runs a 1.66M-bar forward
pass in ~1 s, and - because every matrix product has a fixed shape - a bar's
filtered probabilities are identical to the last bit whether or not later
bars exist.

### The feature space

Thirteen causal features, one per phenomenon, each with its reason in
`regimes/dataset.py` (`FEATURE_SPECS`) and `<tf>/feature_manifest.json`:
20-bar return in volatility units; ln 20-bar realised volatility, its rank
among the last 20 trading days, the rolling lag-1 ACF of |r|; regression slope
/ volatility and R^2 (N = 128); ln OU half-life (M = 256); FFT entropy and
ln(high / low band power) of log returns (the stored Prompt #5 set, N = 256);
wavelet entropy and ln(fast / slow energy) (the stored Prompt #6 set, N =
512); spread and tick-activity ranks among the last 20 trading days. Left out,
with reasons (`EXCLUDED_FEATURES`): the single-bar return (tail-driven,
flickering components), residual and OU Z-scores (fast oscillators, and the
conditioning variable of the mean-reversion tables - circular), OU theta
(= ln 2 / half-life), the OU validity flag (binary: a zero-variance Gaussian
coordinate), and every Prompt #5/#6 near-duplicate (wavelet centroid period
vs entropy 0.995, and so on). No input pair reaches |Spearman| 0.9.

The integrity gate refuses to fit unless the ticks are complete, the bars
were built from them, the regression features are current, and the stored
FFT and wavelet feature sets carry the current tick, bar and engine
versions - and a fresh computation of the last 3000 bars reproduces them.
Missing features are never filled with zero: a bar is scored on its observed
features by the exact Gaussian marginal (`preprocessing.missing_policy`), if
at least 75 % of them and ln RV are present; models are fitted on complete
bars.

### Baselines, nulls and hypotheses

Every model is read against: **one state** (K = 1); **volatility buckets**
(K quantile buckets of ln RV, causal edges from all earlier bars; as a
density, a mixture whose components are the buckets); **random labels** (a
Markov chain with the model's own empirical transition matrix, independent
of the data). HMM persistence is read against row-shuffled features (every
temporal link gone), circular blocks of 1 and 5 trading days of feature rows,
and the Prompt #5/#6 pipeline nulls (random walk, shuffled returns, 1024-bar
block bootstrap) pushed through the same regression, OU, FFT and wavelet
engines. Synthetic controls (`--controls`) check the machinery: a known
3-state HMM, one regime (Gaussian and Student-t), and a continuously drifting
volatility.

Thirteen hypotheses (`REG-H-001` ... `REG-H-013`) are registered in
`results/regime_research/hypotheses_registered.json` before evaluation and
every verdict goes to the research ledger. The number of states is never
chosen by an outcome: the registered K rule (REG-H-004) takes the largest K
whose every step from K = 1 improves the chronological hold-out likelihood by
more than 0.005 nats per bar, with no degenerate flag and a median walk-forward
refit agreement (ARI) of at least 0.5. Incremental information (Steps 58-59)
uses nested ridge / logistic models on the Prompt #6 chronological folds:
model **A** holds every continuous feature - volatility over 5 to 1024 bars,
hour of day, regression, OU, FFT, wavelet, spread and activity - and A+soft,
A+hard, A+volatility-buckets and A+random-states are compared with it.

### What is measured

| Group | Tables (per timeframe / model / K) |
|---|---|
| model selection (offline) | `<tf>/offline/model_selection` (BIC, AIC, hold-out likelihood, silhouette, balance, separation, degeneracy), `baselines_holdout`, `scaling_comparison_*` |
| states | `state_profiles`, `standardized_state_medians`, `feature_contribution` (eta^2, mutual information), `summary.json` descriptions |
| dynamics | `transition_matrix`, `state_durations` (observed vs 1/(1 - A_ii)), `calibration`, `entropy_analysis` |
| time | `yearly_frequency`, `quarterly_frequency`, `intraday_frequency` (hour, session, weekday; Cramer's V) |
| stability | `stability` (every refit: alignment cost, mean / Bhattacharyya / covariance / transition drift, overlap ARI, distance from the first definition), `scheme_comparison` (expanding vs rolling, quarterly vs monthly), `era_definitions` |
| outcomes | `regime_outcomes`, `mean_reversion_by_state`, `ou_by_state`, `spectral_by_state`, `wavelet_by_state`, `spread_activity_by_state`, `future_volatility_by_state` - each beside volatility buckets and random labels |
| information | `incremental_summary` (+ `_nulls`), `mean_reversion_increment`, `volatility_baseline` |
| controls | `null_controls`, `hypothesis_verdicts`; `<tf>/nulls/`, `synthetic_controls/`, `changepoints/`, `cross_timeframe_agreement`, `defensible_states` |
| post hoc (not registered) | `post_hoc/feature_geometry_<tf>*` and `transition_adjacency.csv` from `scripts/regime_feature_geometry.py` |

Notebooks [`26_regime_feature_space`](research/26_regime_feature_space.ipynb),
[`27_gmm_regimes`](research/27_gmm_regimes.ipynb),
[`28_hmm_regimes`](research/28_hmm_regimes.ipynb),
[`29_regime_stability`](research/29_regime_stability.ipynb),
[`30_regime_outcomes`](research/30_regime_outcomes.ipynb) and
[`31_regime_null_controls`](research/31_regime_null_controls.ipynb) load and
display the stored results.

### Findings on the full 2003-2026 history

55 studies, dataset `ticks-2e173ef8e61bd240`: 1h, 30m, 15m and 5m x K-Means,
GMM and HMM x K = 2..6. The 5m GMM and every diagonal GMM were fitted offline
only. Each study has a causal walk-forward from 2008-01-01 to 2026-09-18 (75
quarterly expanding refits, training from 2003). The HMM also has 5-year
rolling refits (at 5m for K = 2 and 3), and monthly refits for K = 3 at 15m-1h.
The ledger holds 2,115 REG-H tests. 1m was not run: 7.9M bars, and no 1m
log-return FFT or wavelet feature sets exist. Cost decisions, all in
`WORKLOG.md`:
- pipeline-null increments for every K at 1h, K = 3 elsewhere;
- persistence nulls for K = 2..6 at 1h and 30m, K = 3 at 15m and 5m.

**No discrete regimes beyond volatility.** What the states are:

| | GMM / HMM (full covariance) | K-Means |
|---|---|---|
| what separates them | trend and linearity: upward drift with a linear path / downward drift / a less linear path | volatility level, and a fast-scale state (high FFT high/low ratio, concentrated wavelet energy, short OU half-life) |
| NMI with same-K volatility buckets | <= 0.02 (HMM), <= 0.05 (GMM) | 0.14-0.33 for K >= 3 |
| Cramer's V with the hour | 1h <= 0.02, 30m <= 0.14, 15m and 5m 0.09-0.18 | up to 0.33 (15m and 5m; at 5m K = 3 the high-volatility state holds 88 % of London-New York overlap bars) |
| spread, activity, FFT, wavelet, OU half-life | the same in every state (30m HMM K = 3: half-life 19.9-20.7 bars, spectral entropy 0.918 in each) | differ only through the features that define the states |

- **The trend states are feature geometry.**
  - R^2 of the 128-bar regression is close to a monotone function of
    |slope / volatility|: Spearman 0.91-0.93, and a 20-bin step function of
    |slope/vol| explains 84-86 % of R^2's variance.
  - The registered redundancy rule saw only the signed pair (|Spearman|
    0.13-0.19), so it let both through.
  - One Gaussian cannot follow a V, so a mixture puts one component on each
    arm and one on the vertex.
  - The post-hoc check `scripts/regime_feature_geometry.py` (1h and 30m;
    descriptive, not in the ledger) finds the same V in a random walk (0.94),
    shuffled returns and the block bootstrap.
  - A GMM on the pair alone gains as much there (K = 3: +0.86-0.91 nats/bar
    on the random walk, +0.71-0.78 on the real series) and finds the same
    three states.
- **Their persistence is the windows'.**
  - HMM median expected durations match the pipeline nulls at every K
    tested (REG-H-005: 0 of 12 beyond them).
  - At K = 3 (real vs random walk / shuffled / bootstrap, in bars): 1h 30 vs
    27 / 58 / 62; 30m 57 vs 56 / 37 / 34; 15m 52 vs 56 / 61 / 101; 5m 45 vs
    41 / 18 / 165.
  - A row shuffle cuts it to 1.5 bars.
  - The HMM's likelihood gain over a GMM (0.85-0.91 nats/bar) is the nulls'
    too (0.54-1.00).
  - Without R^2, the post-hoc HMM finds volatility states instead. Their
    persistence (42-44 bars) exceeds the random walk and shuffled returns
    (14-19) and equals the 1024-block bootstrap at 1h (46; 25 at 30m).
- **Likelihood never settles on a K.**
  - BIC and the hold-out likelihood improve with every K.
  - The registered K rule (REG-H-004) gives the HMM 4 / 6 / 5 / 6 at 1h / 30m /
    15m / 5m and the GMM 5 / 3 / 4 / untested (the 5m GMM has no
    walk-forward). The diagonal GMM gets 1 at every timeframe; K-Means is not
    applicable (no likelihood).
  - The steps are noisy: hold-out fits use one initialisation, and at 1h
    K = 5 fits worse than both 4 and 6.
  - The synthetic controls show why no likelihood rule can separate regimes
    from a continuum. A known 3-state HMM plateaus exactly at K = 3. A
    smoothly drifting volatility improves at every K (+3.4 nats at K = 5),
    with persistent states that move only between neighbours (adjacency
    0.97-0.99).
  - The real HMM at 30m and 1h also moves almost only between neighbours in
    the trend ordering: up <-> down 2e-5 vs 0.011 via "flat"; adjacency
    0.996-0.999 at K = 3-4 on both and K = 5 at 1h (chance 2/K;
    `regime_feature_geometry.py --adjacency`).
  - At 15m and 5m the states divide the slope / R^2 plane in two dimensions
    (a linear-path state, then up and down), so a one-dimensional ordering
    does not apply (0.70 at K = 3).
- **Stable, but that proves little.**
  - Refits agree with their predecessors on the overlap (median ARI
    0.97-0.996).
  - Definitions hold at K = 2-3 and drift at K >= 4 (REG-H-013: 21 stable,
    14 drift).
  - Monthly refits reproduce quarterly ones (ARI 0.98-0.99). Expanding
    training beats 5-year rolling in 16 of 17 (REG-H-012).
  - The two schemes' states agree less as K grows: ARI 0.63-0.89 at
    K = 2-3, 0.28-0.44 at K = 5-6.
  - Filtered HMM confidence is 0.95-0.98, above 0.99 in 79-92 % of bars.
    High entropy marks switches (median 2 bars from one vs 23), not
    volatility.
  - The leave probability is miscalibrated in 13 of 20 (REG-H-009; slopes
    -0.36 to 1.58).
- **What differs by state is mechanical or volatility.**
  - After |Z| > 2 the residual shrinks within 10 bars in 82 % of 30m HMM
    "less linear" bars vs 66-68 % in the trend states.
  - Adding the states to a logistic model with volatility quartiles helps
    (REG-H-001 beyond the pipeline nulls in 13 cases, by <= 0.007 nats), but
    the same states fitted on the random walk and bootstrap produce 52-97 %
    of that gain.
  - REG-H-006: 30 of 55 separate decay beyond volatility buckets. REG-H-003:
    19 of 55 (28 had no such state).
  - Daily ln RV moves in level shifts (PELT: 87 over 6,058 days, median
    segment 47 days). The HMM trend states last hours to days instead
    (K = 3 median run 41 bars at 5m, 69 at 1h).
- **Across timeframes the states disagree.** 30m vs 1h: Cramer's V 0.21-0.39
  (REG-H-010: 2 of 20 above 0.3). 15m or 5m vs 1h and 5m vs 15m: 0.004-0.11.
  The inputs are counted in bars, so "the trend over 128 bars" is a different
  horizon at each timeframe.
- **Information beyond the continuous features is negligible.**
  - Model A holds volatility over 5-1024 bars, the hour, regression, OU, FFT,
    wavelet, spread and activity. A+soft beats A, random states and both
    pipeline nulls in 14 of 825 tests (REG-H-007; 3 at 1h, 9 at 15m, 2 at
    5m). The largest gain is +0.001 R^2 or +0.001 nats, all on
    volatility-type or residual targets, never on returns.
  - A plain one-hot of volatility buckets adds up to +0.039 R^2 to the same
    targets: nonlinearity in volatility, not regimes.
  - Soft probabilities beat hard labels in 206 of 825 (REG-H-008).

Conclusion: the regime models describe two things well.
- Trend states: feature geometry made persistent by trailing windows.
- K-Means states, and the HMM once R^2 is removed: volatility and the daily
  cycle.

The verdicts that "pass" (REG-H-004, -006, -011 and a few of -001 / -007)
pass for those reasons. No regime feature is a candidate beyond volatility at
several horizons and the hour of day. The live-safe machinery works:
filtered features are bit-for-bit prefix-invariant, and a synthetic 3-state
HMM is recovered (filtered accuracy 0.98, transition error 0.003).

---

## Feature factory and predictive feature research

The eighth layer turns every earlier layer into one registered, versioned,
causal feature matrix per timeframe, keeps the outcomes in a separate target
table, and measures which features carry forward information - against nulls
strong enough that 50 pure-noise features earn no candidate status.

```bash
xq build-feature-factory --timeframe 5m --live-safe-only   # alias: build-feature-matrix
xq feature-research --timeframe 1h                         # 12 cached stages, statuses, ledger, figures
xq feature-research --all                                  # 5m, 15m, 30m, 1h, one child process each
xq feature-research --timeframe 1h --feature log_rv_20 --target realized_vol
xq feature-redundancy --timeframe 1h
xq alpha-decay --timeframe 1h --feature ret_1
xq feature-research --compare
```

Configuration: [`config/features.yaml`](config/features.yaml) (families,
windows, interactions, invalid combinations, prior statuses),
[`config/targets.yaml`](config/targets.yaml) and
[`config/alpha_research.yaml`](config/alpha_research.yaml) (IC, nulls, veto
rule, the nine preregistered hypotheses `ALPHA-PR-01..09`, the ledger).

**The factory** (`features/`). A registry entry per feature: id, family,
window, warm-up, source module and series, cost, incremental-update note,
parameter family, prior status (from Prompts #2-#7), live-safe flag, feature
version. Families: returns, volatility, autocorrelation, regression (the
Prompt #3 store), OU, FFT and wavelet (the stored, version-checked Prompt #5 /
#6 sets and causal tails), regime (Prompt #7 walk-forward *filtered*
probabilities only), microstructure (spread, tick activity), time, and
registered interactions (products of trailing z-scores). Non-causal
constructions are catalogued in `feature_catalog.json` only to record their
exclusion; they are never computed. Sources are joined on exact timestamps
(a stray or duplicate timestamp is refused). The matrix is float32 per year,
with a manifest (`factory-<tf>-<hash>`), the dataset lineage and a `CURRENT`
pointer; building twice gives identical values.

**Targets** (`targets/`, namespace `target_*`): future log return, residual
reduction, realised volatility, absolute return and excursions, at several
horizons, stored apart from the features and never joined into the matrix.

**The research** (`research/feature_research.py`, 12 stages, each cached by a
stamp of data versions, config and its code): quality (missingness, drift),
IC, conditioning (volatility terciles, regime states, intraday), mutual
information, redundancy, deciles, interactions, nulls, robustness, cost,
pipeline nulls.
- **IC** from monthly sufficient statistics (one matrix product per month),
  Newey-West standard errors over monthly batches, yearly / quarterly / era /
  recent / causal rolling IC, decay by horizon and an information half-life.
- **Nulls.** A studentized Westfall-Young max-T circular-shift null (200
  shifts, all targets of a feature at once); pipeline nulls - a random walk,
  shuffled whole bars and a sign flip, as full OHLC paths through the same
  engines (a residual claim needs the random-walk control: invariant 9), and a
  1024-bar block bootstrap for reference. A null *vetoes* a claim when it
  reproduces at least half of the real IC with the same sign. 50 noise
  features go through every test.
- **Multiple testing.** Every test gets a permanent `ALPHA-H` id in the ledger
  (preregistered or exploratory), BH q-values per family, Bonferroni and the
  effective number of tests.
- **Statuses** (information, not signals): `strong_candidate`, `candidate`,
  `weak_candidate`, `redundant`, `unstable`, `failed_null`.

Notebooks [`32_feature_catalog`](research/32_feature_catalog.ipynb) to
[`38_feature_null_tests`](research/38_feature_null_tests.ipynb) display the
stored results.

### Findings on the full 2003-2026 history

5m, 15m, 30m and 1h (1m not run: 7.9M bars and no 1m FFT / wavelet sets),
dataset `ticks-2e173ef8e61bd240`, 127-129 registered features and 81,733-83,019
tests per timeframe in 54 test families (the ledger gained 330,790 `ALPHA-H`
rows).

| tf | strong | candidate | weak | redundant | unstable | failed_null | noise features with a candidate status |
|---|---|---|---|---|---|---|---|
| 5m | 43 | 30 | 20 | 11 | 11 | 11 | 0 of 50 |
| 15m | 38 | 26 | 16 | 12 | 0 | 37 | 0 of 50 |
| 30m | 29 | 35 | 11 | 19 | 2 | 33 | 0 of 50 |
| 1h | 27 | 33 | 6 | 15 | 2 | 46 | 0 of 50 |

- **Volatility is the information.** Past volatility ranks future realised
  volatility with rank IC 0.58-0.74 (Parkinson 20 bars, 0.74 at 5m; EWMA
  lambda 0.99, 0.68 at 1h), in every year. Time of day and session (0.13-0.36)
  and tick activity (0.16-0.37) come next: the daily cycle.
- **Direction is short-term reversal and small.** The last bar's return ranks
  the next with IC -0.097 (5m), -0.082 (15m), -0.065 (30m), -0.054 (1h): it
  shrinks with bar length and is strongest where the early feed's quote noise
  is (read with the era splits, not pooled).
- **Residual reversion is mostly mechanical.** The random-walk pipeline null
  reproduces the median residual IC in full; about a third of residual tests
  beat it (1,149 of 3,556 at 5m, 1,310 of 3,612 at 1h), and the statuses
  require that control (invariant 9).
- **FFT, wavelet and regime features are volatility proxies.** At 1h 12 of 14
  FFT, 9 of 10 wavelet and 7 of 10 regime features fail the nulls; what passes
  at 5m (wavelet entropy, FFT residual entropy, regime state probabilities)
  passes on volatility targets. OU half-life / theta relations with
  excursions flip sign by year (`unstable`).
- **Redundancy is large.** Volatility estimators are interchangeable (Parkinson,
  Garman-Klass, 20-bar RV, EWMA 0.94), OU parameters are exact functions of one
  another (|Spearman| ~ 1), residual Z-scores track residual volatility; the
  effective number of features is ~75 of 127-129 (Li-Ji).

## Feature selection and dimensionality reduction

The ninth layer asks which compact subset of those features should enter
supervised learning in the next stage - causal, live-safe, low-redundancy,
stable through time, reasonably predictive, cheap enough to run live, and
chosen in a way that resists data mining. Feature selection is part of model
fitting, so it happens inside a chronological training methodology.

```bash
xq feature-select --timeframe 5m                               # every stage + report + manifests
xq feature-select --timeframe 5m --target residual_reduction   # one target family's view
xq feature-select --timeframe 5m --target future_return --horizon 5
xq feature-selection-report --timeframe 5m                     # summary, SEL-H ledger rows, plots
xq feature-selection-report --compare
xq build-feature-manifest --set standard --timeframe 5m        # verify + reproduce the matrix
```

Configuration: [`config/feature_selection.yaml`](config/feature_selection.yaml).

**Three periods, and what each may be used for.**

| Period | Rows | Used for |
|---|---|---|
| development | 2003-01-01 .. 2018-01-01 | every selection decision: screens, clusters, representatives, stability selection, nested folds (validate 2011-12, 2013-14, 2015-17) |
| validation | 2018-01-01 .. 2022-01-01 | evaluating what development chose, and reading the size plateaus - never to choose a feature |
| reserved test | 2022-01-01 .. end | nothing: its target files are never opened; feature *values* may be computed (causal) |

Targets are purged at each period boundary by their own horizon (an outcome
window may not cross into the next period), training rows end `horizon +
embargo` bars before an evaluation block, and a guard refuses any outcome row
at or after the reserved start. Prompt #8's statuses saw 2022-2026 outcomes,
so they are reported beside the selection but never used by it: the universe
is re-screened on development only.

**Stages** (`research/feature_selection_research.py`, cached like Prompt #8's):

1. `universe` - the quality filter with reason codes (`EXCLUDE_NON_CAUSAL`,
   `EXCLUDE_INVALID`, `EXCLUDE_MISSINGNESS`, `EXCLUDE_UNAVAILABLE`,
   `EXCLUDE_NEAR_CONSTANT`, `EXCLUDE_NUMERICAL`, `EXCLUDE_FAILED_NULL`; drift
   is classified, never a reason) and the development null screen (max-T
   circular shifts, BH, the pipeline veto on development quarters).
2. `redundancy` - clusters on 1 - |Spearman|, a representative per cluster by
   a documented lexicographic rule (passes the null, yearly sign consistency,
   |median yearly IC|, recent relevance, missingness, cost, warm-up - never the
   largest IC alone), parameter-family reduction, non-monotone near-duplicates.
3. `evidence` - effect stability (|median yearly IC| x max(0, 2c - 1)),
   recent relevance, regime robustness, target specificity.
4. `stability` - 50 resamples of half the development quarters plus
   odd / even years and early / middle / late thirds; mRMR and the Lasso on
   each; a run keeps the features entering **before the first of 8 noise
   probes** (at most 20). Selection frequency, Jaccard, rank stability.
5. `nested` - for each chronological split the whole selection is redone on
   its training rows (training-CDF rank design, in-split screens): IC ranking,
   effect stability, mRMR (|rank IC| or copula MI; difference / quotient),
   cluster representatives, Lasso and elastic-net entry order; ridge learning
   curves for k = 5 ... 75; PCA fitted on training rows; cumulative family
   ablation, leave one family out, block permutation importance (correlated
   features permuted together), interaction hierarchy, an L1 logistic path for
   the sign of the return.
6. `sets` - Minimal / Standard / Extended as prefixes of one development
   ranking (selection-stable features first), sized by the validation plateau
   (the smallest k reaching 0.90 / 0.97 of the best rank IC; residual targets
   excluded); a general set and one set per target kind where justified;
   collinearity (condition number, VIF, effective rank), cost, warm-up and
   buffer per set; immutable manifests.
7. `audit` - streaming reconstruction (bars replayed one at a time into a
   rolling buffer, each feature recomputed and compared with the stored
   matrix), the per-feature leakage audit and the reserved-isolation check.

The diagnostic models are linear on rank designs from sufficient statistics:
ridge, the Lasso and a naive elastic net by coordinate descent, an L1 logistic
path by FISTA (scikit-learn is not a dependency). They measure whether a
reduced set keeps forward information; they are not the ML system.

**Manifests** (`results/feature_selection/<tf>/manifests/<set>.json`,
id `FEATURESET_<TF>_<SET>_V<nnn>`): ordered features (configured family order,
then registry order - never a ranking), each with id, family, dtype,
transformation, window, warm-up, live-safe flag, source, feature version,
refresh frequency, declared cost and its selection evidence; the set's
maximum history, effective rank, the periods it was created from and the
reserved test period, config fingerprints and the git commit. A content hash
makes it immutable (a different set under the same id is refused);
`build-feature-manifest` verifies the hash and reproduces the stored matrix
in manifest order.

Every selection experiment - each method x k x fold x target, each PCA
dimension, each stability run, each ablation and candidate set - gets a
permanent `SEL-H` id in the research ledger.

Notebooks [`39_feature_filtering`](research/39_feature_filtering.ipynb) to
[`44_feature_ablation`](research/44_feature_ablation.ipynb) display the
stored results.

### Historical findings (selection 2003-2017, size validation 2018-2021)

Stage #12 audit: these immutable manifests use later outcomes than early
ML scoring folds. The Standard/Minimal counts depend on validation plateaus.
These recorded findings are preserved as historical diagnostics; the
2022+ period was already inspected by feature research and later stages.

| tf | registered | quality | development universe | Minimal | Standard | Extended | general | direction | reversion | volatility | magnitude |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5m | 127 | 124 | 86 | 20 | 20 | 75 | 17 | 19 | 13 | 17 | 17 |
| 15m | 129 | 126 | 78 | 20 | 72 | 72 | 17 | 11 | 6 | 15 | 18 |
| 30m | 129 | 123 | 88 | 50 | 50 | 75 | 16 | 15 | 8 | 15 | 17 |
| 1h | 129 | 115 | 70 | 20 | 30 | 64 | 15 | 9 | 4 | 15 | 17 |

- **Volatility and magnitude carry the information, compactly.** The 17-feature
  5m volatility set reaches validation rank IC 0.43 / 0.69 / 0.75 (h = 1 / 5 /
  20; R^2 0.29 / 0.63 / 0.70), the magnitude set 0.43 / 0.39 / 0.34. At every
  timeframe the first 20 features of the general ranking reach >= 97 % of the
  best validation IC for both kinds; the rest adds nothing.
- **Direction is weak.** 5m validation rank IC 0.053 / 0.045 / 0.035 with R^2
  below zero; it is mostly short-term reversal (returns over 1-10 bars,
  residual Z-scores), whose 5m development IC shrinks from -0.13 (2003-2017) to
  -0.05 (2015-2017) and does not hold in every filtered regime state.
- **Reversion is small and partly mechanical.** 5m validation IC 0.03-0.10 for
  13 features that each beat the random-walk pipeline on development; the set's
  own IC has no null model beside it.
- **The general Standard size is not well defined.** The registered plateau
  rule (mean over targets of IC / best IC) gives 20 / 72 / 50 / 30 at 5m / 15m
  / 30m / 1h because the weak direction targets keep rising until their
  late-ranked features enter; volatility and magnitude settle at 20 everywhere
  (post-hoc sensitivity, not registered).
- **mRMR vs IC ranking:** top-20 Jaccard 0.67-0.82; mRMR swaps duplicate
  volatility estimators for activity and time features; validation IC moves by
  <= 0.005.
- **PCA does not help:** 5 components hold 38 % of the variance and lose up to
  0.07 IC (5m `realized_vol_20`: 0.64 vs 0.71 raw); at 20-30 components PCA
  comes within ~0.04 of the raw features on volatility and matches them only
  on the weak direction targets (within noise).
- **Families:** leaving out the statistics family (returns, volatility, ACF)
  costs 0.058 validation IC on average at 5m, time 0.011 (0.050 on
  `realized_vol_20`), microstructure 0.004; OU and regression matter only on
  residual targets (mechanical); FFT, wavelet, regime and interactions
  <= 0.0003.
- **Stability:** at 5m no noise probe enters the first 20 steps for
  volatility / magnitude, 12-20 features enter before one for direction, 5-9
  for reversion; resample Jaccard 0.61-0.96; the late third of development
  agrees least with the full period (0.43-0.56).
- **Live:** streaming reconstruction matched the stored matrix at every
  timeframe (60 bars; 1.2 s per bar at 5m with an 18,040-bar buffer); the 5m
  Standard set needs 5,520 bars of warm-up (20 trading days), Extended 7,520.

---

## Supervised predictive-model research

The tenth layer asks one question: **can supervised models extract stable
out-of-sample predictive information from the selected features?** Prediction
first, trading later - models output a probability (classification) or an
expected value (regression), never BUY / SELL, a threshold, a position, a stop
or a PnL. Stable weak prediction counts; a strong result in one year does not.

```bash
pip install -e ".[ml]"                                   # scikit-learn, XGBoost, LightGBM, CatBoost, SHAP
xq ml-research --timeframe 5m --dry-run                  # the unit plan and what is already cached
xq ml-research --timeframe 5m [--stages grid,trees]      # cached walk-forward units (child process)
xq ml-train --timeframe 5m --target mean_reversion --horizon 5 --model xgboost --feature-set standard
xq ml-walk-forward --timeframe 5m --target mean_reversion --horizon 5   # every enabled family
xq ml-compare --timeframe 5m --target mean_reversion     # from stored units; trains nothing
xq ml-report --timeframe 5m                              # tables, ML-H ledger rows, figures
xq ml-freeze --timeframe 5m                              # pre-registered rules -> MODEL_SPEC files
xq ml-finalize --timeframe 5m                            # artifacts, reload, streaming, latency
xq ml-final-test --model-spec LGBM_REVERSION_5M_H5_V001  # once; frozen specs only
```

Configuration: [`config/ml.yaml`](config/ml.yaml), fixed before any result
existed (targets, horizons, blocks, model defaults, search grids, freeze rules).

**Targets** (per bar t, horizon h; the Prompt #8 target table plus labels derived
in memory, never stored):

| Target | Task | Definition |
|---|---|---|
| `mean_reversion` | classification | 1{\|eps_t+h\| < \|eps_t\|}, eps the N = 128 regression residual |
| `mean_reversion_half` | classification | 1{\|eps_t+h\| < 0.5 \|eps_t\|} |
| `residual_reduction` | regression | (\|eps_t\| - \|eps_t+h\|) / sigma_t |
| `future_return` | regression | ln(P_t+h / P_t) / (sigma_t sqrt(h)) (rank IC read on the raw return) |
| `direction_up_cost` / `direction_down_cost` | classification | R_t,h > +spread_t / < -spread_t (one spread; a label, not a cost model) |
| `future_volatility` | regression | ln RV_t,h |
| `future_abs_move` | regression | ln \|R_t,h\| (floored) |

One model per target and horizon - no universal model.

**Walk-forward.** Five expanding validation blocks - 2011-12, 2013-14, 2015-17,
2018-19, 2020-21 - each trained on everything before it (a rolling five-year
window is a research variant). Between training rows and the block: a purge of
h bars (the last label's outcome window must end first) plus a 12-bar embargo.
The last 10 % of each training span, purged on both sides, is an *inner slice*
used only for early stopping and to fit calibrators - never to fit the model,
never scored. Preprocessing (median imputation, missing indicators, robust
scaling and clipping for linear models; boosters see NaN natively) is fitted on
the fold's training rows and frozen, and refuses a feature order it was not
fitted with. No date, timestamp or row number is ever a predictor; time enters
only through the cyclical features of the manifests. The reserved period
(2022-) is not in memory: development loading never opens its year files.

**Models.** Constant (the training mean or positive rate) and a reference linear
model (L2 logistic / ridge) at every target x horizon; L1 and elastic-net
logistic, OLS and elastic net at the seven tree-comparison pairs; random
forest, XGBoost, LightGBM and CatBoost there too, LightGBM at every primary pair.
Metrics: AUC, PR AUC, log loss, Brier, ECE, accuracy / precision / recall at 0.5
(secondary) for classification; MAE, RMSE, R^2, Pearson, Spearman, rank IC and
bias for regression. **Skill** is 1 - loss / loss of the constant training mean
(log loss or MSE), positive = better than the baseline, and every comparison is
read block by block (mean difference, SE, wins).

**Stages** (`research/ml_research.py`; each unit = target x horizon x family x
feature set x variant x block writes its predictions and metrics as it
finishes, stamped with data versions, spec and the code of `ml/*.py`; a rerun
skips current units): `grid`, `trees` (with TreeSHAP, block permutation
importance, ALE), `feature_sets` (Extended and target-specific vs Standard),
`variants` (booster NaN vs imputation, class weights), `search` (12 LightGBM +
8 XGBoost random trials on two pairs, the one-step neighbourhood of the best,
a num_leaves complexity curve), `windows` (rolling vs expanding, time-decay
weights with 2- and 5-year half-lives), `learning_curve`, `decay` (a model
frozen in 2011 scored every later year), `retraining` (refits every 1 / 3 / 6 /
12 months over 2018-21), `ablation` (leave one family out, forward family
addition), `nulls` (training labels circularly shifted, noise columns, permuted
features) and `pipeline_null` (residual targets computed on a random walk
pushed through the same feature and target engines - invariant 9).

**Freeze rules** (`ml-freeze`, development results only): a model is eligible
when it beats the constant in at least 4 of the 5 blocks with a positive mean;
the simplest eligible model within one SE of the best is frozen (simplicity
order constant < linear < forest < boosters, then fewer features, Standard
first); a rolling window, time-decay weights, a searched parameter set or a
calibration method replaces the default only when it is better block by block
by more than one SE (searched parameters also need every one-step neighbour
within two SE, and apply only to a model frozen on the set the search ran on,
Standard - trials are compared with the defaults on that set alone). The
constant and the reference linear model are frozen beside
each candidate so the final test compares like with like. Specs
(`frozen/MODEL_SPEC_<id>.json`, ids like `LGBM_REVERSION_5M_H5_V001`) are
content-hashed and immutable.

**Final development models and the final test.** `ml-finalize` fits each frozen
spec on the whole development span, saves it in its native format (XGBoost
UBJSON, LightGBM text, CatBoost `.cbm`, joblib) with its preprocessor,
calibrator and a hashed manifest under `data/models/<id>/`, checks that the
reloaded model reproduces its predictions, replays bars one at a time through
the feature engines (streaming = batch, or the model fails live-readiness) and
times inference. `ml-final-test` accepts only frozen, hash-verified specs,
evaluates each once on 2022- (overall, per year, the last twelve months, by
regime / volatility / session), writes an audit trail of predictions, and
refuses a second look without a reason that is logged in
`final_test/access_log.jsonl`.

Every configuration tried - failed ones included - gets a permanent `ML-H` id
in the research ledger, so the number of trials behind any result is known.

Outputs per timeframe (`results/ml_research/<tf>/`): `units/` (every fold-level
fit: predictions by row and a JSON with metrics, spec and stamp), `tables/` and
`plots/` (the report), `predictions/<target>_h<h>.parquet` (the walk-forward
out-of-sample predictions of every base model of a comparison pair: timestamp,
fold, label, one column per `<family>|<set>` and its Platt probability, with a
sidecar JSON of model ids, feature-set ids and versions - the input for
ensemble research), `frozen/` (specs, `freeze_report.json`, `finalize.json`),
`final_test/` (results, prediction audit trail, access log) and per-problem
copies under `<target>/h<h>/`.

### Findings (walk-forward 2011-2021; one final test 2022-01-03 -> 2026-09-18)

Frozen by the registered rules on development evidence only, then evaluated
once (the reference linear model and the constant were frozen and tested beside
each candidate):

| tf | target (metric) | frozen candidate | development | final test | linear reference, final |
|---|---|---|---|---|---|
| 5m | future volatility h5 (rank IC) | LightGBM, target set | 0.688 | 0.741 | ridge 0.716 |
| 15m | future volatility h5 (rank IC) | LightGBM, target set | 0.647 | 0.702 | ridge 0.678 |
| 5m | absolute move h5 (rank IC) | random forest | 0.392 | 0.433 | ridge 0.417 |
| 15m | absolute move h5 (rank IC) | random forest | 0.372 | 0.411 | ridge 0.402 |
| 5m | mean reversion h5 (log-loss skill) | LightGBM, Extended | +0.143 | +0.136 (AUC 0.736) | logistic +0.005 |
| 15m | mean reversion h5 (log-loss skill) | CatBoost | +0.132 | +0.134 (AUC 0.738) | logistic +0.046 |
| 5m | residual reduction h5 (rank IC) | LightGBM, Extended | 0.459 | 0.465 | ridge 0.089 |
| 15m | residual reduction h5 (rank IC) | random forest | 0.427 | 0.437 | ridge 0.261 |
| 5m | reversion c = 0.5, h5 (log-loss skill) | LightGBM | +0.005 | +0.007 | logistic +0.005 |
| 15m | reversion c = 0.5, h5 (log-loss skill) | LightGBM | +0.064 | +0.061 | logistic +0.012 |
| 5m | up beyond one spread, h5 (log-loss skill) | logistic, Extended | +0.035 | +0.039 (+0.007 beyond the recent base rate) | - |
| 15m | up beyond one spread, h5 (log-loss skill) | logistic, target set | +0.016 | +0.019 (+0.000 beyond it) | - |
| 5m | future return h1 (rank IC) | XGBoost | 0.046 (MSE skill +0.0001) | 0.021 (MSE skill -0.0006) | ridge 0.022 |
| 15m | future return h1 | none eligible (no model beat the constant in 4 of 5 blocks) | | | |

- **Volatility is the predictable target.** Every family beats the constant in
  5/5 blocks; boosters add +0.03 rank IC over ridge (t ~ 7); per-year IC 0.61-
  0.74 in development and 0.62-0.71 in the final test at 5m (15m 0.55-0.72 and
  0.53-0.68), lowest in 2026 (0.62 at 5m, 0.53 at 15m - the most recent months
  are the weakest). Time of day is the one family beyond the statistics block
  that matters (removing it costs 0.044 IC at 5m), then microstructure (0.005);
  regression, OU, FFT, wavelet and regime features add nothing. All families'
  predictions correlate 0.92-1.00 - one signal, not several.
- **Reversion skill is mechanical.** The large numbers come from z-scores of the
  very residual the targets are built from: on a random walk pushed through the
  same pipeline the same LightGBM reaches +0.126 (5m Extended) / +0.112 (15m),
  and on the post-hoc sign-flip null (the real volatility path, random signs)
  +0.136 / +0.126. Beyond the sign-flip null the real data add +0.002 to +0.008
  log-loss skill (4-5 of 5 blocks). Residual reduction: the boosters are within
  -0.031 to +0.015 rank IC of the nulls (below the random walk at 15m); only the
  weak 5m ridge on Standard (0.11) clears them (+0.04-0.06).
- **Direction is weak and mostly magnitude.** The one-spread labels' base rate
  drifts with spreads and volatility, so a constant recalibrated on the recent
  inner slice already earns +0.018 / +0.020-0.023 / +0.009-0.014 (h1 / h5 /
  h20, 5m); beyond it the features add +0.030 / +0.007 / +0.001-0.003. Their
  scores correlate 0.65-0.68 with the volatility and absolute-move scores.
  Future return: rank IC 0.04-0.05 in development (beyond shifted labels by
  +0.05, decaying with the horizon: 0.047 at h1, 0.020 at h20), negative MSE
  skill throughout, and half of it in the final test (0.021).
- **Complexity earns little.** Trees beat linear models where the target is
  nonlinear in volatility (+0.03 IC) and on the mechanical reversion targets
  (the same gap as on the random walk: +0.083 real vs +0.078 random walk, 15m);
  on direction and return they tie or lose. XGBoost, LightGBM and CatBoost are
  interchangeable (predictions correlate 0.93-1.00, 0.70-0.78 on future return);
  random forest is weaker and 40-180x slower to score. Searches found a plateau:
  LightGBM trials within 0.005 skill (reversion) / 0.004 IC (volatility) of each
  other, XGBoost within 0.009 / 0.013, the defaults at or near the top;
  complexity flattens from 7-15 leaves.
- **Feature sets:** Extended helps only reversion (through the residual
  z-scores); volatility +0.002; return worse. Target-specific sets are within
  noise of Standard for volatility and abs move, worse for reversion.
- **Calibration:** reversion probabilities are nearly calibrated raw (ECE
  0.003-0.014 at 5m, 0.014-0.028 at 15m; isotonic ~0.007) and stay so by year
  (Platt ECE 0.007-0.020 at 5m, 0.009-0.033 at 15m), volatility quartile and
  session (one regime state reaches 0.04). The direction labels need Platt /
  isotonic (ECE 0.06-0.08 -> 0.01-0.03) and still drift with their base rate:
  ECE 0.046 / 0.062 in 2011, 0.031 / 0.040 in 2020, 0.03 / 0.05 in the top
  volatility quartile (5m / 15m).
- **Windows, retraining, decay:** rolling 5-year windows and 2-year half-life
  weights help volatility a little (+0.002-0.005 IC, 3-5 of 5 blocks) and hurt
  the return ridge (-0.02, 0 of 5); refitting every 1, 3, 6 or 12 months makes
  no difference; a LightGBM frozen at the end of 2010 trails the refitted ones
  by ~0.06 volatility IC by 2017-18 and halves its 5m reversion skill, while the
  linear models barely decay.
- **Controls:** shifted labels and permuted features collapse every target
  (shifted volatility labels keep part of the daily cycle); five noise columns
  rank 15th or lower of 25 by SHAP at 5m (21st or lower of 77 at 15m) and change
  nothing; no feature has |Spearman| > 0.75 with any target (no leakage canary);
  every frozen model reloads identically and replays bar by bar equal to batch
  (<= 1e-10), at 0.14-1.0 ms per row (random forests 19-28 ms).
- **Trials:** 684 configurations in the ledger (`ML-H`, 24 of them post-hoc
  controls), 2,714 fold-level fits.

Notebooks [`45_ml_baselines`](research/45_ml_baselines.ipynb),
[`46_tree_models`](research/46_tree_models.ipynb),
[`47_probability_calibration`](research/47_probability_calibration.ipynb),
[`48_model_stability`](research/48_model_stability.ipynb),
[`49_feature_importance`](research/49_feature_importance.ipynb) and
[`50_final_oos_evaluation`](research/50_final_oos_evaluation.ipynb) display the
stored results.

---

## Ensembles, meta-models and predictive diversification

The eleventh layer asks: **does combining genuinely distinct models give
predictions that are more stable, better calibrated, less fragile and more
informative out of sample than one model?** Still prediction research - an
ensemble outputs a probability or an expected value plus uncertainty proxies,
never BUY / SELL, a threshold, a size or a PnL - and simple averaging is the
benchmark every other combination has to beat.

```bash
xq ensemble-research --timeframe 5m --report            # gate, meta walk-forward, controls, tables, figures
xq ensemble-build -t 5m --target mean_reversion --horizon 5 --method stacking   # one combination, research only
xq ensemble-build -t 5m --target future_volatility --horizon 5 --method simple_average --models "lightgbm|target" "ridge|standard"
xq ensemble-report --timeframe 5m                       # cross-pair tables, ENS-H ledger rows, joint predictive state
xq ensemble-freeze --timeframe 5m                       # pre-registered rule -> ENSEMBLE_SPEC + constituent specs
xq ensemble-finalize --timeframe 5m                     # constituents fitted, reload / versions / streaming, benchmark
xq ensemble-final-test --ensemble-spec ENS_VOL_5M_H5_V001   # once; logged; a second look at 2022-
```

Configuration: [`config/ensemble.yaml`](config/ensemble.yaml), fixed before any
ensemble result (one later packaging correction is marked in the file).

**Inputs and the meta walk-forward.** The inputs are Prompt #10's walk-forward
out-of-sample predictions (`results/ml_research/<tf>/predictions/`): every row
of the five blocks 2011-2021 was predicted by a model fitted before its block.
They are refused unless they describe the current data (versions, target,
horizon, timeframe, strictly increasing rows and timestamps, contiguous blocks,
no row at or after the reserved start); nothing is ever filled - a row missing a
constituent is left out of that ensemble and counted
(`ensemble_prediction_coverage`). The ensemble research is itself a
walk-forward over those blocks: every method is scored on 2013-2021, and
everything it fits - weights, a stacked meta-model, a calibrator, conditional or
trailing weights, a subset, the "best individual" it is compared with - is
fitted on the earlier blocks' out-of-sample rows only, purged by the horizon
plus a 12-bar embargo at the boundary. One ensemble per target x horizon x
timeframe (never across targets or horizons).

**Eligibility (Step 2).** From the Prompt #10 development evidence, each base
model gets one status: `leakage_failed` (unregistered / non-live-safe features,
a leakage canary, a fold whose training outcomes reach its block),
`deprecated` (feature set or data versions not current), `null_failed` (within
the registered null - shifted labels, the random-walk pipeline - or, for the
drifting one-spread labels, no skill beyond the recalibrated constant),
`untested` (an undefined statistic: never a pass or a fail), `weak` (beats the
constant in fewer than 4 of 5 blocks), `unstable` (worst block below half the
mean), `calibration_failed` (mean ECE > 0.05 or no Brier skill) or `eligible`.
The eligible models minus near-duplicates (prediction correlation >= 0.995;
the simpler model kept) form the *universe* (at most 8). Because the gate reads
all five blocks, a robustness variant rebuilds the universe of each block from
the earlier blocks only (`*_wf_universe`).

**Methods** (complexity order, Step 84): the best individual chosen on earlier
blocks; simple average; median; performance weights (skill against the
constant on earlier blocks, shrunk halfway to equal weights; unshrunk as a
variant); diversity-aware weights (skill x (1 - lambda x mean error
correlation), one weight per duplicate cluster); stacking (logistic on the
constituents' logits / ridge on their values; with a few context inputs as a
variant); weights conditional on the regime probabilities (soft) or on causal
volatility quartiles; and trailing model-health weights updated daily from
labels that had resolved before the day began. Research beside them: ensemble
size (top-*s* and greedy forward selection), family and feature-set
diversification, weaker models added, leave one model / family out, each
model's contribution, calibration before / after averaging, cross-horizon
stacking, disagreement and OOD against error, failure and common-mode failure,
decile shape and alpha decay, stability by year / quarter / regime /
volatility / spread / session.

**Controls.** Ensembles of the shifted-label models (scored on the real
labels) and of the random-walk / sign-flip pipeline models (scored on the
null's own labels - invariant 9); an intentionally useless noise model and a
duplicated model inserted into the universe, with the weight every method gives
them.

**Freeze rule.** Candidates in the order above; one replaces the one retained
so far only when it is better on the primary metric block by block by more
than one SE in at least 3 of the 4 blocks and (probabilities) does not raise
the mean ECE by more than 0.005. What the rule retains is frozen - an
`ENSEMBLE_SPEC_<id>.json` (`ENS_<TARGET>_<TF>_H<h>_V<nnn>`) with its
constituents' frozen `MODEL_SPEC`s (`<FAMILY>_<SET>_<TARGET>_<TF>_H<h>_V<nnn>`)
and spec hashes, the combination fitted on every out-of-sample row 2011-2021, a
final calibrator (simplest of none / sigmoid / isotonic within one SE), the
companions evaluated beside it (simple average, best individual, constant), the
training and refit policy ("frozen between approved versions; no automatic
retraining") and the development evidence. When no ensemble earns its
complexity the single model is frozen as a one-member spec.

**Inference.** `EnsembleModel.load(spec)` verifies the spec hash, each
constituent's spec version and artifact hashes (a constituent of another
version fails: `ModelVersionMismatchError`) and its feature manifest against the
live one (`FeatureVersionMismatchError`); `predict(feature_vector)` /
`predict_proba(...)` run the frozen preprocessing, models, calibrators,
combination and final calibrator. A missing feature or constituent output is
`ENSEMBLE_INVALID` (fail closed; no reduced ensemble). `ensemble-finalize`
replays bars one at a time through the feature engines (streaming = batch) and
measures loading memory, latency, CPU and artifact size.

**Prediction contract.** `xauusd_quant.ensemble.contract`: one record per bar
with `reversion_probability`, `expected_return` (and its vol-scaled form),
`expected_log_volatility` / `expected_volatility`, `expected_log_abs_move` /
`expected_abs_move`, and `uncertainty` (model disagreement, OOD score, regime
entropy, the reversion probability's entropy), each field's source ensemble,
horizon and status; no decision field can exist in a valid record.
`<tf>/joint_predictive_state.parquet` holds the development out-of-sample
records; `prediction_contract_schema.json` the schema.

Every ensemble experiment - methods, variants, sizes, subsets, leave-outs,
controls, calibration orders - gets a permanent `ENS-H` id in the research
ledger.

### Findings (meta walk-forward 2013-2021; 565 ensemble experiments)

- **The constituents are not diverse where it matters.** Predictions differ
  (Spearman 0.16-0.99 between eligible models; least for reversion, where
  feature sets differ), but the errors do not: error correlation 0.90-1.00
  (medians 0.94-0.999) in every pair. For volatility and absolute move the
  predictions themselves correlate 0.90-0.99. Every constituent is beyond its own
  90 % loss quantile at the same time on 0.4-9.9 % of bars - 10^-8 if errors were
  independent, 10 % if identical.
- **Simple averaging never beat the best single model by more than one SE** (13
  pairs); where weaker eligible models sit in the universe it dilutes badly - 5m
  reversion +0.053 log-loss skill against +0.142 for LightGBM on the Extended
  set - and averaging sharp boosters with blunt linear models makes the
  probabilities under-confident (ECE 0.065-0.078 against 0.012-0.015).
  Performance and diversity weights recover part of the dilution and match
  each other to three or four decimals; the best size is 1-4 models in 12 of 13
  pairs.
- **The freeze rule retained an ensemble in 5 of 13 pairs**, all by small
  margins: stacking for 5m volatility (+0.0018 rank IC over the best model, 4/4
  blocks), 5m absolute move and 5m reversion c = 0.5 (+0.0002); regime-conditioned
  weights for 5m residual reduction (+0.003) and stacking for 15m residual
  reduction - both built from minimal / target-set models that have no
  pipeline-null units (their null status is *untested*, so these two cannot be
  read against the random-walk null). The single model was retained for the
  other 8 pairs; 15m future return has no eligible model.
- **Controls:** the noise model gets no raw weight anywhere (stacking share
  0.0003-0.06); a duplicated model gets double weight from naive performance
  weights and exactly its single weight from the diversity-aware weights;
  shifted-label null ensembles stay near zero (volatility 0.13-0.15 rank IC from
  the daily cycle, against 0.64-0.68 real); reversion beside the sign-flip
  pipeline null +0.006; the residual-reduction ensemble of the null-tested models
  falls below both pipeline nulls.
- **Disagreement is not uncertainty here.** Its rank correlation with the loss is
  within +-0.04 for volatility, absolute move, return and direction (OOD within
  +-0.09); for reversion it is *negative* (-0.21 / -0.32): disagreement marks the
  bars where the strong models are confidently right and the weak ones sit at
  0.5. Only for 15m reversion c = 0.5 and residual reduction is it positive
  (+0.18 / +0.14). The top disagreement / OOD tercile does carry 5-30 % higher
  squared error for the regression targets - largely a volatility effect.
- **Weights are stable and near-equal** (no scheme flagged across blocks);
  regime-, volatility-conditioned and trailing weights stay within about
  +-0.001 of the static ones (trailing turnover 0.0003-0.04 per day); other
  horizons' predictions add +0.0003-0.001 to reversion and hurt future return.

### Final test (2022-01-03 -> 2026-09-18; once per frozen spec)

The reserved period had already been evaluated once (Prompt #10's 37
single-model specs), so this is a **logged second look**: the ensembles were
frozen on development data by the rule above, nothing is tuned afterwards, and
every output says so. A first attempt that stopped with a `MemoryError` before
any metric counts as a look; the 5m rerun carries its logged reason. 334,254
labelled 5m rows and 111,421 15m rows, 0 invalid.

| Spec | Frozen | Metric | Ensemble | Simple avg | Best indiv. | Development |
|---|---|---|---|---|---|---|
| VOL 5m h5 | stacking | rank IC | 0.745 | 0.741 | 0.741 | 0.685 |
| ABSMOVE 5m h5 | stacking | rank IC | 0.442 | 0.440 | 0.433 | 0.396 |
| RESIDRED 5m h5 | regime-conditioned | rank IC | 0.093 | 0.092 | 0.089 | 0.114 |
| REVHALF 5m h5 | stacking | log-loss skill | 0.0073 | 0.0071 | 0.0068 | 0.0055 |
| REVERSION 5m h5 | single (LightGBM, Extended) | log-loss skill | 0.137 | 0.050 | = | 0.142 |
| UPCOST 5m h5 | single (logistic, Extended) | log-loss skill | 0.039 | 0.038 | = | 0.029 |
| RETURN 5m h1 | single (XGBoost) | rank IC | 0.021 | 0.020 | = | 0.047 |
| VOL 15m h5 | single (LightGBM, target set) | rank IC | 0.702 | 0.709 | = | 0.643 |
| ABSMOVE 15m h5 | single (random forest) | rank IC | 0.411 | 0.422 | = | 0.374 |
| RESIDRED 15m h5 | stacking | rank IC | 0.269 | 0.243 | 0.263 | 0.251 |
| REVERSION 15m h5 | single (CatBoost) | log-loss skill | 0.135 | 0.106 | = | 0.133 |
| REVHALF 15m h5 | single (LightGBM) | log-loss skill | 0.061 | 0.059 | = | 0.068 |
| UPCOST 15m h5 | single (random forest) | log-loss skill | 0.020 | 0.021 | = | 0.016 |

"Best indiv." is Prompt #10's frozen model for the pair; "=" means the frozen
spec is that model.

- **Where the rule kept an ensemble it held up**: it beat its simple average
  and the best individual on 2022- by the same small margins as in development
  (+0.0005 to +0.009 over the best individual), and its worst year was no worse
  than theirs.
- **Where it kept one model, averaging was not a free lunch either way**: far
  worse for reversion (diluted and under-confident, ECE 0.074 / 0.063 against
  0.013 / 0.014), within 0.001 for 5m direction / return and 15m reversion
  c = 0.5, and *better* at 15m for volatility (+0.006), absolute move (+0.010)
  and direction (+0.001) - there Prompt #10's frozen single models ranked
  below CatBoost and LightGBM / Standard on 2022-, after differing from them by
  ~0.01 or less in development.
- Direction beyond the recent base rate: +0.007 (5m), +0.001 (15m). Future
  return decayed as in Prompt #10. 2026 is the weakest year for every
  volatility-type target. Reversion and residual reduction remain mostly
  mechanical (sign-flip null; untested nulls for the target-set models).

---

## Dataset versioning

Every result can be traced to the exact data behind it:

```
raw file            blake2b fingerprint of the source
  -> tick dataset   ticks-<hash>        over every partition's content digest and row count
    -> bars         bars-<tf>-<hash>    records the tick version it was built from
      -> features   regfeat-<tf>-w<N>-<hash>   records the bar version and settings
```

Each research `summary.json` embeds this lineage under
`provenance.dataset`, with coverage (first/last timestamp, rows, partitions).
A stale link is refused rather than read: bars refuse an incomplete tick
dataset, features refuse bars of another version, and `--resume` refuses a
study whose recorded versions differ. A result computed from an incomplete
dataset carries `PARTIAL DEVELOPMENT SAMPLE — NOT FULL RESEARCH RESULT` in its
own provenance.

---

## Directory structure

```
xauusd-quant/
├── config/
│   ├── data.yaml               # pipeline configuration
│   ├── research.yaml           # statistical layer
│   ├── regression.yaml         # rolling-regression layer
│   ├── ou.yaml                 # Ornstein-Uhlenbeck layer
│   ├── spectral.yaml           # spectral layer
│   ├── wavelet.yaml            # wavelet / time-frequency layer
│   ├── regimes.yaml            # regime layer
│   └── logging.yaml            # dictConfig: rich console + rotating file
├── data/
│   ├── <raw tick file>.csv     # the source, read-only, never modified
│   ├── raw/                    # optional home for additional raw files
│   ├── parquet/                # canonical ticks, year=YYYY/month=MM, _manifest.json
│   ├── bars/                   # 1m/ 5m/ 15m/ 30m/ 1h/, year=YYYY, _manifest.json each
│   ├── features/               # versioned regression features, timeframe=/window=;
│   │                           # spectral/, wavelet/, regime_inputs/, regime/ (live-safe states)
│   ├── regime_models/          # every walk-forward refit, immutable (HMM_5M_K4_00017.json)
│   └── metadata/               # every JSON report
├── src/xauusd_quant/
│   ├── cli.py                  # argparse CLI (`xq`)
│   ├── data/
│   │   ├── inspector.py        # large-file profiling, byte-offset seeking
│   │   ├── source_index.py     # per-month byte runs, content hashes, ordering report
│   │   ├── schema.py           # canonical schema + column detection
│   │   ├── timezones.py        # timezone policy (never silent)
│   │   ├── validator.py        # 14 checks, flags only
│   │   ├── cleaner.py          # drop policy + audit trail
│   │   ├── converter.py        # partition-safe, transactional CSV -> Parquet
│   │   ├── dataset_validation.py   # independent verification + coverage
│   │   ├── versioning.py       # dataset lineage embedded in every result
│   │   ├── resampler.py        # ticks -> OHLC bars, versioned
│   │   ├── diagnostics.py      # bar coverage and gap analysis
│   │   ├── metadata.py         # dataset-level statistics
│   │   └── loader.py           # DuckDB range queries
│   ├── features/
│   │   ├── config.py           # regression.yaml -> frozen dataclasses
│   │   ├── rolling_regression.py   # the rolling OLS engine
│   │   ├── store.py            # versioned, de-duplicated feature cache
│   │   ├── spectral_config.py  # spectral.yaml -> frozen dataclasses
│   │   ├── spectral.py         # causal rolling FFT engine and per-bar features
│   │   ├── spectral_bands.py   # frequency bins, bands, centroid
│   │   ├── spectral_entropy.py # entropy, flatness, concentration
│   │   ├── wavelet_config.py   # wavelet.yaml -> frozen dataclasses
│   │   ├── wavelet.py          # causal MODWT, DWT components, offline CWT, ridges
│   │   ├── wavelet_energy.py   # bands, physical periods, shares, white-noise baseline
│   │   ├── wavelet_entropy.py  # scale entropy
│   │   └── wavelet_causal.py   # rolling causal features and their schema
│   ├── models/
│   │   ├── config.py           # ou.yaml -> frozen dataclasses
│   │   └── ornstein_uhlenbeck.py   # AR(1) <-> OU mapping, rolling fits, simulators
│   ├── regimes/
│   │   ├── config.py           # regimes.yaml -> frozen dataclasses
│   │   ├── dataset.py          # integrity gate, the causal input table, feature specs
│   │   ├── preprocessing.py    # frozen robust scaler, coverage, missing-feature policy
│   │   ├── emissions.py        # Gaussian log-densities with exact marginalisation
│   │   ├── clustering.py gmm.py hmm.py  # K-Means, EM mixtures, blocked HMM filter
│   │   ├── fitting.py          # one fitted model + its scaler; live-safe inference
│   │   ├── causal_inference.py # walk-forward refits, per-bar features, schema guard
│   │   ├── state_alignment.py transitions.py diagnostics.py
│   │   ├── registry.py         # immutable model IDs, content-hash deduplication
│   │   └── synthetic.py changepoints.py
│   ├── research/
│   │   ├── regime_analysis.py  # offline fits, baselines, hold-out, state description
│   │   ├── regime_outcomes.py  # outcomes by state, incremental models A / A+states
│   │   ├── regime_stability.py # refits, schemes, eras, the K rule
│   │   ├── regime_nulls.py     # shuffles, block bootstraps, pipeline nulls, controls
│   │   └── regime_reports.py regime_plots.py
│   ├── research/
│   │   ├── returns.py distributions.py autocorrelation.py volatility.py
│   │   ├── stationarity.py rolling.py conditional.py intraday.py spread.py
│   │   ├── residuals.py residual_stationarity.py residual_decay.py residual_extremes.py
│   │   ├── ou_estimation.py    # per-bar OU table, static fits, controls
│   │   ├── ou_decay.py         # forecasts, extremes, realised vs fitted decay
│   │   ├── ou_diagnostics.py   # innovation diagnostics
│   │   ├── ou_stability.py     # half-life stability, years, quarters, eras
│   │   ├── ou_conditional.py   # volatility, trend, R^2, hour and session
│   │   ├── fft_analysis.py     # single-window spectra, PSD, positive controls, benchmark
│   │   ├── spectral_nulls.py   # the real series and its four null controls
│   │   ├── spectral_stability.py   # persistence, time, regimes, intraday
│   │   ├── spectral_phase.py   # circular statistics, phase buckets and continuation
│   │   ├── spectral_reconstruction.py  # in-window fit, out-of-window extrapolation
│   │   ├── spectral_ic.py      # information coefficients with period stability
│   │   ├── research_ledger.py  # every tested hypothesis and its verdict
│   │   ├── study_io.py         # shared study writer, JSON cleaning, provenance
│   │   ├── spectral_reports.py spectral_plots.py
│   │   ├── wavelet_analysis.py # positive controls, boundary study, slices, vs FFT
│   │   ├── wavelet_stability.py    # persistence, runs, drift, bursts
│   │   ├── wavelet_predictiveness.py   # IC, extremes, nested models A-D
│   │   ├── wavelet_reports.py wavelet_plots.py
│   │   └── reports.py regression_reports.py ou_reports.py plots.py ...
│   └── utils/
│       ├── config.py           # typed, validated YAML configuration
│       ├── logging.py          # logging setup
│       ├── clock.py            # timezone conversion
│       └── paths.py            # path resolution, atomic writes
├── research/                   # notebooks 01-31: load, call, display
├── results/                    # statistical_research/, regression_research/,
│                               # ou_research/, spectral_research/, wavelet_research/,
│                               # regime_research/, research_ledger.parquet, _archive/
├── scripts/                    # thin CLI wrappers, benchmark_regression.py,
│                               # wavelet_baseline_robustness.py (post-hoc WAVE-P-001),
│                               # regime_grid.ps1, regime_feature_geometry.py (post hoc)
├── tests/                      # synthetic-data unit tests + real-data tests
├── notebooks/  experiments/    # research space (git-ignored)
├── WORKLOG.md                  # history of every stage, run, fix and decision
└── pyproject.toml
```

`data/` is git-ignored. Raw market data is never committed and never modified.

### Reports written to `data/metadata/`

| File | Produced by | Contents |
|---|---|---|
| `raw_dataset_report.json` | `inspect` | per-file profile from head/tail/samples |
| `source_index.json` | `source-index`, `convert` | per-month byte runs, hashes, line counts |
| `source_ordering_report.json` | `source-index` | backward jumps explained, with context lines |
| `validation_report_standalone.json` | `validate` | exact check counts over raw input |
| `conversion_manifest.json` | `convert` | copy of the dataset's `_manifest.json` |
| `validation_report.json` | `convert` | exact check counts from the conversion pass |
| `cleaning_report.json` | `convert` | rows in/out/dropped, reason attribution |
| `dataset_validation.json`, `tick_coverage.csv` | `verify-dataset` | integrity checks, coverage and gaps by year |
| `dataset_metadata.json` | `metadata` | full-dataset statistics |
| `bar_diagnostics.json` | `diagnose` | per-timeframe coverage and gaps |
| `regression_feature_coverage.json` | `build-features` | cached windows, versions, fits by year |
| `cleanup_log.json` | recovery | every deleted derived file: path, size, reason |

---

## Data assumptions

These were **measured from the actual file**, not assumed. Re-derive them with
`xq inspect` and `xq detect-schema`.

### Source format

| Property | Value |
|---|---|
| Layout | single CSV, `DateTime,Bid,Ask,Volume,Spread` |
| Size | 33.75 GiB, 731,928,177 data lines |
| Timestamp | `YYYYMMDD HH:MM:SS.mmm` — fixed width, millisecond precision |
| Coverage | 2003-05-05 03:01:03.421 → 2026-09-18 23:59:59.079 |
| Order | **not sorted**: from 2021-01-04 each week's opening hour or two is repeated verbatim (see [results](#what-the-data-turned-out-to-be)) |
| Encoding / terminator | UTF-8, no BOM, LF |

The source `Spread` column equals `Ask - Bid`; it is kept as `spread_source` so
the pipeline can check that, and `spread` is always derived.

### Timezone

The file carries **naive** timestamps with no zone information. The configured
policy is `anchored_dst`: **broker server time = America/New_York wall clock
+ 7 hours**, i.e. UTC+2 in US winter and UTC+3 in US summer, switching on the
**US** DST calendar.

Evidence (all reproducible from the file), for the **whole 23-year span**:

- From 2011, US 08:30 America/New_York releases (NFP, CPI) produce their
  tick-rate spike at **15:30** in this file in *both* DST halves of the year —
  so the offset from New York is constant at +7 h. That still holds during the
  weeks when the EU and US DST calendars disagree, which rules out every
  `Europe/*` zone.
- Before 2011 the feed is too sparse for release spikes. What settles the early
  era is the CME settlement lull (17:00-18:00 New York, the quietest hour of
  the gold day): it sits at broker hour 0 in every era and in both seasons.
- The week opens at Sunday 18:00 New York = Monday 01:00 broker time, the
  standard weekly gold open.

Re-derive this before trusting it on a new export.

This combination has **no IANA equivalent**, which is why the policy is modelled
as an offset from an anchor zone rather than a zone name. It is the common
MT4/MT5 broker "server time" convention.

Handling:

- `timestamp` is always the original wall clock, **never rewritten**, and is
  what partitioning and bars are based on.
- `timestamp_utc` is a separate, optional, derived column
  (`timezone.emit_utc_column`).
- US DST switches at 02:00 New York on a Sunday = 09:00 broker time, which is
  always inside the weekend close, so no real tick is ambiguous or non-existent.
- Set `timezone.mode: naive` to refuse to claim any zone at all.

### Trading schedule

- Daily break **00:00–01:00** broker time (the CME settlement hour).
- **January 2012 is a structural change, not a clock change.** Before 2012 the
  00:00-01:00 hour carries ~1.2 % of each day's ticks; from 2012-01 the
  provider stops exporting it, so the lull becomes a hard daily gap and the
  weekend goes from 48 h (Monday 00:00 open) to 49 h (Monday 01:00 open).
- Holidays produce early closes (e.g. Memorial Day 2024-05-27 stopped at 21:28).

These are configured under `diagnostics.session` and used *only* to classify
gaps. They never filter or modify data.

### Data quality

Over all 731.9M lines: zero malformed rows, zero unparseable timestamps, zero
missing or non-positive prices, zero crossed quotes. The only defect is the
weekly repetition described above (2,683,808 exact duplicate rows, all dropped
by the configured policy and attributed). Early years are far sparser than
recent ones - 2.8M ticks for May-December 2003 against 71.4M for 2025 - and
2003-2010 has frequent intraday gaps, so a large `holiday_or_unknown` count at
1m in that era is real, not a bug.

### Interpretation caveats

- **`Volume` is not traded volume.** 358 distinct values, all multiples of ten,
  in the 90–1000+ range. This is broker-reported liquidity/size, characteristic
  of an OANDA-style feed. Do not treat it as executed contracts.
- **This is a throttled quote feed, not raw full-depth ticks.** The modern
  feed is quantised to ~50 ms (roughly a 20 Hz sample of the quote stream); the
  early feed is seconds-grained. Microstructure research below those scales is
  not supported by this data, and pooling eras mixes different sampling.
- Prices are stored as `float64`. Gold quotes carry three decimals, well within
  float64 exactness at these magnitudes.

---

## Design guarantees

**No look-ahead.** Bars aggregate `[t, t+Δ)` only. A boundary tick belongs to
the next bar. No aggregate is shifted, filled or carried across bars.

**Raw data is never modified.** The pipeline only ever reads
`raw_data_path`. All output goes to separate directories.

**No invented data.** Prices are never interpolated, forward-filled or
fabricated, under any configuration. Missing bars stay missing.

**Bid and ask are preserved.** Both are carried through ticks and bars
(`first_bid`/`first_ask`/`last_bid`/`last_ask`), because the eventual backtester
must fill long entries at the ask and long exits at the bid, and vice versa.
`mid` and `spread` are *derived*; a source spread column, if present, is kept
separately as `spread_source` and never overwrites the derived value.

**Nothing is deleted silently.** The validator only flags. The cleaner drops
only what `cleaning.drop_*` explicitly enables, counts every removal, and
attributes each dropped row to the first matching reason so per-reason counts
sum exactly to `rows_dropped`. Independent per-check tallies are reported
alongside. Defaults drop only structurally unusable rows (no parseable
timestamp, missing quote, non-positive price, non-finite value) — crossed
quotes, extreme spreads and duplicates are kept and flagged.

**Reproducibility.** Same input + same configuration ⇒ same *content*. Each
partition is stably sorted before writing and carries a blake2b content digest
(per-column, over dense values and a null mask) that is independent of how rows
were chunked — so a resumed run and a full run agree. Every report embeds a
`config_fingerprint`, a stable hash of the settings that affect output, and
every research result embeds its [dataset lineage](#dataset-versioning).
Physical Parquet layout may differ between runs; see
[limitation 10](#known-limitations).

**A partition is never half-written, and the live dataset is never deleted
first.** Files are written as `.partial` and renamed only after verification;
rebuilds happen beside the live dataset and are swapped in only when complete
and validated. A crash at any point leaves either the old dataset or the new
one, never a mixture.

**The source's order is never assumed.** Each month is read from its indexed
byte runs, validated as a whole month and sorted, so a late row lands in its
own month wherever it sits in the file.

**Scalability.** Nothing requires the dataset to fit in memory. Conversion,
resampling and diagnostics all stream. The query layer makes full loads the
explicit, opt-in path.

**Honest reporting.** Sampled figures are labelled as estimates; full-pass
figures as exact; approximate quantiles as approximate. `dataset_metadata.json`
warns when validation counts cover only part of the raw input.

---

## Known limitations

1. **The source index needs fixed-position year and month.** `%Y` and `%m`
   must sit at fixed character offsets of `timestamp_format`, which is how one
   pass assigns every line to its month without parsing it. A format without
   that property is refused rather than guessed at.
2. **Bar timeframes must divide a day evenly.** Bars are built one month at a
   time, so a timeframe that does not divide 86400 s would straddle a partition
   boundary. `validate_timeframes` rejects such timeframes rather than producing
   split bars. All configured timeframes (1m–1h) satisfy this.
3. **Quantiles are approximate by default.** `metadata.exact_quantiles: true`
   switches to exact `quantile_cont` at significant cost. The method used is
   always recorded in the output.
4. **Bar diagnostics and the research layers load one timeframe at a time.**
   7.9M 1-minute bars are a few hundred MB for diagnostics, but an OU study at
   1m holds several full-length tables and needs several GB (its extrapolated
   peak is ~4.5 GB); the other timeframes need at most ~1.3 GB. These are the
   components that are not streaming.
5. **The session schedule is configuration, not detection.** Gap classification
   is only as good as `diagnostics.session`. A different broker or instrument
   needs those values updated, or every gap lands in `holiday_or_unknown`.
6. **The timezone conclusion is empirical.** It is strongly evidenced (see
   above) but derived from market microstructure, not from a statement by the
   data provider. Confirm with your broker before relying on exact UTC
   alignment for event studies.
7. **`Volume` semantics are unverified.** Treated as an opaque broker-reported
   quantity; it is summed into bars but no meaning is attached to it.
8. **No multi-instrument support.** The schema and configuration assume one
   instrument per dataset.
9. **Querying during a conversion sees a mixture of months.** Files are only
   ever visible complete (`.partial` does not match `*.parquet`), but a resumed
   conversion replaces months one at a time, so a query spanning them can read
   some months from before the run and some from after. There is no locking.
   Let `convert` finish before querying.
10. **Parquet files are reproducible in content, not byte-for-byte.** Encodings
    and row-group boundaries are the writer's choice, so the files themselves
    can differ between runs even when every value is identical. The per-partition `digest` in the
    manifest is therefore a *content* fingerprint: one hasher per column, fed
    the dense values plus an explicit null mask in row order, combined in
    schema order at close. It is invariant to batch, flush and row-group
    chunking. **Compare digests, not `sha256` of the Parquet files.**
11. **Rolling OU estimates are biased, and the bias depends on the window.** A
    windowed AR(1) fit underestimates `b` (Dickey-Fuller / Kendall bias), most
    at small M, so rolling half-lives run short and even a random walk shows a
    finite one. That is why every OU figure sits beside `reference_ou` (the bias
    on a true OU) and `control_random_walk` (the bias with no reversion at all).
    No bias correction is applied.
12. **Twenty-three years are not one regime.** The feed changes from
    seconds-grained and sparse (2003-2010) to ~50 ms throttled, the daily break
    changes in 2012, and gold's volatility level moves by multiples. Pooled
    full-history statistics are reported, but always beside yearly, quarterly
    and era breakdowns; read a pooled number as an average over regimes.

---

## Stage #12: offline execution and trading economics

Prompt #12 (2026-10-03) authorizes offline simulation and a separate fixed
decision policy, costs, PnL and equity. The forecast contracts are unchanged.
See [Stage #12 interfaces and audit](docs/stage12_execution.md) for actual
commands, supported order semantics, assumptions, evidence blockers and tests.
Historical feature-selection and ensemble results are retained: their global
adaptive choices prevent promoting the existing 2011-2021 scores as clean
out-of-sample evidence. 2022+ was already inspected; no fresh final test is
claimed. Broker access, demo/live trading and deployment remain prohibited.

## Non-goals

The list below records the historical Stages #1-#11 scope. Stage #12 now
implements offline entries/exits, fixed quantity, PnL, tick execution,
declared costs and backtesting as documented above. The other exclusions
(especially broker/live access and performance-based model selection) remain.

Historically **not** implemented before Stage #12, and not to be added here
without a decision to move to the next stage:

regression strategies · Z-score strategies · Ornstein–Uhlenbeck strategies ·
spectral, phase- or wavelet-based signals · regime-based signals or strategies ·
Hurst exponent · variance ratio · neural networks · LSTM · Transformers ·
alpha-combination weights by trading performance · probability trading
thresholds · deep or Bayesian-nonparametric regime models · trading signals ·
BUY / SELL logic · entries/exits · position sizing · stops · PnL · realistic
tick execution · slippage models · spread-adjusted strategies · risk engine ·
execution simulation · portfolio management · backtesting · parameter
optimization by trading performance · Monte Carlo · Exness live data · MT5
integration / live feeds · demo trading · real trading · production
deployment.

Note the distinction in that list: **regression, Ornstein-Uhlenbeck, spectral,
wavelet and regime *strategies*** are non-goals, while the corresponding
*research* layers above are implemented. The wavelet layer's nested models A-D
and the regime layer's models A / A+states are small linear / logistic
research models that measure incremental information; they are not the ML
system. The regime layer's K-Means, mixtures and HMM describe states, and its
walk-forward exists so those descriptions are causal - it evaluates no trade.
The OU layer estimates rolling reversion speeds and half-lives purely
descriptively, always beside a random-walk control and an exact-OU reference.
The feature factory, the alpha research and the feature selection are
implemented research layers too: their statuses and feature sets are
statistical descriptions, and the selection's ridge / Lasso / elastic-net /
L1-logistic / PCA models are diagnostics of forward information. The
supervised layer (random forests, XGBoost, LightGBM, CatBoost and linear
baselines) is implemented as *prediction research*: its models output
probabilities and expected values, are chosen by out-of-sample predictive
skill under pre-registered rules, and are evaluated once on the reserved
period after freezing. The ensemble layer (averages, weights, stacked
logistic / ridge meta-models, conditional and trailing weights) is prediction
research in the same sense: its weights come from out-of-sample predictive skill
on earlier blocks and error diversity, never from a trading result, and its
output - the prediction contract - has no decision field. The line is that
nothing here defines a trade or applies a cost, no number of states, feature
set, model or ensemble is chosen by trading performance, and no probability is
turned into a decision.

`notebooks/`, `results/` and `experiments/` exist for later analysis. Core
logic belongs in `src/`, never in a notebook.

---

## Development

```bash
pytest                       # 984 unit tests + 19 real-data tests
pytest -m "not realdata"     # unit tests only - no 34 GB needed
pytest -m realdata           # against the converted dataset
ruff check src tests scripts
mypy
```

1003 tests in 74 modules; the unit tests use small synthetic datasets and
never touch the real file. The full suite, real-data tests included, runs in
about 22.5 minutes on the development PC (i7-6700) and peaks at ~3.9 GB
resident (up to ~5 GB committed).

**Data foundation**

| Module | Covers |
|---|---|
| `test_schema.py` | column detection, canonical schema, mid/spread derivation |
| `test_timezones.py` | the anchored-DST rule, including EU/US mismatch weeks |
| `test_validator.py` | all 14 checks, cross-batch state, gap detection |
| `test_cleaner.py` | drop policy, first-match attribution, audit integrity |
| `test_source_index.py` | format offsets, every line in exactly one partition, late rows, exact ordering statistics, repeats and ties, CRLF, cache invalidated by content hash |
| `test_converter.py` | partitioning, resume, digests, determinism, reports |
| `test_conversion_integrity.py` | out-of-order and late rows, the duplicate policy, partition-level resume, a crash mid-write, transactional overwrite and interrupted swaps, disk-space refusal, every source line accounted for |
| `test_dataset_validation.py` | tampered files, unknown files, lost months, tick-level gap classes |
| `test_resampler.py` | OHLC correctness, bar boundaries, no look-ahead |
| `test_loader.py` | range queries, guards, metadata, gap classification |
| `test_inspector.py` | sampling, byte-offset search, dialect sniffing |
| `test_config.py` | strict validation, env expansion, fingerprinting |
| `test_paths_and_overwrite.py` | atomic writes, surgical `--overwrite` |

**Statistical research**

| Module | Covers |
|---|---|
| `test_returns.py` | return definitions, session gaps, forward returns as outcomes |
| `test_distributions.py` | moments, tail comparison, normality tests |
| `test_autocorrelation.py` | ACF, effect-size labelling, Ljung-Box vs statsmodels |
| `test_volatility.py` | estimators, regimes, opt-in annualisation |
| `test_stationarity.py` | ADF and KPSS kept separate, segmenting |
| `test_rolling.py` | rolling moments against closed-form values |
| `test_conditional.py` | conditioning without leakage |
| `test_intraday.py` | sessions, hour-of-day, the Int8 overflow regression |

**Rolling regression**

| Module | Covers |
|---|---|
| `test_rolling_regression.py` | the OLS engine, numerics, degenerate windows |
| `test_regression_no_leakage.py` | exact-equality look-ahead checks, null vs NaN |
| `test_residuals.py` | both Z-scores, residual distribution and ACF, buckets |
| `test_residual_decay.py` | AR(1) recovery, controls, censoring, MARE, bootstrap |
| `test_feature_store.py` | loaded = freshly computed, stale caches refused, column subsets |

**Ornstein-Uhlenbeck**

| Module | Covers |
|---|---|
| `test_ou_estimation.py` | static fit against statsmodels, the Dickey-Fuller form, every rolling window against an independent fit, the mapping's closed forms, equilibrium estimated not forced |
| `test_ou_half_life.py` | exact half-lives, validity classes, the reportable cap, censoring instead of averaging, no confident half-life for a random walk |
| `test_ou_simulation.py` | exact innovation variance, stationarity, ACF = b^h, parameter recovery, small-window bias, random-walk controls and their window scaling |
| `test_ou_no_leakage.py` | appending future bars changes no past estimate |
| `test_ou_edge_cases.py` | constant, perfect-fit, negative-b, near-zero-b, b ≥ 1, NaN/inf, tiny and trending series |
| `test_ou_research.py` | first passage, censoring, forecasts, conditioning with its control, the report end to end, `--resume` and report versions, one process per study, a failing study not stopping the grid |

**Spectral**

| Module | Covers |
|---|---|
| `test_fft.py` | conventions (DC, Nyquist, units), single and paired sinusoids recovered in frequency, amplitude and phase, leakage vs components, resolution, aliasing, PSD scaling, the missing-bar policy, constant input, warm-up |
| `test_spectral_entropy.py` | entropy, flatness and concentration of known spectra; a tone against white noise; no stable dominant cycle in noise; bands partition the power |
| `test_phase.py` | circular mean and resultant length, Rayleigh test, circular sectors, phase validity, phase continuation of a real cycle vs noise |
| `test_reconstruction.py` | exact inverse transform, Parseval explained variance, periodic extrapolation of a true cycle vs noise |
| `test_spectral_no_leakage.py` | appending bars or a future spike changes no earlier feature; chunk size never matters |
| `test_spectral_controls.py` | every null the right length and reproducible; shuffled returns keep the distribution; bootstrap blocks are real blocks; the detrended walk's spectrum is red |
| `test_spectral_research.py` | the study end to end: every output, summary keys, feature partitions, ledger rows, `--resume`, the integrity gate; a zero-IQR null still counts in a verdict; IC chance rates and test counts |
| `test_spectral_cli.py` | window choice required, raw price refused, heavy timeframes need `--allow-heavy` |

**Wavelet**

| Module | Covers |
|---|---|
| `test_wavelet.py` | filter lengths and levels, the MODWT energy identity and agreement with `pywt.swt`, causal = circular away from the start, bit-for-bit prefix invariance, NaN reach, DWT reconstruction x = A_J + sum D_j, scale-to-period mapping, cone of influence, a known sinusoid in the CWT, chirp ridges vs noise ridges, padding dependence of a windowed DWT |
| `test_wavelet_energy.py` | physical band periods across bar lengths, empty physical groups, shares, concentration, active scales, the exact white-noise baseline vs simulation, a sinusoid in its band |
| `test_wavelet_entropy.py` | known shares, single scale vs broadband noise |
| `test_wavelet_causality.py` | appending wild future bars changes nothing, a future spike does not leak, prefix invariance at random cut-offs, every feature from a 2N live buffer, schema refuses offline / unregistered columns, declared lookbacks, missing values and gaps, constant input, the causal default beats windowed padding |
| `test_wavelet_nulls.py` | the null list and block-size variants, roles, reproducible and distinct bootstraps, white noise has no stable long-lived dominant scale |
| `test_wavelet_synthetic.py` | a temporary cycle localised (FFT smears it), a chirp followed, two scales separated |
| `test_wavelet_predictiveness.py` | chronological embargoed folds, an informative block raises out-of-sample R^2 and noise does not, missing indicators, logistic = statsmodels, AUC, the post-hoc baseline extensions (trailing, one-hot hours, input untouched) |
| `test_wavelet_research.py` | the study end to end: every table, summary keys, provenance, causal feature partitions and manifest, registered hypotheses and ledger rows, offline outputs labelled, `--resume`, the FFT-feature gate, models A-D on a multi-year source (the `original` baseline is the registered model bit for bit), an undefined (NaN) statistic is `untested` in every rule, WAVE-P-001 verdicts against nulls with the same widened baseline |
| `test_wavelet_cli.py` | window choice required, raw price refused, dry run estimates only, heavy timeframes need `--allow-heavy`, `--controls` writes its tables |

**Regimes**

| Module | Covers |
|---|---|
| `test_hmm.py` | the blocked recursions equal the textbook ones for many K, lengths and block sizes; likelihood and Viterbi equal brute-force enumeration; a bar without evidence equals marginalising every feature; stationary distribution and durations; Baum-Welch recovers a known HMM; a serialised model gives identical probabilities |
| `test_gmm.py` | log-densities equal SciPy (padded chunks included), EM recovery, covariance types, information criteria, posteriors, the registry round trip, K-Means blobs and partially observed rows |
| `test_state_alignment.py` | a permuted model aligns back, ambiguity under perturbation, Bhattacharyya invariant to an affine map, canonical order, agreement measures ignore renumbering, eta^2, degeneracy flags |
| `test_regime_dataset.py` | every feature registered, documented and causal; unknown or duplicate features refused; rolling percentile = brute force with ties and gaps; causal transforms ignore the future; the redundancy rule; scalers from training rows only; the integrity gate on the real 1h data |
| `test_regime_causality.py` | filtered probabilities ignore future bars bit for bit while smoothed ones change; walk-forward outputs prefix-invariant for HMM, GMM and K-Means; scaler and refits see only the past; regime age from past labels; the missing-feature policy; immutable models that restore exactly |
| `test_regime_synthetic.py` | a known 3-state HMM recovered and K = 3 chosen; one Gaussian regime gives no persistent states; heavy tails make mixtures want more components; a continuum cut into neighbouring bands; row shuffles destroy persistence; the K rule stops at the first failing step and reads an unavailable step as untested; an undefined increment is untested; PELT change points |

**Feature factory and predictive research (Prompt #8)**

| Module | Covers |
|---|---|
| `test_feature_leakage.py` | every registered feature and interaction is prefix-invariant bit for bit, wild future bars change nothing before them, a full-sample z-score is caught |
| `test_target_alignment.py` | targets are forward outcomes aligned to their bar, in their own namespace, never joined into the matrix |
| `test_feature_registry.py` | registry fields, feature ids, windows and warm-ups, invalid combinations, the non-causal catalog |
| `test_information_coefficient.py` | monthly sufficient statistics equal direct correlations, Newey-West errors, first-value centring keeps causal slices exact |
| `test_alpha_decay.py` | decay curves and the information half-life of known processes |
| `test_feature_redundancy.py` | clusters, representatives, non-monotone pairs, candidate manifests |
| `test_alpha_nulls_and_ranking.py` | max-T circular shifts, noise features, the pipeline veto, statuses, permanent test ids |
| `test_feature_factory.py` | OHLC pipeline nulls, round trips, deterministic builds, exact-timestamp joins, quality tables |
| `test_feature_research.py` | the stages end to end on synthetic data, stamps and resume |

**Feature selection (Prompt #9)**

| Module | Covers |
|---|---|
| `test_feature_filters.py` | each exclusion reason on a feature built for it, rare binary flags kept, drift classified not excluded, no outcome read |
| `test_redundancy_selection.py` | correlated features cluster, the lexicographic representative rule, parameter-family reduction, effective rank and VIF |
| `test_mrmr.py` | mRMR prefers the relevant non-redundant feature, IC ranking takes near-copies, the hard limit, both schemes |
| `test_selection_stability.py` | quarter resamples, year subsets, the probe-stopping rule, expected frequencies, top-k saturation |
| `test_pca_causality.py` | a fitted PCA is unchanged by appended wild rows, frozen training CDFs, moments add, ridge = least squares, Lasso / elastic net / L1 logistic recover known supports |
| `test_feature_manifest.py` | deterministic order, immutability, tamper detection, the stored matrix reproduced in manifest order, stale versions refused |
| `test_selection_isolation.py` | reserved target files never opened (corrupted on purpose), purging by horizon, the guard, evaluation outcomes never change a training selection |
| `test_live_reconstruction.py` | streaming = batch through a truncated buffer, an altered stored value is caught |

**Supervised predictive research (Prompt #10)**

| Module | Covers |
|---|---|
| `test_ml_splits.py` | no validation date in training or the inner slice, the purge covers h + embargo, rolling look-back, refit schedules, a too-short span refused, the final fold ends at the last labelled row |
| `test_ml_preprocessing.py` | parameters from the fold's fitting rows only, feature-order mismatches fail clearly, missing values handled identically in batch, row by row and after a round trip, clipping |
| `test_model_training.py` | every classifier and regressor scores exactly its block, bit-for-bit reproducibility, the built-in structure found while shifted labels and permuted features find nothing, noise columns, booster NaN modes, time-decay weights |
| `test_calibration.py` | calibrators fitted on the purged inner slice only, Platt repairs a miscalibrated score out of sample, isotonic needs enough rows and stays monotone, reliability bins, round trips |
| `test_model_serialization.py` | every family reloads with identical predictions, artifacts detect tampering, frozen specs are immutable and refuse edits, model ids |
| `test_ml_no_leakage.py` | outcome-like, time and row columns refused, a leakage canary rejected whatever its name, the design gate, derived labels never reach past the purged end |
| `test_final_test_isolation.py` | development loading never opens reserved year files (corrupted on purpose), reserved rows refused without a frozen spec, a second look refused without a logged reason, no selection module imports the final test |
| `test_ml_streaming.py` | streaming predictions equal batch predictions for boosters and linear models; a stored value that differs from its causal recomputation fails |
| `test_ml_freeze.py` | block counts against the baseline, the one-SE simplicity rule, an SE of exactly zero read as a consistent difference, refreezing idempotent and a changed decision refused |

A further 19 tests run against the converted dataset and skip automatically
when it is absent or incomplete: 15 in `test_realdata.py` (sortedness,
partition purity, derived-column exactness, bars recomputed field-by-field from
raw ticks with no look-ahead, the UTC offsets, manifest/disk agreement), the
regime integrity gate on the 1h data in `test_regime_dataset.py`, and the 1h
feature-factory checks in `test_feature_realdata.py`.

`pytest` runs with `-W error::DeprecationWarning` scoped to this package, so a
deprecation in our own code fails the suite rather than accumulating quietly.

### Conventions

- Type hints and docstrings throughout; `pathlib`, never string paths.
- Frozen dataclasses for configuration; no global mutable state.
- Explicit errors with actionable messages.
- Logging reports progress and aggregates — never per-tick lines. A full
  34 GB conversion produces a few hundred log lines.
