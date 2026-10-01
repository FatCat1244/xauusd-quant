# `data/raw/`

Optional home for raw tick files. Nothing in the pipeline requires files to be
here: `raw_data_path` in `config/data.yaml` may point anywhere, including
another drive or a network share.

The raw dataset for this project currently lives one level up, at
`data/2026.9.11XAUUSD_oanda-TICK-No Session.csv` (~13.8 GiB), and is referenced
by path rather than copied.

Raw data is **never** committed to git and is **never** modified by the
pipeline. All output goes to `data/parquet/`, `data/bars/` and
`data/metadata/`.
