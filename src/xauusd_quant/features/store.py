"""Versioned, de-duplicated cache of rolling-regression features.

Layout, per timeframe::

    data/features/regression/timeframe=5m/
        base.parquet                    timestamp, close, log_price, log_return,
                                        trailing_volatility - identical for every window
        window=128/features.parquet     only what depends on the regression window
        window=256/features.parquet
        _manifest.json                  versions, row counts, coverage

The previous cache stored the window-independent columns once per window, and
the aliased ``residual_zscore_rolling`` column twice. Here each is stored once;
``fitted_price``, ``residual_pct`` and the alias are re-derived on load, exactly
as :func:`~xauusd_quant.features.rolling_regression.rolling_regression_features`
derives them, so :meth:`RegressionFeatureStore.load` returns the same frame.

Versioning
----------
Every cached window records the bar dataset version it was computed from and a
fingerprint of the settings that change feature values. :meth:`load` refuses a
cache whose bars or settings no longer match (:class:`StaleFeaturesError`), so
features from one dataset version can never be read against another - the
failure mode that stale 19-month artefacts beside a 23-year dataset would be.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from ..data.resampler import bar_dataset_version, load_bar_manifest
from ..utils.clock import utc_now_iso
from ..utils.config import Config
from ..utils.logging import format_duration, get_logger
from ..utils.paths import atomic_write_text, ensure_dir, human_bytes
from .config import RegressionConfig
from .rolling_regression import rolling_regression_features

__all__ = [
    "BASE_COLUMNS",
    "RegressionFeatureStore",
    "StaleFeaturesError",
    "feature_fingerprint",
]

LOGGER = get_logger("features.store")

STORE_MANIFEST = "_manifest.json"
BASE_FILE = "base.parquet"
WINDOW_FILE = "features.parquet"

#: Columns that do not depend on the regression window.
BASE_COLUMNS: tuple[str, ...] = (
    "timestamp", "close", "log_price", "log_return", "trailing_volatility",
)
#: Columns re-derived on load instead of being stored.
DERIVED_ON_LOAD: tuple[str, ...] = ("fitted_price", "residual_pct", "residual_zscore_rolling")


class StaleFeaturesError(RuntimeError):
    """The cached features were computed from other bars or other settings."""


def feature_fingerprint(regression: RegressionConfig) -> str:
    """Digest of exactly the regression settings that change feature values."""
    payload = regression.to_dict()
    chosen = {
        "schema_version": payload.get("schema_version"),
        "regression": payload.get("regression"),
        "zscore": payload.get("zscore"),
        "numerical": payload.get("numerical"),
        "volatility_window": (payload.get("conditioning") or {}).get("volatility_window"),
    }
    blob = json.dumps(chosen, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.blake2b(blob, digest_size=8).hexdigest()


@dataclass
class FeatureBuild:
    """What one ``build`` call wrote or reused."""

    timeframe: str
    bar_dataset_version: str | None = None
    windows_built: list[int] = field(default_factory=list)
    windows_reused: list[int] = field(default_factory=list)
    rows: int = 0
    bytes_written: int = 0
    duration_seconds: float = 0.0


class RegressionFeatureStore:
    """Build, verify and load the cached regression features."""

    def __init__(self, config: Config, regression: RegressionConfig) -> None:
        self.config = config
        self.regression = regression

    # -- paths ---------------------------------------------------------------
    def timeframe_dir(self, timeframe: str) -> Path:
        return self.regression.features_path / f"timeframe={timeframe}"

    def manifest(self, timeframe: str) -> dict[str, Any] | None:
        path = self.timeframe_dir(timeframe) / STORE_MANIFEST
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return data if isinstance(data, dict) else None

    # -- build ---------------------------------------------------------------
    def _bar_manifest(self, timeframe: str) -> dict[str, Any]:
        manifest = load_bar_manifest(self.config.bars_dir(timeframe))
        if manifest is None or manifest.get("status") != "complete":
            raise RuntimeError(
                f"{timeframe} bars have no complete manifest; run `xq build-bars` first. "
                "Features are only cached against a verified bar dataset."
            )
        return manifest

    def build(
        self,
        timeframe: str,
        windows: list[int],
        *,
        bars: pl.DataFrame | None = None,
        force: bool = False,
    ) -> FeatureBuild:
        """Compute and cache features for *windows*, reusing current ones."""
        started = time.perf_counter()
        bar_manifest = self._bar_manifest(timeframe)
        result = FeatureBuild(
            timeframe=timeframe, bar_dataset_version=bar_manifest.get("bar_dataset_version")
        )
        current = self._current_manifest(timeframe, bar_manifest)
        cached = set((current or {}).get("windows", {}))
        todo = [w for w in windows if force or str(w) not in cached]
        result.windows_reused = [w for w in windows if w not in todo]
        if todo and bars is None:
            from ..research.regression_reports import load_bars

            bars = load_bars(self.config, timeframe, None, None)
        for window in todo:
            assert bars is not None
            frame, diagnostics = rolling_regression_features(
                bars, window=int(window), config=self.regression, timeframe=timeframe
            )
            result.bytes_written += self.save(timeframe, int(window), frame, diagnostics)
            result.windows_built.append(int(window))
        result.rows = int((self.manifest(timeframe) or {}).get("rows") or 0)
        result.duration_seconds = time.perf_counter() - started
        LOGGER.info("[%s] feature store: built %s, reused %s in %s", timeframe,
                    result.windows_built, result.windows_reused,
                    format_duration(result.duration_seconds))
        return result

    def _current_manifest(
        self, timeframe: str, bar_manifest: dict[str, Any]
    ) -> dict[str, Any] | None:
        """The existing manifest if it matches these bars and settings, else None.

        A cache made from other bars or settings is stale as a whole, so its
        files are removed here rather than left beside the new ones.
        """
        directory = self.timeframe_dir(timeframe)
        manifest = self.manifest(timeframe)
        if (
            manifest is not None
            and manifest.get("bar_dataset_version") == bar_manifest.get("bar_dataset_version")
            and manifest.get("feature_fingerprint") == feature_fingerprint(self.regression)
            and (directory / BASE_FILE).exists()
        ):
            return manifest
        if directory.exists():
            for stale in [*directory.rglob("*.parquet"), directory / STORE_MANIFEST]:
                if stale.exists():
                    LOGGER.info("Removing stale feature cache %s", stale)
                    stale.unlink()
        return None

    def save(
        self,
        timeframe: str,
        window: int,
        frame: pl.DataFrame,
        diagnostics: dict[str, Any] | None = None,
    ) -> int:
        """Cache a feature frame computed from the COMPLETE current bar dataset.

        Returns the bytes written. Refuses a frame that does not cover every
        bar, because a cache must never describe part of the history.
        """
        bar_manifest = self._bar_manifest(timeframe)
        if frame.height != int(bar_manifest.get("rows") or -1):
            raise RuntimeError(
                f"{timeframe}: the frame has {frame.height:,} rows but the bar dataset has "
                f"{bar_manifest.get('rows')}; only full-history features are cached."
            )
        directory = ensure_dir(self.timeframe_dir(timeframe))
        manifest = self._current_manifest(timeframe, bar_manifest) or {}
        if not manifest:
            base = frame.select(BASE_COLUMNS)
            _atomic_parquet(base, directory / BASE_FILE)
            manifest = {
                "timeframe": timeframe,
                "bar_dataset_version": bar_manifest.get("bar_dataset_version"),
                "tick_dataset_version": bar_manifest.get("tick_dataset_version"),
                "feature_fingerprint": feature_fingerprint(self.regression),
                "regression_config_fingerprint": self.regression.fingerprint(),
                "rows": base.height,
                "first_timestamp": str(base["timestamp"][0]),
                "last_timestamp": str(base["timestamp"][-1]),
                "rows_by_year": _rows_by_year(base),
                "windows": {},
            }
        specific = frame.drop([*BASE_COLUMNS[1:], *DERIVED_ON_LOAD])
        path = directory / f"window={window}" / WINDOW_FILE
        _atomic_parquet(specific.drop("timestamp"), path)
        valid = frame.filter(pl.col("residual").is_not_null())
        windows = dict(manifest.get("windows", {}))
        windows[str(window)] = {
            "window": int(window),
            "path": path.relative_to(directory).as_posix(),
            "feature_version": _feature_version(
                timeframe, int(window), manifest["bar_dataset_version"],
                manifest["feature_fingerprint"],
            ),
            "rows": frame.height,
            "valid_fits": valid.height,
            "first_valid_timestamp": str(valid["timestamp"][0]) if valid.height else None,
            "valid_fits_by_year": _rows_by_year(valid),
            "degenerate_windows": (diagnostics or {}).get("degenerate_windows", 0),
            "bytes": path.stat().st_size,
            "columns": specific.drop("timestamp").columns,
            "generated_utc": utc_now_iso(),
        }
        manifest["windows"] = {k: windows[k] for k in sorted(windows, key=int)}
        manifest["generated_utc"] = utc_now_iso()
        atomic_write_text(directory / STORE_MANIFEST, json.dumps(manifest, indent=1) + "\n")
        LOGGER.info("[%s w=%d] features cached: %s rows, %s valid fits, %s",
                    timeframe, window, f"{frame.height:,}", f"{valid.height:,}",
                    human_bytes(path.stat().st_size))
        return path.stat().st_size

    # -- load ----------------------------------------------------------------
    def load(self, timeframe: str, window: int,
             columns: Sequence[str] | None = None) -> pl.DataFrame:
        """The feature frame, verified against the current bars and settings.

        *columns* restricts the result - and what is read from disk - to those
        names, in that order. Values are identical to the full frame's; at 23
        years of 1-minute bars the full frame is ~30 columns x 7.9M rows, so a
        caller that needs seven should ask for seven.
        """
        manifest = self.manifest(timeframe)
        entry = (manifest or {}).get("windows", {}).get(str(window))
        if manifest is None or entry is None:
            raise FileNotFoundError(
                f"No cached features for {timeframe} window={window}. Run "
                f"`xq build-features --timeframe {timeframe} --window {window}`."
            )
        current = bar_dataset_version(self.config, timeframe)
        if manifest.get("bar_dataset_version") != current:
            raise StaleFeaturesError(
                f"{timeframe} features were computed from bars "
                f"{manifest.get('bar_dataset_version')} but the current bars are {current}. "
                "Rebuild them with `xq build-features`."
            )
        if manifest.get("feature_fingerprint") != feature_fingerprint(self.regression):
            raise StaleFeaturesError(
                f"{timeframe} features were computed under different regression settings. "
                "Rebuild them with `xq build-features`."
            )
        directory = self.timeframe_dir(timeframe)
        base_path, specific_path = directory / BASE_FILE, directory / entry["path"]
        base_rows = pq.read_metadata(base_path).num_rows
        specific_rows = pq.read_metadata(specific_path).num_rows
        if base_rows != specific_rows or base_rows != entry["rows"]:
            raise StaleFeaturesError(
                f"{timeframe} window={window}: cached tables are misaligned "
                f"({base_rows} vs {specific_rows} rows)."
            )
        primary = self.regression.zscore.resolve_primary(window)
        order = _FULL_ORDER(window, self.regression)
        wanted = list(order if columns is None else columns)
        unknown = [c for c in wanted if c not in order]
        if unknown:
            raise KeyError(f"{timeframe} window={window}: no feature column(s) {unknown}")
        # What each re-derived column is computed from.
        sources = {"fitted_price": ["fitted_log_price"], "residual_pct": ["residual", "close"],
                   "residual_zscore_rolling": [f"residual_zscore_rolling_{primary}"]}
        read = {c for name in wanted for c in sources.get(name, [name])}
        stored = pq.read_schema(specific_path).names
        # The timestamp is always read: it anchors the frame's height.
        frame = pl.read_parquet(
            base_path, columns=[c for c in BASE_COLUMNS if c in read or c == "timestamp"])
        if any(c in read for c in stored):
            frame = frame.hstack(
                pl.read_parquet(specific_path, columns=[c for c in stored if c in read]))
        # Re-derive with the very numpy calls rolling_regression_features uses,
        # so a loaded frame is bit-identical to a freshly computed one.
        derived: list[pl.Series | pl.Expr] = []
        with np.errstate(invalid="ignore", divide="ignore"):
            log_prices = self.regression.regression.price_transform == "log"
            if "fitted_price" in wanted:
                fitted = frame["fitted_log_price"].to_numpy().astype(np.float64)
                fitted_price = np.exp(fitted) if log_prices else fitted.copy()
                derived.append(pl.Series("fitted_price", fitted_price).fill_nan(None))
            if "residual_pct" in wanted:
                residual = frame["residual"].to_numpy().astype(np.float64)
                if log_prices:
                    residual_pct = np.expm1(residual)
                else:
                    close = frame["close"].to_numpy().astype(np.float64)
                    residual_pct = residual / np.where(close != 0, close, np.nan)
                derived.append(pl.Series("residual_pct", residual_pct).fill_nan(None))
        if "residual_zscore_rolling" in wanted:
            derived.append(
                pl.col(f"residual_zscore_rolling_{primary}").alias("residual_zscore_rolling"))
        if derived:
            frame = frame.with_columns(derived)
        missing = [c for c in wanted if c not in frame.columns]
        if columns is not None and missing:
            raise KeyError(f"{timeframe} window={window}: feature column(s) {missing} "
                           "are not in the cache")
        return frame.select([c for c in wanted if c in frame.columns])

    def load_or_build(
        self, timeframe: str, window: int, *, bars: pl.DataFrame | None = None,
        columns: Sequence[str] | None = None,
    ) -> pl.DataFrame:
        try:
            return self.load(timeframe, window, columns)
        except (FileNotFoundError, StaleFeaturesError) as exc:
            LOGGER.info("[%s w=%d] %s; building", timeframe, window, exc)
        self.build(timeframe, [window], bars=bars)
        return self.load(timeframe, window, columns)

    # -- coverage ------------------------------------------------------------
    def coverage(self, timeframes: list[str]) -> list[dict[str, Any]]:
        """One row per cached (timeframe, window) for the coverage report."""
        rows: list[dict[str, Any]] = []
        for timeframe in timeframes:
            manifest = self.manifest(timeframe) or {}
            current = bar_dataset_version(self.config, timeframe)
            for key, entry in (manifest.get("windows") or {}).items():
                rows.append({
                    "timeframe": timeframe,
                    "window": int(key),
                    "rows": entry.get("rows"),
                    "valid_fits": entry.get("valid_fits"),
                    "first_timestamp": manifest.get("first_timestamp"),
                    "first_valid_timestamp": entry.get("first_valid_timestamp"),
                    "last_timestamp": manifest.get("last_timestamp"),
                    "years": len(entry.get("valid_fits_by_year") or {}),
                    "bar_dataset_version": manifest.get("bar_dataset_version"),
                    "current": manifest.get("bar_dataset_version") == current
                    and manifest.get("feature_fingerprint") == feature_fingerprint(
                        self.regression),
                    "feature_version": entry.get("feature_version"),
                    "bytes": entry.get("bytes"),
                })
        return rows


def _FULL_ORDER(window: int, regression: RegressionConfig) -> list[str]:  # noqa: N802
    """Column order of rolling_regression_features, for an identical frame."""
    zscores = [f"residual_zscore_rolling_{w}" for w in sorted(set(regression.zscore.resolve(window)))]
    return [
        "timestamp", "close", "log_price", "regression_intercept", "regression_slope",
        "fitted_log_price", "fitted_price", "residual", "residual_pct", "residual_std_fit",
        "r_squared", "n_observations", "residual_zscore_fit", *zscores,
        "residual_zscore_rolling", "log_return", "trailing_volatility",
    ]


def _feature_version(timeframe: str, window: int, bars: str | None, fingerprint: str) -> str:
    hasher = hashlib.blake2b(digest_size=8)
    hasher.update(f"{timeframe}|{window}|{bars}|{fingerprint}".encode())
    return f"regfeat-{timeframe}-w{window}-{hasher.hexdigest()}"


def _rows_by_year(frame: pl.DataFrame) -> dict[str, int]:
    if frame.is_empty():
        return {}
    counts = frame.group_by(pl.col("timestamp").dt.year().alias("year")).len().sort("year")
    return {str(y): int(n) for y, n in counts.iter_rows()}


def _atomic_parquet(frame: pl.DataFrame, path: Path) -> None:
    """Write via a temporary name so a crash never leaves a truncated cache file."""
    ensure_dir(path.parent)
    tmp = path.with_name(path.name + ".partial")
    tmp.unlink(missing_ok=True)
    frame.write_parquet(tmp, compression="zstd", compression_level=3, statistics=True)
    tmp.replace(path)
