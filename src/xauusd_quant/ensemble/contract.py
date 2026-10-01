r"""The standardized prediction record - the only thing later stages consume
(Prompt #11, Steps 89-91).

Features -> models -> ensembles -> **prediction contract** -> (later) strategy /
execution. A record describes one bar of one timeframe:

.. code-block:: text

    {"timestamp": "2021-12-31T23:00:00", "timeframe": "5m",
     "prediction_version": "PRED_5M_V001-<hash>",
     "reversion_probability": 0.53,        P(|eps_{t+h}| < |eps_t|), h in sources
     "expected_return": 0.00012,           E[ln(P_{t+h}/P_t)] (vol-scaled model x sigma_t sqrt h)
     "expected_return_vol_scaled": 0.03,
     "expected_log_volatility": -7.9,      E[ln(RV_{t,h} + floor)]
     "expected_volatility": 0.00037,       exp(E[ln(RV + floor)]) - floor (a median-type value)
     "expected_log_abs_move": -8.4,
     "expected_abs_move": 0.00022,
     "uncertainty": {"model_disagreement": 0.011, "disagreement_by_output": {...},
                     "ood_score": 0.62, "regime_entropy": 0.4,
                     "reversion_entropy": 0.69},
     "sources": {"reversion_probability": {"ensemble_id": "ENS_REVERSION_5M_H5_V001",
                                           "horizon_bars": 5, "status": "ok"}, ...}}

``timestamp`` is the bar's own timestamp as stored (broker-local wall clock, bar
open time, bars ``[t, t+D)``): every input is known at the bar's close, so the
record is available from ``t + D`` on. A field whose ensemble is missing, invalid
(a constituent failed - ``ENSEMBLE_INVALID``) or not frozen is ``null`` with its
``status`` saying why; nothing is filled.

There is no decision field, by construction: :func:`validate_record` refuses any
key that looks like one (buy, sell, long, short, signal, side, position, size,
lots, threshold, stop, take_profit, pnl, order, action, entry, exit) anywhere in
the record.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np
import polars as pl

__all__ = ["CONTRACT_VERSION", "ContractError", "FIELDS", "FORBIDDEN_KEYS", "SCHEMA",
           "build_record", "contract_frame", "prediction_version", "records_from_frame",
           "validate_record"]

CONTRACT_VERSION = "1.0.0"

#: output field -> (target of the ensemble that fills it, what it is)
FIELDS: dict[str, tuple[str, str]] = {
    "reversion_probability": ("mean_reversion", "P(|eps_{t+h}| < |eps_t|)"),
    "expected_return_vol_scaled": ("future_return", "E[ln(P_{t+h}/P_t)] / (sigma_t sqrt(h))"),
    "expected_log_volatility": ("future_volatility", "E[ln(RV_{t,h} + floor)]"),
    "expected_log_abs_move": ("future_abs_move", "E[ln(|R_{t,h}| + floor)]"),
}
DERIVED = ("expected_return", "expected_volatility", "expected_abs_move")
UNCERTAINTY = ("model_disagreement", "ood_score", "regime_entropy", "reversion_entropy")
STATUSES = ("ok", "not_frozen", "no_eligible_ensemble", "ENSEMBLE_INVALID", "missing_input")
FORBIDDEN_KEYS = ("buy", "sell", "long", "short", "signal", "side", "position", "size", "lots",
                  "threshold", "stop", "take_profit", "takeprofit", "pnl", "order", "action",
                  "entry", "exit", "trade")

SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "xauusd-quant prediction record",
    "version": CONTRACT_VERSION,
    "type": "object",
    "required": ["timestamp", "timeframe", "prediction_version", *FIELDS, *DERIVED,
                 "uncertainty", "sources"],
    "additionalProperties": False,
    "properties": {
        "timestamp": {"type": "string", "description": "bar timestamp as stored (broker-local "
                      "wall clock, bar open time); available from the bar's close"},
        "timeframe": {"type": "string"},
        "prediction_version": {"type": "string"},
        **{f: {"type": ["number", "null"], "description": d} for f, (_, d) in FIELDS.items()},
        "expected_return": {"type": ["number", "null"],
                            "description": "expected_return_vol_scaled x sigma_t x sqrt(h)"},
        "expected_volatility": {"type": ["number", "null"],
                                "description": "exp(expected_log_volatility) - floor: the exp "
                                               "of an expected log, a median-type value"},
        "expected_abs_move": {"type": ["number", "null"],
                              "description": "exp(expected_log_abs_move) - floor"},
        "uncertainty": {"type": "object", "additionalProperties": False,
                        "properties": {
                            "model_disagreement": {"type": ["number", "null"]},
                            "disagreement_by_output": {"type": "object"},
                            "ood_score": {"type": ["number", "null"]},
                            "regime_entropy": {"type": ["number", "null"]},
                            "reversion_entropy": {"type": ["number", "null"]}}},
        "sources": {"type": "object"},
    },
}


class ContractError(ValueError):
    """A record does not satisfy the prediction contract."""


def prediction_version(timeframe: str, ensembles: dict[str, str | None]) -> str:
    """``PRED_<TF>_V<contract major>-<hash of the ensemble ids and spec hashes>``."""
    blob = json.dumps({"contract": CONTRACT_VERSION, "ensembles": ensembles},
                      sort_keys=True).encode()
    return (f"PRED_{timeframe.upper()}_V{CONTRACT_VERSION.split('.')[0].zfill(3)}-"
            f"{hashlib.blake2b(blob, digest_size=6).hexdigest()}")


def _num(v: Any) -> float | None:
    if v is None:
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def _walk_keys(obj: Any, path: str = "") -> list[str]:
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.append(f"{path}.{k}" if path else str(k))
            out += _walk_keys(v, f"{path}.{k}" if path else str(k))
    return out


def validate_record(record: dict[str, Any]) -> dict[str, Any]:
    """Raise :class:`ContractError` unless *record* satisfies the contract; return it."""
    for key in _walk_keys(record):
        leaf = key.rsplit(".", 1)[-1].lower()
        parts = leaf.replace("-", "_").split("_")
        if leaf in FORBIDDEN_KEYS or any(p in FORBIDDEN_KEYS for p in parts):
            raise ContractError(f"{key}: a decision-like field has no place in a prediction "
                                "record (no BUY / SELL, sizes, thresholds or PnL)")
    required = SCHEMA["required"]
    missing = [k for k in required if k not in record]
    if missing:
        raise ContractError(f"missing fields {missing}")
    extra = sorted(set(record) - set(SCHEMA["properties"]))
    if extra:
        raise ContractError(f"unknown fields {extra}")
    if not isinstance(record["timestamp"], str) or not isinstance(record["timeframe"], str):
        raise ContractError("timestamp and timeframe must be strings")
    for f in (*FIELDS, *DERIVED):
        v = record[f]
        if v is not None and (not isinstance(v, (int, float)) or not math.isfinite(float(v))):
            raise ContractError(f"{f}: a finite number or null, got {v!r}")
    p = record["reversion_probability"]
    if p is not None and not 0.0 <= float(p) <= 1.0:
        raise ContractError(f"reversion_probability {p} is not a probability")
    unc = record["uncertainty"]
    if not isinstance(unc, dict):
        raise ContractError("uncertainty must be an object")
    bad = sorted(set(unc) - set(SCHEMA["properties"]["uncertainty"]["properties"]))
    if bad:
        raise ContractError(f"unknown uncertainty fields {bad}")
    for f, (_, _) in FIELDS.items():
        src = (record["sources"] or {}).get(f)
        if not isinstance(src, dict) or src.get("status") not in STATUSES:
            raise ContractError(f"sources.{f}: needs a status in {STATUSES}")
        if (src.get("status") == "ok") != (record[f] is not None):
            raise ContractError(f"{f}: status {src.get('status')!r} with value {record[f]!r}")
    return record


def build_record(*, timestamp: str, timeframe: str, version: str,
                 values: dict[str, float | None], sources: dict[str, dict[str, Any]],
                 sigma: float | None, horizons: dict[str, int], log_floor: float,
                 uncertainty: dict[str, Any]) -> dict[str, Any]:
    """One validated record from the ensembles' outputs at one bar."""
    vals = {f: _num(values.get(f)) for f in FIELDS}
    srcs = {}
    for f in FIELDS:
        s = dict(sources.get(f) or {"status": "not_frozen"})
        if s.get("status") == "ok" and vals[f] is None:
            s["status"] = "missing_input"
        if s.get("status") != "ok":
            vals[f] = None
        srcs[f] = s
    h_ret = horizons.get("expected_return_vol_scaled")
    sig = _num(sigma)
    exp_ret = (vals["expected_return_vol_scaled"] * sig * math.sqrt(h_ret)
               if vals["expected_return_vol_scaled"] is not None and sig is not None and h_ret
               else None)
    lv, la = vals["expected_log_volatility"], vals["expected_log_abs_move"]
    record = {"timestamp": timestamp, "timeframe": timeframe, "prediction_version": version,
              **vals, "expected_return": exp_ret,
              "expected_volatility": None if lv is None else math.exp(lv) - log_floor,
              "expected_abs_move": None if la is None else math.exp(la) - log_floor,
              "uncertainty": {k: (_num(v) if k != "disagreement_by_output" else
                                  {kk: _num(vv) for kk, vv in (v or {}).items()})
                              for k, v in uncertainty.items()},
              "sources": srcs}
    return validate_record(record)


def contract_frame(timestamps: pl.Series, timeframe: str, version: str,
                   columns: dict[str, np.ndarray], statuses: dict[str, str],
                   sigma: np.ndarray | None, horizons: dict[str, int], log_floor: float
                   ) -> pl.DataFrame:
    """The joint predictive state as a table (Step 89): one row per bar, contract columns."""
    n = timestamps.len()
    out: dict[str, Any] = {"timestamp": timestamps, "timeframe": [timeframe] * n,
                           "prediction_version": [version] * n}
    for f in FIELDS:
        v = np.asarray(columns.get(f, np.full(n, np.nan)), dtype=np.float64)
        static = statuses.get(f, "not_frozen")
        if static != "ok":
            v = np.full(n, np.nan)
        out[f] = v
        # per row: a frozen ensemble without a value at this bar is a missing input, not 0
        out[f"status_{f}"] = np.where(np.isfinite(v), "ok",
                                      "missing_input" if static == "ok" else static).tolist()
    h = horizons.get("expected_return_vol_scaled")
    sig = np.asarray(sigma if sigma is not None else np.full(n, np.nan), dtype=np.float64)
    out["expected_return"] = out["expected_return_vol_scaled"] * sig * (np.sqrt(h) if h else np.nan)
    out["expected_volatility"] = np.exp(out["expected_log_volatility"]) - log_floor
    out["expected_abs_move"] = np.exp(out["expected_log_abs_move"]) - log_floor
    for u in UNCERTAINTY:
        out[u] = np.asarray(columns.get(u, np.full(n, np.nan)), dtype=np.float64)
    frame = pl.DataFrame(out)
    return frame.with_columns(pl.col(pl.Float64).fill_nan(None))


def records_from_frame(frame: pl.DataFrame, *, sources: dict[str, dict[str, Any]],
                       horizons: dict[str, int], log_floor: float) -> list[dict[str, Any]]:
    """Contract records (validated) from rows of :func:`contract_frame`."""
    out = []
    for r in frame.iter_rows(named=True):
        unc = {u: r.get(u) for u in UNCERTAINTY}
        srcs = {f: {**(sources.get(f) or {}), "status": r[f"status_{f}"]} for f in FIELDS}
        rec = {"timestamp": str(r["timestamp"]), "timeframe": r["timeframe"],
               "prediction_version": r["prediction_version"],
               **{f: _num(r[f]) for f in FIELDS},
               "expected_return": _num(r["expected_return"]),
               "expected_volatility": _num(r["expected_volatility"]),
               "expected_abs_move": _num(r["expected_abs_move"]),
               "uncertainty": {k: _num(v) for k, v in unc.items()}, "sources": srcs}
        for f in FIELDS:
            if rec[f] is None and srcs[f]["status"] == "ok":
                srcs[f]["status"] = "missing_input"
        out.append(validate_record(rec))
    del horizons, log_floor
    return out
