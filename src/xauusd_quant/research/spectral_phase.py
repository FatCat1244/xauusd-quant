r"""Phase, analysed as the circular quantity it is.

A phase of :math:`-\pi + 0.01` and one of :math:`\pi - 0.01` are 0.02 apart,
not 6.26, so nothing here takes an arithmetic mean of raw phases:

.. math::

    \bar\phi = \operatorname{atan2}\Big(\tfrac1n\sum\sin\phi_i,\;
                                          \tfrac1n\sum\cos\phi_i\Big), \qquad
    R = \Big|\tfrac1n\sum e^{i\phi_i}\Big| \in [0, 1]

``R`` near 1 means the phases agree; near 0, that they are spread out. The
Rayleigh test asks whether ``R`` is larger than uniform phases would give.

Two cautions govern every result here:

* **Overlapping windows share data.** Windows ``h < N`` bars apart hold
  ``N - h`` of the same values, so their phases advance consistently by
  construction, even for white noise. Phase *continuation* is therefore
  measured between disjoint windows (lag >= N) and always beside controls.
* **Phase of a negligible component is noise.** Only components holding at
  least ``phase_min_power_share`` of their window's power carry a phase.

Outcomes (the series ``h`` bars later) are research outcomes, never inputs:
the phase at ``t`` uses bars up to ``t`` only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from ..features.spectral import TWO_PI, RollingSpectrum

__all__ = [
    "circular_mean",
    "circular_summary",
    "phase_advance_consistency",
    "phase_bucket_index",
    "phase_bucket_table",
    "phase_projection",
    "rayleigh_test",
    "resultant_length",
    "wrap_phase",
]


def wrap_phase(angle: np.ndarray | float) -> np.ndarray:
    """Wrap to ``[-pi, pi)``."""
    return (np.asarray(angle, dtype=np.float64) + np.pi) % TWO_PI - np.pi


def circular_mean(angles: np.ndarray) -> float:
    a = np.asarray(angles, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))


def resultant_length(angles: np.ndarray) -> float:
    a = np.asarray(angles, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(np.hypot(np.sin(a).mean(), np.cos(a).mean()))


def rayleigh_test(angles: np.ndarray) -> tuple[float, float]:
    """Rayleigh ``Z = nR^2`` and its p-value (Zar's approximation) against uniformity."""
    a = np.asarray(angles, dtype=np.float64)
    a = a[np.isfinite(a)]
    n = a.size
    if n < 2:
        return float("nan"), float("nan")
    r = resultant_length(a)
    z = n * r * r
    p = np.exp(np.sqrt(1.0 + 4.0 * n + 4.0 * (n * n - (n * r) ** 2)) - (1.0 + 2.0 * n))
    return float(z), float(min(max(p, 0.0), 1.0))


def circular_summary(angles: np.ndarray) -> dict[str, float]:
    a = np.asarray(angles, dtype=np.float64)
    a = a[np.isfinite(a)]
    r = resultant_length(a)
    z, p = rayleigh_test(a)
    return {
        "n": int(a.size),
        "circular_mean": circular_mean(a),
        "resultant_length": r,
        "circular_std": float(np.sqrt(-2.0 * np.log(r))) if r and r > 0 else float("nan"),
        "rayleigh_z": z,
        "rayleigh_p": p,
    }


def phase_bucket_index(phase: np.ndarray, buckets: int) -> np.ndarray:
    """Sector ``j`` covers ``[-pi + 2 pi j / B, -pi + 2 pi (j+1) / B)``; -1 if no phase."""
    p = np.asarray(phase, dtype=np.float64)
    out = np.full(p.size, -1, dtype=np.int64)
    ok = np.isfinite(p)
    out[ok] = np.minimum(((wrap_phase(p[ok]) + np.pi) / TWO_PI * buckets).astype(np.int64),
                         buckets - 1)
    return out


def phase_bucket_table(
    phase: np.ndarray, outcomes: dict[str, np.ndarray], *, buckets: int, source: str,
) -> pl.DataFrame:
    """Mean outcome in each phase sector, with its standard error.

    One row per (outcome, sector). ``spread_over_std`` on the ``*`` rows is
    the range of sector means in units of the outcome's own standard
    deviation - an effect size - and ``f_statistic`` a one-way ANOVA F, to be
    read with care because neighbouring bars' outcomes overlap.
    """
    index = phase_bucket_index(phase, buckets)
    rows: list[dict[str, Any]] = []
    width = TWO_PI / buckets
    for name, values in outcomes.items():
        y = np.asarray(values, dtype=np.float64)
        usable = (index >= 0) & np.isfinite(y)
        if usable.sum() < buckets * 10:
            continue
        idx, yy = index[usable], y[usable]
        counts = np.bincount(idx, minlength=buckets)
        sums = np.bincount(idx, weights=yy, minlength=buckets)
        squares = np.bincount(idx, weights=yy * yy, minlength=buckets)
        with np.errstate(invalid="ignore", divide="ignore"):
            means = sums / counts
            variances = squares / counts - means ** 2
        grand = yy.mean()
        between = float(np.nansum(counts * (means - grand) ** 2) / max(buckets - 1, 1))
        within = float(np.nansum(counts * variances) / max(yy.size - buckets, 1))
        std = float(yy.std(ddof=1))
        for j in range(buckets):
            rows.append({
                "source": source, "outcome": name, "bucket": j,
                "phase_lower": -np.pi + j * width, "phase_upper": -np.pi + (j + 1) * width,
                "observations": int(counts[j]), "mean": float(means[j]),
                "std_error": (float(np.sqrt(variances[j] / counts[j]))
                              if counts[j] > 1 else None),
            })
        rows.append({
            "source": source, "outcome": name, "bucket": -1, "observations": int(yy.size),
            "mean": float(grand), "spread_over_std": (float((np.nanmax(means) - np.nanmin(means))
                                                            / std) if std > 0 else None),
            "f_statistic": between / within if within > 0 else None,
        })
    return pl.DataFrame(rows, infer_schema_length=None)


def phase_advance_consistency(result: RollingSpectrum, lag: int) -> dict[str, Any]:
    r"""Does the dominant component's phase advance as a steady cycle would?

    For bars ``t`` and ``t + lag`` whose dominant component is the same bin
    ``k`` with a valid phase in both, the error
    :math:`\phi_{t+lag} - \phi_t - 2\pi k\,lag/N` is wrapped and summarised
    circularly. A persistent sinusoid gives errors near 0 (``R`` near 1);
    unrelated phases give ``R`` near 0. With ``lag < N`` the windows overlap
    and ``R`` is high for any series; read ``lag >= N`` against controls.
    """
    n = result.fft_window
    bins = result.dominant_bin.astype(np.int64)
    phase = result.dominant_phase.astype(np.float64)
    ok = result.dominant_phase_valid
    if lag <= 0 or lag >= bins.size:
        return {"lag": lag, "pairs": 0}
    same = ok[:-lag] & ok[lag:] & (bins[:-lag] == bins[lag:]) & (bins[:-lag] > 0)
    pairs_possible = int((ok[:-lag] & ok[lag:]).sum())
    k = bins[:-lag][same]
    error = wrap_phase(phase[lag:][same] - phase[:-lag][same] - TWO_PI * k * lag / n)
    summary = circular_summary(error)
    return {
        "lag": lag, "lag_in_windows": lag / n, "pairs_with_phase": pairs_possible,
        "pairs_same_bin": int(same.sum()),
        "same_bin_fraction": (float(same.sum() / pairs_possible) if pairs_possible else None),
        "error_resultant_length": summary["resultant_length"],
        "error_circular_mean": summary["circular_mean"],
        "rayleigh_p": summary["rayleigh_p"],
    }


def phase_projection(
    result: RollingSpectrum, series: np.ndarray, horizons: tuple[int, ...], *,
    target: np.ndarray | None = None, target_name: str = "self",
) -> pl.DataFrame:
    r"""The dominant component carried forward against what actually happened.

    Predicted change of the component over ``h`` bars:
    :math:`\hat\Delta = A\,[\cos(\phi_t + 2\pi f h) - \cos\phi_t]` with the
    phase at the last bar of the window ending at ``t``. It is compared with
    the realised change of *target* (default: the analysed series itself),
    ``target[t+h] - target[t]``: Pearson and rank correlation, and how often
    the signs agree. An outcome study only - nothing about ``t+h`` enters
    the estimate at ``t``.
    """
    from scipy import stats

    y = np.asarray(series if target is None else target, dtype=np.float64)
    amp = result.top_amplitude[:, 0].astype(np.float64)
    phase = result.dominant_phase.astype(np.float64)
    freq = result.dominant_bin.astype(np.float64) / result.fft_window
    ok = result.dominant_phase_valid & np.isfinite(amp) & np.isfinite(y)
    rows = []
    for h in horizons:
        if h >= y.size:
            continue
        idx = np.flatnonzero(ok[:-h])
        idx = idx[np.isfinite(y[idx + h])]
        if idx.size < 30:
            continue
        predicted = amp[idx] * (np.cos(phase[idx] + TWO_PI * freq[idx] * h) - np.cos(phase[idx]))
        actual = y[idx + h] - y[idx]
        nonzero = (predicted != 0) & (actual != 0)
        rows.append({
            "target": target_name, "horizon": h, "observations": int(idx.size),
            "pearson": float(np.corrcoef(predicted, actual)[0, 1]),
            "spearman": float(stats.spearmanr(predicted, actual).statistic),
            "sign_agreement": (float(np.mean(np.sign(predicted[nonzero])
                                             == np.sign(actual[nonzero])))
                               if nonzero.any() else None),
        })
    return pl.DataFrame(rows, infer_schema_length=None)
