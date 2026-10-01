r"""Fourier reconstruction inside a window, and extrapolation beyond it.

Inside the window the decomposition is exact: with the rectangular (untapered)
FFT of the mean-removed window and Parseval's theorem,

.. math::

    \sum_n (x_n-\bar x)^2 = \tfrac1N \sum_{k} c_k |X_k|^2, \qquad
    c_k = 2 \text{ for } 0<k<N/2,\; 1 \text{ at Nyquist},

so the variance explained by a set of bins is their share of that sum, and
the reconstruction is :func:`numpy.fft.irfft` of the kept coefficients. Top-K
here means the K largest coefficients (best K-term approximation), not
spectral peaks.

In-window fit is **not** evidence of anything: N coefficients always fit N
values. The harder test extrapolates: the components found in bars
``t-N+1..t`` are carried to ``t+h``,

.. math::

    \hat x_{t+h} = \bar x + \sum_{k \in K} \tfrac{c_k}{N} |X_k|
                   \cos\big(2\pi k (N-1+h)/N + \arg X_k\big),

which, because every bin completes a whole number of cycles in N bars, is
the window's own periodic continuation. It is scored against the realised
values beside two baselines (last value, window mean) and beside the same
test on null controls. Estimates come from an evenly spaced sample of
windows and are labelled as such.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from ..features.spectral import TWO_PI

__all__ = [
    "extrapolation_table",
    "reconstruct",
    "reconstruction_table",
    "sample_window_ends",
]


def sample_window_ends(valid: np.ndarray, max_windows: int) -> np.ndarray:
    """Evenly spaced bar indices among valid window ends (all of them if few)."""
    ends = np.flatnonzero(valid)
    if ends.size <= max_windows:
        return ends
    return ends[np.linspace(0, ends.size - 1, max_windows).round().astype(np.int64)]


def _gather(values: np.ndarray, ends: np.ndarray, n: int) -> np.ndarray:
    return values[ends[:, None] - (n - 1) + np.arange(n)[None, :]]


def _weights(n: int, bins: np.ndarray) -> np.ndarray:
    return np.where(bins * 2 == n, 1.0, 2.0)


def reconstruct(window: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Mean plus the kept Fourier components of one window (inverse FFT, exact scaling)."""
    x = np.asarray(window, dtype=np.float64)
    spectrum = np.fft.rfft(x - x.mean())
    kept = np.zeros_like(spectrum)
    kept[bins] = spectrum[bins]
    return x.mean() + np.fft.irfft(kept, n=x.size)


def _top(power: np.ndarray, usable: np.ndarray, k: int) -> np.ndarray:
    """Column indices (into the full rfft) of the k largest usable coefficients."""
    sub = power[:, usable]
    k = min(k, usable.size)
    part = np.argpartition(sub, sub.shape[1] - k, axis=1)[:, -k:]
    order = np.argsort(-np.take_along_axis(sub, part, axis=1), axis=1)
    return usable[np.take_along_axis(part, order, axis=1)]


def reconstruction_table(
    values: np.ndarray, ends: np.ndarray, fft_window: int, *, usable_bins: np.ndarray,
    components: tuple[int, ...], source: str,
) -> pl.DataFrame:
    """Share of each window's variance kept by its top-K coefficients, and RMSE."""
    n = fft_window
    block = _gather(np.asarray(values, dtype=np.float64), ends, n)
    keep = np.isfinite(block).all(axis=1)
    block = block[keep]
    if block.shape[0] == 0:
        return pl.DataFrame()
    centred = block - block.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(centred, axis=1)
    power = spectrum.real ** 2 + spectrum.imag ** 2
    all_bins = np.arange(1, power.shape[1])
    weights = _weights(n, np.arange(power.shape[1]))
    total = (power[:, 1:] * weights[1:]).sum(axis=1)
    variance = (centred ** 2).mean(axis=1)
    rows = []
    for k in components:
        chosen = _top(power, usable_bins, k)
        kept = (np.take_along_axis(power, chosen, axis=1) * weights[chosen]).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            explained = kept / total
        rmse = np.sqrt(np.maximum(1.0 - explained, 0.0) * variance)
        rows.append({
            "source": source, "components": k, "windows_sampled": int(block.shape[0]),
            "explained_variance_median": float(np.nanmedian(explained)),
            "explained_variance_mean": float(np.nanmean(explained)),
            "explained_variance_p10": float(np.nanquantile(explained, 0.1)),
            "explained_variance_p90": float(np.nanquantile(explained, 0.9)),
            "rmse_median": float(np.nanmedian(rmse)),
            "rmse_over_window_std_median": float(np.nanmedian(
                np.sqrt(np.maximum(1.0 - explained, 0.0)))),
            "bins_available": int(all_bins.size),
        })
    return pl.DataFrame(rows)


def extrapolation_table(
    values: np.ndarray, ends: np.ndarray, fft_window: int, *, usable_bins: np.ndarray,
    components: tuple[int, ...], horizons: tuple[int, ...], source: str,
) -> pl.DataFrame:
    """Top-K components of each window carried ``h`` bars past its end, scored.

    ``skill_vs_last`` and ``skill_vs_mean`` are ``1 - MSE / MSE_baseline``;
    positive means the extrapolation beat that baseline. ``change_*`` columns
    compare the predicted change ``x_hat[t+h] - x[t]`` with the realised
    change.
    """
    x = np.asarray(values, dtype=np.float64)
    n = fft_window
    max_h = max(horizons)
    ends = ends[ends + max_h < x.size]
    block = _gather(x, ends, n)
    future = x[ends[:, None] + np.arange(1, max_h + 1)[None, :]]
    keep = np.isfinite(block).all(axis=1)
    block, future, ends = block[keep], future[keep], ends[keep]
    if block.shape[0] == 0:
        return pl.DataFrame()
    mean = block.mean(axis=1)
    last = block[:, -1]
    spectrum = np.fft.rfft(block - mean[:, None], axis=1)
    power = spectrum.real ** 2 + spectrum.imag ** 2
    weights = _weights(n, np.arange(power.shape[1]))
    rows: list[dict[str, Any]] = []
    for k in components:
        chosen = _top(power, usable_bins, k)
        coef = np.take_along_axis(spectrum, chosen, axis=1)
        amp = weights[chosen] * np.abs(coef) / n
        arg = np.angle(coef)
        for h in horizons:
            actual = future[:, h - 1]
            ok = np.isfinite(actual)
            predicted = mean + (amp * np.cos(TWO_PI * chosen * (n - 1 + h) / n + arg)).sum(axis=1)
            p, a, last_h, mean_h = predicted[ok], actual[ok], last[ok], mean[ok]
            mse = float(np.mean((p - a) ** 2))
            mse_last = float(np.mean((last_h - a) ** 2))
            mse_mean = float(np.mean((mean_h - a) ** 2))
            dp, da = p - last_h, a - last_h
            nonzero = (dp != 0) & (da != 0)
            rows.append({
                "source": source, "components": k, "horizon": h, "windows_sampled": int(ok.sum()),
                "mae": float(np.mean(np.abs(p - a))), "rmse": float(np.sqrt(mse)),
                "rmse_last_value": float(np.sqrt(mse_last)),
                "rmse_window_mean": float(np.sqrt(mse_mean)),
                "skill_vs_last": 1.0 - mse / mse_last if mse_last > 0 else None,
                "skill_vs_mean": 1.0 - mse / mse_mean if mse_mean > 0 else None,
                "change_correlation": (float(np.corrcoef(dp, da)[0, 1])
                                       if dp.std() > 0 and da.std() > 0 else None),
                "change_sign_agreement": (float(np.mean(np.sign(dp[nonzero])
                                                        == np.sign(da[nonzero])))
                                          if nonzero.any() else None),
            })
    return pl.DataFrame(rows, infer_schema_length=None)
