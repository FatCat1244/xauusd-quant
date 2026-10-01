r"""The central feature registry (Prompt #8, Steps 3-6).

Every column the factory may write has an entry here - no anonymous columns.
:func:`build_registry` expands ``config/features.yaml`` into one
:class:`FeatureSpec` per (timeframe, feature): its id, family, mathematical
definition, source module and series, window, the history it needs before its
first value, dtype, missing-value behaviour, whether it is live-safe and can be
updated incrementally, its computational-cost category, the parameter family
it belongs to (its neighbouring windows), and the status earlier research
gave it with the evidence. Dataset and feature versions are attached when the
factory builds the matrix.

Features that are *not* causal are registered too - so that their exclusion
is explicit and recorded - in :data:`NON_CAUSAL`, and are never computed.

Notation: ``C, O, H, L`` the bar's mid close / open / high / low,
:math:`r_t = \ln C_t - \ln C_{t-1}`, :math:`\sigma_{50,t}` the trailing 50-bar
standard deviation of ``r`` (the Prompt #3 trailing volatility), ``eps`` the
Prompt #3 rolling-regression residual.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from .factory_config import FeatureFactoryConfig

__all__ = [
    "NON_CAUSAL",
    "FeatureSpec",
    "build_registry",
    "feature_id",
    "registry_frame_rows",
]

FLOAT32 = "float32"


@dataclass(frozen=True)
class FeatureSpec:
    """Everything the research needs to know about one feature column."""

    feature_id: str
    name: str
    family: str
    definition: str
    source_module: str
    source_series: str
    timeframe: str
    window: int | None
    min_history: int
    live_safe: bool = True
    dtype: str = FLOAT32
    missing_policy: str = "nan until min_history; nan where the source is undefined"
    cost: str = "cheap"
    incremental_update: str = "yes"
    parameter_family: str | None = None
    prior_status: str = "candidate"
    prior_evidence: str = ""
    invalid_reason: str | None = None
    source_feed: str = "historical_research_feed"
    dataset_version: str | None = None
    feature_version: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def status_excludes_default(self) -> bool:
        """failed_null_control / invalid / non_causal never enter the default candidates."""
        return (not self.live_safe or self.invalid_reason is not None
                or self.prior_status in ("failed_null_control", "invalid", "non_causal"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def feature_id(base: str, timeframe: str, window: int | None = None) -> str:
    """``OU_LOG_HALF_LIFE_5M_256``: base name, timeframe, then the window if any."""
    parts = [base.upper(), timeframe.upper()]
    if window is not None:
        parts.append(str(window))
    return "_".join(parts)


#: Registered only to make their exclusion explicit: never computed, never stored.
NON_CAUSAL: dict[str, str] = {
    "regime_smoothed_probability": "HMM smoothed P(S_t | X_1..T) uses later bars (Prompt #7)",
    "regime_viterbi_state": "Viterbi path is decided by the whole sample (Prompt #7)",
    "wavelet_cwt_scalogram": "two-sided CWT: coefficients at t use bars after t (Prompt #6)",
    "wavelet_ridge": "CWT ridges are traced through later coefficients (Prompt #6)",
    "wavelet_dwt_components": "full-series DWT reconstruction (Prompt #6)",
    "centered_rolling_statistics": "a centred window reads half a window into the future",
    "full_sample_percentile": "a 2003-2026 rank of a value uses the future distribution",
    "full_sample_zscore": "normalisation by the full-history mean / sd",
    "offline_regime_fit_state": "states of the full-sample (offline) regime fit (Prompt #7)",
}


def _spec(cfg: FeatureFactoryConfig, timeframe: str, *, base: str, window: int | None,
          family: str, definition: str, source_module: str, source_series: str,
          min_history: int, cost: str = "cheap", incremental: str = "yes",
          parameter_family: str | None = None, name: str | None = None,
          extra: dict[str, Any] | None = None) -> FeatureSpec:
    column = name or (f"{base}_{window}" if window is not None else base)
    prior = cfg.prior(column)
    return FeatureSpec(
        feature_id=feature_id(base, timeframe, window),
        name=column, family=family, definition=definition, source_module=source_module,
        source_series=source_series, timeframe=timeframe, window=window,
        min_history=int(min_history), cost=cost, incremental_update=incremental,
        parameter_family=parameter_family, prior_status=prior.status,
        prior_evidence=prior.evidence, invalid_reason=cfg.invalid_reason(column, timeframe),
        source_feed=cfg.source_feed, extra=dict(extra or {}))


def _returns(cfg: FeatureFactoryConfig, tf: str, _bar_seconds: float) -> Iterator[FeatureSpec]:
    f = cfg.families.returns
    mod = "features/families.py"
    for h in f.horizons:
        yield _spec(cfg, tf, base="ret", window=h, family="returns",
                    definition=f"ln C_t - ln C_(t-{h})", source_module=mod,
                    source_series="close", min_history=h + 1, parameter_family="ret")
    for h in f.z_horizons:
        yield _spec(cfg, tf, base="ret_z", window=h, family="returns",
                    definition=f"(ln C_t - ln C_(t-{h})) / (sigma_{f.volatility_window},t sqrt {h})",
                    source_module=mod, source_series="close",
                    min_history=max(h, f.volatility_window) + 1, parameter_family="ret_z")
    for w in f.moment_windows:
        yield _spec(cfg, tf, base="ret_skew", window=w, family="returns",
                    definition=f"sample skewness of r over the last {w} bars",
                    source_module=mod, source_series="log_return", min_history=w + 1,
                    parameter_family="ret_skew")
        yield _spec(cfg, tf, base="ret_kurt", window=w, family="returns",
                    definition=f"sample excess kurtosis of r over the last {w} bars",
                    source_module=mod, source_series="log_return", min_history=w + 1,
                    parameter_family="ret_kurt")
    for w in f.range_windows:
        yield _spec(cfg, tf, base="range_pos", window=w, family="returns",
                    definition=f"(ln C_t - min ln L) / (max ln H - min ln L) over the last {w} "
                               "bars (0 = at the low, 1 = at the high)",
                    source_module=mod, source_series="high, low, close", min_history=w,
                    incremental="yes (monotone deque)", parameter_family="range_pos")


def _volatility(cfg: FeatureFactoryConfig, tf: str, bar_seconds: float) -> Iterator[FeatureSpec]:
    f = cfg.families.volatility
    mod = "features/families.py"
    for w in f.rv_windows:
        yield _spec(cfg, tf, base="log_rv", window=w, family="volatility",
                    definition=f"0.5 ln( mean of r^2 over the last {w} bars )",
                    source_module=mod, source_series="log_return", min_history=w + 1,
                    parameter_family="log_rv")
    for lam in f.ewma_lambdas:
        tag = int(round(lam * 100))
        yield _spec(cfg, tf, base="log_ewma_vol", window=tag, family="volatility",
                    definition=f"0.5 ln EWMA(r^2), v_t = {lam} v_(t-1) + {1 - lam:.2f} r_t^2 "
                               "(seeded by the mean of the first span)",
                    source_module=mod, source_series="log_return",
                    min_history=int(math.ceil(3.0 / (1.0 - lam))) + 1,
                    parameter_family="log_ewma_vol",
                    extra={"lambda": lam})
    lag = f.change_lag
    yield _spec(cfg, tf, base="vol_change", window=lag, family="volatility",
                definition=f"log_rv_20(t) - log_rv_20(t - {lag})", source_module=mod,
                source_series="log_return", min_history=20 + lag + 1)
    a, b = f.ratio_pair
    yield _spec(cfg, tf, base="vol_ratio", window=None, name=f"vol_ratio_{a}_{b}",
                family="volatility", definition=f"log_rv_{a} - log_rv_{b}",
                source_module=mod, source_series="log_return", min_history=b + 1)
    pct = cfg.percentile_window(bar_seconds)
    yield _spec(cfg, tf, base="rv_percentile", window=None, family="volatility",
                definition=f"(rank - 0.5) / n of log_rv_20 among the last {cfg.percentile_days} "
                           f"trading days ({pct} bars)",
                source_module=mod, source_series="log_return", min_history=pct + 20,
                incremental="partial (order-statistic tree)")
    w = f.range_window
    yield _spec(cfg, tf, base="log_parkinson", window=w, family="volatility",
                definition=f"0.5 ln mean over {w} bars of (ln H - ln L)^2 / (4 ln 2)",
                source_module=mod, source_series="high, low", min_history=w,
                parameter_family="range_vol")
    yield _spec(cfg, tf, base="log_garman_klass", window=w, family="volatility",
                definition=f"0.5 ln mean over {w} bars of 0.5 (ln H/L)^2 - (2 ln 2 - 1) "
                           "(ln C/O)^2",
                source_module=mod, source_series="open, high, low, close", min_history=w,
                parameter_family="range_vol")


def _autocorrelation(cfg: FeatureFactoryConfig, tf: str,
                     _bar_seconds: float) -> Iterator[FeatureSpec]:
    f = cfg.families.autocorrelation
    mod = "features/families.py"
    for k in f.return_lags:
        yield _spec(cfg, tf, base=f"acf_lag{k}", window=f.window, family="autocorrelation",
                    definition=f"rolling corr(r_t, r_(t-{k})) over the last {f.window} bars",
                    source_module=mod, source_series="log_return",
                    min_history=f.window + k + 1,
                    parameter_family="acf_lag1" if k == 1 else f"acf_lag{k}")
    for w in f.lag1_windows:
        yield _spec(cfg, tf, base="acf_lag1", window=w, family="autocorrelation",
                    definition=f"rolling corr(r_t, r_(t-1)) over the last {w} bars",
                    source_module=mod, source_series="log_return", min_history=w + 2,
                    parameter_family="acf_lag1")
    yield _spec(cfg, tf, base="abs_acf_lag1", window=f.window, family="autocorrelation",
                definition=f"rolling corr(|r_t|, |r_(t-1)|) over {f.window} bars",
                source_module=mod, source_series="log_return", min_history=f.window + 2)
    yield _spec(cfg, tf, base="sq_acf_lag1", window=f.window, family="autocorrelation",
                definition=f"rolling corr(r_t^2, r_(t-1)^2) over {f.window} bars",
                source_module=mod, source_series="log_return", min_history=f.window + 2)


def _regression(cfg: FeatureFactoryConfig, tf: str, _bar_seconds: float) -> Iterator[FeatureSpec]:
    f = cfg.families.regression
    mod = "features/store.py (Prompt #3 rolling OLS of ln C on 0..N-1)"
    sig = f"sigma_{f.volatility_window}"
    for n in f.windows:
        common: dict[str, Any] = {"family": "regression", "source_module": mod, "source_series": "ln close",
                  "cost": "cheap", "incremental": "yes (running sums)"}
        yield _spec(cfg, tf, base="reg_slope_vol", window=n,
                    definition=f"slope of the {n}-bar rolling regression / {sig}",
                    min_history=max(n, f.volatility_window + 1), parameter_family="reg_slope_vol",
                    **common)
        yield _spec(cfg, tf, base="reg_r2", window=n,
                    definition=f"R^2 of the {n}-bar rolling regression",
                    min_history=n, parameter_family="reg_r2", **common)
        yield _spec(cfg, tf, base="reg_resid_z", window=n,
                    definition=f"rolling Z-score of the N={n} residual over its own recent "
                               "values (the Prompt #3 primary residual_zscore_rolling)",
                    min_history=2 * n, parameter_family="reg_resid_z", **common)
        yield _spec(cfg, tf, base="reg_resid_vol", window=n,
                    definition=f"eps_t (N={n}) / {sig}: the residual in volatility units",
                    min_history=max(n, f.volatility_window + 1), parameter_family="reg_resid_vol",
                    **common)
        yield _spec(cfg, tf, base="reg_resid_std_ratio", window=n,
                    definition=f"ln( residual standard error of the N={n} fit / {sig} )",
                    min_history=max(n, f.volatility_window + 1),
                    parameter_family="reg_resid_std_ratio", **common)


def _ou(cfg: FeatureFactoryConfig, tf: str, _bar_seconds: float) -> Iterator[FeatureSpec]:
    f = cfg.families.ou
    mod = (f"models/ornstein_uhlenbeck.py rolling_ou (Prompt #4) of the "
           f"N={f.regression_window} residual")
    base_hist = f.regression_window
    for m in f.windows:
        common: dict[str, Any] = {"family": "ou", "source_module": mod,
                  "source_series": f"eps (N={f.regression_window})",
                  "min_history": base_hist + m + 1, "cost": "moderate",
                  "incremental": "yes (running AR(1) sums)"}
        yield _spec(cfg, tf, base="ou_log_half_life", window=m,
                    definition=f"ln half-life (bars) of the {m}-bar AR(1) fit, valid fits only "
                               "(0 < b < 1)", parameter_family="ou_log_half_life", **common)
        yield _spec(cfg, tf, base="ou_theta", window=m,
                    definition=f"theta = -ln b of the {m}-bar AR(1) fit, valid fits only",
                    parameter_family="ou_theta", **common)
        yield _spec(cfg, tf, base="ou_zscore", window=m,
                    definition=f"(eps_t - mu_t) / stationary sd of the {m}-bar OU fit",
                    parameter_family="ou_zscore", **common)
    m = f.full_window
    common = {"family": "ou", "source_module": mod,
              "source_series": f"eps (N={f.regression_window})",
              "min_history": base_hist + m + 1, "cost": "moderate",
              "incremental": "yes (running AR(1) sums)"}
    yield _spec(cfg, tf, base="ou_b", window=m, definition=f"AR(1) slope b of the {m}-bar fit",
                **common)
    yield _spec(cfg, tf, base="ou_mu_norm", window=m,
                definition=f"OU equilibrium mu / in-window sd of eps ({m} bars)", **common)
    yield _spec(cfg, tf, base="ou_fit_r2", window=m,
                definition=f"R^2 of the {m}-bar AR(1) regression (fit quality)", **common)
    yield _spec(cfg, tf, base="ou_valid", window=m,
                definition=f"1 if the {m}-bar fit is a valid OU (0 < b < 1, finite), else 0",
                **common)
    yield _spec(cfg, tf, base="ou_std_ratio", window=m,
                definition=f"in-window sd / OU stationary sd ({m} bars)", **common)
    yield _spec(cfg, tf, base="ou_innov_z", window=m,
                definition=f"one-step innovation eps_t - (a + b eps_(t-1)) of the fit at t-1, "
                           f"/ its innovation sd ({m} bars)", **common)
    h = f.decay_horizon
    yield _spec(cfg, tf, base=f"ou_decay_ratio{h}", window=m,
                definition=f"expected decay ratio b^{h} of the {m}-bar fit (valid fits only)",
                **common)


def _fft(cfg: FeatureFactoryConfig, tf: str, bar_seconds: float) -> Iterator[FeatureSpec]:
    f = cfg.families.fft
    mod = "features/spectral.py rolling_spectrum (Prompt #5)"
    for n in f.windows:
        common: dict[str, Any] = {"family": "fft", "source_module": mod, "source_series": f.series,
                  "min_history": n + 1, "cost": "moderate",
                  "incremental": "no (one FFT per window; sliding DFT O(N))"}
        yield _spec(cfg, tf, base="fft_entropy", window=n,
                    definition=f"normalised spectral entropy of r over {n} bars (Hann, detrended)",
                    parameter_family="fft_entropy", **common)
        yield _spec(cfg, tf, base="fft_high_low", window=n,
                    definition=f"ln( high-band / low-band power share ) over {n} bars",
                    parameter_family="fft_high_low", **common)
        if n == f.full_window:
            yield _spec(cfg, tf, base="fft_flatness", window=n,
                        definition=f"spectral flatness (geometric / arithmetic mean power), {n}",
                        **common)
            yield _spec(cfg, tf, base="fft_top3_share", window=n,
                        definition=f"power share of the three largest spectral bins, {n}",
                        **common)
            yield _spec(cfg, tf, base="fft_dominant_period", window=n,
                        definition=f"period (bars) of the largest spectral peak, {n}", **common)
            yield _spec(cfg, tf, base="fft_centroid", window=n,
                        definition=f"normalised spectral centroid, {n}", **common)
    n = f.residual_window
    common = {"family": "fft", "source_module": mod, "source_series": "regression residual",
              "min_history": n + cfg.families.ou.regression_window, "cost": "moderate",
              "incremental": "no (one FFT per window)"}
    yield _spec(cfg, tf, base="fft_resid_entropy", window=n,
                definition=f"spectral entropy of the N=128 residual over {n} bars", **common)
    yield _spec(cfg, tf, base="fft_resid_high_low", window=n,
                definition=f"ln( high / low power share ) of the residual over {n} bars", **common)
    if bar_seconds >= 900:          # |eta| is stored at 15m-1h: the daily cycle needs N bars > 1 day
        n = f.abs_innovation_window
        common = {"family": "fft", "source_module": mod, "source_series": "|OU innovation|",
                  "min_history": n + cfg.families.ou.regression_window
                  + cfg.families.ou.full_window, "cost": "moderate",
                  "incremental": "no (one FFT per window)"}
        yield _spec(cfg, tf, base="fft_abs_eta_entropy", window=n,
                    definition=f"spectral entropy of |eta| over {n} bars", **common)
        yield _spec(cfg, tf, base="fft_abs_eta_dominant_period", window=n,
                    definition=f"period (bars) of the largest |eta| spectral peak over {n} bars "
                               "(the daily volatility cycle when it dominates)", **common)


def _wavelet(cfg: FeatureFactoryConfig, tf: str, _bar_seconds: float) -> Iterator[FeatureSpec]:
    f = cfg.families.wavelet
    mod = "features/wavelet_causal.py rolling_wavelet (Prompt #6, one-sided MODWT)"
    for n in f.windows:
        common: dict[str, Any] = {"family": "wavelet", "source_module": mod, "source_series": f.series,
                  "min_history": n + 1, "cost": "moderate",
                  "incremental": "partial (one-sided filters per bar; window energies by sums)"}
        yield _spec(cfg, tf, base="wav_entropy", window=n,
                    definition=f"normalised entropy of the MODWT band energy shares, {n} bars",
                    parameter_family="wav_entropy", **common)
        yield _spec(cfg, tf, base="wav_fast_slow", window=n,
                    definition=f"ln( fast-band energy / slow-band energy ), {n} bars",
                    parameter_family="wav_fast_slow", **common)
        if n == f.full_window:
            yield _spec(cfg, tf, base="wav_dominant_period", window=n,
                        definition=f"period (bars) of the band most above its white-noise share, "
                                   f"{n}", **common)
            yield _spec(cfg, tf, base="wav_run_length", window=n,
                        definition=f"bars the dominant band has held (scale persistence), {n}",
                        **{**common, "min_history": 2 * n + 1})
            yield _spec(cfg, tf, base="wav_top3_share", window=n,
                        definition=f"energy share of the three largest bands (concentration), {n}",
                        **common)
            yield _spec(cfg, tf, base="wav_scale_drift", window=n,
                        definition=f"ln centroid period minus its value k bars earlier "
                                   f"(frequency drift), {n}", **common)


def _regime(cfg: FeatureFactoryConfig, tf: str, _bar_seconds: float) -> Iterator[FeatureSpec]:
    f = cfg.families.regime
    mod = "regimes/causal_inference.py (Prompt #7 walk-forward forward filter)"
    src = f"{f.model} K={f.states} {f.scheme}"
    common: dict[str, Any] = {"family": "regime", "source_module": mod, "source_series": src,
              "min_history": 2000, "cost": "moderate",
              "incremental": "yes (forward filter O(K^2) per bar; periodic refits)"}
    for k in range(f.states):
        yield _spec(cfg, tf, base=f"regime_p{k}", window=None,
                    definition=f"filtered P(S_t = {k} | X_<=t), the walk-forward model of the "
                               "current quarter", **common)
    yield _spec(cfg, tf, base="regime_entropy", window=None,
                definition="entropy of the filtered state probabilities", **common)
    yield _spec(cfg, tf, base="regime_confidence", window=None,
                definition="largest filtered state probability", **common)
    yield _spec(cfg, tf, base="regime_age", window=None,
                definition="ln(1 + bars since the most likely state last changed)", **common)
    for k in range(f.states):
        yield _spec(cfg, tf, base=f"regime_next_p{k}", window=None,
                    definition=f"P(S_(t+1) = {k} | X_<=t) = sum_i p_i A_i{k}", **common)
    yield _spec(cfg, tf, base="regime_leave_prob", window=None,
                definition="P(S_(t+1) != S_t | X_<=t)", **common)


def _microstructure(cfg: FeatureFactoryConfig, tf: str,
                    bar_seconds: float) -> Iterator[FeatureSpec]:
    f = cfg.families.microstructure
    mod = "features/families.py (bar spreads and tick counts)"
    pct = cfg.percentile_window(bar_seconds)
    w = f.change_window
    yield _spec(cfg, tf, base="spread_rel", window=None, family="microstructure",
                definition="median bid-ask spread of the bar / C_t, in basis points",
                source_module=mod, source_series="median_spread", min_history=1)
    yield _spec(cfg, tf, base="spread_percentile", window=None, family="microstructure",
                definition=f"(rank - 0.5) / n of the bar's median spread among the last "
                           f"{cfg.percentile_days} trading days", source_module=mod,
                source_series="median_spread", min_history=pct,
                incremental="partial (order-statistic tree)")
    yield _spec(cfg, tf, base="spread_change", window=w, family="microstructure",
                definition=f"ln( median spread / its mean over the last {w} bars )",
                source_module=mod, source_series="median_spread", min_history=w)
    yield _spec(cfg, tf, base="log_tick_count", window=None, family="microstructure",
                definition="ln(1 + ticks in the bar) (broker quote activity, not volume)",
                source_module=mod, source_series="tick_count", min_history=1)
    yield _spec(cfg, tf, base="activity_percentile", window=None, family="microstructure",
                definition=f"(rank - 0.5) / n of the tick count among the last "
                           f"{cfg.percentile_days} trading days", source_module=mod,
                source_series="tick_count", min_history=pct,
                incremental="partial (order-statistic tree)")
    yield _spec(cfg, tf, base="activity_change", window=w, family="microstructure",
                definition=f"ln( (1 + ticks) / mean(1 + ticks) over the last {w} bars )",
                source_module=mod, source_series="tick_count", min_history=w)


def _time(cfg: FeatureFactoryConfig, tf: str, _bar_seconds: float) -> Iterator[FeatureSpec]:
    mod = "features/families.py (bar timestamp, broker clock = New York + 7 h)"
    common: dict[str, Any] = {"family": "time", "source_module": mod, "source_series": "timestamp",
              "min_history": 0}
    yield _spec(cfg, tf, base="tod_sin", window=None,
                definition="sin(2 pi (hour + minute/60) / 24) of the bar's open time", **common)
    yield _spec(cfg, tf, base="tod_cos", window=None,
                definition="cos(2 pi (hour + minute/60) / 24)", **common)
    yield _spec(cfg, tf, base="dow_sin", window=None,
                definition="sin(2 pi weekday / 5), Monday = 0", **common)
    yield _spec(cfg, tf, base="dow_cos", window=None, definition="cos(2 pi weekday / 5)",
                **common)
    if cfg.families.time.session_indicators:
        for session in ("off_hours", "asia", "london", "london_ny_overlap", "new_york"):
            yield _spec(cfg, tf, base=f"session_{session}", window=None,
                        definition=f"1 if the bar falls in the {session} session "
                                   "(config/research.yaml, first match wins)", **common)


def _interactions(cfg: FeatureFactoryConfig, tf: str, bar_seconds: float,
                  known: dict[str, FeatureSpec]) -> Iterator[FeatureSpec]:
    days = cfg.interaction_standardisation_days
    window = days * cfg.bars_per_day(bar_seconds)
    for ix in cfg.interactions:
        a, b = known.get(ix.a), known.get(ix.b)
        if a is None or b is None:
            continue
        hist = max(a.min_history, b.min_history) + window
        yield replace(
            _spec(cfg, tf, base=ix.name, window=None, family="interaction",
                  definition=f"z({ix.a}) x z({ix.b}), each z = (x - trailing mean) / "
                             f"trailing sd over {days} trading days ({window} bars), "
                             "clipped at +-10",
                  source_module="features/interactions.py",
                  source_series=f"{ix.a}, {ix.b}", min_history=hist,
                  cost="cheap" if a.cost == b.cost == "cheap" else "moderate",
                  incremental="yes (running sums)",
                  extra={"a": ix.a, "b": ix.b, "why": ix.why}),
            prior_status="candidate",
            prior_evidence="registered interaction (Step 43); no earlier evidence")


_BUILDERS = {"returns": _returns, "volatility": _volatility, "autocorrelation": _autocorrelation,
             "regression": _regression, "ou": _ou, "fft": _fft, "wavelet": _wavelet,
             "regime": _regime, "microstructure": _microstructure, "time": _time}


def build_registry(cfg: FeatureFactoryConfig, timeframe: str, bar_seconds: float, *,
                   families: tuple[str, ...] | None = None,
                   interactions: bool = True) -> list[FeatureSpec]:
    """Every causal feature of *timeframe*, in family order, then the interactions."""
    chosen = families or tuple(_BUILDERS)
    specs: list[FeatureSpec] = []
    for family in _BUILDERS:
        if family in chosen:
            specs.extend(_BUILDERS[family](cfg, timeframe, bar_seconds))
    names = [s.name for s in specs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"duplicate feature names in the registry: {dupes}")
    if interactions:
        known = {s.name: s for s in specs}
        specs.extend(_interactions(cfg, timeframe, bar_seconds, known))
    return specs


def registry_frame_rows(specs: list[FeatureSpec]) -> list[dict[str, Any]]:
    """Flat rows for the catalog table (extra is JSON-friendly)."""
    rows = []
    for s in specs:
        row = s.to_dict()
        row["extra"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in s.extra.items()}
        row["excluded_from_default_candidates"] = s.status_excludes_default
        rows.append(row)
    for name, reason in NON_CAUSAL.items():
        rows.append({"feature_id": name.upper(), "name": name, "family": "non_causal",
                     "definition": reason, "live_safe": False, "prior_status": "non_causal",
                     "prior_evidence": reason, "excluded_from_default_candidates": True})
    return rows
