r"""Ornstein-Uhlenbeck mathematics for the residual research.

The model
---------
.. math::

    dX_t = \theta(\mu - X_t)\,dt + \sigma\,dW_t

sampled every :math:`\Delta t` is *exactly* an AR(1):

.. math::

    X_{t+\Delta t} = a + b X_t + \eta_t, \qquad
    b = e^{-\theta\Delta t}, \quad a = \mu(1-b), \quad
    \mathrm{Var}(\eta) = \frac{\sigma^2}{2\theta}\bigl(1 - e^{-2\theta\Delta t}\bigr)

so an OLS fit of the AR(1) maps back to

.. math::

    \theta = -\frac{\ln b}{\Delta t}, \quad \mu = \frac{a}{1-b}, \quad
    \sigma = \sqrt{\mathrm{Var}(\eta)\,\frac{2\theta}{1-b^2}}, \quad
    \mathrm{Std}(X) = \frac{\sigma}{\sqrt{2\theta}} = \frac{\mathrm{Std}(\eta)}{\sqrt{1-b^2}},
    \quad HL_{bars} = -\frac{\ln 2}{\ln b}

- **but only when** :math:`0 < b < 1`. Nothing here forces an AR(1) fit into
that reading. Every fit is classified first (:data:`OU_STATES`):

``valid``
    :math:`0 < b < 1` and the half-life is reportable.
``near_unit_root``
    :math:`0 < b < 1` but the half-life exceeds the configured maximum
    (``b > 0.99993`` for 10,000 bars). Computable, meaningless: the
    equilibrium is unidentified. No theta, mu, sigma or half-life is reported.
``unit_root``
    :math:`|b - 1|` below the tolerance. No finite half-life exists.
``explosive``
    :math:`b > 1`. Deviations grow; there is no reversion to speak of.
``non_positive``
    :math:`b \le 0`. Sign-flipping discrete dynamics with no continuous OU
    counterpart; :math:`\ln b` is never taken.
``outside_valid_range``
    Inside (0, 1) but outside a narrower configured range.
``degenerate``
    The regressor has (almost) no variance - a flat residual window - so
    :math:`b` itself is undefined.
``insufficient_data``
    The window is incomplete (warm-up) or contains missing values.

Only ``valid`` rows receive OU parameters; every other row keeps its AR(1)
coefficients (where they exist) and a state that says why.

Numerics
--------
The rolling estimator uses running sums, not one regression per bar. Sums are
accumulated in blocks, each shifted by a constant taken from the block's first
finite observation, so rounding error does not grow with series length. The
shift depends only on data at or before every output that uses it, so no
output can depend - even in its last bit - on a later observation. The tests
check this to exact equality.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "OU_STATES",
    "AR1Fit",
    "OUMapping",
    "OUParameters",
    "OUValidityRules",
    "RollingAR1",
    "RollingOU",
    "ar1_coefficients",
    "decay_ratio",
    "expected_value",
    "fit_ar1",
    "fit_ar1_pairs",
    "half_life_bars_from_b",
    "innovation_variance",
    "map_ar1_to_ou",
    "rolling_ar1",
    "rolling_ou",
    "simulate_ar1",
    "simulate_ou",
    "simulate_random_walk",
    "stationary_std",
    "theta_from_b",
]

STATE_VALID = "valid"
STATE_NEAR_UNIT_ROOT = "near_unit_root"
STATE_UNIT_ROOT = "unit_root"
STATE_EXPLOSIVE = "explosive"
STATE_NON_POSITIVE = "non_positive"
STATE_OUTSIDE_RANGE = "outside_valid_range"
STATE_DEGENERATE = "degenerate"
STATE_INSUFFICIENT = "insufficient_data"

#: Every state, in code order (the int8 codes used internally index this tuple).
OU_STATES: tuple[str, ...] = (
    STATE_VALID,
    STATE_NEAR_UNIT_ROOT,
    STATE_UNIT_ROOT,
    STATE_EXPLOSIVE,
    STATE_NON_POSITIVE,
    STATE_OUTSIDE_RANGE,
    STATE_DEGENERATE,
    STATE_INSUFFICIENT,
)
_CODE = {name: np.int8(i) for i, name in enumerate(OU_STATES)}
_LN2 = math.log(2.0)


@dataclass(frozen=True, slots=True)
class OUValidityRules:
    """Which AR(1) fits receive an OU reading, and the numerical floors."""

    dt: float = 1.0
    valid_b_min: float = 0.0
    valid_b_max: float = 1.0
    unit_root_tolerance: float = 1e-9
    max_half_life_bars: float = 10_000.0
    near_unit_root_threshold: float = 0.99
    min_regressor_variance: float = 1e-20
    zero_innovation_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt must be > 0")
        if not 0.0 <= self.valid_b_min < self.valid_b_max <= 1.0:
            raise ValueError("need 0 <= valid_b_min < valid_b_max <= 1")
        if self.max_half_life_bars <= 1:
            raise ValueError("max_half_life_bars must be > 1")


# ---------------------------------------------------------------------------
# Closed forms (scalars or arrays; NaN wherever the mapping does not exist)
# ---------------------------------------------------------------------------
def _as_float_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def theta_from_b(b: Any, dt: float = 1.0) -> Any:
    r""":math:`\theta = -\ln(b)/\Delta t` for :math:`0 < b < 1`, else NaN."""
    arr = _as_float_array(b)
    inside = (arr > 0) & (arr < 1)
    # log1p is more precise near b = 1; log is more precise near b = 0.
    safe = np.where(inside, arr, 0.5)
    out = np.where(inside, np.where(safe > 0.5, -np.log1p(safe - 1.0), -np.log(safe)) / dt, np.nan)
    return float(out) if np.ndim(b) == 0 else out


def half_life_bars_from_b(b: Any) -> Any:
    r""":math:`HL = -\ln 2 / \ln b` in bars for :math:`0 < b < 1`, else NaN.

    The formula is never applied outside (0, 1): a negative or unit ``b`` has
    no half-life, and ``ln`` of it would only manufacture a misleading number.
    """
    theta = _as_float_array(theta_from_b(b, 1.0))
    with np.errstate(divide="ignore"):
        out = np.where(theta > 0, _LN2 / np.where(theta > 0, theta, 1.0), np.nan)
    return float(out) if np.ndim(b) == 0 else out


def stationary_std(sigma: Any, theta: Any) -> Any:
    r""":math:`\sigma/\sqrt{2\theta}` for :math:`\theta > 0`, else NaN."""
    s, t = _as_float_array(sigma), _as_float_array(theta)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(t > 0, s / np.sqrt(2.0 * np.where(t > 0, t, 1.0)), np.nan)
    return float(out) if np.ndim(out) == 0 else out


def innovation_variance(theta: float, sigma: float, dt: float = 1.0) -> float:
    r"""Exact discrete innovation variance :math:`\sigma^2(1-e^{-2\theta\Delta t})/(2\theta)`."""
    if theta <= 0:
        raise ValueError("theta must be > 0")
    return sigma * sigma * (-math.expm1(-2.0 * theta * dt)) / (2.0 * theta)


def ar1_coefficients(theta: float, mu: float, dt: float = 1.0) -> tuple[float, float]:
    r"""The exact AR(1) :math:`(a, b)` of an OU process sampled every ``dt``."""
    if theta <= 0:
        raise ValueError("theta must be > 0")
    b = math.exp(-theta * dt)
    return mu * (1.0 - b), b


def decay_ratio(theta: Any, h: Any) -> Any:
    r""":math:`e^{-\theta h}`: the share of a deviation expected to remain after ``h``."""
    t, horizon = _as_float_array(theta), _as_float_array(h)
    out = np.where(t > 0, np.exp(-t * horizon), np.nan)
    return float(out) if np.ndim(out) == 0 else out


def expected_value(x: Any, mu: Any, theta: Any, h: Any) -> Any:
    r""":math:`E[X_{t+h} \mid X_t] = \mu + (X_t - \mu)e^{-\theta h}`.

    A model expectation from quantities known at ``t``; it never looks at
    :math:`X_{t+h}`.
    """
    x_, mu_ = _as_float_array(x), _as_float_array(mu)
    out = mu_ + (x_ - mu_) * _as_float_array(decay_ratio(theta, h))
    return float(out) if np.ndim(out) == 0 else out


# ---------------------------------------------------------------------------
# Classification and mapping (vectorised)
# ---------------------------------------------------------------------------
@dataclass
class OUMapping:
    """OU reading of a set of AR(1) fits, element by element."""

    state_code: np.ndarray
    theta: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    stationary_std: np.ndarray
    half_life_bars: np.ndarray
    mr_per_bar: np.ndarray
    valid: np.ndarray
    near_unit_root: np.ndarray
    numerical_warning: np.ndarray

    @property
    def state(self) -> np.ndarray:
        return np.asarray(OU_STATES, dtype=object)[self.state_code]


def map_ar1_to_ou(
    a: Any,
    b: Any,
    innovation_var: Any,
    *,
    rules: OUValidityRules,
    complete: Any = True,
    degenerate: Any = False,
    zero_innovation: Any = False,
) -> OUMapping:
    """Classify each fit and give OU parameters only to the ``valid`` ones."""
    a_ = np.atleast_1d(_as_float_array(a))
    b_ = np.atleast_1d(_as_float_array(b))
    var = np.atleast_1d(_as_float_array(innovation_var))
    n = b_.size
    complete_ = np.broadcast_to(np.asarray(complete, dtype=bool), (n,))
    degenerate_ = np.broadcast_to(np.asarray(degenerate, dtype=bool), (n,))
    zero_ = np.broadcast_to(np.asarray(zero_innovation, dtype=bool), (n,))

    code = np.full(n, _CODE[STATE_INSUFFICIENT], dtype=np.int8)
    finite_b = np.isfinite(b_)
    code[complete_ & (degenerate_ | ~finite_b)] = _CODE[STATE_DEGENERATE]
    ok = complete_ & ~degenerate_ & finite_b
    tol = rules.unit_root_tolerance
    code[ok & (b_ <= 0)] = _CODE[STATE_NON_POSITIVE]
    code[ok & (b_ > 1 + tol)] = _CODE[STATE_EXPLOSIVE]
    code[ok & (np.abs(b_ - 1.0) <= tol)] = _CODE[STATE_UNIT_ROOT]
    inside = ok & (b_ > 0) & (b_ < 1 - tol)
    half_life = np.full(n, np.nan)
    half_life[inside] = half_life_bars_from_b(b_[inside])
    too_long = inside & ~(half_life <= rules.max_half_life_bars)
    code[too_long] = _CODE[STATE_NEAR_UNIT_ROOT]
    reportable = inside & ~too_long
    in_range = (b_ > rules.valid_b_min) & (b_ < rules.valid_b_max)
    code[reportable & ~in_range] = _CODE[STATE_OUTSIDE_RANGE]
    valid = reportable & in_range
    code[valid] = _CODE[STATE_VALID]

    theta = np.full(n, np.nan)
    mu = np.full(n, np.nan)
    sigma = np.full(n, np.nan)
    stat_std = np.full(n, np.nan)
    theta[valid] = theta_from_b(b_[valid], rules.dt)
    mu[valid] = a_[valid] / (1.0 - b_[valid])
    one_minus_b2 = (1.0 - b_[valid]) * (1.0 + b_[valid])
    stat_var = np.where(zero_[valid], 0.0, var[valid]) / one_minus_b2
    stat_std[valid] = np.sqrt(np.maximum(stat_var, 0.0))
    sigma[valid] = np.sqrt(np.maximum(2.0 * theta[valid] * stat_var, 0.0))
    half_life[~valid] = np.nan

    mr_per_bar = np.where(ok, 1.0 - b_, np.nan)
    near = ok & (b_ > rules.near_unit_root_threshold)
    numerical = degenerate_ | (valid & zero_) | (valid & ~(
        np.isfinite(theta) & np.isfinite(mu) & np.isfinite(sigma) & np.isfinite(stat_std)
    ))
    return OUMapping(
        state_code=code, theta=theta, mu=mu, sigma=sigma, stationary_std=stat_std,
        half_life_bars=half_life, mr_per_bar=mr_per_bar, valid=valid,
        near_unit_root=near, numerical_warning=np.asarray(numerical, dtype=bool),
    )


# ---------------------------------------------------------------------------
# Static fit
# ---------------------------------------------------------------------------
@dataclass
class OUParameters:
    """The OU reading of one static fit."""

    state: str
    valid: bool
    theta: float = float("nan")
    mu: float = float("nan")
    sigma: float = float("nan")
    stationary_std: float = float("nan")
    half_life_bars: float = float("nan")
    mr_per_bar: float = float("nan")
    near_unit_root: bool = False
    numerical_warning: bool = False


@dataclass
class AR1Fit:
    r"""OLS fit of :math:`X_{t+1} = a + b X_t + \eta_t` with OU mapping.

    Two sets of standard errors are kept. OLS assumes independent
    innovations; Newey-West (Bartlett kernel) allows serial correlation and
    heteroskedasticity. When they differ materially the innovations are not
    white, which is itself a finding about the model.
    """

    n_pairs: int = 0
    a: float = float("nan")
    b: float = float("nan")
    se_a: float = float("nan")
    se_b: float = float("nan")
    t_a: float = float("nan")
    t_b: float = float("nan")
    df_statistic: float = float("nan")
    hac_lags: int = 0
    hac_se_a: float = float("nan")
    hac_se_b: float = float("nan")
    r_squared: float = float("nan")
    innovation_var: float = float("nan")
    innovation_std: float = float("nan")
    sample_mean: float = float("nan")
    sample_std: float = float("nan")
    ou: OUParameters = field(default_factory=lambda: OUParameters(STATE_INSUFFICIENT, False))
    mu_se: float = float("nan")
    mu_ci_lower: float = float("nan")
    mu_ci_upper: float = float("nan")
    std_ratio: float = float("nan")
    residuals: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("residuals", None)
        return payload


def fit_ar1(
    x: Any,
    *,
    rules: OUValidityRules | None = None,
    hac_lags: int | None = None,
    confidence_level: float = 0.95,
    keep_residuals: bool = True,
) -> AR1Fit:
    r"""Static AR(1)/OU fit on a series; pairs straddling a missing value are dropped.

    A pair :math:`(X_t, X_{t+1})` is used only when both are finite, so a gap
    in the series (warm-up, a degenerate regression window) never bridges two
    unrelated observations.
    """
    values = np.asarray(x, dtype=np.float64)
    u, v = values[:-1], values[1:]
    ok = np.isfinite(u) & np.isfinite(v)
    finite = values[np.isfinite(values)]
    fit = fit_ar1_pairs(u[ok], v[ok], rules=rules, hac_lags=hac_lags,
                        confidence_level=confidence_level, keep_residuals=keep_residuals)
    if finite.size > 1:
        fit.sample_mean = float(finite.mean())
        fit.sample_std = float(finite.std(ddof=1))
        if fit.ou.valid and fit.ou.stationary_std > 0:
            fit.std_ratio = fit.sample_std / fit.ou.stationary_std
    return fit


def fit_ar1_pairs(
    u: Any,
    v: Any,
    *,
    rules: OUValidityRules | None = None,
    hac_lags: int | None = None,
    confidence_level: float = 0.95,
    keep_residuals: bool = True,
) -> AR1Fit:
    """AR(1) OLS on explicit (X_t, X_{t+1}) pairs, e.g. pooled across days."""
    from scipy import stats

    rules = rules or OUValidityRules()
    u_, v_ = np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
    n = int(u_.size)
    fit = AR1Fit(n_pairs=n)
    if n < 3:
        return fit
    u_mean, v_mean = float(u_.mean()), float(v_.mean())
    uc, vc = u_ - u_mean, v_ - v_mean
    sxx = float(uc @ uc)
    syy = float(vc @ vc)
    if sxx <= rules.min_regressor_variance * n:
        fit.ou = OUParameters(STATE_DEGENERATE, False, numerical_warning=True)
        return fit
    b = float(uc @ vc) / sxx
    alpha = v_mean                      # intercept of the centred regression
    a = alpha - b * u_mean
    resid = vc - b * uc
    sse = float(resid @ resid)
    zero = syy <= 0 or sse <= rules.zero_innovation_tolerance * syy
    if zero:
        sse = 0.0
    var = sse / (n - 2)
    fit.a, fit.b = a, b
    fit.innovation_var = var
    fit.innovation_std = math.sqrt(var)
    fit.r_squared = 1.0 - sse / syy if syy > 0 else float("nan")
    fit.se_b = math.sqrt(var / sxx)
    var_alpha = var / n
    cov_alpha_b = 0.0                   # exactly zero with a centred regressor
    fit.se_a = math.sqrt(var_alpha + u_mean * u_mean * fit.se_b ** 2 - 2 * u_mean * cov_alpha_b)
    fit.t_b = b / fit.se_b if fit.se_b > 0 else float("nan")
    fit.t_a = a / fit.se_a if fit.se_a > 0 else float("nan")
    fit.df_statistic = (b - 1.0) / fit.se_b if fit.se_b > 0 else float("nan")

    lags = hac_lags if hac_lags is not None else max(1, int(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    fit.hac_lags = lags
    hac = _newey_west(uc, resid, sxx, n, lags)
    fit.hac_se_b = math.sqrt(hac[1, 1]) if hac[1, 1] >= 0 else float("nan")
    var_a_hac = hac[0, 0] + u_mean * u_mean * hac[1, 1] - 2.0 * u_mean * hac[0, 1]
    fit.hac_se_a = math.sqrt(var_a_hac) if var_a_hac >= 0 else float("nan")

    mapping = map_ar1_to_ou(a, b, var, rules=rules, zero_innovation=zero)
    state = str(mapping.state[0])
    fit.ou = OUParameters(
        state=state,
        valid=bool(mapping.valid[0]),
        theta=float(mapping.theta[0]),
        mu=float(mapping.mu[0]),
        sigma=float(mapping.sigma[0]),
        stationary_std=float(mapping.stationary_std[0]),
        half_life_bars=float(mapping.half_life_bars[0]),
        mr_per_bar=float(mapping.mr_per_bar[0]),
        near_unit_root=bool(mapping.near_unit_root[0]),
        numerical_warning=bool(mapping.numerical_warning[0]),
    )
    if fit.ou.valid:
        # Delta method for mu = a / (1 - b), with the HAC covariance of (a, b).
        cov_ab = hac[0, 1] - u_mean * hac[1, 1]
        grad_a = 1.0 / (1.0 - b)
        grad_b = a / (1.0 - b) ** 2
        mu_var = grad_a ** 2 * var_a_hac + grad_b ** 2 * hac[1, 1] + 2 * grad_a * grad_b * cov_ab
        if mu_var >= 0:
            fit.mu_se = math.sqrt(mu_var)
            z = float(stats.norm.ppf(0.5 + confidence_level / 2.0))
            fit.mu_ci_lower = fit.ou.mu - z * fit.mu_se
            fit.mu_ci_upper = fit.ou.mu + z * fit.mu_se
    if keep_residuals:
        fit.residuals = resid
    return fit


def _newey_west(
    uc: np.ndarray, resid: np.ndarray, sxx: float, n: int, lags: int
) -> np.ndarray:
    """HAC covariance of (alpha, b) for the regression on a centred regressor.

    Returned as a 2x2 matrix for the parameters (alpha, b), where alpha is the
    intercept of the centred regression; callers convert to (a, b).
    """
    scores = np.column_stack((resid, uc * resid))           # (n, 2)
    meat = scores.T @ scores
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = scores[lag:].T @ scores[:-lag]
        meat += weight * (gamma + gamma.T)
    bread = np.diag([1.0 / n, 1.0 / sxx])
    return bread @ meat @ bread


# ---------------------------------------------------------------------------
# Rolling estimation
# ---------------------------------------------------------------------------
@dataclass
class RollingAR1:
    """Per-bar AR(1) fits over a trailing window of ``window`` observations."""

    window: int
    a: np.ndarray
    b: np.ndarray
    r_squared: np.ndarray
    innovation_var: np.ndarray
    se_b: np.ndarray
    window_mean: np.ndarray
    window_std: np.ndarray
    complete: np.ndarray
    degenerate: np.ndarray
    zero_innovation: np.ndarray

    @property
    def n_pairs(self) -> int:
        return self.window - 1


def rolling_ar1(
    x: Any,
    window: int,
    *,
    chunk_rows: int = 16_384,
    min_regressor_variance: float = 1e-20,
    zero_innovation_tolerance: float = 1e-12,
) -> RollingAR1:
    r"""OLS of :math:`X_{i+1}` on :math:`X_i` over each trailing window.

    The value at ``t`` uses the ``window`` observations :math:`X_{t-w+1..t}`,
    i.e. the ``window-1`` pairs inside them, and nothing else. A window with
    any missing observation is incomplete and yields NaN. The first
    ``window-1`` values are always NaN (warm-up); nothing is back-filled.

    Sums come from prefix sums over blocks of ``chunk_rows`` outputs. Each
    block subtracts a shift equal to the first finite observation of the data
    it reads; that observation lies at or before every output in the block
    that can be complete, so the arithmetic of an output never depends on a
    later value - not even in rounding.
    """
    values = np.asarray(x, dtype=np.float64)
    n = values.size
    if window < 4:
        raise ValueError("an AR(1) window needs at least 4 observations (3 pairs)")
    k = window - 1                                   # pairs per window
    shape = (n,)
    out = {name: np.full(shape, np.nan) for name in (
        "a", "b", "r_squared", "innovation_var", "se_b", "window_mean", "window_std")}
    complete = np.zeros(shape, dtype=bool)
    degenerate = np.zeros(shape, dtype=bool)
    zero_innovation = np.zeros(shape, dtype=bool)
    result = RollingAR1(window=window, complete=complete, degenerate=degenerate,
                        zero_innovation=zero_innovation, **out)
    if n < window:
        return result

    finite = np.isfinite(values)
    for block_start in range(window - 1, n, chunk_rows):
        block_end = min(block_start + chunk_rows, n)
        lo = block_start - (window - 1)
        seg = values[lo:block_end]
        fin = finite[lo:block_end]
        first = np.flatnonzero(fin)
        shift = float(seg[first[0]]) if first.size else 0.0
        y = np.where(fin, seg - shift, 0.0)
        count = np.concatenate(([0], np.cumsum(fin, dtype=np.int64)))
        s1 = np.concatenate(([0.0], np.cumsum(y)))
        s2 = np.concatenate(([0.0], np.cumsum(y * y)))
        s12 = np.concatenate(([0.0], np.cumsum(y[:-1] * y[1:])))
        # Local index of the window's last observation, for every output.
        tl = np.arange(block_start - lo, block_end - lo)
        first_obs = tl - (window - 1)
        full = (count[tl + 1] - count[first_obs]) == window
        su = s1[tl] - s1[first_obs]                   # regressors: X[first .. t-1]
        sv = s1[tl + 1] - s1[first_obs + 1]           # responses: X[first+1 .. t]
        suu = s2[tl] - s2[first_obs]
        svv = s2[tl + 1] - s2[first_obs + 1]
        suv = s12[tl] - s12[first_obs]
        sxx = suu - su * su / k
        syy = svv - sv * sv / k
        sxy = suv - su * sv / k
        flat = sxx <= min_regressor_variance * k
        usable = full & ~flat
        with np.errstate(invalid="ignore", divide="ignore"):
            b = np.where(usable, sxy / np.where(usable, sxx, 1.0), np.nan)
            a_shifted = (sv - b * su) / k
            a = a_shifted + shift * (1.0 - b)
            sse = np.maximum(syy - b * sxy, 0.0)
            zero = usable & ((syy <= 0) | (sse <= zero_innovation_tolerance * syy))
            sse = np.where(zero, 0.0, sse)
            var = sse / (k - 2)
            r2 = np.where(syy > 0, 1.0 - sse / np.where(syy > 0, syy, 1.0), np.nan)
            se_b = np.sqrt(var / np.where(usable, sxx, 1.0))
            s_all1 = s1[tl + 1] - s1[first_obs]
            s_all2 = s2[tl + 1] - s2[first_obs]
            mean = s_all1 / window + shift
            wvar = np.maximum((s_all2 - s_all1 * s_all1 / window) / (window - 1), 0.0)
        idx = slice(block_start, block_end)
        result.a[idx] = np.where(usable, a, np.nan)
        result.b[idx] = b
        result.r_squared[idx] = np.where(usable, r2, np.nan)
        result.innovation_var[idx] = np.where(usable, var, np.nan)
        result.se_b[idx] = np.where(usable, se_b, np.nan)
        result.window_mean[idx] = np.where(full, mean, np.nan)
        result.window_std[idx] = np.where(full, np.sqrt(wvar), np.nan)
        complete[idx] = full
        degenerate[idx] = full & flat
        zero_innovation[idx] = zero
    return result


@dataclass
class RollingOU:
    """Rolling AR(1) fits with their OU reading, Z-score and one-step innovations."""

    fits: RollingAR1
    mapping: OUMapping
    zscore: np.ndarray
    std_ratio: np.ndarray
    innovation: np.ndarray
    innovation_standardized: np.ndarray


def rolling_ou(
    x: Any,
    window: int,
    *,
    rules: OUValidityRules,
    chunk_rows: int = 16_384,
) -> RollingOU:
    r"""Rolling OU estimation, every output causal.

    ``zscore`` is :math:`(X_t - \mu_t)/\sigma_{stat,t}` from the window ending
    at ``t``. ``innovation`` is the one-step forecast error of the fit made at
    ``t-1``: :math:`X_t - (a_{t-1} + b_{t-1}X_{t-1})`, known at ``t``.
    """
    values = np.asarray(x, dtype=np.float64)
    fits = rolling_ar1(
        values, window, chunk_rows=chunk_rows,
        min_regressor_variance=rules.min_regressor_variance,
        zero_innovation_tolerance=rules.zero_innovation_tolerance,
    )
    mapping = map_ar1_to_ou(
        fits.a, fits.b, fits.innovation_var, rules=rules, complete=fits.complete,
        degenerate=fits.degenerate, zero_innovation=fits.zero_innovation,
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        usable = mapping.valid & (mapping.stationary_std > 0)
        zscore = np.where(
            usable, (values - mapping.mu) / np.where(usable, mapping.stationary_std, 1.0), np.nan
        )
        std_ratio = np.where(
            usable, fits.window_std / np.where(usable, mapping.stationary_std, 1.0), np.nan
        )
    innovation = np.full(values.size, np.nan)
    standardized = np.full(values.size, np.nan)
    if values.size > 1:
        predicted = fits.a[:-1] + fits.b[:-1] * values[:-1]
        err = values[1:] - predicted
        defined = np.isfinite(err) & np.isfinite(fits.b[:-1])
        innovation[1:] = np.where(defined, err, np.nan)
        scale = np.sqrt(fits.innovation_var[:-1])
        with np.errstate(invalid="ignore", divide="ignore"):
            ok = defined & (scale > 0)
            standardized[1:] = np.where(ok, err / np.where(ok, scale, 1.0), np.nan)
    return RollingOU(fits=fits, mapping=mapping, zscore=zscore, std_ratio=std_ratio,
                     innovation=innovation, innovation_standardized=standardized)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def simulate_ar1(
    a: float,
    b: float,
    innovation_std: float,
    n: int,
    *,
    x0: float | None = None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    r""":math:`X_{t+1} = a + bX_t + \eta_t`, :math:`\eta \sim N(0, s^2)`.

    ``x0`` defaults to a draw from the stationary distribution when
    :math:`|b| < 1`, so the path is stationary from its first value.
    """
    from scipy.signal import lfilter

    generator = rng or np.random.default_rng(seed)
    if x0 is None:
        if abs(b) < 1:
            x0 = a / (1.0 - b) + innovation_std / math.sqrt(1.0 - b * b) * generator.standard_normal()
        else:
            x0 = 0.0
    if n <= 0:
        return np.empty(0)
    shocks = a + innovation_std * generator.standard_normal(n - 1)
    rest = lfilter([1.0], [1.0, -b], shocks, zi=[b * x0])[0] if n > 1 else np.empty(0)
    return np.concatenate(([x0], rest))


def simulate_ou(
    theta: float,
    mu: float,
    sigma: float,
    n: int,
    *,
    dt: float = 1.0,
    x0: float | None = None,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    r"""Exact discretisation of the OU process - no Euler error at any ``dt``."""
    a, b = ar1_coefficients(theta, mu, dt)
    return simulate_ar1(a, b, math.sqrt(innovation_variance(theta, sigma, dt)), n,
                        x0=x0, seed=seed, rng=rng)


def simulate_random_walk(
    n: int,
    step_std: float,
    *,
    x0: float = 0.0,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """A driftless Gaussian random walk: no mean reversion at all."""
    generator = rng or np.random.default_rng(seed)
    return x0 + np.concatenate(([0.0], np.cumsum(step_std * generator.standard_normal(n - 1))))
