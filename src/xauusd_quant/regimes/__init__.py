"""Unsupervised market-regime discovery (Prompt #7).

Does XAUUSD repeatedly enter statistically distinct states that can be
identified from information available at the time? A regime is a latent state
:math:`S_t \\in \\{0..K-1\\}` that shapes the distribution of the features
:math:`X_t`; the layer estimates :math:`P(S_t = k \\mid X_{\\le t})` and asks
whether those states are persistent, stable, more than volatility buckets and
the time of day, and informative beyond the continuous features.

Module map::

    config.py            typed configuration (config/regimes.yaml)
    dataset.py           the compact causal feature space, its manifest, the gate
    preprocessing.py     scalers fitted on training rows only; the missing-feature policy
    emissions.py         Gaussian state densities (exact marginalisation, deterministic)
    clustering.py        K-Means baseline
    gmm.py               Gaussian mixture (full / diag / tied)
    hmm.py               Gaussian hidden Markov model: filter (live safe), smoother
                         and Viterbi (offline), Baum-Welch
    state_alignment.py   label switching: Hungarian matching on Bhattacharyya distance
    transitions.py       durations, regime age, transition features and calibration
    diagnostics.py       degenerate states, agreement between labelings, contribution
    registry.py          immutable model artefacts (HMM_5M_K4_00017)
    causal_inference.py  walk-forward refits and the live-safe regime feature table
    synthetic.py         known-HMM, single-regime and smooth-continuum controls
    changepoints.py      PELT structural breaks (research only)

Two categories are kept strictly apart. **Offline** results (full-sample
fits, smoothed probabilities, Viterbi paths) are labelled
``OFFLINE / NON-CAUSAL RESEARCH ONLY`` and never stored as features.
**Causal** results come from walk-forward refits (past data only, a frozen
scaler and model between refits) and the forward filter; only they are
written to ``data/features/regime`` - and the schema there refuses anything
else. Nothing here defines a signal, a trade, a position or a cost.
"""

from __future__ import annotations

from .config import MODEL_NAMES, OFFLINE_LABEL, RegimeConfig, load_regime_config

__all__ = ["MODEL_NAMES", "OFFLINE_LABEL", "RegimeConfig", "load_regime_config"]
