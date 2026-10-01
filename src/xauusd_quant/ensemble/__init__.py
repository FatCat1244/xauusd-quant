"""Model ensembles, meta-models and predictive diversification (Prompt #11).

Combines the Prompt #10 walk-forward out-of-sample predictions of genuinely
distinct models - simple and median averages, performance- and diversity-aware
weights, stacked logistic / ridge meta-models, regime- / volatility-conditioned
and trailing-health weights - and asks whether the combination is more stable,
better calibrated and more informative out of sample than one model. Everything
fitted at the second level is fitted on earlier out-of-sample blocks only.

Outputs are probabilities, expected values and uncertainty proxies (model
disagreement, ensemble entropy, OOD score) - never a decision, a threshold, a
size or a PnL. Imported lazily; needs the ``ml`` extra.
"""
