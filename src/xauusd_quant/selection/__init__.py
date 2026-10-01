"""Feature selection and dimensionality reduction (Prompt #9).

Which compact, causal, live-safe, low-redundancy and time-robust subset of the
Prompt #8 feature factory may enter supervised learning (Prompt #10)?

Selection is part of model fitting: every outcome-based statistic is computed
inside a chronological training span (the development period, a nested
training fold, or a block resample of development quarters); the reserved
test period's outcomes are never read. Modules:

``config``        typed ``config/feature_selection.yaml``
``periods``       development / validation / reserved periods, purge, embargo, guard
``data``          development + validation features, outcome-safe targets, probes
``filters``       the quality filter with reason codes and drift classes
``relevance``     in-span IC, stability, max-T null screen, pipeline veto
``redundancy``    correlation, collinearity, effective rank
``clustering``    hierarchical clusters, representatives, parameter families
``mrmr``          mRMR and the simpler rankings it is compared with
``linear``        rank-linear ridge / Lasso / elastic net / L1 logistic diagnostics
``pca``           training-fitted PCA and its predictive comparison
``stability``     chronological resamples, selection frequency, Jaccard, rank stability
``selection_cv``  nested selection and evaluation with purging and embargo
``ablation``      family ablation, leave-one-family-out, block permutation importance
``manifest``      immutable ordered feature-set manifests
``live``          bar-by-bar live reconstruction of a selected set

Nothing here trains the final model, defines a trade or reads a PnL.
"""
