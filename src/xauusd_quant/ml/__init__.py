"""Supervised predictive-model research (Prompt #10).

Can supervised models extract stable out-of-sample information from the
Prompt #9 feature manifests? Models output probabilities, expected values and
uncertainty proxies - never orders, sizes or PnL. Development is walk-forward
over 2003-2021 with purging and embargo; the reserved test period is read only
by :mod:`.final_test`, for frozen model specifications.

Modules: :mod:`.config`, :mod:`.datasets`, :mod:`.splits`,
:mod:`.preprocessing`, :mod:`.models`, :mod:`.calibration`,
:mod:`.evaluation`, :mod:`.training`, :mod:`.explainability`,
:mod:`.diagnostics`, :mod:`.registry`, :mod:`.streaming`, :mod:`.final_test`.
The third-party model libraries are imported lazily (``pip install -e .[ml]``).
"""
