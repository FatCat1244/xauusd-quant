"""The pre-registered ensemble freeze rule (Prompt #11, Steps 54, 59-61, 84; Rule 3).

Candidates in complexity order replace the one retained so far only when they are
better on the primary metric block by block by more than one SE in at least
``min_wins`` blocks, and - for probabilities - do not raise the mean ECE by more than
``max_ece_increase``. What the rule retains is what is frozen: the single model
when no ensemble earns its complexity. Dyadic values keep the block differences
exact (an SE of exactly 0 is a consistent difference, not a missing one).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from xauusd_quant.ensemble.config import load_ensemble_config
from xauusd_quant.research.ensemble_reports import calibration_decision, freeze_decision

BLOCKS = [1, 2, 3, 4]
BEST = [64 / 512, 66 / 512, 62 / 512, 68 / 512]


class _Run:
    """Just what the rule reads from a PairRun."""

    def __init__(self, series: dict[str, list[float]], *, classification: bool = True,
                 ece: dict[str, float] | None = None,
                 log_loss: dict[str, list[float]] | None = None) -> None:
        self.cfg = load_ensemble_config()
        self.series = dict.fromkeys(series)
        self._vals = series
        self._ece = ece or {}
        self._ll = log_loss or {}
        self.eval_blocks = BLOCKS
        self.pair = SimpleNamespace(is_classification=classification)

    def _candidate(self, method: str) -> str:
        return {"performance_weighted": "performance_weighted_s50",
                "diversity_weighted": "diversity_weighted_l50"}.get(method, method)

    def primary(self, name: str, k: int) -> float | None:
        return self._vals[name][BLOCKS.index(k)]

    def block_score(self, name: str, k: int) -> dict[str, Any]:
        out: dict[str, Any] = {"ece": self._ece.get(name, 0.01)}
        if name in self._ll:
            out["log_loss"] = self._ll[name][BLOCKS.index(k)]
        return out


def _plus(values: list[float], deltas: list[float]) -> list[float]:
    return [v + d for v, d in zip(values, deltas, strict=True)]


def test_no_ensemble_beyond_one_se_keeps_the_single_model() -> None:
    run = _Run({"best_individual": BEST,
                "simple_average": _plus(BEST, [1 / 512, -1 / 512, 1 / 512, -1 / 512]),
                "stacking": _plus(BEST, [-2 / 512] * 4)})
    d = freeze_decision(run)                               # type: ignore[arg-type]
    assert d["single_model_retained"] is True
    assert d["frozen_method"] == "best_individual"       # what the rule retained is frozen
    assert all(s["decision"] in ("does not earn its complexity", "not run")
               for s in d["steps"])


def test_a_consistent_gain_in_three_blocks_beyond_one_se_replaces_the_retained() -> None:
    run = _Run({"best_individual": BEST,
                "simple_average": _plus(BEST, [4 / 512, 4 / 512, 4 / 512, -1 / 512]),
                "stacking": _plus(BEST, [5 / 512, 4 / 512, 4 / 512, 0.0])})
    d = freeze_decision(run)                               # type: ignore[arg-type]
    assert d["retained"] == "simple_average" and d["frozen_method"] == "simple_average"
    step = next(s for s in d["steps"] if s["method"] == "stacking")
    assert step["versus"] == "simple_average"            # compared with what was retained
    # +0.5/512 on average, beyond one SE, but better in only 2 of 4 blocks (< min_wins 3)
    assert step["wins"] == 2 and step["decision"] == "does not earn its complexity"


def test_two_wins_are_not_enough_and_ece_must_not_worsen() -> None:
    run = _Run({"best_individual": BEST,
                "simple_average": _plus(BEST, [8 / 512, 8 / 512, -1 / 512, -1 / 512])})
    assert freeze_decision(run)["single_model_retained"] is True   # type: ignore[arg-type]
    worse_ece = _Run({"best_individual": BEST,
                      "simple_average": _plus(BEST, [4 / 512] * 4)},
                     ece={"best_individual": 0.010, "simple_average": 0.020})
    d = freeze_decision(worse_ece)                         # type: ignore[arg-type]
    assert d["single_model_retained"] is True
    assert d["steps"][0]["ece_note"].startswith("ECE 0.0200")
    # an identical gain in every block (SE exactly 0) is a consistent gain, not a missing SE
    same = _Run({"best_individual": BEST, "simple_average": _plus(BEST, [2 / 512] * 4)})
    assert freeze_decision(same)["retained"] == "simple_average"  # type: ignore[arg-type]


def test_regression_ignores_calibration_and_the_calibration_rule_prefers_simple() -> None:
    run = _Run({"best_individual": BEST, "stacking": _plus(BEST, [4 / 512] * 4)},
               classification=False, ece={"stacking": 0.5})
    assert freeze_decision(run)["retained"] == "stacking"   # type: ignore[arg-type]
    ll_none = [0.6900, 0.6905, 0.6898, 0.6902]
    cal = _Run({"calibration:A_calibrate_then_average": BEST,
                "calibration:C_both_sigmoid": BEST, "calibration:C_both_isotonic": BEST},
               log_loss={"calibration:A_calibrate_then_average": ll_none,
                         "calibration:C_both_sigmoid": [v - 0.00001 for v in ll_none],
                         "calibration:C_both_isotonic": [v - 0.003 for v in ll_none]})
    out = calibration_decision(cal)                        # type: ignore[arg-type]
    assert out["calibration"] == "isotonic"               # clearly lower log loss
    tiny = _Run({"calibration:A_calibrate_then_average": BEST,
                 "calibration:C_both_sigmoid": BEST},
                log_loss={"calibration:A_calibrate_then_average": ll_none,
                          "calibration:C_both_sigmoid": [ll_none[0] - 0.0001, ll_none[1] + 0.0001,
                                                         ll_none[2] - 0.0001, ll_none[3]]})
    assert calibration_decision(tiny)["calibration"] == "none"   # type: ignore[arg-type]
