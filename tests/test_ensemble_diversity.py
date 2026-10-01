"""Pin Prompt #11 Steps 6-14, 27-32, 46-48, 55-58 and 77: causal weights and diversity."""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pytest

from xauusd_quant.ensemble.averaging import median_combination, weighted_mean
from xauusd_quant.ensemble.disagreement import binary_entropy, disagreement_measures
from xauusd_quant.ensemble.diversity import diversity_table
from xauusd_quant.ensemble.weighting import (
    apply_state_weights,
    conditional_weights,
    diversity_weights,
    duplicate_clusters,
    dynamic_weights,
    performance_weights,
    turnover,
    weight_entropy,
)


def test_exact_copy_has_identical_predictions_errors_and_correctness() -> None:
    probabilities = np.tile([0.1, 0.3, 0.7, 0.9], 6)
    labels = np.tile([0.0, 1.0, 1.0, 0.0], 6)
    table = diversity_table(
        np.column_stack([probabilities, probabilities.copy()]),
        labels,
        ["original", "copy"],
        task="classification",
    )
    assert len(table) == 1
    record = table[0]
    assert record["rows"] == 24
    assert record["sampled"] is False
    assert record["pearson"] == pytest.approx(1.0)
    assert record["error_correlation"] == pytest.approx(1.0)
    assert record["mean_abs_disagreement"] == pytest.approx(0.0)
    # Both copies have 12 correct and 12 incorrect rows: Q = 144 / 144.
    assert record["q_statistic"] == pytest.approx(1.0)


def test_disagreement_matches_brute_force_pairs_and_propagates_missing() -> None:
    rng = np.random.default_rng(1108)
    predictions = rng.normal(size=(12, 5))
    measures = disagreement_measures(predictions)
    brute = np.mean(
        np.column_stack(
            [
                np.abs(predictions[:, i] - predictions[:, j])
                for i, j in combinations(range(5), 2)
            ]
        ),
        axis=1,
    )
    assert measures["pairwise_disagreement"] == pytest.approx(brute)
    assert measures["prediction_range"] == pytest.approx(
        predictions.max(axis=1) - predictions.min(axis=1)
    )
    assert measures["prediction_std"] == pytest.approx(predictions.std(axis=1))

    missing = predictions.copy()
    missing[3, 2] = np.nan
    missing_measures = disagreement_measures(missing)
    for name, values in missing_measures.items():
        assert np.isnan(values[3])
        keep = np.arange(12) != 3
        np.testing.assert_array_equal(values[keep], measures[name][keep])


def test_binary_entropy_at_half_and_missing() -> None:
    entropy = binary_entropy(np.array([0.5, np.nan]))
    assert entropy[0] == pytest.approx(math.log(2))
    assert np.isnan(entropy[1])


@pytest.mark.parametrize(
    "weights",
    [
        np.array([0.2, 0.2]),
        np.array([[0.5, 0.5], [0.2, 0.2]]),
    ],
)
def test_weighted_mean_refuses_weights_not_summing_to_one(weights: np.ndarray) -> None:
    with pytest.raises(ValueError, match="sum"):
        weighted_mean(np.array([[1.0, 3.0], [2.0, 4.0]]), weights)


@pytest.mark.parametrize("per_row", [False, True])
def test_combinations_propagate_missing_without_renormalising(per_row: bool) -> None:
    predictions = np.array([[1.0, 3.0], [2.0, np.nan]])
    weights = np.array([0.75, 0.25])
    if per_row:
        weights = np.tile(weights, (2, 1))
    mean = weighted_mean(predictions, weights)
    assert mean[0] == pytest.approx(1.5)
    assert np.isnan(mean[1])
    median = median_combination(predictions)
    assert median[0] == pytest.approx(2.0)
    assert np.isnan(median[1])

    # A missing constituent remains missing even when its assigned weight is zero.
    zero_weight = weighted_mean(predictions, np.array([1.0, 0.0]))
    assert np.isnan(zero_weight[1])


def test_performance_weights_positive_skill_shrink_and_equal_fallback() -> None:
    labels = np.zeros(4)
    base = np.ones(4)
    predictions = np.tile([0.0, 0.5, 2.0], (4, 1))
    weights, quality = performance_weights(predictions, labels, base, "regression")
    # Base loss is 4; model losses are 0, 1 and 16.
    assert quality == pytest.approx([1.0, 0.75, -3.0])
    assert weights == pytest.approx([4 / 7, 3 / 7, 0.0])
    assert weights.sum() == pytest.approx(1.0)
    assert weights[2] == 0.0

    shrunk, shrunk_quality = performance_weights(
        predictions, labels, base, "regression", shrink=0.5
    )
    assert shrunk == pytest.approx((weights + 1 / 3) / 2)
    assert shrunk.sum() == pytest.approx(1.0)
    np.testing.assert_array_equal(shrunk_quality, quality)

    unskilled = np.tile([1.0, 2.0, 3.0], (4, 1))
    fallback, fallback_quality = performance_weights(
        unskilled, labels, base, "regression"
    )
    assert (fallback_quality <= 0).all()
    assert fallback == pytest.approx([1 / 3] * 3)


def test_duplicate_clusters_include_constant_copies_and_near_copies() -> None:
    original = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    near = original.copy()
    near[2] = 0.01
    # corr(original, near)^2 = 10 / 10.00008, hence corr > 0.999.
    predictions = np.column_stack(
        [original, original.copy(), near, np.ones(5), np.ones(5), -original]
    )
    assert duplicate_clusters(predictions, threshold=0.999) == [
        [0, 1, 2], [3, 4], [5]
    ]


def test_diversity_cluster_weight_is_unchanged_by_an_exact_copy() -> None:
    original = np.array([-1.0, -0.5, 0.5, 1.0])
    other = np.array([0.5, -1.0, 1.0, -0.5])
    labels = original + np.array([0.125, -0.125, 0.25, -0.25])
    base = np.full(4, 3.0)
    predictions = np.column_stack([original, other])
    weights, details = diversity_weights(
        predictions,
        labels,
        base,
        "regression",
        lam=0.4,
        shrink=0.2,
        duplicate_correlation=0.999,
    )
    copied_weights, copied_details = diversity_weights(
        np.column_stack([original, other, original.copy()]),
        labels,
        base,
        "regression",
        lam=0.4,
        shrink=0.2,
        duplicate_correlation=0.999,
    )
    assert details["clusters"] == [[0], [1]]
    assert copied_details["clusters"] == [[0, 2], [1]]
    assert weights.sum() == pytest.approx(1.0)
    assert copied_weights.sum() == pytest.approx(1.0)
    assert copied_weights[0] == pytest.approx(copied_weights[2])
    assert copied_weights[0] + copied_weights[2] == weights[0]
    assert copied_weights[1] == weights[1]


def test_conditional_weights_use_effective_rows_and_missing_state_fallback() -> None:
    labels = np.zeros(4)
    base = np.ones(4)
    predictions = np.array([[0.0, 2.0], [0.0, 2.0], [2.0, 0.0], [2.0, 0.0]])
    state_prob = np.array([[1.0, 0.0], [1.0, 0.0], [0.5, 0.5], [np.nan, np.nan]])
    state_weights, unconditional, own = conditional_weights(
        predictions,
        labels,
        base,
        "regression",
        state_prob,
        shrink=0.0,
        min_rows=1.0,
    )
    # State 0 has 2.5 effective rows and losses (2, 8), giving skills (0.2, -2.2).
    # State 1 has only 0.5 effective rows, so it uses unconditional equal weights.
    assert own == [True, False]
    assert state_weights[0] == pytest.approx([1.0, 0.0])
    assert unconditional == pytest.approx([0.5, 0.5])
    np.testing.assert_array_equal(state_weights[1], unconditional)
    assert state_weights.sum(axis=1) == pytest.approx([1.0, 1.0])

    applied = apply_state_weights(state_prob, state_weights, unconditional)
    assert applied.sum(axis=1) == pytest.approx(np.ones(4))
    assert applied[0] == pytest.approx([1.0, 0.0])
    assert applied[2] == pytest.approx([0.75, 0.25])
    np.testing.assert_array_equal(applied[3], unconditional)


def test_dynamic_weights_ignore_invalid_rows_instead_of_inventing_losses() -> None:
    """A row without a prediction or a label keeps its place in time but adds no loss and
    does not count toward min_rows (Codex review 2: such rows once got made-up values)."""
    rows = np.arange(8, dtype=np.int64)
    days = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    labels = np.array([1.0, 0.0, 1.0, np.nan, 0.0, 1.0, 0.0, 1.0])
    p = np.array([[0.9, 0.5], [0.1, 0.5], [0.9, 0.5], [0.9, 0.5],
                  [0.1, 0.5], [0.9, 0.5], [0.1, 0.5], [0.9, 0.5]])
    p[2, 1] = np.nan                                       # a missing constituent output
    base = np.full(8, 0.5)
    kw = {"task": "classification", "horizon": 1, "window_days": 5, "shrink": 0.0}
    out = dynamic_weights(p, labels, base, rows, days, np.array([5, 6, 7]), min_rows=2, **kw)
    # resolved before bar 4 (row + 1 < 4): rows 0-2; row 2 is invalid -> rows 0, 1 only
    assert out.update_rows_used.tolist() == [2]
    np.testing.assert_allclose(out.row_weights[0], [1.0, 0.0])     # model 0 had the skill
    strict = dynamic_weights(p, labels, base, rows, days, np.array([5, 6, 7]), min_rows=3, **kw)
    np.testing.assert_allclose(strict.row_weights[0], [0.5, 0.5])  # too few valid rows
    masked = dynamic_weights(p, labels, base, rows, days, np.array([5, 6, 7]), min_rows=2,
                             valid=np.array([False, True, True, True, True, True, True, True]),
                             **kw)
    assert masked.update_rows_used.tolist() == [1]


def test_dynamic_weights_are_bitwise_invariant_when_later_rows_are_appended() -> None:
    rng = np.random.default_rng(1132)
    n = 12
    labels = rng.normal(size=n)
    predictions = np.column_stack(
        [labels + rng.normal(scale=0.1, size=n), rng.normal(size=n)]
    )
    base = np.full(n, 2.0)
    rows = np.arange(n, dtype=np.int64)
    days = np.repeat(np.arange(4), 3)
    evaluate = np.arange(3, n, dtype=np.int64)
    prefix = dynamic_weights(
        predictions,
        labels,
        base,
        rows,
        days,
        evaluate,
        task="regression",
        horizon=1,
        window_days=3,
        min_rows=2,
        shrink=0.25,
    )
    later_labels = np.array([1e6, -1e6, 2e6, -2e6, 3e6, -3e6])
    later_predictions = np.column_stack([later_labels, -later_labels])
    extended = dynamic_weights(
        np.vstack([predictions, later_predictions]),
        np.concatenate([labels, later_labels]),
        np.concatenate([base, np.ones(6)]),
        np.arange(n + 6, dtype=np.int64),
        np.concatenate([days, np.repeat([4, 5], 3)]),
        np.arange(3, n + 6, dtype=np.int64),
        task="regression",
        horizon=1,
        window_days=3,
        min_rows=2,
        shrink=0.25,
    )
    np.testing.assert_array_equal(
        prefix.row_weights.view(np.uint64),
        extended.row_weights[: evaluate.size].view(np.uint64),
    )
    np.testing.assert_array_equal(
        prefix.update_weights.view(np.uint64),
        extended.update_weights[: prefix.update_positions.size].view(np.uint64),
    )
    np.testing.assert_array_equal(
        prefix.update_rows_used,
        extended.update_rows_used[: prefix.update_positions.size],
    )
    assert prefix.update_rows_used.tolist() == [2, 5, 8]


@pytest.mark.parametrize("horizon", [1, 2])
def test_dynamic_weights_exclude_labels_resolving_at_or_after_first_bar(
    horizon: int,
) -> None:
    rows = np.arange(6, dtype=np.int64)
    days = np.array([0, 0, 0, 1, 1, 1])
    evaluate = np.array([4, 5], dtype=np.int64)
    labels = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
    predictions = np.array(
        [[0.0, 2.0], [0.0, 2.0], [0.0, 10.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0]]
    )
    base = np.ones(6)
    result = dynamic_weights(
        predictions,
        labels,
        base,
        rows,
        days,
        evaluate,
        task="regression",
        horizon=horizon,
        window_days=2,
        min_rows=1,
        shrink=0.0,
    )
    # Update uses the day's first bar (3), even though evaluation begins at row 4.
    # h=1 permits rows 0,1; h=2 permits row 0. Row 2 is unresolved in both cases.
    assert result.update_positions.tolist() == [4]
    assert result.update_rows_used.tolist() == [3 - horizon]
    assert result.row_weights == pytest.approx(np.tile([1.0, 0.0], (2, 1)))

    changed_labels = labels.copy()
    changed_labels[2] = -1000.0
    changed = dynamic_weights(
        predictions,
        changed_labels,
        base,
        rows,
        days,
        evaluate,
        task="regression",
        horizon=horizon,
        window_days=2,
        min_rows=1,
        shrink=0.0,
    )
    np.testing.assert_array_equal(
        result.row_weights.view(np.uint64), changed.row_weights.view(np.uint64)
    )

    # Including row 2 would change the weights: losses are (100, 8), base loss 83,
    # so model 0 has negative skill and model 1 has positive skill.
    leaked, _ = performance_weights(
        predictions[:3], labels[:3], base[:3], "regression"
    )
    assert leaked == pytest.approx([0.0, 1.0])
    assert not np.array_equal(result.row_weights[0], leaked)


def test_equal_weight_entropy_effective_number_and_zero_turnover() -> None:
    weights = np.full(5, 1 / 5)
    entropy, effective = weight_entropy(weights)
    assert entropy == pytest.approx(math.log(5))
    assert effective == pytest.approx(5.0)
    repeated = np.tile(weights, (4, 1))
    np.testing.assert_array_equal(turnover(repeated), np.zeros(3))
