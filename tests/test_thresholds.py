import numpy as np
import pytest
from sklearn.metrics import precision_recall_curve as sk_prc

from imbmetrics import best_f1_threshold, threshold_for_recall, threshold_table

HAND_TRUE = [0, 0, 1, 1]
HAND_SCORE = [0.1, 0.4, 0.35, 0.8]


def test_table_hand_example():
    table = threshold_table(HAND_TRUE, HAND_SCORE)
    np.testing.assert_allclose(table.thresholds, [0.1, 0.35, 0.4, 0.8])
    np.testing.assert_allclose(table.precision, [0.5, 2 / 3, 0.5, 1.0])
    np.testing.assert_allclose(table.recall, [1.0, 1.0, 0.5, 0.5])
    np.testing.assert_allclose(table.f1, [2 / 3, 0.8, 0.5, 2 / 3])


def test_best_f1_hand_example():
    best = best_f1_threshold(HAND_TRUE, HAND_SCORE)
    assert best.threshold == pytest.approx(0.35)
    assert best.precision == pytest.approx(2 / 3)
    assert best.recall == pytest.approx(1.0)
    assert best.f1 == pytest.approx(0.8)


@pytest.mark.parametrize(
    "min_recall, expected_threshold",
    [(1.0, 0.35), (0.75, 0.35), (0.5, 0.8), (0.0, 0.8)],
)
def test_threshold_for_recall_hand_example(min_recall, expected_threshold):
    point = threshold_for_recall(HAND_TRUE, HAND_SCORE, min_recall)
    assert point.threshold == pytest.approx(expected_threshold)
    assert point.recall >= min_recall


def test_precision_ties_go_to_the_lowest_threshold():
    # Thresholds 0.8 and 0.9 both have precision 1.0, but 0.8 also has recall 1.0.
    point = threshold_for_recall([1, 1, 0], [0.9, 0.8, 0.1], min_recall=0.5)
    assert point.threshold == pytest.approx(0.8)
    assert point.precision == pytest.approx(1.0)
    assert point.recall == pytest.approx(1.0)


def test_table_has_no_row_without_a_threshold():
    # Here every real precision is below 1, so an argmax over the curve's closing point
    # (precision 1, recall 0) would have no threshold to point at.
    table = threshold_table([0, 1], [0.9, 0.1])
    assert len(table.thresholds) == len(table.precision) == len(table.recall) == len(table.f1) == 2
    assert best_f1_threshold([0, 1], [0.9, 0.1]).threshold == pytest.approx(0.1)


@pytest.mark.parametrize("min_recall", [-0.1, 1.1])
def test_threshold_for_recall_rejects_bad_targets(min_recall):
    with pytest.raises(ValueError):
        threshold_for_recall(HAND_TRUE, HAND_SCORE, min_recall)


def brute_force_rows(y_true, y_score):
    rows = []
    for t in np.unique(y_score):
        flagged = y_score >= t
        tp = np.sum(flagged & (y_true == 1))
        precision = tp / flagged.sum()
        recall = tp / y_true.sum()
        f1 = 0.0 if tp == 0 else 2 * precision * recall / (precision + recall)
        rows.append((t, precision, recall, f1))
    return np.array(rows)


def make_scores(seed, n_decimals, pos_rate):
    rng = np.random.default_rng(seed)
    y_true = (rng.random(2_000) < pos_rate).astype(int)
    y_score = 0.6 * rng.random(2_000) + 0.4 * y_true
    if n_decimals is not None:
        y_score = np.round(y_score, n_decimals)
    return y_true, y_score


@pytest.mark.parametrize("pos_rate", [0.5, 0.05, 0.01])
@pytest.mark.parametrize("n_decimals", [None, 1])
@pytest.mark.parametrize("seed", range(3))
def test_table_matches_brute_force_and_sklearn(seed, n_decimals, pos_rate):
    y_true, y_score = make_scores(seed, n_decimals, pos_rate)
    table = threshold_table(y_true, y_score)
    rows = brute_force_rows(y_true, y_score)

    np.testing.assert_allclose(table.thresholds, rows[:, 0])
    np.testing.assert_allclose(table.precision, rows[:, 1])
    np.testing.assert_allclose(table.recall, rows[:, 2])
    np.testing.assert_allclose(table.f1, rows[:, 3])

    sk_precision, sk_recall, sk_thresholds = sk_prc(y_true, y_score)
    np.testing.assert_allclose(table.precision, sk_precision[:-1])
    np.testing.assert_allclose(table.recall, sk_recall[:-1])
    np.testing.assert_allclose(table.thresholds, sk_thresholds)


@pytest.mark.parametrize("pos_rate", [0.5, 0.05, 0.01])
@pytest.mark.parametrize("n_decimals", [None, 1])
@pytest.mark.parametrize("seed", range(3))
def test_chosen_points_are_optimal(seed, n_decimals, pos_rate):
    y_true, y_score = make_scores(seed, n_decimals, pos_rate)
    rows = brute_force_rows(y_true, y_score)

    best = best_f1_threshold(y_true, y_score)
    assert best.f1 == pytest.approx(rows[:, 3].max())

    for min_recall in (0.5, 0.8, 0.9, 1.0):
        point = threshold_for_recall(y_true, y_score, min_recall)
        eligible = rows[rows[:, 2] >= min_recall]
        assert point.recall >= min_recall
        assert point.precision == pytest.approx(eligible[:, 1].max())