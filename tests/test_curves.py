import numpy as np
import pytest
from sklearn.metrics import precision_recall_curve as sk_prc

from imbmetrics import precision_recall_curve


def test_curve_hand_example():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.35, 0.8]
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    np.testing.assert_allclose(precision, [0.5, 2 / 3, 0.5, 1.0, 1.0])
    np.testing.assert_allclose(recall, [1.0, 1.0, 0.5, 0.5, 0.0])
    np.testing.assert_allclose(thresholds, [0.1, 0.35, 0.4, 0.8])


def test_tied_scores_collapse_to_one_threshold():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.4, 0.8]
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    np.testing.assert_allclose(thresholds, [0.1, 0.4, 0.8])
    np.testing.assert_allclose(precision, [0.5, 2 / 3, 1.0, 1.0])
    np.testing.assert_allclose(recall, [1.0, 1.0, 0.5, 0.0])


def test_order_of_tied_samples_does_not_matter():
    first = precision_recall_curve([0, 0, 1, 1], [0.1, 0.4, 0.4, 0.8])
    second = precision_recall_curve([0, 1, 0, 1], [0.1, 0.4, 0.4, 0.8])
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("pos_rate", [0.5, 0.05, 0.005])
@pytest.mark.parametrize("n_decimals", [None, 1, 0])
@pytest.mark.parametrize("seed", range(5))
def test_curve_matches_sklearn(seed, n_decimals, pos_rate):
    rng = np.random.default_rng(seed)
    y_true = (rng.random(5_000) < pos_rate).astype(int)
    y_score = 0.6 * rng.random(5_000) + 0.4 * y_true
    if n_decimals is not None:
        y_score = np.round(y_score, n_decimals)

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    sk_precision, sk_recall, sk_thresholds = sk_prc(y_true, y_score)

    np.testing.assert_allclose(precision, sk_precision)
    np.testing.assert_allclose(recall, sk_recall)
    np.testing.assert_allclose(thresholds, sk_thresholds)


def test_no_positives_warns_and_sets_recall_to_one():
    with pytest.warns(UserWarning, match="No positive class"):
        precision, recall, _ = precision_recall_curve([0, 0, 0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(precision, [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(recall, [1.0, 1.0, 1.0, 0.0])


@pytest.mark.parametrize(
    "y_true, y_score",
    [
        ([0, 1], [0.1]),
        ([], []),
        ([0, 1], [0.1, np.nan]),
        ([[0, 1]], [[0.1, 0.2]]),
    ],
)
def test_invalid_input_raises(y_true, y_score):
    with pytest.raises(ValueError):
        precision_recall_curve(y_true, y_score)