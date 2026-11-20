import numpy as np
import pytest
from sklearn.metrics import confusion_matrix as sk_cm

from imbmetrics import confusion_matrix


@pytest.mark.parametrize("pos_rate", [0.5, 0.05, 0.005])
@pytest.mark.parametrize("seed", range(5))
def test_confusion_matrix_matches_sklearn(pos_rate, seed):
    rng = np.random.default_rng(seed)
    y_true = (rng.random(10_000) < pos_rate).astype(int)
    y_pred = (rng.random(10_000) < 0.1).astype(int)
    np.testing.assert_array_equal(confusion_matrix(y_true, y_pred), sk_cm(y_true, y_pred))
    
def test_hand_example():
    result = confusion_matrix([0, 0, 0, 0, 1, 1], [0, 0, 1, 0, 1, 0])
    np.testing.assert_array_equal(result, [[3, 1], [1, 1]])
    
def test_labels_not_covering_data_matches_sklearn():
    y_true = [0, 1, 2]
    y_pred = [0, 1, 2]
    np.testing.assert_array_equal(
        confusion_matrix(y_true, y_pred, labels=[0, 1]),
        sk_cm(y_true, y_pred, labels=[0, 1]),
    )
def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        confusion_matrix([0, 1, 1], [0, 1])
        
@pytest.mark.filterwarnings("ignore:A single label")
def test_single_class_matches_sklearn():
    y_true = [1, 1, 1]
    y_pred = [1, 1, 1]
    np.testing.assert_array_equal(
        confusion_matrix(y_true, y_pred),
        sk_cm(y_true, y_pred),
    )