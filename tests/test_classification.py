import numpy as np
import pytest
from sklearn.metrics import precision_recall_fscore_support as sk_prfs

from imbmetrics import precision_recall_f1_support


def test_three_class_hand_example():
    y_true = [0, 1, 2, 2]
    y_pred = [0, 2, 2, 1]
    precision, recall, f1, support = precision_recall_f1_support(y_true, y_pred)
    np.testing.assert_allclose(precision, [1.0, 0.0, 0.5])
    np.testing.assert_allclose(recall, [1.0, 0.0, 0.5])
    np.testing.assert_allclose(f1, [1.0, 0.0, 0.5])
    np.testing.assert_array_equal(support, [1, 1, 2])
    
def test_asymmetric_hand_example():
    y_true = [0, 0, 0, 1, 1]
    y_pred = [0, 1, 1, 1, 1]
    precision, recall, f1, support = precision_recall_f1_support(y_true, y_pred)
    np.testing.assert_allclose(precision, [1.0, 0.5])
    np.testing.assert_allclose(recall, [1 / 3, 1.0])
    np.testing.assert_allclose(f1, [0.5, 2 / 3])
    np.testing.assert_array_equal(support, [3, 2])
    
@pytest.mark.parametrize("pos_rate", [0.5, 0.05, 0.005])
@pytest.mark.parametrize("seed", range(5))
def test_binary_matches_sklearn(pos_rate, seed):
    rng = np.random.default_rng(seed)
    y_true = (rng.random(10_000) < pos_rate).astype(int)
    y_pred = (rng.random(10_000) < 0.1).astype(int)

    precision, recall, f1, support = precision_recall_f1_support(y_true, y_pred)
    sk_precision, sk_recall, sk_f1, sk_support = sk_prfs(y_true, y_pred, zero_division=0)

    np.testing.assert_allclose(precision, sk_precision)
    np.testing.assert_allclose(recall, sk_recall)
    np.testing.assert_allclose(f1, sk_f1)
    np.testing.assert_array_equal(support, sk_support)
    
@pytest.mark.parametrize("zero_division", [0, 1, np.nan])
@pytest.mark.parametrize(
    "y_true, y_pred",
    [
        ([0, 1], [0, 0]),
        ([0, 0, 1, 1, 1], [0, 0, 0, 0, 0]),
        ([0, 1, 2, 2], [0, 0, 0, 0]),
    ],
)
def test_class_never_predicted_matches_sklearn(y_true, y_pred, zero_division):
    ours = precision_recall_f1_support(y_true, y_pred, zero_division=zero_division)
    theirs = sk_prfs(y_true, y_pred, zero_division=zero_division)
    for a, b in zip(ours, theirs):
        np.testing.assert_allclose(a, np.asarray(b))
        
@pytest.mark.parametrize("seed", range(5))
def test_multiclass_matches_sklearn(seed):
    rng = np.random.default_rng(seed)
    y_true = rng.choice(4, size=10_000, p=[0.9, 0.07, 0.025, 0.005])
    y_pred = rng.choice(4, size=10_000, p=[0.8, 0.1, 0.09, 0.01])

    precision, recall, f1, support = precision_recall_f1_support(y_true, y_pred)
    sk_precision, sk_recall, sk_f1, sk_support = sk_prfs(y_true, y_pred, zero_division=0)

    np.testing.assert_allclose(precision, sk_precision)
    np.testing.assert_allclose(recall, sk_recall)
    np.testing.assert_allclose(f1, sk_f1)
    np.testing.assert_array_equal(support, sk_support)
    
def test_labels_subset_counts_all_samples():
    y_true = [0, 0, 1]
    y_pred = [0, 2, 1]
    precision, recall, f1, support = precision_recall_f1_support(
        y_true, y_pred, labels=[0, 1]
    )
    np.testing.assert_allclose(precision, [1.0, 1.0])
    np.testing.assert_allclose(recall, [0.5, 1.0])
    np.testing.assert_allclose(f1, [2 / 3, 1.0])
    np.testing.assert_array_equal(support, [2, 1])