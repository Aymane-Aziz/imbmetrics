import numpy as np

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