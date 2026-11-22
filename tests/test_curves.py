import numpy as np

from imbmetrics import precision_recall_curve


def test_curve_hand_example():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.35, 0.8]
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    np.testing.assert_allclose(precision, [0.5, 2 / 3, 0.5, 1.0, 1.0])
    np.testing.assert_allclose(recall, [1.0, 1.0, 0.5, 0.5, 0.0])
    np.testing.assert_allclose(thresholds, [0.1, 0.35, 0.4, 0.8])