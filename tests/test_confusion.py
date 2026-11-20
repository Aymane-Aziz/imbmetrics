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