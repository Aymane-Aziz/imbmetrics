import numpy as np
import pytest
from sklearn.metrics import precision_recall_fscore_support as sk_prfs

from imbmetrics import precision_recall_f1_average


def test_macro_and_weighted_hand_example():
    y_true = [0, 0, 0, 1, 1]
    y_pred = [0, 1, 1, 1, 1]
    macro = precision_recall_f1_average(y_true, y_pred, average="macro")
    weighted = precision_recall_f1_average(y_true, y_pred, average="weighted")
    np.testing.assert_allclose(macro, [0.75, 2 / 3, 7 / 12])
    np.testing.assert_allclose(weighted, [0.8, 0.6, 17 / 30])
    
def test_unsupported_average_raises():
    with pytest.raises(ValueError):
        precision_recall_f1_average([0, 1], [0, 1], average="micro")  # type: ignore[arg-type]


@pytest.mark.parametrize("zero_division", [0, 1, np.nan])
@pytest.mark.parametrize("average", ["macro", "weighted"])
@pytest.mark.parametrize("labels", [None, [3, 1], [0, 9], [9]])
@pytest.mark.parametrize("pred_p", [[0.8, 0.1, 0.09, 0.01], [0.8, 0.1, 0.1, 0.0]])
def test_averages_match_sklearn(pred_p, labels, average, zero_division):
    rng = np.random.default_rng(0)
    y_true = rng.choice(4, size=10_000, p=[0.9, 0.07, 0.025, 0.005])
    y_pred = rng.choice(4, size=10_000, p=pred_p)

    ours = precision_recall_f1_average(
        y_true, y_pred, average=average, labels=labels, zero_division=zero_division
    )
    theirs = sk_prfs(
        y_true, y_pred, average=average, labels=labels, zero_division=zero_division
    )[:3]

    np.testing.assert_allclose(ours, np.asarray(theirs, dtype=float))