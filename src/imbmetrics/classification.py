import numpy as np
from numpy.typing import ArrayLike, NDArray

from imbmetrics.confusion import confusion_matrix


def _safe_divide(
    numerator: NDArray[np.int64],
    denominator: NDArray[np.int64],
    zero_division: float,
) -> NDArray[np.float64]:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.true_divide(numerator, denominator)
        result[denominator == 0] = zero_division
    return result 

def precision_recall_f1_support(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    labels: list | None = None,
    zero_division: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm)
    support = cm.sum(axis=1)    
    predicted = cm.sum(axis=0)
    precision = _safe_divide(tp, predicted, zero_division)
    recall = _safe_divide(tp, support, zero_division)
    f1 = _safe_divide(2 * tp, support + predicted, zero_division)
    return precision, recall, f1, support