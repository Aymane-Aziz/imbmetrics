from typing import Literal

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
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        cm = confusion_matrix(y_true, y_pred)
        n_reported = cm.shape[0]
    else:
        present = np.unique(np.concatenate([y_true, y_pred]))
        extra = np.setdiff1d(present , labels)
        all_labels = np.concatenate([list(labels), extra.tolist()])
        cm = confusion_matrix(y_true, y_pred, labels=all_labels.tolist())
        n_reported = len(labels)
    tp = np.diag(cm)[:n_reported]
    support = cm.sum(axis=1)[:n_reported]
    predicted = cm.sum(axis=0)[:n_reported]
    precision = _safe_divide(tp, predicted, zero_division)
    recall = _safe_divide(tp, support, zero_division)
    f1 = _safe_divide(2 * tp, support + predicted, zero_division)
    return precision, recall, f1, support

def _nan_average(
    values: NDArray[np.float64],
    weights: NDArray[np.int64] | None = None,
) -> float:
    keep = ~np.isnan(values)
    if not keep.any():
        return float("nan")
    values = values[keep]
    if weights is None or weights[keep].sum() == 0:
        return float(values.mean())
    return float(np.average(values, weights=weights[keep]))


def precision_recall_f1_average(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: Literal["macro", "weighted"] = "macro",
    labels: list | None = None,
    zero_division: float = 0.0,
) -> tuple[float, float, float]:
    if average not in ("macro", "weighted"):
        raise ValueError(f"average must be 'macro' or 'weighted', got {average!r}")
    precision, recall, f1, support = precision_recall_f1_support(
        y_true, y_pred, labels=labels, zero_division=zero_division
    )
    weights = support if average == "weighted" else None
    return (
        _nan_average(precision, weights),
        _nan_average(recall, weights),
        _nan_average(f1, weights),
    )