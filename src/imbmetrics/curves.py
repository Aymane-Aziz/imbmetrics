import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray


def precision_recall_curve(
    y_true: ArrayLike,
    y_score: ArrayLike,
    pos_label: int | str | bool = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    is_positive: NDArray[np.bool_] = np.asarray(y_true) == pos_label
    scores: NDArray[np.float64] = np.asarray(y_score, dtype=np.float64)
    if is_positive.ndim != 1 or is_positive.shape != scores.shape:
        raise ValueError(
            "y_true and y_score must be 1-D and the same length, "
            f"got {is_positive.shape} and {scores.shape}"
        )
    if scores.size == 0:
        raise ValueError("y_true and y_score must not be empty")
    if not np.all(np.isfinite(scores)):
        raise ValueError("y_score must contain only finite values")

    # Sort from the highest score to the lowest (stable, so ties keep their order).
    order = np.argsort(-scores, kind="mergesort")
    is_positive = is_positive[order]
    scores = scores[order]

    # One threshold per distinct score: the last position of each run of equal scores.
    last_of_each_score = np.concatenate([np.flatnonzero(np.diff(scores)), [scores.size - 1]])
    true_positives = np.cumsum(is_positive)[last_of_each_score]
    predicted_positive = last_of_each_score + 1

    precision = true_positives / predicted_positive
    if true_positives[-1] == 0:
        warnings.warn("No positive class found in y_true, recall is set to one for all thresholds.")
        recall = np.ones_like(precision)
    else:
        recall = true_positives / true_positives[-1]

    # Reverse so thresholds increase (recall decreases) and add the final point
    # (precision 1, recall 0) that sklearn also returns.
    return (
        np.concatenate([precision[::-1], [1.0]]),
        np.concatenate([recall[::-1], [0.0]]),
        scores[last_of_each_score][::-1],
    )
    
def average_precision(
    y_true: ArrayLike,
    y_score: ArrayLike,
    pos_label: int | str | bool = 1,
) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label)
    return float(max(0.0, -np.sum(np.diff(recall) * precision[:-1])))


def pr_auc_trapezoid(
    y_true: ArrayLike,
    y_score: ArrayLike,
    pos_label: int | str | bool = 1,
) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_score, pos_label)
    return float(-np.sum(np.diff(recall) * (precision[1:] + precision[:-1]) / 2))