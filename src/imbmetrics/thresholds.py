from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from imbmetrics.curves import precision_recall_curve


@dataclass(frozen=True)
class ThresholdTable:
    """Precision, recall and F1 at every distinct threshold (one row each, thresholds increasing)."""

    thresholds: NDArray[np.float64]
    precision: NDArray[np.float64]
    recall: NDArray[np.float64]
    f1: NDArray[np.float64]


@dataclass(frozen=True)
class OperatingPoint:
    """One chosen threshold and the metrics you get by flagging every score >= threshold."""

    threshold: float
    precision: float
    recall: float
    f1: float


def threshold_table(
    y_true: ArrayLike,
    y_score: ArrayLike,
    pos_label: int | str | bool = 1,
) -> ThresholdTable:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score, pos_label)
    # The curve has one extra closing point (precision 1, recall 0) with no threshold.
    # Drop it so that every row of the table belongs to a real threshold.
    precision = precision[:-1]
    recall = recall[:-1]
    total = precision + recall
    f1 = np.zeros_like(total)
    np.divide(2 * precision * recall, total, out=f1, where=total > 0)
    return ThresholdTable(thresholds, precision, recall, f1)


def _point(table: ThresholdTable, index: int) -> OperatingPoint:
    return OperatingPoint(
        threshold=float(table.thresholds[index]),
        precision=float(table.precision[index]),
        recall=float(table.recall[index]),
        f1=float(table.f1[index]),
    )


def best_f1_threshold(
    y_true: ArrayLike,
    y_score: ArrayLike,
    pos_label: int | str | bool = 1,
) -> OperatingPoint:
    """Threshold with the highest F1. If several tie, the lowest threshold wins."""
    table = threshold_table(y_true, y_score, pos_label)
    return _point(table, int(np.argmax(table.f1)))


def threshold_for_recall(
    y_true: ArrayLike,
    y_score: ArrayLike,
    min_recall: float,
    pos_label: int | str | bool = 1,
) -> OperatingPoint:
    """Highest precision among thresholds with recall >= min_recall.

    If several tie on precision, the lowest threshold wins (it has the higher recall).
    """
    if not 0.0 <= min_recall <= 1.0:
        raise ValueError(f"min_recall must be between 0 and 1, got {min_recall}")
    table = threshold_table(y_true, y_score, pos_label)
    candidates = np.flatnonzero(table.recall >= min_recall)
    best = candidates[np.argmax(table.precision[candidates])]
    return _point(table, int(best))