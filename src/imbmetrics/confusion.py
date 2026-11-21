import numpy as np
from numpy.typing import ArrayLike, NDArray


def confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    labels: list | None = None,
) -> NDArray[np.int64]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape, got {y_true.shape} and {y_pred.shape}")
    classes = np.unique(np.concatenate([y_true, y_pred])) if labels is None else labels
    if labels is not None and not np.any(np.isin(labels, y_true)):
        raise ValueError("At least one label specified must be in y_true")
    mask = np.isin(y_true, classes) & np.isin(y_pred, classes)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    index = {label: i for i, label in enumerate(classes)}
    true_idx = np.array([index[v] for v in y_true], dtype=np.intp)
    pred_idx = np.array([index[v] for v in y_pred], dtype=np.intp)
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (true_idx, pred_idx), 1)
    return cm
