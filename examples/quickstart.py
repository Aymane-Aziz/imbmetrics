import numpy as np

from imbmetrics import (
    average_precision,
    best_f1_threshold,
    confusion_matrix,
    precision_recall_f1_average,
    precision_recall_f1_support,
    threshold_for_recall,
)

# Simulated fraud data: 1% positives, scores that are informative but imperfect.
rng = np.random.default_rng(0)
y_true = (rng.random(10_000) < 0.01).astype(int)
y_score = 1 / (1 + np.exp(-(rng.normal(size=10_000) + 2.0 * y_true - 1.5)))

y_pred = (y_score >= 0.5).astype(int)
print("positives:", y_true.sum(), "of", y_true.size)
print("confusion matrix (rows = true, columns = predicted):")
print(confusion_matrix(y_true, y_pred))

precision, recall, f1, support = precision_recall_f1_support(y_true, y_pred)
print("per-class precision:", precision.round(3))
print("per-class recall:   ", recall.round(3))
print("per-class f1:       ", f1.round(3))
print("support:            ", support)
print("macro (P, R, F1):   ", tuple(round(v, 3) for v in precision_recall_f1_average(y_true, y_pred)))

print("average precision (PR-AUC):", round(average_precision(y_true, y_score), 3))
print("best F1 point:", best_f1_threshold(y_true, y_score))
print("90% recall point:", threshold_for_recall(y_true, y_score, min_recall=0.9))