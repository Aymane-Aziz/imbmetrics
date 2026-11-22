# imbmetrics

[![CI](https://github.com/Aymane-Aziz/imbmetrics/actions/workflows/ci.yml/badge.svg)](https://github.com/Aymane-Aziz/imbmetrics/actions/workflows/ci.yml)

Evaluation metrics for **imbalanced classification**, in pure NumPy: per-class precision, recall and F1, macro and weighted averages, a confusion matrix, the precision-recall curve, PR-AUC and threshold analysis.

Accuracy is misleading when one class is rare: a model that never flags fraud scores 99% accuracy on data with 1% fraud. This library focuses on the numbers that expose the rare class, and every function is tested against scikit-learn.

## Install

```bash
pip install git+https://github.com/Aymane-Aziz/imbmetrics.git@v0.1.0
```

Requires Python 3.10+ and NumPy. scikit-learn is only needed to run the test suite.

## Quickstart

```python
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
```

```text
positives: 89 of 10000
confusion matrix (rows = true, columns = predicted):
[[9267  644]
 [  28   61]]
per-class precision: [0.997 0.087]
per-class recall:    [0.935 0.685]
per-class f1:        [0.965 0.154]
support:             [9911   89]
macro (P, R, F1):    (0.542, 0.81, 0.559)
average precision (PR-AUC): 0.219
best F1 point: OperatingPoint(threshold=0.7418117538141814, precision=0.34782608695652173, recall=0.2696629213483146, f1=0.3037974683544304)
90% recall point: OperatingPoint(threshold=0.2801931379624299, precision=0.027, recall=0.9101123595505618, f1=0.052444156685011324)
```

How to read it:

- At the default 0.5 cutoff the model catches 61 of the 89 frauds (recall 0.685), but only 8.7% of its fraud flags are real (precision 0.087).
- Overall accuracy is 93.3%. A model that never flags anything scores 99.1% on this data, which is why accuracy is the wrong yardstick here.
- Pushing recall to 90% costs precision: only 2.7% of flags would be real fraud.

## API

| Function | Returns |
|---|---|
| `confusion_matrix(y_true, y_pred, labels=None)` | Count matrix, rows = true labels, columns = predictions |
| `precision_recall_f1_support(y_true, y_pred, labels=None, zero_division=0.0)` | Per-class precision, recall, F1 and support arrays |
| `precision_recall_f1_average(y_true, y_pred, average="macro" or "weighted", ...)` | One `(precision, recall, f1)` tuple |
| `precision_recall_curve(y_true, y_score)` | `(precision, recall, thresholds)` at every distinct score |
| `average_precision(y_true, y_score)` | Step-wise PR-AUC (matches scikit-learn's `average_precision_score`) |
| `pr_auc_trapezoid(y_true, y_score)` | PR-AUC by the trapezoid rule (matches `auc(recall, precision)`) |
| `threshold_table(y_true, y_score)` | Precision, recall and F1 at every threshold |
| `best_f1_threshold(y_true, y_score)` | The threshold with the highest F1 |
| `threshold_for_recall(y_true, y_score, min_recall)` | The highest-precision threshold with recall at or above the target |

## Design notes

- **Matches scikit-learn.** Results are compared against scikit-learn on random imbalanced data, including tied scores, label subsets and undefined cases.
- **`labels` selects, it doesn't filter.** In the per-class metrics, `labels` chooses which classes are reported and in what order, but every sample is still counted (same as scikit-learn).
- **`zero_division`.** Sets the value used when a metric is undefined, for example the precision of a class that is never predicted. Use `np.nan` to make it visible; averages skip `nan` values.
- **Tied scores** are handled by evaluating one threshold per distinct score, never per sample.
- **Two PR-AUC definitions.** `average_precision` is step-wise and `pr_auc_trapezoid` uses linear interpolation. They give different numbers, so pick one and say which.
- **No micro average.** For single-label multiclass problems micro-averaged precision, recall and F1 all equal accuracy, which hides the imbalance this library is for.
- **Ties between thresholds** go to the lowest threshold, which flags the most samples and has the higher recall.

## Development

```bash
git clone https://github.com/Aymane-Aziz/imbmetrics.git
cd imbmetrics
pip install -e ".[dev]"
pytest
ruff check .
mypy src
```

CI runs ruff, mypy and pytest on Python 3.10 to 3.14 for every push and pull request.

## License

MIT