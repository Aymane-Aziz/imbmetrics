import numpy as np
import pytest
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score as sk_ap
from sklearn.metrics import precision_recall_curve as sk_prc

from imbmetrics import average_precision, pr_auc_trapezoid


def make_scores(seed, n_decimals, pos_rate):
    rng = np.random.default_rng(seed)
    y_true = (rng.random(5_000) < pos_rate).astype(int)
    y_score = 0.6 * rng.random(5_000) + 0.4 * y_true
    if n_decimals is not None:
        y_score = np.round(y_score, n_decimals)
    return y_true, y_score


def test_average_precision_hand_example():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.35, 0.8]
    assert average_precision(y_true, y_score) == pytest.approx(5 / 6)


def test_trapezoid_hand_example():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.35, 0.8]
    assert pr_auc_trapezoid(y_true, y_score) == pytest.approx(19 / 24)


def test_the_two_definitions_differ():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.35, 0.8]
    assert average_precision(y_true, y_score) != pytest.approx(
        pr_auc_trapezoid(y_true, y_score)
    )


def test_perfect_ranking_gives_one():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.8, 0.9]
    assert average_precision(y_true, y_score) == pytest.approx(1.0)
    assert pr_auc_trapezoid(y_true, y_score) == pytest.approx(1.0)


def test_random_scores_give_average_precision_near_the_positive_rate():
    rng = np.random.default_rng(0)
    y_true = (rng.random(100_000) < 0.05).astype(int)
    y_score = rng.random(100_000)
    assert average_precision(y_true, y_score) == pytest.approx(y_true.mean(), abs=0.01)


@pytest.mark.parametrize("pos_rate", [0.5, 0.05, 0.005])
@pytest.mark.parametrize("n_decimals", [None, 1, 0])
@pytest.mark.parametrize("seed", range(5))
def test_average_precision_matches_sklearn(seed, n_decimals, pos_rate):
    y_true, y_score = make_scores(seed, n_decimals, pos_rate)
    assert average_precision(y_true, y_score) == pytest.approx(sk_ap(y_true, y_score))


@pytest.mark.parametrize("pos_rate", [0.5, 0.05, 0.005])
@pytest.mark.parametrize("n_decimals", [None, 1, 0])
@pytest.mark.parametrize("seed", range(5))
def test_trapezoid_matches_sklearn_auc(seed, n_decimals, pos_rate):
    y_true, y_score = make_scores(seed, n_decimals, pos_rate)
    sk_precision, sk_recall, _ = sk_prc(y_true, y_score)
    assert pr_auc_trapezoid(y_true, y_score) == pytest.approx(auc(sk_recall, sk_precision))