import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from classes.metrics import NPV, F1Score, FBetaScore  # noqa: E402


def _one_hot(labels):
    return np.eye(2, dtype="float32")[labels]


Y_TRUE = [1, 1, 1, 0, 0, 0, 0, 1]
Y_PRED = [1, 1, 0, 0, 0, 1, 0, 0]
# TP=2, FP=1, FN=2, TN=3


def test_f1_matches_hand_computation():
    m = F1Score()
    m.update_state(_one_hot(Y_TRUE), _one_hot(Y_PRED))
    p, r = 2 / 3, 2 / 4
    assert float(m.result()) == pytest.approx(2 * p * r / (p + r), abs=1e-4)


def test_fbeta_with_beta_two_weights_recall():
    m = FBetaScore(beta=2.0)
    m.update_state(_one_hot(Y_TRUE), _one_hot(Y_PRED))
    p, r = 2 / 3, 2 / 4
    assert float(m.result()) == pytest.approx(5 * p * r / (4 * p + r), abs=1e-4)


def test_npv_matches_hand_computation():
    m = NPV()
    m.update_state(_one_hot(Y_TRUE), _one_hot(Y_PRED))
    assert float(m.result()) == pytest.approx(3 / 5, abs=1e-4)


def test_reset_state_clears_counters():
    m = F1Score()
    m.update_state(_one_hot(Y_TRUE), _one_hot(Y_PRED))
    m.reset_state()
    assert float(m.result()) == pytest.approx(0.0, abs=1e-6)
