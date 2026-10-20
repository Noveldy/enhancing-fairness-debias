"""Phase 4.1 metrics: ECE, equal-opportunity gap, per-group rates."""
import numpy as np
import pandas as pd
import pytest

from experiments.metrics import expected_calibration_error, extended_metrics, group_rate_rows


def test_ece_zero_when_calibrated_and_known_value_otherwise():
    y = np.array([0, 1] * 50)
    assert expected_calibration_error(y, np.full(100, 0.5)) == pytest.approx(0.0)
    # every score 0.9 but only half positive: |0.5 - 0.9| = 0.4
    assert expected_calibration_error(y, np.full(100, 0.9)) == pytest.approx(0.4)
    # score exactly 1.0 lands in the last bin instead of overflowing
    assert expected_calibration_error(np.array([1, 1]), np.array([1.0, 1.0])) == pytest.approx(0.0)


def test_gaps_and_group_rates_on_hand_computed_example():
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 1, 0, 0, 0])
    score = y_pred * 0.8 + 0.1
    g = pd.DataFrame({"sex": ["a"] * 4 + ["b"] * 4})
    m = extended_metrics(y_true, y_pred, score, g)
    # a: selection 3/4, TPR 1, FPR 1/2; b: selection 1/4, TPR 1/2, FPR 0
    assert m["sex__dp_diff"] == pytest.approx(0.5)
    assert m["sex__eopp_diff"] == m["sex__tpr_diff"] == pytest.approx(0.5)
    assert m["sex__fpr_diff"] == pytest.approx(0.5)
    assert m["sex__eo_diff"] == pytest.approx(0.5)
    rows = {r["group"]: r for r in group_rate_rows(y_true, y_pred, g)}
    assert rows["a"]["selection_rate"] == 0.75 and rows["b"]["tpr"] == 0.5 and rows["b"]["fpr"] == 0.0


def test_constant_predictor_has_zero_gaps():
    y = np.array([0, 1, 0, 1, 1, 0])
    g = pd.DataFrame({"sex": ["a", "a", "a", "b", "b", "b"]})
    m = extended_metrics(y, np.zeros(6, int), np.full(6, 0.5), g)
    assert m["sex__dp_diff"] == m["sex__eo_diff"] == 0.0 and m["balanced_accuracy"] == 0.5
