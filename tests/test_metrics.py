import numpy as np

from fraud.metrics import classification_report, pick_threshold


def test_pick_threshold_meets_precision_target():
    y = np.array([0, 0, 0, 0, 1, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9])
    t = pick_threshold(y, s, min_precision=0.9)
    report = classification_report(y, s, t)
    assert report["precision"] >= 0.9
    assert t == 0.8


def test_report_counts_and_value_caught():
    y = np.array([0, 1, 1, 0])
    s = np.array([0.1, 0.9, 0.2, 0.8])
    r = classification_report(y, s, 0.5, amounts=[10, 100, 300, 5])
    assert (r["true_positives"], r["false_positives"], r["false_negatives"]) == (1, 1, 1)
    assert r["fraud_value_caught_pct"] == 25.0
