"""Metrics for a rare-event problem. Accuracy is useless at 0.17% fraud, so we report
precision-recall AUC, recall, precision and the share of fraud value caught."""
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_auc_score)


def pick_threshold(y_true: np.ndarray, scores: np.ndarray, min_precision: float) -> float:
    """Highest-recall threshold whose precision is at least min_precision.
    Falls back to the best-F1 threshold if no threshold reaches the target."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    precision, recall = precision[:-1], recall[:-1]  # align with thresholds
    ok = np.where(precision >= min_precision)[0]
    if len(ok):
        best = ok[np.argmax(recall[ok])]
    else:
        f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
        best = int(np.argmax(f1))
    return float(thresholds[best])


def classification_report(y_true, scores, threshold, amounts=None) -> dict:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    report = {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "threshold": float(threshold),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }
    if amounts is not None:
        amounts = np.asarray(amounts)
        fraud_value = amounts[np.asarray(y_true) == 1].sum()
        caught_value = amounts[(np.asarray(y_true) == 1) & (y_pred == 1)].sum()
        report["fraud_value_caught_pct"] = float(100 * caught_value / fraud_value) if fraud_value else 0.0
    return report
