"""Shared evaluation metrics for binary classification."""

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(y_true, y_prob, threshold=0.5):
    """
    Args:
        y_true: 1-D array of 0/1 labels.
        y_prob: 1-D array of predicted probabilities (post-sigmoid).
        threshold: Decision threshold for the hard-label metrics.

    Returns:
        dict: accuracy, precision, recall, f1, roc_auc.
    """
    y_pred = (y_prob >= threshold).astype(int)
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, y_prob)),
    }
