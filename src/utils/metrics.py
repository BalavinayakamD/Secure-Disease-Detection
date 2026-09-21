"""Shared evaluation metrics for binary classification."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
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
        dict: accuracy, balanced_accuracy, majority_baseline, precision, recall,
        f1, roc_auc.

    `majority_baseline` is the accuracy of always predicting the commonest class in
    y_true. On a skewed local test set it can exceed a real model's accuracy, so any
    reported accuracy must be read against it; `balanced_accuracy` is the metric that
    is not fooled by skew (a constant predictor always scores 0.5).
    """
    y_pred = (y_prob >= threshold).astype(int)
    pos_rate = float(np.mean(y_true))
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'majority_baseline': max(pos_rate, 1.0 - pos_rate),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, y_prob)),
    }
