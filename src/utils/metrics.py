"""Shared evaluation metrics for binary classification.

IMPORTANT NOTES:
1. ROC-AUC requires probability scores (0-1 range), not binary predictions
2. Balanced accuracy is the primary metric for imbalanced datasets
3. majority_baseline shows the accuracy of always predicting the most common class
4. All metrics handle edge cases (zero division, single class, etc.)
"""

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
    Comprehensive binary classification metrics.
    
    Args:
        y_true: 1-D array of 0/1 ground truth labels
        y_prob: 1-D array of predicted probabilities in [0, 1] (post-sigmoid scores)
        threshold: Decision threshold for hard classification (default: 0.5)

    Returns:
        dict with keys:
        - accuracy: Overall correctness
        - balanced_accuracy: Average of recall per class (handles imbalance)
        - majority_baseline: Accuracy of always predicting majority class
        - precision: TP / (TP + FP)
        - recall: TP / (TP + FN)  
        - f1: Harmonic mean of precision and recall
        - roc_auc: Area under ROC curve (discrimination ability)
        - confusion_matrix: [[TN, FP], [FN, TP]]

    INTERPRETATION:
    - majority_baseline: On skewed data, a trivial classifier can have high accuracy.
      Always compare accuracy against this baseline.
    - balanced_accuracy: Not fooled by class imbalance. A random classifier scores 0.5.
    - roc_auc: Computed from probabilities, measures ranking quality independent of threshold.
    
    REQUIREMENTS:
    - y_prob must contain probability scores (0-1), not logits or binary predictions
    - y_true must contain binary labels (0 or 1)
    - Both arrays must have the same length
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    
    # Validate inputs
    assert len(y_true) == len(y_prob), "y_true and y_prob must have the same length"
    assert set(y_true).issubset({0, 1}), "y_true must contain only 0 and 1"
    assert np.all((y_prob >= 0) & (y_prob <= 1)), "y_prob must be in [0, 1]"
    
    # Hard predictions from threshold
    y_pred = (y_prob >= threshold).astype(int)
    
    # Majority baseline
    pos_rate = float(np.mean(y_true))
    majority_baseline = max(pos_rate, 1.0 - pos_rate)
    
    # Confusion matrix
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'majority_baseline': majority_baseline,
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, y_prob)),
        'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]],
        'threshold': threshold
    }
