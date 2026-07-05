"""Shared classification metrics for training and evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Compute standard binary classification metrics for the positive class."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def extended_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Compute standard metrics plus PR-AUC, balanced accuracy, MCC, and Brier score."""
    metrics = classification_metrics(y_true, y_pred, y_prob)
    metrics.update(
        {
            "pr_auc": float(average_precision_score(y_true, y_prob)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
            "brier_score": float(brier_score_loss(y_true, y_prob)),
        }
    )
    return metrics


def predictions_at_threshold(y_prob: np.ndarray, threshold: float) -> np.ndarray:
    """Convert predicted probabilities to binary labels at *threshold*."""
    return (y_prob >= threshold).astype(int)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method: str = "f1",
) -> tuple[float, dict[str, float]]:
    """Find the decision threshold that maximizes F1 or Youden's J on validation data."""
    thresholds = np.unique(np.concatenate([[0.05, 0.95], np.round(y_prob, 4)]))
    thresholds = np.clip(thresholds, 0.01, 0.99)

    best_threshold = 0.5
    best_score = -1.0
    best_metrics: dict[str, float] = {}

    for threshold in thresholds:
        y_pred = predictions_at_threshold(y_prob, float(threshold))
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        if method == "youden":
            tn = int(((y_pred == 0) & (y_true == 0)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            specificity = tn / max(tn + fp, 1)
            sensitivity = tp / max(tp + fn, 1)
            score = sensitivity + specificity - 1.0
        else:
            score = float(f1)

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = {
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "score": float(score),
            }

    return best_threshold, best_metrics


def threshold_analysis_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: list[float] | None = None,
) -> list[dict[str, float]]:
    """Return precision, recall, and F1 at a range of decision thresholds."""
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.1, 0.95, 0.05)]

    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        y_pred = predictions_at_threshold(y_prob, threshold)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            }
        )
    return rows


def classify_overfitting(roc_auc_gap: float) -> str:
    """Classify train–test ROC-AUC gap into an overfitting severity label."""
    if roc_auc_gap < 0.02:
        return "Excellent"
    if roc_auc_gap < 0.05:
        return "Minor Overfit"
    if roc_auc_gap < 0.10:
        return "Moderate Overfit"
    return "Severe Overfit"
