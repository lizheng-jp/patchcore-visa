"""Evaluation protocol: the threshold is chosen on val and applied once to test."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, roc_auc_score, roc_curve


def image_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores))


def pixel_auroc(masks: np.ndarray, maps: np.ndarray) -> float:
    """masks: (N, H, W) binary ground truth; maps: (N, H, W) anomaly scores."""
    if masks.shape != maps.shape:
        raise ValueError(f"mask shape {masks.shape} != map shape {maps.shape}")
    return float(roc_auc_score(masks.reshape(-1).astype(np.uint8), maps.reshape(-1)))


def select_threshold(labels: np.ndarray, scores: np.ndarray, method: str = "f1") -> float:
    """Pick a decision threshold from validation data only."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if method == "f1":
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
        return float(thresholds[int(np.argmax(f1[:-1]))])
    if method == "youden":
        fpr, tpr, thresholds = roc_curve(labels, scores)
        return float(thresholds[int(np.argmax(tpr - fpr))])
    raise ValueError(f"unknown threshold method {method!r}")


def thresholded_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Score >= threshold is predicted anomalous (label 1)."""
    labels = np.asarray(labels)
    pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "n": int(labels.size),
        "anomaly_recall": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "anomaly_precision": float(tp / (tp + fp)) if tp + fp else float("nan"),
        "normal_specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "accuracy": float((tp + tn) / labels.size),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def evaluate(
    val_labels: np.ndarray,
    val_scores: np.ndarray,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
    method: str = "f1",
) -> dict:
    """Full protocol for one category. Test data never influences the threshold."""
    threshold = select_threshold(val_labels, val_scores, method)
    official_labels = np.concatenate([val_labels, test_labels])
    official_scores = np.concatenate([val_scores, test_scores])
    return {
        "threshold_method": method,
        "image_auroc_official_test": image_auroc(official_labels, official_scores),
        "image_auroc_test": image_auroc(test_labels, test_scores),
        "val": thresholded_metrics(val_labels, val_scores, threshold),
        "test": thresholded_metrics(test_labels, test_scores, threshold),
    }
