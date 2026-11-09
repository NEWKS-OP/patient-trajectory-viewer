"""Evaluation metrics, written out explicitly."""

from __future__ import annotations

import numpy as np


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    return {
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
    }


def classification_report(y_true, y_pred) -> dict[str, float]:
    c = confusion(y_true, y_pred)
    precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (c["tp"] + c["tn"]) / len(y_true),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        **c,
    }


def roc_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """False/true positive rates when the threshold sweeps over every distinct score."""
    order = np.argsort(-scores, kind="mergesort")
    y, s = y_true[order], scores[order]
    distinct = np.r_[np.where(np.diff(s))[0], len(s) - 1]  # last index of each tied block
    tps = np.cumsum(y)[distinct]
    fps = (distinct + 1) - tps
    tpr = np.r_[0, tps / max(1, y.sum())]
    fpr = np.r_[0, fps / max(1, len(y) - y.sum())]
    return fpr, tpr


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """Area under a curve with the trapezoidal rule."""
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2))


def roc_auc(y_true, scores) -> float:
    fpr, tpr = roc_curve(y_true, scores)
    return auc(fpr, tpr)


def best_f1_threshold(y_true, scores) -> float:
    """Threshold maximising F1 (chosen on validation data, never on the test set)."""
    best_t, best = 0.5, -1.0
    for t in np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 97))):
        f = classification_report(y_true, (scores >= t).astype(int))["f1"]
        if f > best:
            best, best_t = f, float(t)
    return best_t
