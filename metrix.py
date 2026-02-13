from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


@dataclass(frozen=True)
class MetricResult:
    accuracy: float
    auc: float
    precision: float
    recall: float
    f1: float
    mcc: float


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Computes AUC for binary or multiclass classification.
    - Binary: y_score is probability for positive class.
    - Multiclass: y_score is (n_samples, n_classes) probability matrix.
    """
    y_true = np.asarray(y_true)

    # Determine number of classes from y_true
    classes = np.unique(y_true)
    n_classes = len(classes)

    try:
        if n_classes == 2:
            return _safe_float(roc_auc_score(y_true, y_score))
        else:
            return _safe_float(roc_auc_score(y_true, y_score, multi_class="ovr", average="weighted"))
    except Exception:
        # Some models might not provide good scores in edge cases
        return float("nan")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> MetricResult:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = np.unique(y_true)
    n_classes = len(classes)

    avg = "binary" if n_classes == 2 else "weighted"

    acc = _safe_float(accuracy_score(y_true, y_pred))
    prec = _safe_float(precision_score(y_true, y_pred, average=avg, zero_division=0))
    rec = _safe_float(recall_score(y_true, y_pred, average=avg, zero_division=0))
    f1 = _safe_float(f1_score(y_true, y_pred, average=avg, zero_division=0))
    mcc = _safe_float(matthews_corrcoef(y_true, y_pred))

    auc = float("nan")
    if y_proba is not None:
        # For binary: use positive class prob; for multiclass: pass full matrix
        if n_classes == 2 and y_proba.ndim == 2 and y_proba.shape[1] >= 2:
            auc = compute_auc(y_true, y_proba[:, 1])
        else:
            auc = compute_auc(y_true, y_proba)

    return MetricResult(
        accuracy=acc,
        auc=auc,
        precision=prec,
        recall=rec,
        f1=f1,
        mcc=mcc,
    )


def get_confusion_and_report(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, str]:
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)
    return cm, report


def metric_result_to_dict(m: MetricResult) -> Dict[str, float]:
    return {
        "Accuracy": m.accuracy,
        "AUC": m.auc,
        "Precision": m.precision,
        "Recall": m.recall,
        "F1": m.f1,
        "MCC": m.mcc,
    }