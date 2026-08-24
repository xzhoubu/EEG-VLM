

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_pred):

    return {
        'balanced_acc': float(balanced_accuracy_score(y_true, y_pred)),
        'cohen_kappa': float(cohen_kappa_score(y_true, y_pred)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }


def safe_average_precision(y_true, y_score):

    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.size == 0:
        return float('nan')
    if np.all(y_true == 1):
        return 1.0
    if np.all(y_true == 0):
        return float('nan')
    return float(average_precision_score(y_true, y_score))


def safe_roc_auc(y_true, y_score):

    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float('nan')
    return float(roc_auc_score(y_true, y_score))


def binary_paper_metrics(y_true, y_score):

    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    y_pred = (y_score >= 0.5).astype(np.int64)
    return {
        'balanced_acc': float(balanced_accuracy_score(y_true, y_pred)),
        'auprc': safe_average_precision(y_true, y_score),
        'auroc': safe_roc_auc(y_true, y_score),
    }
