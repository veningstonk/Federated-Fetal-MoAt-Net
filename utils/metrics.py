"""
utils/metrics.py
────────────────
Evaluation metrics for multi-class fetal plane classification:
  • accuracy
  • macro AUC (one-vs-rest)
  • macro F1
  • per-class precision / recall / F1
  • confusion matrix (raw + normalized)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


# ─────────────────────────── inference helper ─────────────────────────────────

@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model inference and collect ground-truth labels, predicted labels,
    and softmax probability scores.

    Returns
    -------
    (y_true, y_pred, y_proba) — each a 1-D or 2-D numpy array.
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = np.argmax(probs, axis=1)
        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    y_true  = np.concatenate(all_labels)
    y_pred  = np.concatenate(all_preds)
    y_proba = np.concatenate(all_probs, axis=0)
    return y_true, y_pred, y_proba


# ─────────────────────────── metric computation ───────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute all evaluation metrics.

    Parameters
    ----------
    y_true     : ground-truth integer labels  (N,)
    y_pred     : predicted integer labels     (N,)
    y_proba    : predicted class probabilities (N, C)
    class_names: optional list of class name strings

    Returns
    -------
    Dict with keys: accuracy, auc, f1, precision, recall,
                    confusion_matrix, per_class, report
    """
    num_classes = y_proba.shape[1]

    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # Macro one-vs-rest AUC
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # Per-class metrics
    per_class_f1   = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_rec  = recall_score(y_true, y_pred, average=None, zero_division=0)

    names = class_names or [str(i) for i in range(num_classes)]
    per_class = {
        names[i]: {
            "precision": float(per_class_prec[i]),
            "recall":    float(per_class_rec[i]),
            "f1":        float(per_class_f1[i]),
        }
        for i in range(num_classes)
    }

    report = classification_report(
        y_true, y_pred,
        target_names=names,
        zero_division=0,
    )

    return {
        "accuracy":         float(acc),
        "auc":              float(auc),
        "f1":               float(f1),
        "precision":        float(prec),
        "recall":           float(rec),
        "confusion_matrix": cm,
        "per_class":        per_class,
        "report":           report,
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """Full evaluation pipeline: inference → metrics."""
    y_true, y_pred, y_proba = collect_predictions(model, loader, device)
    return compute_metrics(y_true, y_pred, y_proba, class_names)


def print_metrics(metrics: Dict, prefix: str = "") -> None:
    """Pretty-print a metrics dict."""
    tag = f"[{prefix}] " if prefix else ""
    print(f"\n{tag}Accuracy : {metrics['accuracy']*100:.2f}%")
    print(f"{tag}AUC      : {metrics['auc']*100:.2f}%")
    print(f"{tag}F1 (mac) : {metrics['f1']*100:.2f}%")
    print(f"{tag}Precision: {metrics['precision']*100:.2f}%")
    print(f"{tag}Recall   : {metrics['recall']*100:.2f}%")
    print(f"\n{metrics['report']}")
