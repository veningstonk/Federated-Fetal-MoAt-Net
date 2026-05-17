"""
utils/visualization.py
──────────────────────
Plotting utilities:
  • Training / validation curves (loss, accuracy, AUC)
  • ROC curves (per-class + micro/macro average)
  • Precision-Recall curves
  • Confusion matrix heatmap
  • Grad-CAM heatmaps (target layer: moat_attention.conv_out)
  • Federated round-accuracy curves across strategies
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


PALETTE = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0",
           "#FF9800", "#009688", "#E91E63", "#607D8B"]


# ─────────────────────────── training curves ──────────────────────────────────

def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: str,
    title: str = "Training Curves",
) -> None:
    """
    Plot loss, accuracy, and AUC curves from a training history dict.

    history keys: 'train_loss', 'val_loss', 'train_acc', 'val_acc',
                  'train_auc', 'val_auc'  (all optional except loss/acc)
    """
    metrics_to_plot = []
    if "train_loss" in history:
        metrics_to_plot.append(("loss",     "Loss",     "train_loss",  "val_loss"))
    if "train_acc" in history:
        metrics_to_plot.append(("accuracy", "Accuracy", "train_acc",   "val_acc"))
    if "train_auc" in history:
        metrics_to_plot.append(("auc",      "AUC",      "train_auc",   "val_auc"))

    n = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (_, ylabel, train_key, val_key) in zip(axes, metrics_to_plot):
        epochs = range(1, len(history[train_key]) + 1)
        ax.plot(epochs, history[train_key], label=f"Train {ylabel}", color=PALETTE[0])
        if val_key in history:
            ax.plot(epochs, history[val_key], label=f"Val {ylabel}",
                    color=PALETTE[1], linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────── ROC curves ───────────────────────────────────────

def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "ROC Curves",
) -> None:
    num_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", color=PALETTE[i % len(PALETTE)])

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────── PR curves ────────────────────────────────────────

def plot_pr_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Precision-Recall Curves",
) -> None:
    num_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, name in enumerate(class_names):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        pr_auc = auc(rec, prec)
        ax.plot(rec, prec, label=f"{name} (AUC={pr_auc:.3f})", color=PALETTE[i % len(PALETTE)])

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────── confusion matrix ─────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
    normalize: bool = False,
) -> None:
    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt = ".2f"
    else:
        fmt = "d"

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(title, fontweight="bold", fontsize=12)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────── federated round curves ───────────────────────────────

def plot_federated_comparison(
    round_histories: Dict[str, List[float]],
    save_path: str,
    metric: str = "accuracy",
    title: str = "Federated Strategy Comparison",
) -> None:
    """
    Plot per-round metric for multiple federated strategies on one figure.

    Parameters
    ----------
    round_histories : Dict[strategy_name → list of per-round metric values]
    metric          : label for y-axis
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (strategy, values) in enumerate(round_histories.items()):
        rounds = range(1, len(values) + 1)
        ax.plot(rounds, values, label=strategy, color=PALETTE[i % len(PALETTE)], linewidth=1.8)

    ax.set_xlabel("Communication Round")
    ax.set_ylabel(metric.capitalize())
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────── Grad-CAM ─────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM for Fetal MoAt Net.

    Target layer: moat_attention.conv_out  (the final 1×1 fusion projection
    in the MoAt block — last spatially-resolved feature map before GAP).
    This layer is used per the paper's specification for reproducibility.
    """

    def __init__(self, model: nn.Module, target_layer_name: str = "moat_attention.conv_out"):
        self.model = model
        self.model.eval()
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._hook_handles = []
        self._register_hooks(target_layer_name)

    def _get_layer(self, name: str) -> nn.Module:
        parts = name.split(".")
        layer = self.model
        for p in parts:
            layer = getattr(layer, p)
        return layer

    def _register_hooks(self, name: str) -> None:
        layer = self._get_layer(name)

        def forward_hook(module, inp, out):
            self._activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self._hook_handles.append(layer.register_forward_hook(forward_hook))
        self._hook_handles.append(layer.register_full_backward_hook(backward_hook))

    def generate(self, image_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for the given image tensor (1 × C × H × W).

        Returns
        -------
        heatmap : np.ndarray (H × W), values in [0, 1].
        """
        self.model.zero_grad()
        logits = self.model(image_tensor)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[0, class_idx]
        score.backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)   # GAP of gradients
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()


def save_gradcam_figure(
    model: nn.Module,
    image_tensor: torch.Tensor,
    original_image: np.ndarray,
    save_path: str,
    class_names: List[str],
    true_label: int,
    pred_label: int,
) -> None:
    """
    Overlay Grad-CAM heatmap on the original image and save.
    """
    cam_gen = GradCAM(model)
    heatmap = cam_gen.generate(image_tensor.unsqueeze(0), class_idx=pred_label)
    cam_gen.remove_hooks()

    # Resize heatmap to image size
    from PIL import Image as PILImage
    heatmap_img = PILImage.fromarray((heatmap * 255).astype(np.uint8)).resize(
        (original_image.shape[1], original_image.shape[0]), PILImage.BILINEAR
    )
    heatmap_np = np.array(heatmap_img) / 255.0

    cmap = plt.get_cmap("jet")
    heatmap_colored = cmap(heatmap_np)[:, :, :3]
    overlay = 0.5 * original_image / 255.0 + 0.5 * heatmap_colored

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original_image)
    axes[0].set_title(f"Original\nTrue: {class_names[true_label]}")
    axes[0].axis("off")
    axes[1].imshow(heatmap_np, cmap="jet")
    axes[1].set_title("Grad-CAM\n(moat_attention.conv_out)")
    axes[1].axis("off")
    axes[2].imshow(np.clip(overlay, 0, 1))
    axes[2].set_title(f"Overlay\nPred: {class_names[pred_label]}")
    axes[2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
