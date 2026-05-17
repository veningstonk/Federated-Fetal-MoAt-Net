"""
strategies/proto_fl.py
──────────────────────
Contrastive Prototype FL — inspired by Fiorentino et al. (2025).

Reference
─────────
Fiorentino M.C. et al., "Contrastive prototype federated learning against noisy
labels in fetal standard plane detection," Int. J. Computer Assisted Radiology
and Surgery, vol. 20, pp. 1431–1439, 2025. doi: 10.1007/s11548-025-03400-6

Key idea
─────────
1. The largest client applies SimCLR-style contrastive learning to generate
   robust embeddings and uses k-NN re-labeling to correct noisy labels.
2. Class prototypes (mean embeddings per class) are computed from the
   noise-corrected labels and shared with all smaller clients.
3. Smaller clients use these prototypes as regularization anchors during
   local training, guiding their feature space without requiring their
   own clean labels.

Implementation notes
─────────────────────
• This file implements a practical approximation of the above framework
  compatible with the Flower FL pipeline used in this repository.
• SimCLR pretraining is approximated by a lightweight projection head
  trained with NT-Xent (InfoNCE) loss on the backbone features.
• Prototype sharing is implemented by appending prototype tensors to the
  model state during aggregation — no raw data is shared between clients.
• The prototype regularisation loss encourages the local feature space
  to align with the shared prototype centroids.
"""

import copy
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from flwr.common import Scalar


# ─────────────────────────── SimCLR projection head ───────────────────────────

class SimCLRProjectionHead(nn.Module):
    """2-layer MLP projection head for contrastive learning (128-D output)."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


# ─────────────────────────── NT-Xent loss ─────────────────────────────────────

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    NT-Xent (InfoNCE) contrastive loss for SimCLR.

    Parameters
    ----------
    z1, z2 : normalised projection vectors for two augmented views. (B × D)
    """
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)              # 2B × D
    sim = torch.mm(z, z.T) / temperature         # 2B × 2B
    # Mask out self-similarities
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))

    # Positive pair indices: (i, i+B) and (i+B, i)
    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B, device=z.device),
    ])
    return F.cross_entropy(sim, labels)


# ─────────────────────────── prototype computation ────────────────────────────

@torch.no_grad()
def compute_class_prototypes(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
    feature_dim: int = 512,
) -> torch.Tensor:
    """
    Compute mean class prototype embeddings from the bottleneck layer.

    Returns
    -------
    prototypes : Tensor of shape (num_classes, feature_dim)
    """
    model.eval()
    accum = torch.zeros(num_classes, feature_dim, device=device)
    counts = torch.zeros(num_classes, device=device)

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        # Extract features from bottleneck (before classifier)
        feats = _extract_features(model, images)   # B × feature_dim
        for i in range(num_classes):
            mask = labels == i
            if mask.any():
                accum[i] += feats[mask].sum(0)
                counts[i] += mask.sum().float()

    counts = counts.clamp(min=1)
    return accum / counts.unsqueeze(1)             # num_classes × feature_dim


@torch.no_grad()
def _extract_features(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """
    Extract bottleneck features from FetalMoAtNet (or any model with .bottleneck).
    Falls back to logits if bottleneck attribute is absent.
    """
    if hasattr(model, "backbone") and hasattr(model, "moat_attention"):
        # FetalMoAtNet
        x = model.backbone(images)
        x = model.moat_attention(x)
        x = model.gap(x).flatten(1)
        return model.bottleneck(x)
    else:
        return model(images)


# ─────────────────────────── prototype regularisation loss ────────────────────

def prototype_regularisation_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Encourage each sample's feature embedding to be close to its class prototype.

    Loss = − (1/N) Σ log [ exp(sim(fᵢ, pᵧᵢ)/τ) / Σⱼ exp(sim(fᵢ, pⱼ)/τ) ]
    """
    # features: B × D,  prototypes: C × D
    f_norm = F.normalize(features, dim=1)
    p_norm = F.normalize(prototypes, dim=1)
    logits = torch.mm(f_norm, p_norm.T) / temperature   # B × C
    return F.cross_entropy(logits, labels)


# ─────────────────────────── proto-FL local training ──────────────────────────

def proto_fl_train(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Scalar],
    prototypes: Optional[torch.Tensor] = None,
) -> Dict:
    """
    Contrastive Prototype FL local training step.

    Parameters
    ----------
    model        : local model.
    train_loader : client's DataLoader.
    device       : torch.device.
    config       : Flower config dict.
                   Keys: 'local_epochs', 'lr', 'lambda_proto' (default 0.5)
    prototypes   : shared class prototype tensor (num_classes × feature_dim),
                   or None if not yet available (first round).

    Returns
    -------
    Dict with 'train_loss', 'train_acc'.
    """
    local_epochs  = int(config.get("local_epochs", 3))
    lr            = float(config.get("lr", 1e-3))
    lambda_proto  = float(config.get("lambda_proto", 0.5))

    model.train()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    total_loss, total_correct, total_samples = 0.0, 0, 0

    for _ in range(local_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Classification loss
            logits = model(images)
            ce_loss = criterion(logits, labels)

            # Prototype regularisation (only when prototypes are available)
            proto_loss = torch.tensor(0.0, device=device)
            if prototypes is not None:
                feats = _extract_features(model, images)
                proto_loss = prototype_regularisation_loss(
                    feats, labels, prototypes.to(device)
                )

            loss = ce_loss + lambda_proto * proto_loss
            loss.backward()
            optimizer.step()

            total_loss    += ce_loss.item() * labels.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc  = total_correct / max(total_samples, 1)
    return {"train_loss": avg_loss, "train_acc": avg_acc}
