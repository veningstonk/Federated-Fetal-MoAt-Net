"""
strategies/fedprox.py
─────────────────────
FedProx local training function.

Reference
─────────
Li T. et al., "Federated Optimization in Heterogeneous Networks,"
Proceedings of Machine Learning and Systems (MLSys), vol. 2, pp. 429–450, 2020.
arXiv: 1812.06127

Key modification over FedAvg
─────────────────────────────
FedProx adds a proximal regularization term to each client's local objective:

    L_prox(w) = L_local(w) + (μ/2) · ‖w − w_global‖²

This penalises client updates that deviate far from the received global model,
providing convergence guarantees under statistical heterogeneity (Non-IID data)
and systems heterogeneity (variable local computation).

Tuning μ
─────────
Li et al. recommend tuning μ ∈ {0.001, 0.01, 0.1, 0.5, 1.0}.
Default μ = 0.01 works well for medical image classification tasks.
"""

import copy
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from flwr.common import Scalar


def fedprox_train(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Scalar],
) -> Dict[str, float]:
    """
    FedProx local training step with proximal regularisation.

    Parameters
    ----------
    model        : local model (already set to global weights by the client).
    train_loader : client's local DataLoader.
    device       : torch.device.
    config       : Flower config dict.
                   Expected keys:
                     'local_epochs' (int,   default 3)
                     'lr'           (float, default 1e-3)
                     'mu'           (float, default 0.01)  ← proximal coefficient

    Returns
    -------
    Dict with 'train_loss', 'train_acc', 'proximal_loss'.
    """
    local_epochs = int(config.get("local_epochs", 3))
    lr           = float(config.get("lr", 1e-3))
    mu           = float(config.get("mu", 0.01))

    # Freeze a copy of the global model weights for proximal term computation
    global_model = copy.deepcopy(model)
    global_model.eval()
    for p in global_model.parameters():
        p.requires_grad = False

    model.train()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    total_loss, prox_loss_sum, total_correct, total_samples = 0.0, 0.0, 0, 0

    for _ in range(local_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            logits = model(images)
            ce_loss = criterion(logits, labels)

            # Proximal term: (μ/2) · Σ ‖wᵢ − w_global_i‖²
            prox_term = torch.tensor(0.0, device=device)
            for w_local, w_global in zip(model.parameters(), global_model.parameters()):
                if w_local.requires_grad:
                    prox_term += (w_local - w_global.to(device)).norm(2) ** 2

            loss = ce_loss + (mu / 2.0) * prox_term
            loss.backward()
            optimizer.step()

            total_loss    += ce_loss.item() * labels.size(0)
            prox_loss_sum += prox_term.item() * labels.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss     = total_loss / max(total_samples, 1)
    avg_prox     = prox_loss_sum / max(total_samples, 1)
    avg_acc      = total_correct / max(total_samples, 1)
    return {"train_loss": avg_loss, "proximal_loss": avg_prox, "train_acc": avg_acc}
