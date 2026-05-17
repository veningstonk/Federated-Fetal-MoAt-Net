"""
strategies/fedavg.py
────────────────────
FedAvg local training function.

Reference
─────────
McMahan H.B. et al., "Communication-Efficient Learning of Deep Networks
from Decentralized Data," AISTATS 2017.

FedAvg simply performs standard SGD/Adam local training for E epochs
and returns the updated model parameters. The server computes a weighted
average of all client parameters proportional to dataset size.
"""

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from flwr.common import Scalar


def fedavg_train(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Scalar],
) -> Dict[str, float]:
    """
    FedAvg local training step.

    Parameters
    ----------
    model        : local model (already set to global weights by the client).
    train_loader : client's local DataLoader.
    device       : torch.device.
    config       : Flower config dict.
                   Expected keys:
                     'local_epochs' (int, default 3)
                     'lr'           (float, default 1e-3)

    Returns
    -------
    Dict with 'train_loss' and 'train_acc'.
    """
    local_epochs = int(config.get("local_epochs", 3))
    lr           = float(config.get("lr", 1e-3))

    model.train()
    optimizer  = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion  = nn.CrossEntropyLoss()

    total_loss, total_correct, total_samples = 0.0, 0, 0

    for _ in range(local_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss    += loss.item() * labels.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc  = total_correct / max(total_samples, 1)
    return {"train_loss": avg_loss, "train_acc": avg_acc}
