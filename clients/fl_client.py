"""
clients/fl_client.py
────────────────────
Flower FL client used by all federated learning strategies.

The client wraps a PyTorch model and exposes the standard Flower
NumPyClient interface (get_parameters, set_parameters, fit, evaluate).

Strategy-specific behaviour (FedProx proximal term, SCAFFOLD control
variates) is injected via a `train_fn` callable so that this client
class remains reusable across all strategies.
"""

import copy
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

import flwr as fl
from flwr.common import NDArrays, Scalar

from utils.metrics import evaluate_model


# ─────────────────────────── parameter helpers ────────────────────────────────

def get_model_parameters(model: nn.Module) -> NDArrays:
    """Extract model parameters as a list of numpy arrays."""
    return [p.detach().cpu().numpy() for p in model.parameters()]


def set_model_parameters(model: nn.Module, parameters: NDArrays) -> None:
    """Load a list of numpy arrays into model parameters in-place."""
    params_dict = zip(model.parameters(), parameters)
    for param, new_val in params_dict:
        param.data = torch.tensor(new_val, dtype=param.dtype).to(param.device)


# ─────────────────────────── FL client ────────────────────────────────────────

class FetalFLClient(fl.client.NumPyClient):
    """
    Generic Flower NumPy client for fetal ultrasound FL experiments.

    Parameters
    ----------
    client_id    : int — unique identifier for this client.
    model        : nn.Module — the model to train locally.
    train_subset : Dataset — this client's training data subset.
    val_loader   : DataLoader — shared validation loader (for reporting only).
    train_fn     : Callable — strategy-specific local training function.
                   Signature: train_fn(model, train_loader, device, config) → Dict
    device       : torch.device
    batch_size   : int
    class_names  : List[str]
    num_workers  : int
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_subset: Dataset,
        val_loader: DataLoader,
        train_fn: Callable,
        device: torch.device,
        batch_size: int = 32,
        class_names: Optional[List[str]] = None,
        num_workers: int = 2,
    ):
        self.client_id   = client_id
        self.model       = model.to(device)
        self.val_loader  = val_loader
        self.train_fn    = train_fn
        self.device      = device
        self.class_names = class_names
        self.train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

    # ── Flower interface ───────────────────────────────────────────────────────

    def get_parameters(self, config: Dict) -> NDArrays:
        return get_model_parameters(self.model)

    def set_parameters(self, parameters: NDArrays) -> None:
        set_model_parameters(self.model, parameters)

    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Receive global model, train locally, return updated parameters."""
        self.set_parameters(parameters)
        metrics = self.train_fn(
            model=self.model,
            train_loader=self.train_loader,
            device=self.device,
            config=config,
        )
        updated_params = get_model_parameters(self.model)
        num_examples   = len(self.train_loader.dataset)
        return updated_params, num_examples, metrics

    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Receive global model, evaluate on local validation set."""
        self.set_parameters(parameters)
        metrics = evaluate_model(
            model=self.model,
            loader=self.val_loader,
            device=self.device,
            class_names=self.class_names,
        )
        loss = 1.0 - metrics["accuracy"]  # proxy loss for Flower reporting
        num_examples = len(self.val_loader.dataset)
        return (
            loss,
            num_examples,
            {
                "accuracy": metrics["accuracy"],
                "auc":      metrics["auc"],
                "f1":       metrics["f1"],
            },
        )
