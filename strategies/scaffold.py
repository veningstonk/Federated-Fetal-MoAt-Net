"""
strategies/scaffold.py
──────────────────────
SCAFFOLD local training function with client and server control variates.

Reference
─────────
Karimireddy S.P. et al., "SCAFFOLD: Stochastic Controlled Averaging for
Federated Learning," ICML 2020, PMLR 119:5132–5143.
arXiv: 1910.06378

Key idea
─────────
FedAvg suffers from "client drift" in Non-IID settings because clients
optimise towards their own local optima, causing the average update to
deviate from the true global gradient direction.

SCAFFOLD corrects this drift using per-client control variates (cᵢ) and
a global control variate (c) that are maintained and updated across rounds.

Local update rule (Option II, the practical variant used here):
    yᵢ ← yᵢ − η(∇Fᵢ(yᵢ) − cᵢ + c)     [gradient correction]

Control variate update at the end of each round:
    cᵢ⁺ ← cᵢ − c + (1/ηK)(x − yᵢ)     [K = local steps, x = global params]
    Δcᵢ  = cᵢ⁺ − cᵢ                    [sent to server]

Server aggregation:
    c ← c + (1/N) Σ Δcᵢ                [N = number of sampled clients]

Implementation notes
─────────────────────
• Each client maintains its own control variate stored in-memory between
  rounds as a list of numpy arrays (one per trainable parameter).
• The server control variate is initialised to zero and updated by averaging
  the client deltas received after each round.
• The control variates are passed via the `config` dict using a JSON-serialisable
  format (flat list of floats); for very large models consider quantisation.
"""

import copy
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from flwr.common import Scalar


# ─────────────────────────── control variate store ────────────────────────────

class ControlVariateStore:
    """
    Maintains per-client and server-side SCAFFOLD control variates.

    Usage
    ─────
    store = ControlVariateStore(num_params)
    # At start of round, fetch client and server CVs:
    ci = store.get_client(client_id)
    c  = store.get_server()
    # After round, update:
    store.update_client(client_id, new_ci)
    store.update_server(delta_c_list)
    """

    def __init__(self, num_clients: int):
        self.client_cvs: Dict[int, Optional[List[np.ndarray]]] = {
            i: None for i in range(num_clients)
        }
        self.server_cv: Optional[List[np.ndarray]] = None

    def initialise(self, model: nn.Module) -> None:
        """Initialise all control variates to zero tensors matching model params."""
        zeros = [np.zeros_like(p.detach().cpu().numpy()) for p in model.parameters()
                 if p.requires_grad]
        self.server_cv = copy.deepcopy(zeros)
        for cid in self.client_cvs:
            self.client_cvs[cid] = copy.deepcopy(zeros)

    def get_client(self, client_id: int) -> List[np.ndarray]:
        return self.client_cvs[client_id]

    def get_server(self) -> List[np.ndarray]:
        return self.server_cv

    def update_client(self, client_id: int, new_cv: List[np.ndarray]) -> None:
        self.client_cvs[client_id] = new_cv

    def update_server(self, delta_c_list: List[List[np.ndarray]], num_sampled: int) -> None:
        """Aggregate delta_c from all sampled clients: c ← c + (1/N)·Σ Δcᵢ"""
        for i, delta in enumerate(zip(*delta_c_list)):
            self.server_cv[i] += np.mean(np.stack(delta, axis=0), axis=0)


# ─────────────────────────── local training ───────────────────────────────────

def scaffold_train(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Scalar],
    client_cv: List[np.ndarray],
    server_cv: List[np.ndarray],
) -> Dict:
    """
    SCAFFOLD local training step (Option II).

    Parameters
    ----------
    model        : local model with global weights already loaded.
    train_loader : client's local DataLoader.
    device       : torch.device.
    config       : Flower config dict with 'local_epochs' and 'lr'.
    client_cv    : this client's control variate (list of np.ndarray).
    server_cv    : server control variate (list of np.ndarray).

    Returns
    -------
    Dict with 'train_loss', 'train_acc', and 'new_client_cv' + 'delta_c'.
    """
    local_epochs = int(config.get("local_epochs", 3))
    lr           = float(config.get("lr", 1e-3))

    # Snapshot of global parameters before local training
    x_global = [p.detach().clone() for p in model.parameters() if p.requires_grad]

    # Convert control variates to tensors on device
    ci = [torch.tensor(cv, dtype=torch.float32, device=device) for cv in client_cv]
    c  = [torch.tensor(cv, dtype=torch.float32, device=device) for cv in server_cv]

    model.train()
    # SCAFFOLD uses SGD (not Adam) to keep the gradient correction well-defined
    optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    total_loss, total_correct, total_samples, local_steps = 0.0, 0, 0, 0

    for _ in range(local_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()

            # Apply control variate correction to each gradient
            # ∇ corrected = ∇F(y) − cᵢ + c
            for param, cvi, cv_server in zip(
                filter(lambda p: p.requires_grad, model.parameters()), ci, c
            ):
                if param.grad is not None:
                    param.grad.data += (- cvi + cv_server)

            optimizer.step()

            total_loss    += loss.item() * labels.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)
            local_steps   += 1

    # ── Update client control variate (Option II) ──────────────────────────────
    # cᵢ⁺ ← cᵢ − c + (1 / (K·η)) · (x_global − y_final)
    # where K = total local steps, η = lr
    K = max(local_steps, 1)
    new_ci_list, delta_c_list = [], []
    param_iter = filter(lambda p: p.requires_grad, model.parameters())

    for y_param, x_param, cvi, cv_s in zip(param_iter, x_global, ci, c):
        new_cvi  = cvi - cv_s + (x_param.to(device) - y_param.detach()) / (K * lr)
        delta_c  = new_cvi - cvi
        new_ci_list.append(new_cvi.cpu().numpy())
        delta_c_list.append(delta_c.cpu().numpy())

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc  = total_correct / max(total_samples, 1)

    return {
        "train_loss":    avg_loss,
        "train_acc":     avg_acc,
        "new_client_cv": new_ci_list,
        "delta_c":       delta_c_list,
    }
