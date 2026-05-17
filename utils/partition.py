"""
utils/partition.py
──────────────────
IID and Non-IID (Dirichlet) data partitioning for federated learning experiments.

Reference for Dirichlet Non-IID partitioning:
  Hsieh et al., "Quagmire: Practical Federated Learning for Non-IID Data", NeurIPS 2020.
  Concentration parameter α=0.5 is the standard benchmark setting.
"""

import numpy as np
from typing import Dict, List, Tuple
from torch.utils.data import Dataset, Subset


def partition_iid(
    dataset: Dataset,
    num_clients: int,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """
    Uniformly distribute dataset indices across clients (IID).

    Parameters
    ----------
    dataset     : torch Dataset with __len__ defined.
    num_clients : number of FL clients.
    seed        : random seed for reproducibility.

    Returns
    -------
    Dict mapping client_id → list of sample indices.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset)).tolist()
    client_size = len(indices) // num_clients
    partitions: Dict[int, List[int]] = {}
    for cid in range(num_clients):
        start = cid * client_size
        end = start + client_size if cid < num_clients - 1 else len(indices)
        partitions[cid] = indices[start:end]
    return partitions


def partition_non_iid_dirichlet(
    dataset: Dataset,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """
    Distribute dataset indices across clients using a Dirichlet distribution
    to simulate realistic inter-institutional class imbalance (Non-IID).

    Parameters
    ----------
    dataset     : torch Dataset; must expose get_labels() → List[int].
    num_clients : number of FL clients.
    alpha       : Dirichlet concentration parameter.
                  α→0: extreme heterogeneity; α→∞: approaches IID.
                  α=0.5 is the standard benchmark (Hsieh et al., 2020).
    seed        : random seed for reproducibility.

    Returns
    -------
    Dict mapping client_id → list of sample indices.
    """
    rng = np.random.default_rng(seed)
    labels = np.array(dataset.get_labels())
    num_classes = len(np.unique(labels))

    # Group indices by class
    class_indices: Dict[int, List[int]] = {
        c: np.where(labels == c)[0].tolist() for c in range(num_classes)
    }
    for c in class_indices:
        rng.shuffle(class_indices[c])

    # Sample client proportions per class from Dirichlet
    client_indices: Dict[int, List[int]] = {cid: [] for cid in range(num_clients)}

    for c in range(num_classes):
        proportions = rng.dirichlet(alpha=np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()
        splits = (proportions * len(class_indices[c])).astype(int)
        # Correct rounding error so all samples are assigned
        splits[-1] += len(class_indices[c]) - splits.sum()

        idxs = class_indices[c]
        offset = 0
        for cid in range(num_clients):
            client_indices[cid].extend(idxs[offset: offset + splits[cid]])
            offset += splits[cid]

    return client_indices


def build_client_datasets(
    dataset: Dataset,
    partitions: Dict[int, List[int]],
) -> Dict[int, Subset]:
    """
    Wrap each client's index list into a torch Subset.

    Parameters
    ----------
    dataset    : the full torch Dataset.
    partitions : output of partition_iid or partition_non_iid_dirichlet.

    Returns
    -------
    Dict mapping client_id → Subset.
    """
    return {cid: Subset(dataset, idxs) for cid, idxs in partitions.items()}


def get_partitions(
    dataset: Dataset,
    num_clients: int,
    partition: str = "iid",
    alpha: float = 0.5,
    seed: int = 42,
) -> Dict[int, Subset]:
    """
    Unified entry point: returns client Subsets for either IID or Non-IID.

    Parameters
    ----------
    dataset     : torch Dataset with get_labels() method.
    num_clients : number of FL clients.
    partition   : 'iid' or 'non_iid'.
    alpha       : Dirichlet α for Non-IID (ignored for IID).
    seed        : reproducibility seed.

    Returns
    -------
    Dict mapping client_id → Subset.
    """
    assert partition in ("iid", "non_iid"), "partition must be 'iid' or 'non_iid'"
    if partition == "iid":
        parts = partition_iid(dataset, num_clients, seed)
    else:
        parts = partition_non_iid_dirichlet(dataset, num_clients, alpha, seed)
    return build_client_datasets(dataset, parts)


def partition_summary(client_datasets: Dict[int, Subset], num_classes: int) -> None:
    """Print a per-client class distribution table for inspection."""
    print(f"\n{'Client':>8}  {'Samples':>8}  " +
          "  ".join(f"C{c:02d}" for c in range(num_classes)))
    print("-" * (20 + 6 * num_classes))
    for cid, subset in client_datasets.items():
        labels = [subset.dataset.samples[i][1] for i in subset.indices]
        counts = [labels.count(c) for c in range(num_classes)]
        row = "  ".join(f"{cnt:4d}" for cnt in counts)
        print(f"{cid:>8}  {len(labels):>8}  {row}")
    print()
