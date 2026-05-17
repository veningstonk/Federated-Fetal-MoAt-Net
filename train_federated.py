"""
train_federated.py
──────────────────
Main entry point for federated learning experiments.

Supports:
  • Models   : Fetal MoAt Net, ResNet-101, DenseNet-121, VGG-16, InceptionV3, Xception
  • Strategies: FedAvg, FedProx, SCAFFOLD, Proto-FL (Fiorentino et al. 2025)
  • Datasets  : FPUS23, FETAL_PLANES_DB
  • Partitions: IID, Non-IID (Dirichlet, α=0.5)
  • Clients   : configurable (default: 5, 10, 20)

Usage
─────
# FedAvg, 10 clients, IID
python train_federated.py --strategy fedavg --dataset fpus23 --num_clients 10 --rounds 100

# FedProx, 20 clients, Non-IID, μ=0.01
python train_federated.py --strategy fedprox --dataset fpus23 --num_clients 20 \
                          --partition non_iid --mu 0.01 --rounds 100

# SCAFFOLD, 10 clients, Non-IID
python train_federated.py --strategy scaffold --dataset fpus23 --num_clients 10 \
                          --partition non_iid --rounds 100

# Contrastive Prototype FL
python train_federated.py --strategy proto_fl --dataset fpus23 --num_clients 10 \
                          --partition non_iid --rounds 100

# Run all strategies × all client configs × both partitions at once
python train_federated.py --strategy all --num_clients 5 10 20 \
                          --partition iid non_iid --dataset fpus23 --rounds 100
"""

import argparse
import copy
import json
import os
import sys
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import Metrics, NDArrays, Parameters, Scalar
from flwr.server.strategy import FedAvg as FlowerFedAvg

from data.dataset import (
    FetalUltrasoundDataset, get_dataset_info,
    TRAIN_TRANSFORM, EVAL_TRANSFORM,
)
from models.moat_net import build_moat_net
from models.baselines import build_baseline, BASELINE_REGISTRY
from utils.partition import get_partitions, partition_summary
from utils.metrics import evaluate_model, print_metrics, collect_predictions
from utils.visualization import (
    plot_training_curves,
    plot_roc_curves,
    plot_pr_curves,
    plot_confusion_matrix,
    plot_federated_comparison,
)
from strategies.fedavg   import fedavg_train
from strategies.fedprox  import fedprox_train
from strategies.scaffold import scaffold_train, ControlVariateStore
from strategies.proto_fl import proto_fl_train, compute_class_prototypes
from clients.fl_client   import FetalFLClient, get_model_parameters, set_model_parameters


# ─────────────────────────── argument parser ──────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Federated learning for fetal ultrasound plane classification."
    )
    parser.add_argument("--strategy",   type=str, default="fedavg",
                        choices=["fedavg", "fedprox", "scaffold", "proto_fl", "all"])
    parser.add_argument("--model",      type=str, default="moat_net",
                        choices=["moat_net"] + list(BASELINE_REGISTRY.keys()))
    parser.add_argument("--dataset",    type=str, default="fpus23",
                        choices=["fpus23", "fetal_planes_db"])
    parser.add_argument("--data_root",  type=str, default="data")
    parser.add_argument("--num_clients",type=int, nargs="+", default=[10])
    parser.add_argument("--partition",  type=str, nargs="+", default=["iid"],
                        choices=["iid", "non_iid"])
    parser.add_argument("--rounds",     type=int, default=100)
    parser.add_argument("--local_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--mu",         type=float, default=0.01,
                        help="FedProx proximal coefficient.")
    parser.add_argument("--alpha",      type=float, default=0.5,
                        help="Dirichlet α for Non-IID partitioning.")
    parser.add_argument("--lambda_proto", type=float, default=0.5,
                        help="Prototype regularisation weight (Proto-FL).")
    parser.add_argument("--fraction_fit",type=float, default=1.0,
                        help="Fraction of clients sampled per round.")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--num_workers",type=int, default=2)
    parser.add_argument("--results_dir",type=str, default="results")
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    return parser.parse_args()


# ─────────────────────────── model factory ────────────────────────────────────

def build_model(model_name: str, num_classes: int, freeze_backbone: bool) -> nn.Module:
    if model_name == "moat_net":
        return build_moat_net(num_classes=num_classes, freeze_backbone=freeze_backbone)
    return build_baseline(model_name, num_classes=num_classes, freeze_backbone=freeze_backbone)


# ─────────────────────────── single FL experiment ─────────────────────────────

def run_experiment(
    strategy_name: str,
    model_name: str,
    dataset: str,
    data_root: str,
    num_clients: int,
    partition: str,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    lr: float,
    mu: float,
    alpha: float,
    lambda_proto: float,
    fraction_fit: float,
    seed: int,
    num_workers: int,
    results_dir: str,
    freeze_backbone: bool,
    device: torch.device,
) -> Dict:
    """Run a single FL experiment and return the final metrics dict."""

    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── Dataset ─────────────────────────────────────────────────────────────────
    info        = get_dataset_info(dataset)
    num_classes = info["num_classes"]
    class_names = info["class_names"]

    train_ds = FetalUltrasoundDataset(
        root=data_root, dataset=dataset, split="train",
        transform=TRAIN_TRANSFORM, seed=seed,
    )
    val_ds = FetalUltrasoundDataset(
        root=data_root, dataset=dataset, split="test",
        transform=EVAL_TRANSFORM, seed=seed,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)

    # ── Partition ────────────────────────────────────────────────────────────────
    client_subsets = get_partitions(train_ds, num_clients, partition, alpha, seed)
    partition_summary(client_subsets, num_classes)

    # ── Global model ─────────────────────────────────────────────────────────────
    global_model = build_model(model_name, num_classes, freeze_backbone).to(device)

    # ── Strategy-specific setup ───────────────────────────────────────────────────
    scaffold_store = None
    prototypes     = None

    if strategy_name == "scaffold":
        scaffold_store = ControlVariateStore(num_clients)
        scaffold_store.initialise(global_model)

    # Config dict passed to each client's fit() call
    config = {
        "local_epochs": local_epochs,
        "lr":           lr,
        "mu":           mu,
        "lambda_proto": lambda_proto,
    }

    # ── Training loop (manual simulation without Flower server/client sockets) ────
    # We use an in-process simulation: no network sockets, just Python calls.
    round_metrics: Dict[str, List[float]] = {"accuracy": [], "auc": [], "loss": []}

    run_tag  = (f"{strategy_name}_{model_name}_{dataset}_"
                f"{num_clients}c_{partition}")
    save_dir = os.path.join(results_dir, run_tag)
    os.makedirs(save_dir, exist_ok=True)
    best_ckpt = os.path.join(save_dir, "best_model.pt")
    best_acc  = 0.0

    print(f"\n{'='*60}")
    print(f"  Strategy : {strategy_name.upper()}")
    print(f"  Model    : {model_name}")
    print(f"  Dataset  : {dataset}  |  Clients : {num_clients}  |  {partition.upper()}")
    print(f"  Rounds   : {rounds}   |  Local epochs : {local_epochs}")
    print(f"{'='*60}")

    for rnd in range(1, rounds + 1):
        t0 = time.time()
        global_params = get_model_parameters(global_model)

        # Determine which clients participate this round
        n_sampled = max(1, int(fraction_fit * num_clients))
        sampled   = np.random.choice(num_clients, size=n_sampled, replace=False).tolist()

        client_updates: List[NDArrays] = []
        client_sizes:   List[int]      = []
        delta_c_list:   List           = []   # SCAFFOLD only

        for cid in sampled:
            local_model = copy.deepcopy(global_model)
            set_model_parameters(local_model, global_params)
            local_loader = DataLoader(
                client_subsets[cid], batch_size=batch_size,
                shuffle=True, num_workers=num_workers,
            )

            # ── Strategy-specific local training ──────────────────────────────
            if strategy_name == "fedavg":
                fedavg_train(local_model, local_loader, device, config)

            elif strategy_name == "fedprox":
                fedprox_train(local_model, local_loader, device, config)

            elif strategy_name == "scaffold":
                res = scaffold_train(
                    local_model, local_loader, device, config,
                    client_cv=scaffold_store.get_client(cid),
                    server_cv=scaffold_store.get_server(),
                )
                scaffold_store.update_client(cid, res["new_client_cv"])
                delta_c_list.append(res["delta_c"])

            elif strategy_name == "proto_fl":
                proto_fl_train(local_model, local_loader, device, config,
                               prototypes=prototypes)

            client_updates.append(get_model_parameters(local_model))
            client_sizes.append(len(client_subsets[cid]))

        # ── FedAvg aggregation (weighted by dataset size) ────────────────────
        total_samples = sum(client_sizes)
        agg_params    = []
        for layer_idx in range(len(global_params)):
            weighted = sum(
                client_updates[i][layer_idx] * (client_sizes[i] / total_samples)
                for i in range(len(sampled))
            )
            agg_params.append(weighted)

        set_model_parameters(global_model, agg_params)

        # ── SCAFFOLD: update server control variate ───────────────────────────
        if strategy_name == "scaffold" and delta_c_list:
            scaffold_store.update_server(delta_c_list, n_sampled)

        # ── Proto-FL: recompute prototypes every 5 rounds ────────────────────
        if strategy_name == "proto_fl" and rnd % 5 == 0:
            prototypes = compute_class_prototypes(
                global_model, val_loader, num_classes, device,
                feature_dim=512,
            )

        # ── Global evaluation ─────────────────────────────────────────────────
        metrics = evaluate_model(global_model, val_loader, device, class_names)
        round_metrics["accuracy"].append(metrics["accuracy"])
        round_metrics["auc"].append(metrics["auc"])

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            torch.save(global_model.state_dict(), best_ckpt)

        print(f"  Round {rnd:4d}/{rounds}  "
              f"acc={metrics['accuracy']*100:.2f}%  "
              f"auc={metrics['auc']*100:.2f}%  "
              f"[{time.time()-t0:.1f}s]")

    # ── Final evaluation ──────────────────────────────────────────────────────
    global_model.load_state_dict(torch.load(best_ckpt, map_location=device))
    final_metrics = evaluate_model(global_model, val_loader, device, class_names)
    print_metrics(final_metrics, prefix=run_tag)

    # ── Save results ──────────────────────────────────────────────────────────
    results_out = {
        "strategy":    strategy_name,
        "model":       model_name,
        "dataset":     dataset,
        "num_clients": num_clients,
        "partition":   partition,
        "rounds":      rounds,
        "accuracy":    final_metrics["accuracy"],
        "auc":         final_metrics["auc"],
        "f1":          final_metrics["f1"],
        "per_class":   final_metrics["per_class"],
        "round_accuracy": round_metrics["accuracy"],
        "round_auc":      round_metrics["auc"],
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results_out, f, indent=2)

    # ── Plots ─────────────────────────────────────────────────────────────────
    y_true, y_pred, y_proba = collect_predictions(global_model, val_loader, device)
    plot_roc_curves(y_true, y_proba, class_names,
        save_path=os.path.join(save_dir, "roc_curves.png"),
        title=f"ROC — {strategy_name} / {dataset} / {num_clients}c {partition}")
    plot_pr_curves(y_true, y_proba, class_names,
        save_path=os.path.join(save_dir, "pr_curves.png"),
        title=f"PR — {strategy_name} / {dataset} / {num_clients}c {partition}")
    plot_confusion_matrix(final_metrics["confusion_matrix"], class_names,
        save_path=os.path.join(save_dir, "confusion_matrix.png"),
        title=f"CM — {strategy_name} / {dataset} / {num_clients}c {partition}")
    plot_training_curves(
        {"val_acc": round_metrics["accuracy"], "val_auc": round_metrics["auc"]},
        save_path=os.path.join(save_dir, "round_curves.png"),
        title=f"FL Rounds — {strategy_name} / {dataset} / {num_clients}c {partition}",
    )

    print(f"\nResults saved to: {save_dir}")
    return results_out


# ─────────────────────────── main ─────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    strategies = (
        ["fedavg", "fedprox", "scaffold", "proto_fl"]
        if args.strategy == "all"
        else [args.strategy]
    )

    all_results = []
    for strategy in strategies:
        for nc in args.num_clients:
            for part in args.partition:
                res = run_experiment(
                    strategy_name=strategy,
                    model_name=args.model,
                    dataset=args.dataset,
                    data_root=args.data_root,
                    num_clients=nc,
                    partition=part,
                    rounds=args.rounds,
                    local_epochs=args.local_epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    mu=args.mu,
                    alpha=args.alpha,
                    lambda_proto=args.lambda_proto,
                    fraction_fit=args.fraction_fit,
                    seed=args.seed,
                    num_workers=args.num_workers,
                    results_dir=args.results_dir,
                    freeze_backbone=args.freeze_backbone,
                    device=device,
                )
                all_results.append(res)

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print(f"{'Strategy':<12} {'Dataset':<18} {'Clients':>8} {'Partition':<10} "
          f"{'Acc (%)':>8} {'AUC (%)':>8}")
    print("-"*80)
    for r in all_results:
        print(f"{r['strategy']:<12} {r['dataset']:<18} {r['num_clients']:>8} "
              f"{r['partition']:<10} {r['accuracy']*100:>8.2f} {r['auc']*100:>8.2f}")
    print("="*80)

    # Save master results JSON
    os.makedirs(args.results_dir, exist_ok=True)
    with open(os.path.join(args.results_dir, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to: {args.results_dir}/all_results.json")


if __name__ == "__main__":
    main()
