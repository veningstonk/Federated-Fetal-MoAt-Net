"""
train_centralized.py
────────────────────
Centralized (non-federated) training of Fetal MoAt Net and baseline models
on FPUS23 and FETAL_PLANES_DB.

Usage
─────
# Train Fetal MoAt Net on FPUS23 (default)
python train_centralized.py --dataset fpus23 --epochs 50

# Train a baseline on FETAL_PLANES_DB
python train_centralized.py --dataset fetal_planes_db --model resnet101 --epochs 50

# Full argument list
python train_centralized.py --help
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np

from data.dataset import FetalUltrasoundDataset, get_dataset_info, TRAIN_TRANSFORM, EVAL_TRANSFORM
from models.moat_net import build_moat_net
from models.baselines import build_baseline, BASELINE_REGISTRY
from utils.metrics import evaluate_model, print_metrics
from utils.visualization import (
    plot_training_curves,
    plot_roc_curves,
    plot_pr_curves,
    plot_confusion_matrix,
)


# ─────────────────────────── argument parser ──────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Centralized training of Fetal MoAt Net and baselines."
    )
    parser.add_argument("--dataset",   type=str, default="fpus23",
                        choices=["fpus23", "fetal_planes_db"])
    parser.add_argument("--data_root", type=str, default="data",
                        help="Root directory containing dataset subfolders.")
    parser.add_argument("--model",     type=str, default="moat_net",
                        choices=["moat_net"] + list(BASELINE_REGISTRY.keys()))
    parser.add_argument("--epochs",    type=int, default=50)
    parser.add_argument("--batch_size",type=int, default=32)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--freeze_backbone", action="store_true", default=True)
    return parser.parse_args()


# ─────────────────────────── training loop ────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss    += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total         += labels.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)
        probs  = torch.softmax(logits, dim=1)
        total_loss    += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total         += labels.size(0)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    y_proba = np.concatenate(all_probs)
    y_true  = np.concatenate(all_labels)
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")
    return total_loss / max(total, 1), total_correct / max(total, 1), auc


# ─────────────────────────── main ─────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dataset ────────────────────────────────────────────────────────────────
    info = get_dataset_info(args.dataset)
    num_classes  = info["num_classes"]
    class_names  = info["class_names"]

    train_ds = FetalUltrasoundDataset(
        root=args.data_root, dataset=args.dataset,
        split="train", transform=TRAIN_TRANSFORM, seed=args.seed
    )
    val_ds   = FetalUltrasoundDataset(
        root=args.data_root, dataset=args.dataset,
        split="test", transform=EVAL_TRANSFORM, seed=args.seed
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    print(f"Dataset    : {args.dataset}  ({len(train_ds)} train / {len(val_ds)} val)")
    print(f"Classes    : {class_names}")

    # ── Model ──────────────────────────────────────────────────────────────────
    if args.model == "moat_net":
        model = build_moat_net(num_classes=num_classes,
                               freeze_backbone=args.freeze_backbone)
    else:
        model = build_baseline(args.model, num_classes=num_classes,
                               freeze_backbone=args.freeze_backbone)
    model = model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"Model      : {args.model}  "
          f"({trainable/1e6:.2f}M trainable / {total_p/1e6:.2f}M total params)")

    # ── Training ───────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    run_tag = f"{args.model}_{args.dataset}_centralized"
    save_dir = os.path.join(args.results_dir, run_tag)
    os.makedirs(save_dir, exist_ok=True)

    history = {k: [] for k in ["train_loss", "val_loss",
                                "train_acc",  "val_acc",
                                "train_auc",  "val_auc"]}
    best_val_acc = 0.0
    best_ckpt    = os.path.join(save_dir, "best_model.pt")

    print(f"\nTraining for {args.epochs} epochs …")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc, vl_auc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        history["val_auc"].append(vl_auc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), best_ckpt)

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}  "
              f"vl_loss={vl_loss:.4f}  vl_acc={vl_acc:.4f}  "
              f"vl_auc={vl_auc:.4f}  "
              f"[{time.time()-t0:.1f}s]")

    # ── Final evaluation on best checkpoint ────────────────────────────────────
    print("\nLoading best checkpoint for final evaluation …")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    final_metrics = evaluate_model(model, val_loader, device, class_names)
    print_metrics(final_metrics, prefix=f"{args.model}/{args.dataset}/centralized")

    # ── Save results ───────────────────────────────────────────────────────────
    results = {
        "model":    args.model,
        "dataset":  args.dataset,
        "setting":  "centralized",
        "accuracy": final_metrics["accuracy"],
        "auc":      final_metrics["auc"],
        "f1":       final_metrics["f1"],
        "per_class": final_metrics["per_class"],
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ── Plots ──────────────────────────────────────────────────────────────────
    from utils.metrics import collect_predictions
    y_true, y_pred, y_proba = collect_predictions(model, val_loader, device)

    plot_training_curves(history,
        save_path=os.path.join(save_dir, "training_curves.png"),
        title=f"{args.model} – {args.dataset} – Centralized")
    plot_roc_curves(y_true, y_proba, class_names,
        save_path=os.path.join(save_dir, "roc_curves.png"))
    plot_pr_curves(y_true, y_proba, class_names,
        save_path=os.path.join(save_dir, "pr_curves.png"))
    plot_confusion_matrix(final_metrics["confusion_matrix"], class_names,
        save_path=os.path.join(save_dir, "confusion_matrix.png"))

    print(f"\nResults saved to: {save_dir}")
    print(f"Best model  : {best_ckpt}")
    print(f"Final Acc   : {final_metrics['accuracy']*100:.2f}%")
    print(f"Final AUC   : {final_metrics['auc']*100:.2f}%")


if __name__ == "__main__":
    main()
