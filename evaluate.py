"""
evaluate.py
───────────
Standalone evaluation of any saved model checkpoint.

Usage
─────
python evaluate.py \
    --checkpoint results/fedavg_moat_net_fpus23_10c_iid/best_model.pt \
    --dataset fpus23 \
    --model moat_net \
    --data_root data \
    --gradcam          # optional: generate Grad-CAM figures
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import FetalUltrasoundDataset, get_dataset_info, EVAL_TRANSFORM
from models.moat_net import build_moat_net
from models.baselines import build_baseline, BASELINE_REGISTRY
from utils.metrics import evaluate_model, print_metrics, collect_predictions
from utils.visualization import (
    plot_roc_curves,
    plot_pr_curves,
    plot_confusion_matrix,
    save_gradcam_figure,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved model checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset",    type=str, default="fpus23",
                        choices=["fpus23", "fetal_planes_db"])
    parser.add_argument("--model",      type=str, default="moat_net",
                        choices=["moat_net"] + list(BASELINE_REGISTRY.keys()))
    parser.add_argument("--data_root",  type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--num_workers",type=int, default=2)
    parser.add_argument("--results_dir",type=str, default="results/eval")
    parser.add_argument("--gradcam",    action="store_true",
                        help="Generate Grad-CAM figures (Fetal MoAt Net only).")
    parser.add_argument("--gradcam_samples", type=int, default=4,
                        help="Number of Grad-CAM samples per class.")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.results_dir, exist_ok=True)

    # ── Dataset ────────────────────────────────────────────────────────────────
    info        = get_dataset_info(args.dataset)
    num_classes = info["num_classes"]
    class_names = info["class_names"]

    val_ds = FetalUltrasoundDataset(
        root=args.data_root, dataset=args.dataset, split="test",
        transform=EVAL_TRANSFORM, seed=args.seed,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    # ── Model ──────────────────────────────────────────────────────────────────
    if args.model == "moat_net":
        model = build_moat_net(num_classes=num_classes, freeze_backbone=False)
    else:
        model = build_baseline(args.model, num_classes=num_classes, freeze_backbone=False)

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    metrics = evaluate_model(model, val_loader, device, class_names)
    print_metrics(metrics, prefix=f"{args.model}/{args.dataset}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    y_true, y_pred, y_proba = collect_predictions(model, val_loader, device)

    plot_roc_curves(y_true, y_proba, class_names,
        save_path=os.path.join(args.results_dir, "roc_curves.png"))
    plot_pr_curves(y_true, y_proba, class_names,
        save_path=os.path.join(args.results_dir, "pr_curves.png"))
    plot_confusion_matrix(metrics["confusion_matrix"], class_names,
        save_path=os.path.join(args.results_dir, "confusion_matrix.png"))
    plot_confusion_matrix(metrics["confusion_matrix"], class_names,
        save_path=os.path.join(args.results_dir, "confusion_matrix_norm.png"),
        normalize=True)

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    if args.gradcam and args.model == "moat_net":
        import random
        from torchvision.transforms.functional import to_tensor
        from PIL import Image
        gradcam_dir = os.path.join(args.results_dir, "gradcam")
        os.makedirs(gradcam_dir, exist_ok=True)

        model.eval()
        # Sample a few images per class
        by_class = {c: [] for c in range(num_classes)}
        for img_path, label in val_ds.samples:
            by_class[label].append(img_path)

        for cls_idx, cls_name in enumerate(class_names):
            paths = by_class.get(cls_idx, [])
            sampled = random.sample(paths, min(args.gradcam_samples, len(paths)))
            for i, img_path in enumerate(sampled):
                orig = np.array(Image.open(img_path).convert("RGB").resize((224, 224)))
                tensor = EVAL_TRANSFORM(Image.open(img_path).convert("RGB")).to(device)
                save_gradcam_figure(
                    model=model,
                    image_tensor=tensor,
                    original_image=orig,
                    save_path=os.path.join(gradcam_dir, f"{cls_name}_{i:02d}.png"),
                    class_names=class_names,
                    true_label=cls_idx,
                    pred_label=cls_idx,
                )
        print(f"Grad-CAM figures saved to: {gradcam_dir}")
    elif args.gradcam and args.model != "moat_net":
        print("Note: Grad-CAM is only implemented for moat_net.")

    print(f"\nEvaluation complete. Results saved to: {args.results_dir}")
    print(f"Accuracy : {metrics['accuracy']*100:.2f}%")
    print(f"AUC      : {metrics['auc']*100:.2f}%")
    print(f"F1       : {metrics['f1']*100:.2f}%")


if __name__ == "__main__":
    main()
