"""
models/baselines.py
───────────────────
Centralised baseline model wrappers used for comparison:
  • ResNet-101
  • DenseNet-121
  • VGG-16
  • InceptionV3  (via timm)
  • Xception     (via timm)

All models share the same classifier head design used in the paper:
  GAP → Dense(1024, ReLU) → Dense(512, ReLU) → Dense(num_classes)
"""

import torch
import torch.nn as nn
from torchvision import models
import timm


# ─────────────────────────── shared head builder ──────────────────────────────

def _make_head(in_features: int, num_classes: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )


# ─────────────────────────── ResNet-101 ───────────────────────────────────────

class ResNet101Baseline(nn.Module):
    def __init__(self, num_classes: int = 4, freeze_backbone: bool = True):
        super().__init__()
        base = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        if freeze_backbone:
            for p in base.parameters():
                p.requires_grad = False
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.gap       = nn.AdaptiveAvgPool2d(1)
        self.head      = _make_head(2048, num_classes)

    def forward(self, x):
        x = self.gap(self.features(x)).flatten(1)
        return self.head(x)


# ─────────────────────────── DenseNet-121 ─────────────────────────────────────

class DenseNet121Baseline(nn.Module):
    def __init__(self, num_classes: int = 4, freeze_backbone: bool = True):
        super().__init__()
        base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for p in base.features.parameters():
                p.requires_grad = False
        self.features = base.features
        self.gap       = nn.AdaptiveAvgPool2d(1)
        self.head      = _make_head(1024, num_classes)

    def forward(self, x):
        x = torch.relu(self.features(x))
        x = self.gap(x).flatten(1)
        return self.head(x)


# ─────────────────────────── VGG-16 ───────────────────────────────────────────

class VGG16Baseline(nn.Module):
    def __init__(self, num_classes: int = 4, freeze_backbone: bool = True):
        super().__init__()
        base = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        if freeze_backbone:
            for p in base.features.parameters():
                p.requires_grad = False
        self.features = base.features
        self.gap       = nn.AdaptiveAvgPool2d(1)
        self.head      = _make_head(512, num_classes)

    def forward(self, x):
        x = self.gap(self.features(x)).flatten(1)
        return self.head(x)


# ─────────────────────────── InceptionV3 ──────────────────────────────────────

class InceptionV3Baseline(nn.Module):
    def __init__(self, num_classes: int = 4, freeze_backbone: bool = True):
        super().__init__()
        # timm gives a cleaner InceptionV3 without auxiliary head complications
        self.backbone = timm.create_model(
            "inception_v3", pretrained=True, num_classes=0, global_pool="avg"
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = _make_head(self.backbone.num_features, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


# ─────────────────────────── Xception ─────────────────────────────────────────

class XceptionBaseline(nn.Module):
    def __init__(self, num_classes: int = 4, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "xception", pretrained=True, num_classes=0, global_pool="avg"
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.head = _make_head(self.backbone.num_features, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


# ─────────────────────────── registry ─────────────────────────────────────────

BASELINE_REGISTRY = {
    "resnet101":   ResNet101Baseline,
    "densenet121": DenseNet121Baseline,
    "vgg16":       VGG16Baseline,
    "inceptionv3": InceptionV3Baseline,
    "xception":    XceptionBaseline,
}


def build_baseline(name: str, num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """
    Factory function for baseline models.

    Parameters
    ----------
    name          : one of the keys in BASELINE_REGISTRY.
    num_classes   : output dimension.
    freeze_backbone: freeze pretrained weights in feature extractor.
    """
    assert name in BASELINE_REGISTRY, (
        f"Unknown baseline '{name}'. Choose from {list(BASELINE_REGISTRY)}"
    )
    return BASELINE_REGISTRY[name](num_classes=num_classes, freeze_backbone=freeze_backbone)
