"""
models/moat_net.py
──────────────────
Fetal MoAt Net: MobileNetV2 backbone + MoAt Custom Attention Block + Classifier Head.

Architecture
────────────
  Backbone   : MobileNetV2 (ImageNet pretrained, base layers frozen)
  Attention  : MoAt block — Q/K/V projections via 1×1 conv,
               scaled dot-product attention, learnable residual γ
  Head       : GAP → Dense(1024,ReLU) → Dense(512,ReLU) → Dense(num_classes, Softmax)

Parameters : ~4.1 M  |  FLOPs : ~1.2 B

Reference
─────────
Mushtaq G. and Veningston K., "Fetal MoAt Net: a light-weight deep learning model
for fetal diagnostic plane classification," Int. J. Comput. Appl., 2025.
doi: 10.1080/1206212X.2025.2543550
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    _HAS_WEIGHTS_ENUM = True
except ImportError:
    from torchvision.models import mobilenet_v2
    _HAS_WEIGHTS_ENUM = False


# ─────────────────────────── MoAt Attention Block ─────────────────────────────

class MoAtAttention(nn.Module):
    """
    MoAt Custom Attention Block (Eq. 1-7 from the paper).

    For input feature map F ∈ R^{H×W×C}:
      Q = φ_I(F),   K = φ_II(F),   V = φ_III(F)   [1×1 conv projections]
      S  = Q·Kᵀ
      Sₛ = S / √d
      A  = softmax(Sₛ)
      F_att = A·V
      F_out = conv₁ₓ₁(concat(F, γ·F_att))

    Parameters
    ----------
    in_channels : number of input feature channels (C).
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.d = in_channels  # projection dimensionality = C

        # Three 1×1 conv projections → Q, K, V
        self.proj_q = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.proj_k = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.proj_v = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

        # Learnable residual scaling (γ), initialized to 0
        self.gamma = nn.Parameter(torch.zeros(1))

        # Output fusion projection: concat(F, γ·F_att) → C channels
        self.conv_out = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)
        self.bn_out   = nn.BatchNorm2d(in_channels)

        self._init_weights()

    def _init_weights(self):
        for m in [self.proj_q, self.proj_k, self.proj_v, self.conv_out]:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W          # number of spatial positions

        # ── Projections ────────────────────────────────────────────────────────
        Q = self.proj_q(x).view(B, C, N)          # B × C × N
        K = self.proj_k(x).view(B, C, N)          # B × C × N
        V = self.proj_v(x).view(B, C, N)          # B × C × N

        # ── Scaled dot-product attention ───────────────────────────────────────
        # S = Q · Kᵀ  →  B × N × N
        S = torch.bmm(Q.permute(0, 2, 1), K)      # B × N × N
        S = S / math.sqrt(self.d)                  # scaled scores
        A = F.softmax(S, dim=-1)                   # attention weights, sums to 1

        # ── Feature aggregation: F_att = A · Vᵀ ───────────────────────────────
        F_att = torch.bmm(V, A.permute(0, 2, 1))  # B × C × N
        F_att = F_att.view(B, C, H, W)            # restore spatial dims

        # ── Residual fusion with learnable γ ──────────────────────────────────
        fused = torch.cat([x, self.gamma * F_att], dim=1)   # B × 2C × H × W
        out   = self.bn_out(self.conv_out(fused))            # B × C × H × W

        return F.relu(out, inplace=True)


# ─────────────────────────── Fetal MoAt Net ───────────────────────────────────

class FetalMoAtNet(nn.Module):
    """
    Fetal MoAt Net.

    Parameters
    ----------
    num_classes : number of output classes (4 for FPUS23, 6 for FETAL_PLANES_DB).
    freeze_backbone : if True, freeze all MobileNetV2 base layers.
    """

    def __init__(self, num_classes: int = 4, freeze_backbone: bool = True):
        super().__init__()

        # ── Backbone: MobileNetV2 ──────────────────────────────────────────────
        # Load MobileNetV2; fall back to random weights if offline (weights downloaded during training)
        try:
            if _HAS_WEIGHTS_ENUM:
                mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
            else:
                mobilenet = mobilenet_v2(pretrained=True)
        except Exception:
            import warnings
            warnings.warn(
                "Pretrained MobileNetV2 weights unavailable (no network). "
                "Random weights used — download weights before training.",
                RuntimeWarning,
            )
            mobilenet = mobilenet_v2(weights=None)
        self.backbone = mobilenet.features   # output: B × 1280 × 7 × 7

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        backbone_channels = 1280

        # ── MoAt Attention Block ───────────────────────────────────────────────
        self.moat_attention = MoAtAttention(in_channels=backbone_channels)

        # ── Classifier Head ────────────────────────────────────────────────────
        self.gap         = nn.AdaptiveAvgPool2d(1)
        self.bottleneck  = nn.Sequential(
            nn.Linear(backbone_channels, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.classifier  = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone feature extraction
        features = self.backbone(x)                  # B × 1280 × 7 × 7

        # MoAt attention refinement
        attended = self.moat_attention(features)     # B × 1280 × 7 × 7

        # Global average pooling + head
        pooled = self.gap(attended).flatten(1)       # B × 1280
        out    = self.bottleneck(pooled)             # B × 512
        logits = self.classifier(out)               # B × num_classes
        return logits

    def get_attention_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return attended feature maps for Grad-CAM visualization."""
        features = self.backbone(x)
        return self.moat_attention(features)


def build_moat_net(num_classes: int, freeze_backbone: bool = True) -> FetalMoAtNet:
    """Convenience factory."""
    return FetalMoAtNet(num_classes=num_classes, freeze_backbone=freeze_backbone)
