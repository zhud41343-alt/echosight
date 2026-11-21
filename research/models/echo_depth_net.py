"""
EchoDepthNet: Edge-Efficient Depth-Aware Visual-to-Binaural Spatial Sonification Network
Designed for single-GPU training (e.g., RTX 4060 8GB) and real-time ONNX/WebGPU edge deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class DifferentiableSobelPrior(nn.Module):
    """Extracts high-frequency structural edge priors directly inside the neural network."""

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Rec. 709 luminance
        gray = 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]
        gx = F.conv2d(gray, self.kx, padding=1)
        gy = F.conv2d(gray, self.ky, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-6)
        return torch.clamp(mag, 0.0, 1.0)


class FrequencyEdgeFusionBlock(nn.Module):
    """
    FG-EDFM (Residual): Frequency-Guided Edge-Depth Fusion Module with Residual Skip Connection.
    Preserves low-frequency global depth semantics via an identity/1x1 shortcut while adding
    gated high-frequency Sobel edge refinement to sharpen obstacle boundaries without gradient decay.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.shortcut = (
            nn.Identity()
            if in_ch == out_ch
            else nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        )
        self.conv_feat = nn.Sequential(
            nn.Conv2d(in_ch + 1, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(out_ch + 1, out_ch, kernel_size=1),
            nn.Sigmoid(),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.25))
        self.act = nn.SiLU(inplace=True)

    def forward(self, feat: torch.Tensor, edge_map: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(feat)
        edge_resized = F.interpolate(edge_map, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.conv_feat(torch.cat([feat, edge_resized], dim=1))
        attn = self.gate(torch.cat([fused, edge_resized], dim=1))
        return self.act(identity + self.res_scale * (fused * attn))


class EchoDepthNet(nn.Module):
    """
    Dual-Head Lightweight Architecture:
      1) depth_map: Dense normalized inverse depth (proximity) in [0, 1], shape [B, 1, H, W]
      2) acoustic_matrix: 3D HRTF Spatial Sonification Hazard Matrix in [0, 1], shape [B, 1, 16, 48]
    """

    def __init__(self, pretrained_backbone: bool = False, use_edge_fusion: bool = True):
        super().__init__()
        self.use_edge_fusion = use_edge_fusion
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained_backbone else None
        backbone = mobilenet_v3_small(weights=weights).features

        # Multi-scale encoder stages from MobileNetV3-Small
        self.stage1 = backbone[:2]   # stride 4,  16 ch
        self.stage2 = backbone[2:4]  # stride 8,  24 ch
        self.stage3 = backbone[4:9]  # stride 16, 48 ch
        self.stage4 = backbone[9:]   # stride 32, 576 ch

        self.sobel_prior = DifferentiableSobelPrior()

        # Lightweight FPN Decoder
        self.lat4 = nn.Conv2d(576, 96, kernel_size=1)
        self.lat3 = nn.Conv2d(48, 96, kernel_size=1)
        self.lat2 = nn.Conv2d(24, 64, kernel_size=1)
        self.lat1 = nn.Conv2d(16, 32, kernel_size=1)

        self.fuse3 = FrequencyEdgeFusionBlock(96, 64) if use_edge_fusion else self._plain_block(96, 64)
        self.fuse2 = FrequencyEdgeFusionBlock(64, 32) if use_edge_fusion else self._plain_block(64, 32)
        self.fuse1 = FrequencyEdgeFusionBlock(32, 32) if use_edge_fusion else self._plain_block(32, 32)

        # Head 1: Dense Proximity / Inverse Depth Head
        self.depth_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Head 2: 16x48 Acoustic Spatial Hazard Head (aligned with EchoSight 3D HRTF engine)
        self.acoustic_head = nn.Sequential(
            nn.Conv2d(33, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((16, 48)),
            nn.Conv2d(24, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _plain_block(in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        h, w = x.shape[-2:]
        edge_map = self.sobel_prior(x)

        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)

        p4 = self.lat4(f4)
        p3 = self.lat3(f3) + F.interpolate(p4, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.fuse3(p3, edge_map) if self.use_edge_fusion else self.fuse3(p3)

        p2 = self.lat2(f2) + F.interpolate(d3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.fuse2(p2, edge_map) if self.use_edge_fusion else self.fuse2(p2)

        p1 = self.lat1(f1) + F.interpolate(d2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.fuse1(p1, edge_map) if self.use_edge_fusion else self.fuse1(p1)

        depth_low = self.depth_head(d1)
        depth_map = F.interpolate(depth_low, size=(h, w), mode="bilinear", align_corners=False)

        # Fuse structural features + proximity to predict 16x48 acoustic sonification matrix
        acoustic_matrix = self.acoustic_head(torch.cat([d1, depth_low], dim=1))
        return depth_map, acoustic_matrix


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = EchoDepthNet(pretrained_backbone=False, use_edge_fusion=True)
    dummy = torch.randn(2, 3, 256, 256)
    depth, acoustic = net(dummy)
    print(f"EchoDepthNet trainable params: {count_parameters(net) / 1e6:.2f} M")
    print(f"Depth map shape: {tuple(depth.shape)}, Acoustic matrix shape: {tuple(acoustic.shape)}")
