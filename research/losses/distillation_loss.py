"""
Composite Loss Functions for EchoDepthNet:
  1. Scale-Invariant Logarithmic Depth Loss (SILog)
  2. Multi-Scale Edge-Preserving Gradient Matching Loss
  3. Acoustic Spatial Hazard Matrix Alignment Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SILogLoss(nn.Module):
    def __init__(self, variance_focus: float = 0.85, eps: float = 1e-4):
        super().__init__()
        self.variance_focus = variance_focus
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        d = torch.log(pred.clamp(min=self.eps)) - torch.log(target.clamp(min=self.eps))
        return torch.sqrt(torch.mean(d ** 2) - self.variance_focus * (torch.mean(d) ** 2) + 1e-6)


class MultiScaleGradientLoss(nn.Module):
    """Encourages crisp obstacle boundary transitions in predicted depth maps."""

    def __init__(self, scales: int = 3):
        super().__init__()
        self.scales = scales

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=pred.device)
        for s in range(self.scales):
            step = 2 ** s
            p = pred[:, :, ::step, ::step]
            t = target[:, :, ::step, ::step]
            diff = p - t
            gx = torch.abs(diff[:, :, :, 1:] - diff[:, :, :, :-1]).mean()
            gy = torch.abs(diff[:, :, 1:, :] - diff[:, :, :-1, :]).mean()
            loss = loss + gx + gy
        return loss / self.scales


class EchoSightCompositeLoss(nn.Module):
    def __init__(self, w_l1: float = 1.8, w_silog: float = 0.45, w_grad: float = 0.6, w_acoustic: float = 1.0):
        super().__init__()
        self.silog = SILogLoss()
        self.grad = MultiScaleGradientLoss()
        self.w_l1 = w_l1
        self.w_silog = w_silog
        self.w_grad = w_grad
        self.w_acoustic = w_acoustic

    def forward(
        self,
        pred_depth: torch.Tensor,
        target_depth: torch.Tensor,
        pred_acoustic: torch.Tensor,
        target_acoustic: torch.Tensor,
    ):
        l_silog = self.silog(pred_depth, target_depth)
        l_grad = self.grad(pred_depth, target_depth)
        l_l1 = F.l1_loss(pred_depth, target_depth)
        l_acoustic = F.smooth_l1_loss(pred_acoustic, target_acoustic)
        total = (
            self.w_l1 * l_l1
            + self.w_silog * l_silog
            + self.w_grad * l_grad
            + self.w_acoustic * l_acoustic
        )
        return total, {
            "total": float(total.item()),
            "l1": float(l_l1.item()),
            "silog": float(l_silog.item()),
            "grad": float(l_grad.item()),
            "acoustic": float(l_acoustic.item()),
        }
