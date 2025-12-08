# models/losses/CombinedLoss.py

import torch
import torch.nn as nn

from .sam_loss import sam_loss
from .VGGPerceptualLoss import VGGPerceptualLoss


class CombinedLoss(nn.Module):
    """
    Combined loss for *BAND targets only* (RGB / multispectral bands):

        total = alpha * L1(pred, target)
              + beta  * SAM(pred, target)
              + gamma * VGG(pred_rgb, target_rgb)

    This loss is NOT meant for vegetation indices (NDVI, etc.).
    Indices must use IndexStructureLoss directly.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.1,
        rgb_indices=None,         # which channels act as RGB for VGG
        reduction: str = "mean",
        **kwargs,                # allows safe extra params from config
    ):
        super().__init__()

        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unsupported reduction: {reduction}")

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # --- Base losses ---
        self.l1 = nn.L1Loss(reduction=reduction)
        self.sam = sam_loss
        self.vgg = VGGPerceptualLoss()

        # --- Default RGB mapping if not provided ---
        if rgb_indices is None:
            rgb_indices = [6, 2, 1]   # your original default
        self.rgb_indices = tuple(rgb_indices)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")

        # --- L1 term ---
        l1_val = self.l1(pred, target)

        # --- SAM term ---
        sam_val = self.sam(pred, target)

        # --- VGG perceptual term (on RGB-like channels only) ---
        pred_rgb = pred[:, self.rgb_indices, :, :]
        target_rgb = target[:, self.rgb_indices, :, :]
        vgg_val = self.vgg(pred_rgb, target_rgb)

        return (
            self.alpha * l1_val
          + self.beta  * sam_val
          + self.gamma * vgg_val
        )

