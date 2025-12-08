import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------
#  Utilities
# -----------------------------------------------------------

def _spatial_gradients(x: torch.Tensor):
    """
    Compute simple forward finite-difference gradients along x and y.

    x: (B, C, H, W)

    Returns:
        gx, gy with shape (B, C, H, W-1) and (B, C, H-1, W)
        for use in gradient-based losses (shapes match between pred/target).
    """
    gx = x[..., :, 1:] - x[..., :, :-1]   # grad along width (x)
    gy = x[..., 1:, :] - x[..., :-1, :]   # grad along height (y)
    return gx, gy


def _gaussian_window(window_size: int, sigma: float, device=None, dtype=None):
    """
    1D Gaussian window used to build a 2D separable kernel for SSIM.
    """
    gauss = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    gauss = torch.exp(-0.5 * (gauss / sigma) ** 2)
    gauss = gauss / gauss.sum()
    return gauss


def _create_ssim_kernel(window_size: int, sigma: float, channels: int, device=None, dtype=None):
    """
    Build 2D Gaussian kernel of shape (C, 1, k, k) for depthwise conv.
    """
    g1d = _gaussian_window(window_size, sigma, device=device, dtype=dtype)
    g2d = g1d[:, None] * g1d[None, :]  # outer product -> (k, k)
    g2d = g2d / g2d.sum()
    kernel = g2d.view(1, 1, window_size, window_size)
    kernel = kernel.repeat(channels, 1, 1, 1)   # (C, 1, k, k)
    return kernel


# -----------------------------------------------------------
#  Losses
# -----------------------------------------------------------
class GradientLoss(nn.Module):
    """
    Enforces similarity of spatial gradients between pred and target.

    Works for any number of channels (indices).
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")

        pred_gx, pred_gy = _spatial_gradients(pred)
        tgt_gx, tgt_gy = _spatial_gradients(target)

        diff_x = torch.abs(pred_gx - tgt_gx)  # (B, C, H, W-1)
        diff_y = torch.abs(pred_gy - tgt_gy)  # (B, C, H-1, W)

        if self.reduction == "mean":
            # scalar: mean error on x-gradients + mean error on y-gradients
            return diff_x.mean() + diff_y.mean()

        elif self.reduction == "sum":
            # scalar: sum of both
            return diff_x.sum() + diff_y.sum()

        else:  # "none"
            # If you ever need per-pixel maps, return them separately
            # (or change this to whatever interface you prefer)
            return diff_x, diff_y




class MultiScaleGradientLoss(nn.Module):
    """
    Multi-scale gradient loss: compute GradientLoss at different resolutions.

    At each scale:
      - compute gradients
      - compute L1 difference of gradients

    Then average across scales.
    """

    def __init__(self, num_scales: int = 3, reduction: str = "mean"):
        super().__init__()
        self.num_scales = num_scales
        self.base_grad_loss = GradientLoss(reduction=reduction)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")

        total = 0.0
        weight_sum = 0.0

        current_pred = pred
        current_target = target

        for s in range(self.num_scales):
            scale_weight = 1.0 / (2.0 ** s)   # higher weight at finer scales
            loss_s = self.base_grad_loss(current_pred, current_target)

            total = total + scale_weight * loss_s
            weight_sum = weight_sum + scale_weight

            # Downsample for next scale
            if s < self.num_scales - 1:
                current_pred = F.avg_pool2d(current_pred, kernel_size=2, stride=2, ceil_mode=False)
                current_target = F.avg_pool2d(current_target, kernel_size=2, stride=2, ceil_mode=False)

                # Stop if spatial size becomes too small
                if current_pred.shape[-1] < 4 or current_pred.shape[-2] < 4:
                    break

        return total / max(weight_sum, 1e-8)


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index Measure (SSIM) loss for 1 or more channels.

    Returns:
        loss = 1 - SSIM(pred, target)

    Assumes inputs in a reasonable numeric range (e.g. [-1, 1] or [0, 1]),
    but does not enforce it; you can normalize NDVI / indices beforehand
    if you want.
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        channels: int = 1,
        reduction: str = "mean",
        C1: float = 0.01 ** 2,
        C2: float = 0.03 ** 2,
    ):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.window_size = window_size
        self.sigma = sigma
        self.channels = channels
        self.reduction = reduction
        self.C1 = C1
        self.C2 = C2

        self.register_buffer(
            "kernel",
            _create_ssim_kernel(window_size, sigma, channels=channels, device="cpu", dtype=torch.float32)
        )

    def _conv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Depthwise conv with Gaussian kernel, per channel.
        Input:  (B, C, H, W)
        Output: (B, C, H', W')
        """
        C = x.shape[1]
        if C != self.channels:
            # recreate kernel for new number of channels, preserving device/dtype
            self.channels = C
            self.kernel = _create_ssim_kernel(
                self.window_size,
                self.sigma,
                channels=C,
                device=x.device,
                dtype=x.dtype,
            )
        return F.conv2d(
            x,
            self.kernel.to(device=x.device, dtype=x.dtype),
            padding=self.window_size // 2,
            groups=C,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")

        mu_x = self._conv(pred)
        mu_y = self._conv(target)

        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = self._conv(pred * pred) - mu_x2
        sigma_y2 = self._conv(target * target) - mu_y2
        sigma_xy = self._conv(pred * target) - mu_xy

        C1 = self.C1
        C2 = self.C2

        num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)

        ssim_map = num / (den + 1e-12)      # (B, C, H, W)
        loss_map = 1.0 - ssim_map           # 0 → perfect, ~1 → dissimilar

        if self.reduction == "mean":
            return loss_map.mean()
        elif self.reduction == "sum":
            return loss_map.sum()
        else:  # "none"
            return loss_map


class IndexStructureLoss(nn.Module):
    """
    Combined loss for NDVI / vegetation indices:

        total = alpha * L1(pred, target)
              + beta  * MultiScaleGradientLoss(pred, target)
              + gamma * SSIMLoss(pred, target)

    Works for any number of channels (C = number of indices or 1 for NDVI).

    This is a "perceptual-like" structural loss specialized for
    remote sensing indices (no VGG, no ImageNet assumption).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.0,
        num_scales: int = 3,
        window_size: int = 11,
        sigma: float = 1.5,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.l1 = nn.L1Loss(reduction=reduction)
        self.ms_grad = MultiScaleGradientLoss(num_scales=num_scales, reduction=reduction)
        self.ssim = SSIMLoss(
            window_size=window_size,
            sigma=sigma,
            channels=1,        # will adapt automatically to input channels
            reduction=reduction,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")

        l1_val = self.l1(pred, target)
        grad_val = self.ms_grad(pred, target)
        ssim_val = self.ssim(pred, target)

        return self.alpha * l1_val + self.beta * grad_val + self.gamma * ssim_val

