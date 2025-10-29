import torch
import torch.nn.functional as F

def sam_loss(pred, target, eps=1e-8):
  """
  pred, target: shape (B, C, H, W)
  """
  B, C, H, W = pred.shape
  pred_flat = pred.view(B, C, -1) # (B, C, N)
  target_flat = target.view(B, C, -1)

  dot = torch.sum(pred_flat * target_flat, dim=1) # (B, N)
  norm_pred = torch.norm(pred_flat, dim=1)
  norm_target = torch.norm(target_flat, dim=1)

  cos = dot / (norm_pred * norm_target + eps)  # (B, N)
  angle = torch.acos(torch.clamp(cos, -1.0, 1.0))  # (B, N)
  return torch.mean(angle)
