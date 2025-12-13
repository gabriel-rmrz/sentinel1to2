import torch
import torch.nn as nn

class SAMLoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dot = (pred * target).sum(dim=1)
        norm_p = torch.norm(pred, dim=1)
        norm_t = torch.norm(target, dim=1)
        cos = dot / (norm_p * norm_t + self.eps)
        return torch.mean(torch.acos(torch.clamp(cos, -1.0, 1.0)))

