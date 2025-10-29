#from .models.losses.sam_loss import sam_loss
from .sam_loss import sam_loss
from torchvision.models import vgg16, VGG16_Weights
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGPerceptualLoss(nn.Module):
  def __init__(self):
    super().__init__()
    # Usa weights espliciti per evitare warning
    weights = VGG16_Weights.DEFAULT
    vgg = vgg16(weights=weights).features[:16].eval()
    for param in vgg.parameters():
      param.requires_grad = False
    self.vgg = vgg
    self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
    self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

  def forward(self, pred, target):
    # Sposta mean e std sul device degli input
    mean = self.mean.to(pred.device)
    std = self.std.to(pred.device)
  
    # Normalizzazione
    pred = (pred - mean) / std
    target = (target - mean) / std
  
    # Sposta anche la VGG sul device corretto (se non già fatto)
    self.vgg = self.vgg.to(pred.device)
  
    # Calcolo feature e perdita
    pred_feat = self.vgg(pred)
    target_feat = self.vgg(target)
    return F.l1_loss(pred_feat, target_feat)
