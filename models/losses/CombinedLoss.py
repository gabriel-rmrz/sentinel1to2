from models.losses.sam_loss import sam_loss
from models.losses.VGGPerceptualLoss import VGGPerceptualLoss
from torchvision.models import vgg16, VGG16_Weights
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
  def __init__(self, alpha=1.0, beta=0.1, gamma=0.01):
    super().__init__()
    self.l1 = nn.L1Loss()
    self.sam = sam_loss
    self.vgg = VGGPerceptualLoss()
    self.alpha = alpha  # peso L1
    self.beta = beta    # peso SAM
    self.gamma = gamma  # peso VGG

  def forward(self, pred, target):
    l1 = self.l1(pred, target)
    sam = self.sam(pred, target)
    # Prendi solo 3 bande RGB per VGG (es. bande 3,2,1 per Sentinel-2 RGB)
    pred_rgb = pred[:, [6, 2, 1], :, :]  # es. B4-B3-B2
    target_rgb = target[:, [6, 2, 1], :, :]
    vgg = self.vgg(pred_rgb, target_rgb)
    return self.alpha * l1 + self.beta * sam + self.gamma * vgg
