import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN discriminator as in pix2pix.

    Input is concatenation [source, target] or [source, generated]:
      - source:  (B, C_in,  H, W)
      - target:  (B, C_out, H, W)
      - D input: (B, C_in + C_out, H, W)
    """

    def __init__(self, in_channels: int, base_channels: int = 64):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 1, 1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_conv = nn.Conv2d(base_channels * 8, 1, 4, 1, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.out_conv(x)
        return x  # logits (no sigmoid)

