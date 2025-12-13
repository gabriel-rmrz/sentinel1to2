from __future__ import annotations

import torch.nn as nn
from typing import Dict, Any

from ..models.losses.CombinedLoss import CombinedLoss
from ..models.losses.index_structure_losses import IndexStructureLoss
from ..models.losses.VGGPerceptualLoss import VGGPerceptualLoss
from ..models.losses.sam_loss import SAMLoss  # assuming you have a class wrapper


def get_loss(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create the training loss from config.

    Supported losses (case-insensitive):
      - L1Loss, MAE
      - MSELoss
      - SmoothL1Loss / Huber
      - CombinedLoss        (bands only)
      - IndexStructureLoss  (indices only)
      - VGGPerceptualLoss   (bands only)
      - SAM                 (bands only)

    The factory automatically:
      - validates target type compatibility
      - redirects CombinedLoss → IndexStructureLoss for indices
      - passes all parameters from config["training"]["loss"]["parameters"]

    Returns
    -------
    torch.nn.Module
        Instantiated loss module.
    """
    loss_cfg = config.get("training", {}).get("loss", {})
    name = loss_cfg.get("name", "L1Loss")
    params = dict(loss_cfg.get("parameters", {}) or {})

    target_type = config["target"]["type"].lower()
    name_lower = str(name).lower()

    # -------------------------------------------------
    # Index targets: force structural index loss
    # -------------------------------------------------
    if target_type == "indices":
        if name_lower in (
            "indexstructureloss",
            "index_structure",
            "combinedloss",
            "combined",
        ):
            return IndexStructureLoss(**params)

        if name_lower in ("l1", "l1loss", "mae"):
            return nn.L1Loss(**params)

        if name_lower in ("mse", "mseloss"):
            return nn.MSELoss(**params)

        raise ValueError(
            f"Loss '{name}' is not compatible with target.type='indices'. "
            f"Use IndexStructureLoss or L1/MSE."
        )

    # -------------------------------------------------
    # Band targets
    # -------------------------------------------------
    if target_type == "bands":
        if name_lower in ("combinedloss", "combined"):
            return CombinedLoss(target_type="bands", **params)

        if name_lower in ("vggperceptualloss", "vgg"):
            return VGGPerceptualLoss()

        if name_lower in ("sam", "samloss"):
            return SAMLoss(**params)

        if name_lower in ("l1", "l1loss", "mae"):
            return nn.L1Loss(**params)

        if name_lower in ("mse", "mseloss"):
            return nn.MSELoss(**params)

        if name_lower in ("smoothl1", "huber"):
            return nn.SmoothL1Loss(**params)

        raise ValueError(
            f"Unknown loss '{name}' for target.type='bands'."
        )

    # -------------------------------------------------
    # Safety net
    # -------------------------------------------------
    raise ValueError(f"Unknown target.type '{target_type}' in config.")

