from __future__ import annotations

from typing import Any, Dict
import torch.nn as nn

import segmentation_models_pytorch as smp


def _out_channels_from_target(config: Dict[str, Any]) -> int:
    target = config.get("target", {})
    ttype = str(target.get("type", "")).lower()

    if ttype == "bands":
        bands = target.get("selected_bands", [])
        if not bands:
            raise ValueError("target.type='bands' but target.selected_bands is empty.")
        return len(bands)

    if ttype == "indices":
        inds = target.get("selected_indices", [])
        if not inds:
            raise ValueError("target.type='indices' but target.selected_indices is empty.")
        return len(inds)

    raise ValueError(f"Unknown target.type '{ttype}'. Expected 'bands' or 'indices'.")


def get_model(config: Dict[str, Any]) -> nn.Module:
    """
    Build the *generator* model.

    Supported model.name values (case-insensitive):
      - "SMP_UNet"  -> segmentation_models_pytorch.Unet
      - "UNet"      -> local models/UNet.py (custom)

    Notes
    -----
    - Output channels are always inferred from config["target"].
    - For SMP_UNet we set `classes=<out_channels>`.
    - For local UNet we set `out_channels=<out_channels>`.
    """
    model_cfg = config.get("model", {})
    name = str(model_cfg.get("name", "")).strip()
    params = dict(model_cfg.get("parameters", {}) or {})

    if not name:
        raise KeyError("Missing config['model']['name'].")

    out_channels = _out_channels_from_target(config)
    name_lower = name.lower()

    # ------------------------------------------------------------
    # SMP Unet (segmentation_models_pytorch)
    # ------------------------------------------------------------
    if name_lower in ("smp_unet", "smpunet", "smp-unet"):
        # avoid accidental mismatch if user hard-codes classes
        if "classes" in params and int(params["classes"]) != out_channels:
            raise ValueError(
                f"SMP_UNet got classes={params['classes']} but target implies out_channels={out_channels}."
            )
        params["classes"] = out_channels
        return smp.Unet(**params)

    # ------------------------------------------------------------
    # Local UNet (models/UNet.py)
    # ------------------------------------------------------------
    if name_lower in ("unet", "local_unet", "plain_unet"):
        from ..models.UNet import UNet  # your file is UNet.py

        # Your UNet signature: UNet(in_channels=6, out_channels=1, init_features=128)
        # We'll infer out_channels from target, and take in_channels/init_features from params.
        if "out_channels" in params and int(params["out_channels"]) != out_channels:
            raise ValueError(
                f"UNet got out_channels={params['out_channels']} but target implies out_channels={out_channels}."
            )
        params["out_channels"] = out_channels
        return UNet(**params)

    supported = ["SMP_UNet", "UNet"]
    raise ValueError(f"Unknown model '{name}'. Supported: {supported}.")

