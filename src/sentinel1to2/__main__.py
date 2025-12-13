import argparse
import logging
from pathlib import Path

import torch
import yaml

from .prepare_input_data import prepare_input_data
from .train_model import train_model
from .evaluate_model import evaluate_model
from .batch_run_inference import batch_run_inference
from .performance import performance

from .tools.get_steps import get_steps
from .tools.get_model import get_model
from .tools.parse_args import parse_args

from .models.patchgan import PatchGANDiscriminator

# NEW: split-config + run/dataset path resolving
from .tools.config_utils import load_yaml, deep_merge, resolve_paths, save_yaml
from .tools.loss_factory import get_loss  # (recommended) move loss factory out of __main__


def _ensure_dirs(config: dict) -> None:
    """Create main folders for dataset cache and run outputs."""
    dataset_cache_dir = Path(config["paths"]["dataset_cache_dir"])
    run_dir = Path(config["paths"]["run_dir"])

    # Dataset cache
    (dataset_cache_dir / "h5").mkdir(parents=True, exist_ok=True)
    (dataset_cache_dir / "lists").mkdir(parents=True, exist_ok=True)
    (dataset_cache_dir / "norm").mkdir(parents=True, exist_ok=True)

    # Run outputs
    (run_dir / "config").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (run_dir / "inference").mkdir(parents=True, exist_ok=True)


def _load_and_resolve_config(args) -> dict:
    """Load config from legacy -c or split dataset/experiment configs, then resolve paths."""
    if getattr(args, "config", None) is not None and args.config is not None:
        config = load_yaml(Path(args.config))
    else:
        ds_cfg = getattr(args, "dataset_config", None)
        ex_cfg = getattr(args, "experiment_config", None)
        if ds_cfg is None or ex_cfg is None:
            raise ValueError(
                "Provide either -c CONFIG.yaml (legacy) OR both --dataset-config and --experiment-config."
            )
        config = deep_merge(load_yaml(Path(ds_cfg)), load_yaml(Path(ex_cfg)))

    config = resolve_paths(config)
    _ensure_dirs(config)

    # snapshot resolved config (run-level, always)
    run_dir = Path(config["paths"]["run_dir"])
    save_yaml(config, run_dir / "config" / "resolved.yaml")

    return config


def _setup_device(config: dict) -> torch.device:
    prefer_gpu = config.get("runtime", {}).get("prefer_gpu", True)
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_gan_components(config: dict, device: torch.device):
    """
    Build optional pix2pix discriminator and its optimizer.
    Returns: (discriminator, optimizer_D)
    """
    gan_cfg = config.get("training", {}).get("gan", {})
    gan_mode = gan_cfg.get("mode", "none")

    if gan_mode != "pix2pix":
        return None, None

    in_ch = int(config["model"]["parameters"]["in_channels"])
    target_type = config["target"]["type"]
    if target_type == "bands":
        out_ch = len(config["target"]["selected_bands"])
    elif target_type == "indices":
        out_ch = len(config["target"]["selected_indices"])
    else:
        raise ValueError(f"Unknown target.type: {target_type}")

    discriminator = PatchGANDiscriminator(in_channels=in_ch + out_ch).to(device)

    lr = float(config["training"]["optimizer"]["parameters"]["lr"])
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    return discriminator, optimizer_D


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")

    args = parse_args(argparse)
    steps = get_steps(args)

    config = _load_and_resolve_config(args)
    device = _setup_device(config)

    run_dir = Path(config["paths"]["run_dir"])
    dataset_cache_dir = Path(config["paths"]["dataset_cache_dir"])

    logging.info("Resolved paths:")
    logging.info(f"  dataset_cache_dir: {dataset_cache_dir}")
    logging.info(f"  run_dir          : {run_dir}")
    logging.info(f"  device           : {device}")

    logging.info("Carrying out the following steps:")
    for st, i_st in steps.items():
        print(f"\t- {i_st}. {st}")

    # -------------------------
    # 1) Preprocessing (dataset cache)
    # -------------------------
    if "preprocessing" in steps:
        logging.info(f"Step {steps['preprocessing']}: preprocessing")
        prepare_input_data(config)  # should write to dataset_cache_dir/*
        if not args.all_steps:
            return

    # -------------------------
    # 2) Build model + loss + optimizers
    # -------------------------
    model = get_model(config).to(device)
    criterion = get_loss(config)

    lr = float(config["training"]["optimizer"]["parameters"]["lr"])
    optimizer_G = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))

    discriminator, optimizer_D = _build_gan_components(config, device)
    gan_mode = config.get("training", {}).get("gan", {}).get("mode", "none")

    logging.info(f"Model: {config['model']['name']}")
    logging.info(f"GAN mode: {gan_mode}")

    # -------------------------
    # 3) Training (run outputs)
    # -------------------------
    if "training" in steps:
        logging.info(f"Step {steps['training']}: training")
        train_losses, val_losses = train_model(
            model=model,
            device=device,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer_G=optimizer_G,
            discriminator=discriminator,
            optimizer_D=optimizer_D,
            epochs=int(config["training"]["n_epochs"]),
            patience=int(config["training"]["patience"]),
        )
        logging.info("Training step finished")

    # -------------------------
    # 4) Inference (run outputs)
    # -------------------------
    if "inference" in steps:
        logging.info(f"Step {steps['inference']}: inference")
        batch_run_inference(config, device=device, sample_type="val")
        batch_run_inference(config, device=device, sample_type="test")
        logging.info("Inference step finished")

    # -------------------------
    # 5) Evaluation (run outputs)
    # -------------------------
    if "evaluation" in steps:
        logging.info(f"Step {steps['evaluation']}: evaluation")
        evaluate_model(model, config, device)
        logging.info("Evaluation finished")

    # -------------------------
    # 6) Performance (run outputs)
    # -------------------------
    if "performance" in steps:
        logging.info(f"Step {steps['performance']}: performance")
        performance(config, "val")
        logging.info("Performance step finished")


if __name__ == "__main__":
    main()

