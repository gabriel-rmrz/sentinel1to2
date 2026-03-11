import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .prepare_input_data import prepare_input_data
from .train_model import train_model
from .evaluate_model import evaluate_model
from .batch_run_inference import batch_run_inference
from .performance import performance

from .tools.get_steps import get_steps
from .tools.get_model import get_model
from .tools.parse_args import parse_args
from .tools.scene_split_dataset import scene_split_dataset

from .models.patchgan import PatchGANDiscriminator

from .tools.config_utils import load_yaml, deep_merge, resolve_paths, save_yaml
from .tools.loss_factory import get_loss


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
    """
    Refactored-only mode:
      requires --dataset-config and --experiment-config
    """
    ds_cfg = getattr(args, "dataset_config", None)
    ex_cfg = getattr(args, "experiment_config", None)

    if ds_cfg is None or ex_cfg is None:
        raise ValueError("Refactored mode requires --dataset-config and --experiment-config.")

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
    target_type = str(config["target"]["type"]).lower()

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

def _build_train_val_loaders(config: dict, device: torch.device) -> tuple[DataLoader, DataLoader]:
    dataset_cache_dir = Path(config["paths"]["dataset_cache_dir"])
    train_h5 = dataset_cache_dir / "h5" / config["training"]["data"]["train_dataset"]
    val_h5 = dataset_cache_dir / "h5" / config["training"]["data"]["val_dataset"]

    train_ds = scene_split_dataset(train_h5)
    val_ds = scene_split_dataset(val_h5)

    bs = int(config["training"]["data"]["batch_size"])
    nw = int(config["training"]["data"]["n_workers"])
    pin = (device.type == "cuda")

    # More overlap between CPU loading and GPU compute
    prefetch = int(config["training"]["data"].get("prefetch_factor", 4))
    drop_last = bool(config["training"]["data"].get("drop_last", True))

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        #pin_memory=pin,
        pin_memory=False,
        persistent_workers=(nw > 0),
        prefetch_factor=prefetch if nw > 0 else None,
        drop_last=drop_last,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        #pin_memory=pin,
        pin_memory=False,
        persistent_workers=(nw > 0),
        prefetch_factor=prefetch if nw > 0 else None,
        drop_last=False,
    )
    return train_loader, val_loader




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
        prepare_input_data(config)
        if not args.all_steps:
            return

    # -------------------------
    # 2) Model + loss + optimizers
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
    # 2.5) DataLoaders
    # -------------------------
    train_loader = None
    val_loader = None
    if ("training" in steps) or ("evaluation" in steps):
        logging.info("Building DataLoaders from dataset cache")
        train_loader, val_loader = _build_train_val_loaders(config, device)

    # -------------------------
    # 3) Training
    # -------------------------
    if "training" in steps:
        logging.info(f"Step {steps['training']}: training")
        train_model(
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
    # 4) Inference
    # -------------------------
    if "inference" in steps:
        logging.info(f"Step {steps['inference']}: inference")
        #batch_run_inference(config, device=device, sample_type="val")
        batch_run_inference(config, device=device, sample_type="test")
        logging.info("Inference step finished")

    # -------------------------
    # 5) Evaluation
    # -------------------------
    if "evaluation" in steps:
        logging.info(f"Step {steps['evaluation']}: evaluation")
        num_samples = int(config.get("evaluation", {}).get("num_samples", 5))
        evaluate_model(model, config, device, val_loader, num_samples=num_samples)
        logging.info("Evaluation finished")

    # -------------------------
    # 6) Performance
    # -------------------------
    if "performance" in steps:
        logging.info(f"Step {steps['performance']}: performance")
        performance(config, "test")
        logging.info("Performance step finished")


if __name__ == "__main__":
    main()

