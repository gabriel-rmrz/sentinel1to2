from __future__ import annotations

import logging
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _write_list_to_csv(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["value"])
        for v in values:
            w.writerow([v])


def _save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def train_model(
    model: torch.nn.Module,
    device: torch.device,
    config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,              # reconstruction / structural loss (CombinedLoss, IndexStructureLoss, etc.)
    optimizer_G: torch.optim.Optimizer,
    discriminator: torch.nn.Module | None = None,  # PatchGAN or None
    optimizer_D: torch.optim.Optimizer | None = None,
    epochs: int = 100,
    patience: int = 5,
):
    """
    Training loop for:
      - supervised mode (no GAN)
      - pix2pix mode (Generator + PatchGAN Discriminator)

    All outputs are written under:
      config["paths"]["run_dir"]/
        checkpoints/
        metrics/
    """
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
    logger = logging.getLogger(__name__)

    # -----------------------------
    # Output paths (run_dir)
    # -----------------------------
    if "paths" not in config or "run_dir" not in config["paths"]:
        raise KeyError(
            "config['paths']['run_dir'] not found. Make sure __main__.py resolves paths via resolve_paths(config)."
        )

    run_dir = Path(config["paths"]["run_dir"])
    ckpt_dir = run_dir / "checkpoints"
    metrics_dir = run_dir / "metrics"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = Path(config.get("training", {}).get("checkpoint_best", ckpt_dir / "best.pth"))
    last_ckpt_path = Path(config.get("training", {}).get("checkpoint_last", ckpt_dir / "last.pth"))

    # optional: save every N epochs
    save_every = int(config.get("training", {}).get("save_every", 0))  # 0 disables

    # -----------------------------
    # GAN configuration
    # -----------------------------
    gan_cfg = config.get("training", {}).get("gan", {})
    gan_mode = gan_cfg.get("mode", "none")
    use_gan = gan_mode == "pix2pix" and discriminator is not None and optimizer_D is not None

    if use_gan:
        adv_criterion = nn.BCEWithLogitsLoss()
        lambda_recon = float(gan_cfg.get("lambda_recon", 100.0))
        lambda_adv = float(gan_cfg.get("lambda_adv", 1.0))
        logger.info(f"Using pix2pix mode with λ_recon={lambda_recon}, λ_adv={lambda_adv}")
    else:
        adv_criterion = None
        lambda_recon = 1.0
        lambda_adv = 0.0
        logger.info("Using pure supervised mode (no GAN).")

    # -----------------------------
    # Training loop
    # -----------------------------
    best_val_loss = float("inf")
    no_improve = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")

        # ===== TRAIN =====
        model.train()
        if use_gan:
            discriminator.train()

        epoch_train_losses: list[float] = []

        for inputs, targets, _scene, _patch_idx in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if not use_gan:
                optimizer_G.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer_G.step()
                epoch_train_losses.append(float(loss.item()))
                continue

            # -------------------------
            # pix2pix: Train Discriminator
            # -------------------------
            optimizer_D.zero_grad(set_to_none=True)

            with torch.no_grad():
                fake = model(inputs)

            real_pair = torch.cat([inputs, targets], dim=1)
            fake_pair = torch.cat([inputs, fake], dim=1)

            pred_real = discriminator(real_pair)
            pred_fake = discriminator(fake_pair)

            real_labels = torch.ones_like(pred_real)
            fake_labels = torch.zeros_like(pred_fake)

            loss_D_real = adv_criterion(pred_real, real_labels)
            loss_D_fake = adv_criterion(pred_fake, fake_labels)
            loss_D = 0.5 * (loss_D_real + loss_D_fake)

            loss_D.backward()
            optimizer_D.step()

            # -------------------------
            # pix2pix: Train Generator
            # -------------------------
            optimizer_G.zero_grad(set_to_none=True)
            fake = model(inputs)

            fake_pair = torch.cat([inputs, fake], dim=1)
            pred_fake = discriminator(fake_pair)

            adv_loss_G = adv_criterion(pred_fake, torch.ones_like(pred_fake))
            recon_loss = criterion(fake, targets)

            loss_G = lambda_adv * adv_loss_G + lambda_recon * recon_loss
            loss_G.backward()
            optimizer_G.step()

            epoch_train_losses.append(float(loss_G.item()))

        train_loss = float(np.mean(epoch_train_losses)) if epoch_train_losses else float("nan")
        train_losses.append(train_loss)

        # ===== VALIDATION =====
        model.eval()
        val_epoch_losses: list[float] = []

        with torch.no_grad():
            for inputs, targets, _scene, _patch_idx in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                outputs = model(inputs)
                val_loss = criterion(outputs, targets)
                val_epoch_losses.append(float(val_loss.item()))

        val_loss = float(np.mean(val_epoch_losses)) if val_epoch_losses else float("nan")
        val_losses.append(val_loss)

        # ===== CHECKPOINTING =====
        # Always save last
        _save_checkpoint(model, last_ckpt_path)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            _save_checkpoint(model, best_ckpt_path)
            logger.info(f"✔ New best checkpoint saved: {best_ckpt_path}")
        else:
            no_improve += 1

        # Optional periodic checkpoint
        if save_every > 0 and (epoch + 1) % save_every == 0:
            epoch_ckpt_path = ckpt_dir / f"epoch_{epoch+1:03d}.pth"
            _save_checkpoint(model, epoch_ckpt_path)

        # Write metrics each epoch (safe for Condor preemption)
        _write_list_to_csv(metrics_dir / "train_losses.csv", train_losses)
        _write_list_to_csv(metrics_dir / "val_losses.csv", val_losses)

        logger.info(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # ===== EARLY STOPPING =====
        if no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1} (patience={patience})")
            break

    return train_losses, val_losses

