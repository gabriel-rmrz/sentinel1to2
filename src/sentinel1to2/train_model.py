import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error


def train_model(
    model,
    device,
    config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,              # reconstruction / structural loss (CombinedLoss, IndexStructureLoss, etc.)
    optimizer_G,
    discriminator=None,     # PatchGAN or None
    optimizer_D=None,       # optimizer for discriminator or None
    epochs: int = 100,
    patience: int = 5,
):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s]: %(message)s",
    )

    job_dir = Path(config["job"]["dir"])
    job_data_dir = job_dir / "data"
    job_data_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = np.inf
    no_improve = 0
    train_losses = []
    val_losses = []

    gan_cfg = config["training"].get("gan", {})
    gan_mode = gan_cfg.get("mode", "none")
    use_gan = gan_mode == "pix2pix" and discriminator is not None and optimizer_D is not None

    if use_gan:
        adv_criterion = nn.BCEWithLogitsLoss()
        lambda_recon = float(gan_cfg.get("lambda_recon", 100.0))
        lambda_adv = float(gan_cfg.get("lambda_adv", 1.0))
        logging.info(f"Using pix2pix mode with λ_recon={lambda_recon}, λ_adv={lambda_adv}")
    else:
        adv_criterion = None
        lambda_recon = 1.0
        lambda_adv = 0.0
        logging.info("Using pure supervised mode (no GAN).")

    for epoch in range(epochs):
        logging.info(f"Epoch {epoch+1}/{epochs}")
        # ===================== TRAIN =====================
        model.train()
        if use_gan:
            discriminator.train()

        epoch_train_losses = []

        for inputs, targets, _scene, _patch_idx in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            if not use_gan:
                # ----- Supervised UNet only -----
                optimizer_G.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer_G.step()
                epoch_train_losses.append(loss.item())

            else:
                # ----- 1) Train D -----
                optimizer_D.zero_grad()

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

                # ----- 2) Train G -----
                optimizer_G.zero_grad()
                fake = model(inputs)

                fake_pair = torch.cat([inputs, fake], dim=1)
                pred_fake = discriminator(fake_pair)

                # adversarial term: want D(x, G(x)) ≈ 1
                adv_loss_G = adv_criterion(pred_fake, torch.ones_like(pred_fake))
                # reconstruction / structural loss (your criterion)
                recon_loss = criterion(fake, targets)

                loss_G = lambda_adv * adv_loss_G + lambda_recon * recon_loss
                loss_G.backward()
                optimizer_G.step()

                epoch_train_losses.append(loss_G.item())

        train_loss = float(np.mean(epoch_train_losses))
        train_losses.append(train_loss)

        # ===================== VALIDATION =====================
        model.eval()
        val_epoch_losses = []

        with torch.no_grad():

            for inputs, targets, _scene, _patch_idx in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)

                # usually we only care about reconstruction loss on val
                val_loss = criterion(outputs, targets)
                val_epoch_losses.append(val_loss.item())

        val_loss = float(np.mean(val_epoch_losses))
        val_losses.append(val_loss)

        # ===================== EARLY STOPPING & SAVE =====================
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), job_data_dir / config["training"]["model_output"])
        else:
            no_improve += 1

        logging.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if no_improve >= patience:
            logging.info(f"Early stopping at epoch {epoch+1}")
            return train_losses, val_losses

    return train_losses, val_losses

