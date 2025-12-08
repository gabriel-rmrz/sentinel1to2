import random
import csv
import torch
import torch.nn as nn
import yaml
import argparse
from pathlib import Path
import logging

from torch.utils.data import DataLoader, random_split
from .prepare_input_data import prepare_input_data
from .train_model import train_model
from .evaluate_model import evaluate_model
from .tools.scene_split_dataset import scene_split_dataset
from .tools.get_steps import get_steps
from .tools.get_model import get_model
from .tools.parse_args import parse_args
from .models.losses.CombinedLoss import CombinedLoss
from .models.losses.VGGPerceptualLoss import VGGPerceptualLoss
from .batch_run_inference import batch_run_inference
import torch.nn as nn


from .models.patchgan import PatchGANDiscriminator
from .models.losses.index_structure_losses import IndexStructureLoss
from .models.losses.CombinedLoss import CombinedLoss
from .models.losses.VGGPerceptualLoss import VGGPerceptualLoss
from .models.losses.sam_loss import sam_loss  # add this import

from .performance import performance
def get_loss(config) -> nn.Module:
    loss_cfg = config.get("training", {}).get("loss", {})
    name = loss_cfg.get("name", "L1Loss")
    params = dict(loss_cfg.get("parameters", {}) or {})

    target_type = config["target"]["type"]
    name_lower = str(name).lower()

    if target_type == "bands" and name_lower in ("combinedloss", "combined"):
        return CombinedLoss(target_type="bands", **params)

    if target_type == "indices" and name_lower in ("combinedloss", "combined", "indexstructureloss", "index_structure"):
        # Redirigimos automáticamente CombinedLoss → IndexStructureLoss
        return IndexStructureLoss(**params)

    # ----------------- Pérdidas estándar de PyTorch ------------

    if name_lower in ("l1", "l1loss", "mae"):
        return nn.L1Loss(**params)

    if name_lower in ("mse", "mseloss"):
        return nn.MSELoss(**params)

    if name_lower in ("smoothl1", "huber"):
        return nn.SmoothL1Loss(**params)

    raise ValueError(f"Unknown loss '{name}' in config['training']['loss']['name']")
'''
def get_loss(config):
  if config['target']['type'] == "bands":
    loss = CombinedLoss(**config["training"]["loss"]["parameters"])
  elif  config['target']['type'] == 'indices':
    loss = nn.L1Loss()
  return loss
'''


def check_step_requirements():
  pass

def write_list_to_csv(path, list_out):
  with open(path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows([[x] for x in list_out])
  return

def write_two_lists_to_csv(path, list1, list2):
  with open(path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["train_losses", "val_losses"])   # optional header
    for a, b in zip(list1, list2):
      writer.writerow([a, b])
  return

def main() -> None:
  # Configurazioni
  logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s]: %(message)s",
  )
  args = parse_args(argparse)
  steps = get_steps(args)
  with open(args.config, 'r') as file:
    config = yaml.safe_load(file)
  job_dir = Path(config["job"]["dir"])
  job_data_dir = job_dir / 'data' 
  job_outputs_dir = job_dir / 'outputs' 
  
  job_data_dir.mkdir(parents=True, exist_ok=True)
  job_outputs_dir.mkdir(parents=True, exist_ok=True)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  logging.info(f"Carrying out the following steps:")
  for st, i_st in steps.items():
    print(f"\t- {i_st}. {st}")
  if "preprocessing" in steps.keys():
    logging.info(f"Step {steps['preprocessing']}: preprocessing.")
    prepare_input_data(config)
    if not args.all_steps:
      return

  # Caricamento dataset


  model = get_model(config).to(device)         # this is your UNet (generator)
  criterion = get_loss(config)                 # reconstruction / structural loss
  
  optimizer_G = torch.optim.Adam(
      model.parameters(),
      lr=float(config["training"]["optimizer"]["parameters"]["lr"]),
      betas=(0.5, 0.999),
  )
  
  gan_cfg = config["training"].get("gan", {})
  gan_mode = gan_cfg.get("mode", "none")
  
  discriminator = None
  optimizer_D = None
  
  target_type = config["target"]["type"]

  if target_type == "bands":
    out_ch = len(config["target"]["selected_bands"])
  elif target_type == "indices":
    out_ch = len(config["target"]["selected_indices"])

  if gan_mode == "pix2pix":
      in_ch = config["model"]["parameters"]["in_channels"]
      out_ch = out_ch 
      discriminator = PatchGANDiscriminator(in_channels=in_ch + out_ch).to(device)
      optimizer_D = torch.optim.Adam(
          discriminator.parameters(),
          lr=float(config["training"]["optimizer"]["parameters"]["lr"]),
          #lr=config["training"]["learning_rate"],
          betas=(0.5, 0.999),
      )
  















  
  logging.info(f"Making use of model {config['model']['name']}")

  # TODO: Add test_loader 
  # TODO: Put the 3 data_loaders in a function 
  # DataLoaders
  val_loader, train_loader = None, None
  if "training" in steps.keys() or "evaluation" in steps.keys(): 
    logging.info(f"Loading validation data")
    val_dataset = scene_split_dataset(job_data_dir / config["training"]["data"]["val_dataset"])
    val_loader = DataLoader(
      val_dataset,
      batch_size=config["training"]["data"]["batch_size"],
      shuffle=True,
      num_workers=config["training"]["data"]["n_workers"],
      persistent_workers=False,
      pin_memory= torch.cuda.is_available()
    )

  if 'training' in steps.keys():
    logging.info(f"Loading training data")
    train_dataset = scene_split_dataset(job_data_dir / config["training"]["data"]["train_dataset"])
    train_loader = DataLoader(
      train_dataset, batch_size=config["training"]["data"]["batch_size"],
      shuffle=True,
      num_workers=config["training"]["data"]["n_workers"],
      pin_memory=torch.cuda.is_available()
    )
    #print( sum(p.numel() for p in model.parameters() if p.requires_grad) )

    #optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["optimizer"]["parameters"]["lr"]))
    #criterion = nn.L1Loss() # nn.MSELoss()  # Per regressione

    
    # Addestramento
    # TODO: epochs and patience are not necessary if we have the cofig (?)
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
      epochs=config["training"]["n_epochs"],
      patience=config["training"]["patience"],
    )
    output_train_losses_path = job_outputs_dir / "train_losses.csv"
    output_val_losses_path = job_outputs_dir / "val_losses.csv"
    logging.info(f"Writing metrics to {output_train_losses_path}")
    write_list_to_csv(output_train_losses_path, train_losses)
    logging.info(f"Writing metrics to {output_val_losses_path}")
    write_list_to_csv(output_val_losses_path, val_losses)
    logging.info(f"Training step finished")
  if "inference" in steps.keys():
    # TODO: use the cofig as parameter instead of the model_path, data_dir
    batch_run_inference(
      config,
      device=device,
      sample_type='val' # use 'val' to load validation scene list
    )
    batch_run_inference(
      config,
      device=device
    )
    logging.info(f"Inference step finished")
  if "evaluation" in steps.keys():
    if not "training" in steps.keys():
      model.load_state_dict(torch.load(job_data_dir / config["training"]["model_output"], map_location=device))
    evaluate_model(model, config, device, val_loader, num_samples= 1000000)
    logging.info(f"Evaluation finished")
  if "performance" in steps.keys():
    performance(config, 'val')
    #performance(config, 'test')
    logging.info(f"Performance step finished")
  
  return

  


if __name__=='__main__':
  main()
