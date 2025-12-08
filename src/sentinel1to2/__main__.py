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
#from .process_scenes import process_scenes
from .performance import performance

def get_loss(config):
  if config['target']['type'] == "bands":
    loss = CombinedLoss(**config["training"]["loss"]["parameters"])
  elif  config['target']['type'] == 'indices':
    loss = nn.L1Loss()
  return loss


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
  
  logging.info(f"Making use of model {config['model']['name']}")
  model = get_model(config)

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

  model = model.to(device)
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

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["optimizer"]["parameters"]["lr"]))
    #criterion = nn.L1Loss() # nn.MSELoss()  # Per regressione

    # TODO: Select the loss between different options
    criterion = get_loss(config)
    #criterion = CombinedLoss(alpha=1.0, beta=2, gamma=0.1)
    
    # Addestramento
    # TODO: epochs and patience are not necessary if we have the cofig (?)
    train_losses, val_losses = train_model(
      model,
      device,
      config,
      train_loader,
      val_loader,
      criterion,
      optimizer,
      epochs=config["training"]["n_epochs"],
      patience=config["training"]["patience"]
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
