import segmentation_models_pytorch as smp
import random
import torch
import yaml
import argparse
from pathlib import Path
import logging

from torch.utils.data import DataLoader, random_split
from .train_model import train_model
from .evaluate_model import evaluate_model
from .tools.scene_split_dataset import scene_split_dataset
from .models.losses.CombinedLoss import CombinedLoss
from .batch_run_inference import batch_run_inference
from .process_scenes import process_scenes
from .performance import performance

def check_step_requirements():
  pass

def get_steps(args):
  step_choices = ["preprocessing", "training", "evaluation", "inference", "performance"]
  steps = {} 
  if args.step not in step_choices:
    return {}
  if args.step == "preprocessing":
    steps[0] = "preprocessing"
    return steps

  if args.all_steps:
    steps[1] = "training"
    if args.step == "training":
      return steps
    else:
      steps[2] = "evaluation"
      if args.step == "evaluation":
        return steps
      else:
        steps[3] = "inference"
        if args.step == "inference":
          return steps
        else: 
          steps[4] = "performance"
          return steps
  elif args.step == "training":
    steps[1] = "training"
    return steps
  elif args.step == "evaluation":
    steps[1] = "evaluation"
    return steps
  elif args.step == "inference":
    steps[1] = "inference"
    return steps
  elif args.step == "performance": 
    steps[1] = "performance"
    return steps
  return {}

def parse_args(argparse):
  parser = argparse.ArgumentParser(description="Sentinel 1 to Sentinel 2 images translator")
  parser.add_argument(
      "-c",
      "--config",
      type = Path,
      default= "configs/default_config.yaml",
      help="YAML cofiguration file",
      )

  parser.add_argument(
      "-a",
      "--all_steps",
      type = bool,
      default= False,
      help="Set to True if you want to run all the scripts preceeding the selected step.",
      )

  parser.add_argument(
      "-s",
      "--step",
      choices=["preprocessing", "training", "evaluation", "inference", "performance"],
      required = True,
      help="The step to run: training, evaluation, inference or performance.",
      )

  return parser.parse_args()

def get_model(config):
  m_name = config["model"]["name"]
  if m_name == "SMP_UNet": 
     model = smp.Unet(**config["model"]["parameters"])
  '''
  model = smp.Unet(
    encoder_name="efficientnet-b0",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7                 # meno blocchi dell’encoder
    encoder_weights='imagenet',             # o None
    in_channels=4,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=9,                      # model output channels (number of classes in your dataset)
  #    decoder_channels=(128, 64, 32),
  #    encoder_depth=3

  )
  '''
  return model

  #model = UNet(in_channels=4, init_features=64, out_channels=1).to(device)
  #decoder_channels=(256, 128, 64, 32, 16), decoder_attention_type=None,
def main() -> None:
  print("running run")
  # TODO: Stop the script (step) if the propertly formated data is not available.
  args = parse_args(argparse)
  steps = get_steps(args)
  with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

  # Configurazioni
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  batch_size = 4
  learning_rate = 1e-4
  n_workers = 4
  
  # Caricamento dataset
  train_dataset = scene_split_dataset(config["training"]["train_dataset"])
  val_dataset = scene_split_dataset(config["training"]["val_dataset"])
  
  model = get_model(config)
  #print(model)
  # TODO: Add test_loader 
  # TODO: Put the 3 data_loaders in a function 
  # DataLoaders
  val_loader, train_loader = None, None
  if "training" in steps.values() or "evaluation" in steps.values():
    val_loader = DataLoader(
      val_dataset,
      batch_size=config["data_loader"]["batch_size"],
      shuffle=True,
      num_workers=config["data_loader"]["n_workers"],
      persistent_workers=False,
      pin_memory= torch.cuda.is_available()
    )

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  print(steps)
  if 'training' in steps.values():
    train_loader = DataLoader(
      train_dataset,
      batch_size=config["data_loader"]["batch_size"],
      shuffle=True,
      num_workers=config["data_loader"]["n_workers"],
      pin_memory=torch.cuda.is_available()
    )
    print(device)
    print( sum(p.numel() for p in model.parameters() if p.requires_grad) )
    #model = UNet(in_channels=4, init_features=32, out_channels=1).to(device)
    #model =  EffUNet(in_channels=6, classes=1)
    #model = CustomUNet(in_channels=6, init_features=32, depth=5, dropout_rate=0.05, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["optimizer"]["parameters"]["lr"]))
    #criterion = nn.L1Loss() # nn.MSELoss()  # Per regressione

    # TODO: Select the loss between different options
    criterion = CombinedLoss(**config["training"]["loss"]["parameters"])
    #criterion = CombinedLoss(alpha=1.0, beta=2, gamma=0.1)
    
    # Addestramento
    train_losses, val_losses = train_model(
      model,
      device,
      train_loader,
      val_loader,
      criterion,
      optimizer,
      epochs=50,
      patience=10
    )
    print(f"Training finished")
  if "evaluation" in steps.values():
    if not "training" in steps.values():
      model.load_state_dict(torch.load("best_model.pth"))
    evaluate_model(model, device, val_loader, num_samples= 100000)
  if "inference" in steps.values():
    batch_run_inference(
      model_path="best_model.pth",
      data_dir="data/inference/",
      output_dir="data/output_combined/",
      device="cpu"
    )
  if "performance" in steps.values():
    data_dir = "data/inference"
    pred_dir = "data/output_combined"
    results = process_scenes(data_dir, pred_dir)
    performance()
  
  return

  


if __name__=='__main__':
  main()
