from pathlib import Path
import torch
import numpy as np
import segmentation_models_pytorch as smp

from .tools.infer_on_scene import infer_on_scene
from .tools.get_model import get_model
from .tools.save_geotiff import save_geotiff
from .tools.load_and_stack_full import load_and_stack_full

def inference(config, scene_folder, device='cuda', prefix='test'):
  job_dir = Path(config["job"]["dir"])
  job_data_dir = job_dir / 'data'
  model_path = job_data_dir / config['training']['model_output']
  data_dir = config['preprocessing']['input_dir']
  output_dir = job_data_dir / config['inference']['output_dir']
  params_file = job_data_dir / config['preprocessing']['norm_params_file']
  # === Normalizzazione ===
  norm_params = np.load(params_file)
  MEAN = norm_params["mean"]
  STD = norm_params["std"]
  device = torch.device(device if torch.cuda.is_available() else 'cpu')
  print(f"Using device: {device}")

  #model = smp.Unet(encoder_name="efficientnet-b0", in_channels=4, classes=9)
  model = get_model(config)
  model.load_state_dict(torch.load(model_path, map_location=device))
  model.to(device)

  # Carica stack input
  dsm, s1, wc, _s2_selected, _ind, profile = load_and_stack_full(scene_folder, data_dir, MEAN, STD)
  input_stack = np.concatenate([dsm,s1,wc], axis=0)
  print(f"Input shape: {input_stack.shape}")

  # Inference
  output = infer_on_scene(model, input_stack, device)

  # Salva TIFF
  output_dir.mkdir(parents=True, exist_ok=True)
  out_path = output_dir / f"{prefix}_{scene_folder}_pred.tif"
  save_geotiff(output, profile, out_path)
  print(f"✅ Output salvato in: {out_path}")
