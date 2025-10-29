import os
from .inference import inference

def batch_run_inference(model_path, data_dir, output_dir, device='cuda'):
  scene_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
  for scene_folder in scene_folders:
    print(f"\n🔍 Inference su scena: {scene_folder}")
    inference(
      scene_folder=scene_folder,
      model_path=model_path,
      data_dir=data_dir,
      output_dir=output_dir,
      device=device
    )
