import os
import torch
import itertools
from .inference import inference

def batch_run_inference(model_path, data_dir, output_dir, loader= None, device='cuda',prefix='test'):
  # Get list (if we don't want to run over all the files in the val/test directory)
  scene_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
  print(scene_folders)
  if loader != None:
    print("here")
    with torch.no_grad():
      chosen_scenes = []
      for inputs, targets, scenes, patch_idx in loader:
        chosen_scenes.append(scenes)
      chosen_scenes = list(set(itertools.chain(*chosen_scenes)))
      scene_folders_tmp = []
      for sf in scene_folders:
        for cs in chosen_scenes:
          if cs in sf:
            scene_folders_tmp.append(sf)
      scene_folders = scene_folders_tmp
  for scene_folder in scene_folders:
    print(f"\n🔍 Inference su scena: {scene_folder}")
    inference(
      scene_folder=scene_folder,
      model_path=model_path,
      data_dir=data_dir,
      output_dir=output_dir,
      device=device,
      prefix=prefix
    )
