import logging
from pathlib import Path
import torch
import itertools
from .inference import inference

def batch_run_inference(config, device='cuda',prefix='test'):
  logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s]: %(message)s",
  )
  job_dir = Path(config["job"]["dir"])
  job_data_dir = job_dir / 'data'
  data_dir = config['preprocessing']['input_dir']
  # Get list (if we don't want to run over all the files in the val/test directory)
  all_scenes = sorted([f.name for f in Path(data_dir).iterdir() if f.is_dir()])
  if prefix == 'val':
    with open(job_data_dir / 'lists/validation_scene_list.csv', 'r') as file:
      all_scenes = [scene.strip() for scene in file.readlines()]
      #all_scenes = list(file)
  sample_size = config['inference']['sample_size']

  if sample_size == 0 or sample_size > len(all_scenes):
    sample_scenes = all_scenes
    sample_size = len(all_scenes)
  else:
    sample_scenes = all_scenes[:sample_size]
  logging.info(f"Sampling {sample_size} out of {len(all_scenes)} scenes available in {data_dir}")
  '''
  if loader != None:
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
  '''
  for scene_folder in sample_scenes:
    print(f"\n🔍 Inference su scena: {scene_folder}")
    inference(
      config,
      scene_folder=scene_folder,
      device=device,
      prefix=prefix
    )
