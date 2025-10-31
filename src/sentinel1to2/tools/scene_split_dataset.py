import h5py
import torch
from torch.utils.data import Dataset

class scene_split_dataset(Dataset):
  def __init__(self, hdf5_path):
    self.hdf5_path = hdf5_path
    with h5py.File(hdf5_path, 'r') as hf:
      # Leggi la lista delle scene dai metadati
      self.scene_list = [s.decode() for s in hf['metadata/scene_list'][:]]
      
      # Crea una mappa degli indici
      self.index_map = []
      for scene in self.scene_list:
        scene_group = hf.get(f'scene_{scene}')
        if scene_group is None:
          print(f"Warning: Scene {scene} non trovata, salto.")
          continue
        if 'inputs' not in scene_group:
          print(f"Warning: Scene {scene} senza 'inputs', salto.")
          continue
        num_patches = scene_group['inputs'].shape[0]
        self.index_map.extend([(scene, i) for i in range(num_patches)])

  def __len__(self):
    return len(self.index_map)

  def __getitem__(self, idx):
    scene, patch_idx = self.index_map[idx]
    with h5py.File(self.hdf5_path, 'r') as hf:
      inputs = torch.from_numpy(hf[f'scene_{scene}/inputs'][patch_idx])
      targets = torch.from_numpy(hf[f'scene_{scene}/targets'][patch_idx])
    return inputs, targets
