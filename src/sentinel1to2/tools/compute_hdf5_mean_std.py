import os
import h5py
import numpy as np
def compute_hdf5_mean_std(hdf5_path):
  input_data = []
  with h5py.File(hdf5_path, 'r') as f:
    for scene in f:
      if scene == 'metadata':
        continue
      if 'inputs' not in f[scene]:
         print(f"[SKIP] {scene} non contiene 'inputs'")
         continue
      inputs = f[scene]["inputs"][:]
      input_data.append(inputs)

  if len(input_data) == 0:
    raise ValueError("Nessuna scena valida trovata per calcolare mean/std")

  input_data = np.concatenate(input_data, axis=0)  # [N, C, H, W]
  mean = input_data.mean(axis=(0, 2, 3))
  std = input_data.std(axis=(0, 2, 3))

  return mean, std
