import os
import h5py
import numpy as np
from tools.load_and_stack_full import load_and_stack_full
def process_scene(folder, data_dir, hdf5_file, crop_size=128, stride=128, mean = None, std = None):
  try:
    dsm, s1, worldcover, indices, _profile = load_and_stack_full(folder, data_dir, mean, std)

    input_patches = []
    target_patches = []

    H, W = indices.shape[1], indices.shape[2]
    for top in range(0, H - crop_size + 1, stride):
      for left in range(0, W - crop_size + 1, stride):
        input_patch = np.concatenate([
          dsm[:, top:top+crop_size, left:left+crop_size],
          s1[:, top:top+crop_size, left:left+crop_size],
          worldcover[:, top:top+crop_size, left:left+crop_size]
        ], axis=0)
        target_patch = indices[:, top:top+crop_size, left:left+crop_size]  # Es. NDVI come target

        if np.isnan(input_patch).any() or np.isnan(target_patch).any():
          continue

        input_patches.append(input_patch)
        target_patches.append(target_patch)

    if len(input_patches) == 0:
      print(f"[WARN] Nessuna patch valida per {folder}, scena ignorata.")
      return folder, 0

    grp = hdf5_file.create_group(f"scene_{folder}")
    grp.create_dataset("inputs", data=np.stack(input_patches), dtype=np.float32)
    grp.create_dataset("targets", data=np.stack(target_patches), dtype=np.float32)

    return folder, len(input_patches)

  except Exception as e:
      print(f"Error processing {folder}: {str(e)}")
      return folder, 0
