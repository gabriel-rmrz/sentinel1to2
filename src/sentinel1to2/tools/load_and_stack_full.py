import os
import numpy as np
import rasterio
from .compute_vegetation_indices import compute_vegetation_indices

def load_and_stack_full(folder, data_dir, MEAN=None, STD=None):
  base_name = folder.split("_")[1]

  paths = {
    'dsm': os.path.join(data_dir, folder, f"{base_name}_dsm.tif"),
    's1': os.path.join(data_dir, folder, f"{base_name}_s1.tif"),
    's2': os.path.join(data_dir, folder, f"{base_name}_s2.tif"),
    'worldcover': os.path.join(data_dir, folder, f"{base_name}_worldcover.tif")
  }

  # === DSM ===
  with rasterio.open(paths['dsm']) as src:
    dsm = src.read(1).astype(np.float32)[np.newaxis, ...]
    if MEAN is not None and STD is not None:
      dsm = (dsm - MEAN[0, None, None]) / STD[0, None, None]
    profile = src.profile

  # === Sentinel-1 (es. VV, VH) ===
  with rasterio.open(paths['s1']) as src:
    s1 = src.read((3, 4)).astype(np.float32)  # bands VV/VH
    s1 = np.nan_to_num(s1, nan=0.0, posinf=0.0, neginf=0.0)
    if MEAN is not None and STD is not None:
      s1 = (s1 - MEAN[1:3, None, None]) / STD[1:3, None, None]

  # === WorldCover ===
  with rasterio.open(paths['worldcover']) as src:
    wc = src.read(1).astype(np.float32)[np.newaxis, ...]
    if MEAN is not None and STD is not None:
      wc = (wc - MEAN[3]) / STD[3]



  # === Sentinel-2 ===
  with rasterio.open(paths['s2']) as src:
    s2 = src.read().astype(np.float32)

  
  s2_selected  = s2[ np.r_[1,2,3,4,5,6,7,10,11] ]/10000
  indices, _ind_names = compute_vegetation_indices(s2_selected)


  #print(f"[{folder}] target min/max:", indices.min(), indices.max())

  return dsm, s1, wc, s2_selected, indices, profile
