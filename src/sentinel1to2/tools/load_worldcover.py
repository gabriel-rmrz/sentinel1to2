import os
import rasterio
import numpy as np

def load_worldcover(scene, day, real_dir):
  wc_path = os.path.join(real_dir, f"{scene}_{day}", f"{day}_worldcover.tif")
  if not os.path.exists(wc_path):
    print(f"⚠️ WorldCover non trovato per {scene}_{day}")
    return None
  with rasterio.open(wc_path) as src:
    return src.read(1).astype(np.int16)
