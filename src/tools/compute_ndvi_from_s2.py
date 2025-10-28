import rasterio
import numpy as np

def compute_ndvi_from_s2(s2_path):
  with rasterio.open(s2_path) as src:
    s2 = src.read().astype(np.float32)
    #red = s2[3]  # banda 4 (indice 3)
    #nir = s2[7]  # banda 8 (indice 7)
    #ndvi = (nir - red) / (nir + red + 1e-6)
    #ndvi = np.clip(ndvi, -1, 1)
    indices = s2[ np.r_[1,2,3,4,5,6,7,10,11] ]/10000
    #ndvi = indices[KKK]
    ndvi = indices
    print(ndvi.shape)
    profile = src.profile
  return ndvi, profile
