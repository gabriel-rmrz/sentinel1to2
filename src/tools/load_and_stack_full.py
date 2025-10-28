import os
import numpy as np
import rasterio

def load_and_stack_full(folder, data_dir, MEAN, STD):
  base_name = folder.split("_")[1]

  paths = {
    'dsm': os.path.join(data_dir, folder, f"{base_name}_dsm.tif"),
    's1': os.path.join(data_dir, folder, f"{base_name}_s1.tif"),
    's2': os.path.join(data_dir, folder, f"{base_name}_s2.tif"),
    'worldcover': os.path.join(data_dir, folder, f"{base_name}_worldcover.tif")
  }

  # === DSM ===
  with rasterio.open(paths['dsm']) as src:
    dsm = src.read(1)[np.newaxis, ...].astype(np.float32)
    dsm = (dsm - MEAN[0, None, None]) / STD[0, None, None]
    profile = src.profile

  # === Sentinel-1 (es. VV, VH) ===
  with rasterio.open(paths['s1']) as src:
    s1 = src.read((3, 4)).astype(np.float32)  # Assumendo canali 3=VV, 4=VH
    s1 = (s1 - MEAN[1:3, None, None]) / STD[1:3, None, None]

  # === WorldCover ===
  with rasterio.open(paths['worldcover']) as src:
    worldcover = src.read(1)[np.newaxis, ...].astype(np.float32)
    worldcover = (worldcover - MEAN[3, None, None]) / STD[3, None, None]

  # === Sentinel-2 ===
  with rasterio.open(paths['s2']) as src:
    s2 = src.read().astype(np.float32)

  # Bande Sentinel-2 standardizzate
  blue   = s2[1]  # B2
  green  = s2[2]  # B3
  red    = s2[3]  # B4
  b5 = s2[4]  # B5
  rededge = s2[5] # B6
  nir    = s2[7]  # B8
  swir   = s2[11] # B12

  eps = 1e-6

  # === Indici Spettrali ===
  ndvi = (nir - red) / (nir + red + eps)
  """
  gndvi = (nir - green) / (nir + green + eps)
  ndre = (nir - rededge) / (nir + rededge + eps)
  reci = (nir / (rededge + eps)) - 1
  msi = swir / (nir + eps)
  ndwi = (green - nir) / (green + nir + eps)
  evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + eps)
  savi = ((nir - red) / (nir + red + 0.5)) * (1.5)
  arvi = (nir - (2 * red - blue)) / (nir + (2 * red - blue) + eps)
  cire = nir / (rededge + eps)
  bsi = ((red + swir) - (nir + blue)) / ((red + swir) + (nir + blue) + eps)
  ndsi = (green - swir) / (green + swir + eps)
  mcari = [(b5 - red) - 0.2*(b5 - green)] * (b5 / red)
  """

  indices = s2[ np.r_[1,2,3,4,5,6,7,10,11] ]/10000
  #np.stack([
     #np.clip(ndvi, -1, 1) #np.clip(ndvi, -1, 1), #In teoria mi interessano i soli valori tra 0 e 1
     #np.clip(gndvi, -1, 1),
     #np.clip(ndre, -1, 1),
     #np.clip(reci, -1, 10),
     #np.clip(msi, 0, 10),
     #np.clip(ndwi, -1, 1),
     #np.clip(evi, 0, 2),
     #np.clip(savi, -1, 1),
     #np.clip(arvi, -1, 1),
     #np.clip(cire, 0, 10),
     #np.clip(bsi, -1, 1),
     #np.clip(ndsi, -1, 1),
     #np.clip(mcari, 0, 10)
  #], axis=0).astype(np.float32)


  #print(f"[{folder}] target min/max:", indices.min(), indices.max())

  return dsm, s1, worldcover, indices, profile
