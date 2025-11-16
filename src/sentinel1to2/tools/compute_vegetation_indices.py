import numpy as np

def compute_vegetation_indices(s2):
  """
  inputs:
    s2: We are considering only the bands relevants for our pipeline. If we have all sentinel bands, we are suposed to filter
        leaving only the 'selected' band defined below:
          band_names = ["b1","blue", "green", "red", "b5", "rededge", "b7", "nir","b8a","b9", "b10", "swir", "b12"]
          selected_bands = [1,2,3,4,5,6,7,10,11]
  """
  # Bande Sentinel-2 standardizzate
  blue   = s2[0]  # B2
  green  = s2[1]  # B3
  red    = s2[2]  # B4
  b5 = s2[3]  # B5
  rededge = s2[4] # B6
  nir    = s2[6]  # B8
  swir   = s2[8] # B12

  eps = 1e-6
  # === Indici Spettrali ===
  ndvi = (nir - red) / (nir + red + eps)
  gndvi = (nir - green) / (nir + green + eps)
  ndre = (nir - rededge) / (nir + rededge + eps)
  reci = (nir / (rededge + eps)) - 1
  msi = swir / (nir + eps)
  ndwi = (green - nir) / (green + nir + eps)
  evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + eps)
  savi = ((nir - red) / (nir + red + 0.5)) * (1.5)
  arvi = (nir - (2 * red - blue)) / (nir + (2 * red - blue) + eps)
  #cire = nir / (rededge + eps)
  cig = (nir - green) / (green + eps)
  cire = (nir - rededge) / (rededge + eps)
  bsi = ((red + swir) - (nir + blue)) / ((red + swir) + (nir + blue) + eps)
  ndsi = (green - swir) / (green + swir + eps)
  mcari = ((b5 - red) - 0.2*(b5 - green)) * b5 / (red + eps)
  #msavi = (2*nir + 1 - np.sqrt(np.power(2*nir+1,2) - 8 *(nir - red)))/2.
  ind_names = ["ndvi", "gndvi", "ndre", "reci", "msi", "ndwi", "evi", "savi", "arvi", "cig", "cire", "bsi", "ndsi", "mcari"]
  #ind_names = ["ndvi", "gndvi", "ndre", "reci", "msi", "ndwi", "evi", "savi", "arvi", "cig", "cire", "bsi", "ndsi", "mcari", "msavi"]

  #s2_selected = s2[ np.r_[1,2,3,4,5,6,7,10,11] ]/10000
  #print(np.clip(mcari, 0, 10).shape)
  indices = np.stack([
      np.clip(ndvi, -1, 1), #np.clip(ndvi, -1, 1), #In teoria mi interessano i soli valori tra 0 e 1
      np.clip(gndvi, -1, 1),
      np.clip(ndre, -1, 1),
      np.clip(reci, -1, 10),
      np.clip(msi, 0, 10),
      np.clip(ndwi, -1, 1),
      np.clip(evi, 0, 2),
      np.clip(savi, -1, 1),
      np.clip(arvi, -1, 1),
      np.clip(cig, -1, 1),
      np.clip(cire, -1, 1),
      #np.clip(cire, 0, 10),
      np.clip(bsi, -1, 1),
      np.clip(ndsi, -1, 1),
      np.clip(mcari, 0, 10),
      #np.clip(msavi, -1, 1)
  ], axis=0).astype(np.float32)

  return indices, ind_names 
