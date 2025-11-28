import numpy as np

def compute_vegetation_indices(config, s2):
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
  #TODO: the order of vi have to match the order of the indices for the 
  # method load_and_stack to work properly. Change the logic

  vi = config['params']['vegetation_indices']
  ind_list = []
  ndvi = (nir - red) / (nir + red + eps)
  ind_list.append(ndvi)
  gndvi = (nir - green) / (nir + green + eps)
  ind_list.append(gndvi)
  ndre = (nir - rededge) / (nir + rededge + eps)
  ind_list.append(ndre)
  reci = (nir / (rededge + eps)) - 1
  ind_list.append(reci)
  msi = swir / (nir + eps)
  ind_list.append(msi)
  ndwi = (green - nir) / (green + nir + eps)
  ind_list.append(ndwi)
  evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + eps)
  ind_list.append(evi)
  savi = ((nir - red) / (nir + red + 0.5)) * (1.5)
  ind_list.append(savi)
  arvi = (nir - (2 * red - blue)) / (nir + (2 * red - blue) + eps)
  ind_list.append(arvi)
  cig = (nir - green) / (green + eps)
  ind_list.append(cig)
  cire = (nir - rededge) / (rededge + eps)
  ind_list.append(cire) 
  bsi = ((red + swir) - (nir + blue)) / ((red + swir) + (nir + blue) + eps)
  ind_list.append(bsi)
  ndsi = (green - swir) / (green + swir + eps)
  ind_list.append(ndsi)
  mcari = ((b5 - red) - 0.2*(b5 - green)) * b5 / (red + eps)
  ind_list.append(mcari) 
  #print(ind_list)
  #msavi = (2*nir + 1 - np.sqrt(np.power(2*nir+1,2) - 8 *(nir - red)))/2.
  #ind_names = vi #["ndvi", "gndvi", "ndre", "reci", "msi", "ndwi", "evi", "savi", "arvi", "cig", "cire", "bsi", "ndsi", "mcari"]
  #ind_names = ["ndvi", "gndvi", "ndre", "reci", "msi", "ndwi", "evi", "savi", "arvi", "cig", "cire", "bsi", "ndsi", "mcari", "msavi"]

  '''
  # === Indici Spettrali ===
  ind_list = []
  ndvi = (nir - red) / (nir + red + eps)
  ind_list.append(np.clip(ndvi, -1, 1))
  gndvi = (nir - green) / (nir + green + eps)
  ind_list.append(np.clip(gndvi, -1, 1))
  ndre = (nir - rededge) / (nir + rededge + eps)
  ind_list.append(np.clip(ndre, -1, 1))
  reci = (nir / (rededge + eps)) - 1
  ind_list.append(np.clip(reci, -1, 10))
  msi = swir / (nir + eps)
  ind_list.append(np.clip(msi, 0, 10))
  ndwi = (green - nir) / (green + nir + eps)
  ind_list.append(np.clip(ndwi, -1, 1))
  evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + eps)
  ind_list.append(np.clip(evi, 0, 2))
  savi = ((nir - red) / (nir + red + 0.5)) * (1.5)
  ind_list.append(np.clip(savi, -1, 1))
  arvi = (nir - (2 * red - blue)) / (nir + (2 * red - blue) + eps)
  ind_list.append(np.clip(arvi, -1, 1))
  cig = (nir - green) / (green + eps)
  ind_list.append(np.clip(cig, -1, 1))
  cire = (nir - rededge) / (rededge + eps)
  ind_list.append(np.clip(cire, 0, 10)) 
  bsi = ((red + swir) - (nir + blue)) / ((red + swir) + (nir + blue) + eps)
  ind_list.append(np.clip(bsi, -1, 1))
  ndsi = (green - swir) / (green + swir + eps)
  ind_list.append(np.clip(ndsi, -1, 1))
  mcari = ((b5 - red) - 0.2*(b5 - green)) * b5 / (red + eps)
  ind_list.append(np.clip(mcari, 0, 10)) 
  #print(ind_list)
  #msavi = (2*nir + 1 - np.sqrt(np.power(2*nir+1,2) - 8 *(nir - red)))/2.
  #ind_names = vi #["ndvi", "gndvi", "ndre", "reci", "msi", "ndwi", "evi", "savi", "arvi", "cig", "cire", "bsi", "ndsi", "mcari"]
  #ind_names = ["ndvi", "gndvi", "ndre", "reci", "msi", "ndwi", "evi", "savi", "arvi", "cig", "cire", "bsi", "ndsi", "mcari", "msavi"]
  '''

  indices = np.stack(ind_list, axis=0).astype(np.float32)

  return indices, vi 
